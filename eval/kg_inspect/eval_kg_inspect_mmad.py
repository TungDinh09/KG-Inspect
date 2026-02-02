# eval_kg_inspect_mmad.py

import argparse
import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, List, Tuple, Set

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==== pipeline & dependencies ====
from kg_inspect.pipeline_inspect import InspectionPipeline
from kg_inspect.rag.rag_manager import initialize_rag
from kg_inspect.utils.prompt import Prompt
from lightrag.base import QueryParam

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

# ==== helper functions ====
from eval.helper.summary import (
    setup_logging,
    parse_conversation,
    build_user_query,
    parse_answer,  # kept as fallback
    compute_metrics_from_df,
)

from eval.helper.format import extract_mcq_answer
from eval.helper.backup import load_existing_results


# =========================================================
# Constants & helpers for question/options handling
# =========================================================

SORRY_TEXT = "Sorry, I'm not able to provide an answer to that question.[no-context]"


def question_has_lettered_options(q_text: str) -> bool:
    """
    Detect whether the question text already contains lettered options like:
        A. ...
        B. ...
    using a line-based regex.
    """
    if not q_text:
        return False

    # Match patterns like:
    #   A. something
    #   B. something
    return re.search(r"(?mi)^\s*A\.\s+.+\n\s*B\.\s+.+", q_text) is not None


def build_options_text_from_question(q: Dict[str, Any]) -> str | None:
    """
    From a question object (returned by parse_conversation), build:

        Options:
        A. ...
        B. ...
        [C. ...]
        [D. ...]

    if options exist in q["options"] or q["Options"].
    """
    opts = q.get("options") or q.get("Options")
    if not isinstance(opts, dict):
        return None

    letters = [k for k in ("A", "B", "C", "D") if k in opts]
    if not letters:
        return None

    lines = ["Options:"]
    for L in letters:
        lines.append(f"{L}. {opts[L]}")
    return "\n".join(lines)


def save_accuracy_plot(
    checkpoints: List[int],
    accuracies: List[float],
    output_dir: str,
    correct: int,
    total: int,
) -> None:
    """
    Plot cumulative accuracy vs number of processed images and save as PNG.

    - checkpoints: number of images processed at each checkpoint
    - accuracies:  accuracy values (0.0 - 1.0)
    - correct:     total correct answers so far
    - total:       total questions processed so far
    """
    if not checkpoints or not accuracies:
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(checkpoints, accuracies, marker="o")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Images processed")
    plt.ylabel("Accuracy")
    current_acc = accuracies[-1] * 100.0
    plt.title(
        f"Accuracy trend (current: {correct}/{total} = {current_acc:.2f}%)"
    )
    plt.grid(True)
    plt.tight_layout()

    last_checkpoint = checkpoints[-1]
    out_path = os.path.join(
        output_dir, f"accuracy_until_{last_checkpoint:05d}_images.png"
    )
    plt.savefig(out_path)
    plt.close()

    logging.info(
        f"Saved accuracy plot at {last_checkpoint} images "
        f"({correct}/{total} = {current_acc:.2f}%) → {out_path}"
    )


# =========================================================
# Main evaluation logic
# =========================================================
async def evaluate(
    data_path: str,
    json_path: str,
    output_dir: str,
    device: str,
    mode: str,
    normal_flag: str = "good",
    resume: bool = False,
    save_every: int = 1000,
    min_images_for_metrics: int = 10,
    max_retries: int = 5,
):
    """
    Run KG-Inspect evaluation (InspectionPipeline) on MMAD-style DS-MVTec & VisA.

    Optimizations:
      - Each image runs CNNInspect (_inspect_images) only once,
        even if multiple questions reference the same image.
      - Metrics are computed/printed only when the number of unique images
        >= min_images_for_metrics.
      - Every 100 processed images, a cumulative accuracy plot is generated.
      - Retry multiple times if the model returns SORRY/no-context or
        the answer cannot be parsed, ensuring MCQ options A/B/C/D
        are always present in the prompt.
    """

    os.makedirs(output_dir, exist_ok=True)

    answers_csv_path = os.path.join(output_dir, "answers_kg_inspect_DS-MVTec_VisA.csv")

    # Set DEVICE env for CNN/LLM if provided
    if device:
        os.environ["DEVICE"] = device

    # Initialize RAG + Pipeline
    logging.info("Initializing KGInspect / LightRAG...")
    rag = await initialize_rag()
    pipeline = InspectionPipeline(rag=rag)

    # Load MMAD-like JSON
    with open(json_path, "r", encoding="utf-8") as f:
        chat_ad: Dict[str, Any] = json.load(f)

    # Resume or start fresh
    if resume:
        records, processed_keys = load_existing_results(answers_csv_path)
    else:
        records: List[Dict[str, Any]] = []
        processed_keys: Set[Tuple[str, int]] = set()

    logging.info(f"Total items in JSON: {len(chat_ad)}")

    # Select effective mode
    allowed_modes = {"local", "global", "hybrid", "naive", "mix", "bypass"}

    if mode not in allowed_modes:
        logging.warning(f"Invalid mode '{mode}', falling back to 'hybrid'.")
        effective_mode = "hybrid"
    else:
        effective_mode = mode

    logging.info(f"Running evaluation with RAG mode: {effective_mode}")

    new_record_count = 0

    # Cache per image to avoid repeated CNNInspect runs
    image_cache: Dict[str, Dict[str, Any]] = {}

    # Accuracy tracking
    correct_so_far = 0
    total_so_far = 0
    images_processed = 0
    plot_interval = 100

    checkpoints: List[int] = []
    accuracies: List[float] = []

    try:
        for image_rel_path, text_gt in tqdm(chat_ad.items(), desc="Evaluating images"):
            dataset_name = image_rel_path.split("/")[0]
            if dataset_name not in ["MVTec-AD", "VisA"]:
                continue

            image_abs_path = os.path.join(data_path, image_rel_path)
            logging.info(
                f"Processing image: {image_rel_path} from dataset: {dataset_name}"
            )

            if not os.path.isfile(image_abs_path):
                logging.warning(f"Image file not found, skipping: {image_abs_path}")
                continue

            questions, answers, question_types = parse_conversation(text_gt)
            if not questions:
                logging.warning(f"No questions found for {image_rel_path}")
                continue

            pending_indices: List[int] = []
            for q_idx in range(len(questions)):
                question_index = q_idx + 1
                key = (image_rel_path, question_index)
                if key not in processed_keys:
                    pending_indices.append(q_idx)

            if not pending_indices:
                logging.info(
                    f"[RESUME] All questions already processed for image: {image_rel_path}"
                )
                continue

            if image_abs_path not in image_cache:
                logging.info(f"Running CNNInspect once for image: {image_abs_path}")
                try:
                    inspection = await pipeline._inspect_images([image_abs_path])
                except Exception as e:
                    logging.exception(
                        f"Error during _inspect_images for {image_abs_path}: {e}"
                    )
                    inspection = {
                        "paths": [image_abs_path],
                        "labels": ["object"],
                        "anomaly_flags": [False],
                        "confidences": [0.0],
                        "scores": [None],
                        "thresholds": [None],
                    }

                try:
                    visual_context = Prompt.build_visual_context(
                        image_paths=inspection["paths"],
                        labels=inspection["labels"],
                        anomaly_flags=inspection["anomaly_flags"],
                    )
                except Exception as e:
                    logging.exception(
                        f"Error during Prompt.build_visual_context for {image_abs_path}: {e}"
                    )
                    visual_context = ""

                image_cache[image_abs_path] = {
                    "inspection": inspection,
                    "visual_context": visual_context,
                }

            inspection_data = image_cache[image_abs_path]
            visual_context: str = inspection_data.get("visual_context", "")

            new_questions_for_image = 0

            for q_idx in pending_indices:
                question_index = q_idx + 1
                key = (image_rel_path, question_index)

                q = questions[q_idx]
                correct_ans = answers[q_idx]
                q_type = question_types[q_idx]

                original_q_text = str(q.get("text", "")).strip()

                # Ensure the model always receives lettered options
                q_for_model = dict(q)
                q_text_full = original_q_text

                if not question_has_lettered_options(q_text_full):
                    opt_text = build_options_text_from_question(q_for_model)
                    if opt_text:
                        q_text_full = f"{q_text_full}\n\n{opt_text}"
                    elif str(q_type).lower() == "anomaly detection":
                        q_text_full = (
                            f"{q_text_full}\n\nOptions:\nA. Yes.\nB. No."
                        )

                q_for_model["text"] = q_text_full

                user_query = build_user_query(q_for_model)
                logging.info(
                    f"Image: {image_rel_path} | Q#{question_index} | "
                    f"Type: {q_type} | GT: {correct_ans}"
                )

                param = QueryParam(
                    mode=effective_mode,
                    stream=False,
                    enable_rerank=False,
                    conversation_history=[],
                )

                augmented_query = Prompt.build_augmented_query(
                    user_query=user_query,
                    visual_context=visual_context,
                )

                best_response_str = ""
                best_model_ans = ""
                attempt = 0

                while attempt < max_retries:
                    attempt += 1
                    try:
                        result = await rag.aquery(
                            augmented_query,
                            images=[image_abs_path],
                            param=param,
                            system_prompt=None,
                        )
                    except Exception as e:
                        logging.exception(
                            f"Error during rag.aquery for {image_rel_path} "
                            f"Q#{question_index} (attempt {attempt}): {e}"
                        )
                        result = ""

                    response_str = str(result) if result is not None else ""
                    model_ans = extract_mcq_answer(response_str)

                    if not model_ans:
                        try:
                            model_ans = parse_answer(
                                response_str,
                                q_for_model.get("options"),
                            )
                        except Exception:
                            model_ans = ""

                    if response_str.strip() != SORRY_TEXT and model_ans:
                        best_response_str = response_str
                        best_model_ans = model_ans
                        break

                    best_response_str = response_str
                    best_model_ans = model_ans

                response_str = best_response_str
                model_ans = best_model_ans

                is_correct = (model_ans == correct_ans)

                record = {
                    "dataset": dataset_name,
                    "image": image_rel_path,
                    "question_index": question_index,
                    "question_type": q_type,
                    "question_text": original_q_text,
                    "correct_answer": correct_ans,
                    "model_answer": model_ans,
                    "is_correct": is_correct,
                    "raw_output": response_str,
                }
                records.append(record)
                processed_keys.add(key)
                new_record_count += 1
                new_questions_for_image += 1

                total_so_far += 1
                if is_correct:
                    correct_so_far += 1

                if save_every > 0 and new_record_count % save_every == 0:
                    df_partial = pd.DataFrame(records)
                    df_partial.to_csv(
                        answers_csv_path, index=False, encoding="utf-8-sig"
                    )

            if new_questions_for_image > 0:
                images_processed += 1

                if images_processed % plot_interval == 0 and total_so_far > 0:
                    acc_now = correct_so_far / total_so_far
                    checkpoints.append(images_processed)
                    accuracies.append(acc_now)
                    save_accuracy_plot(
                        checkpoints,
                        accuracies,
                        output_dir,
                        correct_so_far,
                        total_so_far,
                    )

    finally:
        try:
            await rag.finalize_storages()
        except Exception as e:
            logging.warning(f"Error during rag.finalize_storages(): {e}")

    if not records:
        logging.error("No records collected.")
        return

    df = pd.DataFrame(records)
    df.to_csv(answers_csv_path, index=False, encoding="utf-8-sig")

    metrics_df = compute_metrics_from_df(
        df,
        normal_flag=normal_flag,
        datasets_to_keep=["MVTec-AD", "VisA"],
    )

    metrics_csv_path = os.path.join(output_dir, "metrics_kg_inspect_MVTec-AD_VisA.csv")
    metrics_df.to_csv(metrics_csv_path, index=True, encoding="utf-8-sig")

    logging.info(f"Metrics saved to: {metrics_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KG-Inspect (InspectionPipeline) on MMAD-style DS-MVTec & VisA."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Root directory of MMAD images.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to mmad.json annotation file.")
    parser.add_argument("--output_dir", type=str, default="./kg_inspect_eval", help="Output directory.")
    parser.add_argument("--device", type=str, default=None, help="Device string (cuda / cpu).")
    parser.add_argument("--mode", type=str, default="hybrid", help="RAG mode.")
    parser.add_argument("--normal_flag", type=str, default="good", help="Substring indicating normal images.")
    parser.add_argument("--log_file", type=str, default="kg_inspect_eval_log.txt", help="Log file name.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing answers CSV.")
    parser.add_argument("--save_every", type=int, default=500, help="Save every N new records.")
    parser.add_argument("--min_images_for_metrics", type=int, default=10, help="Minimum images for metrics.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries on invalid answers.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, args.log_file))

    asyncio.run(
        evaluate(
            data_path=args.data_path,
            json_path=args.json_path,
            output_dir=args.output_dir,
            device=args.device,
            mode=args.mode,
            normal_flag=args.normal_flag,
            resume=args.resume,
            save_every=args.save_every,
            min_images_for_metrics=args.min_images_for_metrics,
            max_retries=args.max_retries,
        )
    )


if __name__ == "__main__":
    main()
