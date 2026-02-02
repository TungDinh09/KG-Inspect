import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm

# ==== pipeline & dependencies ====
from kg_inspect.pipeline_inspect import InspectionPipeline
from kg_inspect.rag.rag_manager import initialize_rag
from kg_inspect.utils.prompt import Prompt
from lightrag.base import QueryParam

# ==== helper functions (reuse your existing helpers) ====
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from eval.helper.summary import (
    setup_logging,
    compute_metrics_from_df,
)
from eval.helper.format import extract_mcq_answer
from eval.helper.summary import parse_answer  # fallback parser

SORRY_TEXT = "Sorry, I'm not able to provide an answer to that question.[no-context]"


async def build_visual_context_once(pipeline: InspectionPipeline, image_path: str) -> str:
    try:
        inspection = await pipeline._inspect_images([image_path])
        visual_context = Prompt.build_visual_context(
            image_paths=inspection["paths"],
            labels=inspection["labels"],
            anomaly_flags=inspection["anomaly_flags"],
        )
        return visual_context
    except Exception as e:
        logging.exception(f"Visual context failed for {image_path}: {e}")
        return ""


# ---------- NEW: helpers to fetch Options from JSON ----------
def _norm_path(p: str) -> str:
    # Normalize to keep JSON keys <-> CSV keys stable (slash/backslash)
    return p.replace("\\", "/").lstrip("./")


def _find_question_block(js_item: dict, q_text: str) -> Optional[dict]:
    """Find the question block inside 'conversation' or 'annotated_conversation' matching the question text."""
    for key in ("conversation", "annotated_conversation"):
        arr = js_item.get(key) or []
        for blk in arr:
            if str(blk.get("Question", "")).strip() == str(q_text).strip():
                return blk
    return None


def build_options_text_from_json(
    json_index: Dict[str, dict],
    image_rel: str,
    q_text: str
) -> Optional[str]:
    """
    Return a string in the format:

        Options:
        A. ...
        B. ...
        [C. ...]
        [D. ...]

    if options are found in the JSON; otherwise return None.
    """
    key_candidates = []
    # JSON keys may be full path ("VisA/pcb2/test/good/0011.JPG") or a shorter suffix ("test/good/0011.JPG")
    rel_norm = _norm_path(image_rel)
    key_candidates.append(rel_norm)

    # Also try using the suffix after dataset prefix (if present)
    parts = rel_norm.split("/")
    if len(parts) >= 3:
        key_candidates.append("/".join(parts[-3:]))

    js_item = None
    for k in key_candidates:
        if k in json_index:
            js_item = json_index[k]
            break

        # Some JSONs store full keys at a higher level; try matching by suffix
        matches = [kk for kk in json_index.keys() if _norm_path(kk).endswith(rel_norm)]
        if matches:
            js_item = json_index[matches[0]]
            break

    if not js_item:
        return None

    block = _find_question_block(js_item, q_text)
    if not block:
        return None

    opts = block.get("Options") or {}
    if not opts:
        return None

    letters = [k for k in ("A", "B", "C", "D") if k in opts]
    if not letters:
        return None

    lines = ["Options:"]
    for L in letters:
        lines.append(f"{L}. {opts[L]}")
    return "\n".join(lines)


def question_has_lettered_options(q_text: str) -> bool:
    # Detect at least “A.” and “B.”
    return re.search(r"(?mi)^\s*A\.\s+.+\n\s*B\.\s+.+", q_text or "") is not None
# --------------------------------------------------------------


async def reeval_rows(
    df: pd.DataFrame,
    data_root: str,
    json_index: Dict[str, dict],
    mode: str = "hybrid",
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Re-evaluate rows where raw_output == SORRY_TEXT (or model_answer is missing),
    retrying up to `max_retries`.
    """
    rag = await initialize_rag()
    pipeline = InspectionPipeline(rag=rag)

    vctx_cache: Dict[str, str] = {}

    if "retries" not in df.columns:
        df["retries"] = 0
    if "retry_status" not in df.columns:
        df["retry_status"] = "untouched"

    need_retry_mask = (
        df["raw_output"].astype(str).str.strip() == SORRY_TEXT
    ) | (
        df["model_answer"].isna() | (df["model_answer"].astype(str).str.strip() == "")
    )

    rows_to_retry = df[need_retry_mask].copy()
    if rows_to_retry.empty:
        logging.info("No rows need retry. Returning original DataFrame.")
        await rag.finalize_storages()
        return df

    logging.info(f"Rows to retry: {len(rows_to_retry)}")

    try:
        for idx, row in tqdm(rows_to_retry.iterrows(), total=len(rows_to_retry), desc="Re-evaluating"):
            image_rel = str(row["image"])
            q_text = str(row.get("question_text", ""))
            q_type = str(row.get("question_type", "")).lower()
            correct_ans = str(row.get("correct_answer", ""))

            image_abs = os.path.join(data_root, image_rel)
            if not os.path.isfile(image_abs):
                logging.warning(f"Image file not found, skip: {image_abs}")
                df.loc[idx, ["retry_status", "retries"]] = ["missing_image", 0]
                continue

            if image_abs not in vctx_cache:
                vctx_cache[image_abs] = await build_visual_context_once(pipeline, image_abs)

            visual_context = (vctx_cache.get(image_abs, "") or "").rstrip() + "\n\n"

            # ========== NEW: ensure the question includes Options ==========
            q_text_full = q_text.strip()
            if not question_has_lettered_options(q_text_full):
                opt_text = build_options_text_from_json(json_index, image_rel, q_text)
                if opt_text:
                    q_text_full = f"{q_text_full}\n\n{opt_text}"
                elif q_type == "anomaly detection":
                    # Safe fallback for anomaly detection when JSON has no options
                    q_text_full = f"{q_text_full}\n\nOptions:\nA. Yes.\nB. No."
            # =============================================================

            param = QueryParam(
                mode=mode,
                stream=False,
                enable_rerank=False,
                conversation_history=[],
            )

            best_response = None
            best_ans = None
            attempt = 0

            while attempt < max_retries:
                attempt += 1
                try:
                    augmented_query = Prompt.build_augmented_query(
                        user_query=q_text_full,
                        visual_context=visual_context,
                    )
                    result = await rag.aquery(
                        augmented_query,
                        images=[image_abs],
                        param=param,
                        system_prompt=None,  # keep as-is
                    )
                except Exception as e:
                    logging.exception(f"rag.aquery failed (attempt {attempt}) for {image_rel}: {e}")
                    result = ""

                response_str = str(result) if result is not None else ""
                parsed = extract_mcq_answer(response_str)
                if not parsed:
                    try:
                        parsed = parse_answer(response_str, None)
                    except Exception:
                        parsed = ""

                if response_str.strip() != SORRY_TEXT and parsed:
                    best_response = response_str
                    best_ans = parsed
                    break

            df.loc[idx, "retries"] = attempt
            if best_response is not None:
                df.loc[idx, "raw_output"] = best_response
                df.loc[idx, "model_answer"] = best_ans
                if correct_ans:
                    df.loc[idx, "is_correct"] = (best_ans == correct_ans)
                df.loc[idx, "retry_status"] = "updated"
            else:
                df.loc[idx, "retry_status"] = "skipped_after_max_retries"

    finally:
        try:
            await rag.finalize_storages()
        except Exception as e:
            logging.warning(f"Error during rag.finalize_storages(): {e}")

    return df


async def main_async(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    setup_logging(log_path)

    df = pd.read_csv(args.answers_csv)

    # Load JSON annotations (mmad.json)
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    # JSON can be a dict (image -> object). If it is a list, convert it to a dict index.
    if isinstance(data_json, list):
        json_index = {_norm_path(item.get("image_path") or item.get("image") or ""): item for item in data_json}
    else:
        # Already a dict: normalize keys and reuse
        json_index = {_norm_path(k): v for k, v in data_json.items()}

    updated_df = await reeval_rows(
        df,
        data_root=args.data_path,
        json_index=json_index,
        mode=args.mode,
        max_retries=args.max_retries,
    )

    out_answers = os.path.join(
        args.output_dir,
        os.path.basename(args.answers_csv).replace(".csv", "_reval.csv")
    )
    updated_df.to_csv(out_answers, index=False, encoding="utf-8-sig")
    logging.info(f"Saved updated answers CSV → {out_answers}")

    try:
        metrics_df = compute_metrics_from_df(
            updated_df,
            normal_flag=args.normal_flag,
            datasets_to_keep=["MVTec-AD", "VisA"],
        )
        out_metrics = os.path.join(args.output_dir, "metrics_kg_inspect_reval.csv")
        metrics_df.to_csv(out_metrics, index=True, encoding="utf-8-sig")
        logging.info("Final metrics table (after re-eval):\n" + metrics_df.to_string())
        logging.info(f"Saved metrics → {out_metrics}")
    except Exception as e:
        logging.exception(f"Metrics computation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate failed rows in a KG-Inspect answers CSV and recompute metrics."
    )
    parser.add_argument(
        "--answers_csv",
        type=str,
        required=True,
        help="Path to the original answers CSV (from a previous evaluation run).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root folder of images (MMAD layout).",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to mmad.json (annotation).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kg_inspect_reval",
        help="Output directory for updated CSV and metrics.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        help="RAG mode (local/global/hybrid/mix/naive/bypass).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries for rows with SORRY/no output.",
    )
    parser.add_argument(
        "--normal_flag",
        type=str,
        default="good",
        help="Substring indicating normal images in path (used by metrics helper).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="reval_log.txt",
        help="Log file name inside output_dir.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

# Example:
# python eval_retry_failed_rows.py \
#   --answers_csv /path/to/answers_kg_inspect_DS-MVTec_VisA.csv \
#   --data_path   /path/to/MMAD_root \
#   --json_path   /path/to/mmad.json \
#   --output_dir  ./kg_inspect_reval \
#   --mode        hybrid \
#   --max_retries 5
