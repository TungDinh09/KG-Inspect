import logging
import os
import re
from difflib import get_close_matches
from typing import Dict, Any, List, Tuple
import yaml
import pandas as pd
import sys

# ---------------------------
# Logging setup
# ---------------------------

def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info(f"Logging to {log_path}")




def parse_conversation(text_gt: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Parse MMAD-style conversation block into:
      - questions: list of dicts with 'text' and 'options' (A-E)
      - answers: correct answers (A-E)
      - question_types: e.g. "Anomaly Detection", "Defect Classification", ...
    """
    questions = []
    answers = []
    question_types = []

    # Find key starting with "conversation"
    for key, conversation in text_gt.items():
        if not key.startswith("conversation"):
            continue
        for qa in conversation:
            options_items = list(qa["Options"].items())  # dict of { 'A': '...', ... }
            options_text = ""
            for opt_key, opt_val in options_items:
                options_text += f"{opt_key}. {opt_val}\n"

            question_text = qa["Question"]
            q_type = qa.get("type", "Unknown")

            questions.append(
                {
                    "text": question_text,
                    "options": qa["Options"],          # keep dict
                    "options_text": options_text.strip()
                }
            )
            answers.append(qa["Answer"])
            question_types.append(q_type)
        break

    return questions, answers, question_types


# ---------------------------
# Prompt building & parsing
# ---------------------------

BASE_INSTRUCTION = """
You are an industrial inspector who checks products by images.
You must judge whether there is a defect in the query image and answer the questions about it.

Each question provides multiple choices (A, B, C, D, E).
You MUST answer with ONLY the option's letter wrapped in the pattern:

Answer: X.

For example:
Answer: A.
Answer: C.

DO NOT output anything else.
""".strip()


def build_user_query(question: Dict[str, Any]) -> str:
    """
    Build the user_query text for MultimodalRAGPipeline.run(...)
    """
    q_text = question["text"]
    options_text = question["options_text"]
    user_query = (
        BASE_INSTRUCTION
        + "\n\n"
        + "Question:\n"
        + q_text
        + "\n\nChoices:\n"
        + options_text
        + "\n\nPlease respond with 'Answer: X.' only."
    )
    return user_query


ANSWER_PATTERN_SINGLE = re.compile(r"\b([A-E])\b")


def parse_answer(response_text: str, options: Dict[str, str]) -> str:
    """
    Parse the model's response to extract a single letter A-E.
    If regex fails, try fuzzy matching against option text.
    """
    if not response_text:
        return ""

    # 1) Regex to find letters A-E
    matches = ANSWER_PATTERN_SINGLE.findall(response_text)
    if matches:
        # take the last match, similar to GPT script
        return matches[-1]

    # 2) Fallback: fuzzy match against option sentences
    logging.warning(f"Failed to regex parse answer from response: {response_text!r}")
    option_values = list(options.values())
    closest_matches = get_close_matches(response_text, option_values, n=1, cutoff=0.0)
    if closest_matches:
        closest = closest_matches[0]
        for k, v in options.items():
            if v == closest:
                return k

    return ""


# ---------------------------
# Metrics computation
# ---------------------------

def normalize_question_type(qt: str) -> str:
    """
    Map fine-grained types to your desired table columns.
    (You can extend this mapping if needed.)
    """
    if qt in ["Object Structure", "Object Details"]:
        return "Object Analysis"
    return qt


TARGET_COLUMNS = [
    "Anomaly Detection",
    "Defect Classification",
    "Defect Localization",
    "Defect Description",
    "Defect Analysis",
    "Object Classification",
    "Object Analysis",
]


def compute_metrics_from_df(
    df: pd.DataFrame,
    normal_flag: str = "good",
    datasets_to_keep: List[str] = None,
) -> pd.DataFrame:
    """
    df: per-question DataFrame với cột:
        ['dataset','image','question_type','correct_answer','model_answer','is_correct', ...]
    Trả về bảng metrics theo đúng logic bản gốc MMAD:
      - Chuẩn hoá question_type (gộp Object Structure/Details -> Object Analysis)
      - Chuẩn hoá dataset (DS-MVTec -> MVTec-AD)
      - 'Anomaly Detection' = balanced accuracy = (acc_normal + acc_abnormal)/2 * 100
      - Average = mean của riêng các cột câu hỏi (không gộp Recall/Precision/F1/Overkill/Miss)
    """
    if datasets_to_keep is None:
        datasets_to_keep = ["MVTec-AD", "VisA"]

    # --- chuẩn hoá loại câu hỏi
    df = df.copy()
    df["question_type"] = df["question_type"].apply(normalize_question_type)

    # --- chuẩn hoá tên dataset để khớp cách tính bản gốc
    def _norm_ds(name: str) -> str:
        if name in ("DS-MVTec", "MVTec", "MVTec-AD"):
            return "MVTec-AD"
        return name

    df["dataset"] = df["dataset"].apply(_norm_ds)

    # --- lọc dataset như mong muốn
    df = df[df["dataset"].isin(datasets_to_keep)].copy()
    if df.empty:
        raise ValueError("No data found for datasets: " + ", ".join(datasets_to_keep))

    # Cột đích (câu hỏi) giống bản của bạn
    question_cols = [
        "Anomaly Detection",
        "Defect Classification",
        "Defect Localization",
        "Defect Description",
        "Defect Analysis",
        "Object Classification",
        "Object Analysis",
    ]

    metrics_df = pd.DataFrame(index=datasets_to_keep)

    for ds in datasets_to_keep:
        df_ds = df[df["dataset"] == ds]

        # 1) tính accuracy từng loại câu hỏi (tạm thời)
        for qt in question_cols:
            df_q = df_ds[df_ds["question_type"] == qt]
            acc = 100.0 * (df_q["is_correct"].sum() / len(df_q)) if len(df_q) else 0.0
            metrics_df.at[ds, qt] = acc

        det = df_ds[df_ds["question_type"] == "Anomaly Detection"].copy()
        if len(det):
            det["is_normal"] = det["image"].str.contains(normal_flag)
            normal_total = det["is_normal"].sum()
            abnormal_total = (~det["is_normal"]).sum()

            tn = (det["is_normal"] & det["is_correct"]).sum()
            tp = ((~det["is_normal"]) & det["is_correct"]).sum()

            acc_normal = tn / normal_total if normal_total > 0 else 0.0
            acc_abnormal = tp / abnormal_total if abnormal_total > 0 else 0.0
            balanced = (acc_normal + acc_abnormal) / 2.0 * 100.0
            metrics_df.at[ds, "Anomaly Detection"] = balanced

            # các chỉ số phát hiện (giữ nguyên như bạn đang làm)
            fp = (det["is_normal"] & (~det["is_correct"])).sum()
            fn = ((~det["is_normal"]) & (~det["is_correct"])).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            overkill = (1 - acc_normal) * 100.0
            miss = (1 - acc_abnormal) * 100.0

            metrics_df.at[ds, "Recall"] = recall * 100.0
            metrics_df.at[ds, "Precision"] = precision * 100.0
            metrics_df.at[ds, "F1"] = f1 * 100.0
            metrics_df.at[ds, "Overkill"] = overkill
            metrics_df.at[ds, "Miss"] = miss
        else:
            for col in ["Recall", "Precision", "F1", "Overkill", "Miss"]:
                metrics_df.at[ds, col] = 0.0

        metrics_df.at[ds, "Average"] = metrics_df.loc[ds, question_cols].mean()

    metrics_df.loc["Average"] = metrics_df.mean()

    return metrics_df
