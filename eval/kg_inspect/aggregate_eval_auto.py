#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Aggregate MMAD/KG-Inspect Eval CSVs (no argparse, run directly)
-------------------------------------------------------------------
- Recursively scans all CSVs under `kg_inspect_eval_result/`
- Cleans columns (drop Unnamed*, retries, retry_status)
- Merges & de-duplicates by (dataset, image, question_index)
- Normalizes dataset names (DS-MVTec -> MVTec-AD)
- Computes metrics using `compute_metrics_from_df`
- Writes merged & metrics CSVs back into the same folder

Run:
    python aggregate_eval_auto.py
"""

import os
import sys
from typing import List, Set

import pandas as pd

# ================== CONFIG ==================
EVAL_DIR: str = "kg_inspect_eval_result"
COMBINED_OUTPUT: str = os.path.join(EVAL_DIR, "answers_kg_inspect_merged.csv")
METRICS_OUTPUT: str = os.path.join(EVAL_DIR, "metrics_kg_inspect_merged.csv")

DATASETS_TO_KEEP = ["MVTec-AD", "VisA"]
NORMAL_FLAG: str = "good"

REQUIRED_COLS: Set[str] = {
    "dataset",
    "image",
    "question_index",
    "question_type",
    "question_text",
    "correct_answer",
    "model_answer",
    "is_correct",
    "raw_output",
}
# ===========================================


def _import_compute_metrics():
    """Import compute_metrics_from_df từ repo hoặc summary.py cùng thư mục."""
    try:
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if ROOT_DIR not in sys.path:
            sys.path.insert(0, ROOT_DIR)
        from eval.helper.summary import compute_metrics_from_df  # type: ignore
        print("[INFO] Using compute_metrics_from_df from eval.helper.summary")
        return compute_metrics_from_df
    except Exception as e1:
        print(f"[WARN] Cannot import from eval.helper.summary: {e1}")

    try:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        import summary  # type: ignore
        compute_metrics_from_df = getattr(summary, "compute_metrics_from_df")
        print("[INFO] Using compute_metrics_from_df from local summary.py")
        return compute_metrics_from_df
    except Exception as e2:
        print(f"[ERROR] Cannot import compute_metrics_from_df: {e2}")
        raise ImportError(
            "Failed to import compute_metrics_from_df. "
            "Ensure eval/helper/summary.py exists or place a compatible summary.py next to this script."
        )


def clean_answer_df(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ cột Unnamed, retries, retry_status."""
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    drop_cols = [c for c in ["retries", "retry_status"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _normalize_dataset(name: str) -> str:
    if pd.isna(name):
        return name
    n = str(name).strip()
    if n in ("DS-MVTec", "MVTec", "MVTecAD", "MVTec-AD"):
        return "MVTec-AD"
    return n


def _collect_csvs_recursive(root: str) -> List[str]:
    """Thu thập mọi .csv trong thư mục root và các thư mục con."""
    csvs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(dirpath, fname)
            # Bỏ qua file output của chính script (tránh đọc lại)
            if os.path.abspath(fpath) in {
                os.path.abspath(COMBINED_OUTPUT),
                os.path.abspath(METRICS_OUTPUT),
            }:
                continue
            csvs.append(fpath)
    return csvs


def load_all_csvs(folder: str) -> pd.DataFrame:
    """
    Load mọi CSV (đệ quy) và merge.
    - Skip CSV thiếu REQUIRED_COLS (thường là metrics/summary)
    - De-dup theo (dataset, image, question_index)
    - Chuẩn hoá tên dataset
    """
    if not os.path.isdir(folder):
        raise RuntimeError(f"Folder not found: {folder}")

    csv_files = _collect_csvs_recursive(folder)
    if not csv_files:
        raise RuntimeError(f"No CSV files found under (recursively): {folder}")

    print(f"[INFO] Found {len(csv_files)} CSV files (recursive).")
    dfs: List[pd.DataFrame] = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}. Skipping.")
            continue

        if df.empty:
            print(f"[SKIP] Empty CSV: {path}")
            continue

        df = clean_answer_df(df)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f"[SKIP] {os.path.relpath(path, folder)} missing {missing} (looks like metrics/summary).")
            continue

        dfs.append(df)
        print(f"[LOAD] {os.path.relpath(path, folder)}: {len(df)} rows")

    if not dfs:
        raise RuntimeError("No valid answer CSVs loaded. Check your folder contents.")

    merged = pd.concat(dfs, ignore_index=True)

    if "dataset" in merged.columns:
        merged["dataset"] = merged["dataset"].map(_normalize_dataset)

    # De-dup
    key_cols = ["dataset", "image", "question_index"]
    if all(k in merged.columns for k in key_cols):
        before = len(merged)
        merged = (
            merged.sort_values(by=key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )
        print(f"[INFO] De-duplicated {before - len(merged)} rows by {key_cols}.")
    else:
        print("[WARN] Missing keys for dedup; merged rows may contain duplicates.")

    return merged


def main() -> None:
    compute_metrics_from_df = _import_compute_metrics()

    os.makedirs(EVAL_DIR, exist_ok=True)

    print(f"[INFO] Scanning folder (recursive): {EVAL_DIR}")
    merged_df = load_all_csvs(EVAL_DIR)

    merged_df.to_csv(COMBINED_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {COMBINED_OUTPUT}")

    print(f"[INFO] Computing metrics for datasets: {', '.join(DATASETS_TO_KEEP)}")
    metrics_df = compute_metrics_from_df(
        merged_df,
        normal_flag=NORMAL_FLAG,
        datasets_to_keep=DATASETS_TO_KEEP,
    )

    if isinstance(metrics_df, pd.DataFrame) and "dataset" in metrics_df.columns:
        metrics_df = metrics_df.set_index("dataset")
        metrics_df.index.name = ""

    metrics_df.to_csv(METRICS_OUTPUT, index=True, encoding="utf-8-sig")
    print(f"[SAVE] {METRICS_OUTPUT}")

    print("\n[METRICS]\n")
    try:
        print(metrics_df.to_string())
    except Exception:
        print(metrics_df)


if __name__ == "__main__":
    main()
