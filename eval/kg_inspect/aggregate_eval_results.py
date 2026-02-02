import argparse
import os
import sys
from typing import List

import pandas as pd

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, ROOT_DIR)

# Import the metric computation helper used in evaluation
from eval.helper.summary import compute_metrics_from_df  # type: ignore


REQUIRED_COLS = {
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


def clean_answer_df(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Clean a DataFrame loaded from a CSV file:
      - Remove redundant index columns like 'Unnamed: 0', 'Unnamed: 0.1', ...
      - Remove 'retries' and 'retry_status' columns if they exist.

    The input CSV file will be OVERWRITTEN so the columns
    are actually removed from disk, not only in memory.
    """
    original_cols = list(df.columns)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    drop_cols = [c for c in ["retries", "retry_status"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    cleaned_cols = list(df.columns)

    if original_cols != cleaned_cols:
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(
                f"[INFO] Cleaned columns {set(original_cols) - set(cleaned_cols)} "
                f"and rewrote file: {path}"
            )
        except Exception as e:
            print(
                f"[WARN] Failed to rewrite cleaned CSV for {path}: {e}. "
                "Continue using the cleaned DataFrame in memory."
            )

    return df


def merge_answer_csvs(csv_paths: List[str]) -> pd.DataFrame:
    """
    Load and merge multiple answers_*.csv files into a single DataFrame.
    - Only keep files that contain all REQUIRED_COLS.
    - Remove redundant index columns, retries, retry_status.
    - Drop duplicates based on (dataset, image, question_index).
    """
    dfs = []

    for path in csv_paths:
        if not os.path.isfile(path):
            print(f"[WARN] CSV not found, skip: {path}")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[ERROR] Failed to read CSV {path}: {e}")
            continue

        if df.empty:
            print(f"[WARN] CSV is empty, skip: {path}")
            continue

        # Clean columns and overwrite file if modified
        df = clean_answer_df(df, path)

        # Check required columns (after cleaning)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(
                f"[WARN] {path} is missing columns {missing}. "
                "This may be a metrics/summary file, skipping."
            )
            continue

        dfs.append(df)
        print(f"[INFO] Loaded {len(df)} rows from {path}")

    if not dfs:
        raise RuntimeError(
            "No valid CSV files loaded. "
            "Please check the --answers list (you may have passed metrics/summary files by mistake)."
        )

    merged = pd.concat(dfs, ignore_index=True)

    # Drop duplicates by dataset + image + question_index
    key_cols = ["dataset", "image", "question_index"]
    if all(col in merged.columns for col in key_cols):
        before = len(merged)
        merged = (
            merged.sort_values(by=key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
            .reset_index(drop=True)
        )
        after = len(merged)
        print(f"[INFO] Dropped {before - after} duplicate rows.")
    else:
        print(
            "[WARN] Columns (dataset, image, question_index) are incomplete; "
            "cannot reliably drop duplicates."
        )

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multiple KG-Inspect evaluation CSV files (answers) "
            "from different machines and recompute global metrics "
            "(optionally filtered by dataset)."
        )
    )

    parser.add_argument(
        "--answers",
        nargs="+",
        required=True,
        help=(
            "List of paths to answers_*.csv files from different machines. "
            "Example: machine1/answers.csv machine2/answers.csv ..."
        ),
    )

    parser.add_argument(
        "--combined_output",
        type=str,
        default="answers_kg_inspect_merged.csv",
        help="Output path for the merged answers CSV.",
    )

    parser.add_argument(
        "--metrics_output",
        type=str,
        default="metrics_kg_inspect_merged.csv",
        help="Output path for the recomputed metrics CSV.",
    )

    parser.add_argument(
        "--normal_flag",
        type=str,
        default="good",
        help=(
            "Substring indicating normal images in the image path "
            "(same meaning as normal_flag in the original evaluation script)."
        ),
    )

    parser.add_argument(
        "--datasets_to_keep",
        nargs="+",
        default=["MVTec-AD", "VisA"],
        help=(
            "List of datasets to keep when computing metrics. "
            "Example: --datasets_to_keep MVTec-AD VisA"
        ),
    )

    args = parser.parse_args()

    # 1) Merge answer CSV files
    print("[INFO] Merging answer CSVs...")
    merged_df = merge_answer_csvs(args.answers)

    # 2) Save merged answers CSV
    os.makedirs(os.path.dirname(args.combined_output) or ".", exist_ok=True)
    merged_df.to_csv(args.combined_output, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved merged answers CSV to: {args.combined_output}")

    # 3) Compute metrics for selected datasets
    print(
        f"[INFO] Computing metrics for datasets: "
        f"{', '.join(args.datasets_to_keep)}"
    )

    metrics_df = compute_metrics_from_df(
        merged_df,
        normal_flag=args.normal_flag,
        datasets_to_keep=args.datasets_to_keep,
    )

    if "dataset" in metrics_df.columns:
        metrics_df = metrics_df.set_index("dataset")
        metrics_df.index.name = ""  # empty index header to match the expected format

    os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
    metrics_df.to_csv(args.metrics_output, index=True, encoding="utf-8-sig")
    print(f"[INFO] Saved metrics CSV to: {args.metrics_output}")

    print("[INFO] Done. Metrics table:")
    try:
        print(metrics_df.to_string())
    except Exception:
        print(metrics_df)


if __name__ == "__main__":
    main()

# Example usage:
# python eval/kg_inspect/aggregate_eval_results.py \
#   --answers \
#     kg_inspect_eval_result/kg_inspect_bottle_eval/kg_inspect_bottle_eval.csv \
#     kg_inspect_eval_result/kg_inspect_cable_eval/kg_inspect_cable_eval.csv \
#     kg_inspect_eval_result/kg_inspect_capsule_eval/kg_inspect_capsule_eval.csv \
#     kg_inspect_eval_result/kg_inspect_carpet_eval/kg_inspect_carpet_eval.csv \
#     kg_inspect_eval_result/kg_inspect_grid_eval/kg_inspect_grid_eval.csv \
#     kg_inspect_eval_result/kg_inspect_hazard_eval/kg_inspect_hazard_eval.csv \
#     kg_inspect_eval_result/kg_inspect_metal_nut_eval/kg_inspect_metal_nut_eval.csv \
#     kg_inspect_eval_result/kg_inspect_pill_eval/kg_inspect_pill_eval.csv \
#     kg_inspect_eval_result/kg_inspect_screw_eval/kg_inspect_screw_eval.csv \
#     kg_inspect_eval_result/kg_inspect_toothbrush_eval/kg_inspect_toothbrush_eval.csv \
#     kg_inspect_eval_result/kg_inspect_transistor_eval/kg_inspect_transistor_eval.csv \
#     kg_inspect_eval_result/kg_inspect_zipper_eval/kg_inspect_zipper_eval.csv \
#   --combined_output kg_inspect_eval_result/answers_kg_inspect_merged.csv \
#   --metrics_output  kg_inspect_eval_result/metrics_kg_inspect_merged.csv \
#   --datasets_to_keep MVTec-AD VisA \
#   --normal_flag good
