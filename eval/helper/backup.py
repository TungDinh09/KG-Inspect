
import logging
import os

import pandas as pd

def load_existing_results(
    answers_csv_path: str,
) -> tuple[list[dict], set[tuple[str, int]]]:
    """
    Nếu đã có file answers CSV cũ, load vào để:
      - records: list[dict] các dòng cũ
      - processed_keys: set[(image, question_index)] để biết câu nào đã xử lý

    Nếu file không tồn tại → trả về ([], set()).
    """
    if not os.path.isfile(answers_csv_path):
        logging.info(f"No existing answers file found at {answers_csv_path}, starting fresh.")
        return [], set()

    logging.info(f"Resuming from existing answers file: {answers_csv_path}")
    df_prev = pd.read_csv(answers_csv_path, encoding="utf-8-sig")
    records = df_prev.to_dict(orient="records")

    processed_keys: set[tuple[str, int]] = set()
    for row in records:
        try:
            img = row["image"]
            q_idx = int(row["question_index"])
            processed_keys.add((img, q_idx))
        except Exception as e:
            logging.warning(f"Bad row in existing CSV (skip from processed_keys): {e} | row={row}")

    logging.info(f"Loaded {len(records)} previous records, {len(processed_keys)} processed keys.")
    return records, processed_keys