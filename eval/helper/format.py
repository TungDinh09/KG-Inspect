import re

def extract_mcq_answer(text: str) -> str:
    """
    Trích đáp án A/B/C/D từ output của model, ngay cả khi model trả lời kèm giải thích.

    Ưu tiên theo thứ tự:
        1. Dòng có pattern "A:" / "B:" / "C:" / "D:"
        2. Pattern độc lập: "Answer is C", "C is correct", "option C", "→ C"
        3. Pattern đơn ký tự A/B/C/D nếu nó xuất hiện rõ ràng (ít dùng)
    """

    if not text:
        return ""

    lines = text.split("\n")

    # --- 1) Ưu tiên dòng có dạng "A:" / "B:" / "C:" / "D:" ---
    for line in lines:
        match = re.search(r"\b([ABCD])\s*:", line, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # --- 2) Bắt các pattern dạng: "answer is C", "correct answer: C", "option C" ---
    match = re.search(
        r"(?i)(answer\s+is|correct\s+answer|option)\s*([ABCD])\b",
        text
    )
    if match:
        return match.group(2).upper()

    # --- 3) Bắt pattern "C is correct" / "C is the answer" ---
    match = re.search(r"\b([ABCD])\s+is\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # --- 4) Bắt ký tự đơn A/B/C/D nếu đứng tách biệt ---
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1).upper()

    return ""
