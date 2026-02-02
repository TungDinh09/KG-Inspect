from __future__ import annotations
from typing import Optional, List
import os
import re


class Prompt:
    VISUAL_HEADING: str = "=== VISUAL INSPECTION REPORT ==="

    _WIN_PATH_RE = re.compile(r"([A-Za-z]:\\[^ \n\r\t]+)")
    _UNC_PATH_RE = re.compile(r"(\\\\[^ \n\r\t]+)")
    _POSIX_PATH_RE = re.compile(r"(/[^ \n\r\t]+)")

    @staticmethod
    def _shorten_path(p: str) -> str:
        p = p.strip().strip('"').strip("'")
        if not p:
            return p
        p2 = p.replace("/", "\\")
        base = os.path.basename(p2)
        return base if base else p

    @classmethod
    def _denoise_visual_context(cls, visual_context: str) -> str:
        if not visual_context:
            return ""

        out_lines: List[str] = []
        for line in visual_context.splitlines():
            s = line.strip()
            if not s:
                out_lines.append("")
                continue

            if s.startswith("[Image ") and "]" in s:
                if "] " in s:
                    prefix, rest = s.split("] ", 1)
                    rest = rest.strip()
                    if rest:
                        rest = cls._shorten_path(rest)
                        out_lines.append(f"{prefix}] {rest}")
                    else:
                        out_lines.append(f"{prefix}]")
                else:
                    out_lines.append(s)
                continue

            s = cls._WIN_PATH_RE.sub(lambda m: cls._shorten_path(m.group(1)), s)
            s = cls._UNC_PATH_RE.sub(lambda m: cls._shorten_path(m.group(1)), s)
            s = cls._POSIX_PATH_RE.sub(lambda m: cls._shorten_path(m.group(1)), s)

            out_lines.append(s)

        return "\n".join(out_lines).strip()

    @staticmethod
    def build_image_summary(object_label: Optional[str], is_anomaly: bool) -> str:
        obj = object_label or "object"
        defect_word = "defective"
        if is_anomaly:
            return f"This {obj} is {defect_word}."
        return f"This {obj} is not {defect_word}."

    @classmethod
    def build_visual_context(
        cls,
        image_paths: List[str],
        labels: List[str],
        anomaly_flags: List[bool],
    ) -> str:
        if not image_paths:
            return ""

        lines: List[str] = []
        for i, (path, label, anomaly) in enumerate(zip(image_paths, labels, anomaly_flags), start=1):
            summary = cls.build_image_summary(label, anomaly)
            lines.append(f"[Image {i}] {path}")
            lines.append(summary)
            lines.append("")

        return cls._denoise_visual_context("\n".join(lines).strip())

    @classmethod
    def build_augmented_query(
        cls,
        user_query: str,
        visual_context: str,
    ) -> str:
        uq = (user_query or "").strip()
        vc = cls._denoise_visual_context(visual_context or "")
        if not vc.strip():
            return uq

        print("visual_context: ", vc)
        return (
            f"{cls.VISUAL_HEADING}\n"
            f"{vc}\n\n"
            f"Answer this question\n"
            f"{uq}\n\n"
        )
