# kg_inspect/pipeline/pipeline_inspect.py

from __future__ import annotations

from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel

from lightrag.base import QueryParam

from kg_inspect.kg_inspect import KGInspect
from kg_inspect.cnn_inspect import CNNInspect
from kg_inspect.utils.prompt import Prompt
console = Console()


class InspectionPipeline:
    """
    Orchestrator kết hợp:
      - CNNInspect (ConvNeXt + CutPaste) cho ảnh
      - KGInspect (LightRAG + VLM) cho RAG + reasoning

    Luồng:
      - Nếu không có ảnh:
          → Gọi thẳng rag.aquery(user_query, ...)
      - Nếu có ảnh:
          → Mỗi ảnh đi qua CNNInspect
          → Prompt sinh thêm text "This {object} is (not) defective."
          → Ghép vào câu hỏi người dùng (augmented_query)
          → Gọi rag.aquery(augmented_query, images=...) để VLM xử lý
    """

    def __init__(
        self,
        rag: KGInspect,
        cnn: Optional[CNNInspect] = None,
    ) -> None:
        self.rag = rag
        self.cnn = cnn or CNNInspect()

    
    async def _inspect_images(
        self, image_paths: List[str]
    ) -> Dict[str, List[Any]]:
        """
        Chạy CNNInspect cho từng ảnh và gom kết quả thành các list song song.

        Returns:
            {
                "paths": [...],
                "labels": [...],
                "anomaly_flags": [...],
                "confidences": [...],
                "scores": [...],
                "thresholds": [...],
            }
        """
        labels: List[str] = []
        anomaly_flags: List[bool] = []
        confidences: List[float] = []
        scores: List[Optional[float]] = []
        thresholds: List[Optional[float]] = []

        for path in image_paths:
            out = await self.cnn.run(path)
            conv = out.get("convnext", {}) or {}
            cp = out.get("cutpaste", {}) or {}

            label = conv.get("label") or "object"
            confidence = float(conv.get("confidence", 0.0))
            is_anomaly = bool(cp.get("is_anomaly", False))
            score = cp.get("score")
            threshold = cp.get("threshold")

            labels.append(label)
            anomaly_flags.append(is_anomaly)
            confidences.append(confidence)
            scores.append(float(score) if score is not None else None)
            thresholds.append(float(threshold) if threshold is not None else None)

        return {
            "paths": image_paths,
            "labels": labels,
            "anomaly_flags": anomaly_flags,
            "confidences": confidences,
            "scores": scores,
            "thresholds": thresholds,
        }

    
    async def run(
        self,
        user_query: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        mode: str = "hybrid",
        query_param: Optional[QueryParam] = None,
    ) -> Dict[str, Any]:
        """
        Điểm vào chính của pipeline.

        Args:
            user_query: câu hỏi của người dùng (text).
            images: list đường dẫn ảnh (nếu có).
            system_prompt: system prompt cho VLM/RAG.
            mode: mode query của LightRAG (mặc định: "hybrid").
            query_param: nếu muốn custom thêm, có thể truyền; nếu None sẽ tạo mới.

        Returns:
            dict: kết quả thô từ rag.aquery (raw_data + llm_response wrapper của bạn).
        """

        if query_param is None:
            query_param = QueryParam(
                mode=mode,
                stream=False,        # bạn có thể bật True nếu muốn stream
                enable_rerank=False, # tuỳ config
            )

        # ======= Trường hợp KHÔNG có ảnh =======
        if not images:
            console.print(
                Panel(
                    "[bold cyan]No images provided.[/bold cyan] "
                    "Running pure text RAG query.",
                    title="🔍 InspectionPipeline",
                    border_style="cyan",
                )
            )
            return await self.rag.aquery(
                user_query, param=query_param, system_prompt=system_prompt
            )

        console.print(
            Panel(
                f"[bold]Images provided:[/bold] {len(images)}\n"
                f"- Will run ConvNeXt + CutPaste + RAG/VLM.",
                title="🧪 Visual + Text Inspection",
                border_style="green",
            )
        )

        inspection = await self._inspect_images(images)

        visual_context = Prompt.build_visual_context(
            image_paths=inspection["paths"],
            labels=inspection["labels"],
            anomaly_flags=inspection["anomaly_flags"],
        )

        console.print(
            Panel(
                f"[bold]Visual context:[/bold]\n{visual_context}",
                title="🧿 CNNInspect Summary",
                border_style="magenta",
            )
        )

        # Ghép câu hỏi người dùng với context từ ảnh bằng Prompt
        augmented_query = Prompt.build_augmented_query(
            user_query=user_query,
            visual_context=visual_context,
        )

        # Gọi VLM + RAG
        result = await self.rag.aquery(
            augmented_query,
            images=images,             # VLM vẫn nhìn được ảnh
            param=query_param,
            system_prompt=system_prompt,
        )

        if isinstance(result, dict):
            result.setdefault("inspection_meta", {})
            result["inspection_meta"]["images"] = images
            result["inspection_meta"]["visual_context"] = visual_context
            result["inspection_meta"]["inspection"] = inspection

        return result
