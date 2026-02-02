# kg_inspect/gradio/services/pipeline_service.py
from __future__ import annotations

import os
import json
import traceback
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel

from lightrag.base import QueryParam
from kg_inspect.pipeline_inspect import InspectionPipeline
from kg_inspect.rag.rag_manager import initialize_rag

console = Console()

_PIPELINE: Optional[InspectionPipeline] = None


async def _get_pipeline() -> InspectionPipeline:
    """
    Khởi tạo (nếu chưa) và trả về InspectionPipeline dùng chung.
    Không finalize storages sau mỗi request để server chạy nhanh hơn.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    console.print(
        Panel(
            "Initializing KGInspect (LightRAG + VLM + VectorDB + KG)...",
            title="⚙️ INIT",
            border_style="blue",
        )
    )
    rag = await initialize_rag()
    _PIPELINE = InspectionPipeline(rag=rag)
    console.print(
        Panel(
            "KGInspect / InspectionPipeline is ready.",
            title="✅ INIT DONE",
            border_style="green",
        )
    )
    return _PIPELINE


async def run_pipeline(
    image_paths: List[str],
    user_query: str,
    history: List[Dict[str, str]],
    mode: str = "hybrid",
    enable_lightrag: bool = True,
) -> tuple[str, str, str]:
    
    try:
        pipeline = await _get_pipeline()

        allowed_modes = {"local", "global", "hybrid", "naive", "mix", "bypass"}
        if mode not in allowed_modes:
            console.print(
                Panel(
                    f"Mode không hợp lệ: '{mode}', fallback về 'hybrid'.",
                    border_style="yellow",
                    title="Mode Warning",
                )
            )
            effective_mode = "hybrid"
        else:
            effective_mode = mode

        if not enable_lightrag and effective_mode != "bypass":
            console.print(
                Panel(
                    "enable_lightrag=False → chuyển mode sang 'bypass' (tạm thời).",
                    border_style="yellow",
                    title="LightRAG Disabled",
                )
            )
            effective_mode = "bypass"


        param = QueryParam(
            mode=effective_mode,
            stream=False,
            enable_rerank=False,
            conversation_history=history or [],
        )



        result: Any = await pipeline.run(
            user_query=user_query,
            images=image_paths or None,
            system_prompt=None,
            mode=effective_mode,
            query_param=param,
        )

        status_str = (
            f"✅ Pipeline chạy thành công (mode={effective_mode}, "
        )

        answer_str = ""
        debug_str = ""

        if isinstance(result, dict):
            answer_candidate = (
                result.get("answer")
                or result.get("response")
                or result.get("llm_response")
                or result.get("output")
            )

            if isinstance(answer_candidate, dict):
                answer_str = (
                    answer_candidate.get("content")
                    or answer_candidate.get("text")
                    or json.dumps(answer_candidate, ensure_ascii=False, indent=2)
                )
            elif isinstance(answer_candidate, str):
                answer_str = answer_candidate
            else:
                answer_str = json.dumps(result, ensure_ascii=False, indent=2)

            debug_str = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            answer_str = str(result)
            debug_str = str(result)

        return status_str, answer_str, debug_str

    except Exception as e:
        tb = traceback.format_exc()
        status_str = f"❌ Lỗi khi chạy pipeline: {e}"
        debug_str = tb

        console.print(
            Panel(
                tb,
                border_style="red",
                title="Pipeline Error",
            )
        )
        return status_str, "", debug_str

async def clear_rag_cache() -> tuple[str, str]:
    """
    Clear ALL LightRAG cache from the running shared rag instance.

    Returns:
        (status_str, debug_str)
    """
    try:
        rag = await initialize_rag()

        await rag.aclear_cache()
        return "✅ Cleared ALL caches.", json.dumps({"cleared": "all"}, ensure_ascii=False)

    except Exception as e:
        tb = traceback.format_exc()
        console.print(Panel(tb, border_style="red", title="Clear Cache Error"))
        return f"❌ Clear cache failed: {e}", tb