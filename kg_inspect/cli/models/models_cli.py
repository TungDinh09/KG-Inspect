
# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn

from kg_inspect.rag.config import configure_logging
configure_logging()

from kg_inspect.pipeline_inspect import InspectionPipeline
from kg_inspect.rag.rag_manager import initialize_rag


console = Console()



def _run_async(coro):
    return asyncio.run(coro)



@click.group(help="Quản lý & chạy các mô hình inspection (CNN + RAG/VLM) trong KG-Inspect.")
def models():
    pass



@models.command("devices", help="Xem thiết bị hiện tại (DEVICE env) và gợi ý đặt CUDA/CPU.")
def devices_cmd():
    device = os.getenv("DEVICE", "cpu")
    table = Table(title="Runtime Device")
    table.add_column("ENV", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("DEVICE", device)
    console.print(table)

    console.print(
        Panel(
            "Đổi thiết bị tạm thời cho lệnh bằng: [bold]DEVICE=cuda models run ...[/bold]\n"
            "Hoặc đặt ENV toàn cục:\n"
            "  - PowerShell: [bold]$Env:DEVICE='cuda'[/bold]\n"
            "  - Bash:      [bold]export DEVICE=cuda[/bold]",
            title="Gợi ý",
            border_style="blue",
        )
    )



@models.command(
    "run",
    help=(
        "Chạy toàn bộ InspectionPipeline:\n"
        "  - Nếu KHÔNG truyền --images: Text-only RAG (LightRAG + VLM)\n"
        "  - Nếu CÓ --images: CNNInspect (ConvNeXt + CutPaste) + RAG/VLM"
    ),
)
@click.option(
    "--query",
    "-q",
    required=True,
    help="Câu truy vấn văn bản của người dùng.",
)
@click.option(
    "--images",
    "-i",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Đường dẫn tới ảnh (có thể truyền nhiều lần). Nếu không truyền → text-only RAG.",
)
@click.option(
    "--mode",
    default="hybrid",
    show_default=True,
    help="Chế độ truy vấn LightRAG (ví dụ: hybrid, local, global, dense ... tùy bạn cấu hình trong QueryParam).",
)
@click.option(
    "--system-prompt",
    type=str,
    default=None,
    help="System prompt cho VLM/RAG (ưu tiên nếu dùng cùng --system-prompt-file).",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Đọc system prompt từ file (UTF-8). Nếu truyền cả --system-prompt thì file sẽ được ưu tiên.",
)
def run_cmd(
    query: str,
    images: List[str],  # click sẽ inject tuple, nhưng ta xử lý như list
    mode: str,
    system_prompt: Optional[str],
    system_prompt_file: Optional[str],
):
    """
    Điểm vào chính cho pipeline:
      - Khởi tạo LightRAG (KGInspect) qua initialize_rag()
      - Bọc vào InspectionPipeline
      - Chạy pipeline.run(...)
      - In kết quả JSON thô để bạn tự parse/hiển thị thêm ở tầng trên nếu muốn.

    Nếu không truyền --images thì pipeline sẽ chạy nhánh text-only RAG.
    """

    image_paths = list(images) if images else None


    final_system_prompt: Optional[str] = system_prompt
    if system_prompt_file is not None:
        try:
            with open(system_prompt_file, "r", encoding="utf-8") as f:
                final_system_prompt = f.read()
        except Exception as e:
            console.print(
                Panel(
                    f"Không thể đọc system prompt từ file: {e}",
                    border_style="red",
                    title="Lỗi đọc file system prompt",
                )
            )
            sys.exit(1)

    async def _job():
        console.print(
            Rule("[bold cyan]INSPECTION PIPELINE RUN[/bold cyan]")
        )

        try:
            console.print(
                Panel(
                    "Initializing KGInspect (LightRAG + VLM + VectorDB + KG)...",
                    title="⚙️ INIT",
                    border_style="blue",
                )
            )
            rag = await initialize_rag()
            console.print(
                Panel(
                    "KGInspect instance is ready.",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(
                Panel(
                    f"Không thể khởi tạo RAG/KGInspect: {e}",
                    border_style="red",
                    title="Lỗi khởi tạo",
                )
            )
            return


        pipeline = InspectionPipeline(rag=rag)

        
        try:
            result: Dict[str, Any] = await pipeline.run(
                user_query=query,
                images=image_paths,          # None hoặc [] → text-only; list path → multimodal
                system_prompt=final_system_prompt,
                mode=mode,
                # query_param: để None cho pipeline tự tạo QueryParam(mode=mode, ...)
            )

           
            console.print(
                Panel(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    title="Pipeline Result",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(
                Panel(
                    f"Lỗi khi chạy InspectionPipeline: {e}",
                    border_style="red",
                    title="Pipeline Error",
                )
            )
        finally:
            # 4) Shutdown storages
            console.print(
                Panel(
                    "Shutting down LightRAG storages...",
                    border_style="yellow",
                )
            )
            try:
                await rag.finalize_storages()
                console.print(
                    Panel(
                        "System shutdown complete.",
                        border_style="green",
                    )
                )
            except Exception as e:
                console.print(
                    Panel(
                        f"Shutdown warning: {e}",
                        border_style="yellow",
                    )
                )

        console.print(
            Rule("[bold cyan]PIPELINE RUN COMPLETE[/bold cyan]")
        )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as p:
            p.add_task(
                description="Đang chạy InspectionPipeline (text-only hoặc CNN + RAG/VLM)...",
                total=None,
            )
            _run_async(_job())
    except Exception as e:
        console.print(
            Panel(
                f"Đã xảy ra lỗi: {e}",
                border_style="red",
            )
        )
        sys.exit(1)

