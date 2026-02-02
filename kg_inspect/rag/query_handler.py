# kg_inspect/lightrag/query_handler.py
import json
from typing import Optional

from lightrag.base import QueryParam
from kg_inspect.kg_inspect import KGInspect


from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table

console = Console()


async def print_stream(stream):
    """Sử dụng rich.Live để in stream một cách mượt mà trong một panel."""
    response_text = Text()
    with Live(response_text, refresh_per_second=10, vertical_overflow="visible") as live:
        async for chunk in stream:
            response_text.append(chunk)
            live.update(response_text)


async def handle_query(
    rag: KGInspect,
    user_query: str,
    mode: str = "hybrid",  # luôn default là hybrid
):
    """
    Thực hiện một query bằng aquery_data với mode=hybrid,
    hiển thị rõ:
    - Input truyền vào (system_prompt, user_query, mode, QueryParam)
    - Kết quả thô trả về (status, message, metadata, data)
    - Thống kê & sample chunks / entities / relationships

    KHÔNG fallback sang mode 'naive'. Nếu lỗi (KG/LLM...), in lỗi ra và kết thúc.
    """

    # 1) In phần "header" – những gì bạn truyền vào
    console.print(
        Panel(
            f"[bold cyan]User Query:[/bold cyan] {user_query}\n"
            f"[bold cyan]Mode (requested):[/bold cyan] {mode}",
            title="🚀 Starting RAG Query",
            border_style="blue",
        )
    )

    try:
        
        param = QueryParam(
            mode=mode,
        )

        console.print(
            Panel(
                f"[bold]QueryParam input to aquery_data:[/bold]\n\n{repr(param)}",
                title="🧩 QueryParam",
                border_style="magenta",
            )
        )

       
        result = await rag.aquery_data(user_query, param=param)
        used_mode = mode  # giữ lại để show trong Summary

        # 4) In raw result (JSON đẹp) để bạn nhìn full cấu trúc
        console.print(
            Panel(
                f"[bold]Raw result from aquery_data (mode used: {used_mode}):[/bold]",
                border_style="green",
                title="📦 Raw Result",
            )
        )
        console.print_json(json.dumps(result, ensure_ascii=False, indent=2))

        # 5) Chuẩn hoá schema: hỗ trợ cả 2 kiểu trả về
        # Kiểu 1 (giống JSON bạn đang có):
        #   { "entities": [...], "relationships": [...], "chunks": [...], "metadata": {...} }
        # Kiểu 2 (một số version khác):
        #   { "status": ..., "message": ..., "data": { ... }, "metadata": {...} }
        if "data" in result:
            # Schema kiểu mới: có lớp "data"
            status = result.get("status", "unknown")
            message = result.get("message", "")
            data = result.get("data") or {}
            metadata = result.get("metadata") or {}
        else:
            # Schema kiểu bạn đang thấy: entities / relationships / chunks ở top-level
            status = result.get("status", "unknown")  # có thì in, không thì thôi
            message = result.get("message", "")
            metadata = result.get("metadata") or {}
            # data là phần còn lại trừ metadata / status / message
            data = {
                k: v
                for k, v in result.items()
                if k not in ("status", "message", "metadata")
            }

        console.print(
            Panel(
                f"[bold]Status:[/bold] {status}\n"
                f"[bold]Message:[/bold] {message}\n"
                f"[bold]Metadata keys:[/bold] {list(metadata.keys())}\n"
                f"[bold]Mode actually used:[/bold] {used_mode}",
                title="ℹ️ Summary",
                border_style="yellow",
            )
        )

        
        chunks = data.get("chunks", []) or []
        entities = data.get("entities", []) or []
        relationships = data.get("relationships", []) or []

        stats_panel = Panel(
            f"[bold]Chunks:[/bold] {len(chunks)}\n"
            f"[bold]Entities:[/bold] {len(entities)}\n"
            f"[bold]Relationships:[/bold] {len(relationships)}",
            title="📊 Data Stats",
            border_style="cyan",
        )
        console.print(stats_panel)

        # 7) In sample vài phần tử đầu để dễ đọc hơn
        def show_sample_list(items, title, fields, max_items=5):
            if not items:
                return
            table = Table(title=title, show_lines=True)
            for f in fields:
                table.add_column(f, overflow="fold")

            for item in items[:max_items]:
                row = [str(item.get(f, "")) for f in fields]
                table.add_row(*row)

            console.print(table)

        show_sample_list(
            chunks,
            title="📄 Sample Chunks",
            fields=["chunk_id", "file_path", "content"],
        )

        show_sample_list(
            entities,
            title="🧱 Sample Entities",
            fields=["entity_name", "entity_type", "description"],
        )

        show_sample_list(
            relationships,
            title="🔗 Sample Relationships",
            fields=["src_id", "tgt_id", "keywords", "description"],
        )

    except Exception:
        console.print(
            Panel(
                "An error occurred during the query process",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        console.print_exception(show_locals=True)
    finally:
        console.print("\n[dim]----------------------------------------[/dim]")