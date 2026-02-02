import asyncio
from functools import wraps
import os
import sys
import click

from rich.console import Console
from rich.panel import Panel
from kg_inspect.rag.rag_manager import initialize_rag
from kg_inspect.rag.data_operations import (
    insert_custom_kg,
    insert_document,
    delete_all_data,
    test_neo4j_connection
)
from lightrag.utils import logger
from kg_inspect.rag.query_handler import handle_query
import asyncio


console = Console()

def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@click.group()
def lightrag():
    """
    Tương tác và quản lý LightRAG Knowledge Graph.
    """
    pass

@lightrag.command('insert-custom-kg')
@click.argument('filepath', type=click.Path(exists=True, readable=True))
@coro
async def insert_kg_command(filepath):
    """Chèn một custom Knowledge Graph từ file JSON."""
    rag = None
    try:
        rag = await initialize_rag()
        await insert_custom_kg(rag, filepath)
        console.print(f"[bold green]✅ Đã chèn thành công knowledge graph từ '{filepath}'[/bold green]")
    finally:
        if rag:
            await rag.finalize_storages()



@lightrag.command("insert-doc")
@click.argument("filepath", type=click.Path(exists=True, readable=True))
@coro
async def insert_doc_command(filepath: str):
    """
    Chèn một tài liệu (TXT/PDF) vào LightRAG.
    - Bên trong sẽ tự đọc file, trích text, rồi gọi rag.ainsert(...)
    """
    rag = None
    console.print(f"[bold cyan]--- Bắt đầu quá trình chèn tài liệu ---[/bold cyan]")
    console.print(f"📄 File: [bold]{filepath}[/bold]")

    try:
        rag = await initialize_rag()

        success = await insert_document(rag, filepath)

        if success:
            console.print(
                f"[bold green]✅ Đã gửi tài liệu '{filepath}' vào pipeline xử lý của LightRAG.[/bold green]"
            )
            console.print(
                "[green]➡ Nếu cần kiểm tra chi tiết trạng thái embedding/KG, hãy xem thêm log trong file lightrag_app.log.[/green]"
            )
        else:
            console.print(
                f"[bold red]❌ Thao tác chèn tài liệu từ '{filepath}' đã không hoàn thành thành công.[/bold red]"
            )
            console.print(
                "[red]⚠ Vui lòng xem lại log ở bên trên hoặc trong file lightrag_app.log để biết chi tiết lỗi.[/red]"
            )

    except Exception:
        console.print(
            "[bold red]💥 Lỗi nghiêm trọng xảy ra trong lệnh insert-doc:[/bold red]"
        )
        console.print_exception(show_locals=False)

    finally:
        if rag is not None:
            console.print("🔻 Đang đóng các kết nối và lưu trữ (finalize storages)...")
            try:
                await rag.finalize_storages()
            except Exception:
                console.print("[bold red]⚠ Lỗi khi finalize storages:[/bold red]")
                console.print_exception(show_locals=False)

        console.print("[bold cyan]--- Kết thúc quá trình insert-doc ---[/bold cyan]")

        os._exit(0)



@lightrag.command('query')
@click.argument('text')
@click.option(
    '--mode',
    type=click.Choice(['naive', 'local', 'global', 'hybrid'], case_sensitive=False),
    default='hybrid',
    help='Chế độ truy vấn RAG.'
)
@coro
async def query_command(text, mode):
    """Truy vấn RAG với một câu hỏi và nhận lại câu trả lời."""
    rag = None
    if not text:
        console.print("[bold red]Lỗi: Cần cung cấp văn bản để truy vấn.[/bold red]")
        return
    try:
        rag = await initialize_rag()
        await handle_query(rag, text, mode=mode)
    finally:
        if rag:
            await rag.finalize_storages()

@lightrag.command('test-connection-kg')
def test_connection_command():
    """Kiểm tra kết nối đến cơ sở dữ liệu Neo4j."""
    test_neo4j_connection()

@lightrag.command('delete')
def delete_command():
    """Xóa TOÀN BỘ dữ liệu trong rag_storage và Neo4j."""
    # ... (code xác nhận và xóa như cũ)
    console.print(
        Panel(
            "Hành động này sẽ [underline]xóa vĩnh viễn[/underline] tất cả dữ liệu trong Neo4j và thư mục rag_storage.",
            title="[bold yellow]CẢNH BÁO[/bold yellow]",
            border_style="yellow"
        )
    )
    if click.confirm(click.style("Bạn có chắc chắn muốn tiếp tục không?", fg='red', bold=True)):
        delete_all_data()
        console.print("[bold green]✅ Đã xóa xong toàn bộ dữ liệu.[/bold green]")
    else:
        console.print("[bold yellow]ℹ️ Hành động xóa đã bị hủy.[/bold yellow]")
        

@lightrag.command("clear-cache")
@click.option(
    "--sync",
    is_flag=True,
    default=False,
    help="Use synchronous cache clearing (rag.clear_cache) instead of async (rag.aclear_cache).",
)
@coro
async def clear_cache_command(sync):
    """
    Clear ALL LightRAG caches.
    """
    rag = None
    try:
        rag = await initialize_rag()

        console.print(
            Panel(
                "Clearing ALL caches.",
                title="🧹 CLEAR CACHE",
                border_style="cyan",
            )
        )

        if sync:
            rag.clear_cache()
        else:
            await rag.aclear_cache()

        console.print("[bold green]✅ All caches cleared.[/bold green]")

    except Exception:
        console.print("[bold red]💥 Error while clearing cache:[/bold red]")
        console.print_exception(show_locals=False)
        raise
    finally:
        if rag:
            await rag.finalize_storages()
