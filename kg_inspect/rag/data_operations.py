import os
import json
import shutil
from typing import Optional
from pathlib import Path
from kg_inspect.kg_inspect import KGInspect
from neo4j import GraphDatabase
from . import config
from neo4j import GraphDatabase, exceptions
from rich.console import Console
from pypdf import PdfReader
import sys

console = Console()

async def insert_custom_kg(rag: KGInspect, kg_file_path: str):
    """Đọc file JSON và chèn custom knowledge graph vào KGInspect."""
    if not os.path.exists(kg_file_path):
        print(f"Error: Knowledge graph file not found at {kg_file_path}")
        return

    print(f"Inserting custom knowledge graph from: {kg_file_path}")
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        custom_kg = json.load(f)
    
    await rag.ainsert_custom_kg(custom_kg)
    print("Custom knowledge graph inserted successfully.")

def load_txt(path: Path) -> str:
    """Đọc nội dung file .txt với UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: Path) -> str:
    """
    Đọc PDF và trả về toàn bộ text bằng pypdf.
    Yêu cầu: pip install pypdf
    """
    reader = PdfReader(str(path))
    texts = []

    for page in reader.pages:
        page_text: Optional[str] = page.extract_text()
        if page_text:
            texts.append(page_text)

    return "\n\n".join(texts)


async def insert_document(rag: KGInspect, doc_file_path: str) -> bool:
    path = Path(doc_file_path)

    if not path.exists():
        console.print(f"[bold red]Lỗi: Không tìm thấy file tài liệu tại {doc_file_path}[/bold red]")
        return False

    console.print(f"Đang chèn tài liệu từ: {doc_file_path}")

    try:
        ext = path.suffix.lower()

        if ext == ".txt":
            content = load_txt(path)
        elif ext == ".pdf":
            content = load_pdf(path)
        else:
            console.print(
                f"[bold red]❌ Định dạng file '{ext}' hiện chưa được hỗ trợ (chỉ hỗ trợ .txt, .pdf).[/bold red]"
            )
            return False

        if not content.strip():
            console.print(
                "[bold yellow]⚠️ Cảnh báo: File tài liệu trống hoặc không trích được text. Bỏ qua việc chèn.[/bold yellow]"
            )
            return False

        try:
            result = await rag.ainsert(content)
        except Exception:
            console.print(
                "[bold red]❌ Lightrag ném exception khi xử lý tài liệu.[/bold red]"
            )
            console.print_exception(show_locals=False)
            return False

        if not result:
            console.print(
                "[bold red]❌ Xảy ra lỗi trong quá trình xử lý tài liệu. "
                "Vui lòng kiểm tra log LightRAG để biết chi tiết.[/bold red]"
            )
            return False

        return True

    except Exception:
        console.print(
            "[bold red]❌ Đã xảy ra một lỗi không mong muốn khi chèn tài liệu:[/bold red]"
        )
        
        return False


def _clear_neo4j_database():
    """Xóa tất cả các node và relationship trong database Neo4j."""
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
        with driver.session() as session:
            print("Connecting to Neo4j to clear data...")
            # Query để xóa tất cả các node và mối quan hệ của chúng
            query = "MATCH (n) DETACH DELETE n"
            session.run(query)
            print("Successfully cleared all data from Neo4j database.")
        driver.close()
    except Exception as e:
        print(f"An error occurred while clearing Neo4j database: {e}")

def _delete_rag_storage_files():
    """Xóa tất cả các file trong thư mục rag_storage."""
    working_dir = config.WORKING_DIR
    if os.path.exists(working_dir):
        try:
            shutil.rmtree(working_dir)
            print(f"Successfully deleted storage directory: {working_dir}")
            os.makedirs(working_dir) # Tạo lại thư mục rỗng
        except OSError as e:
            print(f"Error deleting storage directory {working_dir}: {e}")
    else:
        print(f"Storage directory not found, nothing to delete: {working_dir}")

def delete_all_data():
    """Thực hiện xóa toàn bộ dữ liệu từ Neo4j và thư mục rag_storage."""
    print("\n--- STARTING COMPLETE DATA DELETION ---")
    _clear_neo4j_database()
    _delete_rag_storage_files()
    print("--- COMPLETE DATA DELETION FINISHED ---\n")
    
def test_neo4j_connection():
    """Kiểm tra kết nối đến cơ sở dữ liệu Neo4j."""
    try:
        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
        with driver.session() as session:
            console.print("Đang kết nối tới Neo4j...")
            session.run("RETURN 1")
            console.print("[bold green]✅ Kết nối Neo4j thành công![/bold green]")
        driver.close()
        return True
    except exceptions.ServiceUnavailable:
        console.print(f"[bold red]❌ Lỗi kết nối Neo4j: Không thể kết nối tới {config.NEO4J_URI}.[/bold red]")
        return False
    except exceptions.AuthError:
        console.print(f"[bold red]❌ Lỗi xác thực Neo4j: Sai username hoặc password.[/bold red]")
        return False
    except Exception as e:
        console.print(f"[bold red]❌ Đã xảy ra lỗi không xác định khi kết nối Neo4j:[/bold red]")
        console.print_exception(show_locals=False)
        return False