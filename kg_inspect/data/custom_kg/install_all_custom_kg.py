import os
import asyncio
import argparse
import sys
from rich.console import Console

try:
    from kg_inspect.rag.rag_manager import initialize_rag
    from kg_inspect.rag.data_operations import insert_custom_kg
except ImportError:
    print("IMPORT ERROR: Cannot find module 'kg_inspect'.")
    print("Please make sure you are running the command from the project root directory (where the 'kg_inspect' folder exists).")
    sys.exit(1)

console = Console()

async def install_all_custom_kgs_async(custom_kg_dir):
    """
    Scan the entire directory, initialize RAG once,
    and insert all detected Custom Knowledge Graph files.
    """
    # 1. Resolve absolute path
    abs_custom_kg_dir = os.path.abspath(custom_kg_dir)
    console.print(f"[bold]📂 Scanning directory:[/bold] {abs_custom_kg_dir}")
    
    if not os.path.exists(abs_custom_kg_dir):
        console.print(f"[bold red]❌ Directory does not exist: {abs_custom_kg_dir}[/bold red]")
        return

    json_files = []
    for folder in os.listdir(abs_custom_kg_dir):
        folder_path = os.path.join(abs_custom_kg_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("_custom_kg.json"):
                    json_files.append(os.path.join(folder_path, file))
    
    if not json_files:
        console.print("[bold yellow]⚠️ No '*_custom_kg.json' files found in subdirectories![/bold yellow]")
        return

    console.print(f"[bold blue]ℹ️ Found {len(json_files)} KG files. Initializing RAG...[/bold blue]")

    rag = None
    try:
        rag = await initialize_rag()
        console.print("[bold green]🚀 RAG initialized. Starting insertion...[/bold green]")

        # 4. Loop and insert KGs
        success_count = 0
        for filepath in json_files:
            filename = os.path.basename(filepath)
            try:
                console.print(f"[cyan]⏳ Processing: {filename}...[/cyan]")
                
                await insert_custom_kg(rag, filepath)
                
                console.print(f"[bold green]✅ Success: {filename}[/bold green]")
                success_count += 1
            
            except Exception as e:
                console.print(f"[bold red]❌ Failed to insert {filename}: {e}[/bold red]")

        console.print(
            f"[bold]📊 Result: {success_count}/{len(json_files)} files installed successfully.[/bold]"
        )

    except Exception as e:
        console.print(f"[bold red]🔥 Critical error during RAG initialization: {e}[/bold red]")
    
    finally:
        if rag:
            console.print("[blue]Finalizing storage connections...[/blue]")
            await rag.finalize_storages()
            console.print("[bold green]✨ Done![/bold green]")

def main():
    parser = argparse.ArgumentParser(
        description="Tool for batch installing Custom Knowledge Graphs into LightRAG."
    )
    
    parser.add_argument(
        "directory",
        nargs="?",  # Optional argument
        default="./kg_inspect/data/custom_kg",  # Default path
        help="Path to the directory containing Custom KG folders (default: ./kg_inspect/data/custom_kg)"
    )

    args = parser.parse_args()

    try:
        asyncio.run(install_all_custom_kgs_async(args.directory))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]⛔ Interrupted by user.[/bold yellow]")

if __name__ == "__main__":
    main()
