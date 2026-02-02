import click
from rich.console import Console
from rich.panel import Panel

from kg_inspect.cli.lightrag.lightrag_cli import lightrag
from kg_inspect.cli.models.models_cli import models

from kg_inspect.rag.config import configure_logging
configure_logging()


from kg_inspect.cli.lightrag.lightrag_cli import lightrag


console = Console()

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """
    KG-Inspect: Một CLI tool để quản lý Knowledge Graph.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


main.add_command(lightrag)
main.add_command(models)