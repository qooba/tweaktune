"""Command for exploring tweaktune datasets."""

import sys
from pathlib import Path

try:
    from rich.console import Console
except ImportError:
    Console = None

console = Console() if Console else None


def explore_data(
    data_file: str,
    filter_expr: str,
    sample: int,
    search: str,
    field: str,
    format: str,
    verbose: bool,
):
    """Interactively explore a dataset."""
    data_path = Path(data_file)

    if not data_path.exists():
        if console:
            console.print(f"[bold red]Error:[/bold red] File '{data_file}' not found.")
        else:
            print(f"Error: File '{data_file}' not found.")
        sys.exit(1)

    # Only support JSONL format for now
    if format == "auto":
        if not str(data_path).endswith(".jsonl"):
            if console:
                console.print(
                    "[yellow]Warning:[/yellow] Only JSONL format is currently supported for interactive exploration."
                )
                console.print(
                    f"\n[dim]For now, use:[/dim] [cyan]tweaktune sample {data_file}[/cyan]\n"
                )
            else:
                print(
                    "Warning: Only JSONL format is currently supported for interactive exploration."
                )
                print(f"\nFor now, use: tweaktune sample {data_file}\n")
            sys.exit(1)

    # Call the Rust TUI explorer
    try:
        from tweaktune import run_explorer

        if verbose and console:
            console.print(f"\n[bold cyan]Launching Explorer[/bold cyan]: {data_path}\n")

        run_explorer(str(data_path))

    except ImportError as e:
        if console:
            console.print(f"[bold red]Error:[/bold red] Failed to import explorer: {e}")
            console.print("\n[dim]Make sure tweaktune is properly installed.[/dim]\n")
        else:
            print(f"Error: Failed to import explorer: {e}")
            print("\nMake sure tweaktune is properly installed.\n")
        sys.exit(1)
    except Exception as e:
        if console:
            console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            print(f"Error: {e}")
        sys.exit(1)
