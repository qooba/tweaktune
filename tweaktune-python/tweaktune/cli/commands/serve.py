"""Command for serving pipeline with web UI."""

import sys
from pathlib import Path

try:
    from rich.console import Console
except ImportError:
    Console = None

console = Console() if Console else None


def serve_pipeline(
    pipeline_file: str,
    host: str,
    port: int,
    reload: bool,
    verbose: bool
):
    """Serve pipeline with web UI."""
    pipeline_path = Path(pipeline_file)

    if not pipeline_path.exists():
        if console:
            console.print(f"[bold red]Error:[/bold red] Pipeline file '{pipeline_file}' not found.")
        else:
            print(f"Error: Pipeline file '{pipeline_file}' not found.")
        sys.exit(1)

    # Check if nicegui is available
    try:
        import nicegui
    except ImportError:
        if console:
            console.print("[bold red]Error:[/bold red] nicegui is not installed.")
            console.print("\n[yellow]Install with:[/yellow] [cyan]pip install 'tweaktune[ui]'[/cyan]\n")
        else:
            print("Error: nicegui is not installed.")
            print("\nInstall with: pip install 'tweaktune[ui]'\n")
        sys.exit(1)

    if console:
        console.print(f"\n[bold cyan]Starting Web UI[/bold cyan]")
        console.print(f"  Pipeline: {pipeline_path}")
        console.print(f"  URL: [link]http://{host}:{port}[/link]")
        console.print(f"  Auto-reload: {'Enabled' if reload else 'Disabled'}")
        console.print()
    else:
        print(f"\nStarting Web UI")
        print(f"  Pipeline: {pipeline_path}")
        print(f"  URL: http://{host}:{port}")
        print(f"  Auto-reload: {'Enabled' if reload else 'Disabled'}")
        print()

    # Placeholder for web UI implementation
    # This would integrate with the existing NiceGUI support

    if console:
        console.print("[yellow]Web UI serving not yet fully integrated via CLI.[/yellow]")
        console.print("\n[dim]For now, you can use the web UI by running your pipeline with:[/dim]")
        console.print("  [cyan]pipeline.ui()[/cyan] [dim]in your Python script[/dim]\n")
    else:
        print("Web UI serving not yet fully integrated via CLI.")
        print("\nFor now, you can use the web UI by running your pipeline with:")
        print("  pipeline.ui() in your Python script\n")
