"""Command for showing tweaktune installation information."""

import platform
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

import tweaktune

console = Console() if Console else None


def show_info(system: bool):
    """Show tweaktune installation information."""
    if console:
        console.print()
        console.print("[bold cyan]Tweaktune Information[/bold cyan]")
        console.print()

        # Version info table
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Version", tweaktune.__version__)
        table.add_row("Installation Path", str(Path(tweaktune.__file__).parent))
        table.add_row(
            "Python Version",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )

        console.print(table)
        console.print()

        if system:
            console.print("[bold]System Information[/bold]")
            console.print()

            sys_table = Table(show_header=False, box=None)
            sys_table.add_column("Key", style="bold")
            sys_table.add_column("Value")

            sys_table.add_row("Platform", platform.platform())
            sys_table.add_row("Processor", platform.processor() or "Unknown")
            sys_table.add_row("Python Implementation", platform.python_implementation())
            sys_table.add_row("Python Compiler", platform.python_compiler())

            console.print(sys_table)
            console.print()

        # Optional dependencies
        optional_deps = {
            "click": "CLI support",
            "rich": "Rich terminal output",
            "nicegui": "Web UI",
            "questionary": "Interactive prompts",
        }

        console.print("[bold]Optional Dependencies[/bold]")
        console.print()

        dep_table = Table(show_header=True)
        dep_table.add_column("Package", style="cyan")
        dep_table.add_column("Status")
        dep_table.add_column("Purpose")

        for package, purpose in optional_deps.items():
            try:
                __import__(package)
                status = "[green]✓ Installed[/green]"
            except ImportError:
                status = "[dim]✗ Not installed[/dim]"
            dep_table.add_row(package, status, purpose)

        console.print(dep_table)
        console.print()

    else:
        print("\nTweaktune Information\n")
        print(f"Version: {tweaktune.__version__}")
        print(f"Installation Path: {Path(tweaktune.__file__).parent}")
        print(
            f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        if system:
            print("\nSystem Information\n")
            print(f"Platform: {platform.platform()}")
            print(f"Processor: {platform.processor() or 'Unknown'}")
            print(f"Python Implementation: {platform.python_implementation()}")
            print(f"Python Compiler: {platform.python_compiler()}")

        print("\nOptional Dependencies\n")
        optional_deps = {
            "click": "CLI support",
            "rich": "Rich terminal output",
            "nicegui": "Web UI",
            "questionary": "Interactive prompts",
        }

        for package, purpose in optional_deps.items():
            try:
                __import__(package)
                status = "✓ Installed"
            except ImportError:
                status = "✗ Not installed"
            print(f"  {package:20} {status:20} {purpose}")

        print()
