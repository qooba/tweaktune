"""Command for listing available templates."""

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

from tweaktune.cli.commands.new import TEMPLATES

console = Console() if Console else None


def list_templates(detailed: bool):
    """List all available project templates."""
    if console and detailed:
        console.print("\n[bold cyan]Available Tweaktune Templates[/bold cyan]\n")

        for name, info in TEMPLATES.items():
            console.print(f"[bold]{name}[/bold]")
            console.print(f"  Name: {info['name']}")
            console.print(f"  Description: {info['description']}")
            console.print("  Features:")
            for feature in info["features"]:
                console.print(f"    • {feature}")
            console.print()

        console.print("[dim]Usage:[/dim]")
        console.print("  [cyan]tweaktune new my-project --template <template-name>[/cyan]\n")

    elif console:
        table = Table(title="Available Templates", show_header=True, header_style="bold magenta")
        table.add_column("Template", style="cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Description")

        for name, info in TEMPLATES.items():
            table.add_row(name, info["name"], info["description"])

        console.print()
        console.print(table)
        console.print(
            "\n[dim]Usage:[/dim] [cyan]tweaktune new my-project --template <template-name>[/cyan]\n"
        )

    else:
        print("\nAvailable Tweaktune Templates\n")

        max_name_len = max(len(name) for name in TEMPLATES.keys())

        for name, info in TEMPLATES.items():
            print(f"  {name:{max_name_len}} - {info['description']}")

        if detailed:
            print("\nDetails:\n")
            for name, info in TEMPLATES.items():
                print(f"{name}:")
                print(f"  Name: {info['name']}")
                print(f"  Description: {info['description']}")
                print("  Features:")
                for feature in info["features"]:
                    print(f"    - {feature}")
                print()

        print("\nUsage: tweaktune new my-project --template <template-name>\n")
