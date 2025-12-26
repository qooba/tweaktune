"""Command for creating new tweaktune projects."""

import shutil
import sys
from pathlib import Path
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table
except ImportError:
    click = None
    Console = None
    Table = None

console = Console() if Console else None

TEMPLATES = {
    "text-generation": {
        "name": "Text Generation",
        "description": "Simple text generation pipeline",
        "features": ["Basic prompts", "Single LLM call", "JSONL output"]
    },
    "conversational": {
        "name": "Conversational Dataset",
        "description": "Multi-turn conversational dataset generation",
        "features": ["Multi-turn conversations", "System messages", "Chat templates"]
    },
    "function-calling": {
        "name": "Function Calling",
        "description": "Function calling dataset generation",
        "features": ["Tool definitions", "Tool calls", "Tool responses"]
    },
    "dpo-dataset": {
        "name": "DPO Dataset",
        "description": "Direct Preference Optimization dataset",
        "features": ["Chosen/rejected pairs", "Preference ranking", "RLHF training"]
    },
    "grpo-dataset": {
        "name": "GRPO Dataset",
        "description": "Group Relative Policy Optimization dataset",
        "features": ["Group preferences", "Relative ranking", "Advanced RLHF"]
    },
    "sft-training": {
        "name": "SFT Training",
        "description": "Supervised Fine-Tuning dataset",
        "features": ["Instruction-response pairs", "Quality filtering", "Deduplication"]
    },
    "custom": {
        "name": "Custom Template",
        "description": "Blank template for custom pipelines",
        "features": ["Minimal boilerplate", "Full customization", "Example steps"]
    }
}


def get_templates_dir() -> Path:
    """Get the templates directory path."""
    package_dir = Path(__file__).parent.parent.parent
    templates_dir = package_dir / "project_templates"

    if not templates_dir.exists():
        # Fall back to repository structure during development
        templates_dir = package_dir.parent.parent / "project-templates"

    return templates_dir


def create_project(
    project_name: str,
    template: str,
    interactive: bool,
    force: bool,
    verbose: bool
):
    """Create a new tweaktune project."""
    if console:
        console.print(f"\n[bold cyan]Creating new tweaktune project:[/bold cyan] {project_name}\n")
    else:
        print(f"\nCreating new tweaktune project: {project_name}\n")

    # Interactive mode
    if interactive:
        template = _interactive_template_selection()

    # Get template directory
    templates_dir = get_templates_dir()
    template_path = templates_dir / template

    if not template_path.exists():
        if console:
            console.print(f"[bold red]Error:[/bold red] Template '{template}' not found.")
            console.print("\n[yellow]Available templates:[/yellow]")
            _print_template_list(False)
        else:
            print(f"Error: Template '{template}' not found.")
            print("\nAvailable templates:")
            for name, info in TEMPLATES.items():
                print(f"  {name:20} - {info['description']}")
        sys.exit(1)

    # Create project directory
    project_path = Path(project_name)

    if project_path.exists() and not force:
        if console:
            console.print(f"[bold red]Error:[/bold red] Directory '{project_name}' already exists.")
            console.print("[yellow]Use --force to overwrite[/yellow]")
        else:
            print(f"Error: Directory '{project_name}' already exists.")
            print("Use --force to overwrite")
        sys.exit(1)

    if project_path.exists() and force:
        shutil.rmtree(project_path)

    # Copy template
    try:
        shutil.copytree(template_path, project_path)
    except Exception as e:
        if console:
            console.print(f"[bold red]Error copying template:[/bold red] {e}")
        else:
            print(f"Error copying template: {e}")
        sys.exit(1)

    # Success message
    template_info = TEMPLATES.get(template, {})

    if console:
        console.print(f"[bold green]✓[/bold green] Project created successfully!\n")
        console.print(f"[bold]Template:[/bold] {template_info.get('name', template)}")
        console.print(f"[dim]{template_info.get('description', '')}[/dim]\n")

        console.print("[bold]Next steps:[/bold]")
        console.print(f"  [cyan]cd {project_name}[/cyan]")
        console.print(f"  [cyan]pip install -r requirements.txt[/cyan]")
        console.print(f"  [cyan]python pipeline.py[/cyan]")
        console.print("\n[bold]Or run with tweaktune CLI:[/bold]")
        console.print(f"  [cyan]tweaktune run {project_name}/pipeline.py[/cyan]\n")
    else:
        print(f"✓ Project created successfully!\n")
        print(f"Template: {template_info.get('name', template)}")
        print(f"{template_info.get('description', '')}\n")
        print("Next steps:")
        print(f"  cd {project_name}")
        print(f"  pip install -r requirements.txt")
        print(f"  python pipeline.py")
        print("\nOr run with tweaktune CLI:")
        print(f"  tweaktune run {project_name}/pipeline.py\n")


def init_project(template: str, force: bool, verbose: bool):
    """Initialize a project in the current directory."""
    current_dir = Path.cwd()

    if console:
        console.print(f"\n[bold cyan]Initializing tweaktune project in current directory[/bold cyan]\n")
    else:
        print("\nInitializing tweaktune project in current directory\n")

    templates_dir = get_templates_dir()
    template_path = templates_dir / template

    if not template_path.exists():
        if console:
            console.print(f"[bold red]Error:[/bold red] Template '{template}' not found.")
        else:
            print(f"Error: Template '{template}' not found.")
        sys.exit(1)

    # Check for existing files
    template_files = list(template_path.glob('*'))
    existing_files = [f for f in template_files if (current_dir / f.name).exists()]

    if existing_files and not force:
        if console:
            console.print("[bold yellow]Warning:[/bold yellow] The following files already exist:")
            for f in existing_files:
                console.print(f"  - {f.name}")
            console.print("\n[yellow]Use --force to overwrite[/yellow]")
        else:
            print("Warning: The following files already exist:")
            for f in existing_files:
                print(f"  - {f.name}")
            print("\nUse --force to overwrite")
        sys.exit(1)

    # Copy files
    for item in template_path.glob('*'):
        dest = current_dir / item.name
        try:
            if item.is_dir():
                if dest.exists() and force:
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        except Exception as e:
            if console:
                console.print(f"[bold red]Error copying {item.name}:[/bold red] {e}")
            else:
                print(f"Error copying {item.name}: {e}")
            sys.exit(1)

    if console:
        console.print(f"[bold green]✓[/bold green] Project initialized successfully!\n")
    else:
        print("✓ Project initialized successfully!\n")


def _interactive_template_selection() -> str:
    """Interactive template selection wizard."""
    try:
        import questionary
    except ImportError:
        print("Error: 'questionary' is required for interactive mode.")
        print("Install with: pip install 'tweaktune[cli]'")
        sys.exit(1)

    # Template selection
    choices = [
        questionary.Choice(
            title=f"{info['name']}: {info['description']}",
            value=name
        )
        for name, info in TEMPLATES.items()
    ]

    template = questionary.select(
        "What type of dataset do you want to generate?",
        choices=choices
    ).ask()

    return template


def _print_template_list(detailed: bool):
    """Print available templates."""
    if console and detailed:
        for name, info in TEMPLATES.items():
            console.print(f"\n[bold cyan]{name}[/bold cyan]")
            console.print(f"  [bold]{info['name']}[/bold]")
            console.print(f"  {info['description']}")
            console.print("  [dim]Features:[/dim]")
            for feature in info['features']:
                console.print(f"    • {feature}")
    elif console:
        table = Table(title="Available Templates", show_header=True)
        table.add_column("Template", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Description")

        for name, info in TEMPLATES.items():
            table.add_row(name, info['name'], info['description'])

        console.print(table)
    else:
        print("\nAvailable templates:\n")
        for name, info in TEMPLATES.items():
            print(f"  {name:20} - {info['description']}")
        print()
