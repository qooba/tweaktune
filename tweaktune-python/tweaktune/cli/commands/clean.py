"""Command for cleaning temporary files."""

from pathlib import Path

try:
    from rich.console import Console
except ImportError:
    Console = None

console = Console() if Console else None


def clean_files(cache: bool, logs: bool, all: bool, dry_run: bool, verbose: bool):
    """Clean temporary files and caches."""
    if not (cache or logs or all):
        if console:
            console.print("[yellow]No cleanup option specified.[/yellow]")
            console.print("\n[dim]Use one of:[/dim]")
            console.print("  [cyan]--cache[/cyan]  Clean cache files")
            console.print("  [cyan]--logs[/cyan]   Clean log files")
            console.print("  [cyan]--all[/cyan]    Clean everything\n")
        else:
            print("No cleanup option specified.")
            print("\nUse one of:")
            print("  --cache  Clean cache files")
            print("  --logs   Clean log files")
            print("  --all    Clean everything\n")
        return

    if console:
        console.print(f"\n[bold cyan]Cleaning Files[/bold cyan] {'(dry run)' if dry_run else ''}\n")
    else:
        print(f"\nCleaning Files {'(dry run)' if dry_run else ''}\n")

    files_to_clean = []

    # Cache files
    if cache or all:
        # Placeholder for cache location
        cache_dirs = [
            Path.home() / ".cache" / "tweaktune",
            Path.cwd() / ".tweaktune" / "cache",
        ]
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                files_to_clean.extend(cache_dir.rglob("*"))

    # Log files
    if logs or all:
        # Placeholder for log location
        log_files = [
            Path.cwd() / "tweaktune.log",
            Path.cwd() / ".tweaktune" / "logs",
        ]
        for log_path in log_files:
            if log_path.exists():
                if log_path.is_file():
                    files_to_clean.append(log_path)
                else:
                    files_to_clean.extend(log_path.rglob("*"))

    if not files_to_clean:
        if console:
            console.print("[yellow]No files to clean.[/yellow]\n")
        else:
            print("No files to clean.\n")
        return

    if console:
        console.print(f"Found {len(files_to_clean)} files to clean:")
        for f in files_to_clean[:10]:  # Show first 10
            console.print(f"  • {f}")
        if len(files_to_clean) > 10:
            console.print(f"  ... and {len(files_to_clean) - 10} more")
        console.print()
    else:
        print(f"Found {len(files_to_clean)} files to clean:")
        for f in files_to_clean[:10]:
            print(f"  - {f}")
        if len(files_to_clean) > 10:
            print(f"  ... and {len(files_to_clean) - 10} more")
        print()

    if dry_run:
        if console:
            console.print("[yellow]Dry run - no files deleted.[/yellow]\n")
        else:
            print("Dry run - no files deleted.\n")
        return

    # Actually delete files
    deleted = 0
    for f in files_to_clean:
        try:
            if f.is_file():
                f.unlink()
                deleted += 1
        except Exception as e:
            if verbose:
                if console:
                    console.print(f"[red]Error deleting {f}:[/red] {e}")
                else:
                    print(f"Error deleting {f}: {e}")

    if console:
        console.print(f"[green]✓[/green] Cleaned {deleted} files.\n")
    else:
        print(f"✓ Cleaned {deleted} files.\n")
