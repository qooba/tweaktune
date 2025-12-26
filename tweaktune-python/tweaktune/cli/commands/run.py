"""Command for running tweaktune pipelines."""

import sys
import time
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
except ImportError:
    Console = None
    Progress = None

console = Console() if Console else None


def run_pipeline(
    pipeline_file: str,
    workers: Optional[int],
    use_tui: bool,
    output: Optional[str],
    resume: bool,
    limit: Optional[int],
    verbose: bool
):
    """Run a tweaktune pipeline."""
    pipeline_path = Path(pipeline_file)

    if not pipeline_path.exists():
        if console:
            console.print(f"[bold red]Error:[/bold red] Pipeline file '{pipeline_file}' not found.")
        else:
            print(f"Error: Pipeline file '{pipeline_file}' not found.")
        sys.exit(1)

    if console:
        console.print(f"\n[bold cyan]Running Tweaktune Pipeline[/bold cyan]")
        console.print(f"  File: {pipeline_path}")
        if workers is not None:
            console.print(f"  Workers: {workers}")
        else:
            console.print(f"  Workers: [dim](from pipeline)[/dim]")
        if limit:
            console.print(f"  Limit: {limit} items")
        if resume:
            console.print(f"  Mode: [yellow]Resume[/yellow]")
        console.print()
    else:
        print(f"\nRunning Tweaktune Pipeline")
        print(f"  File: {pipeline_path}")
        if workers is not None:
            print(f"  Workers: {workers}")
        else:
            print(f"  Workers: (from pipeline)")
        if limit:
            print(f"  Limit: {limit} items")
        if resume:
            print(f"  Mode: Resume")
        print()

    # Load and execute the pipeline
    namespace = {}
    try:
        with open(pipeline_path) as f:
            code = compile(f.read(), pipeline_path, 'exec')
            exec(code, namespace)
    except Exception as e:
        if console:
            console.print(f"[bold red]Error loading pipeline:[/bold red] {e}")
        else:
            print(f"Error loading pipeline: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Get the pipeline object
    pipeline = namespace.get('pipeline')
    if pipeline is None:
        if console:
            console.print("[bold red]Error:[/bold red] No 'pipeline' variable found in script")
            console.print("[yellow]Make sure your script defines a pipeline:[/yellow]")
            console.print("  [dim]from tweaktune import Pipeline[/dim]")
            console.print("  [dim]pipeline = Pipeline()[/dim]")
        else:
            print("Error: No 'pipeline' variable found in script")
            print("Make sure your script defines a pipeline:")
            print("  from tweaktune import Pipeline")
            print("  pipeline = Pipeline()")
        sys.exit(1)

    # Configure workers only if explicitly provided
    if workers is not None:
        pipeline.with_workers(workers)

    # Override output if specified
    if output:
        # Note: This would require pipeline API support
        if verbose:
            if console:
                console.print(f"[dim]Output path: {output}[/dim]")
            else:
                print(f"Output path: {output}")

    # Run the pipeline
    try:
        start_time = time.time()

        if use_tui and console and Progress:
            # Rich progress bar mode
            _run_with_progress(pipeline, verbose)
        else:
            # Simple console mode
            if console:
                with console.status("[bold green]Processing pipeline...") as status:
                    pipeline.run()
            else:
                print("Processing pipeline...")
                pipeline.run()

        elapsed = time.time() - start_time

        if console:
            console.print(f"\n[bold green]✓ Pipeline completed successfully![/bold green]")
            console.print(f"  Elapsed time: {_format_duration(elapsed)}\n")
        else:
            print(f"\n✓ Pipeline completed successfully!")
            print(f"  Elapsed time: {_format_duration(elapsed)}\n")

    except KeyboardInterrupt:
        if console:
            console.print("\n\n[yellow]Pipeline interrupted by user[/yellow]")
        else:
            print("\n\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        if console:
            console.print(f"\n[bold red]Pipeline failed:[/bold red] {e}")
        else:
            print(f"\nPipeline failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _run_with_progress(pipeline, verbose: bool):
    """Run pipeline with rich progress bar."""
    # This is a placeholder for future TUI integration with Rust backend
    # For now, just run the pipeline with a simple progress indicator

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running pipeline...", total=None)

        try:
            # Note: This would integrate with pipeline progress callbacks
            # when implemented in the Rust core
            pipeline.run()
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            raise


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
