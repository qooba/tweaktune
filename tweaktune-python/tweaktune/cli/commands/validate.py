"""Command for validating tweaktune pipelines."""

import sys
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

console = Console() if Console else None


def validate_pipeline(
    pipeline_file: str,
    check_llm: bool,
    dry_run: bool,
    check_cost: bool,
    fix: bool,
    verbose: bool
):
    """Validate a pipeline before running."""
    pipeline_path = Path(pipeline_file)

    if not pipeline_path.exists():
        if console:
            console.print(f"[bold red]Error:[/bold red] Pipeline file '{pipeline_file}' not found.")
        else:
            print(f"Error: Pipeline file '{pipeline_file}' not found.")
        sys.exit(1)

    if console:
        console.print(f"\n[bold cyan]Validating Pipeline[/bold cyan]: {pipeline_path}\n")
    else:
        print(f"\nValidating Pipeline: {pipeline_path}\n")

    checks_passed = 0
    checks_failed = 0
    warnings = []

    # Check 1: Python syntax
    if console:
        console.print("[cyan]Checking Python syntax...[/cyan]", end=" ")
    else:
        print("Checking Python syntax... ", end="")

    try:
        with open(pipeline_path) as f:
            compile(f.read(), pipeline_path, 'exec')
        if console:
            console.print("[green]✓[/green]")
        else:
            print("✓")
        checks_passed += 1
    except SyntaxError as e:
        if console:
            console.print(f"[red]✗[/red]")
            console.print(f"  [red]Syntax error at line {e.lineno}: {e.msg}[/red]")
        else:
            print("✗")
            print(f"  Syntax error at line {e.lineno}: {e.msg}")
        checks_failed += 1

    # Check 2: Pipeline object exists
    if console:
        console.print("[cyan]Checking pipeline definition...[/cyan]", end=" ")
    else:
        print("Checking pipeline definition... ", end="")

    try:
        namespace = {}
        with open(pipeline_path) as f:
            exec(compile(f.read(), pipeline_path, 'exec'), namespace)

        if 'pipeline' in namespace:
            if console:
                console.print("[green]✓[/green]")
            else:
                print("✓")
            checks_passed += 1
        else:
            if console:
                console.print("[red]✗[/red]")
                console.print("  [red]No 'pipeline' variable found[/red]")
            else:
                print("✗")
                print("  No 'pipeline' variable found")
            checks_failed += 1
    except Exception as e:
        if console:
            console.print(f"[red]✗[/red] {e}")
        else:
            print(f"✗ {e}")
        checks_failed += 1

    # Check 3: LLM connections (if requested)
    if check_llm:
        if console:
            console.print("[cyan]Testing LLM connections...[/cyan]", end=" ")
        else:
            print("Testing LLM connections... ", end="")

        # Placeholder for LLM connection testing
        if console:
            console.print("[yellow]⚠[/yellow] (not implemented yet)")
        else:
            print("⚠ (not implemented yet)")
        warnings.append("LLM connection testing not yet implemented")

    # Check 4: Dry run (if requested)
    if dry_run:
        if console:
            console.print("\n[bold]Dry run (processing first 10 items)...[/bold]")
        else:
            print("\nDry run (processing first 10 items)...")

        # Placeholder for dry run
        warnings.append("Dry run not yet implemented")

    # Check 5: Cost estimation (if requested)
    if check_cost:
        if console:
            console.print("\n[bold]Estimating costs...[/bold]")
        else:
            print("\nEstimating costs...")

        # Placeholder for cost estimation
        warnings.append("Cost estimation not yet implemented")

    # Summary
    if console:
        console.print(f"\n[bold]Validation Summary:[/bold]")
        console.print(f"  [green]Passed:[/green] {checks_passed}")
        console.print(f"  [red]Failed:[/red] {checks_failed}")
        if warnings:
            console.print(f"  [yellow]Warnings:[/yellow] {len(warnings)}")
            for warning in warnings:
                console.print(f"    • {warning}")
    else:
        print(f"\nValidation Summary:")
        print(f"  Passed: {checks_passed}")
        print(f"  Failed: {checks_failed}")
        if warnings:
            print(f"  Warnings: {len(warnings)}")
            for warning in warnings:
                print(f"    - {warning}")

    if checks_failed > 0:
        sys.exit(1)
    else:
        if console:
            console.print(f"\n[bold green]✓ Pipeline is valid and ready to run![/bold green]\n")
        else:
            print(f"\n✓ Pipeline is valid and ready to run!\n")
