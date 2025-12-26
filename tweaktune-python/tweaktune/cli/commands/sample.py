"""Command for sampling records from datasets."""

import sys
import json
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.syntax import Syntax
except ImportError:
    Console = None
    Syntax = None

console = Console() if Console else None


def sample_dataset(
    data_file: str,
    number: int,
    random: bool,
    filter_expr: Optional[str],
    output: Optional[str],
    pretty: bool,
    field: Optional[str],
    format: str,
    verbose: bool
):
    """Sample and display records from a dataset."""
    data_path = Path(data_file)

    if not data_path.exists():
        if console:
            console.print(f"[bold red]Error:[/bold red] File '{data_file}' not found.")
        else:
            print(f"Error: File '{data_file}' not found.")
        sys.exit(1)

    if console:
        console.print(f"\n[bold cyan]Sampling Dataset[/bold cyan]: {data_path}")
        console.print(f"  Number: {number}")
        console.print(f"  Method: {'Random' if random else 'First N'}")
        if filter_expr:
            console.print(f"  Filter: {filter_expr}")
        if field:
            console.print(f"  Field: {field}")
        console.print()
    else:
        print(f"\nSampling Dataset: {data_path}")
        print(f"  Number: {number}")
        print(f"  Method: {'Random' if random else 'First N'}")
        if filter_expr:
            print(f"  Filter: {filter_expr}")
        if field:
            print(f"  Field: {field}")
        print()

    # Placeholder - load actual data when implemented
    sample_records = []

    # Try to read JSONL format as fallback
    if data_path.suffix == '.jsonl' or format == 'jsonl':
        try:
            with open(data_path) as f:
                for i, line in enumerate(f):
                    if i >= number:
                        break
                    try:
                        record = json.loads(line)
                        if field and field in record:
                            sample_records.append({field: record[field]})
                        else:
                            sample_records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            if console:
                console.print(f"[red]Error reading file:[/red] {e}")
            else:
                print(f"Error reading file: {e}")
            sys.exit(1)

    if not sample_records:
        if console:
            console.print("[yellow]No records found or format not supported yet.[/yellow]")
            console.print("\n[dim]Full sampling support will include:[/dim]")
            console.print("  • Random sampling")
            console.print("  • Filtering before sampling")
            console.print("  • All formats (JSONL, Parquet, CSV)\n")
        else:
            print("No records found or format not supported yet.")
            print("\nFull sampling support will include:")
            print("  - Random sampling")
            print("  - Filtering before sampling")
            print("  - All formats (JSONL, Parquet, CSV)\n")
        return

    # Display samples
    if console and Syntax and pretty:
        for i, record in enumerate(sample_records, 1):
            console.print(f"\n[bold]Record {i}:[/bold]")
            json_str = json.dumps(record, indent=2, ensure_ascii=False)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            console.print(syntax)
    else:
        for i, record in enumerate(sample_records, 1):
            print(f"\nRecord {i}:")
            if pretty:
                print(json.dumps(record, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(record, ensure_ascii=False))

    # Save to file if requested
    if output:
        try:
            with open(output, 'w') as f:
                for record in sample_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            if console:
                console.print(f"\n[green]✓[/green] Saved {len(sample_records)} records to {output}\n")
            else:
                print(f"\n✓ Saved {len(sample_records)} records to {output}\n")
        except Exception as e:
            if console:
                console.print(f"\n[red]Error saving samples:[/red] {e}\n")
            else:
                print(f"\nError saving samples: {e}\n")
            sys.exit(1)
