#!/usr/bin/env python3
"""
Tweaktune CLI - Command-line interface for dataset synthesis.

This CLI provides commands for creating, running, validating, and exploring
tweaktune pipelines for LLM training data generation.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Error: 'click' is not installed. Install with: pip install 'tweaktune[cli]'")
    sys.exit(1)

from tweaktune import __version__


# ASCII Art Logo
LOGO = """
╔════════════════════════════════════════╗
║                                        ║
║   ████████╗██╗    ██╗███████╗ █████╗  ║
║   ╚══██╔══╝██║    ██║██╔════╝██╔══██╗ ║
║      ██║   ██║ █╗ ██║█████╗  ███████║ ║
║      ██║   ██║███╗██║██╔══╝  ██╔══██║ ║
║      ██║   ╚███╔███╔╝███████╗██║  ██║ ║
║      ╚═╝    ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝ ║
║                                        ║
║      ██╗  ██╗████████╗██╗   ██╗███╗   ██╗███████╗    ║
║      ██║ ██╔╝╚══██╔══╝██║   ██║████╗  ██║██╔════╝    ║
║      █████╔╝    ██║   ██║   ██║██╔██╗ ██║█████╗      ║
║      ██╔═██╗    ██║   ██║   ██║██║╚██╗██║██╔══╝      ║
║      ██║  ██╗   ██║   ╚██████╔╝██║ ╚████║███████╗    ║
║      ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ║
║                                        ║
║   Dataset Synthesis for LLM Training  ║
║                                        ║
╚════════════════════════════════════════╝
"""


@click.group()
@click.version_option(version=__version__, prog_name="tweaktune")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    Tweaktune - Dataset synthesis for LLM training.

    A powerful toolkit for generating synthetic training data for Large Language Models.
    Create conversational datasets, function calling examples, preference pairs, and more.

    Examples:

        # Create a new project
        tweaktune new my-project --template conversational

        # Run a pipeline with interactive TUI
        tweaktune run pipeline.py --workers 4

        # Explore generated data
        tweaktune explore output.jsonl

        # Get statistics
        tweaktune stats output.jsonl --detailed

    For more information: https://github.com/qooba/tweaktune
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


# ============================================================================
# Command: new
# ============================================================================

@cli.command()
@click.argument('project_name')
@click.option('--template', '-t',
              type=click.Choice([
                  'text-generation',
                  'conversational',
                  'function-calling',
                  'dpo-dataset',
                  'grpo-dataset',
                  'sft-training',
                  'custom'
              ]),
              default='text-generation',
              help='Project template to use')
@click.option('--interactive', '-i', is_flag=True, help='Interactive project setup')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing directory')
@click.pass_context
def new(ctx, project_name, template, interactive, force):
    """
    Create a new tweaktune project from a template.

    This command scaffolds a new project with example pipeline code,
    configuration files, and templates based on your chosen template.

    Examples:

        tweaktune new my-dataset
        tweaktune new my-dataset --template conversational
        tweaktune new my-dataset --interactive
    """
    from tweaktune.cli.commands.new import create_project
    create_project(project_name, template, interactive, force, ctx.obj['verbose'])


# ============================================================================
# Command: init
# ============================================================================

@cli.command()
@click.option('--template', '-t',
              type=click.Choice([
                  'text-generation',
                  'conversational',
                  'function-calling',
                  'dpo-dataset',
                  'grpo-dataset',
                  'sft-training',
                  'custom'
              ]),
              default='text-generation',
              help='Project template to use')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files')
@click.pass_context
def init(ctx, template, force):
    """
    Initialize a tweaktune project in the current directory.

    Similar to 'new' but creates files in the current directory
    instead of creating a new subdirectory.

    Examples:

        cd my-existing-project
        tweaktune init --template conversational
    """
    from tweaktune.cli.commands.new import init_project
    init_project(template, force, ctx.obj['verbose'])


# ============================================================================
# Command: templates
# ============================================================================

@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed template information')
def templates(detailed):
    """
    List all available project templates.

    Shows all available templates for creating new projects,
    with descriptions of what each template generates.

    Examples:

        tweaktune templates
        tweaktune templates --detailed
    """
    from tweaktune.cli.commands.templates import list_templates
    list_templates(detailed)


# ============================================================================
# Command: run
# ============================================================================

@cli.command()
@click.argument('pipeline_file', type=click.Path(exists=True))
@click.option('--workers', '-w', type=int, default=None, help='Number of parallel workers (overrides pipeline setting)')
@click.option('--tui/--no-tui', default=True, help='Use interactive TUI interface')
@click.option('--output', '-o', type=click.Path(), help='Output file path (overrides pipeline setting)')
@click.option('--resume', is_flag=True, help='Resume a previously interrupted pipeline')
@click.option('--limit', '-l', type=int, help='Limit number of items to process')
@click.option('--quiet', '-q', is_flag=True, help='Minimal output (no TUI, less logging)')
@click.pass_context
def run(ctx, pipeline_file, workers, tui, output, resume, limit, quiet):
    """
    Run a tweaktune pipeline with progress monitoring.

    Executes a pipeline script with real-time progress tracking.
    By default uses an interactive TUI to show step progress, logs,
    and statistics.

    If --workers is not specified, uses the value configured in the pipeline script.

    Examples:

        # Run with default workers from script
        tweaktune run pipeline.py

        # Override with 8 workers
        tweaktune run pipeline.py --workers 8

        # Run quietly
        tweaktune run pipeline.py --quiet --no-tui

        # Resume interrupted pipeline
        tweaktune run pipeline.py --resume

        # Process only first 100 items
        tweaktune run pipeline.py --limit 100
    """
    from tweaktune.cli.commands.run import run_pipeline
    run_pipeline(
        pipeline_file,
        workers,
        tui and not quiet,
        output,
        resume,
        limit,
        ctx.obj['verbose']
    )


# ============================================================================
# Command: validate
# ============================================================================

@cli.command()
@click.argument('pipeline_file', type=click.Path(exists=True))
@click.option('--check-llm', is_flag=True, help='Test LLM API connections')
@click.option('--dry-run', is_flag=True, help='Simulate pipeline with first 10 items')
@click.option('--check-cost', is_flag=True, help='Estimate API costs')
@click.option('--fix', is_flag=True, help='Attempt to auto-fix common issues')
@click.pass_context
def validate(ctx, pipeline_file, check_llm, dry_run, check_cost, fix):
    """
    Validate a pipeline before running.

    Performs pre-flight checks on your pipeline to catch issues
    before you start processing data. Checks syntax, templates,
    datasets, LLM connections, and more.

    Examples:

        # Basic validation
        tweaktune validate pipeline.py

        # Validate and test LLM connections
        tweaktune validate pipeline.py --check-llm

        # Dry run with first 10 items
        tweaktune validate pipeline.py --dry-run

        # Check estimated costs
        tweaktune validate pipeline.py --check-cost
    """
    from tweaktune.cli.commands.validate import validate_pipeline
    validate_pipeline(
        pipeline_file,
        check_llm,
        dry_run,
        check_cost,
        fix,
        ctx.obj['verbose']
    )


# ============================================================================
# Command: explore
# ============================================================================

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--filter', '-f', help='Filter expression (e.g., "status==completed")')
@click.option('--sample', '-n', type=int, help='Load only N random records')
@click.option('--search', '-s', help='Search text in records')
@click.option('--field', help='Focus on specific field')
@click.option('--format', type=click.Choice(['jsonl', 'parquet', 'csv', 'auto']),
              default='auto', help='Input file format')
@click.pass_context
def explore(ctx, data_file, filter, sample, search, field, format):
    """
    Interactively explore generated datasets.

    Opens an interactive TUI for browsing, filtering, and searching
    through your generated data. Navigate with keyboard, apply filters,
    and export subsets.

    Examples:

        # Explore a dataset
        tweaktune explore output.jsonl

        # Pre-filter data
        tweaktune explore output.jsonl --filter "quality_score>0.9"

        # Sample 1000 random records
        tweaktune explore output.jsonl --sample 1000

        # Search for specific text
        tweaktune explore output.jsonl --search "machine learning"

    Navigation:
        ↑/↓     - Navigate records
        f       - Apply filter
        s       - Search
        e       - Export filtered data
        q       - Quit
    """
    from tweaktune.cli.commands.explore import explore_data
    explore_data(
        data_file,
        filter,
        sample,
        search,
        field,
        format,
        ctx.obj['verbose']
    )


# ============================================================================
# Command: stats
# ============================================================================

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--detailed', '-d', is_flag=True, help='Show detailed statistics')
@click.option('--compare', type=click.Path(exists=True), help='Compare with another dataset')
@click.option('--field', '-f', multiple=True, help='Compute stats for specific fields')
@click.option('--output', '-o', type=click.Path(), help='Save statistics to file (JSON)')
@click.option('--format', type=click.Choice(['jsonl', 'parquet', 'csv', 'auto']),
              default='auto', help='Input file format')
@click.pass_context
def stats(ctx, data_file, detailed, compare, field, output, format):
    """
    Show statistics about a dataset.

    Computes and displays statistics about your generated data including
    record counts, field distributions, quality scores, token counts,
    and cost estimates.

    Examples:

        # Basic statistics
        tweaktune stats output.jsonl

        # Detailed analysis
        tweaktune stats output.jsonl --detailed

        # Compare two datasets
        tweaktune stats output.jsonl --compare baseline.jsonl

        # Stats for specific fields
        tweaktune stats output.jsonl --field quality_score --field length

        # Save to JSON
        tweaktune stats output.jsonl --output stats.json
    """
    from tweaktune.cli.commands.stats import show_statistics
    show_statistics(
        data_file,
        detailed,
        compare,
        list(field) if field else None,
        output,
        format,
        ctx.obj['verbose']
    )


# ============================================================================
# Command: convert
# ============================================================================

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--input-format', type=click.Choice(['jsonl', 'parquet', 'csv', 'arrow', 'auto']),
              default='auto', help='Input file format')
@click.option('--output-format', type=click.Choice(['jsonl', 'parquet', 'csv', 'arrow', 'hf']),
              help='Output file format (auto-detected from extension)')
@click.option('--fields', help='Comma-separated list of fields to include')
@click.option('--filter', '-f', help='Filter expression')
@click.option('--compression', type=click.Choice(['none', 'gzip', 'snappy', 'zstd']),
              help='Compression for output file')
@click.pass_context
def convert(ctx, input_file, output_file, input_format, output_format, fields, filter, compression):
    """
    Convert between dataset formats.

    Convert datasets between different file formats including JSONL,
    Parquet, CSV, Arrow, and HuggingFace datasets.

    Examples:

        # JSONL to Parquet
        tweaktune convert data.jsonl data.parquet

        # Select specific fields
        tweaktune convert data.jsonl data.csv --fields id,text,label

        # With filtering
        tweaktune convert data.jsonl filtered.parquet --filter "score>0.8"

        # To HuggingFace dataset
        tweaktune convert data.jsonl ./hf_dataset --output-format hf

        # With compression
        tweaktune convert data.jsonl data.parquet --compression zstd
    """
    from tweaktune.cli.commands.convert import convert_dataset
    convert_dataset(
        input_file,
        output_file,
        input_format,
        output_format,
        fields,
        filter,
        compression,
        ctx.obj['verbose']
    )


# ============================================================================
# Command: sample
# ============================================================================

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--number', '-n', type=int, default=10, help='Number of samples to show')
@click.option('--random', '-r', is_flag=True, help='Random sampling instead of first N')
@click.option('--filter', '-f', help='Filter expression before sampling')
@click.option('--output', '-o', type=click.Path(), help='Save samples to file')
@click.option('--pretty', '-p', is_flag=True, help='Pretty-print JSON output')
@click.option('--field', help='Show only specific field')
@click.option('--format', type=click.Choice(['jsonl', 'parquet', 'csv', 'auto']),
              default='auto', help='Input file format')
@click.pass_context
def sample(ctx, data_file, number, random, filter, output, pretty, field, format):
    """
    Sample and display records from a dataset.

    Quick way to peek at your data, either the first N records
    or a random sample.

    Examples:

        # Show first 10 records
        tweaktune sample data.jsonl

        # Show 20 random records
        tweaktune sample data.jsonl -n 20 --random

        # Filter and sample
        tweaktune sample data.jsonl -n 5 --filter "quality_score>0.9"

        # Show only conversation field
        tweaktune sample data.jsonl --field conversation --pretty

        # Save samples to file
        tweaktune sample data.jsonl -n 100 -o samples.jsonl
    """
    from tweaktune.cli.commands.sample import sample_dataset
    sample_dataset(
        data_file,
        number,
        random,
        filter,
        output,
        pretty,
        field,
        format,
        ctx.obj['verbose']
    )


# ============================================================================
# Command: serve
# ============================================================================

@cli.command()
@click.argument('pipeline_file', type=click.Path(exists=True))
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', '-p', type=int, default=8080, help='Port to bind to')
@click.option('--public', is_flag=True, help='Make server publicly accessible (0.0.0.0)')
@click.option('--reload', is_flag=True, help='Auto-reload on file changes')
@click.pass_context
def serve(ctx, pipeline_file, host, port, public, reload):
    """
    Run pipeline with web UI interface.

    Starts a web server with an interactive UI for running and
    monitoring your pipeline. Uses NiceGUI for the interface.

    Examples:

        # Start local server
        tweaktune serve pipeline.py

        # Custom port
        tweaktune serve pipeline.py --port 3000

        # Public access
        tweaktune serve pipeline.py --public

        # With auto-reload
        tweaktune serve pipeline.py --reload
    """
    from tweaktune.cli.commands.serve import serve_pipeline

    if public:
        host = '0.0.0.0'

    serve_pipeline(
        pipeline_file,
        host,
        port,
        reload,
        ctx.obj['verbose']
    )


# ============================================================================
# Command: clean
# ============================================================================

@cli.command()
@click.option('--cache', is_flag=True, help='Clean cache files')
@click.option('--logs', is_flag=True, help='Clean log files')
@click.option('--all', '-a', is_flag=True, help='Clean all temporary files')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted')
@click.pass_context
def clean(ctx, cache, logs, all, dry_run):
    """
    Clean temporary files and caches.

    Remove temporary files, caches, and logs created by tweaktune.

    Examples:

        # Clean cache files
        tweaktune clean --cache

        # Clean logs
        tweaktune clean --logs

        # Clean everything
        tweaktune clean --all

        # Dry run
        tweaktune clean --all --dry-run
    """
    from tweaktune.cli.commands.clean import clean_files
    clean_files(cache, logs, all, dry_run, ctx.obj['verbose'])


# ============================================================================
# Command: info
# ============================================================================

@cli.command()
@click.option('--system', is_flag=True, help='Show system information')
def info(system):
    """
    Show tweaktune installation information.

    Displays version, installation path, and optionally system information.

    Examples:

        tweaktune info
        tweaktune info --system
    """
    from tweaktune.cli.commands.info import show_info
    show_info(system)


# ============================================================================
# Helper function for running CLI
# ============================================================================

def main():
    """Main entry point for the CLI."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        import traceback
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
