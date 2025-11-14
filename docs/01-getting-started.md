# Getting Started with tweaktune

## Installation

Install tweaktune using pip:

```bash
pip install tweaktune
```

## What is tweaktune?

tweaktune is a Rust-powered, Python-facing library designed to synthesize datasets for training and fine-tuning AI models, especially Language Models. It provides:

- **Pipeline-based architecture** for composing data processing steps
- **Parallel processing** with configurable worker pools
- **LLM integration** for generating synthetic data
- **Flexible data sources** (Parquet, CSV, JSONL, Arrow, databases, APIs)
- **Dynamic templating** with Jinja2
- **Quality checks** and deduplication
- **Function calling dataset generation**

## Your First Pipeline

Here's a simple example that generates 10 synthetic records:

```python
from tweaktune import Pipeline

(Pipeline()
    .with_workers(2)
    .with_template("output", """{"id": {{index}}, "random": {{"1,100"|random_range}}}""")
    .iter_range(10)
        .write_jsonl(path="output.jsonl", template="output")
    .run())
```

This pipeline:
1. Creates a pipeline with 2 parallel workers
2. Defines a Jinja template for output
3. Iterates 10 times (index 0-9)
4. Writes each iteration to a JSONL file

## Core Concepts

### Pipeline

The `Pipeline` class is the main entry point. You configure resources (datasets, LLMs, templates) on the pipeline, then start iteration.

```python
from tweaktune import Pipeline

pipeline = Pipeline()
pipeline.with_workers(5)  # Configure parallel workers
pipeline.with_template("my_template", "Hello {{name}}")  # Add template
```

### Iteration

Pipelines must start with an iteration method:

- `iter_range(n)` - Iterate n times with an `index` variable (0 to n-1)
- `iter_dataset(name)` - Iterate over each row in a dataset

```python
# Iterate 100 times
pipeline.iter_range(100)

# Iterate over dataset rows
pipeline.iter_dataset("my_dataset")
```

### Steps

After starting iteration, chain steps together. Each step processes the context:

```python
(Pipeline()
    .with_workers(1)
    .iter_range(5)
        .add_column("greeting", lambda data: "Hello World")
        .write_jsonl(path="greetings.jsonl", value="greeting")
    .run())
```

### Context

The context is a dictionary-like object that flows through the pipeline steps. Each step can read from and write to the context:

- `data` - Dictionary containing your data columns
- `index` - Current iteration index (when using `iter_range`)
- `status` - Step execution status

## Pipeline Execution

Always end your pipeline with `.run()`:

```python
(Pipeline()
    .with_workers(2)
    .iter_range(10)
        .add_column("value", lambda data: data.get("index", 0) * 2)
        .write_jsonl(path="output.jsonl", value="value")
    .run())  # Execute the pipeline
```

## Next Steps

- Learn about [Pipeline Basics](02-pipeline-basics.md)
- Explore [Data Sources](03-data-sources.md)
- Check out the [examples](/examples) directory
