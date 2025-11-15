# Pipeline Basics

## Pipeline Architecture

A tweaktune pipeline consists of three main parts:

1. **Configuration** - Set up resources (datasets, LLMs, templates, workers)
2. **Iteration** - Define how to iterate (range or dataset)
3. **Steps** - Chain processing steps together

```python
from tweaktune import Pipeline

(Pipeline()
    # 1. Configuration
    .with_workers(5)
    .with_template("output", """{"result": {{value}}}""")

    # 2. Iteration
    .iter_range(100)

    # 3. Steps
        .add_column("value", lambda data: data["index"] * 2)
        .filter(lambda data: data["value"] > 50)
        .write_jsonl(path="output.jsonl", template="output")

    .run())
```

## Worker Configuration

Configure parallel processing with workers:

```python
pipeline = Pipeline()
pipeline.with_workers(10)  # 10 parallel workers
```

More workers = faster processing, but also more resource usage. Start with 2-5 workers and adjust based on your system.

## Pipeline Naming and Metadata

Name your pipeline and optionally track metadata:

```python
from tweaktune import Pipeline, Metadata

metadata = Metadata(path="./.tweaktune", enabled=True)

(Pipeline(name="my_pipeline", metadata=metadata)
    .with_workers(5)
    .iter_range(100)
        .write_jsonl(path="output.jsonl", value="data")
    .run())
```

With metadata enabled, tweaktune tracks:
- Pipeline runs
- Processed items
- Deduplication state (hashes, simhashes, embeddings)

## Iteration Methods

### iter_range()

Iterate a fixed number of times:

```python
# Iterate 100 times (index: 0-99)
.iter_range(100)

# Start at 10, stop at 20, step by 2 (index: 10, 12, 14, 16, 18)
.iter_range(start=10, stop=20, step=2)

# Alternative syntax
.iter_range(10, 20, 2)
```

The `index` variable is available in the context:

```python
.iter_range(10)
    .add_column("id", lambda data: f"item_{data['index']}")
```

### iter_dataset()

Iterate over dataset rows:

```python
(Pipeline()
    .with_jsonl_dataset("items", "items.jsonl")
    .iter_dataset("items")
        .add_column("processed", lambda data: True)
        .write_jsonl(path="processed.jsonl", value="items")
    .run())
```

When iterating over a dataset, the entire row is available in the context using the dataset name as the key.

## Step Chaining

Steps are executed sequentially for each iteration:

```python
.iter_range(10)
    .add_column("a", lambda data: 1)       # Step 1
    .add_column("b", lambda data: 2)       # Step 2
    .add_column("c", lambda data: data["a"] + data["b"])  # Step 3
    .write_jsonl(path="output.jsonl", value="c")  # Step 4
```

## Logging and Debugging

Control logging levels:

```python
(Pipeline()
    .iter_range(10)
        .log("debug")  # Set log level to debug
        .add_column("value", lambda data: data["index"])
        .write_jsonl(path="output.jsonl", value="value")
    .run())
```

Log levels: `debug`, `info`, `warn`, `error`

Quick debugging:

```python
.iter_range(10)
    .debug()  # Equivalent to .log("debug")
    .add_column("value", lambda data: data["index"])
```

## Print Step

Print values during execution:

```python
.iter_range(10)
    .add_column("value", lambda data: data["index"] * 2)
    .print(columns=["index", "value"])  # Print specific columns
    .write_jsonl(path="output.jsonl", value="value")
```

Or use a template:

```python
.iter_range(10)
    .add_column("value", lambda data: data["index"] * 2)
    .print(template="my_template")  # Print rendered template
```

## Error Handling

Steps can fail. Failed items are marked with `StepStatus.Failed` and typically skipped by subsequent steps.

You can check status in custom steps:

```python
from tweaktune.common import StepStatus

def my_validator(context):
    if context["data"]["value"] < 0:
        context["status"] = StepStatus.FAILED.value
    return context

.iter_range(10)
    .add_column("value", lambda data: data["index"] - 5)
    .map(my_validator)
    .write_jsonl(path="output.jsonl", value="value")  # Only writes non-failed items
```

## Pipeline Builder Pattern

The pipeline uses a builder pattern. Most methods return `self` (or `PipelineRunner`), allowing method chaining:

```python
# This
pipeline = Pipeline()
pipeline.with_workers(5)
pipeline.with_template("t1", "...")
runner = pipeline.iter_range(10)
runner.add_column("a", ...)
runner.run()

# Is equivalent to this
(Pipeline()
    .with_workers(5)
    .with_template("t1", "...")
    .iter_range(10)
        .add_column("a", ...)
    .run())
```

## Next Steps

- Learn about [Data Sources](03-data-sources.md)
- Explore [Templates](04-templates.md)
- See [Pipeline Steps](06-pipeline-steps.md) for all available steps
