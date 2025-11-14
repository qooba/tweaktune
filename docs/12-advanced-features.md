# Advanced Features

Advanced pipeline patterns and techniques.

## Conditional Branching

Use `ifelse` for complex conditional logic:

```python
from tweaktune.chain import Chain

.add_column("score", lambda data: calculate_score(data))
.ifelse(
    condition="score > 80",
    then_chain=Chain()
        .add_column("category", lambda data: "excellent")
        .add_column("priority", lambda data: 1)
        .generate_text(template="high_quality_prompt", llm="gpt4", output="response"),
    else_chain=Chain()
        .add_column("category", lambda data: "standard")
        .add_column("priority", lambda data: 5)
        .generate_text(template="standard_prompt", llm="gpt4", output="response")
)
```

Nested conditions:

```python
.ifelse(
    condition="type == 'premium'",
    then_chain=Chain()
        .add_column("discount", lambda data: 0.3)
        .ifelse(
            condition="amount > 1000",
            then_chain=Chain().add_column("extra_bonus", lambda data: 100),
            else_chain=Chain().add_column("extra_bonus", lambda data: 0)
        ),
    else_chain=Chain()
        .add_column("discount", lambda data: 0.1)
)
```

## Multi-Stage Pipelines

Chain multiple processing stages:

```python
(Pipeline()
    .with_workers(10)
    .with_llm_openai("gpt4", api_key, "gpt-4")

    # Stage 1: Generate questions
    .iter_range(1000)
        .generate_text(template="question_prompt", llm="gpt4", output="question")
        .check_language(input="question", language="english", precision=0.9)
        .check_simhash(input="question", threshold=5)
        .write_jsonl(path="questions.jsonl", value="question")
    .run())

# Stage 2: Generate answers (separate pipeline)
(Pipeline()
    .with_workers(10)
    .with_llm_openai("gpt4", api_key, "gpt-4")
    .with_jsonl_dataset("questions", "questions.jsonl")

    .iter_dataset("questions")
        .generate_text(
            template="Given question: {{questions.question}}, provide answer",
            llm="gpt4",
            output="answer"
        )
        .add_column("qa_pair", lambda data: {
            "question": data["questions"]["question"],
            "answer": data["answer"]
        })
        .write_jsonl(path="qa_pairs.jsonl", value="qa_pair")
    .run())
```

## Mixed Dataset Processing

Combine multiple data sources:

```python
.with_jsonl_dataset("questions", "questions.jsonl")
.with_jsonl_dataset("contexts", "contexts.jsonl")
.with_tools_dataset("tools", [tool1, tool2, tool3])
.with_mixed_dataset("combined", ["questions", "contexts", "tools"])

.iter_dataset("combined")
    .add_column("question", lambda data: data["combined"]["questions"]["text"])
    .add_column("context", lambda data: data["combined"]["contexts"]["text"])
    .add_column("tools", lambda data: data["combined"]["tools"])
    # Process combined data...
```

## Dynamic Template Selection

Choose templates based on data:

```python
.with_template("template_a", "Template A: {{value}}")
.with_template("template_b", "Template B: {{value}}")

.add_column("template_name", lambda data:
    "template_a" if data["type"] == "A" else "template_b"
)

# Can't directly select template, but can render both and choose:
.render("template_a", output="result_a")
.render("template_b", output="result_b")
.add_column("result", lambda data:
    data["result_a"] if data["type"] == "A" else data["result_b"]
)
```

## Multiple LLM Calls

Generate multiple outputs:

```python
.with_llm_openai("gpt4", api_key, "gpt-4")
.with_template("question_gen", "Generate question about: {{topic}}")
.with_template("answer_gen", "Generate answer for: {{topic}}")

.iter_range(100)
    .add_column("topic", lambda data: f"Topic {data['index']}")

    .generate_text(template="question_gen", llm="gpt4", output="question")
    .generate_text(template="answer_gen", llm="gpt4", output="answer")

    # Both available now
    .add_column("qa", lambda data: {
        "q": data["question"],
        "a": data["answer"]
    })
```

## Chunk Processing

Split long text into chunks:

```python
.add_column("long_text", lambda data: load_long_document(data["index"]))

.chunk(
    capacity=(500, 1000),  # Min 500, max 1000 chars
    input="long_text",
    output="chunks"
)

# chunks is now a list of text chunks
.add_column("chunk_count", lambda data: len(data["chunks"]))
```

Process each chunk (requires custom step):

```python
class ChunkProcessor:
    def __init__(self, llm_name):
        self.llm_name = llm_name

    def process(self, context):
        chunks = context["data"].get("chunks", [])
        summaries = []

        for chunk in chunks:
            # Process each chunk (simplified - normally would use LLM API)
            summary = f"Summary of: {chunk[:50]}..."
            summaries.append(summary)

        context["data"]["summaries"] = summaries
        return context

.step(ChunkProcessor("gpt4"))
```

## Web UI Mode

Run pipeline with interactive web interface:

```python
# Requires: pip install nicegui

(Pipeline()
    .with_workers(5)
    .iter_range(100)
        .add_column("value", lambda data: data["index"])
        .write_jsonl(path="output.jsonl", value="value")
    .ui(host="0.0.0.0", port=8080)  # Instead of .run()
)

# Open http://localhost:8080 to monitor pipeline
```

## Expression-Based Transformations

Use string expressions instead of lambdas:

```python
# Math operations
.add_column("total", "price * quantity", is_json=False)
.add_column("discounted", "total * 0.9", is_json=False)

# String operations
.add_column("full_name", "first_name + ' ' + last_name")

# Conditional expressions
.filter("age >= 18 and status == 'active'")

# Object access
.add_column("city", "address.city")
.add_column("first_item", "items[0].name")
```

Expressions are evaluated by the Rust engine for performance.

## SQL Transformations

Use SQL on datasets:

```python
.with_parquet_dataset(
    "sales",
    "sales.parquet",
    sql="""
        SELECT
            product_name,
            SUM(quantity) as total_quantity,
            AVG(price) as avg_price,
            COUNT(*) as num_sales
        FROM sales
        WHERE sale_date >= '2024-01-01'
        GROUP BY product_name
        HAVING total_quantity > 100
        ORDER BY total_quantity DESC
    """
)
```

SQL supports:
- Filtering (WHERE)
- Aggregation (GROUP BY, HAVING)
- Sorting (ORDER BY)
- Joins
- CTEs (WITH)

Powered by Polars SQL engine.

## Multiple Workers

Optimize worker count for your workload:

```python
# CPU-bound tasks (data transformation)
.with_workers(multiprocessing.cpu_count())

# I/O-bound tasks (LLM API calls)
.with_workers(20)  # More workers than CPUs

# Memory-intensive tasks
.with_workers(2)  # Fewer workers to limit memory usage
```

## Combining Steps

Chain multiple operations efficiently:

```python
.add_columns({
    "a": lambda data: 1,
    "b": lambda data: 2,
    "c": lambda data: 3
})
.into_list(inputs=["a", "b", "c"], output="values")
```

Instead of:

```python
.add_column("a", lambda data: 1)
.add_column("b", lambda data: 2)
.add_column("c", lambda data: 3)
.into_list(inputs=["a", "b", "c"], output="values")
```

## Error Recovery

Handle failures gracefully:

```python
class RobustProcessor:
    def process(self, context):
        data = context["data"]

        try:
            # Risky operation
            result = risky_function(data)
            data["result"] = result
        except ValueError as e:
            # Specific error handling
            data["result"] = None
            data["error"] = "ValueError"
        except Exception as e:
            # General error handling
            context["status"] = StepStatus.FAILED.value
            data["error"] = str(e)

        return context

.step(RobustProcessor())
```

## Best Practices

1. **Use workers wisely** - More isn't always better
2. **Batch API calls** when possible
3. **Filter early** - Remove unwanted data early in the pipeline
4. **Use expressions** for simple transformations (faster than lambdas)
5. **Leverage SQL** for complex data transformations
6. **Monitor memory** with large datasets
7. **Chain efficiently** - Combine related operations
8. **Handle errors** - Don't let failures break the pipeline
9. **Log appropriately** - Use debug mode during development
10. **Test incrementally** - Start with small iterations, then scale

## Performance Tips

1. **Reduce dataset size** with SQL filters
2. **Use Parquet** for faster reads
3. **Enable metadata** only when needed
4. **Limit max_tokens** in LLM calls
5. **Use simhash** instead of embeddings when possible
6. **Batch write** by buffering in custom steps
7. **Profile bottlenecks** with timing logs
8. **Scale workers** based on bottleneck type

## Next Steps

- Review [examples](/examples) for real-world patterns
- Check the [GitHub repository](https://github.com/qooba/tweaktune)
- Join community discussions
