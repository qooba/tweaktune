# Validation & Quality

Ensure data quality with validation and deduplication steps.

## JSON Schema Validation

Validate JSON against a schema:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 120}
    },
    "required": ["name", "age"]
}

.add_column("schema", lambda data: schema)
.add_column("instance", lambda data: {"name": "Alice", "age": 25})
.validate_json(schema="schema", instance="instance")
```

## Tool/Function Validation

Validate function calling format:

```python
.validate_tools(instances="tool_calls")
```

Ensures tool calls have correct structure:
```json
{
  "name": "function_name",
  "parameters": {...}
}
```

## Conversation Validation

Validate conversation format:

```python
.validate_conversation(instances="messages")
```

Checks for proper message structure:
```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

## Custom Validation

Use Python functions for custom validation:

```python
def validate_email(context):
    data = context["data"]
    email = data.get("email", "")

    if "@" not in email or "." not in email:
        raise ValueError("Invalid email format")

    return True

.validate(validate_email)
```

## Language Detection

Filter by detected language:

```python
.check_language(
    input="text",
    language="english",
    precision=0.9,
    detect_languages=["english", "french", "german", "spanish", "polish"]
)
```

Parameters:
- `input` - Text field to check
- `language` - Expected language
- `precision` - Confidence threshold (0-1)
- `detect_languages` - Languages to consider

Items with wrong language are marked as failed.

## Deduplication

tweaktune provides three deduplication methods. All require metadata to be enabled:

```python
from tweaktune import Metadata

metadata = Metadata(path="./.tweaktune", enabled=True)
pipeline = Pipeline(metadata=metadata)
```

### Exact Hash

Deduplicate by exact match:

```python
.check_hash(input="content")
```

Use for:
- Exact duplicate detection
- Ensuring unique tool calls
- Preventing repeated API calls

### SimHash (Fuzzy)

Deduplicate similar content:

```python
.check_simhash(
    input="text",
    threshold=3  # Hamming distance
)
```

Parameters:
- `threshold` - Lower = stricter matching (0-64)
  - 0-3: Very similar
  - 4-10: Similar
  - 11+: Different

Use for:
- Near-duplicate text detection
- Similar question filtering

### Semantic Embeddings

Deduplicate by semantic meaning:

```python
# First, configure embeddings model
.with_embeddings_e5(
    name="e5-small",
    model_repo="intfloat/e5-small"
)

# Then use for deduplication
.check_embedding(
    input="text",
    embedding="e5-small",
    threshold=0.01,  # Cosine distance
    similarity_output="similarity"  # Optional
)
```

Parameters:
- `threshold` - Lower = stricter matching
  - 0.0-0.1: Very similar
  - 0.1-0.3: Similar
  - 0.3+: Different

Use for:
- Semantic duplicate detection
- Paraphrase detection
- Topic clustering

### Embeddings Configuration

OpenAI embeddings:

```python
.with_embeddings_api(
    name="openai-embed",
    base_url="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small"
)
```

Local E5 embeddings:

```python
.with_embeddings_e5(
    name="e5-small",
    model_repo="intfloat/e5-small"
)
```

## Deduplication Example

Complete example with all deduplication methods:

```python
from tweaktune import Pipeline, Metadata

metadata = Metadata(path="./.tweaktune", enabled=True)

(Pipeline(name="dedup_example", metadata=metadata)
    .with_workers(5)
    .with_embeddings_e5("e5-small", "intfloat/e5-small")
    .with_template("output", """{"question": {{question|jstr}}}""")
    .iter_range(1000)
        .add_column("question", lambda data: generate_question(data["index"]))

        # Exact deduplication
        .check_hash(input="question")

        # Fuzzy deduplication (catch minor variations)
        .check_simhash(input="question", threshold=3)

        # Semantic deduplication (catch paraphrases)
        .check_embedding(
            input="question",
            embedding="e5-small",
            threshold=0.05
        )

        .write_jsonl(path="unique_questions.jsonl", template="output")
    .run())
```

## Tool Normalization

Normalize tool/function schemas:

```python
.normalize_tools(
    instances="raw_tools",
    output="normalized_tools"
)
```

Converts various tool formats to a standard structure.

## Quality Filtering

Combine validation steps for quality filtering:

```python
.iter_range(1000)
    # Generate content
    .generate_text(template="prompt", llm="gpt4", output="text")

    # Quality checks
    .check_language(
        input="text",
        language="english",
        precision=0.95,
        detect_languages=["english"]
    )
    .filter(lambda data: len(data["text"]) > 50)  # Length check
    .filter(lambda data: len(data["text"]) < 500)
    .check_simhash(input="text", threshold=5)  # Dedup

    .write_jsonl(path="quality_filtered.jsonl", value="text")
```

## Metadata Tracking

When metadata is enabled, tweaktune tracks:

```python
# .tweaktune/state/state.db contains:
# - runs: Pipeline execution history
# - items: Processed items
# - hashes: Exact hash deduplication
# - simhashes: Fuzzy hash deduplication
# - embeddings: Semantic similarity vectors
```

Query the database:

```python
import sqlite3

conn = sqlite3.connect(".tweaktune/state/state.db")

# Get run history
runs = conn.execute("SELECT * FROM runs").fetchall()

# Count unique items
count = conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]

# Get all simhashes
simhashes = conn.execute("SELECT * FROM simhashes").fetchall()
```

## Best Practices

1. **Enable metadata** for deduplication to work
2. **Use exact hash** for identical content
3. **Use simhash** for near-duplicates (typos, minor edits)
4. **Use embeddings** for semantic duplicates (paraphrases)
5. **Combine methods** for comprehensive deduplication
6. **Tune thresholds** based on your data
7. **Monitor quality** by sampling outputs
8. **Filter early** in the pipeline to save processing time

## Next Steps

- Learn about [Conversation & Tools](09-conversation-tools.md)
- Explore [Metadata & Tracking](11-metadata-tracking.md)
- See [examples](/examples) for complete pipelines
