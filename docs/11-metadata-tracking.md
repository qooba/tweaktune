# Metadata & Tracking

Track pipeline execution and enable deduplication with metadata.

## Enabling Metadata

```python
from tweaktune import Pipeline, Metadata

metadata = Metadata(
    path="./.tweaktune",  # Storage directory
    enabled=True
)

pipeline = Pipeline(name="my_pipeline", metadata=metadata)
```

## What's Tracked

When metadata is enabled, tweaktune stores:

1. **Pipeline runs** - Execution history
2. **Processed items** - All items that passed through
3. **Hashes** - Exact duplicate detection
4. **SimHashes** - Fuzzy duplicate detection
5. **Embeddings** - Semantic similarity vectors

## Storage Structure

```
.tweaktune/
└── state/
    └── state.db  # SQLite database
```

## Database Schema

### runs table

```sql
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_name TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    status TEXT,
    total_items INTEGER
);
```

### items table

```sql
CREATE TABLE items (
    id TEXT PRIMARY KEY,  -- UUID
    run_id INTEGER,
    data TEXT,  -- JSON
    status TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
```

### hashes table

```sql
CREATE TABLE hashes (
    hash TEXT PRIMARY KEY,
    item_id TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(id)
);
```

### simhashes table

```sql
CREATE TABLE simhashes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    simhash INTEGER,
    item_id TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(id)
);
```

### embeddings table

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding_name TEXT,
    vector BLOB,  -- Vector as bytes
    item_id TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES items(id)
);
```

## Querying Metadata

```python
import sqlite3
import json

# Connect to database
conn = sqlite3.connect(".tweaktune/state/state.db")

# Get all runs
runs = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
for run in runs:
    print(f"Run: {run[1]} at {run[2]}")

# Count total items
count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
print(f"Total items: {count}")

# Get unique hashes
unique = conn.execute("SELECT COUNT(DISTINCT hash) FROM hashes").fetchone()[0]
print(f"Unique items: {unique}")

# Get items from latest run
latest_run = conn.execute(
    "SELECT id FROM runs ORDER BY started_at DESC LIMIT 1"
).fetchone()[0]

items = conn.execute(
    "SELECT data FROM items WHERE run_id = ?",
    (latest_run,)
).fetchall()

for item_data in items:
    data = json.loads(item_data[0])
    print(data)

conn.close()
```

## Deduplication States

Deduplication steps check the database:

1. **check_hash** - Looks up exact hash in `hashes` table
2. **check_simhash** - Compares with stored simhashes
3. **check_embedding** - Compares vector with stored embeddings

If a duplicate is found, the item is marked as `StepStatus.FAILED` and skipped.

## Metadata Example

```python
from tweaktune import Pipeline, Metadata

metadata = Metadata(path="./.tweaktune", enabled=True)

(Pipeline(name="dedup_pipeline", metadata=metadata)
    .with_workers(5)
    .with_embeddings_e5("e5-small", "intfloat/e5-small")

    .iter_range(1000)
        .add_column("text", lambda data: generate_text(data["index"]))

        # These steps use metadata for deduplication
        .check_hash(input="text")
        .check_simhash(input="text", threshold=3)
        .check_embedding(input="text", embedding="e5-small", threshold=0.05)

        .write_jsonl(path="unique.jsonl", value="text")
    .run())

# Check results
import sqlite3
conn = sqlite3.connect(".tweaktune/state/state.db")

total = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
unique_hash = conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0]
unique_simhash = conn.execute("SELECT COUNT(*) FROM simhashes").fetchone()[0]
unique_embedding = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

print(f"Total items: {total}")
print(f"Unique by hash: {unique_hash}")
print(f"Unique by simhash: {unique_simhash}")
print(f"Unique by embedding: {unique_embedding}")

conn.close()
```

## Clearing Metadata

To start fresh, delete the metadata directory:

```bash
rm -rf .tweaktune
```

Or selectively clear tables:

```python
import sqlite3

conn = sqlite3.connect(".tweaktune/state/state.db")
conn.execute("DELETE FROM hashes")
conn.execute("DELETE FROM simhashes")
conn.execute("DELETE FROM embeddings")
conn.commit()
conn.close()
```

## Metadata Per Pipeline

Use different metadata paths for different pipelines:

```python
# Pipeline 1
metadata1 = Metadata(path="./.tweaktune/pipeline1", enabled=True)
pipeline1 = Pipeline(name="pipeline1", metadata=metadata1)

# Pipeline 2
metadata2 = Metadata(path="./.tweaktune/pipeline2", enabled=True)
pipeline2 = Pipeline(name="pipeline2", metadata=metadata2)
```

## Disabling Metadata

```python
# Metadata disabled (default)
metadata = Metadata(path="", enabled=False)
pipeline = Pipeline(metadata=metadata)

# Or simply don't provide metadata
pipeline = Pipeline()
```

When disabled:
- No tracking
- Deduplication steps won't work
- Faster execution (no DB writes)

## Performance Considerations

1. **Database writes** add overhead
2. **Embeddings** require computation and storage
3. **SimHash** is faster than embeddings
4. **Exact hash** is fastest
5. Use **batch sizes** appropriate for your data

## Monitoring Progress

```python
# In another terminal/notebook
import sqlite3
import time

conn = sqlite3.connect(".tweaktune/state/state.db")

while True:
    count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"Processed: {count} items", end="\r")
    time.sleep(1)
```

## Best Practices

1. **Enable metadata** when you need deduplication
2. **Use appropriate paths** to organize different pipelines
3. **Clean old data** periodically to save space
4. **Monitor database size** for large pipelines
5. **Backup metadata** before major changes
6. **Query metadata** to analyze pipeline results

## Next Steps

- Learn about [Advanced Features](12-advanced-features.md)
- See [Validation & Quality](08-validation-quality.md) for deduplication
- Explore [examples](/examples) for real-world usage
