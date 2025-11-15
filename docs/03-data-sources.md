# Data Sources

tweaktune supports loading data from multiple sources. All dataset methods support optional SQL queries for filtering and transformation.

## Parquet Files

Load data from Parquet files:

```python
(Pipeline()
    .with_parquet_dataset("items", "data/items.parquet")
    .iter_dataset("items")
        .write_jsonl(path="output.jsonl", value="items")
    .run())
```

With SQL filtering:

```python
.with_parquet_dataset(
    "items",
    "data/items.parquet",
    sql="SELECT * FROM items WHERE price > 10.0"
)
```

## CSV Files

Load CSV files with configurable delimiters and headers:

```python
# With header
.with_csv_dataset(
    "data",
    "data/file.csv",
    delimiter=",",
    has_header=True
)

# Without header (uses column names: column_0, column_1, ...)
.with_csv_dataset(
    "data",
    "data/file.csv",
    delimiter=";",
    has_header=False
)
```

## JSONL Files

Load newline-delimited JSON:

```python
.with_jsonl_dataset("records", "data/records.jsonl")
```

With SQL transformation:

```python
.with_jsonl_dataset(
    "records",
    "data/records.jsonl",
    sql="SELECT name, price FROM records WHERE category = 'electronics'"
)
```

## JSON Files

Load JSON arrays:

```python
.with_json_dataset("items", "data/items.json")
```

## Arrow Datasets

Load HuggingFace datasets or PyArrow data:

```python
from datasets import Dataset

# From HuggingFace Dataset
my_dataset = Dataset.from_dict({"name": ["Alice", "Bob"], "age": [25, 30]})
pipeline.with_arrow_dataset("people", my_dataset)

# From PyArrow RecordBatchReader
import pyarrow as pa
reader = ...  # Your PyArrow reader
pipeline.with_arrow_dataset("data", reader)
```

## HuggingFace Datasets

Load directly from HuggingFace:

```python
# Requires: pip install datasets
.with_hf_dataset(
    "squad",
    dataset_path="squad",
    dataset_name=None,
    dataset_split="train"
)
```

## Database Connections

Load from SQL databases using ConnectorX:

```python
# Requires: pip install connectorx

# PostgreSQL
.with_db_dataset(
    "users",
    conn="postgresql://user:pass@localhost:5432/mydb",
    query="SELECT * FROM users WHERE active = true"
)

# SQLite
.with_db_dataset(
    "items",
    conn="sqlite:///path/to/database.db",
    query="SELECT * FROM items"
)
```

Supported databases: PostgreSQL, MySQL, SQLite, SQL Server, and more. See [ConnectorX docs](https://sfu-db.github.io/connector-x/databases.html).

## Python Dictionaries

Create datasets from Python lists of dictionaries:

```python
data = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 35}
]

.with_dicts_dataset("people", data)
```

With SQL:

```python
.with_dicts_dataset(
    "people",
    data,
    sql="SELECT name FROM people WHERE age > 25"
)
```

## Python Functions (Tools)

Create datasets from Python functions for function calling:

```python
from pydantic import Field
from typing import Optional

def search_products(
    query: str = Field(..., description="Search query"),
    category: Optional[str] = Field(None, description="Product category"),
    min_price: Optional[float] = Field(None, description="Minimum price")
):
    """Search for products in the catalog."""
    pass

def get_weather(
    location: str = Field(..., description="City name")
):
    """Get current weather for a location."""
    pass

pipeline.with_tools_dataset("tools", [search_products, get_weather])
```

The functions are converted to JSON schema automatically using type hints and Pydantic Field descriptions.

## Pydantic Models

Create datasets from Pydantic models for structured output:

```python
from pydantic import BaseModel, Field
from enum import Enum

class ItemType(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    food = "food"

class Item(BaseModel):
    """Product item."""
    id: int = Field(..., description="Item ID")
    name: str = Field(..., description="Item name")
    price: float = Field(..., description="Item price")
    type: ItemType = Field(..., description="Item category")

pipeline.with_pydantic_models_dataset("schemas", [Item])
```

## OpenAPI Specifications

Load function schemas from OpenAPI specs:

```python
.with_openapi_dataset("api", "path/to/openapi.json")
# or from URL
.with_openapi_dataset("api", "https://example.com/openapi.json")
```

## Internal Datasets

Use built-in datasets:

```python
from tweaktune import InternalDatasetType

.with_internal_dataset(InternalDatasetType.Openings)
```

Available internal datasets:
- `InternalDatasetType.Openings` - Conversation opening phrases

Access internal dataset sub-collections:

```python
.iter_range(10)
    .sample("openings::question", 1, "question")
    .sample("openings::ask", 1, "ask")
    .sample("openings::neutral", 1, "neutral")
```

## Mixed Datasets

Combine multiple datasets:

```python
.with_jsonl_dataset("functions", "functions.jsonl")
.with_jsonl_dataset("personas", "personas.jsonl")
.with_mixed_dataset("mixed", ["functions", "personas"])

.iter_dataset("mixed")
    .write_jsonl(path="output.jsonl", value="mixed")
```

Each row in a mixed dataset contains all source datasets as nested objects.

## Sampling from Datasets

Once datasets are loaded, sample from them in your pipeline:

```python
(Pipeline()
    .with_parquet_dataset("products", "products.parquet")
    .iter_range(100)
        .sample(dataset="products", size=3, output="sampled_products")
        .write_jsonl(path="output.jsonl", value="sampled_products")
    .run())
```

For tools datasets, use `sample_tools`:

```python
.with_tools_dataset("tools", [func1, func2, func3])
.iter_range(50)
    .sample_tools(dataset="tools", size=2, output="selected_tools")
```

## Next Steps

- Learn about [Templates](04-templates.md)
- See [Pipeline Steps](06-pipeline-steps.md) for data manipulation
- Explore [Conversation & Tools](09-conversation-tools.md) for function calling
