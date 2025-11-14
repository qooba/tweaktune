# Custom Steps

Create custom pipeline steps for specialized processing logic.

## Custom Step Class

Create a class with a `process` method:

```python
class CustomStep:
    def process(self, context):
        # Access data
        data = context["data"]

        # Modify data
        data["custom_field"] = "custom_value"
        data["timestamp"] = time.time()

        # Return modified context
        return context

# Use in pipeline
.step(CustomStep())
```

## Map Function

For simple transformations, use `.map()`:

```python
def my_transform(context):
    context["data"]["doubled"] = context["data"]["value"] * 2
    return context

.map(my_transform)
```

## Add Column with Lambda

Simplest approach for adding computed columns:

```python
.add_column("computed", lambda data: data["a"] + data["b"])
```

## State Management

Custom steps can maintain internal state:

```python
class Counter:
    def __init__(self):
        self.count = 0

    def process(self, context):
        self.count += 1
        context["data"]["item_number"] = self.count
        return context

counter = Counter()
.step(counter)
```

## Error Handling

Mark items as failed:

```python
from tweaktune.common import StepStatus

class Validator:
    def process(self, context):
        data = context["data"]

        # Validation logic
        if not self.is_valid(data):
            context["status"] = StepStatus.FAILED.value
            return context

        # Process valid data
        data["validated"] = True
        return context

    def is_valid(self, data):
        return "required_field" in data and data["required_field"] is not None

.step(Validator())
```

## API Integration Example

```python
import requests

class APIEnricher:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key

    def process(self, context):
        data = context["data"]

        # Call external API
        try:
            response = requests.get(
                f"{self.api_url}/enrich",
                params={"query": data["search_term"]},
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()

            # Add API response to context
            data["enriched_data"] = response.json()
        except Exception as e:
            # Mark as failed on error
            context["status"] = StepStatus.FAILED.value
            data["error"] = str(e)

        return context

enricher = APIEnricher("https://api.example.com", os.environ["API_KEY"])
.step(enricher)
```

## Retry Logic

```python
import time

class RetryStep:
    def __init__(self, max_retries=3, delay=1.0):
        self.max_retries = max_retries
        self.delay = delay

    def process(self, context):
        for attempt in range(self.max_retries):
            try:
                return self._try_process(context)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    context["status"] = StepStatus.FAILED.value
                    context["data"]["error"] = str(e)
                    return context
                time.sleep(self.delay * (attempt + 1))

        return context

    def _try_process(self, context):
        # Your processing logic here
        data = context["data"]
        # ... do something that might fail
        return context

.step(RetryStep(max_retries=3, delay=1.0))
```

## Data Transformation Example

```python
class DataCleaner:
    def process(self, context):
        data = context["data"]

        # Clean text fields
        for field in ["title", "description", "content"]:
            if field in data:
                data[field] = self._clean_text(data[field])

        # Normalize numbers
        if "price" in data:
            data["price"] = round(float(data["price"]), 2)

        # Remove null values
        data = {k: v for k, v in data.items() if v is not None}
        context["data"] = data

        return context

    def _clean_text(self, text):
        if not isinstance(text, str):
            return text
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove special characters
        text = text.strip()
        return text

.step(DataCleaner())
```

## Custom Validator

Use `.validate()` for custom validation:

```python
def my_validator(context):
    data = context["data"]

    # Check required fields
    required = ["name", "email", "age"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate email format
    if "@" not in data["email"]:
        raise ValueError("Invalid email format")

    # Validate age range
    if not (0 <= data["age"] <= 120):
        raise ValueError("Age must be between 0 and 120")

    return True

.validate(my_validator)
```

If validator raises an exception, the item is marked as failed.

## Conditional Processing

```python
class ConditionalProcessor:
    def process(self, context):
        data = context["data"]

        if data.get("type") == "premium":
            data["discount"] = 0.2
            data["priority"] = "high"
        elif data.get("type") == "regular":
            data["discount"] = 0.1
            data["priority"] = "normal"
        else:
            data["discount"] = 0.0
            data["priority"] = "low"

        return context

.step(ConditionalProcessor())
```

Or use the `.ifelse()` step for simpler cases.

## Batch Processing

```python
class BatchProcessor:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.batch = []

    def process(self, context):
        data = context["data"]

        # Add to batch
        self.batch.append(data)

        # Process when batch is full
        if len(self.batch) >= self.batch_size:
            processed = self._process_batch(self.batch)
            data["batch_result"] = processed
            self.batch = []

        return context

    def _process_batch(self, batch):
        # Batch processing logic
        return {"processed": len(batch)}

.step(BatchProcessor(batch_size=10))
```

## Database Operations

```python
import sqlite3

class DatabaseWriter:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def process(self, context):
        data = context["data"]

        try:
            self.conn.execute(
                "INSERT INTO results (name, value) VALUES (?, ?)",
                (data.get("name"), data.get("value"))
            )
            self.conn.commit()
            data["db_inserted"] = True
        except Exception as e:
            context["status"] = StepStatus.FAILED.value
            data["error"] = str(e)

        return context

    def __del__(self):
        self.conn.close()

.step(DatabaseWriter("results.db"))
```

## Logging in Custom Steps

```python
import logging

class LoggingStep:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process(self, context):
        data = context["data"]

        self.logger.info(f"Processing item: {data.get('id')}")

        try:
            # Processing logic
            result = self._process_data(data)
            data["result"] = result
            self.logger.debug(f"Result: {result}")
        except Exception as e:
            self.logger.error(f"Error processing {data.get('id')}: {e}")
            context["status"] = StepStatus.FAILED.value

        return context

    def _process_data(self, data):
        # Processing logic
        return data

.step(LoggingStep())
```

## Best Practices

1. **Always return context** - The `process` method must return the context
2. **Use `StepStatus.FAILED`** to mark failed items rather than raising exceptions
3. **Keep steps focused** - Each step should do one thing well
4. **Manage resources** - Close connections, files in `__del__` or use context managers
5. **Add logging** for debugging and monitoring
6. **Handle errors gracefully** - Don't let exceptions break the pipeline
7. **Document your steps** - Add docstrings explaining what the step does
8. **Make steps reusable** - Parameterize behavior in `__init__`

## Next Steps

- Learn about [Validation & Quality](08-validation-quality.md)
- See [Pipeline Steps](06-pipeline-steps.md) for built-in steps
- Explore [examples](/examples) for real-world use cases
