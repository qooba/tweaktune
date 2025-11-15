"""
Data Sources Example

Demonstrates loading data from various sources:
- Python dictionaries
- JSONL files
- CSV files
- Parquet files (if available)
"""

from tweaktune import Pipeline
import json

def main():
    # Prepare sample data
    sample_data = [
        {"name": "Alice", "age": 25, "city": "New York"},
        {"name": "Bob", "age": 30, "city": "San Francisco"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
        {"name": "Diana", "age": 28, "city": "Boston"},
        {"name": "Eve", "age": 32, "city": "Seattle"}
    ]

    # Write sample JSONL file
    with open("02_sample.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")

    print("Example 1: Loading from Python dictionaries")
    (Pipeline()
        .with_workers(1)
        .with_dicts_dataset("people", sample_data)
        .with_template("output", """{"name": {{person[0].name|jstr}}, "age": {{person[0].age}}}""")
        .iter_range(5)
            .sample("people", 1, "person")
            .write_jsonl(path="02_from_dicts.jsonl", template="output")
        .run())
    print("Written to 02_from_dicts.jsonl\n")

    print("Example 2: Loading from JSONL file")
    (Pipeline()
        .with_workers(1)
        .with_jsonl_dataset("people", "02_sample.jsonl")
        .with_template("output", """{"person": {{person|jstr}}}""")
        .iter_dataset("people")
            .write_jsonl(path="02_from_jsonl.jsonl", template="output")
        .run())
    print("Written to 02_from_jsonl.jsonl\n")

    print("Example 3: Loading with SQL filter")
    (Pipeline()
        .with_workers(1)
        .with_jsonl_dataset(
            "people",
            "02_sample.jsonl",
            sql="SELECT * FROM people WHERE age > 28 ORDER BY age"
        )
        .with_template("output", """{"name": {{people.name|jstr}}, "age": {{people.age}}}""")
        .iter_dataset("people")
            .write_jsonl(path="02_filtered.jsonl", template="output")
        .run())
    print("Written to 02_filtered.jsonl (filtered age > 28)\n")

    # Verify outputs
    with open("02_filtered.jsonl", "r") as f:
        lines = f.readlines()
        print(f"Filtered dataset has {len(lines)} records (expected 3)")
        for line in lines:
            data = json.loads(line)
            print(f"  - {data['name']}, age {data['age']}")

if __name__ == "__main__":
    main()
