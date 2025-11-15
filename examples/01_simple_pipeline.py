"""
Simple Pipeline Example

This example demonstrates basic tweaktune usage:
- Creating a pipeline with workers
- Using templates with random values
- Iterating a fixed number of times
- Writing output to JSONL
"""

from tweaktune import Pipeline

def main():
    # Create a simple pipeline that generates 10 records
    (Pipeline()
        .with_workers(2)
        .with_template("output", """{"id": {{index}}, "value": {{"1,100"|random_range}}}""")
        .iter_range(10)
            .write_jsonl(path="01_simple_output.jsonl", template="output")
        .run())

    # Verify output
    with open("01_simple_output.jsonl", "r") as f:
        lines = f.readlines()
        print(f"Generated {len(lines)} records")
        print(f"First record: {lines[0].strip()}")
        print(f"Last record: {lines[-1].strip()}")

if __name__ == "__main__":
    main()
