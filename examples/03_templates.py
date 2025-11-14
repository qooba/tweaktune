"""
Templates Example

Demonstrates various template usage:
- Inline templates
- Template files (Jinja2)
- Template directories
- Template filters
"""

from tweaktune import Pipeline
import os
import tempfile

def main():
    # Create temporary directory for templates
    temp_dir = tempfile.mkdtemp()

    # Create a template file
    template_file = os.path.join(temp_dir, "greeting.j2")
    with open(template_file, "w") as f:
        f.write("""{"greeting": "Hello {{name}}!", "id": {{index}}}""")

    print("Example 1: Inline template with filters")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"text": {{text|jstr}}, "number": {{number}}}""")
        .iter_range(3)
            .add_column("text", lambda data: f"Hello World {data['index']}")
            .add_column("number", lambda data: data["index"] * 10)
            .write_jsonl(path="03_inline.jsonl", template="output")
        .run())
    print("Written to 03_inline.jsonl\n")

    print("Example 2: Template from file")
    (Pipeline()
        .with_workers(1)
        .with_j2_template("greeting", template_file)
        .iter_range(3)
            .add_column("name", lambda data: f"Person_{data['index']}")
            .write_jsonl(path="03_from_file.jsonl", template="greeting")
        .run())
    print("Written to 03_from_file.jsonl\n")

    print("Example 3: Random values with template filter")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "id": {{index}},
  "random_small": {{"1,10"|random_range}},
  "random_large": {{"100,1000"|random_range}},
  "name": "Item_{{index}}"
}
""")
        .iter_range(5)
            .write_jsonl(path="03_random.jsonl", template="output")
        .run())
    print("Written to 03_random.jsonl\n")

    print("Example 4: Complex template with arrays")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "items": {{items|tojson}},
  "count": {{items|length}},
  "first_item": {{items[0]|jstr}}
}
""")
        .iter_range(3)
            .add_column("items", lambda data: [f"item_{i}" for i in range(data["index"] + 1)])
            .write_jsonl(path="03_complex.jsonl", template="output")
        .run())
    print("Written to 03_complex.jsonl\n")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    # Show sample output
    print("Sample output from 03_complex.jsonl:")
    with open("03_complex.jsonl", "r") as f:
        import json
        for line in f:
            print(f"  {json.loads(line)}")

if __name__ == "__main__":
    main()
