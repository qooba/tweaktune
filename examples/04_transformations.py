"""
Data Transformations Example

Demonstrates various data transformation steps:
- add_column with lambda and expressions
- mutate for modifying columns
- filter for conditional selection
- map for custom transformations
- into_list for combining columns
"""

from tweaktune import Pipeline
import json

def main():
    print("Example 1: Add columns with lambda and expressions")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "id": {{id}},
  "price": {{price}},
  "quantity": {{quantity}},
  "total": {{total}},
  "tax": {{tax}},
  "final": {{final}}
}
""")
        .iter_range(5)
            .add_column("id", lambda data: data["index"] + 1)
            .add_column("price", lambda data: 10.0 + data["index"] * 5)
            .add_column("quantity", lambda data: data["index"] + 1)
            .add_column("total", "price * quantity", is_json=False)
            .add_column("tax", "total * 0.1", is_json=False)
            .add_column("final", "total + tax", is_json=False)
            .write_jsonl(path="04_calculations.jsonl", template="output")
        .run())
    print("Written to 04_calculations.jsonl\n")

    print("Example 2: Filter data")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"value": {{value}}, "category": {{category|jstr}}}""")
        .iter_range(20)
            .add_random("value", 1, 100)
            .add_column("category", lambda data:
                "high" if data["value"] > 75 else
                "medium" if data["value"] > 25 else "low"
            )
            .filter("value > 50")  # Only keep values > 50
            .write_jsonl(path="04_filtered.jsonl", template="output")
        .run())
    print("Written to 04_filtered.jsonl (only values > 50)\n")

    print("Example 3: Mutate existing columns")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"original": {{original}}, "doubled": {{doubled}}, "squared": {{squared}}}""")
        .iter_range(5)
            .add_column("original", lambda data: data["index"] + 1)
            .add_column("doubled", lambda data: data["original"])
            .mutate("doubled", lambda val: val * 2)
            .add_column("squared", lambda data: data["original"])
            .mutate("squared", "squared * squared", is_json=False)
            .write_jsonl(path="04_mutated.jsonl", template="output")
        .run())
    print("Written to 04_mutated.jsonl\n")

    print("Example 4: Custom map function")
    def enrich_data(context):
        data = context["data"]
        value = data.get("value", 0)

        # Add multiple computed fields
        data["is_even"] = value % 2 == 0
        data["category"] = "small" if value < 30 else "medium" if value < 70 else "large"
        data["formatted"] = f"Value: {value:03d}"

        return context

    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "value": {{value}},
  "is_even": {{is_even}},
  "category": {{category|jstr}},
  "formatted": {{formatted|jstr}}
}
""")
        .iter_range(10)
            .add_random("value", 1, 100)
            .map(enrich_data)
            .write_jsonl(path="04_enriched.jsonl", template="output")
        .run())
    print("Written to 04_enriched.jsonl\n")

    print("Example 5: Combine columns into list")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"values": {{values|tojson}}, "sum": {{sum}}}""")
        .iter_range(3)
            .add_column("a", lambda data: (data["index"] + 1) * 10)
            .add_column("b", lambda data: (data["index"] + 1) * 20)
            .add_column("c", lambda data: (data["index"] + 1) * 30)
            .into_list(inputs=["a", "b", "c"], output="values")
            .add_column("sum", lambda data: sum(data["values"]))
            .write_jsonl(path="04_combined.jsonl", template="output")
        .run())
    print("Written to 04_combined.jsonl\n")

    # Show sample output
    print("Sample from 04_enriched.jsonl:")
    with open("04_enriched.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"  {json.loads(line)}")

if __name__ == "__main__":
    main()
