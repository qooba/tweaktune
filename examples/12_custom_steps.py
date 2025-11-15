"""
Custom Steps Example

Demonstrates creating custom pipeline steps:
- Custom step classes
- Map functions
- State management
- Error handling

Based on test_steps.py (step, map tests)
"""

from tweaktune import Pipeline
from tweaktune.common import StepStatus
import json
import time

def main():
    print("Example 1: Simple custom step class")

    class GreetingStep:
        def process(self, context):
            data = context["data"]
            name = data.get("name", "World")
            data["greeting"] = f"Hello, {name}!"
            data["processed_at"] = time.time()
            return context

    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"name": {{name|jstr}}, "greeting": {{greeting|jstr}}}""")
        .iter_range(5)
            .add_column("name", lambda data: f"Person_{data['index']}")
            .step(GreetingStep())
            .write_jsonl(path="12_greetings.jsonl", template="output")
        .run())
    print("Generated with custom step\n")

    print("Example 2: Stateful custom step")

    class Counter:
        def __init__(self):
            self.count = 0
            self.total_value = 0

        def process(self, context):
            data = context["data"]
            self.count += 1
            value = data.get("value", 0)
            self.total_value += value

            data["item_number"] = self.count
            data["running_total"] = self.total_value
            data["average"] = self.total_value / self.count

            return context

    counter = Counter()

    (Pipeline()
        .with_workers(1)  # Must use 1 worker for stateful steps!
        .with_template("output", """
{
  "value": {{value}},
  "item_number": {{item_number}},
  "running_total": {{running_total}},
  "average": {{average}}
}
""")
        .iter_range(5)
            .add_column("value", lambda data: (data["index"] + 1) * 10)
            .step(counter)
            .write_jsonl(path="12_stateful.jsonl", template="output")
        .run())
    print("Generated with stateful step\n")

    print("Example 3: Error handling step")

    class ValidatingStep:
        def process(self, context):
            data = context["data"]

            try:
                # Simulate validation
                value = data.get("value", 0)

                if value < 0:
                    raise ValueError("Value cannot be negative")

                if value > 100:
                    raise ValueError("Value too large")

                # Add validation metadata
                data["validated"] = True
                data["validation_status"] = "passed"

            except ValueError as e:
                # Mark as failed instead of crashing
                context["status"] = StepStatus.FAILED.value
                data["validated"] = False
                data["validation_status"] = str(e)

            return context

    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "value": {{value}},
  "validated": {{validated}},
  "status": {{validation_status|jstr}}
}
""")
        .iter_range(10)
            .add_column("value", lambda data: (data["index"] - 5) * 20)
            .step(ValidatingStep())
            .write_jsonl(path="12_validated.jsonl", template="output")
        .run())
    print("Generated with validation step\n")

    print("Example 4: Map function")

    def enrich_data(context):
        data = context["data"]
        value = data.get("value", 0)

        # Add multiple computed fields
        data["squared"] = value ** 2
        data["is_even"] = value % 2 == 0
        data["category"] = "low" if value < 30 else "medium" if value < 70 else "high"
        data["formatted"] = f"Value: {value:04d}"

        return context

    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "value": {{value}},
  "squared": {{squared}},
  "is_even": {{is_even}},
  "category": {{category|jstr}},
  "formatted": {{formatted|jstr}}
}
""")
        .iter_range(5)
            .add_random("value", 1, 100)
            .map(enrich_data)
            .write_jsonl(path="12_enriched.jsonl", template="output")
        .run())
    print("Generated with map function\n")

    print("Example 5: Data transformation step")

    class DataCleaner:
        def process(self, context):
            data = context["data"]

            # Clean text field
            if "text" in data:
                text = data["text"]
                # Remove extra whitespace
                text = " ".join(text.split())
                # Strip
                text = text.strip()
                # Capitalize
                text = text.capitalize()
                data["text"] = text

            # Round numbers
            if "price" in data:
                data["price"] = round(float(data["price"]), 2)

            return context

    (Pipeline()
        .with_workers(1)
        .with_template("output", """{"text": {{text|jstr}}, "price": {{price}}}""")
        .iter_range(5)
            .add_column("text", lambda data: f"  hello   world  {data['index']}  ")
            .add_column("price", lambda data: 10.12345 + data["index"])
            .step(DataCleaner())
            .write_jsonl(path="12_cleaned.jsonl", template="output")
        .run())
    print("Generated with cleaning step\n")

    # Show results
    print("Stateful step results:")
    with open("12_stateful.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            print(f"  Item #{data['item_number']}: value={data['value']}, "
                  f"total={data['running_total']}, avg={data['average']:.1f}")

    print("\nValidated results:")
    with open("12_validated.jsonl", "r") as f:
        passed = 0
        failed = 0
        for line in f:
            data = json.loads(line)
            if data["validated"]:
                passed += 1
            else:
                failed += 1
        print(f"  Passed: {passed}, Failed: {failed}")

if __name__ == "__main__":
    main()
