"""
Conditional Logic Example

Demonstrates the ifelse step for conditional branching.
Based on test_steps.py (ifelse tests)
"""

from tweaktune import Pipeline
from tweaktune.chain import Chain
import json

def main():
    print("Example 1: Simple ifelse with lambda condition")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "value": {{value}},
  "category": {{category|jstr}},
  "message": {{message|jstr}}
}
""")
        .iter_range(10)
            .add_random("value", 1, 100)

            .ifelse(
                condition=lambda data: data["value"] > 50,
                then_chain=Chain()
                    .add_column("category", lambda data: "high")
                    .add_column("message", lambda data: "This is a high value"),
                else_chain=Chain()
                    .add_column("category", lambda data: "low")
                    .add_column("message", lambda data: "This is a low value")
            )

            .write_jsonl(path="13_simple_ifelse.jsonl", template="output")
        .run())
    print("Generated with lambda condition\n")

    print("Example 2: ifelse with expression condition")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "age": {{age}},
  "is_adult": {{is_adult}},
  "category": {{category|jstr}},
  "discount": {{discount}}
}
""")
        .iter_range(10)
            .add_random("age", 1, 100)

            .ifelse(
                condition="age >= 18",  # String expression
                then_chain=Chain()
                    .add_column("is_adult", lambda data: True)
                    .add_column("category", lambda data: "adult")
                    .add_column("discount", lambda data: 0.1),
                else_chain=Chain()
                    .add_column("is_adult", lambda data: False)
                    .add_column("category", lambda data: "minor")
                    .add_column("discount", lambda data: 0.2)
            )

            .write_jsonl(path="13_expression_ifelse.jsonl", template="output")
        .run())
    print("Generated with expression condition\n")

    print("Example 3: Nested ifelse")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "score": {{score}},
  "grade": {{grade|jstr}},
  "message": {{message|jstr}},
  "pass": {{pass}}
}
""")
        .iter_range(15)
            .add_random("score", 0, 100)

            # Outer condition: pass/fail
            .ifelse(
                condition="score >= 60",
                then_chain=Chain()
                    .add_column("pass", lambda data: True)
                    # Inner condition: grade within passing
                    .ifelse(
                        condition="score >= 90",
                        then_chain=Chain()
                            .add_column("grade", lambda data: "A")
                            .add_column("message", lambda data: "Excellent!"),
                        else_chain=Chain()
                            .ifelse(
                                condition="score >= 75",
                                then_chain=Chain()
                                    .add_column("grade", lambda data: "B")
                                    .add_column("message", lambda data: "Good job!"),
                                else_chain=Chain()
                                    .add_column("grade", lambda data: "C")
                                    .add_column("message", lambda data: "Passed")
                            )
                    ),
                else_chain=Chain()
                    .add_column("pass", lambda data: False)
                    .add_column("grade", lambda data: "F")
                    .add_column("message", lambda data: "Failed")
            )

            .write_jsonl(path="13_nested_ifelse.jsonl", template="output")
        .run())
    print("Generated with nested conditions\n")

    print("Example 4: Complex business logic")
    (Pipeline()
        .with_workers(1)
        .with_template("output", """
{
  "customer_type": {{customer_type|jstr}},
  "amount": {{amount}},
  "discount": {{discount}},
  "priority": {{priority}},
  "shipping": {{shipping|jstr}}
}
""")
        .iter_range(10)
            .add_column("customer_type", lambda data:
                ["regular", "premium", "vip"][data["index"] % 3]
            )
            .add_random("amount", 10, 500)

            # Different logic based on customer type
            .ifelse(
                condition="customer_type == 'vip'",
                then_chain=Chain()
                    .add_column("discount", lambda data: 0.30)
                    .add_column("priority", lambda data: 1)
                    .add_column("shipping", lambda data: "express"),
                else_chain=Chain()
                    .ifelse(
                        condition="customer_type == 'premium'",
                        then_chain=Chain()
                            .add_column("discount", lambda data: 0.20)
                            .add_column("priority", lambda data: 2)
                            .add_column("shipping", lambda data: "standard"),
                        else_chain=Chain()
                            .add_column("discount", lambda data: 0.10)
                            .add_column("priority", lambda data: 3)
                            .add_column("shipping", lambda data: "economy")
                    )
            )

            # Apply discount
            .add_column("final_amount", lambda data:
                data["amount"] * (1 - data["discount"])
            )

            .with_template("full_output", """
{
  "customer_type": {{customer_type|jstr}},
  "amount": {{amount}},
  "discount": {{discount}},
  "final_amount": {{final_amount}},
  "priority": {{priority}},
  "shipping": {{shipping|jstr}}
}
""")
            .write_jsonl(path="13_business_logic.jsonl", template="full_output")
        .run())
    print("Generated with business logic\n")

    # Show results
    print("Simple ifelse results:")
    with open("13_simple_ifelse.jsonl", "r") as f:
        high = low = 0
        for line in f:
            data = json.loads(line)
            if data["category"] == "high":
                high += 1
            else:
                low += 1
        print(f"  High values: {high}, Low values: {low}")

    print("\nNested ifelse grade distribution:")
    with open("13_nested_ifelse.jsonl", "r") as f:
        grades = {"A": 0, "B": 0, "C": 0, "F": 0}
        for line in f:
            data = json.loads(line)
            grades[data["grade"]] += 1
        for grade, count in sorted(grades.items()):
            print(f"  Grade {grade}: {count}")

    print("\nBusiness logic by customer type:")
    with open("13_business_logic.jsonl", "r") as f:
        by_type = {}
        for line in f:
            data = json.loads(line)
            ctype = data["customer_type"]
            if ctype not in by_type:
                by_type[ctype] = {"count": 0, "total_discount": 0}
            by_type[ctype]["count"] += 1
            by_type[ctype]["total_discount"] += data["discount"]

        for ctype, stats in sorted(by_type.items()):
            avg_discount = stats["total_discount"] / stats["count"]
            print(f"  {ctype}: {stats['count']} orders, avg discount: {avg_discount:.0%}")

if __name__ == "__main__":
    main()
