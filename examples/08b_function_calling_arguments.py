"""
Function Calling with Custom Argument Sampling

Demonstrates how to use custom datasets for tool arguments to generate
more controlled and realistic function calling examples.

This example shows:
1. Defining custom argument datasets for specific tool parameters
2. Sampling argument values from these datasets
3. Creating diverse function calling scenarios with controlled inputs

Usage:
    python 08b_function_calling_arguments.py
"""

from tweaktune import Pipeline
from pydantic import Field
from typing import Optional
import json


def search_products(
    query: str = Field(..., description="Search query for products"),
    category: Optional[str] = Field(None, description="Product category filter"),
    min_price: Optional[float] = Field(None, description="Minimum price filter"),
    max_price: Optional[float] = Field(None, description="Maximum price filter")
):
    """Search for products in the catalog by query and filters."""
    pass


def main():
    print("Generating function calling dataset with custom argument sampling...\n")

    # Define custom datasets for different arguments
    categories_data = [
        {"name": "electronics", "description": "Electronic devices and gadgets", "popular": True},
        {"name": "books", "description": "Books and publications", "popular": True},
        {"name": "clothing", "description": "Apparel and accessories", "popular": False},
        {"name": "home", "description": "Home and garden items", "popular": False},
        {"name": "sports", "description": "Sports and fitness equipment", "popular": True},
    ]

    queries_data = [
        {"value": "laptop", "intent": "find computer", "category_hint": "electronics"},
        {"value": "wireless headphones", "intent": "find audio device", "category_hint": "electronics"},
        {"value": "science fiction novel", "intent": "find book", "category_hint": "books"},
        {"value": "running shoes", "intent": "find footwear", "category_hint": "sports"},
        {"value": "yoga mat", "intent": "find exercise equipment", "category_hint": "sports"},
        {"value": "coffee maker", "intent": "find kitchen appliance", "category_hint": "home"},
        {"value": "winter jacket", "intent": "find outerwear", "category_hint": "clothing"},
    ]

    price_ranges_data = [
        {"min": 0, "max": 50, "range": "budget"},
        {"min": 50, "max": 200, "range": "mid"},
        {"min": 200, "max": 1000, "range": "premium"},
        {"min": 1000, "max": None, "range": "luxury"},
    ]

    (Pipeline()
        .with_workers(1)

        # Define the tool
        .with_tools_dataset("tools", [search_products])

        # Define custom argument datasets
        .with_tool_argument_dicts_dataset(
            "search_products",  # Tool name
            "category",         # Argument name
            categories_data
        )
        .with_tool_argument_dicts_dataset(
            "search_products",
            "query",
            queries_data
        )

        # Templates for formatting output
        .with_template("output", """{"tool": {{tool[0]|tojson}}, "sampled_arguments": {{function_arguments|tojson}}, "category_name": "{{function_arguments.category[0].name}}", "category_popular": {{function_arguments.category[0].popular}}, "query_value": "{{function_arguments.query[0].value}}", "query_intent": "{{function_arguments.query[0].intent}}"}""")

        .iter_range(15)
            # Sample the tool
            .sample_tools("tools", 1, "tool")

            # Sample argument values from custom datasets
            .sample_tool_arguments(
                tool_name="tool[0].name",  # Reference to tool name
                size=1,  # Sample 1 value per argument
                output="function_arguments"
            )

            # Write results
            .write_jsonl(path="08b_function_calling_arguments.jsonl", template="output")
        .run())

    print("Generated dataset with custom argument sampling\n")

    # Display samples
    print("Sample outputs (first 3):")
    with open("08b_function_calling_arguments.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            print(f"\n--- Example {i + 1} ---")
            print(f"Tool: {data['tool']['name']}")
            print(f"Query: {data['query_value']} (intent: {data['query_intent']})")
            print(f"Category: {data['category_name']} (popular: {data['category_popular']})")
            print(f"Full sampled arguments:")
            print(f"  - category: {data['sampled_arguments']['category'][0]}")
            print(f"  - query: {data['sampled_arguments']['query'][0]}")

    print("\n" + "="*60)
    print("Key Features Demonstrated:")
    print("="*60)
    print("1. Custom argument datasets with rich metadata")
    print("2. Sampling multiple arguments per tool")
    print("3. Accessing sampled values in templates")
    print("4. Combining tool sampling with argument sampling")
    print("\nOutput file: 08b_function_calling_arguments.jsonl")


if __name__ == "__main__":
    main()
