"""
Function Calling Dataset Generation

Creates datasets for training models on function calling.
Based on test_tools.py and test_steps.py (render_conversation tests).

Requires OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python 08_function_calling.py
"""

from tweaktune import Pipeline
from pydantic import Field
from typing import Optional
import os
import json

# Define example tools
def search_products(
    query: str = Field(..., description="Search query for products"),
    category: Optional[str] = Field(None, description="Product category filter"),
    min_price: Optional[float] = Field(None, description="Minimum price filter"),
    max_price: Optional[float] = Field(None, description="Maximum price filter")
):
    """Search for products in the catalog by query and filters."""
    pass

def get_product_details(
    product_id: str = Field(..., description="Unique product identifier")
):
    """Get detailed information about a specific product."""
    pass

def place_order(
    user_id: str = Field(..., description="User ID placing the order"),
    product_id: str = Field(..., description="Product ID to order"),
    quantity: int = Field(..., description="Quantity to order"),
    delivery_address: str = Field(..., description="Delivery address")
):
    """Place an order for a product."""
    pass

def get_order_status(
    order_id: str = Field(..., description="Order ID to check")
):
    """Check the status of an existing order."""
    pass

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    print("Generating function calling dataset...")

    (Pipeline()
        .with_workers(3)
        .with_llm_openai("gpt4", api_key, "gpt-4o-mini")

        # Add tools dataset
        .with_tools_dataset("tools", [
            search_products,
            get_product_details,
            place_order,
            get_order_status
        ])

        # Templates
        .with_template("system", "You are a helpful shopping assistant with access to tools.")
        .with_template("user_prompt", """
Generate a realistic user request that would require using this function: {{tool[0].name}}
Function description: {{tool[0].description}}
""")
        .with_template("args_prompt", """
For the function "{{tool[0].name}}" with description "{{tool[0].description}}",
generate realistic arguments for this user request: {{user_question}}
Return only JSON with the arguments, no explanation.
""")
        .with_template("final_prompt", """
Based on this tool result: {{tool_response|tojson}}
Provide a natural response to the user for their request: {{user_question}}
""")

        .iter_range(10)
            # Sample a random tool
            .sample_tools("tools", 1, "tool")

            # Generate user question
            .generate_text(
                template="user_prompt",
                llm="gpt4",
                output="user_question",
                max_tokens=80,
                temperature=0.8
            )

            # Generate tool arguments
            .generate_json(
                template="args_prompt",
                llm="gpt4",
                output="tool_args",
                max_tokens=150,
                temperature=0.7
            )

            # Format tool call
            .render_tool_call(
                tool="tool[0].name",
                arguments="tool_args",
                output="tool_call"
            )

            # Mock tool response
            .add_column("tool_response", lambda data: {
                "status": "success",
                "message": f"Executed {data['tool'][0]['name']}"
            })

            .generate_text(
                template="final_prompt",
                llm="gpt4",
                output="assistant_answer",
                max_tokens=100,
                temperature=0.7
            )

            # Build complete conversation
            .render_conversation(
                conversation="@system:system|@user:user_question|@assistant:tool_calls([tool_call])|@tool:tool_response|@assistant:assistant_answer",
                tools="tool",
                output="conversation"
            )

            .write_jsonl(path="08_function_calling.jsonl", value="conversation")
        .run())

    print("Generated function calling dataset\n")

    # Display sample
    print("Sample conversation:")
    with open("08_function_calling.jsonl", "r") as f:
        first = json.loads(f.readline())
        messages = first["messages"]
        for msg in messages:
            role = msg["role"]
            if "tool_calls" in msg:
                print(f"  [{role}] Tool calls: {len(msg['tool_calls'])} call(s)")
                for call in msg["tool_calls"]:
                    print(f"    - {call['function']['name']}({call['function']['arguments']})")
            elif "content" in msg:
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                print(f"  [{role}] {content}")

if __name__ == "__main__":
    main()
