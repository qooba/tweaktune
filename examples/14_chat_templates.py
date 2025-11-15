"""
Chat Templates Example

Demonstrates using ChatTemplateBuilder to format conversations
for model fine-tuning.

Based on test_chat_template_builder.py
"""

from tweaktune import ChatTemplateBuilder
from tweaktune.chat_templates import bielik
from pydantic import Field
import json

def search_products(
    query: str = Field(..., description="Search query"),
    category: str = Field(..., description="Category name")
):
    """Search for products in the catalog."""
    pass

def main():
    print("Example 1: Build chat template with tools")

    # Create chat template with tool definitions
    chat_template = (ChatTemplateBuilder(template=bielik)
        .with_tools([search_products])
        .with_bos_token("<s>")
        .build())

    # Sample conversation with tool use
    messages = [
        {
            "role": "user",
            "content": "Find me some laptops"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search_products",
                        "arguments": {"query": "laptop", "category": "electronics"}
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": '{"results": ["Laptop A", "Laptop B"], "count": 2}'
        },
        {
            "role": "assistant",
            "content": "I found 2 laptops for you: Laptop A and Laptop B."
        }
    ]

    # Render the conversation
    result = chat_template.render(messages)

    print("Formatted conversation:")
    print(result[0]["text"])
    print("\n" + "="*60 + "\n")

    print("Example 2: Simple conversation without tools")

    simple_template = """
{%- for message in messages %}
<|{{ message.role }}|>
{{ message.content }}
{% endfor -%}
"""

    chat_template = (ChatTemplateBuilder(template=simple_template)
        .with_bos_token("")
        .build())

    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "I don't have access to weather data."}
    ]

    result = chat_template.render(messages)
    print("Simple formatted conversation:")
    print(result[0]["text"])
    print("\n" + "="*60 + "\n")

    print("Example 3: Multi-turn with tool calls")

    chat_template = (ChatTemplateBuilder(template=bielik)
        .with_tools([search_products])
        .with_bos_token("<s>")
        .build())

    messages = [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant."
        },
        {
            "role": "user",
            "content": "I need a laptop for programming"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "search_products",
                        "arguments": {"query": "programming laptop", "category": "computers"}
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": '{"results": ["Dell XPS 15", "MacBook Pro", "ThinkPad X1"], "count": 3}'
        },
        {
            "role": "assistant",
            "content": "I found 3 great laptops for programming: Dell XPS 15, MacBook Pro, and ThinkPad X1. Would you like details on any of these?"
        },
        {
            "role": "user",
            "content": "Tell me about the Dell XPS 15"
        },
        {
            "role": "assistant",
            "content": "The Dell XPS 15 is an excellent choice for programming with its powerful processor and high-resolution display."
        }
    ]

    result = chat_template.render(messages)
    print("Multi-turn conversation:")
    print(result[0]["text"][:500] + "...\n")

    print("Example 4: Process JSONL file")

    # Create sample JSONL file
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"}
            ]
        }
    ]

    with open("14_conversations.jsonl", "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    # Process with chat template
    # Note: This requires the 'datasets' package
    try:
        dataset = chat_template.render_jsonl(
            path="14_conversations.jsonl",
            tokenize=False  # Set to True if you have a tokenizer configured
        )

        print(f"Processed {len(dataset)} conversations from JSONL")
        print(f"First conversation preview:")
        print(dataset[0]["text"][:200] + "...")

    except ImportError:
        print("Skipping JSONL rendering (requires 'datasets' package)")
        print("Install with: pip install datasets")

    print("\nChat template examples complete")

if __name__ == "__main__":
    main()
