# Conversation & Tools

Generate datasets for function calling and conversation fine-tuning.

## Tools Dataset

Create a dataset from Python functions:

```python
from pydantic import Field
from typing import Optional

def search_products(
    query: str = Field(..., description="Search query"),
    category: Optional[str] = Field(None, description="Product category"),
    min_price: Optional[float] = Field(None, description="Minimum price"),
    max_price: Optional[float] = Field(None, description="Maximum price")
):
    """Search for products in the catalog."""
    pass

def get_product_details(
    product_id: str = Field(..., description="Product ID")
):
    """Get detailed information about a product."""
    pass

.with_tools_dataset("tools", [search_products, get_product_details])
```

Functions are automatically converted to JSON schema with:
- Function name from the function name
- Description from the docstring
- Parameters from type hints and Field descriptions

## Sampling Tools

Sample random tools:

```python
.sample_tools(dataset="tools", size=2, output="selected_tools")
```

## Sampling Tool Arguments

For more control over function arguments, you can provide custom datasets for specific tool parameters and sample from them:

### Define Argument Datasets

```python
from tweaktune import Pipeline

# Define a tool with parameters
def search_products(
    query: str = Field(..., description="Search query"),
    category: Optional[str] = Field(None, description="Product category"),
    min_price: Optional[float] = Field(None, description="Minimum price")
):
    """Search for products in the catalog."""
    pass

# Create custom dataset for the 'category' argument
.with_tools_dataset("tools", [search_products])
.with_tool_argument_dicts_dataset(
    "search_products",  # Tool name
    "category",         # Argument name
    [
        {"name": "electronics", "description": "Electronic devices and gadgets"},
        {"name": "books", "description": "Books and publications"},
        {"name": "clothing", "description": "Apparel and accessories"}
    ]
)
```

### Sample Argument Values

After defining argument datasets, sample values for a specific tool's arguments:

```python
.sample_tools("tools", 1, "tool")  # Sample a tool
.sample_tool_arguments(
    tool_name="tool[0].name",        # Reference to tool name
    size=1,                          # Number of values to sample per argument
    output="function_arguments"      # Output key
)
```

This creates a structured output with sampled values for each argument that has a custom dataset:

```json
{
  "function_arguments": {
    "category": [
      {
        "name": "electronics",
        "description": "Electronic devices and gadgets"
      }
    ]
  }
}
```

### Complete Example

```python
(Pipeline()
    .with_workers(1)
    .with_tools_dataset("tools", [search_products])

    # Define custom values for specific arguments
    .with_tool_argument_dicts_dataset(
        "search_products",
        "category",
        [
            {"name": "electronics", "description": "Electronics category"},
            {"name": "books", "description": "Books category"}
        ]
    )
    .with_tool_argument_dicts_dataset(
        "search_products",
        "query",
        [
            {"value": "laptop", "intent": "find computer"},
            {"value": "notebook", "intent": "find paper product or computer"}
        ]
    )

    .iter_range(10)
        .sample_tools("tools", 1, "tool")
        .sample_tool_arguments("tool[0].name", 1, "args")

        # Access sampled values
        .add_column("category_name", lambda data: data["args"]["category"][0]["name"])
        .add_column("query_value", lambda data: data["args"]["query"][0]["value"])

    .run())
```

**Use Cases:**
- Generate synthetic function calling datasets with realistic argument values
- Create diverse test cases by sampling from predefined argument pools
- Control the distribution of argument values in training data
- Combine with LLM generation for hybrid synthetic data creation

## Tool Call Formatting

Format a tool call:

```python
.add_column("tool_name", lambda data: "search_products")
.add_column("arguments", lambda data: '{"query": "laptop", "max_price": 1000}')

.render_tool_call(
    tool="tool_name",
    arguments="arguments",
    output="tool_call"
)
```

Result:
```json
{
  "function": {
    "name": "search_products",
    "arguments": {"query": "laptop", "max_price": 1000}
  }
}
```

Extract tool from dataset:

```python
.sample_tools("tools", 1, "tools")
.render_tool_call(
    tool="tools[0].name",
    arguments="arguments",
    output="tool_call"
)
```

## Conversation Format

Tweaktune provides two ways to define conversations:

1. **String format** - Compact syntax using role prefixes (legacy)
2. **Conv() builder** - Fluent Python API (recommended)

### String Format

Build OpenAI-style conversations using string syntax:

```python
.add_column("system", lambda data: "You are a helpful assistant.")
.add_column("question", lambda data: "What's the weather?")
.add_column("answer", lambda data: "I'll check that for you.")

.render_conversation(
    conversation="@system:system|@user:question|@assistant:answer",
    output="messages"
)
```

Result:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I'll check that for you."}
  ]
}
```

### Conv() Builder (Recommended)

The `Conv()` builder provides a cleaner, more Pythonic API with better IDE support and type safety:

```python
from tweaktune import Conv

.add_column("system", lambda data: "You are a helpful assistant.")
.add_column("question", lambda data: "What's the weather?")
.add_column("answer", lambda data: "I'll check that for you.")

.render_conversation(
    conversation=Conv()
        .system("system")
        .user("question")
        .assistant("answer"),
    output="messages"
)
```

**Benefits of Conv() builder:**
- Better IDE autocomplete and type hints
- More readable, especially for complex multi-turn conversations
- Chainable method calls for natural flow
- Less error-prone than string parsing

### Multi-turn Conversations with Conv()

```python
from tweaktune import Conv

.add_column("system", lambda data: "You are a math tutor.")
.add_column("q1", lambda data: "What is 2 + 2?")
.add_column("a1", lambda data: "2 + 2 equals 4.")
.add_column("q2", lambda data: "What about 3 + 5?")
.add_column("a2", lambda data: "3 + 5 equals 8.")

.render_conversation(
    conversation=Conv()
        .system("system")
        .user("q1")
        .assistant("a1")
        .user("q2")
        .assistant("a2"),
    output="conversation"
)
```

### Reasoning with Conv()

Add thinking/reasoning steps before the final answer:

```python
from tweaktune import Conv

.add_column("system", lambda data: "You are a problem-solving assistant.")
.add_column("problem", lambda data: "How can I optimize my code?")
.add_column("thinking", lambda data: "Let me analyze the problem step by step...")
.add_column("solution", lambda data: "Here are three optimization strategies...")

.render_conversation(
    conversation=Conv()
        .system("system")
        .user("problem")
        .think("thinking")      # Adds reasoning_content
        .assistant("solution"),
    output="conversation"
)
```

Result:
```json
{
  "messages": [
    {"role": "user", "content": "How can I optimize my code?"},
    {
      "role": "assistant",
      "reasoning_content": "Let me analyze the problem step by step...",
      "content": ""
    },
    {"role": "assistant", "content": "Here are three optimization strategies..."}
  ]
}
```

### Tool Calls with Conv()

```python
from tweaktune import Conv

.render_conversation(
    conversation=Conv()
        .system("system")
        .user("question")
        .tool_calls(["call1", "call2"])  # Multiple tool calls
        .tool("tool_response1")
        .tool("tool_response2")
        .assistant("final_answer"),
    tools="available_tools",
    output="conversation"
)
```

### Conv() Methods

All Conv() builder methods:

- `.system(content)` - Add system message
- `.user(content)` - Add user message
- `.assistant(content)` - Add assistant message
- `.tool(content)` - Add tool response message
- `.tool_calls(calls)` - Add tool calls (accepts list of call names or single string)
- `.think(content)` - Add reasoning/thinking content

Each method accepts a string that references a key in the pipeline context (e.g., "system", "question", "answer").

## Role Aliases (String Format)

Shorthand notation:

```python
.render_conversation(
    conversation="@s:system_prompt|@u:user_message|@a:assistant_reply|@t:tool_result",
    output="conversation"
)
```

Aliases:
- `@system` or `@s` - System message
- `@user` or `@u` - User message
- `@assistant` or `@a` - Assistant message
- `@tool` or `@t` - Tool result

## Tool Calls in Conversations (String Format)

```python
.render_conversation(
    conversation="@system:system|@user:question|@assistant:tool_calls([call1, call2])|@tool:result1|@tool:result2|@assistant:final_answer",
    output="conversation",
    tools="available_tools"
)
```

Result:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "tool_calls": [
        {"function": {"name": "...", "arguments": {...}}},
        {"function": {"name": "...", "arguments": {...}}}
      ]
    },
    {"role": "tool", "content": "..."},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## Reasoning Content (String Format)

Add reasoning/thinking steps:

```python
.render_conversation(
    conversation="@user:question|@assistant:think(reasoning)|@assistant:answer",
    output="conversation"
)
```

Result:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "reasoning_content": "...",
      "content": ""
    },
    {"role": "assistant", "content": "..."}
  ]
}
```

## Custom Separator

The default separator is `|`. Optionally, you can use a custom separator:

```python
.render_conversation(
    conversation="@system:system\n@user:question\n@assistant:answer",
    separator="\n",  # Custom separator
    output="conversation"
)
```

This is useful when conversation definitions span multiple lines or when you prefer a different delimiter.

## Complete Example with Conv()

Here's a complete example using the Conv() builder to generate a function calling dataset:

```python
from tweaktune import Conv, Pipeline
from pydantic import Field
import os

def get_weather(location: str = Field(..., description="City name")):
    """Get current weather for a location."""
    pass

(Pipeline()
    .with_workers(3)
    .with_llm_openai("gpt4", os.environ["OPENAI_API_KEY"], "gpt-4o-mini")
    .with_tools_dataset("tools", [get_weather])

    .iter_range(10)
        # Sample a tool
        .sample_tools("tools", 1, "tool")

        # Create context data
        .add_column("system", lambda data: "You are a helpful assistant with access to tools.")
        .add_column("user_question", lambda data: "What's the weather in San Francisco?")
        .add_column("tool_args", lambda data: '{"location": "San Francisco"}')

        # Format tool call
        .render_tool_call(
            tool="tool[0].name",
            arguments="tool_args",
            output="tool_call"
        )

        # Mock tool response
        .add_column("tool_response", lambda data: '{"temp": 72, "condition": "sunny"}')
        .add_column("final_answer", lambda data: "It's sunny and 72°F in San Francisco!")

        # Build conversation using Conv() builder
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("user_question")
                .tool_calls(["tool_call"])
                .tool("tool_response")
                .assistant("final_answer"),
            tools="tool",
            output="conversation"
        )

        .write_jsonl(path="function_calling.jsonl", value="conversation")
    .run())
```

This produces conversations like:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What's the weather in San Francisco?"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "name": "get_weather",
            "arguments": {"location": "San Francisco"}
          }
        }
      ]
    },
    {"role": "tool", "content": "{\"temp\": 72, \"condition\": \"sunny\"}"},
    {"role": "assistant", "content": "It's sunny and 72°F in San Francisco!"}
  ],
  "tools": [...]
}
```

## Complete Example (String Format)

The same example using string format:

```python
from tweaktune import Pipeline

def get_weather(location: str = Field(..., description="City name")):
    """Get current weather."""
    pass

def set_alarm(
    time: str = Field(..., description="Time in HH:MM format"),
    label: Optional[str] = Field(None, description="Alarm label")
):
    """Set an alarm."""
    pass

(Pipeline()
    .with_workers(5)
    .with_llm_openai("gpt4", os.environ["OPENAI_API_KEY"], "gpt-4")
    .with_tools_dataset("tools", [get_weather, set_alarm])

    .with_template("system", "You are a helpful assistant with access to tools.")
    .with_template("user_prompt", "Generate a user request for: {{tool[0].description}}")

    .iter_range(100)
        # Sample a tool
        .sample_tools("tools", 1, "tool")

        # Generate user question
        .generate_text(
            template="user_prompt",
            llm="gpt4",
            output="user_question"
        )

        # Generate tool arguments
        .generate_json(
            template="Generate arguments for {{tool[0].name}}",
            llm="gpt4",
            output="arguments",
            response_format=tool[0].schema  # Use tool schema
        )

        # Format tool call
        .render_tool_call(
            tool="tool[0].name",
            arguments="arguments",
            output="tool_call"
        )

        # Mock tool response
        .add_column("tool_response", lambda data: '{"status": "success"}')

        # Generate final answer
        .generate_text(
            template="Based on {{tool_response}}, answer: {{user_question}}",
            llm="gpt4",
            output="assistant_answer"
        )

        # Build conversation
        .render_conversation(
            conversation="@system:system|@user:user_question|@assistant:tool_calls([tool_call])|@tool:tool_response|@assistant:assistant_answer",
            tools="tool",
            output="conversation"
        )

        .write_jsonl(path="function_calling.jsonl", value="conversation")
    .run())
```

## Rendered vs Structured Tool Calls

### Structured (default)

```python
.render_conversation(
    conversation="@assistant:tool_calls([call])",
    output="conv"
)
```

Result:
```json
{
  "role": "assistant",
  "tool_calls": [{"function": {"name": "...", "arguments": {...}}}]
}
```

### Rendered (custom format)

```python
.with_template("call_format", """<tool_call>{{call.function|tojson}}</tool_call>""")
.render("call_format", output="formatted_call")

.render_conversation(
    conversation="@assistant:formatted_call",
    output="conv"
)
```

Result:
```json
{
  "role": "assistant",
  "content": "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"
}
```

## Validating Conversations

Ensure conversation format is correct:

```python
.validate_conversation(instances="conversation")
```

Validates:
- Message structure
- Role values
- Required fields
- Tool call format

## Validating Tools

Check tool/function format:

```python
.validate_tools(instances="tools")
```

Ensures tools have:
- Name
- Description
- Parameters schema

## Pydantic Models for Schemas

Use Pydantic for structured output:

```python
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """Search results."""
    query: str = Field(..., description="Search query")
    results: list[dict] = Field(..., description="List of results")
    count: int = Field(..., description="Number of results")

.with_pydantic_models_dataset("schemas", [SearchResult])
```

## Reinforcement Learning Formats

Generate datasets for various RL fine-tuning methods.

### Supervised Fine-Tuning (SFT)

Format conversations for standard supervised fine-tuning:

```python
.add_column("system", lambda data: "You are a helpful assistant.")
.add_column("question", lambda data: "Hello, who won the world series in 2020?")
.add_column("call1", lambda data: {"name": "get_who_won", "arguments": {"year": 2020}})
.add_column("response", lambda data: '{"winner": "Los Angeles Dodgers", "year": 2020}')
.add_column("answer", lambda data: "The Los Angeles Dodgers won the World Series in 2020.")

.render_sft(
    conversation="@s:system|@u:question|@a:tool_calls([call1])|@t:response|@a:answer",
    output="conversation"
)
```

Result:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who won the world series in 2020?"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_who_won", "arguments": {"year": 2020}}}]},
    {"role": "tool", "content": "{\"winner\": \"Los Angeles Dodgers\", \"year\": 2020}"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."}
  ]
}
```

### Direct Preference Optimization (DPO)

Format conversations with chosen and rejected responses:

```python
.add_column("system", lambda data: "You are a helpful assistant.")
.add_column("question", lambda data: "Hello, who won the world series in 2020?")
.add_column("call1_chosen", lambda data: {"name": "get_who_won", "arguments": {"year": 2020}})
.add_column("call1_rejected", lambda data: {"name": "get_who_won", "arguments": {"year": 2021}})

.render_dpo(
    conversation="@s:system|@u:question",
    chosen="call1_chosen",
    rejected="call1_rejected",
    output="conversation"
)
```

Result:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who won the world series in 2020?"}
  ],
  "chosen": "<tool_call>{\"name\":\"get_who_won\",\"arguments\":{\"year\":2020}}</tool_call>",
  "rejected": "<tool_call>{\"name\":\"get_who_won\",\"arguments\":{\"year\":2021}}</tool_call>"
}
```

The `chosen` and `rejected` fields contain the preferred and non-preferred responses in tool call format.

### Group Relative Policy Optimization (GRPO)

Format conversations with solution and validator:

```python
.add_column("system", lambda data: "You are a helpful assistant.")
.add_column("question", lambda data: "Hello, who won the world series in 2020?")
.add_column("solution", lambda data: {"name": "get_who_won", "arguments": {"year": 2020}})

.render_grpo(
    conversation="@s:system|@u:question",
    solution="solution",
    validator_id="tool_use",
    output="conversation"
)
```

Result:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who won the world series in 2020?"}
  ],
  "solution": "{\"arguments\": {\"year\": 2020}, \"name\": \"get_who_won\"}",
  "validator_id": "tool_use"
}
```

The `solution` field contains the correct response, and `validator_id` identifies the validation method to use.

## Next Steps

- Learn about [Chat Templates](10-chat-templates.md)
- See [Validation & Quality](08-validation-quality.md)
- Explore [examples](/examples) for complete datasets
