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

Build OpenAI-style conversations:

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

## Role Aliases

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

## Tool Calls in Conversations

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

## Reasoning Content

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

## Complete Example

Generate function calling dataset:

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
