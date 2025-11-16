# Pipeline Steps

Complete reference of all available pipeline steps.

## Data Steps

### sample

Sample random items from a dataset:

```python
.sample(dataset="products", size=3, output="sampled_products")
```

Parameters:
- `dataset` - Dataset name
- `size` - Number of items to sample
- `output` - Output variable name

### sample_tools

Sample tools (similar to sample but preserves tool structure):

```python
.sample_tools(dataset="tools", size=2, output="selected_tools")
```

### read

Read entire dataset (not recommended for large datasets):

```python
.read(dataset="items", output="all_items")
```

## Transformation Steps

### add_column

Add a new column using lambda or expression:

```python
# Using lambda
.add_column("greeting", lambda data: f"Hello {data['name']}")

# Using expression
.add_column("total", "price * quantity", is_json=False)

# Using expression with JSON
.add_column("metadata", "{'type': 'product', 'id': index}")
```

Parameters:
- `output` - Column name
- `func` - Lambda function or string expression
- `is_json` - Whether expression returns JSON (default: True)

### add_columns

Add multiple columns at once:

```python
.add_columns({
    "full_name": lambda data: f"{data['first']} {data['last']}",
    "age_group": "age // 10 * 10"
})
```

### add_random

Add a random integer:

```python
.add_random(output="random_value", start=1, stop=100)
```

### mutate

Modify an existing column:

```python
# Using lambda
.mutate("price", lambda price: price * 1.1)

# Using expression
.mutate("price", "price * 1.1", is_json=False)
```

### into_list

Combine multiple columns into a list:

```python
.add_column("a", lambda data: 1)
.add_column("b", lambda data: 2)
.add_column("c", lambda data: 3)
.into_list(inputs=["a", "b", "c"], output="my_list")
# Result: my_list = [1, 2, 3]
```

### chunk

Split text into chunks:

```python
.chunk(
    capacity=(100, 200),  # Min 100, max 200 chars
    input="long_text",
    output="chunks"
)
```

## Filtering Steps

### filter

Filter rows based on condition:

```python
# Using lambda
.filter(lambda data: data["age"] > 18)

# Using expression
.filter("age > 18")
```

## Control Flow Steps

### ifelse

Conditional branching:

```python
from tweaktune.chain import Chain

.ifelse(
    condition=lambda data: data["value"] > 10,
    then_chain=Chain()
        .add_column("category", lambda data: "high")
        .add_column("discount", lambda data: 0.2),
    else_chain=Chain()
        .add_column("category", lambda data: "low")
        .add_column("discount", lambda data: 0.0)
)
```

Using expression:

```python
.ifelse(
    condition="value > 10",
    then_chain=Chain().add_column("result", lambda data: "pass"),
    else_chain=Chain().add_column("result", lambda data: "fail")
)
```

### map

Apply custom function to context:

```python
def process(context):
    context["data"]["processed"] = True
    context["data"]["timestamp"] = time.time()
    return context

.map(process)
```

### step

Add custom step class:

```python
class MyStep:
    def process(self, context):
        context["data"]["custom"] = "value"
        return context

.step(MyStep())
```

## Rendering Steps

### render

Render a template to a variable:

```python
.with_template("greeting", "Hello {{name}}!")
.render(template="greeting", output="message")
```

### render_conversation

Render conversation in OpenAI format:

```python
.render_conversation(
    conversation="""
        @system:system_prompt
        @user:user_message
        @assistant:assistant_response
    """,
    output="conversation"
)
```

Format: `@role:variable_name` separated by `|` (or custom separator)

Role aliases:
- `@system` or `@s` - System message
- `@user` or `@u` - User message
- `@assistant` or `@a` - Assistant message
- `@tool` or `@t` - Tool result

Special formats:
- `@assistant:tool_calls([call1, call2])` - Tool calls
- `@assistant:think(reasoning)` - Reasoning content

```python
.render_conversation(
    conversation="""
        @s:system
        @u:question
        @a:tool_calls([tool_call1, tool_call2])
        @t:tool_result
        @a:think(reasoning)
        @a:final_answer
    """,
    output="messages",
    tools="available_tools",  # Optional tools list
    separator="\n"  # Default: "|"
)
```

### render_tool_call

Format a tool call:

```python
.render_tool_call(
    tool="tool_name",
    arguments="arguments_json",
    output="formatted_call"
)

# Or extract tool name from variable
.render_tool_call(
    tool="tools[0].name",
    arguments="call_args",
    output="tool_call"
)
```

### render_sft

Render conversation for Supervised Fine-Tuning (SFT):

```python
.render_sft(
    conversation="""
        @s:system
        @u:question
        @a:tool_calls([call1])
        @t:response
        @a:answer
    """,
    output="conversation",
    separator="\n"
)
```

Same format as `render_conversation` - creates standard OpenAI-style conversation format.

### render_dpo

Render conversation for Direct Preference Optimization (DPO):

```python
.render_dpo(
    conversation="""
        @s:system
        @u:question
    """,
    chosen="call1_chosen",
    rejected="call1_rejected",
    output="conversation",
    separator="\n"
)
```

Creates conversation with `chosen` and `rejected` fields for preference learning.

Parameters:
- `conversation` - Conversation template (context messages)
- `chosen` - Variable containing the preferred response
- `rejected` - Variable containing the non-preferred response
- `output` - Output variable name
- `separator` - Message separator (default: "|")

### render_grpo

Render conversation for Group Relative Policy Optimization (GRPO):

```python
.render_grpo(
    conversation="""
        @s:system
        @u:question
    """,
    solution="solution",
    validator_id="tool_use",
    output="conversation",
    separator="\n"
)
```

Creates conversation with `solution` and `validator_id` fields for RL training.

Parameters:
- `conversation` - Conversation template (context messages)
- `solution` - Variable containing the correct solution
- `validator_id` - Identifier for the validation method
- `output` - Output variable name
- `separator` - Message separator (default: "|")

## Generation Steps

### generate_text

Generate text using LLM:

```python
.generate_text(
    template="prompt_template",
    llm="gpt4",
    output="generated_text",
    system_template="system_prompt",  # Optional
    max_tokens=1024,
    temperature=0.7
)
```

### generate_json

Generate JSON using LLM:

```python
.generate_json(
    template="prompt_template",
    llm="gpt4",
    output="result",
    json_path="data",  # Extract field from response
    system_template="system_prompt",  # Optional
    max_tokens=500,
    temperature=0.7
)
```

### generate_structured

Generate with Pydantic schema:

```python
from pydantic import BaseModel, Field

class Response(BaseModel):
    title: str = Field(..., description="Title")
    score: int = Field(..., ge=1, le=5)

.generate_structured(
    template="prompt",
    llm="gpt4",
    output="result",
    response_format=Response,
    max_tokens=200,
    temperature=0.7
)
```

## Validation Steps

### validate_json

Validate JSON against schema:

```python
.validate_json(
    schema="json_schema",  # Variable containing JSON schema
    instance="data_to_validate"
)
```

### validate_tools

Validate tool/function calling format:

```python
.validate_tools(instances="tool_calls")
```

### validate_conversation

Validate conversation format:

```python
.validate_conversation(instances="messages")
```

### validate

Custom Python validator:

```python
def my_validator(context):
    data = context["data"]
    if "required_field" not in data:
        raise ValueError("Missing required field")
    if data["value"] < 0:
        raise ValueError("Value must be positive")
    return True

.validate(my_validator)
```

## Quality Steps

### check_hash

Deduplicate by exact hash:

```python
.check_hash(input="content_to_check")
```

Marks item as failed if hash was seen before (when metadata is enabled).

### check_simhash

Fuzzy deduplication using simhash:

```python
.check_simhash(
    input="text_content",
    threshold=3  # Hamming distance threshold
)
```

Lower threshold = stricter matching.

### check_embedding

Semantic similarity deduplication:

```python
.check_embedding(
    input="text_content",
    embedding="e5-small",  # Embedding model name
    threshold=0.01,  # Cosine distance threshold
    similarity_output="similarity_score"  # Optional
)
```

### check_language

Filter by language:

```python
.check_language(
    input="text",
    language="english",
    precision=0.9,
    detect_languages=["english", "french", "german", "spanish", "polish"]
)
```

### normalize_tools

Normalize tool format:

```python
.normalize_tools(
    instances="raw_tools",
    output="normalized_tools"
)
```

## Judge Steps

### judge_conversation

Evaluate conversation quality:

```python
from tweaktune import JudgeType

.judge_conversation(
    input="conversation",
    llm="gpt4",
    output="evaluation",
    language="en",
    judge_type=JudgeType.ToolsCalling,
    attach_to_conversation=False,  # Add scores to conversation
    max_tokens=500,
    temperature=0.1
)
```

Judge types:
- `JudgeType.ToolsCalling` - Evaluate tool usage
- `JudgeType.Conversation` - General conversation quality

## Output Steps

### write_jsonl

Write to JSONL file:

```python
# Using template
.write_jsonl(path="output.jsonl", template="output_template")

# Using value directly
.write_jsonl(path="output.jsonl", value="data_variable")
```

### write_csv

Write to CSV file:

```python
.write_csv(
    path="output.csv",
    columns=["name", "age", "city"],
    delimiter=","
)
```

### print

Print values:

```python
# Print specific columns
.print(columns=["name", "value"])

# Print rendered template
.print(template="output_template")
```

## Logging Steps

### log

Set log level:

```python
.log("debug")  # Levels: debug, info, warn, error
```

### debug

Shortcut for debug logging:

```python
.debug()  # Equivalent to .log("debug")
```

## Next Steps

- Learn about [Custom Steps](07-custom-steps.md)
- See [Validation & Quality](08-validation-quality.md)
- Explore [Conversation & Tools](09-conversation-tools.md)
