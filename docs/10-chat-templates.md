# Chat Templates

Use `ChatTemplateBuilder` to format conversations for model fine-tuning.

## Basic Usage

```python
from tweaktune import ChatTemplateBuilder

# Load a template
template_str = open("template.jinja").read()

chat_template = (ChatTemplateBuilder(template=template_str)
    .with_bos_token("<s>")
    .build())

# Render messages
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]

result = chat_template.render(messages)
```

## From File

```python
chat_template = (ChatTemplateBuilder(path="templates/chatml.jinja")
    .with_bos_token("<|im_start|>")
    .build())
```

## With Tools

Add function definitions:

```python
from pydantic import Field

def search_products(
    query: str = Field(..., description="Search query")
):
    """Search for products."""
    pass

chat_template = (ChatTemplateBuilder(template=template_str)
    .with_tools([search_products])
    .with_bos_token("<s>")
    .build())
```

Using JSON directly:

```python
tools_json = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

chat_template = (ChatTemplateBuilder(template=template_str)
    .with_tools_json(tools_json)
    .build())
```

## Rendering Messages

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What's 2+2?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "calculate",
                    "arguments": {"expression": "2+2"}
                }
            }
        ]
    },
    {"role": "tool", "content": '{"result": 4}'},
    {"role": "assistant", "content": "The answer is 4."}
]

result = chat_template.render(messages)
```

Result is a list of dicts with `"text"` key containing formatted conversation.

## With Tokenizer

Add HuggingFace tokenizer for training:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

chat_template = (ChatTemplateBuilder(template=template_str)
    .with_tokenizer(
        tokenizer=tokenizer,
        truncation=True,
        max_length=2048,
        padding=False
    )
    .build())

# Render and tokenize
result = chat_template.render(messages, tokenize=True)
# Returns dicts with input_ids, attention_mask, labels
```

## Render from JSONL

Process entire JSONL file:

```python
# conversations.jsonl contains one conversation per line:
# {"messages": [{"role": "user", "content": "..."}, ...]}
# {"messages": [{"role": "user", "content": "..."}, ...]}

dataset = chat_template.render_jsonl(
    path="conversations.jsonl",
    tokenize=True
)

# Returns HuggingFace Dataset ready for training
```

With configuration:

```python
config = {
    "max_length": 2048,
    "truncation": True
}

dataset = chat_template.render_jsonl(
    path="conversations.jsonl",
    op_config=config,
    tokenize=True
)
```

## Built-in Templates

tweaktune includes pre-built templates:

```python
from tweaktune.chat_templates import bielik

chat_template = (ChatTemplateBuilder(template=bielik)
    .with_tools([my_function])
    .build())
```

## Example: Format Training Data

```python
from tweaktune import ChatTemplateBuilder
from tweaktune.chat_templates import bielik
from pydantic import Field

def get_weather(location: str = Field(..., description="City name")):
    """Get weather information."""
    pass

# Build template
chat_template = (ChatTemplateBuilder(template=bielik)
    .with_tools([get_weather])
    .with_bos_token("<s>")
    .build())

# Render conversation
messages = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "tool_calls": [{
            "function": {
                "name": "get_weather",
                "arguments": {"location": "Paris"}
            }
        }]
    },
    {"role": "tool", "content": '{"temp": 18, "condition": "sunny"}'},
    {"role": "assistant", "content": "It's sunny and 18°C in Paris."}
]

formatted = chat_template.render(messages)
print(formatted[0]["text"])
```

## Create Training Dataset

```python
from tweaktune import Pipeline, ChatTemplateBuilder
from transformers import AutoTokenizer

# Prepare chat template
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
chat_template = (ChatTemplateBuilder(path="template.jinja")
    .with_tokenizer(tokenizer, truncation=True, max_length=2048, padding=False)
    .build())

# Generate conversations with pipeline
(Pipeline()
    .with_llm_openai("gpt4", api_key, "gpt-4")
    .iter_range(1000)
        # ... generate conversation ...
        .write_jsonl(path="conversations.jsonl", value="conversation")
    .run())

# Format for training
train_dataset = chat_template.render_jsonl(
    path="conversations.jsonl",
    tokenize=True
)

# Use with your training framework
# train_dataset.push_to_hub("username/dataset")
# or
# trainer = Trainer(train_dataset=train_dataset, ...)
```

## Template Variables

Templates have access to:

- `messages` - List of message dicts
- `tools` - List of tool definitions (if provided)
- `bos_token` - Beginning of sequence token
- `add_generation_prompt` - Boolean (optional)

Example template:

```jinja
{% for message in messages %}
{{bos_token}}{{message.role}}
{{message.content}}
{% if message.tool_calls %}
{% for call in message.tool_calls %}
<tool_call>{{call.function.name}}({{call.function.arguments|tojson}})</tool_call>
{% endfor %}
{% endif %}
{% endfor %}
```

## Best Practices

1. **Match your model's format** - Use the template for your specific model
2. **Include BOS token** - Most models need it
3. **Test rendering** - Verify output before training
4. **Use truncation** - Prevent overly long sequences
5. **Tokenize once** - Do it during data prep, not training
6. **Validate messages** - Ensure proper format before rendering

## Next Steps

- Learn about [Metadata & Tracking](11-metadata-tracking.md)
- See [Conversation & Tools](09-conversation-tools.md)
- Explore [examples](/examples) for complete workflows
