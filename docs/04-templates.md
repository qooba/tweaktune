# Templates

tweaktune uses Jinja2 templating for dynamic content generation. Templates can be defined inline, loaded from files, or organized in directories.

## Inline Templates

Define templates directly in your code:

```python
(Pipeline()
    .with_template("greeting", "Hello {{name}}!")
    .with_template("output", """{"message": "{{greeting}}"}""")
    .iter_range(5)
        .add_column("name", lambda data: f"Person {data['index']}")
        .render("greeting", output="greeting")
        .write_jsonl(path="output.jsonl", template="output")
    .run())
```

## Template from File

Load a single Jinja2 template from file:

```python
# templates/my_template.j2
# {"name": "{{name}}", "value": {{value}}}

.with_j2_template("output", "templates/my_template.j2")
```

## Templates from Directory

Load all `.j2` files from a directory:

```python
# Directory structure:
# templates/
#   greeting.j2
#   output.j2
#   system.j2

.with_templates("templates")

.iter_range(5)
    .render("greeting.j2", output="msg")  # Use filename as template name
```

Alternative method with full path specification:

```python
.with_j2_templates("templates")
```

## Templates from YAML

Define multiple templates in a YAML file:

```python
# templates.yaml
# templates:
#   greeting: "Hello {{name}}!"
#   output: |
#     {"message": "{{greeting}}", "id": {{id}}}

.with_j2_templates("templates.yaml")
```

## Template Configuration

Pass configuration to templates using `op_config`:

```python
config = {
    "system_prompt": "You are a helpful assistant.",
    "max_length": 100
}

.with_j2_template("prompt", "templates/prompt.j2", op_config=config)
```

Access config in template:

```jinja2
{{op_config.system_prompt}}

Max length: {{op_config.max_length}}
```

## Using Templates

### In write_jsonl

```python
.with_template("output", """{"result": {{value}}}""")
.write_jsonl(path="output.jsonl", template="output")
```

### In render Step

```python
.with_template("greeting", "Hello {{name}}!")
.render(template="greeting", output="message")
```

### Direct Value Output

Skip template rendering and write a value directly:

```python
.write_jsonl(path="output.jsonl", value="my_data")
```

## Template Variables

### Context Data

Access context data in templates:

```python
.add_column("name", lambda data: "Alice")
.add_column("age", lambda data: 25)
.with_template("output", """{"name": "{{name}}", "age": {{age}}}""")
```

### Arrays and Objects

```python
.sample(dataset="items", size=3, output="items")
.with_template("output", """{"first_item": "{{items[0]}}"}""")
```

### Index Variable

When using `iter_range`, the `index` is available:

```python
.with_template("output", """{"id": {{index}}, "value": "item_{{index}}"}""")
```

## Custom Filters

tweaktune provides custom Jinja filters:

### jstr

Serializes JSON objects to properly escaped strings:

```python
# Use for JSON objects that need to be serialized to strings
.with_template("output", """{"metadata": {{json_object|jstr}}}""")

# For simple string values, use quotes directly
.with_template("output", """{"text": "{{simple_string}}"}""")
```

The `jstr` filter performs double JSON serialization, converting a JSON object into a properly escaped string representation. Use it only when you need to serialize complex JSON structures. For simple string values, use regular quoted interpolation like `"{{variable}}"`.

### tojson

Convert to JSON:

```python
.with_template("output", """{"items": {{my_list|tojson}}}""")
```

### random_range

Generate random integers:

```python
.with_template("output", """{"random": {{"1,100"|random_range}}}""")
# Generates random number between 1 and 100
```

## Multi-line Templates

Use triple quotes for readability:

```python
.with_template("system", """You are a helpful AI assistant.
You should always be polite and professional.
Answer questions concisely.""")

.with_template("output", """
{
  "messages": [
    {"role": "system", "content": "{{system}}"},
    {"role": "user", "content": "{{question}}"},
    {"role": "assistant", "content": "{{answer}}"}
  ]
}
""")
```

## Template Examples

### Simple Key-Value

```python
.with_template("output", """{"key": "{{value}}"}""")
```

### Nested JSON

```python
.with_template("output", """
{
  "user": {
    "name": "{{name}}",
    "age": {{age}}
  },
  "items": {{items|tojson}}
}
""")
```

### Conditional Content

```python
.with_template("output", """
{
  "name": "{{name}}"
  {% if age %}
  , "age": {{age}}
  {% endif %}
}
""")
```

### Loops

```python
.with_template("output", """
{
  "items": [
    {% for item in items %}
    {"name": "{{item.name}}", "price": {{item.price}}}
    {% if not loop.last %},{% endif %}
    {% endfor %}
  ]
}
""")
```

### Conversation Format

```python
.with_template("chat", """
[
  {"role": "system", "content": "{{system}}"},
  {"role": "user", "content": "{{question}}"},
  {"role": "assistant", "content": "{{answer}}"}
]
""")
```

## Best Practices

1. **Use simple strings `"{{variable}}"` for string values** - only use `jstr` when serializing JSON objects to strings
2. **Use `tojson` for complex objects** like arrays and nested dictionaries
3. **Validate JSON** using an online validator or `json.loads()` when testing
4. **Keep templates readable** with proper indentation and line breaks
5. **Store complex templates in files** rather than inline strings
6. **Use YAML for template collections** to keep related templates together

## Next Steps

- Learn about [LLM Integration](05-llm-integration.md)
- See [Conversation & Tools](09-conversation-tools.md) for conversation templates
- Explore [Chat Templates](10-chat-templates.md) for model-specific formatting
