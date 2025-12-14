# Function Calling / Tool Use Examples

This document provides examples of generating function calling datasets using tweaktune.

## Basic Function Calling from Python Functions

Generate tool use conversations from Python functions:

```python
from pydantic import Field
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def get_weather(
    location: str = Field(..., description="City name"),
    unit: str = Field("celsius", description="Temperature unit: celsius or fahrenheit")
):
    """Get current weather for a location"""
    pass

def search_web(
    query: str = Field(..., description="Search query"),
    num_results: int = Field(5, description="Number of results to return")
):
    """Search the web for information"""
    pass

def calculate(
    expression: str = Field(..., description="Mathematical expression to evaluate")
):
    """Calculate a mathematical expression"""
    pass

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/function_calling.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="function-calling-generation", metadata=Metadata(description="Generate tool use data"))
        .with_workers(4)
        .with_tools_dataset("available_tools", [get_weather, search_web, calculate])
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("question_prompt", "Generate a question that requires using this tool: {{selected_tools[0].name}}")
        .with_template("args_prompt", "Generate appropriate arguments for calling {{selected_tools[0].name}} to answer: {{question}}")
        .iter_range(100)
        # Sample tools
        .sample_tools("available_tools", size=1, output="selected_tools")
        .add_literal("system", "You are a helpful assistant with access to tools.")
        # Generate question
        .generate_text(template="question_prompt", llm="gpt4", output="question")
        # Generate tool call arguments
        .generate_json(
            template="args_prompt",
            llm="gpt4",
            output="tool_args",
            json_path="$"
        )
        # Render tool call
        .render_tool_call(
            tool="selected_tools[0].name",
            arguments="tool_args",
            output="tool_call"
        )
        # Simulate tool response
        .add_literal("tool_response", '{"result": "Mocked response"}')
        # Generate final answer
        .add_literal("final_answer", "Based on the tool response, here is the answer.")
        # Build conversation
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("question")
                .tool_calls(["tool_call"])
                .tool("tool_response")
                .assistant("final_answer"),
            tools="selected_tools",
            output="conversation"
        )
        .validate_tools("selected_tools")
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

**Output Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {
      "role": "assistant",
      "tool_calls": [
        {"function": {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}}
      ]
    },
    {"role": "tool", "content": "{\"result\": \"Mocked response\"}"},
    {"role": "assistant", "content": "Based on the tool response, here is the answer."}
  ],
  "tools": [...]
}
```

## From OpenAPI Specification

Load tools from OpenAPI/Swagger specs:

```python
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/openapi_function_calling.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="openapi-tool-generation", metadata=Metadata(description="Generate from OpenAPI"))
        .with_workers(4)
        .with_openapi_dataset("api_tools", "openapi.json")
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("scenario_prompt", "Generate a realistic scenario requiring API endpoint: {{selected_tools[0].name}}")
        .with_template("args_prompt", "Generate API call arguments for {{selected_tools[0].name}} in this scenario: {{scenario}}")
        .iter_range(50)
        # Sample API endpoints
        .sample_tools("api_tools", size=1, output="selected_tools")
        .add_literal("system", "You are an API assistant.")
        # Generate scenario
        .generate_text(template="scenario_prompt", llm="gpt4", output="scenario")
        # Generate arguments
        .generate_json(template="args_prompt", llm="gpt4", output="api_args", json_path="$")
        # Render tool call
        .render_tool_call(tool="selected_tools[0].name", arguments="api_args", output="api_call")
        # Mock response
        .add_literal("api_response", '{"status": "success"}')
        .add_literal("answer", "API call completed successfully.")
        # Build conversation
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("scenario")
                .tool_calls(["api_call"])
                .tool("api_response")
                .assistant("answer"),
            tools="selected_tools",
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

## Multi-Tool Conversations

Generate conversations using multiple tools:

```python
from pydantic import Field
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def get_user_info(user_id: str = Field(..., description="User ID")):
    """Get user information"""
    pass

def get_order_history(user_id: str = Field(..., description="User ID")):
    """Get user's order history"""
    pass

def recommend_products(user_id: str = Field(..., description="User ID")):
    """Recommend products based on user history"""
    pass

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/multi_tool_conversations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="multi-tool-conversations", metadata=Metadata(description="Multi-tool use"))
        .with_workers(4)
        .with_tools_dataset("tools", [get_user_info, get_order_history, recommend_products])
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("task_prompt", "Generate a customer service task requiring multiple tool calls")
        .with_template("call1_args_prompt", "Generate arguments for first tool call: {{selected_tools[0].name}} for task: {{task}}")
        .with_template("call2_args_prompt", "Generate arguments for second tool call: {{selected_tools[1].name}} based on previous response")
        .iter_range(50)
        # Sample multiple tools
        .sample_tools("tools", size=3, output="selected_tools")
        .add_literal("system", "You are a customer service assistant.")
        # Generate task
        .generate_text(template="task_prompt", llm="gpt4", output="task")
        # First tool call
        .generate_json(template="call1_args_prompt", llm="gpt4", output="args1", json_path="$")
        .render_tool_call(tool="selected_tools[0].name", arguments="args1", output="call1")
        .add_literal("response1", '{"user_id": "123", "name": "John"}')
        # Second tool call
        .generate_json(template="call2_args_prompt", llm="gpt4", output="args2", json_path="$")
        .render_tool_call(tool="selected_tools[1].name", arguments="args2", output="call2")
        .add_literal("response2", '{"orders": []}')
        # Final answer
        .add_literal("answer", "Based on user info and order history, here's what I found.")
        # Build conversation
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("task")
                .tool_calls(["call1"])
                .tool("response1")
                .tool_calls(["call2"])
                .tool("response2")
                .assistant("answer"),
            tools="selected_tools",
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

## From Pydantic Models

Convert Pydantic models to tool definitions:

```python
from pydantic import BaseModel, Field
from typing import List
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

class CreateUser(BaseModel):
    """Create a new user"""
    username: str = Field(description="Unique username")
    email: str = Field(description="Email address")
    age: int = Field(description="User age", ge=18)

class SearchUsers(BaseModel):
    """Search for users"""
    query: str = Field(description="Search query")
    filters: List[str] = Field(default=[], description="Filter criteria")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/pydantic_tools.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="pydantic-tool-generation", metadata=Metadata(description="Pydantic tools"))
        .with_workers(4)
        .with_pydantic_models_dataset("tools", [CreateUser, SearchUsers])
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("request_prompt", "Generate a request using tool: {{selected_tools[0].name}}")
        .with_template("args_prompt", "Generate arguments for {{selected_tools[0].name}} to fulfill: {{request}}")
        .iter_range(50)
        .sample_tools("tools", size=1, output="selected_tools")
        .add_literal("system", "You are a system administrator assistant.")
        # Generate request
        .generate_text(template="request_prompt", llm="gpt4", output="request")
        # Generate arguments
        .generate_json(template="args_prompt", llm="gpt4", output="args", json_path="$")
        # Render tool call
        .render_tool_call(tool="selected_tools[0].name", arguments="args", output="tool_call")
        .add_literal("response", '{"status": "success"}')
        .add_literal("answer", "Operation completed.")
        # Build conversation
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("request")
                .tool_calls(["tool_call"])
                .tool("response")
                .assistant("answer"),
            tools="selected_tools",
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

## With Seed Data (Task-Based)

Generate tool use from task descriptions:

```python
from pydantic import Field
from tweaktune import Pipeline, Metadata, Conv
import os
from pathlib import Path

def send_email(to: str, subject: str, body: str):
    """Send an email"""
    pass

def schedule_meeting(attendees: list, date: str, time: str):
    """Schedule a meeting"""
    pass

def create_task(title: str, description: str, deadline: str):
    """Create a task"""
    pass

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    output_path = Path("output/task_based_tools.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    (Pipeline(name="task-based-tools", metadata=Metadata(description="Task-based tool use"))
        .with_workers(4)
        .with_jsonl_dataset("tasks", "tasks.jsonl")  # {task: "Schedule a meeting with team"}
        .with_tools_dataset("tools", [send_email, schedule_meeting, create_task])
        .with_llm_openai("gpt4", api_key, "gpt-4")
        .with_template("tool_selection_prompt", "Which tool should be used for: {{task}}? Choose from: {{available_tools}}")
        .with_template("args_prompt", "Generate arguments for {{selected_tool}} to complete: {{task}}")
        .iter_dataset("tasks")
        .add_literal("available_tools", "send_email, schedule_meeting, create_task")
        .add_literal("system", "You are a productivity assistant.")
        # Determine which tool to use
        .generate_text(template="tool_selection_prompt", llm="gpt4", output="selected_tool")
        # Generate arguments
        .generate_json(template="args_prompt", llm="gpt4", output="args", json_path="$")
        # Get the actual tool object (simplified - in practice, you'd match the name)
        .sample_tools("tools", size=1, output="tools_for_render")
        # Render tool call
        .render_tool_call(tool="tools_for_render[0].name", arguments="args", output="tool_call")
        .add_literal("tool_response", '{"status": "completed"}')
        .add_literal("answer", "Task completed successfully.")
        # Build conversation
        .render_conversation(
            conversation=Conv()
                .system("system")
                .user("task")
                .tool_calls(["tool_call"])
                .tool("tool_response")
                .assistant("answer"),
            tools="tools_for_render",
            output="conversation"
        )
        .validate_conversation("conversation")
        .write_jsonl(path=str(output_path), value="conversation")
        .run()
    )

if __name__ == "__main__":
    main()
```

## Best Practices

1. **Tool Definitions**: Use Pydantic Field descriptions for clear tool documentation
2. **Tool Sampling**: Use `.sample_tools()` to select appropriate tools per conversation
3. **Validation**: Always validate with `.validate_tools()` and `.validate_conversation()`
4. **Realistic Arguments**: Generate context-appropriate arguments for tool calls
5. **Tool Responses**: Simulate realistic tool responses (or use real API calls)
6. **Multi-tool**: Include conversations with multiple sequential tool calls
7. **System Prompt**: Clearly state tool availability in system message
8. **Output Format**: Use `.render_conversation()` with `tools` parameter for OpenAI format
9. **Error Cases**: Consider generating examples with tool errors/failures
10. **Diversity**: Sample different tools and argument combinations

## Tool Dataset Methods

- `.with_tools_dataset(name, [func1, func2])` - From Python functions
- `.with_openapi_dataset(name, "spec.json")` - From OpenAPI/Swagger
- `.with_pydantic_models_dataset(name, [Model1, Model2])` - From Pydantic models

## Sampling Tools

```python
.sample_tools("dataset", size=3, output="selected_tools")  # Random sampling
```

## Rendering Tool Calls

```python
.render_tool_call(
    tool="tool_name",           # Tool name (string or reference)
    arguments="args_json",      # JSON string of arguments
    output="tool_call"          # Output field name
)
```

## Reference

For comprehensive examples, see:
- `/home/jovyan/SpeakLeash/tweaktune/tweaktune-python/tests/test_tools.py`
- `/home/jovyan/SpeakLeash/tweaktune/tweaktune-python/tests/test_steps.py` (lines 256-391)
