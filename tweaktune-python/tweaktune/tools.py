import re
import json
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
import inspect
from typing import Callable, Dict, Any, List

def class_to_schema(model: BaseModel) -> dict:
    """
    Converts a Pydantic model to JSON schema.
    """
    schema = model.model_json_schema()
    schema = normalize_schema(schema)
    name = schema.pop("title", None)

    return {
        "type": "json_schema",
        "name": name,
        "schema": schema,
        "strict": True,
    }

def pydantic_to_json_schema(model: BaseModel) -> dict:
    return json.dumps(class_to_schema(model), ensure_ascii=False)

def function_to_schema(func: callable) -> BaseModel:
    """
    Converts a function with annotated parameters to json schema https://json-schema.org/
    including descriptions from Field(..., description=...).
    """
    func_name = func.__name__
    func_params = func.__annotations__

    sig = inspect.signature(func)
    model_fields = {}
    param_descriptions = {}

    for param_name, param in sig.parameters.items():
        if param_name in ["args", "kwargs"]:
            continue
        if param_name == "return":
            continue

        param_type = func_params.get(param_name, param.annotation)
        default = param.default

        if isinstance(default, FieldInfo):
            description = default.description
            model_fields[param_name] = (param_type, default)
            if description:
                param_descriptions[param_name] = description
        else:
            model_fields[param_name] = (param_type, Field(...))

    schema = create_model(func_name, **model_fields).model_json_schema()
    schema = normalize_schema(schema)
    schema.pop("title", None)

    return {
        "type": "function",
        "name": func_name,
        "description": func.__doc__,
        "parameters": schema,
        "strict": True,
    }

def function_to_json_schema(func: callable) -> BaseModel:
    return json.dumps(function_to_schema(func), ensure_ascii=False)

def normalize_schema(schema):
    defs = schema.pop("$defs", {})

    if "properties" in schema:
        for prop in schema["properties"].values():
            if "$ref" in prop:
                ref_path = prop.pop("$ref")
                if ref_path.startswith("#/$defs/"):
                    def_key = ref_path.split("/")[-1]
                    if def_key in defs:
                        prop.update(defs[def_key])

            if "anyOf" in prop:
                prop["type"] = [t["type"] for t in prop.pop("anyOf")]

            prop.pop("title", None)
        schema["properties"] = json.dumps(schema["properties"], ensure_ascii=False)
    schema["additionalProperties"] = False
    return schema

class ToolFunction(BaseModel):
    name: str = Field(..., description="Name of the tool function")
    arguments: Dict[str,Any] = Field(..., description="Parameters for the tool function in JSON schema format")

class ToolCall(BaseModel):
    function: ToolFunction = Field(..., description="Name of the function to call")

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system', 'tool')")
    content: str | None = Field(..., description="Content of the message")
    tool_calls: List[ToolCall] | None = Field(None, description="List of tool calls made in this message")

class Tools:
    def __init__(self, tools: list[Callable]):
        self._tools = [(function_to_schema(func), func) for func in tools]
        self._tools = {tool[0]["name"]:tool for tool in self._tools }
        self._messages = []

    @property
    def tools(self):
        return [tool[0] for tool in self._tools.values()]
    
    @property
    def messages(self):
        return self._messages
    
    @messages.setter
    def messages(self, messages: list):
        self._messages = [Message(**m) for m in messages]

    def __getitem__(self, item):
        if item in self._tools:
            return self._tools[item]
        raise KeyError(f"Tool {item} not found. Available tools: {list(self._tools.keys())}")

    def __call__(self, response: str):
        try:
            matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
            results = []
            tool_calls = []
            for m in matches:
                data = json.loads(m.strip())
                name = data.get("name")
                tool_calls.append(ToolCall(function=ToolFunction(**data)))
                
                if name in self._tools:
                    tool = self._tools[name]
                    func = tool[1]
                    results.append(func(**data.get("arguments", {})))

            self._messages.append(Message(role="assistant", content=None, tool_calls=tool_calls))
            for result in results:
                self._messages.append(Message(role="tool", content=json.dumps(result, ensure_ascii=False), tool_calls=None))
                
            return results
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            raise ValueError(f"Invalid tool call format: {response}") from e

