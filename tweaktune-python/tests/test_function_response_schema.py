import json

from pydantic import BaseModel, Field

from tweaktune import Pipeline
from tweaktune.tools import function_to_schema


class MyResponse(BaseModel):
    id: int = Field(..., description="identifier")
    name: str


def my_tool(x: int) -> MyResponse:
    """Example tool that returns MyResponse"""
    return MyResponse(id=x, name=str(x))


def test_function_to_schema_includes_response():
    schema = function_to_schema(my_tool, include_response=True)
    assert schema["type"] == "function"
    assert schema["name"] == "my_tool"
    assert "parameters" in schema
    assert "response" in schema
    resp = schema["response"]
    assert resp["type"] == "json_schema"
    assert resp["name"] == "MyResponse"
    assert "schema" in resp


def test_tools_sample_normalized(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    OUTPUT_TEMPLATE = (
        """{"function": {{function[0]}}, "all_functions": {{normalized_all_functions|tojson}} }"""
    )

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_tools_dataset("functions", [my_tool])
        .with_template("output", OUTPUT_TEMPLATE)
        .iter_range(10)
        .sample_tools("functions", 1, "function")
        .sample_tools("functions", 1, "all_functions")
        .normalize_tools("function", "normalized_function")
        .normalize_tools("all_functions", "normalized_all_functions")
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    item = json.loads(lines[0])
    assert "function" in item
    assert "all_functions" in item
    assert len(lines) == 10
    print(json.loads(lines[0]))
