import json

from tweaktune import Pipeline


def test_tojson(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | tojson}}")
        .iter_range(number)
        .add_column("value", lambda data: {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}, "bool": True, "none": None})
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert lines[0] == '{"bool":true,"key":"value","list":[1,2,3],"nested":{"a":1},"none":null}\n'

def test_totoon(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | totoon}}")
        .iter_range(number)
        .add_column("value", lambda data: {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}, "bool": True, "none": None})
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    print(lines)
    assert lines[0] == 'bool: true\\nkey: value\\nlist[3]: 1,2,3\\nnested:\\n  a: 1\\nnone: null\n'

def test_jstr(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | jstr}}")
        .iter_range(number)
        .add_column("value", lambda data: {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}, "bool": True, "none": None})
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    print(lines)
    assert lines[0] == 'bool: true\\nkey: value\\nlist[3]: 1,2,3\\nnested:\\n  a: 1\\nnone: null\n'

