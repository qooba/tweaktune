import json

from tweaktune import Pipeline


def test_tojson(request, output_dir, metadata):
    """Test tojson filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | tojson}}")
        .iter_range(number)
        .add_column(
            "value",
            lambda data: {
                "key": "value",
                "list": [1, 2, 3],
                "nested": {"a": 1},
                "bool": True,
                "none": None,
            },
        )
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert lines[0] == '{"bool":true,"key":"value","list":[1,2,3],"nested":{"a":1},"none":null}\n'


def test_totoon(request, output_dir, metadata):
    """Test totoon filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | totoon}}")
        .iter_range(number)
        .add_column(
            "value",
            lambda data: {
                "key": "value",
                "list": [1, 2, 3],
                "nested": {"a": 1},
                "bool": True,
                "none": None,
            },
        )
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert lines[0] == "bool: true\\nkey: value\\nlist[3]: 1,2,3\\nnested:\\n  a: 1\\nnone: null\n"


def test_jstr(request, output_dir, metadata):
    """Test jstr filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | jstr}}")
        .iter_range(number)
        .add_column(
            "value",
            lambda data: {
                "key": "value",
                "list": [1, 2, 3],
                "nested": {"a": 1},
                "bool": True,
                "none": None,
            },
        )
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert (
        lines[0]
        == '"{\\"bool\\":true,\\"key\\":\\"value\\",\\"list\\":[1,2,3],\\"nested\\":{\\"a\\":1},\\"none\\":null}"\n'
    )


def test_hash(request, output_dir, metadata):
    """Test hash filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | hash}}")
        .iter_range(number)
        .add_column(
            "value",
            lambda data: {
                "key": "value",
                "list": [1, 2, 3],
                "nested": {"a": 1},
                "bool": True,
                "none": None,
            },
        )
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert lines[0] == "926ab999\n"


def test_blake3_hash(request, output_dir, metadata):
    """Test blake3_hash filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | blake3_hash}}")
        .iter_range(number)
        .add_column(
            "value",
            lambda data: {
                "key": "value",
                "list": [1, 2, 3],
                "nested": {"a": 1},
                "bool": True,
                "none": None,
            },
        )
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert lines[0] == "027df616bce2329eee2a253cd03e9bf227bb2f74882fb5387ef8d6ab3c5619a3\n"


def test_random_range(request, output_dir, metadata):
    """Test random_range filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{'1,10' | random_range}}")
        .iter_range(number)
        .add_column(
            "value",
            lambda data: {
                "key": "value",
                "list": [1, 2, 3],
                "nested": {"a": 1},
                "bool": True,
                "none": None,
            },
        )
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert int(lines[0]) >= 1 and int(lines[0]) < 10


def test_shuffle(request, output_dir, metadata):
    """Test shuffle filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | shuffle}}")
        .iter_range(number)
        .add_column("value", lambda data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert set(json.loads(lines[0])) == set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_tool_call(request, output_dir, metadata):
    """Test jstr filter functionality of the pipeline."""
    number = 1
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (
        Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", "{{value | tool_call}}")
        .iter_range(number)
        .add_column(
            "value",
            lambda data: {
                "name": "get_weather",
                "arguments": {"location": "New York", "unit": "Celsius"},
            },
        )
        .write_jsonl(path=output_file, template="output")
        .run()
    )

    lines = open(output_file).readlines()
    assert (
        lines[0]
        == '"<tool_call>{\\"arguments\\":{\\"location\\":\\"New York\\",\\"unit\\":\\"Celsius\\"},\\"name\\":\\"get_weather\\"}</tool_call>"\n'
    )
