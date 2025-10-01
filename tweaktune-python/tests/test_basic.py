import json
from tweaktune import Pipeline

def test_basic(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", """{"hello": "{{value}}", "number": {{"1,10"|random_range}}}""")
    .iter_range(number)
        .add_random("value", 1, 100)
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    print(lines)
    assert len(lines) == number

def test_basic_j2(request, output_dir, j2_file, metadata):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_j2_template("output", j2_file)
    .iter_range(number)
        .add_column("value", lambda data: "world")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    assert len(lines) == number
    item = json.loads(lines[0])
    assert item["hello"] == "world"

def test_basic_j2_dir(request, output_dir, j2_dir, metadata):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_templates(j2_dir)
    .iter_range(number)
        .add_column("value", lambda data: "world")
        .write_jsonl(path=output_file, template="example.j2")
    .run())

    lines = open(output_file, "r").readlines()
    assert len(lines) == number
    item = json.loads(lines[0])
    assert item["hello"] == "world"


def test_basic_j2_yaml(request, output_dir, j2_file_yaml, metadata):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"
    print(open(j2_file_yaml, "r").read())

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_j2_templates(j2_file_yaml)
    .iter_range(number)
        .add_column("value", lambda data: "world")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    assert len(lines) == number
    item = json.loads(lines[0])
    assert item["hello"] == "world"


#def test_basic_j2_https(request, output_dir):
#    """Test the basic functionality of the pipeline."""
#    number = 5
#    output_file = f"{output_dir}/{request.node.name}.jsonl"
#    j2_file ="https://raw.githubusercontent.com/qooba/tweaktune/refs/heads/tools/tweaktune-python/tweaktune/templates/function_calling/output_template.j2"
#
#    (Pipeline()
#        .with_workers(1)
#        .with_j2_template("output", j2_file)
#    .iter_range(number)
#        .add_column("question", lambda data: "How are you ?")
#        .add_column("function", lambda data: [{"name": "search_products"}])
#        .add_column("call", lambda data: "CALL")
#        .add_column("all_functions", lambda data: [{"name": "search_products"}])
#        .write_jsonl(path=output_file, template="output")
#        .debug()
#    .run())
#
#    lines = open(output_file, "r").readlines()
#    item = lines[0]
##    item = json.loads(ttt)
#
#    #assert len(lines) == number
#    #assert item["hello"] == "world"
