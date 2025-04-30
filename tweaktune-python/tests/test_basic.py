import pytest
from tweaktune import Pipeline

def test_basic(request, output_dir):
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_template("output", """{"hello": "world"}""")\
    .iter_range(number)\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()

    
    assert len(lines) == number