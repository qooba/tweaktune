import json
from tweaktune import Pipeline

def test_metadata(request, output_dir):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name)
        .with_workers(1)
        .with_template("output", """{"hello": "{{value}}"}""")
    .iter_range(number)
        .log("info")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number

