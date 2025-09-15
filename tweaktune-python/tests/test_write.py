import pytest
from tweaktune import Pipeline
import json
import csv

def test_write_json(request, output_dir):
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(request.node.name)
        .with_workers(1)
        .with_template("output", """{"hello": "world"}""")
    .iter_range(number)
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    hello = json.loads(lines[0])["hello"]

    assert hello == "world"
    assert len(lines) == number

def test_write_json_render(request, output_dir):
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(request.node.name)
        .with_workers(1)
        .with_template("output", """{"hello": "world"}""")
    .iter_range(number)
        .render(template="output", output="output")
        .write_jsonl(path=output_file, value="output")
    .run())

    lines = open(output_file, "r").readlines()
    hello = json.loads(lines[0])["hello"]

    assert hello == "world"
    assert len(lines) == number


def test_write_csv(request, output_dir):
    number = 5
    output_file = f"{output_dir}/{request.node.name}.csv"

    (Pipeline(request.node.name)
        .with_workers(1)
        .with_template("output", """world""")
    .iter_range(number)
        .render(template="output", output="hello")
        .write_csv(path=output_file, columns=["hello"], delimeter=";" )
    .run())

    lines = open(output_file, "r").readlines()
    
    reader = csv.DictReader(lines)
    hello = next(reader)["world"]
    assert hello == "world"
    assert len(lines) == number