import json
from tweaktune import Pipeline
import random

from tweaktune.chain import Chain

def test_step_sample(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"


    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_items": {{sampled_items[0]|jstr}} }""")
    .iter_range(10)
        .sample(dataset="items", size=1, output="sampled_items")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "my_items" in item
    print("ITEM:", item)
    assert "name" in item["my_items"]
    assert len(lines) == 10

def test_step_py(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    class CustomStep:
        def process(self, context):
            context["data"]["hello"] = "world"
            context["data"]["my_custom"] = context["data"]["sampled_items"][0]
            return context


    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_items": {{sampled_items[0]|jstr}}, "hello": {{hello|jstr}}, "my_custom": {{my_custom|jstr}} }""")
    .iter_range(10)
        .sample(dataset="items", size=1, output="sampled_items")
        .step(CustomStep())
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "my_items" in item
    assert "name" in item["my_items"]
    assert len(lines) == 10
    assert item["hello"] == "world"
    assert "name" in item["my_custom"]


def test_step_map(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    def test_map(context):
        context["data"]["hello"] = "world"
        context["data"]["my_custom"] = context["data"]["sampled_items"][0]
        return context


    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_items": {{sampled_items[0]|jstr}}, "hello": {{hello|jstr}}, "my_custom": {{my_custom|jstr}} }""")
    .iter_range(10)
        .sample(dataset="items", size=1, output="sampled_items")
        .map(test_map)
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "my_items" in item
    assert "name" in item["my_items"]
    assert len(lines) == 10
    assert item["hello"] == "world"
    assert "name" in item["my_custom"]

def test_step_add_column(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_random": {{my_random|jstr}} }""")
    .iter_range(10)
        .add_column("my_random", lambda data: f"random_{random.randint(0,9)}")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert len(lines) == 10
    assert "my_random" in item
    assert item["my_random"].startswith("random_")

def test_step_filter(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_random": {{my_random}} }""")
    .iter_range(10)
        .add_column("my_random", lambda data: random.randint(0,9))
        .filter(lambda data: data["my_random"] % 2 == 0)
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    for line in lines:
        item = json.loads(line)
        assert "my_random" in item
        assert item["my_random"] % 2 == 0

def test_step_mutate(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_random": {{my_random}} }""")
    .iter_range(10)
        .add_column("my_random", lambda data: random.randint(0,9))
        .mutate("my_random", lambda my_random: 10)
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    for line in lines:
        item = json.loads(line)
        assert "my_random" in item
        assert item["my_random"] == 10

def test_step_render(request, output_dir):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"
    
    (Pipeline()
        .with_workers(1)
        .with_template("my_template", """HELLO WORLD""")
        .with_template("output", """{"hello": {{my|jstr}}}""")
    .iter_range(number)
        .render(template="my_template", output="my")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    line = json.loads(lines[0])
    assert "hello" in line
    assert line["hello"] == "HELLO WORLD"
    assert len(lines) == number
    
def test_step_ifelse_then(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_1": {{my_1}}, "my_then": {{my_then}}, "my_else": {{my_else}} }""")
    .iter_range(10)
        .add_column("my_1", lambda data: 1)
        .ifelse(
            condition=lambda data: data["my_1"] == 1,
            then_chain=Chain().add_column("my_then", lambda data: True).add_column("my_else", lambda data: False),
            else_chain=Chain().add_column("my_else", lambda data: True).add_column("my_then", lambda data: False)
        )
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    for line in lines:
        item = json.loads(line)
        assert "my_1" in item
        assert "my_then" in item
        assert "my_else" in item
        assert item["my_1"] == 1
        assert item["my_then"] is True
        assert item["my_else"] is False

def test_step_ifelse_else(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline()
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_1": {{my_1}}, "my_then": {{my_then}}, "my_else": {{my_else}} }""")
    .iter_range(10)
        .add_column("my_1", lambda data: 1)
        .ifelse(
            condition=lambda data: data["my_1"] == 2,
            then_chain=Chain().add_column("my_then", lambda data: True).add_column("my_else", lambda data: False),
            else_chain=Chain().add_column("my_else", lambda data: True).add_column("my_then", lambda data: False)
        )
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    for line in lines:
        item = json.loads(line)
        assert "my_1" in item
        assert "my_then" in item
        assert "my_else" in item
        assert item["my_1"] == 1
        assert item["my_then"] is False
        assert item["my_else"] is True
