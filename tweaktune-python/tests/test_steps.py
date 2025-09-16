import json
from tweaktune import Pipeline
import random

from tweaktune.chain import Chain

def test_step_sample(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"


    (Pipeline(name=request.node.name, metadata=metadata)
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

def test_step_py(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    class CustomStep:
        def process(self, context):
            context["data"]["hello"] = "world"
            context["data"]["my_custom"] = context["data"]["sampled_items"][0]
            return context


    (Pipeline(name=request.node.name, metadata=metadata)
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


def test_step_map(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    def test_map(context):
        context["data"]["hello"] = "world"
        context["data"]["my_custom"] = context["data"]["sampled_items"][0]
        return context


    (Pipeline(name=request.node.name, metadata=metadata)
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

def test_step_add_column_lambda(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
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

def test_step_add_column(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"new_column_3": {{new_column_3}} }""")
    .iter_range(10)
        .add_column("new_column_2", "1 + 1")
        .add_column("new_column_3", "new_column_2 + 1")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert len(lines) == 10
    assert "new_column_3" in item
    assert item["new_column_3"] == 3


def test_step_filter_lambda(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
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

def test_step_filter(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_random": {{my_random}} }""")
    .iter_range(10)
        .add_column("my_random", lambda data: random.randint(0,9))
        .filter(condition="my_random % 2 == 0")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    for line in lines:
        item = json.loads(line)
        assert "my_random" in item
        assert item["my_random"] % 2 == 0

def test_step_mutate_lambda(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
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

def test_step_mutate(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"val": {{val}} }""")
    .iter_range(10)
        .add_column("val_10", func="10")
        .mutate("val", func="val_10 / 2")
        .mutate("val", func="val - 5")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    for line in lines:
        item = json.loads(line)
        assert "val" in item
        assert item["val"] == 0


def test_step_render(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
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


def test_step_render_conversation(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    def get_who_won(year: int) -> dict:
        """Example tool that returns who won the world series in a given year"""
        winners = {
            2020: "Los Angeles Dodgers",
            2019: "Washington Nationals",
            2018: "Boston Red Sox",
            2017: "Houston Astros",
            2016: "Chicago Cubs",
        }
        return {"winner": winners.get(year, "Unknown"), "year": year}

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_tools_dataset("tools", [get_who_won])
        .iter_range(1)
        .sample_tools("tools", 1, "tools")
        .add_column("system", lambda data: "You are a helpful assistant.")
        .add_column("question", lambda data: "Hello, who won the world series in 2020?")
        .add_column("call1", lambda data: "{ \"name\": \"get_who_won\", \"arguments\": { \"year\": 2020 } }")
        .add_column("response", lambda data: "{\"winner\": \"Los Angeles Dodgers\", \"year\": 2020}")
        .add_column("thinking", lambda data: "I should look up who won the world series in 2020.")
        .add_column("answer", lambda data: "The Los Angeles Dodgers won the World Series in 2020.")
        .render_conversation(conversation="""@system:system|@user:question|@assistant:tool_calls([call1])|@tool:response|@assistant:think(thinking)|@assistant:answer
            """,tools="tools", output="conversation")
        .write_jsonl(path=output_file, value="conversation")
    .run())

    lines = open(output_file, "r").readlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    messages = line["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, who won the world series in 2020?"
    assert messages[2]["role"] == "assistant"
    assert "tool_calls" in messages[2]
    assert messages[2]["tool_calls"] == [{"function":{"name": "get_who_won", "arguments": {"year": 2020}}}]
    assert messages[3]["role"] == "tool"
    assert "winner" in messages[3]["content"]
    assert messages[4]["role"] == "assistant"
    assert "I should look up" in messages[4]["reasoning_content"]
    assert messages[5]["role"] == "assistant"
    assert "Los Angeles Dodgers" in messages[5]["content"]

def test_step_render_conversation_aliases(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .iter_range(1)
        .add_column("system", lambda data: "You are a helpful assistant.")
        .add_column("question", lambda data: "Hello, who won the world series in 2020?")
        .add_column("call1", lambda data: "{ \"name\": \"get_who_won\", \"arguments\": { \"year\": 2020 } }")
        .add_column("response", lambda data: "{\"winner\": \"Los Angeles Dodgers\", \"year\": 2020}")
        .add_column("answer", lambda data: "The Los Angeles Dodgers won the World Series in 2020.")
        .render_conversation(conversation="""
            @s:system
            @u:question
            @a:tool_calls([call1])
            @t:response
            @a:answer
        """, output="conversation", separator="\n")
        .write_jsonl(path=output_file, value="conversation")
    .run())

    lines = open(output_file, "r").readlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    messages = line["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, who won the world series in 2020?"
    assert messages[2]["role"] == "assistant"
    assert "tool_calls" in messages[2]
    assert messages[2]["tool_calls"] == [{"function":{"name": "get_who_won", "arguments": {"year": 2020}}}]
    assert messages[3]["role"] == "tool"
    assert "winner" in messages[3]["content"]
    assert messages[4]["role"] == "assistant"
    assert "Los Angeles Dodgers" in messages[4]["content"]


def test_step_check_language(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", """{"question": {{question}}""")
        .iter_range(1)
        .add_column("question", lambda data: "Hello, who won the world series in 2020?")
        .check_language(input="question", language="english", precision=0.8, detect_languages=["english", "french", "german", "spanish"])
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    assert len(lines) == 1

def test_step_check_language_polish(request, output_dir, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_template("output", """{"question": {{question}}""")
        .iter_range(1)
        .add_column("question", lambda data: "Kto wygrał światową serię w 2020 roku?")
        .check_language(input="question", language="polish", precision=0.9, detect_languages=["english", "polish", "french", "german", "spanish"])
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    assert len(lines) == 1

def test_step_ifelse_then_lambda(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
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

def test_step_ifelse_then(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_1": {{my_1}}, "my_then": {{my_then}}, "my_else": {{my_else}} }""")
    .iter_range(10)
        .add_column("my_1", lambda data: 1)
        .ifelse(
            condition="my_1 == 1",
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


def test_step_ifelse_else_lambda(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
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

def test_step_ifelse_else(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_1": {{my_1}}, "my_then": {{my_then}}, "my_else": {{my_else}} }""")
    .iter_range(10)
        .add_column("my_1", lambda data: 1)
        .ifelse(
            condition="my_1 == 2",
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


def test_step_into_list(request, output_dir, data_dir, arrow_dataset, metadata):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    (Pipeline(name=request.node.name, metadata=metadata)
        .with_workers(1)
        .with_arrow_dataset("items", arrow_dataset())
        .with_template("output", """{"my_list": {{my_list|tojson}} }""")
    .iter_range(10)
        .add_column("item_1", lambda data: 1)
        .add_column("item_2", lambda data: 2)
        .into_list(inputs=["item_1", "item_2"], output="my_list")
        .write_jsonl(path=output_file, template="output")
    .run())

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert len(lines) == 10
    assert "my_list" in item
    assert item["my_list"] == [1, 2]
