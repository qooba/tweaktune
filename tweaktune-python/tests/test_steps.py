import pytest
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline
from enum import Enum
import tempfile
import shutil
import polars as pl
import os

def test_step_sample(request, output_dir, data_dir, arrow_dataset):
    """Test the basic functionality of the pipeline."""
    output_file = f"{output_dir}/{request.node.name}.jsonl"


    Pipeline()\
        .with_workers(1)\
        .with_arrow_dataset("items", arrow_dataset())\
        .with_template("output", """{"my_items": {{sampled_items[0]|jstr}} }""")\
    .iter_range(10)\
        .sample(dataset="items", size=1, output="sampled_items")\
        .write_jsonl(path=output_file, template="output")\
    .run()

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


    Pipeline()\
        .with_workers(1)\
        .with_arrow_dataset("items", arrow_dataset())\
        .with_template("output", """{"my_items": {{sampled_items[0]|jstr}}, "hello": {{hello|jstr}}, "my_custom": {{my_custom|jstr}} }""")\
    .iter_range(10)\
        .sample(dataset="items", size=1, output="sampled_items")\
        .step(CustomStep())\
        .write_jsonl(path=output_file, template="output")\
    .run()

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


    Pipeline()\
        .with_workers(1)\
        .with_arrow_dataset("items", arrow_dataset())\
        .with_template("output", """{"my_items": {{sampled_items[0]|jstr}}, "hello": {{hello|jstr}}, "my_custom": {{my_custom|jstr}} }""")\
    .iter_range(10)\
        .sample(dataset="items", size=1, output="sampled_items")\
        .map(test_map)\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "my_items" in item
    assert "name" in item["my_items"]
    assert len(lines) == 10
    assert item["hello"] == "world"
    assert "name" in item["my_custom"]


def test_step_render(request, output_dir):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"
    
    Pipeline()\
        .with_workers(1)\
        .with_template("my_template", """HELLO WORLD""")\
        .with_template("output", """{"hello": {{my|jstr}}}""")\
    .iter_range(number)\
        .render(template="my_template", output="my")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    line = json.loads(lines[0])
    assert "hello" in line
    assert line["hello"] == "HELLO WORLD"
    assert len(lines) == number
    

