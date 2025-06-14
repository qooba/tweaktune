import pytest
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline
from enum import Enum
import tempfile
import shutil


def test_basic(request, output_dir):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_template("output", """{"hello": "{{value}}"}""")\
    .iter_range(number)\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number

def test_basic_j2(request, output_dir, j2_file):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_j2_template("output", j2_file)\
    .iter_range(number)\
        .add_column("value", lambda data: "world")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    assert len(lines) == number
    item = json.loads(lines[0])
    assert item["hello"] == "world"

