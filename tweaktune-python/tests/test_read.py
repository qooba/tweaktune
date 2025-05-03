import pytest
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline
from enum import Enum
import tempfile
import shutil


def test_mixed(request, output_dir):
    """Test the basic functionality of the pipeline."""
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_json_dataset("functions","./datasets/manus_tools.json")\
        .with_jsonl_dataset("personas","./datasets/personas_micro.jsonl")\
        .with_mixed_dataset("mixed",["functions", "personas"])\
        .with_template("output", """{"mixed": {{mixed|jstr}} }""")\
    .iter_dataset("mixed")\
        .write_jsonl(path=output_file, template="output")\
        .run()


    lines = open(output_file, "r").readlines()
    item = json.loads(lines[0])
    assert "mixed" in item
    assert "functions" in item["mixed"]
    assert "personas" in item["mixed"]

