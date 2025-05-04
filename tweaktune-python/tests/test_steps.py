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

def test_sample(request, output_dir, data_dir, arrow_dataset):
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
    assert "name" in item["my_items"]
    assert len(lines) == 10

