import pytest
from pydantic import BaseModel, Field
from typing import List, Optional
from tweaktune import Pipeline
from enum import Enum
import tempfile
import shutil

class ItemType(str, Enum):
    """
    Enum for item types.
    """
    electronics = "electronics"
    clothing = "clothing"
    food = "food"


def my_function(
    id: int = Field(..., description="The ID of the item"),
    name: str = Field(..., description="The name of the item"),
    description: str = Field(..., description="The description of the item"),
    price: Optional[float] = Field(..., description="The price of the item"),
    quantity: int = Field(..., description="The quantity of the item"),
    type: ItemType = Field(..., description="The type of the item"),
):
    """This is a sample function."""
    pass


def test_tools(request, data_dir, output_dir):
    number = 5
    output_file = f"{output_dir}/{request.node.name}.jsonl"

    Pipeline()\
        .with_workers(1)\
        .with_tools_dataset("tools", [my_function])\
        .with_template("output", """{"tool": {{tool|jstr}} }""")\
    .iter_range(number)\
        .sample(dataset="tools", size=1, output="tool")\
        .write_jsonl(path=output_file, template="output")\
    .run()

    lines = open(output_file, "r").readlines()
    
    assert len(lines) == number