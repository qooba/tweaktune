import json
import re
from typing import Any, Dict, Type

from pydantic import BaseModel


def parse(result: str, model: Type[BaseModel]):
    result_dict: Dict[str, Any]
    try:
        result_dict = json.loads(result)
    except Exception:
        try:
            result_dict = extract_json_block_md(result)
        except Exception:
            result_dict = extract_json_block(result)

    return model(**result_dict)


def extract_json_block_md(text):
    pattern = r"```json(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_block = match.group(1).strip()
        return json.loads(json_block)
    else:
        raise ValueError("No JSON block found")


def extract_json_block(text):
    pattern = r"{(.*?)}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_block = match.group(1).strip()
        return json.loads(f"{{{json_block}}}")
    else:
        print(text)
        raise ValueError("No JSON block found")
