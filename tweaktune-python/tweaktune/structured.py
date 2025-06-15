import json
import re
from pydantic import BaseModel

def parse(result: str, model: BaseModel):
    try:
        result = json.loads(result)
    except Exception as ex:
        try:
            result = extract_json_block_md(result)
        except Exception as ex:
            result = extract_json_block(result)

    return model(**result)

def extract_json_block_md(text):
    pattern = r'```json(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_block = match.group(1).strip()
        return json.loads(json_block)
    else:
        raise ValueError("No JSON block found")
    
def extract_json_block(text):
    pattern = r'{(.*?)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        json_block = match.group(1).strip()
        return json.loads(f"{{{json_block}}}")
    else:
        print(text)
        raise ValueError("No JSON block found")