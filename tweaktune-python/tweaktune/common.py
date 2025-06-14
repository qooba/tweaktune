import json
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
import inspect
import io
import pyarrow as pa
import pyarrow.ipc as ipc

def record_batches_to_ipc_bytes(reader: pa.RecordBatchReader) -> bytes:
    """
    Converts a RecordBatchReader to bytes using IPC format.
    """
    sink = io.BytesIO()
    writer = ipc.new_stream(sink, reader.schema)
    for batch in reader:
        writer.write_batch(batch)
    writer.close()
    return sink.getvalue()

def package_installation_hint(package_name: str):
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    BOLD = "\033[1m"
    print(f"\t{BOLD}Please install:{ENDC}\t{OKGREEN}{package_name}{ENDC}{BOLD}{ENDC}")

def pydantic_to_json_schema(model: BaseModel) -> dict:
    """
    Converts a Pydantic model to JSON schema.
    """
    schema = model.model_json_schema()
    schema = normalize_schema(schema)
    name = schema.pop("title", None)

    return json.dumps({
        "type": "json_schema",
        "name": name,
        "schema": schema,
        "strict": True,
    })

def function_to_json_schema(func: callable) -> BaseModel:
    """
    Converts a function with annotated parameters to json schema https://json-schema.org/
    including descriptions from Field(..., description=...).
    """
    func_name = func.__name__
    func_params = func.__annotations__

    sig = inspect.signature(func)
    model_fields = {}
    param_descriptions = {}

    for param_name, param in sig.parameters.items():
        if param_name in ["args", "kwargs"]:
            continue
        if param_name == "return":
            continue

        param_type = func_params.get(param_name, param.annotation)
        default = param.default

        if isinstance(default, FieldInfo):
            description = default.description
            model_fields[param_name] = (param_type, default)
            if description:
                param_descriptions[param_name] = description
        else:
            model_fields[param_name] = (param_type, Field(...))

    schema = create_model(func_name, **model_fields).model_json_schema()
    schema = normalize_schema(schema)
    schema.pop("title", None)

    return json.dumps({
        "type": "function",
        "name": func_name,
        "description": func.__doc__,
        "parameters": schema,
        "strict": True,
    })

def normalize_schema(schema):
    defs = schema.pop("$defs", {})

    if "properties" in schema:
        for prop in schema["properties"].values():
            if "$ref" in prop:
                ref_path = prop.pop("$ref")
                if ref_path.startswith("#/$defs/"):
                    def_key = ref_path.split("/")[-1]
                    if def_key in defs:
                        prop.update(defs[def_key])

            if "anyOf" in prop:
                prop["type"] = [t["type"] for t in prop.pop("anyOf")]

            prop.pop("title", None)
        schema["properties"] = json.dumps(schema["properties"], ensure_ascii=False)
    schema["additionalProperties"] = False
    return schema
