from enum import Enum
from .tweaktune import StepTest, StepConfigTest
from .tweaktune import Step, Jsonl, Parquet, Csv, Arrow, Lang, PipelineBuilder, IterBy, Dataset, LLM, Embeddings, Template
import json
from typing import List, overload, Optional, Union, Tuple
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
import inspect

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
    schema["additionalProperties"] = False
    return schema


class PyStepWrapper:
    def __init__(self, step):
        self.step = step

    def process(self, context):
        context = json.loads(context)
        return json.dumps(self.step.process(context))
    

class UnslothWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def process(self, messages: dict):
        inputs = self.tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
        output_ids = self.model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True)
        input_size = inputs[0].shape[0]
        output_text = self.tokenizer.decode(output_ids[0][input_size:], skip_special_tokens=False)
        return output_text
    
class PyStepValidatorWrapper:
    def __init__(self, func):
        self.func = func

    def process(self, context):
        context = json.loads(context)
        return self.func(context)


class Pipeline:
    def __init__(self):
        self.builder = PipelineBuilder()

    def with_dataset(self, dataset: Dataset):
        if dataset.__class__ == Dataset.Jsonl:
            self.builder.with_jsonl_dataset(dataset.name, dataset.path)
        elif dataset.__class__ == Dataset.Json:
            self.builder.with_json_dataset(dataset.name, dataset.path)
        elif dataset.__class__ == Dataset.Mixed:
            self.builder.with_mixed_dataset(dataset.name, dataset.datasets)
        elif dataset.__class__ == Dataset.Parquet:
            self.builder.with_parquet_dataset(dataset.name, dataset.path)
        elif dataset.__class__ == Dataset.Csv:
            self.builder.with_csv_dataset(dataset.name, dataset.path, dataset.delimiter, dataset.has_header)
        elif dataset.__class__ == Dataset.Arrow:
            try:
                from datasets.arrow_dataset import Dataset as ArrowDataset
                from pyarrow.lib import RecordBatchReader

                if type(dataset) is ArrowDataset:
                    self.builder.with_arrow_dataset(dataset.name, dataset.data.to_reader())
                elif type(dataset) is RecordBatchReader:
                    self.builder.with_arrow_dataset(dataset.name, dataset)
                else:
                    raise ValueError("Invalid dataset type")

                return self

            except ModuleNotFoundError:
                package_installation_hint("datasets")
                package_installation_hint("pyarrow")
                raise

        else:
            raise ValueError("Invalid dataset type")
        
        return self

    def with_openapi_dataset(self, name: str, path_or_url: str):
        """Adds an OpenAPI dataset to the pipeline."""
        self.builder.with_openapi_dataset(name, path_or_url)
        return self

    def with_tools_dataset(self, name: str, tools: List[callable]):
        """Converts a list of functions to json schema and adds them to the pipeline."""
        json_list = [function_to_json_schema(tool) for tool in tools]
        self.builder.with_json_list_dataset(name, json_list)
        return self
    
    def with_pydantic_models_dataset(self, name: str, models: List[BaseModel]):
        """Converts a list of Pydantic models to json schema and adds them to the pipeline."""
        json_list = [pydantic_to_json_schema(model) for model in models]
        self.builder.with_json_list_dataset(name, json_list)
        return self
    
    def with_dicts_dataset(self, name: str, dicts: List[dict]):
        """Converts a list of dictionaries to json schema and adds them to the pipeline."""
        json_list = [json.dumps(d) for d in dicts]
        self.builder.with_json_list_dataset(name, json_list)
        return self

    def with_jsonl_dataset(self, name: str, path: str):
        """Adds a jsonl dataset to the pipeline."""
        self.builder.with_jsonl_dataset(name, path)
        return self
    
    def with_json_dataset(self, name: str, path: str):
        """Adds a json dataset to the pipeline."""
        self.builder.with_json_dataset(name, path)
        return self

    def with_mixed_dataset(self, name: str, datasets: List[str]):
        """Adds a mixed dataset to the pipeline."""
        self.builder.with_mixed_dataset(name, datasets)
        return self

    def with_polars_dataset(self, name: str, path: str, sql: str):
        """Adds a polars dataset to the pipeline."""
        self.builder.with_polars_dataset(name, path, sql)
        return self

    def with_parquet_dataset(self, name: str, path: str):
        """Adds a parquet dataset to the pipeline."""
        self.builder.with_parquet_dataset(name, path)
        return self
    
    def with_csv_dataset(self, name: str, path: str, delimiter: str, has_header: bool):
        """Adds a csv dataset to the pipeline."""
        self.builder.with_csv_dataset(name, path, delimiter, has_header)
        return self
    
    def with_arrow_dataset(self, name: str, dataset):
        """Adds an arrow dataset to the pipeline."""
        try:
            from datasets.arrow_dataset import Dataset as ArrowDataset
            from pyarrow.lib import RecordBatchReader

            if type(dataset) is ArrowDataset:
                self.builder.with_arrow_dataset(name, dataset.data.to_reader())
            elif type(dataset) is RecordBatchReader:
                self.builder.with_arrow_dataset(name, dataset)
            else:
                raise ValueError("Invalid dataset type")

            return self

        except ModuleNotFoundError:
            package_installation_hint("datasets")
            package_installation_hint("pyarrow")
            raise

    
    def with_template(self, name: str, template: str):
        """Adds a template to the pipeline."""
        self.builder.with_jinja_template(name, template)
        return self
    
    def with_llm(self, llm: LLM):
        """Adds a LLM to the pipeline."""
        if llm.__class__ == LLM.OpenAI:
            self.builder.with_llm_api(llm.name, llm.base_url, llm.api_key, llm.model, llm.max_tokens)
        else:
            raise ValueError("Invalid LLM type")
        
        return self
    
    def with_llm_api(self, name: str, base_url: str, api_key: str, model: str, max_tokens: int = 2048):
        """Adds an OpenAI LLM to the pipeline."""
        self.builder.with_llm_api(name, base_url, api_key, model, max_tokens)
        return self
    
    def with_llm_unsloth(self, name: str, 
                         model_name: str, 
                         load_in_4bit: bool = True, 
                         dtype = None, 
                         max_seq_length: int = 2048, 
                         hf_token: str = None,
                         chat_template: Union[str, Tuple[str, str]] = "chatml",
                         mapping: dict = None,
                         map_eos_token: bool = True,
                         ):
        try:
            from unsloth import FastLanguageModel
            from unsloth.chat_templates import get_chat_template

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                token = hf_token
            )

            if not mapping:
                mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}

            tokenizer = get_chat_template(
               tokenizer,
               chat_template = chat_template,
               mapping = mapping,
               map_eos_token = map_eos_token
            )
            FastLanguageModel.for_inference(model)
            self.builder.with_llm_unsloth(name, UnslothWrapper(model, tokenizer))

            return self
        except ModuleNotFoundError:
            package_installation_hint("unsloth")
            raise

    def with_embedings(self, embeddings: Embeddings):
        if embeddings.__class__ == Embeddings.OpenAI:
            self.builder.with_embeddings_api(embeddings.name, embeddings.model, embeddings.base_url, embeddings.api_key)
        else:
            raise ValueError("Invalid Embeddings type")
        
        return self
    
    def with_workers(self, workers: int):
        self.builder.with_workers(workers)
        return self
    
    def from_yaml(self, path_or_url: str):
        #TODO: Implement fetch configuration from yaml
        return self
    
    def iter(self, iter_by: IterBy):
        if iter_by.__class__ == IterBy.Range:
            self.builder.iter_by_range(iter_by.start, iter_by.stop, iter_by.step)
        elif iter_by.__class__ == IterBy.Dataset:
            self.builder.iter_by_dataset(iter_by.name)
        else:
            raise ValueError("Invalid IterBy type")
        
        return PipelineRunner(self.builder)
    
    def iter_dataset(self, name: str):
        self.builder.iter_by_dataset(name)
        return PipelineRunner(self.builder)
    
    def iter_range(self, *args, **kwargs):
        start = kwargs.get('start', 0)
        stop = kwargs.get('stop', 0)
        step = kwargs.get('step', 1)

        if len(args) == 1:
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
        elif len(args) == 3:
            start = args[0]
            stop = args[1]
            step = args[2]

        self.builder.iter_by_range(start, stop, step)
        return PipelineRunner(self.builder)


class PipelineRunner:

    def __init__(self, builder: PipelineBuilder):
        builder.compile()
        self.builder = builder
        self.step_index = 0

    def __then(self, step: Step):
        if step.__class__ == Step.Py:
            self.builder.add_py_step(step.name, PyStepWrapper(step.py_func))
        elif step.__class__ == Step.TextGeneration:
            self.builder.add_text_generation_step(step.name, step.template, step.llm, step.output, step.system_template)
        elif step.__class__ == Step.JsonGeneration:
            self.builder.add_json_generation_step(step.name, step.template, step.llm, step.output, step.json_path, step.system_template)
        elif step.__class__ == Step.DataSampler:
            self.builder.add_data_sampler_step(step.name, step.dataset, step.size)
        elif step.__class__ == Step.Judge:
            self.builder.add_judge_step(step.name, step.template, step.llm)
        elif step.__class__ == Step.PyValidator:
            self.builder.add_py_validator_step(step.name, PyStepValidatorWrapper(step.py_func))
        else:
            raise ValueError("Invalid Step type")
        
        self.step_index += 1
        return self
    
    def __name(self, name: str):
        return f"{name}--{self.step_index}"

    def step(self, step, name: str = "PY-STEP"):
        self.builder.add_py_step(self.__name(name), PyStepWrapper(step))
        self.step_index += 1
        return self
    
    def map(self, func, name: str = "PY-STEP"):
        name = self.__name(name)
        step = type(name.replace("-","_"), (object,), {'process': lambda self, context: func(context)})()
        self.builder.add_py_step(name, PyStepWrapper(step))
        self.step_index += 1
        return self


    def generate_text(self, template: str, llm: str, output: str, system_template: str = None, name: str = "GENERATE-TEXT"):
        self.builder.add_text_generation_step(self.__name(name), template, llm, output, system_template)
        self.step_index += 1
        return self

    def generate_json(self, template: str, llm: str, output: str, json_path: str = None, system_template: str = None, name: str = "GENERATE-JSON"):
        self.builder.add_json_generation_step(self.__name(name), template, llm, output, json_path, system_template)
        self.step_index += 1
        return self
    
    def sample(self, dataset: str, size: int, output: str, name: str = "SAMPLE"):
        self.builder.add_data_sampler_step(self.__name(name), dataset, size, output)
        self.step_index += 1
        return self
    
    def chunk(self, capacity: Tuple[int, int], input: str, output: str, name: str = "CHUNK"):
        self.builder.add_chunk_step(self.__name(name), capacity, input, output)
        self.step_index += 1
        return self
    
    def read(self, dataset: str, output: str, name: str = "SAMPLE"):
        self.builder.add_data_read_step(self.__name(name), dataset, output)
        self.step_index += 1
        return self
    
    def judge(self, template: str, llm: str, name: str = "JUDGE"):
        self.builder.add_judge_step(self.__name(name), template, llm)
        self.step_index += 1
        return self
    
    def validate(self, py_func, name: str = "VALIDATE"):
        self.builder.add_py_validator_step(self.__name(name), PyStepValidatorWrapper(py_func))
        self.step_index += 1
        return self
    
    def write_jsonl(self, path: str, template: str, name: str = "WRITE-JSONL"):
        self.builder.add_write_jsonl_step(self.__name(name), path, template)
        return self
    
    def write_csv(self, path: str, columns: List[str], delimeter: str, name: str = "WRITE-JSONL"):
        self.builder.add_write_csv_step(self.__name(name), path, columns, delimeter)
        return self

    def print(self, *args, **kwargs):
        template = kwargs.get('template', None)
        columns = kwargs.get('columns', None)
        if len(args) == 1:
            columns = args[0]
            
        name = "PRINT"
        self.builder.add_print_step(self.__name(name), template=template, columns=columns)
        return self

    def debug(self, target: str = None):
        self.builder.log("debug", target)
        return self

    def log(self, level: str = None, target: str = None):
        self.builder.log(level, target)
        return self
    
    def run(self):
        return self.builder.run()