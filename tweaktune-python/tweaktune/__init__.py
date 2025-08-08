from tweaktune.tweaktune import PipelineBuilder, IterBy, LLM, Embeddings
from tweaktune.tweaktune import ChatTemplateBuilder as _ChatTemplateBuilder
from tweaktune.common import LogLevel, StepStatus, record_batches_to_ipc_bytes, package_installation_hint
from tweaktune.tools import pydantic_to_json_schema, function_to_json_schema
from tweaktune.wrappers import PyStepWrapper, UnslothWrapper, MistralrsWrapper, PyStepValidatorWrapper, PyConditionWrapper
from tweaktune.chain import Chain
import json
import sys
import os
import inspect
from typing import Dict, List, Union, Tuple, Callable, Optional, Any
from pydantic import BaseModel

def step_item(name: str):
    frame = inspect.currentframe().f_back
    args_info = {str(k):v for k,v in inspect.getargvalues(frame).locals.items() if k != 'self'}
    return StepItem(name=name, func=frame.f_code.co_name, args=args_info)

class StepItem(BaseModel):
    name: str
    func: str
    args: Dict[str, Any] = {}
    children: List['StepItem'] = []

    def add_child(self, child: 'StepItem'):
        self.children.append(child)

def config_item(name: str):
    frame = inspect.currentframe().f_back
    args_info = {str(k):str(v).replace('"','') for k,v in inspect.getargvalues(frame).locals.items() if k != 'ipc_data' and k != 'self'}
    return ConfigItem(name=name, func=frame.f_code.co_name, args=args_info)
    
class ConfigItem(BaseModel):
    name: str
    func: str
    args: Dict[str, str] = {}

def start_item(name: str):
    frame = inspect.currentframe().f_back
    args_info = {str(k):str(v).replace('"','') for k,v in inspect.getargvalues(frame).locals.items() if k != 'ipc_data' and k != 'self'}
    return ConfigItem(name=name, func=frame.f_code.co_name, args=args_info)
    
class StartItem(BaseModel):
    name: str
    func: str
    args: Dict[str, str] = {}

class GraphConfig(BaseModel):
    llms: List[ConfigItem] = []
    datasets: List[ConfigItem] = []
    templates: List[ConfigItem] = []
    workers: int = 1

class Graph(BaseModel):
    config: GraphConfig = GraphConfig()
    steps: List[StepItem] = []
    start: Optional[StartItem] = None



class Pipeline:
    def __init__(self):
        self.builder = PipelineBuilder()
        self.graph = Graph()

    def with_openapi_dataset(self, name: str, path_or_url: str):
        """Adds an OpenAPI dataset to the pipeline."""
        self.builder.with_openapi_dataset(name, path_or_url)
        self.graph.config.datasets.append(config_item(name))
        return self

    def with_tools_dataset(self, name: str, tools: List[callable]):
        """Converts a list of functions to json schema and adds them to the pipeline."""
        json_list = [function_to_json_schema(tool) for tool in tools]
        self.builder.with_json_list_dataset(name, json_list, None)
        self.graph.config.datasets.append(config_item(name))
        return self
    
    def with_pydantic_models_dataset(self, name: str, models: List[BaseModel]):
        """Converts a list of Pydantic models to json schema and adds them to the pipeline."""
        json_list = [pydantic_to_json_schema(model) for model in models]
        self.builder.with_json_list_dataset(name, json_list, None)
        self.graph.config.datasets.append(config_item(name))
        return self
    
    def with_dicts_dataset(self, name: str, dicts: List[dict], sql: str = None):
        """Converts a list of dictionaries to json schema and adds them to the pipeline."""
        json_list = [json.dumps(d) for d in dicts]
        self.builder.with_json_list_dataset(name, json_list, sql)
        self.graph.config.datasets.append(config_item(name))
        return self

    def with_jsonl_dataset(self, name: str, path: str, sql: str = None):
        """Adds a jsonl dataset to the pipeline."""
        self.builder.with_jsonl_dataset(name, path, sql)
        self.graph.config.datasets.append(config_item(name))
        return self
    
    def with_json_dataset(self, name: str, path: str, sql: str = None):
        """Adds a json dataset to the pipeline."""
        self.builder.with_json_dataset(name, path, sql)
        self.graph.config.datasets.append(config_item(name))
        return self

    def with_mixed_dataset(self, name: str, datasets: List[str]):
        """Adds a mixed dataset to the pipeline."""
        self.builder.with_mixed_dataset(name, datasets)
        self.graph.config.datasets.append(config_item(name))
        return self

    def with_polars_dataset(self, name: str, path: str, sql: str):
        """Adds a polars dataset to the pipeline."""
        self.builder.with_polars_dataset(name, path, sql)
        self.graph.config.datasets.append(config_item(name))
        return self

    def with_parquet_dataset(self, name: str, path: str, sql: str = None):
        """Adds a parquet dataset to the pipeline."""
        self.builder.with_parquet_dataset(name, path, sql)
        self.graph.config.datasets.append(config_item(name))
        return self
    
    def with_csv_dataset(self, name: str, path: str, delimiter: str, has_header: bool, sql: str = None):
        """Adds a csv dataset to the pipeline."""
        self.builder.with_csv_dataset(name, path, delimiter, has_header, sql)
        self.graph.config.datasets.append(config_item(name))
        return self

    def with_db_dataset(self, name: str, conn: str, query: str):
        """Adds a database dataset to the pipeline.
        The database connection string should be in the format:
        "postgresql://user:password@host:port/database"
        The query should be a valid SQL query.
        The supported databases are:
        https://sfu-db.github.io/connector-x/databases.html
        """
        try:
            import connectorx as cx
            table  = cx.read_sql(conn, query, return_type="arrow")
            ipc_data = record_batches_to_ipc_bytes(table.to_reader())
            self.builder.with_ipc_dataset(name, ipc_data)
            self.graph.config.datasets.append(config_item(name))
            return self
        except ModuleNotFoundError:
            package_installation_hint("connectorx")
            raise

    def with_hf_dataset(self, name: str, dataset_path: str, dataset_name: str = None, dataset_split = "train", sql: str = None):
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_path, name=dataset_name, split=dataset_split)
            ipc_data = record_batches_to_ipc_bytes(dataset.data.to_reader())
            self.builder.with_ipc_dataset(name, ipc_data, sql)
            self.graph.config.datasets.append(config_item(name))
            return self
        except ModuleNotFoundError:
            package_installation_hint("datasets")
            raise

    def with_arrow_dataset(self, name: str, dataset, sql: str = None):
        """Adds an arrow dataset to the pipeline."""
        try:
            from datasets.arrow_dataset import Dataset as ArrowDataset
            from pyarrow.lib import RecordBatchReader

            if type(dataset) is ArrowDataset:
                ipc_data = record_batches_to_ipc_bytes(dataset.data.to_reader())
                self.builder.with_ipc_dataset(name, ipc_data, sql)
            elif type(dataset) is RecordBatchReader:
                ipc_data = record_batches_to_ipc_bytes(dataset)
                self.builder.with_ipc_dataset(name, ipc_data, sql)
            else:
                raise ValueError("Invalid dataset type")
            
            self.graph.config.datasets.append(config_item(name))
            return self

        except ModuleNotFoundError:
            package_installation_hint("datasets")
            package_installation_hint("pyarrow")
            raise
    
    def with_template(self, name: str, template: str):
        """Adds a template to the pipeline."""
        self.builder.with_jinja_template(name, template)
        self.graph.config.templates.append(config_item(name))
        return self

    def with_templates(self, path: str = "templates", op_config: Optional[dict] = None):
        """Adds a templates from dir to the pipeline."""
        op_config = json.dumps(op_config, ensure_ascii=False) if op_config else None
        self.builder.with_dir_templates(path, op_config)
        self.graph.config.templates.append(config_item("DIR-TEMPLATES"))
        return self
    
    def with_j2_template(self, name: str, path: str, op_config: Optional[dict] = None):
        """Adds a template from file to the pipeline."""
        op_config = json.dumps(op_config, ensure_ascii=False) if op_config else None
        self.builder.with_j2_template(name, path, op_config)
        self.graph.config.templates.append(config_item(name))
        return self
    
    def with_j2_templates(self, path: str, op_config: Optional[dict] = None):
        """Adds a template from file to the pipeline."""
        op_config = json.dumps(op_config, ensure_ascii=False) if op_config else None
        self.builder.with_j2_templates(path, op_config)
        self.graph.config.templates.append(config_item("J2-TEMPLATES"))
        return self
    
    def with_llm(self, llm: LLM):
        """Adds a LLM to the pipeline."""
        if llm.__class__ == LLM.OpenAI:
            self.builder.with_llm_api(llm.name, llm.base_url, llm.api_key, llm.model, llm.max_tokens)
            self.graph.config.llms.append(config_item(llm.name))
        else:
            raise ValueError("Invalid LLM type")
        
        return self
    
    def with_llm_api(self, name: str, base_url: str, api_key: str, model: str, max_tokens: int = 2048, temperature: float = 0.7):
        """Adds an OpenAI LLM to the pipeline."""
        self.builder.with_llm_api(name, base_url, api_key, model, max_tokens, temperature)
        self.graph.config.llms.append(config_item(name))
        return self
    
    def with_llm_openai(self, name: str, api_key: str, model: str, max_tokens: int = 2048, temperature: float = 0.7):
        """Adds an OpenAI LLM to the pipeline."""
        self.builder.with_llm_openai(name, api_key, model, max_tokens, temperature)
        self.graph.config.llms.append(config_item(name))
        return self

    def with_llm_azure_openai(self, name: str, api_key: str, endpoint: str, deployment_name: str, api_version: str, max_tokens: int = 2048, temperature: float = 0.7):
        """Adds an OpenAI LLM to the pipeline."""
        self.builder.with_llm_azure_openai(name, api_key, endpoint, deployment_name, api_version, max_tokens, temperature)
        self.graph.config.llms.append(config_item(name))
        return self
    
    def with_llm_mistralrs(self, name: str, 
                         model_id: str, 
                         in_situ_quant: str
                         ):
        try:
            from mistralrs import ChatCompletionRequest, Runner, Which
            runner = Runner(
                which=Which.Plain(model_id=model_id),
                in_situ_quant=in_situ_quant,
            )
            self.builder.with_llm_mistralrs(name, MistralrsWrapper(runner))
            self.graph.config.llms.append(config_item(name))
            return self
        except ModuleNotFoundError:
            package_installation_hint("mistralrs")
            raise

    
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
            self.graph.config.llms.append(config_item(name))

            return self
        except ModuleNotFoundError:
            package_installation_hint("unsloth")
            raise

    def with_embedings(self, embeddings: Embeddings):
        if embeddings.__class__ == Embeddings.OpenAI:
            self.builder.with_embeddings_api(embeddings.name, embeddings.model, embeddings.base_url, embeddings.api_key)
            self.graph.config.llms.append(config_item("EMBEDDINGS"))
        else:
            raise ValueError("Invalid Embeddings type")
        
        return self
    
    def with_workers(self, workers: int):
        self.builder.with_workers(workers)
        self.graph.config.workers = workers
        return self
    
    def from_yaml(self, path_or_url: str):
        #TODO: Implement fetch configuration from yaml
        return self
    
    def iter(self, iter_by: IterBy):
        if iter_by.__class__ == IterBy.Range:
            self.builder.iter_by_range(iter_by.start, iter_by.stop, iter_by.step)
            self.graph.start = start_item("ITER-RANGE")
        elif iter_by.__class__ == IterBy.Dataset:
            self.builder.iter_by_dataset(iter_by.name)
            self.graph.start = start_item("ITER-DATASET")
        else:
            raise ValueError("Invalid IterBy type")
        
        return PipelineRunner(self.builder, self.graph)
    
    def iter_dataset(self, name: str):
        self.builder.iter_by_dataset(name)
        self.graph.start = start_item("ITER-DATASET")
        return PipelineRunner(self.builder, self.graph)
    
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
        self.graph.start = start_item("ITER-RANGE")
        return PipelineRunner(self.builder, self.graph)


class PipelineRunner:

    def __init__(self, builder: PipelineBuilder, graph: Graph = None):
        self.builder = builder
        self.step_index = 0
        self.graph = graph if graph else Graph()

    def __name(self, name: str):
        return f"{name}--{self.step_index}"

    def step(self, step, name: str = "PY-STEP"):
        self.builder.add_py_step(self.__name(name), PyStepWrapper(step))
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def ifelse(self, condition: Callable, then_chain: Chain, else_chain: Chain, name: str = "PY-IFELSE"):
        name = self.__name(name)
        step = type(name.replace("-","_"), (object,), {'check': lambda self, context: condition(context)})()
        self.builder.add_ifelse_step(name, PyConditionWrapper(step), then_chain.steps_chain, else_chain.steps_chain)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def map(self, func: Callable, name: str = "PY-MAP"):
        name = self.__name(name)
        step = type(name.replace("-","_"), (object,), {'process': lambda self, context: func(context)})()
        self.builder.add_py_step(name, PyStepWrapper(step))
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def add_column(self, output: str, func: Callable, name: str = "PY-ADD-COLUMN"):
        def wrapper(context):
            if output in context["data"]:
                print("Warning: Output column already exists, overwriting it.")
            context["data"][output] = func(context["data"])
            return context

        self.map(wrapper, name=name)
        del self.graph.steps[-1]
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self

    def filter(self, condition: Callable, name: str = "PY-FILTER"):
        def condition_wrapper(context):
            if not condition(context["data"]):
                context["status"] = StepStatus.FAILED.value
            return context

        self.map(condition_wrapper, name=name)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def mutate(self, output: str, func: Callable, name: str = "PY-ADD-COLUMN"):
        def wrapper(context):
            context["data"][output] = func(context["data"][output])
            return context

        self.map(wrapper, name=name)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self

    def generate_text(self, template: str, llm: str, output: str, system_template: str = None, max_tokens: int = 1024, temperature: float = 0.1, name: str = "GENERATE-TEXT"):
        self.builder.add_text_generation_step(self.__name(name), template, llm, output, system_template, max_tokens, temperature)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self

    def generate_json(self, template: str, llm: str, output: str, json_path: str = None, system_template: str = None, response_format: BaseModel = None, schema_template: str = None, max_tokens: int = 1024, temperature: float = 0.1, name: str = "GENERATE-JSON"):
        schema = None
        if not schema_template and response_format:
            schema = {
                "name": response_format.__class__.__name__,
                "schema": response_format.model_json_schema(),
                "strict": True
            }
            schema["schema"]["additionalProperties"] = False
            schema = json.dumps(schema)

        self.builder.add_json_generation_step(self.__name(name), template, llm, output, json_path, system_template, schema, max_tokens, temperature, schema_template)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def generate_structured(self, template: str, llm: str, output: str, response_format: BaseModel, system_template: str = None, max_tokens: int = 1024, temperature: float = 0.1, name: str = "GENERATE-JSON"):
        return self.generate_json(template, llm, output, json_path=None, system_template=system_template, response_format=response_format, max_tokens=max_tokens, temperature=temperature, name=name)
    
    def sample(self, dataset: str, size: int, output: str, name: str = "SAMPLE"):
        self.builder.add_data_sampler_step(self.__name(name), dataset, size, output)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self

    def render(self, template: str, output: str, name: str = "RENDER"):
        self.builder.add_render_step(self.__name(name), template, output)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def validate_json(self, schema: str, instance: str, name: str = "VALIDATE-JSON"):
        self.builder.add_validatejson_step(self.__name(name), schema, instance)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self

    def chunk(self, capacity: Tuple[int, int], input: str, output: str, name: str = "CHUNK"):
        self.builder.add_chunk_step(self.__name(name), capacity, input, output)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def read(self, dataset: str, output: str, name: str = "SAMPLE"):
        self.builder.add_data_read_step(self.__name(name), dataset, output)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self
    
    def judge(self, template: str, llm: str, name: str = "JUDGE"):
        self.builder.add_judge_step(self.__name(name), template, llm)
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self

    def validate(self, py_func, name: str = "VALIDATE"):
        self.builder.add_py_validator_step(self.__name(name), PyStepValidatorWrapper(py_func))
        self.graph.steps.append(step_item(name=self.__name(name)))
        self.step_index += 1
        return self

    def write_jsonl(self, path: str, template: str, name: str = "WRITE-JSONL"):
        self.builder.add_write_jsonl_step(self.__name(name), path, template)
        self.graph.steps.append(step_item(name=self.__name(name)))
        return self
    
    def write_csv(self, path: str, columns: List[str], delimeter: str, name: str = "WRITE-JSONL"):
        self.builder.add_write_csv_step(self.__name(name), path, columns, delimeter)
        self.graph.steps.append(step_item(name=self.__name(name)))
        return self

    def print(self, *args, **kwargs):
        template = kwargs.get('template', None)
        columns = kwargs.get('columns', None)
        if len(args) == 1:
            columns = args[0]
            
        name = "PRINT"
        self.builder.add_print_step(self.__name(name), template=template, columns=columns)
        self.graph.steps.append(step_item(name=self.__name(name)))
        return self

    def debug(self, target: str = None):
        self.log(LogLevel.DEBUG.value, target)
        return self

    def log(self, level: str = LogLevel.ERROR.value, target: str = None):
        file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.builder.log(level, target, file)
        return self
    
    def run(self):
        self.builder.compile()
        return self.builder.run()
    
    def ui(self, host: str = "0.0.0.0", port: int = 8080):
        self.builder.compile()
        try:
            from tweaktune.ui import run_ui
            run_ui(self.builder, self.graph, host, port)
        except ModuleNotFoundError:
            package_installation_hint("nicegui")
            raise
        return self
    

class ChatTemplateBuilder:
    def __init__(self, path: str = None, template: str = None):
        if not path and not template:
            raise ValueError("Either path or template must be provided.")
        if path and template:
            raise ValueError("Only one of path or template can be provided.")
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Template file not found: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                template = f.read()
        if not template:
            raise ValueError("Template cannot be None or empty.")
        if not isinstance(template, str):
            raise TypeError("Template must be a string.")
        self.builder = _ChatTemplateBuilder(template)

    def with_tools_json(self, tools):
        self.builder.with_tools(json.dumps(tools, ensure_ascii=False))
        return self
    
    def with_tools(self, tools: List[callable]):
        """Converts a list of functions to json schema and adds them to the pipeline."""
        json_list = [json.loads(function_to_json_schema(tool)) for tool in tools]
        for tool in json_list:
            tool["parameters"]["properties"] = json.loads(tool["parameters"]["properties"])
        return self.with_tools_json(json_list)
    
    def build(self):
        """Builds the chat template."""
        self.builder.build()
        return ChatTemplate(self.builder)

class ChatTemplate:
    def __init__(self, builder: _ChatTemplateBuilder):
        self.builder = builder

    def render(self, messages: List[dict]):
        """Renders the chat template with the given context."""
        messages = json.dumps(messages, ensure_ascii=False)
        return self.builder.render(messages)
    
    def render_jsonl(self, path: str, op_config: Optional[dict] = None):
        """Renders the chat template with the given context from a JSONL file."""
        try:
            from datasets import Dataset
        except ModuleNotFoundError:
            package_installation_hint("datasets")
            raise
        op_config = json.dumps(op_config, ensure_ascii=False) if op_config else None
        dataset = {"text": self.builder.render_jsonl(path, op_config)}
        dataset = Dataset.from_dict(dataset)
        return dataset