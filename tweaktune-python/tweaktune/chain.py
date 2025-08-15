import json
from typing import Callable
from tweaktune import StepStatus
from pydantic import BaseModel
from tweaktune.wrappers import PyStepWrapper
from tweaktune.tweaktune import StepsChain

class Chain:
    def __init__(self):
        self.steps_chain = StepsChain()
        self.step_index = 0

    def __name(self, name: str):
        return f"{name}--{self.step_index}"   

    def write_jsonl(self, path: str, template: str, name: str = "WRITE-JSONL"):
        self.steps_chain.add_jsonl_writer_step(self.__name(name), path, template)
        self.step_index += 1
        return self

    def sample(self, dataset: str, size: int, output: str, name: str = "SAMPLE"):
        self.steps_chain.add_data_sampler_step(name, dataset, size, output)
        self.step_index += 1
        return self
    
    def generate_text(self, template: str, llm: str, output: str, system_template: str = None, max_tokens: int = 1024, temperature: float = 0.1, name: str = "GENERATE-TEXT"):
        self.steps_chain.add_text_generation_step(self.__name(name), template, llm, output, system_template, max_tokens, temperature)
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

        self.steps_chain.add_json_generation_step(self.__name(name), template, llm, output, json_path, system_template, schema_template, schema, max_tokens, temperature)   
        self.step_index += 1
        return self
    
    def print(self, *args, **kwargs):
        template = kwargs.get('template', None)
        columns = kwargs.get('columns', None)
        if len(args) == 1:
            columns = args[0]
            
        name = "PRINT"
        self.steps_chain.add_print_step(self.__name(name), template, columns)
        self.step_index += 1
        return self

    def step(self, step, name: str = "PY-STEP"):
        self.steps_chain.add_py_step(self.__name(name),PyStepWrapper(step))
        self.step_index += 1
        return self

    def map(self, func: Callable, name: str = "PY-MAP"):
        name = self.__name(name)
        step = type(name.replace("-","_"), (object,), {'process': lambda self, context: func(context)})()
        self.steps_chain.add_py_step(self.__name(name),PyStepWrapper(step))
        self.step_index += 1
        return self
    
    def add_column(self, output: str, func: Callable, name: str = "PY-ADD-COLUMN"):
        def wrapper(context):
            if output in context["data"]:
                print("Warning: Output column already exists, overwriting it.")
            context["data"][output] = func(context["data"])
            return context

        self.map(wrapper, name=name)
        self.step_index += 1
        return self

    def filter(self, condition: Callable, name: str = "PY-FILTER"):
        def condition_wrapper(context):
            if not condition(context["data"]):
                context["status"] = StepStatus.FAILED.value
            return context

        self.map(condition_wrapper, name=name)
        self.step_index += 1
        return self
    
    def mutate(self, output: str, func: Callable, name: str = "PY-ADD-COLUMN"):
        def wrapper(context):
            context["data"][output] = func(context["data"][output])
            return context

        self.map(wrapper, name=name)
        self.step_index += 1
        return self
    

    

