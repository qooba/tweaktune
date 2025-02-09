from enum import Enum
import pyarrow as pa
from .tweaktune import StepTest, StepConfigTest
from .tweaktune import Step, Jsonl, Parquet, Csv, Arrow, Lang, PipelineBuilder, IterBy, Dataset, LLM, Embeddings, Template
from datasets.arrow_dataset import Dataset as ArrowDataset
from pyarrow.lib import RecordBatchReader
import json
from typing import List, overload

#def hello():
#    return "Hello, World!"
#
#def get_buffer(step: Step):
#    b = step.create_arrow_buffer()
#    reader = pa.ipc.open_stream(b)
#
#    d = reader.read_all()
#    return d.to_pandas()
#
#def read_buffer(step: Step, data: pa.lib.Table):
#    sink = pa.BufferOutputStream()
#    writer = pa.ipc.new_stream(sink, data.schema)
#    writer.write_table(data)
#    writer.close()
#    buffer = sink.getvalue().to_pybytes()
#    step.read_pyarrow(buffer)



class PyStepWrapper:
    def __init__(self, step):
        self.step = step

    def process(self, context):
        context = json.loads(context)
        return json.dumps(self.step.process(context))
    
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
            self.builder.with_json_dataset(dataset.name, dataset.path)
        elif dataset.__class__ == Dataset.Parquet:
            self.builder.with_parquet_dataset(dataset.name, dataset.path)
        elif dataset.__class__ == Dataset.Csv:
            self.builder.with_csv_dataset(dataset.name, dataset.path, dataset.delimiter, dataset.has_header)
        elif dataset.__class__ == Dataset.Arrow:
            if type(dataset.dataset) is ArrowDataset:
                self.builder.with_arrow_dataset(dataset.name, dataset.dataset.data.to_reader())
            elif type(dataset.dataset) is RecordBatchReader:
                self.builder.with_arrow_dataset(dataset.name, dataset.dataset)
            else:
                raise ValueError("Invalid dataset type")
        else:
            raise ValueError("Invalid dataset type")
        
        return self
    
    def with_jsonl_dataset(self, name: str, path: str):
        self.builder.with_json_dataset(name, path)
        return self
    
    def with_parquet_dataset(self, name: str, path: str):
        self.builder.with_parquet_dataset(name, path)
        return self
    
    def with_csv_dataset(self, name: str, path: str, delimiter: str, has_header: bool):
        self.builder.with_csv_dataset(name, path, delimiter, has_header)
        return self
    
    def with_arrow_dataset(self, name: str, dataset):
        if type(dataset) is ArrowDataset:
            self.builder.with_arrow_dataset(name, dataset.data.to_reader())
        elif type(dataset) is RecordBatchReader:
            self.builder.with_arrow_dataset(name, dataset)
        else:
            raise ValueError("Invalid dataset type")

        return self

    def with_template(self, name: str, template: str):
        self.builder.with_jinja_template(name, template)
        return self
    
    def with_llm(self, llm: LLM):
        if llm.__class__ == LLM.OpenAI:
            self.builder.with_openai_llm(llm.name, llm.base_url, llm.api_key, llm.model, llm.max_tokens)
        else:
            raise ValueError("Invalid LLM type")
        
        return self
    
    def with_openai_llm(self, name: str, base_url: str, api_key: str, model: str, max_tokens: int = 250):
        self.builder.with_openai_llm(name, base_url, api_key, model, max_tokens)
        return self

    def with_embedings(self, embeddings: Embeddings):
        if embeddings.__class__ == Embeddings.OpenAI:
            self.builder.with_embeddings(embeddings.name, embeddings.model, embeddings.base_url, embeddings.api_key)
        else:
            raise ValueError("Invalid Embeddings type")
        
        return self
    
    def with_workers(self, workers: int):
        self.builder.with_workers(workers)
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

    def generate_text(self, template: str, llm: str, output: str, system_template: str = None, name: str = "GENERATE-TEXT"):
        self.builder.add_text_generation_step(self.__name(name), template, llm, output, system_template)
        self.step_index += 1
        return self
    
    def sample(self, dataset: str, size: str, name: str = "SAMPLE"):
        self.builder.add_data_sampler_step(self.__name(name), dataset, size)
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
    
    def run(self):
        return self.builder.run()