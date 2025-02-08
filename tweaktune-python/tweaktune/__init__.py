from enum import Enum
import pyarrow as pa
from .tweaktune import StepTest, StepConfigTest
from .tweaktune import Step, Jsonl, Parquet, Csv, Arrow, Lang, PipelineBuilder, IterBy, Dataset, LLM, Embeddings, Template
from datasets.arrow_dataset import Dataset as ArrowDataset
from pyarrow.lib import RecordBatchReader

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


class Pipeline:
    def __init__(self):
        self.builder = PipelineBuilder()

    def with_dataset(self, dataset: Dataset):
        if dataset.__class__ == Dataset.Jsonl:
            self.builder.with_json_dataset(dataset.name, dataset.path)
        elif dataset.__class__ == Dataset.Parquet:
            self.builder.with_parquet_dataset(dataset.name, dataset.path)
        elif dataset.__class__ == Dataset.Csv:
            self.builder.with_csv_dataset(dataset.name, dataset.path)
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

    def with_template(self, template: Template):
        if template.__class__ == Template.Jinja:
            self.builder.with_jinja_template(template.name, template.template)
        else:
            raise ValueError("Invalid template type")
        return self
    
    def with_llm(self, llm: LLM):
        if llm.__class__ == LLM.OpenAI:
            self.builder.with_openai_llm(llm.name, llm.model, llm.base_url, llm.api_key)
        else:
            raise ValueError("Invalid LLM type")
        
        return self

    def with_embedings(self, embeddings: Embeddings):
        if embeddings.__class__ == Embeddings.OpenAI:
            self.builder.with_embeddings(embeddings.name, embeddings.model, embeddings.base_url, embeddings.api_key)
        else:
            raise ValueError("Invalid Embeddings type")
        
        return self
    
    def iter(self, iter_by: IterBy, workers: int = 1):
        if iter_by.__class__ == IterBy.Range:
            self.builder.iter_by_range(iter_by.range)
        elif iter_by.__class__ == IterBy.Dataset:
            self.builder.iter_by_dataset(iter_by.name)
        else:
            raise ValueError("Invalid IterBy type")
        
        return PipelineRunner(self.builder)
   
class PipelineRunner:

    def __init__(self, builder: PipelineBuilder):
        self.builder = builder

    def then(self, step: Step):
        if step.__class__ == Step.Py:
            self.builder.add_py_step(step.name, step.py_func)
        elif step.__class__ == Step.TextGeneration:
            self.builder.add_text_generation_step(step.name, step.template, step.llm)
        elif step.__class__ == Step.DataSampler:
            self.builder.add_data_sampler_step(step.name, step.dataset, step.size)
        elif step.__class__ == Step.Judge:
            self.builder.add_judge_step(step.name, step.template, step.llm)
        elif step.__class__ == Step.Validator:
            self.builder.add_validator_step(step.name, step.template, step.llm)
        else:
            raise ValueError("Invalid Step type")
        
        return self

    def run(self):
        return self.builder.run()