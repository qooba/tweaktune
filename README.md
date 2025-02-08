

```python
personas = Jsonl("/home/jovyan/SpeakLeash/swaggset/datasets/persona_pl_full.jsonl")

Pipeline(workers=5)\ 
    .iter_range(range=1000)\
    .add_dataset_jsonl(name="tools", path="path")\ #
    .add_dataset_parquet(name="tools", path="path")\
    .add_dataset_list(name="personas", items=personas)\
    .add_template(name="template_name", content="template_content")\
    .add_llm(name="llm_name", api_base="api_base", api_key="api_key", model="model")\
    .build()\ #PipelineRunner
    .add_random(dataset_name="personas", input_columns=["persona"], output_columns=["persona"])\
    .add_random(dataset_name="tools", input_columns=["tool.name"], output_columns=["tool_name"])\
    .generate_text("template_name", "llm_name", "output_name", "json_key")
    .judge("template_name","llm_name","output_name")
    .validate_schema(...)
    .add_multiple_random(dataset_name="tools", min_number=2, max_number=5, output_column="other_tools")\
    .write_jsonl("path", ["output_name"]

Pipeline(workers=5).iter_list(tools).add_random(personas, ["persona"])
```


```python
import tweaktune
from tweaktune import Step, StepConfig, Jsonl, Lang
import json
import pyarrow as pa

t = """sdasdasd
adsadasdsa
sadsadasd
sadsadsa
"""

config = StepConfig(t)
#############
step = Step(config)
step.embed("dsadsad", Lang.Fra)
#############
r=step.create_record_batch()
r.to_pandas()
#############
r=step.create_record_batch_vec()
r[0].to_pandas()
#############
a=step.create_record_array()
a.to_pandas()
#############
import pandas as pd
arr = pa.array(pd.Series([1, 2]))

step.read_array(arr)

import pyarrow as pa
n_legs = pa.array([2, 2, 4, 4, 5, 100])
animals = pa.array(["Flamingo", "Parrot", "Dog", "Horse", "Brittle stars", "Centipede"])
names = ["n_legs", "animals"]
rb = pa.record_batch({"n_legs": n_legs, "animals": animals})
step.read_record_batch(rb)

table = pa.table({"numbers": [10,20,30,40,50,100 ,101]})
rd = table.to_reader()
step.read_reader(rd)

```

```python
import tweaktune
from tweaktune import Pipeline, IterBy, Dataset, LLM, Embeddings, Template, Step
import json

template = """{"test": "{{data.test}}", "index": "{{data.index}}", "status": "{{status}}"}"""

class CustomStep:

    def process(self, context):
        context["data"]["test"] = "world1"
        return context

def validate_mod(context):
    return context["data"]["index"] % 2 == 0

p = Pipeline()\
        .with_dataset(Dataset.Jsonl("my_jsonl", "/home/jovyan/SpeakLeash/swaggset/datasets/persona_pl_full.jsonl"))\
        .with_dataset(Dataset.Parquet("my_pq", "/home/jovyan/SpeakLeash/swaggset/datasets/persona_pl_full.parquet"))\
        .with_dataset(Dataset.Csv("my_csv", "/home/jovyan/SpeakLeash/swaggset/datasets/persona_pl_full.csv", delimiter=",", has_header=True))\
        .with_template(Template.Jinja("my_template", template))\
        .with_llm(LLM.OpenAI("my_llm", "Bielik", "http://localhost:8000", "key"))\
    .iter(IterBy.Range(11))\
        .step(CustomStep())\
        .validate(lambda context: context["data"]["index"] % 2 == 0)\
        .generate_text("my_template","my_llm", "new_output")\
        .write_jsonl("/home/jovyan/SpeakLeash/tweaktune/notebooks/output.jsonl","my_template")\
        .write_csv("/home/jovyan/SpeakLeash/tweaktune/notebooks/output.csv",["index","status","test"],";")\
    .run()
```