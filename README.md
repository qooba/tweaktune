

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



class CustomStep:

    def process(self, context):
        
        context["data"]["test"] = "world " + str(context["data"]["index"])
        return context

def validate_mod(context):
    return context["data"]["index"] % 2 == 0

p = Pipeline()\
        .with_workers(1)\
        .with_jsonl_dataset("my_jsonl", "/home/jovyan/SpeakLeash/swaggset/datasets/persona_pl_full.jsonl")\
        .with_parquet_dataset("my_pq", "/home/jovyan/SpeakLeash/swaggset/datasets/persona_pl_full.parquet")\
        .with_csv_dataset("my_csv", "/home/jovyan/SpeakLeash/swaggset/datasets/persona_pl_full.csv", delimiter=",", has_header=True)\
        .with_template("json_template", """{"test": "{{test}}", "index": "{{index}}", "output": "{{new_output}}"}""")\
        .with_template("calc_template", """Ile jest 2 + {{index}}. Zwróć sam wynik nie dodawaj nic więcej.""")\
        .with_openai_llm("my_llm", "http://localhost:8093", "api_key", "speakleash/Bielik-11B-v2.3-Instruct")\
    .iter_range(3)\
        .step(CustomStep())\
        .print(["test", "index"])\
        .validate(lambda context: context["data"]["index"] % 2 == 0)\
        .validate(validate_mod)\
        .generate_text("calc_template","my_llm", "new_output")\
        .print(["test", "index", "new_output"])\
        .write_jsonl("/home/jovyan/SpeakLeash/tweaktune/notebooks/output.jsonl","json_template")\
        .write_csv("/home/jovyan/SpeakLeash/tweaktune/notebooks/output.csv",["index","test"],",")\
        .print(template="json_template")\
        .print()\
    .run()    
```


```
<s><|im_start|>system
You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.
[{'name': 'find_recipes', 'description': 'Find recipes based on dietary restrictions, meal type, and preferred ingredients. Note that the provided function is in Python 3 syntax.', 'parameters': {'type': 'dict', 'properties': {'diet': {'type': 'string', 'description': "The dietary restrictions, e.g., 'vegan', 'gluten-free'."}, 'meal_type': {'type': 'string', 'description': "The type of meal, e.g., 'dinner', 'breakfast'."}, 'ingredients': {'type': 'array', 'items': {'type': 'string'}, 'description': 'The preferred ingredients. If left blank, it will default to return general recipes.'}}, 'required': ['diet', 'meal_type']}}]

<|im_end|>
<|im_start|>user
What are some gluten-free recipes for dinner?<|im_end|>
<|im_start|>assistant
```
