

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