use crate::steps::{PyStep, PyValidator, StepType};
use arrow::{
    array::{Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
    ffi_stream::ArrowArrayStreamReader,
    pyarrow::PyArrowType,
};
use futures::stream::{self, StreamExt};
use pyo3::{pyclass, pymethods, PyAny, PyObject};
use std::{collections::HashMap, os::unix::process, sync::Arc};
use tokio::runtime::Runtime;
use tweaktune_core::{
    datasets::{ArrowDataset, CsvDataset, DatasetType, JsonlDataset, ParquetDataset},
    embeddings::{EmbeddingsType, OpenAIEmbeddings},
    llms::{LLMType, OpenAILLM},
    steps::{
        CsvWriterStep, JsonlWriterStep, PrintStep, Step as StepCore, StepContext, StepStatus,
        TextGenerationStep,
    },
    templates::Templates,
};

#[pyclass]
pub struct PipelineBuilder {
    workers: usize,
    datasets: Resources<DatasetType>,
    templates: Templates,
    llms: Resources<LLMType>,
    embeddings: Resources<EmbeddingsType>,
    steps: Vec<StepType>,
    iter_by: IterBy,
}

#[pymethods]
impl PipelineBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            workers: 1,
            datasets: Resources {
                resources: HashMap::new(),
            },
            templates: Templates::default(),
            llms: Resources {
                resources: HashMap::new(),
            },
            embeddings: Resources {
                resources: HashMap::new(),
            },
            steps: vec![],
            iter_by: IterBy::Range {
                start: 0,
                stop: 0,
                step: 1,
            },
        }
    }

    pub fn with_workers(&mut self, workers: usize) {
        self.workers = workers;
    }

    pub fn with_json_dataset(&mut self, name: String, path: String) {
        self.datasets.add(
            name.clone(),
            DatasetType::Jsonl(JsonlDataset::new(name, path)),
        );
    }

    pub fn with_parquet_dataset(&mut self, name: String, path: String) {
        self.datasets.add(
            name.clone(),
            DatasetType::Parquet(ParquetDataset::new(name, path)),
        );
    }

    pub fn with_arrow_dataset(
        &mut self,
        name: String,
        mut reader: PyArrowType<ArrowArrayStreamReader>,
    ) {
        let mut batches = Vec::new();
        for batch in reader.0.by_ref() {
            batches.push(batch.unwrap());
        }

        self.datasets.add(
            name.clone(),
            DatasetType::Arrow(ArrowDataset::new(name, batches)),
        );
    }

    pub fn with_csv_dataset(
        &mut self,
        name: String,
        path: String,
        delimiter: String,
        has_header: bool,
    ) {
        self.datasets.add(
            name.clone(),
            DatasetType::Csv(CsvDataset::new(
                name,
                path,
                delimiter.as_bytes()[0],
                has_header,
            )),
        );
    }

    pub fn with_openai_llm(
        &mut self,
        name: String,
        base_url: String,
        api_key: String,
        model: String,
    ) {
        self.llms.add(
            name.clone(),
            LLMType::OpenAI(OpenAILLM::new(name, base_url, api_key, model)),
        );
    }

    pub fn with_openai_embeddings(
        &mut self,
        name: String,
        base_url: String,
        api_key: String,
        model: String,
    ) {
        self.embeddings.add(
            name.clone(),
            EmbeddingsType::OpenAI(OpenAIEmbeddings::new(name, base_url, api_key, model)),
        );
    }

    pub fn with_jinja_template(&mut self, name: String, template: String) {
        self.templates.add(name, template);
    }

    pub fn iter_by_range(&mut self, start: usize, stop: usize, step: usize) {
        self.iter_by = IterBy::Range { start, stop, step };
    }

    pub fn iter_by_dataset(&mut self, name: String) {
        self.iter_by = IterBy::Dataset { name };
    }

    pub fn add_py_step(&mut self, name: String, py_func: PyObject) {
        self.steps.push(StepType::Py(PyStep::new(name, py_func)));
    }

    pub fn add_py_validator_step(&mut self, name: String, py_func: PyObject) {
        self.steps
            .push(StepType::PyValidator(PyValidator::new(name, py_func)));
    }

    pub fn add_text_generation_step(
        &mut self,
        name: String,
        template: String,
        llm: String,
        output: String,
    ) {
        self.steps
            .push(StepType::TextGeneration(TextGenerationStep::new(
                name, template, llm, output,
            )));
    }

    pub fn add_write_jsonl_step(&mut self, name: String, path: String, template: String) {
        self.steps.push(StepType::JsonWriter(JsonlWriterStep::new(
            name, path, template,
        )));
    }

    pub fn add_print_step(
        &mut self,
        name: String,
        template: Option<String>,
        columns: Option<Vec<String>>,
    ) {
        self.steps
            .push(StepType::Print(PrintStep::new(name, template, columns)));
    }

    pub fn add_write_csv_step(
        &mut self,
        name: String,
        path: String,
        columns: Vec<String>,
        delimiter: String,
    ) {
        self.steps.push(StepType::CsvWriter(CsvWriterStep::new(
            name, path, columns, delimiter,
        )));
    }

    pub fn compile(&self) {
        self.templates.compile().unwrap();
    }

    pub fn run(&self) {
        match &self.iter_by {
            IterBy::Range { start, stop, step } => {
                Runtime::new().unwrap().block_on(
                    stream::iter((*start..*stop).step_by(*step).map(|i| async move {
                        let mut context = StepContext::new();
                        context.set("index", i);
                        context.set_status(StepStatus::Running);
                        process_steps(self, context).await;
                        // process
                    }))
                    .buffered(self.workers)
                    .collect::<Vec<_>>(),
                );
            }
            IterBy::Dataset { name: _ } => {
                todo!()
                // let dataset = self.datasets.get(name).unwrap();
                // let data = dataset.read_all().unwrap();
                // process
            }
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

async fn process_steps(pipeline: &PipelineBuilder, mut context: StepContext) {
    for step in &pipeline.steps {
        if matches!(context.get_status(), StepStatus::Failed) {
            break;
        }

        match step {
            StepType::Py(py_step) => {
                context = py_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await
                    .unwrap();
            }
            StepType::TextGeneration(text_generation_step) => {
                context = text_generation_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await
                    .unwrap();
            }
            StepType::PyValidator(py_validator) => {
                context = py_validator
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await
                    .unwrap();
            }
            StepType::JsonWriter(jsonl_writer_step) => {
                context = jsonl_writer_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await
                    .unwrap();
            }
            StepType::CsvWriter(csv_writer_step) => {
                context = csv_writer_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await
                    .unwrap();
            }
            StepType::Print(print_step) => {
                context = print_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await
                    .unwrap();
            }
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub enum IterBy {
    Range {
        start: usize,
        stop: usize,
        step: usize,
    },
    Dataset {
        name: String,
    },
}

#[pyclass]
#[derive(Debug)]
pub enum Template {
    Jinja { name: String, template: String },
}

#[pyclass]
#[derive(Debug)]
pub enum LLM {
    OpenAI {
        name: String,
        model: String,
        base_url: String,
        api_key: String,
    },
}

#[pyclass]
#[derive(Debug)]
pub enum Embeddings {
    OpenAI {
        name: String,
        model: String,
        base_url: String,
        api_key: String,
    },
}

#[pyclass]
#[derive(Debug)]
pub enum Step {
    Py {
        name: String,
        py_func: PyObject,
    },
    TextGeneration {
        name: String,
        template: String,
        llm: String,
        output: String,
    },
    DataSampler {
        name: String,
        dataset: String,
        size: usize,
    },
    Judge {
        name: String,
        template: String,
        llm: String,
    },
    PyValidator {
        name: String,
        py_func: PyObject,
    },
    JsonlWriter {
        name: String,
        path: String,
        template: String,
    },
}

#[pyclass]
#[derive(Debug)]
pub enum Dataset {
    Jsonl {
        name: String,
        path: String,
    },
    Parquet {
        name: String,
        path: String,
    },
    Arrow {
        name: String,
        dataset: PyObject,
    },
    Csv {
        name: String,
        path: String,
        delimiter: String,
        has_header: bool,
    },
}

#[derive(Default, Clone)]
pub struct Resources<T> {
    resources: HashMap<String, T>,
}

impl<T> Resources<T> {
    pub fn add(&mut self, name: String, resource: T) {
        self.resources.insert(name, resource);
    }

    pub fn list(&self) -> Vec<String> {
        self.resources.keys().cloned().collect()
    }

    pub fn get(&self, name: &str) -> Option<&T> {
        self.resources.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<T> {
        self.resources.remove(name)
    }
}
