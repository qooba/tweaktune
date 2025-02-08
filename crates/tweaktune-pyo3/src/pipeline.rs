use crate::steps::{PyStep, StepType};
use arrow::{
    array::{Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
    ffi_stream::ArrowArrayStreamReader,
    pyarrow::PyArrowType,
};
use futures::stream::{self, StreamExt};
use pyo3::{pyclass, pymethods, PyObject};
use std::{collections::HashMap, sync::Arc};
use tokio::runtime::Runtime;
use tweaktune_core::{
    datasets::{ArrowDataset, CsvDataset, DatasetType, JsonlDataset, ParquetDataset},
    embeddings::{EmbeddingsType, OpenAIEmbeddings},
    llms::{LLMType, OpenAILLM},
    steps::{Step as StepCore, StepContext, TextGenerationStep},
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
            iter_by: IterBy::Range { range: 0 },
        }
    }

    pub fn set_workers(&mut self, workers: usize) {
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

    pub fn iter_by_range(&mut self, range: usize) {
        self.iter_by = IterBy::Range { range };
    }

    pub fn iter_by_dataset(&mut self, name: String) {
        self.iter_by = IterBy::Dataset { name };
    }

    pub fn add_py_step(&mut self, name: String, py_func: PyObject) {
        self.steps.push(StepType::Py(PyStep::new(name, py_func)));
    }

    pub fn add_text_generation_step(&mut self, name: String, template: String, llm: String) {
        self.steps
            .push(StepType::TextGeneration(TextGenerationStep::new(
                name, template, llm,
            )));
    }

    pub fn run(&self) {
        match &self.iter_by {
            IterBy::Range { range } => {
                Runtime::new().unwrap().block_on(
                    stream::iter((0..*range).map(|i| async move {
                        let mut context = StepContext::new();
                        context.set("index", i);

                        for step in &self.steps {
                            match step {
                                StepType::Py(py_step) => {
                                    context = py_step
                                        .process(
                                            &self.datasets.resources,
                                            &self.templates,
                                            &self.llms.resources,
                                            &self.embeddings.resources,
                                            &context,
                                        )
                                        .await
                                        .unwrap();
                                }
                                StepType::TextGeneration(text_generation_step) => {
                                    text_generation_step
                                        .process(
                                            &self.datasets.resources,
                                            &self.templates,
                                            &self.llms.resources,
                                            &self.embeddings.resources,
                                            &context,
                                        )
                                        .await
                                        .unwrap();
                                }
                            }
                        }
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

#[pyclass]
#[derive(Debug)]
pub enum IterBy {
    Range { range: usize },
    Dataset { name: String },
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
    Validator {
        name: String,
        template: String,
        llm: String,
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
