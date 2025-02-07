use arrow::{ffi_stream::ArrowArrayStreamReader, pyarrow::PyArrowType};
use pyo3::{pyclass, pymethods, PyObject, PyRef};
use std::collections::HashMap;
use tweaktune_core::{
    datasets::{ArrowDataset, CsvDataset, DatasetType, JsonlDataset, ParquetDataset},
    embeddings::{EmbeddingsType, OpenAIEmbeddings},
    llms::{LLMType, OpenAILLM},
    templates::Templates,
};

#[pyclass]
pub struct PipelineBuilder {
    workers: usize,
    datasets: Resources<DatasetType>,
    templates: Templates,
    llms: Resources<LLMType>,
    embeddings: Resources<EmbeddingsType>,
    steps: Vec<String>,
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

    pub fn add_step(&mut self, step: &str) {
        self.steps.push(step.to_string());
    }

    pub fn run(&self) {
        println!("{:?}", self.steps);
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

#[derive(Default)]
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
