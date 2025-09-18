pub mod conversations;
pub mod generators;
pub mod logic;
pub mod py;
pub mod quality;
pub mod validators;
pub mod writers;
use crate::{
    common::{df_to_values, OptionToResult},
    datasets::{Dataset, DatasetType},
    embeddings::{self, EmbeddingsType},
    llms::{self, LLMType},
    state::State,
    steps::{
        conversations::{RenderConversationStep, RenderToolCallStep},
        generators::{JsonGenerationStep, TextGenerationStep},
        logic::{FilterStep, MutateStep},
        py::{PyStep, PyValidator},
        quality::{CheckHashStep, CheckLanguageStep, CheckSimHashStep},
        validators::{
            ConversationValidateStep, ToolsNormalizeStep, ToolsValidateStep, ValidateJsonStep,
        },
        writers::{CsvWriterStep, JsonlWriterStep},
    },
    templates::Templates,
};
use anyhow::Result;
use log::error;
use pyo3::prelude::*;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use text_splitter::{Characters, TextSplitter};

pub type StepContextData = serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepContext {
    pub id: uuid::Uuid,
    status: StepStatus,
    pub data: StepContextData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl StepContext {
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            data: json!({}),
            status: StepStatus::Pending,
        }
    }

    pub fn set_status(&mut self, status: StepStatus) {
        self.status = status;
    }

    pub fn get_status(&self) -> &StepStatus {
        &self.status
    }

    pub fn set<T: serde::Serialize>(&mut self, key: &str, value: T) {
        self.data[key] = serde_json::to_value(value).unwrap();
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key)
    }
}

impl Default for StepContext {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Step {
    fn process(
        &self,
        datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        llms: &HashMap<String, llms::LLMType>,
        embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        state: Option<State>,
    ) -> impl std::future::Future<Output = Result<StepContext>>;
}

pub enum StepType {
    IfElse(IfElseStep),
    Py(PyStep),
    PyValidator(PyValidator),
    TextGeneration(TextGenerationStep),
    JsonGeneration(JsonGenerationStep),
    JsonWriter(JsonlWriterStep),
    CsvWriter(CsvWriterStep),
    Print(PrintStep),
    DataSampler(DataSamplerStep),
    Chunk(ChunkStep),
    Render(RenderStep),
    ValidateJson(ValidateJsonStep),
    ValidateTools(ToolsValidateStep),
    NormalizeTools(ToolsNormalizeStep),
    ConversationValidate(ConversationValidateStep),
    IntoList(IntoListStep),
    RenderConversation(RenderConversationStep),
    Filter(FilterStep),
    Mutate(MutateStep),
    CheckLanguage(CheckLanguageStep),
    RenderToolCall(RenderToolCallStep),
    CheckHash(CheckHashStep),
    CheckSimHash(CheckSimHashStep),
}

pub struct IfElseStep {
    pub name: String,
    pub py_condition: Option<PyObject>,
    pub condition_key: Option<String>,
    pub then_steps: Vec<StepType>,
    pub else_steps: Option<Vec<StepType>>,
}

impl IfElseStep {
    pub fn new(
        name: String,
        py_condition: Option<PyObject>,
        condition_key: Option<String>,
        then_steps: Vec<StepType>,
        else_steps: Option<Vec<StepType>>,
    ) -> Self {
        Self {
            name,
            py_condition,
            condition_key,
            then_steps,
            else_steps,
        }
    }

    pub async fn check(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, LLMType>,
        _embeddings: &HashMap<String, EmbeddingsType>,
        context: &StepContext,
    ) -> Result<bool> {
        let json = serde_json::to_string(context)?;

        let result = if let Some(condition) = &self.py_condition {
            let result: PyResult<bool> = Python::with_gil(|py| {
                let result: bool = condition.call_method1(py, "check", (json,))?.extract(py)?;
                Ok(result)
            });

            anyhow::Ok(result?)
        } else if let Some(key) = &self.condition_key {
            let rendered = templates.render(key.clone(), context.data.clone())?;
            if let Ok(v) = serde_json::from_str::<bool>(&rendered) {
                anyhow::Ok(v)
            } else {
                error!(target: "ifelsestep", "🐔 Condition is not a boolean: {}", rendered);
                return Err(anyhow::anyhow!("Condition is not a boolean"));
            }
        } else {
            Err(anyhow::anyhow!(
                "Either py_condition or condition_key must be provided"
            ))
        };

        match result {
            Ok(result) => Ok(result),
            Err(e) => {
                error!(target: "ifelsestep", "🐔 {:?}", e);
                let mut context = context.clone();
                context.set_status(StepStatus::Failed);
                Ok(false)
            }
        }
    }
}

impl Step for IfElseStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, LLMType>,
        _embeddings: &HashMap<String, EmbeddingsType>,
        _context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        unreachable!("Use check method to evaluate condition");
    }
}

pub struct RenderStep {
    pub name: String,
    pub template: String,
    pub output: String,
}

impl RenderStep {
    pub fn new(name: String, template: String, output: String) -> Self {
        Self {
            name,
            template,
            output,
        }
    }
}

impl Step for RenderStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let rendered = templates.render(self.template.clone(), context.data.clone())?;
        context.set(&self.output, rendered);
        Ok(context)
    }
}

pub struct PrintStep {
    pub name: String,
    pub template: Option<String>,
    pub columns: Option<Vec<String>>,
}

impl PrintStep {
    pub fn new(name: String, template: Option<String>, columns: Option<Vec<String>>) -> Self {
        Self {
            name,
            template,
            columns,
        }
    }
}

impl Step for PrintStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        let mut row = if let Some(template) = self.template.clone() {
            templates.render(template.clone(), context.data.clone())?
        } else if let Some(columns) = self.columns.clone() {
            let mut row = String::new();
            for (i, column) in columns.iter().enumerate() {
                if let Some(value) = context.data.get(column) {
                    if i > 0 {
                        row.push_str(" | ");
                    }
                    row.push_str(&value.to_string());
                }
            }

            row
        } else {
            context.data.to_string()
        };

        row.push('\n');

        Python::with_gil(|py| {
            let sys = py.import("sys").unwrap();
            let stdout = sys.getattr("stdout").unwrap();
            let write = stdout.getattr("write").unwrap();
            write.call1((row,)).unwrap();
        });

        // println!("{}", row);

        Ok(context.clone())
    }
}

pub struct DataSamplerStep {
    pub name: String,
    pub dataset: String,
    pub size: Option<usize>,
    pub output: String,
}

// pub fn map_to_json(record_batches: &[RecordBatch]) -> Vec<Vec<serde_json::Value>> {
//     let json_rows = record_batches
//         .iter()
//         .map(|record_batch| {
//             let jv: Vec<serde_json::Value> = serde_arrow::from_record_batch(record_batch).unwrap();
//             jv
//         })
//         .collect();
//     json_rows
// }

// pub fn flat_map_to_json(record_batches: &[RecordBatch]) -> Vec<serde_json::Value> {
//     let json_rows = record_batches
//         .iter()
//         .flat_map(|record_batch| {
//             let jv: Vec<serde_json::Value> = serde_arrow::from_record_batch(record_batch).unwrap();
//             jv
//         })
//         .collect();
//     json_rows
// }

impl DataSamplerStep {
    pub fn new(name: String, dataset: String, size: Option<usize>, output: String) -> Self {
        Self {
            name,
            dataset,
            size,
            output,
        }
    }
}

impl Step for DataSamplerStep {
    async fn process(
        &self,
        datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let dataset_type = datasets
            .get(&self.dataset)
            .ok_or_err(&self.dataset)
            .unwrap();

        let json_rows = if let DatasetType::Mixed(mixed_dataset) = dataset_type {
            mixed_dataset.sample(self.size.unwrap(), datasets)?
        } else {
            let df = match dataset_type {
                DatasetType::Polars(polars_dataset) => polars_dataset.df(),
                DatasetType::Json(json_dataset) => json_dataset.df(),
                DatasetType::JsonList(json_list_dataset) => json_list_dataset.df(),
                DatasetType::OpenApi(openapi_dataset) => openapi_dataset.df(),
                DatasetType::Ipc(ipc_dataset) => ipc_dataset.df(),
                DatasetType::Csv(csv_dataset) => csv_dataset.df(),
                DatasetType::Parquet(parquet_dataset) => parquet_dataset.df(),
                DatasetType::Jsonl(jsonl_dataset) => jsonl_dataset.df(),
                DatasetType::Mixed(_mixed_dataset) => unreachable!(),
            };

            let df = df
                .sample_n_literal(
                    self.size.unwrap_or(df.size()),
                    false,
                    false,
                    Some(rand::rng().next_u64()),
                )
                .unwrap();

            df_to_values(&df)?
        };

        context.set(&self.output, json_rows);
        Ok(context)
    }
}

pub struct ChunkStep {
    pub name: String,
    pub capacity: (usize, usize),
    pub input: String,
    pub output: String,
    pub text_splitter: TextSplitter<Characters>,
}

impl ChunkStep {
    pub fn new(name: String, capacity: (usize, usize), input: String, output: String) -> Self {
        let range = capacity.0..capacity.1;
        let text_splitter = TextSplitter::new(range);

        Self {
            name,
            capacity,
            input,
            output,
            text_splitter,
        }
    }
}

impl Step for ChunkStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let text = context
            .get(&self.input)
            .ok_or_else(|| anyhow::anyhow!("Input not found"))?;
        let chunks: Vec<serde_json::Value> = self
            .text_splitter
            .chunks(&text.to_string())
            .collect::<Vec<&str>>()
            .iter()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();

        context.set(&self.output, chunks);
        Ok(context)
    }
}

pub struct IntoListStep {
    pub name: String,
    pub inputs: Vec<String>,
    pub output: String,
}

impl IntoListStep {
    pub fn new(name: String, inputs: Vec<String>, output: String) -> Self {
        Self {
            name,
            inputs,
            output,
        }
    }
}

impl Step for IntoListStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let list = self
            .inputs
            .iter()
            .map(|input| context.get(input).cloned().expect("🐔 Input not found"))
            .collect::<Vec<serde_json::Value>>();
        context.set(&self.output, list);
        Ok(context)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        println!("hello");
    }

    #[test]
    fn schema_validate() {
        use serde_json::Value;

        let raw_schema = r#"{
            "type": "object",
            "properties": {
                    "critic": {
                        "description": "Nazwisko krytyka",
                        "type": "string"
                    },
                    "title": {
                        "description": "Tytuł filmu",
                        "type": "string"
                    }
            }
        }"#;

        let instance_json = r#"{
            "critic": "Jan Kowalski",
            "title": "Incepcja"
        }"#;

        let full_schema: Value = serde_json::from_str(raw_schema).unwrap();
        let instance: Value = serde_json::from_str(instance_json).unwrap();

        // instance.size is an integer but schema expects a string -> validation should succeed
        assert!(jsonschema::is_valid(&full_schema, &instance));
        println!("hello");
    }

    #[test]
    fn schema_validate2() {
        use serde_json::Value;

        let raw_schema = r#"{
            "type": "object",
            "properties": {
                "color": {
                    "description": "Typ kolorystyczny obrazów do wyszukania",
                    "enum": [
                        "color",
                        "black-and-white"
                    ],
                    "type": "string"
                },
                "keywords": {
                    "description": "Słowa kluczowe do wyszukiwania obrazów",
                    "items": {
                        "type": "string"
                    },
                    "type": "array"
                },
                "license": {
                    "description": "Typ licencji obrazów do wyszukania",
                    "enum": [
                        "public",
                        "commercial",
                        "any"
                    ],
                    "type": "string"
                },
                "size": {
                    "description": "Rozmiar obrazów do wyszukania",
                    "enum": [
                        "small",
                        "medium",
                        "large"
                    ],
                    "type": "string"
                }
            }
        }"#;

        let instance_json = r#"{
            "color": "black-and-white",
            "keywords": ["cat", "dog"],
            "license": "public",
            "size": 1
        }"#;

        let full_schema: Value = serde_json::from_str(raw_schema).unwrap();
        let instance: Value = serde_json::from_str(instance_json).unwrap();

        assert!(!jsonschema::is_valid(&full_schema, &instance));
        println!("hello");
    }
}
