use crate::{
    common::{df_to_values, extract_json, OptionToResult, ResultExt},
    datasets::{Dataset, DatasetType},
    embeddings::{self, EmbeddingsType},
    llms::{self, LLMType, LLM},
    templates::Templates,
};
use anyhow::Result;
use log::debug;
use pyo3::prelude::*;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::{collections::HashMap, fs::File};
use text_splitter::{Characters, TextSplitter};

pub type StepContextData = serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepContext {
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
    ) -> impl std::future::Future<Output = Result<StepContext>>;
}

pub enum StepType {
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
}

pub struct PyStep {
    pub name: String,
    pub py_func: PyObject,
}

impl PyStep {
    pub fn new(name: String, py_func: PyObject) -> Self {
        Self { name, py_func }
    }
}

impl Step for PyStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, LLMType>,
        _embeddings: &HashMap<String, EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let json = serde_json::to_string(context)?;

        let result: PyResult<String> = Python::with_gil(|py| {
            let result: String = self
                .py_func
                .call_method1(py, "process", (json,))?
                .extract(py)?;
            Ok(result)
        });

        match result {
            Ok(result) => {
                let result: StepContext = serde_json::from_str(&result)?;
                Ok(result)
            }
            Err(e) => {
                debug!("{:?}", e);
                let mut context = context.clone();
                context.set_status(StepStatus::Failed);
                Ok(context)
            }
        }
    }
}

pub struct PyValidator {
    pub name: String,
    pub py_func: PyObject,
}

impl PyValidator {
    pub fn new(name: String, py_func: PyObject) -> Self {
        Self { name, py_func }
    }
}

impl Step for PyValidator {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, LLMType>,
        _embeddings: &HashMap<String, EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let json = serde_json::to_string(context)?;

        let result: PyResult<bool> = Python::with_gil(|py| {
            let result: bool = self
                .py_func
                .call_method1(py, "process", (json,))?
                .extract(py)?;
            Ok(result)
        });

        let result = result.map_tt_err("VALIDATOR MUST RETURN BOOL")?;
        let mut context = context.clone();
        if !result {
            context.set_status(StepStatus::Failed);
        }

        Ok(context)
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
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let rendered = templates.render(self.template.clone(), context.data.clone())?;
        context.set(&self.output, rendered);
        Ok(context)
    }
}

pub struct ValidateJsonStep {
    pub name: String,
    pub schema: String,
    pub instance: String,
}

impl ValidateJsonStep {
    pub fn new(name: String, schema: String, instance: String) -> Self {
        Self {
            name,
            schema,
            instance,
        }
    }
}

impl Step for ValidateJsonStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let schema = templates.render(self.schema.clone(), context.data.clone())?;
        let full_schema: Value = serde_json::from_str(&schema).unwrap();

        let properties = if let Value::String(v) = full_schema["properties"].clone() {
            serde_json::from_str(&v).unwrap()
        } else {
            full_schema["properties"].clone()
        };

        let schema_value = json!({
            "type": "object",
            "properties": properties,
            "required": full_schema["required"],
            "additionalProperties": false,
        });

        let instance_json = templates.render(self.instance.clone(), context.data.clone())?;
        match serde_json::from_str(&instance_json) {
            Ok(instance) => {
                let is_valid = jsonschema::is_valid(&schema_value, &instance);

                if !is_valid {
                    debug!(target: "generate", "Failed to validate JSON: {} with schema {}", instance, schema_value);
                    context.set_status(StepStatus::Failed);
                }

                Ok(context)
            }
            Err(e) => {
                debug!(target: "generate", "Failed to render instance: {}", e);
                context.set_status(StepStatus::Failed);
                Ok(context)
            }
        }
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

pub struct TextGenerationStep {
    pub name: String,
    pub template: String,
    pub system_template: Option<String>,
    pub llm: String,
    pub output: String,
}

impl TextGenerationStep {
    pub fn new(
        name: String,
        template: String,
        llm: String,
        output: String,
        system_template: Option<String>,
    ) -> Self {
        Self {
            name,
            template,
            llm,
            output,
            system_template,
        }
    }

    pub async fn generate(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        json_schema: Option<String>,
    ) -> Result<Option<String>> {
        let template = templates.render(self.template.clone(), context.data.clone());
        let template = match template {
            Ok(t) => t,
            Err(e) => {
                debug!(target: "generate", "Failed to render template: {}", e);
                return Ok(None);
            }
        };

        let llm = llms.get(&self.llm).expect("LLM");
        let result = match llm {
            llms::LLMType::OpenAI(llm) => match llm.call(template, json_schema).await {
                Ok(response) => Some(response.choices[0].message.content.clone()),
                Err(e) => {
                    debug!(target: "generate", "Failed to generate text: {}", e);
                    None
                }
            },
            llms::LLMType::Unsloth(llm) => match llm.call(template, json_schema).await {
                Ok(response) => Some(response.choices[0].message.content.clone()),
                Err(e) => {
                    debug!(target: "generate", "Failed to generate text: {}", e);
                    None
                }
            },
        };

        Ok(result)
    }
}

impl Step for TextGenerationStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let result = self
            .generate(_datasets, templates, llms, _embeddings, &context, None)
            .await?;

        match result {
            Some(value) => {
                context.data[self.output.clone()] = serde_json::to_value(value)?;
            }
            None => {
                context.set_status(StepStatus::Failed);
            }
        };
        Ok(context)
    }
}

pub struct JsonGenerationStep {
    pub name: String,
    pub generation_step: TextGenerationStep,
    pub output: String,
    pub json_path: Option<String>,
    pub json_schema: Option<String>,
}

impl JsonGenerationStep {
    pub fn new(
        name: String,
        template: String,
        llm: String,
        output: String,
        json_path: Option<String>,
        system_template: Option<String>,
        json_schema: Option<String>,
    ) -> Self {
        Self {
            generation_step: TextGenerationStep::new(
                name.clone(),
                template,
                llm,
                output.clone(),
                system_template,
            ),
            output,
            name,
            json_path,
            json_schema,
        }
    }
}

impl Step for JsonGenerationStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let result = self
            .generation_step
            .generate(
                _datasets,
                templates,
                llms,
                _embeddings,
                &context,
                self.json_schema.clone(),
            )
            .await?;

        match result {
            Some(value) => match extract_json(&value) {
                Ok(mut value) => {
                    if let Some(json_path) = &self.json_path {
                        json_path.split(".").for_each(|key| {
                            value = value[key].clone();
                        });
                    }

                    debug!(target:"generate", "Generated VALUE: {}", value);
                    context.data[self.output.clone()] = value;
                }
                Err(e) => {
                    debug!(target:"generate", "Failed to extract JSON: {}", e);
                    context.set_status(StepStatus::Failed);
                }
            },
            None => {
                context.set_status(StepStatus::Failed);
            }
        };

        Ok(context)
    }
}

pub struct JsonlWriterStep {
    pub name: String,
    pub path: String,
    pub template: String,
}

impl JsonlWriterStep {
    pub fn new(name: String, path: String, template: String) -> Self {
        Self {
            name,
            path,
            template,
        }
    }
}

impl Step for JsonlWriterStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let file = File::options().append(true).create(true).open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);
        let row = templates.render(self.template.clone(), context.data.clone());
        let mut context = context.clone();
        match row {
            Ok(r) => {
                writeln!(writer, "{}", r)?;
                writer.flush()?;
            }
            Err(e) => {
                debug!("Failed to render template: {}", e);
                context.set_status(StepStatus::Failed);
            }
        };

        Ok(context)
    }
}

pub struct CsvWriterStep {
    pub name: String,
    pub path: String,
    pub columns: Vec<String>,
    pub delimeter: String,
}

impl CsvWriterStep {
    pub fn new(name: String, path: String, columns: Vec<String>, delimeter: String) -> Self {
        Self {
            name,
            path,
            columns,
            delimeter,
        }
    }
}

impl Step for CsvWriterStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let file = File::options().append(true).create(true).open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);
        let mut row = String::new();
        for (i, column) in self.columns.iter().enumerate() {
            if let Some(value) = context.get(column) {
                if i > 0 {
                    row.push_str(&self.delimeter);
                }
                row.push_str(&value.to_string());
            }
        }

        writeln!(writer, "{}", row)?;
        writer.flush()?;

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

#[cfg(test)]
mod tests {
    use polars::prelude::full;

    #[test]
    fn it_works() {
        println!("hello");
    }

    #[test]
    fn schema_validate() {
        use serde_json::{json, Value};

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

        assert!(jsonschema::is_valid(&full_schema, &instance));
        println!("hello");
    }

    #[test]
    fn schema_validate2() {
        use serde_json::{json, Value};

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

        assert!(jsonschema::is_valid(&full_schema, &instance));
        println!("hello");
    }
}
