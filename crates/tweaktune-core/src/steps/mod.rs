use crate::{
    common::{extract_json, OptionToResult, ResultExt},
    datasets::{Dataset, DatasetType},
    embeddings,
    llms::{self, LLM},
    templates::Templates,
};
use anyhow::Result;
use arrow::{array::RecordBatch, json, row};
use log::{debug, error, info};
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use rand::seq::SliceRandom;
use reqwest::header::CONTENT_DISPOSITION;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{BufReader, Write};
use std::{collections::HashMap, fs::File};

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
        let llms::LLMType::OpenAI(llm) = llm;
        let result = match llm.call(template).await {
            Ok(response) => Some(response.choices[0].message.content.clone()),
            Err(e) => {
                debug!(target: "generate", "Failed to generate text: {}", e);
                None
            }
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
            .generate(_datasets, templates, llms, _embeddings, &context)
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
}

impl JsonGenerationStep {
    pub fn new(
        name: String,
        template: String,
        llm: String,
        output: String,
        json_path: Option<String>,
        system_template: Option<String>,
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
            .generate(_datasets, templates, llms, _embeddings, &context)
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
    pub size: usize,
    pub output: String,
    json_batches: Vec<Vec<serde_json::Value>>,
}

pub fn map_to_json(record_batches: &[RecordBatch]) -> Vec<Vec<serde_json::Value>> {
    let json_rows = record_batches
        .iter()
        .map(|record_batch| {
            let jv: Vec<serde_json::Value> = serde_arrow::from_record_batch(record_batch).unwrap();
            jv
        })
        .collect();
    json_rows
}

pub fn flat_map_to_json(record_batches: &[RecordBatch]) -> Vec<serde_json::Value> {
    let json_rows = record_batches
        .iter()
        .flat_map(|record_batch| {
            let jv: Vec<serde_json::Value> = serde_arrow::from_record_batch(record_batch).unwrap();
            jv
        })
        .collect();
    json_rows
}

impl DataSamplerStep {
    pub fn new(
        name: String,
        dataset: String,
        size: usize,
        output: String,
        datasets: &HashMap<String, DatasetType>,
    ) -> Self {
        let dataset_type = datasets.get(&dataset).ok_or_err(&dataset).unwrap();
        let json_batches = match dataset_type {
            DatasetType::Jsonl(jsonl_dataset) => vec![jsonl_dataset.read_all_json().unwrap()],
            DatasetType::Json(json_dataset) => vec![json_dataset.read_all_json().unwrap()],
            DatasetType::Mixed(mixed_dataset) => {
                vec![mixed_dataset.read_all_json(datasets).unwrap()]
            }
            DatasetType::Csv(csv_dataset) => map_to_json(&csv_dataset.read_all(None).unwrap()),
            DatasetType::Parquet(parquet_dataset) => {
                map_to_json(&parquet_dataset.read_all(None).unwrap())
            }
            DatasetType::Arrow(arrow_dataset) => {
                map_to_json(&arrow_dataset.read_all(None).unwrap())
            }
        };

        Self {
            name,
            dataset,
            size,
            output,
            json_batches,
        }
    }
}

impl Step for DataSamplerStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let mut rng = rand::thread_rng();
        let json_rows = self.json_batches.choose(&mut rng).unwrap();
        let json_rows: Vec<Vec<serde_json::Value>> = json_rows
            .choose_multiple(&mut rng, self.size)
            .cloned()
            .map(|row| vec![row])
            .collect();

        context.set(&self.output, json_rows);
        Ok(context)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        println!("hello");
    }
}
