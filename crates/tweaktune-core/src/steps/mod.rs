use crate::{common::ResultExt, datasets::DatasetType, embeddings, llms, templates::Templates};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashMap, fs::File, io::Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepContext {
    status: StepStatus,
    data: serde_json::Value,
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
    pub llm: String,
    pub output: String,
}

impl TextGenerationStep {
    pub fn new(name: String, template: String, llm: String, output: String) -> Self {
        Self {
            name,
            template,
            llm,
            output,
        }
    }
}

impl Step for TextGenerationStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let template = templates.render(self.template.clone(), context.clone());
        let mut context = context.clone();
        context.data[self.output.clone()] = serde_json::to_value(template.unwrap()).unwrap();
        println!("{:?}", context);
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
        let file = File::options().append(true).open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);
        let row = templates.render(self.template.clone(), context.clone())?;
        writeln!(writer, "{}", row)?;
        writer.flush()?;

        Ok(context.clone())
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        println!("hello");
    }
}
