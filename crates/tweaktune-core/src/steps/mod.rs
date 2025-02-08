use crate::{datasets::DatasetType, embeddings, llms, templates::Templates};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepContext {
    data: serde_json::Value,
}

impl StepContext {
    pub fn new() -> Self {
        Self { data: json!({}) }
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
    async fn process(
        &self,
        datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        llms: &HashMap<String, llms::LLMType>,
        embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext>;
}

pub struct TextGenerationStep {
    pub name: String,
    pub template: String,
    pub llm: String,
}

impl TextGenerationStep {
    pub fn new(name: String, template: String, llm: String) -> Self {
        Self {
            name,
            template,
            llm,
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
        println!("{:?}", context);
        let template = templates.render(self.template.clone(), context.clone());
        println!("{:?}", template);
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
