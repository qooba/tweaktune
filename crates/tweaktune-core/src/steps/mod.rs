use std::collections::HashMap;

use anyhow::Result;
use arrow::array::RecordBatch;

use crate::{datasets::DatasetType, embeddings, llms, templates::Templates};

pub trait Step {
    fn process(
        &self,
        datasets: &HashMap<String, DatasetType>,
        templates: Templates,
        llms: HashMap<String, llms::LLMType>,
        embeddings: HashMap<String, embeddings::EmbeddingsType>,
        batch: RecordBatch,
    ) -> Result<RecordBatch>;
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
    fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: Templates,
        _llms: HashMap<String, llms::LLMType>,
        _embeddings: HashMap<String, embeddings::EmbeddingsType>,
        batch: RecordBatch,
    ) -> Result<RecordBatch> {
        let items: Vec<serde_json::Value> = serde_arrow::from_record_batch(&batch).unwrap();
        let template = templates.render(self.template.clone(), items);
        println!("{:?}", template);
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        println!("hello");
    }
}
