use std::collections::HashMap;

use crate::{
    datasets::DatasetType, embeddings::EmbeddingsType, llms::LLMType, templates::Templates,
    tokenizers::TokenizerWrapper,
};

pub mod common;
pub mod config;
pub mod datasets;
pub mod embeddings;
pub mod llms;
pub mod readers;
pub mod state;
pub mod steps;
pub mod templates;
pub mod tokenizers;

pub struct PipelineResources {
    pub datasets: Resources<DatasetType>,
    pub embeddings: Resources<EmbeddingsType>,
    pub llms: Resources<LLMType>,
    pub templates: Templates,
    pub tokenizers: Resources<TokenizerWrapper>,
}

impl Default for PipelineResources {
    fn default() -> Self {
        Self {
            datasets: Resources {
                resources: HashMap::new(),
            },
            embeddings: Resources {
                resources: HashMap::new(),
            },
            llms: Resources {
                resources: HashMap::new(),
            },
            templates: Templates::default(),
            tokenizers: Resources {
                resources: HashMap::new(),
            },
        }
    }
}

#[derive(Default, Clone)]
pub struct Resources<T> {
    pub resources: HashMap<String, T>,
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
