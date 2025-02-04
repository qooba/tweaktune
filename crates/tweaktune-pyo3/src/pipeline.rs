use std::collections::HashMap;

use tweaktune_core::{datasets::Dataset, llms::LLM, templates::Template};

pub struct Pipeline {
    pub workers: usize,
    pub datasets: Resources<Box<dyn Dataset>>,
    pub templates: Resources<Box<dyn Template>>,
    pub llms: Resources<Box<dyn LLM>>,
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
