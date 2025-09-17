use crate::{
    common::dedup::hash_value,
    datasets::DatasetType,
    embeddings::{self},
    llms::{self},
    state::State,
    steps::{Step, StepContext, StepStatus},
    templates::Templates,
};
use anyhow::Result;
use lingua::{LanguageDetector, LanguageDetectorBuilder};
use log::error;
use std::collections::HashMap;

pub struct CheckLanguageStep {
    pub name: String,
    pub input: String,
    pub language: String,
    pub precision: f64,
    pub detector: LanguageDetector,
}

impl CheckLanguageStep {
    pub fn new(
        name: String,
        input: String,
        language: String,
        precision: f64,
        detect_languages: Vec<String>,
    ) -> Self {
        let languages = detect_languages
            .iter()
            .filter_map(|lang| lang.parse().ok())
            .collect::<Vec<_>>();
        let detector = LanguageDetectorBuilder::from_languages(&languages).build();
        Self {
            name,
            input,
            language,
            precision,
            detector,
        }
    }
}

impl Step for CheckLanguageStep {
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

        match context.data.get(&self.input) {
            Some(value) => {
                if let Some(text) = value.as_str() {
                    let detected = self
                        .detector
                        .compute_language_confidence(text, self.language.parse()?);
                    if detected < self.precision {
                        error!(target: "steps_quality", "🐔 Language detection failed: {} < {}", detected, self.precision);
                        context.set_status(StepStatus::Failed);
                    }
                } else {
                    error!(target: "steps_quality", "🐔 Language detection input is not a string");
                    context.set_status(StepStatus::Failed);
                }
            }
            None => {
                error!(target: "steps_quality", "🐔 Language detection input not found");
                context.set_status(StepStatus::Failed);
            }
        }

        Ok(context)
    }
}

pub struct CheckHashStep {
    pub name: String,
    pub input: String,
}

impl CheckHashStep {
    pub fn new(name: String, input: String) -> Self {
        Self { name, input }
    }
}

impl Step for CheckHashStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        state: Option<State>,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        match context.data.get(&self.input) {
            Some(value) => {
                let hash = hash_value(value)?;
                if let Some(state) = state.as_ref() {
                    if let Err(e) = state
                        .add_hash(&context.id.to_string(), &self.input, &hash.clone())
                        .await
                    {
                        error!(target: "steps_quality", "🐔 Hash validation failed to add hash: {}", e);
                        context.set_status(StepStatus::Failed);
                    }
                }
            }
            None => {
                error!(target: "steps_quality", "🐔 Hash validation input not found");
                context.set_status(StepStatus::Failed);
            }
        }

        Ok(context)
    }
}
