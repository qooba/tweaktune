use crate::{
    embeddings::{e5::E5Model, Embeddings, EmbeddingsType},
    steps::{Step, StepContext, StepStatus},
    templates::embed,
    PipelineResources,
};
use anyhow::Result;
use log::error;

pub struct EmbeddingStep {
    pub name: String,
    pub embedding: String,
    pub input: String,
    pub output: String,
}

impl EmbeddingStep {
    pub fn new(name: String, embedding: String, input: String, output: String) -> Self {
        Self {
            name,
            embedding,
            input,
            output,
        }
    }
}

impl Step for EmbeddingStep {
    async fn process(
        &self,
        resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        match context.data.get(&self.input) {
            Some(value) => {
                let embedding = resources
                    .embeddings
                    .get(&self.embedding)
                    .ok_or_else(|| anyhow::anyhow!("Embedding not found: {}", self.embedding))?;

                match embedding {
                    EmbeddingsType::E5(spec) => {
                        let text = if let Some(text) = value.as_str() {
                            text
                        } else {
                            error!(target: "steps_embeddings", "🐔 Embedding input is not a string");
                            context.set_status(StepStatus::Failed);
                            return Ok(context);
                        };

                        let instance = E5Model::lazy(spec.clone())?;
                        let guard = instance
                            .lock()
                            .map_err(|e| anyhow::anyhow!("lock error: {:?}", e))?;

                        let emb = guard.embed(vec![text.to_string()])?;
                        context.set(&self.output, emb);
                    }
                    _ => {
                        error!(target: "steps_embeddings", "🐔 Unsupported embedding type");
                        context.set_status(StepStatus::Failed);
                    }
                }
            }
            None => {
                // error!(target: "steps_quality", "🐔 Hash validation input not found");
                context.set_status(StepStatus::Failed);
            }
        }

        Ok(context)
    }
}
