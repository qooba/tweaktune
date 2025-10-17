use crate::{
    embeddings::{e5::E5Model, Embeddings, EmbeddingsType},
    steps::{Step, StepContext, StepStatus},
    PipelineResources,
};
use anyhow::Result;
use log::{error, info};

pub struct CheckEmbeddingStep {
    pub name: String,
    pub embedding: String,
    pub input: String,
    pub treshold: f32,
    pub similarity_output: Option<String>,
}

impl CheckEmbeddingStep {
    pub fn new(
        name: String,
        embedding: String,
        input: String,
        treshold: f32,
        similarity_output: Option<String>,
    ) -> Self {
        Self {
            name,
            embedding,
            input,
            treshold,
            similarity_output,
        }
    }
}

impl Step for CheckEmbeddingStep {
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
                        if let Some(state) = resources.state.as_ref() {
                            let nearest = state
                                .knn_embeddings(&self.input.clone(), &emb[0], 1)
                                .await?;

                            if !nearest.is_empty() && (nearest[0].1 - 1.0).abs() < self.treshold {
                                info!(target: "steps_embeddings", "✅ Similar embedding found for input");
                                context.set_status(StepStatus::Failed);
                            } else {
                                state
                                    .add_embedding(&context.id.to_string(), &self.input, &emb[0])
                                    .await?;
                                if let Some(output) = &self.similarity_output {
                                    if !nearest.is_empty() {
                                        let similarity = (nearest[0].1 - 1.0).abs();
                                        context.set(output, similarity);
                                    } else {
                                        context.set(output, 0.0);
                                    }
                                }
                            }
                            // if let Err(e) = state
                            //     .add_hash(&context.id.to_string(), &self.input, &hash.clone())
                            //     .await
                            // {
                            //     error!(target: "steps_quality", "🐔 Hash validation failed to add hash: {}", e);
                            //     context.set_status(StepStatus::Failed);
                            // }
                        }
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
