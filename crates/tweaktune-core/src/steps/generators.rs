use crate::{
    common::extract_json,
    datasets::DatasetType,
    embeddings::{self},
    llms::{self, LLM},
    steps::{Step, StepContext, StepStatus},
    templates::Templates,
};
use anyhow::Result;
use log::{debug, error};
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct TextGenerationStep {
    pub name: String,
    pub template: String,
    pub system_template: Option<String>,
    pub llm: String,
    pub output: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl TextGenerationStep {
    pub fn new(
        name: String,
        template: String,
        llm: String,
        output: String,
        system_template: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Self {
        Self {
            name,
            template,
            llm,
            output,
            system_template,
            max_tokens,
            temperature,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn generate(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Option<String>> {
        let template = templates.render(self.template.clone(), context.data.clone());
        let template = match template {
            Ok(t) => t,
            Err(e) => {
                error!(target: "text_generation_step", "🐔 Failed to render template: {}", e);
                return Ok(None);
            }
        };

        let llm = llms.get(&self.llm).expect("LLM");
        let result = match llm {
            llms::LLMType::Api(llm) => match llm
                .call(template, json_schema, max_tokens, temperature)
                .await
            {
                Ok(response) => Some(response.choices[0].message.content.clone()),
                Err(e) => {
                    error!(target: "text_generation_step", "🐔 Failed to generate text: {}", e);
                    None
                }
            },
            llms::LLMType::Unsloth(llm) => match llm
                .call(template, json_schema, max_tokens, temperature)
                .await
            {
                Ok(response) => Some(response.choices[0].message.content.clone()),
                Err(e) => {
                    error!(target: "text_generation_step", "🐔 Failed to generate text: {}", e);
                    None
                }
            },
            llms::LLMType::Mistralrs(llm) => match llm
                .call(template, json_schema, max_tokens, temperature)
                .await
            {
                Ok(response) => Some(response.choices[0].message.content.clone()),
                Err(e) => {
                    error!(target: "text_generation_step", "🐔 Failed to generate text: {}", e);
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
            .generate(
                _datasets,
                templates,
                llms,
                _embeddings,
                &context,
                None,
                self.max_tokens,
                self.temperature,
            )
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
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub schema_key: Option<String>,
}

#[allow(clippy::too_many_arguments)]
impl JsonGenerationStep {
    pub fn new(
        name: String,
        template: String,
        llm: String,
        output: String,
        json_path: Option<String>,
        system_template: Option<String>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        schema_key: Option<String>,
    ) -> Self {
        Self {
            generation_step: TextGenerationStep::new(
                name.clone(),
                template,
                llm,
                output.clone(),
                system_template,
                max_tokens,
                temperature,
            ),
            output,
            name,
            json_path,
            json_schema,
            max_tokens,
            temperature,
            schema_key,
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

        let json_schema = if let Some(schema_key) = &self.schema_key {
            let schema = templates.render(schema_key.clone(), context.data.clone())?;

            let full_schema: Value = serde_json::from_str(&schema).unwrap();

            let properties = if let Value::String(v) = full_schema["properties"].clone() {
                serde_json::from_str(&v).unwrap()
            } else {
                full_schema["properties"].clone()
            };

            let schema = json!({
                "name": "OUTPUT",
                "schema":{
                    "type": "object",
                    "properties": properties,
                    "required": full_schema["required"],
                    "additionalProperties": false,
                },
                "strict": true
            })
            .to_string();

            debug!(target: "json_generation_step", "🤗 RENDERED SCHEMA: {}", schema);
            Some(schema)
        } else if let Some(schema) = &self.json_schema {
            debug!(target: "json_generation_step", "🤗 PROVIDED SCHEMA: {}", schema);
            Some(schema.clone())
        } else {
            None
        };

        let result = self
            .generation_step
            .generate(
                _datasets,
                templates,
                llms,
                _embeddings,
                &context,
                json_schema,
                self.max_tokens,
                self.temperature,
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

                    debug!(target:"json_generation_step", "🤗 Generated VALUE: {}", value);
                    context.data[self.output.clone()] = value;
                }
                Err(e) => {
                    error!(target:"json_generation_step", "🐔 Failed to extract JSON: {}", e);
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
