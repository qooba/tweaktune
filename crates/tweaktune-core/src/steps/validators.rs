use crate::common::validators::validate_function_call_format;
use crate::{
    datasets::DatasetType,
    embeddings::{self},
    llms::{self},
    steps::{Step, StepContext, StepStatus},
    templates::Templates,
};
use anyhow::Result;
use log::error;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct ValidateJsonStep {
    pub name: String,
    pub schema: String,
    pub instance: String,
}

impl ValidateJsonStep {
    pub fn new(name: String, schema: String, instance: String) -> Self {
        Self {
            name,
            schema,
            instance,
        }
    }
}

impl Step for ValidateJsonStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let schema = templates.render(self.schema.clone(), context.data.clone())?;
        let full_schema: Value = serde_json::from_str(&schema).unwrap();

        let properties = if let Value::String(v) = full_schema["properties"].clone() {
            serde_json::from_str(&v).unwrap()
        } else {
            full_schema["properties"].clone()
        };

        let schema_value = json!({
            "type": "object",
            "properties": properties,
            "required": full_schema["required"],
            "additionalProperties": false,
        });

        let instance_json = templates.render(self.instance.clone(), context.data.clone())?;

        match serde_json::from_str(&instance_json) {
            Ok(instance) => {
                let is_valid = jsonschema::is_valid(&schema_value, &instance);

                if !is_valid {
                    error!(target: "validate_json_step", "🐔 Failed to validate JSON: {} with schema {}", instance, schema_value);
                    context.set_status(StepStatus::Failed);
                }

                Ok(context)
            }
            Err(e) => {
                error!(target: "validate_json_step", "🐔 Failed to render instance: {}", e);
                error!(target: "validate_json_step", "🐔 INSTANCE_JSON: {}", &instance_json);
                context.set_status(StepStatus::Failed);
                Ok(context)
            }
        }
    }
}

pub struct ToolsValidateStep {
    pub name: String,
    pub instances: String,
}

impl ToolsValidateStep {
    pub fn new(name: String, instances: String) -> Self {
        Self { name, instances }
    }
}

impl Step for ToolsValidateStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let instance_json = templates.render(self.instances.clone(), context.data.clone())?;

        match serde_json::from_str::<Value>(&instance_json) {
            Ok(value) => {
                let instances: Vec<Value> = match value {
                    Value::Array(arr) => arr,
                    other => vec![other],
                };

                // Validate each instance using the centralized validator. Fail on any error.
                for inst in &instances {
                    if let Err(e) = validate_function_call_format(inst) {
                        error!(target: "tools_validation_step", "🐔 Tool instance failed validation: {} - error: {}", inst, e);
                        context.set_status(StepStatus::Failed);
                        return Ok(context);
                    }
                }

                // All instances validated successfully; store them in context.
                context.set(&self.name, instances);

                Ok(context)
            }
            Err(e) => {
                error!(target: "tools_validation_step", "🐔 Failed to render instance: {}", e);
                context.set_status(StepStatus::Failed);
                Ok(context)
            }
        }
    }
}
