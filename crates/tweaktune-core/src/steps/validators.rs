use crate::common::validators::{
    normalize_tool, validate_function_call_conversation, validate_function_call_format,
    validate_tool_format_messages,
};
use crate::steps::{Step, StepContext, StepStatus};
use crate::PipelineResources;
use anyhow::Result;
use log::error;
use serde_json::{json, Value};

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
        resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let schema = resources
            .templates
            .render(self.schema.clone(), context.data.clone())?;
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

        let instance_json = resources
            .templates
            .render(self.instance.clone(), context.data.clone())?;

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
        _resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let value = context
            .data
            .get(self.instances.clone())
            .expect("Failed to get instances");

        let instances: Vec<Value> = match value {
            Value::Array(arr) => arr.clone(),
            other => vec![other.clone()],
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
}

pub struct ToolsNormalizeStep {
    pub name: String,
    pub instances: String,
    pub output: String,
}

impl ToolsNormalizeStep {
    pub fn new(name: String, instances: String, output: String) -> Self {
        Self {
            name,
            instances,
            output,
        }
    }
}

impl Step for ToolsNormalizeStep {
    async fn process(
        &self,
        _resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let value = context
            .data
            .get(self.instances.clone())
            .expect("Failed to get instances");
        let instances = match value {
            Value::Array(arr) => arr.clone(),
            other => vec![other.clone()],
        };

        // Validate each instance using the centralized validator. Fail on any error.
        let mut normalized: Vec<Value> = Vec::new();
        for inst in &instances {
            match normalize_tool(inst) {
                Ok(norm) => normalized.push(norm),
                Err(e) => {
                    error!(target: "tools_normalize_step", "🐔 Tool instance failed normalization: {} - error: {}", inst, e);
                    context.set_status(StepStatus::Failed);
                    return Ok(context);
                }
            }
        }
        let instances = normalized;

        context.set(&self.output, instances);

        Ok(context)
    }
}

pub struct ConversationValidateStep {
    pub name: String,
    pub conversation: String,
}

impl ConversationValidateStep {
    pub fn new(name: String, conversation: String) -> Self {
        Self { name, conversation }
    }
}

impl Step for ConversationValidateStep {
    async fn process(
        &self,
        _resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let value = context
            .data
            .get(self.conversation.clone())
            .expect("Failed to get conversation");

        if let Some(_conv) = value.get("conversation") {
            if let Err(e) = validate_function_call_conversation(value) {
                error!(target: "conversation_validation_step", "🐔 Conversation validation failed: {}", e);
                context.set_status(StepStatus::Failed);
                return Ok(context);
            }
        } else if let Some(_messages) = value.get("messages") {
            if let Err(e) = validate_tool_format_messages(value) {
                error!(target: "conversation_validation_step", "🐔 Conversation validation failed: {}", e);
                context.set_status(StepStatus::Failed);
                return Ok(context);
            }
        } else {
            error!(target: "conversation_validation_step", "🐔 Conversation does not contain 'conversation' or 'messages' field");
            context.set_status(StepStatus::Failed);
            return Ok(context);
        }

        Ok(context)
    }
}
