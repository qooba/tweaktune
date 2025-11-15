use crate::{
    common::validators::validate_tool_format_messages,
    steps::{Step, StepContext, StepStatus},
    PipelineResources,
};
use anyhow::Result;
use log::error;
use serde_json::{json, Value};
use std::borrow::Cow;

/// Represents a conversation role for efficient matching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversationRole {
    User,
    Assistant,
    System,
    Tool,
}

impl ConversationRole {
    /// Parse role from string with O(1) match instead of multiple comparisons
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "@user" | "@u" => Ok(Self::User),
            "@assistant" | "@a" => Ok(Self::Assistant),
            "@system" | "@s" => Ok(Self::System),
            "@tool" | "@t" => Ok(Self::Tool),
            _ => anyhow::bail!(
                "Invalid role '{}'. Allowed: @user/@u, @assistant/@a, @system/@s, @tool/@t",
                s
            ),
        }
    }

    /// Get the JSON role string
    fn as_json_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Tool => "tool",
        }
    }
}

/// Parse function call syntax like "func_name(content)"
/// Returns (function_name, content_inside_parens)
fn parse_function_call(input: &str) -> Option<(&str, &str)> {
    let open_paren = input.find('(')?;
    let close_paren = input.rfind(')')?;

    if close_paren <= open_paren {
        return None;
    }

    let func_name = input[..open_paren].trim();
    let content = input[open_paren + 1..close_paren].trim();

    Some((func_name, content))
}

pub struct RenderToolCallStep {
    pub name: String,
    pub tool_name: String,
    pub arguments: String,
    pub output: String,
    pub additional_template: Option<String>,
}

impl RenderToolCallStep {
    pub fn new(
        name: String,
        tool_name: String,
        arguments: String,
        output: String,
        additional_template: Option<String>,
    ) -> Self {
        Self {
            name,
            tool_name,
            arguments,
            additional_template,
            output,
        }
    }
}

impl Step for RenderToolCallStep {
    async fn process(
        &self,
        resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let arguments = context.data.get(&self.arguments).unwrap();
        let arguments = if let Value::String(v) = arguments {
            serde_json::from_str(v).unwrap()
        } else {
            arguments.clone()
        };

        let rendered = json!({
            "function": {
                "name": context.data.get(&self.tool_name).unwrap().clone(),
                "arguments": arguments,
            }
        });
        context.set(&self.output, rendered);

        if let Some(tmpl) = self.additional_template.as_ref() {
            let rendered = resources
                .templates
                .render(tmpl.clone(), context.data.clone())?;

            context.set(&self.output, rendered);
        }
        Ok(context)
    }
}

pub struct RenderConversationStep {
    pub name: String,
    pub conversation: String,
    pub tools: Option<String>,
    pub separator: String,
    pub output: String,
}

impl RenderConversationStep {
    pub fn new(
        name: String,
        conversation: String,
        tools: Option<String>,
        separator: Option<String>,
        output: String,
    ) -> Self {
        let separator = separator.unwrap_or_else(|| "|".to_string());
        if separator == "@" {
            error!(target: "conversation_step", "🐔 The separator '@' is not allowed as it conflicts with role prefixes. Using '|' instead.");
        }
        Self {
            name,
            conversation,
            tools,
            separator,
            output,
        }
    }

    /// Parse a single conversation step with zero-allocation string parsing
    fn parse_step(&self, step_str: &str, context: &StepContext) -> Result<Value> {
        // Use splitn to stop after finding role and key (no Vec allocation)
        let mut parts = step_str.splitn(2, ':');
        let role_str = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing role in step: '{}'", step_str))?
            .trim();
        let key = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("Missing key in step: '{}'", step_str))?
            .trim();

        // Parse role once with efficient enum match
        let role = ConversationRole::from_str(role_str)?;

        match role {
            ConversationRole::User => {
                let value = context
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in context", key))?;
                Ok(json!({
                    "role": role.as_json_str(),
                    "content": value
                }))
            }
            ConversationRole::Assistant => self.parse_assistant_step(key, context),
            ConversationRole::Tool => {
                let value = context
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in context", key))?;
                Ok(json!({
                    "role": role.as_json_str(),
                    "content": value
                }))
            }
            ConversationRole::System => {
                let value = context
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in context", key))?;
                Ok(json!({
                    "role": role.as_json_str(),
                    "content": value
                }))
            }
        }
    }

    /// Parse assistant step with special handling for tool_calls() and think()
    fn parse_assistant_step(&self, key: &str, context: &StepContext) -> Result<Value> {
        // Check for special function syntax using optimized parser
        if let Some((func_name, content)) = parse_function_call(key) {
            match func_name {
                "tool_calls" => {
                    // Remove optional surrounding brackets
                    let content = content.trim_matches(&['[', ']', ' '][..]);

                    // Parse comma-separated keys
                    let keys: Vec<_> = content
                        .split(',')
                        .map(|s| s.trim().trim_matches('"'))
                        .filter(|s| !s.is_empty())
                        .collect();

                    let mut calls = Vec::with_capacity(keys.len());
                    for k in keys {
                        let v = context
                            .get(k)
                            .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in context", k))?;

                        // Normalize into { "function": { "name": ..., "arguments": ... } }
                        let call_obj = match v {
                            Value::String(s) => {
                                // Try to parse JSON string
                                if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                                    if parsed.get("function").is_some() {
                                        parsed
                                    } else if parsed.get("name").is_some() {
                                        json!({ "function": parsed })
                                    } else {
                                        json!({ "function": { "name": parsed } })
                                    }
                                } else {
                                    json!({ "function": { "name": s } })
                                }
                            }
                            Value::Object(_) => {
                                if v.get("function").is_some() {
                                    v.clone()
                                } else if v.get("name").is_some() {
                                    json!({ "function": v })
                                } else {
                                    json!({ "function": { "name": v } })
                                }
                            }
                            other => json!({ "function": { "name": other } }),
                        };

                        calls.push(call_obj);
                    }

                    return Ok(json!({
                        "role": "assistant",
                        "tool_calls": calls
                    }));
                }
                "think" => {
                    let inner = content.trim_matches('"');
                    let val = context
                        .get(inner)
                        .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in context", inner))?;

                    return Ok(json!({
                        "role": "assistant",
                        "reasoning_content": val
                    }));
                }
                _ => {
                    // Unknown function, treat as regular key
                }
            }
        }

        // Regular assistant message
        let value = context
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in context", key))?;
        Ok(json!({
            "role": "assistant",
            "content": value
        }))
    }
}

impl Step for RenderConversationStep {
    async fn process(
        &self,
        _resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        // Use Cow to avoid allocation when separator is "\n"
        let conversation_trimmed = self.conversation.trim();
        let conversation: Cow<str> = if self.separator != "\n" {
            Cow::Owned(conversation_trimmed.replace('\n', ""))
        } else {
            Cow::Borrowed(conversation_trimmed)
        };

        // Parse steps with zero-allocation string splitting
        let conversations_steps = conversation
            .split(self.separator.as_str())
            .map(|s| self.parse_step(s.trim(), &context))
            .collect::<Result<Vec<Value>, _>>()?;

        let rendered = if let Some(t) = &self.tools {
            let tools = context
                .get(t)
                .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in context", t))?;

            // Normalize tool properties (parse JSON strings) - optimized to clone only once per tool
            let tools = match tools {
                Value::Array(ref arr) => {
                    let mut normalized_tools = Vec::with_capacity(arr.len());

                    for tool in arr {
                        let mut normalized = tool.clone(); // Clone once per tool

                        // Parse stringified properties if present
                        if let Value::Object(ref mut obj) = normalized {
                            if let Some(Value::Object(ref mut params)) = obj.get_mut("parameters") {
                                if let Some(Value::String(ref props_str)) = params.get("properties")
                                {
                                    match serde_json::from_str::<Value>(props_str) {
                                        Ok(props_json) => {
                                            params.insert("properties".to_string(), props_json);
                                        }
                                        Err(_) => {
                                            error!(
                                                target: "conversation_step",
                                                "🐔 Failed to parse tool properties JSON: {}",
                                                props_str
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        normalized_tools.push(normalized);
                    }

                    Value::Array(normalized_tools)
                }
                _ => {
                    error!(target: "conversation_validation_step", "🐔 Invalid tool format, expected array");
                    context.set_status(StepStatus::Failed);
                    return Ok(context);
                }
            };

            json!({ "messages": conversations_steps, "tools": tools, "id": context.id })
        } else {
            json!({ "messages": conversations_steps, "id": context.id })
        };

        if let Err(e) = validate_tool_format_messages(&rendered) {
            error!(target: "conversation_validation_step", "🐔 Conversation validation failed: {}", e);
            context.set_status(StepStatus::Failed);
            return Ok(context);
        }

        context.set(&self.output, rendered);
        Ok(context)
    }
}

pub struct RenderDPOStep {
    pub name: String,
    pub messages: String,
    pub chosen: String,
    pub rejected: String,
    pub tools: Option<String>,
    pub output: String,
}

impl RenderDPOStep {
    pub fn new(
        name: String,
        messages: String,
        chosen: String,
        rejected: String,
        tools: Option<String>,
        output: String,
    ) -> Self {
        Self {
            name,
            messages,
            chosen,
            rejected,
            tools,
            output,
        }
    }
}

impl Step for RenderDPOStep {
    async fn process(
        &self,
        resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let messages = resources
            .templates
            .render(self.messages.clone(), context.data.clone())?;
        let chosen = resources
            .templates
            .render(self.chosen.clone(), context.data.clone())?;
        let rejected = resources
            .templates
            .render(self.rejected.clone(), context.data.clone())?;
        let dpo = if let Some(tools_template) = &self.tools {
            let tools = Some(
                resources
                    .templates
                    .render(tools_template.clone(), context.data.clone())?,
            );

            json!({
                "messages": messages,
                "chosen": chosen,
                "rejected": rejected,
                "tools": tools,
            })
        } else {
            json!({
                "messages": messages,
                "chosen": chosen,
                "rejected": rejected,
            })
        };

        context.set(&self.output, dpo);
        Ok(context)
    }
}

pub struct RenderGRPOStep {
    pub name: String,
    pub messages: String,
    pub solution: String,
    pub validator_id: String,
    pub tools: Option<String>,
    pub output: String,
}

impl RenderGRPOStep {
    pub fn new(
        name: String,
        messages: String,
        solution: String,
        validator_id: String,
        tools: Option<String>,
        output: String,
    ) -> Self {
        Self {
            name,
            messages,
            solution,
            validator_id,
            tools,
            output,
        }
    }
}

impl Step for RenderGRPOStep {
    async fn process(
        &self,
        resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let messages = resources
            .templates
            .render(self.messages.clone(), context.data.clone())?;
        let solution = resources
            .templates
            .render(self.solution.clone(), context.data.clone())?;

        let grpo = if let Some(tools_template) = &self.tools {
            let tools = Some(
                resources
                    .templates
                    .render(tools_template.clone(), context.data.clone())?,
            );

            json!({
                "messages": messages,
                "solution": solution,
                "validator_id": self.validator_id,
                "tools": tools,
            })
        } else {
            json!({
                "messages": messages,
                "solution": solution,
                "validator_id": self.validator_id,
            })
        };

        context.set(&self.output, grpo);
        Ok(context)
    }
}
