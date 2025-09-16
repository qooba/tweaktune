use crate::{
    common::validators::validate_tool_format_messages,
    datasets::DatasetType,
    embeddings, llms,
    steps::{self, Step, StepContext, StepStatus},
    templates::Templates,
};
use anyhow::Result;
use log::error;
use serde_json::{json, Value};
use std::collections::HashMap;

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

    fn parse_step(&self, conv_step: Vec<&str>, context: &StepContext) -> Result<Value> {
        if conv_step.len() != 2 {
            anyhow::bail!(
                "Invalid conversation step format for step: {:?}, expected format: @role:key",
                conv_step
            );
        }

        let role = conv_step[0];
        if role != "@user"
            && role != "@assistant"
            && role != "@system"
            && role != "@tool"
            && role != "@u"
            && role != "@a"
            && role != "@s"
            && role != "@t"
        {
            anyhow::bail!("Invalid role in conversation step: {}, allowed roles are: user, assistant, system, tool", role);
        }

        Ok(if role == "@user" || role == "@u" {
            let value = context
                .get(conv_step[1])
                .ok_or_else(|| anyhow::anyhow!("Key not found in context: {}", conv_step[1]))?;

            json!({
                "role": "user",
                "content": value
            })
        } else if role == "@assistant" || role == "@a" {
            // handle assistant special forms: tool_calls([...]) and think(...)
            if conv_step[1].starts_with("tool_calls(") {
                // extract inside of parentheses
                let mut inner = conv_step[1]
                    .trim_start_matches("tool_calls(")
                    .trim_end_matches(')')
                    .trim();

                // allow optional surrounding brackets [ ... ]
                if inner.starts_with('[') && inner.ends_with(']') {
                    inner = inner.trim_start_matches('[').trim_end_matches(']').trim();
                }

                let keys = inner
                    .split(',')
                    .map(|s| s.trim().trim_matches('"'))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>();

                let mut calls: Vec<Value> = Vec::new();
                for k in keys {
                    let v = context
                        .get(k)
                        .ok_or_else(|| anyhow::anyhow!("Key not found in context: {}", k))?
                        .clone();

                    // Normalize into an object of shape { "function": { "name": ..., "arguments": ... } }
                    let call_obj = match v {
                        Value::String(s) => {
                            // try to parse a JSON string first
                            if let Ok(parsed) = serde_json::from_str::<Value>(&s) {
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
                                v
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
            } else if conv_step[1].starts_with("think(") {
                let inner = conv_step[1]
                    .trim_start_matches("think(")
                    .trim_end_matches(')')
                    .trim()
                    .trim_matches('"');

                let val = context
                    .get(inner)
                    .ok_or_else(|| anyhow::anyhow!("Key not found in context: {}", inner))?
                    .clone();

                return Ok(json!({
                    "role": "assistant",
                    "reasoning_content": val
                }));
            } else {
                let value = context
                    .get(conv_step[1])
                    .ok_or_else(|| anyhow::anyhow!("Key not found in context: {}", conv_step[1]))?
                    .clone();
                return Ok(json!({
                    "role": "assistant",
                    "content": value
                }));
            }
        } else if role == "@tool" || role == "@t" {
            let value = context
                .get(conv_step[1])
                .ok_or_else(|| anyhow::anyhow!("Key not found in context: {}", conv_step[1]))?;

            json!({
                "role": "tool",
                "content": value
            })
        } else if role == "@system" || role == "@s" {
            let value = context
                .get(conv_step[1])
                .ok_or_else(|| anyhow::anyhow!("Key not found in context: {}", conv_step[1]))?;

            json!({
                "role": "system",
                "content": value
            })
        } else {
            anyhow::bail!("Invalid role in conversation step: {}", role);
        })
    }
}

impl Step for RenderConversationStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let mut context = context.clone();

        let conversation = self.conversation.trim();
        let conversation = if self.separator != "\n" {
            conversation.replace("\n", "")
        } else {
            conversation.to_string()
        };

        let conversations_steps = conversation
            .split(self.separator.as_str())
            .map(|s| {
                self.parse_step(
                    s.split(":").map(|s| s.trim()).collect::<Vec<&str>>(),
                    &context,
                )
            })
            .collect::<Result<Vec<Value>, _>>()?;

        let rendered = if let Some(t) = &self.tools {
            let tools = context
                .get(t)
                .ok_or_else(|| anyhow::anyhow!("Key not found in context: {}", t))?;
            json!({ "messages": conversations_steps, "tools": tools, "id": context.id })
        } else {
            json!({ "messages": conversations_steps, "id": context.id })
        };

        if let Err(e) = validate_tool_format_messages(&rendered) {
            error!(target: "conversation_validation_step", "🐔 Conversation validation failed: {}", e);
            context.set_status(StepStatus::Failed);
            return Ok(context);
        }

        context.set(&self.output, rendered.to_string());
        Ok(context)
    }
}
