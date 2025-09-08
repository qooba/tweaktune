pub mod embed;
use crate::common::{ktmur, OptionToResult, ResultExt};
use crate::readers::build_reader;
use crate::steps::StepContextData;
use anyhow::{bail, Result};
use log::{debug, error};
use minijinja::Environment;
use rand::rng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::io::BufRead;
use std::io::Cursor;
use std::sync::{OnceLock, RwLock};

static ENVIRONMENT: RwLock<OnceLock<Environment>> = RwLock::new(OnceLock::new());

static CHATTEMPLATE_ENVIRONMENT: RwLock<OnceLock<Environment>> = RwLock::new(OnceLock::new());

#[derive(Default, Clone, Deserialize)]
pub struct Templates {
    pub templates: HashMap<String, String>,
}

impl Templates {
    pub fn add(&mut self, name: String, template: String) {
        self.templates.insert(name, template);
    }

    pub fn add_inline(&mut self, step_type: &str, name: &str, template: &str) -> String {
        let kv = ktmur(step_type, name, template);
        self.templates.insert(kv.0.clone(), kv.1);
        kv.0
    }

    pub fn list(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }

    pub fn remove(&mut self, name: &str) {
        self.templates.remove(name);
    }

    pub fn compile(&self) -> Result<()> {
        let mut e = Environment::new();
        e.add_filter("jstr", |value: String| {
            let val = serde_json::to_string(&value);
            match val {
                Ok(v) => v,
                Err(_) => {
                    error!(target: "templates_err", "🐔 Failed to convert to JSON string");
                    value
                }
            }
        });

        e.add_filter("tool_call", |value: String| {
            let val = serde_json::to_string(&value);
            match val {
                Ok(v) => format!(
                    "\"<tool_call>{}</tool_call>\"",
                    v.strip_prefix('"')
                        .unwrap_or(&v)
                        .strip_suffix('"')
                        .unwrap_or(&v)
                ),
                Err(_) => {
                    error!(target: "templates_err", "🐔 Failed to convert to JSON string");
                    value
                }
            }
        });

        e.add_filter("tool_call_args", |value: String| {
            let val = serde_json::to_string(&value);
            match val {
                Ok(v) => v
                    .strip_prefix('"')
                    .unwrap_or(&v)
                    .strip_suffix('"')
                    .unwrap_or(&v)
                    .to_string(),
                Err(_) => {
                    error!(target: "templates_err", "🐔 Failed to convert to JSON string");
                    value
                }
            }
        });

        e.add_filter("shuffle", |value: String| {
            match serde_json::from_str::<Vec<serde_json::Value>>(&value) {
                Ok(arr) => {
                    let mut arr = arr;
                    arr.shuffle(&mut rng());
                    let val = serde_json::to_string(&arr);
                    match val {
                        Ok(v) => v,
                        Err(_) => {
                            error!(target: "templates_err", "🐔 Failed to convert shuffled array to JSON string");
                            value
                        }
                    }
                }
                Err(_) => {
                    error!(target: "templates_err", "🐔 Failed to shuffle array");
                    value
                }
            }
        });

        e.add_filter("hash", |value: String| {
            let mut cursor = Cursor::new(value.clone());
            let hash = murmur3::murmur3_32(&mut cursor, 0);
            match hash {
                Ok(hash) => format!("{:x}", hash),
                Err(_) => {
                    error!(target: "templates_err", "🐔 Failed to hash value");
                    value
                }
            }
        });

        e.add_filter("deserialize", |value: String| {
            let val: serde_json::error::Result<Value> = serde_json::from_str(&value);
            match val {
                Ok(v) => serde_json::to_string(&v).unwrap(),
                Err(_) => {
                    error!(target: "templates_err", "🐔 Failed to deserialize JSON");
                    value
                }
            }
        });

        e.add_filter("dict2items", |value: String| {
            let items: HashMap<String, Value> = serde_json::from_str(&value).unwrap();
            let items: Vec<(String, Value)> = items.into_iter().collect();
            serde_json::to_string(&items).unwrap()
        });

        for (k, v) in self.templates.clone() {
            e.add_template_owned(k, v).map_anyhow_err()?;
        }
        let mut lock = ENVIRONMENT.write().unwrap();
        *lock = OnceLock::new();
        lock.set(e).map_anyhow_err()?;
        Ok(())
    }

    pub fn render(&self, name: String, items: StepContextData) -> Result<String> {
        let environment = ENVIRONMENT
            .read()
            .map_anyhow_err()?
            .get()
            .cloned()
            .ok_or_err("ENVIRONMENT")?;
        let tmpl = match environment.get_template(&name) {
            Ok(t) => {
                debug!(target:"templates", "🤗 Template found: {}", name);
                t
            }
            Err(e) => {
                error!(target:"templates_err", "🐔 Template not found: {}", name);
                bail!("Template not found: {}", e);
            }
        };
        let rendered_template = match tmpl.render(items) {
            Ok(t) => t,
            Err(e) => {
                error!(target:"templates_err", "🐔 Failed to render template: {}", e);
                bail!("Failed to render template: {}", e);
            }
        };
        debug!(target:"templates", "-------------------\nRENDERED TEMPLATE 📝:\n-------------------\n{}\n-------------------\n", rendered_template);
        Ok(rendered_template)
    }
}

pub type ChatTemplateContext = serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
pub struct ChatTemplate {
    context: ChatTemplateContext,
}

impl ChatTemplate {
    pub fn new(template: String) -> Self {
        let mut env = CHATTEMPLATE_ENVIRONMENT.write().unwrap();
        if let Some(e) = env.get_mut() {
            e.add_template_owned("chat_template".to_string(), template)
                .map_anyhow_err()
                .unwrap();
        } else {
            let mut e = Environment::new();
            e.add_template_owned("chat_template".to_string(), template)
                .map_anyhow_err()
                .unwrap();
            env.set(e).map_anyhow_err().unwrap();
        }

        ChatTemplate {
            context: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    pub fn with_tools(mut self, tools: String) -> Self {
        let tools = serde_json::from_str(&tools).unwrap();
        self.add_data("tools", tools);
        self
    }

    fn add_data(&mut self, key: &str, value: serde_json::Value) {
        if let serde_json::Value::Object(ref mut map) = self.context {
            map.insert(key.to_string(), value);
        } else {
            error!(target:"templates_err", "🐔 Context is not an object");
        }
    }

    pub fn render(&self, messages: String) -> Result<String> {
        let mut messages = serde_json::from_str(&messages).unwrap();
        let messages = if let serde_json::Value::Object(ref mut map) = messages {
            map["messages"].clone()
        } else {
            messages
        };

        let env = CHATTEMPLATE_ENVIRONMENT
            .read()
            .map_anyhow_err()?
            .get()
            .cloned()
            .ok_or_err("CHATTEMPLATE_ENVIRONMENT")?;
        let tmpl = match env.get_template("chat_template") {
            Ok(t) => t,
            Err(e) => {
                error!(target:"templates_err", "🐔 Chat template not found: {}", e);
                bail!("Chat template not found: {}", e);
            }
        };

        let mut context = self.context.clone();
        if let serde_json::Value::Object(ref mut map) = context {
            map.insert("messages".to_string(), messages);
        } else {
            error!(target:"templates_err", "🐔 Context is not an object");
        }

        let rendered_template = match tmpl.render(context) {
            Ok(t) => t,
            Err(e) => {
                error!(target:"templates_err", "🐔 Failed to render chat template: {}", e);
                bail!("Failed to render chat template: {}", e);
            }
        };
        debug!(target:"templates", "-------------------\nRENDERED CHAT TEMPLATE 📝:\n-------------------\n{}\n-------------------\n", rendered_template);
        Ok(rendered_template)
    }

    pub fn render_jsonl(&self, path: &str, op_config: Option<String>) -> Result<Vec<String>> {
        let mut reader = build_reader(path, op_config)?;
        let mut output = vec![];

        let mut buf = String::new();
        while reader.inner.read_line(&mut buf)? != 0 {
            let line = buf.trim_end().to_string();
            buf.clear();
            if line.trim().is_empty() {
                continue;
            }
            let rendered = self.render(line)?;
            output.push(rendered);
        }

        Ok(output)
    }
}
