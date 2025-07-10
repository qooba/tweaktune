use crate::common::{OptionToResult, ResultExt};
use crate::steps::StepContextData;
use anyhow::{bail, Result};
use log::{debug, error, info};
use minijinja::Environment;
use rand::rng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{OnceLock, RwLock};

static ENVIRONMENT: RwLock<OnceLock<Environment>> = RwLock::new(OnceLock::new());

#[derive(Default, Clone, Deserialize)]
pub struct Templates {
    pub templates: HashMap<String, String>,
}

impl Templates {
    pub fn add(&mut self, name: String, template: String) {
        self.templates.insert(name, template);
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
                info!(target:"templates", "🤗 Template found: {}", name);
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
        info!(target:"templates", "-------------------\nRENDERED TEMPLATE 📝:\n-------------------\n{}\n-------------------\n", rendered_template);
        Ok(rendered_template)
    }
}
