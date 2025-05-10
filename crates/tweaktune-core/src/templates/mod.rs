use crate::common::{OptionToResult, ResultExt};
use crate::steps::StepContextData;
use anyhow::{bail, Result};
use log::debug;
use minijinja::Environment;
use rand::rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{OnceLock, RwLock};

static ENVIRONMENT: RwLock<OnceLock<Environment>> = RwLock::new(OnceLock::new());

#[derive(Default, Clone)]
pub struct Templates {
    templates: HashMap<String, String>,
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
            serde_json::to_string(&value).unwrap()
        });

        e.add_filter("shuffle", |value: String| {
            match serde_json::from_str::<Vec<serde_json::Value>>(&value) {
                Ok(arr) => {
                    let mut arr = arr;
                    arr.shuffle(&mut rng());
                    serde_json::to_string(&arr).unwrap()
                }
                Err(_) => {
                    log::debug!("Failed to shuffle array");
                    value
                }
            }
        });

        e.add_filter("hash", |value: String| {
            let mut cursor = Cursor::new(value);
            let hash = murmur3::murmur3_32(&mut cursor, 0).unwrap();
            format!("{:x}", hash)
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
                debug!(target:"template", "Template found: {}", name);
                t
            }
            Err(e) => {
                debug!(target:"template", "Template not found: {}", name);
                bail!("Template not found: {}", e);
            }
        };
        let rendered_template = match tmpl.render(items) {
            Ok(t) => t,
            Err(e) => {
                debug!(target:"template", "Failed to render template: {}", e);
                bail!("Failed to render template: {}", e);
            }
        };
        debug!(target:"template", "-------------------\nRENDERED TEMPLATE 📝:\n-------------------\n{}\n-------------------\n", rendered_template);
        Ok(rendered_template)
    }
}
