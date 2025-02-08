use crate::common::{OptionToResult, ResultExt};
use crate::steps::StepContext;
use anyhow::Result;
use minijinja::Environment;
use std::collections::HashMap;
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
        for (k, v) in self.templates.clone() {
            e.add_template_owned(k, v).map_anyhow_err()?;
        }
        let mut lock = ENVIRONMENT.write().unwrap();
        *lock = OnceLock::new();
        lock.set(e).map_anyhow_err()?;
        Ok(())
    }

    pub fn render(&self, name: String, items: StepContext) -> Result<String> {
        let environment = ENVIRONMENT
            .read()
            .map_anyhow_err()?
            .get()
            .cloned()
            .ok_or_err("ENVIRONMENT")?;
        let tmpl = environment.get_template(&name).map_anyhow_err()?;
        tmpl.render(items).map_anyhow_err()
    }
}
