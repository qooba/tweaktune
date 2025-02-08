use std::{collections::HashMap, hash::Hash, sync::Arc};

use minijinja::Environment;
use std::sync::OnceLock;

use crate::steps::StepContext;

static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();

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

    pub fn render(&self, name: String, items: StepContext) -> String {
        let environment = ENVIRONMENT.get_or_init(|| {
            let mut e = Environment::new();
            self.templates.clone().into_iter().for_each(|(k, v)| {
                e.add_template_owned(k, v).unwrap();
            });
            e
        });

        let tmpl = environment.get_template(&name).unwrap();
        tmpl.render(items).unwrap()
    }
}
