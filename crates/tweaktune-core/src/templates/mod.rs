use minijinja::Environment;

pub struct Templates<'a> {
    environment: Environment<'a>,
}

impl<'a> Templates<'a> {
    pub fn new() -> Self {
        let environment = Environment::new();
        Self { environment }
    }

    pub fn add(&mut self, name: &'a str, template: &'a str) {
        self.environment.add_template(name, template).unwrap();
    }

    pub fn list(&self) -> Vec<String> {
        self.environment
            .templates()
            .map(|t| t.0.to_string())
            .collect()
    }

    pub fn remove(&mut self, name: &str) {
        self.environment.remove_template(name)
    }

    pub fn render(&self, name: &'a str, input: serde_json::Value) -> String {
        let tmpl = self.environment.get_template(name).unwrap();
        tmpl.render(input).unwrap()
    }
}

impl Default for Templates<'_> {
    fn default() -> Self {
        Self::new()
    }
}
