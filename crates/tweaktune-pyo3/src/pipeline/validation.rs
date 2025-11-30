use crate::pipeline::{IterBy, PipelineBuilder};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Color, ContentArrangement, Table};
use tweaktune_core::steps::StepType;

#[derive(Debug)]
pub struct ValidationError {
    pub category: String,
    pub item: String,
    pub error: String,
}

#[derive(Default)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_error(&mut self, category: String, item: String, error: String) {
        self.errors.push(ValidationError {
            category,
            item,
            error,
        });
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn generate_report(&self) -> String {
        if self.errors.is_empty() {
            return "✓ Pre-flight validation passed\n".to_string();
        }

        let mut output = String::new();
        output.push('\n');
        output
            .push_str("╭─────────────────────────────────────────────────────────────────────╮\n");
        output
            .push_str("│  ❌ PRE-FLIGHT VALIDATION FAILED                                    │\n");
        output
            .push_str("╰─────────────────────────────────────────────────────────────────────╯\n");
        output.push('\n');

        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        table.set_header(vec![
            Cell::from("Category").fg(Color::Cyan),
            Cell::from("Item").fg(Color::Cyan),
            Cell::from("Error").fg(Color::Cyan),
        ]);

        for error in &self.errors {
            table.add_row(vec![
                Cell::from(&error.category).fg(Color::Yellow),
                Cell::from(&error.item),
                Cell::from(&error.error).fg(Color::Red),
            ]);
        }

        output.push_str(&table.to_string());
        output.push('\n');
        output.push_str(&format!(
            "Found {} validation error(s)\n",
            self.errors.len()
        ));
        output.push_str("Please fix these issues before running the pipeline.\n");
        output.push('\n');

        output
    }
}

impl PipelineBuilder {
    /// Perform pre-flight validation before executing the pipeline
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Validate iteration configuration
        self.validate_iteration(&mut result);

        // Validate all steps
        self.validate_steps(&mut result);

        // Validate templates can compile
        self.validate_templates(&mut result);

        result
    }

    fn validate_iteration(&self, result: &mut ValidationResult) {
        match &self.iter_by {
            IterBy::Range { start, stop, step } => {
                if start >= stop {
                    result.add_error(
                        "Iteration".to_string(),
                        "Range".to_string(),
                        format!("Invalid range: start ({}) >= stop ({})", start, stop),
                    );
                }
                if *step == 0 {
                    result.add_error(
                        "Iteration".to_string(),
                        "Range".to_string(),
                        "Step cannot be zero".to_string(),
                    );
                }
            }
            IterBy::Dataset { name } => {
                if !self.resources.datasets.resources.contains_key(name) {
                    result.add_error(
                        "Iteration".to_string(),
                        "Dataset".to_string(),
                        format!("Dataset '{}' not found", name),
                    );
                }
            }
        }
    }

    fn validate_steps(&self, result: &mut ValidationResult) {
        for (idx, step) in self.steps.iter().enumerate() {
            let step_name = format!("Step #{}", idx + 1);
            self.validate_step(step, &step_name, result);
        }
    }

    fn validate_step(&self, step: &StepType, step_name: &str, result: &mut ValidationResult) {
        match step {
            StepType::TextGeneration(s) => {
                // Check template exists
                if !self.resources.templates.templates.contains_key(&s.template) {
                    result.add_error(
                        "Template".to_string(),
                        step_name.to_string(),
                        format!("Template '{}' not found", s.template),
                    );
                }
                // Check LLM exists
                if !self.resources.llms.resources.contains_key(&s.llm) {
                    result.add_error(
                        "LLM".to_string(),
                        step_name.to_string(),
                        format!("LLM '{}' not found", s.llm),
                    );
                }
                // Check system template if specified
                if let Some(sys_tmpl) = &s.system_template {
                    if !self.resources.templates.templates.contains_key(sys_tmpl) {
                        result.add_error(
                            "Template".to_string(),
                            step_name.to_string(),
                            format!("System template '{}' not found", sys_tmpl),
                        );
                    }
                }
            }
            StepType::JsonGeneration(s) => {
                // Check template exists
                if !self
                    .resources
                    .templates
                    .templates
                    .contains_key(&s.generation_step.template)
                {
                    result.add_error(
                        "Template".to_string(),
                        step_name.to_string(),
                        format!("Template '{}' not found", s.generation_step.template),
                    );
                }
                // Check LLM exists
                if !self
                    .resources
                    .llms
                    .resources
                    .contains_key(&s.generation_step.llm)
                {
                    result.add_error(
                        "LLM".to_string(),
                        step_name.to_string(),
                        format!("LLM '{}' not found", s.generation_step.llm),
                    );
                }
                // Check system template if specified
                if let Some(sys_tmpl) = &s.generation_step.system_template {
                    if !self.resources.templates.templates.contains_key(sys_tmpl) {
                        result.add_error(
                            "Template".to_string(),
                            step_name.to_string(),
                            format!("System template '{}' not found", sys_tmpl),
                        );
                    }
                }
            }
            StepType::DataSampler(s) => {
                // Check dataset exists
                if !self.resources.datasets.resources.contains_key(&s.dataset) {
                    result.add_error(
                        "Dataset".to_string(),
                        step_name.to_string(),
                        format!("Dataset '{}' not found", s.dataset),
                    );
                }
            }
            StepType::Render(s) => {
                // Check template exists
                if !self.resources.templates.templates.contains_key(&s.template) {
                    result.add_error(
                        "Template".to_string(),
                        step_name.to_string(),
                        format!("Template '{}' not found", s.template),
                    );
                }
            }
            StepType::CheckEmbedding(s) => {
                // Check embedding exists
                if !self
                    .resources
                    .embeddings
                    .resources
                    .contains_key(&s.embedding)
                {
                    result.add_error(
                        "Embedding".to_string(),
                        step_name.to_string(),
                        format!("Embedding '{}' not found", s.embedding),
                    );
                }
            }
            StepType::JudgeConversation(s) => {
                // Check LLM exists
                if !self
                    .resources
                    .llms
                    .resources
                    .contains_key(&s.json_generation_step.generation_step.llm)
                {
                    result.add_error(
                        "LLM".to_string(),
                        step_name.to_string(),
                        format!(
                            "LLM '{}' not found",
                            s.json_generation_step.generation_step.llm
                        ),
                    );
                }
            }
            StepType::IfElse(s) => {
                // Recursively validate then/else steps
                for (idx, then_step) in s.then_steps.iter().enumerate() {
                    let then_name = format!("{}->Then#{}", step_name, idx + 1);
                    self.validate_step(then_step, &then_name, result);
                }
                if let Some(else_steps) = &s.else_steps {
                    for (idx, else_step) in else_steps.iter().enumerate() {
                        let else_name = format!("{}->Else#{}", step_name, idx + 1);
                        self.validate_step(else_step, &else_name, result);
                    }
                }
            }
            // Other step types don't have resource dependencies to validate
            _ => {}
        }
    }

    fn validate_templates(&self, result: &mut ValidationResult) {
        // Try to compile all templates
        if let Err(e) = self.resources.templates.compile() {
            result.add_error(
                "Template Compilation".to_string(),
                "All Templates".to_string(),
                format!("Template compilation failed: {}", e),
            );
        }
    }
}
