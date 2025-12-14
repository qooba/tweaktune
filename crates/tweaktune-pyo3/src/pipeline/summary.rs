use crate::pipeline::{IterBy, PipelineBuilder};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, ContentArrangement, Table};

impl PipelineBuilder {
    pub(crate) fn get_summary(&self) -> String {
        let mut output = String::new();

        // Resources summary table
        let mut resources_table = Table::new();
        resources_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        resources_table.set_header(vec![
            Cell::from("Resource Type"),
            Cell::from("Count"),
            Cell::from("Names"),
        ]);

        // Datasets
        let dataset_names = self.resources.datasets.list();
        resources_table.add_row(vec![
            Cell::from("Datasets"),
            Cell::from(dataset_names.len().to_string()),
            Cell::from(if dataset_names.is_empty() {
                "-".to_string()
            } else {
                dataset_names.join(", ")
            }),
        ]);

        // LLMs
        let llm_names = self.resources.llms.list();
        resources_table.add_row(vec![
            Cell::from("LLMs"),
            Cell::from(llm_names.len().to_string()),
            Cell::from(if llm_names.is_empty() {
                "-".to_string()
            } else {
                llm_names.join(", ")
            }),
        ]);

        // Embeddings
        let embedding_names = self.resources.embeddings.list();
        resources_table.add_row(vec![
            Cell::from("Embeddings"),
            Cell::from(embedding_names.len().to_string()),
            Cell::from(if embedding_names.is_empty() {
                "-".to_string()
            } else {
                embedding_names.join(", ")
            }),
        ]);

        // Templates
        let template_count = self.resources.templates.templates.len();
        resources_table.add_row(vec![
            Cell::from("Templates"),
            Cell::from(template_count.to_string()),
            Cell::from(if template_count == 0 {
                "-".to_string()
            } else if template_count > 5 {
                format!("{} templates", template_count)
            } else {
                self.resources
                    .templates
                    .templates
                    .keys()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            }),
        ]);

        // Tokenizers
        let tokenizer_names = self.resources.tokenizers.list();
        resources_table.add_row(vec![
            Cell::from("Tokenizers"),
            Cell::from(tokenizer_names.len().to_string()),
            Cell::from(if tokenizer_names.is_empty() {
                "-".to_string()
            } else {
                tokenizer_names.join(", ")
            }),
        ]);

        output.push_str("Pipeline Resources:\n");
        output.push_str(&resources_table.to_string());
        output.push('\n');

        // Pipeline configuration table
        let mut config_table = Table::new();
        config_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        config_table.set_header(vec![Cell::from("Configuration"), Cell::from("Value")]);

        config_table.add_row(vec![
            Cell::from("Workers"),
            Cell::from(self.workers.to_string()),
        ]);

        // Iteration type
        let iter_info = match &self.iter_by {
            IterBy::Range { start, stop, step } => {
                format!("Range: {}..{} (step: {})", start, stop, step)
            }
            IterBy::Dataset { name } => {
                format!("Dataset: {}", name)
            }
        };
        config_table.add_row(vec![Cell::from("Iteration"), Cell::from(iter_info)]);

        config_table.add_row(vec![
            Cell::from("Steps"),
            Cell::from(self.steps.len().to_string()),
        ]);

        output.push_str("Pipeline Configuration:\n");
        output.push_str(&config_table.to_string());
        output.push('\n');

        // Steps table
        if !self.steps.is_empty() {
            let mut steps_table = Table::new();
            steps_table
                .load_preset(UTF8_FULL)
                .apply_modifier(UTF8_ROUND_CORNERS)
                .set_content_arrangement(ContentArrangement::Dynamic);

            steps_table.set_header(vec![
                Cell::from("#"),
                Cell::from("Step Type"),
                Cell::from("Details"),
            ]);

            for (idx, step) in self.steps.iter().enumerate() {
                let (step_type, details) = get_step_info(step);
                steps_table.add_row(vec![
                    Cell::from((idx + 1).to_string()),
                    Cell::from(step_type),
                    Cell::from(details),
                ]);
            }

            output.push_str("Pipeline Steps:\n");
            output.push_str(&steps_table.to_string());
        }

        output
    }
}

fn get_step_info(step: &tweaktune_core::steps::StepType) -> (String, String) {
    use tweaktune_core::steps::StepType;

    match step {
        StepType::TextGeneration(s) => (
            "TextGeneration".to_string(),
            format!("LLM: {}, Output: {}", s.llm, s.output),
        ),
        StepType::JsonGeneration(s) => (
            "JsonGeneration".to_string(),
            format!("LLM: {}, Output: {}", s.generation_step.llm, s.output),
        ),
        StepType::DataSampler(s) => (
            "DataSampler".to_string(),
            format!("Dataset: {}, Output: {}", s.dataset, s.output),
        ),
        StepType::Print(_) => ("Print".to_string(), "-".to_string()),
        StepType::JsonWriter(s) => ("JsonWriter".to_string(), format!("Path: {}", s.path)),
        StepType::CsvWriter(s) => ("CsvWriter".to_string(), format!("Path: {}", s.path)),
        StepType::Render(s) => ("Render".to_string(), format!("Output: {}", s.output)),
        StepType::RenderConversation(s) => (
            "RenderConversation".to_string(),
            format!("Output: {}", s.output),
        ),
        StepType::RenderToolCall(s) => (
            "RenderToolCall".to_string(),
            format!("Output: {}", s.output),
        ),
        StepType::RenderDPO(s) => ("RenderDPO".to_string(), format!("Output: {}", s.output)),
        StepType::RenderGRPO(s) => ("RenderGRPO".to_string(), format!("Output: {}", s.output)),
        StepType::ValidateJson(_) => ("ValidateJson".to_string(), "-".to_string()),
        StepType::CheckJson(_) => ("CheckJson".to_string(), "-".to_string()),
        StepType::ValidateTools(_) => ("ValidateTools".to_string(), "-".to_string()),
        StepType::NormalizeTools(s) => (
            "NormalizeTools".to_string(),
            format!("Output: {}", s.output),
        ),
        StepType::ConversationValidate(_) => ("ConversationValidate".to_string(), "-".to_string()),
        StepType::Filter(_) => ("Filter".to_string(), "-".to_string()),
        StepType::Mutate(s) => ("Mutate".to_string(), format!("Output: {}", s.output)),
        StepType::Chunk(s) => ("Chunk".to_string(), format!("Output: {}", s.output)),
        StepType::IntoList(s) => ("IntoList".to_string(), format!("Output: {}", s.output)),
        StepType::CheckLanguage(s) => (
            "CheckLanguage".to_string(),
            format!("Language: {}", s.language),
        ),
        StepType::CheckHash(_) => ("CheckHash".to_string(), "-".to_string()),
        StepType::CheckSimHash(_) => ("CheckSimHash".to_string(), "-".to_string()),
        StepType::CheckEmbedding(s) => (
            "CheckEmbedding".to_string(),
            format!("Embedding: {}", s.embedding),
        ),
        StepType::JudgeConversation(s) => (
            "JudgeConversation".to_string(),
            format!(
                "LLM: {}, Output: {}",
                s.json_generation_step.generation_step.llm, s.json_generation_step.output
            ),
        ),
        StepType::IfElse(_) => ("IfElse".to_string(), "Conditional branching".to_string()),
        StepType::Py(s) => ("Python".to_string(), format!("Name: {}", s.name)),
        StepType::PyValidator(s) => ("PyValidator".to_string(), format!("Name: {}", s.name)),
        StepType::ToolArgumentsSampler(tool_arguments_sampler_step) => (
            "ToolArgumentsSampler".to_string(),
            format!(
                "ToolKey: {}, Output: {}",
                tool_arguments_sampler_step.tool_key, tool_arguments_sampler_step.output
            ),
        ),
        StepType::Literal(s) => ("Literal".to_string(), format!("Name: {}", s.name)),
    }
}
