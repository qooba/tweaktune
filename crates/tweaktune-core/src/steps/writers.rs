use crate::{
    datasets::DatasetType,
    embeddings::{self},
    llms::{self},
    steps::{Step, StepContext, StepStatus},
    templates::Templates,
};
use anyhow::Result;
use log::error;
use std::io::Write;
use std::{collections::HashMap, fs::File};

pub struct JsonlWriterStep {
    pub name: String,
    pub path: String,
    pub template: String,
}

impl JsonlWriterStep {
    pub fn new(name: String, path: String, template: String) -> Self {
        Self {
            name,
            path,
            template,
        }
    }
}

impl Step for JsonlWriterStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let file = File::options().append(true).create(true).open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);
        let row = templates.render(self.template.clone(), context.data.clone());
        let mut context = context.clone();
        match row {
            Ok(r) => {
                let r = r.replace("\\n", "\n").replace('\n', "\\n");
                writeln!(writer, "{}", r)?;
                writer.flush()?;
            }
            Err(e) => {
                error!(target: "json_writer_step", "🐔 Failed to render template: {}", e);
                context.set_status(StepStatus::Failed);
            }
        };

        Ok(context)
    }
}

pub struct CsvWriterStep {
    pub name: String,
    pub path: String,
    pub columns: Vec<String>,
    pub delimeter: String,
}

impl CsvWriterStep {
    pub fn new(name: String, path: String, columns: Vec<String>, delimeter: String) -> Self {
        Self {
            name,
            path,
            columns,
            delimeter,
        }
    }
}

impl Step for CsvWriterStep {
    async fn process(
        &self,
        _datasets: &HashMap<String, DatasetType>,
        _templates: &Templates,
        _llms: &HashMap<String, llms::LLMType>,
        _embeddings: &HashMap<String, embeddings::EmbeddingsType>,
        context: &StepContext,
    ) -> Result<StepContext> {
        let file = File::options().append(true).create(true).open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);
        let mut row = String::new();
        for (i, column) in self.columns.iter().enumerate() {
            if let Some(value) = context.get(column) {
                if i > 0 {
                    row.push_str(&self.delimeter);
                }
                row.push_str(&value.to_string());
            }
        }

        let row = row.replace("\\n", "\n").replace('\n', "\\n");
        writeln!(writer, "{}", row)?;
        writer.flush()?;

        Ok(context.clone())
    }
}
