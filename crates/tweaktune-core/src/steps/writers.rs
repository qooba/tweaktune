use crate::{
    steps::{Step, StepContext, StepStatus},
    PipelineResources,
};
use anyhow::Result;
use log::error;
use std::fs::File;
use std::io::Write;

pub struct JsonlWriterStep {
    pub name: String,
    pub path: String,
    pub template: Option<String>,
    pub value: Option<String>,
}

impl JsonlWriterStep {
    pub fn new(
        name: String,
        path: String,
        template: Option<String>,
        value: Option<String>,
    ) -> Self {
        Self {
            name,
            path,
            template,
            value,
        }
    }
}

impl Step for JsonlWriterStep {
    async fn process(
        &self,
        resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let file = File::options().append(true).create(true).open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);
        let row = if let Some(template) = &self.template {
            resources
                .templates
                .render(template.clone(), context.data.clone())
        } else if let Some(value) = &self.value {
            if let Some(v) = context.get(value) {
                if let Some(inner) = v.as_str() {
                    Ok(inner.to_string())
                } else {
                    Ok(v.to_string())
                }
            } else {
                error!(target: "json_writer_step", "🐔 Value '{}' not found in context for JsonlWriterStep", value);
                let mut context = context.clone();
                context.set_status(StepStatus::Failed);
                return Ok(context);
            }
        } else {
            error!(target: "json_writer_step", "🐔 You must set either a template or a value");
            let mut context = context.clone();
            context.set_status(StepStatus::Failed);
            return Ok(context);
        };

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
        _resources: &PipelineResources,
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
