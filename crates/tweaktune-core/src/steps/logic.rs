use crate::{
    state::State,
    steps::{Step, StepContext, StepStatus},
    PipelineResources,
};
use anyhow::Result;
use log::error;

pub struct FilterStep {
    pub name: String,
    pub condition: String,
}

impl FilterStep {
    pub fn new(name: String, condition: String) -> Self {
        Self { name, condition }
    }
}

impl Step for FilterStep {
    async fn process(
        &self,
        resources: &PipelineResources,
        context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        let rendered = resources
            .templates
            .render(self.condition.clone(), context.data.clone())?;
        if let Ok(v) = serde_json::from_str::<bool>(&rendered) {
            if !v {
                context.set_status(StepStatus::Failed);
            }
        }

        Ok(context)
    }
}

pub struct MutateStep {
    pub name: String,
    pub condition: String,
    pub output: String,
    pub fail_if_exists: bool,
}

impl MutateStep {
    pub fn new(name: String, condition: String, output: String, fail_if_exists: bool) -> Self {
        Self {
            name,
            condition,
            output,
            fail_if_exists,
        }
    }
}

impl Step for MutateStep {
    async fn process(
        &self,
        resources: &PipelineResources,
        context: &StepContext,
        _state: Option<State>,
    ) -> Result<StepContext> {
        let mut context = context.clone();
        if self.fail_if_exists && context.data.get(&self.output).is_some() {
            error!(target: "mutatestep", "🐔 Output key '{}' already exists in context data", self.output);
            context.set_status(StepStatus::Failed);
            return Ok(context);
        }

        let rendered = resources
            .templates
            .render(self.condition.clone(), context.data.clone())?;
        match serde_json::from_str::<serde_json::Value>(&rendered) {
            Ok(v) => {
                context.set(&self.output, v);
            }
            Err(_) => {
                context.set(&self.output, serde_json::Value::String(rendered));
            }
        }

        Ok(context)
    }
}
