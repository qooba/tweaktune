use crate::{
    common::ResultExt,
    datasets::DatasetType,
    embeddings::EmbeddingsType,
    llms::LLMType,
    state::State,
    steps::{Step, StepContext, StepStatus},
    templates::Templates,
    PipelineResources,
};
use anyhow::Result;
use log::error;
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct PyStep {
    pub name: String,
    pub py_func: PyObject,
}

impl PyStep {
    pub fn new(name: String, py_func: PyObject) -> Self {
        Self { name, py_func }
    }
}

impl Step for PyStep {
    async fn process(
        &self,
        _resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let json = serde_json::to_string(context)?;

        let result: PyResult<String> = Python::with_gil(|py| {
            let result: String = self
                .py_func
                .call_method1(py, "process", (json,))?
                .extract(py)?;
            Ok(result)
        });

        match result {
            Ok(result) => {
                let result: StepContext = serde_json::from_str(&result)?;
                Ok(result)
            }
            Err(e) => {
                error!(target: "pystep", "🐔 {:?}", e);
                let mut context = context.clone();
                context.set_status(StepStatus::Failed);
                Ok(context)
            }
        }
    }
}

pub struct PyValidator {
    pub name: String,
    pub py_func: PyObject,
}

impl PyValidator {
    pub fn new(name: String, py_func: PyObject) -> Self {
        Self { name, py_func }
    }
}

impl Step for PyValidator {
    async fn process(
        &self,
        _resources: &PipelineResources,
        context: &StepContext,
    ) -> Result<StepContext> {
        let json = serde_json::to_string(context)?;

        let result: PyResult<bool> = Python::with_gil(|py| {
            let result: bool = self
                .py_func
                .call_method1(py, "process", (json,))?
                .extract(py)?;
            Ok(result)
        });

        let result = result.map_tt_err("VALIDATOR MUST RETURN BOOL")?;
        let mut context = context.clone();
        if !result {
            context.set_status(StepStatus::Failed);
        }

        Ok(context)
    }
}
