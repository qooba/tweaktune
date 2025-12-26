use crate::{
    common::ResultExt,
    steps::{Step, StepContext, StepStatus},
    PipelineResources,
};
use anyhow::Result;
use log::error;
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};

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
        let result: Result<StepContext> = Python::with_gil(|py| {
            let py_context = pythonize(py, context)
                .map_err(|e| anyhow::anyhow!("Failed to pythonize context: {:?}", e))?;
            let result = self.py_func.call_method1(py, "process", (py_context,))?;
            depythonize(result.bind(py))
                .map_err(|e| anyhow::anyhow!("Failed to depythonize result: {:?}", e))
        });

        match result {
            Ok(result) => Ok(result),
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
        let result: Result<bool> = Python::with_gil(|py| {
            let py_context = pythonize(py, context)
                .map_err(|e| anyhow::anyhow!("Failed to pythonize context: {:?}", e))?;
            let py_result = self.py_func.call_method1(py, "process", (py_context,))?;
            let result: bool = py_result.extract(py)?;
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
