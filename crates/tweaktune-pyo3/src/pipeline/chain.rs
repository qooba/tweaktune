use log::debug;
use pyo3::{pyclass, pymethods, types::PyAny, Py, Python};
use tweaktune_core::steps::{
    generators::{JsonGenerationStep, TextGenerationStep},
    py::PyStep,
    DataSamplerStep, PrintStep, StepType,
};
use tweaktune_core::templates::Templates;

#[pyclass]
#[derive(Debug)]
pub struct StepsChain {
    pub(super) steps: Vec<Step>,
}

#[pymethods]
impl StepsChain {
    #[new]
    pub fn new() -> Self {
        StepsChain { steps: Vec::new() }
    }

    pub fn add_py_step(&mut self, named: String, py_func: Py<PyAny>) {
        debug!("Added Python step: {}", &named);
        self.steps.push(Step::Py {
            name: named,
            py_func,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_text_generation_step(
        &mut self,
        name: String,
        template: String,
        llm: String,
        output: String,
        system_template: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        enable_thinking: Option<bool>,
    ) {
        debug!(
            "Added text generation step with llm: {}, template: {}",
            &llm, &template
        );
        self.steps.push(Step::TextGeneration {
            name,
            template,
            llm,
            output,
            system_template,
            max_tokens,
            temperature,
            enable_thinking,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_json_generation_step(
        &mut self,
        name: String,
        template: String,
        llm: String,
        output: String,
        json_path: Option<String>,
        system_template: Option<String>,
        schema_template: Option<String>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        enable_thinking: Option<bool>,
    ) {
        debug!(
            "Added JSON generation step with template: {}, llm: {}",
            &llm, &template
        );
        self.steps.push(Step::JsonGeneration {
            name,
            template,
            llm,
            output,
            json_path,
            system_template,
            json_schema,
            max_tokens,
            temperature,
            schema_template,
            enable_thinking,
        });
    }

    pub fn add_print_step(
        &mut self,
        name: String,
        template: Option<String>,
        columns: Option<Vec<String>>,
    ) {
        debug!("Added print step");
        self.steps.push(Step::Print {
            name,
            template,
            columns,
        });
    }

    pub fn add_data_sampler_step(
        &mut self,
        name: String,
        dataset: String,
        size: usize,
        output: String,
    ) {
        debug!(
            "Added data sampler on dataset: {} with size: {}",
            &dataset, &size
        );
        self.steps.push(Step::DataSampler {
            name,
            dataset,
            size,
            output,
        });
    }

    pub fn add_py_validator_step(&mut self, name: String, py_func: Py<PyAny>) {
        debug!("Added Python validator step: {}", &name);
        self.steps.push(Step::PyValidator { name, py_func });
    }

    pub fn add_jsonl_writer_step(&mut self, name: String, path: String, template: String) {
        debug!("Added JSONL writer step: {}", &name);
        self.steps.push(Step::JsonlWriter {
            name,
            path,
            template,
        });
    }

    pub fn add_new_column_step(&mut self, _name: String, _mutation: String, _output: String) {
        todo!()
    }

    pub fn add_filter_step(&mut self, _name: String, _condition: String) {
        todo!()
    }

    pub fn add_mutate_step(&mut self, _name: String, _mutation: String, _output: String) {
        todo!()
    }
}

impl Default for StepsChain {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass]
#[derive(Debug)]
pub enum Step {
    Py {
        name: String,
        py_func: Py<PyAny>,
    },
    TextGeneration {
        name: String,
        template: String,
        llm: String,
        output: String,
        system_template: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        enable_thinking: Option<bool>,
    },
    JsonGeneration {
        name: String,
        template: String,
        llm: String,
        output: String,
        json_path: Option<String>,
        system_template: Option<String>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        schema_template: Option<String>,
        enable_thinking: Option<bool>,
    },
    Print {
        name: String,
        template: Option<String>,
        columns: Option<Vec<String>>,
    },
    DataSampler {
        name: String,
        dataset: String,
        size: usize,
        output: String,
    },
    Judge {
        name: String,
        template: String,
        llm: String,
    },
    PyValidator {
        name: String,
        py_func: Py<PyAny>,
    },
    JsonlWriter {
        name: String,
        path: String,
        template: String,
    },
}

pub(super) fn map_step(step: &Step, templates: &mut Templates) -> StepType {
    match step {
        Step::Py { name, py_func } => Python::attach(|py| {
            let py_obj: Py<PyAny> = py_func.clone_ref(py);
            StepType::Py(PyStep::new(name.clone(), py_obj))
        }),
        Step::TextGeneration {
            name,
            template,
            llm,
            output,
            system_template,
            max_tokens,
            temperature,
            enable_thinking,
        } => StepType::TextGeneration(TextGenerationStep::new(
            name.clone(),
            template.clone(),
            llm.clone(),
            output.clone(),
            system_template.clone(),
            *max_tokens,
            *temperature,
            *enable_thinking,
        )),
        Step::JsonGeneration {
            name,
            template,
            llm,
            output,
            json_path,
            system_template,
            json_schema,
            max_tokens,
            temperature,
            enable_thinking,
            schema_template,
        } => {
            let schema_key = schema_template
                .as_ref()
                .map(|schema| templates.add_inline("json_generation_step", name, schema));

            StepType::JsonGeneration(JsonGenerationStep::new(
                name.clone(),
                template.clone(),
                llm.clone(),
                output.clone(),
                json_path.clone(),
                system_template.clone(),
                json_schema.clone(),
                *max_tokens,
                *temperature,
                schema_key,
                *enable_thinking,
            ))
        }
        Step::Print {
            name,
            template,
            columns,
        } => StepType::Print(PrintStep::new(
            name.clone(),
            template.clone(),
            columns.clone(),
        )),
        Step::DataSampler {
            name,
            dataset,
            size,
            output,
        } => StepType::DataSampler(DataSamplerStep::new(
            name.clone(),
            dataset.clone(),
            Some(*size),
            output.clone(),
        )),
        _ => unimplemented!(), // Handle other step types as needed
    }
}
