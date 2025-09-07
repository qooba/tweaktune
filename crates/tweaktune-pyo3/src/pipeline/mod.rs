use crate::common::ResultExt;
use crate::logging::{BusEvent, ChannelWriter, LogsCollector};
use anyhow::{bail, Result};
use chrono::Local;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info};
use pyo3::{pyclass, pymethods, PyObject, PyRef, PyResult, Python};
use serde::de;
use serde_json::json;
use simplelog::*;
use std::fs::{create_dir_all, File};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::{collections::HashMap, sync::Arc};
use tweaktune_core::common::{deserialize, run_async, SerializationType};
use tweaktune_core::datasets::{
    CsvDataset, Dataset as DatasetTrait, IpcDataset, JsonlDataset, MixedDataset, ParquetDataset,
    PolarsDataset,
};
use tweaktune_core::llms::{ApiLLMMode, MistralrsLLM, UnslothLLM};
use tweaktune_core::readers::read_to_string;
use tweaktune_core::steps::conversations::RenderConversationStep;
use tweaktune_core::steps::validators::{
    ConversationValidateStep, ToolsNormalizeStep, ToolsValidateStep,
};
use tweaktune_core::steps::IntoListStep;
use tweaktune_core::steps::{validators::ValidateJsonStep, ChunkStep, IfElseStep, RenderStep};
use tweaktune_core::{
    common::OptionToResult,
    datasets::{DatasetType, JsonDataset, JsonListDataset, OpenApiDataset},
    embeddings::{EmbeddingsType, OpenAIEmbeddings},
    llms::{ApiLLM, LLMType},
    steps::{
        generators::{JsonGenerationStep, TextGenerationStep},
        py::{PyStep, PyValidator},
        writers::{CsvWriterStep, JsonlWriterStep},
        DataSamplerStep, PrintStep, Step as StepCore, StepContext, StepStatus, StepType,
    },
    templates::Templates,
};

#[pyclass]
pub struct PipelineBuilder {
    workers: usize,
    datasets: Resources<DatasetType>,
    templates: Templates,
    llms: Resources<LLMType>,
    embeddings: Resources<EmbeddingsType>,
    steps: Vec<StepType>,
    iter_by: IterBy,
    running: Arc<AtomicBool>,
    logs_collector: Arc<LogsCollector>,
}

#[pymethods]
impl PipelineBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            workers: 1,
            datasets: Resources {
                resources: HashMap::new(),
            },
            templates: Templates::default(),
            llms: Resources {
                resources: HashMap::new(),
            },
            embeddings: Resources {
                resources: HashMap::new(),
            },
            steps: vec![],
            iter_by: IterBy::Range {
                start: 0,
                stop: 0,
                step: 1,
            },
            running: Arc::new(AtomicBool::new(false)),
            logs_collector: Arc::new(LogsCollector::new()),
        }
    }

    pub fn with_workers(&mut self, workers: usize) {
        self.workers = workers;
        debug!("Setting workers to {}", workers);
    }

    pub fn with_openapi_dataset(&mut self, name: String, path_or_url: String) -> PyResult<()> {
        debug!("Added OPEN_API dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::OpenApi(OpenApiDataset::new(name, path_or_url)?),
        );
        Ok(())
    }

    #[pyo3(signature = (name, json_list, sql=None))]
    pub fn with_json_list_dataset(
        &mut self,
        name: String,
        json_list: Vec<String>,
        sql: Option<String>,
    ) -> PyResult<()> {
        debug!("Added JSON_LIST dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::JsonList(JsonListDataset::new(name, json_list, sql)?),
        );
        Ok(())
    }

    #[pyo3(signature = (name, path, sql=None))]
    pub fn with_jsonl_dataset(
        &mut self,
        name: String,
        path: String,
        sql: Option<String>,
    ) -> PyResult<()> {
        debug!("Added JSONL dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::Jsonl(JsonlDataset::new(name, path, sql)?),
        );
        Ok(())
    }

    pub fn with_polars_dataset(&mut self, name: String, path: String, sql: String) -> PyResult<()> {
        debug!("Added POLARS dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::Polars(PolarsDataset::new(name, path, Some(sql))?),
        );
        Ok(())
    }

    #[pyo3(signature = (name, path, sql=None))]
    pub fn with_json_dataset(
        &mut self,
        name: String,
        path: String,
        sql: Option<String>,
    ) -> PyResult<()> {
        debug!("Added JSON dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::Json(JsonDataset::new(name, path, sql)?),
        );
        Ok(())
    }

    pub fn with_mixed_dataset(&mut self, name: String, datasets: Vec<String>) -> PyResult<()> {
        debug!("Added MIXED dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::Mixed(MixedDataset::new(name, datasets, &self.datasets.resources)?),
        );
        Ok(())
    }

    #[pyo3(signature = (name, path, sql=None))]
    pub fn with_parquet_dataset(
        &mut self,
        name: String,
        path: String,
        sql: Option<String>,
    ) -> PyResult<()> {
        debug!("Added Parquet dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::Parquet(ParquetDataset::new(name, path, sql)?),
        );
        Ok(())
    }

    #[pyo3(signature = (name, ipc_data, sql=None))]
    pub fn with_ipc_dataset(
        &mut self,
        name: String,
        ipc_data: &[u8],
        sql: Option<String>,
    ) -> PyResult<()> {
        debug!("Added Ipc dataset: {}", &name);

        self.datasets.add(
            name.clone(),
            DatasetType::Ipc(IpcDataset::new(name, ipc_data, sql)?),
        );
        Ok(())
    }

    #[pyo3(signature = (name, path, delimiter, has_header, sql=None))]
    pub fn with_csv_dataset(
        &mut self,
        name: String,
        path: String,
        delimiter: String,
        has_header: bool,
        sql: Option<String>,
    ) -> PyResult<()> {
        debug!("Added CSV dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::Csv(CsvDataset::new(
                name,
                path,
                delimiter.as_bytes()[0],
                has_header,
                sql,
            )?),
        );
        Ok(())
    }

    pub fn with_llm_api(
        &mut self,
        name: String,
        base_url: String,
        api_key: String,
        model: String,
        max_tokens: u32,
        temperature: f32,
    ) {
        debug!("Added LLM API: {}", &name);
        self.llms.add(
            name.clone(),
            LLMType::Api(ApiLLM::new(
                name,
                ApiLLMMode::Api {
                    base_url,
                    api_key,
                    model,
                },
                max_tokens,
                temperature,
            )),
        );
    }

    pub fn with_llm_openai(
        &mut self,
        name: String,
        api_key: String,
        model: String,
        max_tokens: u32,
        temperature: f32,
    ) {
        debug!("Added LLM API: {}", &name);
        self.llms.add(
            name.clone(),
            LLMType::Api(ApiLLM::new(
                name,
                ApiLLMMode::OpenAI { api_key, model },
                max_tokens,
                temperature,
            )),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_llm_azure_openai(
        &mut self,
        name: String,
        api_key: String,
        endpoint: String,
        deployment_name: String,
        api_version: String,
        max_tokens: u32,
        temperature: f32,
    ) {
        debug!("Added LLM API: {}", &name);
        self.llms.add(
            name.clone(),
            LLMType::Api(ApiLLM::new(
                name,
                ApiLLMMode::AzureOpenAI {
                    api_key,
                    endpoint,
                    deployment_name,
                    api_version,
                },
                max_tokens,
                temperature,
            )),
        );
    }

    pub fn with_llm_unsloth(&mut self, name: String, py_func: PyObject) {
        debug!("Added LLM UNSLOTH: {}", &name);
        self.llms.add(
            name.clone(),
            LLMType::Unsloth(UnslothLLM::new(name, py_func)),
        );
    }

    pub fn with_llm_mistralrs(&mut self, name: String, py_func: PyObject) {
        debug!("Added LLM MISTRALRS: {}", &name);
        self.llms.add(
            name.clone(),
            LLMType::Mistralrs(MistralrsLLM::new(name, py_func)),
        );
    }

    pub fn with_embeddings_api(
        &mut self,
        name: String,
        base_url: String,
        api_key: String,
        model: String,
    ) {
        debug!("Added OpenAI embeddings: {}", &name);
        self.embeddings.add(
            name.clone(),
            EmbeddingsType::OpenAI(OpenAIEmbeddings::new(name, base_url, api_key, model)),
        );
    }

    pub fn with_jinja_template(&mut self, name: String, template: String) {
        debug!("Added Jinja template: {}", &name);
        self.templates.add(name, template);
    }

    #[pyo3(signature = (path, op_config=None))]
    pub fn with_dir_templates(&mut self, path: String, op_config: Option<String>) {
        if let Ok(entries) = std::fs::read_dir(&path) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".j2") || name.ends_with(".jinja") {
                        let template_path = entry.path();
                        if let Some(template_path_str) = template_path.to_str() {
                            if let Ok(template) =
                                read_to_string(template_path_str, op_config.clone())
                            {
                                debug!("Added Jinja template: {}", &name);
                                self.templates.add(name.to_string(), template);
                            } else {
                                error!("Failed to read template file: {}", template_path.display());
                            }
                        } else {
                            error!(
                                "Failed to convert template path to string: {}",
                                template_path.display()
                            );
                        }
                    }
                }
            }
        } else {
            error!("Failed to read directory: {}", path);
        }
    }

    #[pyo3(signature = (name, path, op_config=None))]
    pub fn with_j2_template(&mut self, name: String, path: String, op_config: Option<String>) {
        debug!("Added Jinja template: {}", &name);
        let template = read_to_string(&path, op_config).unwrap();
        self.templates.add(name, template);
    }

    #[pyo3(signature = (path, op_config=None))]
    pub fn with_j2_templates(&mut self, path: String, op_config: Option<String>) {
        let serialization_type = if path.ends_with(".json") {
            SerializationType::JSON
        } else if path.ends_with(".yaml") || path.ends_with(".yml") {
            SerializationType::YAML
        } else {
            error!("Unsupported template file format. Supported formats are .json and .yaml/.yml");
            return;
        };
        let template = read_to_string(&path, op_config).unwrap();
        let templates = deserialize::<Templates>(&template, serialization_type).unwrap();
        for (name, template) in templates.templates {
            debug!("Adding template: {}", &name);
            self.templates.add(name, template);
        }
    }

    pub fn iter_by_range(&mut self, start: usize, stop: usize, step: usize) {
        self.iter_by = IterBy::Range { start, stop, step };
    }

    pub fn iter_by_dataset(&mut self, name: String) {
        self.iter_by = IterBy::Dataset { name };
    }

    pub fn add_py_step(&mut self, name: String, py_func: PyObject) {
        debug!("Added Python step: {}", &name);
        self.steps.push(StepType::Py(PyStep::new(name, py_func)));
    }

    pub fn add_ifelse_step(
        &mut self,
        name: String,
        condition: PyObject,
        then_steps: PyRef<StepsChain>,
        else_steps: PyRef<StepsChain>,
    ) {
        debug!("Added Ifelse step: {}", &name);

        let then_steps = then_steps
            .steps
            .iter()
            .map(|step| map_step(step, &mut self.templates))
            .collect::<Vec<_>>();

        let else_steps = if !else_steps.steps.is_empty() {
            Some(
                else_steps
                    .steps
                    .iter()
                    .map(|step| map_step(step, &mut self.templates))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        self.steps.push(StepType::IfElse(IfElseStep::new(
            name, condition, then_steps, else_steps,
        )));
    }

    pub fn add_py_validator_step(&mut self, name: String, py_func: PyObject) {
        debug!("Added Python validator step: {}", &name);
        self.steps
            .push(StepType::PyValidator(PyValidator::new(name, py_func)));
    }

    pub fn add_into_list_step(&mut self, name: String, inputs: Vec<String>, output: String) {
        debug!("Added IntoList step: {}", &name);
        self.steps
            .push(StepType::IntoList(IntoListStep::new(name, inputs, output)));
    }

    pub fn add_validate_conversation_step(&mut self, name: String, conversation: String) {
        debug!("Added conversation validation step: {}", &name);
        self.steps.push(StepType::ConversationValidate(
            ConversationValidateStep::new(name, conversation),
        ));
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, template, llm, output, system_template=None, max_tokens=None, temperature=None))]
    pub fn add_text_generation_step(
        &mut self,
        name: String,
        template: String,
        llm: String,
        output: String,
        system_template: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) {
        debug!(
            "Added text generation step with llm: {}, template: {}",
            &llm, &template
        );
        self.steps
            .push(StepType::TextGeneration(TextGenerationStep::new(
                name,
                template,
                llm,
                output,
                system_template,
                max_tokens,
                temperature,
            )));
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, template, llm, output, json_path=None, system_template=None, json_schema=None, max_tokens=None, temperature=None, schema_template=None))]
    pub fn add_json_generation_step(
        &mut self,
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
    ) {
        debug!(
            "Added JSON generation step with template: {}, llm: {}",
            &llm, &template
        );

        let schema_key = if let Some(schema) = &schema_template {
            let schema_key = format!("json_generation_step_{}_{}", name, schema);
            self.templates
                .add(schema_key.clone(), format!("{{{{{}}}}}", schema.clone()));
            Some(schema_key)
        } else {
            None
        };

        self.steps
            .push(StepType::JsonGeneration(JsonGenerationStep::new(
                name,
                template,
                llm,
                output,
                json_path,
                system_template,
                json_schema,
                max_tokens,
                temperature,
                schema_key,
            )));
    }

    #[pyo3(signature = (name, path, template=None, value=None))]
    pub fn add_write_jsonl_step(
        &mut self,
        name: String,
        path: String,
        template: Option<String>,
        value: Option<String>,
    ) {
        debug!("Added JSONL writer step: {}", &name);
        self.steps.push(StepType::JsonWriter(JsonlWriterStep::new(
            name, path, template, value,
        )));
    }

    #[pyo3(signature = (name, template=None, columns=None))]
    pub fn add_print_step(
        &mut self,
        name: String,
        template: Option<String>,
        columns: Option<Vec<String>>,
    ) {
        debug!("Added print step: {}", &name);
        self.steps
            .push(StepType::Print(PrintStep::new(name, template, columns)));
    }

    pub fn add_write_csv_step(
        &mut self,
        name: String,
        path: String,
        columns: Vec<String>,
        delimiter: String,
    ) {
        debug!("Added CSV writer step: {}", &name);
        self.steps.push(StepType::CsvWriter(CsvWriterStep::new(
            name, path, columns, delimiter,
        )));
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
        self.steps.push(StepType::DataSampler(DataSamplerStep::new(
            name,
            dataset,
            Some(size),
            output,
        )));
    }

    pub fn add_tool_sampler_step(
        &mut self,
        name: String,
        dataset: String,
        size: usize,
        output: String,
    ) {
        self.add_data_sampler_step(name.clone(), dataset, size, output.clone());
        self.add_validatetools_step(name.clone(), output);
    }

    pub fn add_data_read_step(&mut self, name: String, dataset: String, output: String) {
        debug!("Added data read on dataset: {}", &dataset);
        self.steps.push(StepType::DataSampler(DataSamplerStep::new(
            name, dataset, None, output,
        )));
    }

    pub fn add_chunk_step(
        &mut self,
        name: String,
        capacity: (usize, usize),
        input: String,
        output: String,
    ) {
        debug!("Added data chunking step");
        self.steps.push(StepType::Chunk(ChunkStep::new(
            name, capacity, input, output,
        )));
    }

    pub fn add_render_step(&mut self, name: String, template: String, output: String) {
        debug!("Added render step");
        self.steps
            .push(StepType::Render(RenderStep::new(name, template, output)));
    }

    pub fn add_render_conversation_step(
        &mut self,
        name: String,
        conversation: String,
        tools: Option<String>,
        output: String,
    ) {
        debug!("Added render conversation step");
        self.steps
            .push(StepType::RenderConversation(RenderConversationStep::new(
                name,
                conversation,
                tools,
                output,
            )));
    }

    pub fn add_validatejson_step(&mut self, name: String, schema: String, instance: String) {
        debug!("Added validate JSON step");

        let schema_key = format!("validatejson_schema_{}_{}", name, schema);
        let instance_key = format!("validatejson_instance_{}_{}", name, instance);
        self.templates.add(
            schema_key.clone(),
            format!("{{{{{}|tojson}}}}", schema.clone()),
        );
        self.templates.add(
            instance_key.clone(),
            format!("{{{{{}|tojson}}}}", instance.clone()),
        );
        self.steps
            .push(StepType::ValidateJson(ValidateJsonStep::new(
                name,
                schema_key,
                instance_key,
            )));
    }

    pub fn add_validatetools_step(&mut self, name: String, instances: String) {
        debug!("Added validate tools step");

        self.steps
            .push(StepType::ValidateTools(ToolsValidateStep::new(
                name, instances,
            )));
    }

    pub fn add_normalizetools_step(&mut self, name: String, instances: String, output: String) {
        debug!("Added normalize tools step");

        self.steps
            .push(StepType::NormalizeTools(ToolsNormalizeStep::new(
                name, instances, output,
            )));
    }

    pub fn compile(&self) {
        self.templates.compile().unwrap();
    }

    #[pyo3(signature = (level=None, target=None, file=None))]
    pub fn log(&self, level: Option<&str>, target: Option<&str>, file: Option<&str>) {
        let level = match level {
            Some("debug") => log::LevelFilter::Debug,
            Some("info") => log::LevelFilter::Info,
            Some("warn") => log::LevelFilter::Warn,
            Some("error") => log::LevelFilter::Error,
            _ => log::LevelFilter::Info,
        };

        let config = ConfigBuilder::new()
            .set_level_color(Level::Error, Some(Color::Rgb(191, 0, 0)))
            .set_level_color(Level::Warn, Some(Color::Rgb(255, 127, 0)))
            .set_level_color(Level::Info, Some(Color::Rgb(192, 192, 0)))
            .set_level_color(Level::Debug, Some(Color::Rgb(63, 127, 0)))
            .set_level_color(Level::Trace, Some(Color::Rgb(127, 127, 255)))
            .build();

        if level == log::LevelFilter::Error {
            if let Err(e) = CombinedLogger::init(vec![
                Box::new((*self.logs_collector).clone()) as Box<dyn simplelog::SharedLogger>
            ]) {
                debug!("Initialize logger issue: {}", e);
            }
        } else {
            create_dir_all(".tweaktune").unwrap();

            let now = Local::now();
            let filename = if let Some(f) = file {
                format!(
                    ".tweaktune/log_{}_{}.log",
                    f,
                    now.format("%Y-%m-%d_%H-%M-%S")
                )
            } else {
                format!(".tweaktune/log_{}.log", now.format("%Y-%m-%d_%H-%M-%S"))
            };

            if let Err(e) = CombinedLogger::init(vec![
                WriteLogger::new(level, config.clone(), File::create(&filename).unwrap()),
                Box::new((*self.logs_collector).clone()) as Box<dyn simplelog::SharedLogger>,
            ]) {
                debug!("Initialize logger issue: {}", e);
            } else {
                println!("📑 LOGGING INTO FILE {}", &filename);
            }
        }

        // env_logger::builder().filter(target, level).init();
    }

    pub fn stop(&self) -> PyResult<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    #[pyo3(signature = (bus=None))]
    pub fn run(&self, bus: Option<PyObject>) -> PyResult<()> {
        self.running.store(true, Ordering::SeqCst);
        let r = self.running.clone();
        match ctrlc::set_handler(move || {
            r.store(false, std::sync::atomic::Ordering::SeqCst);
        }) {
            Ok(_) => {
                debug!("Ctrl-C handler set");
            }
            Err(e) => {
                debug!("Error setting Ctrl-C handler: {}", e);
            }
        }

        let sender = if let Some(bus) = bus {
            let bus_logger = Python::with_gil(|py| {
                let py_obj: PyObject = bus.clone_ref(py);
                py_obj
            });

            let (log_sender, log_receiver) = mpsc::channel::<String>();
            let sender = Arc::new(log_sender);
            let channel_writer = ChannelWriter::new(sender.clone());

            WriteLogger::init(
                log::LevelFilter::Info,
                ConfigBuilder::new().build(),
                channel_writer,
            )
            .unwrap();

            thread::spawn(move || {
                for message in log_receiver {
                    Python::with_gil(|py| {
                        bus_logger.call_method1(py, "put", (message,)).unwrap();
                    });
                }
            });

            Some(sender.clone())
        } else {
            None
        };

        let result = run_async(async {
            let successfull_iterations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            match &self.iter_by {
                IterBy::Range { start, stop, step } => {
                    debug!("Iterating by range: {}..{}..{}", start, stop, step);
                    let bar = ProgressBar::new((stop - start) as u64);

                    bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",)
                    .unwrap().progress_chars("#>-"));

                    let iter_results = stream::iter((*start..*stop).step_by(*step).map(|i| {
                        let bar = &bar;
                        if !self.running.load(std::sync::atomic::Ordering::SeqCst) {
                            bar.finish_with_message("Interrupted");
                            std::process::exit(1);
                        }

                        let sender = sender.clone();
                        let value = successfull_iterations.clone();
                        async move {
                            let mut context = StepContext::new();
                            context.set("index", i);
                            context.set_status(StepStatus::Running);
                            if let Err(e) = process_steps(self, context, None).await {
                                return Err(format!("Error processing step: {} - {}", i, e));
                            } else {
                                value.fetch_add(1, Ordering::SeqCst);
                            }

                            bar.inc(1);

                            if let Some(sender) = &sender {
                                sender
                                    .send(BusEvent::build(
                                        "progress",
                                        json!({"index": i, "total": (stop - start) / step}),
                                    ))
                                    .unwrap();
                            }
                            Ok(())
                        }
                    }))
                    .buffered(self.workers)
                    .collect::<Vec<_>>()
                    .await;

                    for result in iter_results {
                        if let Err(e) = result {
                            bail!(e)
                        }
                    }
                }
                IterBy::Dataset { name } => {
                    debug!("Iterating by dataset: {}", name);
                    let bar = ProgressBar::new(0);

                    bar.set_style(
                        ProgressStyle::with_template(
                            "{spinner:.green} [{elapsed_precise}] ({pos})",
                        )
                        .unwrap(),
                    );

                    let dataset = self.datasets.get(name).ok_or_err(name)?;
                    let mut inc = 0;
                    // macros to reduce duplicated iteration logic for datasets
                    macro_rules! process_dataset {
                        ($dataset:expr) => {{
                            let iter_results = stream::iter($dataset.stream()?.map(|json_row| {
                                let bar = &bar;
                                let sender = sender.clone();
                                process_progress_bar(bar, &self.running);
                                let value = successfull_iterations.clone();
                                async move {
                                    if let Err(e) =
                                        map_record_batches(self, name, &json_row.unwrap()).await
                                    {
                                        return Err(format!(
                                            "Error processing step: {} - {}",
                                            name, e
                                        ));
                                    } else {
                                        value.fetch_add(1, Ordering::SeqCst);
                                    }
                                    bar.inc(1);
                                    inc += 1;
                                    send_progress_event(&sender, inc);
                                    Ok(())
                                }
                            }))
                            .buffered(self.workers)
                            .collect::<Vec<_>>()
                            .await;
                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        }};
                    }

                    macro_rules! process_dataset_mix {
                        ($dataset:expr) => {{
                            let iter_results =
                                stream::iter($dataset.stream_mix(&self.datasets.resources)?.map(
                                    |json_row| {
                                        let bar = &bar;
                                        let sender = sender.clone();
                                        process_progress_bar(bar, &self.running);
                                        let value = successfull_iterations.clone();
                                        async move {
                                            if let Err(e) =
                                                map_record_batches(self, name, &json_row.unwrap())
                                                    .await
                                            {
                                                return Err(format!(
                                                    "Error processing step: {} - {}",
                                                    name, e
                                                ));
                                            } else {
                                                value.fetch_add(1, Ordering::SeqCst);
                                            }
                                            bar.inc(1);
                                            inc += 1;
                                            send_progress_event(&sender, inc);
                                            Ok(())
                                        }
                                    },
                                ))
                                .buffered(self.workers)
                                .collect::<Vec<_>>()
                                .await;
                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        }};
                    }
                    match dataset {
                        DatasetType::Jsonl(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::Json(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::JsonList(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::OpenApi(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::Polars(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::Ipc(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::Csv(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::Parquet(dataset) => {
                            process_dataset!(dataset)
                        }

                        DatasetType::Mixed(dataset) => {
                            process_dataset_mix!(dataset)
                        }
                    }
                }
            }

            info!(
                "🚀 Finished all iterations, processed {} items",
                successfull_iterations.load(Ordering::SeqCst)
            );

            if let Some(sender) = &sender {
                sender
                    .send(BusEvent::build("finished", json!({"message": "Finished"})))
                    .unwrap();
            }

            Ok::<_, anyhow::Error>(())
        });

        println!("{}", self.logs_collector.summary_table());

        result.map_pyerr()
    }
}

fn send_progress_event(sender: &Option<Arc<mpsc::Sender<String>>>, inc: i32) {
    if let Some(sender) = sender {
        let event = BusEvent::build("progress", json!({"inc": inc,}));
        if let Err(e) = sender.send(event) {
            error!("Failed to send progress event: {}", e);
        }
    }
}

fn process_progress_bar(bar: &ProgressBar, running: &Arc<AtomicBool>) {
    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        bar.finish_with_message("Interrupted");
        std::process::exit(1);
    }
    bar.inc_length(1);
}

async fn map_record_batches(
    pipeline: &PipelineBuilder,
    dataset_name: &str,
    json_row: &serde_json::Value,
) -> Result<()> {
    let mut context = StepContext::new();

    context.set(dataset_name, json_row);
    context.set_status(StepStatus::Running);
    process_steps(pipeline, context, None).await?;
    Ok(())
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

async fn process_steps(
    pipeline: &PipelineBuilder,
    mut context: StepContext,
    steps: Option<&Vec<StepType>>,
) -> Result<StepContext> {
    let steps = if let Some(steps) = steps {
        steps
    } else {
        &pipeline.steps
    };

    for step in steps {
        if matches!(context.get_status(), StepStatus::Failed) {
            break;
        }

        match step {
            StepType::IfElse(if_step) => {
                let check_result = if_step
                    .check(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;

                if check_result {
                    context = Box::pin(process_steps(
                        pipeline,
                        context.clone(),
                        Some(&if_step.then_steps),
                    ))
                    .await?;
                } else if let Some(else_steps) = &if_step.else_steps {
                    context = Box::pin(process_steps(pipeline, context.clone(), Some(else_steps)))
                        .await?;
                }
            }
            StepType::Py(py_step) => {
                context = py_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::TextGeneration(text_generation_step) => {
                context = text_generation_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::JsonGeneration(json_generation_step) => {
                context = json_generation_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::PyValidator(py_validator) => {
                context = py_validator
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::JsonWriter(jsonl_writer_step) => {
                context = jsonl_writer_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::CsvWriter(csv_writer_step) => {
                context = csv_writer_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::Print(print_step) => {
                context = print_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::DataSampler(data_sampler_step) => {
                context = data_sampler_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::Chunk(chunk_step) => {
                context = chunk_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::Render(render_step) => {
                context = render_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::ValidateJson(validate_json_step) => {
                context = validate_json_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::ValidateTools(tools_validate_step) => {
                context = tools_validate_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::NormalizeTools(tools_normalize_step) => {
                context = tools_normalize_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::ConversationValidate(conversation_validate_step) => {
                context = conversation_validate_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::IntoList(into_list_step) => {
                context = into_list_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
            StepType::RenderConversation(render_conversation_step) => {
                context = render_conversation_step
                    .process(
                        &pipeline.datasets.resources,
                        &pipeline.templates,
                        &pipeline.llms.resources,
                        &pipeline.embeddings.resources,
                        &context,
                    )
                    .await?;
            }
        }
    }

    Ok(context)
}

#[pyclass]
#[derive(Debug)]
pub enum IterBy {
    Range {
        start: usize,
        stop: usize,
        step: usize,
    },
    Dataset {
        name: String,
    },
}

#[pyclass]
#[derive(Debug)]
pub enum Template {
    Jinja { name: String, template: String },
}

#[pyclass]
#[derive(Debug)]
pub enum LLM {
    OpenAI {
        name: String,
        model: String,
        base_url: String,
        api_key: String,
        max_tokens: u32,
    },
}

#[pyclass]
#[derive(Debug)]
pub enum Embeddings {
    OpenAI {
        name: String,
        model: String,
        base_url: String,
        api_key: String,
    },
}

#[pyclass]
#[derive(Debug)]
pub struct StepsChain {
    steps: Vec<Step>,
}

#[pymethods]
impl StepsChain {
    #[new]
    pub fn new() -> Self {
        StepsChain { steps: Vec::new() }
    }

    pub fn add_py_step(&mut self, named: String, py_func: PyObject) {
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

    pub fn add_judge_step(&mut self, name: String, template: String, llm: String) {
        debug!("Added judge step: {}", &name);
        self.steps.push(Step::Judge {
            name,
            template,
            llm,
        });
    }

    pub fn add_py_validator_step(&mut self, name: String, py_func: PyObject) {
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
        py_func: PyObject,
    },
    TextGeneration {
        name: String,
        template: String,
        llm: String,
        output: String,
        system_template: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
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
        py_func: PyObject,
    },
    JsonlWriter {
        name: String,
        path: String,
        template: String,
    },
}

#[pyclass]
#[derive(Debug)]
pub enum Dataset {
    Jsonl {
        name: String,
        path: String,
    },
    Parquet {
        name: String,
        path: String,
    },
    Arrow {
        name: String,
        dataset: PyObject,
    },
    Csv {
        name: String,
        path: String,
        delimiter: String,
        has_header: bool,
    },
}

#[derive(Default, Clone)]
pub struct Resources<T> {
    resources: HashMap<String, T>,
}

impl<T> Resources<T> {
    pub fn add(&mut self, name: String, resource: T) {
        self.resources.insert(name, resource);
    }

    pub fn list(&self) -> Vec<String> {
        self.resources.keys().cloned().collect()
    }

    pub fn get(&self, name: &str) -> Option<&T> {
        self.resources.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<T> {
        self.resources.remove(name)
    }
}

fn map_step(step: &Step, templates: &mut Templates) -> StepType {
    match step {
        Step::Py { name, py_func } => Python::with_gil(|py| {
            let py_obj: PyObject = py_func.clone_ref(py);
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
        } => StepType::TextGeneration(TextGenerationStep::new(
            name.clone(),
            template.clone(),
            llm.clone(),
            output.clone(),
            system_template.clone(),
            *max_tokens,
            *temperature,
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
            schema_template,
        } => {
            let schema_key = if let Some(schema) = &schema_template {
                let schema_key = format!("json_generation_step_{}_{}", name, schema);
                templates.add(schema_key.clone(), format!("{{{{{}}}}}", schema.clone()));
                Some(schema_key)
            } else {
                None
            };

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
