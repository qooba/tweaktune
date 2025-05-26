use crate::common::ResultExt;
use anyhow::{bail, Result};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use pyo3::{pyclass, pymethods, PyObject, PyResult};
use std::sync::atomic::AtomicBool;
use std::{collections::HashMap, sync::Arc};
use tokio::runtime::Runtime;
use tweaktune_core::datasets::{
    CsvDataset, Dataset as DatasetTrait, IpcDataset, JsonlDataset, MixedDataset, ParquetDataset,
    PolarsDataset,
};
use tweaktune_core::llms::{MistralrsLLM, UnslothLLM};
use tweaktune_core::steps::{ChunkStep, RenderStep, ValidateJsonStep};
use tweaktune_core::{
    common::OptionToResult,
    datasets::{DatasetType, JsonDataset, JsonListDataset, OpenApiDataset},
    embeddings::{EmbeddingsType, OpenAIEmbeddings},
    llms::{LLMType, OpenAILLM},
    steps::{
        CsvWriterStep, DataSamplerStep, JsonGenerationStep, JsonlWriterStep, PrintStep, PyStep,
        PyValidator, Step as StepCore, StepContext, StepStatus, StepType, TextGenerationStep,
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
    pub fn with_jsonl_dataset(&mut self, name: String, path: String, sql: Option<String>) -> PyResult<()> {
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
    pub fn with_json_dataset(&mut self, name: String, path: String, sql: Option<String>) -> PyResult<()> {
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
            DatasetType::Mixed(
                MixedDataset::new(name, datasets, &self.datasets.resources)?,
            ),
        );
        Ok(())
    }

    #[pyo3(signature = (name, path, sql=None))]
    pub fn with_parquet_dataset(&mut self, name: String, path: String, sql: Option<String>) -> PyResult<()> {
        debug!("Added Parquet dataset: {}", &name);
        self.datasets.add(
            name.clone(),
            DatasetType::Parquet(ParquetDataset::new(name, path, sql)?),
        );
        Ok(())
    }

    #[pyo3(signature = (name, ipc_data, sql=None))]
    pub fn with_ipc_dataset(&mut self, name: String, ipc_data: &[u8], sql: Option<String>) -> PyResult<()> {
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
            DatasetType::Csv(
                CsvDataset::new(name, path, delimiter.as_bytes()[0], has_header, sql)?,
            ),
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
            LLMType::OpenAI(OpenAILLM::new(
                name,
                base_url,
                api_key,
                model,
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

    pub fn add_py_validator_step(&mut self, name: String, py_func: PyObject) {
        debug!("Added Python validator step: {}", &name);
        self.steps
            .push(StepType::PyValidator(PyValidator::new(name, py_func)));
    }

    #[pyo3(signature = (name, template, llm, output, system_template=None))]
    pub fn add_text_generation_step(
        &mut self,
        name: String,
        template: String,
        llm: String,
        output: String,
        system_template: Option<String>,
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
            )));
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, template, llm, output, json_path=None, system_template=None, json_schema=None))]
    pub fn add_json_generation_step(
        &mut self,
        name: String,
        template: String,
        llm: String,
        output: String,
        json_path: Option<String>,
        system_template: Option<String>,
        json_schema: Option<String>,
    ) {
        debug!(
            "Added JSON generation step with template: {}, llm: {}",
            &llm, &template
        );
        self.steps
            .push(StepType::JsonGeneration(JsonGenerationStep::new(
                name,
                template,
                llm,
                output,
                json_path,
                system_template,
                json_schema,
            )));
    }

    pub fn add_write_jsonl_step(&mut self, name: String, path: String, template: String) {
        debug!("Added JSONL writer step: {}", &name);
        self.steps.push(StepType::JsonWriter(JsonlWriterStep::new(
            name, path, template,
        )));
    }

    #[pyo3(signature = (name, template=None, columns=None))]
    pub fn add_print_step(
        &mut self,
        name: String,
        template: Option<String>,
        columns: Option<Vec<String>>,
    ) {
        debug!("Added print step");
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

    pub fn add_validatejson_step(&mut self, name: String, schema: String, instance: String) {
        debug!("Added render step");

        let schema_key  = format!("validatejson_schema_{}_{}", name, schema);
        let instance_key  = format!("validatejson_instance_{}_{}", name, instance);
        self.templates.add(schema_key.clone(), format!("{{{{{}}}}}",schema.clone()));
        self.templates.add(instance_key.clone(), format!("{{{{{}}}}}",instance.clone()));
        self.steps
            .push(StepType::ValidateJson(ValidateJsonStep::new(name, schema_key, instance_key)));
    }


    pub fn compile(&self) {
        self.templates.compile().unwrap();
    }

    #[pyo3(signature = (level=None, target=None))]
    pub fn log(&self, level: Option<&str>, target: Option<&str>) {
        let level = match level {
            Some("debug") => log::LevelFilter::Debug,
            Some("info") => log::LevelFilter::Info,
            Some("warn") => log::LevelFilter::Warn,
            Some("error") => log::LevelFilter::Error,
            _ => log::LevelFilter::Info,
        };

        env_logger::builder().filter(target, level).init();
    }

    pub fn run(&self) -> PyResult<()> {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
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

        let result = Runtime::new()?.block_on(async {
            match &self.iter_by {
                IterBy::Range { start, stop, step } => {
                    debug!("Iterating by range: {}..{}..{}", start, stop, step);
                    let bar = ProgressBar::new((stop - start) as u64);

                     bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",)
                    .unwrap().progress_chars("#>-"));

                    let iter_results = stream::iter((*start..*stop).step_by(*step).map(|i| {
                        let bar = &bar;
                        if !running.load(std::sync::atomic::Ordering::SeqCst) {
                            bar.finish_with_message("Interrupted");
                            std::process::exit(1);
                        }

                        async move {

                            let mut context = StepContext::new();
                            context.set("index", i);
                            context.set_status(StepStatus::Running);
                            if let Err(e) = process_steps(self, context).await {
                                return Err(format!("Error processing step: {} - {}", i ,e));
                            }
                            bar.inc(1);
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

                    bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] ({pos})",).unwrap());

                    let dataset = self.datasets.get(name).ok_or_err(name)?;
                    match dataset {
                        DatasetType::Jsonl(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;
                    
                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        }

                        DatasetType::Json(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;
                    
                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        },

                        DatasetType::JsonList(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;
                    
                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        },

                        DatasetType::OpenApi(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;
                    
                             for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        },

                        DatasetType::Polars(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;
                    
                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        },

                        DatasetType::Ipc(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;

                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        },

                        DatasetType::Csv(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;

                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        },

                        DatasetType::Parquet(dataset) => {
                            let iter_results = stream::iter(dataset.stream()?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;

                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                        },

                        DatasetType::Mixed(dataset) => {
                            let iter_results = stream::iter(dataset.stream_mix(&self.datasets.resources)?.map(|json_row|{
                                let bar= &bar;
                                process_progress_bar(bar, &running);
                                async move {
                                    if let Err(e) = map_record_batches(self,name, &json_row.unwrap()).await {
                                        return Err(format!("Error processing step: {} - {}", name ,e));
                                    }
                                    bar.inc(1);
                                    Ok(())
                            }},)).buffered(self.workers).collect:: <Vec<_> >().await;

                            for result in iter_results {
                                if let Err(e) = result {
                                    bail!(e)
                                }
                            }
                    }

                    }
                }
            }

            Ok::<_, anyhow::Error>(())
        });

        result.map_pyerr()
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
    process_steps(pipeline, context).await?;
    Ok(())
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

async fn process_steps(pipeline: &PipelineBuilder, mut context: StepContext) -> Result<()> {
    for step in &pipeline.steps {
        if matches!(context.get_status(), StepStatus::Failed) {
            break;
        }

        match step {
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
            
        }
    }

    Ok(())
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
    },
    DataSampler {
        name: String,
        dataset: String,
        size: usize,
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
