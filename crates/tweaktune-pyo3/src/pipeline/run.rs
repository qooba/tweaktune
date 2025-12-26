use crate::common::ResultExt;
use crate::logging::{BusEvent, ChannelWriter};
use crate::pipeline::error_tracker::ErrorTracker;
use crate::pipeline::{IterBy, PipelineBuilder};
use anyhow::{bail, Result};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info};
use pyo3::{types::PyAny, Py, PyResult, Python};
use serde_json::json;
use simplelog::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use tweaktune_core::common::{run_async, OptionToResult};
use tweaktune_core::datasets::{Dataset as DatasetTrait, DatasetType};
use tweaktune_core::steps::{Step as StepTrait, StepContext, StepStatus, StepType};

impl PipelineBuilder {
    pub(crate) fn run_internal(&self, bus: Option<Py<PyAny>>) -> PyResult<()> {
        // Print TweakTune logo
        println!("\n{}", Self::get_logo());

        // Print pipeline summary
        println!("{}", self.get_summary());

        // Pre-flight validation
        println!("Running pre-flight validation...");
        let validation_result = self.validate();
        if !validation_result.is_valid() {
            println!("{}", validation_result.generate_report());
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Pre-flight validation failed. See details above.",
            ));
        }
        println!("✓ Pre-flight validation passed\n");

        // Initialize error tracker with adaptive threshold
        // For small iteration counts, fail fast (threshold = 1)
        // For larger counts, allow more retries (threshold = 3)
        let max_consecutive = match &self.iter_by {
            IterBy::Range { start, stop, step } => {
                let total_iterations = (stop - start) / step;
                if total_iterations < 3 {
                    1 // Fail fast for small iteration counts
                } else {
                    3 // Allow retries for larger iteration counts
                }
            }
            IterBy::Dataset { .. } => 3, // Unknown size, use default
        };
        let error_tracker = ErrorTracker::new(max_consecutive);

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
            let bus_logger = Python::attach(|py| {
                let py_obj: Py<PyAny> = bus.clone_ref(py);
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
                    Python::attach(|py| {
                        bus_logger.call_method1(py, "put", (message,)).unwrap();
                    });
                }
            });

            Some(sender.clone())
        } else {
            None
        };

        let log_path = self.log_path.clone();

        let result = run_async(async {
            if self.metadata.enabled {
                if let Some(state) = &self.resources.state {
                    state
                        .add_run(
                            &self.id.to_string(),
                            log_path.as_ref().expect("Log path not set"),
                            None,
                        )
                        .await?;
                }
            }

            let successfull_iterations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            match &self.iter_by {
                IterBy::Range { start, stop, step } => {
                    debug!("Iterating by range: {}..{}..{}", start, stop, step);
                    let bar = ProgressBar::new((stop - start) as u64);

                    bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",)
                    .unwrap().progress_chars("#>-"));
                    bar.tick(); // Show progress bar immediately

                    let iter_results = stream::iter((*start..*stop).step_by(*step).map(|i| {
                        let bar = &bar;
                        if !self.running.load(std::sync::atomic::Ordering::SeqCst) {
                            bar.finish_with_message("Interrupted");
                            std::process::exit(1);
                        }

                        let sender = sender.clone();
                        let value = successfull_iterations.clone();
                        let rid = self.id.to_string();
                        let tracker = error_tracker.clone();
                        async move {
                            let mut context = StepContext::new();
                            context.set("index", i);
                            context.set_status(StepStatus::Running);
                            let item_id = context.id.to_string();
                            if self.metadata.enabled {
                                if let Some(state) = &self.resources.state {
                                    state
                                        .add_item(&item_id, &rid, i as i64, None)
                                        .await
                                        .unwrap();
                                }
                            }

                            match process_steps(self, context, None, &tracker, i).await {
                                Ok(_) => {
                                    value.fetch_add(1, Ordering::SeqCst);
                                    tracker.reset_consecutive();

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
                                Err(e) => {
                                    if let Some(state) = &self.resources.state {
                                        state.delete_item(&item_id).await.ok();
                                    }

                                    // Check if we should stop early
                                    if let Some((step_name, error_msg, count)) =
                                        tracker.should_stop()
                                    {
                                        bar.finish_with_message(
                                            "Stopped due to configuration error",
                                        );
                                        return Err(format!(
                                            "CONFIGURATION_ERROR|||{}|||{}|||{}",
                                            step_name, error_msg, count
                                        ));
                                    }

                                    Err(format!("Error processing step: {} - {}", i, e))
                                }
                            }
                        }
                    }))
                    .buffered(self.workers)
                    .collect::<Vec<_>>()
                    .await;

                    for result in iter_results {
                        if let Err(e) = result {
                            // Check if it's a configuration error
                            if e.starts_with("CONFIGURATION_ERROR|||") {
                                let parts: Vec<&str> = e.split("|||").collect();
                                if parts.len() >= 4 {
                                    let step_name = parts[1];
                                    let error_msg = parts[2];
                                    let count: usize = parts[3].parse().unwrap_or(0);
                                    println!(
                                        "{}",
                                        error_tracker.generate_config_error_alert(
                                            step_name, error_msg, count
                                        )
                                    );
                                }
                            }
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
                    bar.tick(); // Show progress bar immediately

                    let dataset = self.resources.datasets.get(name).ok_or_err(name)?;
                    let mut inc = 0;
                    // macros to reduce duplicated iteration logic for datasets
                    macro_rules! process_dataset {
                        ($dataset:expr) => {{
                            let iter_results = stream::iter($dataset.stream()?.map(|json_row| {
                                let bar = &bar;
                                let sender = sender.clone();
                                process_progress_bar(bar, &self.running);
                                let value = successfull_iterations.clone();
                                let tracker = error_tracker.clone();
                                let current_inc = inc;
                                async move {
                                    match map_record_batches(
                                        self,
                                        name,
                                        &json_row.unwrap(),
                                        &current_inc,
                                        &tracker,
                                        current_inc as usize,
                                    )
                                    .await
                                    {
                                        Ok(_) => {
                                            value.fetch_add(1, Ordering::SeqCst);
                                            tracker.reset_consecutive();
                                            bar.inc(1);
                                            send_progress_event(&sender, current_inc);
                                            Ok(())
                                        }
                                        Err(e) => {
                                            // Check if we should stop early
                                            if let Some((step_name, error_msg, count)) =
                                                tracker.should_stop()
                                            {
                                                bar.finish_with_message(
                                                    "Stopped due to configuration error",
                                                );
                                                return Err(format!(
                                                    "CONFIGURATION_ERROR|||{}|||{}|||{}",
                                                    step_name, error_msg, count
                                                ));
                                            }
                                            Err(format!("Error processing step: {} - {}", name, e))
                                        }
                                    }
                                }
                            }))
                            .buffered(self.workers)
                            .collect::<Vec<_>>()
                            .await;
                            for result in iter_results {
                                if let Err(e) = result {
                                    // Check if it's a configuration error
                                    if e.starts_with("CONFIGURATION_ERROR|||") {
                                        let parts: Vec<&str> = e.split("|||").collect();
                                        if parts.len() >= 4 {
                                            let step_name = parts[1];
                                            let error_msg = parts[2];
                                            let count: usize = parts[3].parse().unwrap_or(0);
                                            println!(
                                                "{}",
                                                error_tracker.generate_config_error_alert(
                                                    step_name, error_msg, count
                                                )
                                            );
                                        }
                                    }
                                    bail!(e)
                                }
                                inc += 1;
                            }
                        }};
                    }

                    macro_rules! process_dataset_mix {
                        ($dataset:expr) => {{
                            let iter_results = stream::iter(
                                $dataset
                                    .stream_mix(&self.resources.datasets.resources)?
                                    .map(|json_row| {
                                        let bar = &bar;
                                        let sender = sender.clone();
                                        process_progress_bar(bar, &self.running);
                                        let value = successfull_iterations.clone();
                                        let tracker = error_tracker.clone();
                                        let current_inc = inc;
                                        async move {
                                            match map_record_batches(
                                                self,
                                                name,
                                                &json_row.unwrap(),
                                                &current_inc,
                                                &tracker,
                                                current_inc as usize,
                                            )
                                            .await
                                            {
                                                Ok(_) => {
                                                    value.fetch_add(1, Ordering::SeqCst);
                                                    tracker.reset_consecutive();
                                                    bar.inc(1);
                                                    send_progress_event(&sender, current_inc);
                                                    Ok(())
                                                }
                                                Err(e) => {
                                                    // Check if we should stop early
                                                    if let Some((step_name, error_msg, count)) =
                                                        tracker.should_stop()
                                                    {
                                                        bar.finish_with_message(
                                                            "Stopped due to configuration error",
                                                        );
                                                        return Err(format!(
                                                            "CONFIGURATION_ERROR|||{}|||{}|||{}",
                                                            step_name, error_msg, count
                                                        ));
                                                    }
                                                    Err(format!(
                                                        "Error processing step: {} - {}",
                                                        name, e
                                                    ))
                                                }
                                            }
                                        }
                                    }),
                            )
                            .buffered(self.workers)
                            .collect::<Vec<_>>()
                            .await;
                            for result in iter_results {
                                if let Err(e) = result {
                                    // Check if it's a configuration error
                                    if e.starts_with("CONFIGURATION_ERROR|||") {
                                        let parts: Vec<&str> = e.split("|||").collect();
                                        if parts.len() >= 4 {
                                            let step_name = parts[1];
                                            let error_msg = parts[2];
                                            let count: usize = parts[3].parse().unwrap_or(0);
                                            println!(
                                                "{}",
                                                error_tracker.generate_config_error_alert(
                                                    step_name, error_msg, count
                                                )
                                            );
                                        }
                                    }
                                    bail!(e)
                                }
                                inc += 1;
                            }
                        }};
                    }
                    match dataset {
                        DatasetType::Jsonl(dataset) => process_dataset!(dataset),
                        DatasetType::Json(dataset) => process_dataset!(dataset),
                        DatasetType::JsonList(dataset) => process_dataset!(dataset),
                        DatasetType::OpenApi(dataset) => process_dataset!(dataset),
                        DatasetType::Polars(dataset) => process_dataset!(dataset),
                        DatasetType::Ipc(dataset) => process_dataset!(dataset),
                        DatasetType::Csv(dataset) => process_dataset!(dataset),
                        DatasetType::Parquet(dataset) => process_dataset!(dataset),
                        DatasetType::Mixed(dataset) => process_dataset_mix!(dataset),
                        DatasetType::PhfSet(phf_set_dataset) => process_dataset!(phf_set_dataset),
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

        // Display error summary if there were any errors
        if error_tracker.error_count() > 0 {
            println!("{}", error_tracker.generate_summary());
        }

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
    inc: &i32,
    error_tracker: &ErrorTracker,
    iteration: usize,
) -> Result<()> {
    let mut context = StepContext::new();

    context.set(dataset_name, json_row);
    context.set("index", inc);
    context.set_status(StepStatus::Running);
    let item_id = context.id.to_string();
    if pipeline.metadata.enabled {
        if let Some(state) = &pipeline.resources.state {
            state
                .add_item(&item_id, &pipeline.id.to_string(), *inc as i64, None)
                .await
                .unwrap();
        }
    }

    if let Err(e) = process_steps(pipeline, context, None, error_tracker, iteration).await {
        if let Some(state) = &pipeline.resources.state {
            state.delete_item(&item_id).await.ok();
        }
        return Err(e);
    }
    Ok(())
}

async fn process_steps(
    pipeline: &PipelineBuilder,
    mut context: StepContext,
    steps: Option<&Vec<StepType>>,
    error_tracker: &ErrorTracker,
    iteration: usize,
) -> Result<StepContext> {
    let steps = if let Some(steps) = steps {
        steps
    } else {
        &pipeline.steps
    };

    for (step_idx, step) in steps.iter().enumerate() {
        if matches!(context.get_status(), StepStatus::Failed) {
            break;
        }

        let step_name = format!("step-{}", step_idx);

        // macro to collapse the repeated `step.process(...).await?` pattern
        macro_rules! process_common {
            ($step_ident:ident, $step_name:expr) => {{
                match $step_ident.process(&pipeline.resources, &context).await {
                    Ok(ctx) => {
                        context = ctx;
                    }
                    Err(e) => {
                        error_tracker.track_error($step_name.clone(), e.to_string(), iteration);
                        return Err(e);
                    }
                }
            }};
        }

        match step {
            StepType::IfElse(if_step) => {
                let check_result = if_step
                    .check(
                        &pipeline.resources.datasets.resources,
                        &pipeline.resources.templates,
                        &pipeline.resources.llms.resources,
                        &pipeline.resources.embeddings.resources,
                        &context,
                    )
                    .await?;

                if check_result {
                    context = Box::pin(process_steps(
                        pipeline,
                        context.clone(),
                        Some(&if_step.then_steps),
                        error_tracker,
                        iteration,
                    ))
                    .await?;
                } else if let Some(else_steps) = &if_step.else_steps {
                    context = Box::pin(process_steps(
                        pipeline,
                        context.clone(),
                        Some(else_steps),
                        error_tracker,
                        iteration,
                    ))
                    .await?;
                }
            }
            StepType::Py(py_step) => process_common!(py_step, step_name),
            StepType::TextGeneration(text_generation_step) => {
                process_common!(text_generation_step, step_name)
            }
            StepType::JsonGeneration(json_generation_step) => {
                process_common!(json_generation_step, step_name)
            }
            StepType::PyValidator(py_validator) => process_common!(py_validator, step_name),
            StepType::JsonWriter(jsonl_writer_step) => {
                process_common!(jsonl_writer_step, step_name)
            }
            StepType::CsvWriter(csv_writer_step) => process_common!(csv_writer_step, step_name),
            StepType::Print(print_step) => process_common!(print_step, step_name),
            StepType::DataSampler(data_sampler_step) => {
                process_common!(data_sampler_step, step_name)
            }
            StepType::Chunk(chunk_step) => process_common!(chunk_step, step_name),
            StepType::Render(render_step) => process_common!(render_step, step_name),
            StepType::ValidateJson(validate_json_step) => {
                process_common!(validate_json_step, step_name)
            }
            StepType::ValidateTools(tools_validate_step) => {
                process_common!(tools_validate_step, step_name)
            }
            StepType::NormalizeTools(tools_normalize_step) => {
                process_common!(tools_normalize_step, step_name)
            }
            StepType::ConversationValidate(conversation_validate_step) => {
                process_common!(conversation_validate_step, step_name)
            }
            StepType::IntoList(into_list_step) => process_common!(into_list_step, step_name),
            StepType::RenderConversation(render_conversation_step) => {
                process_common!(render_conversation_step, step_name)
            }
            StepType::Filter(filter_step) => process_common!(filter_step, step_name),
            StepType::Mutate(mutate_step) => process_common!(mutate_step, step_name),
            StepType::CheckLanguage(check_language_step) => {
                process_common!(check_language_step, step_name)
            }
            StepType::RenderToolCall(render_tool_call_step) => {
                process_common!(render_tool_call_step, step_name)
            }
            StepType::CheckHash(check_hash_step) => process_common!(check_hash_step, step_name),
            StepType::CheckSimHash(check_sim_hash_step) => {
                process_common!(check_sim_hash_step, step_name)
            }
            StepType::CheckEmbedding(embedding_step) => process_common!(embedding_step, step_name),
            StepType::JudgeConversation(judge_conversation_step) => {
                process_common!(judge_conversation_step, step_name)
            }
            StepType::RenderDPO(render_dpostep) => process_common!(render_dpostep, step_name),
            StepType::RenderGRPO(render_grpostep) => process_common!(render_grpostep, step_name),
            StepType::CheckJson(check_json_step) => process_common!(check_json_step, step_name),
            StepType::ToolArgumentsSampler(tool_arguments_sampler_step) => {
                process_common!(tool_arguments_sampler_step, step_name)
            }
            StepType::Literal(literal_step) => process_common!(literal_step, step_name),
        }
    }

    Ok(context)
}
