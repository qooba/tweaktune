use crate::common::{hf_hub_get, hf_hub_get_multiple, hf_hub_get_path};
use crate::common::{parse_device, ResultExt};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::t5;
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

static SEQ2SEQ_INSTANCES: OnceCell<Mutex<HashMap<String, Arc<Mutex<Seq2SeqModel>>>>> =
    OnceCell::new();
const DTYPE: DType = DType::F32;

#[derive(Clone, Debug, Copy)]
pub enum Which {
    T5Base,
    T5Small,
    T5Large,
    T5_3B,
    Mt5Base,
    Mt5Small,
    Mt5Large,
}

#[derive(Clone, Debug)]
pub struct Seq2SeqSpec {
    pub name: String,
    pub device: Option<String>,
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub model_file: Option<String>,
    pub tokenizer_file: Option<String>,
    pub config_file: Option<String>,
    pub decode: bool,
    pub disable_cache: bool,
    pub prompt: Option<String>,
    pub decoder_prompt: Option<String>,
    pub hf_token: Option<String>,
    pub normalize_embeddings: bool,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub which: Which,
}

pub struct Seq2SeqModel {
    pub spec: Seq2SeqSpec,
    pub model: t5::T5ForConditionalGeneration,
    pub tokenizer: Tokenizer,
    pub config: t5::Config,
    pub device: Device,
}

impl Seq2SeqModel {
    pub fn load(spec: Seq2SeqSpec) -> Result<Seq2SeqModel> {
        let sp = spec.clone();
        let device = parse_device(spec.device)?;
        let (default_model, default_revision) = match spec.which {
            Which::T5Base => ("t5-base", "main"),
            Which::T5Small => ("t5-small", "refs/pr/15"),
            Which::T5Large => ("t5-large", "main"),
            Which::T5_3B => ("t5-3b", "main"),
            Which::Mt5Base => ("google/mt5-base", "refs/pr/5"),
            Which::Mt5Small => ("google/mt5-small", "refs/pr/6"),
            Which::Mt5Large => ("google/mt5-large", "refs/pr/2"),
        };
        let default_model = default_model.to_string();
        let default_revision = default_revision.to_string();
        let (model_id, revision) = match (spec.model_id.to_owned(), spec.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let _candle_config = hf_hub_get(
            &model_id,
            "config.json",
            spec.hf_token.clone(),
            Some(revision),
        )?;

        let (tokenizer_repo_id, tokenizer_file) = match &spec.tokenizer_file {
            None => match spec.which {
                Which::Mt5Base => ("lmz/mt5-tokenizers", "mt5-base.tokenizer.json"),
                Which::Mt5Small => ("lmz/mt5-tokenizers", "mt5-small.tokenizer.json"),
                Which::Mt5Large => ("lmz/mt5-tokenizers", "mt5-large.tokenizer.json"),
                _ => (model_id.as_str(), "tokenizer.json"),
            },
            Some(f) => (model_id.as_str(), f.as_str()),
        };

        let _tokenizer = hf_hub_get(
            tokenizer_repo_id,
            tokenizer_file,
            spec.hf_token.clone(),
            None,
        )?;
        let weights_filename = match &spec.model_file {
            Some(f) => f.split(',').map(|v| v.into()).collect::<Vec<_>>(),
            None => {
                if model_id == "google/flan-t5-xxl" || model_id == "google/flan-ul2" {
                    hf_hub_get_multiple(
                        &model_id,
                        "model.safetensors.index.json",
                        spec.hf_token.clone(),
                        None,
                    )?
                } else {
                    vec![hf_hub_get_path(
                        &model_id,
                        "model.safetensors",
                        spec.hf_token.clone(),
                        None,
                    )?]
                }
            }
        };
        let mut config: t5::Config = serde_json::from_slice(&_candle_config)?;
        config.use_cache = !spec.disable_cache;
        let mut tokenizer = Tokenizer::from_bytes(_tokenizer).map_err(E::msg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_filename, DTYPE, &device)? };
        let model = t5::T5ForConditionalGeneration::load(vb, &config)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(E::msg)?;

        Ok(Self {
            device,
            config,
            model,
            spec: sp,
            tokenizer,
        })
    }

    pub fn lazy(spec: Seq2SeqSpec) -> Result<Arc<Mutex<Seq2SeqModel>>> {
        let name = spec.name.clone();

        if SEQ2SEQ_INSTANCES.get().is_none() {
            let _ = SEQ2SEQ_INSTANCES.set(Mutex::new(HashMap::new()));
        }

        let map = SEQ2SEQ_INSTANCES.get().expect("SEQ2SEQ_INSTANCES");
        {
            let guard = map.lock().map_anyhow_err()?;
            if let Some(existing) = guard.get(&name) {
                return Ok(existing.clone());
            }
        }

        let seq2seq_model = Seq2SeqModel::load(spec)?;
        let arc = Arc::new(Mutex::new(seq2seq_model));
        let mut guard = map.lock().map_anyhow_err()?;
        guard.insert(name, arc.clone());
        Ok(arc)
    }

    pub fn forward(&self, prompt: String, decoder_prompt: Option<String>) -> Result<String> {
        let mut model = self.model.clone();

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let input_token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let mut output_token_ids = [self
            .config
            .decoder_start_token_id
            .unwrap_or(self.config.pad_token_id) as u32]
        .to_vec();
        if let Some(decoder_prompt) = &decoder_prompt {
            print!("{decoder_prompt}");
            output_token_ids.extend(
                self.tokenizer
                    .encode(decoder_prompt.to_string(), false)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec(),
            );
        }
        let temperature = if self.spec.temperature <= 0. {
            None
        } else {
            Some(self.spec.temperature)
        };
        let mut logits_processor = LogitsProcessor::new(299792458, temperature, self.spec.top_p);
        let encoder_output = model.encode(&input_token_ids)?;

        for index in 0.. {
            if output_token_ids.len() > 512 {
                break;
            }
            let decoder_token_ids = if index == 0 || !self.config.use_cache {
                Tensor::new(output_token_ids.as_slice(), &self.device)?.unsqueeze(0)?
            } else {
                let last_token = *output_token_ids.last().unwrap();
                Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?
            };
            let logits = model
                .decode(&decoder_token_ids, &encoder_output)?
                .squeeze(0)?;
            let logits = if self.spec.repeat_penalty == 1. {
                logits
            } else {
                let start_at = output_token_ids
                    .len()
                    .saturating_sub(self.spec.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.spec.repeat_penalty,
                    &output_token_ids[start_at..],
                )?
            };

            let next_token_id = logits_processor.sample(&logits)?;
            if next_token_id as usize == self.config.eos_token_id {
                break;
            }
            output_token_ids.push(next_token_id);
        }

        let output = self
            .tokenizer
            .decode(&output_token_ids, true)
            .map_anyhow_err()?;

        Ok(output)
    }
}
