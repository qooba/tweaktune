use crate::common::hf_hub_get;
use crate::common::{parse_device, OptionToResult, ResultExt};
use crate::embeddings::Embeddings;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use once_cell::sync::OnceCell;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::{PaddingParams, Tokenizer};

pub const E5_MODEL_REPO: &str = "intfloat/e5-small-v2";

static E5_INSTANCES: OnceCell<Mutex<HashMap<String, Arc<Mutex<E5Model>>>>> = OnceCell::new();

#[derive(Deserialize, Debug, Clone)]
pub struct E5Spec {
    pub name: String,
    pub model_repo: Option<String>,
    pub device: Option<String>,
    pub hf_token: Option<String>,
}

pub struct E5Model {
    pub spec: E5Spec,
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub normalize_embeddings: Option<bool>,
    pub device: Device,
}

impl E5Model {
    pub fn lazy(spec: E5Spec) -> Result<Arc<Mutex<E5Model>>> {
        let name = spec.name.clone();

        if E5_INSTANCES.get().is_none() {
            let _ = E5_INSTANCES.set(Mutex::new(HashMap::new()));
        }

        let map = E5_INSTANCES.get().expect("E5_INSTANCES");
        {
            let guard = map.lock().map_anyhow_err()?;
            if let Some(existing) = guard.get(&name) {
                return Ok(existing.clone());
            }
        }

        let e5_model = E5Model::load(spec)?;
        let arc = Arc::new(Mutex::new(e5_model));
        let mut guard = map.lock().map_anyhow_err()?;
        guard.insert(name, arc.clone());
        Ok(arc)
    }

    pub fn load(spec: E5Spec) -> Result<E5Model> {
        let spec_clone = spec.clone();
        let model_repo = spec.model_repo.clone().expect("model_repo");
        let weights = hf_hub_get(
            &model_repo,
            "model.safetensors",
            spec.hf_token.clone(),
            None,
        )?;
        let tokenizer = hf_hub_get(&model_repo, "tokenizer.json", spec.hf_token.clone(), None)?;
        let candle_config = hf_hub_get(&model_repo, "config.json", spec.hf_token, None)?;
        let candle_config: BertConfig = serde_json::from_slice(&candle_config)?;

        let device = parse_device(spec.device)?;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer).map_anyhow_err()?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let model = BertModel::load(vb, &candle_config)?;
        Ok(E5Model {
            spec: spec_clone,
            model,
            tokenizer,
            normalize_embeddings: Some(true),
            device,
        })
    }
}

impl Embeddings for E5Model {
    fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let device = &self.device;
        let tokens = self
            .tokenizer
            .encode_batch(input.clone(), true)
            .map_anyhow_err()?;

        let token_ids: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let attention_mask: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = if let Some(true) = self.normalize_embeddings {
            embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?
        } else {
            embeddings
        };
        let embeddings_data: Vec<Vec<f32>> = embeddings.to_vec2()?;
        Ok(embeddings_data)
    }
}
