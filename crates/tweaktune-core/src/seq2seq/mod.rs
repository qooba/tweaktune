use crate::common::{hf_hub_get, hf_hub_get_multiple, hf_hub_get_path};
use crate::common::{parse_device, OptionToResult, ResultExt};
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::t5;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

#[derive(Clone, Debug, Copy)]
enum Which {
    T5Base,
    T5Small,
    T5Large,
    T5_3B,
    Mt5Base,
    Mt5Small,
    Mt5Large,
}

pub struct Seq2SeqSpec {
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

struct Seq2SeqModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}

impl Seq2SeqModelBuilder {
    pub fn load(spec: Seq2SeqSpec) -> Result<(Self, Tokenizer)> {
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
        let tokenizer = Tokenizer::from_bytes(_tokenizer).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }

    pub fn build_conditional_generation(&self) -> Result<t5::T5ForConditionalGeneration> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }
}

/*
fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let (builder, mut tokenizer) = T5ModelBuilder::load(&args)?;
    let device = &builder.device;
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    match args.prompt {
        Some(prompt) => {
            let tokens = tokenizer
                .encode(prompt, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
            if !args.decode {
                let mut model = builder.build_encoder()?;
                let start = std::time::Instant::now();
                let ys = model.forward(&input_token_ids)?;
                println!("{ys}");
                println!("Took {:?}", start.elapsed());
            } else {
                let mut model = builder.build_conditional_generation()?;
                let mut output_token_ids = [builder
                    .config
                    .decoder_start_token_id
                    .unwrap_or(builder.config.pad_token_id)
                    as u32]
                .to_vec();
                if let Some(decoder_prompt) = &args.decoder_prompt {
                    print!("{decoder_prompt}");
                    output_token_ids.extend(
                        tokenizer
                            .encode(decoder_prompt.to_string(), false)
                            .map_err(E::msg)?
                            .get_ids()
                            .to_vec(),
                    );
                }
                let temperature = if args.temperature <= 0. {
                    None
                } else {
                    Some(args.temperature)
                };
                let mut logits_processor = LogitsProcessor::new(299792458, temperature, args.top_p);
                let encoder_output = model.encode(&input_token_ids)?;
                let start = std::time::Instant::now();

                for index in 0.. {
                    if output_token_ids.len() > 512 {
                        break;
                    }
                    let decoder_token_ids = if index == 0 || !builder.config.use_cache {
                        Tensor::new(output_token_ids.as_slice(), device)?.unsqueeze(0)?
                    } else {
                        let last_token = *output_token_ids.last().unwrap();
                        Tensor::new(&[last_token], device)?.unsqueeze(0)?
                    };
                    let logits = model
                        .decode(&decoder_token_ids, &encoder_output)?
                        .squeeze(0)?;
                    let logits = if args.repeat_penalty == 1. {
                        logits
                    } else {
                        let start_at = output_token_ids.len().saturating_sub(args.repeat_last_n);
                        candle_transformers::utils::apply_repeat_penalty(
                            &logits,
                            args.repeat_penalty,
                            &output_token_ids[start_at..],
                        )?
                    };

                    let next_token_id = logits_processor.sample(&logits)?;
                    if next_token_id as usize == builder.config.eos_token_id {
                        break;
                    }
                    output_token_ids.push(next_token_id);
                    if let Some(text) = tokenizer.id_to_token(next_token_id) {
                        let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                        print!("{text}");
                        std::io::stdout().flush()?;
                    }
                }
                let dt = start.elapsed();
                println!(
                    "\n{} tokens generated ({:.2} token/s)\n",
                    output_token_ids.len(),
                    output_token_ids.len() as f64 / dt.as_secs_f64(),
                );
            }
        }
        None => {
            let mut model = builder.build_encoder()?;
            let sentences = [
                "The cat sits outside",
                "A man is playing guitar",
                "I love pasta",
                "The new movie is awesome",
                "The cat plays in the garden",
                "A woman watches TV",
                "The new movie is so great",
                "Do you like pizza?",
            ];
            let n_sentences = sentences.len();
            let mut all_embeddings = Vec::with_capacity(n_sentences);
            for sentence in sentences {
                let tokens = tokenizer
                    .encode(sentence, true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let token_ids = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
                let embeddings = model.forward(&token_ids)?;
                println!("generated embeddings {:?}", embeddings.shape());
                // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
                let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
                let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
                let embeddings = if args.normalize_embeddings {
                    normalize_l2(&embeddings)?
                } else {
                    embeddings
                };
                println!("pooled embeddings {:?}", embeddings.shape());
                all_embeddings.push(embeddings)
            }

            let mut similarities = vec![];
            for (i, e_i) in all_embeddings.iter().enumerate() {
                for (j, e_j) in all_embeddings
                    .iter()
                    .enumerate()
                    .take(n_sentences)
                    .skip(i + 1)
                {
                    let sum_ij = (e_i * e_j)?.sum_all()?.to_scalar::<f32>()?;
                    let sum_i2 = (e_i * e_i)?.sum_all()?.to_scalar::<f32>()?;
                    let sum_j2 = (e_j * e_j)?.sum_all()?.to_scalar::<f32>()?;
                    let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                    similarities.push((cosine_similarity, i, j))
                }
            }
            similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
            for &(score, i, j) in similarities[..5].iter() {
                println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
            }
        }
    }
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
*/
