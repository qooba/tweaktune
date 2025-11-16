pub mod e5;
use crate::embeddings::e5::E5Spec;
use anyhow::Result;

pub trait Embeddings {
    fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>>;
}

#[derive(Clone)]
pub enum EmbeddingsType {
    OpenAI(OpenAIEmbeddings),
    E5(E5Spec),
}

#[derive(Clone)]
pub struct OpenAIEmbeddings {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAIEmbeddings {
    pub fn new(name: String, base_url: String, api_key: String, model: String) -> Self {
        Self {
            name,
            base_url,
            api_key,
            model,
        }
    }
}

impl Embeddings for OpenAIEmbeddings {
    fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::blocking::Client::new();
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = serde_json::json!({
            "input": input,
            "model": self.model,
        });
        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!(
                "OpenAI API request failed with status: {}",
                resp.status()
            ));
        }

        let resp_json: serde_json::Value = resp.json()?;
        let embeddings: Vec<Vec<f32>> = resp_json["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))
                    .and_then(|arr| {
                        arr.iter()
                            .map(|v| {
                                v.as_f64()
                                    .map(|f| f as f32)
                                    .ok_or_else(|| anyhow::anyhow!("Invalid float value"))
                            })
                            .collect()
                    })
            })
            .collect::<Result<Vec<Vec<f32>>>>()?;

        Ok(embeddings)
    }
}

#[allow(dead_code)]
fn quantize_f32_to_f16(rows: &[Vec<f32>]) -> Vec<Vec<u16>> {
    rows.iter()
        .map(|vec| {
            vec.iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect()
        })
        .collect()
}

#[allow(dead_code)]
fn dequantize_f16_to_f32(input: &[Vec<u16>]) -> Vec<f32> {
    input
        .iter()
        .flat_map(|vec| vec.iter().map(|&x| half::f16::from_bits(x).to_f32()))
        .collect()
}

#[allow(dead_code)]
fn f16_to_blob(input: &[Vec<u16>]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(input.len() * input[0].len() * 2);
    for row in input {
        for &val in row {
            blob.extend_from_slice(&val.to_le_bytes());
        }
    }
    blob
}

#[allow(dead_code)]
fn blob_to_f16(blob: &[u8], dim: usize) -> Vec<Vec<u16>> {
    let num_rows = blob.len() / (dim * 2);
    let mut result = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let start = i * dim * 2;
        let end = start + dim * 2;
        let row_bytes = &blob[start..end];
        let row: Vec<u16> = row_bytes
            .chunks(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        result.push(row);
    }
    result
}
