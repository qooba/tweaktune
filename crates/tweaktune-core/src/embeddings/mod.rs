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
