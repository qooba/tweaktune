use anyhow::Result;
use log::{debug, error};
use pyo3::prelude::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::HashMap, sync::OnceLock};

static HTTP_CLIENT: OnceLock<Client> = OnceLock::new();

pub trait LLM {
    fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>>;

    fn call(
        &self,
        prompt: String,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>>;
}

pub enum LLMType {
    Api(ApiLLM),
    Unsloth(UnslothLLM),
    Mistralrs(MistralrsLLM),
}

pub enum ApiLLMMode {
    Api {
        api_key: String,
        model: String,
        base_url: String,
    },
    OpenAI {
        api_key: String,
        model: String,
    },
    AzureOpenAI {
        api_key: String,
        endpoint: String,
        deployment_name: String,
        api_version: String,
    },
}

pub struct MistralrsLLM {
    pub name: String,
    pub py_func: PyObject,
}

impl MistralrsLLM {
    pub fn new(name: String, py_func: PyObject) -> Self {
        Self { name, py_func }
    }

    async fn process(
        &self,
        messages: Vec<HashMap<String, String>>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String> {
        let result: PyResult<String> = Python::with_gil(|py| {
            let result: String = self
                .py_func
                .call_method1(
                    py,
                    "process",
                    (messages, json_schema, max_tokens, temperature),
                )?
                .extract(py)?;
            Ok(result)
        });

        match result {
            Ok(result) => Ok(result),
            Err(e) => {
                error!(target: "mistralrs_llm", "🐔 {:?}", e);
                Err(anyhow::anyhow!("Error processing messages: {:?}", e))
            }
        }
    }
}

impl LLM for MistralrsLLM {
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<ChatCompletionResponse> {
        let messages: Vec<HashMap<String, String>> = messages
            .into_iter()
            .map(|msg| {
                let mut map = HashMap::new();
                map.insert("role".to_string(), msg.role);
                map.insert("content".to_string(), msg.content);
                map
            })
            .collect();

        let result = self
            .process(messages, json_schema, max_tokens, temperature)
            .await?;
        let response = ChatCompletionResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: result,
                },
            }],
        };
        Ok(response)
    }

    fn call(
        &self,
        prompt: String,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>> {
        self.chat_completion(
            vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            json_schema,
            max_tokens,
            temperature,
        )
    }
}

pub struct UnslothLLM {
    pub name: String,
    pub py_func: PyObject,
}

impl UnslothLLM {
    pub fn new(name: String, py_func: PyObject) -> Self {
        Self { name, py_func }
    }

    async fn process(
        &self,
        messages: Vec<HashMap<String, String>>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String> {
        let result: PyResult<String> = Python::with_gil(|py| {
            let result: String = self
                .py_func
                .call_method1(
                    py,
                    "process",
                    (messages, json_schema, max_tokens, temperature),
                )?
                .extract(py)?;
            Ok(result)
        });

        match result {
            Ok(result) => Ok(result),
            Err(e) => {
                error!(target: "unsloth_llm", "🐔 {:?}", e);
                Err(anyhow::anyhow!("Error processing messages: {:?}", e))
            }
        }
    }
}

impl LLM for UnslothLLM {
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<ChatCompletionResponse> {
        let messages: Vec<HashMap<String, String>> = messages
            .into_iter()
            .map(|msg| {
                let mut map = HashMap::new();
                map.insert("role".to_string(), msg.role);
                map.insert("content".to_string(), msg.content);
                map
            })
            .collect();

        let result = self
            .process(messages, json_schema, max_tokens, temperature)
            .await?;
        let response = ChatCompletionResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: result,
                },
            }],
        };
        Ok(response)
    }

    fn call(
        &self,
        prompt: String,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>> {
        self.chat_completion(
            vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            json_schema,
            max_tokens,
            temperature,
        )
    }
}

#[derive(Clone)]
pub struct ApiLLM {
    pub name: String,
    pub url: String,
    pub api_key_header: (String, String),
    pub model: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl ApiLLM {
    pub fn new(name: String, mode: ApiLLMMode, max_tokens: u32, temperature: f32) -> Self {
        HTTP_CLIENT.get_or_init(Client::new);

        let (url, api_key_header, model) = match mode {
            ApiLLMMode::Api {
                api_key,
                model,
                base_url,
            } => (
                format!("{}/v1/chat/completions", base_url),
                ("Authorization".to_string(), format!("Bearer {}", api_key)),
                Some(model),
            ),
            ApiLLMMode::OpenAI { api_key, model } => (
                "https://api.openai.com/v1/chat/completions".to_string(),
                ("Authorization".to_string(), format!("Bearer {}", api_key)),
                Some(model),
            ),
            ApiLLMMode::AzureOpenAI {
                api_key,
                endpoint,
                deployment_name,
                api_version,
            } => (
                format!(
                    "{}/openai/deployments/{}?api-version={}",
                    endpoint, deployment_name, api_version
                ),
                ("api-key".to_string(), api_key),
                None,
            ),
        };

        Self {
            name,
            url,
            api_key_header,
            model,
            max_tokens,
            temperature,
        }
    }
}

impl LLM for ApiLLM {
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<ChatCompletionResponse> {
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: if let Some(mt) = max_tokens {
                mt
            } else {
                self.max_tokens
            },
            stream: None,
            seed: None,
            temperature: if temperature.is_some() {
                temperature
            } else {
                Some(self.temperature)
            },
            top_p: None,
            response_format: if json_schema.is_some() {
                Some(json!({"type": "json_schema", "json_schema": json_schema
                .map(|schema| serde_json::from_str::<serde_json::Value>(&schema).unwrap_or_default()) }))
            } else {
                None
            },
        };

        let response = HTTP_CLIENT
            .get()
            .expect("HTTP client not initialized")
            .post(&self.url)
            .header(&self.api_key_header.0, &self.api_key_header.1)
            .json(&request)
            .send()
            .await?
            .json::<ChatCompletionResponse>()
            .await?;
        Ok(response)
    }

    fn call(
        &self,
        prompt: String,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>> {
        self.chat_completion(
            vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            json_schema,
            max_tokens,
            temperature,
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub stream: Option<bool>,
    pub seed: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub response_format: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
}

#[cfg(test)]
mod tests {

    #[tokio::test]
    async fn test_openai_invoke() {
        println!("hello");
    }

    #[test]
    fn it_works() {
        println!("hello");
    }
}
