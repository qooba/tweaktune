use anyhow::Result;
use log::debug;
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
    OpenAI(OpenAILLM),
    Unsloth(UnslothLLM),
    Mistralrs(MistralrsLLM),
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
                debug!(target: "mistralrs_llm", "{:?}", e);
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
                debug!(target: "unsloth_llm", "{:?}", e);
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
pub struct OpenAILLM {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl OpenAILLM {
    pub fn new(
        name: String,
        base_url: String,
        api_key: String,
        model: String,
        max_tokens: u32,
        temperature: f32,
    ) -> Self {
        HTTP_CLIENT.get_or_init(Client::new);
        Self {
            name,
            base_url,
            api_key,
            model,
            max_tokens,
            temperature,
        }
    }
}

impl LLM for OpenAILLM {
    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        json_schema: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<ChatCompletionResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
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
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
    pub model: String,
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
