use anyhow::Result;
use log::debug;
use pyo3::prelude::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::OnceLock};

static HTTP_CLIENT: OnceLock<Client> = OnceLock::new();

pub trait LLM {
    fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>>;

    fn call(
        &self,
        prompt: String,
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>>;
}

pub enum LLMType {
    OpenAI(OpenAILLM),
    Unsloth(UnslothLLM),
}

pub struct UnslothLLM {
    pub name: String,
    pub py_func: PyObject,
}

impl UnslothLLM {
    pub fn new(name: String, py_func: PyObject) -> Self {
        Self { name, py_func }
    }

    async fn process(&self, messages: Vec<HashMap<String, String>>) -> Result<String> {
        let result: PyResult<String> = Python::with_gil(|py| {
            let result: String = self
                .py_func
                .call_method1(py, "process", (messages,))?
                .extract(py)?;
            Ok(result)
        });

        match result {
            Ok(result) => Ok(result),
            Err(e) => {
                debug!("{:?}", e);
                Err(anyhow::anyhow!("Error processing messages: {:?}", e))
            }
        }
    }
}

impl LLM for UnslothLLM {
    async fn chat_completion(&self, messages: Vec<ChatMessage>) -> Result<ChatCompletionResponse> {
        let messages: Vec<HashMap<String, String>> = messages
            .into_iter()
            .map(|msg| {
                let mut map = HashMap::new();
                map.insert("role".to_string(), msg.role);
                map.insert("content".to_string(), msg.content);
                map
            })
            .collect();

        let result = self.process(messages).await?;
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
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>> {
        self.chat_completion(vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
        }])
    }
}

#[derive(Clone)]
pub struct OpenAILLM {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
}

impl OpenAILLM {
    pub fn new(
        name: String,
        base_url: String,
        api_key: String,
        model: String,
        max_tokens: u32,
    ) -> Self {
        HTTP_CLIENT.get_or_init(Client::new);
        Self {
            name,
            base_url,
            api_key,
            model,
            max_tokens,
        }
    }
}

impl LLM for OpenAILLM {
    async fn chat_completion(&self, messages: Vec<ChatMessage>) -> Result<ChatCompletionResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens,
            stream: None,
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
    ) -> impl std::future::Future<Output = Result<ChatCompletionResponse>> {
        self.chat_completion(vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
        }])
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub stream: Option<bool>,
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

    use super::*;

    #[tokio::test]
    async fn test_openai_invoke() {
        println!("hello");
    }

    #[test]
    fn it_works() {
        println!("hello");
    }
}
