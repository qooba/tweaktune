use anyhow::Result;
use reqwest::Client;
use serde::{de, Deserialize, Serialize};
use std::sync::OnceLock;
use tokio;

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

#[derive(Clone)]
pub enum LLMType {
    OpenAI(OpenAILLM),
}

#[derive(Clone)]
pub struct OpenAILLM {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAILLM {
    pub fn new(name: String, base_url: String, api_key: String, model: String) -> Self {
        HTTP_CLIENT.get_or_init(Client::new);
        Self {
            name,
            base_url,
            api_key,
            model,
        }
    }
}

impl LLM for OpenAILLM {
    async fn chat_completion(&self, messages: Vec<ChatMessage>) -> Result<ChatCompletionResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: 100,
            stream: None,
        };
        let response = HTTP_CLIENT
            .get()
            .unwrap()
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .unwrap()
            .json::<ChatCompletionResponse>()
            .await
            .unwrap();
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
