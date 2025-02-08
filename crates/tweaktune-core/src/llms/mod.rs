use anyhow::Result;

pub trait LLM {
    fn load(&self) -> Result<()>;
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
        Self {
            name,
            base_url,
            api_key,
            model,
        }
    }
}
