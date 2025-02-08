use anyhow::Result;

pub trait Embeddings {
    fn load(&self) -> Result<()>;
}

#[derive(Clone)]
pub enum EmbeddingsType {
    OpenAI(OpenAIEmbeddings),
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
