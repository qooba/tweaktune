use anyhow::Result;

pub trait LLM {
    fn load(&self) -> Result<()>;
}
