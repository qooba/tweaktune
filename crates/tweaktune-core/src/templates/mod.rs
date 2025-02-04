use anyhow::Result;

pub trait Template {
    fn load(&self) -> Result<()>;
}
