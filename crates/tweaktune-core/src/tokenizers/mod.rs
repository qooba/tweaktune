use tokenizers::{Encoding, Tokenizer};

pub struct TokenizerWrapper {
    pub tokenizer: Tokenizer,
}
impl TokenizerWrapper {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn encode(&self, text: &str) -> Result<Encoding, tokenizers::Error> {
        self.tokenizer.encode(text, true)
    }

    pub fn count(&self, text: &str) -> Result<usize, tokenizers::Error> {
        let encoding = self.encode(text)?;
        Ok(encoding.len())
    }
}
