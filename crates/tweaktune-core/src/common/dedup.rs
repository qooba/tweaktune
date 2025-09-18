use anyhow::Result;
use serde_json::Value;
use simhash::{hamming_distance, simhash};
use unicode_normalization::UnicodeNormalization;

fn normalize_text(text: &str) -> String {
    let lower = text.nfkc().collect::<String>().to_lowercase();
    let collapsed = regex::Regex::new(r"\s+")
        .unwrap()
        .replace_all(&lower, " ")
        .to_string();
    collapsed.trim().to_string()
}

fn hash_exact(bytes: &[u8]) -> blake3::Hash {
    blake3::hash(bytes)
}

pub fn hash_value(value: &Value) -> String {
    match value {
        Value::String(s) => hash_exact(s.as_bytes()).to_string(),
        _ => {
            let cann = canonicalize_json(value).expect("Failed to canonicalize JSON");
            hash_exact(cann.as_bytes()).to_string()
        }
    }
}

fn simhash64(text: &str) -> u64 {
    let normalized = normalize_text(text);
    simhash(&normalized)
}

pub fn simhash_value(value: &Value) -> u64 {
    match value {
        Value::String(s) => simhash64(s),
        _ => {
            let cann = canonicalize_json(value).expect("Failed to canonicalize JSON");
            simhash64(&cann)
        }
    }
}

fn is_similar(hash1: u64, hash2: u64, threshold: u32) -> bool {
    hamming_distance(hash1, hash2) <= threshold
}

fn has_similar(existing_text: Vec<String>, new_text: &str, threshold: u32) -> bool {
    let new_hash = simhash64(new_text);
    for text in existing_text {
        let existing_hash = simhash64(&text);
        if is_similar(existing_hash, new_hash, threshold) {
            return true;
        }
    }
    false
}
pub fn deduplicate_texts(texts: Vec<String>, similarity_threshold: u32) -> Vec<String> {
    let mut unique_texts = Vec::new();
    for text in texts {
        if !has_similar(unique_texts.clone(), &text, similarity_threshold) {
            unique_texts.push(text);
        }
    }
    unique_texts
}

fn canonicalize_json(input: &Value) -> Option<String> {
    serde_json_canonicalizer::to_string(input).ok()
}

fn word_shingles(text: &str, shingle_size: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut shingles = Vec::new();
    if words.len() < shingle_size {
        return shingles;
    }
    for i in 0..=words.len() - shingle_size {
        let shingle = words[i..i + shingle_size].join(" ");
        shingles.push(shingle);
    }
    shingles
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simhash_normalization() {
        let a = "Hello,   World!";
        let b = "hello, world!";
        assert_eq!(simhash64(a), simhash64(b));
    }

    #[test]
    fn test_simhash_is_similar_exact() {
        let a = "Some Unique Text";
        let b = "Some Unique Text";
        let ha = simhash64(a);
        let hb = simhash64(b);
        // identical texts must be within zero hamming distance
        assert!(is_similar(ha, hb, 0));
    }

    #[test]
    fn test_deduplicate_texts_basic() {
        let texts = vec![
            "Hello World".to_string(),
            "hello   world".to_string(),
            "Completely different".to_string(),
            "HELLO world".to_string(),
        ];
        let deduped = deduplicate_texts(texts, 0);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0], "Hello World");
        assert_eq!(deduped[1], "Completely different");
    }

    #[test]
    fn test_call_hash_canonicalization_tool_call() -> Result<()> {
        let call1 = json!({"name": "mytool", "arguments": {"a": 1, "b": 2}});
        let call2 = json!({"name": "mytool", "arguments": {"b": 2, "a": 1}});
        let h1 = hash_value(&call1);
        let h2 = hash_value(&call2);
        assert_eq!(h1, h2, "call_hash should be independent of JSON key order");
        Ok(())
    }
}
