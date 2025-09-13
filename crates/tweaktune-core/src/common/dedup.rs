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

fn simhash64(text: &str) -> u64 {
    let normalized = normalize_text(text);
    simhash(&normalized)
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

pub fn call_hash(tool: &str, args: &Value) -> String {
    let cann = canonicalize_json(args).unwrap_or_default();
    let combined = format!("tool={};args={}", tool, cann);
    hash_exact(combined.as_bytes()).to_string()
}

pub fn io_hash(tool: &str, args: &Value, result: Option<&Value>) -> String {
    let cann_args = canonicalize_json(args).unwrap_or_default();
    let cann_result = match result {
        Some(res) => canonicalize_json(res).unwrap_or_default(),
        None => "null".to_string(),
    };
    let combined = format!("tool={};args={};result={}", tool, cann_args, cann_result);
    hash_exact(combined.as_bytes()).to_string()
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
    fn test_call_hash_canonicalization_order_independent() {
        let args1 = json!({"a": 1, "b": 2});
        let args2 = json!({"b": 2, "a": 1});
        let h1 = call_hash("mytool", &args1);
        let h2 = call_hash("mytool", &args2);
        assert_eq!(h1, h2, "call_hash should be independent of JSON key order");
    }

    #[test]
    fn test_call_hash_tool_difference() {
        let args = json!({"foo": "bar"});
        let h1 = call_hash("tool1", &args);
        let h2 = call_hash("tool2", &args);
        assert_ne!(h1, h2, "different tool names must produce different hashes");
    }

    #[test]
    fn test_io_hash_none_and_null_equal() {
        let args = json!({"x": 1});
        let h_none = io_hash("t", &args, None);
        let h_null = io_hash("t", &args, Some(&json!(null)));
        assert_eq!(h_none, h_null, "None result and explicit null should hash the same");
    }

    #[test]
    fn test_io_hash_result_changes() {
        let args = json!({"x": 1});
        let h1 = io_hash("t", &args, Some(&json!("ok")));
        let h2 = io_hash("t", &args, Some(&json!("changed")));
        assert_ne!(h1, h2, "different result values must change the io_hash");
    }

    #[test]
    fn test_call_and_io_different() {
        let args = json!({"a": 1});
        let ch = call_hash("t", &args);
        let ih = io_hash("t", &args, None);
        assert_ne!(ch, ih, "call_hash and io_hash should differ because io_hash includes result");
    }
}


