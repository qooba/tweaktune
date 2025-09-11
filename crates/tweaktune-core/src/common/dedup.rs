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

fn canonicalize_json(input: String) -> Option<String> {
    serde_json_canonicalizer::to_string(&input).ok()
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
