#![cfg(feature = "integration-tests")]
use core::time;
//RUN_E5_INTEGRATION=1 cargo test -p tweaktune-core --features integration-tests -- --nocapture
use candle_transformers::models::flux::model;
use std::{env, println};
use tweaktune_core::embeddings::e5::{E5Model, E5Spec, E5_MODEL_REPO};
use tweaktune_core::embeddings::Embeddings;
use tweaktune_core::templates::embed;

#[test]
fn e5_integration_test() -> Result<(), anyhow::Error> {
    if env::var("RUN_E5_INTEGRATION").unwrap_or_default() != "1" {
        eprintln!("Skipping e5 integration test because RUN_E5_INTEGRATION != 1");
        return Ok(());
    }

    let model_repo = "intfloat/multilingual-e5-small";
    let spec = E5Spec {
        name: "test".to_string(),
        model_repo: Some(model_repo.to_string()),
        device: None,
        hf_token: env::var("HF_TOKEN").ok(),
    };

    // Create or get the named instance.
    let instance = E5Model::lazy(spec)?;
    let guard = instance
        .lock()
        .map_err(|e| anyhow::anyhow!("lock error: {:?}", e))?;

    // Simple sanity: call embed with an input
    let input = vec!["hello world".to_string()];
    let timer = std::time::Instant::now();
    let mut emb = Vec::new();
    for _ in 0..3 {
        emb = guard.embed(input.clone())?;
    }
    let elapsed = timer.elapsed();
    println!("3 inferences took {:?}", elapsed);
    println!("Embeddings: {:?}", emb);
    assert!(!emb.is_empty());
    Ok(())
}
