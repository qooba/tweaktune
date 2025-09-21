#![cfg(feature = "integration-tests")]
//RUN_E5_INTEGRATION=1 cargo test -p tweaktune-core --features integration-tests -- --nocapture
use std::{env, println};
use tweaktune_core::embeddings::e5::{E5Model, E5Spec, E5_MODEL_REPO};
use tweaktune_core::embeddings::Embeddings;

#[test]
fn e5_integration_test() -> Result<(), anyhow::Error> {
    if env::var("RUN_E5_INTEGRATION").unwrap_or_default() != "1" {
        eprintln!("Skipping e5 integration test because RUN_E5_INTEGRATION != 1");
        return Ok(());
    }

    let spec = E5Spec {
        name: "test".to_string(),
        model_repo: Some(E5_MODEL_REPO.to_string()),
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
    let emb = guard.embed(input)?;
    println!("Embeddings: {:?}", emb);
    assert!(!emb.is_empty());
    Ok(())
}
