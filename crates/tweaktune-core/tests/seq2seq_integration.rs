#![cfg(feature = "integration-tests")]
//RUN_SEQ2SEQ_INTEGRATION=1 cargo test -p tweaktune-core --features integration-tests -- --nocapture
use std::{env, println};
use tweaktune_core::embeddings::Embeddings;
use tweaktune_core::seq2seq::{Seq2SeqModel, Seq2SeqSpec};

#[test]
fn seq2seq_integration_test() -> Result<(), anyhow::Error> {
    if env::var("RUN_SEQ2SEQ_INTEGRATION").unwrap_or_default() != "1" {
        eprintln!("Skipping seq2seq integration test because RUN_SEQ2SEQ_INTEGRATION != 1");
        return Ok(());
    }

    let spec = Seq2SeqSpec::default();

    // Create or get the named instance.
    let instance = Seq2SeqModel::lazy(spec)?;
    let guard = instance
        .lock()
        .map_err(|e| anyhow::anyhow!("lock error: {:?}", e))?;

    // Simple sanity: call embed with an input
    let input = "translate to Polish: A beautiful cat sits on the windowsill.".to_string();
    let timer = std::time::Instant::now();
    let mut output = String::new();
    for _ in 0..1 {
        output = guard.forward(input.clone(), None)?;
    }
    let elapsed = timer.elapsed();
    println!("3 inferences took {:?}", elapsed);
    println!("Output: {:?}", output);
    assert!(!output.is_empty());
    Ok(())
}
