use pyo3::prelude::*;
use tweaktune_pyo3::steps::{Arrow, Csv, Jsonl, Lang, Parquet, Step, StepConfig};

#[pymodule]
#[pyo3(name = "tweaktune")]
fn tweaktune(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // let steps_module = PyModule::new(py, "steps")?;
    // steps_module.add_class::<Step>()?;
    // steps_module.add_class::<StepConfig>()?;
    // steps_module.add_class::<Jsonl>()?;
    // m.add_submodule(&steps_module)?;

    m.add_class::<Step>()?;
    m.add_class::<StepConfig>()?;
    m.add_class::<Jsonl>()?;
    m.add_class::<Parquet>()?;
    m.add_class::<Csv>()?;
    m.add_class::<Arrow>()?;
    m.add_class::<Lang>()?;

    // let llms_module = PyModule::new_bound(py, "llms")?;
    // llms_module.add_class::<Quantized>()?;
    // llms_module.add_class::<Gemma>()?;

    // m.add_submodule(&llms_module)?;

    Ok(())
}
