use anyhow::Result;
use arrow::json::reader::{infer_json_schema, ReaderBuilder};
use arrow::record_batch::RecordBatch;
use std::fs::File;
use std::sync::Arc;

pub trait Dataset {
    fn read_all(&self) -> Result<Vec<RecordBatch>>;
    fn read_next(&self) -> Result<Option<RecordBatch>>;
}

pub enum DatasetType {
    Jsonl,
    A(Vec<RecordBatch>),
}

pub struct JsonlDataset {
    path: String,
}

impl JsonlDataset {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl Dataset for JsonlDataset {
    fn read_all(&self) -> Result<Vec<RecordBatch>> {
        // Open file to infer schema.
        let file_for_infer = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file_for_infer);
        let (inferred_schema, _) = infer_json_schema(buf_reader, None)?;

        // Reopen file to create reader.
        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let reader = ReaderBuilder::new(Arc::new(inferred_schema)).build(buf_reader)?;

        // Collect all record batches.
        reader
            .map(|batch| batch.map_err(anyhow::Error::from))
            .collect()
    }

    fn read_next(&self) -> Result<Option<RecordBatch>> {
        // Open file to infer schema.
        let file_for_infer = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file_for_infer);
        let (inferred_schema, _) = infer_json_schema(buf_reader, None)?;

        // Reopen file to create reader.
        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let mut reader = ReaderBuilder::new(Arc::new(inferred_schema)).build(buf_reader)?;
        match reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}
