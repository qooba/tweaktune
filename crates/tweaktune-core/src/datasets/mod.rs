use anyhow::Result;
use arrow::csv::reader::{infer_schema_from_files, ReaderBuilder as CsvReaderBuilder};
use arrow::json::reader::{infer_json_schema, ReaderBuilder as JsonReaderBuilder};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

pub trait Dataset {
    fn read_all(&self) -> Result<Vec<RecordBatch>>;
    fn read_next(&mut self) -> Result<Option<RecordBatch>>;
}

pub trait Writer {
    fn write(&self, record_batch: RecordBatch) -> Result<()>;
}

#[derive(Clone)]
pub enum DatasetType {
    Jsonl(JsonlDataset),
    Csv(CsvDataset),
    Parquet(ParquetDataset),
    Arrow(ArrowDataset),
}

#[derive(Clone)]
pub struct JsonlDataset {
    name: String,
    path: String,
}

impl JsonlDataset {
    pub fn new(name: String, path: String) -> Self {
        Self { name, path }
    }
}

impl Dataset for JsonlDataset {
    fn read_all(&self) -> Result<Vec<RecordBatch>> {
        let file_for_infer = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file_for_infer);
        let (inferred_schema, _) = infer_json_schema(buf_reader, None)?;

        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let reader = JsonReaderBuilder::new(Arc::new(inferred_schema)).build(buf_reader)?;

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch?);
        }
        Ok(batches)
    }

    fn read_next(&mut self) -> Result<Option<RecordBatch>> {
        let file_for_infer = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file_for_infer);
        let (inferred_schema, _) = infer_json_schema(buf_reader, None)?;

        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let mut reader = JsonReaderBuilder::new(Arc::new(inferred_schema)).build(buf_reader)?;
        match reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

impl Writer for JsonlDataset {
    fn write(&self, record_batch: RecordBatch) -> Result<()> {
        let file = File::options().append(true).open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);

        let json_rows: Vec<serde_json::Value> = serde_arrow::from_record_batch(&record_batch)?;
        for row in json_rows {
            writeln!(writer, "{}", row)?;
        }

        writer.flush()?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct CsvDataset {
    name: String,
    path: String,
    delimiter: u8,
    has_header: bool,
}

impl CsvDataset {
    pub fn new(name: String, path: String, delimiter: u8, has_header: bool) -> Self {
        Self {
            name,
            path,
            delimiter,
            has_header,
        }
    }
}

impl Dataset for CsvDataset {
    fn read_all(&self) -> Result<Vec<RecordBatch>> {
        let inferred_schema =
            infer_schema_from_files(&[self.path.clone()], self.delimiter, None, self.has_header)?;

        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let reader = CsvReaderBuilder::new(Arc::new(inferred_schema)).build(buf_reader)?;

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch?);
        }
        Ok(batches)
    }

    fn read_next(&mut self) -> Result<Option<RecordBatch>> {
        let inferred_schema =
            infer_schema_from_files(&[self.path.clone()], self.delimiter, None, self.has_header)?;

        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let mut reader = CsvReaderBuilder::new(Arc::new(inferred_schema)).build(buf_reader)?;

        match reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

#[derive(Clone)]
pub struct ParquetDataset {
    name: String,
    path: String,
}

impl ParquetDataset {
    pub fn new(name: String, path: String) -> Self {
        Self { name, path }
    }
}

impl Dataset for ParquetDataset {
    fn read_all(&self) -> Result<Vec<RecordBatch>> {
        let file = File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.build().unwrap();

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch?);
        }
        Ok(batches)
    }

    fn read_next(&mut self) -> Result<Option<RecordBatch>> {
        let file = File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let mut reader = builder.build().unwrap();

        match reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}

#[derive(Clone)]
pub struct ArrowDataset {
    name: String,
    records: Vec<RecordBatch>,
    index: usize,
}

impl ArrowDataset {
    pub fn new(name: String, records: Vec<RecordBatch>) -> Self {
        Self {
            name,
            records,
            index: 0,
        }
    }
}

impl Dataset for ArrowDataset {
    fn read_all(&self) -> Result<Vec<RecordBatch>> {
        Ok(self.records.clone())
    }

    fn read_next(&mut self) -> Result<Option<RecordBatch>> {
        if self.index >= self.records.len() {
            return Ok(None);
        }

        self.index += 1;

        Ok(self.records.get(self.index).cloned())
    }
}
