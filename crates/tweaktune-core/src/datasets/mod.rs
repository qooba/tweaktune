use anyhow::Result;
use arrow::csv::reader::{infer_schema_from_files, ReaderBuilder as CsvReaderBuilder};
use arrow::datatypes::SchemaRef;
use arrow::json::reader::{self, infer_json_schema, ReaderBuilder as JsonReaderBuilder};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use std::f32::consts::E;
use std::fs::File;
use std::io::{BufReader, Write};
use std::sync::Arc;

pub trait Dataset {
    fn read_all(&self, batch_size: Option<usize>) -> Result<Vec<RecordBatch>>;
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

    pub fn create_stream(
        &self,
        batch_size: Option<usize>,
    ) -> Result<arrow::json::reader::Reader<BufReader<File>>> {
        let file_for_infer = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file_for_infer);
        let (inferred_schema, _) = infer_json_schema(buf_reader, None)?;

        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let mut reader = JsonReaderBuilder::new(Arc::new(inferred_schema));
        if let Some(size) = batch_size {
            reader = reader.with_batch_size(size);
        }

        let reader = reader.build(buf_reader)?;

        Ok(reader)
    }
}

impl Dataset for JsonlDataset {
    fn read_all(&self, batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        let reader = self.create_stream(batch_size)?;
        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch?);
        }
        Ok(batches)
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

    pub fn create_stream(
        &self,
        batch_size: Option<usize>,
    ) -> Result<arrow::csv::reader::BufReader<BufReader<BufReader<File>>>> {
        let inferred_schema =
            infer_schema_from_files(&[self.path.clone()], self.delimiter, None, self.has_header)?;

        let file = File::open(&self.path)?;
        let buf_reader = std::io::BufReader::new(file);
        let mut reader = CsvReaderBuilder::new(Arc::new(inferred_schema));
        if let Some(size) = batch_size {
            reader = reader.with_batch_size(size);
        }

        let reader = reader.build(buf_reader)?;

        Ok(reader)
    }
}

impl Dataset for CsvDataset {
    fn read_all(&self, batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        let reader = self.create_stream(batch_size)?;
        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch?);
        }
        Ok(batches)
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

    pub fn create_stream(&self, batch_size: Option<usize>) -> Result<ParquetRecordBatchReader> {
        let file = File::open(&self.path)?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        if let Some(size) = batch_size {
            builder = builder.with_batch_size(size);
        }

        let reader = builder.build().unwrap();

        Ok(reader)
    }
}

impl Dataset for ParquetDataset {
    fn read_all(&self, batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        let reader = self.create_stream(batch_size)?;

        let mut batches = Vec::new();
        for batch in reader {
            batches.push(batch?);
        }
        Ok(batches)
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
    fn read_all(&self, batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        if let Some(size) = batch_size {
            if self.records.is_empty() {
                return Ok(vec![]);
            }

            let schema: SchemaRef = self.records[0].schema();
            let contcatenated_batches = arrow::compute::concat_batches(&schema, &self.records)?;
            let num_rows = contcatenated_batches.num_rows();
            let mut batches = Vec::new();
            for start in (0..num_rows).step_by(size) {
                let end = std::cmp::min(start + size, num_rows);
                let batch = contcatenated_batches.slice(start, end - start);
                batches.push(batch);
            }

            Ok(batches)
        } else {
            Ok(self.records.clone())
        }
    }
}
