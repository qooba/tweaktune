use crate::config::read_config;
use anyhow::Result;
use arrow::csv::reader::{infer_schema_from_files, ReaderBuilder as CsvReaderBuilder};
use arrow::datatypes::SchemaRef;
use arrow::json::reader::{infer_json_schema, ReaderBuilder as JsonReaderBuilder};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
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
    _name: String,
    path: String,
}

impl JsonlDataset {
    pub fn new(name: String, path: String) -> Self {
        Self { _name: name, path }
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
    _name: String,
    path: String,
    delimiter: u8,
    has_header: bool,
}

impl CsvDataset {
    pub fn new(name: String, path: String, delimiter: u8, has_header: bool) -> Self {
        Self {
            _name: name,
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
    _name: String,
    path: String,
}

impl ParquetDataset {
    pub fn new(name: String, path: String) -> Self {
        Self { _name: name, path }
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
    _name: String,
    records: Vec<RecordBatch>,
}

impl ArrowDataset {
    pub fn new(name: String, records: Vec<RecordBatch>) -> Self {
        Self {
            _name: name,
            records,
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

#[derive(Clone, Debug)]
pub struct OpenApiDataset {
    _name: String,
    path_or_url: String,
    open_api_spec: OpenApiSpec,
}

impl OpenApiDataset {
    pub fn new(name: String, path_or_url: String) -> Self {
        let config = read_config::<OpenApiSpec>(&path_or_url, None).unwrap();

        // let definitions: Vec<serde_json::Value> =
        // serde_json::from_value(config["definitions"].clone()).unwrap();

        Self {
            _name: name,
            path_or_url,
            open_api_spec: config,
        }
    }
}

impl Dataset for OpenApiDataset {
    fn read_all(&self, _batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        /*
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
        */
        todo!()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiSpec {
    paths: HashMap<String, OpenApiPath>,
    components: OpenApiComponents,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiComponents {
    schemas: HashMap<String, OpenApiComponentSchema>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiComponentSchema {
    #[serde(rename = "type")]
    type_: Option<String>,
    properties: HashMap<String, OpenApiComponentSchemaProperty>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiComponentSchemaProperty {
    #[serde(rename = "type")]
    type_: Option<String>,
    description: Option<String>,
    format: Option<String>,
    example: Option<Value>,
    #[serde(rename = "enum")]
    enum_: Option<Vec<String>>,
    #[serde(rename = "$ref")]
    ref_: Option<String>,
    items: Option<OpenApiSchemaRef>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiPath {
    get: Option<OpenApiPathItem>,
    post: Option<OpenApiPathItem>,
    put: Option<OpenApiPathItem>,
    delete: Option<OpenApiPathItem>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiPathItem {
    tags: Vec<String>,
    summary: Option<String>,
    description: Option<String>,
    parameters: Option<Vec<OpenApiParameter>>,
    #[serde(rename = "requestBody")]
    request_body: Option<OpenApiRequestBody>,
    responses: HashMap<String, OpenApiResponse>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiResponse {
    description: Option<String>,
    content: Option<HashMap<String, OpenApiBodySchema>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiRequestBody {
    description: Option<String>,
    content: Option<HashMap<String, OpenApiBodySchema>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiBodySchema {
    schema: OpenApiSchemaRef,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiSchemaRef {
    #[serde(rename = "$ref")]
    ref_: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiParameter {
    name: String,
    #[serde(rename = "in")]
    in_: String,
    description: String,
    required: Option<bool>,
    schema: OpenApiSchema,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiSchema {
    #[serde(rename = "type")]
    type_: String,
    default: Option<Value>,
    #[serde(rename = "enum")]
    enum_: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenApiProperty {
    description: String,
    type_: String,
}

#[cfg(test)]
mod tests {
    use crate::datasets::OpenApiDataset;

    #[test]
    fn it_works() {
        let url = "https://petstore3.swagger.io/api/v3/openapi.json";
        let spec = OpenApiDataset::new("test".to_string(), url.to_string());
        println!("{:?}", spec);
        println!("hello");
    }
}
