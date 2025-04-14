use crate::config::read_config;
use crate::steps::{flat_map_to_json, map_to_json};
use anyhow::Result;
use arrow::csv::reader::{infer_schema_from_files, ReaderBuilder as CsvReaderBuilder};
use arrow::datatypes::SchemaRef;
use arrow::json::reader::{infer_json_schema, ReaderBuilder as JsonReaderBuilder};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
    Json(JsonDataset),
    Csv(CsvDataset),
    Parquet(ParquetDataset),
    Arrow(ArrowDataset),
    Mixed(MixedDataset),
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

    pub fn create_json_stream(&self) -> Result<impl Iterator<Item = Result<Value>>> {
        let file = File::open(&self.path)?;
        let buf_reader = BufReader::new(file);
        let stream = serde_json::Deserializer::from_reader(buf_reader).into_iter::<Value>();
        Ok(stream.map(|value| value.map_err(anyhow::Error::from)))
    }

    pub fn read_all_json(&self) -> Result<Vec<Value>> {
        let stream = self.create_json_stream()?;
        let mut records = Vec::new();
        for record in stream {
            records.push(record?);
        }
        Ok(records)
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

#[derive(Clone)]
pub struct JsonDataset {
    _name: String,
    path: String,
}

impl JsonDataset {
    pub fn new(name: String, path: String) -> Self {
        Self { _name: name, path }
    }

    pub fn read_all_json(&self) -> Result<Vec<Value>> {
        let file = File::open(&self.path)?;
        let buf_reader = BufReader::new(file);
        let values: Vec<Value> = serde_json::from_reader(buf_reader)?;
        Ok(values)
    }
}

impl Dataset for JsonDataset {
    fn read_all(&self, _batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct MixedDataset {
    _name: String,
    datasets: Vec<String>,
}

impl MixedDataset {
    pub fn new(name: String, datasets: Vec<String>) -> Self {
        Self {
            _name: name,
            datasets,
        }
    }

    pub fn read_all_json(&self, datasets: &HashMap<String, DatasetType>) -> Result<Vec<Value>> {
        let values: Vec<Vec<Value>> = self
            .datasets
            .iter()
            .map(|dataset| {
                let dataset = datasets.get(dataset).unwrap();
                match dataset {
                    DatasetType::Json(json_dataset) => json_dataset.read_all_json().unwrap(),
                    DatasetType::Jsonl(jsonl_dataset) => jsonl_dataset.read_all_json().unwrap(),
                    DatasetType::Csv(csv_dataset) => {
                        flat_map_to_json(&csv_dataset.read_all(None).unwrap())
                    }
                    DatasetType::Parquet(parquet_dataset) => {
                        flat_map_to_json(&parquet_dataset.read_all(None).unwrap())
                    }
                    DatasetType::Arrow(arrow_dataset) => {
                        flat_map_to_json(&arrow_dataset.read_all(None).unwrap())
                    }
                    _ => unimplemented!(),
                }
            })
            .collect();

        let mut cartesian_product = vec![HashMap::new()];

        for (i, dataset_values) in values.iter().enumerate() {
            let dataset_name = &self.datasets[i];
            let mut new_product = Vec::new();

            for record in dataset_values {
                for existing_combination in &cartesian_product {
                    let mut new_combination = existing_combination.clone();
                    new_combination.insert(dataset_name.clone(), record.clone());
                    new_product.push(new_combination);
                }
            }

            cartesian_product = new_product;
        }

        let result: Vec<Value> = cartesian_product
            .into_iter()
            .map(|combination| serde_json::to_value(combination).unwrap())
            .collect();

        Ok(result)
    }
}

impl Dataset for MixedDataset {
    fn read_all(&self, _batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        unimplemented!()
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

    pub fn read_all_json(&self) -> Result<Vec<Value>> {
        let mut functions = Vec::new();

        for (path, path_item) in &self.open_api_spec.paths {
            if let Some(get_item) = &path_item.get {
                let mut parameters = HashMap::new();
                let mut required = Vec::new();

                if let Some(params) = &get_item.parameters {
                    for param in params {
                        let mut property = HashMap::new();
                        property.insert(
                            "type".to_string(),
                            Value::String(param.schema.type_.clone()),
                        );
                        if let Some(description) = &param.description {
                            property.insert(
                                "description".to_string(),
                                Value::String(description.clone()),
                            );
                        }
                        if let Some(enum_values) = &param.schema.enum_ {
                            property.insert(
                                "enum".to_string(),
                                Value::Array(
                                    enum_values
                                        .iter()
                                        .map(|v| Value::String(v.clone()))
                                        .collect(),
                                ),
                            );
                        }
                        parameters.insert(param.name.clone(), property);
                        if param.required.unwrap_or(false) {
                            required.push(param.name.clone());
                        }
                    }
                }

                let function = json!({
                    "type": "function",
                    "function": {
                        "name": format!("{}_{}", "get", path.replace("/", "_").trim_start_matches('_')),
                        "description": get_item.summary.clone().unwrap_or_default(),
                        "parameters": {
                            "type": "object",
                            "properties": parameters,
                            "required": required,
                            "additionalProperties": false
                        },
                        "strict": true
                    }
                });

                functions.push(function);
            }
        }

        Ok(functions)
    }
}

impl Dataset for OpenApiDataset {
    fn read_all(&self, _batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        unimplemented!();
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
    tags: Option<Vec<String>>,
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
    description: Option<String>,
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
    use anyhow::Result;

    #[test]
    fn it_works() -> Result<()> {
        //let url = "https://petstore3.swagger.io/api/v3/openapi.json";
        let url = "http://localhost:8085/openapi.json";
        let spec = OpenApiDataset::new("test".to_string(), url.to_string());
        let funcs = spec.read_all_json().unwrap();
        for func in funcs {
            println!("{}\n", func);
        }
        Ok(())
    }
}
