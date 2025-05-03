use crate::config::read_config;
use crate::readers::read_file_with_opendal;
use crate::steps::flat_map_to_json;
use anyhow::Result;
use arrow::csv::reader::{Format, ReaderBuilder as CsvReaderBuilder};
use arrow::datatypes::SchemaRef;
use arrow::json::reader::{infer_json_schema, ReaderBuilder as JsonReaderBuilder};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use polars::lazy::frame::IntoLazy;
use polars::prelude::*;
use polars_plan::plans::ScanSources;
use polars_utils::mmap::MemSlice;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
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
    JsonList(JsonListDataset),
    Csv(CsvDataset),
    Parquet(ParquetDataset),
    Arrow(ArrowDataset),
    Mixed(MixedDataset),
    OpenApi(OpenApiDataset),
    Polars(PolarsDataset),
}

#[derive(Clone)]
pub struct PolarsDataset {
    name: String,
    path: String,
    sql: String,
}

impl PolarsDataset {
    pub fn new(name: String, path: String, sql: String) -> Self {
        Self { name, path, sql }
    }

    pub fn read_all_json(&self) -> Result<Vec<Value>> {
        let op_reader = read_file_with_opendal(&self.path)?;
        let mut reader = op_reader.inner;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let sources = ScanSources::Buffers(Arc::new([MemSlice::from_vec(buf)]));

        let df = if self.path.ends_with(".json") {
            LazyJsonLineReader::new_with_sources(sources).finish()?
        } else if self.path.ends_with(".csv") {
            LazyCsvReader::new_with_sources(sources).finish()?
        } else if self.path.ends_with(".parquet") || self.path.ends_with(".pq") {
            let args = ScanArgsParquet::default();
            LazyFrame::scan_parquet_sources(sources, args)?
        } else {
            return Err(anyhow::anyhow!(
                "Unsupported file extension for PolarsDataset"
            ));
        };

        let lf = df.lazy();
        let mut ctx = polars::sql::SQLContext::new();
        ctx.register(&self.name, lf);

        let result = ctx.execute(&self.sql)?;
        let result_df = result.collect()?;
        let columns = result_df.get_column_names();
        let mut json_rows = Vec::new();

        for idx in 0..result_df.height() {
            let row = result_df.get_row(idx)?;
            let mut obj = serde_json::Map::new();
            for (col, val) in columns.iter().zip(row.0.iter()) {
                obj.insert(col.to_string(), anyvalue_to_json(val));
            }
            json_rows.push(Value::Object(obj));
        }

        Ok(json_rows)
    }
}

impl Dataset for PolarsDataset {
    fn read_all(&self, _batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        unimplemented!()
    }
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
    ) -> Result<arrow::json::reader::Reader<opendal::StdReader>> {
        let op_reader = read_file_with_opendal(&self.path)?;
        let (inferred_schema, _) = infer_json_schema(op_reader.inner, None)?;

        let op_reader = read_file_with_opendal(&self.path)?;
        let mut reader = JsonReaderBuilder::new(Arc::new(inferred_schema));
        if let Some(size) = batch_size {
            reader = reader.with_batch_size(size);
        }

        let reader = reader.build(op_reader.inner)?;

        Ok(reader)
    }

    pub fn create_json_stream(&self) -> Result<impl Iterator<Item = Result<Value>>> {
        let op_reader = read_file_with_opendal(&self.path)?;
        let stream = serde_json::Deserializer::from_reader(op_reader.inner).into_iter::<Value>();
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
    ) -> Result<arrow::csv::reader::BufReader<BufReader<opendal::StdReader>>> {
        let format = Format::default()
            .with_delimiter(self.delimiter)
            .with_header(self.has_header);

        let op_reader = read_file_with_opendal(&self.path)?;
        let inferred_schema = format.infer_schema(op_reader.inner, None)?.0;

        let op_reader = read_file_with_opendal(&self.path)?;
        let mut reader = CsvReaderBuilder::new(Arc::new(inferred_schema));
        if let Some(size) = batch_size {
            reader = reader.with_batch_size(size);
        }

        let reader = reader.build(op_reader.inner)?;

        Ok(reader)
    }

    pub fn create_json_stream(&self) -> Result<impl Iterator<Item = Result<Value>>> {
        let op_reader = read_file_with_opendal(&self.path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_header)
            .from_reader(op_reader.inner);

        let headers = rdr.headers()?.clone();
        let iter = rdr.into_records().map(move |result| {
            let record = result?;
            let mut map = serde_json::Map::new();
            for (header, field) in headers.iter().zip(record.iter()) {
                map.insert(header.to_string(), Value::String(field.to_string()));
            }
            Ok(Value::Object(map))
        });
        Ok(iter)
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
        let op_reader = read_file_with_opendal(&self.path)?;
        let mut reader = op_reader.inner;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let bb = bytes::Bytes::from(buf);
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(bb).unwrap();

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
        let op_reader = read_file_with_opendal(&self.path)?;
        let values: Vec<Value> = serde_json::from_reader(op_reader.inner)?;
        Ok(values)
    }
}

impl Dataset for JsonDataset {
    fn read_all(&self, _batch_size: Option<usize>) -> Result<Vec<RecordBatch>> {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct JsonListDataset {
    _name: String,
    json_list: Vec<String>,
}

impl JsonListDataset {
    pub fn new(name: String, json_list: Vec<String>) -> Self {
        Self {
            _name: name,
            json_list,
        }
    }

    pub fn read_all_json(&self) -> Result<Vec<Value>> {
        let values: Vec<Value> = self
            .json_list
            .iter()
            .map(|json_str| serde_json::from_str(json_str).unwrap())
            .collect();
        Ok(values)
    }
}

impl Dataset for JsonListDataset {
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
    _path_or_url: String,
    open_api_spec: OpenApiSpec,
}

impl OpenApiDataset {
    pub fn new(name: String, path_or_url: String) -> Self {
        let config = read_config::<OpenApiSpec>(&path_or_url, None).unwrap();

        // let definitions: Vec<serde_json::Value> =
        // serde_json::from_value(config["definitions"].clone()).unwrap();

        Self {
            _name: name,
            _path_or_url: path_or_url,
            open_api_spec: config,
        }
    }

    fn build_function_from_path_item(
        &self,
        _path: &str,
        _method: &str,
        item: &OpenApiPathItem,
    ) -> Value {
        let mut parameters = HashMap::new();
        let mut required = Vec::new();

        if let Some(params) = &item.parameters {
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

        if let Some(request_body) = &item.request_body {
            if let Some(content) = &request_body.content {
                for schema_ref in content.values() {
                    if let Some(schema) = &schema_ref.schema.ref_ {
                        let schema = schema.replace("#/components/schemas/", "");
                        if let Some(component) = self.open_api_spec.components.schemas.get(&schema)
                        {
                            let mut property = HashMap::new();
                            if let Some(t) = component.type_.clone() {
                                property.insert("type".to_string(), Value::String(t));
                            }
                            if let Some(description) = &component.description {
                                property.insert(
                                    "description".to_string(),
                                    Value::String(description.clone()),
                                );
                            }
                            required.push("request_body".to_string());

                            let mut props = HashMap::new();
                            if let Some(component_properties) = &component.properties {
                                for (key, value) in component_properties {
                                    let mut prop = HashMap::new();

                                    if let Some(t) = value.type_.clone() {
                                        prop.insert("type".to_string(), Value::String(t));
                                    }

                                    if let Some(description) = &value.description {
                                        prop.insert(
                                            "description".to_string(),
                                            Value::String(description.clone()),
                                        );
                                    }

                                    if let Some(any_of) = &value.any_of {
                                        let types = any_of
                                            .iter()
                                            .map(|v| v.type_.clone().unwrap_or_default())
                                            .collect::<Vec<String>>();

                                        prop.insert("type".to_string(), json!(types));
                                    }

                                    if let Some(enum_values) = &value.ref_ {
                                        let enum_values =
                                            enum_values.replace("#/components/schemas/", "");
                                        if let Some(enums) =
                                            self.open_api_spec.components.schemas.get(&enum_values)
                                        {
                                            if let Some(e) = &enums.enum_ {
                                                prop.insert("enum".to_string(), json!(e.clone()));
                                                prop.insert(
                                                    "type".to_string(),
                                                    Value::String(
                                                        enums.type_.as_ref().unwrap().clone(),
                                                    ),
                                                );
                                            }
                                        }
                                    }

                                    props.insert(key.clone(), prop);
                                }
                            }

                            property.insert("properties".to_string(), json!(props));
                            parameters.insert("request_body".to_string(), property);
                        }
                    }
                }
            }
        }

        json!({
            "type": "function",
            "name": item.summary.as_ref().unwrap().replace(" ", "_").to_lowercase(),
            "description": item.description.clone().unwrap_or_default(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
                "additionalProperties": false
            },
            "strict": true
        })
    }

    pub fn read_all_json(&self) -> Result<Vec<Value>> {
        let mut functions = Vec::new();

        for (path, path_item) in &self.open_api_spec.paths {
            if let Some(get_item) = &path_item.get {
                let function = self.build_function_from_path_item(path, "get", get_item);
                functions.push(function);
            }

            if let Some(post_item) = &path_item.post {
                let function = self.build_function_from_path_item(path, "post", post_item);
                functions.push(function);
            }

            if let Some(put_item) = &path_item.put {
                let function = self.build_function_from_path_item(path, "put", put_item);
                functions.push(function);
            }

            if let Some(delete_item) = &path_item.delete {
                let function = self.build_function_from_path_item(path, "delete", delete_item);
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
    description: Option<String>,
    title: Option<String>,
    properties: Option<HashMap<String, OpenApiComponentSchemaProperty>>,
    required: Option<Vec<String>>,
    #[serde(rename = "enum")]
    enum_: Option<Vec<String>>,
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
    #[serde(rename = "anyOf")]
    any_of: Option<Vec<OpenApiComponentSchemaProperty>>,
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

fn anyvalue_to_json(val: &AnyValue) -> Value {
    match val {
        AnyValue::Null => Value::Null,
        AnyValue::Boolean(b) => Value::Bool(*b),
        AnyValue::Int64(i) => Value::from(*i),
        AnyValue::UInt64(u) => Value::from(*u),
        AnyValue::Float64(f) => Value::from(*f),
        AnyValue::String(s) => Value::from(*s),
        AnyValue::Int8(i) => Value::from(*i),
        AnyValue::Int16(i) => Value::from(*i),
        AnyValue::Int32(i) => Value::from(*i),
        AnyValue::UInt8(u) => Value::from(*u),
        AnyValue::UInt16(u) => Value::from(*u),
        AnyValue::UInt32(u) => Value::from(*u),
        AnyValue::Float32(f) => Value::from(*f),
        AnyValue::Date(d) => Value::from(*d),
        AnyValue::Datetime(dt, _, _) => Value::from(*dt),
        AnyValue::Time(t) => Value::from(*t),
        AnyValue::Duration(d, _) => Value::from(*d),
        AnyValue::Decimal(val, scale) => {
            let factor = 10f64.powi(*scale as i32);
            Value::from(*val as f64 / factor)
        }
        _ => Value::String(val.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use crate::datasets::OpenApiDataset;
    use anyhow::Result;
    use arrow::json::reader::ReaderBuilder as ArrowJsonReaderBuilder;
    use serde_arrow::from_record_batch;
    // use serde_json;

    #[test]
    fn it_works() -> Result<()> {
        //let url = "https://petstore3.swagger.io/api/v3/openapi.json";
        let url = "http://localhost:8085/openapi.json";
        let spec = OpenApiDataset::new("test".to_string(), url.to_string());

        // println!(
        //     "spec: {:?}",
        //     serde_json::to_string(&spec.open_api_spec).unwrap()
        // );

        let funcs = spec.read_all_json().unwrap();
        for func in funcs {
            println!("{}\n", func);
        }
        Ok(())
    }
}
