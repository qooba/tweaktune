use crate::common::{create_rows_stream, df_to_values};
use crate::config::read_config;
use crate::readers::build_reader;
use anyhow::Result;
use polars::prelude::*;
use polars_utils::mmap::MemSlice;
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::sync::Arc;

pub trait Dataset {
    fn df(&self) -> &DataFrame;
    fn stream(&self) -> Result<impl Iterator<Item = Result<Value>> + '_> {
        create_rows_stream(self.df())
    }
}

#[derive(Clone)]
pub enum DatasetType {
    Json(JsonDataset),
    Jsonl(JsonlDataset),
    JsonList(JsonListDataset),
    OpenApi(OpenApiDataset),
    Polars(PolarsDataset),
    Ipc(IpcDataset),
    Csv(CsvDataset),
    Parquet(ParquetDataset),
    Mixed(MixedDataset),
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct JsonlDataset {
    name: String,
    path: String,
    sql: Option<String>,
    df: DataFrame,
}

impl JsonlDataset {
    pub fn new(name: String, path: String, sql: Option<String>) -> Result<Self> {
        let op_reader = build_reader(&path, None)?;
        let mut reader = op_reader.inner;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let sources = ScanSources::Buffers(Arc::new([MemSlice::from_vec(buf)]));
        let df = LazyJsonLineReader::new_with_sources(sources).finish()?;

        let df = if let Some(s) = sql.clone() {
            let mut ctx = polars::sql::SQLContext::new();
            ctx.register(&name, df);
            ctx.execute(&s)?
        } else {
            df
        };

        let df = df.collect()?;

        Ok(Self {
            name,
            path,
            sql,
            df,
        })
    }
}

impl Dataset for JsonlDataset {
    fn df(&self) -> &DataFrame {
        &self.df
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ParquetDataset {
    name: String,
    path: String,
    sql: Option<String>,
    df: DataFrame,
}

impl ParquetDataset {
    pub fn new(name: String, path: String, sql: Option<String>) -> Result<Self> {
        let op_reader = build_reader(&path, None)?;
        let mut reader = op_reader.inner;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let cursor = Cursor::new(buf);
        let df = ParquetReader::new(cursor).finish()?;

        let df = if let Some(s) = sql.clone() {
            let mut ctx = polars::sql::SQLContext::new();
            ctx.register(&name, df.lazy());
            ctx.execute(&s)?.collect()?
        } else {
            df
        };

        Ok(Self {
            name,
            path,
            sql,
            df,
        })
    }
}

impl Dataset for ParquetDataset {
    fn df(&self) -> &DataFrame {
        &self.df
    }
}

#[derive(Clone)]
pub struct CsvDataset {
    _name: String,
    df: DataFrame,
}

impl CsvDataset {
    pub fn new(
        name: String,
        path: String,
        delimiter: u8,
        has_header: bool,
        sql: Option<String>,
    ) -> Result<Self> {
        let op_reader = build_reader(&path, None)?;
        let mut reader = op_reader.inner;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        let sources = ScanSources::Buffers(Arc::new([MemSlice::from_vec(buf)]));
        let df = LazyCsvReader::new_with_sources(sources)
            .with_separator(delimiter)
            .with_has_header(has_header)
            .finish()?;

        let df = if let Some(s) = sql.clone() {
            let mut ctx = polars::sql::SQLContext::new();
            ctx.register(&name, df);
            ctx.execute(&s)?
        } else {
            df
        };

        let df = df.collect()?;

        Ok(Self { _name: name, df })
    }
}

impl Dataset for CsvDataset {
    fn df(&self) -> &DataFrame {
        &self.df
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct PolarsDataset {
    name: String,
    path: String,
    sql: Option<String>,
    df: DataFrame,
}

impl PolarsDataset {
    pub fn new(name: String, path: String, sql: Option<String>) -> Result<Self> {
        let op_reader = build_reader(&path, None)?;
        let mut reader = op_reader.inner;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;

        let df = if path.ends_with(".jsonl") || path.ends_with(".ndjson") {
            let sources = ScanSources::Buffers(Arc::new([MemSlice::from_vec(buf)]));
            LazyJsonLineReader::new_with_sources(sources).finish()?
        } else if path.ends_with(".csv") {
            let sources = ScanSources::Buffers(Arc::new([MemSlice::from_vec(buf)]));
            LazyCsvReader::new_with_sources(sources).finish().unwrap()
        } else if path.ends_with(".parquet") || path.ends_with(".pq") {
            let cursor = Cursor::new(buf);
            let df = ParquetReader::new(cursor).finish()?;
            df.lazy()
        } else {
            return Err(anyhow::anyhow!(
                "Unsupported file extension for PolarsDataset"
            ));
        };

        let df = if let Some(s) = sql.clone() {
            let mut ctx = polars::sql::SQLContext::new();
            ctx.register(&name, df);
            ctx.execute(&s)?
        } else {
            df
        };

        let df = df.collect()?;

        Ok(Self {
            name,
            path,
            sql,
            df,
        })
    }
}

impl Dataset for PolarsDataset {
    fn df(&self) -> &DataFrame {
        &self.df
    }
}

#[derive(Clone)]
pub struct IpcDataset {
    _name: String,
    df: DataFrame,
}

impl IpcDataset {
    pub fn new(name: String, ipc_data: &[u8], sql: Option<String>) -> Result<Self> {
        let cursor = Cursor::new(ipc_data);
        let df = IpcStreamReader::new(cursor).finish().unwrap();

        let df = if let Some(s) = sql.clone() {
            let mut ctx = polars::sql::SQLContext::new();
            ctx.register(&name, df.lazy());
            ctx.execute(&s)?.collect()?
        } else {
            df
        };
        Ok(Self { _name: name, df })
    }
}

impl Dataset for IpcDataset {
    fn df(&self) -> &DataFrame {
        &self.df
    }
}

#[derive(Clone)]
pub struct JsonDataset {
    _name: String,
    df: DataFrame,
}

impl JsonDataset {
    pub fn new(name: String, path: String, sql: Option<String>) -> Result<Self> {
        let mut op_reader = build_reader(&path, None)?;
        let mut buf = String::new();
        op_reader.inner.read_to_string(&mut buf)?;
        let cursor = std::io::Cursor::new(buf.as_bytes());
        let df: DataFrame = JsonReader::new(cursor).finish()?;

        let df = if let Some(s) = sql.clone() {
            let mut ctx = polars::sql::SQLContext::new();
            ctx.register(&name, df.lazy());
            ctx.execute(&s)?.collect()?
        } else {
            df
        };

        Ok(Self { _name: name, df })
    }
}

impl Dataset for JsonDataset {
    fn df(&self) -> &DataFrame {
        &self.df
    }
}

#[derive(Clone)]
pub struct JsonListDataset {
    _name: String,
    df: DataFrame,
}

impl JsonListDataset {
    pub fn new(name: String, json_list: Vec<String>, sql: Option<String>) -> Result<Self> {
        let json_array = format!("[{}]", json_list.join(","));
        let cursor = std::io::Cursor::new(json_array.as_bytes());
        let df: DataFrame = JsonReader::new(cursor).finish()?;

        let df = if let Some(s) = sql.clone() {
            let mut ctx = polars::sql::SQLContext::new();
            ctx.register(&name, df.lazy());
            ctx.execute(&s)?.collect()?
        } else {
            df
        };
        Ok(Self { _name: name, df })
    }
}

impl Dataset for JsonListDataset {
    fn df(&self) -> &DataFrame {
        &self.df
    }
}

#[derive(Clone)]
pub struct MixedDataset {
    _name: String,
    selected_datasets: Vec<String>,
    indexes: Vec<Vec<usize>>,
}

impl MixedDataset {
    pub fn new(
        name: String,
        selected_datasets: Vec<String>,
        datasets: &HashMap<String, DatasetType>,
    ) -> Result<Self> {
        let values: Vec<&DataFrame> = selected_datasets
            .iter()
            .map(|dataset| {
                let dataset = datasets.get(dataset).unwrap();
                match dataset {
                    DatasetType::Json(json_dataset) => json_dataset.df(),
                    DatasetType::JsonList(json_list_dataset) => json_list_dataset.df(),
                    DatasetType::OpenApi(open_api_dataset) => open_api_dataset.df(),
                    DatasetType::Polars(polars_dataset) => polars_dataset.df(),
                    DatasetType::Ipc(ipc_dataset) => ipc_dataset.df(),
                    DatasetType::Csv(csv_dataset) => csv_dataset.df(),
                    DatasetType::Parquet(parquet_dataset) => parquet_dataset.df(),
                    DatasetType::Jsonl(jsonl_dataset) => jsonl_dataset.df(),
                    _ => unimplemented!(),
                }
            })
            .collect();

        // Prepare cartesian product of all indexes of selected datasets
        let mut index_product: Vec<Vec<usize>> = vec![vec![]];
        for df in &values {
            let len = df.height();
            let mut new_product = Vec::new();
            for prefix in &index_product {
                for idx in 0..len {
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(idx);
                    new_product.push(new_prefix);
                }
            }
            index_product = new_product;
        }
        // index_product now contains all combinations of row indices for the selected datasets

        Ok(Self {
            _name: name,
            selected_datasets,
            indexes: index_product,
        })
    }

    fn fetch_selected_indexes(
        &self,
        datasets: &HashMap<String, DatasetType>,
        indexes: Vec<usize>,
    ) -> Result<Value> {
        let mut mix_obj = serde_json::Map::new();
        for (ix, value) in indexes.iter().enumerate() {
            let dataset_name = &self.selected_datasets[ix];
            let val = *value as i64;
            let dataset = datasets.get(dataset_name).unwrap();
            let df = match dataset {
                DatasetType::Json(json_dataset) => json_dataset.df().slice(val, 1),
                DatasetType::JsonList(json_list_dataset) => json_list_dataset.df().slice(val, 1),
                DatasetType::OpenApi(open_api_dataset) => open_api_dataset.df().slice(val, 1),
                DatasetType::Polars(polars_dataset) => polars_dataset.df().slice(val, 1),
                DatasetType::Ipc(ipc_dataset) => ipc_dataset.df().slice(val, 1),
                DatasetType::Csv(csv_dataset) => csv_dataset.df().slice(val, 1),
                DatasetType::Parquet(parquet_dataset) => parquet_dataset.df().slice(val, 1),
                DatasetType::Jsonl(jsonl_dataset) => jsonl_dataset.df().slice(val, 1),
                DatasetType::Mixed(_mixed_dataset) => unimplemented!(),
            };

            let df_values = df_to_values(&df).unwrap();
            let df_values = df_values.first().unwrap();
            mix_obj.insert(dataset_name.clone(), df_values.clone());
        }

        Ok(Value::Object(mix_obj))
    }

    pub fn sample(
        &self,
        size: usize,
        datasets: &HashMap<String, DatasetType>,
    ) -> Result<Vec<Value>> {
        let total = self.indexes.len();
        if size > total {
            return Err(anyhow::anyhow!(
                "Sample size exceeds total number of combinations"
            ));
        }
        let samples = self
            .indexes
            .choose_multiple(&mut rand::rng(), size)
            .cloned()
            .collect::<Vec<_>>();
        let num = samples.len();

        let v: Vec<Value> = (0..num)
            .map(move |idx| {
                let indexes = samples[idx].clone();
                self.fetch_selected_indexes(datasets, indexes).unwrap()
            })
            .collect();

        Ok(v)
    }

    pub fn stream_mix<'a>(
        &'a self,
        datasets: &'a HashMap<String, DatasetType>,
    ) -> Result<impl Iterator<Item = Result<Value>> + 'a> {
        let num = self.indexes.len();

        Ok((0..num).map(move |idx| {
            let indexes = self.indexes[idx].clone();
            Ok(self.fetch_selected_indexes(datasets, indexes).unwrap())
        }))
    }
}

impl Dataset for MixedDataset {
    fn df(&self) -> &DataFrame {
        unimplemented!()
    }
}

#[derive(Clone, Debug)]
pub struct OpenApiDataset {
    _name: String,
    _path_or_url: String,
    _open_api_spec: OpenApiSpec,
    df: DataFrame,
}

impl OpenApiDataset {
    pub fn new(name: String, path_or_url: String) -> Result<Self> {
        let config = read_config::<OpenApiSpec>(&path_or_url, None).unwrap();
        let json = openapi_read_all_json(&config)?;
        let json_array = serde_json::to_string(&json)?;
        let cursor = std::io::Cursor::new(json_array.as_bytes());
        let df: DataFrame = JsonReader::new(cursor).finish()?;

        Ok(Self {
            _name: name,
            _path_or_url: path_or_url,
            _open_api_spec: config,
            df,
        })
    }
}

fn openapi_build_function_from_path_item(
    _path: &str,
    _method: &str,
    item: &OpenApiPathItem,
    open_api_spec: &OpenApiSpec,
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
                    if let Some(component) = open_api_spec.components.schemas.get(&schema) {
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
                                        open_api_spec.components.schemas.get(&enum_values)
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

fn openapi_read_all_json(open_api_spec: &OpenApiSpec) -> Result<Vec<Value>> {
    let mut functions = Vec::new();

    for (path, path_item) in &open_api_spec.paths {
        if let Some(get_item) = &path_item.get {
            let function =
                openapi_build_function_from_path_item(path, "get", get_item, open_api_spec);
            functions.push(function);
        }

        if let Some(post_item) = &path_item.post {
            let function =
                openapi_build_function_from_path_item(path, "post", post_item, open_api_spec);
            functions.push(function);
        }

        if let Some(put_item) = &path_item.put {
            let function =
                openapi_build_function_from_path_item(path, "put", put_item, open_api_spec);
            functions.push(function);
        }

        if let Some(delete_item) = &path_item.delete {
            let function =
                openapi_build_function_from_path_item(path, "delete", delete_item, open_api_spec);
            functions.push(function);
        }
    }

    Ok(functions)
}

impl Dataset for OpenApiDataset {
    fn df(&self) -> &DataFrame {
        &self.df
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

#[cfg(test)]
mod tests {
    use crate::datasets::OpenApiDataset;
    use anyhow::Result;
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    // use serde_json;

    #[test]
    fn it_works() -> Result<()> {
        //let url = "https://petstore3.swagger.io/api/v3/openapi.json";
        let url = "http://localhost:8085/openapi.json";
        let _spec = OpenApiDataset::new("test".to_string(), url.to_string());

        // println!(
        //     "spec: {:?}",
        //     serde_json::to_string(&spec.open_api_spec).unwrap()
        // );

        // let funcs = spec.read_all_json().unwrap();
        // for func in funcs {
        //     println!("{}\n", func);
        // }
        Ok(())
    }
}
