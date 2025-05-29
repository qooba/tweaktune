use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use log::debug;
use once_cell::sync::OnceCell;
use polars::prelude::*;
use polars_arrow::array::ListArray;
use rand::distr::Alphanumeric;
use rand::{Rng, RngCore};
use regex::Regex;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::io::{self, Result as IoResult};
use std::{env, fs, path::PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum CommonError {
    #[error("Error {0}")]
    StringError(String),
}

pub struct Errors {}

impl Errors {
    pub fn str(error: &str) -> Box<dyn std::error::Error> {
        Box::new(CommonError::StringError(error.to_string()))
    }

    pub fn anyhow(error: &str) -> anyhow::Error {
        anyhow!(error.to_string())
    }

    pub fn io(error: &str) -> io::Error {
        io::Error::new(io::ErrorKind::Other, format!("{error:?}"))
    }
}

pub trait OptionToResult<T> {
    fn ok_or_err(self, name: &str) -> Result<T>;

    fn expect_tt(self, name: &str) -> T;
}

impl<T> OptionToResult<T> for Option<T> {
    fn ok_or_err(self, name: &str) -> Result<T> {
        self.ok_or(anyhow!("🐔 {:?} - value not found", name))
    }

    fn expect_tt(self, name: &str) -> T {
        self.unwrap_or_else(|| panic!("🐔 {:?} - value not found", name))
    }
}

pub trait ResultExt<T, E> {
    fn map_tt_err(self, message: &str) -> Result<T>;

    fn map_anyhow_err(self) -> Result<T>;

    fn map_io_err(self) -> IoResult<T>;

    fn map_box_err(self) -> Result<T, Box<dyn std::error::Error>>;

    fn map_str_err(self) -> Result<T, String>;
}

impl<T, E: std::fmt::Debug> ResultExt<T, E> for Result<T, E> {
    fn map_tt_err(self, message: &str) -> Result<T> {
        self.map_err(|_e| anyhow!("🐔 {:?}", message))
    }

    fn map_anyhow_err(self) -> Result<T> {
        self.map_err(|e| anyhow!("{:?}", e))
    }

    fn map_io_err(self) -> IoResult<T> {
        self.map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{e:?}")))
    }

    fn map_box_err(self) -> Result<T, Box<dyn std::error::Error>> {
        Ok(self.map_err(|e| Box::new(CommonError::StringError(format!("{e:?}"))))?)
    }

    fn map_str_err(self) -> Result<T, String> {
        self.map_err(|e| format!("{e:?}"))
    }
}

pub fn random_base64(num_bytes: usize) -> String {
    let mut buffer = vec![0u8; num_bytes];
    rand::rng().fill_bytes(&mut buffer);
    general_purpose::STANDARD.encode(&buffer)
}

pub fn generate_token(length: usize) -> String {
    let mut rng = rand::rng();
    (0..length)
        .map(|_| rng.sample(Alphanumeric) as char)
        .collect()
}

pub fn hf_hub_get_path(
    hf_repo_id: &str,
    filename: &str,
    hf_token: Option<String>,
    revision: Option<String>,
) -> Result<PathBuf> {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

    let mut api_builder = ApiBuilder::new();

    if let Some(token) = hf_token {
        api_builder = api_builder.with_token(Some(token));
    }

    let repo = if let Some(rev) = revision {
        Repo::with_revision(hf_repo_id.to_string(), RepoType::Model, rev)
    } else {
        Repo::new(hf_repo_id.to_string(), RepoType::Model)
    };

    let api = api_builder.build()?.repo(repo);
    let path = api.get(filename)?;

    Ok(path)
}

pub fn hf_hub_get(
    hf_repo_id: &str,
    filename: &str,
    hf_token: Option<String>,
    revision: Option<String>,
) -> Result<Vec<u8>> {
    let path = hf_hub_get_path(hf_repo_id, filename, hf_token, revision)?;
    let data = fs::read(path)?;
    Ok(data)
}

pub fn hf_hub_get_multiple(
    hf_repo_id: &str,
    json_file: &str,
    hf_token: Option<String>,
    revision: Option<String>,
) -> Result<Vec<PathBuf>> {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

    let mut api_builder = ApiBuilder::new();

    if let Some(token) = hf_token {
        api_builder = api_builder.with_token(Some(token));
    }

    let repo = if let Some(rev) = revision {
        Repo::with_revision(hf_repo_id.to_string(), RepoType::Model, rev)
    } else {
        Repo::new(hf_repo_id.to_string(), RepoType::Model)
    };

    let api = api_builder.build()?.repo(repo);
    let json_path = api.get(json_file)?;

    let json_file = std::fs::File::open(json_path)?;
    let json: serde_json::Value = serde_json::from_reader(&json_file)?;
    let weight_map = match json.get("weight_map") {
        None => Err(anyhow!("no weight map in {json_file:?}")),
        Some(serde_json::Value::Object(map)) => Ok(map),
        Some(_) => Err(anyhow!("weight map in {json_file:?} is not a map")),
    }?;
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| api.get(v).map_err(anyhow::Error::msg))
        .collect::<Result<Vec<_>>>()?;

    Ok(safetensors_files)
}

// https://docs.rs/once_cell/latest/once_cell/#lateinit
pub struct LateInit<T> {
    cell: OnceCell<T>,
}

impl<T> LateInit<T> {
    pub fn init(&self, value: T) {
        assert!(self.cell.set(value).is_ok())
    }
}

impl<T> Default for LateInit<T> {
    fn default() -> Self {
        LateInit {
            cell: OnceCell::default(),
        }
    }
}

impl<T> std::ops::Deref for LateInit<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.cell.get().unwrap()
    }
}

pub fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

pub enum SerializationType {
    JSON,
    YAML,
}

pub fn decode_and_deserialize<T>(b64_data: &str, serialization_type: SerializationType) -> Result<T>
where
    T: DeserializeOwned,
{
    let decoded_data = general_purpose::STANDARD.decode(b64_data)?;
    let object = match serialization_type {
        SerializationType::JSON => serde_json::from_slice(&decoded_data)?,
        SerializationType::YAML => serde_yaml::from_slice(&decoded_data)?,
    };

    Ok(object)
}

pub fn serialize_and_encode<T>(object: T, serialization_type: SerializationType) -> Result<String>
where
    T: serde::Serialize,
{
    let object_str = match serialization_type {
        SerializationType::JSON => serde_json::to_string(&object)?,
        SerializationType::YAML => serde_yaml::to_string(&object)?,
    };

    let encoded_data = general_purpose::STANDARD.encode(object_str);
    Ok(encoded_data)
}

pub fn env_or_some<T>(key: &str, some: Option<T>) -> T
where
    T: std::str::FromStr,
{
    if let Ok(value) = env::var(key) {
        if let Ok(parsed) = value.parse::<T>() {
            return parsed;
        }
    }

    some.unwrap()
}

pub fn some_or_env<T>(some: Option<T>, key: &str) -> T
where
    T: std::str::FromStr,
{
    if let Some(parsed) = some {
        return parsed;
    } else if let Ok(value) = env::var(key) {
        if let Ok(parsed) = value.parse::<T>() {
            return parsed;
        }
    }

    todo!();
}

pub fn env_or_some_or_fn<T, F>(key: &str, some: Option<T>, func: F) -> T
where
    T: std::str::FromStr,
    F: Fn() -> T,
{
    if let Ok(value) = env::var(key) {
        if let Ok(parsed) = value.parse::<T>() {
            return parsed;
        }
    }

    some.unwrap_or_else(func)
}

pub fn unwrap_str(val: Option<String>, default: &str) -> String {
    val.unwrap_or(String::from(default))
}

fn extract_json_block_md(text: &str) -> Result<Value> {
    let json_str = extract_json_regex(text, r"(?s)```json\s*(.*?)\s*```")?;
    let json: Value = serde_json::from_str(&json_str)?;
    Ok(json)
}

fn extract_json_block(text: &str) -> Result<Value> {
    let json_str = extract_json_regex(text, r"(?s)\{(.*?)\}\s*")?;
    let json_str = format!("{{ {} }}", json_str.trim());
    let json: Value = serde_json::from_str(&json_str)?;
    Ok(json)
}

fn extract_json_regex(text: &str, re: &str) -> Result<String> {
    let re = Regex::new(re)?;
    let captures = re
        .captures(text)
        .ok_or_else(|| anyhow!("No JSON block found, captures"))?;
    let json_str = captures
        .get(1)
        .ok_or_else(|| anyhow!("No JSON block found"))?
        .as_str();
    Ok(json_str.to_string())
}

pub fn extract_json(text: &str) -> Result<Value> {
    let text = text.replace("<|im_end|>", "");
    let value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_e) => match extract_json_block_md(&text) {
            Ok(v) => v,
            Err(_e) => match extract_json_block(&text) {
                Ok(v) => v,
                Err(e) => {
                    debug!(target: "extract_json", "EXTRACT JSON {}", &text);
                    return Err(anyhow!("Failed to extract JSON: {}", e));
                }
            },
        },
    };
    Ok(value)
}

pub fn df_to_values(df: &DataFrame) -> Result<Vec<Value>> {
    let columns = df.get_column_names();
    let mut json_rows = Vec::new();

    for idx in 0..df.height() {
        let row = df.get_row(idx)?;
        let mut obj = serde_json::Map::new();
        for (col, val) in columns.iter().zip(row.0.iter()) {
            obj.insert(col.to_string(), anyvalue_to_json(val));
        }
        json_rows.push(Value::Object(obj));
    }

    Ok(json_rows)
}

pub fn create_rows_stream(df: &DataFrame) -> Result<impl Iterator<Item = Result<Value>> + '_> {
    let columns = df.get_column_names();
    let height = df.height();
    Ok((0..height).map(move |idx| {
        let row = df.get_row(idx).unwrap();
        let mut obj = serde_json::Map::new();
        for (col, val) in columns.iter().zip(row.0.iter()) {
            obj.insert(col.to_string(), anyvalue_to_json(val));
        }
        Ok(Value::Object(obj))
    }))
}

pub fn anyvalue_to_json(val: &AnyValue) -> Value {
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
        AnyValue::Int128(val) => Value::from(*val as i32),
        AnyValue::DatetimeOwned(dt, _time_unit, _pl_small_str) => Value::from(*dt),
        AnyValue::List(series) => {
            let list = series
                .iter()
                .map(|v| anyvalue_to_json(&v))
                .collect::<Vec<_>>();
            Value::Array(list)
        }
        AnyValue::Struct(_, struct_array, fields) => map_struct_to_value(struct_array, fields),
        AnyValue::StringOwned(pl_small_str) => Value::from(pl_small_str.to_string()),
        AnyValue::Binary(items) => {
            let list = items.iter().map(|v| Value::from(*v)).collect::<Vec<_>>();
            Value::Array(list)
        }
        AnyValue::BinaryOwned(items) => {
            let list = items.iter().map(|v| Value::from(*v)).collect::<Vec<_>>();
            Value::Array(list)
        }
        _ => Value::String(val.to_string()),
    }
}

fn map_struct_to_value(
    struct_array: &polars::prelude::StructArray,
    fields: &[polars::prelude::Field],
) -> Value {
    let mut obj = serde_json::Map::new();
    struct_array
        .values()
        .iter()
        .zip((*fields).iter())
        .for_each(|(v, f)| {
            let field_name = f.name();
            match v.dtype() {
                ArrowDataType::Boolean => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::BooleanArray>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::Bool(x.unwrap_or(false)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Int8 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Int8Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Int16 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Int16Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Int32 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Int32Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Int64 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Int64Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Int128 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Int128Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0) as i64))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::UInt8 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::UInt8Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::UInt16 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::UInt16Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::UInt32 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::UInt32Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::UInt64 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::UInt64Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Float16 => {
                    println!("FLOAT16");
                    todo!()
                }
                ArrowDataType::Float32 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Float32Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0.0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Float64 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Float64Array>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(*x.unwrap_or(&0.0)))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Utf8Array<i64>>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(x.unwrap_or("")))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Utf8View => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::Utf8ViewArray>()
                        .unwrap()
                        .iter()
                        .map(|x| Value::from(x.unwrap_or("")))
                        .collect::<Vec<_>>();

                    let v = v[0].to_owned();
                    obj.insert(field_name.to_string(), v);
                }
                ArrowDataType::Struct(f) => {
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::StructArray>()
                        .unwrap();
                    // Convert ArrowField to polars Field
                    let fields: Vec<polars::prelude::Field> = f
                        .iter()
                        .map(|af| {
                            polars::prelude::Field::new(
                                af.name.clone(),
                                arrow_data_type_to_polars_data_type(&af.dtype),
                            )
                        })
                        .collect();
                    let struct_obj = map_struct_to_value(v, &fields);
                    obj.insert(field_name.to_string(), struct_obj);
                }
                ArrowDataType::Timestamp(_time_unit, _pl_small_str) => {
                    println!("TIMESTAMP");
                    todo!()
                }
                ArrowDataType::Date32 => {
                    println!("DATE32");
                    todo!()
                }
                ArrowDataType::Date64 => {
                    println!("DATE64");
                    todo!()
                }
                ArrowDataType::Time32(_time_unit) => {
                    println!("TIME32");
                    todo!()
                }
                ArrowDataType::Time64(_time_unit) => {
                    println!("TIME64");
                    todo!()
                }
                ArrowDataType::Duration(_time_unit) => {
                    println!("DURATION");
                    todo!()
                }
                ArrowDataType::Interval(_interval_unit) => {
                    println!("INTERVAL");
                    todo!()
                }
                ArrowDataType::Binary => {
                    println!("BINARY");
                    todo!()
                }
                ArrowDataType::FixedSizeBinary(_) => {
                    println!("FIXED SIZE BINARY");
                    todo!()
                }
                ArrowDataType::LargeBinary => {
                    println!("LARGE BINARY");
                    todo!()
                }
                ArrowDataType::FixedSizeList(_field, _) => {
                    println!("FIXED SIZE LIST");
                    todo!()
                }

                ArrowDataType::List(field) | ArrowDataType::LargeList(field) => {
                    let dt = field.dtype();
                    let v = v
                        .as_any()
                        .downcast_ref::<polars_arrow::array::ListArray<i64>>()
                        .unwrap();
                    let v = map_array_to_value(v, dt.clone());
                    if let Some(v) = v {
                        obj.insert(field_name.to_string(), v);
                    };
                }
                ArrowDataType::Map(_field, _) => {
                    println!("MAP");
                    todo!()
                }
                ArrowDataType::Dictionary(_integer_type, _arrow_data_type, _) => {
                    println!("DICTIONARY");
                    todo!()
                }
                ArrowDataType::Decimal(_, _) => {
                    println!("DECIMAL");
                    todo!()
                }
                ArrowDataType::Decimal256(_, _) => {
                    println!("DECIMAL256");
                    todo!()
                }
                ArrowDataType::Extension(_extension_type) => {
                    println!("EXTENSION");
                    todo!()
                }
                ArrowDataType::BinaryView => {
                    println!("BINARY VIEW");
                    todo!()
                }
                ArrowDataType::Union(_union_type) => {
                    println!("UNION");
                    todo!()
                }
                ArrowDataType::Unknown | ArrowDataType::Null => {}
            }
        });
    Value::Object(obj)
}

fn map_array_to_value(array: &ListArray<i64>, arrow_type: ArrowDataType) -> Option<Value> {
    match arrow_type {
        ArrowDataType::Boolean => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let bool_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::BooleanArray>()
                            .unwrap();

                        bool_val.iter().map(|x| x.unwrap()).collect::<Vec<bool>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Int8 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let int8_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Int8Array>()
                            .unwrap();

                        int8_val.iter().map(|x| *x.unwrap()).collect::<Vec<i8>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Int16 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let int16_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Int16Array>()
                            .unwrap();

                        int16_val.iter().map(|x| *x.unwrap()).collect::<Vec<i16>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Int32 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let int32_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Int32Array>()
                            .unwrap();

                        int32_val.iter().map(|x| *x.unwrap()).collect::<Vec<i32>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Int64 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let int64_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Int64Array>()
                            .unwrap();

                        int64_val.iter().map(|x| *x.unwrap()).collect::<Vec<i64>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Int128 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let int128_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Int128Array>()
                            .unwrap();

                        int128_val
                            .iter()
                            .map(|x| *x.unwrap() as i64)
                            .collect::<Vec<i64>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::UInt8 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let uint8_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::UInt8Array>()
                            .unwrap();

                        uint8_val.iter().map(|x| *x.unwrap()).collect::<Vec<u8>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::UInt16 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let uint16_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::UInt16Array>()
                            .unwrap();

                        uint16_val.iter().map(|x| *x.unwrap()).collect::<Vec<u16>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::UInt32 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let uint32_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::UInt32Array>()
                            .unwrap();

                        uint32_val.iter().map(|x| *x.unwrap()).collect::<Vec<u32>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::UInt64 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let uint64_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::UInt64Array>()
                            .unwrap();

                        uint64_val.iter().map(|x| *x.unwrap()).collect::<Vec<u64>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Float32 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let float32_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Float32Array>()
                            .unwrap();

                        float32_val
                            .iter()
                            .map(|x| *x.unwrap())
                            .collect::<Vec<f32>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Float64 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let float64_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Float64Array>()
                            .unwrap();

                        float64_val
                            .iter()
                            .map(|x| *x.unwrap())
                            .collect::<Vec<f64>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let utf8_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Utf8Array<i64>>()
                            .unwrap();

                        utf8_val
                            .iter()
                            .map(|x| x.unwrap().to_string())
                            .collect::<Vec<String>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        ArrowDataType::Utf8View => {
            let v = array
                .iter()
                .map(|x| match x {
                    Some(val) => {
                        let utf8_val = val
                            .as_any()
                            .downcast_ref::<polars_arrow::array::Utf8ViewArray>()
                            .unwrap();

                        utf8_val
                            .iter()
                            .map(|x| x.unwrap().to_string())
                            .collect::<Vec<String>>()
                    }
                    None => vec![],
                })
                .collect::<Vec<_>>();
            let v = v.last();
            v.map(|v| Value::from(v.clone()))
        }
        _ => {
            println!("Unsupported ArrowDataType: {:?}", arrow_type);
            todo!()
        }
    }
}

fn arrow_data_type_to_polars_data_type(
    arrow_data_type: &polars_arrow::datatypes::ArrowDataType,
) -> polars::prelude::DataType {
    polars::prelude::DataType::from_arrow(arrow_data_type, false, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct Person {
        name: String,
    }

    #[tokio::test]
    async fn test_decode_and_deserialize_json() -> Result<()> {
        let b64_data = serialize_and_encode(
            Person {
                name: "___".to_string(),
            },
            SerializationType::JSON,
        )?;

        let person: Person = decode_and_deserialize(&b64_data, SerializationType::JSON)?;

        assert_eq!(person.name, "___");

        Ok(())
    }

    #[tokio::test]
    async fn test_decode_and_deserialize_yaml() -> Result<()> {
        let b64_data = serialize_and_encode(
            Person {
                name: "___".to_string(),
            },
            SerializationType::YAML,
        )?;

        let person: Person = decode_and_deserialize(&b64_data, SerializationType::YAML)?;

        assert_eq!(person.name, "___");

        Ok(())
    }

    #[tokio::test]
    async fn test_random_base64() -> Result<()> {
        let _random_b64 = random_base64(32);

        Ok(())
    }

    #[tokio::test]
    async fn test_generate_token() -> Result<()> {
        let _token = generate_token(50);

        Ok(())
    }

    #[tokio::test]
    async fn test_extract_json_md() -> Result<()> {
        let tekst = r#"W oparciu o podane informacje, przygotowałem przykładowe pytanie Anny Kowalskiej, pasjonatki kulinarnej, skierowane do chatbota z funkcją obliczania miesięcznej raty spłaty kredytu.
    
    ```json
    {
      "message": "Anna Kowalska pytanie o ratę kredytu:",
      "question": {
        "name": "Anna Kowalska",
        " laughed_last": "2023-05-25 15:28:30",
        "case": "kulinarny kredyt",
        "loan_amount": 50000,
        "interest_rate": 0.05,
        "repayment_term": 5
      }
    }
    ```
    
    W tym przykładzie:
    - Pytanie zaczyna się od przedstawienia się użytkownika i kontekstu pytania (w tym prz
        "#;
        let json = extract_json(tekst)?;
        println!("{:?}", json);

        Ok(())
    }

    #[tokio::test]
    async fn test_extract_json() -> Result<()> {
        let tekst = r#"Na podstawie podanych informacji, przygotowałem przykładowe pytanie, które może zadać Pani Alicja Straczyńska-Błach, wykorzystując funkcję `search_book_reviews`:
    
     JSON:
    {
      "message": "Czy możesz polecić książki o wielkich kompozytorach muzyki klasycznej, szczególnie z okresu romantyzmu?"
    }
    
    W tym przykładzie:
    - **book_title** lub **book_author** zostały pominięte, ponieważ Pani Alicja nie szuka konkretnego tytułu ani autora, ale raczej ogólnego tematu.
    - Założyłem, że chce znaleźć książki związane z wielkimi kompozytorami muzyki klasycznej, a okres romantyzmu jest dla niej istotny ze względu na
    
        "#;
        let json = extract_json(tekst)?;
        println!("{:?}", json);

        Ok(())
    }
}
