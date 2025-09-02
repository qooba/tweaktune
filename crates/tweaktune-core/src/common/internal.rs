use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use log::error;
use once_cell::sync::Lazy;
use once_cell::sync::OnceCell;
use polars::prelude::*;
use rand::distr::Alphanumeric;
use rand::{Rng, RngCore};
use regex::Regex;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::io::{self, Result as IoResult};
use std::{env, fs, path::PathBuf};
use tokio::runtime::Runtime;

static RUNTIME: Lazy<Runtime> = Lazy::new(|| Runtime::new().unwrap());

pub fn enter_runtime() -> tokio::runtime::EnterGuard<'static> {
    RUNTIME.enter()
}

pub fn run_async<T: std::future::Future>(fut: T) -> T::Output {
    RUNTIME.block_on(fut)
}

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
        io::Error::other(format!("{error:?}"))
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
        self.map_err(|e| io::Error::other(format!("{e:?}")))
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

pub fn deserialize<T>(data: &str, serialization_type: SerializationType) -> Result<T>
where
    T: DeserializeOwned,
{
    let object = match serialization_type {
        SerializationType::JSON => serde_json::from_str(data)?,
        SerializationType::YAML => serde_yaml::from_str(data)?,
    };

    Ok(object)
}

pub fn serialize<T>(object: T, serialization_type: SerializationType) -> Result<String>
where
    T: serde::Serialize,
{
    let object_str = match serialization_type {
        SerializationType::JSON => serde_json::to_string(&object)?,
        SerializationType::YAML => serde_yaml::to_string(&object)?,
    };

    Ok(object_str)
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
                    error!(target: "extract_json", "🐔 EXTRACT JSON {}", &text);
                    return Err(anyhow!("Failed to extract JSON: {}", e));
                }
            },
        },
    };
    Ok(value)
}

pub fn df_to_values(df: &DataFrame) -> Result<Vec<Value>> {
    let mut buffer = Vec::new();
    let mut df = df.clone();
    JsonWriter::new(&mut buffer)
        .with_json_format(JsonFormat::Json)
        .finish(&mut df)
        .unwrap();
    let json_str = String::from_utf8(buffer.clone())
        .map_err(|e| anyhow!("Failed to convert DataFrame to string: {}", e))?;

    let df_json = serde_json::from_str(&json_str)
        .map_err(|e| anyhow!("Failed to parse DataFrame to JSON: {}", e))?;

    if let Value::Array(arr) = df_json {
        Ok(arr)
    } else {
        Err(anyhow!("DataFrame JSON is not an array"))
    }
}

pub fn create_rows_stream(df: &DataFrame) -> Result<impl Iterator<Item = Result<Value>> + '_> {
    let height = df.height();
    Ok((0..height).map(move |idx| {
        let row = df.slice(idx as i64, 1);
        let values = df_to_values(&row)?;
        Ok(values[0].clone())
    }))
}

// Parse simple python function definitions from a string and convert them to
// JSON-schema-like objects compatible with the `function_to_schema` output
// used on the Python side. This implementation is intentionally conservative
// and only understands simple annotated parameters and Field(..., description=..)
// style defaults. It returns a JSON array where each item represents a
// function with `type`, `name`, `description`, `parameters`, and `strict`.
pub fn python_functions_to_schemas(code: &str) -> Result<Value> {
    // Simple line-based parser: find lines starting with `def `, extract the
    // parameter block (handles parentheses across lines), and optionally the
    // following docstring (triple quotes). This is more robust than a single
    // regex for inputs with embedded commas in defaults.
    let mut functions = Vec::new();
    let lines: Vec<&str> = code.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        if trimmed.starts_with("def ") {
            // capture function name
            if let Some(name_end) = trimmed[4..].find('(') {
                let name = trimmed[4..4 + name_end]
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_string();

                // capture params across lines until matching ')'
                // include parentheses in the captured string so nested
                // parentheses are visible to the later depth-aware splitter
                let mut params = String::new();
                let mut depth = 0i32;
                let mut started = false;
                // iterate characters from current line pos
                let start_idx = trimmed.find('(').unwrap_or(0);
                for ch in trimmed.chars().skip(start_idx) {
                    if ch == '(' {
                        depth += 1;
                        started = true;
                        // keep the '('
                        params.push(ch);
                        continue;
                    }
                    if ch == ')' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                        // keep the ')'
                        params.push(ch);
                        continue;
                    }
                    if started {
                        params.push(ch);
                    }
                }
                // if depth still > 0, continue across following lines
                if depth > 0 {
                    let mut j = i + 1;
                    while j < lines.len() && depth > 0 {
                        for ch in lines[j].chars() {
                            if ch == '(' {
                                depth += 1;
                            }
                            if ch == ')' {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            params.push(ch);
                        }
                        j += 1;
                    }
                }

                // attempt to capture docstring on following non-empty line
                let mut doc = String::new();
                let mut k = i + 1;
                while k < lines.len() && lines[k].trim().is_empty() {
                    k += 1;
                }
                if k < lines.len() {
                    let s = lines[k].trim_start();
                    if s.starts_with("\"\"\"") || s.starts_with("'''") {
                        let quote = &s[..3];
                        // capture until closing quote occurs
                        let mut collected = String::new();
                        // include the rest of the line after opening quotes
                        collected.push_str(&s[3..]);
                        let mut closed = s[3..].contains(quote);
                        let mut l = k + 1;
                        while l < lines.len() && !closed {
                            let ln = lines[l];
                            if ln.contains(quote) {
                                let idx = ln.find(quote).unwrap();
                                collected.push_str(&ln[..idx]);
                                closed = true;
                                break;
                            } else {
                                collected.push_str(ln);
                                collected.push('\n');
                            }
                            l += 1;
                        }
                        doc = collected;
                    }
                }

                // build properties map
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();

                // split params by commas at the outermost parameter level
                // we included the outer '(' so the function parameter level is depth==1
                let mut parts = Vec::new();
                let mut cur = String::new();
                let mut d = 0i32;
                for ch in params.chars() {
                    match ch {
                        '(' => {
                            d += 1;
                            cur.push(ch);
                        }
                        ')' => {
                            d -= 1;
                            cur.push(ch);
                        }
                        ',' if d == 1 => {
                            parts.push(cur.trim().to_string());
                            cur.clear();
                        }
                        c => cur.push(c),
                    }
                }
                if !cur.trim().is_empty() {
                    parts.push(cur.trim().to_string());
                }

                for part in parts.iter() {
                    if part.trim().is_empty() {
                        continue;
                    }
                    // trim surrounding parentheses that may remain from the outer capture
                    let part = part
                        .trim()
                        .trim_start_matches('(')
                        .trim_end_matches(')')
                        .trim();
                    // parse name:type = default
                    let (left, default_opt) = match part.find('=') {
                        Some(idx) => (part[..idx].trim(), Some(part[idx + 1..].trim())),
                        None => (part.trim(), None),
                    };
                    let (pname, ptype) = match left.find(':') {
                        Some(idx) => (left[..idx].trim().to_string(), left[idx + 1..].trim()),
                        None => (left.trim().to_string(), "Any"),
                    };

                    let mut prop = serde_json::Map::new();
                    let p_json_type = match ptype {
                        "int" => json!("integer"),
                        "float" => json!("number"),
                        "str" | "string" => json!("string"),
                        "bool" => json!("boolean"),
                        "dict" | "Dict" => json!({"type":"object"}),
                        "list" | "List" | "[]" => json!({"type":"array"}),
                        _ => json!("string"),
                    };
                    if p_json_type.is_object() {
                        if let Some(map) = p_json_type.as_object() {
                            for (k, v) in map.iter() {
                                prop.insert(k.clone(), v.clone());
                            }
                        }
                    } else {
                        prop.insert("type".to_string(), p_json_type);
                    }

                    if let Some(def) = default_opt {
                        if def.contains("description") {
                            if let Some(c) = Regex::new(r#"description\s*=\s*"([^"]*)""#)
                                .ok()
                                .and_then(|re| re.captures(def))
                            {
                                if let Some(m) = c.get(1) {
                                    prop.insert("description".to_string(), json!(m.as_str()));
                                }
                            } else if let Some(c) = Regex::new(r#"description\s*=\s*'([^']*)'"#)
                                .ok()
                                .and_then(|re| re.captures(def))
                            {
                                if let Some(m) = c.get(1) {
                                    prop.insert("description".to_string(), json!(m.as_str()));
                                }
                            }
                        }
                        let def_trim = def.trim();
                        if def_trim.starts_with("Field(") && def_trim.contains("...") {
                            required.push(pname.clone());
                        }
                    } else {
                        required.push(pname.clone());
                    }

                    properties.insert(pname, Value::Object(prop));
                }

                let mut params_obj = serde_json::Map::new();
                params_obj.insert("type".to_string(), json!("json_schema"));
                params_obj.insert("name".to_string(), json!(format!("{}_parameters", name)));
                let mut schema_obj = serde_json::Map::new();
                schema_obj.insert("type".to_string(), json!("object"));
                schema_obj.insert("properties".to_string(), Value::Object(properties));
                schema_obj.insert("additionalProperties".to_string(), json!(false));
                if !required.is_empty() {
                    schema_obj.insert(
                        "required".to_string(),
                        Value::Array(required.into_iter().map(|r| json!(r)).collect()),
                    );
                }
                params_obj.insert("schema".to_string(), Value::Object(schema_obj));

                let func_obj = json!({
                    "type": "function",
                    "name": name,
                    "description": if doc.is_empty() { Value::Null } else { json!(doc.trim()) },
                    "parameters": Value::Object(params_obj),
                    "strict": true
                });
                functions.push(func_obj);
            }
        }
        i += 1;
    }

    Ok(Value::Array(functions))
}

// ...existing code...

#[allow(dead_code)]
fn arrow_data_type_to_polars_data_type(
    arrow_data_type: &polars_arrow::datatypes::ArrowDataType,
) -> polars::prelude::DataType {
    polars::prelude::DataType::from_arrow(arrow_data_type, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let code = r#"
def add(a: int, b: int):
    """Add two integers"""
    return a + b
"#;

        let res = python_functions_to_schemas(code).unwrap();
        // expect one function in array
        let arr = res.as_array().expect("expected array");
        assert_eq!(arr.len(), 1);
        let func = &arr[0];
        assert_eq!(func["type"], "function");
        assert_eq!(func["name"], "add");
        // description should be present
        assert!(func.get("description").is_some());
        // parameters schema
        let params = &func["parameters"];
        assert_eq!(params["type"], "json_schema");
        let schema = &params["schema"];
        assert_eq!(schema["type"], "object");
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("a"));
        assert!(props.contains_key("b"));
        // required contains both
        let required = schema["required"].as_array().unwrap();
        let reqs: Vec<String> = required
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(reqs.contains(&"a".to_string()) && reqs.contains(&"b".to_string()));
    }

    #[test]
    fn test_field_description_and_default() {
        let code = r#"
def greet(name: str = Field(..., description="user name"), excited: bool = False):
    '''Greet a user'''
    pass
"#;

        let res = python_functions_to_schemas(code).unwrap();
        let arr = res.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let func = &arr[0];

        assert_eq!(func["name"], "greet");
        let params = &func["parameters"];
        let schema = &params["schema"];

        let props = schema["properties"].as_object().unwrap();
        // name should have description
        let name_prop = props.get("name").unwrap();
        assert_eq!(name_prop["type"], "string");
        assert_eq!(name_prop["description"], "user name");
        // name used Field(..., ...) -> should be required; excited has default -> not required
        let required = schema.get("required").and_then(|v| v.as_array());
        let reqs: Vec<String> = required
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default();
        assert!(reqs.contains(&"name".to_string()));
        assert!(!reqs.contains(&"excited".to_string()));
    }
}
