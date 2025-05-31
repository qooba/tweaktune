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
    let mut buffer = Vec::new();
    let mut df = df.clone();
    JsonWriter::new(&mut buffer)
        .with_json_format(JsonFormat::Json)
        .finish(&mut df)
        .unwrap();
    let json_str = String::from_utf8(buffer.clone())
        .map_err(|e| anyhow!("Failed to convert DataFrame to string: {}", e))?;

    println!("DATAFRAME JSON STR: {}", json_str);

    let df_json = serde_json::from_str(&json_str)
        .map_err(|e| anyhow!("Failed to parse DataFrame to JSON: {}", e))?;

    println!("DATAFRAME JSON: {}", df_json);
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
