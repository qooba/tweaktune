use crate::common::enter_runtime;
use crate::config::read_config_str;
use anyhow::Result;
use opendal::blocking::{Operator, StdReader};
use opendal::services::{AzblobConfig, FsConfig, GcsConfig, HttpConfig, S3Config};
use opendal::Operator as AsyncOperator;
use serde::Deserialize;
use std::io::Read;
use std::path::Path;

pub struct OpReader {
    pub inner: StdReader,
    // pub content_length: u64,
}

impl OpReader {
    pub fn new(reader: StdReader) -> Self {
        Self {
            inner: reader,
            // content_length,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::large_enum_variant)]
pub enum OpConfig {
    Fs(FsConfig),
    S3(S3Config),
    Gcs(GcsConfig),
    Azblob(AzblobConfig),
    Http(HttpConfig),
}

#[allow(dead_code, unreachable_code, clippy::field_reassign_with_default)]
pub fn path_to_operator(path: &str) -> Result<OpConfig> {
    if path.starts_with("s3://") {
        let mut config = S3Config::default();
        config.bucket = path.trim_start_matches("s3://").to_string();
        todo!("S3 configuration is not fully implemented yet");
        // Ok(OpConfig::S3(config))
    } else if path.starts_with("gcs://") {
        let mut config = GcsConfig::default();
        config.bucket = path.trim_start_matches("gcs://").to_string();
        todo!("GCS configuration is not fully implemented yet");
        // Ok(OpConfig::Gcs(config))
    } else if path.starts_with("azblob://") {
        let mut config = AzblobConfig::default();
        config.container = path.trim_start_matches("azblob://").to_string();
        todo!("Azure Blob Storage configuration is not fully implemented yet");
        // Ok(OpConfig::Azblob(config))
    } else if path.starts_with("http://") || path.starts_with("https://") {
        todo!("HTTP configuration is not fully implemented yet");
        let mut config = HttpConfig::default();
        let url = url::Url::parse(path)?;
        if let Some(host) = url.host_str() {
            if let Some(p) = url.path().strip_prefix('/') {
                if let Some(pos) = p.rfind('/') {
                    let p = &p[..pos];
                    config.endpoint = Some(format!("{}://{}/{}/", url.scheme(), host, p));
                }
            }
        }

        // if let Some(pos) = path.rfind('/') {
        //     println!("ENDPOINT: {}", &path[..pos]);
        //     config.endpoint = Some(path[..pos].to_string());
        // }
        Ok(OpConfig::Http(config))
    } else {
        let p = Path::new(path);
        let dir = p.parent().unwrap().to_str().unwrap();
        let mut config = FsConfig::default();
        config.root = Some(dir.to_string());
        Ok(OpConfig::Fs(config))
    }
}

pub fn build_reader(path: &str, op_config: Option<String>) -> Result<OpReader> {
    let p = Path::new(path);
    // let dir = p.parent().unwrap().to_str().unwrap();
    let file_name = p.file_name().unwrap().to_str().unwrap();
    // TODO: implement the other operators
    // let builder = Fs::default().root(dir);
    // let json_config = format!("{{\"type\":\"Fs\", \"root\": \"{}\"}}", dir);
    let operator = match op_config {
        Some(config) => {
            let op_config: OpConfig = serde_json::from_str(&config)?;
            match op_config {
                OpConfig::Fs(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::S3(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::Gcs(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::Azblob(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::Http(config) => AsyncOperator::from_config(config)?.finish(),
            }
        }
        None => {
            let op_config = path_to_operator(path)?;
            match op_config {
                OpConfig::Fs(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::S3(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::Gcs(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::Azblob(config) => AsyncOperator::from_config(config)?.finish(),
                OpConfig::Http(config) => AsyncOperator::from_config(config)?.finish(),
            }
            // let builder = Fs::default().root(dir);
            // Operator::new(builder)?.finish()
        }
    };

    let _guard = enter_runtime();
    let op = Operator::new(operator)?;
    // let content_length = op.stat(file_name)?.content_length();
    let reader = op.reader(file_name)?.into_std_read(..)?;
    Ok(OpReader {
        inner: reader,
        // content_length,
    })
}

pub fn read_to_string(path: &str, op_config: Option<String>) -> Result<String> {
    if path.starts_with("http://") || path.starts_with("https://") {
        read_config_str(&path.to_string(), None)
    } else {
        let mut reader = build_reader(path, op_config)?;
        let mut content = String::new();
        reader.inner.read_to_string(&mut content)?;
        Ok(content)
    }
}
