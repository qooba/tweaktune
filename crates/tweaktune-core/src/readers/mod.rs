use anyhow::Result;
use opendal::services::{AzblobConfig, Fs, FsConfig, GcsConfig, HttpConfig, S3Config};
use opendal::Operator;
use opendal::StdReader;
use serde::Deserialize;
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

pub fn build_reader(path: &str, op_config: Option<String>) -> Result<OpReader> {
    let p = Path::new(path);
    let dir = p.parent().unwrap().to_str().unwrap();
    let file_name = p.file_name().unwrap().to_str().unwrap();
    // let builder = Fs::default().root(dir);
    // let json_config = format!("{{\"type\":\"Fs\", \"root\": \"{}\"}}", dir);
    let operator = match op_config {
        Some(config) => {
            let op_config: OpConfig = serde_json::from_str(&config)?;
            match op_config {
                OpConfig::Fs(config) => Operator::from_config(config)?.finish(),
                OpConfig::S3(config) => Operator::from_config(config)?.finish(),
                OpConfig::Gcs(config) => Operator::from_config(config)?.finish(),
                OpConfig::Azblob(config) => Operator::from_config(config)?.finish(),
                OpConfig::Http(config) => Operator::from_config(config)?.finish(),
            }
        }
        None => {
            let builder = Fs::default().root(dir);
            Operator::new(builder)?.finish()
        }
    };

    let op = operator.blocking();
    // let content_length = op.stat(file_name)?.content_length();
    let reader = op.reader(file_name)?.into_std_read(..)?;
    Ok(OpReader {
        inner: reader,
        // content_length,
    })
}
