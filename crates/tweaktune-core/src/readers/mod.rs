use anyhow::Result;
use opendal::services::Fs;
use opendal::Operator;
use opendal::StdReader;
use std::path::Path;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::io::StreamReader;

pub fn read_file_with_opendal(path: &str) -> Result<StdReader> {
    let p = Path::new(path);
    let dir = p.parent().unwrap().to_str().unwrap();

    let builder = Fs::default().root(dir);
    let operator: Operator = Operator::new(builder)?.finish();

    let op = operator.blocking();
    let r = op.reader(path)?.into_std_read(..)?;
    Ok(r)
}

pub struct JsonlReader {
    pub path: String,
}

impl JsonlReader {
    pub async fn load(&self) -> Result<Vec<String>> {
        let path = Path::new(&self.path);
        let dir = path.parent().unwrap().to_str().unwrap();
        let file = path.file_name().unwrap().to_str().unwrap();

        let builder = Fs::default().root(dir);
        let op: Operator = Operator::new(builder)?.finish();
        let reader = op.reader(file).await?;

        let bytes_stream = reader.into_bytes_stream(..).await?;
        let stream_reader = StreamReader::new(bytes_stream);
        let mut buf_reader = BufReader::new(stream_reader);

        let mut line = String::new();

        let mut lines = Vec::new();
        while buf_reader.read_line(&mut line).await? != 0 {
            lines.push(line.trim_end().to_string());
            line.clear();
        }

        Ok(lines)
    }
}
