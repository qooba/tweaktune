pub mod actors;
pub mod pipeline;
use anyhow::Result;
use opendal::services::Fs;
use opendal::Operator;
use std::fs::read;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::io::StreamReader;

#[tokio::main]
async fn main() -> Result<()> {
    let builder = Fs::default().root("/home/jovyan/SpeakLeash/swaggset/datasets/");
    let op: Operator = Operator::new(builder)?.finish();

    let reader = op.reader("persona_pl_full.jsonl").await?;

    // let ttt = reader.into_bytes_stream(1024..2048).await?;

    let bytes_stream = reader.into_bytes_stream(..).await?;
    let stream_reader = StreamReader::new(bytes_stream);
    let mut buf_reader = BufReader::new(stream_reader);
    let mut line = String::new();

    let mut i = 0;
    while buf_reader.read_line(&mut line).await? != 0 {
        i += 1;
        println!("{}", i);
        println!("{}", line.trim_end());
        line.clear();
    }

    //let text = reader.read(0..1024).await?.to_vec();
    //let t = std::str::from_utf8(&text)?;
    //println!("{}", t);
    //op.write("hello.txt", "Hello, World!").await?;

    Ok(())
}
