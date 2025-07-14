use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::sync::{mpsc, Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusEvent {
    pub event_type: String,
    pub data: Value,
}

impl BusEvent {
    pub fn build(event_type: &str, data: Value) -> String {
        serde_json::to_string(&BusEvent {
            event_type: event_type.to_string(),
            data,
        })
        .unwrap()
    }
}

pub struct ChannelWriter {
    pub sender: Arc<mpsc::Sender<String>>,
    buffer: Mutex<String>,
}

impl ChannelWriter {
    pub fn new(sender: Arc<mpsc::Sender<String>>) -> Self {
        ChannelWriter {
            sender,
            buffer: Mutex::new(String::new()),
        }
    }
}

impl Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.push_str(&String::from_utf8_lossy(buf));
        while let Some(pos) = buffer.find('\n') {
            let line = buffer.drain(..=pos).collect::<String>();
            let line = BusEvent::build("log", Value::String(line));
            self.sender
                .send(line)
                .map_err(|_| std::io::Error::other("Failed to send message"))?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
