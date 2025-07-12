use std::io::Write;
use std::sync::{mpsc, Mutex};

pub struct ChannelWriter {
    pub sender: mpsc::Sender<String>,
    buffer: Mutex<String>,
}

impl ChannelWriter {
    pub fn new(sender: mpsc::Sender<String>) -> Self {
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
