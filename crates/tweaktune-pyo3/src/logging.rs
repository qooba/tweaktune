use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, ContentArrangement, Table};
use log::{Level, Log, Metadata, Record};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use simplelog::{Config, LevelFilter, SharedLogger};
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

#[derive(Debug, Clone)]
pub struct CollectedLogEntry {
    pub level: Level,
    pub target: String,
    pub message: String,
}

/// LogsCollector collects all log messages in-memory and can render a
/// summary table using comfy-table.
#[derive(Clone)]
pub struct LogsCollector {
    entries: Arc<Mutex<Vec<CollectedLogEntry>>>,
}

impl LogsCollector {
    pub fn new() -> Self {
        LogsCollector {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a clone of the internal Arc so the collector can be shared
    /// with other parts of the program (for example to register as a logger).
    pub fn shared(&self) -> Arc<Mutex<Vec<CollectedLogEntry>>> {
        self.entries.clone()
    }

    /// Build a comfy-table summary string with counts grouped by (level, message).
    /// It shows total counts, counts per level, and top messages.
    pub fn summary_table(&self) -> String {
        let entries = self.entries.lock().unwrap();

        // Build counts grouped by (level, message)
        use std::collections::BTreeMap;

        let mut grouped: BTreeMap<(String, String), usize> = BTreeMap::new();
        let mut total = 0usize;
        let mut per_level: std::collections::BTreeMap<String, usize> = BTreeMap::new();

        for e in entries.iter() {
            if e.level > Level::Info {
                continue;
            }

            total += 1;
            let key = (format!("{:?}", e.level), e.message.clone());
            *grouped.entry(key).or_insert(0) += 1;
            *per_level.entry(format!("{:?}", e.level)).or_insert(0) += 1;
        }

        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            // .set_width(200)
            .set_content_arrangement(ContentArrangement::Dynamic);
        table.set_header(vec![
            Cell::from("Level"),
            Cell::from("Message"),
            Cell::from("Count"),
        ]);

        // Sort grouped by count desc
        let mut items: Vec<((String, String), usize)> = grouped.into_iter().collect();
        items.sort_by(|a, b| b.1.cmp(&a.1));

        for ((level, msg), count) in items.iter() {
            table.add_row(vec![
                Cell::from(level.clone()),
                Cell::from(msg.clone()),
                Cell::from(count.to_string()),
            ]);
        }

        // Summary table
        let mut summary = Table::new();
        summary.set_header(vec![Cell::from("Metric"), Cell::from("Value")]);
        summary.add_row(vec![Cell::from("Total"), Cell::from(total.to_string())]);
        for (lvl, cnt) in per_level.iter() {
            summary.add_row(vec![Cell::from(lvl.clone()), Cell::from(cnt.to_string())]);
        }

        let mut out = String::new();
        // out.push_str("Summary:\n");
        // out.push_str(&summary.to_string());
        // out.push_str("\nDetails:\n");
        out.push_str(&table.to_string());
        out
    }
}

impl Log for LogsCollector {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let entry = CollectedLogEntry {
            level: record.level(),
            target: record.target().to_string(),
            message: record.args().to_string(),
        };

        if let Ok(mut guard) = self.entries.lock() {
            guard.push(entry);
        }
    }

    fn flush(&self) {}
}

impl SharedLogger for LogsCollector {
    fn level(&self) -> LevelFilter {
        LevelFilter::Info
    }

    fn config(&self) -> Option<&Config> {
        None
    }

    fn as_log(self: Box<Self>) -> Box<dyn Log> {
        self
    }
}
