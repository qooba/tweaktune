use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// A single message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Represents a single JSONL line with messages and other properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationData {
    pub messages: Vec<Message>,
    #[serde(flatten)]
    pub other_properties: HashMap<String, Value>,
}

/// Manages the JSONL file and its data
pub struct DataManager {
    file_path: PathBuf,
    lines: Vec<ConversationData>,
}

impl DataManager {
    /// Load JSONL file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path)
            .context(format!("Failed to open file: {}", file_path.display()))?;

        let reader = BufReader::new(file);
        let mut lines = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.context(format!("Failed to read line {}", line_num + 1))?;

            if line.trim().is_empty() {
                continue;
            }

            let data: ConversationData = serde_json::from_str(&line)
                .context(format!("Failed to parse JSON at line {}", line_num + 1))?;

            lines.push(data);
        }

        Ok(Self { file_path, lines })
    }

    /// Get all conversations
    pub fn get_conversations(&self) -> &[ConversationData] {
        &self.lines
    }

    /// Get a specific conversation
    pub fn get_conversation(&self, index: usize) -> Option<&ConversationData> {
        self.lines.get(index)
    }

    /// Update a message in a conversation
    pub fn update_message(&mut self, line_index: usize, message_index: usize, new_content: String) {
        if let Some(conv) = self.lines.get_mut(line_index) {
            if let Some(msg) = conv.messages.get_mut(message_index) {
                msg.content = new_content;
            }
        }
    }

    /// Save all data back to file with backup
    pub fn save(&self) -> Result<()> {
        // Create backup
        let backup_path = self.file_path.with_extension("jsonl.bak");
        std::fs::copy(&self.file_path, &backup_path).context("Failed to create backup")?;

        // Write to file
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.file_path)
            .context("Failed to open file for writing")?;

        for line in &self.lines {
            let json = serde_json::to_string(line).context("Failed to serialize line")?;
            writeln!(file, "{}", json).context("Failed to write line")?;
        }

        file.flush().context("Failed to flush file")?;

        Ok(())
    }

    /// Get number of conversations
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}
