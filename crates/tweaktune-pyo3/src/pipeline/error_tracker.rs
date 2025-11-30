use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Color, ContentArrangement, Table};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct ErrorInfo {
    pub step_name: String,
    pub error_message: String,
    pub iteration: usize,
}

#[derive(Clone)]
pub struct ErrorTracker {
    errors: Arc<Mutex<Vec<ErrorInfo>>>,
    consecutive_failures: Arc<Mutex<HashMap<String, usize>>>,
    max_consecutive_before_stop: usize,
}

impl ErrorTracker {
    pub fn new(max_consecutive_before_stop: usize) -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
            consecutive_failures: Arc::new(Mutex::new(HashMap::new())),
            max_consecutive_before_stop,
        }
    }

    /// Track an error that occurred during step processing
    pub fn track_error(&self, step_name: String, error_message: String, iteration: usize) {
        // Store the error
        if let Ok(mut errors) = self.errors.lock() {
            errors.push(ErrorInfo {
                step_name: step_name.clone(),
                error_message: error_message.clone(),
                iteration,
            });
        }

        // Track consecutive failures for this specific error
        let error_key = format!("{}::{}", step_name, error_message);
        if let Ok(mut consecutive) = self.consecutive_failures.lock() {
            let count = consecutive.entry(error_key).or_insert(0);
            *count += 1;
        }
    }

    /// Reset consecutive failure count (called when an iteration succeeds)
    pub fn reset_consecutive(&self) {
        if let Ok(mut consecutive) = self.consecutive_failures.lock() {
            consecutive.clear();
        }
    }

    /// Check if we should stop execution due to configuration errors
    /// Returns Some(error_info) if we should stop, None otherwise
    pub fn should_stop(&self) -> Option<(String, String, usize)> {
        if let Ok(consecutive) = self.consecutive_failures.lock() {
            for (key, count) in consecutive.iter() {
                if *count >= self.max_consecutive_before_stop {
                    // Extract step_name and error_message from key
                    if let Some((step_name, error_message)) = key.split_once("::") {
                        return Some((step_name.to_string(), error_message.to_string(), *count));
                    }
                }
            }
        }
        None
    }

    /// Get total error count
    pub fn error_count(&self) -> usize {
        if let Ok(errors) = self.errors.lock() {
            errors.len()
        } else {
            0
        }
    }

    /// Generate a detailed error summary table
    pub fn generate_summary(&self) -> String {
        let errors = if let Ok(errors) = self.errors.lock() {
            errors.clone()
        } else {
            return String::new();
        };

        if errors.is_empty() {
            return String::new();
        }

        // Group errors by (step_name, error_message)
        let mut error_groups: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for error in &errors {
            error_groups
                .entry((error.step_name.clone(), error.error_message.clone()))
                .or_default()
                .push(error.iteration);
        }

        let mut output = String::new();
        output.push('\n');

        // Header table
        let mut header_table = Table::new();
        header_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);
        header_table.add_row(vec![Cell::from("  ⚠️  ERROR SUMMARY")]);
        output.push_str(&header_table.to_string());
        output.push('\n');

        // Create errors table
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        table.set_header(vec![
            Cell::from("Step").fg(Color::Cyan),
            Cell::from("Error").fg(Color::Cyan),
            Cell::from("Count").fg(Color::Cyan),
            Cell::from("Iterations").fg(Color::Cyan),
        ]);

        // Sort by count (descending)
        let mut sorted_groups: Vec<_> = error_groups.iter().collect();
        sorted_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        for ((step_name, error_message), iterations) in sorted_groups {
            let count = iterations.len();
            let iterations_str = if count <= 5 {
                iterations
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                format!(
                    "{} - {} (+ {} more)",
                    iterations[0],
                    iterations[1],
                    count - 2
                )
            };

            let error_display = if error_message.len() > 50 {
                format!("{}...", &error_message[..47])
            } else {
                error_message.clone()
            };

            table.add_row(vec![
                Cell::from(step_name),
                Cell::from(error_display).fg(Color::Red),
                Cell::from(count.to_string()).fg(Color::Yellow),
                Cell::from(iterations_str),
            ]);
        }

        output.push_str(&table.to_string());
        output.push_str(&format!(
            "\nTotal errors: {} across {} unique error types\n",
            errors.len(),
            error_groups.len()
        ));

        output
    }

    /// Generate a configuration error alert (when stopping early)
    pub fn generate_config_error_alert(
        &self,
        step_name: &str,
        error_message: &str,
        count: usize,
    ) -> String {
        let mut output = String::new();
        output.push('\n');

        // Header table
        let mut header_table = Table::new();
        header_table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);
        header_table.add_row(vec![Cell::from("  🛑 CONFIGURATION ERROR DETECTED")]);
        output.push_str(&header_table.to_string());
        output.push('\n');

        // Error details table
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic);

        table.add_row(vec![
            Cell::from("Step").fg(Color::Cyan),
            Cell::from(step_name),
        ]);

        let error_display = if error_message.len() > 60 {
            format!("{}...", &error_message[..57])
        } else {
            error_message.to_string()
        };

        table.add_row(vec![
            Cell::from("Error").fg(Color::Cyan),
            Cell::from(error_display).fg(Color::Red),
        ]);
        table.add_row(vec![
            Cell::from("Pattern").fg(Color::Cyan),
            Cell::from(format!("Failed in {}/{} iterations (100%)", count, count))
                .fg(Color::Yellow),
        ]);

        output.push_str(&table.to_string());
        output.push_str(
            "\nThis appears to be a configuration issue that will affect all iterations.\n",
        );
        output.push_str("Pipeline execution stopped to prevent wasted iterations.\n\n");
        output.push_str("Common causes:\n");
        output.push_str("  • Missing or incorrectly named templates, datasets, or LLMs\n");
        output.push_str("  • Template syntax errors\n");
        output.push_str("  • Invalid JSON schemas\n");
        output.push_str("  • Missing required fields in context\n\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_tracking() {
        let tracker = ErrorTracker::new(3);

        // Track same error 3 times
        tracker.track_error("step1".to_string(), "Error A".to_string(), 0);
        tracker.track_error("step1".to_string(), "Error A".to_string(), 1);
        tracker.track_error("step1".to_string(), "Error A".to_string(), 2);

        assert!(tracker.should_stop().is_some());
        assert_eq!(tracker.error_count(), 3);
    }

    #[test]
    fn test_different_errors_dont_trigger_stop() {
        let tracker = ErrorTracker::new(3);

        tracker.track_error("step1".to_string(), "Error A".to_string(), 0);
        tracker.track_error("step1".to_string(), "Error B".to_string(), 1);
        tracker.track_error("step1".to_string(), "Error C".to_string(), 2);

        assert!(tracker.should_stop().is_none());
    }

    #[test]
    fn test_reset_consecutive() {
        let tracker = ErrorTracker::new(3);

        tracker.track_error("step1".to_string(), "Error A".to_string(), 0);
        tracker.track_error("step1".to_string(), "Error A".to_string(), 1);
        tracker.reset_consecutive();
        tracker.track_error("step1".to_string(), "Error A".to_string(), 2);

        assert!(tracker.should_stop().is_none());
    }
}
