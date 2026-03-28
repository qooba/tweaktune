use super::data::DataManager;
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use tui_textarea::{Input, TextArea};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Normal,
    Edit,
    Command,
}

pub struct ExplorerApp<'a> {
    data_manager: DataManager,
    current_line: usize,
    current_message: usize,
    mode: Mode,
    text_area: TextArea<'a>,
    command_input: String,
    status_message: String,
    expanded_properties: bool,
    scroll_offset: usize,
}

impl<'a> ExplorerApp<'a> {
    pub fn new(data_manager: DataManager) -> Self {
        let mut text_area = TextArea::default();
        text_area.set_block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Edit Message "),
        );

        Self {
            data_manager,
            current_line: 0,
            current_message: 0,
            mode: Mode::Normal,
            text_area,
            command_input: String::new(),
            status_message: String::new(),
            expanded_properties: false,
            scroll_offset: 0,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let result = self.run_app(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    fn run_app<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        loop {
            terminal.draw(|f| self.ui(f))?;

            if let Event::Key(key) = event::read()? {
                match self.mode {
                    Mode::Normal => {
                        if self.handle_normal_mode(key)? {
                            break;
                        }
                    }
                    Mode::Edit => {
                        if self.handle_edit_mode(key)? {
                            // Exit edit mode
                            self.mode = Mode::Normal;
                        }
                    }
                    Mode::Command => {
                        if self.handle_command_mode(key)? {
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn handle_normal_mode(&mut self, key: event::KeyEvent) -> Result<bool> {
        match key.code {
            KeyCode::Char('q') => {
                return Ok(true);
            }
            KeyCode::Char(':') => {
                self.mode = Mode::Command;
                self.command_input.clear();
            }
            KeyCode::Char('i') | KeyCode::Char('I') => {
                self.enter_edit_mode();
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.current_message > 0 {
                    self.current_message -= 1;
                    self.adjust_scroll();
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if let Some(conv) = self.data_manager.get_conversation(self.current_line) {
                    if self.current_message + 1 < conv.messages.len() {
                        self.current_message += 1;
                        self.adjust_scroll();
                    }
                }
            }
            KeyCode::PageUp | KeyCode::Char('u')
                if key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            KeyCode::PageDown | KeyCode::Char('d')
                if key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if self.current_line > 0 {
                    self.current_line -= 1;
                    self.current_message = 0;
                    self.scroll_offset = 0;
                }
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if self.current_line + 1 < self.data_manager.len() {
                    self.current_line += 1;
                    self.current_message = 0;
                    self.scroll_offset = 0;
                }
            }
            KeyCode::Enter => {
                self.expanded_properties = !self.expanded_properties;
            }
            _ => {}
        }

        Ok(false)
    }

    fn handle_edit_mode(&mut self, key: event::KeyEvent) -> Result<bool> {
        match key.code {
            KeyCode::Esc => {
                self.mode = Mode::Normal;
                return Ok(true);
            }
            KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Ctrl+S: Save changes to file
                self.save_current_edit();
                match self.data_manager.save() {
                    Ok(_) => {
                        self.status_message = "Saved!".to_string();
                    }
                    Err(e) => {
                        self.status_message = format!("Error saving: {}", e);
                    }
                }
            }
            _ => {
                let input = Input::from(key);
                self.text_area.input(input);
            }
        }

        Ok(false)
    }

    fn handle_command_mode(&mut self, key: event::KeyEvent) -> Result<bool> {
        match key.code {
            KeyCode::Esc => {
                self.mode = Mode::Normal;
                self.command_input.clear();
            }
            KeyCode::Enter => {
                let result = self.execute_command();
                self.mode = Mode::Normal;
                if let Err(e) = result {
                    self.status_message = format!("Error: {}", e);
                }
                if self.command_input == "q" {
                    return Ok(true);
                }
                self.command_input.clear();
            }
            KeyCode::Char(c) => {
                self.command_input.push(c);
            }
            KeyCode::Backspace => {
                self.command_input.pop();
            }
            _ => {}
        }

        Ok(false)
    }

    fn execute_command(&mut self) -> Result<()> {
        match self.command_input.as_str() {
            "q" => {
                // Will be handled in handle_command_mode
            }
            _ => {
                self.status_message = format!("Unknown command: {}", self.command_input);
            }
        }

        Ok(())
    }

    fn enter_edit_mode(&mut self) {
        if let Some(conv) = self.data_manager.get_conversation(self.current_line) {
            if let Some(msg) = conv.messages.get(self.current_message) {
                self.text_area = TextArea::from(msg.content.lines());
                self.text_area.set_block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Edit Message (Ctrl+S to save, Esc to exit) "),
                );
                self.mode = Mode::Edit;
            }
        }
    }

    fn save_current_edit(&mut self) {
        let content = self.text_area.lines().join("\n");
        self.data_manager
            .update_message(self.current_line, self.current_message, content);
    }

    fn adjust_scroll(&mut self) {
        // Implement scroll adjustment if needed
    }

    fn ui(&self, f: &mut Frame) {
        let size = f.area();

        // Create main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header
                Constraint::Min(10),   // Main content
                Constraint::Length(3), // Status bar
            ])
            .split(size);

        // Header
        self.render_header(f, chunks[0]);

        // Main content
        if self.mode == Mode::Edit {
            f.render_widget(&self.text_area, chunks[1]);
        } else {
            self.render_main_content(f, chunks[1]);
        }

        // Status bar
        self.render_status_bar(f, chunks[2]);
    }

    fn render_header(&self, f: &mut Frame, area: Rect) {
        let title = format!(
            " Conversation {}/{} | Message {}/{} ",
            self.current_line + 1,
            self.data_manager.len(),
            self.current_message + 1,
            self.data_manager
                .get_conversation(self.current_line)
                .map(|c| c.messages.len())
                .unwrap_or(0)
        );

        let header = Paragraph::new(title).style(
            Style::default()
                .bg(Color::Blue)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );

        f.render_widget(header, area);
    }

    fn render_main_content(&self, f: &mut Frame, area: Rect) {
        if let Some(conv) = self.data_manager.get_conversation(self.current_line) {
            let chunks = if self.expanded_properties {
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                    .split(area)
            } else {
                Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Percentage(100)])
                    .split(area)
            };

            // Render messages
            self.render_messages(f, chunks[0], conv);

            // Render other properties if expanded
            if self.expanded_properties {
                self.render_properties(f, chunks[1], conv);
            }
        } else {
            let text = Paragraph::new("No data available")
                .block(Block::default().borders(Borders::ALL).title(" Messages "));
            f.render_widget(text, area);
        }
    }

    fn render_messages(&self, f: &mut Frame, area: Rect, conv: &super::data::ConversationData) {
        // Build text with all messages
        let mut lines = Vec::new();

        for (idx, msg) in conv.messages.iter().enumerate() {
            let role_color = match msg.role.as_str() {
                "user" => Color::Green,
                "assistant" => Color::Blue,
                "system" => Color::Yellow,
                _ => Color::White,
            };

            let is_current = idx == self.current_message;

            // Role header style: colored, bold, underlined
            let role_style = Style::default()
                .fg(role_color)
                .add_modifier(Modifier::BOLD | Modifier::UNDERLINED);

            // Content style: white text, no highlighting
            let content_style = Style::default().fg(Color::White);

            // Add a separator line before each message (except first)
            if idx > 0 {
                lines.push(Line::from(""));
            }

            // Add role header with marker for current message
            let marker = if is_current { "► " } else { "  " };
            lines.push(Line::from(vec![
                Span::styled(marker, Style::default().fg(role_color)),
                Span::styled(msg.role.to_uppercase(), role_style),
            ]));

            // Split content into lines and wrap them
            for content_line in msg.content.lines() {
                // Wrap long lines to fit the width (accounting for borders and padding)
                let max_width = area.width.saturating_sub(4) as usize;
                if content_line.chars().count() > max_width {
                    // Wrap the line
                    let words: Vec<&str> = content_line.split_whitespace().collect();
                    let mut current_line = String::new();

                    for word in words {
                        if current_line.chars().count() + word.chars().count() + 1 > max_width
                            && !current_line.is_empty()
                        {
                            lines.push(Line::from(Span::styled(
                                current_line.clone(),
                                content_style,
                            )));
                            current_line.clear();
                        }
                        if !current_line.is_empty() {
                            current_line.push(' ');
                        }
                        current_line.push_str(word);
                    }
                    if !current_line.is_empty() {
                        lines.push(Line::from(Span::styled(current_line, content_style)));
                    }
                } else {
                    lines.push(Line::from(Span::styled(
                        content_line.to_string(),
                        content_style,
                    )));
                }
            }
        }

        let paragraph = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Messages (↑/↓ to navigate, i to edit) "),
            )
            .wrap(Wrap { trim: false })
            .scroll((self.scroll_offset as u16, 0));

        f.render_widget(paragraph, area);
    }

    fn render_properties(&self, f: &mut Frame, area: Rect, conv: &super::data::ConversationData) {
        let json = serde_json::to_string_pretty(&conv.other_properties).unwrap_or_default();

        let paragraph = Paragraph::new(json)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Other Properties (Enter to collapse) "),
            )
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        let mode_text = match self.mode {
            Mode::Normal => "NORMAL",
            Mode::Edit => "EDIT",
            Mode::Command => "COMMAND",
        };

        let status_text = if self.mode == Mode::Command {
            format!(":{}", self.command_input)
        } else if !self.status_message.is_empty() {
            self.status_message.clone()
        } else {
            "q: quit | ←/→: lines | ↑/↓: msgs | Ctrl+U/D: scroll | i: edit | Ctrl+S: save (in edit) | Enter: expand"
                .to_string()
        };

        let status = Paragraph::new(vec![Line::from(vec![
            Span::styled(
                format!(" {} ", mode_text),
                Style::default()
                    .fg(Color::Black)
                    .bg(match self.mode {
                        Mode::Normal => Color::Blue,
                        Mode::Edit => Color::Green,
                        Mode::Command => Color::Yellow,
                    })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::raw(status_text),
        ])])
        .block(Block::default().borders(Borders::ALL));

        f.render_widget(status, area);
    }
}

/// Run the explorer TUI
pub fn run_explorer(file_path: &str) -> Result<()> {
    let data_manager = DataManager::load(file_path)?;
    let mut app = ExplorerApp::new(data_manager);
    app.run()
}
