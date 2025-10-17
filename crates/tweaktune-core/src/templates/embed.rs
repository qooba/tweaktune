use include_dir::{include_dir, Dir};

pub static TEMPLATES_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/templates");

pub enum ChateTemplate {
    Bielik,
}

pub fn chat_templates(name: &str) -> Option<&'static str> {
    TEMPLATES_DIR
        .get_file(format!("chat_templates/{name}.j2"))
        .and_then(|f| f.contents_utf8())
}

pub fn conversation_templates(name: &str) -> Option<&'static str> {
    TEMPLATES_DIR
        .get_file(format!("conversations/{name}.j2"))
        .and_then(|f| f.contents_utf8())
}

pub fn judge_templates(name: &str, language: &str) -> Option<&'static str> {
    TEMPLATES_DIR
        .get_file(format!("judges/{name}_{language}.j2"))
        .and_then(|f| f.contents_utf8())
}
