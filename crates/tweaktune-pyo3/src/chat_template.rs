use pyo3::prelude::*;
use tweaktune_core::templates::ChatTemplate;

#[pyclass]
pub struct ChatTemplateBuilder {
    template: String,
    tools: Option<String>,
    chat_template: Option<ChatTemplate>,
}

#[pymethods]
impl ChatTemplateBuilder {
    #[new]
    pub fn new(template: String) -> PyResult<Self> {
        Ok(ChatTemplateBuilder {
            template,
            tools: None,
            chat_template: None,
        })
    }

    pub fn with_tools(&mut self, tools: String) {
        self.tools = Some(tools);
    }

    fn build(&mut self) {
        let mut chat_template = ChatTemplate::new(self.template.clone());

        if let Some(tools) = &self.tools {
            chat_template = chat_template.with_tools(tools.clone());
        }

        self.chat_template = Some(chat_template);
    }

    pub fn render(&mut self, messages: String) -> PyResult<String> {
        if self.chat_template.is_none() {
            self.build();
        }

        self.chat_template
            .as_mut()
            .expect("Chat template not built")
            .render(messages)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to render chat template: {}",
                    e
                ))
            })
    }
}
