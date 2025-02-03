/*use crate::steps::Step;
use futures::stream::StreamExt;
use opendal::Operator;
use std::sync::Arc;

struct Pipeline {
    steps: Vec<Box<dyn Step>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline { steps: Vec::new() }
    }

    pub fn add_stage(&mut self, stage: Box<dyn Step>) {
        self.steps.push(stage);
    }

    pub fn run(&self) {
        for stage in &self.steps {
            stage.load().unwrap();
        }
    }
}
*/
