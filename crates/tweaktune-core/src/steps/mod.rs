use anyhow::Result;
use std::collections::HashMap;
use tweaktune_abstractions::EntityValue;

pub trait Step {
    fn load(&self) -> Result<()>;
}

pub trait ProcessingStep: Step {
    fn process(&self, input: StepInput) -> Result<StepOutput>;
    fn inputs(&self) -> Vec<String>;
    fn outputs(&self) -> Vec<String>;
}

pub trait GenerationStep: Step {
    fn process(&self, offset: usize) -> Result<GenerationStepOutput>;
    fn outputs(&self) -> Vec<String>;
}

pub trait ZipStep: Step {
    fn process(
        &self,
        inputs: HashMap<String, GenerationStepOutput>,
    ) -> Result<GenerationStepOutput>;
    fn outputs(&self) -> Vec<String>;
}

type GenerationStepOutput = Vec<(Vec<HashMap<String, EntityValue>>, bool)>;

type StepInput = Vec<HashMap<String, EntityValue>>;

type StepOutput = Vec<HashMap<String, EntityValue>>;

struct LoadDataFromDict {
    name: String,
    data: Vec<HashMap<String, EntityValue>>,
    batch_size: usize,
    next_step: Option<Box<dyn Step>>,
}

impl Step for LoadDataFromDict {
    fn load(&self) -> Result<()> {
        Ok(())
    }
}

impl GenerationStep for LoadDataFromDict {
    fn process(&self, offset: usize) -> Result<GenerationStepOutput> {
        let mut data: Vec<_> = self
            .data
            .iter()
            .skip(offset)
            .cloned()
            .collect::<Vec<_>>()
            .chunks(self.batch_size)
            .map(|x| (x.to_vec(), false))
            .collect();

        let last = data.last_mut().unwrap();
        last.1 = true;

        Ok(data)
    }

    fn outputs(&self) -> Vec<String> {
        vec![self.name.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let data = vec![
            HashMap::from([
                ("id".to_string(), EntityValue::INT32(1)),
                ("active".to_string(), EntityValue::BOOL(true)),
            ]),
            HashMap::from([
                ("id".to_string(), EntityValue::INT32(1)),
                ("active".to_string(), EntityValue::BOOL(true)),
            ]),
            HashMap::from([
                ("id".to_string(), EntityValue::INT32(1)),
                ("active".to_string(), EntityValue::BOOL(true)),
            ]),
            HashMap::from([
                ("id".to_string(), EntityValue::INT32(1)),
                ("active".to_string(), EntityValue::BOOL(true)),
            ]),
            HashMap::from([
                ("id".to_string(), EntityValue::INT32(1)),
                ("active".to_string(), EntityValue::BOOL(true)),
            ]),
            HashMap::from([
                ("id".to_string(), EntityValue::INT32(1)),
                ("active".to_string(), EntityValue::BOOL(true)),
            ]),
            HashMap::from([
                ("id".to_string(), EntityValue::INT32(1)),
                ("active".to_string(), EntityValue::BOOL(true)),
            ]),
        ];

        let step = LoadDataFromDict {
            name: "test".to_string(),
            data,
            batch_size: 3,
            next_step: None,
        };

        step.load().unwrap();
        let result = step.process(0).unwrap();
        println!("{:?}", result);
    }
}
