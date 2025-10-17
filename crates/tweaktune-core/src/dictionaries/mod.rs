pub mod openings;
use polars::prelude::*;

pub fn phf_to_df(set: &phf::Set<&'static str>, column_name: &str) -> DataFrame {
    let series: Vec<&str> = set.iter().cloned().collect();
    let s = Series::new(column_name.into(), series);
    DataFrame::new(vec![s.into()]).unwrap()
}

pub fn get_question_words() -> DataFrame {
    phf_to_df(&openings::QUESTION, "question")
}

pub fn get_ask_words() -> DataFrame {
    phf_to_df(&openings::ASK, "ask")
}

pub fn get_neutral_words() -> DataFrame {
    phf_to_df(&openings::NEUTRAL, "neutral")
}
