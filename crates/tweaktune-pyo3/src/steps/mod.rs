use arrow::array::{Array, ArrayData, ArrayRef, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use minijinja::{context, Environment};
use pyo3::ffi::PyThreadState;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tweaktune_abstractions::EntityValue;
use tweaktune_core::datasets::{ArrowDataset, CsvDataset, Dataset, JsonlDataset, ParquetDataset};
use tweaktune_core::readers::JsonlReader;

#[pyclass]
#[derive(Debug)]
pub enum Lang {
    Deu,
    Eng,
    Fra,
}

#[pyclass]
pub struct StepConfig {
    #[pyo3(get, set)]
    pub name: String,
}

#[pymethods]
impl StepConfig {
    #[new]
    pub fn new(name: String) -> Self {
        StepConfig { name }
    }
}

#[pyclass]
pub struct Step {
    name: String,
}

#[pymethods]
impl Step {
    #[new]
    pub fn new(config: PyRef<StepConfig>) -> PyResult<Self> {
        let name = config.name.clone();
        Ok(Step { name })
    }

    pub fn embed(&self, input: String, lang: PyRef<Lang>) -> PyResult<String> {
        let l = format!("{:?}", lang);
        Ok(input + &self.name + &l)
    }

    pub fn persona(&self, input: Vec<HashMap<String, String>>) -> PyResult<String> {
        println!("{:?}", input);
        Ok(self.name.to_string())
    }

    pub fn template(
        &self,
        name: String,
        template: String,
        input: HashMap<String, String>,
    ) -> PyResult<String> {
        let mut env = Environment::new();
        env.add_template(&name, &template).unwrap();
        let tmpl = env.get_template(&name).unwrap();
        let vvv = tmpl.render(input).unwrap();

        Ok(vvv)
    }

    pub fn create_arrow_buffer(&self, py: Python) -> PyObject {
        let arrow = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let arrow1 = StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]);
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let batch = RecordBatch::try_new(
            schema.clone().into(),
            vec![Arc::new(arrow), Arc::new(arrow1)],
        )
        .unwrap();

        let mut buffer = Vec::new();
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();

        PyBytes::new(py, &buffer).into()
    }

    pub fn create_record_batch(&self) -> PyArrowType<RecordBatch> {
        let arrow = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let arrow1 = StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]);
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let batch = RecordBatch::try_new(
            schema.clone().into(),
            vec![Arc::new(arrow), Arc::new(arrow1)],
        )
        .unwrap();

        PyArrowType(batch)
    }

    pub fn create_record_batch_vec(&self) -> PyArrowType<Vec<RecordBatch>> {
        let arrow = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let arrow1 = StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]);
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]);

        let batch = RecordBatch::try_new(
            schema.clone().into(),
            vec![Arc::new(arrow), Arc::new(arrow1)],
        )
        .unwrap();

        let items: Vec<serde_json::Value> = serde_arrow::from_record_batch(&batch).unwrap();
        println!("{:?}", items);

        PyArrowType(vec![batch])
    }

    pub fn create_record_array(&self) -> PyArrowType<ArrayData> {
        let arrow = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        PyArrowType(arrow.to_data())
    }

    pub fn read_record_batch(&self, record_batch: PyArrowType<RecordBatch>) {
        println!("{:?}", record_batch.0);
    }

    pub fn read_array(&self, record_batch: PyArrowType<ArrayData>) {
        println!("{:?}", record_batch.0);
    }

    pub fn read_reader(&self, mut reader: PyArrowType<ArrowArrayStreamReader>) {
        let t = reader.0.next().unwrap().unwrap().clone();
        println!("{:?}", t);
    }

    pub fn read_pyarrow(&self, py: Python, buffer: Py<PyBytes>) -> PyResult<Vec<i64>> {
        let buffer = Cursor::new(buffer.as_bytes(py));

        let mut reader = StreamReader::try_new(buffer, None).unwrap();
        let batch = reader.next().unwrap().unwrap();

        println!("{:?}", batch.schema());

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let data = array.values().to_vec();

        Ok(data)
    }

    pub fn read_pyarrow_str(&self, py: Python, buffer: Py<PyBytes>) -> PyResult<String> {
        let buffer = Cursor::new(buffer.as_bytes(py));

        let mut reader = StreamReader::try_new(buffer, None).unwrap();
        let batch = reader.next().unwrap().unwrap();

        println!("{:?}", batch.schema());

        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let data = array.values().to_vec();
        Ok("".to_string())
    }
}

#[pyclass]
pub struct Jsonl {
    // reader: JsonlReader,
    dataset: JsonlDataset,
}

#[pymethods]
impl Jsonl {
    #[new]
    pub fn new(name: String, path: String) -> PyResult<Self> {
        Ok(Jsonl {
            dataset: JsonlDataset::new(name, path),
        })
    }

    pub fn load(&self) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        // let data = Runtime::new().unwrap().block_on(self.dataset.read_all())?;
        let data = self.dataset.read_all().unwrap();
        Ok(PyArrowType(data))
    }
}

#[pyclass]
pub struct Csv {
    dataset: CsvDataset,
}

#[pymethods]
impl Csv {
    #[new]
    pub fn new(name: String, path: String, delimiter: String, has_header: bool) -> PyResult<Self> {
        Ok(Csv {
            dataset: CsvDataset::new(name, path, delimiter.as_bytes()[0], has_header),
        })
    }

    pub fn load(&self) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        let data = self.dataset.read_all().unwrap();
        Ok(PyArrowType(data))
    }
}

#[pyclass]
pub struct Parquet {
    dataset: ParquetDataset,
}

#[pymethods]
impl Parquet {
    #[new]
    pub fn new(name: String, path: String) -> PyResult<Self> {
        Ok(Parquet {
            dataset: ParquetDataset::new(name, path),
        })
    }

    pub fn load(&self) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        let data = self.dataset.read_all().unwrap();
        Ok(PyArrowType(data))
    }
}

#[pyclass]
pub struct Arrow {
    dataset: ArrowDataset,
}

#[pymethods]
impl Arrow {
    #[new]
    pub fn new(name: String, mut reader: PyArrowType<ArrowArrayStreamReader>) -> PyResult<Self> {
        let mut batches = Vec::new();
        for batch in reader.0.by_ref() {
            batches.push(batch.unwrap());
        }

        Ok(Arrow {
            dataset: ArrowDataset::new(name, batches),
        })
    }

    pub fn load(&self) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        let data = self.dataset.read_all().unwrap();
        Ok(PyArrowType(data))
    }
}
