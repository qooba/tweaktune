use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use tweaktune_core::datasets::{ArrowDataset, CsvDataset, Dataset, JsonlDataset, ParquetDataset};

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

    #[pyo3(signature = (batch_size=None))]
    pub fn load(&self, batch_size: Option<usize>) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        // let data = Runtime::new().unwrap().block_on(self.dataset.read_all())?;
        let data = self.dataset.read_all(batch_size).unwrap();
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

    #[pyo3(signature = (batch_size=None))]
    pub fn load(&self, batch_size: Option<usize>) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        let data = self.dataset.read_all(batch_size).unwrap();
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

    #[pyo3(signature = (batch_size=None))]
    pub fn load(&self, batch_size: Option<usize>) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        let data = self.dataset.read_all(batch_size).unwrap();
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

    #[pyo3(signature = (batch_size=None))]
    pub fn load(&self, batch_size: Option<usize>) -> PyResult<PyArrowType<Vec<RecordBatch>>> {
        let data = self.dataset.read_all(batch_size).unwrap();
        Ok(PyArrowType(data))
    }
}
