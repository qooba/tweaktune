use pyo3::{exceptions::PyTypeError, PyErr};

pub trait ResultExt<T, E> {
    fn map_pyerr(self) -> Result<T, PyErr>;
}

impl<T, E: std::fmt::Debug> ResultExt<T, E> for Result<T, E> {
    fn map_pyerr(self) -> Result<T, PyErr> {
        self.map_err(|e| {
            let err = format!("{:?}", e);
            PyErr::new::<PyTypeError, _>(err)
        })
    }
}
