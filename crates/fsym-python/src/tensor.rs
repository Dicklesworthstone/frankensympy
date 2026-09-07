//! Tensor and differential geometry Python bindings (WS20).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use fsym_core::{Expr, Symbol};
use fsym_tensor::{IndexVariance, MetricTensor, TensorExpr, TensorIndex};

use crate::expr::PyExpr;

#[pyclass(name = "TensorIndex", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyTensorIndex {
    pub inner: TensorIndex,
}

#[pymethods]
impl PyTensorIndex {
    #[new]
    #[pyo3(signature = (name, is_up=true))]
    pub fn new(name: &str, is_up: bool) -> Self {
        let variance = if is_up {
            IndexVariance::Upper
        } else {
            IndexVariance::Lower
        };
        Self {
            inner: TensorIndex {
                symbol: Symbol::new(name),
                variance,
            },
        }
    }

    #[staticmethod]
    pub fn upper(name: &str) -> Self {
        Self {
            inner: TensorIndex::upper(name),
        }
    }

    #[staticmethod]
    pub fn lower(name: &str) -> Self {
        Self {
            inner: TensorIndex::lower(name),
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.symbol.name.clone()
    }

    #[getter]
    pub fn is_up(&self) -> bool {
        self.inner.variance == IndexVariance::Upper
    }

    pub fn flip(&self) -> Self {
        Self {
            inner: self.inner.flip_variance(),
        }
    }

    fn __repr__(&self) -> String {
        let prefix = if self.is_up() { "^" } else { "_" };
        format!("TensorIndex({}{})", prefix, self.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Tensor", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyTensor {
    pub inner: TensorExpr,
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (name, dimension, indices, components=None))]
    pub fn new(
        name: &str,
        dimension: usize,
        indices: Vec<PyTensorIndex>,
        components: Option<Vec<PyExpr>>,
    ) -> PyResult<Self> {
        let raw_indices: Vec<TensorIndex> = indices.into_iter().map(|idx| idx.inner).collect();
        let tensor = match components {
            Some(comps) => {
                let raw_comps: Vec<Expr> = comps.into_iter().map(|c| c.inner).collect();
                TensorExpr::with_components(name, dimension, raw_indices, raw_comps)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            }
            None => {
                let mut t = TensorExpr::new(name, raw_indices);
                t.dimension = dimension;
                t
            }
        };
        Ok(Self { inner: tensor })
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[getter]
    pub fn dimension(&self) -> usize {
        self.inner.dimension
    }

    #[getter]
    pub fn indices(&self) -> Vec<PyTensorIndex> {
        self.inner
            .indices
            .iter()
            .cloned()
            .map(|inner| PyTensorIndex { inner })
            .collect()
    }

    #[getter]
    pub fn components(&self) -> Option<Vec<PyExpr>> {
        self.inner
            .components
            .as_ref()
            .map(|comps| comps.iter().cloned().map(PyExpr::from_expr).collect())
    }

    #[pyo3(signature = (other, new_name=None))]
    pub fn outer_product(&self, other: &PyTensor, new_name: Option<String>) -> PyResult<PyTensor> {
        let name =
            new_name.unwrap_or_else(|| format!("({}*{})", self.inner.name, other.inner.name));
        let prod = self
            .inner
            .outer_product(&other.inner, name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: prod })
    }

    #[pyo3(signature = (upper_name, lower_name, new_name=None))]
    pub fn self_contract(
        &self,
        upper_name: &str,
        lower_name: &str,
        new_name: Option<String>,
    ) -> PyResult<PyTensor> {
        let name = new_name.unwrap_or_else(|| self.inner.name.clone());
        let contracted = self
            .inner
            .self_contract(&Symbol::new(upper_name), &Symbol::new(lower_name), name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: contracted })
    }

    #[pyo3(signature = (other, upper_name, lower_name, new_name=None))]
    pub fn contract(
        &self,
        other: &PyTensor,
        upper_name: &str,
        lower_name: &str,
        new_name: Option<String>,
    ) -> PyResult<PyTensor> {
        let prod = self.outer_product(other, Some("tmp_prod".to_string()))?;
        prod.self_contract(upper_name, lower_name, new_name)
    }

    #[pyo3(signature = (target_name, new_name=None))]
    pub fn contract_index(
        &self,
        target_name: &str,
        new_name: Option<String>,
    ) -> PyResult<PyTensor> {
        let name = new_name.unwrap_or_else(|| self.inner.name.clone());
        let res = self
            .inner
            .contract_index(&Symbol::new(target_name), name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: res })
    }

    fn __repr__(&self) -> String {
        let idx_str = self
            .inner
            .indices
            .iter()
            .map(|idx| {
                let p = if idx.variance == IndexVariance::Upper {
                    "^"
                } else {
                    "_"
                };
                format!("{}{}", p, idx.symbol.name)
            })
            .collect::<Vec<_>>()
            .join("");
        format!("Tensor({}{})", self.inner.name, idx_str)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Metric", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyMetric {
    pub inner: MetricTensor,
}

#[pymethods]
impl PyMetric {
    #[staticmethod]
    pub fn minkowski_4d(name: &str) -> Self {
        Self {
            inner: MetricTensor::minkowski_4d(name),
        }
    }

    #[staticmethod]
    pub fn euclidean(name: &str, dimension: usize) -> PyResult<Self> {
        MetricTensor::euclidean(name, dimension)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    pub fn diagonal(name: &str, diag_entries: Vec<PyExpr>) -> PyResult<Self> {
        let raw_diag: Vec<Expr> = diag_entries.into_iter().map(|e| e.inner).collect();
        MetricTensor::diagonal(name, raw_diag)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn dimension(&self) -> usize {
        self.inner.dimension
    }

    #[getter]
    pub fn matrix(&self) -> Vec<PyExpr> {
        self.inner
            .matrix
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    #[getter]
    pub fn inverse(&self) -> Vec<PyExpr> {
        self.inner
            .inverse
            .iter()
            .cloned()
            .map(PyExpr::from_expr)
            .collect()
    }

    pub fn lower_vector(&self, vec_tensor: &PyTensor) -> PyResult<PyTensor> {
        let lowered = self
            .inner
            .lower_vector(&vec_tensor.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: lowered })
    }

    pub fn raise_covector(&self, covec_tensor: &PyTensor) -> PyResult<PyTensor> {
        let raised = self
            .inner
            .raise_covector(&covec_tensor.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyTensor { inner: raised })
    }

    pub fn inner_product(&self, u: &PyTensor, v: &PyTensor) -> PyResult<PyExpr> {
        let ip = self
            .inner
            .inner_product(&u.inner, &v.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyExpr::from_expr(ip))
    }

    pub fn norm_squared(&self, v: &PyTensor) -> PyResult<PyExpr> {
        let ns = self
            .inner
            .norm_squared(&v.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyExpr::from_expr(ns))
    }

    fn __repr__(&self) -> String {
        format!("Metric({}, dim={})", self.inner.name, self.inner.dimension)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}
