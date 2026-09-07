//! Sets Python bindings.

#![forbid(unsafe_code)]

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::hash::{DefaultHasher, Hash, Hasher};

use fsym_sets::SymSet as NativeSet;

use crate::expr::PyExpr;

/// Python binding for symbolic sets.
#[pyclass(name = "SymSet", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PySymSet {
    pub inner: NativeSet,
}

impl std::fmt::Display for PySymSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

#[pymethods]
impl PySymSet {
    /// The empty set singleton: ∅.
    #[staticmethod]
    pub fn empty() -> Self {
        Self {
            inner: NativeSet::empty(),
        }
    }

    /// The universal set singleton: 𝕌.
    #[staticmethod]
    pub fn universal() -> Self {
        Self {
            inner: NativeSet::universal(),
        }
    }

    /// Construct an interval with specified bounds and openness.
    #[staticmethod]
    #[pyo3(signature = (start, end, left_open = false, right_open = false))]
    pub fn interval(
        start: PyExpr,
        end: PyExpr,
        left_open: bool,
        right_open: bool,
    ) -> PyResult<Self> {
        let inner = NativeSet::interval_full_checked(start.inner, end.inner, left_open, right_open)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Construct a finite discrete set from explicit elements.
    #[staticmethod]
    pub fn finite(elements: Vec<PyExpr>) -> Self {
        Self {
            inner: NativeSet::finite(elements.into_iter().map(|e| e.inner)),
        }
    }

    /// Construct the union of multiple sets.
    #[staticmethod]
    pub fn union_many(sets: Vec<PySymSet>) -> Self {
        let inner = sets
            .into_iter()
            .map(|s| s.inner)
            .fold(NativeSet::EmptySet, |acc, s| acc.union(s));
        Self { inner }
    }

    /// Construct the intersection of multiple sets.
    #[staticmethod]
    pub fn intersection_many(sets: Vec<PySymSet>) -> Self {
        let inner = sets
            .into_iter()
            .map(|s| s.inner)
            .reduce(|acc, s| acc.intersection(s))
            .unwrap_or(NativeSet::UniversalSet);
        Self { inner }
    }

    /// Construct the complement of a set against the universe.
    #[staticmethod]
    pub fn complement_of(target: PySymSet) -> Self {
        Self {
            inner: NativeSet::Complement(Box::new(target.inner)),
        }
    }

    /// Union with another set.
    pub fn union(&self, other: &PySymSet) -> PySymSet {
        PySymSet {
            inner: self.inner.clone().union(other.inner.clone()),
        }
    }

    /// Intersection with another set.
    pub fn intersection(&self, other: &PySymSet) -> PySymSet {
        PySymSet {
            inner: self.inner.clone().intersection(other.inner.clone()),
        }
    }

    /// Relative complement (difference): self \ other.
    pub fn difference(&self, other: &PySymSet) -> PySymSet {
        PySymSet {
            inner: self.inner.clone().difference(other.inner.clone()),
        }
    }

    /// Symmetric difference: (self \ other) ∪ (other \ self).
    pub fn symmetric_difference(&self, other: &PySymSet) -> PySymSet {
        PySymSet {
            inner: self.inner.clone().symmetric_difference(other.inner.clone()),
        }
    }

    /// Universal complement: 𝕌 \ self.
    pub fn complement(&self) -> PySymSet {
        PySymSet {
            inner: self.inner.clone().complement(),
        }
    }

    /// Three-valued membership: True, False, or None when undecidable.
    pub fn contains(&self, elem: &PyExpr) -> Option<bool> {
        self.inner.contains(&elem.inner)
    }

    /// Check if the set is definitely empty.
    pub fn is_empty_set(&self) -> Option<bool> {
        self.inner.is_empty_set()
    }

    /// Check if self is a subset of other.
    pub fn is_subset(&self, other: &PySymSet) -> Option<bool> {
        self.inner.is_subset(&other.inner)
    }

    /// Check if self is a superset of other.
    pub fn is_superset(&self, other: &PySymSet) -> Option<bool> {
        self.inner.is_superset(&other.inner)
    }

    /// Check if self and other are disjoint.
    pub fn is_disjoint(&self, other: &PySymSet) -> Option<bool> {
        self.inner.is_disjoint(&other.inner)
    }

    /// Exact Lebesgue measure of the set on the real line.
    pub fn measure(&self) -> Option<PyExpr> {
        self.inner.measure().map(PyExpr::from_expr)
    }

    /// Topological interior of the set.
    pub fn interior(&self) -> Option<PySymSet> {
        self.inner.interior().map(|inner| PySymSet { inner })
    }

    /// Topological closure of the set.
    pub fn closure(&self) -> Option<PySymSet> {
        self.inner.closure().map(|inner| PySymSet { inner })
    }

    /// Topological boundary of the set.
    pub fn boundary(&self) -> Option<PySymSet> {
        self.inner.boundary().map(|inner| PySymSet { inner })
    }

    /// Whether the set is open in the standard topology.
    pub fn is_open(&self) -> Option<bool> {
        self.inner.is_open()
    }

    /// Whether the set is closed in the standard topology.
    pub fn is_closed(&self) -> Option<bool> {
        self.inner.is_closed()
    }

    /// Whether the set is compact.
    pub fn is_compact(&self) -> Option<bool> {
        self.inner.is_compact()
    }

    /// The kind of set node.
    #[getter]
    pub fn kind(&self) -> String {
        match &self.inner {
            NativeSet::EmptySet => "EmptySet".to_string(),
            NativeSet::UniversalSet => "UniversalSet".to_string(),
            NativeSet::Interval { .. } => "Interval".to_string(),
            NativeSet::FiniteSet(_) => "FiniteSet".to_string(),
            NativeSet::Union(_) => "Union".to_string(),
            NativeSet::Intersection(_) => "Intersection".to_string(),
            NativeSet::Complement(_) => "Complement".to_string(),
        }
    }

    /// Interval start bound.
    #[getter]
    pub fn start(&self) -> Option<PyExpr> {
        match &self.inner {
            NativeSet::Interval { start, .. } => Some(PyExpr::from_expr(start.clone())),
            _ => None,
        }
    }

    /// Interval end bound.
    #[getter]
    pub fn end(&self) -> Option<PyExpr> {
        match &self.inner {
            NativeSet::Interval { end, .. } => Some(PyExpr::from_expr(end.clone())),
            _ => None,
        }
    }

    /// Interval left-open flag.
    #[getter]
    pub fn left_open(&self) -> Option<bool> {
        match &self.inner {
            NativeSet::Interval { left_open, .. } => Some(*left_open),
            _ => None,
        }
    }

    /// Interval right-open flag.
    #[getter]
    pub fn right_open(&self) -> Option<bool> {
        match &self.inner {
            NativeSet::Interval { right_open, .. } => Some(*right_open),
            _ => None,
        }
    }

    /// Elements of a FiniteSet.
    #[getter]
    pub fn elements(&self) -> Option<Vec<PyExpr>> {
        match &self.inner {
            NativeSet::FiniteSet(elems) => {
                Some(elems.iter().cloned().map(PyExpr::from_expr).collect())
            }
            _ => None,
        }
    }

    /// Child sets for Union, Intersection, or Complement.
    #[getter]
    pub fn args(&self) -> Vec<PySymSet> {
        match &self.inner {
            NativeSet::Union(sets) | NativeSet::Intersection(sets) => sets
                .iter()
                .cloned()
                .map(|inner| PySymSet { inner })
                .collect(),
            NativeSet::Complement(inner) => {
                vec![PySymSet {
                    inner: (**inner).clone(),
                }]
            }
            NativeSet::Interval { start, end, .. } => {
                vec![
                    PySymSet {
                        inner: NativeSet::finite(vec![start.clone()]),
                    },
                    PySymSet {
                        inner: NativeSet::finite(vec![end.clone()]),
                    },
                ]
            }
            NativeSet::FiniteSet(elems) => elems
                .iter()
                .cloned()
                .map(|e| PySymSet {
                    inner: NativeSet::finite(vec![e]),
                })
                .collect(),
            NativeSet::EmptySet | NativeSet::UniversalSet => Vec::new(),
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __richcmp__(&self, other: &PySymSet, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyValueError::new_err(
                "only == and != comparisons are supported for symbolic sets",
            )),
        }
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        (hasher.finish() & 0x7FFF_FFFF_FFFF_FFFF) as isize
    }
}
