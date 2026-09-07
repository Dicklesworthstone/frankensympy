//! Geometry Python bindings.

use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use fsym_geometry::{
    Circle as NativeCircle, Line2D as NativeLine2D, Line3D as NativeLine3D,
    Plane3D as NativePlane3D, Point2D as NativePoint2D, Point3D as NativePoint3D,
    Polygon2D as NativePolygon2D, Ray2D as NativeRay2D, Ray3D as NativeRay3D,
    Segment2D as NativeSegment2D, Segment3D as NativeSegment3D, Sphere as NativeSphere,
    Triangle2D as NativeTriangle2D,
};

use crate::expr::PyExpr;

#[pyclass(name = "Point2D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPoint2D {
    pub inner: NativePoint2D,
}

#[pymethods]
impl PyPoint2D {
    #[new]
    pub fn new(x: PyExpr, y: PyExpr) -> Self {
        Self {
            inner: NativePoint2D::new(x.inner, y.inner),
        }
    }

    #[getter]
    pub fn x(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.x.clone())
    }

    #[getter]
    pub fn y(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.y.clone())
    }

    pub fn distance_squared(&self, other: &PyPoint2D) -> PyExpr {
        PyExpr::from_expr(self.inner.distance_squared(&other.inner))
    }

    pub fn midpoint(&self, other: &PyPoint2D) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.midpoint(&other.inner),
        }
    }

    pub fn dot(&self, other: &PyPoint2D) -> PyExpr {
        PyExpr::from_expr(self.inner.dot(&other.inner))
    }

    fn __getitem__(&self, idx: isize) -> PyResult<PyExpr> {
        match idx {
            0 | -2 => Ok(self.x()),
            1 | -1 => Ok(self.y()),
            _ => Err(PyIndexError::new_err("point index out of range")),
        }
    }

    fn __len__(&self) -> usize {
        2
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __eq__(&self, other: &PyPoint2D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Point3D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPoint3D {
    pub inner: NativePoint3D,
}

#[pymethods]
impl PyPoint3D {
    #[new]
    pub fn new(x: PyExpr, y: PyExpr, z: PyExpr) -> Self {
        Self {
            inner: NativePoint3D::new(x.inner, y.inner, z.inner),
        }
    }

    #[getter]
    pub fn x(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.x.clone())
    }

    #[getter]
    pub fn y(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.y.clone())
    }

    #[getter]
    pub fn z(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.z.clone())
    }

    pub fn distance_squared(&self, other: &PyPoint3D) -> PyExpr {
        PyExpr::from_expr(self.inner.distance_squared(&other.inner))
    }

    pub fn midpoint(&self, other: &PyPoint3D) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.midpoint(&other.inner),
        }
    }

    pub fn dot(&self, other: &PyPoint3D) -> PyExpr {
        PyExpr::from_expr(self.inner.dot(&other.inner))
    }

    fn __getitem__(&self, idx: isize) -> PyResult<PyExpr> {
        match idx {
            0 | -3 => Ok(self.x()),
            1 | -2 => Ok(self.y()),
            2 | -1 => Ok(self.z()),
            _ => Err(PyIndexError::new_err("point index out of range")),
        }
    }

    fn __len__(&self) -> usize {
        3
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __eq__(&self, other: &PyPoint3D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Segment2D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PySegment2D {
    pub inner: NativeSegment2D,
}

#[pymethods]
impl PySegment2D {
    #[new]
    pub fn new(p1: PyPoint2D, p2: PyPoint2D) -> Self {
        Self {
            inner: NativeSegment2D::new(p1.inner, p2.inner),
        }
    }

    #[getter]
    pub fn p1(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.p1.clone(),
        }
    }

    #[getter]
    pub fn p2(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.p2.clone(),
        }
    }

    pub fn midpoint(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.midpoint(),
        }
    }

    pub fn length_squared(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.length_squared())
    }

    fn __repr__(&self) -> String {
        format!("Segment2D({}, {})", self.inner.p1, self.inner.p2)
    }

    fn __str__(&self) -> String {
        format!("Segment2D({}, {})", self.inner.p1, self.inner.p2)
    }

    fn __eq__(&self, other: &PySegment2D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Segment3D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PySegment3D {
    pub inner: NativeSegment3D,
}

#[pymethods]
impl PySegment3D {
    #[new]
    pub fn new(p1: PyPoint3D, p2: PyPoint3D) -> Self {
        Self {
            inner: NativeSegment3D::new(p1.inner, p2.inner),
        }
    }

    #[getter]
    pub fn p1(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.p1.clone(),
        }
    }

    #[getter]
    pub fn p2(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.p2.clone(),
        }
    }

    pub fn midpoint(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.midpoint(),
        }
    }

    pub fn length_squared(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.length_squared())
    }

    fn __repr__(&self) -> String {
        format!("Segment3D({}, {})", self.inner.p1, self.inner.p2)
    }

    fn __str__(&self) -> String {
        format!("Segment3D({}, {})", self.inner.p1, self.inner.p2)
    }

    fn __eq__(&self, other: &PySegment3D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Line2D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyLine2D {
    pub inner: NativeLine2D,
}

#[pymethods]
impl PyLine2D {
    #[new]
    pub fn new(p1: PyPoint2D, p2: PyPoint2D) -> PyResult<Self> {
        let inner = NativeLine2D::new(p1.inner, p2.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn p1(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.p1().clone(),
        }
    }

    #[getter]
    pub fn p2(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.p2().clone(),
        }
    }

    pub fn intersection(&self, other: &PyLine2D) -> PyResult<PyPoint2D> {
        let pt = self
            .inner
            .intersection(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyPoint2D { inner: pt })
    }

    fn __repr__(&self) -> String {
        format!("Line2D({}, {})", self.inner.p1(), self.inner.p2())
    }

    fn __str__(&self) -> String {
        format!("Line2D({}, {})", self.inner.p1(), self.inner.p2())
    }

    fn __eq__(&self, other: &PyLine2D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Line3D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyLine3D {
    pub inner: NativeLine3D,
}

#[pymethods]
impl PyLine3D {
    #[new]
    pub fn new(p1: PyPoint3D, p2: PyPoint3D) -> PyResult<Self> {
        let inner = NativeLine3D::new(p1.inner, p2.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn p1(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.p1().clone(),
        }
    }

    #[getter]
    pub fn p2(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.p2().clone(),
        }
    }

    pub fn direction(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.direction(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Line3D({}, {})", self.inner.p1(), self.inner.p2())
    }

    fn __str__(&self) -> String {
        format!("Line3D({}, {})", self.inner.p1(), self.inner.p2())
    }

    fn __eq__(&self, other: &PyLine3D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Ray2D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyRay2D {
    pub inner: NativeRay2D,
}

#[pymethods]
impl PyRay2D {
    #[new]
    pub fn new(source: PyPoint2D, point: PyPoint2D) -> PyResult<Self> {
        let inner = NativeRay2D::new(source.inner, point.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn source(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    pub fn point(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.point().clone(),
        }
    }

    pub fn direction(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.direction(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Ray2D({}, {})", self.inner.source(), self.inner.point())
    }

    fn __str__(&self) -> String {
        format!("Ray2D({}, {})", self.inner.source(), self.inner.point())
    }

    fn __eq__(&self, other: &PyRay2D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Ray3D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyRay3D {
    pub inner: NativeRay3D,
}

#[pymethods]
impl PyRay3D {
    #[new]
    pub fn new(source: PyPoint3D, point: PyPoint3D) -> PyResult<Self> {
        let inner = NativeRay3D::new(source.inner, point.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn source(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.source().clone(),
        }
    }

    #[getter]
    pub fn point(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.point().clone(),
        }
    }

    pub fn direction(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.direction(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Ray3D({}, {})", self.inner.source(), self.inner.point())
    }

    fn __str__(&self) -> String {
        format!("Ray3D({}, {})", self.inner.source(), self.inner.point())
    }

    fn __eq__(&self, other: &PyRay3D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Circle", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyCircle {
    pub inner: NativeCircle,
}

#[pymethods]
impl PyCircle {
    #[new]
    pub fn new(center: PyPoint2D, radius: PyExpr) -> PyResult<Self> {
        let inner = NativeCircle::new(center.inner, radius.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn center(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.center().clone(),
        }
    }

    #[getter]
    pub fn radius(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.radius().clone())
    }

    pub fn area(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.area())
    }

    pub fn circumference(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.circumference())
    }

    fn __repr__(&self) -> String {
        format!("Circle({}, {})", self.inner.center(), self.inner.radius())
    }

    fn __str__(&self) -> String {
        format!("Circle({}, {})", self.inner.center(), self.inner.radius())
    }

    fn __eq__(&self, other: &PyCircle) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish() as isize
    }
}

#[pyclass(name = "Sphere", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PySphere {
    pub inner: NativeSphere,
}

#[pymethods]
impl PySphere {
    #[new]
    pub fn new(center: PyPoint3D, radius: PyExpr) -> PyResult<Self> {
        let inner = NativeSphere::new(center.inner, radius.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn center(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.center().clone(),
        }
    }

    #[getter]
    pub fn radius(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.radius().clone())
    }

    pub fn volume(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.volume())
    }

    pub fn surface_area(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.surface_area())
    }

    fn __repr__(&self) -> String {
        format!("Sphere({}, {})", self.inner.center(), self.inner.radius())
    }

    fn __str__(&self) -> String {
        format!("Sphere({}, {})", self.inner.center(), self.inner.radius())
    }

    fn __eq__(&self, other: &PySphere) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish() as isize
    }
}

#[pyclass(name = "Triangle2D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyTriangle2D {
    pub inner: NativeTriangle2D,
}

#[pymethods]
impl PyTriangle2D {
    #[new]
    pub fn new(p1: PyPoint2D, p2: PyPoint2D, p3: PyPoint2D) -> Self {
        Self {
            inner: NativeTriangle2D::new(p1.inner, p2.inner, p3.inner),
        }
    }

    #[getter]
    pub fn p1(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.p1.clone(),
        }
    }

    #[getter]
    pub fn p2(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.p2.clone(),
        }
    }

    #[getter]
    pub fn p3(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.p3.clone(),
        }
    }

    pub fn centroid(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.centroid(),
        }
    }

    pub fn double_signed_area(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.double_signed_area())
    }

    pub fn is_right(&self) -> Option<bool> {
        self.inner.is_right()
    }

    pub fn is_isosceles(&self) -> Option<bool> {
        self.inner.is_isosceles()
    }

    pub fn is_equilateral(&self) -> Option<bool> {
        self.inner.is_equilateral()
    }

    pub fn is_collinear(&self) -> Option<bool> {
        self.inner.is_collinear()
    }

    fn __repr__(&self) -> String {
        format!(
            "Triangle2D({}, {}, {})",
            self.inner.p1, self.inner.p2, self.inner.p3
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Triangle2D({}, {}, {})",
            self.inner.p1, self.inner.p2, self.inner.p3
        )
    }

    fn __eq__(&self, other: &PyTriangle2D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Polygon2D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPolygon2D {
    pub inner: NativePolygon2D,
}

#[pymethods]
impl PyPolygon2D {
    #[new]
    pub fn new(vertices: Vec<PyPoint2D>) -> PyResult<Self> {
        let native_verts = vertices.into_iter().map(|p| p.inner).collect();
        let inner =
            NativePolygon2D::new(native_verts).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn vertices(&self) -> Vec<PyPoint2D> {
        self.inner
            .vertices()
            .iter()
            .cloned()
            .map(|inner| PyPoint2D { inner })
            .collect()
    }

    pub fn centroid(&self) -> PyPoint2D {
        PyPoint2D {
            inner: self.inner.centroid(),
        }
    }

    pub fn double_signed_area(&self) -> PyExpr {
        PyExpr::from_expr(self.inner.double_signed_area())
    }

    pub fn edge_lengths_squared(&self) -> Vec<PyExpr> {
        self.inner
            .edge_lengths_squared()
            .into_iter()
            .map(PyExpr::from_expr)
            .collect()
    }

    pub fn is_convex(&self) -> Option<bool> {
        self.inner.is_convex()
    }

    fn __repr__(&self) -> String {
        let v_str = self
            .inner
            .vertices()
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Polygon2D([{}])", v_str)
    }

    fn __str__(&self) -> String {
        let v_str = self
            .inner
            .vertices()
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join(", ");
        format!("Polygon2D([{}])", v_str)
    }

    fn __eq__(&self, other: &PyPolygon2D) -> bool {
        self.inner == other.inner
    }
}

#[pyclass(name = "Plane3D", module = "fsym_python")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPlane3D {
    pub inner: NativePlane3D,
}

#[pymethods]
impl PyPlane3D {
    #[new]
    pub fn new(point: PyPoint3D, normal: PyPoint3D) -> PyResult<Self> {
        let inner = NativePlane3D::new(point.inner, normal.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_three_points(p1: PyPoint3D, p2: PyPoint3D, p3: PyPoint3D) -> PyResult<Self> {
        let inner = NativePlane3D::from_three_points(p1.inner, p2.inner, p3.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn point(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.point().clone(),
        }
    }

    #[getter]
    pub fn normal(&self) -> PyPoint3D {
        PyPoint3D {
            inner: self.inner.normal().clone(),
        }
    }

    pub fn eval_at_point(&self, q: &PyPoint3D) -> PyExpr {
        PyExpr::from_expr(self.inner.eval_at_point(&q.inner))
    }

    pub fn distance_squared(&self, q: &PyPoint3D) -> PyExpr {
        PyExpr::from_expr(self.inner.distance_squared(&q.inner))
    }

    pub fn intersection_line(&self, line: &PyLine3D) -> PyResult<PyPoint3D> {
        let pt = self
            .inner
            .intersection_line(&line.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyPoint3D { inner: pt })
    }

    pub fn is_parallel(&self, other: &PyPlane3D) -> PyResult<bool> {
        self.inner
            .is_parallel(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn is_perpendicular(&self, other: &PyPlane3D) -> PyResult<bool> {
        self.inner
            .is_perpendicular(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn intersection_plane(&self, other: &PyPlane3D) -> PyResult<PyLine3D> {
        let l = self
            .inner
            .intersection_plane(&other.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyLine3D { inner: l })
    }

    fn __repr__(&self) -> String {
        format!(
            "Plane3D(p={}, n={})",
            self.inner.point(),
            self.inner.normal()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Plane3D(p={}, n={})",
            self.inner.point(),
            self.inner.normal()
        )
    }

    fn __eq__(&self, other: &PyPlane3D) -> bool {
        self.inner == other.inner
    }
}
