//! # fsym-geometry
//!
//! Symbolic 2D and 3D geometry: points, lines, segments, rays, circles, polygons,
//! intersections, and distances (WS20).

#![forbid(unsafe_code)]

use fsym_core::{BigRational, Expr};
use fsym_simplify::simplify;
use serde::de::{Error as _, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// Maximum number of vertices admitted into a single polygon.
///
/// This is a trust-boundary limit, not a claim about the mathematical maximum polygon size.
pub const MAX_POLYGON_VERTICES: usize = 8_192;

/// Initial allocation allowed from an untrusted Serde sequence length hint.
const INITIAL_POLYGON_VERTEX_RESERVE: usize = 16;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GeometryError {
    #[error("Coincident points cannot define a unique line")]
    CoincidentPoints,
    #[error("Invalid radius: radius cannot be negative")]
    NegativeRadius,
    #[error("Lines are parallel, no unique intersection")]
    ParallelLines,
    #[error("A symbolic degeneracy predicate is undecidable without additional assumptions")]
    SymbolicDegeneracyUndetermined,
    #[error("A polygon must have at least 3 vertices")]
    DegeneratePolygon,
    #[error("Polygon has {actual} vertices, exceeding the maximum of {max}")]
    PolygonVertexLimitExceeded { actual: usize, max: usize },
    #[error("Normal vector cannot be zero")]
    DegeneratePlane,
}

/// 2D Symbolic Point.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Point2D {
    pub x: Expr,
    pub y: Expr,
}

impl Point2D {
    pub fn new(x: Expr, y: Expr) -> Self {
        Self { x, y }
    }

    /// Squared Euclidean distance between two points: (x2 - x1)^2 + (y2 - y1)^2.
    pub fn distance_squared(&self, other: &Self) -> Expr {
        let dx = Expr::Add(vec![
            other.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.x.clone()]),
        ]);
        let dy = Expr::Add(vec![
            other.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.y.clone()]),
        ]);
        let dx2 = Expr::Pow(Arc::new(dx), Arc::new(Expr::from_i64(2)));
        let dy2 = Expr::Pow(Arc::new(dy), Arc::new(Expr::from_i64(2)));
        simplify(&Expr::Add(vec![dx2, dy2]))
    }
}

impl fmt::Display for Point2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point2D({}, {})", self.x, self.y)
    }
}

/// 3D Symbolic Point.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Point3D {
    pub x: Expr,
    pub y: Expr,
    pub z: Expr,
}

impl Point3D {
    pub fn new(x: Expr, y: Expr, z: Expr) -> Self {
        Self { x, y, z }
    }

    /// Squared Euclidean distance between two 3D points: (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2.
    pub fn distance_squared(&self, other: &Self) -> Expr {
        let dx = Expr::Add(vec![
            other.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.x.clone()]),
        ]);
        let dy = Expr::Add(vec![
            other.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.y.clone()]),
        ]);
        let dz = Expr::Add(vec![
            other.z.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.z.clone()]),
        ]);
        let dx2 = Expr::Pow(Arc::new(dx), Arc::new(Expr::from_i64(2)));
        let dy2 = Expr::Pow(Arc::new(dy), Arc::new(Expr::from_i64(2)));
        let dz2 = Expr::Pow(Arc::new(dz), Arc::new(Expr::from_i64(2)));
        simplify(&Expr::Add(vec![dx2, dy2, dz2]))
    }
}

impl fmt::Display for Point3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point3D({}, {}, {})", self.x, self.y, self.z)
    }
}

/// 3D Symbolic Segment between two endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment3D {
    pub p1: Point3D,
    pub p2: Point3D,
}

impl Segment3D {
    pub fn new(p1: Point3D, p2: Point3D) -> Self {
        Self { p1, p2 }
    }

    /// Midpoint of the 3D segment: ((x1 + x2)/2, (y1 + y2)/2, (z1 + z2)/2).
    pub fn midpoint(&self) -> Point3D {
        let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
        let mx = simplify(&Expr::Mul(vec![
            half.clone(),
            Expr::Add(vec![self.p1.x.clone(), self.p2.x.clone()]),
        ]));
        let my = simplify(&Expr::Mul(vec![
            half.clone(),
            Expr::Add(vec![self.p1.y.clone(), self.p2.y.clone()]),
        ]));
        let mz = simplify(&Expr::Mul(vec![
            half,
            Expr::Add(vec![self.p1.z.clone(), self.p2.z.clone()]),
        ]));
        Point3D::new(mx, my, mz)
    }

    /// Squared length of the 3D segment.
    pub fn length_squared(&self) -> Expr {
        self.p1.distance_squared(&self.p2)
    }
}

impl fmt::Display for Segment3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Segment3D({}, {})", self.p1, self.p2)
    }
}

/// 2D Symbolic Line passing through two distinct points `p1` and `p2`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Line2D {
    p1: Point2D,
    p2: Point2D,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line2DWire {
    p1: Point2D,
    p2: Point2D,
}

impl<'de> Deserialize<'de> for Line2D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Line2DWire::deserialize(deserializer)?;
        Self::new(wire.p1, wire.p2).map_err(serde::de::Error::custom)
    }
}

impl Line2D {
    pub fn new(p1: Point2D, p2: Point2D) -> Result<Self, GeometryError> {
        if p1 == p2 {
            return Err(GeometryError::CoincidentPoints);
        }
        let dx = coordinate_difference(&p2.x, &p1.x);
        let dy = coordinate_difference(&p2.y, &p1.y);
        match classify_zero_vector([&dx, &dy]) {
            ZeroVectorStatus::Zero => Err(GeometryError::CoincidentPoints),
            ZeroVectorStatus::NonZero => Ok(Self { p1, p2 }),
            ZeroVectorStatus::Unknown => Err(GeometryError::SymbolicDegeneracyUndetermined),
        }
    }

    pub fn p1(&self) -> &Point2D {
        &self.p1
    }

    pub fn p2(&self) -> &Point2D {
        &self.p2
    }

    /// Computes intersection point with another 2D line.
    pub fn intersection(&self, other: &Self) -> Result<Point2D, GeometryError> {
        // Line 1: (x1, y1) to (x2, y2)
        // Line 2: (x3, y3) to (x4, y4)
        // D = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        let (x1, y1) = (&self.p1.x, &self.p1.y);
        let (x2, y2) = (&self.p2.x, &self.p2.y);
        let (x3, y3) = (&other.p1.x, &other.p1.y);
        let (x4, y4) = (&other.p2.x, &other.p2.y);

        let dx12 = simplify(&Expr::Add(vec![
            x1.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), x2.clone()]),
        ]));
        let dy12 = simplify(&Expr::Add(vec![
            y1.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), y2.clone()]),
        ]));
        let dx34 = simplify(&Expr::Add(vec![
            x3.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), x4.clone()]),
        ]));
        let dy34 = simplify(&Expr::Add(vec![
            y3.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), y4.clone()]),
        ]));

        let denom = simplify(&Expr::Add(vec![
            Expr::Mul(vec![dx12.clone(), dy34.clone()]),
            Expr::Mul(vec![Expr::from_i64(-1), dy12.clone(), dx34.clone()]),
        ]));

        match numeric_value(&denom) {
            Some(value) if value.numer().is_zero() => {
                return Err(GeometryError::ParallelLines);
            }
            Some(_) => {}
            None => return Err(GeometryError::SymbolicDegeneracyUndetermined),
        }

        let d12 = simplify(&Expr::Add(vec![
            Expr::Mul(vec![x1.clone(), y2.clone()]),
            Expr::Mul(vec![Expr::from_i64(-1), y1.clone(), x2.clone()]),
        ]));
        let d34 = simplify(&Expr::Add(vec![
            Expr::Mul(vec![x3.clone(), y4.clone()]),
            Expr::Mul(vec![Expr::from_i64(-1), y3.clone(), x4.clone()]),
        ]));

        let px_num = simplify(&Expr::Add(vec![
            Expr::Mul(vec![d12.clone(), dx34]),
            Expr::Mul(vec![Expr::from_i64(-1), dx12, d34.clone()]),
        ]));

        let py_num = simplify(&Expr::Add(vec![
            Expr::Mul(vec![d12, dy34]),
            Expr::Mul(vec![Expr::from_i64(-1), dy12, d34]),
        ]));

        let px = expr_div(px_num, denom.clone());
        let py = expr_div(py_num, denom);

        Ok(Point2D::new(px, py))
    }
}

fn numeric_value(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Integer(value) => Some(BigRational::from_integer(value.clone())),
        Expr::Rational(value) => Some(value.clone()),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZeroVectorStatus {
    Zero,
    NonZero,
    Unknown,
}

fn coordinate_difference(minuend: &Expr, subtrahend: &Expr) -> Expr {
    Expr::Add(vec![
        minuend.clone(),
        Expr::Mul(vec![Expr::from_i64(-1), subtrahend.clone()]),
    ])
}

/// Classifies whether an exact coordinate vector is zero without treating symbolic unknowns as
/// proof of nonzero-ness. One exact nonzero component establishes the whole vector as nonzero;
/// otherwise every component must reduce to an exact numeric zero to establish degeneracy.
fn classify_zero_vector<'a>(components: impl IntoIterator<Item = &'a Expr>) -> ZeroVectorStatus {
    let mut saw_unknown = false;
    for component in components {
        let simplified = simplify(component);
        match numeric_value(&simplified) {
            Some(value) if !value.numer().is_zero() => return ZeroVectorStatus::NonZero,
            Some(_) => {}
            None => saw_unknown = true,
        }
    }
    if saw_unknown {
        ZeroVectorStatus::Unknown
    } else {
        ZeroVectorStatus::Zero
    }
}

fn expr_div(num: Expr, den: Expr) -> Expr {
    match (&num, &den) {
        (Expr::Integer(a), Expr::Integer(b)) => {
            if b.is_zero() {
                let inv = Expr::Pow(Arc::new(den), Arc::new(Expr::from_i64(-1)));
                return simplify(&Expr::Mul(vec![num, inv]));
            }
            let r = BigRational::new(a.clone(), b.clone());
            if r.is_integer() {
                Expr::Integer(r.to_integer())
            } else {
                Expr::Rational(r)
            }
        }
        (Expr::Rational(a), Expr::Rational(b)) => {
            if b.numer().is_zero() {
                let inv = Expr::Pow(Arc::new(den), Arc::new(Expr::from_i64(-1)));
                return simplify(&Expr::Mul(vec![num, inv]));
            }
            let r = a / b;
            if r.is_integer() {
                Expr::Integer(r.to_integer())
            } else {
                Expr::Rational(r)
            }
        }
        _ => {
            let inv = Expr::Pow(Arc::new(den), Arc::new(Expr::from_i64(-1)));
            simplify(&Expr::Mul(vec![num, inv]))
        }
    }
}

/// 2D Symbolic Segment between two endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment2D {
    pub p1: Point2D,
    pub p2: Point2D,
}

impl Segment2D {
    pub fn new(p1: Point2D, p2: Point2D) -> Self {
        Self { p1, p2 }
    }

    /// Midpoint of the segment: ((x1 + x2)/2, (y1 + y2)/2).
    pub fn midpoint(&self) -> Point2D {
        let half = Expr::Rational(BigRational::new(1.into(), 2.into()));
        let mx = simplify(&Expr::Mul(vec![
            half.clone(),
            Expr::Add(vec![self.p1.x.clone(), self.p2.x.clone()]),
        ]));
        let my = simplify(&Expr::Mul(vec![
            half,
            Expr::Add(vec![self.p1.y.clone(), self.p2.y.clone()]),
        ]));
        Point2D::new(mx, my)
    }

    /// Squared length of the segment.
    pub fn length_squared(&self) -> Expr {
        self.p1.distance_squared(&self.p2)
    }
}

/// 2D Symbolic Ray emanating from `source` passing through `point`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ray2D {
    source: Point2D,
    point: Point2D,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ray2DWire {
    source: Point2D,
    point: Point2D,
}

impl<'de> Deserialize<'de> for Ray2D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Ray2DWire::deserialize(deserializer)?;
        Self::new(wire.source, wire.point).map_err(serde::de::Error::custom)
    }
}

impl Ray2D {
    pub fn new(source: Point2D, point: Point2D) -> Result<Self, GeometryError> {
        if source == point {
            return Err(GeometryError::CoincidentPoints);
        }
        let dx = coordinate_difference(&point.x, &source.x);
        let dy = coordinate_difference(&point.y, &source.y);
        match classify_zero_vector([&dx, &dy]) {
            ZeroVectorStatus::Zero => Err(GeometryError::CoincidentPoints),
            ZeroVectorStatus::NonZero => Ok(Self { source, point }),
            ZeroVectorStatus::Unknown => Err(GeometryError::SymbolicDegeneracyUndetermined),
        }
    }

    pub fn source(&self) -> &Point2D {
        &self.source
    }

    pub fn point(&self) -> &Point2D {
        &self.point
    }

    /// Direction vector (dx, dy) = (point.x - source.x, point.y - source.y).
    pub fn direction(&self) -> Point2D {
        let dx = simplify(&Expr::Add(vec![
            self.point.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.source.x.clone()]),
        ]));
        let dy = simplify(&Expr::Add(vec![
            self.point.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.source.y.clone()]),
        ]));
        Point2D::new(dx, dy)
    }
}

impl fmt::Display for Ray2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ray2D({}, {})", self.source, self.point)
    }
}

/// 2D Symbolic Triangle defined by three vertices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Triangle2D {
    pub p1: Point2D,
    pub p2: Point2D,
    pub p3: Point2D,
}

impl Triangle2D {
    pub fn new(p1: Point2D, p2: Point2D, p3: Point2D) -> Self {
        Self { p1, p2, p3 }
    }

    /// Centroid of the triangle: ((x1 + x2 + x3)/3, (y1 + y2 + y3)/3).
    pub fn centroid(&self) -> Point2D {
        let third = Expr::Rational(BigRational::new(1.into(), 3.into()));
        let cx = simplify(&Expr::Mul(vec![
            third.clone(),
            Expr::Add(vec![
                self.p1.x.clone(),
                self.p2.x.clone(),
                self.p3.x.clone(),
            ]),
        ]));
        let cy = simplify(&Expr::Mul(vec![
            third,
            Expr::Add(vec![
                self.p1.y.clone(),
                self.p2.y.clone(),
                self.p3.y.clone(),
            ]),
        ]));
        Point2D::new(cx, cy)
    }

    /// Double signed area via shoelace formula: x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2).
    pub fn double_signed_area(&self) -> Expr {
        let t1 = Expr::Mul(vec![
            self.p1.x.clone(),
            Expr::Add(vec![
                self.p2.y.clone(),
                Expr::Mul(vec![Expr::from_i64(-1), self.p3.y.clone()]),
            ]),
        ]);
        let t2 = Expr::Mul(vec![
            self.p2.x.clone(),
            Expr::Add(vec![
                self.p3.y.clone(),
                Expr::Mul(vec![Expr::from_i64(-1), self.p1.y.clone()]),
            ]),
        ]);
        let t3 = Expr::Mul(vec![
            self.p3.x.clone(),
            Expr::Add(vec![
                self.p1.y.clone(),
                Expr::Mul(vec![Expr::from_i64(-1), self.p2.y.clone()]),
            ]),
        ]);
        simplify(&Expr::Add(vec![t1, t2, t3]))
    }

    /// Decides whether the three vertices are collinear when their exact signed area is numeric.
    ///
    /// A symbolic nonzero-looking expression is not evidence of non-collinearity: it may vanish
    /// under a specialization. Such cases remain `None` until assumptions discharge the predicate.
    pub fn is_collinear(&self) -> Option<bool> {
        let area = self.double_signed_area();
        if area.is_zero() {
            Some(true)
        } else {
            numeric_value(&area).map(|value| value.numer().is_zero())
        }
    }
}

/// 2D General Symbolic Polygon defined by an ordered list of vertices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Polygon2D {
    vertices: Vec<Point2D>,
}

struct BoundedPolygonVertices(Vec<Point2D>);

struct BoundedPolygonVerticesVisitor;

fn try_reserve_polygon_vertices<E>(vertices: &mut Vec<Point2D>, additional: usize) -> Result<(), E>
where
    E: serde::de::Error,
{
    vertices
        .try_reserve_exact(additional)
        .map_err(|_| E::custom("polygon vertex allocation refused"))
}

fn next_polygon_vertex_capacity(current: usize) -> Option<usize> {
    if current >= MAX_POLYGON_VERTICES {
        return None;
    }
    Some(
        current
            .checked_mul(2)
            .unwrap_or(MAX_POLYGON_VERTICES)
            .clamp(1, MAX_POLYGON_VERTICES),
    )
}

impl<'de> Deserialize<'de> for BoundedPolygonVertices {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_seq(BoundedPolygonVerticesVisitor)
            .map(Self)
    }
}

impl<'de> Visitor<'de> for BoundedPolygonVerticesVisitor {
    type Value = Vec<Point2D>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "at most {MAX_POLYGON_VERTICES} polygon vertices")
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let hint = sequence.size_hint();
        if let Some(hint) = hint
            && hint > MAX_POLYGON_VERTICES
        {
            return Err(A::Error::invalid_length(hint, &self));
        }

        let mut vertices = Vec::new();
        let initial_reserve = hint
            .unwrap_or(0)
            .min(INITIAL_POLYGON_VERTEX_RESERVE)
            .min(MAX_POLYGON_VERTICES);
        if initial_reserve != 0 {
            try_reserve_polygon_vertices::<A::Error>(&mut vertices, initial_reserve)?;
        }

        while vertices.len() < MAX_POLYGON_VERTICES {
            let Some(vertex) = sequence.next_element::<Point2D>()? else {
                return Ok(vertices);
            };
            if vertices.len() == vertices.capacity() {
                let target_capacity = next_polygon_vertex_capacity(vertices.capacity())
                    .ok_or_else(|| {
                        A::Error::custom("polygon vertex capacity invariant violated")
                    })?;
                let additional = target_capacity.checked_sub(vertices.len()).ok_or_else(|| {
                    A::Error::custom("polygon vertex capacity accounting overflow")
                })?;
                if additional == 0 {
                    return Err(A::Error::custom(
                        "polygon vertex capacity invariant violated",
                    ));
                }
                try_reserve_polygon_vertices::<A::Error>(&mut vertices, additional)?;
            }
            vertices.push(vertex);
        }

        if sequence.next_element::<IgnoredAny>()?.is_some() {
            return Err(A::Error::invalid_length(
                MAX_POLYGON_VERTICES.saturating_add(1),
                &self,
            ));
        }
        Ok(vertices)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Polygon2DWire {
    vertices: BoundedPolygonVertices,
}

impl<'de> Deserialize<'de> for Polygon2D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Polygon2DWire::deserialize(deserializer)?;
        Self::new(wire.vertices.0).map_err(D::Error::custom)
    }
}

impl Polygon2D {
    pub fn new(vertices: Vec<Point2D>) -> Result<Self, GeometryError> {
        if vertices.len() < 3 {
            return Err(GeometryError::DegeneratePolygon);
        }
        if vertices.len() > MAX_POLYGON_VERTICES {
            return Err(GeometryError::PolygonVertexLimitExceeded {
                actual: vertices.len(),
                max: MAX_POLYGON_VERTICES,
            });
        }
        Ok(Self { vertices })
    }

    pub fn vertices(&self) -> &[Point2D] {
        &self.vertices
    }

    /// Double signed area via the Shoelace formula: sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i).
    pub fn double_signed_area(&self) -> Expr {
        let n = self.vertices.len();
        let mut terms = Vec::with_capacity(n);
        for i in 0..n {
            let next = (i + 1) % n;
            let xi = &self.vertices[i].x;
            let yi = &self.vertices[i].y;
            let x_next = &self.vertices[next].x;
            let y_next = &self.vertices[next].y;

            let term = Expr::Add(vec![
                Expr::Mul(vec![xi.clone(), y_next.clone()]),
                Expr::Mul(vec![Expr::from_i64(-1), x_next.clone(), yi.clone()]),
            ]);
            terms.push(term);
        }
        simplify(&Expr::Add(terms))
    }

    /// Centroid of the polygon vertices.
    pub fn centroid(&self) -> Point2D {
        let n = self.vertices.len();
        let scale = Expr::Rational(BigRational::new(1.into(), (n as i64).into()));
        let mut sum_x = Vec::with_capacity(n);
        let mut sum_y = Vec::with_capacity(n);
        for v in &self.vertices {
            sum_x.push(v.x.clone());
            sum_y.push(v.y.clone());
        }
        let cx = simplify(&Expr::Mul(vec![scale.clone(), Expr::Add(sum_x)]));
        let cy = simplify(&Expr::Mul(vec![scale, Expr::Add(sum_y)]));
        Point2D::new(cx, cy)
    }
}

impl fmt::Display for Polygon2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Polygon2D({} vertices)", self.vertices.len())
    }
}

/// 2D Symbolic Circle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Circle {
    center: Point2D,
    radius: Expr,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CircleWire {
    center: Point2D,
    radius: Expr,
}

impl<'de> Deserialize<'de> for Circle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CircleWire::deserialize(deserializer)?;
        Self::new(wire.center, wire.radius).map_err(serde::de::Error::custom)
    }
}

impl Circle {
    pub fn new(center: Point2D, radius: Expr) -> Result<Self, GeometryError> {
        if numeric_value(&radius).is_some_and(|value| value < BigRational::from_integer(0.into())) {
            return Err(GeometryError::NegativeRadius);
        }
        Ok(Self { center, radius })
    }

    pub fn center(&self) -> &Point2D {
        &self.center
    }

    pub fn radius(&self) -> &Expr {
        &self.radius
    }

    /// Exact circle area: π * r^2.
    pub fn area(&self) -> Expr {
        let r2 = Expr::Pow(Arc::new(self.radius.clone()), Arc::new(Expr::from_i64(2)));
        simplify(&Expr::Mul(vec![Expr::Const(fsym_core::Constant::Pi), r2]))
    }

    /// Exact circle circumference: 2 * π * r.
    pub fn circumference(&self) -> Expr {
        simplify(&Expr::Mul(vec![
            Expr::from_i64(2),
            Expr::Const(fsym_core::Constant::Pi),
            self.radius.clone(),
        ]))
    }
}

impl fmt::Display for Circle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Circle({}, r={})", self.center, self.radius)
    }
}

/// 3D Symbolic Sphere.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Sphere {
    center: Point3D,
    radius: Expr,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SphereWire {
    center: Point3D,
    radius: Expr,
}

impl<'de> Deserialize<'de> for Sphere {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SphereWire::deserialize(deserializer)?;
        Self::new(wire.center, wire.radius).map_err(serde::de::Error::custom)
    }
}

impl Sphere {
    pub fn new(center: Point3D, radius: Expr) -> Result<Self, GeometryError> {
        if numeric_value(&radius).is_some_and(|value| value < BigRational::from_integer(0.into())) {
            return Err(GeometryError::NegativeRadius);
        }
        Ok(Self { center, radius })
    }

    pub fn center(&self) -> &Point3D {
        &self.center
    }

    pub fn radius(&self) -> &Expr {
        &self.radius
    }

    /// Exact sphere volume: (4/3) * π * r^3.
    pub fn volume(&self) -> Expr {
        let four_thirds = Expr::Rational(BigRational::new(4.into(), 3.into()));
        let r3 = Expr::Pow(Arc::new(self.radius.clone()), Arc::new(Expr::from_i64(3)));
        simplify(&Expr::Mul(vec![
            four_thirds,
            Expr::Const(fsym_core::Constant::Pi),
            r3,
        ]))
    }

    /// Exact sphere surface area: 4 * π * r^2.
    pub fn surface_area(&self) -> Expr {
        let r2 = Expr::Pow(Arc::new(self.radius.clone()), Arc::new(Expr::from_i64(2)));
        simplify(&Expr::Mul(vec![
            Expr::from_i64(4),
            Expr::Const(fsym_core::Constant::Pi),
            r2,
        ]))
    }
}

impl fmt::Display for Sphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sphere({}, r={})", self.center, self.radius)
    }
}

/// 3D Symbolic Line passing through two distinct points `p1` and `p2`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Line3D {
    p1: Point3D,
    p2: Point3D,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line3DWire {
    p1: Point3D,
    p2: Point3D,
}

impl<'de> Deserialize<'de> for Line3D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Line3DWire::deserialize(deserializer)?;
        Self::new(wire.p1, wire.p2).map_err(serde::de::Error::custom)
    }
}

impl Line3D {
    pub fn new(p1: Point3D, p2: Point3D) -> Result<Self, GeometryError> {
        if p1 == p2 {
            return Err(GeometryError::CoincidentPoints);
        }
        let dx = coordinate_difference(&p2.x, &p1.x);
        let dy = coordinate_difference(&p2.y, &p1.y);
        let dz = coordinate_difference(&p2.z, &p1.z);
        match classify_zero_vector([&dx, &dy, &dz]) {
            ZeroVectorStatus::Zero => Err(GeometryError::CoincidentPoints),
            ZeroVectorStatus::NonZero => Ok(Self { p1, p2 }),
            ZeroVectorStatus::Unknown => Err(GeometryError::SymbolicDegeneracyUndetermined),
        }
    }

    pub fn p1(&self) -> &Point3D {
        &self.p1
    }

    pub fn p2(&self) -> &Point3D {
        &self.p2
    }

    /// Direction vector (dx, dy, dz) = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z).
    pub fn direction(&self) -> Point3D {
        let dx = simplify(&Expr::Add(vec![
            self.p2.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.p1.x.clone()]),
        ]));
        let dy = simplify(&Expr::Add(vec![
            self.p2.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.p1.y.clone()]),
        ]));
        let dz = simplify(&Expr::Add(vec![
            self.p2.z.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.p1.z.clone()]),
        ]));
        Point3D::new(dx, dy, dz)
    }
}

impl fmt::Display for Line3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Line3D({}, {})", self.p1, self.p2)
    }
}

/// 3D Symbolic Ray emanating from `source` passing through `point`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ray3D {
    source: Point3D,
    point: Point3D,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ray3DWire {
    source: Point3D,
    point: Point3D,
}

impl<'de> Deserialize<'de> for Ray3D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Ray3DWire::deserialize(deserializer)?;
        Self::new(wire.source, wire.point).map_err(serde::de::Error::custom)
    }
}

impl Ray3D {
    pub fn new(source: Point3D, point: Point3D) -> Result<Self, GeometryError> {
        if source == point {
            return Err(GeometryError::CoincidentPoints);
        }
        let dx = coordinate_difference(&point.x, &source.x);
        let dy = coordinate_difference(&point.y, &source.y);
        let dz = coordinate_difference(&point.z, &source.z);
        match classify_zero_vector([&dx, &dy, &dz]) {
            ZeroVectorStatus::Zero => Err(GeometryError::CoincidentPoints),
            ZeroVectorStatus::NonZero => Ok(Self { source, point }),
            ZeroVectorStatus::Unknown => Err(GeometryError::SymbolicDegeneracyUndetermined),
        }
    }

    pub fn source(&self) -> &Point3D {
        &self.source
    }

    pub fn point(&self) -> &Point3D {
        &self.point
    }

    /// Direction vector (dx, dy, dz) = (point.x - source.x, point.y - source.y, point.z - source.z).
    pub fn direction(&self) -> Point3D {
        let dx = simplify(&Expr::Add(vec![
            self.point.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.source.x.clone()]),
        ]));
        let dy = simplify(&Expr::Add(vec![
            self.point.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.source.y.clone()]),
        ]));
        let dz = simplify(&Expr::Add(vec![
            self.point.z.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.source.z.clone()]),
        ]));
        Point3D::new(dx, dy, dz)
    }
}

impl fmt::Display for Ray3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ray3D({}, {})", self.source, self.point)
    }
}

/// 3D Symbolic Plane passing through `point` with `normal` vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plane3D {
    point: Point3D,
    normal: Point3D,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plane3DWire {
    point: Point3D,
    normal: Point3D,
}

impl<'de> Deserialize<'de> for Plane3D {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = Plane3DWire::deserialize(deserializer)?;
        Self::new(wire.point, wire.normal).map_err(serde::de::Error::custom)
    }
}

impl Plane3D {
    pub fn new(point: Point3D, normal: Point3D) -> Result<Self, GeometryError> {
        match classify_zero_vector([&normal.x, &normal.y, &normal.z]) {
            ZeroVectorStatus::Zero => Err(GeometryError::DegeneratePlane),
            ZeroVectorStatus::NonZero => Ok(Self { point, normal }),
            ZeroVectorStatus::Unknown => Err(GeometryError::SymbolicDegeneracyUndetermined),
        }
    }

    pub fn from_three_points(p1: Point3D, p2: Point3D, p3: Point3D) -> Result<Self, GeometryError> {
        // v1 = p2 - p1, v2 = p3 - p1
        let v1x = simplify(&Expr::Add(vec![
            p2.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), p1.x.clone()]),
        ]));
        let v1y = simplify(&Expr::Add(vec![
            p2.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), p1.y.clone()]),
        ]));
        let v1z = simplify(&Expr::Add(vec![
            p2.z.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), p1.z.clone()]),
        ]));

        let v2x = simplify(&Expr::Add(vec![
            p3.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), p1.x.clone()]),
        ]));
        let v2y = simplify(&Expr::Add(vec![
            p3.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), p1.y.clone()]),
        ]));
        let v2z = simplify(&Expr::Add(vec![
            p3.z.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), p1.z.clone()]),
        ]));

        // cross product: n = v1 x v2
        let nx = simplify(&Expr::Add(vec![
            Expr::Mul(vec![v1y.clone(), v2z.clone()]),
            Expr::Mul(vec![Expr::from_i64(-1), v1z.clone(), v2y.clone()]),
        ]));
        let ny = simplify(&Expr::Add(vec![
            Expr::Mul(vec![v1z, v2x.clone()]),
            Expr::Mul(vec![Expr::from_i64(-1), v1x.clone(), v2z]),
        ]));
        let nz = simplify(&Expr::Add(vec![
            Expr::Mul(vec![v1x, v2y]),
            Expr::Mul(vec![Expr::from_i64(-1), v1y, v2x]),
        ]));

        let normal = Point3D::new(nx, ny, nz);
        Self::new(p1, normal)
    }

    pub fn point(&self) -> &Point3D {
        &self.point
    }

    pub fn normal(&self) -> &Point3D {
        &self.normal
    }

    /// Evaluates plane equation E(q) = n . (q - p). If 0, q is on the plane.
    pub fn eval_at_point(&self, q: &Point3D) -> Expr {
        let dx = Expr::Add(vec![
            q.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.point.x.clone()]),
        ]);
        let dy = Expr::Add(vec![
            q.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.point.y.clone()]),
        ]);
        let dz = Expr::Add(vec![
            q.z.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), self.point.z.clone()]),
        ]);

        let dot = Expr::Add(vec![
            Expr::Mul(vec![self.normal.x.clone(), dx]),
            Expr::Mul(vec![self.normal.y.clone(), dy]),
            Expr::Mul(vec![self.normal.z.clone(), dz]),
        ]);
        simplify(&dot)
    }

    /// Computes squared perpendicular distance from a 3D point to this plane:
    /// d^2 = (n . (q - p))^2 / (nx^2 + ny^2 + nz^2).
    pub fn distance_squared(&self, q: &Point3D) -> Expr {
        let eval = self.eval_at_point(q);
        let num = Expr::Pow(Arc::new(eval), Arc::new(Expr::from_i64(2)));
        let nx2 = Expr::Pow(Arc::new(self.normal.x.clone()), Arc::new(Expr::from_i64(2)));
        let ny2 = Expr::Pow(Arc::new(self.normal.y.clone()), Arc::new(Expr::from_i64(2)));
        let nz2 = Expr::Pow(Arc::new(self.normal.z.clone()), Arc::new(Expr::from_i64(2)));
        let den = simplify(&Expr::Add(vec![nx2, ny2, nz2]));
        expr_div(simplify(&num), den)
    }

    /// Computes intersection point with a Line3D.
    pub fn intersection_line(&self, line: &Line3D) -> Result<Point3D, GeometryError> {
        let dir = line.direction();
        // n . dir
        let denom = simplify(&Expr::Add(vec![
            Expr::Mul(vec![self.normal.x.clone(), dir.x.clone()]),
            Expr::Mul(vec![self.normal.y.clone(), dir.y.clone()]),
            Expr::Mul(vec![self.normal.z.clone(), dir.z.clone()]),
        ]));

        match numeric_value(&denom) {
            Some(value) if value.numer().is_zero() => {
                return Err(GeometryError::ParallelLines);
            }
            Some(_) => {}
            None => {
                if denom.is_zero() {
                    return Err(GeometryError::ParallelLines);
                }
                return Err(GeometryError::SymbolicDegeneracyUndetermined);
            }
        }

        // t = n . (p_plane - p_line1) / (n . dir)
        let diff_x = Expr::Add(vec![
            self.point.x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), line.p1.x.clone()]),
        ]);
        let diff_y = Expr::Add(vec![
            self.point.y.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), line.p1.y.clone()]),
        ]);
        let diff_z = Expr::Add(vec![
            self.point.z.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), line.p1.z.clone()]),
        ]);

        let numer = simplify(&Expr::Add(vec![
            Expr::Mul(vec![self.normal.x.clone(), diff_x]),
            Expr::Mul(vec![self.normal.y.clone(), diff_y]),
            Expr::Mul(vec![self.normal.z.clone(), diff_z]),
        ]));

        let t = expr_div(numer, denom);

        let ix = simplify(&Expr::Add(vec![
            line.p1.x.clone(),
            Expr::Mul(vec![t.clone(), dir.x]),
        ]));
        let iy = simplify(&Expr::Add(vec![
            line.p1.y.clone(),
            Expr::Mul(vec![t.clone(), dir.y]),
        ]));
        let iz = simplify(&Expr::Add(vec![
            line.p1.z.clone(),
            Expr::Mul(vec![t, dir.z]),
        ]));

        Ok(Point3D::new(ix, iy, iz))
    }
}

impl fmt::Display for Plane3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Plane3D(p={}, n={})", self.point, self.normal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::de::DeserializeSeed;
    use serde::de::value::Error as ValueError;
    use std::cell::Cell;
    use std::rc::Rc;

    struct HintPointSeq {
        remaining: usize,
        claimed: Option<usize>,
        next_calls: Rc<Cell<usize>>,
        vertex: serde_json::Value,
    }

    impl HintPointSeq {
        fn new(remaining: usize, claimed: Option<usize>, next_calls: Rc<Cell<usize>>) -> Self {
            Self {
                remaining,
                claimed,
                next_calls,
                vertex: serde_json::to_value(Point2D::new(Expr::from_i64(0), Expr::from_i64(0)))
                    .expect("test point must serialize"),
            }
        }
    }

    impl<'de> SeqAccess<'de> for HintPointSeq {
        type Error = serde_json::Error;

        fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where
            T: DeserializeSeed<'de>,
        {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            self.next_calls.set(self.next_calls.get() + 1);
            seed.deserialize(self.vertex.clone()).map(Some)
        }

        fn size_hint(&self) -> Option<usize> {
            self.claimed
        }
    }

    #[test]
    fn test_point_distance() {
        let p1 = Point2D::new(Expr::from_i64(0), Expr::from_i64(0));
        let p2 = Point2D::new(Expr::from_i64(3), Expr::from_i64(4));
        let dist_sq = p1.distance_squared(&p2);
        assert_eq!(dist_sq, Expr::from_i64(25));
    }

    #[test]
    fn test_line_intersection() {
        // Line 1: from (0, 0) to (2, 2) (y = x)
        // Line 2: from (0, 2) to (2, 0) (y = 2 - x)
        // Intersection: (1, 1)
        let l1 = Line2D::new(
            Point2D::new(Expr::from_i64(0), Expr::from_i64(0)),
            Point2D::new(Expr::from_i64(2), Expr::from_i64(2)),
        )
        .unwrap();
        let l2 = Line2D::new(
            Point2D::new(Expr::from_i64(0), Expr::from_i64(2)),
            Point2D::new(Expr::from_i64(2), Expr::from_i64(0)),
        )
        .unwrap();

        let pt = l1.intersection(&l2).unwrap();
        assert_eq!(pt.x, Expr::from_i64(1));
        assert_eq!(pt.y, Expr::from_i64(1));
    }

    #[test]
    fn test_triangle_centroid_and_collinearity() {
        let p1 = Point2D::new(Expr::from_i64(0), Expr::from_i64(0));
        let p2 = Point2D::new(Expr::from_i64(6), Expr::from_i64(0));
        let p3 = Point2D::new(Expr::from_i64(0), Expr::from_i64(6));

        let tri = Triangle2D::new(p1, p2, p3);
        assert_eq!(tri.is_collinear(), Some(false));
        let centroid = tri.centroid();
        assert_eq!(centroid.x, Expr::from_i64(2));
        assert_eq!(centroid.y, Expr::from_i64(2));

        // Collinear test
        let c1 = Point2D::new(Expr::from_i64(0), Expr::from_i64(0));
        let c2 = Point2D::new(Expr::from_i64(1), Expr::from_i64(1));
        let c3 = Point2D::new(Expr::from_i64(2), Expr::from_i64(2));
        let tri_col = Triangle2D::new(c1, c2, c3);
        assert_eq!(tri_col.is_collinear(), Some(true));
    }

    #[test]
    fn symbolic_geometry_preserves_degeneracy_uncertainty() {
        let a = Expr::symbol("a");
        let first = Line2D::new(
            Point2D::new(Expr::from_i64(0), Expr::from_i64(0)),
            Point2D::new(Expr::from_i64(1), a.clone()),
        )
        .unwrap();
        let second = Line2D::new(
            Point2D::new(Expr::from_i64(0), Expr::from_i64(1)),
            Point2D::new(Expr::from_i64(1), Expr::symbol("b")),
        )
        .unwrap();
        assert_eq!(
            first.intersection(&second),
            Err(GeometryError::SymbolicDegeneracyUndetermined)
        );

        let symbolic_triangle = Triangle2D::new(
            Point2D::new(Expr::from_i64(0), Expr::from_i64(0)),
            Point2D::new(Expr::from_i64(1), a),
            Point2D::new(Expr::from_i64(2), Expr::symbol("b")),
        );
        assert_eq!(symbolic_triangle.is_collinear(), None);
    }

    #[test]
    fn test_point3d_and_segment3d() {
        let p1 = Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(0));
        let p2 = Point3D::new(Expr::from_i64(2), Expr::from_i64(4), Expr::from_i64(4));
        assert_eq!(p1.distance_squared(&p2), Expr::from_i64(36));

        let seg = Segment3D::new(p1, p2);
        assert_eq!(seg.length_squared(), Expr::from_i64(36));
        let mid = seg.midpoint();
        assert_eq!(mid.x, Expr::from_i64(1));
        assert_eq!(mid.y, Expr::from_i64(2));
        assert_eq!(mid.z, Expr::from_i64(2));
    }

    #[test]
    fn test_sphere_and_circle_metrics() {
        let origin = Point2D::new(Expr::from_i64(0), Expr::from_i64(0));
        let circle = Circle::new(origin, Expr::from_i64(3)).unwrap();
        // Area = 9 * pi
        let expected_area = Expr::Mul(vec![
            Expr::from_i64(9),
            Expr::Const(fsym_core::Constant::Pi),
        ]);
        assert_eq!(circle.area(), expected_area);
        // Circumference = 6 * pi
        let expected_circ = Expr::Mul(vec![
            Expr::from_i64(6),
            Expr::Const(fsym_core::Constant::Pi),
        ]);
        assert_eq!(circle.circumference(), expected_circ);

        let origin3d = Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(0));
        let sphere = Sphere::new(origin3d, Expr::from_i64(3)).unwrap();
        // Surface area = 36 * pi
        let expected_sa = Expr::Mul(vec![
            Expr::from_i64(36),
            Expr::Const(fsym_core::Constant::Pi),
        ]);
        assert_eq!(sphere.surface_area(), expected_sa);
        // Volume = 36 * pi: (4/3) * pi * 27 = 36 * pi
        let expected_vol = Expr::Mul(vec![
            Expr::from_i64(36),
            Expr::Const(fsym_core::Constant::Pi),
        ]);
        assert_eq!(sphere.volume(), expected_vol);
    }

    #[test]
    fn test_polygon2d_area_centroid_and_wire_invariants() {
        // Square [0,0], [2,0], [2,2], [0,2] -> Area 4, double signed area 8, Centroid (1, 1)
        let vertices = vec![
            Point2D::new(Expr::from_i64(0), Expr::from_i64(0)),
            Point2D::new(Expr::from_i64(2), Expr::from_i64(0)),
            Point2D::new(Expr::from_i64(2), Expr::from_i64(2)),
            Point2D::new(Expr::from_i64(0), Expr::from_i64(2)),
        ];
        let poly = Polygon2D::new(vertices).unwrap();
        assert_eq!(poly.double_signed_area(), Expr::from_i64(8));
        let c = poly.centroid();
        assert_eq!(c.x, Expr::from_i64(1));
        assert_eq!(c.y, Expr::from_i64(1));

        // Less than 3 vertices rejected
        assert_eq!(
            Polygon2D::new(vec![
                Point2D::new(Expr::from_i64(0), Expr::from_i64(0)),
                Point2D::new(Expr::from_i64(1), Expr::from_i64(1)),
            ]),
            Err(GeometryError::DegeneratePolygon)
        );

        // Wire decoding validation
        let poly_wire = serde_json::to_value(&poly).unwrap();
        assert_eq!(
            serde_json::from_value::<Polygon2D>(poly_wire.clone()).unwrap(),
            poly
        );

        // Sphere wire decoding validation
        let origin3d = Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(0));
        let valid_sphere = Sphere::new(origin3d, Expr::from_i64(2)).unwrap();
        let mut sphere_wire = serde_json::to_value(&valid_sphere).unwrap();
        assert_eq!(
            serde_json::from_value::<Sphere>(sphere_wire.clone()).unwrap(),
            valid_sphere
        );
        sphere_wire.as_object_mut().unwrap().insert(
            "radius".to_owned(),
            serde_json::to_value(Expr::from_i64(-5)).unwrap(),
        );
        assert!(serde_json::from_value::<Sphere>(sphere_wire).is_err());
    }

    #[test]
    fn polygon_vertex_limit_distrusts_sequence_hints_and_stops_early() {
        let point = Point2D::new(Expr::from_i64(0), Expr::from_i64(0));
        assert_eq!(
            Polygon2D::new(vec![point; MAX_POLYGON_VERTICES + 1]),
            Err(GeometryError::PolygonVertexLimitExceeded {
                actual: MAX_POLYGON_VERTICES + 1,
                max: MAX_POLYGON_VERTICES,
            })
        );

        for hostile_hint in [Some(MAX_POLYGON_VERTICES + 1), Some(usize::MAX)] {
            let next_calls = Rc::new(Cell::new(0));
            let error = BoundedPolygonVerticesVisitor
                .visit_seq(HintPointSeq::new(
                    MAX_POLYGON_VERTICES + 2,
                    hostile_hint,
                    Rc::clone(&next_calls),
                ))
                .expect_err("an over-limit hint must fail before reading elements");
            assert!(error.to_string().contains("at most 8192"));
            assert_eq!(next_calls.get(), 0);
        }

        for underreported_hint in [None, Some(1), Some(MAX_POLYGON_VERTICES)] {
            let next_calls = Rc::new(Cell::new(0));
            let error = BoundedPolygonVerticesVisitor
                .visit_seq(HintPointSeq::new(
                    MAX_POLYGON_VERTICES + 2,
                    underreported_hint,
                    Rc::clone(&next_calls),
                ))
                .expect_err("an underreported sequence must not bypass the vertex limit");
            assert!(error.to_string().contains("at most 8192"));
            assert_eq!(
                next_calls.get(),
                MAX_POLYGON_VERTICES + 1,
                "the decoder must stop after observing one excess element"
            );
        }

        let next_calls = Rc::new(Cell::new(0));
        let vertices = BoundedPolygonVerticesVisitor
            .visit_seq(HintPointSeq::new(
                MAX_POLYGON_VERTICES,
                None,
                Rc::clone(&next_calls),
            ))
            .expect("the exact vertex limit is admitted by the streaming decoder");
        assert_eq!(vertices.len(), MAX_POLYGON_VERTICES);
        assert_eq!(next_calls.get(), MAX_POLYGON_VERTICES);
    }

    #[test]
    fn polygon_vertex_capacity_growth_and_allocation_failure_are_bounded() {
        assert_eq!(next_polygon_vertex_capacity(0), Some(1));
        assert_eq!(next_polygon_vertex_capacity(1), Some(2));
        assert_eq!(next_polygon_vertex_capacity(2), Some(4));
        assert_eq!(
            next_polygon_vertex_capacity(MAX_POLYGON_VERTICES - 1),
            Some(MAX_POLYGON_VERTICES)
        );
        assert_eq!(next_polygon_vertex_capacity(MAX_POLYGON_VERTICES), None);

        let mut vertices = Vec::<Point2D>::new();
        let error = try_reserve_polygon_vertices::<ValueError>(&mut vertices, usize::MAX)
            .expect_err("an impossible reservation must fail");
        assert!(error.to_string().contains("allocation refused"));
        assert!(vertices.is_empty());
    }

    #[test]
    fn symbolic_line_and_plane_construction_requires_a_proven_nonzero_vector() {
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let one = Expr::from_i64(1);

        assert_eq!(
            Line2D::new(
                Point2D::new(zero.clone(), zero.clone()),
                Point2D::new(x.clone(), zero.clone()),
            ),
            Err(GeometryError::SymbolicDegeneracyUndetermined)
        );
        assert!(
            Line2D::new(
                Point2D::new(x.clone(), zero.clone()),
                Point2D::new(x.clone(), one.clone()),
            )
            .is_ok(),
            "one exact nonzero coordinate difference proves distinctness"
        );

        assert_eq!(
            Line3D::new(
                Point3D::new(zero.clone(), zero.clone(), zero.clone()),
                Point3D::new(x.clone(), zero.clone(), zero.clone()),
            ),
            Err(GeometryError::SymbolicDegeneracyUndetermined)
        );
        assert!(
            Line3D::new(
                Point3D::new(x.clone(), zero.clone(), zero.clone()),
                Point3D::new(x.clone(), zero.clone(), one.clone()),
            )
            .is_ok(),
            "one exact nonzero coordinate difference proves distinctness"
        );

        let origin = Point3D::new(zero.clone(), zero.clone(), zero.clone());
        assert_eq!(
            Plane3D::new(
                origin.clone(),
                Point3D::new(x.clone(), zero.clone(), zero.clone()),
            ),
            Err(GeometryError::SymbolicDegeneracyUndetermined)
        );
        assert!(
            Plane3D::new(
                origin.clone(),
                Point3D::new(x.clone(), one.clone(), zero.clone()),
            )
            .is_ok(),
            "one exact nonzero normal component proves a nonzero vector"
        );

        let symbolic_collinearity = Plane3D::from_three_points(
            origin.clone(),
            Point3D::new(x.clone(), zero.clone(), zero.clone()),
            Point3D::new(zero.clone(), one, zero.clone()),
        );
        assert_eq!(
            symbolic_collinearity,
            Err(GeometryError::SymbolicDegeneracyUndetermined)
        );

        let cancelled = Expr::Add(vec![
            x.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), x.clone()]),
        ]);
        assert_eq!(
            Plane3D::new(origin, Point3D::new(cancelled, zero.clone(), zero.clone()),),
            Err(GeometryError::DegeneratePlane),
            "algebraically zero components remain a typed zero-vector refusal"
        );

        let uncertain_plane_wire = serde_json::json!({
            "point": Point3D::new(zero.clone(), zero.clone(), zero.clone()),
            "normal": Point3D::new(x, zero.clone(), zero),
        });
        let error = serde_json::from_value::<Plane3D>(uncertain_plane_wire)
            .expect_err("wire decoding must replay the symbolic degeneracy check");
        assert!(error.to_string().contains("undecidable"));
    }

    #[test]
    fn test_line3d_and_plane3d_metrics_and_wire_invariants() {
        // Line3D through (0,0,0) and (1,2,3)
        let p1 = Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(0));
        let p2 = Point3D::new(Expr::from_i64(1), Expr::from_i64(2), Expr::from_i64(3));
        let line = Line3D::new(p1.clone(), p2.clone()).unwrap();
        let dir = line.direction();
        assert_eq!(dir.x, Expr::from_i64(1));
        assert_eq!(dir.y, Expr::from_i64(2));
        assert_eq!(dir.z, Expr::from_i64(3));

        // Coincident points rejected for Line3D
        assert_eq!(
            Line3D::new(p1.clone(), p1.clone()),
            Err(GeometryError::CoincidentPoints)
        );

        // Plane3D from 3 points: (0,0,0), (1,0,0), (0,1,0) -> XY-plane, normal = (0,0,1)
        let pt1 = Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(0));
        let pt2 = Point3D::new(Expr::from_i64(1), Expr::from_i64(0), Expr::from_i64(0));
        let pt3 = Point3D::new(Expr::from_i64(0), Expr::from_i64(1), Expr::from_i64(0));
        let xy_plane = Plane3D::from_three_points(pt1, pt2, pt3).unwrap();
        assert_eq!(xy_plane.normal().z, Expr::from_i64(1));

        // Point (0,0,5) has distance^2 = 25 to XY-plane
        let q = Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(5));
        assert_eq!(xy_plane.distance_squared(&q), Expr::from_i64(25));

        // Line from (0,0,5) to (1,1,4) intersects XY-plane at (5, 5, 0)
        let l_intersect = Line3D::new(
            Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(5)),
            Point3D::new(Expr::from_i64(1), Expr::from_i64(1), Expr::from_i64(4)),
        )
        .unwrap();
        let inter_pt = xy_plane.intersection_line(&l_intersect).unwrap();
        assert_eq!(inter_pt.x, Expr::from_i64(5));
        assert_eq!(inter_pt.y, Expr::from_i64(5));
        assert_eq!(inter_pt.z, Expr::from_i64(0));

        // Parallel line (0,0,1) to (1,0,1) to XY plane returns ParallelLines error
        let l_parallel = Line3D::new(
            Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(1)),
            Point3D::new(Expr::from_i64(1), Expr::from_i64(0), Expr::from_i64(1)),
        )
        .unwrap();
        assert_eq!(
            xy_plane.intersection_line(&l_parallel),
            Err(GeometryError::ParallelLines)
        );

        // Degenerate plane with zero normal rejected
        let zero_normal = Point3D::new(Expr::from_i64(0), Expr::from_i64(0), Expr::from_i64(0));
        assert_eq!(
            Plane3D::new(p1.clone(), zero_normal),
            Err(GeometryError::DegeneratePlane)
        );

        // Serde roundtrips
        let plane_wire = serde_json::to_value(&xy_plane).unwrap();
        assert_eq!(
            serde_json::from_value::<Plane3D>(plane_wire).unwrap(),
            xy_plane
        );
        let line_wire = serde_json::to_value(&line).unwrap();
        assert_eq!(serde_json::from_value::<Line3D>(line_wire).unwrap(), line);
    }

    #[test]
    fn test_ray2d_and_ray3d_construction_direction_and_serde() {
        // Ray2D
        let s2 = Point2D::new(Expr::from_i64(1), Expr::from_i64(2));
        let p2 = Point2D::new(Expr::from_i64(4), Expr::from_i64(6));
        let ray2 = Ray2D::new(s2.clone(), p2.clone()).unwrap();
        assert_eq!(ray2.source(), &s2);
        assert_eq!(ray2.point(), &p2);
        let dir2 = ray2.direction();
        assert_eq!(dir2.x, Expr::from_i64(3));
        assert_eq!(dir2.y, Expr::from_i64(4));

        // Coincident points error
        assert_eq!(
            Ray2D::new(s2.clone(), s2.clone()),
            Err(GeometryError::CoincidentPoints)
        );

        // Ray2D Serde roundtrip
        let ray2_wire = serde_json::to_value(&ray2).unwrap();
        assert_eq!(serde_json::from_value::<Ray2D>(ray2_wire).unwrap(), ray2);

        // Ray3D
        let s3 = Point3D::new(Expr::from_i64(1), Expr::from_i64(2), Expr::from_i64(3));
        let p3 = Point3D::new(Expr::from_i64(3), Expr::from_i64(5), Expr::from_i64(7));
        let ray3 = Ray3D::new(s3.clone(), p3.clone()).unwrap();
        assert_eq!(ray3.source(), &s3);
        assert_eq!(ray3.point(), &p3);
        let dir3 = ray3.direction();
        assert_eq!(dir3.x, Expr::from_i64(2));
        assert_eq!(dir3.y, Expr::from_i64(3));
        assert_eq!(dir3.z, Expr::from_i64(4));

        // Coincident points error
        assert_eq!(
            Ray3D::new(s3.clone(), s3.clone()),
            Err(GeometryError::CoincidentPoints)
        );

        // Ray3D Serde roundtrip
        let ray3_wire = serde_json::to_value(&ray3).unwrap();
        assert_eq!(serde_json::from_value::<Ray3D>(ray3_wire).unwrap(), ray3);
    }
}
