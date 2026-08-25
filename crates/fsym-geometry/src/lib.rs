//! # fsym-geometry
//!
//! Symbolic 2D and 3D geometry: points, lines, segments, rays, circles, polygons,
//! intersections, and distances (WS20).

#![forbid(unsafe_code)]

use fsym_core::{BigRational, Expr};
use fsym_simplify::simplify;
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

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
        Ok(Self { p1, p2 })
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
}

impl fmt::Display for Circle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Circle({}, r={})", self.center, self.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn circle_rejects_provably_negative_radius() {
        let origin = Point2D::new(Expr::from_i64(0), Expr::from_i64(0));
        assert_eq!(
            Circle::new(origin.clone(), Expr::from_i64(-1)),
            Err(GeometryError::NegativeRadius)
        );
        assert!(Circle::new(origin, Expr::symbol("r")).is_ok());
    }

    #[test]
    fn geometry_wire_decode_cannot_bypass_constructor_invariants() {
        let origin = Point2D::new(Expr::from_i64(0), Expr::from_i64(0));
        let valid_line = Line2D::new(
            origin.clone(),
            Point2D::new(Expr::from_i64(1), Expr::from_i64(1)),
        )
        .unwrap();
        let mut line_wire = serde_json::to_value(&valid_line).unwrap();
        assert_eq!(
            serde_json::from_value::<Line2D>(line_wire.clone()).unwrap(),
            valid_line
        );
        let duplicate_point = line_wire.get("p1").unwrap().clone();
        line_wire
            .as_object_mut()
            .unwrap()
            .insert("p2".to_owned(), duplicate_point);
        assert!(serde_json::from_value::<Line2D>(line_wire).is_err());

        let valid_circle = Circle::new(origin, Expr::from_i64(1)).unwrap();
        let mut circle_wire = serde_json::to_value(&valid_circle).unwrap();
        assert_eq!(
            serde_json::from_value::<Circle>(circle_wire.clone()).unwrap(),
            valid_circle
        );
        circle_wire.as_object_mut().unwrap().insert(
            "radius".to_owned(),
            serde_json::to_value(Expr::from_i64(-1)).unwrap(),
        );
        assert!(serde_json::from_value::<Circle>(circle_wire).is_err());
    }
}
