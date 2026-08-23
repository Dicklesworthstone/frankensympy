//! # fsym-geometry
//!
//! Symbolic 2D and 3D geometry: points, lines, segments, rays, circles, polygons,
//! intersections, and distances.

#![forbid(unsafe_code)]

use fsym_core::Expr;
use fsym_simplify::simplify;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GeometryError {
    #[error("Coincident points cannot define a unique line")]
    CoincidentPoints,
    #[error("Invalid radius: radius cannot be negative")]
    NegativeRadius,
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

/// 2D Symbolic Circle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Circle {
    pub center: Point2D,
    pub radius: Expr,
}

impl Circle {
    pub fn new(center: Point2D, radius: Expr) -> Self {
        Self { center, radius }
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
}
