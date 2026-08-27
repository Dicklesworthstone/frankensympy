//! Certified real ball arithmetic (WS11).
//!
//! A real ball $\mathcal{B}(m, r) = [m - r, m + r]$ represents a certified enclosure
//! of a real number with rational midpoint $m \in \mathbb{Q}$ and non-negative rational radius $r \in \mathbb{Q}_{\ge 0}$.

#![forbid(unsafe_code)]

use crate::{BigInt, BigRational};
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BallError {
    #[error("Division by ball containing zero: {0}")]
    DivisionByZero(String),
    #[error("Negative radius is invalid: {0}")]
    NegativeRadius(String),
}

/// Certified real ball $\mathcal{B}(m, r) = [m - r, m + r]$.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct RealBall {
    midpoint: BigRational,
    radius: BigRational,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealBallWire {
    midpoint: BigRational,
    radius: BigRational,
}

impl<'de> Deserialize<'de> for RealBall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RealBallWire::deserialize(deserializer)?;
        Self::new(wire.midpoint, wire.radius).map_err(serde::de::Error::custom)
    }
}

impl RealBall {
    /// Constructs a certified ball with given midpoint and non-negative radius.
    pub fn new(midpoint: BigRational, radius: BigRational) -> Result<Self, BallError> {
        if radius < BigRational::zero() {
            return Err(BallError::NegativeRadius(radius.to_string()));
        }
        Ok(Self { midpoint, radius })
    }

    /// Construct an exact point ball with zero radius: $\mathcal{B}(q, 0) = [q, q]$.
    pub fn exact(q: BigRational) -> Self {
        Self {
            midpoint: q,
            radius: BigRational::zero(),
        }
    }

    /// Construct an exact integer ball: $\mathcal{B}(n, 0)$.
    pub fn from_i64(n: i64) -> Self {
        Self::exact(BigRational::from_integer(BigInt::from(n)))
    }

    /// Lower bound $m - r$.
    pub fn lower(&self) -> BigRational {
        &self.midpoint - &self.radius
    }

    /// Upper bound $m + r$.
    pub fn upper(&self) -> BigRational {
        &self.midpoint + &self.radius
    }

    /// Midpoint $m$.
    pub fn midpoint(&self) -> &BigRational {
        &self.midpoint
    }

    /// Radius $r$.
    pub fn radius(&self) -> &BigRational {
        &self.radius
    }

    /// Width / diameter $2r$.
    pub fn width(&self) -> BigRational {
        self.diameter()
    }

    /// Checks if this ball is completely disjoint from another ball.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.upper() < other.lower() || other.upper() < self.lower()
    }

    /// Diameter $2r$.
    pub fn diameter(&self) -> BigRational {
        &self.radius * BigRational::from_integer(BigInt::from(2))
    }

    /// Check if point $x \in \mathcal{B}(m, r)$.
    pub fn contains(&self, x: &BigRational) -> bool {
        x >= &self.lower() && x <= &self.upper()
    }

    /// Check if another ball is entirely contained within this ball: $B_2 \subseteq B_1$.
    pub fn contains_ball(&self, other: &Self) -> bool {
        other.lower() >= self.lower() && other.upper() <= self.upper()
    }

    /// Check if 0 is contained within the ball ($0 \in [m - r, m + r]$).
    pub fn contains_zero(&self) -> bool {
        self.lower() <= BigRational::zero() && self.upper() >= BigRational::zero()
    }

    /// Strictly positive: lower bound > 0.
    pub fn is_positive(&self) -> bool {
        self.lower() > BigRational::zero()
    }

    /// Strictly negative: upper bound < 0.
    pub fn is_negative(&self) -> bool {
        self.upper() < BigRational::zero()
    }

    /// Certified ball addition: $[m_1 - r_1, m_1 + r_1] + [m_2 - r_2, m_2 + r_2] = [m_1 + m_2, r_1 + r_2]$.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            midpoint: &self.midpoint + &other.midpoint,
            radius: &self.radius + &other.radius,
        }
    }

    /// Certified ball subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            midpoint: &self.midpoint - &other.midpoint,
            radius: &self.radius + &other.radius,
        }
    }

    /// Certified ball negation.
    pub fn neg(&self) -> Self {
        Self {
            midpoint: -&self.midpoint,
            radius: self.radius.clone(),
        }
    }

    /// Certified ball multiplication:
    /// $\mathcal{B}(m_1, r_1) \cdot \mathcal{B}(m_2, r_2) = \mathcal{B}(m_1 m_2, |m_1| r_2 + |m_2| r_1 + r_1 r_2)$.
    pub fn mul(&self, other: &Self) -> Self {
        let new_midpoint = &self.midpoint * &other.midpoint;
        let abs_m1 = self.midpoint.abs();
        let abs_m2 = other.midpoint.abs();

        let new_radius =
            (&abs_m1 * &other.radius) + (&abs_m2 * &self.radius) + (&self.radius * &other.radius);

        Self {
            midpoint: new_midpoint,
            radius: new_radius,
        }
    }

    /// Certified ball inversion: $1 / \mathcal{B}(m, r)$ when $0 \notin \mathcal{B}(m, r)$.
    pub fn inv(&self) -> Result<Self, BallError> {
        if self.contains_zero() {
            return Err(BallError::DivisionByZero(self.to_string()));
        }
        let low = self.lower();
        let high = self.upper();

        let inv_low = BigRational::one() / &high;
        let inv_high = BigRational::one() / &low;

        let mid = (&inv_low + &inv_high) / BigRational::from_integer(BigInt::from(2));
        let rad = (&inv_high - &inv_low) / BigRational::from_integer(BigInt::from(2));

        Ok(Self {
            midpoint: mid,
            radius: rad.abs(),
        })
    }

    /// Certified ball division: $\mathcal{B}_1 / \mathcal{B}_2$.
    pub fn div(&self, other: &Self) -> Result<Self, BallError> {
        let inv_other = other.inv()?;
        Ok(self.mul(&inv_other))
    }

    /// Intersect two balls: returns tightest enclosing ball of the intersection if non-empty.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let low = self.lower().max(other.lower());
        let high = self.upper().min(other.upper());

        if low > high {
            None
        } else {
            let mid = (&low + &high) / BigRational::from_integer(BigInt::from(2));
            let rad = (&high - &low) / BigRational::from_integer(BigInt::from(2));
            Some(Self {
                midpoint: mid,
                radius: rad,
            })
        }
    }
}

impl fmt::Display for RealBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} ± {}]", self.midpoint, self.radius)
    }
}
