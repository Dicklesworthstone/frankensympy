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

    /// Computes the canonical BLAKE3 content digest of this certified ball.
    pub fn digest(&self) -> [u8; 32] {
        let serialized = serde_json::to_vec(self).expect("RealBall is serializable");
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.real_ball.v1:");
        hasher.update(&serialized);
        *hasher.finalize().as_bytes()
    }
}

impl fmt::Display for RealBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} ± {}]", self.midpoint, self.radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(i: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(i))
    }

    #[test]
    fn new_rejects_negative_radius() {
        let err = RealBall::new(q(0), BigRational::from_integer(BigInt::from(-1))).unwrap_err();
        assert!(matches!(err, BallError::NegativeRadius(_)));
        // Zero radius is allowed: an exact point ball.
        assert!(RealBall::new(q(0), q(0)).is_ok());
    }

    #[test]
    fn exact_and_from_i64_construct_point_balls() {
        let b = RealBall::exact(BigRational::new(1.into(), 3.into()));
        assert_eq!(b.lower(), BigRational::new(1.into(), 3.into()));
        assert_eq!(b.upper(), BigRational::new(1.into(), 3.into()));
        assert!(b.radius().is_zero());

        let b = RealBall::from_i64(5);
        assert_eq!(b.lower(), q(5));
        assert_eq!(b.upper(), q(5));
    }

    #[test]
    fn contains_and_contains_ball_and_contains_zero() {
        let ball = RealBall::new(q(0), q(2)).unwrap();
        assert!(ball.contains(&q(0)));
        assert!(ball.contains(&q(2)));
        assert!(ball.contains(&BigRational::from_integer(BigInt::from(-2))));
        assert!(!ball.contains(&q(3)));
        assert!(!ball.contains(&BigRational::from_integer(BigInt::from(-3))));
        assert!(ball.contains_zero());

        // contains_ball is inclusive on the boundaries.
        let inner = RealBall::new(q(0), q(1)).unwrap();
        let boundary = RealBall::new(q(2), q(0)).unwrap();
        let outside = RealBall::new(q(5), q(0)).unwrap();
        assert!(ball.contains_ball(&inner));
        assert!(ball.contains_ball(&boundary));
        assert!(!ball.contains_ball(&outside));
    }

    #[test]
    fn is_positive_and_is_negative_use_strict_inequalities() {
        // The boundary point zero is neither strictly positive nor
        // strictly negative.
        let zero = RealBall::exact(q(0));
        assert!(!zero.is_positive());
        assert!(!zero.is_negative());

        // A ball that includes zero but is mostly positive is not
        // strictly positive.
        let cross = RealBall::new(q(0), q(1)).unwrap();
        assert!(!cross.is_positive());
        assert!(!cross.is_negative());

        // A ball entirely above zero.
        let pos = RealBall::new(q(2), q(1)).unwrap();
        assert!(pos.is_positive());
        assert!(!pos.is_negative());
    }

    #[test]
    fn add_sub_neg_preserve_radius_rules() {
        let a = RealBall::new(q(1), q(2)).unwrap();
        let b = RealBall::new(q(3), q(4)).unwrap();
        let sum = a.add(&b);
        assert_eq!(sum.midpoint(), &q(4));
        // Radii ADD under ball addition, not the triangle inequality.
        assert_eq!(sum.radius(), &q(6));

        let diff = a.sub(&b);
        assert_eq!(
            diff.midpoint(),
            &BigRational::from_integer(BigInt::from(-2))
        );
        assert_eq!(diff.radius(), &q(6));

        let neg = a.neg();
        assert_eq!(neg.midpoint(), &BigRational::from_integer(BigInt::from(-1)));
        assert_eq!(neg.radius(), &q(2));
    }

    #[test]
    fn mul_uses_abs_midpoint_radius_formula() {
        // (m1, r1) * (m2, r2) has midpoint m1*m2 and radius
        // |m1|*r2 + |m2|*r1 + r1*r2.
        let a = RealBall::new(q(2), q(1)).unwrap();
        let b = RealBall::new(q(3), q(1)).unwrap();
        let prod = a.mul(&b);
        assert_eq!(prod.midpoint(), &q(6));
        // 2*1 + 3*1 + 1*1 = 6
        assert_eq!(prod.radius(), &q(6));

        // Negative midpoint: (m1=-2, r1=1)*(m2=3, r2=1) has midpoint
        // -6 and radius |-2|*1 + 3*1 + 1*1 = 6.
        let neg = RealBall::new(BigRational::from_integer(BigInt::from(-2)), q(1)).unwrap();
        let prod = neg.mul(&b);
        assert_eq!(
            prod.midpoint(),
            &BigRational::from_integer(BigInt::from(-6))
        );
        assert_eq!(prod.radius(), &q(6));
    }

    #[test]
    fn inv_rejects_balls_containing_zero() {
        let crossing = RealBall::new(q(0), q(1)).unwrap();
        assert!(matches!(crossing.inv(), Err(BallError::DivisionByZero(_))));

        let zero_point = RealBall::exact(q(0));
        assert!(matches!(
            zero_point.inv(),
            Err(BallError::DivisionByZero(_))
        ));
    }

    #[test]
    fn inv_is_consistent_for_positive_balls() {
        // 1 / [3, 5] should land in [1/5, 1/3] with midpoint 4/15.
        let ball = RealBall::new(q(3), q(2)).unwrap();
        let inv = ball.inv().unwrap();
        // Lower = 1/5, upper = 1/3.
        assert!(inv.lower() <= BigRational::new(1.into(), 5.into()));
        assert!(inv.upper() >= BigRational::new(1.into(), 3.into()));
    }

    #[test]
    fn intersect_returns_none_for_disjoint() {
        let a = RealBall::new(q(0), q(1)).unwrap();
        let b = RealBall::new(q(5), q(1)).unwrap();
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn intersect_returns_tightest_common_enclosure() {
        let a = RealBall::new(q(0), q(3)).unwrap();
        // Ball a = [-3, 3]
        let b = RealBall::new(q(2), q(3)).unwrap();
        // Ball b = [-1, 5]
        let intersection = a.intersect(&b).unwrap();
        // max(-3, -1) = -1, min(3, 5) = 3
        assert_eq!(
            intersection.lower(),
            BigRational::from_integer(BigInt::from(-1))
        );
        assert_eq!(intersection.upper(), q(3));
    }

    #[test]
    fn digest_is_deterministic_and_distinct() {
        let b1 = RealBall::new(q(1), q(2)).unwrap();
        let b2 = RealBall::new(q(1), q(2)).unwrap();
        let b3 = RealBall::new(q(1), q(3)).unwrap();
        assert_eq!(b1.digest(), b2.digest());
        assert_ne!(b1.digest(), b3.digest());
        assert_ne!(b1.digest(), [0u8; 32]);
    }
}
