//! Exact algebraic number representation and certified root refinement (WS11).
//!
//! An algebraic number $\alpha \in \mathbb{R}$ is represented by:
//! 1. A square-free defining polynomial $P(x) \in \mathbb{Q}[x]$ such that $P(\alpha) = 0$.
//! 2. A certified root isolating ball $\mathcal{B}(m, r)$ containing exactly one real root of $P$.

#![forbid(unsafe_code)]

use crate::ball::{BallError, RealBall};
use crate::{BigInt, BigRational};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AlgebraicError {
    #[error("Zero polynomial cannot define an algebraic number")]
    ZeroPolynomial,
    #[error("Root isolating interval contains no root (or is not isolating): {0}")]
    InvalidIsolatingInterval(String),
    #[error("Ball arithmetic error: {0}")]
    Ball(#[from] BallError),
}

/// Exact real algebraic number with certified root isolating ball.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AlgebraicNumber {
    /// Coefficients in increasing degree: $c_0 + c_1 x + \dots + c_n x^n$.
    pub min_poly_coeffs: Vec<BigRational>,
    /// Certified ball enclosing the unique target root.
    pub isolating_ball: RealBall,
}

impl AlgebraicNumber {
    /// Constructs an algebraic number with defining polynomial and root isolating ball.
    pub fn new(
        mut min_poly_coeffs: Vec<BigRational>,
        isolating_ball: RealBall,
    ) -> Result<Self, AlgebraicError> {
        while min_poly_coeffs.len() > 1 && min_poly_coeffs.last().is_some_and(|c| c.is_zero()) {
            min_poly_coeffs.pop();
        }
        if min_poly_coeffs.is_empty()
            || (min_poly_coeffs.len() == 1 && min_poly_coeffs[0].is_zero())
        {
            return Err(AlgebraicError::ZeroPolynomial);
        }

        // Verify root existence by intermediate value theorem: P(low) * P(high) <= 0
        let low = isolating_ball.lower();
        let high = isolating_ball.upper();
        let p_low = Self::eval_poly_at(&min_poly_coeffs, &low);
        let p_high = Self::eval_poly_at(&min_poly_coeffs, &high);

        if &p_low * &p_high > BigRational::zero() {
            return Err(AlgebraicError::InvalidIsolatingInterval(format!(
                "P({}) = {}, P({}) = {} (no sign change)",
                low, p_low, high, p_high
            )));
        }

        Ok(Self {
            min_poly_coeffs,
            isolating_ball,
        })
    }

    /// Construct exact rational as a degree-1 algebraic number: $x - q = 0$.
    pub fn from_rational(q: BigRational) -> Self {
        Self {
            min_poly_coeffs: vec![-q.clone(), BigRational::one()],
            isolating_ball: RealBall::exact(q),
        }
    }

    /// Construct exact integer as an algebraic number: $x - n = 0$.
    pub fn from_i64(n: i64) -> Self {
        Self::from_rational(BigRational::from_integer(BigInt::from(n)))
    }

    /// Degree of defining polynomial.
    pub fn degree(&self) -> usize {
        self.min_poly_coeffs.len() - 1
    }

    /// Evaluates polynomial at a given rational point.
    fn eval_poly_at(coeffs: &[BigRational], x: &BigRational) -> BigRational {
        let mut acc = BigRational::zero();
        for c in coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    /// Evaluates polynomial over a certified ball using interval arithmetic.
    pub fn eval_ball(&self, ball: &RealBall) -> RealBall {
        let mut acc = RealBall::exact(BigRational::zero());
        for c in self.min_poly_coeffs.iter().rev() {
            acc = acc.mul(ball).add(&RealBall::exact(c.clone()));
        }
        acc
    }

    /// Refines the root isolating interval using bisection and certified enclosure steps.
    pub fn refine_step(&mut self) {
        let low = self.isolating_ball.lower();
        let high = self.isolating_ball.upper();
        let mid = (&low + &high) / BigRational::from_integer(BigInt::from(2));

        let p_mid = Self::eval_poly_at(&self.min_poly_coeffs, &mid);
        if p_mid.is_zero() {
            self.isolating_ball = RealBall::exact(mid);
            return;
        }

        let p_low = Self::eval_poly_at(&self.min_poly_coeffs, &low);
        if &p_low * &p_mid <= BigRational::zero() {
            let new_mid = (&low + &mid) / BigRational::from_integer(BigInt::from(2));
            let new_rad = (&mid - &low) / BigRational::from_integer(BigInt::from(2));
            self.isolating_ball = RealBall::new(new_mid, new_rad).expect("valid ball");
        } else {
            let new_mid = (&mid + &high) / BigRational::from_integer(BigInt::from(2));
            let new_rad = (&high - &mid) / BigRational::from_integer(BigInt::from(2));
            self.isolating_ball = RealBall::new(new_mid, new_rad).expect("valid ball");
        }
    }

    /// Refines the root isolating ball until its radius is at most `target_radius`.
    pub fn refine_to_radius(&mut self, target_radius: &BigRational) {
        while &self.isolating_ball.radius > target_radius {
            self.refine_step();
        }
    }

    /// Exact sign of algebraic number: returns -1, 0, or 1.
    pub fn sign(&mut self) -> i8 {
        if self.isolating_ball.is_positive() {
            return 1;
        }
        if self.isolating_ball.is_negative() {
            return -1;
        }
        // Check if 0 is a root: P(0) == c_0 == 0
        if self.min_poly_coeffs[0].is_zero() && self.isolating_ball.contains_zero() {
            return 0;
        }
        // Refine until isolating interval excludes 0
        while self.isolating_ball.contains_zero() && !self.isolating_ball.radius.is_zero() {
            self.refine_step();
        }

        if self.isolating_ball.is_positive() {
            1
        } else if self.isolating_ball.is_negative() {
            -1
        } else {
            0
        }
    }
}

impl fmt::Display for AlgebraicNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AlgebraicNumber(deg={}, ball={})",
            self.degree(),
            self.isolating_ball
        )
    }
}
