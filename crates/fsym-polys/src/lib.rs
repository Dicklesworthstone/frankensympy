//! # fsym-polys
//!
//! Polynomial algebra, polynomial rings, GCD, factorization, and root isolation
//! for FrankenSymPy.

#![forbid(unsafe_code)]

use fsym_core::{Expr, Symbol};
use num_rational::BigRational;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PolyError {
    #[error("Division by zero polynomial")]
    DivisionByZero,
    #[error("Incompatible polynomial ring generators: expected {0}, got {1}")]
    IncompatibleGenerators(String, String),
}

/// Univariate polynomial represented by coefficient vector:
/// `c_0 + c_1 * x + ... + c_n * x^n`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnivariatePoly {
    pub gen_sym: Symbol,
    /// Coefficients ordered by increasing degree: `coeffs[k]` is coefficient of `gen_sym^k`.
    pub coeffs: Vec<BigRational>,
}

impl UnivariatePoly {
    pub fn new(gen_sym: Symbol, mut coeffs: Vec<BigRational>) -> Self {
        while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            coeffs.push(BigRational::zero());
        }
        Self { gen_sym, coeffs }
    }

    pub fn zero(gen_sym: Symbol) -> Self {
        Self {
            gen_sym,
            coeffs: vec![BigRational::zero()],
        }
    }

    pub fn one(gen_sym: Symbol) -> Self {
        Self {
            gen_sym,
            coeffs: vec![BigRational::one()],
        }
    }

    pub fn degree(&self) -> Option<usize> {
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_zero()
    }

    pub fn leading_coeff(&self) -> &BigRational {
        self.coeffs.last().unwrap()
    }

    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        if self.gen_sym != other.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                self.gen_sym.name.clone(),
                other.gen_sym.name.clone(),
            ));
        }
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut new_coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self
                .coeffs
                .get(i)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            let b = other
                .coeffs
                .get(i)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            new_coeffs.push(a + b);
        }
        Ok(Self::new(self.gen_sym.clone(), new_coeffs))
    }

    pub fn mul(&self, other: &Self) -> Result<Self, PolyError> {
        if self.gen_sym != other.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                self.gen_sym.name.clone(),
                other.gen_sym.name.clone(),
            ));
        }
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero(self.gen_sym.clone()));
        }
        let deg_a = self.coeffs.len() - 1;
        let deg_b = other.coeffs.len() - 1;
        let mut res = vec![BigRational::zero(); deg_a + deg_b + 1];
        for (i, c_a) in self.coeffs.iter().enumerate() {
            for (j, c_b) in other.coeffs.iter().enumerate() {
                res[i + j] = &res[i + j] + (c_a * c_b);
            }
        }
        Ok(Self::new(self.gen_sym.clone(), res))
    }

    /// Polynomial division with remainder: self = quotient * divisor + remainder.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolyError> {
        if self.gen_sym != divisor.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                self.gen_sym.name.clone(),
                divisor.gen_sym.name.clone(),
            ));
        }
        if divisor.is_zero() {
            return Err(PolyError::DivisionByZero);
        }
        if self.is_zero() {
            return Ok((
                Self::zero(self.gen_sym.clone()),
                Self::zero(self.gen_sym.clone()),
            ));
        }

        let deg_div = divisor.degree().unwrap();
        let deg_self = match self.degree() {
            Some(d) => d,
            None => {
                return Ok((
                    Self::zero(self.gen_sym.clone()),
                    Self::zero(self.gen_sym.clone()),
                ));
            }
        };

        if deg_self < deg_div {
            return Ok((Self::zero(self.gen_sym.clone()), self.clone()));
        }

        let mut remainder = self.clone();
        let mut quotient_coeffs = vec![BigRational::zero(); deg_self - deg_div + 1];
        let lc_div = divisor.leading_coeff().clone();

        while let Some(deg_rem) = remainder.degree() {
            if deg_rem < deg_div {
                break;
            }
            let shift = deg_rem - deg_div;
            let lc_rem = remainder.leading_coeff().clone();
            let factor = &lc_rem / &lc_div;
            quotient_coeffs[shift] = &quotient_coeffs[shift] + &factor;

            let mut sub_coeffs = vec![BigRational::zero(); shift + divisor.coeffs.len()];
            for (i, c) in divisor.coeffs.iter().enumerate() {
                sub_coeffs[shift + i] = c * &factor;
            }
            let sub_poly = Self::new(self.gen_sym.clone(), sub_coeffs);
            let neg_sub_coeffs = sub_poly.coeffs.into_iter().map(|c| -c).collect();
            let neg_sub = Self::new(self.gen_sym.clone(), neg_sub_coeffs);
            remainder = remainder.add(&neg_sub)?;
        }

        Ok((Self::new(self.gen_sym.clone(), quotient_coeffs), remainder))
    }

    /// Greatest Common Divisor using Euclidean algorithm, monic normalized.
    pub fn gcd(&self, other: &Self) -> Result<Self, PolyError> {
        if self.gen_sym != other.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                self.gen_sym.name.clone(),
                other.gen_sym.name.clone(),
            ));
        }
        let mut a = self.clone();
        let mut b = other.clone();
        while !b.is_zero() {
            let (_, rem) = a.div_rem(&b)?;
            a = b;
            b = rem;
        }
        if a.is_zero() {
            return Ok(a);
        }
        let lc = a.leading_coeff().clone();
        let monic_coeffs = a.coeffs.into_iter().map(|c| c / &lc).collect();
        Ok(Self::new(self.gen_sym.clone(), monic_coeffs))
    }

    /// Convert polynomial back to symbolic Expr.
    pub fn to_expr(&self) -> Expr {
        if self.is_zero() {
            return Expr::from_i64(0);
        }
        let mut terms = Vec::new();
        for (deg, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            let coeff_expr = if coeff.is_integer() {
                Expr::Integer(coeff.to_integer())
            } else {
                Expr::Rational(coeff.clone())
            };
            if deg == 0 {
                terms.push(coeff_expr);
            } else if deg == 1 {
                if coeff.is_one() {
                    terms.push(Expr::Sym(self.gen_sym.clone()));
                } else {
                    terms.push(Expr::Mul(vec![coeff_expr, Expr::Sym(self.gen_sym.clone())]));
                }
            } else if coeff.is_one() {
                terms.push(Expr::Pow(
                    std::sync::Arc::new(Expr::Sym(self.gen_sym.clone())),
                    std::sync::Arc::new(Expr::from_i64(deg as i64)),
                ));
            } else {
                terms.push(Expr::Mul(vec![
                    coeff_expr,
                    Expr::Pow(
                        std::sync::Arc::new(Expr::Sym(self.gen_sym.clone())),
                        std::sync::Arc::new(Expr::from_i64(deg as i64)),
                    ),
                ]));
            }
        }
        if terms.len() == 1 {
            terms.pop().unwrap()
        } else {
            Expr::Add(terms)
        }
    }
}

impl fmt::Display for UnivariatePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poly({}, {})", self.to_expr(), self.gen_sym)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_poly_arithmetic() {
        let x = Symbol::new("x");
        let p1 = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(1)),
                BigRational::from_integer(BigInt::from(2)),
            ],
        ); // 1 + 2x
        let p2 = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(3)),
                BigRational::from_integer(BigInt::from(4)),
            ],
        ); // 3 + 4x
        let p_sum = p1.add(&p2).unwrap();
        assert_eq!(
            p_sum.coeffs,
            vec![
                BigRational::from_integer(BigInt::from(4)),
                BigRational::from_integer(BigInt::from(6))
            ]
        );
    }

    #[test]
    fn test_poly_div_and_gcd() {
        let x = Symbol::new("x");
        // (x - 1)(x + 2) = x^2 + x - 2 -> coeffs: [-2, 1, 1]
        let p1 = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(-2)),
                BigRational::from_integer(BigInt::from(1)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );
        // (x - 1)(x + 3) = x^2 + 2x - 3 -> coeffs: [-3, 2, 1]
        let p2 = UnivariatePoly::new(
            x.clone(),
            vec![
                BigRational::from_integer(BigInt::from(-3)),
                BigRational::from_integer(BigInt::from(2)),
                BigRational::from_integer(BigInt::from(1)),
            ],
        );
        let gcd = p1.gcd(&p2).unwrap();
        // gcd should be (x - 1) -> coeffs: [-1, 1]
        assert_eq!(
            gcd.coeffs,
            vec![
                BigRational::from_integer(BigInt::from(-1)),
                BigRational::from_integer(BigInt::from(1))
            ]
        );
    }
}
