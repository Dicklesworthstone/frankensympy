//! Sparse multivariate polynomial arithmetic over $\mathbb{Q}[x_1, \ldots, x_n]$ (WS08).

#![forbid(unsafe_code)]

use crate::PolyError;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// Term ordering policy for multivariate monomials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TermOrder {
    /// Lexicographic ordering ($x_1 > x_2 > \ldots > x_n$).
    Lex,
    /// Graded Lexicographic ordering (total degree first, then Lex).
    DegLex,
    /// Graded Reverse Lexicographic ordering (total degree first, then reverse Lex).
    DegRevLex,
}

/// Sparse multivariate polynomial represented as a sorted map from exponent vectors to coefficients:
/// $\sum_{\alpha} c_\alpha \mathbf{x}^\alpha$.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MultivariatePoly {
    pub generators: Vec<Symbol>,
    /// Exponent vector (in generator order) mapped to non-zero coefficient.
    pub terms: BTreeMap<Vec<u32>, BigRational>,
}

impl MultivariatePoly {
    /// Creates a multivariate polynomial with canonicalized terms (dropping zeros).
    pub fn new(generators: Vec<Symbol>, raw_terms: BTreeMap<Vec<u32>, BigRational>) -> Self {
        let n_vars = generators.len();
        let mut terms = BTreeMap::new();
        for (exp, coeff) in raw_terms {
            if coeff.is_zero() {
                continue;
            }
            let mut normalized_exp = exp;
            normalized_exp.resize(n_vars, 0);
            terms.insert(normalized_exp, coeff);
        }
        Self { generators, terms }
    }

    /// Construct constant polynomial 0.
    pub fn zero(generators: Vec<Symbol>) -> Self {
        Self {
            generators,
            terms: BTreeMap::new(),
        }
    }

    /// Construct constant polynomial 1.
    pub fn one(generators: Vec<Symbol>) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(vec![0; generators.len()], BigRational::one());
        Self { generators, terms }
    }

    /// Construct generator variable polynomial $x_k$.
    pub fn var(generators: Vec<Symbol>, sym: &Symbol) -> Result<Self, PolyError> {
        let idx = generators.iter().position(|s| s == sym).ok_or_else(|| {
            PolyError::IncompatibleGenerators(sym.name.clone(), "missing from generators".into())
        })?;
        let mut exp = vec![0; generators.len()];
        exp[idx] = 1;
        let mut terms = BTreeMap::new();
        terms.insert(exp, BigRational::one());
        Ok(Self { generators, terms })
    }

    /// Check if polynomial is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Check if polynomial is the constant one polynomial.
    pub fn is_one(&self) -> bool {
        if self.terms.len() != 1 {
            return false;
        }
        let zero_exp = vec![0; self.generators.len()];
        self.terms.get(&zero_exp).is_some_and(|c| c.is_one())
    }

    /// Total degree of polynomial.
    pub fn total_degree(&self) -> Option<u32> {
        if self.is_zero() {
            None
        } else {
            self.terms.keys().map(|exp| exp.iter().sum::<u32>()).max()
        }
    }

    /// Degree in a specific generator variable index.
    pub fn degree_in(&self, var_idx: usize) -> u32 {
        self.terms
            .keys()
            .map(|exp| *exp.get(var_idx).unwrap_or(&0))
            .max()
            .unwrap_or(0)
    }

    /// Evaluates the polynomial given values for all generators.
    pub fn eval(&self, values: &[BigRational]) -> Result<BigRational, PolyError> {
        if values.len() != self.generators.len() {
            return Err(PolyError::General(format!(
                "Evaluation requires {} values, got {}",
                self.generators.len(),
                values.len()
            )));
        }
        let mut result = BigRational::zero();
        for (exp, coeff) in &self.terms {
            let mut monomial_val = coeff.clone();
            for (v_idx, &deg) in exp.iter().enumerate() {
                if deg > 0 {
                    let val_pow = values[v_idx].pow(deg as i32);
                    monomial_val *= val_pow;
                }
            }
            result += monomial_val;
        }
        Ok(result)
    }

    /// Addition of multivariate polynomials with matching generators.
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        self.check_same_generators(other)?;
        let mut new_terms = self.terms.clone();
        for (exp, coeff) in &other.terms {
            let entry = new_terms
                .entry(exp.clone())
                .or_insert_with(BigRational::zero);
            *entry += coeff;
            if entry.is_zero() {
                new_terms.remove(exp);
            }
        }
        Ok(Self {
            generators: self.generators.clone(),
            terms: new_terms,
        })
    }

    /// Subtraction of multivariate polynomials with matching generators.
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        self.check_same_generators(other)?;
        let mut new_terms = self.terms.clone();
        for (exp, coeff) in &other.terms {
            let entry = new_terms
                .entry(exp.clone())
                .or_insert_with(BigRational::zero);
            *entry -= coeff;
            if entry.is_zero() {
                new_terms.remove(exp);
            }
        }
        Ok(Self {
            generators: self.generators.clone(),
            terms: new_terms,
        })
    }

    /// Multiplication of multivariate polynomials.
    pub fn mul(&self, other: &Self) -> Result<Self, PolyError> {
        self.check_same_generators(other)?;
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero(self.generators.clone()));
        }
        let n_vars = self.generators.len();
        let mut new_terms = BTreeMap::new();
        for (exp_a, c_a) in &self.terms {
            for (exp_b, c_b) in &other.terms {
                let mut exp_res = vec![0u32; n_vars];
                for i in 0..n_vars {
                    exp_res[i] = exp_a[i] + exp_b[i];
                }
                let prod_c = c_a * c_b;
                let entry = new_terms
                    .entry(exp_res.clone())
                    .or_insert_with(BigRational::zero);
                *entry += prod_c;
                if entry.is_zero() {
                    new_terms.remove(&exp_res);
                }
            }
        }
        Ok(Self {
            generators: self.generators.clone(),
            terms: new_terms,
        })
    }

    /// Metered multiplication with cancellation checkpoints and work accounting.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Self, PolyError> {
        meter
            .checkpoint()
            .map_err(|e| PolyError::General(e.to_string()))?;
        let work = (self.terms.len() as u64).saturating_mul(other.terms.len() as u64);
        meter
            .charge(Dimension::ComputeSteps, work)
            .map_err(|e| PolyError::General(e.to_string()))?;
        self.mul(other)
    }

    /// Exponentiation $P^k$.
    pub fn pow(&self, mut exp: u32) -> Result<Self, PolyError> {
        if exp == 0 {
            return Ok(Self::one(self.generators.clone()));
        }
        let mut base = self.clone();
        let mut res = Self::one(self.generators.clone());
        while exp > 0 {
            if exp % 2 == 1 {
                res = res.mul(&base)?;
            }
            if exp > 1 {
                base = base.mul(&base)?;
            }
            exp /= 2;
        }
        Ok(res)
    }

    /// Partial derivative with respect to variable at `var_idx`.
    pub fn derivative(&self, var_idx: usize) -> Result<Self, PolyError> {
        if var_idx >= self.generators.len() {
            return Err(PolyError::General(format!(
                "Invalid variable index {} for generators count {}",
                var_idx,
                self.generators.len()
            )));
        }
        let mut new_terms = BTreeMap::new();
        for (exp, coeff) in &self.terms {
            let deg = exp[var_idx];
            if deg == 0 {
                continue;
            }
            let mut new_exp = exp.clone();
            new_exp[var_idx] -= 1;
            let mult = BigRational::from_integer(BigInt::from(deg as i64));
            new_terms.insert(new_exp, coeff * &mult);
        }
        Ok(Self {
            generators: self.generators.clone(),
            terms: new_terms,
        })
    }

    /// Convert `Expr` into a multivariate polynomial in the given generators.
    pub fn from_expr(expr: &Expr, generators: &[Symbol]) -> Result<Self, PolyError> {
        let n_vars = generators.len();
        match expr {
            Expr::Integer(n) => {
                let mut terms = BTreeMap::new();
                if !n.is_zero() {
                    terms.insert(vec![0; n_vars], BigRational::from_integer(n.clone()));
                }
                Ok(Self {
                    generators: generators.to_vec(),
                    terms,
                })
            }
            Expr::Rational(r) => {
                let mut terms = BTreeMap::new();
                if !r.is_zero() {
                    terms.insert(vec![0; n_vars], r.clone());
                }
                Ok(Self {
                    generators: generators.to_vec(),
                    terms,
                })
            }
            Expr::Sym(s) => {
                let idx = generators.iter().position(|g| g == s).ok_or_else(|| {
                    PolyError::IncompatibleGenerators(
                        s.name.clone(),
                        "missing from generators".into(),
                    )
                })?;
                let mut exp = vec![0; n_vars];
                exp[idx] = 1;
                let mut terms = BTreeMap::new();
                terms.insert(exp, BigRational::one());
                Ok(Self {
                    generators: generators.to_vec(),
                    terms,
                })
            }
            Expr::Add(terms_expr) => {
                let mut sum = Self::zero(generators.to_vec());
                for t in terms_expr {
                    let pt = Self::from_expr(t, generators)?;
                    sum = sum.add(&pt)?;
                }
                Ok(sum)
            }
            Expr::Mul(factors_expr) => {
                let mut prod = Self::one(generators.to_vec());
                for f in factors_expr {
                    let pf = Self::from_expr(f, generators)?;
                    prod = prod.mul(&pf)?;
                }
                Ok(prod)
            }
            Expr::Pow(base, exp) => {
                let p_base = Self::from_expr(base, generators)?;
                if let Expr::Integer(n) = exp.as_ref()
                    && let Ok(k) = usize::try_from(n)
                {
                    return p_base.pow(k as u32);
                }
                Err(PolyError::NonPolynomialExpression(format!(
                    "Non-integer exponent in multivariate polynomial: {expr}"
                )))
            }
            _ => Err(PolyError::NonPolynomialExpression(format!(
                "Unsupported expression form for multivariate polynomial: {expr}"
            ))),
        }
    }

    /// Converts back to an exact `Expr`.
    pub fn to_expr(&self) -> Expr {
        if self.is_zero() {
            return Expr::from_i64(0);
        }
        let mut terms_expr = Vec::new();
        for (exp, coeff) in &self.terms {
            let mut factors = Vec::new();
            if !coeff.is_one() || exp.iter().all(|&d| d == 0) {
                let c_expr = if coeff.is_integer() {
                    Expr::Integer(coeff.to_integer())
                } else {
                    Expr::Rational(coeff.clone())
                };
                factors.push(c_expr);
            }
            for (idx, &deg) in exp.iter().enumerate() {
                if deg == 1 {
                    factors.push(Expr::Sym(self.generators[idx].clone()));
                } else if deg > 1 {
                    factors.push(Expr::Pow(
                        Arc::new(Expr::Sym(self.generators[idx].clone())),
                        Arc::new(Expr::from_i64(deg as i64)),
                    ));
                }
            }
            let term = match factors.len() {
                0 => Expr::from_i64(1),
                1 => factors.pop().unwrap(),
                _ => Expr::Mul(factors),
            };
            terms_expr.push(term);
        }

        match terms_expr.len() {
            0 => Expr::from_i64(0),
            1 => terms_expr.pop().unwrap(),
            _ => Expr::Add(terms_expr),
        }
    }

    fn check_same_generators(&self, other: &Self) -> Result<(), PolyError> {
        if self.generators != other.generators {
            Err(PolyError::IncompatibleGenerators(
                format!("{:?}", self.generators),
                format!("{:?}", other.generators),
            ))
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for MultivariatePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Poly({}, {:?})",
            self.to_expr(),
            self.generators.iter().map(|s| &s.name).collect::<Vec<_>>()
        )
    }
}
