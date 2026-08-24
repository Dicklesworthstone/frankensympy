//! Sparse multivariate polynomial arithmetic over $\mathbb{Q}[x_1, \ldots, x_n]$ (WS08).

#![forbid(unsafe_code)]

use crate::PolyError;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

const MAX_MULTIVARIATE_TERMS: usize = 4_096;
const MAX_MULTIVARIATE_TERM_PRODUCTS: u64 = 1_000_000;
const MAX_MULTIVARIATE_GENERATORS: usize = 256;
const MAX_MULTIVARIATE_GENERATOR_NAME_BYTES: usize = 65_536;
const MAX_MULTIVARIATE_EXPR_DEPTH: usize = 256;
const MAX_MULTIVARIATE_EXPR_NODES: usize = 262_144;

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultivariatePoly {
    pub generators: Vec<Symbol>,
    /// Exponent vector (in generator order) mapped to non-zero coefficient.
    pub terms: BTreeMap<Vec<u32>, BigRational>,
}

#[derive(Serialize)]
struct MultivariatePolyWireRef<'a> {
    schema_version: u32,
    generators: &'a [Symbol],
    terms: Vec<MultivariateTermWireRef<'a>>,
}

#[derive(Serialize)]
struct MultivariateTermWireRef<'a> {
    exponents: &'a [u32],
    coefficient: &'a BigRational,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MultivariatePolyWire {
    schema_version: u32,
    generators: Vec<Symbol>,
    terms: Vec<MultivariateTermWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MultivariateTermWire {
    exponents: Vec<u32>,
    coefficient: BigRational,
}

impl Serialize for MultivariatePoly {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate_shape().map_err(serde::ser::Error::custom)?;
        let terms = self
            .terms
            .iter()
            .map(|(exponents, coefficient)| MultivariateTermWireRef {
                exponents,
                coefficient,
            })
            .collect();
        MultivariatePolyWireRef {
            schema_version: 1,
            generators: &self.generators,
            terms,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MultivariatePoly {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MultivariatePolyWire::deserialize(deserializer)?;
        if wire.schema_version != 1 {
            return Err(serde::de::Error::custom(format!(
                "unsupported multivariate polynomial schema version {}",
                wire.schema_version
            )));
        }
        validate_generators(&wire.generators).map_err(serde::de::Error::custom)?;
        if wire.terms.len() > MAX_MULTIVARIATE_TERMS {
            return Err(serde::de::Error::custom(format!(
                "multivariate polynomial exceeds the term limit of {MAX_MULTIVARIATE_TERMS}"
            )));
        }

        let mut terms = BTreeMap::new();
        for term in wire.terms {
            if term.exponents.len() != wire.generators.len() {
                return Err(serde::de::Error::custom(
                    "multivariate exponent-vector width does not match the generator count",
                ));
            }
            if term.coefficient.is_zero() {
                return Err(serde::de::Error::custom(
                    "multivariate canonical wire cannot contain a zero coefficient",
                ));
            }
            if terms.insert(term.exponents, term.coefficient).is_some() {
                return Err(serde::de::Error::custom(
                    "multivariate canonical wire contains duplicate exponent vectors",
                ));
            }
        }
        Ok(Self {
            generators: wire.generators,
            terms,
        })
    }
}

impl MultivariatePoly {
    /// Ordered polynomial-ring generators.
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }

    /// Canonical non-zero terms keyed by exponent vector.
    pub fn terms(&self) -> &BTreeMap<Vec<u32>, BigRational> {
        &self.terms
    }

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
            let entry = terms
                .entry(normalized_exp.clone())
                .or_insert_with(BigRational::zero);
            *entry += coeff;
            if entry.is_zero() {
                terms.remove(&normalized_exp);
            }
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
        validate_generators(&generators).map_err(PolyError::General)?;
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
            let mut maximum = 0u32;
            for exponents in self.terms.keys() {
                let degree = exponents
                    .iter()
                    .try_fold(0u32, |sum, exponent| sum.checked_add(*exponent))?;
                maximum = maximum.max(degree);
            }
            Some(maximum)
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
        self.validate_shape()?;
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
                    let signed_degree = i32::try_from(deg).map_err(|_| {
                        PolyError::General(
                            "polynomial evaluation exponent exceeds the supported i32 range"
                                .to_string(),
                        )
                    })?;
                    let val_pow = values[v_idx].pow(signed_degree);
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
            if new_terms.len() > MAX_MULTIVARIATE_TERMS {
                return Err(PolyError::General(format!(
                    "multivariate result exceeds the term limit of {MAX_MULTIVARIATE_TERMS}"
                )));
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
            if new_terms.len() > MAX_MULTIVARIATE_TERMS {
                return Err(PolyError::General(format!(
                    "multivariate result exceeds the term limit of {MAX_MULTIVARIATE_TERMS}"
                )));
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
        self.checked_term_products(other)?;
        let n_vars = self.generators.len();
        let mut new_terms = BTreeMap::new();
        for (exp_a, c_a) in &self.terms {
            for (exp_b, c_b) in &other.terms {
                let mut exp_res = vec![0u32; n_vars];
                for i in 0..n_vars {
                    exp_res[i] = exp_a[i].checked_add(exp_b[i]).ok_or_else(|| {
                        PolyError::General(
                            "multivariate exponent exceeds the u32 representation".to_string(),
                        )
                    })?;
                }
                let prod_c = c_a * c_b;
                let entry = new_terms
                    .entry(exp_res.clone())
                    .or_insert_with(BigRational::zero);
                *entry += prod_c;
                if entry.is_zero() {
                    new_terms.remove(&exp_res);
                }
                if new_terms.len() > MAX_MULTIVARIATE_TERMS {
                    return Err(PolyError::General(format!(
                        "multivariate result exceeds the term limit of {MAX_MULTIVARIATE_TERMS}"
                    )));
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
        self.check_same_generators(other)?;
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero(self.generators.clone()));
        }
        let work = self.checked_term_products(other)?;
        meter
            .charge(Dimension::ComputeSteps, work)
            .map_err(|e| PolyError::General(e.to_string()))?;
        self.mul(other)
    }

    /// Exponentiation $P^k$.
    pub fn pow(&self, mut exp: u32) -> Result<Self, PolyError> {
        self.validate_shape()?;
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
        self.validate_shape()?;
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
        validate_generators(generators).map_err(PolyError::General)?;
        let mut visited_nodes = 0usize;
        Self::from_expr_at(expr, generators, 0, &mut visited_nodes)
    }

    fn from_expr_at(
        expr: &Expr,
        generators: &[Symbol],
        depth: usize,
        visited_nodes: &mut usize,
    ) -> Result<Self, PolyError> {
        if depth > MAX_MULTIVARIATE_EXPR_DEPTH {
            return Err(PolyError::NonPolynomialExpression(format!(
                "expression exceeds the conversion depth limit of {MAX_MULTIVARIATE_EXPR_DEPTH}"
            )));
        }
        *visited_nodes = visited_nodes.checked_add(1).ok_or_else(|| {
            PolyError::NonPolynomialExpression(
                "expression node counter overflowed during conversion".to_string(),
            )
        })?;
        if *visited_nodes > MAX_MULTIVARIATE_EXPR_NODES {
            return Err(PolyError::NonPolynomialExpression(format!(
                "expression exceeds the conversion node limit of {MAX_MULTIVARIATE_EXPR_NODES}"
            )));
        }

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
                        "expression symbol".into(),
                        "missing from bounded generator list".into(),
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
                    let pt = Self::from_expr_at(t, generators, depth + 1, visited_nodes)?;
                    sum = sum.add(&pt)?;
                }
                Ok(sum)
            }
            Expr::Mul(factors_expr) => {
                let mut prod = Self::one(generators.to_vec());
                for f in factors_expr {
                    let pf = Self::from_expr_at(f, generators, depth + 1, visited_nodes)?;
                    prod = prod.mul(&pf)?;
                }
                Ok(prod)
            }
            Expr::Pow(base, exp) => {
                let k = match exp.as_ref() {
                    Expr::Integer(n) => n
                        .to_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or_else(|| {
                            PolyError::NonPolynomialExpression(
                                "polynomial exponent is negative or exceeds u32".to_string(),
                            )
                        })?,
                    _ => {
                        return Err(PolyError::NonPolynomialExpression(
                            "multivariate polynomial exponent is not an integer".to_string(),
                        ));
                    }
                };
                *visited_nodes = visited_nodes.checked_add(1).ok_or_else(|| {
                    PolyError::NonPolynomialExpression(
                        "expression node counter overflowed during conversion".to_string(),
                    )
                })?;
                if *visited_nodes > MAX_MULTIVARIATE_EXPR_NODES {
                    return Err(PolyError::NonPolynomialExpression(format!(
                        "expression exceeds the conversion node limit of {MAX_MULTIVARIATE_EXPR_NODES}"
                    )));
                }
                let p_base = Self::from_expr_at(base, generators, depth + 1, visited_nodes)?;
                p_base.pow(k)
            }
            _ => Err(PolyError::NonPolynomialExpression(
                "unsupported expression form for multivariate polynomial".to_string(),
            )),
        }
    }

    /// Converts back to an exact `Expr`.
    pub fn to_expr(&self) -> Result<Expr, PolyError> {
        self.validate_shape()?;
        if self.is_zero() {
            return Ok(Expr::from_i64(0));
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

        Ok(match terms_expr.len() {
            0 => Expr::from_i64(0),
            1 => terms_expr.pop().unwrap(),
            _ => Expr::Add(terms_expr),
        })
    }

    fn check_same_generators(&self, other: &Self) -> Result<(), PolyError> {
        self.validate_shape()?;
        other.validate_shape()?;
        if self.generators != other.generators {
            Err(PolyError::IncompatibleGenerators(
                format!("{:?}", self.generators),
                format!("{:?}", other.generators),
            ))
        } else {
            Ok(())
        }
    }

    fn checked_term_products(&self, other: &Self) -> Result<u64, PolyError> {
        let lhs_terms = u64::try_from(self.terms.len())
            .map_err(|_| PolyError::General("left term count exceeds u64".to_string()))?;
        let rhs_terms = u64::try_from(other.terms.len())
            .map_err(|_| PolyError::General("right term count exceeds u64".to_string()))?;
        let products = lhs_terms.checked_mul(rhs_terms).ok_or_else(|| {
            PolyError::General("multivariate term-product count overflowed".to_string())
        })?;
        if products > MAX_MULTIVARIATE_TERM_PRODUCTS {
            return Err(PolyError::General(format!(
                "multivariate multiplication exceeds the term-product limit of {MAX_MULTIVARIATE_TERM_PRODUCTS}"
            )));
        }
        Ok(products)
    }

    pub(crate) fn validate_shape(&self) -> Result<(), PolyError> {
        validate_generators(&self.generators).map_err(PolyError::General)?;
        if self.terms.len() > MAX_MULTIVARIATE_TERMS {
            return Err(PolyError::General(format!(
                "multivariate polynomial exceeds the term limit of {MAX_MULTIVARIATE_TERMS}"
            )));
        }
        if self
            .terms
            .keys()
            .any(|exponents| exponents.len() != self.generators.len())
        {
            return Err(PolyError::General(
                "multivariate exponent-vector width does not match the generator count".to_string(),
            ));
        }
        if self.terms.values().any(|coefficient| coefficient.is_zero()) {
            return Err(PolyError::General(
                "multivariate canonical form cannot contain a zero coefficient".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_generators(generators: &[Symbol]) -> Result<(), String> {
    if generators.len() > MAX_MULTIVARIATE_GENERATORS {
        return Err(format!(
            "multivariate generator count exceeds {MAX_MULTIVARIATE_GENERATORS}"
        ));
    }
    let mut names = std::collections::BTreeSet::new();
    let mut name_bytes = 0usize;
    for generator in generators {
        if !names.insert(&generator.name) {
            return Err("multivariate generator list contains a duplicate".to_string());
        }
        name_bytes = name_bytes
            .checked_add(generator.name.len())
            .ok_or_else(|| "multivariate generator-name byte count overflowed".to_string())?;
    }
    if name_bytes > MAX_MULTIVARIATE_GENERATOR_NAME_BYTES {
        return Err(format!(
            "multivariate generator names exceed {MAX_MULTIVARIATE_GENERATOR_NAME_BYTES} bytes"
        ));
    }
    Ok(())
}

impl fmt::Display for MultivariatePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_expr() {
            Ok(expr) => write!(
                f,
                "Poly({}, {:?})",
                expr,
                self.generators.iter().map(|s| &s.name).collect::<Vec<_>>()
            ),
            Err(error) => write!(f, "Poly(<invalid: {error}>)"),
        }
    }
}
