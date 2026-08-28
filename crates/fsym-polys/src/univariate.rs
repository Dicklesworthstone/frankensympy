//! Dense univariate polynomial arithmetic over $\mathbb{Q}$ and $\mathbb{Z}$ (WS08).

#![forbid(unsafe_code)]

use crate::PolyError;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use serde::de::{IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::sync::Arc;

const MAX_UNIVARIATE_COEFFICIENTS: usize = 65_536;
const MAX_UNIVARIATE_POWER: u32 = 65_535;

/// Univariate polynomial represented by dense coefficient vector:
/// `c_0 + c_1 * x + ... + c_n * x^n`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnivariatePoly {
    pub gen_sym: Symbol,
    /// Coefficients ordered by increasing degree: `coeffs[k]` is coefficient of `gen_sym^k`.
    pub coeffs: Vec<BigRational>,
}

#[derive(Serialize)]
struct UnivariatePolyWireRef<'a> {
    gen_sym: &'a Symbol,
    coeffs: &'a [BigRational],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnivariatePolyWire {
    gen_sym: Symbol,
    #[serde(deserialize_with = "deserialize_bounded_coefficients")]
    coeffs: Vec<BigRational>,
}

fn deserialize_bounded_coefficients<'de, D>(deserializer: D) -> Result<Vec<BigRational>, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundedCoefficientVisitor;

    impl<'de> Visitor<'de> for BoundedCoefficientVisitor {
        type Value = Vec<BigRational>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "at most {MAX_UNIVARIATE_COEFFICIENTS} rational coefficients"
            )
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let size_hint = seq.size_hint().unwrap_or(0);
            if size_hint > MAX_UNIVARIATE_COEFFICIENTS {
                return Err(serde::de::Error::invalid_length(size_hint, &self));
            }

            let mut coeffs = Vec::with_capacity(size_hint);
            while coeffs.len() < MAX_UNIVARIATE_COEFFICIENTS {
                match seq.next_element()? {
                    Some(coefficient) => coeffs.push(coefficient),
                    None => return Ok(coeffs),
                }
            }

            if seq.next_element::<IgnoredAny>()?.is_some() {
                return Err(serde::de::Error::invalid_length(
                    MAX_UNIVARIATE_COEFFICIENTS + 1,
                    &self,
                ));
            }
            Ok(coeffs)
        }
    }

    deserializer.deserialize_seq(BoundedCoefficientVisitor)
}

impl Serialize for UnivariatePoly {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.validate_shape().map_err(serde::ser::Error::custom)?;
        UnivariatePolyWireRef {
            gen_sym: &self.gen_sym,
            coeffs: &self.coeffs,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UnivariatePoly {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = UnivariatePolyWire::deserialize(deserializer)?;
        let poly = Self {
            gen_sym: wire.gen_sym,
            coeffs: wire.coeffs,
        };
        poly.validate_shape().map_err(serde::de::Error::custom)?;
        Ok(poly)
    }
}

impl UnivariatePoly {
    /// Constructs a canonical univariate polynomial, trimming trailing zeros.
    pub fn new(gen_sym: Symbol, mut coeffs: Vec<BigRational>) -> Self {
        while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            coeffs.push(BigRational::zero());
        }
        Self { gen_sym, coeffs }
    }

    /// Validates the canonical dense representation at a trust boundary.
    pub fn validate_shape(&self) -> Result<(), PolyError> {
        if self.coeffs.is_empty() {
            return Err(PolyError::General(
                "univariate polynomial coefficient vector is empty".to_string(),
            ));
        }
        if self.coeffs.len() > MAX_UNIVARIATE_COEFFICIENTS {
            return Err(PolyError::General(format!(
                "univariate polynomial exceeds the coefficient limit of {MAX_UNIVARIATE_COEFFICIENTS}"
            )));
        }
        if self.coeffs.len() > 1 && self.coeffs.last().is_some_and(Zero::is_zero) {
            return Err(PolyError::General(
                "univariate polynomial has a noncanonical trailing zero coefficient".to_string(),
            ));
        }
        Ok(())
    }

    /// Construct constant polynomial 0.
    pub fn zero(gen_sym: Symbol) -> Self {
        Self {
            gen_sym,
            coeffs: vec![BigRational::zero()],
        }
    }

    /// Construct constant polynomial 1.
    pub fn one(gen_sym: Symbol) -> Self {
        Self {
            gen_sym,
            coeffs: vec![BigRational::one()],
        }
    }

    /// Construct monomial $c \cdot x^k$ within the dense representation limit.
    pub fn monomial(gen_sym: Symbol, coeff: BigRational, degree: usize) -> Result<Self, PolyError> {
        if coeff.is_zero() {
            return Ok(Self::zero(gen_sym));
        }
        let coefficient_count = degree.checked_add(1).ok_or_else(|| {
            PolyError::General("univariate monomial degree overflowed".to_string())
        })?;
        if coefficient_count > MAX_UNIVARIATE_COEFFICIENTS {
            return Err(PolyError::General(format!(
                "univariate monomial exceeds the coefficient limit of {MAX_UNIVARIATE_COEFFICIENTS}"
            )));
        }
        let mut coeffs = vec![BigRational::zero(); coefficient_count];
        coeffs[degree] = coeff;
        Ok(Self { gen_sym, coeffs })
    }

    /// Degree of polynomial ($\text{deg}(0) = \text{None}$).
    pub fn degree(&self) -> Option<usize> {
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Check if this polynomial is the zero polynomial.
    pub fn is_zero(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_zero()
    }

    /// Check if this polynomial is the constant one polynomial.
    pub fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }

    /// Leading coefficient.
    pub fn leading_coeff(&self) -> &BigRational {
        self.coeffs.last().expect("coeffs is non-empty")
    }

    /// Check if this polynomial is monic ($\text{LC}(P) = 1$ and not zero polynomial).
    pub fn is_monic(&self) -> bool {
        self.coeffs.last().is_some_and(One::is_one)
    }

    /// Evaluate polynomial at given point $x = v$.
    pub fn eval(&self, point: &BigRational) -> BigRational {
        // Horner's method: c_0 + x * (c_1 + x * (...))
        let mut acc = BigRational::zero();
        for c in self.coeffs.iter().rev() {
            acc = acc * point + c;
        }
        acc
    }

    /// Differentiate with respect to generator symbol.
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero(self.gen_sym.clone());
        }
        let mut deriv_coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        for (i, c) in self.coeffs.iter().enumerate().skip(1) {
            let mult = BigRational::from_integer(BigInt::from(i as i64));
            deriv_coeffs.push(c * &mult);
        }
        Self::new(self.gen_sym.clone(), deriv_coeffs)
    }

    /// Indefinite integration with respect to the generator symbol, with integration constant $C$.
    pub fn integrate(&self, constant: BigRational) -> Result<Self, PolyError> {
        self.validate_shape()?;
        let next_len = self
            .coeffs
            .len()
            .checked_add(1)
            .ok_or_else(|| PolyError::General("univariate integration degree overflow".to_string()))?;
        if next_len > MAX_UNIVARIATE_COEFFICIENTS {
            return Err(PolyError::General(format!(
                "univariate integration exceeds coefficient limit of {MAX_UNIVARIATE_COEFFICIENTS}"
            )));
        }
        let mut int_coeffs = Vec::with_capacity(next_len);
        int_coeffs.push(constant);
        for (i, c) in self.coeffs.iter().enumerate() {
            let divisor = BigRational::from_integer(BigInt::from((i + 1) as i64));
            int_coeffs.push(c / &divisor);
        }
        Ok(Self::new(self.gen_sym.clone(), int_coeffs))
    }

    /// Computes the polynomial discriminant.
    ///
    /// - Degree 2 ($a x^2 + b x + c$): $\Delta = b^2 - 4 a c$
    /// - Degree 1 ($a x + b$): $\Delta = 1$
    pub fn discriminant(&self) -> Result<BigRational, PolyError> {
        self.validate_shape()?;
        match self.degree() {
            Some(2) => {
                let c = &self.coeffs[0];
                let b = &self.coeffs[1];
                let a = &self.coeffs[2];
                let four = BigRational::from_integer(BigInt::from(4));
                Ok((b * b) - (four * a * c))
            }
            Some(1) => Ok(BigRational::one()),
            _ => Err(PolyError::General(
                "discriminant currently supported for degree 1 and 2 univariate polynomials".to_string(),
            )),
        }
    }

    /// Addition of polynomials in the same generator.
    pub fn add(&self, other: &Self) -> Result<Self, PolyError> {
        self.validate_shape()?;
        other.validate_shape()?;
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

    /// Subtraction of polynomials in the same generator.
    pub fn sub(&self, other: &Self) -> Result<Self, PolyError> {
        self.validate_shape()?;
        other.validate_shape()?;
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
            new_coeffs.push(a - b);
        }
        Ok(Self::new(self.gen_sym.clone(), new_coeffs))
    }

    /// Multiplication of polynomials in the same generator.
    pub fn mul(&self, other: &Self) -> Result<Self, PolyError> {
        self.validate_shape()?;
        other.validate_shape()?;
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
        let result_len = deg_a
            .checked_add(deg_b)
            .and_then(|degree| degree.checked_add(1))
            .ok_or_else(|| PolyError::General("univariate result degree overflowed".to_string()))?;
        if result_len > MAX_UNIVARIATE_COEFFICIENTS {
            return Err(PolyError::General(format!(
                "univariate result exceeds the coefficient limit of {MAX_UNIVARIATE_COEFFICIENTS}"
            )));
        }
        let mut res = vec![BigRational::zero(); result_len];
        for (i, c_a) in self.coeffs.iter().enumerate() {
            if c_a.is_zero() {
                continue;
            }
            for (j, c_b) in other.coeffs.iter().enumerate() {
                if c_b.is_zero() {
                    continue;
                }
                res[i + j] = &res[i + j] + (c_a * c_b);
            }
        }
        Ok(Self::new(self.gen_sym.clone(), res))
    }

    /// Metered multiplication with safe-point cancellation and step charging.
    pub fn metered_mul<M: BudgetMeter>(
        &self,
        other: &Self,
        meter: &mut M,
    ) -> Result<Self, PolyError> {
        self.validate_shape()?;
        other.validate_shape()?;
        meter
            .checkpoint()
            .map_err(|e| PolyError::General(e.to_string()))?;
        let steps = (self.coeffs.len() as u64).saturating_mul(other.coeffs.len() as u64);
        meter
            .charge(Dimension::ComputeSteps, steps)
            .map_err(|e| PolyError::General(e.to_string()))?;
        self.mul(other)
    }

    /// Polynomial power $P^k$.
    pub fn pow(&self, mut exp: u32) -> Result<Self, PolyError> {
        self.validate_shape()?;
        if exp > MAX_UNIVARIATE_POWER {
            return Err(PolyError::General(format!(
                "univariate exponent {exp} exceeds the limit of {MAX_UNIVARIATE_POWER}"
            )));
        }
        if let Some(degree) = self.degree() {
            let result_degree = degree
                .checked_mul(usize::try_from(exp).map_err(|_| {
                    PolyError::General("univariate exponent conversion failed".to_string())
                })?)
                .ok_or_else(|| {
                    PolyError::General("univariate result degree overflowed".to_string())
                })?;
            if result_degree >= MAX_UNIVARIATE_COEFFICIENTS {
                return Err(PolyError::General(format!(
                    "univariate result exceeds the coefficient limit of {MAX_UNIVARIATE_COEFFICIENTS}"
                )));
            }
        }
        if exp == 0 {
            return Ok(Self::one(self.gen_sym.clone()));
        }
        let mut base = self.clone();
        let mut res = Self::one(self.gen_sym.clone());
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

    /// Polynomial division with remainder: `self = quotient * divisor + remainder`.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolyError> {
        self.validate_shape()?;
        divisor.validate_shape()?;
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

        let mut remainder_coeffs = self.coeffs.clone();
        let mut quotient_coeffs = vec![BigRational::zero(); deg_self - deg_div + 1];
        let lc_div = divisor.leading_coeff();

        for i in (0..=(deg_self - deg_div)).rev() {
            let cur_deg = i + deg_div;
            if cur_deg >= remainder_coeffs.len() {
                continue;
            }
            let cur_lead = remainder_coeffs[cur_deg].clone();
            if cur_lead.is_zero() {
                continue;
            }
            let q_coeff = &cur_lead / lc_div;
            quotient_coeffs[i] = q_coeff.clone();

            for (j, d_coeff) in divisor.coeffs.iter().enumerate() {
                remainder_coeffs[i + j] = &remainder_coeffs[i + j] - (&q_coeff * d_coeff);
            }
        }

        Ok((
            Self::new(self.gen_sym.clone(), quotient_coeffs),
            Self::new(self.gen_sym.clone(), remainder_coeffs),
        ))
    }

    /// Converts an algebraic expression to a univariate polynomial.
    pub fn from_expr(expr: &Expr, gen_sym: &Symbol) -> Result<Self, PolyError> {
        match expr {
            Expr::Integer(n) => Ok(Self::new(
                gen_sym.clone(),
                vec![BigRational::from_integer(n.clone())],
            )),
            Expr::Rational(r) => Ok(Self::new(gen_sym.clone(), vec![r.clone()])),
            Expr::Sym(s) => {
                if s == gen_sym {
                    Ok(Self::new(
                        gen_sym.clone(),
                        vec![BigRational::zero(), BigRational::one()],
                    ))
                } else {
                    Err(PolyError::NonPolynomialExpression(format!(
                        "Encountered variable {} other than generator {}",
                        s.name, gen_sym.name
                    )))
                }
            }
            Expr::Add(terms) => {
                let mut sum = Self::zero(gen_sym.clone());
                for t in terms {
                    let pt = Self::from_expr(t, gen_sym)?;
                    sum = sum.add(&pt)?;
                }
                Ok(sum)
            }
            Expr::Mul(factors) => {
                let mut prod = Self::one(gen_sym.clone());
                for f in factors {
                    let pf = Self::from_expr(f, gen_sym)?;
                    prod = prod.mul(&pf)?;
                }
                Ok(prod)
            }
            Expr::Pow(base, exp) => {
                let p_base = Self::from_expr(base, gen_sym)?;
                if let Expr::Integer(n) = exp.as_ref() {
                    if let Ok(k) = usize::try_from(n)
                        && let Ok(k_u32) = u32::try_from(k)
                    {
                        return p_base.pow(k_u32);
                    } else if p_base.degree() == Some(0)
                        && !p_base.is_zero()
                        && let Ok(k) = usize::try_from(&(-n))
                        && let Ok(k_u32) = u32::try_from(k)
                    {
                        let inv_c = BigRational::one() / p_base.leading_coeff();
                        let inv_poly = Self::new(gen_sym.clone(), vec![inv_c]);
                        return inv_poly.pow(k_u32);
                    }
                }
                Err(PolyError::NonPolynomialExpression(format!(
                    "Non-integer power expression in polynomial: {expr}"
                )))
            }
            _ => Err(PolyError::NonPolynomialExpression(format!(
                "Unsupported expression form for polynomial: {expr}"
            ))),
        }
    }

    /// Converts univariate polynomial back into an exact `Expr`.
    pub fn to_expr(&self) -> Expr {
        if self.is_zero() {
            return Expr::from_i64(0);
        }
        let mut terms = Vec::new();
        for (deg, coeff) in self.coeffs.iter().enumerate() {
            if coeff.is_zero() {
                continue;
            }
            let c_expr = if coeff.is_integer() {
                Expr::Integer(coeff.to_integer())
            } else {
                Expr::Rational(coeff.clone())
            };

            let term = match deg {
                0 => c_expr,
                1 => {
                    if coeff.is_one() {
                        Expr::Sym(self.gen_sym.clone())
                    } else {
                        Expr::Mul(vec![c_expr, Expr::Sym(self.gen_sym.clone())])
                    }
                }
                d => {
                    let pow_expr = Expr::Pow(
                        Arc::new(Expr::Sym(self.gen_sym.clone())),
                        Arc::new(Expr::from_i64(d as i64)),
                    );
                    if coeff.is_one() {
                        pow_expr
                    } else {
                        Expr::Mul(vec![c_expr, pow_expr])
                    }
                }
            };
            terms.push(term);
        }

        match terms.len() {
            0 => Expr::from_i64(0),
            1 => terms.pop().unwrap(),
            _ => Expr::Add(terms),
        }
    }
}

impl fmt::Display for UnivariatePoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poly({}, {})", self.to_expr(), self.gen_sym.name)
    }
}
