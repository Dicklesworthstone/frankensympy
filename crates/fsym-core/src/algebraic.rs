//! Exact algebraic number representation and certified root refinement (WS11).
//!
//! An algebraic number $\alpha \in \mathbb{R}$ is represented by:
//! 1. A square-free defining polynomial $P(x) \in \mathbb{Q}[x]$ such that $P(\alpha) = 0$.
//! 2. A certified root isolating ball $\mathcal{B}(m, r)$ containing exactly one real root of $P$.

#![forbid(unsafe_code)]

use crate::ball::{BallError, RealBall};
use crate::{BigInt, BigRational};
use num_traits::{One, Zero};
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use thiserror::Error;

const MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS: usize = 1_024;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AlgebraicError {
    #[error("Zero polynomial cannot define an algebraic number")]
    ZeroPolynomial,
    #[error("Defining polynomial is not square-free")]
    NonSquareFreePolynomial,
    #[error(
        "Defining polynomial has {0} coefficients, exceeding the supported limit of {MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS}"
    )]
    PolynomialCoefficientLimitExceeded(usize),
    #[error("Root isolating interval contains no root (or is not isolating): {0}")]
    InvalidIsolatingInterval(String),
    #[error("Target radius cannot be negative: {0}")]
    NegativeTargetRadius(String),
    #[error("Refinement iteration limit exceeded ({0} steps)")]
    IterationLimitExceeded(usize),
    #[error("Division by zero algebraic number")]
    DivisionByZero,
    #[error("Ball arithmetic error: {0}")]
    Ball(#[from] BallError),
}

/// Trims trailing zeros from coefficient vector.
fn poly_trim(mut coeffs: Vec<BigRational>) -> Vec<BigRational> {
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
    }
    if coeffs.is_empty() {
        coeffs.push(BigRational::zero());
    }
    coeffs
}

/// Normalizes a non-zero polynomial to be monic.
fn poly_monic(coeffs: &[BigRational]) -> Vec<BigRational> {
    let trimmed = poly_trim(coeffs.to_vec());
    if poly_is_zero(&trimmed) {
        return trimmed;
    }
    let lc = trimmed.last().unwrap().clone();
    trimmed.into_iter().map(|c| c / &lc).collect()
}

/// Checks if polynomial is identically zero.
fn poly_is_zero(coeffs: &[BigRational]) -> bool {
    coeffs.is_empty() || (coeffs.len() == 1 && coeffs[0].is_zero())
}

/// Degree of polynomial.
fn poly_degree(coeffs: &[BigRational]) -> usize {
    if coeffs.is_empty() {
        0
    } else {
        coeffs.len() - 1
    }
}

/// Formal polynomial derivative $\frac{d}{dx} P(x)$.
fn poly_derivative(coeffs: &[BigRational]) -> Vec<BigRational> {
    if coeffs.len() <= 1 {
        return vec![BigRational::zero()];
    }
    let mut deriv = Vec::with_capacity(coeffs.len() - 1);
    for (deg, coeff) in coeffs.iter().enumerate().skip(1) {
        let k = BigRational::from_integer(BigInt::from(deg as u64));
        deriv.push(coeff * &k);
    }
    poly_trim(deriv)
}

/// Negates polynomial coefficients.
fn poly_negate(coeffs: &[BigRational]) -> Vec<BigRational> {
    coeffs.iter().map(|c| -c).collect()
}

/// Polynomial division $(Q, R)$ with $A = Q \cdot B + R$.
fn poly_div_rem(a: &[BigRational], b: &[BigRational]) -> (Vec<BigRational>, Vec<BigRational>) {
    let a_trimmed = poly_trim(a.to_vec());
    let b_trimmed = poly_trim(b.to_vec());
    if poly_is_zero(&b_trimmed) {
        return (vec![BigRational::zero()], a_trimmed);
    }
    let deg_b = poly_degree(&b_trimmed);
    let deg_a = poly_degree(&a_trimmed);
    if poly_is_zero(&a_trimmed) || deg_a < deg_b {
        return (vec![BigRational::zero()], a_trimmed);
    }
    let lead_b = &b_trimmed[deg_b];
    let mut quot = vec![BigRational::zero(); deg_a - deg_b + 1];
    let mut rem = a_trimmed;
    while !poly_is_zero(&rem) && poly_degree(&rem) >= deg_b {
        let deg_r = poly_degree(&rem);
        let lead_r = rem[deg_r].clone();
        let factor = &lead_r / lead_b;
        let shift = deg_r - deg_b;
        quot[shift] += &factor;
        for (j, b_coeff) in b_trimmed.iter().enumerate() {
            rem[j + shift] -= &factor * b_coeff;
        }
        rem = poly_trim(rem);
    }
    (poly_trim(quot), rem)
}

/// Polynomial remainder $A \pmod B$.
fn poly_rem(a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
    poly_div_rem(a, b).1
}

/// Monic polynomial GCD over $\mathbb{Q}[x]$.
fn poly_gcd(a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
    let mut u = poly_trim(a.to_vec());
    let mut v = poly_trim(b.to_vec());
    while !poly_is_zero(&v) {
        let r = poly_rem(&u, &v);
        u = v;
        v = r;
    }
    poly_monic(&u)
}

/// Monic square-free part $P / \gcd(P, P')$.
fn poly_square_free_part(a: &[BigRational]) -> Vec<BigRational> {
    let trimmed = poly_trim(a.to_vec());
    if poly_is_zero(&trimmed) || poly_degree(&trimmed) == 0 {
        return poly_monic(&trimmed);
    }
    let deriv = poly_derivative(&trimmed);
    let g = poly_gcd(&trimmed, &deriv);
    let (sf, _) = poly_div_rem(&trimmed, &g);
    poly_monic(&sf)
}

/// Evaluates $P(x_0 - y)$ as a polynomial in $y$ of formal degree `deg_p`.
fn poly_eval_shifted_neg_y(coeffs: &[BigRational], x0: &BigRational) -> Vec<BigRational> {
    // Horner in Q[y] with multiplier (x0 - y)
    let mut acc = vec![BigRational::zero()];
    for c in coeffs.iter().rev() {
        let mut next = vec![BigRational::zero(); acc.len() + 1];
        for (i, a) in acc.iter().enumerate() {
            next[i] += a * x0;
            next[i + 1] -= a;
        }
        next[0] += c;
        acc = next;
    }
    acc
}

/// Evaluates $y^m P(x_0 / y) = \sum_{i=0}^m c_i x_0^i y^{m-i}$ as a polynomial in $y$.
fn poly_eval_homogenized_quotient_y(coeffs: &[BigRational], x0: &BigRational) -> Vec<BigRational> {
    let m = poly_degree(coeffs);
    let mut out = vec![BigRational::zero(); m + 1];
    let mut x_pow = BigRational::one();
    for (i, c) in coeffs.iter().enumerate() {
        out[m - i] = c * &x_pow;
        x_pow *= x0;
    }
    out
}

/// Computes the exact $(m, n)$-Sylvester resultant over $\mathbb{Q}$.
fn sylvester_resultant_fixed_degrees(
    a_coeffs: &[BigRational],
    deg_a: usize,
    b_coeffs: &[BigRational],
    deg_b: usize,
) -> BigRational {
    let dim = deg_a + deg_b;
    if dim == 0 {
        return BigRational::one();
    }
    let mut mat = vec![vec![BigRational::zero(); dim]; dim];
    for row in 0..deg_b {
        for k in 0..=deg_a {
            let val = a_coeffs.get(k).cloned().unwrap_or_else(BigRational::zero);
            mat[row][row + (deg_a - k)] = val;
        }
    }
    for row in 0..deg_a {
        for k in 0..=deg_b {
            let val = b_coeffs.get(k).cloned().unwrap_or_else(BigRational::zero);
            mat[deg_b + row][row + (deg_b - k)] = val;
        }
    }
    // Exact Gaussian elimination over BigRational
    let mut det = BigRational::one();
    let mut sign = 1i64;
    for i in 0..dim {
        let mut pivot_row = i;
        while pivot_row < dim && mat[pivot_row][i].is_zero() {
            pivot_row += 1;
        }
        if pivot_row == dim {
            return BigRational::zero();
        }
        if pivot_row != i {
            mat.swap(i, pivot_row);
            sign = -sign;
        }
        let pivot = mat[i][i].clone();
        det *= &pivot;
        let inv_pivot = BigRational::one() / &pivot;
        let (upper, lower) = mat.split_at_mut(i + 1);
        let pivot_slice = &upper[i];
        for r in lower.iter_mut() {
            if !r[i].is_zero() {
                let factor = &r[i] * &inv_pivot;
                r[i] = BigRational::zero();
                for (target, p_val) in r.iter_mut().skip(i + 1).zip(pivot_slice.iter().skip(i + 1)) {
                    *target -= &factor * p_val;
                }
            }
        }
    }
    if sign == -1 { -det } else { det }
}

/// Exact Newton interpolation over distinct rational nodes $(x_k, y_k)$.
fn interpolate_rational_poly(xs: &[BigRational], ys: &[BigRational]) -> Vec<BigRational> {
    let n = xs.len();
    if n == 0 {
        return vec![BigRational::zero()];
    }
    let mut dd = ys.to_vec();
    for j in 1..n {
        for i in (j..n).rev() {
            let num = &dd[i] - &dd[i - 1];
            let den = &xs[i] - &xs[i - j];
            dd[i] = num / den;
        }
    }
    // Convert Newton form to standard monomial basis using Horner backwards
    let mut poly = vec![dd[n - 1].clone()];
    for k in (0..n - 1).rev() {
        let mut next = vec![BigRational::zero(); poly.len() + 1];
        for (i, c) in poly.iter().enumerate() {
            next[i] -= c * &xs[k];
            next[i + 1] += c;
        }
        next[0] += &dd[k];
        poly = next;
    }
    poly_trim(poly)
}

/// Builds the Sturm sequence $(f_0, f_1, \dots, f_k)$ for a univariate polynomial $P$.
pub fn sturm_sequence(poly: &[BigRational]) -> Vec<Vec<BigRational>> {
    let p0 = poly_trim(poly.to_vec());
    if poly_is_zero(&p0) {
        return Vec::new();
    }
    let p1 = poly_derivative(&p0);
    if poly_is_zero(&p1) {
        return vec![p0];
    }
    let mut seq = vec![p0, p1];
    loop {
        let last_idx = seq.len() - 1;
        let rem = poly_rem(&seq[last_idx - 1], &seq[last_idx]);
        if poly_is_zero(&rem) {
            break;
        }
        let neg_rem = poly_negate(&rem);
        seq.push(neg_rem);
        if poly_degree(seq.last().unwrap()) == 0 {
            break;
        }
    }
    seq
}

/// Number of sign variations in the Sturm sequence evaluated at rational point $x$ (ignoring zeroes).
pub fn sign_variations(seq: &[Vec<BigRational>], x: &BigRational) -> usize {
    let mut signs: Vec<i8> = Vec::new();
    for p in seq {
        let val = AlgebraicNumber::eval_poly_at(p, x);
        if val > BigRational::zero() {
            signs.push(1);
        } else if val < BigRational::zero() {
            signs.push(-1);
        }
    }
    let mut count = 0;
    for i in 0..signs.len().saturating_sub(1) {
        if signs[i] != signs[i + 1] {
            count += 1;
        }
    }
    count
}

/// Exact count of distinct real roots of $P$ in the closed interval $[a, b]$ via Sturm's theorem.
pub fn count_real_roots_in_interval(
    seq: &[Vec<BigRational>],
    a: &BigRational,
    b: &BigRational,
) -> usize {
    if a > b || seq.is_empty() {
        return 0;
    }
    if a == b {
        return if AlgebraicNumber::eval_poly_at(&seq[0], a).is_zero() {
            1
        } else {
            0
        };
    }
    let v_a = sign_variations(seq, a);
    let v_b = sign_variations(seq, b);
    let mut roots = v_a.saturating_sub(v_b);
    if AlgebraicNumber::eval_poly_at(&seq[0], a).is_zero() {
        roots += 1;
    }
    roots
}

fn sturm_sequence_is_square_free(seq: &[Vec<BigRational>]) -> bool {
    seq.last()
        .is_some_and(|last| !poly_is_zero(last) && poly_degree(last) == 0)
}

fn deserialize_defining_polynomial<'de, D>(deserializer: D) -> Result<Vec<BigRational>, D::Error>
where
    D: Deserializer<'de>,
{
    struct CoefficientsVisitor;

    impl<'de> Visitor<'de> for CoefficientsVisitor {
        type Value = Vec<BigRational>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "at most {MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS} rational coefficients"
            )
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            if sequence
                .size_hint()
                .is_some_and(|size| size > MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS)
            {
                return Err(serde::de::Error::custom(format!(
                    "defining polynomial exceeds the coefficient limit of {MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS}"
                )));
            }

            let capacity = sequence
                .size_hint()
                .unwrap_or(0)
                .min(MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS);
            let mut coefficients = Vec::new();
            coefficients.try_reserve_exact(capacity).map_err(|_| {
                serde::de::Error::custom(format!(
                    "could not reserve {capacity} defining-polynomial coefficients"
                ))
            })?;
            while let Some(coefficient) = sequence.next_element()? {
                if coefficients.len() == MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS {
                    return Err(serde::de::Error::custom(format!(
                        "defining polynomial exceeds the coefficient limit of {MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS}"
                    )));
                }
                if coefficients.len() == coefficients.capacity() {
                    coefficients.try_reserve(1).map_err(|_| {
                        serde::de::Error::custom(
                            "could not grow the defining-polynomial coefficient buffer",
                        )
                    })?;
                }
                coefficients.push(coefficient);
            }
            Ok(coefficients)
        }
    }

    deserializer.deserialize_seq(CoefficientsVisitor)
}

/// Exact real algebraic number with certified root isolating ball.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct AlgebraicNumber {
    /// Coefficients in increasing degree: $c_0 + c_1 x + \dots + c_n x^n$.
    defining_poly_coeffs: Vec<BigRational>,
    /// Certified ball enclosing the unique target root.
    isolating_ball: RealBall,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AlgebraicNumberWire {
    #[serde(deserialize_with = "deserialize_defining_polynomial")]
    defining_poly_coeffs: Vec<BigRational>,
    isolating_ball: RealBall,
}

impl<'de> Deserialize<'de> for AlgebraicNumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AlgebraicNumberWire::deserialize(deserializer)?;
        Self::new(wire.defining_poly_coeffs, wire.isolating_ball).map_err(serde::de::Error::custom)
    }
}

impl AlgebraicNumber {
    /// Constructs an algebraic number with defining polynomial and certified root isolating ball.
    ///
    /// Verifies that the defining polynomial is square-free and, via Sturm's
    /// theorem, that the isolating interval contains exactly one real root.
    pub fn new(
        defining_poly_coeffs: Vec<BigRational>,
        isolating_ball: RealBall,
    ) -> Result<Self, AlgebraicError> {
        if defining_poly_coeffs.len() > MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS {
            return Err(AlgebraicError::PolynomialCoefficientLimitExceeded(
                defining_poly_coeffs.len(),
            ));
        }
        let coeffs = poly_trim(defining_poly_coeffs);
        if poly_is_zero(&coeffs) {
            return Err(AlgebraicError::ZeroPolynomial);
        }

        let low = isolating_ball.lower();
        let high = isolating_ball.upper();
        let sturm_seq = sturm_sequence(&coeffs);
        if !sturm_sequence_is_square_free(&sturm_seq) {
            return Err(AlgebraicError::NonSquareFreePolynomial);
        }
        let root_count = count_real_roots_in_interval(&sturm_seq, &low, &high);

        if root_count != 1 {
            return Err(AlgebraicError::InvalidIsolatingInterval(format!(
                "isolating interval [{}, {}] contains {} real roots of defining polynomial, expected exactly 1",
                low, high, root_count
            )));
        }

        Ok(Self {
            defining_poly_coeffs: coeffs,
            isolating_ball,
        })
    }

    /// Access the defining polynomial coefficients in ascending degree.
    pub fn defining_poly_coeffs(&self) -> &[BigRational] {
        &self.defining_poly_coeffs
    }

    /// Access the certified root isolating ball.
    pub fn isolating_ball(&self) -> &RealBall {
        &self.isolating_ball
    }

    /// Construct exact rational as a degree-1 algebraic number: $x - q = 0$.
    pub fn from_rational(q: BigRational) -> Self {
        Self {
            defining_poly_coeffs: vec![-q.clone(), BigRational::one()],
            isolating_ball: RealBall::exact(q),
        }
    }

    /// Construct exact integer as an algebraic number: $x - n = 0$.
    pub fn from_i64(n: i64) -> Self {
        Self::from_rational(BigRational::from_integer(BigInt::from(n)))
    }

    /// Degree of defining polynomial.
    pub fn degree(&self) -> usize {
        self.defining_poly_coeffs.len() - 1
    }

    /// Evaluates polynomial at a given rational point using Horner's method.
    pub fn eval_poly_at(coeffs: &[BigRational], x: &BigRational) -> BigRational {
        let mut acc = BigRational::zero();
        for c in coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    /// Evaluates polynomial over a certified ball using interval arithmetic.
    pub fn eval_ball(&self, ball: &RealBall) -> RealBall {
        let mut acc = RealBall::exact(BigRational::zero());
        for c in self.defining_poly_coeffs.iter().rev() {
            acc = acc.mul(ball).add(&RealBall::exact(c.clone()));
        }
        acc
    }

    /// Refines the root isolating interval using bisection and certified enclosure steps.
    pub fn refine_step(&mut self) {
        let low = self.isolating_ball.lower();
        let high = self.isolating_ball.upper();
        let mid = (&low + &high) / BigRational::from_integer(BigInt::from(2));

        let p_mid = Self::eval_poly_at(&self.defining_poly_coeffs, &mid);
        if p_mid.is_zero() {
            self.isolating_ball = RealBall::exact(mid);
            return;
        }

        let p_low = Self::eval_poly_at(&self.defining_poly_coeffs, &low);
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
    pub fn refine_to_radius(&mut self, target_radius: &BigRational) -> Result<(), AlgebraicError> {
        if target_radius < &BigRational::zero() {
            return Err(AlgebraicError::NegativeTargetRadius(
                target_radius.to_string(),
            ));
        }
        let mut steps = 0;
        const MAX_REFINE_STEPS: usize = 10_000;
        while self.isolating_ball.radius() > target_radius {
            if steps >= MAX_REFINE_STEPS {
                return Err(AlgebraicError::IterationLimitExceeded(steps));
            }
            self.refine_step();
            steps += 1;
        }
        Ok(())
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
        if self.defining_poly_coeffs[0].is_zero() && self.isolating_ball.contains_zero() {
            return 0;
        }
        // Refine until isolating interval excludes 0
        while self.isolating_ball.contains_zero() && !self.isolating_ball.radius().is_zero() {
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

    /// Exact negation $-\alpha$: defining polynomial $P(-x)$ with negated isolating ball.
    pub fn neg(&self) -> Self {
        let mut neg_coeffs = Vec::with_capacity(self.defining_poly_coeffs.len());
        for (deg, c) in self.defining_poly_coeffs.iter().enumerate() {
            if deg % 2 == 1 {
                neg_coeffs.push(-c);
            } else {
                neg_coeffs.push(c.clone());
            }
        }
        Self {
            defining_poly_coeffs: poly_monic(&neg_coeffs),
            isolating_ball: self.isolating_ball.neg(),
        }
    }

    /// Refines two algebraic operands and builds a certified `AlgebraicNumber` from a square-free
    /// resultant polynomial `sf_coeffs` and an interval combination closure.
    fn isolate_binary_result(
        &self,
        other: &Self,
        sf_coeffs: Vec<BigRational>,
        combine_balls: impl Fn(&RealBall, &RealBall) -> Result<RealBall, AlgebraicError>,
    ) -> Result<Self, AlgebraicError> {
        let sturm_seq = sturm_sequence(&sf_coeffs);
        if !sturm_sequence_is_square_free(&sturm_seq) {
            return Err(AlgebraicError::NonSquareFreePolynomial);
        }
        let mut lhs = self.clone();
        let mut rhs = other.clone();
        const MAX_ISOLATE_STEPS: usize = 2_000;
        for _ in 0..MAX_ISOLATE_STEPS {
            let cand = combine_balls(&lhs.isolating_ball, &rhs.isolating_ball)?;
            let low = cand.lower();
            let high = cand.upper();
            if count_real_roots_in_interval(&sturm_seq, &low, &high) == 1 {
                let final_ball = if Self::eval_poly_at(&sf_coeffs, &low).is_zero() {
                    RealBall::exact(low)
                } else if Self::eval_poly_at(&sf_coeffs, &high).is_zero() {
                    RealBall::exact(high)
                } else {
                    cand
                };
                return Self::new(sf_coeffs, final_ball);
            }
            lhs.refine_step();
            rhs.refine_step();
        }
        Err(AlgebraicError::IterationLimitExceeded(MAX_ISOLATE_STEPS))
    }

    /// Exact algebraic addition $\alpha + \beta$ via resultant $\text{Res}_y(P(x - y), Q(y))$
    /// and Sturm-certified interval isolation.
    pub fn add(&self, other: &Self) -> Result<Self, AlgebraicError> {
        let m = self.degree();
        let n = other.degree();
        let max_deg = m.saturating_mul(n);
        if max_deg.saturating_add(1) > MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS {
            return Err(AlgebraicError::PolynomialCoefficientLimitExceeded(
                max_deg.saturating_add(1),
            ));
        }
        let mut xs = Vec::with_capacity(max_deg + 1);
        let mut ys = Vec::with_capacity(max_deg + 1);
        for k in 0..=max_deg {
            let x_k = BigRational::from_integer(BigInt::from(k as i64));
            let shifted_p = poly_eval_shifted_neg_y(&self.defining_poly_coeffs, &x_k);
            let res_k = sylvester_resultant_fixed_degrees(
                &shifted_p,
                m,
                &other.defining_poly_coeffs,
                n,
            );
            xs.push(x_k);
            ys.push(res_k);
        }
        let resultant_poly = interpolate_rational_poly(&xs, &ys);
        let sf = poly_square_free_part(&resultant_poly);
        self.isolate_binary_result(other, sf, |a, b| Ok(a.add(b)))
    }

    /// Exact algebraic subtraction $\alpha - \beta$.
    pub fn sub(&self, other: &Self) -> Result<Self, AlgebraicError> {
        self.add(&other.neg())
    }

    /// Exact algebraic multiplication $\alpha \cdot \beta$ via resultant
    /// $\text{Res}_y(y^m P(x / y), Q(y))$ and Sturm-certified interval isolation.
    pub fn mul(&self, other: &Self) -> Result<Self, AlgebraicError> {
        let mut a_check = self.clone();
        let mut b_check = other.clone();
        if a_check.sign() == 0 || b_check.sign() == 0 {
            return Ok(Self::from_i64(0));
        }
        let m = self.degree();
        let n = other.degree();
        let max_deg = m.saturating_mul(n);
        if max_deg.saturating_add(1) > MAX_ALGEBRAIC_POLYNOMIAL_COEFFICIENTS {
            return Err(AlgebraicError::PolynomialCoefficientLimitExceeded(
                max_deg.saturating_add(1),
            ));
        }
        let mut xs = Vec::with_capacity(max_deg + 1);
        let mut ys = Vec::with_capacity(max_deg + 1);
        for k in 0..=max_deg {
            let x_k = BigRational::from_integer(BigInt::from(k as i64));
            let hom_p = poly_eval_homogenized_quotient_y(&self.defining_poly_coeffs, &x_k);
            let res_k = sylvester_resultant_fixed_degrees(
                &hom_p,
                m,
                &other.defining_poly_coeffs,
                n,
            );
            xs.push(x_k);
            ys.push(res_k);
        }
        let resultant_poly = interpolate_rational_poly(&xs, &ys);
        let sf = poly_square_free_part(&resultant_poly);
        self.isolate_binary_result(other, sf, |a, b| Ok(a.mul(b)))
    }

    /// Exact algebraic reciprocal $1 / \alpha$ for $\alpha \ne 0$.
    pub fn inv(&self) -> Result<Self, AlgebraicError> {
        let mut a = self.clone();
        if a.sign() == 0 {
            return Err(AlgebraicError::DivisionByZero);
        }
        let mut rev = a.defining_poly_coeffs.clone();
        rev.reverse();
        let sf = poly_square_free_part(&rev);
        let sturm_seq = sturm_sequence(&sf);
        const MAX_ISOLATE_STEPS: usize = 2_000;
        for _ in 0..MAX_ISOLATE_STEPS {
            if !a.isolating_ball.contains_zero() {
                let cand = a.isolating_ball.inv()?;
                let low = cand.lower();
                let high = cand.upper();
                if count_real_roots_in_interval(&sturm_seq, &low, &high) == 1 {
                    let final_ball = if Self::eval_poly_at(&sf, &low).is_zero() {
                        RealBall::exact(low)
                    } else if Self::eval_poly_at(&sf, &high).is_zero() {
                        RealBall::exact(high)
                    } else {
                        cand
                    };
                    return Self::new(sf, final_ball);
                }
            }
            a.refine_step();
        }
        Err(AlgebraicError::IterationLimitExceeded(MAX_ISOLATE_STEPS))
    }

    /// Exact algebraic division $\alpha / \beta$.
    pub fn checked_div(&self, other: &Self) -> Result<Self, AlgebraicError> {
        let inv_other = other.inv()?;
        self.mul(&inv_other)
    }

    /// Refines and returns a certified `RealBall` enclosure with radius $\le 10^{-(\text{precision\_digits} + 1)}$.
    pub fn to_real_ball(&self, precision_digits: u32) -> Result<RealBall, AlgebraicError> {
        let target = BigRational::new(
            BigInt::one(),
            BigInt::from(10).pow(precision_digits.saturating_add(1)),
        );
        let mut refined = self.clone();
        refined.refine_to_radius(&target)?;
        Ok(refined.isolating_ball)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn q(i: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(i))
    }

    #[test]
    fn from_i64_constructs_degree_one_algebraic_with_exact_ball() {
        // from_i64 must produce an algebraic number whose isolating
        // ball is the exact point [n, n] and whose degree is 1.
        let a = AlgebraicNumber::from_i64(5);
        assert_eq!(a.degree(), 1);
        assert_eq!(a.isolating_ball().lower(), q(5));
        assert_eq!(a.isolating_ball().upper(), q(5));
        assert!(a.isolating_ball().radius().is_zero());
    }

    #[test]
    fn from_rational_constructs_degree_one_algebraic_for_non_integer() {
        // 1/3 is a degree-1 algebraic with isolating ball [1/3, 1/3].
        let third = BigRational::new(1.into(), 3.into());
        let a = AlgebraicNumber::from_rational(third.clone());
        assert_eq!(a.degree(), 1);
        assert_eq!(a.isolating_ball().lower(), third);
        assert_eq!(a.isolating_ball().upper(), third);
    }

    #[test]
    fn degree_reports_correct_value_for_exact_rationals() {
        // Every exact rational is a linear algebraic number.
        assert_eq!(AlgebraicNumber::from_i64(0).degree(), 1);
        assert_eq!(AlgebraicNumber::from_i64(-7).degree(), 1);
        let a = AlgebraicNumber::from_rational(BigRational::new(22.into(), 7.into()));
        assert_eq!(a.degree(), 1);
    }

    #[test]
    fn sturm_sequence_constant_polynomial_is_empty() {
        // The constant polynomial 0 has no Sturm sequence; the
        // function returns an empty Vec. This pins the boundary
        // behavior expected by the broader certificate pipeline.
        let poly = vec![q(0)];
        assert!(sturm_sequence(&poly).is_empty());
    }

    #[test]
    fn sturm_sequence_for_x_squared_minus_2_starts_with_polynomial_then_derivative() {
        // For P(x) = x^2 - 2, the Sturm sequence starts with
        // P_0 = x^2 - 2 and P_1 = P_0' = 2x. The exact length of
        // the tail depends on the sign of the final remainder; pin
        // the documented head entries and assert the sequence ends
        // with a non-zero constant.
        let poly = vec![BigRational::from_integer(BigInt::from(-2)), q(0), q(1)];
        let seq = sturm_sequence(&poly);
        assert!(seq.len() >= 2);
        assert_eq!(seq[0], poly);
        assert_eq!(seq[1], vec![q(0), q(2)]);
        let last = seq.last().unwrap();
        assert_eq!(last.len(), 1);
        assert!(!last[0].is_zero());
    }

    #[test]
    fn test_algebraic_number_field_arithmetic_and_real_ball_refinement() {
        // sqrt(2): root of x^2 - 2 in [1, 2]
        let sqrt2 = AlgebraicNumber::new(
            vec![q(-2), q(0), q(1)],
            RealBall::new(
                BigRational::new(3.into(), 2.into()),
                BigRational::new(1.into(), 2.into()),
            )
            .unwrap(),
        )
        .unwrap();
        // sqrt(3): root of x^2 - 3 in [1, 2]
        let sqrt3 = AlgebraicNumber::new(
            vec![q(-3), q(0), q(1)],
            RealBall::new(
                BigRational::new(3.into(), 2.into()),
                BigRational::new(1.into(), 2.into()),
            )
            .unwrap(),
        )
        .unwrap();

        // sqrt(2) * sqrt(2) == 2 (root of x^2 - 4 in positive interval, or exact 2)
        let two_alg = sqrt2.mul(&sqrt2).unwrap();
        assert_eq!(
            AlgebraicNumber::eval_poly_at(two_alg.defining_poly_coeffs(), &q(2)),
            q(0)
        );
        let two_ball = two_alg.to_real_ball(20).unwrap();
        assert!(two_ball.contains(&q(2)));

        // sqrt(2) + sqrt(3) has minimal polynomial x^4 - 10*x^2 + 1
        let sum = sqrt2.add(&sqrt3).unwrap();
        assert_eq!(sum.degree(), 4);
        assert_eq!(
            sum.defining_poly_coeffs(),
            &[q(1), q(0), q(-10), q(0), q(1)]
        );
        // High-precision enclosure of sqrt(2) + sqrt(3) ≈ 3.1462643699419723
        let sum_ball = sum.to_real_ball(15).unwrap();
        let approx = BigRational::new(
            BigInt::from(3_146_264_369_941_972i64),
            BigInt::from(1_000_000_000_000_000i64),
        );
        let tol = BigRational::new(BigInt::one(), BigInt::from(10).pow(14));
        assert!(
            sum_ball.lower() <= &approx + &tol && &approx - &tol <= sum_ball.upper(),
            "sqrt(2)+sqrt(3) ball {sum_ball} must overlap reference {approx}"
        );

        // (sqrt(3) - sqrt(2)) * (sqrt(3) + sqrt(2)) == 1
        let diff = sqrt3.sub(&sqrt2).unwrap();
        let prod = diff.mul(&sum).unwrap();
        let prod_ball = prod.to_real_ball(20).unwrap();
        assert!(
            prod_ball.contains(&q(1)),
            "(sqrt(3)-sqrt(2))*(sqrt(3)+sqrt(2)) ball {prod_ball} must contain 1"
        );

        // 1 / sqrt(2) == sqrt(2) / 2 (minimal polynomial x^2 - 1/2)
        let inv_sqrt2 = sqrt2.inv().unwrap();
        assert_eq!(
            inv_sqrt2.defining_poly_coeffs(),
            &[BigRational::new((-1).into(), 2.into()), q(0), q(1)]
        );
        assert!(matches!(
            AlgebraicNumber::from_i64(0).inv(),
            Err(AlgebraicError::DivisionByZero)
        ));
    }
}
