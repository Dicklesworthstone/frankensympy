//! Polynomial factorization and square-free decomposition (WS09).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::univariate::UnivariatePoly;
use fsym_core::{BigInt, BigRational, Symbol};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

/// Factor with multiplicity: $(f(x), k)$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactorTerm {
    pub poly: UnivariatePoly,
    pub multiplicity: usize,
}

/// Verifiable factorization result: $P(x) = \text{scale} \cdot \prod f_i(x)^{e_i}$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactorizationResult {
    pub scale: BigRational,
    pub factors: Vec<FactorTerm>,
}

impl FactorizationResult {
    /// Reconstructs the expanded product polynomial from the factored representation.
    pub fn expand(&self, sym: Symbol) -> Result<UnivariatePoly, PolyError> {
        let mut prod = UnivariatePoly::new(sym.clone(), vec![self.scale.clone()]);
        for factor in &self.factors {
            factor.poly.validate_shape()?;
            if factor.poly.gen_sym != sym {
                return Err(PolyError::IncompatibleGenerators(
                    sym.name.clone(),
                    factor.poly.gen_sym.name.clone(),
                ));
            }
            if factor.multiplicity == 0 {
                return Err(PolyError::IdentityCheckFailed(
                    "factor multiplicity must be positive".to_string(),
                ));
            }
            let exponent = u32::try_from(factor.multiplicity).map_err(|_| {
                PolyError::IdentityCheckFailed(format!(
                    "factor multiplicity {} exceeds the supported exponent range",
                    factor.multiplicity
                ))
            })?;
            let factor_pow = factor.poly.pow(exponent)?;
            prod = prod.mul(&factor_pow)?;
        }
        Ok(prod)
    }
}

/// Computes the square-free decomposition of a univariate polynomial using Yun's algorithm:
/// $P(x) = c \cdot f_1^1 \cdot f_2^2 \cdots f_k^k$ where each $f_i$ is square-free and pairwise coprime.
pub fn square_free_decomposition(poly: &UnivariatePoly) -> Result<FactorizationResult, PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        return Ok(FactorizationResult {
            scale: BigRational::zero(),
            factors: Vec::new(),
        });
    }

    let lc = poly.leading_coeff().clone();
    let monic_p = poly.to_monic();
    if monic_p.degree() == Some(0) {
        return Ok(FactorizationResult {
            scale: lc,
            factors: Vec::new(),
        });
    }

    let p_prime = monic_p.derivative();
    let c = monic_p.gcd(&p_prime)?;

    if c.degree() == Some(0) {
        // Polynomial is already square-free
        return Ok(FactorizationResult {
            scale: lc,
            factors: vec![FactorTerm {
                poly: monic_p,
                multiplicity: 1,
            }],
        });
    }

    let (mut w, _) = monic_p.div_rem(&c)?;
    let (mut y, _) = p_prime.div_rem(&c)?;

    let mut factors = Vec::new();
    let mut i = 1;

    while !w.is_one() {
        let y_sub_w_prime = y.sub(&w.derivative())?;
        let a_i = w.gcd(&y_sub_w_prime)?;

        if !a_i.is_one() {
            factors.push(FactorTerm {
                poly: a_i.clone(),
                multiplicity: i,
            });
        }

        let (next_w, _) = w.div_rem(&a_i)?;
        let (next_y, _) = y_sub_w_prime.div_rem(&a_i)?;

        w = next_w;
        y = next_y;
        i += 1;
    }

    Ok(FactorizationResult { scale: lc, factors })
}

/// Independently verify a polynomial factorization certificate.
///
/// Acceptance criteria:
/// 1. The product of factors $\text{scale} \cdot \prod f_i(x)^{e_i}$ equals $P(x)$ exactly.
/// 2. Each factor $f_i(x)$ is monic and square-free ($\gcd(f_i, f_i') = 1$).
/// 3. All factors $f_i(x), f_j(x)$ are pairwise coprime ($\gcd(f_i, f_j) = 1$ for $i \neq j$).
pub fn verify_factorization_certificate(
    poly: &UnivariatePoly,
    factorization: &FactorizationResult,
) -> Result<(), PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        if !factorization.scale.is_zero() || !factorization.factors.is_empty() {
            return Err(PolyError::IdentityCheckFailed(
                "the zero polynomial requires canonical scale zero with no factors".to_string(),
            ));
        }
    } else if factorization.scale.is_zero() {
        return Err(PolyError::IdentityCheckFailed(
            "a nonzero polynomial cannot have factorization scale zero".to_string(),
        ));
    }

    // Admit the certificate shape before exponentiation. This prevents a
    // malformed multiplicity from triggering disproportionate dense work and
    // enforces the canonical monic, positive-degree factor convention.
    let target_degree = poly.degree().unwrap_or(0);
    let mut reconstructed_degree = 0usize;
    for factor in &factorization.factors {
        factor.poly.validate_shape()?;
        if factor.poly.gen_sym != poly.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                poly.gen_sym.name.clone(),
                factor.poly.gen_sym.name.clone(),
            ));
        }
        if factor.multiplicity == 0 {
            return Err(PolyError::IdentityCheckFailed(
                "factor multiplicity must be positive".to_string(),
            ));
        }
        u32::try_from(factor.multiplicity).map_err(|_| {
            PolyError::IdentityCheckFailed(format!(
                "factor multiplicity {} exceeds the supported exponent range",
                factor.multiplicity
            ))
        })?;
        let factor_degree = factor.poly.degree().ok_or_else(|| {
            PolyError::IdentityCheckFailed("zero polynomial cannot be a factor".to_string())
        })?;
        if factor_degree == 0 {
            return Err(PolyError::IdentityCheckFailed(
                "constant factors must be absorbed into the scale".to_string(),
            ));
        }
        if !factor.poly.leading_coeff().is_one() {
            return Err(PolyError::IdentityCheckFailed(format!(
                "factor `{}` is not monic",
                factor.poly
            )));
        }
        let degree_contribution =
            factor_degree
                .checked_mul(factor.multiplicity)
                .ok_or_else(|| {
                    PolyError::IdentityCheckFailed(
                        "factorization degree calculation overflowed".to_string(),
                    )
                })?;
        reconstructed_degree = reconstructed_degree
            .checked_add(degree_contribution)
            .ok_or_else(|| {
                PolyError::IdentityCheckFailed(
                    "factorization degree calculation overflowed".to_string(),
                )
            })?;
        if reconstructed_degree > target_degree {
            return Err(PolyError::IdentityCheckFailed(format!(
                "factorization degree {reconstructed_degree} exceeds polynomial degree {target_degree}"
            )));
        }
    }
    if reconstructed_degree != target_degree {
        return Err(PolyError::IdentityCheckFailed(format!(
            "factorization degree {reconstructed_degree} does not match polynomial degree {target_degree}"
        )));
    }

    // 1. Reconstruct product and check equality
    let reconstructed = factorization.expand(poly.gen_sym.clone())?;
    if &reconstructed != poly {
        return Err(PolyError::IdentityCheckFailed(format!(
            "Factorization product `{reconstructed}` does not match original polynomial `{poly}`"
        )));
    }

    // 2. Verify square-freeness of each factor
    for factor in &factorization.factors {
        let f_prime = factor.poly.derivative();
        let g = factor.poly.gcd(&f_prime)?;
        if !g.is_one() {
            return Err(PolyError::IdentityCheckFailed(format!(
                "Factor `{}` is not square-free: gcd with derivative is `{g}`",
                factor.poly
            )));
        }
    }

    // 3. Verify pairwise coprimality
    for (i, f_i) in factorization.factors.iter().enumerate() {
        for (_j, f_j) in factorization.factors.iter().enumerate().skip(i + 1) {
            let g = f_i.poly.gcd(&f_j.poly)?;
            if !g.is_one() {
                return Err(PolyError::IdentityCheckFailed(format!(
                    "Factors `{}` and `{}` are not coprime: gcd is `{g}`",
                    f_i.poly, f_j.poly
                )));
            }
        }
    }

    Ok(())
}

fn integer_divisors(n: &BigInt, limit: usize) -> Vec<BigInt> {
    let abs_n = if n < &BigInt::zero() {
        -n.clone()
    } else {
        n.clone()
    };
    if abs_n.is_zero() {
        return Vec::new();
    }
    if abs_n.is_one() {
        return vec![BigInt::one()];
    }
    if let Ok(val) = u64::try_from(abs_n.clone()) {
        let mut divs = Vec::new();
        let mut d = 1u64;
        while d * d <= val && divs.len() < limit {
            if val % d == 0 {
                divs.push(BigInt::from(d));
                if d * d != val {
                    divs.push(BigInt::from(val / d));
                }
            }
            d += 1;
        }
        divs.sort();
        divs
    } else {
        vec![BigInt::one(), abs_n]
    }
}

fn find_rational_roots(poly: &UnivariatePoly) -> Vec<BigRational> {
    if poly.degree() == Some(0) || poly.is_zero() {
        return Vec::new();
    }
    let mut roots = Vec::new();
    let mut current = poly.clone();
    if current.coeffs[0].is_zero() {
        roots.push(BigRational::zero());
        if let Ok(m) = UnivariatePoly::monomial(current.gen_sym.clone(), BigRational::one(), 1)
            && let Ok((q, _)) = current.div_rem(&m)
        {
            current = q;
        }
    }
    if current.degree() == Some(0) {
        return roots;
    }
    let mut denom_lcm = BigInt::one();
    for c in &current.coeffs {
        let d = c.denom();
        let gcd_d = denom_lcm.gcd(d);
        if !gcd_d.is_zero() {
            denom_lcm = (&denom_lcm * d) / gcd_d;
        }
    }
    let mut int_coeffs: Vec<BigInt> = current
        .coeffs
        .iter()
        .map(|c| (c * BigRational::from_integer(denom_lcm.clone())).to_integer())
        .collect();
    while int_coeffs.len() > 1 && int_coeffs.last().is_some_and(|c| c.is_zero()) {
        int_coeffs.pop();
    }
    if int_coeffs.len() <= 1 {
        return roots;
    }
    let a0 = &int_coeffs[0];
    let an = &int_coeffs[int_coeffs.len() - 1];
    let p_divs = integer_divisors(a0, 500);
    let q_divs = integer_divisors(an, 100);

    for p in &p_divs {
        for q in &q_divs {
            if q.is_zero() {
                continue;
            }
            for sign in &[1i64, -1i64] {
                let candidate_p = if *sign == -1 { -p.clone() } else { p.clone() };
                let candidate = BigRational::new(candidate_p, q.clone());
                let val = current.eval(&candidate);
                if val.is_zero() && !roots.contains(&candidate) {
                    roots.push(candidate);
                }
            }
        }
    }
    roots
}

fn factor_square_free_monic(poly: &UnivariatePoly) -> Result<Vec<UnivariatePoly>, PolyError> {
    poly.validate_shape()?;
    if poly.degree() == Some(0) || poly.is_zero() {
        return Ok(Vec::new());
    }
    let mut factors = Vec::new();
    let mut rem = poly.clone();
    let roots = find_rational_roots(&rem);
    for r in roots {
        let linear = UnivariatePoly::new(rem.gen_sym.clone(), vec![-r, BigRational::one()]);
        while let Ok((q, remainder)) = rem.div_rem(&linear) {
            if remainder.is_zero() {
                factors.push(linear.clone());
                rem = q;
                if rem.degree() == Some(0) {
                    break;
                }
            } else {
                break;
            }
        }
    }
    if rem.degree() > Some(0) {
        if rem.degree() == Some(2) {
            let b = &rem.coeffs[1];
            let c = &rem.coeffs[0];
            let four = BigRational::from_integer(BigInt::from(4));
            let discr = b * b - four * c;
            if discr >= BigRational::zero() {
                let num_sqrt = discr.numer().sqrt();
                let den_sqrt = discr.denom().sqrt();
                if &num_sqrt * &num_sqrt == *discr.numer()
                    && &den_sqrt * &den_sqrt == *discr.denom()
                {
                    let d = BigRational::new(num_sqrt, den_sqrt);
                    let two = BigRational::from_integer(BigInt::from(2));
                    let r1 = (-b + &d) / &two;
                    let r2 = (-b - &d) / &two;
                    factors.push(UnivariatePoly::new(
                        rem.gen_sym.clone(),
                        vec![-r1, BigRational::one()],
                    ));
                    factors.push(UnivariatePoly::new(
                        rem.gen_sym.clone(),
                        vec![-r2, BigRational::one()],
                    ));
                    return Ok(factors);
                }
            }
        }
        if rem != UnivariatePoly::one(rem.gen_sym.clone()) {
            factors.push(rem);
        }
    }
    Ok(factors)
}

/// Computes the complete factorization of a univariate polynomial over $\mathbb{Q}[x]$:
/// $P(x) = \text{scale} \cdot \prod f_i(x)^{e_i}$.
pub fn factor_polynomial(poly: &UnivariatePoly) -> Result<FactorizationResult, PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        return Ok(FactorizationResult {
            scale: BigRational::zero(),
            factors: Vec::new(),
        });
    }
    let sqf = square_free_decomposition(poly)?;
    let mut factors_vec: Vec<FactorTerm> = Vec::new();

    for sqf_term in sqf.factors {
        let irreducible = factor_square_free_monic(&sqf_term.poly)?;
        for irr in irreducible {
            if let Some(existing) = factors_vec.iter_mut().find(|f| f.poly == irr) {
                existing.multiplicity += sqf_term.multiplicity;
            } else {
                factors_vec.push(FactorTerm {
                    poly: irr,
                    multiplicity: sqf_term.multiplicity,
                });
            }
        }
    }

    let res = FactorizationResult {
        scale: sqf.scale,
        factors: factors_vec,
    };
    verify_factorization_certificate(poly, &res)?;
    Ok(res)
}
