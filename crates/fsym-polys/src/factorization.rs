//! Polynomial factorization and square-free decomposition (WS09).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::univariate::UnivariatePoly;
use fsym_core::{BigRational, Symbol};
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
