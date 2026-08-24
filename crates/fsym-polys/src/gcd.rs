//! Polynomial GCD, Extended Euclidean Algorithm, and Bezout Certificates (WS09).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::univariate::UnivariatePoly;
use fsym_core::BigRational;
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

/// Bezout identity certificate for polynomial GCD:
/// $\gcd(A, B) = U \cdot A + V \cdot B$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BezoutCertificate {
    pub gcd: UnivariatePoly,
    pub u: UnivariatePoly,
    pub v: UnivariatePoly,
}

impl UnivariatePoly {
    /// Monic normalization: divide all coefficients by the leading coefficient.
    pub fn to_monic(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        let lc = self.leading_coeff();
        let new_coeffs = self.coeffs.iter().map(|c| c / lc).collect();
        Self::new(self.gen_sym.clone(), new_coeffs)
    }

    /// Computes the monic GCD of two univariate polynomials using Euclidean algorithm.
    pub fn gcd(&self, other: &Self) -> Result<Self, PolyError> {
        if self.gen_sym != other.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                self.gen_sym.name.clone(),
                other.gen_sym.name.clone(),
            ));
        }
        if self.is_zero() {
            return Ok(other.to_monic());
        }
        if other.is_zero() {
            return Ok(self.to_monic());
        }

        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero() {
            let (_, rem) = a.div_rem(&b)?;
            a = b;
            b = rem;
        }

        Ok(a.to_monic())
    }

    /// Extended Euclidean algorithm producing GCD and Bezout coefficients $U, V$
    /// such that $U \cdot A + V \cdot B = \gcd(A, B)$.
    pub fn extended_gcd(&self, other: &Self) -> Result<BezoutCertificate, PolyError> {
        if self.gen_sym != other.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                self.gen_sym.name.clone(),
                other.gen_sym.name.clone(),
            ));
        }

        let sym = self.gen_sym.clone();
        if other.is_zero() {
            let gcd = self.to_monic();
            let lc = self.leading_coeff();
            let u = if lc.is_zero() {
                Self::one(sym.clone())
            } else {
                Self::new(sym.clone(), vec![BigRational::one() / lc])
            };
            return Ok(BezoutCertificate {
                gcd,
                u,
                v: Self::zero(sym),
            });
        }

        let mut old_r = self.clone();
        let mut r = other.clone();

        let mut old_s = Self::one(sym.clone());
        let mut s = Self::zero(sym.clone());

        let mut old_t = Self::zero(sym.clone());
        let mut t = Self::one(sym.clone());

        while !r.is_zero() {
            let (quotient, remainder) = old_r.div_rem(&r)?;
            old_r = r;
            r = remainder;

            let qs = quotient.mul(&s)?;
            let new_s = old_s.sub(&qs)?;
            old_s = s;
            s = new_s;

            let qt = quotient.mul(&t)?;
            let new_t = old_t.sub(&qt)?;
            old_t = t;
            t = new_t;
        }

        // Normalize so GCD is monic
        let lc = old_r.leading_coeff().clone();
        if !lc.is_zero() && !lc.is_one() {
            let scale = Self::new(sym.clone(), vec![BigRational::one() / &lc]);
            let monic_gcd = old_r.mul(&scale)?;
            let scaled_s = old_s.mul(&scale)?;
            let scaled_t = old_t.mul(&scale)?;
            Ok(BezoutCertificate {
                gcd: monic_gcd,
                u: scaled_s,
                v: scaled_t,
            })
        } else {
            Ok(BezoutCertificate {
                gcd: old_r,
                u: old_s,
                v: old_t,
            })
        }
    }
}

/// Independently verify a Bezout certificate for $\gcd(A, B)$.
pub fn verify_bezout_certificate(
    a: &UnivariatePoly,
    b: &UnivariatePoly,
    cert: &BezoutCertificate,
) -> Result<(), PolyError> {
    // 1. Verify linear combination: U * A + V * B == GCD
    let ua = cert.u.mul(a)?;
    let vb = cert.v.mul(b)?;
    let sum = ua.add(&vb)?;
    if sum != cert.gcd {
        return Err(PolyError::IdentityCheckFailed(format!(
            "Bezout certificate linear combination failed: U*A + V*B (`{sum}`) != GCD (`{}`)",
            cert.gcd
        )));
    }

    // 2. Verify that GCD divides A and B
    if !cert.gcd.is_zero() {
        let (_, rem_a) = a.div_rem(&cert.gcd)?;
        if !rem_a.is_zero() {
            return Err(PolyError::IdentityCheckFailed(format!(
                "GCD `{}` does not divide A `{a}`: remainder `{rem_a}`",
                cert.gcd
            )));
        }

        let (_, rem_b) = b.div_rem(&cert.gcd)?;
        if !rem_b.is_zero() {
            return Err(PolyError::IdentityCheckFailed(format!(
                "GCD `{}` does not divide B `{b}`: remainder `{rem_b}`",
                cert.gcd
            )));
        }
    }

    Ok(())
}
