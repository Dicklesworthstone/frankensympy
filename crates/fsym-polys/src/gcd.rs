//! Polynomial GCD, Extended Euclidean Algorithm, and Bezout Certificates (WS09).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::multivariate::{MultivariatePoly, TermOrder};
use crate::univariate::UnivariatePoly;
use fsym_budget::Unbounded;
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

/// Exact divisibility certificate for a monic multivariate common divisor.
///
/// This certificate proves only `divisor | A` and `divisor | B`. It does not prove that the
/// divisor is greatest. Use [`crate::checked_multivariate_gcd`] for a complete GCD/LCM claim
/// with independently checked principal-intersection evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultivariateDivisibilityCertificate {
    pub divisor: MultivariatePoly,
    pub quotient_a: MultivariatePoly,
    pub quotient_b: MultivariatePoly,
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
        self.validate_shape()?;
        other.validate_shape()?;
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
        self.validate_shape()?;
        other.validate_shape()?;
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

/// Checks a Bezout certificate for $\gcd(A, B)$ using exact polynomial identities.
pub fn verify_bezout_certificate(
    a: &UnivariatePoly,
    b: &UnivariatePoly,
    cert: &BezoutCertificate,
) -> Result<(), PolyError> {
    a.validate_shape()?;
    b.validate_shape()?;
    cert.gcd.validate_shape()?;
    cert.u.validate_shape()?;
    cert.v.validate_shape()?;

    for polynomial in [b, &cert.gcd, &cert.u, &cert.v] {
        if polynomial.gen_sym != a.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                a.gen_sym.name.clone(),
                polynomial.gen_sym.name.clone(),
            ));
        }
    }

    // The zero polynomial is the gcd only when both inputs are zero. Without this admission,
    // G = U = V = 0 would satisfy the linear-combination check for every input while the later
    // divisibility checks were skipped.
    if cert.gcd.is_zero() {
        if a.is_zero() && b.is_zero() {
            return Ok(());
        }
        return Err(PolyError::IdentityCheckFailed(
            "zero cannot be the GCD of a nonzero polynomial".to_string(),
        ));
    }

    // Over Q[x], nonzero associates differ by a unit. Monic normalization is therefore part of
    // the exact GCD claim rather than presentation metadata.
    if !cert.gcd.leading_coeff().is_one() {
        return Err(PolyError::IdentityCheckFailed(
            "GCD certificate must report a monic polynomial".to_string(),
        ));
    }

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

    // 2. Verify that GCD divides A and B. Together with the Bezout identity, every common divisor
    // divides the reported GCD; the monic check above chooses the canonical associate.
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

    Ok(())
}

impl MultivariatePoly {
    /// Computes the monic GCD of two multivariate polynomials in $\mathbb{Q}[x_1, \ldots, x_n]$.
    ///
    /// Uses the complete, independently checked principal-intersection path in
    /// [`crate::checked_multivariate_gcd`], including its degree, term, coefficient,
    /// allocation and work limits. Call that API directly to supply cancellation-aware
    /// meters and retain the accepted evidence instead of just its polynomial value.
    pub fn gcd(&self, other: &Self) -> Result<Self, PolyError> {
        let (gcd, _) =
            crate::checked_multivariate_gcd(self, other, &mut Unbounded, &mut Unbounded)?
                .into_parts();
        Ok(gcd)
    }

    /// Computes the monic least common multiple (LCM) of two multivariate polynomials in $\mathbb{Q}[x_1, \ldots, x_n]$.
    ///
    /// Shares the checked and bounded path used by [`Self::gcd`].
    pub fn lcm(&self, other: &Self) -> Result<Self, PolyError> {
        let (_, lcm) =
            crate::checked_multivariate_gcd(self, other, &mut Unbounded, &mut Unbounded)?
                .into_parts();
        Ok(lcm)
    }

    /// Computes a GCD candidate with an exact certificate that it divides both inputs.
    ///
    /// The candidate comes from [`Self::gcd`], but the certificate intentionally grants only the
    /// weaker common-divisor claim. This serialized certificate omits the complete
    /// intersection evidence, so verifying it does not establish greatestness.
    pub fn gcd_candidate_with_divisibility_certificate(
        &self,
        other: &Self,
    ) -> Result<MultivariateDivisibilityCertificate, PolyError> {
        let divisor = self.gcd(other)?;
        let (mut q_a_vec, rem_a) = self.div_rem(std::slice::from_ref(&divisor), TermOrder::Lex)?;
        let (mut q_b_vec, rem_b) = other.div_rem(std::slice::from_ref(&divisor), TermOrder::Lex)?;
        if !rem_a.is_zero() || !rem_b.is_zero() || q_a_vec.len() != 1 || q_b_vec.len() != 1 {
            return Err(PolyError::General(
                "common-divisor certificate division failed".to_string(),
            ));
        }
        let cert = MultivariateDivisibilityCertificate {
            divisor,
            quotient_a: q_a_vec.pop().ok_or_else(|| {
                PolyError::General("missing first divisibility quotient".to_string())
            })?,
            quotient_b: q_b_vec.pop().ok_or_else(|| {
                PolyError::General("missing second divisibility quotient".to_string())
            })?,
        };
        verify_multivariate_divisibility_certificate(self, other, &cert)?;
        Ok(cert)
    }
}

/// Checks that a monic multivariate polynomial divides both inputs using exact identities.
///
/// Acceptance does not establish that the divisor is greatest. In particular, the unit
/// polynomial is a valid certificate for every pair in the same polynomial ring.
pub fn verify_multivariate_divisibility_certificate(
    a: &MultivariatePoly,
    b: &MultivariatePoly,
    cert: &MultivariateDivisibilityCertificate,
) -> Result<(), PolyError> {
    a.validate_shape()?;
    b.validate_shape()?;
    cert.divisor.validate_shape()?;
    cert.quotient_a.validate_shape()?;
    cert.quotient_b.validate_shape()?;

    for polynomial in [b, &cert.divisor, &cert.quotient_a, &cert.quotient_b] {
        if polynomial.generators != a.generators {
            return Err(PolyError::IncompatibleGenerators(
                format!("{:?}", a.generators),
                format!("{:?}", polynomial.generators),
            ));
        }
    }

    // 1. Check the common divisor is monic.
    if !cert.divisor.is_zero() {
        let lc = cert
            .divisor
            .leading_coeff(TermOrder::Lex)
            .ok_or_else(|| PolyError::General("divisor has no leading coefficient".to_string()))?;
        if !lc.is_one() {
            return Err(PolyError::General(
                "common divisor must be monic".to_string(),
            ));
        }
    }

    // 2. Check divisor * quotient_a == A and divisor * quotient_b == B.
    let prod_a = cert.divisor.mul(&cert.quotient_a)?;
    if prod_a != *a {
        return Err(PolyError::IdentityCheckFailed(
            "common-divisor quotient check failed: D * Q_A != A".to_string(),
        ));
    }
    let prod_b = cert.divisor.mul(&cert.quotient_b)?;
    if prod_b != *b {
        return Err(PolyError::IdentityCheckFailed(
            "common-divisor quotient check failed: D * Q_B != B".to_string(),
        ));
    }

    Ok(())
}
