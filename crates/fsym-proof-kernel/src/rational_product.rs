//! Independent exact checking of finite rational products on integer ranges.
//!
//! A telescoper certifies `N(k)/D(k) = c*R(k+1)/R(k)` by dense coefficient
//! convolution. Every original inverse guard is supplied separately by the
//! caller and must be reconstructed completely before its integer zeros are
//! excluded. A cancelled factor never removes an original domain obligation.
//! No generator, factor search, numerical sampling, gamma identity, analytic
//! continuation, or infinite-product convergence assumption enters this checker.

mod modular;

use modular::check_factors;
pub use modular::{MAX_PRODUCT_DOMAIN_PRIME, RationalProductModularFactor};

use crate::rational_sum::{
    MAX_SUM_COEFFICIENT_BITS, RationalSumPole, RationalSumQuadratic, RationalSumVerificationError,
    Reference, validate,
};
use fsym_budget::{BudgetMeter, MeterError};
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};
use thiserror::Error;

pub const MAX_DIRECT_PRODUCT_TERMS: u64 = 2001;
const MAX_GUARDS: usize = 128;
const MAX_ENDPOINT_BITS: u64 = 4096;

/// Complete factor product for an integer-domain check. Linear roots are
/// retained explicitly; quadratic factors are checked for irreducibility;
/// higher-degree factors have independent modular no-integer-root witnesses.
/// The leading scalar is taken from the original polynomial, not this witness.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RationalProductFactors {
    pub roots: Vec<RationalSumPole>,
    pub quadratics: Vec<RationalSumQuadratic>,
    pub modular: Vec<RationalProductModularFactor>,
}

/// Original-source domain obligations, in their original retained order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalProductDomain {
    pub denominator: RationalProductFactors,
    pub guards: Vec<RationalProductFactors>,
}

/// Untrusted finite-product witness. The expected source, guards and bounds
/// are independent checker inputs; no candidate can substitute any of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RationalProductCertificate {
    /// An independently recomputed, bounded exact product.
    Direct { value: BigRational },
    /// A real factor is zero, and every original factor is defined. The
    /// reciprocal of such a range is undefined and is explicitly rejected.
    Zero {
        at: BigInt,
        domain: RationalProductDomain,
    },
    /// `R = numerator/denominator`, with complete domain witnesses for R.
    Telescoping {
        scale: BigRational,
        numerator: Vec<BigRational>,
        denominator: Vec<BigRational>,
        domain: RationalProductDomain,
        numerator_factors: RationalProductFactors,
        denominator_factors: RationalProductFactors,
    },
}

#[derive(Debug, Error)]
pub enum RationalProductVerificationError {
    #[error("rational product verification refused: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalSumVerificationError),
}

impl From<RationalSumVerificationError> for RationalProductVerificationError {
    fn from(error: RationalSumVerificationError) -> Self {
        match error {
            RationalSumVerificationError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalProductVerificationError>;

fn invalid(reason: &'static str) -> RationalProductVerificationError {
    RationalProductVerificationError::Invalid(reason)
}

fn has_integer_root(factors: &RationalProductFactors, lower: &BigInt, upper: &BigInt) -> bool {
    factors.roots.iter().any(|root| {
        root.root.is_integer() && root.root.numer() >= lower && root.root.numer() <= upper
    })
}

fn check_domain(
    denominator: &[BigRational],
    guards: &[Vec<BigRational>],
    domain: &RationalProductDomain,
    lower: &BigInt,
    upper: &BigInt,
    reference: &mut Reference<'_, impl BudgetMeter>,
) -> Result<()> {
    if domain.guards.len() != guards.len() {
        return Err(invalid("original inverse guard count mismatch"));
    }
    check_factors(denominator, &domain.denominator, reference)?;
    if has_integer_root(&domain.denominator, lower, upper) {
        return Err(invalid("original factor has an integer pole in the range"));
    }
    for (guard, witness) in guards.iter().zip(&domain.guards) {
        check_factors(guard, witness, reference)?;
        if has_integer_root(witness, lower, upper) {
            return Err(invalid("original inverse guard vanishes in the range"));
        }
    }
    Ok(())
}

fn power(
    base: &BigRational,
    exponent: &BigInt,
    reference: &mut Reference<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    validate(std::slice::from_ref(base))?;
    if base.is_one() || exponent.is_zero() {
        return Ok(BigRational::one());
    }
    if base == &-BigRational::one() {
        return Ok(if (exponent % BigInt::from(2)).is_zero() {
            BigRational::one()
        } else {
            -BigRational::one()
        });
    }
    if base.is_zero() {
        return Err(invalid("a nonzero telescoper requires a nonzero scalar"));
    }
    // For a reduced rational with |base| != 1, a numerator or denominator
    // has magnitude >= 2. At least exponent+1 output bits are unavoidable.
    let mut exponent = exponent
        .to_u64()
        .filter(|value| *value < MAX_SUM_COEFFICIENT_BITS)
        .ok_or_else(|| invalid("scalar power exceeds the exact output bit bound"))?;
    let mut result = BigRational::one();
    let mut factor = base.clone();
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = reference.mul(&result, &factor)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            factor = reference.mul(&factor, &factor)?;
        }
    }
    Ok(result)
}

/// Check a complete finite rational product. Reversed ranges use the Karr
/// convention `product(a,b) = 1/product(b+1,a-1)`; adjacent bounds give one.
/// Every integer factor must be defined even when an earlier factor is zero.
pub fn verify_rational_product(
    numerator: &[BigRational],
    denominator: &[BigRational],
    guards: &[Vec<BigRational>],
    lower: &BigInt,
    upper: &BigInt,
    certificate: &RationalProductCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<BigRational> {
    meter.checkpoint()?;
    if lower.bits() > MAX_ENDPOINT_BITS || upper.bits() > MAX_ENDPOINT_BITS {
        return Err(invalid("integer endpoint height limit"));
    }
    if guards.len() > MAX_GUARDS {
        return Err(invalid("original inverse guard count limit"));
    }
    for polynomial in [numerator, denominator]
        .into_iter()
        .chain(guards.iter().map(Vec::as_slice))
    {
        validate(polynomial)?;
    }
    if denominator.iter().all(Zero::is_zero) || guards.iter().any(|g| g.iter().all(Zero::is_zero)) {
        return Err(invalid("original source is nowhere defined"));
    }
    let mut reference = Reference::new(meter);
    reference.reserve(4)?;
    let reversed = upper < lower;
    let (lo, hi) = if reversed {
        (upper + BigInt::one(), lower - BigInt::one())
    } else {
        (lower.clone(), upper.clone())
    };
    if hi < lo {
        return match certificate {
            RationalProductCertificate::Direct { value } if value.is_one() => {
                reference.work(1)?;
                Ok(BigRational::one())
            }
            _ => Err(invalid("empty product requires the exact unit witness")),
        };
    }
    let count = &hi - &lo + BigInt::one();
    match certificate {
        RationalProductCertificate::Direct { value } => {
            validate(std::slice::from_ref(value))?;
            let length = count
                .to_u64()
                .filter(|length| *length <= MAX_DIRECT_PRODUCT_TERMS)
                .ok_or_else(|| invalid("direct product term count limit"))?;
            let mut point = BigRational::from_integer(lo);
            let mut actual = BigRational::one();
            for _ in 0..length {
                for guard in guards {
                    if reference.evaluate(guard, &point)?.is_zero() {
                        return Err(invalid("original inverse guard vanishes in the range"));
                    }
                }
                let top = reference.evaluate(numerator, &point)?;
                let bottom = reference.evaluate(denominator, &point)?;
                let factor = reference.quotient(top, bottom)?;
                actual = reference.mul(&actual, &factor)?;
                point = reference.add(&point, &BigRational::one())?;
            }
            if reversed {
                actual = reference.quotient(BigRational::one(), actual)?;
            }
            if &actual != value {
                return Err(invalid("direct product value mismatch"));
            }
            reference.work(1)?;
            Ok(actual)
        }
        RationalProductCertificate::Zero { at, domain } => {
            check_domain(denominator, guards, domain, &lo, &hi, &mut reference)?;
            if reversed || at < &lo || at > &hi {
                return Err(invalid(
                    "zero-factor witness is outside a forward product range",
                ));
            }
            if !reference
                .evaluate(numerator, &BigRational::from_integer(at.clone()))?
                .is_zero()
            {
                return Err(invalid("proposed zero factor is not zero"));
            }
            reference.work(1)?;
            Ok(BigRational::zero())
        }
        RationalProductCertificate::Telescoping {
            scale,
            numerator: top,
            denominator: bottom,
            domain,
            numerator_factors,
            denominator_factors,
        } => {
            for polynomial in [
                top.as_slice(),
                bottom.as_slice(),
                std::slice::from_ref(scale),
            ] {
                validate(polynomial)?;
            }
            if scale.is_zero() {
                return Err(invalid("zero scalar in multiplicative identity"));
            }
            check_domain(denominator, guards, domain, &lo, &hi, &mut reference)?;
            check_factors(top, numerator_factors, &mut reference)?;
            check_factors(bottom, denominator_factors, &mut reference)?;
            let after_hi = &hi + BigInt::one();
            if has_integer_root(numerator_factors, &lo, &after_hi)
                || has_integer_root(denominator_factors, &lo, &after_hi)
            {
                return Err(invalid(
                    "telescoper is zero or undefined on an integer endpoint chain",
                ));
            }
            let top_shift = reference.shift_one(top)?;
            let bottom_shift = reference.shift_one(bottom)?;
            let lhs = reference.product(numerator, top)?;
            let lhs = reference.product(&lhs, &bottom_shift)?;
            let rhs = reference.product(denominator, bottom)?;
            let rhs = reference.product(&rhs, &top_shift)?;
            let rhs = reference.product(&rhs, std::slice::from_ref(scale))?;
            if lhs != rhs {
                return Err(invalid("multiplicative polynomial identity mismatch"));
            }
            let start = BigRational::from_integer(lo);
            let end = BigRational::from_integer(after_hi);
            let start_top = reference.evaluate(top, &start)?;
            let start_bottom = reference.evaluate(bottom, &start)?;
            let end_top = reference.evaluate(top, &end)?;
            let end_bottom = reference.evaluate(bottom, &end)?;
            let top_ratio = reference.mul(&end_top, &start_bottom)?;
            let bottom_ratio = reference.mul(&end_bottom, &start_top)?;
            let endpoint_ratio = reference.quotient(top_ratio, bottom_ratio)?;
            let scalar = power(scale, &count, &mut reference)?;
            let mut value = reference.mul(&scalar, &endpoint_ratio)?;
            if reversed {
                value = reference.quotient(BigRational::one(), value)?;
            }
            reference.work(1)?;
            Ok(value)
        }
    }
}
