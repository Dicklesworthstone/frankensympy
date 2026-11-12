//! Complete cancellation of an exact rational source over an ordered QQ ring.
//!
//! Original-source admission retains every inverse guard, even below held zero
//! products, nested reciprocals and zeroth powers. The checked complete GCD is
//! bound to that exact numerator and denominator. Separate coefficient checks
//! establish both quotient identities; dividing by a greatest common divisor
//! makes the resulting pair coprime. Its denominator is monic, so normalization
//! is unique and zero is represented by 0/1. No factorization is requested.
//!
//! This is equality on the original source domain. The output expression may
//! extend across removed poles, but the immutable receipt retains all original
//! guards. It is an in-process composition receipt, not a portable proof claim.
//!
//! Source, polynomial and complete-GCD envelopes remain unchanged: at most eight
//! typed variables, 4096 source nodes/terms, degree256, 4096-bit coefficients,
//! and the existing auxiliary-ring bounds for general GCDs. Both phases retain
//! cumulative 20M-compute/128MiB logical allocation caps through nested calls
//! and publication, in addition to the caller's budget and cancellation ledger.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_VARIABLES, PolynomialWork, copy_poly, refused, ring_storage, same_ring, validate_poly,
};
use crate::{
    MultivariatePoly, PolyError, TermOrder, VerifiedMultivariateGcd, VerifiedRationalSource,
    bounded_polynomial_from_expr, checked_multivariate_gcd, checked_polynomial_division,
    metered_rational_source, verify_bounded_division, verify_rational_source,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::One;
use std::mem::size_of;
use std::sync::Arc;

/// Untrusted reduced values with immutable source and complete-GCD evidence.
/// The GCD must use the exact original source fraction, before cancellation.
#[derive(Debug, Clone)]
pub struct RationalCancellationCertificate {
    pub original: Arc<VerifiedRationalSource>,
    pub gcd: Arc<VerifiedMultivariateGcd>,
    pub numerator: MultivariatePoly,
    pub denominator: MultivariatePoly,
}

/// Original-source-bound coprime pair in QQ[variables], with monic denominator.
/// Original guards remain authoritative even when a cancelled pole no longer
/// occurs in the reduced pair. There is no unchecked constructor.
#[derive(Debug, Clone)]
pub struct VerifiedRationalCancellation {
    certificate: RationalCancellationCertificate,
}

impl VerifiedRationalCancellation {
    pub fn source(&self) -> &Expr {
        self.certificate.original.source()
    }

    pub fn variables(&self) -> &[Symbol] {
        self.certificate.original.variables()
    }

    pub fn numerator(&self) -> &MultivariatePoly {
        &self.certificate.numerator
    }

    pub fn denominator(&self) -> &MultivariatePoly {
        &self.certificate.denominator
    }

    pub fn guards(&self) -> &[MultivariatePoly] {
        self.certificate.original.guards()
    }

    pub fn certificate(&self) -> &RationalCancellationCertificate {
        &self.certificate
    }

    /// Move the reduced pair and its original-domain evidence without cloning
    /// a source, coefficient or guard outside the request's accounting.
    pub fn into_parts(
        self,
    ) -> (
        MultivariatePoly,
        MultivariatePoly,
        Arc<VerifiedRationalSource>,
    ) {
        (
            self.certificate.numerator,
            self.certificate.denominator,
            self.certificate.original,
        )
    }

    /// Lift only accepted values, prepaying bounded output storage and checking
    /// every recovered coefficient before publication. The supplied phase
    /// meters should be the same ledgers used for source admission/cancellation.
    /// No unmetered simplifier or expression expansion participates.
    pub fn metered_expression(
        &self,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<Expr, PolyError> {
        let mut generation = PolynomialWork::new(generator);
        let mut verification = PolynomialWork::new(verifier);
        let mut nodes = 5usize;
        for polynomial in [self.numerator(), self.denominator()] {
            validate_poly(polynomial, &mut generation)?;
            for powers in polynomial.terms().keys() {
                generation.tick(powers.len() + 1)?;
                nodes += 2 + 4 * powers.iter().filter(|&&power| power != 0).count();
                if nodes > 4096 {
                    return Err(refused("cancelled expression exceeds 4096 output nodes"));
                }
            }
        }
        generation.allocate(nodes * (2048 + ring_storage(self.variables())))?;
        let numerator = self.numerator().to_expr()?;
        let denominator = self.denominator().to_expr()?;
        for (expression, expected) in [
            (&numerator, self.numerator()),
            (&denominator, self.denominator()),
        ] {
            let recovered =
                bounded_polynomial_from_expr(expression, self.variables(), &mut verification)?;
            same_polynomial(&recovered, expected, &mut verification)?;
        }
        let expression = if denominator.is_one() {
            numerator
        } else {
            let inverse = Expr::Pow(Arc::new(denominator), Arc::new(Expr::from_i64(-1)));
            if numerator.is_one() {
                inverse
            } else {
                Expr::Mul(vec![numerator, inverse])
            }
        };
        generation.tick(1)?;
        verification.tick(1)?;
        Ok(expression)
    }
}

fn check_polynomial(
    polynomial: &MultivariatePoly,
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if polynomial.generators().len() > MAX_VARIABLES {
        return Err(refused("cancellation output exceeds eight variables"));
    }
    let mut bytes = 0usize;
    for variable in polynomial.generators() {
        bytes = bytes
            .checked_add(variable.name.len())
            .filter(|&value| value <= 65_536)
            .ok_or_else(|| refused("cancellation generator names exceed 65536 bytes"))?;
        work.tick(variable.name.len() + 1)?;
    }
    // Bound untrusted diagnostic payloads before comparing or formatting rings.
    same_ring(polynomial.generators(), variables)?;
    validate_poly(polynomial, work)
}

fn same_polynomial(
    actual: &MultivariatePoly,
    expected: &MultivariatePoly,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_polynomial(actual, expected.generators(), work)?;
    if actual.terms().len() != expected.terms().len() {
        return Err(refused(
            "cancellation witness changed original term coverage",
        ));
    }
    for ((powers, value), (original_powers, original_value)) in
        actual.terms().iter().zip(expected.terms())
    {
        let limbs = value.numer().limb_count()
            + value.denom().limb_count()
            + original_value.numer().limb_count()
            + original_value.denom().limb_count();
        work.tick(powers.len() + limbs as usize + 1)?;
        if powers != original_powers || value != original_value {
            return Err(refused(
                "cancellation witness changed an exact original coefficient",
            ));
        }
    }
    Ok(())
}

/// Compare the exact held source after independent source preflight. Pay for
/// each borrowed child and scalar/name comparison; algebraic equality cannot
/// substitute for the original expression or its retained domain obligations.
fn same_source(
    actual: &Expr,
    expected: &Expr,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.allocate(128)?;
    let mut pending = vec![(actual, expected)];
    let mut visited = 0usize;
    while let Some((actual, expected)) = pending.pop() {
        visited += 1;
        work.tick(1)?;
        if visited > 4096 {
            return Err(refused(
                "cancellation source comparison exceeds its node bound",
            ));
        }
        let equal = match (actual, expected) {
            (Expr::Integer(a), Expr::Integer(b)) => {
                work.tick((a.limb_count() + b.limb_count()) as usize + 1)?;
                a == b
            }
            (Expr::Rational(a), Expr::Rational(b)) => {
                work.tick(
                    (a.numer().limb_count()
                        + a.denom().limb_count()
                        + b.numer().limb_count()
                        + b.denom().limb_count()) as usize
                        + 1,
                )?;
                a == b
            }
            (Expr::Sym(a), Expr::Sym(b)) => {
                work.tick(a.name.len() + b.name.len() + 1)?;
                a == b
            }
            (Expr::Add(a), Expr::Add(b)) | (Expr::Mul(a), Expr::Mul(b)) => {
                if a.len() != b.len() || a.len() > 4096 - visited - pending.len() {
                    return Err(refused(
                        "cancellation evidence changed original source children",
                    ));
                }
                work.allocate(a.len() * 128)?;
                pending.extend(a.iter().zip(b));
                true
            }
            (Expr::Pow(a, x), Expr::Pow(b, y)) => {
                if 2 > 4096 - visited - pending.len() {
                    return Err(refused(
                        "cancellation source comparison exceeds its node bound",
                    ));
                }
                work.allocate(256)?;
                pending.push((a.as_ref(), b.as_ref()));
                pending.push((x.as_ref(), y.as_ref()));
                true
            }
            _ => false,
        };
        if !equal {
            return Err(refused(
                "cancellation evidence changed the exact held source",
            ));
        }
    }
    Ok(())
}

fn scalar_multiple(
    polynomial: &MultivariatePoly,
    scalar: &BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let mut result = copy_poly(polynomial, work)?;
    for value in result.terms.values_mut() {
        *value = work.multiply(value, scalar)?;
    }
    Ok(result)
}

fn checked_quotient(
    polynomial: &MultivariatePoly,
    divisor: &MultivariatePoly,
    generator: &mut PolynomialWork<'_, impl BudgetMeter>,
    verifier: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let (mut quotients, remainder) = checked_polynomial_division(
        polynomial,
        std::slice::from_ref(divisor),
        TermOrder::Lex,
        generator,
        verifier,
    )?;
    if quotients.len() != 1 || !remainder.is_zero() {
        return Err(refused("cancellation division must have zero remainder"));
    }
    quotients
        .pop()
        .ok_or_else(|| refused("missing exact cancellation quotient"))
}

/// Independently bind the complete original rational source, its typed ring and
/// guards, then check both normalized quotient identities against the greatest
/// common divisor. This checks full reduction, not merely equality of ratios.
pub fn verify_rational_cancel(
    source: &Expr,
    variables: &[Symbol],
    certificate: &RationalCancellationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalCancellation, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    // This borrowed admission precedes any source comparison or owned copy.
    // Its independent recurrence checks every original inverse guard, including
    // guards hidden by zero products, outer zero powers and nested reciprocals.
    let original = verify_rational_source(
        source,
        variables,
        certificate.original.certificate(),
        &mut work,
    )?;
    work.tick(ring_storage(variables) + 1)?;
    if certificate.original.variables() != variables {
        return Err(refused(
            "cancellation evidence changed the exact original source or ring",
        ));
    }
    same_source(certificate.original.source(), source, &mut work)?;
    same_polynomial(
        certificate.gcd.left(),
        &original.fraction().numerator,
        &mut work,
    )?;
    same_polynomial(
        certificate.gcd.right(),
        &original.fraction().denominator,
        &mut work,
    )?;
    check_polynomial(&certificate.numerator, variables, &mut work)?;
    check_polynomial(&certificate.denominator, variables, &mut work)?;
    if certificate
        .denominator
        .terms()
        .last_key_value()
        .is_none_or(|(_, coefficient)| !coefficient.is_one())
    {
        return Err(refused("cancelled denominator must be nonzero and monic"));
    }
    let leading = original
        .fraction()
        .denominator
        .terms()
        .last_key_value()
        .ok_or(PolyError::DivisionByZero)?
        .1;
    work.allocate(size_of::<MultivariatePoly>() + ring_storage(variables))?;
    let zero = MultivariatePoly::zero(variables.to_vec());
    for (input, normalized) in [
        (&original.fraction().numerator, &certificate.numerator),
        (&original.fraction().denominator, &certificate.denominator),
    ] {
        // The complete GCD is monic, hence the unnormalized denominator
        // quotient has exactly the original denominator's leading coefficient.
        // Undo the candidate's normalization with checked scalar arithmetic,
        // then replay the whole coefficient identity independently of division.
        let quotient = scalar_multiple(normalized, leading, &mut work)?;
        verify_bounded_division(
            input,
            std::slice::from_ref(certificate.gcd.gcd()),
            std::slice::from_ref(&quotient),
            &zero,
            TermOrder::Lex,
            &mut work,
        )?;
    }
    work.allocate(
        size_of::<VerifiedRationalCancellation>() + size_of::<VerifiedRationalSource>() + 64,
    )?;
    let accepted = VerifiedRationalCancellation {
        certificate: RationalCancellationCertificate {
            original: Arc::new(original),
            gcd: Arc::clone(&certificate.gcd),
            numerator: copy_poly(&certificate.numerator, &mut work)?,
            denominator: copy_poly(&certificate.denominator, &mut work)?,
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Completely cancel an original exact rational expression, including sums and
/// nested reciprocals, without invoking irreducible factorization. All lower
/// receipts and normalized output remain private until acceptance completes.
pub fn checked_rational_cancel(
    source: &Expr,
    variables: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRationalCancellation, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let original = metered_rational_source(source, variables, &mut generation, &mut verification)?;
    let gcd = checked_multivariate_gcd(
        &original.fraction().numerator,
        &original.fraction().denominator,
        &mut generation,
        &mut verification,
    )?;
    let mut numerator = checked_quotient(
        &original.fraction().numerator,
        gcd.gcd(),
        &mut generation,
        &mut verification,
    )?;
    let mut denominator = checked_quotient(
        &original.fraction().denominator,
        gcd.gcd(),
        &mut generation,
        &mut verification,
    )?;
    let leading = original
        .fraction()
        .denominator
        .terms()
        .last_key_value()
        .ok_or(PolyError::DivisionByZero)?
        .1;
    for coefficient in numerator
        .terms
        .values_mut()
        .chain(denominator.terms.values_mut())
    {
        *coefficient = generation.divide(coefficient, leading)?;
    }
    generation.allocate(
        size_of::<RationalCancellationCertificate>()
            + size_of::<VerifiedRationalSource>()
            + size_of::<VerifiedMultivariateGcd>()
            + 64,
    )?;
    let candidate = RationalCancellationCertificate {
        original: Arc::new(original),
        gcd: Arc::new(gcd),
        numerator,
        denominator,
    };
    let accepted = verify_rational_cancel(source, variables, &candidate, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
