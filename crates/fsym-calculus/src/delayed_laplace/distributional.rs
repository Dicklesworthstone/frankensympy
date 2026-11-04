//! Complete bounded causal time shifts of rational impulse and regular parts.
//!
//! The generator proposes one distributional inverse for each exact delay.
//! Acceptance independently decomposes the original sums/products and checks
//! the complete polynomial-plus-regular transform identity for every group.
//! Typed substitution shifts both components; only the regular function may
//! receive a Heaviside multiplier on the public causal surface. In particular,
//! exp(-a*s)*s^k gives delta^(k)(t-a), without a step at its support.
//!
//! This separate receipt does not change the ordinary-only delayed API. It
//! shares the bounded input grammar, original domain obligations, cumulative
//! budgets and output limits, and does not claim general distribution algebra
//! or a portable proof-kernel certificate.

use super::{
    Capped, MAX_BITS, MAX_TERMS, Result, admit, bits, charge, copy_expression, generate_groups,
    generate_plans, number, prepay_shift, reference_groups, shift_expression,
};
use crate::rational_laplace::distributional::{
    DistributionalLaplaceCertificate, generate_distributional_laplace_certificate,
    verify_distributional_laplace_certificate,
};
use crate::rational_laplace::{RationalLaplaceError, admit_rational_laplace_source};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::{Signed, Zero};

/// Untrusted complete polynomial and regular inverse at one claimed delay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedDistributionalLaplaceTermCertificate {
    pub delay: BigRational,
    pub distributional: DistributionalLaplaceCertificate,
}

/// Ordered coverage of every exact delay group in the original source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayedDistributionalLaplaceCertificate {
    pub terms: Vec<DelayedDistributionalLaplaceTermCertificate>,
}

/// Separately shifted ordinary and impulse distributions at one causal delay.
#[derive(Debug, Clone)]
pub struct VerifiedDelayedDistributionalLaplacePart {
    delay: BigRational,
    shifted_regular_function: Expr,
    shifted_impulses: Expr,
}

impl VerifiedDelayedDistributionalLaplacePart {
    pub fn delay(&self) -> &BigRational {
        &self.delay
    }
    /// The public causal surface attaches H(t-delay) to this component only.
    pub fn shifted_regular_function(&self) -> &Expr {
        &self.shifted_regular_function
    }
    /// These shifted delta derivatives must not acquire a step multiplier.
    pub fn shifted_impulses(&self) -> &Expr {
        &self.shifted_impulses
    }
    pub fn into_parts(self) -> (BigRational, Expr, Expr) {
        (
            self.delay,
            self.shifted_regular_function,
            self.shifted_impulses,
        )
    }
}

/// Checked complete causal inverse on a sufficient common right half-plane.
#[derive(Debug, Clone)]
pub struct VerifiedDelayedDistributionalLaplace {
    source: Expr,
    transform_variable: Symbol,
    time_variable: Symbol,
    certificate: DelayedDistributionalLaplaceCertificate,
    parts: Vec<VerifiedDelayedDistributionalLaplacePart>,
    sufficient_abscissa: BigRational,
}

impl VerifiedDelayedDistributionalLaplace {
    pub fn source(&self) -> &Expr {
        &self.source
    }
    pub fn transform_variable(&self) -> &Symbol {
        &self.transform_variable
    }
    pub fn time_variable(&self) -> &Symbol {
        &self.time_variable
    }
    pub fn certificate(&self) -> &DelayedDistributionalLaplaceCertificate {
        &self.certificate
    }
    pub fn parts(&self) -> &[VerifiedDelayedDistributionalLaplacePart] {
        &self.parts
    }
    pub fn sufficient_abscissa(&self) -> &BigRational {
        &self.sufficient_abscissa
    }
    pub fn into_parts(self) -> Vec<VerifiedDelayedDistributionalLaplacePart> {
        self.parts
    }
}

fn verify_inner(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    certificate: DelayedDistributionalLaplaceCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedDelayedDistributionalLaplace> {
    admit(source, variable, time, meter)?;
    if certificate.terms.len() > MAX_TERMS {
        return Err(RationalLaplaceError::ResourceLimit(
            "delayed distributional certificate term count",
        ));
    }
    for term in &certificate.terms {
        if term.delay.is_negative()
            || bits(&term.delay) > MAX_BITS
            || term.delay.denom().is_zero()
            || term.delay.denom().is_negative()
        {
            return Err(RationalLaplaceError::Verification(
                "invalid claimed distributional time delay",
            ));
        }
    }
    let groups = reference_groups(source, variable, meter)?.ok_or(
        RationalLaplaceError::Verification("source is outside the delayed rational grammar"),
    )?;
    if groups.len() != certificate.terms.len() {
        return Err(RationalLaplaceError::Verification(
            "missing or duplicated delayed distributional term",
        ));
    }
    // No earlier candidate or supported component can mask a later invalid
    // original rational child, including children erased by zero or ^0.
    for group in &groups {
        admit_rational_laplace_source(&group.source, variable, time, meter)?;
    }
    charge(meter, groups.len(), groups.len() * 512)?;
    let mut parts = Vec::with_capacity(groups.len());
    let mut abscissa = None;
    let mut output_nodes = 0usize;
    for (group, candidate) in groups.into_iter().zip(&certificate.terms) {
        if !group.delay.metered_cmp(&candidate.delay, meter)?.is_eq() {
            return Err(RationalLaplaceError::Verification(
                "changed or reordered distributional time delay",
            ));
        }
        let checked = verify_distributional_laplace_certificate(
            &group.source,
            variable,
            time,
            &candidate.distributional,
            meter,
        )?;
        let replace_bound = match &abscissa {
            None => true,
            Some(previous) => checked
                .sufficient_abscissa()
                .metered_cmp(previous, meter)?
                .is_gt(),
        };
        if replace_bound {
            abscissa = Some(number(checked.sufficient_abscissa(), meter)?);
        }
        // Both components and all groups consume one complete-output bound.
        // Account before copying either expression or constructing its shift.
        prepay_shift(
            checked.regular_function(),
            time,
            &group.delay,
            &mut output_nodes,
            meter,
        )?;
        prepay_shift(
            checked.impulses(),
            time,
            &group.delay,
            &mut output_nodes,
            meter,
        )?;
        let (shifted_regular_function, shifted_impulses) = if group.delay.is_zero() {
            checked.into_parts()
        } else {
            let negative = BigRational::zero().metered_sub(&group.delay, meter)?;
            let displacement = Expr::Add(vec![Expr::Sym(time.clone()), rational_expr(negative)]);
            (
                shift_expression(checked.regular_function(), time, &displacement),
                shift_expression(checked.impulses(), time, &displacement),
            )
        };
        parts.push(VerifiedDelayedDistributionalLaplacePart {
            delay: group.delay,
            shifted_regular_function,
            shifted_impulses,
        });
    }
    let source = copy_expression(source, meter)?;
    charge(meter, 2, variable.name.len() + time.name.len() + 256)?;
    let receipt = VerifiedDelayedDistributionalLaplace {
        source,
        transform_variable: variable.clone(),
        time_variable: time.clone(),
        certificate,
        parts,
        sufficient_abscissa: abscissa.unwrap_or_else(BigRational::zero),
    };
    meter.checkpoint()?;
    Ok(receipt)
}

/// Accept an untrusted candidate by value, without an unmetered deep copy.
/// Neither the reported delays nor the impulse coefficients drive coverage.
pub fn verify_delayed_distributional_laplace_certificate(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    certificate: DelayedDistributionalLaplaceCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedDelayedDistributionalLaplace> {
    let mut meter = Capped::new(verifier);
    let result = verify_inner(
        source,
        transform_variable,
        time_variable,
        certificate,
        &mut meter,
    );
    if let Some(error) = meter.stopped {
        return Err(error.into());
    }
    result
}

/// None is a bounded grammar/degree miss before any rational factorization.
/// Unsupported rational blocks, invalid originals and resource/cancellation
/// failures stay terminal; no accepted partial inverse is ever returned.
pub fn metered_inverse_delayed_distributional_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Option<VerifiedDelayedDistributionalLaplace>> {
    let mut generation = Capped::new(generator);
    let mut verification = Capped::new(verifier);
    let result = (|| {
        admit(source, transform_variable, time_variable, &mut generation)?;
        let Some(plans) = generate_plans(source, transform_variable, &mut generation)? else {
            return Ok(None);
        };
        let Some(groups) = generate_groups(plans, transform_variable, &mut generation)? else {
            return Ok(None);
        };
        // Admit all original rational sources before the first factor search
        // or unsupported regular tail can terminate candidate generation.
        for group in &groups {
            admit_rational_laplace_source(
                &group.source,
                transform_variable,
                time_variable,
                &mut generation,
            )?;
        }
        charge(&mut generation, groups.len(), groups.len() * 512)?;
        let mut terms = Vec::with_capacity(groups.len());
        for group in groups {
            let distributional = generate_distributional_laplace_certificate(
                &group.source,
                transform_variable,
                time_variable,
                &mut generation,
            )?;
            terms.push(DelayedDistributionalLaplaceTermCertificate {
                delay: group.delay,
                distributional,
            });
        }
        let checked = verify_inner(
            source,
            transform_variable,
            time_variable,
            DelayedDistributionalLaplaceCertificate { terms },
            &mut verification,
        )?;
        generation.checkpoint()?;
        verification.checkpoint()?;
        Ok(Some(checked))
    })();
    if let Some(error) = generation.stopped.or(verification.stopped) {
        return Err(error.into());
    }
    result
}

pub fn inverse_delayed_distributional_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
) -> Result<Option<VerifiedDelayedDistributionalLaplace>> {
    metered_inverse_delayed_distributional_laplace(
        source,
        transform_variable,
        time_variable,
        &mut Unbounded,
        &mut Unbounded,
    )
}
