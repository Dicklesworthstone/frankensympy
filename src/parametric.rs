//! Original-source root families over a complete declared parameter domain.
//!
//! Unknowns and parameters are separate ordered typed coordinates. Uniform
//! and complete-cover acceptance comes from the independent core checker;
//! adaptive subdivision returns only an untrusted candidate. Accepted family
//! receipts can subsequently supply source-bound solution sensitivities.

#![forbid(unsafe_code)]

use crate::certification::{
    CertificationError, MAX_CERTIFICATION_NAME_BYTES, MAX_CERTIFICATION_SOURCE_BYTES,
    ResidualProblem, charge,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::parametric_cover::{ParametricRootCoverOptions, propose_parametric_root_cover};
use fsym_core::{
    Expr, MAX_ROOT_BOX_VARIABLES, ParametricRootBoxReceipt, ParametricRootBoxWitness,
    ParametricRootCoverError, ParametricRootCoverReceipt, ParametricRootCoverWitness, RealBall,
    RootBoxError, Symbol, verify_parametric_root_box, verify_parametric_root_cover,
};
use std::mem::size_of;
use thiserror::Error;

/// Refusal, resource exhaustion and incomplete coverage never grant a family.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParametricError {
    #[error(transparent)]
    Certification(#[from] CertificationError),
    #[error(transparent)]
    Cover(#[from] ParametricRootCoverError),
}

impl From<RootBoxError> for ParametricError {
    fn from(error: RootBoxError) -> Self {
        Self::Certification(error.into())
    }
}

fn cover_error(error: ParametricRootCoverError) -> ParametricError {
    match error {
        // Preserve the same cancellation and budget classification as a
        // uniform request; refinement cannot turn an operational refusal
        // into an incomplete mathematical answer or restart its ledger.
        ParametricRootCoverError::Verification(error) => error.into(),
        error => error.into(),
    }
}

fn check_dimensions(
    residuals: usize,
    unknowns: usize,
    parameters: usize,
) -> Result<usize, ParametricError> {
    let total = unknowns
        .checked_add(parameters)
        .ok_or(RootBoxError::DimensionMismatch)?;
    if unknowns == 0 || parameters == 0 || residuals != unknowns || total > MAX_ROOT_BOX_VARIABLES {
        return Err(RootBoxError::DimensionMismatch.into());
    }
    Ok(total)
}

fn owned_text<'a>(
    inputs: impl Iterator<Item = &'a str>,
    count: usize,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<String>, CertificationError> {
    charge(meter, 1, (count * size_of::<String>()) as u64, 1)?;
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(count)
        .map_err(|_| CertificationError::AllocationFailure)?;
    for text in inputs {
        charge(meter, text.len() as u64 + 1, text.len() as u64, 1)?;
        let mut copy = String::new();
        copy.try_reserve_exact(text.len())
            .map_err(|_| CertificationError::AllocationFailure)?;
        copy.push_str(text);
        owned.push(copy);
    }
    Ok(owned)
}

/// A square residual system in nonempty unknown and parameter groups.
///
/// There are at most sixteen total coordinates. Registration retains the
/// original source and checks its bounded envelope; it does not establish
/// differentiability, existence, uniqueness or a valid parameter domain.
/// Same-name symbols with different typed identities remain distinct. No
/// unchecked deserializer or mutable coordinate accessor exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricProblem {
    source: ResidualProblem,
    unknown_count: usize,
}

impl ParametricProblem {
    /// Take ownership without simplifying source domains or rebinding symbols.
    pub fn from_exprs(
        residuals: Vec<Expr>,
        mut unknowns: Vec<Symbol>,
        parameters: Vec<Symbol>,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, ParametricError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let unknown_count = unknowns.len();
        let total = check_dimensions(residuals.len(), unknown_count, parameters.len())?;
        charge(
            meter,
            parameters.len() as u64 + 1,
            (total * size_of::<Symbol>()) as u64,
            1,
        )?;
        unknowns
            .try_reserve_exact(parameters.len())
            .map_err(|_| CertificationError::AllocationFailure)?;
        unknowns.extend(parameters);
        let source = ResidualProblem::new(residuals, unknowns, meter)?;
        Ok(Self {
            source,
            unknown_count,
        })
    }

    /// Parse bounded raw expressions with separate ordered coordinate groups.
    ///
    /// The same parser and source bounds as [`ResidualProblem`] apply. Held
    /// zero factors, cancellation and zero powers retain domain obligations.
    /// Text names denote plain symbols; use [`Self::from_exprs`] for explicit
    /// assumptions/dummy identities. Copying and parsing share the caller's
    /// ledger, and no oversized source/name is copied before admission.
    pub fn parse_sources(
        residuals: &[&str],
        unknowns: &[&str],
        parameters: &[&str],
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, ParametricError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let total = check_dimensions(residuals.len(), unknowns.len(), parameters.len())?;
        for source in residuals {
            if source.len() > MAX_CERTIFICATION_SOURCE_BYTES {
                return Err(CertificationError::SourceLimit("textual residual bytes").into());
            }
        }
        for name in unknowns.iter().chain(parameters) {
            if name.len() > MAX_CERTIFICATION_NAME_BYTES {
                return Err(CertificationError::InvalidVariable.into());
            }
        }
        let residuals = owned_text(residuals.iter().copied(), residuals.len(), meter)?;
        let variables = owned_text(unknowns.iter().chain(parameters).copied(), total, meter)?;
        let source = ResidualProblem::parse_sources(&residuals, &variables, meter)?;
        Ok(Self {
            source,
            unknown_count: unknowns.len(),
        })
    }

    pub fn residuals(&self) -> &[Expr] {
        self.source.residuals()
    }

    pub fn unknowns(&self) -> &[Symbol] {
        &self.source.variables()[..self.unknown_count]
    }

    pub fn parameters(&self) -> &[Symbol] {
        &self.source.variables()[self.unknown_count..]
    }

    /// Independently certify one nonsingular unknown root for every parameter.
    ///
    /// The unknown box requires strict interior inclusion; fixed parameter
    /// coordinates are permitted. The center and preconditioner are untrusted.
    /// The original source is evaluated on the full product domain, including
    /// parameter endpoints and any domains hidden by algebraic cancellation.
    pub fn verify_root_box(
        &self,
        witness: &ParametricRootBoxWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<ParametricRootBoxReceipt, ParametricError> {
        verify_parametric_root_box(
            self.residuals(),
            self.unknowns(),
            self.parameters(),
            witness,
            meter,
        )
        .map_err(Into::into)
    }

    /// Generate a candidate complete parameter cover in the same unknown box.
    ///
    /// Only parameter coordinates are subdivided. Successful generation does
    /// not authorize publication: call [`Self::verify_cover`] with an independent
    /// verifier meter. Exhaustion or an unresolved leaf returns no partial cover.
    pub fn propose_cover(
        &self,
        unknown_domain: &[RealBall],
        parameter_domain: &[RealBall],
        options: ParametricRootCoverOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<ParametricRootCoverWitness, ParametricError> {
        propose_parametric_root_cover(
            self.residuals(),
            self.unknowns(),
            self.parameters(),
            unknown_domain,
            parameter_domain,
            options,
            meter,
        )
        .map_err(cover_error)
    }

    /// Independently check every leaf and exact coverage of the parameter box.
    ///
    /// Every leaf must retain the original unknown box and source/coordinate
    /// groups. Partial coverage, repeated references, source holes and invalid
    /// local witnesses cannot produce an accepted family or sensitivity input.
    pub fn verify_cover(
        &self,
        unknown_domain: &[RealBall],
        parameter_domain: &[RealBall],
        witness: &ParametricRootCoverWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<ParametricRootCoverReceipt, ParametricError> {
        verify_parametric_root_cover(
            self.residuals(),
            self.unknowns(),
            self.parameters(),
            unknown_domain,
            parameter_domain,
            witness,
            meter,
        )
        .map_err(cover_error)
    }
}
