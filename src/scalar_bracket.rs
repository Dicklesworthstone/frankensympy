//! Source-bound scalar root existence and portable exact sign brackets.
//!
//! The generator supplies only an exact subinterval. The independent core
//! checker establishes continuity on the complete original interval, containment,
//! requested width, and opposite endpoint signs or an exact endpoint zero.
//! Repeated and nonsmooth roots are allowed; uniqueness, multiplicity and complete
//! enumeration are separate claims that this receipt never grants.

#![forbid(unsafe_code)]

use crate::certification::{CertificationError, ResidualProblem};
use fsym_budget::BudgetMeter;
use fsym_calculus::scalar_bracket::{ScalarRootBracketOptions, propose_scalar_root_bracket};
use fsym_core::root_certificate::scalar_bracket_artifact;
use fsym_core::{
    BigRational, Expr, ScalarRootBracketReceipt, ScalarRootBracketWitness, Symbol,
    verify_scalar_root_bracket,
};

// Native Sessions and host bridges share one source-bound canonical codec.
// The product API and version-1 wire format remain unchanged.
pub use scalar_bracket_artifact::{
    MAX_SCALAR_ROOT_BRACKET_BYTES, SCALAR_ROOT_BRACKET_SCHEMA_VERSION, ScalarBracketError,
};

fn scalar_source(problem: &ResidualProblem) -> Result<(&Expr, &Symbol), ScalarBracketError> {
    if problem.residuals().len() != 1 || problem.variables().len() != 1 {
        return Err(ScalarBracketError::ProblemDimension);
    }
    Ok((&problem.residuals()[0], &problem.variables()[0]))
}

impl ResidualProblem {
    /// Propose a bounded continuous scalar sign bracket on ordinary resources.
    ///
    /// Endpoints and width tolerance are exact rationals. The complete original
    /// source must be continuous on the requested ordered closed interval, even
    /// when a narrow candidate would avoid a source pole. Equal endpoints still
    /// require an exactly verified zero. A proposed witness is not an accepted
    /// root; use [`Self::verify_scalar_root_bracket`] with an independent meter.
    pub fn propose_scalar_root_bracket(
        &self,
        lower: &BigRational,
        upper: &BigRational,
        absolute_tolerance: &BigRational,
        options: ScalarRootBracketOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<ScalarRootBracketWitness, ScalarBracketError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let (source, variable) = scalar_source(self)?;
        Ok(propose_scalar_root_bracket(
            source,
            variable,
            lower,
            upper,
            absolute_tolerance,
            options,
            meter,
        )?)
    }

    /// Independently verify at least one real zero in the proposed subinterval.
    ///
    /// No supplied function value, derivative or acceptance flag is trusted.
    /// This check supports odd repeated and continuous nonsmooth roots without
    /// asserting a nonzero derivative, uniqueness or a complete root set. The
    /// caller can pass a protected verifier meter from the proposal's cumulative
    /// ledger; no failure resets that allowance or invokes another solver.
    pub fn verify_scalar_root_bracket(
        &self,
        lower: &BigRational,
        upper: &BigRational,
        absolute_tolerance: &BigRational,
        witness: &ScalarRootBracketWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<ScalarRootBracketReceipt, ScalarBracketError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let (source, variable) = scalar_source(self)?;
        Ok(verify_scalar_root_bracket(
            source,
            variable,
            lower,
            upper,
            absolute_tolerance,
            witness,
            meter,
        )?)
    }
}

/// Export an accepted bracket's original statement and untrusted witness.
///
/// The artifact contains no root value, enclosure, endpoint evaluations or
/// accepted flag. Raw source structure, typed variable identity, ordered request
/// endpoints, exact tolerance and context are bound. Receiving hosts independently
/// supply the expected statement and re-evaluate the source before accepting it.
pub fn export_scalar_root_bracket(
    problem: &ResidualProblem,
    receipt: &ScalarRootBracketReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, ScalarBracketError> {
    scalar_bracket_artifact::export_scalar_root_bracket(
        problem.residuals(),
        problem.variables(),
        receipt,
        context_digest,
        meter,
    )
}

/// Replay a portable bracket against the receiver's exact original statement.
///
/// Byte/depth limits and cumulative decoder costs precede borrowed parsing.
/// Source, coordinate, interval, tolerance and context mismatches refuse before
/// witness construction. Canonical re-encoding rejects alternative encodings,
/// extra fields and cached evidence. Only the independent core continuity and
/// endpoint-sign checker can grant a receipt; replay never runs candidate search.
#[allow(clippy::too_many_arguments)]
pub fn replay_scalar_root_bracket(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_lower: &BigRational,
    expected_upper: &BigRational,
    expected_tolerance: &BigRational,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<ScalarRootBracketReceipt, ScalarBracketError> {
    scalar_bracket_artifact::replay_scalar_root_bracket(
        bytes,
        expected_problem.residuals(),
        expected_problem.variables(),
        expected_lower,
        expected_upper,
        expected_tolerance,
        expected_context,
        meter,
    )
}
