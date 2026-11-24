//! Source-bound proper real definite integration and portable partition witnesses.
//!
//! The adaptive generator proposes only a partition. The independent core
//! checker re-evaluates every original source child on every closed cell,
//! establishes complete coverage, and checks the absolute enclosure width.
//! Neither a symbolic antiderivative nor an exported acceptance flag is trusted.

#![forbid(unsafe_code)]

use crate::certification::{CertificationError, ResidualProblem};
use fsym_budget::BudgetMeter;
use fsym_calculus::quadrature::{QuadratureOptions, propose_integral_enclosure};
use fsym_core::root_certificate::integral_artifact;
use fsym_core::{
    BigRational, Expr, IntegralEnclosureReceipt, QuadratureWitness, Symbol,
    verify_integral_enclosure,
};

// One bounded canonical codec serves native Sessions and host bridges. The
// public native API, schema, source binding and mathematical replay are retained.
pub use integral_artifact::{
    INTEGRAL_CERTIFICATE_SCHEMA_VERSION, IntegrationError, MAX_INTEGRAL_CERTIFICATE_BYTES,
};

fn integral_source(problem: &ResidualProblem) -> Result<(&Expr, &Symbol), IntegrationError> {
    if problem.residuals().len() != 1 || problem.variables().len() != 1 {
        return Err(IntegrationError::ProblemDimension);
    }
    Ok((&problem.residuals()[0], &problem.variables()[0]))
}

impl ResidualProblem {
    /// Propose a proper-real integral partition using ordinary work resources.
    ///
    /// This bounded lane requires one original residual and one typed axis.
    /// Endpoints and tolerance are exact rationals; every original source
    /// domain obligation is retained. A returned partition is only a candidate.
    /// Improper integrals, complex contours and principal values are outside
    /// this claim. Unsupported sources or exhausted resources refuse.
    pub fn propose_integral(
        &self,
        lower: &BigRational,
        upper: &BigRational,
        absolute_tolerance: &BigRational,
        options: QuadratureOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<QuadratureWitness, IntegrationError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let (source, variable) = integral_source(self)?;
        Ok(propose_integral_enclosure(
            source,
            variable,
            lower,
            upper,
            absolute_tolerance,
            options,
            meter,
        )?)
    }

    /// Independently certify complete coverage and the exact enclosure width.
    ///
    /// The caller may supply a protected verifier meter from the same ledger
    /// used for proposal. Reversed limits negate the accepted enclosure; equal
    /// limits still require the original source to be admitted at the point.
    pub fn verify_integral(
        &self,
        lower: &BigRational,
        upper: &BigRational,
        absolute_tolerance: &BigRational,
        witness: &QuadratureWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<IntegralEnclosureReceipt, IntegrationError> {
        meter.checkpoint().map_err(CertificationError::from)?;
        let (source, variable) = integral_source(self)?;
        Ok(verify_integral_enclosure(
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

/// Export an accepted integral's exact statement and untrusted partition.
///
/// The format contains no enclosure, cached value or accepted flag. The source
/// must match the supplied problem exactly, including raw operand order and
/// symbol identity. The receiver supplies its own expected ordered endpoints,
/// tolerance and assumption context, then independently replays the witness.
pub fn export_integral_certificate(
    problem: &ResidualProblem,
    receipt: &IntegralEnclosureReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, IntegrationError> {
    integral_artifact::export_integral_certificate(
        problem.residuals(),
        problem.variables(),
        receipt,
        context_digest,
        meter,
    )
}

/// Replay the shared canonical partition using independent source verification.
#[allow(clippy::too_many_arguments)]
pub fn replay_integral_certificate(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_lower: &BigRational,
    expected_upper: &BigRational,
    expected_tolerance: &BigRational,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<IntegralEnclosureReceipt, IntegrationError> {
    integral_artifact::replay_integral_certificate(
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
