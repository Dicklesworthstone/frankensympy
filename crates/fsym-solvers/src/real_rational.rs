//! Complete finite real rational systems with the original pole set retained.
//!
//! Rational source localization and complete real univariate representation
//! already have separate independent checkers. This facade composes those
//! receipts instead of cancelling denominators before solving. The checked
//! equation `u * product(original guards) = 1` gives every original solution
//! exactly one auxiliary coordinate, so projecting that coordinate away is a
//! bijection, including when the complete real solution set is empty.
//!
//! The accepted value retains both receipts. It is an in-process exact
//! cross-check, not a portable proof, a multiplicity statement about the
//! original ideal, or a complete complex solution set. Positive-dimensional
//! and unrecognized finite systems remain `None`; resource, cancellation and
//! verification failures are errors. Both phases consume the same caller
//! meters and retain their existing intrinsic bounds.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::real_roots::VerifiedRealRoots;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{
    RationalSystemAdmission, VerifiedRationalLocalization, metered_localize_rational_system,
};

use crate::SolverError;
use crate::real_rur::{VerifiedRealRurSystem, metered_solve_real_rur_system};

/// A complete distinct real solution set in the original coordinate order.
///
/// Every indexed real root of `eliminant()` gives one solution by evaluating
/// `coordinates()[i]` at that root. The original pole restrictions remain in
/// `localization()` when an auxiliary equation was needed. The auxiliary can
/// serve as the internal RUR parameter, but is never an additional returned
/// solution coordinate.
#[derive(Debug, Clone)]
pub struct VerifiedRealRationalSystem {
    localization: Option<Box<VerifiedRationalLocalization>>,
    representation: VerifiedRealRurSystem,
    // Only `accept` constructs this type, after checking both backing slice
    // lengths and the exact typed prefix. No public method can mutate them.
    original_width: usize,
}

impl VerifiedRealRationalSystem {
    pub fn variables(&self) -> &[Symbol] {
        &self.representation.variables()[..self.original_width]
    }

    pub fn coordinates(&self) -> &[Vec<BigRational>] {
        &self.representation.coordinates()[..self.original_width]
    }

    pub fn parameter(&self) -> &Symbol {
        self.representation.parameter()
    }

    pub fn eliminant(&self) -> &[BigRational] {
        self.representation.eliminant()
    }

    pub fn roots(&self) -> &VerifiedRealRoots {
        self.representation.roots()
    }

    pub fn localization(&self) -> Option<&VerifiedRationalLocalization> {
        self.localization.as_deref()
    }

    /// The complete underlying polynomial receipt. Its variables and
    /// coordinates include the auxiliary when localization was required;
    /// callers publishing original solutions use `variables`/`coordinates`.
    pub fn representation(&self) -> &VerifiedRealRurSystem {
        &self.representation
    }

    pub fn all_complex_roots_real(&self) -> bool {
        self.representation.all_complex_roots_real()
    }
}

fn refuse(message: impl std::fmt::Display) -> SolverError {
    SolverError::IncompleteSolutionSet(format!("real rational system refused: {message}"))
}

fn tick(meter: &mut impl BudgetMeter, steps: u64) -> Result<(), SolverError> {
    meter.checkpoint().map_err(refuse)?;
    meter
        .charge(Dimension::ComputeSteps, steps.max(1))
        .map_err(refuse)
}

fn bind_variables(
    actual: &[Symbol],
    expected: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<(), SolverError> {
    tick(meter, 1)?;
    if actual.len() != expected.len() {
        return Err(refuse("accepted coordinate count changed"));
    }
    for (actual, expected) in actual.iter().zip(expected) {
        tick(
            meter,
            actual.name.len() as u64 + expected.name.len() as u64 + 128,
        )?;
        if actual != expected {
            return Err(refuse("accepted typed coordinate order changed"));
        }
    }
    Ok(())
}

/// Bind the two independently accepted theorems to exactly the same
/// augmented equations. Matching only variable names or a root count would
/// allow an unrelated complete polynomial solution set to be projected.
fn bind_localization(
    localization: &VerifiedRationalLocalization,
    representation: &VerifiedRealRurSystem,
    meter: &mut impl BudgetMeter,
) -> Result<(), SolverError> {
    bind_variables(representation.variables(), localization.variables(), meter)?;
    let expected = localization.polynomials();
    let actual = representation.equations();
    tick(meter, 1)?;
    if actual.len() != expected.len() {
        return Err(refuse("accepted localization equation count changed"));
    }
    for (actual, expected) in actual.iter().zip(expected) {
        bind_variables(actual.generators(), expected.generators(), meter)?;
        tick(meter, 1)?;
        if actual.terms().len() != expected.terms().len() {
            return Err(refuse("accepted localization equation changed"));
        }
        for ((actual_power, actual_value), (expected_power, expected_value)) in
            actual.terms().iter().zip(expected.terms())
        {
            tick(
                meter,
                actual_power.len() as u64
                    + expected_power.len() as u64
                    + actual_value.numer().limb_count()
                    + actual_value.denom().limb_count()
                    + expected_value.numer().limb_count()
                    + expected_value.denom().limb_count()
                    + 1,
            )?;
            if actual_power != expected_power || actual_value != expected_value {
                return Err(refuse("accepted localization equation changed"));
            }
        }
    }
    meter.checkpoint().map_err(refuse)
}

fn accept(
    variables: &[Symbol],
    localization: Option<Box<VerifiedRationalLocalization>>,
    representation: VerifiedRealRurSystem,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRealRationalSystem, SolverError> {
    tick(verifier, 1)?;
    let expected_width = variables.len() + usize::from(localization.is_some());
    if representation.variables().len() != expected_width
        || representation.coordinates().len() != expected_width
    {
        return Err(refuse("accepted solution projection has an invalid width"));
    }
    let original = representation
        .variables()
        .get(..variables.len())
        .ok_or_else(|| refuse("accepted solution is missing original coordinates"))?;
    bind_variables(original, variables, verifier)?;
    if let Some(localization) = localization.as_deref() {
        bind_localization(localization, &representation, verifier)?;
    }
    generator.checkpoint().map_err(refuse)?;
    verifier.checkpoint().map_err(refuse)?;
    Ok(VerifiedRealRationalSystem {
        localization,
        representation,
        original_width: variables.len(),
    })
}

/// Solve a bounded finite rational system over the reals without losing any
/// original denominator obligation.
///
/// Polynomial inputs retain the RUR limit of eight variables and 64
/// equations. Variable-denominator inputs reserve one variable/equation for
/// the checked localization, hence admit at most seven variables and 63
/// original equations. Every original source is admitted even behind zero
/// products, zero powers or an inconsistent equation. All coordinate maps
/// use ascending rational coefficients and the complete real root list.
///
/// The existing localization and RUR intrinsic resource envelopes still
/// apply, in addition to these cumulative caller meters. Neither recognition
/// misses nor failures replenish a meter or authorize a partial solution set.
pub fn metered_solve_real_rational_system(
    sources: &[Expr],
    variables: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealRationalSystem>, SolverError> {
    generator_meter.checkpoint().map_err(refuse)?;
    verifier_meter.checkpoint().map_err(refuse)?;
    if variables.is_empty() || variables.len() > 8 || sources.len() > 64 {
        return Err(SolverError::InvalidSystem(
            "real rational systems require 1..8 variables and at most 64 equations; localization reserves one of each".into(),
        ));
    }
    let localization =
        match metered_localize_rational_system(sources, variables, generator_meter, verifier_meter)
            .map_err(refuse)?
        {
            RationalSystemAdmission::Polynomial => None,
            RationalSystemAdmission::Localized(receipt) => Some(receipt),
        };
    let representation = match localization.as_deref() {
        Some(receipt) => metered_solve_real_rur_system(
            receipt.equations(),
            receipt.variables(),
            generator_meter,
            verifier_meter,
        )?,
        // This is only a routing hint. RUR must still admit every original
        // source before its zero, unit-ideal or dimension shortcuts.
        None => metered_solve_real_rur_system(sources, variables, generator_meter, verifier_meter)?,
    };
    let Some(representation) = representation else {
        generator_meter.checkpoint().map_err(refuse)?;
        verifier_meter.checkpoint().map_err(refuse)?;
        return Ok(None);
    };
    accept(
        variables,
        localization,
        representation,
        generator_meter,
        verifier_meter,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::Unbounded;

    #[test]
    fn unrelated_complete_receipt_cannot_authorize_a_projection() {
        let variables = [Symbol::new("x")];
        let sources = [fsym_core::parse("(x^2-1)/(x-1)").unwrap()];
        let RationalSystemAdmission::Localized(localization) =
            metered_localize_rational_system(&sources, &variables, &mut Unbounded, &mut Unbounded)
                .unwrap()
        else {
            panic!("variable denominator must localize")
        };
        let mut unrelated = localization.equations().to_vec();
        // Same ring, equation count and monomial support; only an exact
        // coefficient changes. Structural shape alone cannot bind the claim.
        unrelated[0] = fsym_core::parse("x^2-4").unwrap();
        let representation = metered_solve_real_rur_system(
            &unrelated,
            localization.variables(),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap()
        .unwrap();
        assert!(
            accept(
                &variables,
                Some(localization),
                representation,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }

    #[test]
    fn a_valid_augmented_receipt_cannot_expose_the_auxiliary_as_an_original_axis() {
        let variables = [Symbol::new("x")];
        let sources = [fsym_core::parse("(x^2-1)/(x-1)").unwrap()];
        let accepted = metered_solve_real_rational_system(
            &sources,
            &variables,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap()
        .unwrap();
        let augmented = accepted.representation.variables().to_vec();
        assert!(
            accept(
                &augmented,
                accepted.localization,
                accepted.representation,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }

    #[test]
    fn a_valid_polynomial_receipt_cannot_rebind_the_original_axis_identity() {
        let variables = [Symbol::new("x")];
        let representation = metered_solve_real_rur_system(
            &[fsym_core::parse("x-1").unwrap()],
            &variables,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap()
        .unwrap();
        let other = Symbol::with_identity(
            "x",
            fsym_core::SymbolIdentity::from_assumptions_digest([92; 32]),
        );
        assert!(
            accept(
                &[other],
                None,
                representation,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}
