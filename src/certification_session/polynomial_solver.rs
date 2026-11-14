//! Exact finite real solution tuples from independently checked polynomial ideals.
//!
//! A separating parameter identifies every distinct real tuple without replacing
//! an algebraic number by a floating approximation. The underlying RUR receipt
//! checks both ideal inclusions, complete real roots and all coordinate maps.
//! Its internal parameter and possible radical reduction do not change the
//! original ordered variables or assert multiplicities of the original system.

use super::real_solver::LedgerView;
use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError, ExactInterval,
    ResidualSystemId, admit_response, charge, render_scalar,
};
use crate::certification::ResidualProblem;
use fsym_budget::{Budget, BudgetError, BudgetMeter, Dimension, MeterError, VerifierLease};
use fsym_core::{BigRational, Symbol};
use fsym_solvers::real_rur::{VerifiedRealRurSystem, metered_solve_real_rur_system};
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::mem::size_of;

/// Complete distinct real tuples, represented through one algebraic parameter.
///
/// `polynomial` is a nonzero monic polynomial with ascending rational
/// coefficients. `roots` isolates every distinct real root in increasing order;
/// an interval represents that exact root, not all points inside the interval.
/// For each root `t`, evaluate `coordinates[i]` at `t` to obtain coordinate `i`
/// in the response's original `variables` order. Coordinate coefficients also
/// use ascending degree. Every resulting tuple is a solution, every real
/// solution occurs once, and empty `roots` means no real solutions.
///
/// The polynomial can have repeated roots. Their multiplicity is deliberately
/// not a multiplicity claim about the original ideal: a checked radical can be
/// used internally while preserving the complete distinct solution set. This
/// description is an output, not an independently replayable certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExactRealPolynomialSolutionSet {
    pub parameter: Symbol,
    pub polynomial: Vec<String>,
    pub roots: Vec<ExactInterval>,
    pub coordinates: Vec<Vec<String>>,
}

/// The older polynomial solver carries refusals as `SolverError` strings.
/// Retain an actual caller-meter failure separately so no string matching can
/// turn cancellation or exhausted verifier fuel into mathematical refusal.
/// Once failed, both views are terminal even if a future generator retries.
struct RecordedMeter<'a, M> {
    inner: &'a mut M,
    failure: &'a Cell<Option<MeterError>>,
}

impl<M: BudgetMeter> RecordedMeter<'_, M> {
    fn remember(&self, result: Result<(), MeterError>) -> Result<(), MeterError> {
        if let Err(error) = result {
            self.failure.set(Some(error));
        }
        result
    }
}

impl<M: BudgetMeter> BudgetMeter for RecordedMeter<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if let Some(error) = self.failure.get() {
            return Err(error);
        }
        let result = self.inner.charge_batch(charges);
        self.remember(result)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if let Some(error) = self.failure.get() {
            return Err(error);
        }
        let result = self.inner.checkpoint();
        self.remember(result)
    }
}

fn checked_solution(
    problem: &ResidualProblem,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedRealRurSystem, CertificationSessionError> {
    let failure = Cell::new(None);
    let result = metered_solve_real_rur_system(
        problem.residuals(),
        problem.variables(),
        &mut RecordedMeter {
            inner: generator,
            failure: &failure,
        },
        &mut RecordedMeter {
            inner: verifier,
            failure: &failure,
        },
    );
    if let Some(error) = failure.get() {
        return Err(error.into());
    }
    result?.ok_or(CertificationSessionError::UnsupportedRealPolynomialSystem)
}

fn coefficients(
    values: &[BigRational],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<String>, CertificationSessionError> {
    charge(meter, values.len(), values.len() * size_of::<String>())?;
    values
        .iter()
        .map(|value| render_scalar(value, meter))
        .collect()
}

pub(super) fn solve(
    system: ResidualSystemId,
    problem: &ResidualProblem,
    context_digest: [u8; 32],
    budget: &mut Budget,
    lease: Option<&VerifierLease>,
) -> Result<CertificationResponse, CertificationSessionError> {
    if !(1..=8).contains(&problem.variables().len()) || problem.residuals().len() > 64 {
        return Err(CertificationSessionError::InvalidRealPolynomialProblem);
    }
    let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
    budget
        .verifier_meter(lease)
        .map_err(MeterError::from)?
        .checkpoint()?;
    let receipt = {
        let ledger = RefCell::new(&mut *budget);
        checked_solution(
            problem,
            &mut LedgerView {
                ledger: &ledger,
                verifier: None,
            },
            &mut LedgerView {
                ledger: &ledger,
                verifier: Some(lease),
            },
        )?
    };
    // Receipt construction is private to the independent native acceptance
    // path. Original source admission precedes unit-ideal and shape shortcuts.
    let polynomial = coefficients(receipt.eliminant(), budget)?;
    charge(
        budget,
        receipt.roots().intervals().len(),
        receipt.roots().intervals().len() * size_of::<ExactInterval>(),
    )?;
    let roots = receipt
        .roots()
        .intervals()
        .iter()
        .map(|root| {
            Ok(ExactInterval {
                lower: render_scalar(&root.lower, budget)?,
                upper: render_scalar(&root.upper, budget)?,
            })
        })
        .collect::<Result<_, CertificationSessionError>>()?;
    charge(
        budget,
        receipt.coordinates().len(),
        receipt.coordinates().len() * size_of::<Vec<String>>(),
    )?;
    let coordinates = receipt
        .coordinates()
        .iter()
        .map(|coordinate| coefficients(coordinate, budget))
        .collect::<Result<_, _>>()?;
    charge(
        budget,
        receipt.variables().len() + 1,
        (receipt.variables().len() + 1) * size_of::<Symbol>()
            + receipt.parameter().name.len()
            + receipt
                .variables()
                .iter()
                .map(|axis| axis.name.len())
                .sum::<usize>(),
    )?;
    let response = CertificationResponse::RealPolynomialSystemSolved {
        schema_version: CERTIFICATION_REQUEST_SCHEMA,
        system,
        variables: receipt.variables().to_vec(),
        evidence: "exact_cross_checked",
        claim: "complete_distinct_real_solution_set_of_original_polynomial_system",
        completeness: "complete",
        domain: "real",
        solution_set: ExactRealPolynomialSolutionSet {
            parameter: receipt.parameter().clone(),
            polynomial,
            roots,
            coordinates,
        },
        context_digest,
    };
    // No root, coordinate or claimed tuple escapes until the entire response
    // has been admitted. Failure changes only cumulative resource consumption.
    admit_response(&response, budget)?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{BudgetLimits, Unbounded};
    use fsym_core::{Expr, SymbolIdentity};

    struct CancelAt {
        remaining: usize,
    }

    impl BudgetMeter for CancelAt {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }

        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.remaining == 0 {
                Err(MeterError::Cancelled)
            } else {
                self.remaining -= 1;
                Ok(())
            }
        }
    }

    fn problem() -> ResidualProblem {
        ResidualProblem::parse_sources(
            &["x-y".into(), "y^2-2".into()],
            &["x".into(), "y".into()],
            &mut Unbounded,
        )
        .unwrap()
    }

    #[test]
    fn cancellation_in_either_lane_remains_typed_through_solver_errors() {
        let problem = problem();
        for remaining in [0, 1, 20] {
            for cancel_verifier in [false, true] {
                let mut cancelled = CancelAt { remaining };
                let result = if cancel_verifier {
                    checked_solution(&problem, &mut Unbounded, &mut cancelled)
                } else {
                    checked_solution(&problem, &mut cancelled, &mut Unbounded)
                };
                assert!(matches!(
                    result,
                    Err(CertificationSessionError::Meter(MeterError::Cancelled))
                ));
            }
        }
    }

    #[test]
    fn recorded_failure_is_terminal_and_shared_across_both_views() {
        let failure = Cell::new(None);
        let mut cancelled = CancelAt { remaining: 0 };
        let mut ordinary = Budget::new(BudgetLimits::uniform(100, 0));
        let before = ordinary.snapshot();
        let mut first = RecordedMeter {
            inner: &mut cancelled,
            failure: &failure,
        };
        assert_eq!(first.checkpoint(), Err(MeterError::Cancelled));
        let mut second = RecordedMeter {
            inner: &mut ordinary,
            failure: &failure,
        };
        assert_eq!(
            second.charge(Dimension::ComputeSteps, 1),
            Err(MeterError::Cancelled)
        );
        assert_eq!(ordinary.snapshot(), before);
    }

    #[test]
    fn missing_or_foreign_verifier_authority_precedes_all_generation() {
        let problem = problem();
        let mut budget = Budget::new(BudgetLimits::uniform(100_000, 100_000));
        let mut other = Budget::new(BudgetLimits::uniform(100_000, 100_000));
        let foreign = other.verifier_lease().unwrap();
        let before = budget.snapshot();
        for lease in [None, Some(&foreign)] {
            let result = solve(
                ResidualSystemId::new("system:polynomial").unwrap(),
                &problem,
                [0; 32],
                &mut budget,
                lease,
            );
            assert!(matches!(
                result,
                Err(CertificationSessionError::Meter(MeterError::Budget(
                    BudgetError::VerifierPoolAccessDenied
                )))
            ));
            assert_eq!(budget.snapshot(), before);
        }
    }

    #[test]
    fn same_print_axes_keep_distinct_typed_identities_through_the_response() {
        let first = Symbol::new("same");
        let second =
            Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([81; 32]));
        let x = Expr::Sym(first.clone());
        let y = Expr::Sym(second.clone());
        let problem = ResidualProblem::new(
            vec![x.clone() * x.clone() - Expr::from_i64(2), y - x],
            vec![first.clone(), second.clone()],
            &mut Unbounded,
        )
        .unwrap();
        let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 100_000_000));
        let lease = budget.verifier_lease().unwrap();
        let response = solve(
            ResidualSystemId::new("system:typed").unwrap(),
            &problem,
            [43; 32],
            &mut budget,
            Some(&lease),
        )
        .unwrap();
        let CertificationResponse::RealPolynomialSystemSolved {
            variables,
            solution_set,
            context_digest,
            ..
        } = response
        else {
            panic!("expected typed polynomial solution")
        };
        assert_eq!(variables, [first, second.clone()]);
        assert_eq!(solution_set.parameter, second);
        assert_eq!(solution_set.coordinates, [vec!["0", "1"], vec!["0", "1"]]);
        assert_eq!(solution_set.roots.len(), 2);
        assert_eq!(context_digest, [43; 32]);
    }
}
