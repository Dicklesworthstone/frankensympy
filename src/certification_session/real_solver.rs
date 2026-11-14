//! Complete real solution sets from frozen, original rational sources.
//!
//! The existing solver independently checks polynomial reductions, complete
//! real-root isolation, and sign-chart decisions. Its two synchronous meters
//! borrow one session ledger for each atomic operation: generator work cannot
//! spend the protected verifier pool, and switching lanes never resets fuel.
//! The response describes exact indexed algebraic points and open gaps. It
//! does not turn rational isolating bounds into approximate solution points.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError, ExactInterval,
    ResidualSystemId, admit_response, charge, render_scalar,
};
use crate::certification::ResidualProblem;
use fsym_budget::{Budget, BudgetError, BudgetMeter, Dimension, MeterError, VerifierLease};
use fsym_core::real_roots::{RealRootError, VerifiedRealRoots};
use fsym_solvers::inequality::IneqOp;
use fsym_solvers::real_equation::{
    RealEquationError, VerifiedRealEquation, VerifiedRealInequality, metered_solve_real_equation,
    metered_solve_real_inequality,
};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::mem::size_of;

/// Comparison of the registered original residual with zero over the reals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealInequalityRelation {
    Lt,
    Le,
    Gt,
    Ge,
}

impl From<RealInequalityRelation> for IneqOp {
    fn from(relation: RealInequalityRelation) -> Self {
        match relation {
            RealInequalityRelation::Lt => Self::Lt,
            RealInequalityRelation::Le => Self::Le,
            RealInequalityRelation::Gt => Self::Gt,
            RealInequalityRelation::Ge => Self::Ge,
        }
    }
}

/// An exact real set encoded by a complete algebraic sign decomposition.
///
/// `polynomial` lists rational coefficients in ascending degree. It is monic,
/// nonzero and square-free. `roots[i]` isolates its i-th distinct real root;
/// it denotes that exact algebraic point, not the interval's whole contents.
/// A point interval denotes an exact rational root. There are no omitted roots.
/// `point_inclusions[i]` decides membership of that root. `gap_inclusions`
/// decides the open gaps in order, including both unbounded outer gaps, and
/// has one more entry than `roots`. With no roots, its sole entry decides
/// membership of the entire real line. Constant one is the empty root basis.
///
/// This is an output description, not an independently replayable proof or an
/// unchecked constructor for a native verified solver receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExactRealSolutionSet {
    pub polynomial: Vec<String>,
    pub roots: Vec<ExactInterval>,
    pub gap_inclusions: Vec<bool>,
    pub point_inclusions: Vec<bool>,
}

/// The backend interleaves generation and reference checks, but never runs
/// them concurrently or calls arbitrary user code. Each borrow lasts only
/// for one charge/checkpoint. A failed borrow refuses access rather than
/// panicking or falling back to a fresh/unbounded meter.
struct LedgerView<'a, 'budget> {
    ledger: &'a RefCell<&'budget mut Budget>,
    verifier: Option<&'a VerifierLease>,
}

impl BudgetMeter for LedgerView<'_, '_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let mut ledger = self
            .ledger
            .try_borrow_mut()
            .map_err(|_| MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        match self.verifier {
            Some(lease) => ledger.verifier_meter(lease)?.charge_batch(charges),
            None => ledger.charge_batch(charges),
        }
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let mut ledger = self
            .ledger
            .try_borrow_mut()
            .map_err(|_| MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        match self.verifier {
            Some(lease) => ledger.verifier_meter(lease)?.checkpoint(),
            None => ledger.checkpoint(),
        }
    }
}

enum Accepted {
    Equation(VerifiedRealEquation),
    Inequality(VerifiedRealInequality),
}

fn root_basis(
    roots: &VerifiedRealRoots,
    meter: &mut impl BudgetMeter,
) -> Result<(Vec<String>, Vec<ExactInterval>), CertificationSessionError> {
    charge(
        meter,
        roots.coefficients().len(),
        roots.coefficients().len() * size_of::<String>(),
    )?;
    let polynomial = roots
        .coefficients()
        .iter()
        .map(|value| render_scalar(value, meter))
        .collect::<Result<_, _>>()?;
    charge(
        meter,
        roots.intervals().len(),
        roots.intervals().len() * size_of::<ExactInterval>(),
    )?;
    let intervals = roots
        .intervals()
        .iter()
        .map(|root| {
            Ok(ExactInterval {
                lower: render_scalar(&root.lower, meter)?,
                upper: render_scalar(&root.upper, meter)?,
            })
        })
        .collect::<Result<_, CertificationSessionError>>()?;
    Ok((polynomial, intervals))
}

pub(super) fn solve(
    system: ResidualSystemId,
    problem: &ResidualProblem,
    relation: Option<RealInequalityRelation>,
    context_digest: [u8; 32],
    budget: &mut Budget,
    lease: Option<&VerifierLease>,
) -> Result<CertificationResponse, CertificationSessionError> {
    if problem.residuals().len() != 1 || problem.variables().len() != 1 {
        return Err(CertificationSessionError::InvalidRealSolverProblem);
    }
    let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
    // Check live ledger authority before any generator or response work.
    budget
        .verifier_meter(lease)
        .map_err(MeterError::from)?
        .checkpoint()?;
    charge(budget, 1, 4096)?;
    let accepted = {
        let ledger = RefCell::new(&mut *budget);
        let mut generator = LedgerView {
            ledger: &ledger,
            verifier: None,
        };
        let mut verifier = LedgerView {
            ledger: &ledger,
            verifier: Some(lease),
        };
        let expression = &problem.residuals()[0];
        let variable = &problem.variables()[0];
        match relation {
            None => Accepted::Equation(
                metered_solve_real_equation(expression, variable, &mut generator, &mut verifier)?
                    .ok_or(CertificationSessionError::UnsupportedRealSolverSource)?,
            ),
            Some(relation) => Accepted::Inequality(
                metered_solve_real_inequality(
                    expression,
                    variable,
                    relation.into(),
                    &mut generator,
                    &mut verifier,
                )?
                .ok_or(CertificationSessionError::UnsupportedRealSolverSource)?,
            ),
        }
    };
    charge(
        budget,
        1,
        problem.variables()[0].name.len() + size_of::<fsym_core::Symbol>(),
    )?;
    let variable = problem.variables()[0].clone();
    let response = match accepted {
        Accepted::Equation(receipt) => {
            let (polynomial, roots) = root_basis(receipt.roots(), budget)?;
            let count = roots.len();
            charge(budget, count + 1, count + 1)?;
            let gap_inclusions = vec![receipt.is_complement(); count + 1];
            charge(budget, count, count)?;
            let point_inclusions = vec![!receipt.is_complement(); count];
            CertificationResponse::RealEquationSolved {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system,
                variable,
                evidence: "exact_cross_checked",
                claim: "complete_real_solution_set_of_original_rational_equation",
                completeness: "complete",
                domain: "real",
                solution_set: ExactRealSolutionSet {
                    polynomial,
                    roots,
                    gap_inclusions,
                    point_inclusions,
                },
                context_digest,
            }
        }
        Accepted::Inequality(receipt) => {
            let (polynomial, roots) = root_basis(receipt.roots(), budget)?;
            charge(
                budget,
                receipt.gap_inclusions().len(),
                receipt.gap_inclusions().len(),
            )?;
            let gap_inclusions = receipt.gap_inclusions().to_vec();
            charge(
                budget,
                receipt.point_inclusions().len(),
                receipt.point_inclusions().len(),
            )?;
            let point_inclusions = receipt.point_inclusions().to_vec();
            let relation = match receipt.operation() {
                IneqOp::Lt => RealInequalityRelation::Lt,
                IneqOp::Le => RealInequalityRelation::Le,
                IneqOp::Gt => RealInequalityRelation::Gt,
                IneqOp::Ge => RealInequalityRelation::Ge,
            };
            CertificationResponse::RealInequalitySolved {
                schema_version: CERTIFICATION_REQUEST_SCHEMA,
                system,
                variable,
                relation,
                evidence: "exact_cross_checked",
                claim: "complete_real_solution_set_of_original_rational_inequality",
                completeness: "complete",
                domain: "real",
                solution_set: ExactRealSolutionSet {
                    polynomial,
                    roots,
                    gap_inclusions,
                    point_inclusions,
                },
                context_digest,
            }
        }
    };
    // No partially rendered set or accepted claim escapes on output failure.
    admit_response(&response, budget)?;
    Ok(response)
}

pub(super) fn error_code(error: &RealEquationError) -> &'static str {
    match error {
        RealEquationError::Meter(MeterError::Cancelled)
        | RealEquationError::Roots(RealRootError::Meter(MeterError::Cancelled)) => "cancelled",
        RealEquationError::Meter(MeterError::Budget(_))
        | RealEquationError::Roots(RealRootError::Meter(MeterError::Budget(_))) => {
            "budget_exhausted"
        }
        RealEquationError::ResourceLimit(_)
        | RealEquationError::Roots(RealRootError::ResourceLimit(_)) => "resource_limit_exceeded",
        RealEquationError::InvalidReduction(_) | RealEquationError::Roots(_) => "refused",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{BudgetLimits, Unbounded};

    #[test]
    fn scoped_lanes_keep_atomic_shared_charges_and_disjoint_compute_pools() {
        let mut limits = BudgetLimits::uniform(10, 20);
        limits.dimensions[Dimension::ComputeSteps.index()] = 3;
        limits.dimensions[Dimension::MemoryBytes.index()] = 5;
        let mut budget = Budget::new(limits);
        let lease = budget.verifier_lease().unwrap();
        {
            let ledger = RefCell::new(&mut budget);
            let mut generator = LedgerView {
                ledger: &ledger,
                verifier: None,
            };
            let mut verifier = LedgerView {
                ledger: &ledger,
                verifier: Some(&lease),
            };
            generator.charge(Dimension::ComputeSteps, 3).unwrap();
            assert!(generator.charge(Dimension::ComputeSteps, 1).is_err());
            verifier
                .charge_batch(&[(Dimension::ComputeSteps, 2), (Dimension::MemoryBytes, 4)])
                .unwrap();
            let before = ledger.borrow().snapshot();
            assert!(
                verifier
                    .charge_batch(&[(Dimension::ComputeSteps, 7), (Dimension::MemoryBytes, 2),])
                    .is_err()
            );
            assert_eq!(ledger.borrow().snapshot(), before);
            // A new view has no fresh allowance of its own.
            let mut second = LedgerView {
                ledger: &ledger,
                verifier: Some(&lease),
            };
            second.charge(Dimension::ComputeSteps, 5).unwrap();
        }
        let snapshot = budget.snapshot();
        assert_eq!(snapshot.dimensions[Dimension::ComputeSteps.index()], 0);
        assert_eq!(snapshot.dimensions[Dimension::MemoryBytes.index()], 1);
        assert_eq!(snapshot.verifier_pool, 13);
    }

    #[test]
    fn inaccessible_ledger_and_foreign_verifier_authority_refuse_without_panics() {
        let mut budget = Budget::new(BudgetLimits::uniform(100, 100));
        let mut foreign = Budget::new(BudgetLimits::uniform(100, 100));
        let foreign_lease = foreign.verifier_lease().unwrap();
        let before = budget.snapshot();
        {
            let ledger = RefCell::new(&mut budget);
            let mut verifier = LedgerView {
                ledger: &ledger,
                verifier: Some(&foreign_lease),
            };
            assert_eq!(
                verifier.checkpoint(),
                Err(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))
            );
            assert!(verifier.charge(Dimension::ComputeSteps, 1).is_err());
            let mut generator = LedgerView {
                ledger: &ledger,
                verifier: None,
            };
            let held = ledger.borrow_mut();
            assert!(generator.checkpoint().is_err());
            assert!(generator.charge(Dimension::ComputeSteps, 1).is_err());
            drop(held);
        }
        assert_eq!(budget.snapshot(), before);
    }

    #[test]
    fn missing_or_foreign_lease_is_checked_before_generator_or_output_costs() {
        let problem =
            ResidualProblem::parse_sources(&["x".into()], &["x".into()], &mut Unbounded).unwrap();
        let mut budget = Budget::new(BudgetLimits::uniform(100_000, 100_000));
        let mut foreign = Budget::new(BudgetLimits::uniform(100_000, 100_000));
        let foreign_lease = foreign.verifier_lease().unwrap();
        let before = budget.snapshot();
        for lease in [None, Some(&foreign_lease)] {
            let result = solve(
                ResidualSystemId::new("system:real").unwrap(),
                &problem,
                None,
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
}
