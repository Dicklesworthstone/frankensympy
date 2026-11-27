//! Numerical Hessians propose witnesses; only the original-source checker
//! proves stationary points and local curvature.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedObjective;
use fsym_core::{BigInt, BigRational, RootBoxError, StationaryPointKind, Symbol, parse};
use fsym_numeric_consumer::{
    ObjectiveConsumerError, exact_binary64_rational, propose_stationary_box_with_meter,
};
use std::error::Error;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn objective(source: &str, variables: &[&str]) -> VerifiedObjective {
    let source = parse(source).unwrap();
    let variables: Vec<_> = variables.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedObjective::try_compile(&source, &variables).unwrap()
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(1 << 40, 1 << 40))
}

#[test]
fn exact_coupled_witness_has_separate_protected_verification_and_rejects_mutants() {
    let objective = objective("3*(x-2)^2+2*(x-2)*(y+1)+4*(y+1)^2+7", &["y", "x"]);
    let mut budget = budget();
    let protected = budget.verifier_remaining();
    let witness = propose_stationary_box_with_meter(
        &objective,
        &[-1.0, 2.0],
        &[q(1, 10), q(1, 20)],
        24,
        &mut budget,
    )
    .unwrap();
    assert_eq!(witness.center, [q(-1, 1), q(2, 1)]);
    assert_eq!(witness.domain[0].radius(), &q(1, 10));
    assert_eq!(witness.domain[1].radius(), &q(1, 20));
    assert_eq!(budget.verifier_remaining(), protected);
    let lease = budget.verifier_lease().unwrap();
    let checked = fsym_core::verify_stationary_point(
        objective.objective(),
        objective.variables(),
        &witness,
        &mut budget.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(checked.kind(), StationaryPointKind::StrictMinimum);
    assert!(checked.stationary_enclosure()[0].contains(&q(-1, 1)));
    assert!(checked.stationary_enclosure()[1].contains(&q(2, 1)));
    assert!(budget.verifier_remaining() < protected);

    let mut forged = witness;
    forged.preconditioner[0].fill(q(0, 1));
    assert_eq!(
        fsym_core::certify_stationary_point(objective.objective(), objective.variables(), &forged),
        Err(RootBoxError::NotContractive)
    );
}

#[test]
fn stationary_maxima_and_saddles_retain_their_true_local_classification() {
    for (objective, point, kind) in [
        (
            objective("3-x^2", &["x"]),
            vec![0.0],
            StationaryPointKind::StrictMaximum,
        ),
        (
            objective("x^2-y^2", &["x", "y"]),
            vec![0.0, 0.0],
            StationaryPointKind::Saddle,
        ),
    ] {
        let witness = propose_stationary_box_with_meter(
            &objective,
            &point,
            &vec![q(1, 10); point.len()],
            24,
            &mut budget(),
        )
        .unwrap();
        let checked = fsym_core::certify_stationary_point(
            objective.objective(),
            objective.variables(),
            &witness,
        )
        .unwrap();
        assert_eq!(checked.kind(), kind);
    }
}

#[test]
fn a_candidate_does_not_assert_inclusion_and_preserves_exact_binary64_coordinates() {
    let objective = objective("x^2", &["x"]);
    let point = 0.1_f64;
    let witness =
        propose_stationary_box_with_meter(&objective, &[point], &[q(1, 1000)], 24, &mut budget())
            .unwrap();
    assert_eq!(witness.center[0], exact_binary64_rational(point).unwrap());
    assert_ne!(witness.center[0], q(1, 10));
    assert_eq!(witness.preconditioner, [vec![q(1, 2)]]);
    assert!(matches!(
        fsym_core::certify_stationary_point(objective.objective(), objective.variables(), &witness),
        Err(RootBoxError::NoInteriorInclusion { .. })
    ));
}

#[test]
fn singular_hessians_and_invalid_controls_refuse_before_protected_work() {
    let regular = objective("x^2", &["x"]);
    for (point, radii, precision) in [
        (vec![], vec![q(1, 10)], 24),
        (vec![f64::NAN], vec![q(1, 10)], 24),
        (vec![0.0], vec![], 24),
        (vec![0.0], vec![q(0, 1)], 24),
        (vec![0.0], vec![q(-1, 1)], 24),
        (vec![0.0], vec![q(1, 10)], 7),
        (vec![0.0], vec![q(1, 10)], 129),
    ] {
        let mut meter = budget();
        let protected = meter.verifier_remaining();
        assert!(
            propose_stationary_box_with_meter(&regular, &point, &radii, precision, &mut meter)
                .is_err()
        );
        assert_eq!(meter.verifier_remaining(), protected);
    }
    let singular = objective("x^4", &["x"]);
    assert!(matches!(
        propose_stationary_box_with_meter(&singular, &[0.0], &[q(1, 10)], 24, &mut budget()),
        Err(ObjectiveConsumerError::LinearSolve(_))
    ));
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    cancel_at: usize,
}

impl Interrupt {
    fn new(cancel_at: usize) -> Self {
        Self {
            budget: budget(),
            checkpoints: 0,
            cancel_at,
        }
    }
}

impl BudgetMeter for Interrupt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let index = self.checkpoints;
        self.checkpoints += 1;
        if index >= self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_proposal_safe_point_propagates_cancellation_without_a_candidate() {
    let objective = objective("x^2", &["x"]);
    let mut measured = Interrupt::new(usize::MAX);
    propose_stationary_box_with_meter(&objective, &[0.0], &[q(1, 10)], 24, &mut measured).unwrap();
    assert!(measured.checkpoints > 10);
    for cancel_at in 0..measured.checkpoints {
        let mut meter = Interrupt::new(cancel_at);
        let protected = meter.budget.verifier_remaining();
        let error =
            propose_stationary_box_with_meter(&objective, &[0.0], &[q(1, 10)], 24, &mut meter)
                .unwrap_err();
        let mut cause: Option<&(dyn Error + 'static)> = Some(&error);
        let mut cancelled = false;
        while let Some(current) = cause {
            if current.downcast_ref::<MeterError>() == Some(&MeterError::Cancelled) {
                cancelled = true;
                break;
            }
            cause = current.source();
        }
        assert!(cancelled, "safe point {cancel_at}: {error:?}");
        assert_eq!(meter.budget.verifier_remaining(), protected);
    }
}

#[test]
fn widely_scaled_hessian_yields_independently_classified_extrema_and_saddles() {
    for (formula, kind, sign) in [
        (
            "2^59*(x-1)^2 + (y+1)^2/2",
            StationaryPointKind::StrictMinimum,
            1,
        ),
        ("2^59*(x-1)^2 - (y+1)^2/2", StationaryPointKind::Saddle, -1),
    ] {
        let objective = objective(formula, &["x", "y"]);
        let mut budget = budget();
        let protected = budget.verifier_remaining();
        let witness = propose_stationary_box_with_meter(
            &objective,
            &[1.0, -1.0],
            &[q(1, 8), q(1, 8)],
            24,
            &mut budget,
        )
        .unwrap();
        assert_eq!(
            witness.preconditioner,
            [vec![q(1, 1_i64 << 60), q(0, 1)], vec![q(0, 1), q(sign, 1)],]
        );
        assert_eq!(budget.verifier_remaining(), protected);
        let lease = budget.verifier_lease().unwrap();
        let checked = fsym_core::verify_stationary_point(
            objective.objective(),
            objective.variables(),
            &witness,
            &mut budget.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
        assert_eq!(checked.kind(), kind);
        assert!(budget.verifier_remaining() < protected);
    }
}
