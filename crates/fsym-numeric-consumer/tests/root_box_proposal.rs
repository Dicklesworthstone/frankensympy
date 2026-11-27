//! Candidate-only root-box construction stays outside the acceptance boundary.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_core::{BigInt, BigRational, Expr, RootBoxError, Symbol};
use fsym_numeric_consumer::{SystemConsumerError, propose_root_box_with_meter};

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn system() -> VerifiedResidualSystem {
    VerifiedResidualSystem::try_compile(
        &[Expr::symbol("x") - Expr::from_i64(1)],
        &[Symbol::new("x")],
    )
    .unwrap()
}

#[test]
fn exact_radii_and_untrusted_preconditioners_have_separate_verification() {
    let system = system();
    let mut limits = BudgetLimits::uniform(10_000_000, 2_000_000);
    // Each independent verification reserves the real ball kernels' 64 MiB
    // envelope; both checks below share this ordinary allocation ledger.
    limits.dimensions[Dimension::MemoryBytes.index()] = 256 * 1024 * 1024;
    let mut budget = Budget::new(limits);
    let lease = budget.verifier_lease().unwrap();
    let witness =
        propose_root_box_with_meter(&system, &[1.0], &[q(1, 10)], 24, &mut budget).unwrap();
    assert_eq!(witness.domain[0].radius(), &q(1, 10));
    assert_eq!(witness.center, [q(1, 1)]);
    assert_eq!(witness.preconditioner, [vec![q(1, 1)]]);
    assert_eq!(budget.verifier_remaining(), 2_000_000);
    let receipt = fsym_core::verify_root_box(
        system.residuals(),
        system.vars(),
        &witness,
        &mut budget.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert!(budget.verifier_remaining() < 2_000_000);
    assert!(receipt.root_enclosure()[0].contains(&q(1, 1)));

    let mut forged = witness;
    forged.preconditioner[0][0] = q(0, 1);
    assert_eq!(
        fsym_core::verify_root_box(
            system.residuals(),
            system.vars(),
            &forged,
            &mut budget.verifier_meter(&lease).unwrap(),
        ),
        Err(RootBoxError::NotContractive)
    );
}

#[test]
fn a_constructible_box_does_not_assert_that_it_contains_a_root() {
    let system = system();
    let mut budget = Budget::new(BudgetLimits::uniform(10_000_000, 100));
    let witness =
        propose_root_box_with_meter(&system, &[0.0], &[q(1, 10)], 24, &mut budget).unwrap();
    assert_eq!(budget.verifier_remaining(), 100);
    assert!(fsym_core::certify_root_box(system.residuals(), system.vars(), &witness).is_err());
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    cancel_at: usize,
}

impl Interrupt {
    fn new(cancel_at: usize) -> Self {
        Self {
            budget: Budget::new(BudgetLimits::uniform(10_000_000, 137)),
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
fn cancellation_at_every_witness_safe_point_returns_no_candidate_or_verifier_charge() {
    let system = system();
    let mut measured = Interrupt::new(usize::MAX);
    propose_root_box_with_meter(&system, &[1.0], &[q(1, 10)], 24, &mut measured).unwrap();
    assert!(measured.checkpoints > 10);
    for cancel_at in 0..measured.checkpoints {
        let mut meter = Interrupt::new(cancel_at);
        let error =
            propose_root_box_with_meter(&system, &[1.0], &[q(1, 10)], 24, &mut meter).unwrap_err();
        assert!(
            matches!(
                error,
                SystemConsumerError::Meter(MeterError::Cancelled)
                    | SystemConsumerError::Evaluation(EvalError::Meter(MeterError::Cancelled))
                    | SystemConsumerError::Certification(RootBoxError::Meter(
                        MeterError::Cancelled
                    ))
            ),
            "safe point {cancel_at}: {error:?}"
        );
        assert_eq!(meter.budget.verifier_remaining(), 137);
    }
}

#[test]
fn invalid_exact_radii_refuse_without_consuming_the_protected_pool() {
    let system = system();
    for radii in [vec![], vec![q(0, 1)], vec![q(-1, 1)], vec![q(1, 10); 2]] {
        let mut budget = Budget::new(BudgetLimits::uniform(10_000_000, 137));
        assert!(propose_root_box_with_meter(&system, &[1.0], &radii, 24, &mut budget).is_err());
        assert_eq!(budget.verifier_remaining(), 137);
    }
}

#[test]
fn one_pivoted_factorization_preserves_every_inverse_column_in_source_order() {
    let vars = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let z = Expr::Sym(vars[2].clone());
    let system = VerifiedResidualSystem::try_compile(
        &[
            Expr::from_i64(2) * y.clone(),
            Expr::from_i64(4) * z.clone(),
            x + y + z,
        ],
        &vars,
    )
    .unwrap();
    let witness = propose_root_box_with_meter(
        &system,
        &[0.0; 3],
        &[q(1, 8), q(1, 8), q(1, 8)],
        24,
        &mut fsym_budget::Unbounded,
    )
    .unwrap();
    assert_eq!(
        witness.preconditioner,
        [
            vec![q(-1, 2), q(-1, 4), q(1, 1)],
            vec![q(1, 2), q(0, 1), q(0, 1)],
            vec![q(0, 1), q(1, 4), q(0, 1)],
        ]
    );
    fsym_core::certify_root_box(system.residuals(), system.vars(), &witness).unwrap();
}
