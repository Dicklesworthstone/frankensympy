//! Cumulative construction/admission and independent sparse-program checks.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompileError, CompiledExpr, CompiledOp};
use fsym_calculus::sparse_jacobian::{JacobianVerificationError, SparseJacobian};
use fsym_calculus::verified::VerifiedCompilationError;
use fsym_core::{Expr, Symbol};

const AMPLE: u64 = 1 << 40;
const DIMENSIONS: [Dimension; 3] = [
    Dimension::ComputeSteps,
    Dimension::MemoryBytes,
    Dimension::AllocationCount,
];

fn budget(ordinary: [u64; 3], verifier: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(AMPLE, verifier);
    for (dimension, amount) in DIMENSIONS.into_iter().zip(ordinary) {
        limits.dimensions[dimension.index()] = amount;
    }
    Budget::new(limits)
}

fn remaining(ledger: &Budget) -> [u64; 3] {
    DIMENSIONS.map(|dimension| ledger.remaining(dimension))
}

fn sources() -> (Vec<Expr>, Vec<Symbol>) {
    (
        vec![
            fsym_core::parse("x^2+y").unwrap(),
            fsym_core::parse("sin(y)").unwrap(),
        ],
        vec![Symbol::new("y"), Symbol::new("x")],
    )
}

fn exhausted(error: &VerifiedCompilationError, expected: Dimension) -> bool {
    matches!(error,
        VerifiedCompilationError::Compilation(CompileError::Meter(MeterError::Budget(
            BudgetError::Exhausted { dimension, .. }
        ))) | VerifiedCompilationError::Jacobian(JacobianVerificationError::Compile(
            CompileError::Meter(MeterError::Budget(BudgetError::Exhausted { dimension, .. }))
        )) if *dimension == expected
    ) || matches!(error,
        VerifiedCompilationError::Jacobian(JacobianVerificationError::Eval(
            fsym_calculus::compile::EvalError::Meter(MeterError::Budget(
                BudgetError::Exhausted { dimension, .. }
            ))
        )) if *dimension == expected
    )
}

#[test]
fn protected_admission_checks_hand_derived_sparse_values_and_source_order() {
    let (residuals, variables) = sources();
    let mut ledger = budget([AMPLE; 3], AMPLE);
    let lease = ledger.verifier_lease().unwrap();
    let candidate =
        SparseJacobian::try_build_with_meter(&residuals, &variables, &mut ledger).unwrap();
    assert_eq!(ledger.verifier_remaining(), AMPLE);
    let before_admission = remaining(&ledger);
    let verified = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
        candidate,
        &mut ledger.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(
        ledger.remaining(Dimension::ComputeSteps),
        before_admission[0]
    );
    assert!(ledger.remaining(Dimension::MemoryBytes) < before_admission[1]);
    assert!(ledger.verifier_remaining() < AMPLE);
    assert_eq!(verified.residuals(), residuals);
    assert_eq!(verified.vars(), variables);
    assert_eq!(verified.jacobian_pattern(), [(0, 0), (0, 1), (1, 0)]);
    assert_eq!(verified.receipt().proofs_verified, 3);
    let reserve = ledger.verifier_remaining();
    let result = verified
        .evaluate_sparse_with_meter(&[0.0, 2.0], &mut ledger)
        .unwrap();
    assert_eq!(result.residuals, [4.0, 0.0]);
    assert_eq!(result.jacobian, [1.0, 4.0, 1.0]);
    assert_eq!(ledger.verifier_remaining(), reserve);
    assert!(ledger.remaining(Dimension::ComputeSteps) < before_admission[0]);
}

#[test]
fn complete_compilation_is_cumulative_and_each_resource_can_refuse_publication() {
    let (residuals, variables) = sources();
    let mut probe = budget([AMPLE; 3], 37);
    VerifiedResidualSystem::try_compile_with_meter(&residuals, &variables, &mut probe).unwrap();
    let cost = remaining(&probe).map(|left| AMPLE - left);
    assert!(cost.iter().all(|value| *value > 0));

    let mut shared = budget(cost.map(|value| value * 2), 37);
    for _ in 0..2 {
        VerifiedResidualSystem::try_compile_with_meter(&residuals, &variables, &mut shared)
            .unwrap();
    }
    assert_eq!(remaining(&shared), [0; 3]);
    assert_eq!(shared.verifier_remaining(), 37);
    assert!(exhausted(
        &VerifiedResidualSystem::try_compile_with_meter(&residuals, &variables, &mut shared)
            .unwrap_err(),
        Dimension::ComputeSteps,
    ));
    for (index, dimension) in DIMENSIONS.into_iter().enumerate() {
        let mut limits = [AMPLE; 3];
        limits[index] = cost[index] - 1;
        let mut ledger = budget(limits, 37);
        let error =
            VerifiedResidualSystem::try_compile_with_meter(&residuals, &variables, &mut ledger)
                .unwrap_err();
        assert!(exhausted(&error, dimension), "{dimension:?}: {error}");
        assert!(ledger.remaining(dimension) < limits[index]);
        assert_eq!(ledger.verifier_remaining(), 37);
    }
}

struct InterruptAt {
    ledger: Budget,
    checkpoints: usize,
    stop: Option<usize>,
}

impl InterruptAt {
    fn new(stop: Option<usize>) -> Self {
        Self {
            ledger: budget([AMPLE; 3], 37),
            checkpoints: 0,
            stop,
        }
    }
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.ledger, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.ledger, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.stop.is_some_and(|stop| self.checkpoints >= stop) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_generation_and_owned_admission_safe_point_can_cancel() {
    let residuals = vec![fsym_core::parse("x^2").unwrap()];
    let variables = vec![Symbol::new("x")];
    let candidate = SparseJacobian::try_build(&residuals, &variables).unwrap();
    for admission in [false, true] {
        let run = |meter: &mut InterruptAt| -> Result<(), VerifiedCompilationError> {
            if admission {
                VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
                    candidate.clone(),
                    meter,
                )
                .map(drop)
            } else {
                SparseJacobian::try_build_with_meter(&residuals, &variables, meter)
                    .map(drop)
                    .map_err(Into::into)
            }
        };
        let mut observed = InterruptAt::new(None);
        run(&mut observed).unwrap();
        assert!(observed.checkpoints > 10);
        for point in 1..=observed.checkpoints {
            let mut interrupted = InterruptAt::new(Some(point));
            let error = run(&mut interrupted).unwrap_err();
            assert!(
                matches!(
                    error,
                    VerifiedCompilationError::Compilation(CompileError::Meter(
                        MeterError::Cancelled
                    )) | VerifiedCompilationError::Jacobian(JacobianVerificationError::Compile(
                        CompileError::Meter(MeterError::Cancelled)
                    )) | VerifiedCompilationError::Jacobian(JacobianVerificationError::Eval(
                        fsym_calculus::compile::EvalError::Meter(MeterError::Cancelled)
                    ))
                ),
                "admission={admission}, checkpoint={point}: {error}"
            );
            assert_eq!(interrupted.ledger.verifier_remaining(), 37);
        }
    }
}

#[test]
fn metered_admission_rejects_forged_bytecode_derivatives_and_missing_coordinates() {
    let candidate =
        SparseJacobian::try_build(&[fsym_core::parse("x^2").unwrap()], &[Symbol::new("x")])
            .unwrap();
    for alteration in 0..3 {
        let mut forged = candidate.clone();
        match alteration {
            0 => {
                forged.entries[0].compiled_deriv =
                    CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(42.0)]).unwrap()
            }
            1 => forged.entries[0].symbolic_deriv = Expr::from_i64(0),
            _ => {
                forged.entries.clear();
                forged.pattern.nonzeros.clear();
                forged.pattern.nnz = 0;
            }
        }
        let mut ledger = budget([AMPLE; 3], AMPLE);
        let error =
            VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(forged, &mut ledger)
                .unwrap_err();
        assert!(
            matches!(error, VerifiedCompilationError::Jacobian(_)),
            "{error}"
        );
        assert!(ledger.remaining(Dimension::ComputeSteps) < AMPLE);
    }
}
