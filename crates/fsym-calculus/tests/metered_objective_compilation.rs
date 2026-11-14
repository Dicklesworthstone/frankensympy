//! Source-bound objective construction under cumulative and protected budgets.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompileError, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::sparse_hessian::{
    MAX_OBJECTIVE_PAYLOAD_BITS, MAX_OBJECTIVE_SOURCE_DEPTH, MAX_OBJECTIVE_SOURCE_NODES,
    ObjectiveCandidate, ObjectiveCompilationError, VerifiedObjective,
};
use fsym_calculus::sparse_jacobian::JacobianVerificationError;
use fsym_core::{BigInt, Expr, Symbol, SymbolIdentity};
use std::sync::Arc;

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

fn meter_error(error: &ObjectiveCompilationError) -> Option<&MeterError> {
    match error {
        ObjectiveCompilationError::Compilation(CompileError::Meter(error))
        | ObjectiveCompilationError::Compilation(CompileError::InvalidBytecode(
            EvalError::Meter(error),
        ))
        | ObjectiveCompilationError::Verification(JacobianVerificationError::Compile(
            CompileError::Meter(error),
        ))
        | ObjectiveCompilationError::Verification(JacobianVerificationError::Compile(
            CompileError::InvalidBytecode(EvalError::Meter(error)),
        ))
        | ObjectiveCompilationError::Verification(JacobianVerificationError::Eval(
            EvalError::Meter(error),
        )) => Some(error),
        _ => None,
    }
}

fn exhausted(error: &ObjectiveCompilationError, expected: Dimension) -> bool {
    matches!(
        meter_error(error),
        Some(MeterError::Budget(BudgetError::Exhausted { dimension, .. }))
            if *dimension == expected
    )
}

#[test]
fn protected_admission_preserves_hand_derived_value_gradient_hessian_and_axis_order() {
    let source = fsym_core::parse("x^2*y+3*y^2").unwrap();
    let variables = [Symbol::new("y"), Symbol::new("x")];
    let mut ledger = budget([AMPLE; 3], AMPLE);
    let lease = ledger.verifier_lease().unwrap();
    let candidate =
        ObjectiveCandidate::try_build_with_meter(&source, &variables, &mut ledger).unwrap();
    assert_eq!(ledger.verifier_remaining(), AMPLE);
    let before = remaining(&ledger);
    let program = VerifiedObjective::try_from_candidate_with_meter(
        &source,
        &variables,
        candidate,
        &mut ledger.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(ledger.remaining(Dimension::ComputeSteps), before[0]);
    assert!(ledger.remaining(Dimension::MemoryBytes) < before[1]);
    assert!(ledger.remaining(Dimension::AllocationCount) < before[2]);
    assert!(ledger.verifier_remaining() < AMPLE);
    assert_eq!(program.objective(), &source);
    assert_eq!(program.variables(), variables);
    assert_eq!(program.hessian_pattern(), [(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(program.receipt().gradient_proofs_verified, 2);
    assert_eq!(program.receipt().hessian_proofs_verified, 4);
    assert_eq!(program.receipt().zero_entries_checked, 0);

    let reserve = ledger.verifier_remaining();
    let result = program
        .evaluate_with_meter(&[2.0, 3.0], &mut ledger)
        .unwrap();
    assert_eq!(result.value, 30.0);
    assert_eq!(result.gradient, [21.0, 12.0]);
    assert_eq!(result.hessian, [6.0, 6.0, 6.0, 4.0]);
    assert_eq!(
        program
            .hessian_vector_product_with_meter(&[2.0, 3.0], &[2.0, -1.0], &mut ledger)
            .unwrap(),
        [6.0, 8.0]
    );
    assert_eq!(ledger.verifier_remaining(), reserve);
    assert!(ledger.remaining(Dimension::ComputeSteps) < before[0]);
}

#[test]
fn generation_admission_and_complete_compilation_retain_cumulative_resource_cost() {
    let source = fsym_core::parse("x^2").unwrap();
    let variables = [Symbol::new("x")];
    let candidate = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    for phase in 0..3 {
        let run =
            |ledger: &mut Budget| -> Result<(), ObjectiveCompilationError> {
                match phase {
                    0 => ObjectiveCandidate::try_build_with_meter(&source, &variables, ledger)
                        .map(drop),
                    1 => VerifiedObjective::try_from_candidate_with_meter(
                        &source,
                        &variables,
                        candidate.clone(),
                        ledger,
                    )
                    .map(drop),
                    _ => VerifiedObjective::try_compile_with_meter(&source, &variables, ledger)
                        .map(drop),
                }
            };
        let mut probe = budget([AMPLE; 3], 37);
        run(&mut probe).unwrap();
        let cost = remaining(&probe).map(|left| AMPLE - left);
        assert!(cost.iter().all(|value| *value > 0));
        let mut shared = budget(cost.map(|value| value * 2), 37);
        run(&mut shared).unwrap();
        run(&mut shared).unwrap();
        assert_eq!(remaining(&shared), [0; 3], "phase={phase}");
        assert_eq!(shared.verifier_remaining(), 37);
        assert!(exhausted(
            &run(&mut shared).unwrap_err(),
            Dimension::ComputeSteps
        ));
        for (index, dimension) in DIMENSIONS.into_iter().enumerate() {
            let mut limits = [AMPLE; 3];
            limits[index] = cost[index] - 1;
            let mut ledger = budget(limits, 37);
            let error = run(&mut ledger).unwrap_err();
            assert!(exhausted(&error, dimension), "phase={phase}: {error}");
            assert!(ledger.remaining(dimension) < limits[index]);
            assert_eq!(ledger.verifier_remaining(), 37);
        }
    }
}

#[test]
fn generation_cannot_spend_protected_fuel_and_admission_cannot_borrow_generator_fuel() {
    let source = fsym_core::parse("x^2").unwrap();
    let variables = [Symbol::new("x")];
    let mut generator = budget([0, AMPLE, AMPLE], AMPLE);
    let error =
        ObjectiveCandidate::try_build_with_meter(&source, &variables, &mut generator).unwrap_err();
    assert!(exhausted(&error, Dimension::ComputeSteps));
    assert_eq!(generator.verifier_remaining(), AMPLE);

    let candidate = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    let mut verifier = budget([AMPLE; 3], 1);
    let lease = verifier.verifier_lease().unwrap();
    let before = verifier.remaining(Dimension::ComputeSteps);
    let error = VerifiedObjective::try_from_candidate_with_meter(
        &source,
        &variables,
        candidate,
        &mut verifier.verifier_meter(&lease).unwrap(),
    )
    .unwrap_err();
    assert!(exhausted(&error, Dimension::ComputeSteps));
    assert_eq!(verifier.remaining(Dimension::ComputeSteps), before);
    assert_eq!(verifier.verifier_remaining(), 0);
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
fn every_generation_and_independent_admission_safe_point_can_cancel() {
    let source = fsym_core::parse("x^2").unwrap();
    let variables = [Symbol::new("x")];
    let candidate = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    for admission in [false, true] {
        let run = |meter: &mut InterruptAt| -> Result<(), ObjectiveCompilationError> {
            if admission {
                VerifiedObjective::try_from_candidate_with_meter(
                    &source,
                    &variables,
                    candidate.clone(),
                    meter,
                )
                .map(drop)
            } else {
                ObjectiveCandidate::try_build_with_meter(&source, &variables, meter).map(drop)
            }
        };
        let mut observed = InterruptAt::new(None);
        run(&mut observed).unwrap();
        assert!(observed.checkpoints > 10);
        for point in 1..=observed.checkpoints {
            let mut interrupted = InterruptAt::new(Some(point));
            let error = run(&mut interrupted).unwrap_err();
            assert!(
                matches!(meter_error(&error), Some(MeterError::Cancelled)),
                "admission={admission}, checkpoint={point}: {error}"
            );
            assert_eq!(interrupted.ledger.verifier_remaining(), 37);
        }
    }
}

#[test]
fn cancellation_at_a_refusal_boundary_remains_an_operational_outcome() {
    let source = fsym_core::parse("x^2").unwrap();
    let variables = [Symbol::new("x")];
    let mut forged = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    forged.compiled_objective =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(42.0)]).unwrap();
    let unsupported = Expr::Function("unregistered".into(), vec![Expr::Sym(variables[0].clone())]);
    for admission in [false, true] {
        let run = |meter: &mut InterruptAt| -> Result<(), ObjectiveCompilationError> {
            if admission {
                VerifiedObjective::try_from_candidate_with_meter(
                    &source,
                    &variables,
                    forged.clone(),
                    meter,
                )
                .map(drop)
            } else {
                ObjectiveCandidate::try_build_with_meter(&unsupported, &variables, meter).map(drop)
            }
        };
        let mut observed = InterruptAt::new(None);
        assert!(meter_error(&run(&mut observed).unwrap_err()).is_none());
        let mut cancelled = InterruptAt::new(Some(observed.checkpoints));
        assert!(matches!(
            meter_error(&run(&mut cancelled).unwrap_err()),
            Some(MeterError::Cancelled)
        ));
    }
}

#[test]
fn metered_import_rejects_source_axis_proof_bytecode_and_completeness_forgeries() {
    let source = fsym_core::parse("x^2*y+3*y^2").unwrap();
    let variables = [Symbol::new("y"), Symbol::new("x")];
    let candidate = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    for alteration in 0..7 {
        let mut forged = candidate.clone();
        match alteration {
            0 => {
                forged.compiled_objective =
                    CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(30.0)]).unwrap();
            }
            1 => forged.gradient.residuals[0] = Expr::from_i64(30),
            2 => {
                let typed =
                    Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([3; 32]));
                forged.gradient.vars[0] = typed.clone();
                forged.hessian.vars[0] = typed;
            }
            3 => forged.hessian.residuals[0] = Expr::from_i64(21),
            4 => {
                forged.hessian.entries.remove(0);
                forged.hessian.pattern.nonzeros.remove(0);
                forged.hessian.pattern.nnz -= 1;
            }
            5 => {
                forged.hessian.entries[0].compiled_deriv =
                    CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
            }
            _ => forged.gradient.entries[0].derivation.steps.clear(),
        }
        let mut ledger = budget([AMPLE; 3], 37);
        assert!(
            VerifiedObjective::try_from_candidate_with_meter(
                &source,
                &variables,
                forged,
                &mut ledger,
            )
            .is_err(),
            "alteration={alteration}"
        );
        assert!(ledger.remaining(Dimension::ComputeSteps) < AMPLE);
        assert_eq!(ledger.verifier_remaining(), 37);
    }
}

#[test]
fn original_domain_guards_survive_zero_omission_and_zero_hessian_directions() {
    let x = Symbol::new("x");
    let source = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-1))),
    ]);
    let mut ledger = budget([AMPLE; 3], 37);
    let program = VerifiedObjective::try_compile_with_meter(&source, &[x], &mut ledger).unwrap();
    assert_eq!(program.objective(), &source);
    assert_eq!(
        program
            .evaluate_with_meter(&[2.0], &mut ledger)
            .unwrap()
            .value,
        0.0
    );
    assert!(program.evaluate_with_meter(&[0.0], &mut ledger).is_err());
    assert!(
        program
            .hessian_vector_product_with_meter(&[0.0], &[0.0], &mut ledger)
            .is_err()
    );
    assert_eq!(ledger.verifier_remaining(), 37);
}

#[test]
fn metered_preflight_preserves_depth_node_and_exact_payload_limits() {
    let mut deep = Expr::Sym(Symbol::new("x"));
    for _ in 0..=MAX_OBJECTIVE_SOURCE_DEPTH {
        deep = Expr::Function("sin".into(), vec![deep]);
    }
    let wide = Expr::Add(vec![Expr::from_i64(0); MAX_OBJECTIVE_SOURCE_NODES]);
    let large = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Integer(BigInt::one() << MAX_OBJECTIVE_PAYLOAD_BITS as u32),
    ]);
    for (source, expected) in [
        (deep, "expression depth"),
        (wide, "expression nodes"),
        (large, "integer bits"),
    ] {
        let mut ledger = budget([AMPLE; 3], 37);
        let error =
            ObjectiveCandidate::try_build_with_meter(&source, &[Symbol::new("x")], &mut ledger)
                .unwrap_err();
        assert!(
            matches!(
                error,
                ObjectiveCompilationError::ResourceLimit { resource, .. } if resource == expected
            ),
            "{error}"
        );
        assert!(ledger.remaining(Dimension::ComputeSteps) < AMPLE);
    }
}

#[test]
fn same_name_typed_coordinates_and_constant_empty_axes_keep_exact_semantics() {
    let plain = Symbol::new("x");
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([5; 32]));
    let source = Expr::Add(vec![
        Expr::pow(Expr::Sym(plain.clone()), Expr::from_i64(2)),
        Expr::Mul(vec![
            Expr::from_i64(3),
            Expr::pow(Expr::Sym(typed.clone()), Expr::from_i64(2)),
        ]),
    ]);
    let variables = [plain, typed];
    let mut ledger = budget([AMPLE; 3], 37);
    let program =
        VerifiedObjective::try_compile_with_meter(&source, &variables, &mut ledger).unwrap();
    let result = program
        .evaluate_with_meter(&[2.0, 3.0], &mut ledger)
        .unwrap();
    assert_eq!(program.variables(), variables);
    assert_eq!(program.hessian_pattern(), [(0, 0), (1, 1)]);
    assert_eq!(result.value, 31.0);
    assert_eq!(result.gradient, [4.0, 18.0]);
    assert_eq!(result.hessian, [2.0, 6.0]);

    let constant =
        VerifiedObjective::try_compile_with_meter(&Expr::from_i64(7), &[], &mut ledger).unwrap();
    let result = constant.evaluate_with_meter(&[], &mut ledger).unwrap();
    assert_eq!(result.value, 7.0);
    assert!(result.gradient.is_empty());
    assert!(result.hessian.is_empty());
    assert_eq!(constant.receipt().zero_entries_checked, 0);
    assert_eq!(ledger.verifier_remaining(), 37);
}
