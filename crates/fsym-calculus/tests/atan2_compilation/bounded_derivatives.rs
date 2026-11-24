//! Bounded numerical differentiation and independent replay of held powers.
//!
//! The adversarial constants have tiny syntax but would expand to roughly a
//! billion bits if generic integer folding were reached without a governor.
//! No test evaluates those constants or invokes the public symbolic simplifier
//! on them. Raw recipes may remain admitted; required normalization must refuse.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::compile::{CompileError, CompiledExpr, CompiledResidualSystem, EvalError};
use fsym_calculus::proof::{classify_diff_rule, make_diff_term};
use fsym_calculus::sparse_hessian::{
    ObjectiveCandidate, ObjectiveCompilationError, VerifiedObjective,
};
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, SparseJacobianEntry, SparsityPattern,
    verify_sparse_jacobian_certificate_with_meter,
};
use fsym_calculus::{VerifiedResidualSystem, diff};
use fsym_core::{Expr, Symbol, parse};
use fsym_proof_kernel::{Claim, DerivationStep, DerivationTree, ProofRule, StepId};
use fsym_simplify::SimplifyError;
use std::collections::HashMap;
use std::sync::Arc;

const AMPLE: u64 = 1 << 40;
const RESERVE: u64 = 37;
const DIMENSIONS: [Dimension; 3] = [
    Dimension::ComputeSteps,
    Dimension::MemoryBytes,
    Dimension::AllocationCount,
];

fn function(name: &str, arguments: Vec<Expr>) -> Expr {
    Expr::Function(name.into(), arguments)
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn held_power() -> Expr {
    let mut value = Expr::from_i64(2);
    for _ in 0..3 {
        value = power(value, 1000);
    }
    value
}

fn angle(constant: Expr) -> Expr {
    function("atan2", vec![Expr::symbol("t"), constant])
}

fn bounded() -> Budget {
    let mut limits = BudgetLimits::uniform(64 * 1024 * 1024, RESERVE);
    limits.dimensions[Dimension::ComputeSteps.index()] = 8_000_000;
    Budget::new(limits)
}

fn normalization_refusal(error: &CompileError) -> bool {
    matches!(
        error,
        CompileError::DerivativeNormalization(SimplifyError::NumericLimitExceeded { .. })
    )
}

fn compile_meter(error: &CompileError) -> Option<MeterError> {
    match error {
        CompileError::Meter(error) | CompileError::InvalidBytecode(EvalError::Meter(error)) => {
            Some(*error)
        }
        _ => None,
    }
}

fn replay_meter(error: &JacobianVerificationError) -> Option<MeterError> {
    match error {
        JacobianVerificationError::Compile(error) => compile_meter(error),
        JacobianVerificationError::Eval(EvalError::Meter(error)) => Some(*error),
        _ => None,
    }
}

fn raw_derivative(constant: &Expr, historical: bool) -> Expr {
    let t = Expr::symbol("t");
    if historical {
        Expr::Mul(vec![
            Expr::Add(vec![
                Expr::Mul(vec![constant.clone(), Expr::from_i64(1)]),
                Expr::Mul(vec![Expr::from_i64(-1), t.clone(), Expr::from_i64(0)]),
            ]),
            power(Expr::Add(vec![power(constant.clone(), 2), power(t, 2)]), -1),
        ])
    } else {
        Expr::Add(vec![
            Expr::Mul(vec![
                function("_fsym_atan2_dy", vec![t.clone(), constant.clone()]),
                Expr::from_i64(1),
            ]),
            Expr::Mul(vec![
                function("_fsym_atan2_dx", vec![t, constant.clone()]),
                Expr::from_i64(0),
            ]),
        ])
    }
}

fn sparse(source: Expr, derivative: Option<Expr>) -> SparseJacobian {
    let variable = Symbol::new("t");
    let entries = derivative
        .into_iter()
        .map(|derivative| {
            let lhs = make_diff_term(&source, &variable);
            let rule = classify_diff_rule(&source, &variable);
            let derivation = DerivationTree {
                steps: vec![DerivationStep {
                    id: StepId(1),
                    rule: ProofRule::DefinitionalReduction {
                        lhs: lhs.clone(),
                        rhs: derivative.clone(),
                        rule_name: rule.into(),
                    },
                    claim: Claim::equality(lhs, derivative.clone()),
                }],
                root: StepId(1),
            };
            let mapping = HashMap::from([(variable.clone(), 0)]);
            SparseJacobianEntry {
                row: 0,
                col: 0,
                compiled_deriv: CompiledExpr::try_compile(&derivative, &mapping).unwrap(),
                symbolic_deriv: derivative,
                derivation,
            }
        })
        .collect::<Vec<_>>();
    let nonzeros = if entries.is_empty() {
        vec![]
    } else {
        vec![(0, 0)]
    };
    SparseJacobian {
        residuals: vec![source],
        vars: vec![variable],
        pattern: SparsityPattern {
            num_rows: 1,
            num_cols: 1,
            nnz: entries.len(),
            nonzeros,
        },
        entries,
        column_colors: vec![0],
        num_colors: 1,
    }
}

#[test]
fn radial_cancellation_uses_identity_for_large_continued_fraction_coefficients() {
    use fsym_core::{BigInt, BigRational};

    // Adjacent Fibonacci ratios have a long common continued fraction.
    // Sorting them by numeric Rational::Ord would run an unmetered Euclidean
    // chain. Cancellation only requires canonical component identity.
    let mut previous = BigInt::from(1);
    let mut current = BigInt::from(1);
    loop {
        let next = previous.clone() + current.clone();
        if next.bits() > 4096 {
            break;
        }
        previous = current;
        current = next;
    }
    assert_eq!(current.bits(), 4096);
    let older = current.clone() - previous.clone();
    let first = BigRational::new(current, previous.clone());
    let second = BigRational::new(previous, older);
    let variable = Symbol::new("x");
    let source = function(
        "atan2",
        vec![
            Expr::Mul(vec![Expr::Rational(first), Expr::Sym(variable.clone())]),
            Expr::Mul(vec![Expr::Rational(second), Expr::Sym(variable.clone())]),
        ],
    );
    let mut dense_meter = Interruptible::new();
    let program = CompiledResidualSystem::try_compile_with_meter(
        std::slice::from_ref(&source),
        std::slice::from_ref(&variable),
        &mut dense_meter,
    )
    .unwrap();
    assert_derivative_envelope(&dense_meter.ledger);
    let mut residual = [f64::NAN];
    let mut jacobian = [f64::NAN];
    program
        .try_eval_system(&[1.0], &mut residual, &mut jacobian)
        .unwrap();
    assert!(residual[0].is_finite());
    assert_eq!(jacobian[0], 0.0);
    assert!(
        program
            .try_eval_system(&[0.0], &mut residual, &mut jacobian)
            .is_err()
    );

    let mut sparse_meter = Interruptible::new();
    let candidate =
        SparseJacobian::try_build_with_meter(&[source], &[variable], &mut sparse_meter).unwrap();
    assert_eq!(candidate.pattern.nnz, 0);
    assert!(candidate.entries.is_empty());
    assert_derivative_envelope(&sparse_meter.ledger);
    // The independent verifier retains its separate multiplicity matcher;
    // it must establish the omitted axis without numeric rational ordering.
    let mut replay_meter = Interruptible::new();
    verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut replay_meter).unwrap();
    assert_derivative_envelope(&replay_meter.ledger);
}

#[test]
fn dense_numerical_differentiation_refuses_tiny_power_forests_before_expansion() {
    let source = angle(held_power());
    let mut meter = bounded();
    let error = CompiledResidualSystem::try_compile_with_meter(
        std::slice::from_ref(&source),
        &[Symbol::new("t")],
        &mut meter,
    )
    .unwrap_err();
    assert!(normalization_refusal(&error), "{error:?}");
    assert!(meter.remaining(Dimension::ComputeSteps) > 0);
    assert_eq!(meter.verifier_remaining(), RESERVE);
    // The convenience entrypoint has the same intrinsic envelope even though
    // its caller deliberately supplied no finite resource ledger.
    let error = CompiledResidualSystem::try_compile(&[source], &[Symbol::new("t")]).unwrap_err();
    assert!(normalization_refusal(&error), "{error:?}");
}

#[test]
fn nested_sparse_chain_rules_refuse_before_normalizing_a_huge_coordinate() {
    let source = function("sin", vec![angle(held_power())]);
    let mut meter = bounded();
    let error = SparseJacobian::try_build_with_meter(&[source], &[Symbol::new("t")], &mut meter)
        .unwrap_err();
    assert!(normalization_refusal(&error), "{error:?}");
    assert!(meter.remaining(Dimension::ComputeSteps) > 0);
    assert_eq!(meter.verifier_remaining(), RESERVE);
}

#[test]
fn objective_generation_and_two_stage_admission_bound_gradient_normalization() {
    let constant = held_power();
    let source = angle(constant.clone());
    let variables = [Symbol::new("t")];
    let mut meter = bounded();
    let error =
        VerifiedObjective::try_compile_with_meter(&source, &variables, &mut meter).unwrap_err();
    assert!(
        matches!(&error, ObjectiveCompilationError::Compilation(error) if normalization_refusal(error)),
        "{error:?}"
    );

    // A hostile saved objective can carry an individually valid raw gradient
    // and an unproved Hessian. Reconstructing the checked gradient sources is
    // itself governed before any second-stage witness can be admitted.
    let first = raw_derivative(&constant, false);
    let candidate = ObjectiveCandidate {
        compiled_objective: CompiledExpr::try_compile(
            &source,
            &HashMap::from([(variables[0].clone(), 0)]),
        )
        .unwrap(),
        gradient: sparse(source.clone(), Some(first.clone())),
        hessian: sparse(first, Some(Expr::from_i64(0))),
    };
    let mut meter = bounded();
    let error = VerifiedObjective::try_from_candidate_with_meter(
        &source, &variables, candidate, &mut meter,
    )
    .unwrap_err();
    assert!(
        matches!(
            &error,
            ObjectiveCompilationError::Compilation(error)
                | ObjectiveCompilationError::Verification(JacobianVerificationError::Compile(error))
                if normalization_refusal(error)
        ),
        "{error:?}"
    );
    assert!(meter.remaining(Dimension::ComputeSteps) > 0);
    assert_eq!(meter.verifier_remaining(), RESERVE);
}

#[test]
fn current_and_historical_raw_power_recipes_replay_without_rewriting_stored_code() {
    for historical in [false, true] {
        let constant = held_power();
        let candidate = sparse(
            angle(constant.clone()),
            Some(raw_derivative(&constant, historical)),
        );
        let retained = serde_json::to_value(&candidate).unwrap();
        let mut meter = bounded();
        verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut meter).unwrap();
        let program = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
            candidate, &mut meter,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(program.sparse_jacobian()).unwrap(),
            retained
        );
        let bytes = program.to_json_bytes().unwrap();
        let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
        assert_eq!(restored.to_json_bytes().unwrap(), bytes);
        assert_eq!(
            serde_json::to_value(restored.sparse_jacobian()).unwrap(),
            retained
        );
        assert!(meter.remaining(Dimension::ComputeSteps) > 0);
        // Compilation/replay does not promise that this huge original
        // coordinate has a finite binary64 value; do not evaluate it here.
    }
}

#[test]
fn nested_stored_current_and_historical_recipes_cannot_trigger_unbounded_reference_folding() {
    for historical in [false, true] {
        let constant = held_power();
        let argument = angle(constant.clone());
        let source = function("sin", vec![argument.clone()]);
        let derivative = Expr::Mul(vec![
            function("cos", vec![argument]),
            raw_derivative(&constant, historical),
        ]);
        let candidate = sparse(source, Some(derivative));
        let mut meter = bounded();
        let error =
            verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut meter).unwrap_err();
        assert!(
            matches!(&error, JacobianVerificationError::Compile(error) if normalization_refusal(error)),
            "{error:?}"
        );
        assert!(meter.remaining(Dimension::ComputeSteps) > 0);
        assert_eq!(meter.verifier_remaining(), RESERVE);
    }
}

#[test]
fn omitted_present_axes_are_checked_with_the_same_bounded_reference_normalizer() {
    for source in [
        angle(held_power()),
        function("sin", vec![angle(held_power())]),
    ] {
        let candidate = sparse(source, None);
        let mut meter = bounded();
        let error =
            verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut meter).unwrap_err();
        assert!(
            matches!(&error, JacobianVerificationError::Compile(error) if normalization_refusal(error)),
            "{error:?}"
        );
        assert!(meter.remaining(Dimension::ComputeSteps) > 0);
        assert_eq!(meter.verifier_remaining(), RESERVE);
    }
}

fn piecewise_source() -> Expr {
    function(
        "Piecewise",
        vec![
            function(
                "Tuple",
                vec![
                    power(Expr::symbol("x"), 2),
                    function(
                        "StrictGreaterThan",
                        vec![Expr::symbol("x"), Expr::from_i64(0)],
                    ),
                ],
            ),
            function(
                "Tuple",
                vec![
                    Expr::Mul(vec![Expr::from_i64(-1), Expr::symbol("x")]),
                    Expr::symbol("True"),
                ],
            ),
        ],
    )
}

#[test]
fn moderate_numerical_normal_forms_preserve_the_existing_symbolic_recipe() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let mapping = HashMap::from([(variables[0].clone(), 0), (variables[1].clone(), 1)]);
    let mut sources = [
        "3*x^2+4*x-7",
        "(x+1)*(x-2)",
        "(2^4)^3*x",
        "sin(x)^2+cos(x)^2",
        "exp(x)*exp(2*x)",
        "x^y",
    ]
    .map(|source| parse(source).unwrap())
    .to_vec();
    sources.push(piecewise_source());
    for source in sources {
        let mut meter = Budget::new(BudgetLimits::uniform(AMPLE, RESERVE));
        let program = CompiledResidualSystem::try_compile_with_meter(
            std::slice::from_ref(&source),
            &variables,
            &mut meter,
        )
        .unwrap();
        for (column, variable) in variables.iter().enumerate() {
            let expected = CompiledExpr::try_compile(&diff(&source, variable), &mapping).unwrap();
            assert_eq!(
                program.compiled_jacobian[column], expected,
                "{source:?}, {variable:?}"
            );
        }
        assert_eq!(meter.verifier_remaining(), RESERVE);
    }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-12 * expected.abs().max(1.0),
        "{actual} differs from {expected}"
    );
}

#[test]
fn bounded_dynamic_power_objective_retains_hand_derived_hessian() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let objective = VerifiedObjective::try_compile(&parse("x^y").unwrap(), &variables).unwrap();
    let value = objective.evaluate(&[2.0, 3.0]).unwrap();
    close(value.value, 8.0);
    for (actual, expected) in value.gradient.iter().zip([12.0, 8.0 * 2.0_f64.ln()]) {
        close(*actual, expected);
    }
    let mixed = 4.0 * (1.0 + 3.0 * 2.0_f64.ln());
    assert_eq!(value.hessian.len(), 4);
    for (actual, expected) in
        value
            .hessian
            .iter()
            .zip([12.0, mixed, mixed, 8.0 * 2.0_f64.ln().powi(2)])
    {
        close(*actual, expected);
    }
    assert!(objective.evaluate(&[-2.0, 3.0]).is_err());
}

#[test]
fn bounded_dense_piecewise_derivatives_preserve_branches_without_claiming_a_seam_proof() {
    let source = piecewise_source();
    let variables = [Symbol::new("x")];
    let piecewise =
        CompiledResidualSystem::try_compile(std::slice::from_ref(&source), &variables).unwrap();
    for (point, expected_value, expected_first) in [(2.0, 4.0, 4.0), (-2.0, 2.0, -1.0)] {
        let mut values = [f64::NAN];
        let mut jacobian = [f64::NAN];
        piecewise
            .try_eval_system(&[point], &mut values, &mut jacobian)
            .unwrap();
        close(values[0], expected_value);
        close(jacobian[0], expected_first);
    }

    // Preserving the existing dense branch recipe does not establish a
    // derivative at the branch seam. The independent objective proof lane
    // still refuses this unsupported theorem instead of admitting it on the
    // strength of the numerical generator alone.
    let error = VerifiedObjective::try_compile(&source, &variables).unwrap_err();
    assert!(
        matches!(
            error,
            ObjectiveCompilationError::Verification(JacobianVerificationError::ProofFailed {
                row: 0,
                col: 0,
                ..
            })
        ),
        "{error:?}"
    );
}

struct Interruptible {
    ledger: Budget,
    checkpoints: usize,
    stop: Option<usize>,
    stopped: bool,
}

impl Interruptible {
    fn new() -> Self {
        Self {
            ledger: Budget::new(BudgetLimits::uniform(AMPLE, RESERVE)),
            checkpoints: 0,
            stop: None,
            stopped: false,
        }
    }
}

impl BudgetMeter for Interruptible {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after terminal cancellation");
        BudgetMeter::charge(&mut self.ledger, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after terminal cancellation");
        BudgetMeter::charge_batch(&mut self.ledger, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.stopped {
            return Err(MeterError::Cancelled);
        }
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn nested_chain_rules_share_one_intrinsic_derivative_envelope() {
    let variable = Symbol::new("x");
    let mut source = Expr::Sym(variable.clone());
    for _ in 0..32 {
        source = function("sin", vec![source]);
    }
    let scalar =
        CompiledExpr::try_compile(&source, &HashMap::from([(variable.clone(), 0)])).unwrap();
    assert_eq!(scalar.ops().len(), 33);

    // A compact source can cause cumulative work across many independently
    // modest chain-rule normalizations. The whole derivative must share one
    // intrinsic envelope instead of restarting it for every recursive child.
    // An implementation that eliminates repeated work can finish inside that
    // envelope; otherwise only a typed intrinsic refusal is permitted.
    let mut meter = Interruptible::new();
    match CompiledResidualSystem::try_compile_with_meter(&[source], &[variable], &mut meter) {
        Ok(program) => {
            let mut residuals = [f64::NAN];
            let mut jacobian = [f64::NAN];
            program
                .try_eval_system(&[0.0], &mut residuals, &mut jacobian)
                .unwrap();
            close(residuals[0], 0.0);
            close(jacobian[0], 1.0);
        }
        Err(error) => assert_intrinsic_budget_refusal(&error, &meter.ledger),
    }
    assert_derivative_envelope(&meter.ledger);
    assert!(AMPLE - meter.ledger.remaining(Dimension::ComputeSteps) > 1_000_000);
}

#[test]
fn shallow_product_expansion_cannot_forward_an_intrinsic_overcharge_to_the_caller() {
    let variable = Symbol::new("x");
    let source = Expr::Mul(vec![Expr::Sym(variable.clone()); 70]);
    CompiledExpr::try_compile(&source, &HashMap::from([(variable.clone(), 0)])).unwrap();
    // This source has depth one and its predicted derivative fits the 8192
    // operation shape limit. The admitted product-rule storage/work envelope
    // exceeds the private derivative budget before any expansion can begin.
    let mut meter = Interruptible::new();
    let error = CompiledResidualSystem::try_compile_with_meter(&[source], &[variable], &mut meter)
        .unwrap_err();
    assert_intrinsic_budget_refusal(&error, &meter.ledger);
    assert_derivative_envelope(&meter.ledger);
}

fn assert_intrinsic_budget_refusal(error: &CompileError, ledger: &Budget) {
    let Some(MeterError::Budget(BudgetError::Exhausted {
        dimension,
        requested,
        remaining,
    })) = compile_meter(error)
    else {
        panic!("expected intrinsic derivative exhaustion, got {error:?}");
    };
    assert!(requested > remaining);
    assert!(ledger.remaining(dimension) > requested);
}

fn assert_derivative_envelope(ledger: &Budget) {
    // Dense source compilation and its small argument/buffer setup precede
    // the derivative envelope. Allow at most 64 KiB / 64K steps for this
    // separately charged prefix rather than duplicating buffer bookkeeping.
    for (dimension, cap, prefix) in [
        (Dimension::ComputeSteps, 20_000_000, 65_536),
        (Dimension::MemoryBytes, 64 * 1024 * 1024, 65_536),
        (Dimension::AllocationCount, 1_000_000, 1_024),
    ] {
        let used = AMPLE - ledger.remaining(dimension);
        assert!(used <= cap + prefix, "{dimension:?}: used {used}");
    }
    assert_eq!(ledger.verifier_remaining(), RESERVE);
}

#[test]
fn dense_derivative_compilation_preserves_exact_budget_and_cancellation_boundaries() {
    let source = parse("sin(x*x)+exp(x)").unwrap();
    let variables = [Symbol::new("x")];
    let mut probe = Interruptible::new();
    CompiledResidualSystem::try_compile_with_meter(
        std::slice::from_ref(&source),
        &variables,
        &mut probe,
    )
    .unwrap();
    assert!(probe.checkpoints > 3);
    for dimension in DIMENSIONS {
        let used = AMPLE - probe.ledger.remaining(dimension);
        assert!(used > 0);
        let mut limits = BudgetLimits::uniform(AMPLE, RESERVE);
        limits.dimensions[dimension.index()] = used;
        let mut exact = Budget::new(limits);
        CompiledResidualSystem::try_compile_with_meter(
            std::slice::from_ref(&source),
            &variables,
            &mut exact,
        )
        .unwrap();
        assert_eq!(exact.remaining(dimension), 0);
        assert_eq!(exact.verifier_remaining(), RESERVE);
        limits.dimensions[dimension.index()] -= 1;
        let error = CompiledResidualSystem::try_compile_with_meter(
            std::slice::from_ref(&source),
            &variables,
            &mut Budget::new(limits),
        )
        .unwrap_err();
        assert!(
            matches!(compile_meter(&error), Some(MeterError::Budget(BudgetError::Exhausted { dimension: actual, .. })) if actual == dimension),
            "{error:?}"
        );
    }
    for stop in [1, probe.checkpoints / 2, probe.checkpoints] {
        let mut cancelled = Interruptible::new();
        cancelled.stop = Some(stop);
        let error = CompiledResidualSystem::try_compile_with_meter(
            std::slice::from_ref(&source),
            &variables,
            &mut cancelled,
        )
        .unwrap_err();
        assert_eq!(compile_meter(&error), Some(MeterError::Cancelled));
        assert!(cancelled.stopped);
        assert_eq!(cancelled.checkpoints, stop);
        assert_eq!(cancelled.ledger.verifier_remaining(), RESERVE);
    }
}

#[test]
fn current_historical_and_omitted_entry_replay_share_exact_operational_limits() {
    let constant = Expr::from_i64(1);
    let candidates = [
        sparse(
            angle(constant.clone()),
            Some(raw_derivative(&constant, false)),
        ),
        sparse(
            angle(constant.clone()),
            Some(raw_derivative(&constant, true)),
        ),
        sparse(
            function("atan2", vec![Expr::symbol("t"), Expr::symbol("t")]),
            None,
        ),
    ];
    for candidate in candidates {
        let mut probe = Interruptible::new();
        verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut probe).unwrap();
        assert!(probe.checkpoints > 3);
        for dimension in DIMENSIONS {
            let used = AMPLE - probe.ledger.remaining(dimension);
            assert!(used > 0);
            let mut limits = BudgetLimits::uniform(AMPLE, RESERVE);
            limits.dimensions[dimension.index()] = used;
            let mut exact = Budget::new(limits);
            verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut exact).unwrap();
            assert_eq!(exact.remaining(dimension), 0);
            assert_eq!(exact.verifier_remaining(), RESERVE);
            limits.dimensions[dimension.index()] -= 1;
            let error = verify_sparse_jacobian_certificate_with_meter(
                &candidate,
                &[],
                &mut Budget::new(limits),
            )
            .unwrap_err();
            assert!(
                matches!(replay_meter(&error), Some(MeterError::Budget(BudgetError::Exhausted { dimension: actual, .. })) if actual == dimension),
                "{error:?}"
            );
        }
        for stop in [1, probe.checkpoints / 2, probe.checkpoints] {
            let mut cancelled = Interruptible::new();
            cancelled.stop = Some(stop);
            let error =
                verify_sparse_jacobian_certificate_with_meter(&candidate, &[], &mut cancelled)
                    .unwrap_err();
            assert_eq!(
                replay_meter(&error),
                Some(MeterError::Cancelled),
                "{error:?}"
            );
            assert!(cancelled.stopped);
            assert_eq!(cancelled.checkpoints, stop);
            assert_eq!(cancelled.ledger.verifier_remaining(), RESERVE);
        }
    }
}
