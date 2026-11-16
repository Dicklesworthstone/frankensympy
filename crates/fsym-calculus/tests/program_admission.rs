//! Source-bound admission of portable programs into a receiving caller.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded,
};
use fsym_calculus::compile::{CompiledExpr, CompiledOp};
use fsym_calculus::program_artifact::{
    MAX_PROGRAM_CANDIDATE_INTEGER_BITS, MAX_PROGRAM_CANDIDATE_NAME_BYTES, ProgramAdmissionError,
    ProgramArtifactError, admit_objective_candidate, admit_residual_candidate,
    decode_objective_candidate, decode_residual_candidate, encode_objective_candidate,
    encode_residual_candidate,
};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::SparseJacobian;
use fsym_calculus::verified::{MAX_VERIFIED_SOURCE_DEPTH, VerifiedResidualSystem};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_proof_kernel::Claim;
use std::error::Error;

const AMPLE: u64 = 1 << 40;
const DIMENSIONS: [Dimension; 3] = [
    Dimension::ComputeSteps,
    Dimension::MemoryBytes,
    Dimension::AllocationCount,
];

fn axes() -> [Symbol; 2] {
    [
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32])),
        Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([7; 32])),
    ]
}

fn sources(variables: &[Symbol; 2]) -> [Expr; 2] {
    [
        Expr::Add(vec![
            Expr::pow(Expr::Sym(variables[0].clone()), Expr::from_i64(2)),
            Expr::Sym(variables[1].clone()),
        ]),
        Expr::from_i64(5),
    ]
}

fn objective(variables: &[Symbol; 2]) -> Expr {
    let x = Expr::Sym(variables[0].clone());
    let y = Expr::Sym(variables[1].clone());
    Expr::Add(vec![
        Expr::pow(x.clone(), Expr::from_i64(2)),
        Expr::Mul(vec![x, y.clone()]),
        Expr::Mul(vec![Expr::from_i64(3), Expr::pow(y, Expr::from_i64(2))]),
    ])
}

fn residual_candidate(residuals: &[Expr], variables: &[Symbol]) -> SparseJacobian {
    let original = VerifiedResidualSystem::try_compile(residuals, variables).unwrap();
    let bytes = encode_residual_candidate(&original, &mut Unbounded).unwrap();
    decode_residual_candidate(&bytes, &mut Unbounded).unwrap()
}

fn objective_candidate(source: &Expr, variables: &[Symbol]) -> ObjectiveCandidate {
    let original = VerifiedObjective::try_compile(source, variables).unwrap();
    let bytes = encode_objective_candidate(&original, &mut Unbounded).unwrap();
    decode_objective_candidate(&bytes, &mut Unbounded).unwrap()
}

#[test]
fn receiving_residual_source_binds_and_all_sparse_products_execute() {
    let variables = axes();
    let residuals = sources(&variables);
    let candidate = residual_candidate(&residuals, &variables);
    let original_proposal = candidate.clone();
    let restored =
        admit_residual_candidate(&residuals, &variables, candidate, &mut Unbounded).unwrap();
    assert_eq!(restored.sparse_jacobian(), &original_proposal);
    let values = restored
        .evaluate_sparse_with_meter(&[2.0, 3.0], &mut Unbounded)
        .unwrap();
    assert_eq!(values.residuals, [7.0, 5.0]);
    assert_eq!(values.jacobian, [4.0, 1.0]);
    assert_eq!(restored.jacobian_pattern(), [(0, 0), (0, 1)]);
    assert_eq!(
        restored
            .jacobian_vector_product(&[2.0, 3.0], &[3.0, -2.0])
            .unwrap(),
        [10.0, 0.0]
    );
    assert_eq!(
        restored
            .jacobian_transpose_vector_product(&[2.0, 3.0], &[2.0, 11.0])
            .unwrap(),
        [8.0, 2.0]
    );
}

#[test]
fn receiving_objective_binds_and_gradient_hessian_products_execute() {
    let variables = axes();
    let source = objective(&variables);
    let candidate = objective_candidate(&source, &variables);
    let original_proposal = candidate.clone();
    let restored =
        admit_objective_candidate(&source, &variables, candidate, &mut Unbounded).unwrap();
    assert_eq!(restored.candidate(), &original_proposal);
    let values = restored.evaluate(&[2.0, 3.0]).unwrap();
    assert_eq!(values.value, 37.0);
    assert_eq!(values.gradient, [7.0, 20.0]);
    assert_eq!(values.hessian, [2.0, 1.0, 1.0, 6.0]);
    assert_eq!(
        restored
            .hessian_vector_product(&[2.0, 3.0], &[3.0, -2.0])
            .unwrap(),
        [4.0, -9.0]
    );
}

#[test]
fn valid_proofs_for_a_different_source_do_not_authorize_replay() {
    let variables = axes();
    let original = sources(&variables);
    let changed = [original[0].clone(), Expr::from_i64(6)];
    assert!(matches!(
        admit_residual_candidate(
            &original,
            &variables,
            residual_candidate(&changed, &variables),
            &mut Unbounded
        ),
        Err(ProgramAdmissionError::SourceMismatch(
            "ordered residual sources"
        ))
    ));
    let source = objective(&variables);
    let changed = Expr::Add(vec![source.clone(), Expr::from_i64(1)]);
    assert!(matches!(
        admit_objective_candidate(
            &source,
            &variables,
            objective_candidate(&changed, &variables),
            &mut Unbounded
        ),
        Err(ProgramAdmissionError::SourceMismatch(
            "ordered residual sources"
        ))
    ));
}

#[test]
fn typed_axis_identity_order_count_and_unused_axes_are_bound() {
    let variables = axes();
    // A constant has valid zero derivatives in every universe. It makes a
    // coordinated axis substitution mathematically valid, but not the source
    // requested by this receiving caller.
    let source = Expr::from_i64(3);
    let changed = [
        variables.iter().rev().cloned().collect::<Vec<_>>(),
        vec![Symbol::new("x"), variables[1].clone()],
        vec![
            Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32])),
            variables[1].clone(),
        ],
        vec![variables[0].clone()],
    ];
    for candidate_variables in changed {
        let residual = residual_candidate(std::slice::from_ref(&source), &candidate_variables);
        assert!(matches!(
            admit_residual_candidate(
                std::slice::from_ref(&source),
                &variables,
                residual,
                &mut Unbounded
            ),
            Err(ProgramAdmissionError::SourceMismatch(
                "ordered typed variables"
            ))
        ));
        let candidate = objective_candidate(&source, &candidate_variables);
        assert!(matches!(
            admit_objective_candidate(&source, &variables, candidate, &mut Unbounded),
            Err(ProgramAdmissionError::SourceMismatch(
                "ordered typed variables"
            ))
        ));
    }
}

#[test]
fn raw_source_order_multiplicity_and_scalar_variants_cannot_be_normalized_away() {
    let variables = axes();
    let x = Expr::Sym(variables[0].clone());
    let y = Expr::Sym(variables[1].clone());
    for (expected, changed) in [
        (
            Expr::Add(vec![x.clone(), y.clone()]),
            Expr::Add(vec![y, x.clone()]),
        ),
        (
            Expr::Add(vec![x.clone(), x.clone()]),
            Expr::Mul(vec![Expr::from_i64(2), x]),
        ),
        (
            Expr::Integer(BigInt::from(1)),
            Expr::Rational(BigRational::from_integer(BigInt::from(1))),
        ),
    ] {
        assert!(matches!(
            admit_residual_candidate(
                std::slice::from_ref(&expected),
                &variables,
                residual_candidate(std::slice::from_ref(&changed), &variables),
                &mut Unbounded
            ),
            Err(ProgramAdmissionError::SourceMismatch(
                "ordered residual sources"
            ))
        ));
        assert!(matches!(
            admit_objective_candidate(
                &expected,
                &variables,
                objective_candidate(&changed, &variables),
                &mut Unbounded
            ),
            Err(ProgramAdmissionError::SourceMismatch(
                "ordered residual sources"
            ))
        ));
    }
}

#[test]
fn original_held_poles_survive_admission_and_zero_direction_evaluation() {
    let variable = Symbol::new("x");
    let variables = std::slice::from_ref(&variable);
    let source = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::pow(Expr::Sym(variable.clone()), Expr::from_i64(-1)),
    ]);
    let sources = std::slice::from_ref(&source);
    let residual = admit_residual_candidate(
        sources,
        variables,
        residual_candidate(sources, variables),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        residual
            .evaluate_sparse_with_meter(&[1.0], &mut Unbounded)
            .unwrap()
            .residuals,
        [0.0]
    );
    assert!(residual.jacobian_vector_product(&[0.0], &[0.0]).is_err());
    assert!(
        residual
            .jacobian_transpose_vector_product(&[0.0], &[0.0])
            .is_err()
    );
    let objective = admit_objective_candidate(
        &source,
        variables,
        objective_candidate(&source, variables),
        &mut Unbounded,
    )
    .unwrap();
    assert!(objective.hessian_vector_product(&[0.0], &[0.0]).is_err());
    assert!(matches!(
        admit_residual_candidate(
            sources,
            variables,
            residual_candidate(&[Expr::from_i64(0)], variables),
            &mut Unbounded
        ),
        Err(ProgramAdmissionError::SourceMismatch(_))
    ));
}

#[test]
fn independent_verifier_rejects_source_matching_derivative_proof_bytecode_and_omission_mutants() {
    let variables = axes();
    let residuals = sources(&variables);
    let original = residual_candidate(&residuals, &variables);
    let mut derivative = original.clone();
    derivative.entries[0].symbolic_deriv = Expr::from_i64(0);
    let mut bytecode = original.clone();
    bytecode.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    let mut proof = original.clone();
    let Claim::Equality { rhs, .. } = &mut proof.entries[0].derivation.steps[0].claim else {
        panic!("fixture must carry an equality");
    };
    *rhs = Expr::from_i64(0);
    let mut omitted = original;
    omitted.entries.remove(0);
    omitted.pattern.nonzeros.remove(0);
    omitted.pattern.nnz -= 1;
    for mutant in [derivative, bytecode, proof, omitted] {
        assert!(matches!(
            admit_residual_candidate(&residuals, &variables, mutant, &mut Unbounded),
            Err(ProgramAdmissionError::Residual(_))
        ));
    }
    let source = objective(&variables);
    let original = objective_candidate(&source, &variables);
    let mut objective_bytecode = original.clone();
    objective_bytecode.compiled_objective =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    let mut hessian = original;
    hessian.hessian.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    for mutant in [objective_bytecode, hessian] {
        assert!(matches!(
            admit_objective_candidate(&source, &variables, mutant, &mut Unbounded),
            Err(ProgramAdmissionError::Objective(_))
        ));
    }
}

#[test]
fn caller_source_is_preflighted_without_trusting_candidate_decode_bounds() {
    let zero = Expr::from_i64(0);
    let candidate = residual_candidate(std::slice::from_ref(&zero), &[]);
    let mut deep = zero;
    for _ in 0..=MAX_VERIFIED_SOURCE_DEPTH {
        deep = Expr::Add(vec![deep]);
    }
    let large = Expr::Integer(
        BigInt::from(1) << u32::try_from(MAX_PROGRAM_CANDIDATE_INTEGER_BITS + 1).unwrap(),
    );
    for (source, resource) in [(deep, "expression depth"), (large, "integer bits")] {
        assert!(matches!(
            admit_residual_candidate(&[source], &[], candidate.clone(), &mut Unbounded),
            Err(ProgramAdmissionError::Transport(ProgramArtifactError::ResourceLimit {
                resource: actual, ..
            })) if actual == resource
        ));
    }
    let variables = [Symbol::new("x")];
    let source = Expr::from_i64(1);
    let candidate = residual_candidate(std::slice::from_ref(&source), &variables);
    assert!(matches!(
        admit_residual_candidate(
            &[source],
            &[Symbol::new(
                "x".repeat(MAX_PROGRAM_CANDIDATE_NAME_BYTES + 1)
            )],
            candidate,
            &mut Unbounded
        ),
        Err(ProgramAdmissionError::Transport(
            ProgramArtifactError::ResourceLimit {
                resource: "name bytes",
                ..
            }
        ))
    ));
}

fn ledger() -> Budget {
    Budget::new(BudgetLimits::uniform(AMPLE, AMPLE))
}

fn meter_error(error: &ProgramAdmissionError) -> Option<MeterError> {
    let mut current: &(dyn Error + 'static) = error;
    loop {
        if let Some(error) = current.downcast_ref::<MeterError>() {
            return Some(*error);
        }
        current = current.source()?;
    }
}

#[test]
fn decode_and_independent_admission_share_a_ledger_with_protected_verifier_compute() {
    let variables = axes();
    let residuals = sources(&variables);
    let source = objective(&variables);
    let original = VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap();
    let objective = VerifiedObjective::try_compile(&source, &variables).unwrap();
    let bytes = encode_residual_candidate(&original, &mut Unbounded).unwrap();
    let objective_bytes = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
    let mut budget = ledger();
    let residual_candidate = decode_residual_candidate(&bytes, &mut budget).unwrap();
    let objective_candidate = decode_objective_candidate(&objective_bytes, &mut budget).unwrap();
    assert_eq!(budget.verifier_remaining(), AMPLE);
    let remaining = budget.remaining(Dimension::ComputeSteps);
    budget.charge(Dimension::ComputeSteps, remaining).unwrap();
    let lease = budget.verifier_lease().unwrap();
    for _ in 0..2 {
        let before = budget.verifier_remaining();
        admit_residual_candidate(
            &residuals,
            &variables,
            residual_candidate.clone(),
            &mut budget.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
        admit_objective_candidate(
            &source,
            &variables,
            objective_candidate.clone(),
            &mut budget.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
        assert!(budget.verifier_remaining() < before);
        assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    }
}

#[test]
fn every_resource_dimension_refuses_at_the_measured_cumulative_boundary() {
    let variables = axes();
    let residuals = sources(&variables);
    let source = objective(&variables);
    let residual = residual_candidate(&residuals, &variables);
    let objective = objective_candidate(&source, &variables);
    let mut residual_budget = ledger();
    admit_residual_candidate(
        &residuals,
        &variables,
        residual.clone(),
        &mut residual_budget,
    )
    .unwrap();
    let mut objective_budget = ledger();
    admit_objective_candidate(
        &source,
        &variables,
        objective.clone(),
        &mut objective_budget,
    )
    .unwrap();
    for dimension in DIMENSIONS {
        let mut limits = BudgetLimits::uniform(AMPLE, AMPLE);
        limits.dimensions[dimension.index()] = AMPLE - residual_budget.remaining(dimension) - 1;
        let error = admit_residual_candidate(
            &residuals,
            &variables,
            residual.clone(),
            &mut Budget::new(limits),
        )
        .unwrap_err();
        assert!(
            matches!(meter_error(&error), Some(MeterError::Budget(BudgetError::Exhausted {
            dimension: actual, ..
        })) if actual == dimension),
            "{error:?}"
        );
        let mut limits = BudgetLimits::uniform(AMPLE, AMPLE);
        limits.dimensions[dimension.index()] = AMPLE - objective_budget.remaining(dimension) - 1;
        let error = admit_objective_candidate(
            &source,
            &variables,
            objective.clone(),
            &mut Budget::new(limits),
        )
        .unwrap_err();
        assert!(
            matches!(meter_error(&error), Some(MeterError::Budget(BudgetError::Exhausted {
            dimension: actual, ..
        })) if actual == dimension),
            "{error:?}"
        );
    }
}

struct InterruptAt {
    checkpoints: usize,
    cancel_at: usize,
}

impl BudgetMeter for InterruptAt {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints >= self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_before_binding_during_verification_and_at_publication_stays_typed() {
    let variables = [Symbol::new("x")];
    let source = Expr::pow(Expr::Sym(variables[0].clone()), Expr::from_i64(2));
    let residuals = std::slice::from_ref(&source);
    let residual = residual_candidate(residuals, &variables);
    let objective = objective_candidate(&source, &variables);
    for objective_lane in [false, true] {
        let run = |meter: &mut InterruptAt| {
            if objective_lane {
                admit_objective_candidate(&source, &variables, objective.clone(), meter).map(|_| ())
            } else {
                admit_residual_candidate(residuals, &variables, residual.clone(), meter).map(|_| ())
            }
        };
        let mut count = InterruptAt {
            checkpoints: 0,
            cancel_at: usize::MAX,
        };
        run(&mut count).unwrap();
        assert!(count.checkpoints > 10);
        for cancel_at in [
            1,
            3,
            10,
            count.checkpoints / 2,
            count.checkpoints - 1,
            count.checkpoints,
        ] {
            let mut interrupted = InterruptAt {
                checkpoints: 0,
                cancel_at,
            };
            let error = run(&mut interrupted).unwrap_err();
            assert_eq!(
                meter_error(&error),
                Some(MeterError::Cancelled),
                "{error:?}"
            );
        }
    }
}
