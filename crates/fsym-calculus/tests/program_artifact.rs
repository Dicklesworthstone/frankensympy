//! Portable candidates remain untrusted until separate independent admission.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded,
};
use fsym_calculus::compile::{CompiledExpr, CompiledOp, MAX_COMPILE_OPS};
use fsym_calculus::program_artifact::{
    MAX_PROGRAM_CANDIDATE_BYTES, MAX_PROGRAM_CANDIDATE_INTEGER_BITS,
    MAX_PROGRAM_CANDIDATE_JSON_DEPTH, PROGRAM_CANDIDATE_SCHEMA_VERSION, ProgramArtifactError,
    decode_objective_candidate, decode_residual_candidate, encode_objective_candidate,
    encode_residual_candidate,
};
use fsym_calculus::sparse_hessian::{ObjectiveCandidate, VerifiedObjective};
use fsym_calculus::sparse_jacobian::SparseJacobian;
use fsym_calculus::verified::{MAX_VERIFIED_SOURCE_DEPTH, VerifiedResidualSystem};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity};
use serde::Serialize;

const AMPLE: u64 = 1 << 40;
const DIMENSIONS: [Dimension; 3] = [
    Dimension::ComputeSteps,
    Dimension::MemoryBytes,
    Dimension::AllocationCount,
];

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn residual_fixture() -> VerifiedResidualSystem {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let y = Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([9; 32]));
    let residuals = [
        Expr::Add(vec![
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
            Expr::Sym(y.clone()),
        ]),
        Expr::Add(vec![
            Expr::Function("sin".into(), vec![Expr::Sym(y.clone())]),
            Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
        ]),
        Expr::from_i64(7),
    ];
    VerifiedResidualSystem::try_compile(&residuals, &[y, x]).unwrap()
}

fn objective_fixture() -> VerifiedObjective {
    let source = fsym_core::parse("x^2*y+3*y^2").unwrap();
    VerifiedObjective::try_compile(&source, &[Symbol::new("y"), Symbol::new("x")]).unwrap()
}

#[derive(Serialize)]
struct Envelope<'a, T> {
    schema_version: u32,
    kind: &'static str,
    candidate: &'a T,
}

fn residual_bytes(candidate: &SparseJacobian) -> Vec<u8> {
    serde_json::to_vec(&Envelope {
        schema_version: PROGRAM_CANDIDATE_SCHEMA_VERSION,
        kind: "residual",
        candidate,
    })
    .unwrap()
}

fn objective_bytes(candidate: &ObjectiveCandidate) -> Vec<u8> {
    serde_json::to_vec(&Envelope {
        schema_version: PROGRAM_CANDIDATE_SCHEMA_VERSION,
        kind: "objective",
        candidate,
    })
    .unwrap()
}

fn constant_bytes() -> Vec<u8> {
    let program = VerifiedResidualSystem::try_compile(&[Expr::from_i64(0)], &[]).unwrap();
    encode_residual_candidate(&program, &mut Unbounded).unwrap()
}

fn replace_constant(source: &str) -> Vec<u8> {
    let original = String::from_utf8(constant_bytes()).unwrap();
    let anchor = r#""residuals":[{"Integer":[0,[]]}]"#;
    assert_eq!(original.matches(anchor).count(), 1);
    original
        .replace(anchor, &format!(r#""residuals":[{source}]"#))
        .into_bytes()
}

#[test]
fn residual_round_trip_preserves_typed_axes_proofs_bytecode_and_all_numeric_products() {
    let original = residual_fixture();
    let bytes = encode_residual_candidate(&original, &mut Unbounded).unwrap();
    let candidate = decode_residual_candidate(&bytes, &mut Unbounded).unwrap();
    assert_eq!(&candidate, original.sparse_jacobian());
    assert!(
        !String::from_utf8(bytes.clone())
            .unwrap()
            .contains("receipt")
    );
    assert!(
        !String::from_utf8(bytes.clone())
            .unwrap()
            .contains("shared_evaluation")
    );
    let restored = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
        candidate,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(restored.vars(), original.vars());
    assert_eq!(restored.receipt(), original.receipt());
    assert_eq!(restored.residual_programs(), original.residual_programs());
    assert_eq!(
        encode_residual_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
    let values = restored
        .evaluate_sparse_with_meter(&[3.0, 2.0], &mut Unbounded)
        .unwrap();
    assert_eq!(values.residuals[0], 7.0);
    close(values.residuals[1], 6.0 + 3.0f64.sin());
    assert_eq!(values.residuals[2], 7.0);
    assert_eq!(
        restored.jacobian_pattern(),
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    );
    for (&actual, expected) in values
        .jacobian
        .iter()
        .zip([1.0, 4.0, 2.0 + 3.0f64.cos(), 3.0])
    {
        close(actual, expected);
    }
    let forward = restored
        .jacobian_vector_product(&[3.0, 2.0], &[2.0, -1.0])
        .unwrap();
    close(forward[0], -2.0);
    close(forward[1], 1.0 + 2.0 * 3.0f64.cos());
    assert_eq!(forward[2], 0.0);
    let transpose = restored
        .jacobian_transpose_vector_product(&[3.0, 2.0], &[2.0, -3.0, 11.0])
        .unwrap();
    close(transpose[0], -4.0 - 3.0 * 3.0f64.cos());
    close(transpose[1], -1.0);
}

#[test]
fn objective_round_trip_preserves_gradient_hessian_products_and_exact_candidate() {
    let original = objective_fixture();
    let bytes = encode_objective_candidate(&original, &mut Unbounded).unwrap();
    let candidate = decode_objective_candidate(&bytes, &mut Unbounded).unwrap();
    assert_eq!(&candidate, original.candidate());
    let restored = VerifiedObjective::try_from_candidate_with_meter(
        original.objective(),
        original.variables(),
        candidate,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(restored.receipt(), original.receipt());
    assert_eq!(
        encode_objective_candidate(&restored, &mut Unbounded).unwrap(),
        bytes
    );
    let result = restored.evaluate(&[2.0, 3.0]).unwrap();
    assert_eq!(result.value, 30.0);
    assert_eq!(result.gradient, [21.0, 12.0]);
    assert_eq!(restored.hessian_pattern(), [(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(result.hessian, [6.0, 6.0, 6.0, 4.0]);
    assert_eq!(
        restored
            .hessian_vector_product(&[2.0, 3.0], &[2.0, -3.0])
            .unwrap(),
        [-6.0, 0.0]
    );
}

#[test]
fn imported_programs_keep_original_domains_even_for_zero_numeric_directions() {
    let x = Symbol::new("x");
    let source = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(-1)),
    ]);
    let residual = VerifiedResidualSystem::try_compile(
        std::slice::from_ref(&source),
        std::slice::from_ref(&x),
    )
    .unwrap();
    let bytes = encode_residual_candidate(&residual, &mut Unbounded).unwrap();
    let candidate = decode_residual_candidate(&bytes, &mut Unbounded).unwrap();
    let residual = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
        candidate,
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
    assert!(
        residual
            .evaluate_sparse_with_meter(&[0.0], &mut Unbounded)
            .is_err()
    );
    assert!(residual.jacobian_vector_product(&[0.0], &[0.0]).is_err());
    assert!(
        residual
            .jacobian_transpose_vector_product(&[0.0], &[0.0])
            .is_err()
    );
    let objective = VerifiedObjective::try_compile(&source, std::slice::from_ref(&x)).unwrap();
    let bytes = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
    let candidate = decode_objective_candidate(&bytes, &mut Unbounded).unwrap();
    let objective =
        VerifiedObjective::try_from_candidate_with_meter(&source, &[x], candidate, &mut Unbounded)
            .unwrap();
    assert_eq!(objective.evaluate(&[1.0]).unwrap().value, 0.0);
    assert!(objective.evaluate(&[0.0]).is_err());
    assert!(objective.hessian_vector_product(&[0.0], &[0.0]).is_err());
}

#[test]
fn decoding_never_grants_math_authority_to_forged_proofs_bytecode_or_sparse_omissions() {
    let original = residual_fixture();
    let mut bytecode = original.sparse_jacobian().clone();
    bytecode.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    let mut source = original.sparse_jacobian().clone();
    source.residuals[0] = Expr::from_i64(0);
    let mut omitted = original.sparse_jacobian().clone();
    omitted.entries.remove(0);
    omitted.pattern.nonzeros.remove(0);
    omitted.pattern.nnz -= 1;
    for mutant in [bytecode, source, omitted] {
        let candidate =
            decode_residual_candidate(&residual_bytes(&mutant), &mut Unbounded).unwrap();
        assert!(
            VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
                candidate,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let original = objective_fixture();
    let mut candidate = original.candidate().clone();
    candidate.compiled_objective =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    let candidate =
        decode_objective_candidate(&objective_bytes(&candidate), &mut Unbounded).unwrap();
    assert!(
        VerifiedObjective::try_from_candidate_with_meter(
            original.objective(),
            original.variables(),
            candidate,
            &mut Unbounded
        )
        .is_err()
    );
    let candidate = decode_objective_candidate(
        &encode_objective_candidate(&original, &mut Unbounded).unwrap(),
        &mut Unbounded,
    )
    .unwrap();
    assert!(
        VerifiedObjective::try_from_candidate_with_meter(
            &Expr::from_i64(0),
            original.variables(),
            candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn schemas_kinds_duplicates_unknown_fields_and_noncanonical_json_refuse() {
    let bytes = constant_bytes();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(
        decode_residual_candidate(&serde_json::to_vec_pretty(&value).unwrap(), &mut Unbounded),
        Err(ProgramArtifactError::NonCanonical)
    ));
    let future = br#"{"schema_version":2,"kind":"residual","candidate":null}"#;
    assert!(matches!(
        decode_residual_candidate(future, &mut Unbounded),
        Err(ProgramArtifactError::UnsupportedVersion { found: 2, .. })
    ));
    assert!(matches!(
        decode_objective_candidate(&bytes, &mut Unbounded),
        Err(ProgramArtifactError::WrongKind {
            expected: "objective"
        })
    ));
    let original = String::from_utf8(bytes).unwrap();
    let duplicate = original.replacen(
        "\"schema_version\":1",
        "\"schema_version\":1,\"schema_version\":1",
        1,
    );
    assert!(matches!(
        decode_residual_candidate(duplicate.as_bytes(), &mut Unbounded),
        Err(ProgramArtifactError::Encoding(_))
    ));
    for extra in [
        "\"receipt\":{},",
        "\"verified\":true,",
        "\"shared_evaluation\":{},",
    ] {
        let forged = original.replacen("\"candidate\":{", &format!("\"candidate\":{{{extra}"), 1);
        assert!(matches!(
            decode_residual_candidate(forged.as_bytes(), &mut Unbounded),
            Err(ProgramArtifactError::Encoding(_))
        ));
    }
    let mut trailing = original.into_bytes();
    trailing.extend_from_slice(b"null");
    assert!(matches!(
        decode_residual_candidate(&trailing, &mut Unbounded),
        Err(ProgramArtifactError::Encoding(_))
    ));
}

#[test]
fn scalar_preflight_rejects_excess_limbs_zero_denominators_and_nonreduced_rationals() {
    let excess = MAX_PROGRAM_CANDIDATE_INTEGER_BITS.div_ceil(32) + 1;
    let digits = vec!["1"; excess].join(",");
    let bytes = replace_constant(&format!(r#"{{"Integer":[1,[{digits}]]}}"#));
    assert!(matches!(
        decode_residual_candidate(&bytes, &mut Unbounded),
        Err(ProgramArtifactError::ResourceLimit {
            resource: "integer limbs",
            ..
        })
    ));
    for source in [
        r#"{"Integer":[1,[]]}"#,
        r#"{"Integer":[0,[1]]}"#,
        r#"{"Integer":[1,[1,0]]}"#,
        r#"{"Rational":[[1,[1]],[0,[]]]}"#,
        r#"{"Rational":[[1,[1]],[-1,[2]]]}"#,
        r#"{"Rational":[[1,[2]],[1,[4]]]}"#,
    ] {
        let result = decode_residual_candidate(&replace_constant(source), &mut Unbounded);
        assert!(
            matches!(result, Err(ProgramArtifactError::NonCanonical)),
            "{source}: {result:?}"
        );
    }
    let canonical = replace_constant(r#"{"Rational":[[1,[1]],[1,[2]]]}"#);
    let candidate = decode_residual_candidate(&canonical, &mut Unbounded).unwrap();
    let accepted = VerifiedResidualSystem::try_from_owned_sparse_jacobian_with_meter(
        candidate,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        accepted
            .evaluate_sparse_with_meter(&[], &mut Unbounded)
            .unwrap()
            .residuals,
        [0.5]
    );
}

#[test]
fn shape_depth_proof_and_bytecode_bounds_precede_full_candidate_materialization() {
    let mut deep = r#"{"Integer":[0,[]]}"#.to_owned();
    for _ in 0..=MAX_VERIFIED_SOURCE_DEPTH {
        deep = format!(r#"{{"Add":[{deep}]}}"#);
    }
    assert!(matches!(
        decode_residual_candidate(&replace_constant(&deep), &mut Unbounded),
        Err(ProgramArtifactError::ResourceLimit {
            resource: "expression depth",
            ..
        })
    ));
    let bytes = String::from_utf8(constant_bytes())
        .unwrap()
        .replace("\"num_rows\":1", "\"num_rows\":65537");
    assert!(matches!(
        decode_residual_candidate(bytes.as_bytes(), &mut Unbounded),
        Err(ProgramArtifactError::ResourceLimit {
            resource: "residual axis",
            ..
        })
    ));
    let bytes = encode_residual_candidate(&residual_fixture(), &mut Unbounded).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let steps = value["candidate"]["entries"][0]["derivation"]["steps"]
        .as_array_mut()
        .unwrap();
    steps.push(steps[0].clone());
    assert!(matches!(
        decode_residual_candidate(&serde_json::to_vec(&value).unwrap(), &mut Unbounded),
        Err(ProgramArtifactError::ResourceLimit {
            resource: "derivative proof steps",
            ..
        })
    ));
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["candidate"]["entries"][0]["compiled_deriv"]["ops"] =
        serde_json::json!(vec!["Neg"; MAX_COMPILE_OPS + 1]);
    assert!(matches!(
        decode_residual_candidate(&serde_json::to_vec(&value).unwrap(), &mut Unbounded),
        Err(ProgramArtifactError::ResourceLimit {
            resource: "bytecode operations",
            ..
        })
    ));
    let too_deep = format!(
        "{}0{}",
        "[".repeat(MAX_PROGRAM_CANDIDATE_JSON_DEPTH + 1),
        "]".repeat(MAX_PROGRAM_CANDIDATE_JSON_DEPTH + 1)
    );
    assert!(matches!(
        decode_residual_candidate(too_deep.as_bytes(), &mut Unbounded),
        Err(ProgramArtifactError::DepthLimit { .. })
    ));
    assert!(matches!(
        decode_residual_candidate(&vec![b' '; MAX_PROGRAM_CANDIDATE_BYTES + 1], &mut Unbounded),
        Err(ProgramArtifactError::ByteLimit { .. })
    ));
}

fn budget(ordinary: [u64; 3]) -> Budget {
    let mut limits = BudgetLimits::uniform(AMPLE, 777);
    for (dimension, amount) in DIMENSIONS.into_iter().zip(ordinary) {
        limits.dimensions[dimension.index()] = amount;
    }
    Budget::new(limits)
}

struct CancelAfter {
    ledger: Budget,
    checkpoints: usize,
}

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.ledger.charge(dimension, amount)
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.ledger.charge_batch(charges)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints >= 5 {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn transport_consumes_cumulative_ordinary_budget_and_refuses_exhaustion_or_cancellation() {
    let residual = residual_fixture();
    let objective = objective_fixture();
    let bytes = encode_residual_candidate(&residual, &mut Unbounded).unwrap();
    let objective_bytes = encode_objective_candidate(&objective, &mut Unbounded).unwrap();
    let mut ledger = budget([AMPLE; 3]);
    encode_residual_candidate(&residual, &mut ledger).unwrap();
    decode_residual_candidate(&bytes, &mut ledger).unwrap();
    encode_objective_candidate(&objective, &mut ledger).unwrap();
    decode_objective_candidate(&objective_bytes, &mut ledger).unwrap();
    let before = DIMENSIONS.map(|dimension| ledger.remaining(dimension));
    decode_residual_candidate(&bytes, &mut ledger).unwrap();
    for (dimension, previous) in DIMENSIONS.into_iter().zip(before) {
        assert!(ledger.remaining(dimension) < previous);
    }
    assert_eq!(ledger.verifier_remaining(), 777);
    for dimension in DIMENSIONS {
        let mut ordinary = [AMPLE; 3];
        ordinary[dimension.index()] = 0;
        for error in [
            encode_residual_candidate(&residual, &mut budget(ordinary)).unwrap_err(),
            decode_residual_candidate(&bytes, &mut budget(ordinary)).unwrap_err(),
            encode_objective_candidate(&objective, &mut budget(ordinary)).unwrap_err(),
            decode_objective_candidate(&objective_bytes, &mut budget(ordinary)).unwrap_err(),
        ] {
            assert!(
                matches!(error, ProgramArtifactError::Budget(MeterError::Budget(BudgetError::Exhausted { dimension: actual, .. })) if actual == dimension)
            );
        }
    }
    let mut cancelled = CancelAfter {
        ledger: budget([AMPLE; 3]),
        checkpoints: 0,
    };
    assert!(matches!(
        encode_residual_candidate(&residual, &mut cancelled),
        Err(ProgramArtifactError::Budget(MeterError::Cancelled))
    ));
    assert!(cancelled.ledger.remaining(Dimension::ComputeSteps) < AMPLE);
    let mut cancelled = CancelAfter {
        ledger: budget([AMPLE; 3]),
        checkpoints: 0,
    };
    assert!(matches!(
        decode_residual_candidate(&bytes, &mut cancelled),
        Err(ProgramArtifactError::Budget(MeterError::Cancelled))
    ));
    assert!(cancelled.ledger.remaining(Dimension::ComputeSteps) < AMPLE);
}

#[test]
fn objective_export_refuses_nonfinite_retained_bytecode() {
    let program = VerifiedObjective::try_compile(&Expr::Const(Constant::Infinity), &[]).unwrap();
    assert!(matches!(
        encode_objective_candidate(&program, &mut Unbounded),
        Err(ProgramArtifactError::NonFiniteBytecode)
    ));
}

#[test]
fn export_checks_retained_rational_height_before_any_output_allocation() {
    // A quotient can round to finite zero even when a component exceeds the
    // narrower portable scalar profile, so compilation alone is no such bound.
    let denominator =
        BigInt::from(1u32) << u32::try_from(MAX_PROGRAM_CANDIDATE_INTEGER_BITS + 1).unwrap();
    let source = Expr::Rational(BigRational::new(BigInt::from(1u32), denominator));
    let program = VerifiedResidualSystem::try_compile(&[source], &[]).unwrap();
    let mut ledger = budget([AMPLE, 0, 0]);
    assert!(matches!(
        encode_residual_candidate(&program, &mut ledger),
        Err(ProgramArtifactError::ResourceLimit {
            resource: "rational bits",
            ..
        })
    ));
    assert_eq!(ledger.remaining(Dimension::MemoryBytes), 0);
    assert_eq!(ledger.remaining(Dimension::AllocationCount), 0);
}
