use fsym_calculus::compile::{CompiledExpr, CompiledOp};
use fsym_calculus::sparse_jacobian::{SparseJacobian, SparsityPattern};
use fsym_calculus::verified::{
    MAX_VERIFIED_ARTIFACT_BYTES, MAX_VERIFIED_ARTIFACT_JSON_DEPTH, MAX_VERIFIED_JACOBIAN_ENTRIES,
    MAX_VERIFIED_SOURCE_DEPTH, VERIFIED_ARTIFACT_SCHEMA_VERSION, VerifiedArtifactError,
    VerifiedCompilationError, VerifiedResidualSystem,
};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};

fn fixture() -> VerifiedResidualSystem {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let residuals = vec![
        Expr::Add(vec![
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
            Expr::Sym(y.clone()),
        ]),
        Expr::Add(vec![
            Expr::pow(Expr::Sym(y.clone()), Expr::from_i64(2)),
            Expr::from_i64(-2),
        ]),
    ];
    VerifiedResidualSystem::try_compile(&residuals, &[x, y]).unwrap()
}

fn encode_untrusted(jacobian: &SparseJacobian) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "schema_version": VERIFIED_ARTIFACT_SCHEMA_VERSION,
        "jacobian": jacobian,
    }))
    .unwrap()
}

#[test]
fn complete_proof_and_exact_executed_bytecode_survive_transport() {
    let original = fixture();
    let encoded = original.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&encoded).unwrap();
    assert_eq!(restored.sparse_jacobian(), original.sparse_jacobian());
    assert_eq!(restored.residuals(), original.residuals());
    assert_eq!(restored.vars(), original.vars());
    assert_eq!(restored.receipt(), original.receipt());
    assert_eq!(restored.receipt().test_points_evaluated, 0);
    assert_eq!(
        restored.compiled().compiled_residuals,
        original.compiled().compiled_residuals
    );
    assert_eq!(
        restored.compiled().compiled_jacobian,
        original.compiled().compiled_jacobian
    );
    let mut residuals = [0.0; 2];
    let mut jacobian = [0.0; 4];
    restored
        .compiled()
        .try_eval_system(&[2.0, 3.0], &mut residuals, &mut jacobian)
        .unwrap();
    assert_eq!(residuals, [7.0, 7.0]);
    assert_eq!(jacobian, [4.0, 1.0, 0.0, 6.0]);
    assert_eq!(restored.to_json_bytes().unwrap(), encoded);
}

#[test]
fn decoding_rechecks_proofs_sources_sparsity_and_bytecode() {
    let original = fixture();
    let mut mutants = Vec::new();
    let mut bytecode = original.sparse_jacobian().clone();
    bytecode.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    mutants.push(bytecode);
    let mut proof = original.sparse_jacobian().clone();
    proof.entries[0].derivation.steps.clear();
    mutants.push(proof);
    let mut source = original.sparse_jacobian().clone();
    source.residuals[0] = Expr::from_i64(0);
    mutants.push(source);
    let mut pattern = original.sparse_jacobian().clone();
    pattern.entries.clear();
    pattern.pattern.nonzeros.clear();
    pattern.pattern.nnz = 0;
    pattern.column_colors.fill(0);
    pattern.num_colors = 1;
    mutants.push(pattern);
    let mut variable_order = original.sparse_jacobian().clone();
    variable_order.vars.swap(0, 1);
    mutants.push(variable_order);
    for mutant in mutants {
        assert!(matches!(
            VerifiedResidualSystem::from_json_bytes(&encode_untrusted(&mutant)),
            Err(VerifiedArtifactError::Admission(_))
        ));
    }
    // Mutating a separately owned bundle cannot alter an admitted program.
    assert_eq!(
        original.compiled().compiled_jacobian[0]
            .try_eval(&[2.0, 3.0])
            .unwrap(),
        4.0
    );
}

#[test]
fn envelope_rejects_receipts_unknown_fields_duplicates_and_future_versions() {
    let system = fixture();
    let bytes = system.to_json_bytes().unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(original.as_object().unwrap().len(), 2);
    assert!(original.get("receipt").is_none());
    for field in ["receipt", "verified", "compiled"] {
        let mut forged = original.clone();
        forged[field] = serde_json::json!(true);
        assert!(matches!(
            VerifiedResidualSystem::from_json_bytes(&serde_json::to_vec(&forged).unwrap()),
            Err(VerifiedArtifactError::Encoding(_))
        ));
    }
    let mut nested = original.clone();
    nested["jacobian"]["verified"] = serde_json::json!(true);
    assert!(matches!(
        VerifiedResidualSystem::from_json_bytes(&serde_json::to_vec(&nested).unwrap()),
        Err(VerifiedArtifactError::Encoding(_))
    ));
    for version in [0, VERIFIED_ARTIFACT_SCHEMA_VERSION + 1, u32::MAX] {
        // The deliberately invalid payload must not be admitted or interpreted
        // using the current schema before an incompatible version is rejected.
        let incompatible = serde_json::json!({
            "jacobian": null,
            "schema_version": version,
        });
        assert_eq!(
            VerifiedResidualSystem::from_json_bytes(&serde_json::to_vec(&incompatible).unwrap())
                .unwrap_err(),
            VerifiedArtifactError::UnsupportedVersion {
                found: version,
                supported: VERIFIED_ARTIFACT_SCHEMA_VERSION,
            }
        );
    }
    let duplicate = br#"{"schema_version":1,"schema_version":1,"jacobian":null}"#;
    assert!(matches!(
        VerifiedResidualSystem::from_json_bytes(duplicate),
        Err(VerifiedArtifactError::Encoding(_))
    ));
    let mut trailing = bytes;
    trailing.extend_from_slice(b" null");
    assert!(matches!(
        VerifiedResidualSystem::from_json_bytes(&trailing),
        Err(VerifiedArtifactError::Encoding(_))
    ));
}

#[test]
fn decoder_refuses_oversized_or_deep_data_before_expression_construction() {
    let oversized = vec![b' '; MAX_VERIFIED_ARTIFACT_BYTES + 1];
    assert_eq!(
        VerifiedResidualSystem::from_json_bytes(&oversized).unwrap_err(),
        VerifiedArtifactError::ByteLimit {
            actual: MAX_VERIFIED_ARTIFACT_BYTES + 1,
            max: MAX_VERIFIED_ARTIFACT_BYTES,
        }
    );
    let depth = MAX_VERIFIED_ARTIFACT_JSON_DEPTH + 1;
    let deep = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
    assert!(matches!(
        VerifiedResidualSystem::from_json_bytes(deep.as_bytes()),
        Err(VerifiedArtifactError::DepthLimit { .. })
    ));
}

#[test]
fn retained_proof_claims_and_premises_are_bounded_before_cloning() {
    let original = fixture();
    let mut payload = Expr::from_i64(0);
    for _ in 0..49 {
        payload = Expr::Add(vec![payload, Expr::from_i64(0)]);
    }
    let mut claim_bundle = original.sparse_jacobian().clone();
    let mut extra = claim_bundle.entries[0].derivation.steps[0].clone();
    extra.id = fsym_proof_kernel::StepId(999);
    extra.claim = fsym_proof_kernel::Claim::equality(payload.clone(), payload.clone());
    claim_bundle.entries[0].derivation.steps.push(extra);
    assert!(matches!(
        VerifiedResidualSystem::try_from_sparse_jacobian(&claim_bundle),
        Err(VerifiedCompilationError::SourceDepthLimit { .. })
    ));
    let mut rule_bundle = original.sparse_jacobian().clone();
    let mut extra = rule_bundle.entries[0].derivation.steps[0].clone();
    extra.id = fsym_proof_kernel::StepId(999);
    extra.rule = fsym_proof_kernel::ProofRule::Substitution {
        template: payload,
        var: Symbol::new("z"),
        step: fsym_proof_kernel::StepId(1),
    };
    rule_bundle.entries[0].derivation.steps.push(extra);
    assert!(matches!(
        VerifiedResidualSystem::try_from_sparse_jacobian(&rule_bundle),
        Err(VerifiedCompilationError::SourceDepthLimit { .. })
    ));
}

#[test]
fn deep_raw_sources_refuse_before_recursive_symbolic_admission() {
    let x = Symbol::new("x");
    let mut source = Expr::Sym(x.clone());
    for _ in 0..MAX_VERIFIED_SOURCE_DEPTH {
        source = Expr::Add(vec![source, Expr::from_i64(0)]);
    }
    let at_boundary =
        VerifiedResidualSystem::try_compile(&[source.clone()], std::slice::from_ref(&x)).unwrap();
    assert_eq!(
        at_boundary.compiled().compiled_jacobian[0]
            .try_eval(&[1.0])
            .unwrap(),
        1.0
    );
    let too_deep = Expr::Add(vec![source, Expr::from_i64(0)]);
    assert!(matches!(
        VerifiedResidualSystem::try_compile(
            std::slice::from_ref(&too_deep),
            std::slice::from_ref(&x)
        ),
        Err(VerifiedCompilationError::SourceDepthLimit { .. })
    ));
    let mut forged = at_boundary.sparse_jacobian().clone();
    forged.residuals[0] = too_deep;
    assert!(matches!(
        VerifiedResidualSystem::try_from_sparse_jacobian(&forged),
        Err(VerifiedCompilationError::SourceDepthLimit { .. })
    ));
}

#[test]
fn zero_dimensions_cannot_hide_an_oversized_other_axis() {
    let excess = MAX_VERIFIED_JACOBIAN_ENTRIES + 1;
    let too_many_rows = SparseJacobian {
        residuals: vec![Expr::from_i64(0); excess],
        vars: Vec::new(),
        pattern: SparsityPattern {
            num_rows: excess,
            num_cols: 0,
            nonzeros: Vec::new(),
            nnz: 0,
        },
        entries: Vec::new(),
        column_colors: Vec::new(),
        num_colors: 0,
    };
    let too_many_columns = SparseJacobian {
        residuals: Vec::new(),
        vars: (0..excess)
            .map(|index| Symbol::new(format!("v{index}")))
            .collect(),
        pattern: SparsityPattern {
            num_rows: 0,
            num_cols: excess,
            nonzeros: Vec::new(),
            nnz: 0,
        },
        entries: Vec::new(),
        column_colors: vec![0; excess],
        num_colors: 1,
    };
    for bundle in [too_many_rows, too_many_columns] {
        assert!(matches!(
            VerifiedResidualSystem::try_compile(&bundle.residuals, &bundle.vars),
            Err(VerifiedCompilationError::AxisLimit { .. })
        ));
        assert!(matches!(
            VerifiedResidualSystem::try_from_sparse_jacobian(&bundle),
            Err(VerifiedCompilationError::AxisLimit { .. })
        ));
        let encoded = encode_untrusted(&bundle);
        assert!(encoded.len() < MAX_VERIFIED_ARTIFACT_BYTES);
        assert!(matches!(
            VerifiedResidualSystem::from_json_bytes(&encoded),
            Err(VerifiedArtifactError::Admission(
                VerifiedCompilationError::AxisLimit { .. }
            ))
        ));
    }
}

#[test]
fn quoted_brackets_escapes_and_unicode_do_not_count_as_json_nesting() {
    let name = format!("variable_{}\\\"_λ", "[{]}".repeat(100));
    let variable = Symbol::new(name);
    let original =
        VerifiedResidualSystem::try_compile(&[Expr::Sym(variable.clone())], &[variable]).unwrap();
    let restored =
        VerifiedResidualSystem::from_json_bytes(&original.to_json_bytes().unwrap()).unwrap();
    assert_eq!(restored.vars(), original.vars());
    assert_eq!(restored.sparse_jacobian(), original.sparse_jacobian());
}

#[test]
fn small_sparse_wire_cannot_request_an_unbounded_dense_program() {
    let side = 257;
    assert!(side * side > MAX_VERIFIED_JACOBIAN_ENTRIES);
    let residuals = vec![Expr::from_i64(0); side];
    let vars = (0..side)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect::<Vec<_>>();
    assert!(matches!(
        VerifiedResidualSystem::try_compile(&residuals, &vars),
        Err(VerifiedCompilationError::DenseJacobianLimit { .. })
    ));
    let jacobian = SparseJacobian {
        residuals,
        vars,
        pattern: SparsityPattern {
            num_rows: side,
            num_cols: side,
            nonzeros: Vec::new(),
            nnz: 0,
        },
        entries: Vec::new(),
        column_colors: vec![0; side],
        num_colors: 1,
    };
    let encoded = encode_untrusted(&jacobian);
    assert!(encoded.len() < MAX_VERIFIED_ARTIFACT_BYTES);
    assert!(matches!(
        VerifiedResidualSystem::from_json_bytes(&encoded),
        Err(VerifiedArtifactError::Admission(
            VerifiedCompilationError::DenseJacobianLimit { .. }
        ))
    ));
}

#[test]
fn empty_system_and_declared_variable_order_round_trip() {
    let empty = VerifiedResidualSystem::try_compile(&[], &[]).unwrap();
    let restored =
        VerifiedResidualSystem::from_json_bytes(&empty.to_json_bytes().unwrap()).unwrap();
    restored
        .compiled()
        .try_eval_system(&[], &mut [], &mut [])
        .unwrap();
    assert_eq!(restored.receipt().proofs_verified, 0);
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let ordered = VerifiedResidualSystem::try_compile(
        &[Expr::Sym(x.clone()), Expr::Sym(y.clone())],
        &[y.clone(), x.clone()],
    )
    .unwrap();
    let decoded =
        VerifiedResidualSystem::from_json_bytes(&ordered.to_json_bytes().unwrap()).unwrap();
    assert_eq!(decoded.vars(), &[y, x]);
    let mut values = [0.0; 2];
    decoded
        .compiled()
        .try_eval_residuals(&[7.0, 11.0], &mut values)
        .unwrap();
    assert_eq!(values, [11.0, 7.0]);
}

#[test]
fn nonfinite_derivative_literals_are_not_silently_encoded_as_null() {
    let x = Symbol::new("x");
    let source = Expr::Mul(vec![Expr::Const(Constant::Infinity), Expr::Sym(x.clone())]);
    let system = VerifiedResidualSystem::try_compile(&[source], &[x]).unwrap();
    assert_eq!(
        system.to_json_bytes().unwrap_err(),
        VerifiedArtifactError::NonFiniteBytecode
    );
}

#[test]
fn malformed_structure_refuses_before_proof_payload_traversal() {
    let x = Symbol::new("x");
    let admitted = VerifiedResidualSystem::try_compile(
        &[Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2))],
        &[x],
    )
    .unwrap();
    let mut bundle = admitted.sparse_jacobian().clone();
    let mut duplicate = bundle.entries[0].clone();
    let mut deep = Expr::from_i64(0);
    for _ in 0..49 {
        deep = Expr::Add(vec![deep, Expr::from_i64(0)]);
    }
    duplicate.symbolic_deriv = deep;
    bundle.entries.push(duplicate);
    bundle.pattern.nonzeros.push((0, 0));
    bundle.pattern.nnz = 2;
    assert!(matches!(
        VerifiedResidualSystem::try_from_sparse_jacobian(&bundle),
        Err(VerifiedCompilationError::Jacobian(
            fsym_calculus::sparse_jacobian::JacobianVerificationError::NonCanonicalPattern { .. }
        ))
    ));
}

#[test]
fn nontrivial_binary64_coefficients_round_trip_without_provenance_drift() {
    // Pinned serde_json's own float-roundtrip tests identify these decimal
    // representations as double-rounding cases in its default fast parser.
    for (numerator, denominator, expected) in [
        (
            225_391_872_104_827_i64,
            4_398_046_511_104_i64,
            51.248178375505404_f64,
        ),
        (
            -6_566_203_950_917_947_i64,
            70_368_744_177_664_i64,
            -93.31137037688033_f64,
        ),
    ] {
        let x = Symbol::new("x");
        let coefficient = Expr::Rational(BigRational::new(
            BigInt::from(numerator),
            BigInt::from(denominator),
        ));
        let system = VerifiedResidualSystem::try_compile(
            &[Expr::Mul(vec![coefficient, Expr::Sym(x.clone())])],
            &[x],
        )
        .unwrap();
        let before = system.sparse_jacobian().entries[0]
            .compiled_deriv
            .try_eval(&[1.0])
            .unwrap();
        assert_eq!(before.to_bits(), expected.to_bits());
        let saved = system.to_json_bytes().unwrap();
        let restored = VerifiedResidualSystem::from_json_bytes(&saved).unwrap();
        let after = restored.sparse_jacobian().entries[0]
            .compiled_deriv
            .try_eval(&[1.0])
            .unwrap();
        let original_ops = system.sparse_jacobian().entries[0].compiled_deriv.ops();
        let restored_ops = restored.sparse_jacobian().entries[0].compiled_deriv.ops();
        assert_eq!(original_ops.len(), restored_ops.len());
        for (original, restored) in original_ops.iter().zip(restored_ops) {
            match (original, restored) {
                (CompiledOp::LoadConst(left), CompiledOp::LoadConst(right)) => {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
                _ => assert_eq!(original, restored),
            }
        }
        assert_eq!(after.to_bits(), before.to_bits());
        assert_eq!(
            restored.compiled().compiled_residuals,
            system.compiled().compiled_residuals
        );
    }
}
