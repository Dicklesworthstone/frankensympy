//! Adversarial acceptance and publication checks for imported sparse Jacobians.

use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, verify_sparse_jacobian_certificate,
};
use fsym_core::{Expr, Symbol};

fn diagonal_system() -> SparseJacobian {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    SparseJacobian::try_build(
        &[
            Expr::pow(Expr::Sym(x.clone()), Expr::from_i64(2)),
            Expr::pow(Expr::Sym(y.clone()), Expr::from_i64(2)),
        ],
        &[x, y],
    )
    .unwrap()
}

#[test]
fn bytecode_provenance_is_required_without_numerical_samples() {
    let valid = diagonal_system();
    let receipt = verify_sparse_jacobian_certificate(&valid, &[]).unwrap();
    assert_eq!(receipt.proofs_verified, 2);
    assert_eq!(receipt.test_points_evaluated, 0);

    for forged in [
        vec![CompiledOp::LoadConst(0.0)],
        vec![CompiledOp::LoadConst(42.0)],
        vec![CompiledOp::LoadVar(1)],
        vec![CompiledOp::LoadConst(f64::INFINITY)],
    ] {
        let mut candidate = valid.clone();
        candidate.entries[0].compiled_deriv = CompiledExpr::try_from_ops(forged).unwrap();
        for points in [vec![], vec![vec![0.0, 0.0]]] {
            assert!(matches!(
                verify_sparse_jacobian_certificate(&candidate, &points),
                Err(JacobianVerificationError::CompiledProvenanceMismatch { row: 0, col: 0 })
            ));
        }
    }
}

#[test]
fn bytecode_operator_and_variable_mutants_cannot_reuse_a_valid_derivative_proof() {
    let valid = diagonal_system();
    let original_ops = valid.entries[0].compiled_deriv.ops();
    for index in 0..original_ops.len() {
        let mut operations = original_ops.to_vec();
        operations[index] = match &operations[index] {
            CompiledOp::LoadVar(_) => CompiledOp::LoadVar(1),
            CompiledOp::LoadConst(value) => CompiledOp::LoadConst(value + 1.0),
            CompiledOp::Pow(value) => CompiledOp::Pow(value + 1.0),
            CompiledOp::Mul(count) => CompiledOp::Add(*count),
            _ => continue,
        };
        let mut candidate = valid.clone();
        candidate.entries[0].compiled_deriv = CompiledExpr::try_from_ops(operations).unwrap();
        assert!(matches!(
            verify_sparse_jacobian_certificate(&candidate, &[]),
            Err(JacobianVerificationError::CompiledProvenanceMismatch { .. })
        ));
    }
}

#[test]
fn malformed_metadata_is_refused_without_panics_or_buffer_publication() {
    let valid = diagonal_system();
    let mut mutants: Vec<(&str, SparseJacobian)> = Vec::new();

    let mut candidate = valid.clone();
    candidate.pattern.num_rows = 0;
    candidate.pattern.num_cols = 0;
    candidate.pattern.nonzeros.clear();
    candidate.pattern.nnz = 0;
    candidate.entries.clear();
    candidate.column_colors.clear();
    candidate.num_colors = 0;
    mutants.push(("zero dimensions hide the whole system", candidate));

    let mut candidate = valid.clone();
    candidate.residuals.pop();
    mutants.push(("truncated residuals", candidate));
    let mut candidate = valid.clone();
    candidate.vars.pop();
    mutants.push(("truncated variables", candidate));
    let mut candidate = valid.clone();
    candidate.vars[1] = candidate.vars[0].clone();
    mutants.push(("duplicate variables", candidate));
    let mut candidate = valid.clone();
    candidate.pattern.nnz += 1;
    mutants.push(("forged nnz", candidate));
    let mut candidate = valid.clone();
    candidate.pattern.nonzeros[0] = (0, 1);
    mutants.push(("pattern coordinate differs from entry", candidate));
    let mut candidate = valid.clone();
    candidate.entries[0].col = 1;
    mutants.push(("entry coordinate differs from pattern", candidate));
    let mut candidate = valid.clone();
    candidate.pattern.nonzeros[1] = candidate.pattern.nonzeros[0];
    candidate.entries[1] = candidate.entries[0].clone();
    mutants.push(("duplicate coordinates", candidate));
    let mut candidate = valid.clone();
    candidate.pattern.nonzeros.swap(0, 1);
    candidate.entries.swap(0, 1);
    mutants.push(("noncanonical coordinate ordering", candidate));
    let mut candidate = valid.clone();
    candidate.entries[1].row = usize::MAX;
    mutants.push(("out of bounds entry", candidate));
    let mut candidate = valid.clone();
    candidate.pattern.nonzeros[1].1 = usize::MAX;
    mutants.push(("out of bounds pattern", candidate));
    let mut candidate = valid.clone();
    candidate.column_colors.pop();
    mutants.push(("truncated color table", candidate));
    let mut candidate = valid.clone();
    candidate.column_colors.push(0);
    mutants.push(("oversized color table", candidate));
    let mut candidate = valid.clone();
    candidate.column_colors[1] = usize::MAX;
    mutants.push(("out of bounds color", candidate));
    let mut candidate = valid.clone();
    candidate.num_colors = 0;
    mutants.push(("zero color count", candidate));
    let mut candidate = valid.clone();
    candidate.num_colors = 2;
    mutants.push(("unused declared color", candidate));
    let mut candidate = valid.clone();
    candidate.num_colors = usize::MAX;
    mutants.push(("overflowing color count", candidate));
    let mut candidate = valid;
    candidate.pattern.num_rows = usize::MAX;
    candidate.pattern.num_cols = 2;
    mutants.push(("overflowing matrix dimensions", candidate));

    for (description, candidate) in mutants {
        assert!(
            verify_sparse_jacobian_certificate(&candidate, &[]).is_err(),
            "accepted {description}"
        );
        let mut sparse = [73.0; 2];
        let mut dense = [73.0; 4];
        let mut compressed = [73.0; 2];
        assert!(candidate.try_eval_sparse(&[2.0, 3.0], &mut sparse).is_err());
        assert!(candidate.try_eval_dense(&[2.0, 3.0], &mut dense).is_err());
        assert!(
            candidate
                .try_eval_compressed(&[2.0, 3.0], &mut compressed)
                .is_err()
        );
        assert_eq!(sparse, [73.0; 2], "published {description}");
        assert_eq!(dense, [73.0; 4], "published {description}");
        assert_eq!(compressed, [73.0; 2], "published {description}");
    }
}

#[test]
fn omitted_nonzero_is_checked_even_when_the_pattern_and_entry_counts_agree() {
    let mut candidate = diagonal_system();
    candidate.entries.pop();
    candidate.pattern.nonzeros.pop();
    candidate.pattern.nnz -= 1;
    assert!(matches!(
        verify_sparse_jacobian_certificate(&candidate, &[]),
        Err(JacobianVerificationError::OmittedNonzero { row: 1, col: 1, .. })
    ));
}

#[test]
fn a_color_collision_is_rejected_before_compressed_output_is_modified() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let mut system = SparseJacobian::try_build(
        &[Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())])],
        &[x, y],
    )
    .unwrap();
    system.column_colors = vec![0, 0];
    system.num_colors = 1;
    assert!(matches!(
        verify_sparse_jacobian_certificate(&system, &[]),
        Err(JacobianVerificationError::ColoringCollision { .. })
    ));
    let mut output = [17.0];
    assert!(
        system
            .try_eval_compressed(&[1.0, 2.0], &mut output)
            .is_err()
    );
    assert_eq!(output, [17.0]);
}

#[test]
fn late_entry_evaluation_errors_preserve_every_output_layout() {
    let mut system = diagonal_system();
    system.entries[1].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(99)]).unwrap();
    let mut sparse = [19.0; 2];
    let mut dense = [19.0; 4];
    let mut compressed = [19.0; 2];
    for result in [
        system.try_eval_sparse(&[2.0, 3.0], &mut sparse),
        system.try_eval_dense(&[2.0, 3.0], &mut dense),
        system.try_eval_compressed(&[2.0, 3.0], &mut compressed),
    ] {
        assert!(matches!(
            result,
            Err(EvalError::VariableOutOfBounds { index: 99, .. })
        ));
    }
    assert_eq!(sparse, [19.0; 2]);
    assert_eq!(dense, [19.0; 4]);
    assert_eq!(compressed, [19.0; 2]);
}

#[test]
fn numerical_diagnostics_reject_nonfinite_inputs_and_results() {
    let system = diagonal_system();
    for point in [
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1.0],
        vec![1.0, 2.0, 3.0],
    ] {
        assert!(matches!(
            verify_sparse_jacobian_certificate(&system, &[point]),
            Err(JacobianVerificationError::InvalidTestPoint { .. })
        ));
    }
    let x = Symbol::new("x");
    let exponential = SparseJacobian::try_build(
        &[Expr::Function("exp".into(), vec![Expr::Sym(x.clone())])],
        &[x],
    )
    .unwrap();
    assert!(verify_sparse_jacobian_certificate(&exponential, &[]).is_ok());
    assert!(matches!(
        verify_sparse_jacobian_certificate(&exponential, &[vec![1_000.0]]),
        Err(JacobianVerificationError::NonFiniteEvaluation { .. })
    ));
}

#[test]
fn valid_empty_and_constant_systems_remain_usable() {
    let half = Expr::Rational(fsym_core::BigRational::new(1.into(), 2.into()));
    for system in [
        SparseJacobian::try_build(&[], &[]).unwrap(),
        SparseJacobian::try_build(&[Expr::from_i64(3)], &[]).unwrap(),
        SparseJacobian::try_build(&[Expr::from_i64(3)], &[Symbol::new("x")]).unwrap(),
        SparseJacobian::try_build(
            &[Expr::pow(Expr::from_i64(2), half.clone())],
            &[Symbol::new("x")],
        )
        .unwrap(),
        SparseJacobian::try_build(&[Expr::pow(Expr::symbol("y"), half)], &[Symbol::new("x")])
            .unwrap(),
    ] {
        let point = vec![1.0; system.vars.len()];
        let receipt =
            verify_sparse_jacobian_certificate(&system, std::slice::from_ref(&point)).unwrap();
        assert_eq!(receipt.nnz, 0);
        let mut dense = vec![91.0; system.pattern.num_rows * system.pattern.num_cols];
        system.try_eval_dense(&point, &mut dense).unwrap();
        assert!(dense.iter().all(|value| *value == 0.0));
    }

    let residual = Expr::Add(vec![
        Expr::symbol("x"),
        Expr::pow(
            Expr::from_i64(2),
            Expr::Rational(fsym_core::BigRational::new(1.into(), 2.into())),
        ),
    ]);
    let system = SparseJacobian::try_build(&[residual], &[Symbol::new("x")]).unwrap();
    verify_sparse_jacobian_certificate(&system, &[]).unwrap();
    let mut derivative = [0.0];
    system.try_eval_dense(&[0.0], &mut derivative).unwrap();
    assert_eq!(derivative, [1.0]);
}

#[test]
fn rational_power_derivative_and_serialized_provenance_round_trip() {
    let x = Symbol::new("x");
    let system = SparseJacobian::try_build(
        &[Expr::Function("asin".into(), vec![Expr::Sym(x.clone())])],
        &[x],
    )
    .unwrap();
    let encoded = serde_json::to_string(&system).unwrap();
    let imported: SparseJacobian = serde_json::from_str(&encoded).unwrap();
    verify_sparse_jacobian_certificate(&imported, &[vec![0.25]]).unwrap();
    let mut output = [0.0];
    imported.try_eval_dense(&[0.25], &mut output).unwrap();
    assert!((output[0] - 1.0 / (1.0_f64 - 0.25 * 0.25).sqrt()).abs() < 1e-12);

    let mut unknown = serde_json::to_value(system).unwrap();
    unknown["verified"] = serde_json::Value::Bool(true);
    assert!(serde_json::from_value::<SparseJacobian>(unknown).is_err());
}

#[test]
fn hostile_expression_depth_is_bounded_before_differentiation_and_verification() {
    let x = Symbol::new("x");
    let mut nested = Expr::Sym(x.clone());
    for _ in 0..=fsym_calculus::compile::MAX_COMPILE_DEPTH {
        nested = Expr::Function("sin".into(), vec![nested]);
    }
    assert!(matches!(
        SparseJacobian::try_build(std::slice::from_ref(&nested), &[x]),
        Err(fsym_calculus::compile::CompileError::DepthLimitExceeded { .. })
    ));
    let mut candidate = diagonal_system();
    candidate.residuals[0] = nested.clone();
    assert!(matches!(
        verify_sparse_jacobian_certificate(&candidate, &[]),
        Err(JacobianVerificationError::Compile(
            fsym_calculus::compile::CompileError::DepthLimitExceeded { .. }
        ))
    ));
    let mut candidate = diagonal_system();
    candidate.entries[0].symbolic_deriv = nested;
    assert!(matches!(
        verify_sparse_jacobian_certificate(&candidate, &[]),
        Err(JacobianVerificationError::Compile(
            fsym_calculus::compile::CompileError::DepthLimitExceeded { .. }
        ))
    ));
}
