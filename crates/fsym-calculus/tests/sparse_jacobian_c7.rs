//! C7 Campaign Gate: Sparse Jacobian analysis, proof replay, and compiled evaluator agreement.

#![forbid(unsafe_code)]

use fsym_calculus::compile::CompiledOp;
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, verify_sparse_jacobian_certificate,
};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, ProofRule};
use std::sync::Arc;

/// Constructs the canonical 4D sparse nonlinear hero fixture:
/// f0(x0, x1, x2, x3) = x0^2 + sin(x1) - 1
/// f1(x0, x1, x2, x3) = x1 * x2 - cos(x2)
/// f2(x0, x1, x2, x3) = exp(x2) + x3^3 - 4
/// f3(x0, x1, x2, x3) = x0 * x3 - 2
fn hero_sparse_residual_fixture() -> (Vec<Expr>, Vec<Symbol>) {
    let x0 = Symbol::new("x0");
    let x1 = Symbol::new("x1");
    let x2 = Symbol::new("x2");
    let x3 = Symbol::new("x3");
    let vars = vec![x0.clone(), x1.clone(), x2.clone(), x3.clone()];

    let f0 = Expr::Add(vec![
        Expr::Pow(Arc::new(Expr::Sym(x0.clone())), Arc::new(Expr::from_i64(2))),
        Expr::Function("sin".into(), vec![Expr::Sym(x1.clone())]),
        Expr::from_i64(-1),
    ]);

    let f1 = Expr::Add(vec![
        Expr::Mul(vec![Expr::Sym(x1.clone()), Expr::Sym(x2.clone())]),
        Expr::Mul(vec![
            Expr::from_i64(-1),
            Expr::Function("cos".into(), vec![Expr::Sym(x2.clone())]),
        ]),
    ]);

    let f2 = Expr::Add(vec![
        Expr::Function("exp".into(), vec![Expr::Sym(x2.clone())]),
        Expr::Pow(Arc::new(Expr::Sym(x3.clone())), Arc::new(Expr::from_i64(3))),
        Expr::from_i64(-4),
    ]);

    let f3 = Expr::Add(vec![
        Expr::Mul(vec![Expr::Sym(x0.clone()), Expr::Sym(x3.clone())]),
        Expr::from_i64(-2),
    ]);

    (vec![f0, f1, f2, f3], vars)
}

#[test]
fn test_sparse_jacobian_c7_hero_fixture_verified_and_evaluates() {
    let (residuals, vars) = hero_sparse_residual_fixture();
    let sparse_jac = SparseJacobian::try_build(&residuals, &vars).unwrap();

    // Structural checks
    assert_eq!(sparse_jac.pattern.num_rows, 4);
    assert_eq!(sparse_jac.pattern.num_cols, 4);
    assert_eq!(sparse_jac.nnz(), 8);
    assert!((sparse_jac.density() - 0.5).abs() < 1e-12);

    // Distance-1 coloring: ring/tridiagonal allows 2 colors
    assert!(sparse_jac.num_colors <= 2);

    // Nonzeros: (0,0), (0,1), (1,1), (1,2), (2,2), (2,3), (3,0), (3,3)
    let expected_nonzeros = vec![
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 3),
    ];
    assert_eq!(sparse_jac.pattern.nonzeros, expected_nonzeros);

    let test_points = vec![
        vec![1.0, 0.5, -0.5, 2.0],
        vec![0.5, 1.2, 0.3, -1.0],
        vec![2.0, -0.8, 1.5, 0.1],
        vec![0.1, 0.2, 0.3, 0.4],
    ];

    // Independent certificate verification: proof replay + omission checks + evaluator agreement
    let receipt = verify_sparse_jacobian_certificate(&sparse_jac, &test_points).unwrap();
    assert_eq!(receipt.num_residuals, 4);
    assert_eq!(receipt.num_vars, 4);
    assert_eq!(receipt.nnz, 8);
    assert_eq!(receipt.proofs_verified, 8);
    assert_eq!(receipt.test_points_evaluated, 4);

    // Test sparse and compressed evaluators
    for pt in &test_points {
        let mut sparse_vals = vec![0.0; 8];
        sparse_jac.try_eval_sparse(pt, &mut sparse_vals).unwrap();

        let mut dense_vals = vec![0.0; 16];
        sparse_jac.try_eval_dense(pt, &mut dense_vals).unwrap();

        // Check nonzeros match dense buffer
        for (k, &(r, c)) in sparse_jac.pattern.nonzeros.iter().enumerate() {
            assert_eq!(sparse_vals[k], dense_vals[r * 4 + c]);
        }

        // Compressed evaluation
        let mut compressed = vec![0.0; 4 * sparse_jac.num_colors];
        sparse_jac.try_eval_compressed(pt, &mut compressed).unwrap();
        for (k, &(r, c)) in sparse_jac.pattern.nonzeros.iter().enumerate() {
            let col_color = sparse_jac.column_colors[c];
            assert_eq!(
                sparse_vals[k],
                compressed[r * sparse_jac.num_colors + col_color]
            );
        }
    }

    // Serde wire round-trip
    let json = serde_json::to_string(&sparse_jac).unwrap();
    let deser: SparseJacobian = serde_json::from_str(&json).unwrap();
    assert_eq!(sparse_jac, deser);
    verify_sparse_jacobian_certificate(&deser, &test_points).unwrap();
}

#[test]
fn test_sparse_jacobian_c7_adversarial_tampering() {
    let (residuals, vars) = hero_sparse_residual_fixture();
    let sparse_jac = SparseJacobian::try_build(&residuals, &vars).unwrap();
    let test_points = vec![vec![1.0, 0.5, -0.5, 2.0]];

    // 1. Mutant: Tampered proof step claim
    let mut bad_proof = sparse_jac.clone();
    bad_proof.entries[0].derivation.steps[0].claim =
        Claim::equality(Expr::from_i64(0), Expr::from_i64(1));
    assert!(matches!(
        verify_sparse_jacobian_certificate(&bad_proof, &test_points),
        Err(JacobianVerificationError::ProofFailed { row: 0, col: 0, .. })
    ));

    // 2. Mutant: Tampered symbolic derivative
    let mut bad_deriv = sparse_jac.clone();
    bad_deriv.entries[0].symbolic_deriv = Expr::from_i64(999);
    assert!(matches!(
        verify_sparse_jacobian_certificate(&bad_deriv, &test_points),
        Err(JacobianVerificationError::ProofFailed { row: 0, col: 0, .. })
    ));

    // 3. Mutant: Tampered proof rule
    let mut bad_rule = sparse_jac.clone();
    bad_rule.entries[0].derivation.steps[0].rule = ProofRule::Reflexivity(Expr::from_i64(0));
    assert!(matches!(
        verify_sparse_jacobian_certificate(&bad_rule, &test_points),
        Err(JacobianVerificationError::ProofFailed { row: 0, col: 0, .. })
    ));

    // 4. Mutant: Omitted nonzero entry from pattern
    let mut omitted_entry = sparse_jac.clone();
    omitted_entry.pattern.nonzeros.remove(0);
    omitted_entry.entries.remove(0);
    omitted_entry.pattern.nnz = omitted_entry.entries.len();
    assert!(matches!(
        verify_sparse_jacobian_certificate(&omitted_entry, &test_points),
        Err(JacobianVerificationError::OmittedNonzero { row: 0, col: 0, .. })
    ));

    // 5. Mutant: Coloring collision (force col 0 and col 1 to share color 0)
    let mut bad_color = sparse_jac.clone();
    bad_color.column_colors[0] = 0;
    bad_color.column_colors[1] = 0; // row 0 has nonzeros at col 0 and col 1!
    assert!(matches!(
        verify_sparse_jacobian_certificate(&bad_color, &test_points),
        Err(JacobianVerificationError::ColoringCollision {
            row: 0,
            color: 0,
            ..
        })
    ));

    // 6. Mutant: Discrepant compiled bytecode (replaces entry with constant 42.0)
    let mut bad_bytecode = sparse_jac.clone();
    let forged_expr =
        fsym_calculus::compile::CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(42.0)])
            .unwrap();
    bad_bytecode.entries[0].compiled_deriv = forged_expr;
    assert!(matches!(
        verify_sparse_jacobian_certificate(&bad_bytecode, &test_points),
        Err(JacobianVerificationError::EvaluatorDisagreement { row: 0, col: 0, .. })
    ));
}
