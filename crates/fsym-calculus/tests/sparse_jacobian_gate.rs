//! Sparse Jacobian fixture with proof replay and compiled evaluator agreement (C7 Gate).
//!
//! Validates:
//! 1. Symbolic structural differentiation with typed proof trees (`DerivationTree`).
//! 2. Independent derivation verifier proof replay (`verify_diff_derivation`) across all entries.
//! 3. Exact sparsity pattern agreement matching symbol dependency graph.
//! 4. Compiled bytecode evaluation (`CompiledResidualSystem`) agreeing with exact analytical evaluations.
//! 5. Central finite differences diagnostic agreement.
//! 6. Adversarial tampering resistance (forged claims, incorrect rule names, mutated formulas).

use fsym_calculus::compile::{CompiledResidualSystem, EvalError};
use fsym_calculus::proof::{
    RULE_DIFF_CONST, make_diff_term, verified_diff, verify_diff_derivation,
};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, KernelError, ProofRule, StepId};
use std::sync::Arc;

fn free_symbols(expr: &Expr) -> std::collections::BTreeSet<Symbol> {
    let mut set = std::collections::BTreeSet::new();
    collect_symbols(expr, &mut set);
    set
}

fn collect_symbols(expr: &Expr, acc: &mut std::collections::BTreeSet<Symbol>) {
    match expr {
        Expr::Sym(s) => {
            acc.insert(s.clone());
        }
        Expr::Add(terms) | Expr::Mul(terms) => {
            for t in terms {
                collect_symbols(t, acc);
            }
        }
        Expr::Pow(b, e) => {
            collect_symbols(b, acc);
            collect_symbols(e, acc);
        }
        Expr::Function(_, args) => {
            for a in args {
                collect_symbols(a, acc);
            }
        }
        _ => {}
    }
}

#[test]
fn test_sparse_jacobian_3d_fixture_proof_replay_and_compiled_agreement() {
    // 3D Sparse nonlinear system:
    // f1(x, y, z) = x^2 + y^2 - 1        (independent of z)
    // f2(x, y, z) = sin(y) + cos(z)       (independent of x)
    // f3(x, y, z) = exp(x) - z^2          (independent of y)
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let z = Symbol::new("z");
    let vars = vec![x.clone(), y.clone(), z.clone()];

    let f1 = Expr::Add(vec![
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
        Expr::Pow(Arc::new(Expr::Sym(y.clone())), Arc::new(Expr::from_i64(2))),
        Expr::from_i64(-1),
    ]);
    let f2 = Expr::Add(vec![
        Expr::Function("sin".into(), vec![Expr::Sym(y.clone())]),
        Expr::Function("cos".into(), vec![Expr::Sym(z.clone())]),
    ]);
    let f3 = Expr::Add(vec![
        Expr::Function("exp".into(), vec![Expr::Sym(x.clone())]),
        Expr::Mul(vec![
            Expr::from_i64(-1),
            Expr::Pow(Arc::new(Expr::Sym(z.clone())), Arc::new(Expr::from_i64(2))),
        ]),
    ]);

    let exprs = vec![f1.clone(), f2.clone(), f3.clone()];

    // 1. Sparsity analysis based on symbol dependency graph
    let mut expected_sparse_zeros = vec![vec![false; 3]; 3];
    for (i, expr) in exprs.iter().enumerate() {
        let syms = free_symbols(expr);
        for (j, var) in vars.iter().enumerate() {
            if !syms.contains(var) {
                expected_sparse_zeros[i][j] = true;
            }
        }
    }

    assert!(expected_sparse_zeros[0][2], "f1 is independent of z");
    assert!(expected_sparse_zeros[1][0], "f2 is independent of x");
    assert!(expected_sparse_zeros[2][1], "f3 is independent of y");
    assert!(!expected_sparse_zeros[0][0]);
    assert!(!expected_sparse_zeros[0][1]);
    assert!(!expected_sparse_zeros[1][1]);
    assert!(!expected_sparse_zeros[1][2]);
    assert!(!expected_sparse_zeros[2][0]);
    assert!(!expected_sparse_zeros[2][2]);

    // 2. Symbolic differentiation with derivation trees and proof replay
    let mut derivation_trees = Vec::new();
    let mut symbolic_derivatives = Vec::new();

    for (i, expr) in exprs.iter().enumerate() {
        for (j, var) in vars.iter().enumerate() {
            let (deriv, tree) = verified_diff(expr, var);

            // Replay and verify proof
            verify_diff_derivation(&tree, expr, var, &deriv)
                .unwrap_or_else(|e| panic!("Proof replay failed for ({i}, {j}): {e:?}"));

            // Check derivation claim integrity
            assert_eq!(tree.steps.len(), 1);
            assert_eq!(tree.root, StepId(1));
            assert_eq!(
                tree.steps[0].claim,
                Claim::equality(make_diff_term(expr, var), deriv.clone())
            );

            // If structurally zero, verify that evaluation is zero
            if expected_sparse_zeros[i][j] {
                assert!(
                    fsym_simplify::simplify(&deriv).is_zero(),
                    "Expected structurally zero derivative for ({i}, {j}), got {deriv}"
                );
            }

            derivation_trees.push(tree);
            symbolic_derivatives.push(deriv);
        }
    }

    assert_eq!(derivation_trees.len(), 9);

    // 3. Compile system into bytecode
    let system = CompiledResidualSystem::try_compile(&exprs, &vars).expect("compile succeeds");
    assert_eq!(system.num_residuals, 3);
    assert_eq!(system.num_vars, 3);

    // 4. Test evaluation agreement across multiple test points
    let test_points: Vec<[f64; 3]> = vec![
        [0.5, 0.5, 1.0],
        [
            0.0,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_6,
        ], // [0, pi/4, pi/6]
        [-0.8, 0.6, 2.0],
        [1.2, -0.4, 0.1],
    ];

    for pt in test_points {
        let mut res = [0.0; 3];
        let mut jac = [0.0; 9];
        system.eval_system(&pt, &mut res, &mut jac);

        let (xv, yv, zv) = (pt[0], pt[1], pt[2]);

        // Analytical residual values
        let expected_res = [
            xv * xv + yv * yv - 1.0,
            yv.sin() + zv.cos(),
            xv.exp() - zv * zv,
        ];

        for i in 0..3 {
            assert!(
                (res[i] - expected_res[i]).abs() < 1e-12,
                "Residual mismatch at pt {pt:?}, dim {i}: got {}, expected {}",
                res[i],
                expected_res[i]
            );
        }

        // Analytical Jacobian values
        let expected_jac = [
            2.0 * xv,
            2.0 * yv,
            0.0, // row 0: df1/dx, df1/dy, df1/dz
            0.0,
            yv.cos(),
            -zv.sin(), // row 1: df2/dx, df2/dy, df2/dz
            xv.exp(),
            0.0,
            -2.0 * zv, // row 2: df3/dx, df3/dy, df3/dz
        ];

        for idx in 0..9 {
            let row = idx / 3;
            let col = idx % 3;
            assert!(
                (jac[idx] - expected_jac[idx]).abs() < 1e-12,
                "Jacobian mismatch at pt {pt:?}, entry ({row}, {col}): got {}, expected {}",
                jac[idx],
                expected_jac[idx]
            );

            // Verify exact zero for sparse zero entries
            if expected_sparse_zeros[row][col] {
                assert_eq!(
                    jac[idx], 0.0,
                    "Sparse zero entry ({row}, {col}) must evaluate to exactly 0.0"
                );
            }
        }

        // 5. Numerical finite-differences diagnostic consistency
        let fd_ok = system.check_with_finite_differences(&pt, 1e-6, 1e-5);
        assert!(
            fd_ok,
            "Finite differences consistency failed at test point {pt:?}"
        );
    }
}

#[test]
fn test_sparse_jacobian_tridiagonal_4d_fixture() {
    // 4D Tridiagonal Sparse System:
    // f1 = x1^2 + x2
    // f2 = x1 + x2^2 + x3
    // f3 = x2 + x3^2 + x4
    // f4 = x3 + x4^2
    let x1 = Symbol::new("x1");
    let x2 = Symbol::new("x2");
    let x3 = Symbol::new("x3");
    let x4 = Symbol::new("x4");
    let vars = vec![x1.clone(), x2.clone(), x3.clone(), x4.clone()];

    let f1 = Expr::Add(vec![
        Expr::Pow(Arc::new(Expr::Sym(x1.clone())), Arc::new(Expr::from_i64(2))),
        Expr::Sym(x2.clone()),
    ]);
    let f2 = Expr::Add(vec![
        Expr::Sym(x1.clone()),
        Expr::Pow(Arc::new(Expr::Sym(x2.clone())), Arc::new(Expr::from_i64(2))),
        Expr::Sym(x3.clone()),
    ]);
    let f3 = Expr::Add(vec![
        Expr::Sym(x2.clone()),
        Expr::Pow(Arc::new(Expr::Sym(x3.clone())), Arc::new(Expr::from_i64(2))),
        Expr::Sym(x4.clone()),
    ]);
    let f4 = Expr::Add(vec![
        Expr::Sym(x3.clone()),
        Expr::Pow(Arc::new(Expr::Sym(x4.clone())), Arc::new(Expr::from_i64(2))),
    ]);

    let exprs = vec![f1, f2, f3, f4];

    // Verify all 16 partial derivatives with proof replay
    for (i, expr) in exprs.iter().enumerate() {
        for (j, var) in vars.iter().enumerate() {
            let (deriv, tree) = verified_diff(expr, var);
            verify_diff_derivation(&tree, expr, var, &deriv)
                .unwrap_or_else(|e| panic!("Tridiagonal proof replay failed ({i}, {j}): {e:?}"));

            // Check tridiagonal sparsity: |i - j| > 1 must be zero
            let is_sparse_zero = (i as isize - j as isize).abs() > 1;
            if is_sparse_zero {
                assert!(
                    fsym_simplify::simplify(&deriv).is_zero(),
                    "Tridiagonal entry ({i}, {j}) must be zero"
                );
            }
        }
    }

    // Compile and evaluate
    let system = CompiledResidualSystem::try_compile(&exprs, &vars).expect("compile succeeds");
    let pt = [1.0, 2.0, 3.0, 4.0];
    let mut res = [0.0; 4];
    let mut jac = [0.0; 16];
    system.eval_system(&pt, &mut res, &mut jac);

    // Expected tridiagonal Jacobian at [1, 2, 3, 4]:
    // Row 0: [2*1, 1,   0,   0  ] = [2, 1, 0, 0]
    // Row 1: [1,   2*2, 1,   0  ] = [1, 4, 1, 0]
    // Row 2: [0,   1,   2*3, 1  ] = [0, 1, 6, 1]
    // Row 3: [0,   0,   1,   2*4] = [0, 0, 1, 8]
    let expected_jac = [
        2.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 6.0, 1.0, 0.0, 0.0, 1.0, 8.0,
    ];

    for idx in 0..16 {
        assert!(
            (jac[idx] - expected_jac[idx]).abs() < 1e-12,
            "Tridiagonal entry {idx} mismatch: got {}, expected {}",
            jac[idx],
            expected_jac[idx]
        );
    }
}

#[test]
fn test_sparse_jacobian_adversarial_tampering_and_mutants_killed() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let f = Expr::Add(vec![
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
        Expr::Sym(y.clone()),
    ]);

    let (deriv, tree) = verified_diff(&f, &x);

    // 1. Positive baseline
    assert!(verify_diff_derivation(&tree, &f, &x, &deriv).is_ok());

    // 2. Mutant: forged derivative expression
    let forged_deriv = Expr::from_i64(100);
    assert!(matches!(
        verify_diff_derivation(&tree, &f, &x, &forged_deriv),
        Err(KernelError::ClaimDiscrepancy { .. })
    ));

    // 3. Mutant: forged wrong variable in claim
    assert!(verify_diff_derivation(&tree, &f, &y, &deriv).is_err());

    // 4. Mutant: tampered proof step claim
    let mut tampered_tree = tree.clone();
    tampered_tree.steps[0].claim = Claim::equality(Expr::from_i64(0), Expr::from_i64(0));
    assert!(matches!(
        verify_diff_derivation(&tampered_tree, &f, &x, &deriv),
        Err(KernelError::ClaimDiscrepancy { .. })
    ));

    // 5. Mutant: rule mismatch
    let mut tampered_rule_tree = tree.clone();
    tampered_rule_tree.steps[0].rule = ProofRule::DefinitionalReduction {
        lhs: make_diff_term(&f, &x),
        rhs: deriv.clone(),
        rule_name: RULE_DIFF_CONST.to_string(),
    };
    assert!(matches!(
        verify_diff_derivation(&tampered_rule_tree, &f, &x, &deriv),
        Err(KernelError::RuleMismatch(_))
    ));

    // 6. Mutant: forged tree root step ID
    let mut invalid_root_tree = tree.clone();
    invalid_root_tree.root = StepId(999);
    assert!(matches!(
        verify_diff_derivation(&invalid_root_tree, &f, &x, &deriv),
        Err(KernelError::UnknownStep(StepId(999)))
    ));

    // 7. Compiler adversarial: input variable out of bounds produces typed refusal
    let vars = vec![x.clone(), y.clone()];
    let system = CompiledResidualSystem::try_compile(std::slice::from_ref(&f), &vars).unwrap();
    let short_point = [1.0];
    let mut res = [0.0; 1];
    let mut jac = [0.0; 2];
    assert!(matches!(
        system.try_eval_system(&short_point, &mut res, &mut jac),
        Err(EvalError::VariableOutOfBounds { .. })
    ));

    // 8. Compiler adversarial: buffer length mismatch produces typed refusal
    let ok_point = [1.0, 2.0];
    let mut bad_res = [0.0; 2]; // expected 1
    assert!(matches!(
        system.try_eval_system(&ok_point, &mut bad_res, &mut jac),
        Err(EvalError::ResidualBufferMismatch { .. })
    ));
    let mut bad_jac = [0.0; 1]; // expected 2
    assert!(matches!(
        system.try_eval_system(&ok_point, &mut res, &mut bad_jac),
        Err(EvalError::JacobianBufferMismatch { .. })
    ));
}
