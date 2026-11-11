//! Shared numeric work stays behind source admission for every residual API.

use fsym_calculus::compile::{CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::verified::{
    ResidualEvaluationError, VerifiedArtifactError, VerifiedResidualSystem,
};
use fsym_core::{Expr, Symbol, parse};

fn fixture() -> VerifiedResidualSystem {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let sine = Expr::Function("sin".into(), vec![Expr::Sym(x.clone())]);
    let exponential = Expr::Function("exp".into(), vec![Expr::Sym(y.clone())]);
    let sum = Expr::Add(vec![sine.clone(), exponential.clone()]);
    let product = Expr::Mul(vec![sine, exponential]);
    let residuals = [
        sum.clone(),
        product.clone(),
        Expr::Add(vec![sum, Expr::from_i64(7)]),
        Expr::Add(vec![product, Expr::from_i64(11)]),
    ];
    VerifiedResidualSystem::try_compile(&residuals, &[x, y, Symbol::new("unused")]).unwrap()
}

fn assert_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
}

fn assert_domain_error<T: std::fmt::Debug>(result: Result<T, ResidualEvaluationError>) {
    assert!(
        matches!(
            result,
            Err(ResidualEvaluationError::Evaluation(
                EvalError::NonFiniteValue
            ))
        ),
        "expected a finite-domain refusal, got {result:?}"
    );
}

#[test]
fn dense_sparse_and_both_products_match_unshared_programs_bit_for_bit() {
    let system = fixture();
    assert!(system.shared_subexpression_count() > 0);
    assert_eq!(system.num_residuals(), 4);
    assert_eq!(system.num_vars(), 3);
    assert_eq!(
        system.jacobian_pattern(),
        [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1)
        ]
    );
    let separate = system.try_dense_program().unwrap();
    let direction = [2.0, -3.0, 5.0];
    let weights = [3.0, -2.0, 4.0, 1.0];
    for point in [[0.0, 0.0, 0.0], [0.25, -0.5, 2.0], [-0.75, 0.3, -4.0]] {
        let mut residual_reference = vec![0.0; 4];
        let mut jacobian_reference = vec![0.0; 12];
        separate
            .try_eval_system(&point, &mut residual_reference, &mut jacobian_reference)
            .unwrap();
        // These programs execute independently, without the batch memo used by
        // any of the verified evaluation entry points.
        let original_values = system
            .residual_programs()
            .iter()
            .map(|program| program.try_eval_finite(&point).unwrap())
            .collect::<Vec<_>>();
        assert_bits(&original_values, &residual_reference);
        let dense = system.try_eval_dense(&point).unwrap();
        assert_bits(&dense.residuals, &residual_reference);
        assert_bits(&dense.jacobian, &jacobian_reference);

        let sparse_reference = system
            .jacobian_pattern()
            .iter()
            .map(|&(row, col)| jacobian_reference[row * 3 + col])
            .collect::<Vec<_>>();
        let sparse = system.evaluate_sparse(&point).unwrap();
        assert_bits(&sparse.residuals, &residual_reference);
        assert_bits(&sparse.jacobian, &sparse_reference);

        let mut forward_reference = vec![0.0; 4];
        let mut transpose_reference = vec![0.0; 3];
        for (&(row, col), &derivative) in system.jacobian_pattern().iter().zip(&sparse_reference) {
            forward_reference[row] += derivative * direction[col];
            transpose_reference[col] += derivative * weights[row];
        }
        let forward = system.jacobian_vector_product(&point, &direction).unwrap();
        let transpose = system
            .jacobian_transpose_vector_product(&point, &weights)
            .unwrap();
        assert_bits(&forward, &forward_reference);
        assert_bits(&transpose, &transpose_reference);

        // Closed forms separately check coordinate placement, including the
        // unused variable and the rectangular transpose output shape.
        let (sine, cosine, exponential) = (point[0].sin(), point[0].cos(), point[1].exp());
        let sum_direction = 2.0 * cosine - 3.0 * exponential;
        let product_direction = 2.0 * cosine * exponential - 3.0 * sine * exponential;
        for (actual, expected) in forward.iter().zip([
            sum_direction,
            product_direction,
            sum_direction,
            product_direction,
        ]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in transpose.iter().zip([
            7.0 * cosine - cosine * exponential,
            7.0 * exponential - sine * exponential,
            0.0,
        ]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn shared_original_source_domains_precede_zero_weight_products_and_publication() {
    let x = Symbol::new("x");
    let shared_log = Expr::Function("log".into(), vec![Expr::Sym(x.clone())]);
    let system = VerifiedResidualSystem::try_compile(
        &[
            Expr::Function("sin".into(), vec![Expr::Sym(x.clone())]),
            shared_log.clone(),
            shared_log,
        ],
        &[x],
    )
    .unwrap();
    assert!(system.shared_subexpression_count() > 0);
    let valid = system.try_eval_dense(&[2.0]).unwrap();
    // The negative point has finite derivative bytecode; only the original
    // residual's log domain forbids publishing a numeric answer.
    assert!(
        system
            .sparse_jacobian()
            .entries
            .iter()
            .all(|entry| { entry.compiled_deriv.try_eval_finite(&[-1.0]).is_ok() })
    );
    for point in [-1.0, 0.0] {
        let mut residuals = [901.0; 3];
        let mut dense = [902.0; 3];
        assert_domain_error(system.try_eval_dense_into(&[point], &mut residuals, &mut dense));
        assert_eq!(residuals, [901.0; 3]);
        assert_eq!(dense, [902.0; 3]);
        let mut sparse = vec![903.0; system.jacobian_pattern().len()];
        assert_domain_error(system.try_eval_sparse_into(&[point], &mut residuals, &mut sparse));
        assert_eq!(residuals, [901.0; 3]);
        assert!(sparse.iter().all(|value| *value == 903.0));
        let mut forward = [904.0; 3];
        assert_domain_error(system.try_jacobian_vector_product_into(
            &[point],
            &[0.0],
            &mut forward,
        ));
        assert_eq!(forward, [904.0; 3]);
        let mut transpose = [905.0];
        assert_domain_error(system.try_jacobian_transpose_vector_product_into(
            &[point],
            &[0.0; 3],
            &mut transpose,
        ));
        assert_eq!(transpose, [905.0]);
    }
    // A failed call cannot poison a subsequent call or reuse an earlier point.
    let repeated = system.try_eval_dense(&[2.0]).unwrap();
    assert_bits(&repeated.residuals, &valid.residuals);
    assert_bits(&repeated.jacobian, &valid.jacobian);
}

#[test]
fn cached_finite_sources_do_not_hide_derivative_domain_errors() {
    let source = parse("x^(1/2)").unwrap();
    let system =
        VerifiedResidualSystem::try_compile(&[source.clone(), source], &[Symbol::new("x")])
            .unwrap();
    assert!(system.shared_subexpression_count() > 0);
    assert!(
        system
            .residual_programs()
            .iter()
            .all(|program| program.try_eval_finite(&[0.0]).unwrap() == 0.0)
    );
    assert_domain_error(system.try_eval_dense(&[0.0]));
    assert_domain_error(system.evaluate_sparse(&[0.0]));
    assert_domain_error(system.jacobian_vector_product(&[0.0], &[0.0]));
    assert_domain_error(system.jacobian_transpose_vector_product(&[0.0], &[0.0, 0.0]));
}

#[test]
fn json_rebuilds_the_plan_after_source_admission_and_rejects_mutated_programs() {
    let system = fixture();
    let bytes = system.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
    assert!(restored.shared_subexpression_count() > 0);
    assert_eq!(
        restored.shared_subexpression_count(),
        system.shared_subexpression_count()
    );
    assert_eq!(restored.residual_programs(), system.residual_programs());
    assert_eq!(restored.sparse_jacobian(), system.sparse_jacobian());
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
    for point in [[0.25, -0.5, 2.0], [-0.75, 0.3, -4.0]] {
        let expected = system.try_eval_dense(&point).unwrap();
        let actual = restored.try_eval_dense(&point).unwrap();
        assert_bits(&actual.residuals, &expected.residuals);
        assert_bits(&actual.jacobian, &expected.jacobian);
        assert_bits(
            &restored
                .jacobian_vector_product(&point, &[2.0, -3.0, 5.0])
                .unwrap(),
            &system
                .jacobian_vector_product(&point, &[2.0, -3.0, 5.0])
                .unwrap(),
        );
        assert_bits(
            &restored
                .jacobian_transpose_vector_product(&point, &[3.0, -2.0, 4.0, 1.0])
                .unwrap(),
            &system
                .jacobian_transpose_vector_product(&point, &[3.0, -2.0, 4.0, 1.0])
                .unwrap(),
        );
    }
    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["jacobian"]["entries"][0]["compiled_deriv"] = serde_json::to_value(
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(999.0)]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        VerifiedResidualSystem::from_json_bytes(&serde_json::to_vec(&forged).unwrap()),
        Err(VerifiedArtifactError::Admission(_))
    ));
    // The serialized proof remains authoritative; the derived memo plan and
    // its cached numeric values never become an artifact admission shortcut.
    let mut forged_plan: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged_plan["shared_evaluation"] = serde_json::json!({"slots": [], "values": [999.0]});
    assert!(matches!(
        VerifiedResidualSystem::from_json_bytes(&serde_json::to_vec(&forged_plan).unwrap()),
        Err(VerifiedArtifactError::Encoding(_))
    ));
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);
}
