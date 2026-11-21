//! Derivative-domain trial refusals backtrack, while initial failures stop.

use fsym_calculus::compile::EvalError;
use fsym_calculus::sparse_hessian::ObjectiveEvaluationError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_calculus::{VerifiedObjective, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol, parse};
use fsym_numeric_consumer::{
    LeastSquaresOptions, MinimizeOptions, ObjectiveConsumerError, SystemConsumerError,
    least_squares, minimize, sparse_least_squares, sparse_minimize,
};
use std::sync::Arc;

fn domain_cases() -> Vec<(&'static str, f64, EvalError)> {
    vec![
        ("sqrt(x)", 1.0, EvalError::NonsmoothSource("sqrt")),
        ("abs(x)", 1.0, EvalError::NonsmoothSource("abs")),
        ("acos(x+1)", -1.0, EvalError::NonsmoothSource("acos")),
        ("atan2(x,-1)", 1.0, EvalError::NonsmoothAngle),
        ("atan2(x,0)", 1.0, EvalError::UndefinedAngle),
    ]
}

fn held_sum(base: Expr, primitive: &str) -> Expr {
    Expr::Add(vec![
        base,
        Expr::Mul(vec![Expr::from_i64(0), parse(primitive).unwrap()]),
    ])
}

#[test]
fn dense_and_sparse_objective_trials_backtrack_before_publishing_an_endpoint() {
    for (primitive, initial, expected) in domain_cases() {
        let source = held_sum(
            Expr::Pow(
                Arc::new(Expr::symbol("x")),
                Arc::new(Expr::from_i64(2)),
            ),
            primitive,
        );
        let objective = VerifiedObjective::try_compile(&source, &[Symbol::new("x")]).unwrap();
        for solve in [minimize, sparse_minimize] {
            assert_eq!(
                solve(&objective, &[0.0], &MinimizeOptions::default()),
                Err(ObjectiveConsumerError::Evaluation(
                    ObjectiveEvaluationError::Evaluation(expected.clone())
                ))
            );
            // The Newton trial reaches the nonsmooth limit point. A domain
            // refusal must cause a shorter step instead of an abort or a
            // publication of that endpoint as an admitted minimum.
            let result = solve(&objective, &[initial], &MinimizeOptions::default()).unwrap();
            assert!(result.point[0] * initial > 0.0, "{primitive}");
            assert!(result.gradient_norm <= 1e-10);
            assert_eq!(result.value, result.point[0] * result.point[0]);
            assert_eq!(result.gradient, [2.0 * result.point[0]]);
            assert!(result.iterations > 1);
            assert!(result.evaluations > result.iterations + 2);
        }
    }
}

#[test]
fn dense_and_sparse_least_squares_retry_math_domains_and_recheck_original_sources() {
    for (primitive, initial, expected) in domain_cases() {
        let source = held_sum(Expr::symbol("x"), primitive);
        let system =
            VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("x")]).unwrap();
        for solve in [least_squares, sparse_least_squares] {
            assert!(matches!(
                solve(&system, &[0.0], &LeastSquaresOptions::default()),
                Err(SystemConsumerError::SparseEvaluation(
                    ResidualEvaluationError::Evaluation(actual)
                )) if actual == expected
            ));
            let result = solve(&system, &[initial], &LeastSquaresOptions::default()).unwrap();
            assert!(result.point[0] * initial > 0.0, "{primitive}");
            assert!(result.gradient_norm <= 1e-10);
            assert_eq!(result.residuals, result.point);
            assert_eq!(result.cost, 0.5 * result.point[0] * result.point[0]);
            assert!(result.iterations > 1);
            assert!(result.evaluations > result.iterations + 2);
        }
    }
}
