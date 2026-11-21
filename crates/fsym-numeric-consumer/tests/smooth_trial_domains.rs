//! Derivative-domain trial refusals backtrack, while initial failures stop.

use fsym_calculus::compile::EvalError;
use fsym_calculus::sparse_hessian::ObjectiveEvaluationError;
use fsym_calculus::verified::ResidualEvaluationError;
use fsym_calculus::{VerifiedObjective, VerifiedResidualSystem};
use fsym_core::{Expr, Symbol};
use fsym_numeric_consumer::{
    LeastSquaresOptions, MinimizeOptions, ObjectiveConsumerError, SystemConsumerError,
    least_squares, minimize, sparse_least_squares, sparse_minimize,
};
use std::sync::Arc;

fn domain_cases() -> Vec<(&'static str, Expr, f64, EvalError)> {
    let x = Expr::symbol("x");
    // Preserve the precise source primitive under test. Both public parsers
    // can lower sqrt to a rational power, which has a different error tag.
    vec![
        (
            "sqrt(x)",
            Expr::Function("sqrt".into(), vec![x.clone()]),
            1.0,
            EvalError::NonsmoothSource("sqrt"),
        ),
        (
            "abs(x)",
            Expr::Function("abs".into(), vec![x.clone()]),
            1.0,
            EvalError::NonsmoothSource("abs"),
        ),
        (
            "acos(x+1)",
            Expr::Function(
                "acos".into(),
                vec![Expr::Add(vec![x.clone(), Expr::from_i64(1)])],
            ),
            -1.0,
            EvalError::NonsmoothSource("acos"),
        ),
        (
            "atan2(x,-1)",
            Expr::Function("atan2".into(), vec![x.clone(), Expr::from_i64(-1)]),
            1.0,
            EvalError::NonsmoothAngle,
        ),
        (
            "atan2(x,0)",
            Expr::Function("atan2".into(), vec![x, Expr::from_i64(0)]),
            1.0,
            EvalError::UndefinedAngle,
        ),
    ]
}

fn held_sum(base: Expr, primitive: Expr) -> Expr {
    Expr::Add(vec![base, Expr::Mul(vec![Expr::from_i64(0), primitive])])
}

#[test]
fn dense_and_sparse_objective_trials_backtrack_before_publishing_an_endpoint() {
    for (label, primitive, initial, expected) in domain_cases() {
        let source = held_sum(
            Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(2))),
            primitive,
        );
        let objective = VerifiedObjective::try_compile(&source, &[Symbol::new("x")]).unwrap();
        for solve in [minimize, sparse_minimize] {
            assert_eq!(
                solve(&objective, &[0.0], &MinimizeOptions::default()),
                Err(ObjectiveConsumerError::Evaluation(
                    ObjectiveEvaluationError::Evaluation(expected.clone())
                )),
                "{label}"
            );
            // The Newton trial reaches the nonsmooth limit point. A domain
            // refusal must cause a shorter step instead of an abort or a
            // publication of that endpoint as an admitted minimum.
            let result = solve(&objective, &[initial], &MinimizeOptions::default())
                .unwrap_or_else(|error| panic!("{label}: {error:?}"));
            assert!(result.point[0] * initial > 0.0, "{label}: {result:?}");
            assert!(result.gradient_norm <= 1e-10, "{label}: {result:?}");
            assert_eq!(result.value, result.point[0] * result.point[0], "{label}");
            assert_eq!(result.gradient, [2.0 * result.point[0]], "{label}");
            assert!(result.iterations > 1, "{label}: {result:?}");
            assert!(
                result.evaluations > result.iterations + 2,
                "{label}: {result:?}"
            );
        }
    }
}

#[test]
fn dense_and_sparse_least_squares_retry_math_domains_and_recheck_original_sources() {
    for (label, primitive, initial, expected) in domain_cases() {
        let source = held_sum(Expr::symbol("x"), primitive);
        let system = VerifiedResidualSystem::try_compile(&[source], &[Symbol::new("x")]).unwrap();
        for solve in [least_squares, sparse_least_squares] {
            let result = solve(&system, &[0.0], &LeastSquaresOptions::default());
            assert!(
                matches!(
                    &result,
                    Err(SystemConsumerError::SparseEvaluation(
                        ResidualEvaluationError::Evaluation(actual)
                    )) if actual == &expected
                ),
                "{label}: {result:?}"
            );
            let result = solve(&system, &[initial], &LeastSquaresOptions::default())
                .unwrap_or_else(|error| panic!("{label}: {error:?}"));
            assert!(result.point[0] * initial > 0.0, "{label}: {result:?}");
            assert!(result.gradient_norm <= 1e-10, "{label}: {result:?}");
            assert_eq!(result.residuals, result.point, "{label}");
            assert_eq!(
                result.cost,
                0.5 * result.point[0] * result.point[0],
                "{label}"
            );
            assert!(result.iterations > 1, "{label}: {result:?}");
            assert!(
                result.evaluations > result.iterations + 2,
                "{label}: {result:?}"
            );
        }
    }
}
