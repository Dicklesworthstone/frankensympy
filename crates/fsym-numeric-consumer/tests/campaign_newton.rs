//! Integration proof for the pinned-consumer boundary
//! (`fra-rc-adapters-iym`): a transcendental campaign-shaped residual
//! system is compiled by fsym-calculus, Newton-refined with every linear
//! solve executed through the pinned `fnp-linalg` consumer, and the
//! accepted iterate is certified under exact ball arithmetic against the
//! isolated-oracle-compatible declared bound. Singular systems refuse
//! through the consumer's typed error.

use fsym_core::{BigInt, Expr, Symbol};
use fsym_numeric_consumer::{NewtonGuards, certified_residual_bound, newton_refine_2x2};

fn transcendental_system() -> (Vec<Expr>, Vec<Symbol>) {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    // f1 = exp(x/2) + y - 3 ; f2 = x + sin(y) - 1.2
    let half_x = Expr::Mul(vec![
        Expr::Rational(fsym_core::BigRational::new(
            BigInt::from(1),
            BigInt::from(2),
        )),
        Expr::Sym(x.clone()),
    ]);
    let f1 = Expr::Add(vec![
        Expr::Function("exp".into(), vec![half_x]),
        Expr::Sym(y.clone()),
        Expr::from_i64(-3),
    ]);
    let f2 = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Function("sin".into(), vec![Expr::Sym(y.clone())]),
        Expr::Rational(fsym_core::BigRational::new(
            BigInt::from(-6),
            BigInt::from(5),
        )),
    ]);
    (vec![f1, f2], vec![x, y])
}

#[test]
fn consumer_newton_converges_and_certifies_transcendental_system() {
    let (exprs, vars) = transcendental_system();
    let system = fsym_calculus::CompiledResidualSystem::try_compile(&exprs, &vars)
        .expect("campaign system compiles");

    let outcome = newton_refine_2x2(&system, [1.0, 1.0], &NewtonGuards::default())
        .expect("consumer-driven Newton converges");

    assert!(
        outcome.residual_norm <= 1e-12,
        "residual norm {} exceeds the declared tolerance",
        outcome.residual_norm
    );
    assert!(
        outcome.final_jacobian_determinant.abs() > 1e-6,
        "nonsingularity certificate requires a bounded-away determinant"
    );
    assert!(outcome.iterations <= 16, "Newton must converge quickly");

    // Certified consistency: the accepted iterate, substituted as exact
    // binary64 rationals, evaluates under certified ball arithmetic to a
    // residual infinity-norm bounded by 1e-12 with ball radii far below.
    certified_residual_bound(&exprs, &vars, &outcome.root)
        .expect("certified residual bound at the accepted iterate");
}

#[test]
fn singular_system_refuses_through_the_consumer() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    // Both equations impose the same constraint: the Jacobian is singular
    // everywhere.
    let degenerate = Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]);
    let exprs = vec![
        Expr::Add(vec![degenerate.clone(), Expr::from_i64(-1)]),
        Expr::Add(vec![degenerate, Expr::from_i64(-1)]),
    ];
    let vars = vec![x, y];
    let system = fsym_calculus::CompiledResidualSystem::try_compile(&exprs, &vars)
        .expect("degenerate system compiles");

    let outcome = newton_refine_2x2(&system, [0.0, 0.0], &NewtonGuards::default());
    assert!(
        matches!(
            outcome,
            Err(fsym_numeric_consumer::ConsumerError::SingularJacobian(_))
        ),
        "degenerate system must refuse through the consumer's typed error, got {outcome:?}"
    );
}
