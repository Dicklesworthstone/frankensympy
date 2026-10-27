//! Complete real equations, original domains, mutation rejection and bounds.
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_solvers::{RealEquationError, metered_solve_real_equation, solve_real_equation};
use std::sync::Arc;

fn pow(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn add(terms: Vec<Expr>) -> Expr {
    Expr::Add(terms)
}
fn mul(factors: Vec<Expr>) -> Expr {
    Expr::Mul(factors)
}
fn quintic(x: &Expr) -> Expr {
    add(vec![
        pow(x.clone(), 5),
        mul(vec![Expr::from_i64(-1), x.clone()]),
        Expr::from_i64(1),
    ])
}
fn quadratic(x: &Expr) -> Expr {
    add(vec![pow(x.clone(), 2), Expr::from_i64(-2)])
}

#[test]
fn unsolvable_quintic_is_complete_and_bound_to_the_original_typed_source() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let expression = quintic(&Expr::Sym(x.clone()));
    let result = solve_real_equation(&expression, &x).unwrap().unwrap();
    assert_eq!(result.expression(), &expression);
    assert_eq!(result.variable(), &x);
    assert_eq!(result.polynomial().gen_sym, x);
    assert!(!result.is_complement());
    assert_eq!(
        result.polynomial().coeffs,
        vec![
            1.into(),
            (-1).into(),
            0.into(),
            0.into(),
            0.into(),
            1.into()
        ]
    );
    assert_eq!(result.roots().intervals().len(), 1);
    let root = &result.roots().intervals()[0];
    assert!(root.lower < BigRational::from_integer((-1).into()));
    assert!(root.upper > BigRational::from_integer((-2).into()));
}

#[test]
fn irrational_domain_holes_remove_all_multiplicities_not_just_rational_poles() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let f = quintic(&xe);
    let forbidden = quadratic(&xe);
    let linear = add(vec![xe, Expr::from_i64(-3)]);
    let original = mul(vec![
        f.clone(),
        pow(forbidden.clone(), 2),
        pow(linear.clone(), 3),
        pow(mul(vec![forbidden, linear]), -1),
    ]);
    let result = solve_real_equation(&original, &x).unwrap().unwrap();
    assert_eq!(result.expression(), &original);
    assert_eq!(result.roots().intervals().len(), 1);
    let expected = solve_real_equation(&f, &x).unwrap().unwrap();
    assert_eq!(result.polynomial(), expected.polynomial());
    assert_eq!(result.roots().count_with_multiplicity(), 1);
}

#[test]
fn nested_inverse_and_held_zero_power_keep_the_original_domain() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let nested = pow(pow(xe.clone(), -1), -1);
    let result = solve_real_equation(&nested, &x).unwrap().unwrap();
    assert!(!result.is_complement());
    assert!(result.roots().intervals().is_empty());
    let held_zero = mul(vec![Expr::from_i64(0), pow(pow(xe, -1), 0)]);
    let result = solve_real_equation(&held_zero, &x).unwrap().unwrap();
    assert!(result.is_complement());
    assert_eq!(result.polynomial().coeffs, vec![0.into(), 1.into()]);
    assert_eq!(result.roots().intervals().len(), 1);
}

#[test]
fn identities_return_the_complete_cofinite_domain_with_overlapping_irrational_holes() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let f = quintic(&xe);
    let q = quadratic(&xe);
    let expression = mul(vec![
        Expr::from_i64(0),
        pow(mul(vec![f.clone(), q.clone()]), -1),
        pow(q, -2),
    ]);
    let result = solve_real_equation(&expression, &x).unwrap().unwrap();
    assert!(result.is_complement());
    assert_eq!(result.roots().intervals().len(), 3);
    assert!(
        result
            .roots()
            .intervals()
            .iter()
            .all(|r| r.multiplicity == 1)
    );
    assert_eq!(result.polynomial().degree(), Some(7));
}

#[test]
fn constants_nowhere_defined_inputs_and_no_real_roots_are_distinct() {
    let x = Symbol::new("x");
    let zero = solve_real_equation(&Expr::from_i64(0), &x)
        .unwrap()
        .unwrap();
    assert!(zero.is_complement());
    assert!(zero.roots().intervals().is_empty());
    let nonzero = solve_real_equation(&Expr::from_i64(7), &x)
        .unwrap()
        .unwrap();
    assert!(!nonzero.is_complement());
    assert!(nonzero.roots().intervals().is_empty());
    let undefined = mul(vec![Expr::from_i64(0), pow(Expr::from_i64(0), -1)]);
    let undefined = solve_real_equation(&undefined, &x).unwrap().unwrap();
    assert!(!undefined.is_complement());
    assert!(undefined.roots().intervals().is_empty());
    let near_real = add(vec![
        pow(Expr::Sym(x.clone()), 2),
        Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(80))),
    ]);
    let result = solve_real_equation(&near_real, &x).unwrap().unwrap();
    assert!(result.roots().intervals().is_empty());
}

#[test]
fn rational_sums_cross_multiply_without_forgetting_either_original_denominator() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let minus = add(vec![xe.clone(), Expr::from_i64(-1)]);
    let plus = add(vec![xe, Expr::from_i64(1)]);
    let equation = add(vec![
        pow(minus.clone(), -1),
        mul(vec![Expr::from_i64(-2), pow(plus.clone(), -1)]),
    ]);
    let result = solve_real_equation(&equation, &x).unwrap().unwrap();
    assert!(!result.is_complement());
    assert_eq!(result.polynomial().coeffs, vec![(-3).into(), 1.into()]);
    let identity = add(vec![
        mul(vec![plus.clone(), pow(mul(vec![minus.clone(), plus]), -1)]),
        mul(vec![Expr::from_i64(-1), pow(minus, -1)]),
    ]);
    let result = solve_real_equation(&identity, &x).unwrap().unwrap();
    assert!(result.is_complement());
    assert_eq!(
        result.polynomial().coeffs,
        vec![(-1).into(), 0.into(), 1.into()]
    );
    assert_eq!(result.roots().intervals().len(), 2);
}

#[test]
fn exact_polynomials_with_many_real_roots_deduplicate_repeated_factors() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let p = mul(vec![
        pow(quadratic(&xe), 3),
        pow(add(vec![xe.clone(), Expr::from_i64(-1)]), 2),
        add(vec![pow(xe, 2), Expr::from_i64(1)]),
    ]);
    let result = solve_real_equation(&p, &x).unwrap().unwrap();
    assert_eq!(result.roots().intervals().len(), 3);
    assert_eq!(result.roots().count_with_multiplicity(), 3);
}

#[test]
fn unsupported_shapes_and_same_name_foreign_symbols_do_not_enter_the_exact_lane() {
    let x = Symbol::new("x");
    let foreign = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32]));
    for expression in [
        Expr::Sym(foreign),
        Expr::Function("sin".to_owned(), vec![Expr::Sym(x.clone())]),
        Expr::Pow(
            Arc::new(Expr::Sym(x.clone())),
            Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
        ),
    ] {
        assert_eq!(solve_real_equation(&expression, &x).unwrap(), None);
    }
}

#[test]
fn original_hidden_growth_and_degree_limits_refuse_before_cancellation() {
    let x = Symbol::new("x");
    for expression in [
        pow(Expr::Sym(x.clone()), 65),
        mul(vec![
            Expr::from_i64(0),
            Expr::Integer(BigInt::one() << 4096u32),
        ]),
        pow(pow(Expr::Sym(x.clone()), 40), 2),
    ] {
        assert!(matches!(
            solve_real_equation(&expression, &x),
            Err(RealEquationError::ResourceLimit(_))
        ));
    }
    let mut deep = Expr::Sym(x.clone());
    for _ in 0..65 {
        deep = pow(deep, 1);
    }
    assert!(matches!(
        solve_real_equation(&deep, &x),
        Err(RealEquationError::ResourceLimit(_))
    ));
}

struct Cancelled;
impl BudgetMeter for Cancelled {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn generator_and_reference_verifier_cancellation_and_exhaustion_are_typed() {
    let x = Symbol::new("x");
    let expression = quintic(&Expr::Sym(x.clone()));
    for result in [
        metered_solve_real_equation(&expression, &x, &mut Cancelled, &mut Unbounded),
        metered_solve_real_equation(&expression, &x, &mut Unbounded, &mut Cancelled),
    ] {
        assert_eq!(result, Err(RealEquationError::Meter(MeterError::Cancelled)));
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_solve_real_equation(&expression, &x, &mut tiny, &mut Unbounded),
        Err(RealEquationError::Meter(MeterError::Budget(_)))
    ));
}
