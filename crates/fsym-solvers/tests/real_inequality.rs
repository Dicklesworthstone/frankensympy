//! Complete exact rational sign charts, original domains and adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_solvers::inequality::IneqOp;
use fsym_solvers::{
    RealEquationError, VerifiedRealInequality, metered_solve_real_inequality, solve_real_inequality,
};
use std::sync::Arc;

fn pow(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn add(terms: Vec<Expr>) -> Expr {
    Expr::Add(terms)
}
fn mul(terms: Vec<Expr>) -> Expr {
    Expr::Mul(terms)
}
fn polynomial(x: &Expr, degree: i64, constant: i64) -> Expr {
    add(vec![pow(x.clone(), degree), Expr::from_i64(constant)])
}
fn solve(expression: &Expr, variable: &Symbol, op: IneqOp) -> VerifiedRealInequality {
    solve_real_inequality(expression, variable, op)
        .unwrap()
        .unwrap()
}

#[test]
fn irreducible_quartic_and_quintic_signs_are_complete_without_radicals() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let xe = Expr::Sym(x.clone());
    let quartic = add(vec![
        pow(xe.clone(), 4),
        mul(vec![Expr::from_i64(-1), xe.clone()]),
        Expr::from_i64(-1),
    ]);
    let result = solve(&quartic, &x, IneqOp::Le);
    assert_eq!(result.expression(), &quartic);
    assert_eq!(result.variable(), &x);
    assert_eq!(result.operation(), IneqOp::Le);
    assert_eq!(result.polynomial().gen_sym, x);
    assert_eq!(result.roots().intervals().len(), 2);
    assert_eq!(result.gap_inclusions(), [false, true, false]);
    assert_eq!(result.point_inclusions(), [true, true]);
    let quintic = add(vec![
        pow(xe.clone(), 5),
        mul(vec![Expr::from_i64(-1), xe]),
        Expr::from_i64(1),
    ]);
    let result = solve(&quintic, &x, IneqOp::Gt);
    assert_eq!(result.roots().intervals().len(), 1);
    assert_eq!(result.gap_inclusions(), [false, true]);
    assert_eq!(result.point_inclusions(), [false]);
}

#[test]
fn exact_irrational_endpoint_ownership_distinguishes_zeros_from_poles() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let expression = mul(vec![
        polynomial(&xe, 2, -2),
        pow(polynomial(&xe, 2, -3), -1),
    ]);
    let result = solve(&expression, &x, IneqOp::Ge);
    assert_eq!(result.roots().intervals().len(), 4);
    assert_eq!(result.gap_inclusions(), [true, false, true, false, true]);
    assert_eq!(result.point_inclusions(), [false, true, true, false]);
    assert_eq!(
        result.polynomial().coeffs,
        vec![6.into(), 0.into(), (-5).into(), 0.into(), 1.into()]
    );
    let strict = solve(&expression, &x, IneqOp::Gt);
    assert_eq!(strict.gap_inclusions(), result.gap_inclusions());
    assert_eq!(strict.point_inclusions(), [false; 4]);
}

#[test]
fn cancellation_never_fills_original_irrational_holes() {
    let x = Symbol::new("x");
    let q = polynomial(&Expr::Sym(x.clone()), 2, -2);
    let expression = mul(vec![pow(q.clone(), 2), pow(q, -1)]);
    let result = solve(&expression, &x, IneqOp::Ge);
    assert_eq!(result.gap_inclusions(), [true, false, true]);
    assert_eq!(result.point_inclusions(), [false, false]);
    let result = solve(&expression, &x, IneqOp::Le);
    assert_eq!(result.gap_inclusions(), [false, true, false]);
    assert_eq!(result.point_inclusions(), [false, false]);
}

#[test]
fn repeated_zeros_keep_even_and_odd_sign_changes_distinct() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let expression = mul(vec![
        pow(xe.clone(), 2),
        pow(add(vec![xe, Expr::from_i64(-1)]), 3),
    ]);
    let result = solve(&expression, &x, IneqOp::Ge);
    assert_eq!(result.roots().intervals().len(), 2);
    assert_eq!(result.gap_inclusions(), [false, false, true]);
    assert_eq!(result.point_inclusions(), [true, true]);
    assert!(
        result
            .roots()
            .intervals()
            .iter()
            .all(|root| root.multiplicity == 1)
    );
}

#[test]
fn rational_sums_preserve_both_original_denominators() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let minus = add(vec![xe.clone(), Expr::from_i64(-1)]);
    let plus = add(vec![xe, Expr::from_i64(1)]);
    // 1/(x-1) - 2/(x+1) = (3-x)/(x²-1).
    let expression = add(vec![
        pow(minus, -1),
        mul(vec![Expr::from_i64(-2), pow(plus, -1)]),
    ]);
    let result = solve(&expression, &x, IneqOp::Ge);
    assert_eq!(result.gap_inclusions(), [true, false, true, false]);
    assert_eq!(result.point_inclusions(), [false, false, true]);
}

#[test]
fn nested_inverse_and_held_zero_powers_retain_original_domains() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let nested = pow(pow(xe.clone(), -1), -1);
    let result = solve(&nested, &x, IneqOp::Ge);
    assert_eq!(result.gap_inclusions(), [false, true]);
    assert_eq!(result.point_inclusions(), [false]);
    let held = pow(pow(xe, -1), 0);
    let result = solve(&held, &x, IneqOp::Gt);
    assert_eq!(result.gap_inclusions(), [true, true]);
    assert_eq!(result.point_inclusions(), [false]);
}

#[test]
fn identically_zero_expressions_return_all_and_only_defined_weak_gaps() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let expression = mul(vec![
        Expr::from_i64(0),
        pow(polynomial(&xe, 2, -2), -1),
        pow(xe, -1),
    ]);
    for op in [IneqOp::Le, IneqOp::Ge] {
        let result = solve(&expression, &x, op);
        assert_eq!(result.roots().intervals().len(), 3);
        assert_eq!(result.gap_inclusions(), [true; 4]);
        assert_eq!(result.point_inclusions(), [false; 3]);
    }
    for op in [IneqOp::Lt, IneqOp::Gt] {
        let result = solve(&expression, &x, op);
        assert_eq!(result.gap_inclusions(), [false; 4]);
        assert_eq!(result.point_inclusions(), [false; 3]);
    }
}

#[test]
fn constants_nowhere_defined_sources_and_complex_only_zeros_remain_distinct() {
    let x = Symbol::new("x");
    for (value, op, included) in [
        (0, IneqOp::Ge, true),
        (0, IneqOp::Gt, false),
        (1, IneqOp::Ge, true),
        (1, IneqOp::Lt, false),
        (-1, IneqOp::Le, true),
        (-1, IneqOp::Gt, false),
    ] {
        let result = solve(&Expr::from_i64(value), &x, op);
        assert!(result.roots().intervals().is_empty());
        assert_eq!(result.gap_inclusions(), [included]);
    }
    let undefined = mul(vec![Expr::from_i64(0), pow(Expr::from_i64(0), -1)]);
    for op in [IneqOp::Lt, IneqOp::Le, IneqOp::Gt, IneqOp::Ge] {
        let result = solve(&undefined, &x, op);
        assert_eq!(result.gap_inclusions(), [false]);
        assert!(result.roots().intervals().is_empty());
    }
    let no_real_zeros = polynomial(&Expr::Sym(x.clone()), 4, 1);
    let result = solve(&no_real_zeros, &x, IneqOp::Gt);
    assert!(result.roots().intervals().is_empty());
    assert_eq!(result.gap_inclusions(), [true]);
}

#[test]
fn extremely_close_critical_points_are_ordered_exactly() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let epsilon = BigRational::new(1.into(), BigInt::from(10).pow(80));
    let expression = mul(vec![xe.clone(), add(vec![xe, Expr::Rational(-epsilon)])]);
    let result = solve(&expression, &x, IneqOp::Ge);
    assert_eq!(result.roots().intervals().len(), 2);
    assert_eq!(result.gap_inclusions(), [true, false, true]);
    assert_eq!(result.point_inclusions(), [true, true]);
}

#[test]
fn unsupported_shapes_foreign_identities_and_hidden_growth_refuse() {
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
        assert_eq!(
            solve_real_inequality(&expression, &x, IneqOp::Lt).unwrap(),
            None
        );
    }
    let xe = Expr::Sym(x.clone());
    for expression in [
        pow(xe.clone(), 65),
        mul(vec![
            Expr::from_i64(0),
            Expr::Integer(BigInt::one() << 4096u32),
        ]),
        mul(vec![
            polynomial(&xe, 33, 1),
            pow(polynomial(&xe, 32, 1), -1),
        ]),
    ] {
        assert!(matches!(
            solve_real_inequality(&expression, &x, IneqOp::Ge),
            Err(RealEquationError::ResourceLimit(_))
        ));
    }
}

#[test]
fn production_budgets_accept_nontrivial_charts_and_preserve_exhaustion() {
    let x = Symbol::new("x");
    let xe = Expr::Sym(x.clone());
    let expression = mul(vec![
        polynomial(&xe, 2, -2),
        pow(polynomial(&xe, 2, -3), -1),
    ]);
    let mut generator = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let result =
        metered_solve_real_inequality(&expression, &x, IneqOp::Ge, &mut generator, &mut verifier)
            .unwrap()
            .unwrap();
    assert_eq!(result.point_inclusions(), [false, true, true, false]);
    for generator_tiny in [true, false] {
        let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
        let result = if generator_tiny {
            metered_solve_real_inequality(&expression, &x, IneqOp::Ge, &mut tiny, &mut Unbounded)
        } else {
            metered_solve_real_inequality(&expression, &x, IneqOp::Ge, &mut Unbounded, &mut tiny)
        };
        assert!(matches!(
            result,
            Err(RealEquationError::Meter(MeterError::Budget(_)))
        ));
    }
}

#[derive(Default)]
struct Interrupt {
    checkpoints: usize,
    cancel_at: usize,
}
impl BudgetMeter for Interrupt {
    fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints == self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_is_observed_before_during_and_at_final_publication_in_both_lanes() {
    let x = Symbol::new("x");
    let expression = polynomial(&Expr::Sym(x.clone()), 2, -2);
    let mut generator = Interrupt::default();
    let mut verifier = Interrupt::default();
    metered_solve_real_inequality(&expression, &x, IneqOp::Ge, &mut generator, &mut verifier)
        .unwrap();
    for (generator_lane, count) in [(true, generator.checkpoints), (false, verifier.checkpoints)] {
        for cancel_at in [1, count / 2, count] {
            let mut interrupted = Interrupt {
                checkpoints: 0,
                cancel_at,
            };
            let result = if generator_lane {
                metered_solve_real_inequality(
                    &expression,
                    &x,
                    IneqOp::Ge,
                    &mut interrupted,
                    &mut Unbounded,
                )
            } else {
                metered_solve_real_inequality(
                    &expression,
                    &x,
                    IneqOp::Ge,
                    &mut Unbounded,
                    &mut interrupted,
                )
            };
            assert!(matches!(
                result,
                Err(RealEquationError::Meter(MeterError::Cancelled))
                    | Err(RealEquationError::Roots(
                        fsym_core::real_roots::RealRootError::Meter(MeterError::Cancelled)
                    ))
            ));
        }
    }
}
