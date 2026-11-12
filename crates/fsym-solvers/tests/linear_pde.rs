//! Characteristic families, independent source checks, and domain adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::MultivariatePoly;
use fsym_solvers::pde::{
    LinearFirstOrderPdeCertificate, PdeCompleteness, PdeError, linear_first_order_pde,
    metered_linear_first_order_pde, verify_linear_first_order_pde,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn same_polynomial(actual: &Expr, expected: &Expr, x: &Symbol, y: &Symbol) {
    let variables = [x.clone(), y.clone()];
    assert_eq!(
        MultivariatePoly::from_expr(actual, &variables).unwrap(),
        MultivariatePoly::from_expr(expected, &variables).unwrap(),
        "{actual} != {expected}"
    );
}

fn original_residual(
    solution: &Expr,
    coefficients: [&BigRational; 3],
    forcing: &Expr,
    x: &Symbol,
    y: &Symbol,
) -> Expr {
    rational_expr(coefficients[0].clone()) * fsym_calculus::diff(solution, x)
        + rational_expr(coefficients[1].clone()) * fsym_calculus::diff(solution, y)
        + rational_expr(coefficients[2].clone()) * solution.clone()
        - forcing.clone()
}

fn fixture() -> LinearFirstOrderPdeCertificate {
    LinearFirstOrderPdeCertificate {
        time_coordinate: e("x/2"),
        invariant_coordinate: e("y-3*x/2"),
        particular: e("x**2*y/4-x**3/8"),
    }
}

#[test]
fn mixed_transport_has_a_hand_derived_particular_and_characteristic_family() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b, c) = (q(2, 1), q(3, 1), q(0, 1));
    let forcing = e("x*y");
    let accepted = linear_first_order_pde(&a, &b, &c, &forcing, &x, &y).unwrap();
    same_polynomial(accepted.particular(), &e("x**2*y/4-x**3/8"), &x, &y);
    same_polynomial(accepted.time_coordinate(), &e("x/2"), &x, &y);
    same_polynomial(accepted.invariant_coordinate(), &e("y-3*x/2"), &x, &y);
    assert_eq!(accepted.homogeneous_coefficient(), &Expr::from_i64(1));
    assert_eq!(accepted.coefficients(), &[a.clone(), b.clone(), c.clone()],);
    assert_eq!(accepted.variables(), &[x.clone(), y.clone()]);
    assert_eq!(accepted.forcing(), &forcing);
    assert_eq!(
        accepted.completeness(),
        PdeCompleteness::VerifiedSolutionFamily,
    );

    // An actual nonconstant choice F(t)=t^3+2t satisfies the original PDE
    // through the separate expression-differentiation path.
    let invariant = accepted.invariant_coordinate().clone();
    let solution =
        accepted.particular().clone() + power(invariant.clone(), 3) + Expr::from_i64(2) * invariant;
    same_polynomial(
        &original_residual(&solution, [&a, &b, &c], &forcing, &x, &y),
        &Expr::from_i64(0),
        &x,
        &y,
    );
}

#[test]
fn rational_reaction_and_transport_recover_a_known_polynomial() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b, c) = (q(2, 3), q(-3, 5), q(7, 11));
    // q=x^2*y/2+y/3-2; these forcing coefficients are hand-derived.
    let forcing = e("7*x**2*y/22+2*x*y/3-3*x**2/10+7*y/33-81/55");
    let expected = e("x**2*y/2+y/3-2");
    let accepted = linear_first_order_pde(&a, &b, &c, &forcing, &x, &y).unwrap();
    same_polynomial(accepted.particular(), &expected, &x, &y);
    same_polynomial(accepted.time_coordinate(), &e("3*x/2"), &x, &y);
    same_polynomial(accepted.invariant_coordinate(), &e("y+9*x/10"), &x, &y);
    same_polynomial(
        &original_residual(accepted.particular(), [&a, &b, &c], &forcing, &x, &y),
        &Expr::from_i64(0),
        &x,
        &y,
    );
    assert!(matches!(accepted.homogeneous_coefficient(), Expr::Function(name, _) if name == "exp"),);
}

#[test]
fn both_single_axis_cases_and_homogeneous_reaction_preserve_freedom() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    for (a, b, expected, time, invariant) in [
        (2, 0, "x**2*y/4", "x/2", "y"),
        (0, -2, "-x*y**2/4", "-y/2", "x"),
    ] {
        let accepted =
            linear_first_order_pde(&q(a, 1), &q(b, 1), &q(0, 1), &e("x*y"), &x, &y).unwrap();
        same_polynomial(accepted.particular(), &e(expected), &x, &y);
        same_polynomial(accepted.time_coordinate(), &e(time), &x, &y);
        same_polynomial(accepted.invariant_coordinate(), &e(invariant), &x, &y);
    }

    for c in [-3, 0, 2] {
        let (a, b, reaction) = (q(2, 1), q(3, 1), q(c, 1));
        let accepted = linear_first_order_pde(&a, &b, &reaction, &e("0"), &x, &y).unwrap();
        same_polynomial(accepted.particular(), &e("0"), &x, &y);
        let solution = accepted.homogeneous_coefficient().clone()
            * power(accepted.invariant_coordinate().clone(), 2);
        let residual = original_residual(&solution, [&a, &b, &reaction], &e("0"), &x, &y);
        let residual = fsym_simplify::try_expand(&residual).unwrap();
        assert!(fsym_simplify::try_simplify(&residual).unwrap().is_zero());
    }
}

#[test]
fn independent_checker_accepts_affine_reparameterization_and_invariant_constants() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b, c) = (q(2, 1), q(3, 1), q(0, 1));
    let mut supplied = fixture();
    supplied.time_coordinate = e("x/2+7");
    supplied.invariant_coordinate = e("2*y-3*x+4");
    supplied.particular = e("x**2*y/4-x**3/8+(y-3*x/2)**4+5");
    let accepted =
        verify_linear_first_order_pde([&a, &b, &c], &e("x*y"), [&x, &y], &supplied, &mut Unbounded)
            .unwrap();
    same_polynomial(accepted.particular(), &supplied.particular, &x, &y);
}

#[test]
fn characteristic_and_tiny_particular_mutations_are_rejected() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b, c) = (q(2, 1), q(3, 1), q(0, 1));
    let epsilon = rational_expr(BigRational::new(BigInt::one(), BigInt::from(10).pow(90)));
    for (field, mutation) in [
        (0, e("0")),
        (0, e("x")),
        (1, e("1")),
        (1, e("y")),
        (1, e("(y-3*x/2)**2")),
        (2, e("x**2*y/4")),
        (2, fixture().particular + epsilon * e("x")),
        (2, fixture().particular + e("x**17")),
    ] {
        let mut altered = fixture();
        match field {
            0 => altered.time_coordinate = mutation,
            1 => altered.invariant_coordinate = mutation,
            _ => altered.particular = mutation,
        }
        assert!(
            verify_linear_first_order_pde(
                [&a, &b, &c],
                &e("x*y"),
                [&x, &y],
                &altered,
                &mut Unbounded,
            )
            .is_err(),
            "accepted changed field {field}",
        );
    }
}

#[test]
fn original_coefficients_forcing_and_axes_are_independently_bound() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    for coefficients in [
        [q(3, 1), q(3, 1), q(0, 1)],
        [q(2, 1), q(4, 1), q(0, 1)],
        [q(2, 1), q(3, 1), q(1, 1)],
    ] {
        assert!(
            verify_linear_first_order_pde(
                [&coefficients[0], &coefficients[1], &coefficients[2]],
                &e("x*y"),
                [&x, &y],
                &fixture(),
                &mut Unbounded,
            )
            .is_err(),
        );
    }
    let (a, b, c) = (q(2, 1), q(3, 1), q(0, 1));
    assert!(
        verify_linear_first_order_pde(
            [&a, &b, &c],
            &e("x*y+1/100000000000000000000"),
            [&x, &y],
            &fixture(),
            &mut Unbounded,
        )
        .is_err(),
    );
    assert!(
        verify_linear_first_order_pde(
            [&a, &b, &c],
            &e("x*y"),
            [&y, &x],
            &fixture(),
            &mut Unbounded,
        )
        .is_err(),
    );
}

#[test]
fn highest_supported_forcing_degree_works_for_transport_and_reaction() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b) = (q(1, 1), q(0, 1));
    let forcing = e("x**16");
    let transport = linear_first_order_pde(&a, &b, &q(0, 1), &forcing, &x, &y).unwrap();
    same_polynomial(transport.particular(), &e("x**17/17"), &x, &y);
    let reaction = linear_first_order_pde(&a, &b, &q(1, 1), &forcing, &x, &y).unwrap();
    same_polynomial(
        &original_residual(reaction.particular(), [&a, &b, &q(1, 1)], &forcing, &x, &y),
        &e("0"),
        &x,
        &y,
    );
    assert!(linear_first_order_pde(&a, &b, &q(0, 1), &e("x**17"), &x, &y).is_err(),);
}

#[test]
fn held_source_domains_and_foreign_nodes_cannot_disappear() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let zero = Expr::from_i64(0);
    let xexpr = Expr::Sym(x.clone());
    for invalid in [
        Expr::Sym(Symbol::new("foreign_parameter")),
        Expr::Function("opaque".into(), vec![xexpr.clone()]),
        Expr::Const(Constant::NaN),
        Expr::Const(Constant::Infinity),
        Expr::Add(Vec::new()),
        Expr::Mul(Vec::new()),
        power(xexpr.clone(), -1),
        power(xexpr.clone(), 0),
        power(zero.clone(), 0),
        power(zero.clone(), -1),
        Expr::Mul(vec![xexpr.clone(), power(xexpr.clone(), -1)]),
        Expr::Mul(vec![zero.clone(), power(xexpr.clone(), -1)]),
        power(Expr::Function("opaque".into(), vec![xexpr]), 0),
    ] {
        assert!(linear_first_order_pde(&q(2, 1), &q(3, 1), &q(0, 1), &invalid, &x, &y,).is_err(),);
        assert!(
            verify_linear_first_order_pde(
                [&q(2, 1), &q(3, 1), &q(0, 1)],
                &invalid,
                [&x, &y],
                &fixture(),
                &mut Unbounded,
            )
            .is_err(),
        );
    }
    // Literal rational reciprocals and nonzero zero-powers are globally defined.
    let forcing = Expr::Add(vec![
        power(Expr::from_i64(2), -1),
        power(Expr::from_i64(3), 0),
    ]);
    let accepted = linear_first_order_pde(&q(1, 1), &q(0, 1), &q(0, 1), &forcing, &x, &y).unwrap();
    same_polynomial(accepted.particular(), &e("3*x/2"), &x, &y);
}

#[test]
fn same_name_typed_coordinates_remain_distinct() {
    let x = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([2; 32]));
    let forcing = Expr::Sym(x.clone()) + Expr::Sym(y.clone());
    let accepted = linear_first_order_pde(&q(1, 1), &q(0, 1), &q(0, 1), &forcing, &x, &y).unwrap();
    assert_eq!(accepted.variables(), &[x.clone(), y.clone()]);
    same_polynomial(
        accepted.particular(),
        &(rational_expr(q(1, 2)) * power(Expr::Sym(x.clone()), 2)
            + Expr::Sym(x.clone()) * Expr::Sym(y.clone())),
        &x,
        &y,
    );
    assert!(linear_first_order_pde(&q(1, 1), &q(0, 1), &q(0, 1), &e("0"), &x, &x,).is_err(),);
}

#[test]
fn zero_transport_and_oversized_original_inputs_refuse() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    assert!(matches!(
        linear_first_order_pde(&q(0, 1), &q(0, 1), &q(1, 1), &e("0"), &x, &y,),
        Err(PdeError::InvalidSource(_)),
    ));
    let huge = BigRational::from_integer(BigInt::one() << 4097u32);
    assert!(matches!(
        linear_first_order_pde(&huge, &q(0, 1), &q(0, 1), &e("0"), &x, &y),
        Err(PdeError::ResourceLimit(_)),
    ));
    let mut deep = e("x");
    for _ in 0..66 {
        deep = power(deep, 1);
    }
    for forcing in [
        deep,
        power(e("x"), i64::MAX),
        power(power(e("2"), 64), 64),
        rational_expr(huge),
    ] {
        assert!(linear_first_order_pde(&q(1, 1), &q(0, 1), &q(1, 1), &forcing, &x, &y,).is_err(),);
    }
}

#[derive(Default)]
struct CountingMeter {
    compute: u64,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CountingMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        for &(dimension, amount) in charges {
            if dimension == Dimension::ComputeSteps {
                self.compute += amount;
            }
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.checkpoints += 1;
        if self
            .cancel_at
            .is_some_and(|limit| self.checkpoints >= limit)
        {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn generation_and_verification_share_cumulative_charges_and_cancellation() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b, c) = (q(2, 1), q(3, 1), q(0, 1));
    let forcing = e("x*y");
    let mut meter = CountingMeter::default();
    let accepted =
        metered_linear_first_order_pde(&a, &b, &c, &forcing, &x, &y, &mut meter).unwrap();
    let first_charges = meter.compute;
    let first_checkpoints = meter.checkpoints;
    assert!(first_charges > 0);
    verify_linear_first_order_pde(
        [&a, &b, &c],
        &forcing,
        [&x, &y],
        accepted.certificate(),
        &mut meter,
    )
    .unwrap();
    assert!(meter.compute > first_charges);
    for cancel_at in [1, 20, first_checkpoints - 1] {
        let mut meter = CountingMeter {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert!(matches!(
            metered_linear_first_order_pde(&a, &b, &c, &forcing, &x, &y, &mut meter,),
            Err(PdeError::Meter(MeterError::Cancelled)),
        ));
    }
    let mut limits = BudgetLimits::uniform(100_000_000, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = first_charges - 1;
    let mut budget = Budget::new(limits);
    assert!(matches!(
        metered_linear_first_order_pde(&a, &b, &c, &forcing, &x, &y, &mut budget),
        Err(PdeError::Meter(MeterError::Budget(_))),
    ));
}
