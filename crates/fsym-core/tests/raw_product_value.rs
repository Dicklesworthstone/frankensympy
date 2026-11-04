//! Raw numeric products must retain value during addition and rendering.

#![forbid(unsafe_code)]

use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol, parse};
use num_traits::{One, Zero};

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

/// An independent scalar interpretation: do not call Expr's constructors,
/// substitution, constant collection or simplification to compare values.
fn exact_at(expression: &Expr, variable: &Symbol, value: &BigRational) -> BigRational {
    match expression {
        Expr::Integer(number) => BigRational::from_integer(number.clone()),
        Expr::Rational(number) => number.clone(),
        Expr::Sym(symbol) => {
            assert_eq!(symbol, variable);
            value.clone()
        }
        Expr::Add(terms) => terms.iter().fold(BigRational::zero(), |sum, term| {
            sum + exact_at(term, variable, value)
        }),
        Expr::Mul(factors) => factors.iter().fold(BigRational::one(), |product, factor| {
            product * exact_at(factor, variable, value)
        }),
        Expr::Pow(base, exponent) => {
            let Expr::Integer(power) = exponent.as_ref() else {
                panic!("the test fragment has only integer powers");
            };
            let power = power.to_i64().unwrap();
            assert!(power.unsigned_abs() <= 16);
            let base = exact_at(base, variable, value);
            let result =
                (0..power.unsigned_abs()).fold(BigRational::one(), |product, _| product * &base);
            if power < 0 {
                BigRational::one() / result
            } else {
                result
            }
        }
        other => panic!("outside the exact test fragment: {other:?}"),
    }
}

#[test]
fn additive_collection_preserves_the_shifted_ode_correction() {
    let x = Expr::symbol("x");
    let raw = Expr::Mul(vec![Expr::from_i64(-1), rational_expr(q(334, 495))]);
    assert_eq!(x.clone() + raw, x + rational_expr(q(-334, 495)));
}

#[test]
fn all_scalar_factors_survive_addition_in_either_order() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    for (factors, expected) in [
        (vec![Expr::from_i64(2), Expr::from_i64(3)], q(6, 1)),
        (vec![rational_expr(q(2, 3)), Expr::from_i64(5)], q(10, 3)),
        (vec![Expr::from_i64(2), rational_expr(q(-3, 7))], q(-6, 7)),
        (vec![Expr::from_i64(-2), rational_expr(q(-3, 7))], q(6, 7)),
        (vec![Expr::from_i64(2), Expr::from_i64(0)], q(0, 1)),
        (vec![Expr::from_i64(0), rational_expr(q(4, 7))], q(0, 1)),
    ] {
        let raw = Expr::Mul(factors);
        for sum in [x.clone() + raw.clone(), raw + x.clone()] {
            for point in [q(0, 1), q(-3, 2), q(7, 3)] {
                assert_eq!(exact_at(&sum, &variable, &point), &point + &expected);
            }
        }
    }
}

#[test]
fn numeric_factor_permutations_and_nested_products_preserve_additive_value() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    let factors = [
        rational_expr(q(-3, 2)),
        rational_expr(q(5, 7)),
        Expr::from_i64(2),
    ];
    let expected = q(-15, 7);
    for indices in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let [a, b, c] = indices.map(|index| factors[index].clone());
        for raw in [
            Expr::Mul(vec![a.clone(), b.clone(), c.clone()]),
            Expr::Mul(vec![a.clone(), Expr::Mul(vec![b.clone(), c.clone()])]),
            Expr::Mul(vec![Expr::Mul(vec![a.clone(), b.clone()]), c.clone()]),
            Expr::Mul(vec![Expr::Mul(vec![a]), Expr::Mul(vec![b, c])]),
        ] {
            let sum = x.clone() + raw;
            assert_eq!(exact_at(&sum, &variable, &q(2, 3)), q(2, 3) + &expected);
        }
    }
}

#[test]
fn normalization_comparison_cannot_turn_a_nonunit_scalar_into_one() {
    let tiny = BigRational::new(BigInt::one(), BigInt::from(10).pow(400));
    for value in [q(334, 495), q(-2, 3), q(0, 1), tiny] {
        let raw = Expr::Mul(vec![Expr::from_i64(1), rational_expr(value.clone())]);
        let difference = raw.clone() + Expr::from_i64(-1);
        assert_eq!(difference, rational_expr(&value - BigRational::one()));
        assert!(!difference.is_zero());
        assert_eq!(raw - rational_expr(value), Expr::from_i64(0));
    }
}

fn assert_display_value(expression: &Expr, variable: &Symbol) {
    let rendered = expression.to_string();
    let parsed = parse(&rendered).unwrap_or_else(|error| panic!("{rendered}: {error}"));
    for point in [q(0, 1), q(-3, 2), q(7, 3)] {
        assert_eq!(
            exact_at(&parsed, variable, &point),
            exact_at(expression, variable, &point),
            "rendering changed the exact value: {rendered}"
        );
    }
}

#[test]
fn sign_aware_add_rendering_negates_only_one_raw_numeric_factor() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    for factors in [
        vec![Expr::from_i64(-1), rational_expr(q(334, 495))],
        vec![Expr::from_i64(-1), rational_expr(q(-334, 495))],
        vec![Expr::from_i64(-2), Expr::from_i64(3), x.clone()],
        vec![Expr::from_i64(-2), Expr::from_i64(-3), x.clone()],
        vec![rational_expr(q(-2, 3)), Expr::from_i64(5), x.clone()],
        vec![Expr::from_i64(-1), Expr::from_i64(0), x.clone()],
    ] {
        let raw = Expr::Mul(factors);
        // Exercise both the initial Add term and the subsequent sign-aware join.
        assert_display_value(&Expr::Add(vec![x.clone(), raw.clone()]), &variable);
        assert_display_value(&Expr::Add(vec![raw, x.clone()]), &variable);
    }
}

#[test]
fn raw_product_display_roundtrips_factor_orders_nesting_and_tiny_rationals() {
    let variable = Symbol::new("x");
    let x = Expr::Sym(variable.clone());
    let tiny = rational_expr(BigRational::new(BigInt::one(), BigInt::from(10).pow(400)));
    let factors = [Expr::from_i64(-2), tiny, x.clone()];
    for indices in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let [a, b, c] = indices.map(|index| factors[index].clone());
        for raw in [
            Expr::Mul(vec![a.clone(), b.clone(), c.clone()]),
            Expr::Mul(vec![a.clone(), Expr::Mul(vec![b.clone(), c.clone()])]),
            Expr::Mul(vec![Expr::Mul(vec![a, b]), c]),
        ] {
            assert_display_value(&raw, &variable);
            assert_display_value(&Expr::Add(vec![x.clone(), raw]), &variable);
        }
    }
}
