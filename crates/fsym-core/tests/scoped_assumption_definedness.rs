//! Held source domains precede every structural assumption deduction.

use fsym_core::assume::{
    Facts, facts_of_symbol, is_integer, is_nonzero, is_real, sign, with_symbol_facts,
};
use fsym_core::elementary::{eval_function, eval_pow};
use fsym_core::{Constant, Expr, Symbol};
use std::collections::HashMap;
use std::sync::Arc;

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn no_finite_fact(source: &Expr) {
    assert_eq!(sign(source), None, "{source:?}");
    assert!(!is_real(source), "{source:?}");
    assert!(!is_integer(source), "{source:?}");
    assert!(!is_nonzero(source), "{source:?}");
}

#[test]
fn a_held_zero_reciprocal_is_not_a_real_number() {
    let reciprocal = power(Expr::from_i64(0), -1);
    no_finite_fact(&reciprocal);
    assert_eq!(
        eval_pow(Expr::from_i64(0), Expr::from_i64(-1)),
        Expr::Const(Constant::ComplexInfinity)
    );
}

#[test]
fn exponential_cannot_claim_nonzero_after_an_undefined_logarithm() {
    let logarithm = function("log", Expr::from_i64(0));
    no_finite_fact(&function("exp", logarithm));
    // The public extended-value convention is log(0)=-oo and exp(-oo)=0.
    // Either that convention or refusing the original logarithm rules out
    // the old unconditional nonzero assertion.
    let evaluated_log = eval_function("log", &[Expr::from_i64(0)]).unwrap();
    assert_eq!(evaluated_log, Expr::Const(Constant::NegativeInfinity));
    assert_eq!(
        eval_function("exp", &[evaluated_log]),
        Some(Expr::from_i64(0))
    );
}

#[test]
fn an_undefined_exponential_argument_cannot_collapse_a_zero_base_power() {
    let exponent = function("exp", power(Expr::from_i64(0), -1));
    let rewritten = eval_pow(Expr::from_i64(0), exponent.clone());
    assert_eq!(
        rewritten,
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(exponent))
    );
    let reciprocal = eval_pow(Expr::from_i64(0), Expr::from_i64(-1));
    let evaluated_exponent = eval_function("exp", &[reciprocal]).unwrap();
    assert_eq!(evaluated_exponent, Expr::Const(Constant::NaN));
    assert_eq!(
        eval_pow(Expr::from_i64(0), evaluated_exponent),
        Expr::Const(Constant::NaN)
    );
}

#[test]
fn literal_infinity_signs_do_not_authorize_arithmetic_or_inversion_facts() {
    for (constant, expected_sign) in [(Constant::Infinity, 1), (Constant::NegativeInfinity, -1)] {
        let infinity = Expr::Const(constant);
        assert_eq!(sign(&infinity), Some(expected_sign));
        assert!(!is_real(&infinity));
        assert!(!is_nonzero(&infinity));
        no_finite_fact(&power(infinity.clone(), -1));
        no_finite_fact(&Expr::Mul(vec![Expr::from_i64(0), infinity.clone()]));
        assert_eq!(eval_pow(infinity, Expr::from_i64(-1)), Expr::from_i64(0));
    }
}

#[test]
fn extended_real_declarations_do_not_erase_the_negative_infinity_case() {
    let x = Symbol::new("extended_real_coordinate");
    let declared = Facts::from_names(&["extended_real"]);
    assert!(!declared.real);
    assert_ne!(declared, Facts::default());
    let exponent = function("exp", Expr::Sym(x.clone()));
    let original = Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(exponent.clone()));
    let rewritten = with_symbol_facts(HashMap::from([(x.clone(), declared)]), || {
        assert_eq!(facts_of_symbol(&x), Some(declared));
        assert!(!is_real(&Expr::Sym(x.clone())));
        assert!(!is_nonzero(&exponent));
        eval_pow(Expr::from_i64(0), exponent)
    });
    assert_eq!(rewritten, original);
    // exp(-oo)=0, and the existing constructor convention is 0**0=1.
    assert_eq!(
        rewritten.subs(&HashMap::from([(
            x,
            Expr::Const(Constant::NegativeInfinity)
        )])),
        Expr::from_i64(1)
    );
}

#[test]
fn finite_real_declarations_still_discharge_the_exponential_power_rule() {
    let x = Symbol::new("finite_real_coordinate");
    for names in [&["real"][..], &["integer"][..], &["nonzero"][..]] {
        let declared = Facts::from_names(names);
        assert!(declared.real);
        with_symbol_facts(HashMap::from([(x.clone(), declared)]), || {
            let exponent = function("exp", Expr::Sym(x.clone()));
            assert!(is_real(&exponent));
            assert!(is_nonzero(&exponent));
            assert_eq!(eval_pow(Expr::from_i64(0), exponent), Expr::from_i64(0));
        });
    }
}

#[test]
fn direct_facts_records_preserve_finite_real_implications() {
    let x = Symbol::new("direct_finite_coordinate");
    for declared in [
        Facts {
            nonzero: true,
            ..Facts::default()
        },
        Facts {
            nonnegative: true,
            ..Facts::default()
        },
        Facts {
            nonpositive: true,
            ..Facts::default()
        },
    ] {
        with_symbol_facts(HashMap::from([(x.clone(), declared)]), || {
            assert!(is_real(&Expr::Sym(x.clone())));
            let exponent = function("exp", Expr::Sym(x.clone()));
            assert!(is_nonzero(&exponent));
            assert_eq!(eval_pow(Expr::from_i64(0), exponent), Expr::from_i64(0));
        });
    }
}

#[test]
fn hidden_undefined_children_survive_zero_powers_and_zero_products() {
    for original in [
        power(power(Expr::from_i64(0), -1), 0),
        power(function("log", Expr::from_i64(0)), 0),
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
    ] {
        no_finite_fact(&original);
        no_finite_fact(&function("exp", original));
    }
    // A defined finite original still supports the existing zero-power
    // convention, even when the original itself is zero.
    for base in [0, 2, -3] {
        let defined = power(Expr::from_i64(base), 0);
        assert_eq!(sign(&defined), Some(1));
        assert!(is_integer(&defined));
        assert!(is_nonzero(&defined));
    }
}

#[test]
fn negative_integer_powers_require_and_retain_nonzero_real_bases() {
    let x = Symbol::new("reciprocal_coordinate");
    let reciprocal = power(Expr::Sym(x.clone()), -2);
    with_symbol_facts(
        HashMap::from([(x.clone(), Facts::from_names(&["real"]))]),
        || no_finite_fact(&reciprocal),
    );
    with_symbol_facts(
        HashMap::from([(x, Facts::from_names(&["real", "nonzero"]))]),
        || {
            assert!(is_real(&reciprocal));
            assert!(is_nonzero(&reciprocal));
            assert!(!is_integer(&reciprocal));
        },
    );
    for (exponent, expected_sign) in [(-3, -1), (-2, 1), (3, -1), (2, 1)] {
        let known = power(Expr::from_i64(-2), exponent);
        assert_eq!(sign(&known), Some(expected_sign));
        assert!(is_real(&known));
        assert!(is_nonzero(&known));
    }
}

#[test]
fn finite_complex_exponentials_remain_nonzero_without_becoming_real() {
    for argument in [
        Expr::Const(Constant::I),
        Expr::Add(vec![Expr::from_i64(2), Expr::Const(Constant::I)]),
        power(Expr::Const(Constant::I), -1),
    ] {
        let exponential = function("exp", argument);
        assert!(is_nonzero(&exponential));
        assert!(!is_real(&exponential));
        assert_eq!(sign(&exponential), None);
    }
    // exp(log(-1)) is -1, so its nonzero guarantee does not imply nonreality.
    assert!(is_nonzero(&function(
        "exp",
        function("log", Expr::from_i64(-1))
    )));
}

#[test]
fn unknown_and_contradictory_arguments_do_not_acquire_finiteness() {
    let x = Symbol::new("unknown_finite_coordinate");
    let exponential = function("exp", Expr::Sym(x.clone()));
    no_finite_fact(&exponential);
    with_symbol_facts(
        HashMap::from([(x, Facts::from_names(&["positive", "negative"]))]),
        || no_finite_fact(&exponential),
    );
    no_finite_fact(&function(
        "exp",
        function("unknown_function", Expr::from_i64(1)),
    ));
}

#[test]
fn malformed_functions_and_complex_poles_cannot_discharge_facts() {
    for expression in [
        Expr::Function("exp".into(), vec![]),
        Expr::Function("exp".into(), vec![Expr::from_i64(0), Expr::from_i64(1)]),
        function("atan", Expr::Const(Constant::I)),
        function(
            "tanh",
            Expr::Mul(vec![
                Expr::Const(Constant::I),
                Expr::Const(Constant::Pi),
                Expr::rational(1, 2).unwrap(),
            ]),
        ),
    ] {
        no_finite_fact(&expression);
        no_finite_fact(&function("exp", expression));
    }
}

#[test]
fn bounded_queries_fail_closed_without_recursive_guard_explosion() {
    let mut nested = Expr::from_i64(1);
    for _ in 0..32 {
        nested = power(nested, -1);
    }
    assert!(is_real(&nested));
    assert!(is_nonzero(&nested));
    for _ in 32..140 {
        nested = power(nested, -1);
    }
    no_finite_fact(&nested);
    let wide = Expr::Add(vec![Expr::from_i64(1); 16_384]);
    no_finite_fact(&wide);
}
