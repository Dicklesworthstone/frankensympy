//! Exact coefficients and refusal boundaries for the generalized series lane.

use fsym_calculus::series::{SeriesError, series_in};
use fsym_core::{BigRational, Constant, Expr, Symbol};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn nonlinear_logarithms_cannot_be_proved_affine_by_three_samples() {
    let t = Symbol::new("t");
    let logarithm = function("log", Expr::Sym(t.clone()));
    // This nonzero cubic vanishes at all three points used by the old
    // recognizer. Its exponential is not the constant one near t = 0+.
    let impostor = logarithm.clone()
        * (logarithm.clone() - Expr::from_i64(1))
        * (logarithm.clone() - Expr::from_i64(2));
    for exponent in [
        impostor.clone(),
        impostor + Expr::from_i64(2) * logarithm.clone() + Expr::from_i64(7),
    ] {
        assert!(matches!(
            series_in(&function("exp", exponent), &t, &q(5, 1), &logarithm),
            Err(SeriesError::Unsupported(_))
        ));
    }
}

#[test]
fn exact_affine_logarithms_still_shift_the_exponential_series() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    // exp(3 log(t) + t) = t^3 exp(t); coefficients follow 1/k!.
    let source = function("exp", Expr::from_i64(3) * logarithm.clone() + te);
    let expanded = series_in(&source, &t, &q(7, 1), &logarithm).unwrap();
    assert_eq!(
        expanded.terms,
        BTreeMap::from([
            (q(3, 1), Expr::from_i64(1)),
            (q(4, 1), Expr::from_i64(1)),
            (q(5, 1), Expr::Rational(q(1, 2))),
            (q(6, 1), Expr::Rational(q(1, 6))),
        ])
    );
    assert_eq!(expanded.order, Some(q(7, 1)));
}

#[test]
fn powers_with_poles_keep_terms_needed_by_later_products() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    for exponent in 2..=8 {
        // (t^-1 + t^2)^m = sum_j binom(m,j) t^(3j-m).
        // The first factor's t^2 term lies beyond the requested order,
        // but products with remaining poles can bring it back below it.
        let source = power(power(te.clone(), -1) + power(te.clone(), 2), exponent);
        let expanded = series_in(&source, &t, &q(2, 1), &logarithm).unwrap();
        let mut expected = BTreeMap::new();
        let mut binomial = 1_i64;
        for j in 0..=exponent {
            if 3 * j - exponent < 2 {
                expected.insert(q(3 * j - exponent, 1), Expr::from_i64(binomial));
            }
            if j < exponent {
                binomial = binomial * (exponent - j) / (j + 1);
            }
        }
        assert_eq!(expanded.terms, expected, "power {exponent}");
        assert_eq!(expanded.order, Some(q(2, 1)));
    }
}

#[test]
fn analytic_laurent_powers_have_the_correct_constant_term() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    // (1/t + sin(t))^2 = t^-2 + 2 + 2/3 t^2 + O(t^4).
    let source = power(power(te.clone(), -1) + function("sin", te), 2);
    let expanded = series_in(&source, &t, &q(1, 1), &logarithm).unwrap();
    assert_eq!(
        expanded.terms,
        BTreeMap::from([(q(-2, 1), Expr::from_i64(1)), (q(0, 1), Expr::from_i64(2))])
    );
    assert_eq!(expanded.order, Some(q(1, 1)));
}

#[test]
fn exponential_cancellation_is_resolved_before_inverting_a_local_series() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    let base = function("exp", te.clone()) - Expr::from_i64(1);
    let inverse = power(base.clone(), -1);
    let expanded = series_in(&inverse, &t, &q(5, 1), &logarithm).unwrap();
    // Bernoulli coefficients: 1/(exp(t)-1) = 1/t - 1/2 + t/12 - t^3/720 + O(t^5).
    assert_eq!(
        expanded.terms,
        BTreeMap::from([
            (q(-1, 1), Expr::from_i64(1)),
            (q(0, 1), Expr::Rational(q(-1, 2))),
            (q(1, 1), Expr::Rational(q(1, 12))),
            (q(3, 1), Expr::Rational(q(-1, 720))),
        ])
    );
    assert_eq!(expanded.order, Some(q(5, 1)));

    // Removing the linear term changes the pole order. The identity
    // (t^2/2 + t^3/6 + ...)*(2/t^2 - 2/(3t) + ...) = 1 fixes both coefficients.
    let twice_canceled = power(base - te.clone(), -1);
    let expanded = series_in(&twice_canceled, &t, &q(0, 1), &logarithm).unwrap();
    assert_eq!(
        expanded.terms,
        BTreeMap::from([
            (q(-2, 1), Expr::from_i64(2)),
            (q(-1, 1), Expr::Rational(q(-2, 3))),
        ])
    );
    assert_eq!(expanded.order, Some(q(0, 1)));

    // An even germ has no residue; cancellation must still reveal its double pole.
    let even = power(function("exp", power(te, 2)) - Expr::from_i64(1), -1);
    let expanded = series_in(&even, &t, &q(0, 1), &logarithm).unwrap();
    assert_eq!(
        expanded.terms,
        BTreeMap::from([(q(-2, 1), Expr::from_i64(1))])
    );
    assert_eq!(expanded.order, Some(q(0, 1)));
}

#[test]
fn exponential_reciprocal_powers_keep_residues_at_shifted_scaled_centers() {
    let x = Symbol::new("x");
    for center in [Expr::from_i64(0), Expr::Rational(q(3, 2))] {
        for scale in [q(1, 1), q(-2, 3)] {
            let local = Expr::Rational(scale.clone()) * (Expr::Sym(x.clone()) - center.clone());
            let base = function("exp", local) - Expr::from_i64(1);
            for multiplicity in [1, 2, 3, 8] {
                let source = power(base.clone(), -multiplicity);
                let (terms, order) =
                    fsym_calculus::series_expansion(&source, &x, &center, 0).unwrap();
                // z=exp(scale*(x-center))-1 gives dx=dz/[scale*(1+z)],
                // so the residue of z^-m is (-1)^(m-1)/scale exactly.
                let sign = if multiplicity % 2 == 1 { 1 } else { -1 };
                let expected = fsym_core::elementary::rational_expr(q(sign, 1) / scale.clone());
                assert_eq!(
                    terms.first().map(|(_, exponent)| exponent),
                    Some(&q(-multiplicity, 1))
                );
                assert_eq!(
                    terms
                        .iter()
                        .find(|(_, exponent)| exponent == &q(-1, 1))
                        .map(|(value, _)| value),
                    Some(&expected),
                    "center {center}, scale {scale}, multiplicity {multiplicity}"
                );
                assert_eq!(order, Some(q(0, 1)));
            }
        }
    }
}

#[test]
fn logarithm_temporaries_do_not_replace_user_parameters() {
    let t = Symbol::new("t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    let parameter = Expr::symbol("_fsym_log_t");
    let second = Expr::symbol("_fsym_log_t_1");
    let source = parameter.clone() + second.clone() + function("exp", te);
    let expanded = series_in(&source, &t, &q(3, 1), &logarithm).unwrap();
    assert_eq!(
        expanded.terms.get(&q(0, 1)),
        Some(&(parameter + second + Expr::from_i64(1)))
    );
    assert_eq!(expanded.terms.get(&q(1, 1)), Some(&Expr::from_i64(1)));
    assert_eq!(expanded.terms.get(&q(2, 1)), Some(&Expr::Rational(q(1, 2))));
}

#[test]
fn logarithms_and_same_named_parameters_remain_distinct() {
    let t = Symbol::new("_fsym_log_t");
    let te = Expr::Sym(t.clone());
    let logarithm = function("log", te.clone());
    let source = logarithm.clone() + te;
    let expanded = series_in(&source, &t, &q(2, 1), &logarithm).unwrap();
    assert_eq!(expanded.terms.get(&q(0, 1)), Some(&logarithm));
    assert_eq!(expanded.terms.get(&q(1, 1)), Some(&Expr::from_i64(1)));
    assert_eq!(expanded.order, None);
}

#[test]
fn public_series_local_coordinate_cannot_capture_parameters_or_centers() {
    let x = Symbol::new("x");
    let parameter = Expr::symbol("_fsym_series_t");
    let sine = function("sin", Expr::Sym(x.clone()));
    let (terms, order) = fsym_calculus::series_expansion(
        &(sine.clone() + parameter.clone()),
        &x,
        &Expr::from_i64(0),
        3,
    )
    .unwrap();
    assert_eq!(
        terms,
        vec![(parameter.clone(), q(0, 1)), (Expr::from_i64(1), q(1, 1))]
    );
    assert_eq!(order, Some(q(3, 1)));

    let (terms, order) = fsym_calculus::series_expansion(&sine, &x, &parameter, 2).unwrap();
    assert_eq!(
        terms,
        vec![
            (function("sin", parameter.clone()), q(0, 1)),
            (function("cos", parameter), q(1, 1)),
        ]
    );
    assert_eq!(order, Some(q(2, 1)));
}

#[test]
fn generalized_series_preflights_depth_before_recursive_symbol_queries() {
    let t = Symbol::new("t");
    let mut source = Expr::Sym(t.clone());
    for _ in 0..80 {
        source = function("sin", source);
    }
    assert!(matches!(
        series_in(&source, &t, &q(2, 1), &Expr::from_i64(0)),
        Err(SeriesError::Unsupported(_))
    ));
}

#[test]
fn limit_scales_cannot_capture_declared_user_parameters() {
    use fsym_core::assume::{Facts, with_symbol_facts};

    let x = Symbol::new("x");
    for name in ["_fsym_w1", "_fsym_w1_1", "_fsym_log_t"] {
        let parameter = Symbol::new(name);
        let value = Expr::Sym(parameter.clone());
        let source = value.clone() + function("exp", -Expr::Sym(x.clone()));
        let actual = with_symbol_facts(
            HashMap::from([(parameter, Facts::from_names(&["positive"]))]),
            || fsym_calculus::limit(&source, &x, &Expr::Const(Constant::Infinity)),
        )
        .unwrap();
        assert_eq!(actual, value, "parameter {name}");
    }
}

#[test]
fn internal_coordinates_do_not_inherit_unrelated_ambient_assumptions() {
    use fsym_core::assume::{Facts, with_symbol_facts};

    let x = Symbol::new("x");
    let source = function("sin", Expr::Const(Constant::Pi) * Expr::Sym(x.clone()));
    // Neither reserved name occurs in the source. Reusing one of them
    // would let the substitution fold sin(pi*t) to zero for an integer t.
    let facts = ["_fsym_series_t", "_fsym_series_t_1"]
        .into_iter()
        .map(|name| (Symbol::new(name), Facts::from_names(&["integer"])))
        .collect();
    let (terms, order) = with_symbol_facts(facts, || {
        fsym_calculus::series_expansion(&source, &x, &Expr::from_i64(0), 3)
    })
    .unwrap();
    assert_eq!(terms, vec![(Expr::Const(Constant::Pi), q(1, 1))]);
    assert_eq!(order, Some(q(3, 1)));
}
