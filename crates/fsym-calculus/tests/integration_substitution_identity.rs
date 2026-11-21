//! Substitution integration must preserve the source's complete symbol universe.

use fsym_calculus::{diff, integrate::antiderivative};
use fsym_core::elementary::eval_pow;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use std::collections::BTreeSet;

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.to_owned(), vec![argument])
}

fn half() -> Expr {
    Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)))
}

fn assert_same_expression(actual: Expr, expected: Expr) {
    let difference = fsym_simplify::simplify(&(actual.clone() - expected.clone()));
    assert!(difference.is_zero(), "expected {expected}, got {actual}");
}

fn assert_substitution_preserves_parameters(variable: Symbol, parameters: &[Symbol]) {
    let x = Expr::Sym(variable.clone());
    let square = eval_pow(x.clone(), Expr::from_i64(2));
    let parameter_sum = parameters
        .iter()
        .map(|parameter| Expr::Sym(parameter.clone()))
        .fold(Expr::from_i64(0), |sum, parameter| sum + parameter);
    let integrand = x * (parameter_sum.clone() + function("cos", square.clone()));
    let expected = half() * (parameter_sum * square.clone() + function("sin", square));
    let result = antiderivative(&integrand, &variable)
        .unwrap_or_else(|| panic!("substitution failed for {integrand}"));

    // Both a known antiderivative and the independent differentiation path
    // detect parameter capture; numerical samples alone would not suffice.
    assert_same_expression(result.clone(), expected);
    assert_same_expression(diff(&result, &variable), integrand.clone());
    assert_eq!(
        result.free_symbols().into_iter().collect::<BTreeSet<_>>(),
        integrand
            .free_symbols()
            .into_iter()
            .collect::<BTreeSet<_>>()
    );
    assert_eq!(antiderivative(&integrand, &variable), Some(result));
}

#[test]
fn user_parameter_cannot_be_captured_by_the_substitution_variable() {
    assert_substitution_preserves_parameters(Symbol::new("x"), &[Symbol::new("_fsym_sub1")]);
}

#[test]
fn occupied_freshness_suffixes_are_skipped_without_losing_parameters() {
    assert_substitution_preserves_parameters(
        Symbol::new("x"),
        &[
            Symbol::new("_fsym_sub1"),
            Symbol::new("_fsym_sub1_0"),
            Symbol::new("_fsym_sub1_1"),
        ],
    );
}

#[test]
fn integration_axis_can_have_the_internal_substitution_name() {
    assert_substitution_preserves_parameters(
        Symbol::new("_fsym_sub1"),
        &[Symbol::new("parameter")],
    );
}

#[test]
fn equal_printed_parameters_keep_their_distinct_typed_identities() {
    assert_substitution_preserves_parameters(
        Symbol::new("x"),
        &[
            Symbol::new("_fsym_sub1"),
            Symbol::with_identity(
                "_fsym_sub1",
                SymbolIdentity::from_assumptions_digest([17; 32]),
            ),
            Symbol::with_identity(
                "_fsym_sub1",
                SymbolIdentity::from_assumptions_digest([23; 32]),
            ),
        ],
    );
}

#[test]
fn typed_axis_and_plain_parameter_with_equal_names_remain_distinct() {
    assert_substitution_preserves_parameters(
        Symbol::with_identity(
            "_fsym_sub1",
            SymbolIdentity::from_assumptions_digest([31; 32]),
        ),
        &[Symbol::new("_fsym_sub1")],
    );
}

#[test]
fn ordinary_substitution_retains_its_existing_result() {
    assert_substitution_preserves_parameters(
        Symbol::new("x"),
        &[Symbol::new("a"), Symbol::new("b")],
    );
}
