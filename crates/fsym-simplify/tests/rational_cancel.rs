//! Native consumer regressions: complete rational functions reach the checked
//! polynomial engine before the old expression expansion or zero shortcuts.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_polys::bounded_polynomial_from_expr;
use fsym_simplify::{SimplifyError, cancel_with, simplify, try_cancel};
use std::sync::Arc;

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn fraction(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, power(denominator, -1)])
}

fn difference(a: Expr, b: Expr) -> Expr {
    Expr::Add(vec![a, Expr::Mul(vec![Expr::from_i64(-1), b])])
}

#[test]
fn complete_nonlinear_multivariate_factor_is_cancelled() {
    let x = Expr::symbol("cancel_native_x");
    let y = Expr::symbol("cancel_native_y");
    let common = Expr::Add(vec![
        Expr::Mul(vec![x.clone(), y.clone()]),
        Expr::from_i64(1),
    ]);
    let numerator = Expr::Add(vec![
        x.clone(),
        Expr::Mul(vec![Expr::from_i64(2), y.clone()]),
    ]);
    let denominator = difference(x, y);
    let source = fraction(
        Expr::Mul(vec![common.clone(), numerator.clone()]),
        Expr::Mul(vec![common, denominator.clone()]),
    );
    assert_eq!(
        try_cancel(&source).unwrap(),
        simplify(&fraction(numerator, denominator))
    );
}

#[test]
fn degree_65_input_does_not_use_the_expression_expansion_limit() {
    let x = Symbol::new("cancel_degree_x");
    let y = Symbol::new("cancel_degree_y");
    let sum = Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]);
    let source = fraction(
        power(sum.clone(), 65),
        Expr::Mul(vec![Expr::from_i64(2), sum]),
    );
    let result = try_cancel(&source).unwrap();
    let polynomial = bounded_polynomial_from_expr(&result, &[x, y], &mut Unbounded).unwrap();
    assert_eq!(polynomial.terms().len(), 65);
    // Independently generate the binomial coefficients of (x+y)^64 / 2.
    let mut binomial = BigInt::from(1);
    for k in 0..=64u32 {
        assert_eq!(
            polynomial.terms().get(&vec![64 - k, k]),
            Some(&BigRational::new(binomial.clone(), BigInt::from(2)))
        );
        if k < 64 {
            binomial = binomial * BigInt::from(64 - k) / BigInt::from(k + 1);
        }
    }
}

#[test]
fn complete_sum_and_nested_reciprocal_are_lowered_from_the_original() {
    let x = Expr::symbol("cancel_sum_x");
    let y = Expr::symbol("cancel_sum_y");
    let plus = Expr::Add(vec![x.clone(), y.clone()]);
    let minus = difference(x.clone(), y.clone());
    let squares = difference(power(x.clone(), 2), power(y, 2));
    let sum = Expr::Add(vec![power(plus.clone(), -1), power(minus.clone(), -1)]);
    let expected = fraction(Expr::Mul(vec![Expr::from_i64(2), x]), squares.clone());
    assert_eq!(try_cancel(&sum).unwrap(), simplify(&expected));
    let nested = power(fraction(squares, minus), -1);
    assert_eq!(try_cancel(&nested).unwrap(), simplify(&power(plus, -1)));
}

#[test]
fn identically_printed_typed_symbols_stay_distinct() {
    let a = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([1; 32]));
    let b = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([2; 32]));
    let a = Expr::Sym(a);
    let b = Expr::Sym(b);
    let source = fraction(
        difference(power(a.clone(), 2), power(b.clone(), 2)),
        difference(a.clone(), b.clone()),
    );
    assert_eq!(
        try_cancel(&source).unwrap(),
        simplify(&Expr::Add(vec![a, b]))
    );
}

#[test]
fn exact_scalar_ring_and_nonrational_function_arguments_are_supported() {
    let scalar = fraction(Expr::from_i64(3), Expr::from_i64(-6));
    assert_eq!(try_cancel(&scalar).unwrap(), Expr::rational(-1, 2).unwrap());
    let x = Expr::symbol("cancel_function_x");
    let source = Expr::Function(
        "sin".into(),
        vec![fraction(
            difference(power(x.clone(), 2), Expr::from_i64(1)),
            difference(x.clone(), Expr::from_i64(1)),
        )],
    );
    assert_eq!(
        try_cancel(&source).unwrap(),
        Expr::Function(
            "sin".into(),
            vec![simplify(&Expr::Add(vec![x, Expr::from_i64(1)]))]
        )
    );
}

#[test]
fn undefined_inverse_is_not_erased_by_zero_or_zeroth_power() {
    let x = Expr::symbol("cancel_undefined_x");
    let undefined = power(difference(x.clone(), x), -1);
    for source in [
        Expr::Mul(vec![Expr::from_i64(0), undefined.clone()]),
        power(undefined, 0),
    ] {
        assert!(try_cancel(&source).is_err());
    }
}

#[test]
fn every_original_child_is_bounded_before_zero_shortcuts() {
    let oversized = Expr::Integer(BigInt::from(1) << 4096u32);
    let source = Expr::Mul(vec![Expr::from_i64(0), oversized]);
    assert!(
        matches!(try_cancel(&source), Err(SimplifyError::General(message)) if message.contains("4096 bits"))
    );
    let excessive_power = Expr::Mul(vec![Expr::from_i64(0), power(Expr::symbol("x"), 257)]);
    assert!(try_cancel(&excessive_power).is_err());
    let oversized_function = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("f".repeat(65_537), vec![]),
    ]);
    assert!(try_cancel(&oversized_function).is_err());
    let too_many_nodes = Expr::Add(vec![Expr::from_i64(0); 4096]);
    assert!(try_cancel(&too_many_nodes).is_err());
    let mut deep = Expr::symbol("deep");
    for _ in 0..65 {
        deep = Expr::Add(vec![deep]);
    }
    assert!(try_cancel(&deep).is_err());
}

#[test]
fn generator_and_verifier_exhaustion_remain_typed_errors() {
    let x = Expr::symbol("cancel_budget_x");
    let source = fraction(
        difference(power(x.clone(), 2), Expr::from_i64(1)),
        difference(x, Expr::from_i64(1)),
    );
    let mut empty_generator = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        cancel_with(&source, &mut empty_generator, &mut Unbounded),
        Err(SimplifyError::BudgetExhausted(_))
    ));
    let mut empty_verifier = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        cancel_with(&source, &mut Unbounded, &mut empty_verifier),
        Err(SimplifyError::BudgetExhausted(_))
    ));
}

#[derive(Default)]
struct Checkpoints {
    count: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.count += 1;
        if self.cancel_at.is_some_and(|limit| self.count >= limit) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn final_verifier_cancellation_prevents_publication() {
    let x = Expr::symbol("cancel_final_x");
    let source = fraction(
        difference(power(x.clone(), 2), Expr::from_i64(1)),
        difference(x, Expr::from_i64(1)),
    );
    let mut measured = Checkpoints::default();
    cancel_with(&source, &mut Unbounded, &mut measured).unwrap();
    assert!(measured.count > 1);
    let mut cancelled = Checkpoints {
        count: 0,
        cancel_at: Some(measured.count),
    };
    assert_eq!(
        cancel_with(&source, &mut Unbounded, &mut cancelled),
        Err(SimplifyError::Cancelled)
    );
}
