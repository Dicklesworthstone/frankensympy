use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, Expr, RealLambertError, RealLambertWitness, Symbol, SymbolIdentity,
    verify_real_lambert_equation,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn scalar(q: &BigRational) -> Expr {
    Expr::Rational(q.clone())
}
fn exp(value: Expr) -> Expr {
    Expr::Function("exp".into(), vec![value])
}
fn power(base: Expr, n: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(n)))
}
fn form(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) -> RealLambertWitness {
    RealLambertWitness {
        linear_slope: q(a, 1),
        linear_offset: q(b, 1),
        exponential_slope: q(c, 1),
        exponential_offset: q(d, 1),
        argument_slope: q(e, 1),
        argument_offset: q(f, 1),
    }
}
fn source(w: &RealLambertWitness, x: &Symbol) -> Expr {
    let sx = Expr::Sym(x.clone());
    Expr::Add(vec![
        Expr::Mul(vec![scalar(&w.linear_slope), sx.clone()]),
        scalar(&w.linear_offset),
        Expr::Mul(vec![
            Expr::Add(vec![
                Expr::Mul(vec![scalar(&w.exponential_slope), sx.clone()]),
                scalar(&w.exponential_offset),
            ]),
            exp(Expr::Add(vec![
                Expr::Mul(vec![scalar(&w.argument_slope), sx]),
                scalar(&w.argument_offset),
            ])),
        ]),
    ])
}
fn branch(root: &Expr) -> Option<i64> {
    match root {
        Expr::Function(name, args) if name == "LambertW" => match args.as_slice() {
            [_] => Some(0),
            [_, Expr::Integer(k)] => k.to_i64(),
            _ => None,
        },
        Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => args.iter().find_map(branch),
        Expr::Pow(a, b) => branch(a).or_else(|| branch(b)),
        _ => None,
    }
}

#[test]
fn positive_argument_has_one_source_bound_principal_root() {
    let x = Symbol::new("positive_lambert");
    let w = form(0, -1, 1, 0, 1, 0);
    let expression = source(&w, &x);
    let result = verify_real_lambert_equation(&expression, &x, &w, &mut Unbounded).unwrap();
    assert_eq!(result.expression(), &expression);
    assert_eq!(result.variable(), &x);
    assert_eq!(result.normal_form(), &w);
    assert_eq!(result.argument(), &Expr::from_i64(1));
    assert_eq!(
        result.roots(),
        &[Expr::Function("LambertW".into(), vec![Expr::from_i64(1)])]
    );
}

#[test]
fn two_negative_argument_branches_are_complete_and_distinct() {
    let x = Symbol::new("two_branches");
    let mut w = form(0, 1, 1, 0, 1, 0);
    w.linear_offset = q(1, 4);
    let result = verify_real_lambert_equation(&source(&w, &x), &x, &w, &mut Unbounded).unwrap();
    assert_eq!(result.argument(), &Expr::rational(-1, 4).unwrap());
    assert_eq!(result.roots().len(), 2);
    assert_eq!(
        result.roots().iter().map(branch).collect::<Vec<_>>(),
        vec![Some(-1), Some(0)]
    );
    assert_ne!(result.roots()[0], result.roots()[1]);
}

#[test]
fn exp_equals_affine_reverses_branches_after_negative_affine_map() {
    // exp(x)=3*x has two positive roots; -W_0(-1/3) is the smaller.
    let x = Symbol::new("exp_affine");
    let w = form(-3, 0, 0, 1, 1, 0);
    let result = verify_real_lambert_equation(&source(&w, &x), &x, &w, &mut Unbounded).unwrap();
    assert_eq!(result.argument(), &Expr::rational(-1, 3).unwrap());
    assert_eq!(
        result.roots().iter().map(branch).collect::<Vec<_>>(),
        vec![Some(0), Some(-1)]
    );
}

#[test]
fn branch_point_and_zero_argument_are_not_duplicated() {
    let x = Symbol::new("boundary_lambert");
    for (w, root) in [
        (form(0, 1, 1, 0, 1, 1), -1), // x*exp(x+1)+1
        (form(1, 1, 0, -1, 1, 0), 0), // x+1-exp(x)
        (form(2, 4, 0, -2, 1, 1), -1),
        (form(0, 0, 2, 6, 3, 4), -3), // exp never vanishes
        (form(2, 6, 0, 0, 3, 4), -3), // vanished exponential coefficient
    ] {
        let result = verify_real_lambert_equation(&source(&w, &x), &x, &w, &mut Unbounded).unwrap();
        assert_eq!(result.roots(), &[Expr::from_i64(root)]);
    }
}

#[test]
fn lower_than_minimum_is_proved_empty_and_nearby_sides_remain_distinct() {
    let x = Symbol::new("minimum_lambert");
    let w = form(0, 1, 1, 0, 1, 0);
    assert!(
        verify_real_lambert_equation(&source(&w, &x), &x, &w, &mut Unbounded)
            .unwrap()
            .roots()
            .is_empty()
    );
    // Exact rational exponent perturbations must not round to the tangent.
    let epsilon = q(1, 10_i64.pow(12));
    for (offset, count) in [(q(1, 1) + &epsilon, 2usize), (q(1, 1) - &epsilon, 0)] {
        let mut near = form(0, 1, 1, 0, 1, 1);
        near.argument_offset = offset;
        assert_eq!(
            verify_real_lambert_equation(&source(&near, &x), &x, &near, &mut Unbounded)
                .unwrap()
                .roots()
                .len(),
            count
        );
    }
}

#[test]
fn certified_comparisons_cover_both_exponent_signs_and_both_root_counts() {
    let x = Symbol::new("comparison_lambert");
    for (coefficient, offset, count) in [
        (q(1, 2), 0, 0), // exp(1)>2, so -1/2 lies below -1/e.
        (q(1, 4), 0, 2), // exp(1)<4.
        (q(4, 1), 2, 0), // 4*exp(-1)>1.
        (q(4, 1), 3, 2), // 4*exp(-2)<1.
    ] {
        let mut w = form(0, 1, 1, 0, 1, offset);
        w.linear_offset = coefficient;
        assert_eq!(
            verify_real_lambert_equation(&source(&w, &x), &x, &w, &mut Unbounded)
                .unwrap()
                .roots()
                .len(),
            count
        );
    }
}

#[test]
fn shifted_scaled_and_negative_slope_forms_keep_exact_arguments() {
    let x = Symbol::new("shifted_lambert");
    let w = form(0, -5, 2, 3, 3, -2);
    let result = verify_real_lambert_equation(&source(&w, &x), &x, &w, &mut Unbounded).unwrap();
    let expected = Expr::Mul(vec![
        Expr::rational(15, 2).unwrap(),
        exp(Expr::rational(13, 2).unwrap()),
    ]);
    assert_eq!(result.argument(), &expected);
    assert_eq!(result.roots().len(), 1);
    assert_eq!(branch(&result.roots()[0]), Some(0));
    // (-x)*exp(-x)=1/4 has one negative real root.
    let mut negative = form(0, -1, -1, 0, -1, 0);
    negative.linear_offset = q(-1, 4);
    let result =
        verify_real_lambert_equation(&source(&negative, &x), &x, &negative, &mut Unbounded)
            .unwrap();
    assert_eq!(result.argument(), &Expr::rational(1, 4).unwrap());
    assert_eq!(result.roots().len(), 1);
}

#[test]
fn exact_principal_special_value_is_lifted_as_a_rational_root() {
    let x = Symbol::new("special_lambert");
    let w = form(0, -1, 1, 0, 1, -1); // x*exp(x-1)=1 has x=1.
    let result = verify_real_lambert_equation(&source(&w, &x), &x, &w, &mut Unbounded).unwrap();
    assert_eq!(result.roots(), &[Expr::from_i64(1)]);
}

#[test]
fn every_changed_coefficient_and_original_source_is_rejected() {
    let x = Symbol::new("mutant_lambert");
    let w = form(0, -5, 2, 3, 3, -2);
    let original = source(&w, &x);
    for index in 0..6 {
        let mut bad = w.clone();
        let coefficient = match index {
            0 => &mut bad.linear_slope,
            1 => &mut bad.linear_offset,
            2 => &mut bad.exponential_slope,
            3 => &mut bad.exponential_offset,
            4 => &mut bad.argument_slope,
            _ => &mut bad.argument_offset,
        };
        *coefficient += q(1, 7);
        assert!(matches!(
            verify_real_lambert_equation(&original, &x, &bad, &mut Unbounded),
            Err(RealLambertError::InvalidWitness(_))
        ));
    }
    let changed = Expr::Add(vec![original, Expr::rational(1, 1_000_000).unwrap()]);
    assert!(matches!(
        verify_real_lambert_equation(&changed, &x, &w, &mut Unbounded),
        Err(RealLambertError::InvalidWitness(_))
    ));
}

#[test]
fn nonlinear_and_mismatched_exponential_sources_cannot_fake_the_normal_form() {
    let x = Symbol::new("false_normal_form");
    let w = form(0, -1, 1, 0, 1, 0);
    let original = source(&w, &x);
    for extra in [
        power(Expr::Sym(x.clone()), 2),
        exp(Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(x.clone())])),
    ] {
        let bad = Expr::Add(vec![original.clone(), extra]);
        assert!(matches!(
            verify_real_lambert_equation(&bad, &x, &w, &mut Unbounded),
            Err(RealLambertError::InvalidWitness(_))
        ));
    }
    for bad in [
        form(1, 0, 1, 0, 1, 0),
        form(0, 1, 0, 1, 1, 0),
        form(0, -1, 1, 0, 0, 1),
    ] {
        assert!(matches!(
            verify_real_lambert_equation(&source(&bad, &x), &x, &bad, &mut Unbounded),
            Err(RealLambertError::InvalidWitness(_))
        ));
    }
}

#[test]
fn original_poles_and_undefined_values_survive_zero_products_and_powers() {
    let x = Symbol::new("source_domains");
    let w = form(0, -1, 1, 0, 1, 0);
    let pole = power(Expr::Sym(x.clone()), -1);
    for extra in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        power(pole, 0),
    ] {
        let bad = Expr::Add(vec![source(&w, &x), extra]);
        assert!(matches!(
            verify_real_lambert_equation(&bad, &x, &w, &mut Unbounded),
            Err(RealLambertError::Unsupported(_))
        ));
    }
    let bad = Expr::Add(vec![
        source(&w, &x),
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
    ]);
    assert!(matches!(
        verify_real_lambert_equation(&bad, &x, &w, &mut Unbounded),
        Err(RealLambertError::InvalidSource(_))
    ));
}

#[test]
fn coordinate_identity_and_unsupported_original_nodes_are_preserved() {
    let x = Symbol::new("same_name");
    let foreign = Symbol::with_identity(
        "same_name",
        SymbolIdentity::from_assumptions_digest([7; 32]),
    );
    let w = form(0, -1, 1, 0, 1, 0);
    assert!(matches!(
        verify_real_lambert_equation(&source(&w, &foreign), &x, &w, &mut Unbounded),
        Err(RealLambertError::Unsupported(_))
    ));
    for extra in [
        Expr::Function("log".into(), vec![Expr::from_i64(0)]),
        Expr::Function("exp".into(), vec![]),
    ] {
        let bad = Expr::Add(vec![
            source(&w, &x),
            Expr::Mul(vec![Expr::from_i64(0), extra]),
        ]);
        assert!(matches!(
            verify_real_lambert_equation(&bad, &x, &w, &mut Unbounded),
            Err(RealLambertError::Unsupported(_))
        ));
    }
}

#[test]
fn oversized_inputs_fail_before_unbounded_recursion_or_integer_arithmetic() {
    let x = Symbol::new("bounded_lambert");
    let w = form(0, -1, 1, 0, 1, 0);
    let mut deep = source(&w, &x);
    for _ in 0..40 {
        deep = power(deep, 1);
    }
    for bad in [
        deep,
        Expr::Integer(BigInt::from(1) << 4096u32),
        power(source(&w, &x), 17),
    ] {
        assert!(matches!(
            verify_real_lambert_equation(&bad, &x, &w, &mut Unbounded),
            Err(RealLambertError::ResourceLimit(_))
        ));
    }
}

#[derive(Default)]
struct Checkpoints {
    calls: usize,
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
        self.calls += 1;
        if self.cancel_at.is_some_and(|n| self.calls >= n) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn caller_budgets_are_cumulative_and_cancellation_never_publishes_partial_roots() {
    let x = Symbol::new("metered_lambert");
    let mut w = form(0, 1, 1, 0, 1, 0);
    w.linear_offset = q(1, 4);
    let original = source(&w, &x);
    let allowance = 512 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(allowance, 0));
    verify_real_lambert_equation(&original, &x, &w, &mut budget).unwrap();
    assert!(budget.remaining(Dimension::ComputeSteps) < allowance);
    assert!(budget.remaining(Dimension::MemoryBytes) < allowance);
    let mut insufficient = Budget::new(BudgetLimits::uniform(1, 0));
    let before = insufficient.snapshot();
    assert!(matches!(
        verify_real_lambert_equation(&original, &x, &w, &mut insufficient),
        Err(RealLambertError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(insufficient.snapshot(), before);
    let mut baseline = Checkpoints::default();
    verify_real_lambert_equation(&original, &x, &w, &mut baseline).unwrap();
    for n in [1, baseline.calls / 2, baseline.calls] {
        let mut meter = Checkpoints {
            calls: 0,
            cancel_at: Some(n),
        };
        assert!(matches!(
            verify_real_lambert_equation(&original, &x, &w, &mut meter),
            Err(RealLambertError::Meter(MeterError::Cancelled))
        ));
    }
}

#[test]
fn explicit_lambert_branches_have_profile_correct_exact_special_values() {
    use fsym_core::{Constant, elementary::eval_function};
    let zero = Expr::from_i64(0);
    let minus_one = Expr::from_i64(-1);
    assert_eq!(
        eval_function("LambertW", std::slice::from_ref(&zero)),
        Some(zero.clone())
    );
    assert_eq!(
        eval_function("LambertW", &[zero.clone(), minus_one.clone()]),
        Some(Expr::Const(Constant::NegativeInfinity))
    );
    let x = Expr::symbol("branch_argument");
    assert_eq!(
        eval_function("LambertW", &[x.clone(), zero]),
        Some(Expr::Function("LambertW".into(), vec![x]))
    );
    assert_eq!(
        eval_function("LambertW", &[Expr::Const(Constant::E)]),
        Some(Expr::from_i64(1))
    );
    assert_eq!(
        eval_function("LambertW", &[Expr::Const(Constant::E), minus_one.clone()]),
        None
    );
    for boundary in [
        Expr::Mul(vec![minus_one.clone(), exp(minus_one.clone())]),
        Expr::Mul(vec![minus_one.clone(), power(Expr::Const(Constant::E), -1)]),
    ] {
        assert_eq!(
            eval_function("LambertW", std::slice::from_ref(&boundary)),
            Some(minus_one.clone())
        );
        assert_eq!(
            eval_function("LambertW", &[boundary, minus_one.clone()]),
            Some(minus_one.clone())
        );
    }
}
