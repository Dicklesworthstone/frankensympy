//! Decimal-pair acceptance is checked against independent exact identities
//! and frozen mathematical constants, with no binary64 reference conversion.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Constant, Expr, MeteredComplexDecimalError, Symbol};
use std::sync::Arc;

fn q(n: i64, d: i64) -> Expr {
    Expr::Rational(BigRational::new(n.into(), d.into()))
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}

fn complex(real: Expr, imag: Expr) -> Expr {
    Expr::Add(vec![real, Expr::Mul(vec![imag, Expr::Const(Constant::I)])])
}

fn decimal(source: &Expr, digits: u32) -> (String, String) {
    source
        .evalf_complex_decimal_metered(digits, &mut Unbounded)
        .unwrap()
}

fn exact_digits(value: i64, digits: usize) -> String {
    format!("{value}.{}e0", "0".repeat(digits - 1))
}

#[test]
fn exact_gaussian_values_retain_one_hundred_significant_digits_per_component() {
    assert_eq!(
        decimal(&complex(q(1, 3), q(-2, 3)), 100),
        (
            format!("3.{}e-1", "3".repeat(99)),
            format!("-6.{}7e-1", "6".repeat(98)),
        )
    );
    assert_eq!(
        decimal(&complex(Expr::from_i64(1), Expr::from_i64(-2)), 100),
        (exact_digits(1, 100), exact_digits(-2, 100))
    );
}

#[test]
fn exponential_phase_matches_independent_one_hundred_digit_taylor_bounds() {
    // Exact alternating rational Taylor sums of degrees 160 and 161
    // enclose cos(1) and sin(1) and give these same 100-digit half-even bins.
    let expected: (String, String) = (
        "5.403023058681397174009366074429766037323104206179222276700972553811003947744717645179518560871830893e-1".into(),
        "8.414709848078965066525023216302989996225630607983710656727517099919104043912396689486397435430526959e-1".into(),
    );
    assert_eq!(
        decimal(&function("exp", Expr::Const(Constant::I)), 100),
        expected
    );
    assert_eq!(
        decimal(
            &power(Expr::Const(Constant::E), Expr::Const(Constant::I)),
            100
        ),
        expected
    );
}

#[test]
fn named_constants_and_component_projection_keep_high_precision() {
    let pi = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068e0";
    let e = "2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427e0";
    let source = complex(Expr::Const(Constant::Pi), Expr::Const(Constant::E));
    assert_eq!(decimal(&source, 100), (pi.into(), e.into()));
    assert_eq!(
        decimal(&function("re", source.clone()), 100),
        (pi.into(), "0".into())
    );
    assert_eq!(
        decimal(&function("im", source), 100),
        (e.into(), "0".into())
    );
}

#[test]
fn principal_square_roots_and_half_integer_powers_keep_their_branch() {
    // Independently: (2+I)^2=3+4I, (2+I)^3=2+11I, and sqrt(-4)=2I.
    let source = complex(Expr::from_i64(3), Expr::from_i64(4));
    assert_eq!(
        decimal(&function("sqrt", source.clone()), 40),
        (exact_digits(2, 40), exact_digits(1, 40))
    );
    assert_eq!(
        decimal(&power(source, q(3, 2)), 40),
        (exact_digits(2, 40), format!("1.1{}e1", "0".repeat(38)))
    );
    assert_eq!(
        decimal(&function("sqrt", Expr::from_i64(-4)), 40),
        ("0".into(), exact_digits(2, 40))
    );
}

#[test]
fn principal_logarithms_and_nonreal_exponents_have_verified_decimal_components() {
    assert_eq!(
        decimal(&function("log", Expr::from_i64(-1)), 30),
        ("0".into(), "3.14159265358979323846264338328e0".into())
    );
    assert_eq!(
        decimal(
            &function("log", complex(Expr::from_i64(1), Expr::from_i64(1))),
            30
        ),
        (
            "3.46573590279972654708616060729e-1".into(),
            "7.85398163397448309615660845820e-1".into(),
        )
    );
    // Principal I**I = exp(-pi/2), a strictly positive real value.
    assert_eq!(
        decimal(
            &power(Expr::Const(Constant::I), Expr::Const(Constant::I)),
            30
        ),
        ("2.07879576350761908546955619835e-1".into(), "0".into())
    );
}

#[test]
fn differently_scaled_components_and_tiny_exponentials_are_not_flushed_to_zero() {
    let scale = BigInt::from(10).pow(150);
    let source = complex(
        Expr::Rational(BigRational::new(BigInt::from(1), scale.clone())),
        Expr::Integer(scale),
    );
    assert_eq!(
        decimal(&source, 40),
        (
            format!("1.{}e-150", "0".repeat(39)),
            format!("1.{}e150", "0".repeat(39)),
        )
    );
    assert_eq!(
        decimal(
            &complex(function("exp", Expr::from_i64(-100)), Expr::from_i64(1)),
            30
        ),
        (
            "3.72007597602083596295969580386e-44".into(),
            exact_digits(1, 30)
        )
    );
}

#[test]
fn exact_ties_are_half_even_and_carry_changes_the_decimal_exponent() {
    for (real, imag, expected_real, expected_imag, digits) in [
        (q(5, 4), q(-27, 20), "1.2e0", "-1.4e0", 2),
        (q(27, 20), q(-5, 4), "1.4e0", "-1.2e0", 2),
        (q(1999, 200), q(-1999, 200), "1.00e1", "-1.00e1", 3),
    ] {
        assert_eq!(
            decimal(&complex(real, imag), digits),
            (expected_real.into(), expected_imag.into())
        );
    }
}

#[test]
fn exact_zero_is_accepted_but_unresolved_transcendental_cancellation_refuses() {
    assert_eq!(decimal(&Expr::from_i64(0), 40), ("0".into(), "0".into()));
    assert_eq!(
        decimal(&Expr::Const(Constant::I), 40),
        ("0".into(), exact_digits(1, 40))
    );
    let held_zero = Expr::Add(vec![
        function("sin", Expr::from_i64(1)),
        Expr::Mul(vec![Expr::from_i64(-1), function("sin", Expr::from_i64(1))]),
    ]);
    // Equality recognition is not part of decimal enclosure acceptance.
    // Both finite refinement refusal and intrinsic work exhaustion are honest.
    let result = held_zero.evalf_complex_decimal_metered(30, &mut Unbounded);
    assert!(matches!(
        result,
        Err(MeteredComplexDecimalError::RoundingInconclusive
            | MeteredComplexDecimalError::ResourceLimit(_))
    ));
}

#[test]
fn held_invalid_children_and_indeterminate_zero_powers_never_disappear() {
    let pole = function("log", Expr::from_i64(0));
    for source in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        power(pole.clone(), Expr::from_i64(0)),
        power(Expr::from_i64(1), pole),
        power(Expr::from_i64(0), Expr::from_i64(0)),
        power(Expr::from_i64(0), Expr::from_i64(-1)),
        function(
            "sqrt",
            Expr::Mul(vec![
                Expr::from_i64(0),
                power(Expr::from_i64(0), Expr::from_i64(-1)),
            ]),
        ),
    ] {
        assert!(
            source
                .evalf_complex_decimal_metered(30, &mut Unbounded)
                .is_err()
        );
    }
}

#[test]
fn bounded_real_function_compositions_preserve_lambert_and_atan2_domains() {
    let w = function("LambertW", Expr::from_i64(1));
    assert_eq!(
        decimal(&complex(w, Expr::from_i64(1)), 30),
        (
            "5.67143290409783872999968662210e-1".into(),
            exact_digits(1, 30)
        )
    );
    let lower = Expr::Function("LambertW".into(), vec![q(-1, 4), Expr::from_i64(-1)]);
    assert_eq!(
        decimal(&complex(lower, Expr::from_i64(1)), 30),
        (
            "-2.15329236411034964916909915009e0".into(),
            exact_digits(1, 30)
        )
    );
    let branch_point = Expr::Mul(vec![
        Expr::from_i64(-1),
        power(Expr::Const(Constant::E), Expr::from_i64(-1)),
    ]);
    for branch in [0, -1] {
        let source = Expr::Function(
            "LambertW".into(),
            vec![branch_point.clone(), Expr::from_i64(branch)],
        );
        assert_eq!(decimal(&source, 30), (exact_digits(-1, 30), "0".into()));
    }
    let angle = Expr::Function("atan2".into(), vec![Expr::from_i64(3), Expr::from_i64(4)]);
    assert_eq!(
        decimal(&complex(angle, Expr::from_i64(1)), 30),
        (
            "6.43501108793284386802809228717e-1".into(),
            exact_digits(1, 30)
        )
    );
    for source in [
        function(
            "LambertW",
            Expr::Mul(vec![Expr::from_i64(0), Expr::Const(Constant::I)]),
        ),
        Expr::Function("LambertW".into(), vec![q(-1, 4), Expr::from_i64(1)]),
        Expr::Function("atan2".into(), vec![Expr::from_i64(0), Expr::from_i64(0)]),
    ] {
        assert!(
            source
                .evalf_complex_decimal_metered(30, &mut Unbounded)
                .is_err()
        );
    }
}

#[test]
fn precision_and_source_envelopes_refuse_before_large_work() {
    for digits in [0, Expr::MAX_COMPLEX_DECIMAL_DIGITS + 1, u32::MAX] {
        assert!(matches!(
            Expr::from_i64(1).evalf_complex_decimal_metered(digits, &mut Unbounded),
            Err(MeteredComplexDecimalError::Unsupported(_))
        ));
    }
    assert_eq!(
        decimal(&complex(Expr::from_i64(1), Expr::from_i64(-2)), 1000),
        (exact_digits(1, 1000), exact_digits(-2, 1000))
    );
    for source in [
        Expr::Integer(BigInt::from(1) << 4096u32),
        Expr::Add(vec![Expr::from_i64(0); 1024]),
        power(
            power(Expr::from_i64(2), Expr::from_i64(1000)),
            Expr::from_i64(1000),
        ),
    ] {
        assert!(matches!(
            source.evalf_complex_decimal_metered(30, &mut Unbounded),
            Err(MeteredComplexDecimalError::ResourceLimit(_))
        ));
    }
    let mut deep = Expr::from_i64(1);
    for _ in 0..34 {
        deep = function("conjugate", deep);
    }
    assert!(matches!(
        deep.evalf_complex_decimal_metered(30, &mut Unbounded),
        Err(MeteredComplexDecimalError::ResourceLimit(_))
    ));
}

#[test]
fn unknown_names_and_malformed_arity_fail_closed_without_printing_source() {
    for source in [
        Expr::Sym(Symbol::new("x")),
        function(&"unknown".repeat(100_000), Expr::from_i64(1)),
        Expr::Function("sin".into(), vec![Expr::from_i64(1), Expr::from_i64(2)]),
        Expr::Const(Constant::Infinity),
    ] {
        assert!(matches!(
            source.evalf_complex_decimal_metered(30, &mut Unbounded),
            Err(MeteredComplexDecimalError::Unsupported(_))
        ));
    }
}

#[test]
fn real_only_fallbacks_do_not_bypass_intermediate_growth_guards() {
    let enormous = power(
        power(Expr::from_i64(2), Expr::from_i64(1000)),
        Expr::from_i64(1000),
    );
    for source in [
        function("atan", enormous.clone()),
        function("LambertW", enormous.clone()),
        Expr::Function("atan2".into(), vec![enormous, Expr::from_i64(1)]),
    ] {
        assert!(matches!(
            source.evalf_complex_decimal_metered(30, &mut Unbounded),
            Err(MeteredComplexDecimalError::ResourceLimit(_))
        ));
    }
}

struct ObservedMeter {
    budget: Budget,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl ObservedMeter {
    fn new(cancel_at: Option<usize>) -> Self {
        Self {
            budget: Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0)),
            checkpoints: 0,
            cancel_at,
        }
    }
}

impl BudgetMeter for ObservedMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at.is_some_and(|stop| self.checkpoints >= stop) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_admission_midwork_and_final_publication_returns_no_pair() {
    let source = complex(q(5, 7), q(-11, 13));
    let mut baseline = ObservedMeter::new(None);
    source
        .evalf_complex_decimal_metered(40, &mut baseline)
        .unwrap();
    for stop in [1, baseline.checkpoints / 2, baseline.checkpoints] {
        let mut meter = ObservedMeter::new(Some(stop));
        assert_eq!(
            source.evalf_complex_decimal_metered(40, &mut meter),
            Err(MeteredComplexDecimalError::Meter(MeterError::Cancelled))
        );
    }
}

#[test]
fn budget_exhaustion_stays_typed_and_consumption_survives_failure_and_reuse() {
    let source = function("exp", Expr::Const(Constant::I));
    let mut zero = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        source.evalf_complex_decimal_metered(40, &mut zero),
        Err(MeteredComplexDecimalError::Meter(MeterError::Budget(_)))
    ));
    let mut budget = Budget::new(BudgetLimits::uniform(64, 0));
    let before = budget.remaining(Dimension::ComputeSteps);
    assert!(matches!(
        source.evalf_complex_decimal_metered(40, &mut budget),
        Err(MeteredComplexDecimalError::Meter(MeterError::Budget(_)))
    ));
    let after = budget.remaining(Dimension::ComputeSteps);
    assert!(after < before);
    let _ = source.evalf_complex_decimal_metered(40, &mut budget);
    assert!(budget.remaining(Dimension::ComputeSteps) <= after);
}

#[test]
fn default_certification_sized_budget_admits_one_hundred_digit_complex_values() {
    let mut budget = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let source = function("exp", Expr::Const(Constant::I));
    let value = source
        .evalf_complex_decimal_metered(100, &mut budget)
        .unwrap();
    assert_eq!(value, decimal(&source, 100));
    assert!(budget.remaining(Dimension::ComputeSteps) < 512 * 1024 * 1024);
    assert!(budget.remaining(Dimension::MemoryBytes) < 512 * 1024 * 1024);
    assert!(budget.remaining(Dimension::AllocationCount) < 512 * 1024 * 1024);
}

#[test]
fn inverse_cut_decimals_keep_exact_axis_branches_in_both_entry_points() {
    // Independent branch identities use pi/2, acosh(2)=ln(2+sqrt(3))
    // and ln(3)/2. The digits agree with the isolated SymPy 1.14 oracle.
    let half_pi = "1.57079632679489661923132169164e0";
    let a = "1.31695789692481670862504634731e0";
    let l = "5.49306144334054845697622618461e-1";
    let pi = "3.14159265358979323846264338328e0";
    let negative_half_pi = format!("-{half_pi}");
    let negative_a = format!("-{a}");
    let negative_l = format!("-{l}");
    for (name, x, y, real, imag) in [
        ("asin", 2, 0, half_pi, negative_a.as_str()),
        ("asin", -2, 0, negative_half_pi.as_str(), a),
        ("acos", 2, 0, "0", a),
        ("acos", -2, 0, pi, negative_a.as_str()),
        ("atan", 0, 2, half_pi, l),
        ("atan", 0, -2, negative_half_pi.as_str(), negative_l.as_str()),
        ("asinh", 0, 2, a, half_pi),
        ("asinh", 0, -2, negative_a.as_str(), negative_half_pi.as_str()),
        ("acosh", 2, 0, a, "0"),
        ("acosh", -2, 0, a, pi),
        ("atanh", 2, 0, l, negative_half_pi.as_str()),
        ("atanh", -2, 0, negative_l.as_str(), half_pi),
    ] {
        let source = function(name, complex(q(x, 1), q(y, 1)));
        let expected = (real.to_string(), imag.to_string());
        assert_eq!(decimal(&source, 30), expected, "{source}");
        assert_eq!(
            source.evalf_complex_decimal(30).unwrap(),
            expected,
            "{source}"
        );
    }
}

#[test]
fn inverse_zero_components_are_exact_without_a_numerical_chop() {
    for (name, argument) in [
        ("asin", 0),
        ("acos", 1),
        ("atan", 0),
        ("asinh", 0),
        ("acosh", 1),
        ("atanh", 0),
    ] {
        let source = function(name, Expr::from_i64(argument));
        assert_eq!(decimal(&source, 80), ("0".into(), "0".into()));
        assert_eq!(
            source.evalf_complex_decimal(80).unwrap(),
            ("0".into(), "0".into())
        );
    }
    let source = function("asin", Expr::Const(Constant::I));
    let expected = ("0".into(), "8.81373587019543025232609324980e-1".into());
    assert_eq!(decimal(&source, 30), expected);
    assert_eq!(source.evalf_complex_decimal(30).unwrap(), expected);
}

#[test]
fn inverse_poles_and_original_undefined_children_remain_terminal() {
    for (name, argument) in [
        ("atan", complex(q(0, 1), q(1, 1))),
        ("atan", complex(q(0, 1), q(-1, 1))),
        ("atanh", q(1, 1)),
        ("atanh", q(-1, 1)),
    ] {
        let pole = function(name, argument);
        for source in [pole.clone(), Expr::Mul(vec![q(0, 1), pole])] {
            assert!(matches!(
                source.evalf_complex_decimal_metered(30, &mut Unbounded),
                Err(MeteredComplexDecimalError::Ball(_))
            ));
            assert!(source.evalf_complex_decimal(30).is_err());
        }
    }
    for name in ["asin", "acos", "atan", "asinh", "acosh", "atanh"] {
        let argument = Expr::Add(vec![
            q(1, 1),
            Expr::Const(Constant::I),
            Expr::Mul(vec![q(0, 1), power(q(0, 1), q(-1, 1))]),
        ]);
        let source = function(name, argument);
        assert!(matches!(
            source.evalf_complex_decimal_metered(30, &mut Unbounded),
            Err(MeteredComplexDecimalError::Ball(_))
        ));
        assert!(source.evalf_complex_decimal(30).is_err());
    }
}

#[test]
fn complex_inverse_admission_does_not_relax_real_only_source_domains() {
    for (name, argument) in [("asin", 2), ("acos", 2), ("acosh", -2), ("atanh", 2)] {
        let source = function(name, Expr::from_i64(argument));
        assert!(source.eval_real_ball(30).is_err());
        assert!(source.eval_complex_ball(30).is_ok());
    }
    // LambertW remains a real-only kernel. A complex child cannot disappear
    // under zero multiplication before that real source domain is checked.
    let source = function(
        "LambertW",
        Expr::Mul(vec![q(0, 1), function("asin", q(2, 1))]),
    );
    assert!(
        source
            .evalf_complex_decimal_metered(30, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn inverse_evaluation_uses_one_cumulative_cancellable_ledger() {
    for name in ["asin", "acos", "atan", "asinh", "acosh", "atanh"] {
        let source = function(name, complex(q(1, 3), q(1, 5)));
        let mut baseline = ObservedMeter::new(None);
        let expected = source
            .evalf_complex_decimal_metered(30, &mut baseline)
            .unwrap();
        let checkpoints = baseline.checkpoints;
        let before = baseline.budget.remaining(Dimension::ComputeSteps);
        assert_eq!(
            source
                .evalf_complex_decimal_metered(30, &mut baseline)
                .unwrap(),
            expected
        );
        assert!(baseline.budget.remaining(Dimension::ComputeSteps) < before);
        for stop in [1, checkpoints / 2, checkpoints] {
            let mut meter = ObservedMeter::new(Some(stop));
            assert_eq!(
                source.evalf_complex_decimal_metered(30, &mut meter),
                Err(MeteredComplexDecimalError::Meter(MeterError::Cancelled))
            );
        }
        let mut insufficient = Budget::new(BudgetLimits::uniform(64, 0));
        assert!(matches!(
            source.evalf_complex_decimal_metered(30, &mut insufficient),
            Err(MeteredComplexDecimalError::Meter(MeterError::Budget(_)))
        ));
    }
}

#[test]
fn inverse_function_arity_remains_closed_in_both_decimal_interfaces() {
    for name in ["asin", "acos", "atan", "asinh", "acosh", "atanh"] {
        for arity in [0, 2] {
            let source = Expr::Function(name.into(), vec![q(1, 1); arity]);
            assert!(matches!(
                source.evalf_complex_decimal_metered(30, &mut Unbounded),
                Err(MeteredComplexDecimalError::Unsupported(_))
            ));
            assert!(matches!(
                source.evalf_complex_decimal(30),
                Err(fsym_core::ComplexDecimalError::UnsupportedSource(_))
            ));
        }
    }
}
