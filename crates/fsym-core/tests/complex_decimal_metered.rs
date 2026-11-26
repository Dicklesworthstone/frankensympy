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
