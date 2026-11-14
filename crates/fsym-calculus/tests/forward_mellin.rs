//! Exact hand-derived Mellin identities, original domains and adversarial replay.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::forward_mellin::{
    ForwardMellinError, MellinKernel, generate_mellin_candidate, mellin_transform,
    metered_mellin_transform, verify_mellin_candidate,
};
use fsym_calculus::rational_mellin::{MellinStrip, MellinStripBound};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use num_traits::{One, Zero};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn finite(n: i64, d: i64) -> MellinStripBound {
    MellinStripBound::Finite(q(n, d))
}
fn expression(source: &str) -> Expr {
    parse(source).unwrap()
}
fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}

/// Independent exact interpreter at rational points where every gamma
/// argument is a positive integer. It never invokes production transforms,
/// differentiation, candidate renderers, or the symbolic simplifier.
fn exact_value(source: &Expr, s: &Symbol, point: &BigRational) -> BigRational {
    match source {
        Expr::Integer(n) => BigRational::from_integer(n.clone()),
        Expr::Rational(q) => q.clone(),
        Expr::Sym(symbol) if symbol == s => point.clone(),
        Expr::Add(children) => children
            .iter()
            .map(|child| exact_value(child, s, point))
            .fold(BigRational::zero(), |a, b| a + b),
        Expr::Mul(children) => children
            .iter()
            .map(|child| exact_value(child, s, point))
            .fold(BigRational::one(), |a, b| a * b),
        Expr::Pow(base, exponent) => {
            let exponent = exact_value(exponent, s, point);
            assert!(exponent.is_integer());
            exact_value(base, s, point)
                .pow(exponent.numer().to_i64().unwrap() as i32)
                .unwrap()
        }
        Expr::Function(name, args) if name == "gamma" && args.len() == 1 => {
            let argument = exact_value(&args[0], s, point);
            assert!(argument.is_integer());
            let n = argument.numer().to_i64().unwrap();
            assert!((1..=16).contains(&n));
            (1..n)
                .map(BigRational::from)
                .fold(BigRational::one(), |a, b| a * b)
        }
        _ => panic!("unexpected exact fixture node {source:?}"),
    }
}

#[test]
fn gamma_kernels_have_exact_values_and_directional_maximal_strips() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let cases = [
        (
            "exp(-x)",
            q(3, 1),
            q(2, 1),
            finite(0, 1),
            MellinStripBound::PositiveInfinity,
        ),
        (
            "exp(-2*x^2)",
            q(4, 1),
            q(1, 8),
            finite(0, 1),
            MellinStripBound::PositiveInfinity,
        ),
        (
            "-6*x^2*exp(-3*x)",
            q(0, 1),
            q(-2, 3),
            finite(-2, 1),
            MellinStripBound::PositiveInfinity,
        ),
        (
            "x^(1/3)*exp(-3*x^(-2))",
            q(-7, 3),
            q(1, 6),
            MellinStripBound::NegativeInfinity,
            finite(-1, 3),
        ),
        (
            "5*x^(-2)*exp(-2*x^(-1))",
            q(1, 1),
            q(5, 2),
            MellinStripBound::NegativeInfinity,
            finite(2, 1),
        ),
    ];
    for (source, point, expected, lower, upper) in cases {
        let result = mellin_transform(&expression(source), &x, &s).unwrap();
        assert_eq!(result.strip(), &MellinStrip { lower, upper }, "{source}");
        assert_eq!(
            exact_value(result.transform(), &s, &point),
            expected,
            "{source}"
        );
    }
}

#[test]
fn beta_kernels_include_reciprocal_powers_and_nonunit_scales() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let cases = [
        ("(1+x)^(-2)", q(1, 1), q(1, 1), finite(0, 1), finite(2, 1)),
        (
            "x^2*(1+2*x^3)^(-2)",
            q(1, 1),
            q(1, 6),
            finite(-2, 1),
            finite(4, 1),
        ),
        (
            "12*x*(2+3*x^2)^(-3)",
            q(1, 1),
            q(1, 4),
            finite(-1, 1),
            finite(5, 1),
        ),
        (
            "(1+2*x^(-2))^(-2)",
            q(-2, 1),
            q(1, 4),
            finite(-4, 1),
            finite(0, 1),
        ),
        (
            "-3*x^(1/2)*(2+4*x^(-1))^(-2)",
            q(-3, 2),
            q(-3, 8),
            finite(-5, 2),
            finite(-1, 2),
        ),
    ];
    for (source, point, expected, lower, upper) in cases {
        let result = mellin_transform(&expression(source), &x, &s).unwrap();
        assert_eq!(result.strip(), &MellinStrip { lower, upper }, "{source}");
        assert_eq!(
            exact_value(result.transform(), &s, &point),
            expected,
            "{source}"
        );
    }
}

#[test]
fn fractional_beta_order_is_checked_as_a_real_positive_axis_power() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let source = power(expression("2+3*x^2"), Expr::Rational(q(-3, 2)));
    let candidate = generate_mellin_candidate(&source, &x, &s, &mut Unbounded).unwrap();
    assert_eq!(
        candidate.strip,
        MellinStrip {
            lower: finite(0, 1),
            upper: finite(3, 1)
        }
    );
    assert!(
        matches!(&candidate.kernel, MellinKernel::Beta { order, constant, scale, .. } if order == &q(3,2) && constant == &q(2,1) && scale == &q(3,1))
    );
    verify_mellin_candidate(&source, &x, &s, &candidate, &mut Unbounded).unwrap();
}

#[test]
fn zero_has_the_whole_strip_and_no_other_pure_power_is_claimed() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let zero = mellin_transform(&Expr::from_i64(0), &x, &s).unwrap();
    assert!(zero.transform().is_zero());
    assert_eq!(
        zero.strip(),
        &MellinStrip {
            lower: MellinStripBound::NegativeInfinity,
            upper: MellinStripBound::PositiveInfinity
        }
    );
    for source in [
        "1",
        "x",
        "x^(-2)",
        "exp(x)",
        "(1-x)^(-1)",
        "sin(x)",
        "exp(-x)+exp(-2*x)",
        "exp(-x)*(1+x)^(-1)",
    ] {
        assert!(
            matches!(
                mellin_transform(&expression(source), &x, &s),
                Err(ForwardMellinError::Unsupported(_))
            ),
            "{source}"
        );
    }
}

#[test]
fn verifier_binds_derivative_and_amplitude_to_the_original_source() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let source = expression("2*x^2*exp(-3*x^2)");
    let candidate = generate_mellin_candidate(&source, &x, &s, &mut Unbounded).unwrap();
    // Different factorization, independently checked by differentiation.
    let equivalent = Expr::Mul(vec![
        Expr::from_i64(4),
        Expr::Rational(q(1, 2)),
        Expr::Sym(x.clone()),
        Expr::Sym(x.clone()),
        expression("exp(-3*x^2)"),
    ]);
    verify_mellin_candidate(&equivalent, &x, &s, &candidate, &mut Unbounded).unwrap();
    for changed in [
        "2*x^2*exp(-4*x^2)",
        "2*x^3*exp(-3*x^2)",
        "3*x^2*exp(-3*x^2)",
    ] {
        assert!(
            matches!(
                verify_mellin_candidate(&expression(changed), &x, &s, &candidate, &mut Unbounded),
                Err(ForwardMellinError::Verification(_))
            ),
            "{changed}"
        );
    }
}

#[test]
fn beta_verifier_rejects_source_scale_and_order_mutations() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let source = expression("3*x*(2+5*x^2)^(-3)");
    let candidate = generate_mellin_candidate(&source, &x, &s, &mut Unbounded).unwrap();
    for changed in [
        "3*x*(2+6*x^2)^(-3)",
        "3*x*(3+5*x^2)^(-3)",
        "3*x*(2+5*x^2)^(-2)",
        "4*x*(2+5*x^2)^(-3)",
    ] {
        assert!(
            verify_mellin_candidate(&expression(changed), &x, &s, &candidate, &mut Unbounded)
                .is_err(),
            "{changed}"
        );
    }
}

#[test]
fn candidate_parameter_formula_and_strip_mutations_never_publish() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    for text in ["exp(-2*x^2)", "3*x*(2+5*x^2)^(-3)"] {
        let source = expression(text);
        let original = generate_mellin_candidate(&source, &x, &s, &mut Unbounded).unwrap();
        let mut mutations = vec![];
        let mut changed = original.clone();
        changed.transform = Expr::Mul(vec![Expr::from_i64(2), changed.transform]);
        mutations.push(changed);
        let mut changed = original.clone();
        changed.transform = expression("gamma(s+1)");
        mutations.push(changed);
        let mut changed = original.clone();
        changed.strip.lower = finite(-100, 1);
        mutations.push(changed);
        let mut changed = original.clone();
        changed.strip.upper = finite(100, 1);
        mutations.push(changed);
        let mut changed = original.clone();
        match &mut changed.kernel {
            MellinKernel::Gamma { coefficient, .. } | MellinKernel::Beta { coefficient, .. } => {
                *coefficient = q(999, 1)
            }
            MellinKernel::Zero => unreachable!(),
        }
        mutations.push(changed);
        for changed in mutations {
            assert!(
                matches!(
                    verify_mellin_candidate(&source, &x, &s, &changed, &mut Unbounded),
                    Err(ForwardMellinError::Verification(_))
                ),
                "{changed:?}"
            );
        }
    }
}

#[test]
fn original_undefined_powers_are_checked_before_zero_or_cancellation() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    for exponent in [0, -1] {
        let invalid = power(Expr::from_i64(0), Expr::from_i64(exponent));
        let source = Expr::Mul(vec![Expr::from_i64(0), invalid, expression("exp(-x)")]);
        assert!(matches!(
            mellin_transform(&source, &x, &s),
            Err(ForwardMellinError::Refused(_))
        ));
    }
    let pole = expression("x-1");
    let hidden = Expr::Mul(vec![
        pole.clone(),
        power(pole, Expr::from_i64(-1)),
        expression("exp(-x)"),
    ]);
    assert!(matches!(
        mellin_transform(&hidden, &x, &s),
        Err(ForwardMellinError::Unsupported(_))
    ));
    let candidate =
        generate_mellin_candidate(&expression("exp(-x)"), &x, &s, &mut Unbounded).unwrap();
    assert!(verify_mellin_candidate(&hidden, &x, &s, &candidate, &mut Unbounded).is_err());
}

#[test]
fn candidate_formula_cannot_hide_undefined_zero_powers_in_affine_arguments() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let source = expression("exp(-x)");
    let original = generate_mellin_candidate(&source, &x, &s, &mut Unbounded).unwrap();
    for exponent in [0, -1] {
        let mut changed = original.clone();
        let undefined = power(Expr::from_i64(0), Expr::from_i64(exponent));
        // A rational normalizer that silently turns 0^0 into one would
        // disguise this as the legitimate gamma(s) answer.
        changed.transform = Expr::Function(
            "gamma".into(),
            vec![Expr::Add(vec![
                Expr::Sym(s.clone()),
                undefined,
                Expr::from_i64(-1),
            ])],
        );
        assert!(matches!(
            verify_mellin_candidate(&source, &x, &s, &changed, &mut Unbounded),
            Err(ForwardMellinError::Verification(_))
        ));
    }
    let zero_at_one = expression("s-1");
    for undefined in [
        power(zero_at_one.clone(), Expr::from_i64(0)),
        power(power(zero_at_one, Expr::from_i64(-1)), Expr::from_i64(0)),
    ] {
        let mut changed = original.clone();
        changed.transform = Expr::Function(
            "gamma".into(),
            vec![Expr::Add(vec![
                Expr::Sym(s.clone()),
                undefined,
                Expr::from_i64(-1),
            ])],
        );
        assert!(matches!(
            verify_mellin_candidate(&source, &x, &s, &changed, &mut Unbounded),
            Err(ForwardMellinError::Verification(_))
        ));
    }
    // A source power on the strictly positive integration axis has a
    // different domain from a candidate power over the whole complex strip.
    let harmless = power(
        power(Expr::Sym(x.clone()), Expr::from_i64(2)),
        Expr::from_i64(0),
    );
    let held = Expr::Function(
        "exp".into(),
        vec![Expr::Mul(vec![
            Expr::from_i64(-1),
            Expr::Sym(x.clone()),
            harmless,
        ])],
    );
    let verified = mellin_transform(&held, &x, &s).unwrap();
    assert_eq!(exact_value(verified.transform(), &s, &q(3, 1)), q(2, 1));
}

#[test]
fn algebraic_inner_rewrites_outside_the_checked_lane_are_explicitly_unsupported() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let root = power(Expr::Sym(x.clone()), Expr::Rational(q(1, 2)));
    let argument = Expr::Mul(vec![Expr::from_i64(-1), root.clone(), root]);
    let source = Expr::Function("exp".into(), vec![argument]);
    assert!(matches!(
        mellin_transform(&source, &x, &s),
        Err(ForwardMellinError::Unsupported(_))
    ));
}

#[test]
fn typed_variable_identity_is_preserved_and_aliasing_is_refused() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let source = expression("exp(-x)");
    assert!(matches!(
        mellin_transform(&source, &x, &x),
        Err(ForwardMellinError::Refused(_))
    ));
    let mut other_x = x.clone();
    other_x.identity = Some(SymbolIdentity::from_assumptions_digest([42; 32]));
    assert!(matches!(
        mellin_transform(&source, &other_x, &s),
        Err(ForwardMellinError::Unsupported(_))
    ));
    let mut other_s = s.clone();
    other_s.identity = Some(SymbolIdentity::from_assumptions_digest([17; 32]));
    let result = mellin_transform(&source, &x, &other_s).unwrap();
    assert_eq!(exact_value(result.transform(), &other_s, &q(3, 1)), q(2, 1));
}

#[test]
fn input_envelopes_and_untrusted_strip_growth_are_bounded_before_cloning() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let huge = Expr::Integer(BigInt::from(1) << 4097u32);
    assert!(matches!(
        mellin_transform(&huge, &x, &s),
        Err(ForwardMellinError::Refused(_))
    ));
    let mut deep = expression("exp(-x)");
    for _ in 0..66 {
        deep = power(deep, Expr::from_i64(1));
    }
    assert!(matches!(
        mellin_transform(&deep, &x, &s),
        Err(ForwardMellinError::Refused(_))
    ));
    let source = expression("exp(-x)");
    let mut candidate = generate_mellin_candidate(&source, &x, &s, &mut Unbounded).unwrap();
    candidate.strip.upper =
        MellinStripBound::Finite(BigRational::from_integer(BigInt::from(1) << 40000u32));
    assert!(matches!(
        verify_mellin_candidate(&source, &x, &s, &candidate, &mut Unbounded),
        Err(ForwardMellinError::Refused(_))
    ));
}

#[derive(Default)]
struct Checkpoints {
    count: usize,
    stop: Option<usize>,
}

impl BudgetMeter for Checkpoints {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.count += 1;
        if self.stop == Some(self.count) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
        self.checkpoint()
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        self.checkpoint()
    }
}

#[test]
fn generation_and_independent_verification_have_separate_terminal_budgets() {
    let (x, s) = (Symbol::new("x"), Symbol::new("s"));
    let source = expression("3*x*(2+5*x^2)^(-3)");
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        metered_mellin_transform(&source, &x, &s, &mut empty, &mut Unbounded),
        Err(ForwardMellinError::Meter(_))
    ));
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        metered_mellin_transform(&source, &x, &s, &mut Unbounded, &mut empty),
        Err(ForwardMellinError::Meter(_))
    ));
    let (mut generation, mut verification) = (Checkpoints::default(), Checkpoints::default());
    metered_mellin_transform(&source, &x, &s, &mut generation, &mut verification).unwrap();
    for stop in [1, generation.count / 2, generation.count] {
        let mut cancelled = Checkpoints {
            count: 0,
            stop: Some(stop),
        };
        assert!(matches!(
            metered_mellin_transform(&source, &x, &s, &mut cancelled, &mut Unbounded),
            Err(ForwardMellinError::Meter(MeterError::Cancelled))
        ));
    }
    for stop in [1, verification.count / 2, verification.count] {
        let mut cancelled = Checkpoints {
            count: 0,
            stop: Some(stop),
        };
        assert!(matches!(
            metered_mellin_transform(&source, &x, &s, &mut Unbounded, &mut cancelled),
            Err(ForwardMellinError::Meter(MeterError::Cancelled))
        ));
    }
}
