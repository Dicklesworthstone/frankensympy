use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::scalar_bracket::{
    ScalarRootBracketOptions, certified_scalar_root_bracket, metered_certified_scalar_root_bracket,
    propose_scalar_root_bracket,
};
use fsym_core::{
    BigInt, BigRational, Constant, Expr, MAX_SCALAR_ROOT_BISECTIONS, RealBall, RootBoxError,
    ScalarRootBracketError, Symbol, verify_scalar_root_bracket,
};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(value: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(value), Arc::new(Expr::from_i64(exponent)))
}
fn options() -> ScalarRootBracketOptions {
    ScalarRootBracketOptions {
        precision_digits: 24,
        max_bisections: 256,
    }
}
fn shifted(x: &Symbol) -> Expr {
    Expr::Add(vec![Expr::Sym(x.clone()), Expr::Rational(q(-1, 3))])
}

#[test]
fn exact_bisection_encloses_an_odd_multiple_root_that_is_not_dyadic() {
    let x = Symbol::new("multiple_x");
    let source = power(shifted(&x), 3);
    let tolerance = q(1, 1_000_000_000);
    let receipt = certified_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &tolerance,
        options(),
    )
    .unwrap();
    assert!(receipt.root_enclosure().contains(&q(1, 3)));
    assert!(receipt.root_enclosure().width() <= tolerance);
    assert!(receipt.endpoint_values()[0].is_negative());
    assert!(receipt.endpoint_values()[1].is_positive());
    assert_eq!(receipt.expression(), &source);
    assert_eq!(receipt.variable(), &x);
}

#[test]
fn a_nondyadic_absolute_value_corner_needs_no_classical_derivative() {
    let x = Symbol::new("corner_x");
    let base = shifted(&x);
    let source = Expr::Add(vec![
        Expr::Mul(vec![Expr::from_i64(2), base.clone()]),
        Expr::Function("Abs".into(), vec![base]),
    ]);
    let tolerance = q(1, 1_000_000);
    let receipt = certified_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &tolerance,
        options(),
    )
    .unwrap();
    // On the two sides this is x-1/3 and 3*(x-1/3), respectively.
    assert!(receipt.root_enclosure().contains(&q(1, 3)));
    assert!(receipt.root_enclosure().width() <= tolerance);
    assert!(receipt.endpoint_values()[0].is_negative());
    assert!(receipt.endpoint_values()[1].is_positive());
}

#[test]
fn reversed_sign_orientation_produces_the_same_exact_bracket() {
    let x = Symbol::new("orientation_x");
    let source = power(shifted(&x), 3);
    let positive = certified_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 10_000),
        options(),
    )
    .unwrap();
    let negative = certified_scalar_root_bracket(
        &Expr::Mul(vec![Expr::from_i64(-1), source]),
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 10_000),
        options(),
    )
    .unwrap();
    assert_eq!(positive.witness(), negative.witness());
    assert!(negative.endpoint_values()[0].is_positive());
    assert!(negative.endpoint_values()[1].is_negative());
}

#[test]
fn exact_endpoint_roots_and_point_requests_do_not_spend_bisection_steps() {
    let x = Symbol::new("endpoint_x");
    let sx = Expr::Sym(x.clone());
    let no_steps = ScalarRootBracketOptions {
        max_bisections: 0,
        ..options()
    };
    for (source, lower, upper, root) in [
        (power(sx.clone(), 2), q(0, 1), q(2, 1), q(0, 1)),
        (
            power(Expr::Add(vec![sx, Expr::from_i64(-2)]), 2),
            q(0, 1),
            q(2, 1),
            q(2, 1),
        ),
        (Expr::from_i64(0), q(-1, 1), q(1, 1), q(-1, 1)),
        (Expr::from_i64(0), q(3, 1), q(3, 1), q(3, 1)),
    ] {
        let receipt = certified_scalar_root_bracket(
            &source,
            &x,
            &lower,
            &upper,
            &q(1, 1_000_000),
            no_steps,
        )
        .unwrap();
        assert_eq!(receipt.root_enclosure(), &RealBall::exact(root));
    }
}

#[test]
fn uncertain_transcendental_midpoints_are_not_published_as_exact_roots() {
    let x = Symbol::new("rounded_x");
    let epsilon = BigRational::new(1.into(), BigInt::from(10).pow(30));
    let source = Expr::Add(vec![
        Expr::Function(
            "sin".into(),
            vec![Expr::Mul(vec![
                Expr::Const(Constant::Pi),
                Expr::Sym(x.clone()),
            ])],
        ),
        Expr::Rational(epsilon),
    ]);
    let tolerance = q(1, 10_000);
    let receipt = certified_scalar_root_bracket(
        &source,
        &x,
        &q(1, 2),
        &q(3, 2),
        &tolerance,
        ScalarRootBracketOptions {
            precision_digits: 8,
            max_bisections: 64,
        },
    )
    .unwrap();
    // f(1)=10^-30 is positive, despite the working enclosure straddling
    // zero. Quarter probes retain a real interval around the nearby root.
    assert!(receipt.root_enclosure().width() > q(0, 1));
    assert!(receipt.root_enclosure().width() <= tolerance);
    assert!(receipt.endpoint_values()[0].is_positive());
    assert!(receipt.endpoint_values()[1].is_negative());
}

#[test]
fn lack_of_a_sign_bracket_and_exhausted_accuracy_remain_explicit() {
    let x = Symbol::new("inconclusive_x");
    let sx = Expr::Sym(x.clone());
    // There really are two roots here, but equal endpoint signs do not
    // establish their existence through this bounded sign-bracket method.
    let two_roots = Expr::Add(vec![power(sx, 2), Expr::from_i64(-1)]);
    assert!(matches!(
        certified_scalar_root_bracket(
            &two_roots,
            &x,
            &q(-2, 1),
            &q(2, 1),
            &q(1, 100),
            options(),
        ),
        Err(ScalarRootBracketError::NoSignChange)
    ));
    assert!(matches!(
        certified_scalar_root_bracket(
            &shifted(&x),
            &x,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1_000_000),
            ScalarRootBracketOptions {
                max_bisections: 1,
                ..options()
            },
        ),
        Err(ScalarRootBracketError::AccuracyNotReached)
    ));
}

#[test]
fn poles_behind_sign_changes_and_held_zeroes_never_enter_a_refinement() {
    let x = Symbol::new("pole_x");
    let sx = Expr::Sym(x.clone());
    let inverse = power(sx.clone(), -1);
    for source in [
        inverse.clone(),
        Expr::Add(vec![
            sx.clone(),
            Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
        ]),
        Expr::Add(vec![power(inverse, 0), Expr::from_i64(-1)]),
        Expr::Mul(vec![
            sx.clone(),
            power(Expr::Add(vec![sx, Expr::from_i64(1)]), -1),
        ]),
    ] {
        assert!(
            certified_scalar_root_bracket(
                &source,
                &x,
                &q(-2, 1),
                &q(2, 1),
                &q(1, 100),
                options(),
            )
            .is_err()
        );
    }
}

#[test]
fn proposed_intervals_are_rechecked_against_the_receivers_original_source() {
    let x = Symbol::new("replay_x");
    let source = shifted(&x);
    let tolerance = q(1, 10_000);
    let proposed = propose_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &tolerance,
        options(),
        &mut Unbounded,
    )
    .unwrap();
    verify_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &tolerance,
        &proposed,
        &mut Unbounded,
    )
    .unwrap();
    let wrong_source = Expr::Add(vec![source.clone(), Expr::from_i64(1)]);
    assert!(matches!(
        verify_scalar_root_bracket(
            &wrong_source,
            &x,
            &q(0, 1),
            &q(1, 1),
            &tolerance,
            &proposed,
            &mut Unbounded,
        ),
        Err(ScalarRootBracketError::NoSignChange)
    ));
    let mut altered = proposed;
    altered.lower = q(0, 1);
    altered.upper = q(0, 1);
    assert!(matches!(
        verify_scalar_root_bracket(
            &source,
            &x,
            &q(0, 1),
            &q(1, 1),
            &tolerance,
            &altered,
            &mut Unbounded,
        ),
        Err(ScalarRootBracketError::NoSignChange)
    ));
}

#[test]
fn invalid_bounds_precision_and_search_limits_cannot_be_relaxed() {
    let x = Symbol::new("options_x");
    let source = Expr::Sym(x.clone());
    assert!(matches!(
        certified_scalar_root_bracket(
            &source,
            &x,
            &q(-1, 1),
            &q(1, 1),
            &q(1, 100),
            ScalarRootBracketOptions {
                max_bisections: MAX_SCALAR_ROOT_BISECTIONS + 1,
                ..options()
            },
        ),
        Err(ScalarRootBracketError::ResourceLimit(_))
    ));
    for precision_digits in [7, 129, u32::MAX] {
        assert!(matches!(
            certified_scalar_root_bracket(
                &source,
                &x,
                &q(-1, 1),
                &q(1, 1),
                &q(1, 100),
                ScalarRootBracketOptions {
                    precision_digits,
                    ..options()
                },
            ),
            Err(ScalarRootBracketError::Enclosure(RootBoxError::InvalidPrecision))
        ));
    }
    assert!(matches!(
        certified_scalar_root_bracket(
            &source,
            &x,
            &q(1, 1),
            &q(-1, 1),
            &q(1, 100),
            options(),
        ),
        Err(ScalarRootBracketError::InvalidBracket(_))
    ));
}

#[test]
fn both_phases_use_their_own_cumulative_budget_and_do_not_replenish_it() {
    let x = Symbol::new("metered_x");
    let source = power(shifted(&x), 3);
    let limit = 96 * 1024 * 1024;
    let mut generator = Budget::new(BudgetLimits::uniform(limit, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(limit, 0));
    let receipt = metered_certified_scalar_root_bracket(
        &source,
        &x,
        &q(0, 1),
        &q(1, 1),
        &q(1, 100_000),
        options(),
        &mut generator,
        &mut verifier,
    )
    .unwrap();
    assert!(receipt.root_enclosure().contains(&q(1, 3)));
    for budget in [&generator, &verifier] {
        assert!(budget.remaining(Dimension::ComputeSteps) < limit);
        assert!(budget.remaining(Dimension::MemoryBytes) < limit);
    }
    let before = generator.snapshot();
    assert!(matches!(
        propose_scalar_root_bracket(
            &source,
            &x,
            &q(0, 1),
            &q(1, 1),
            &q(1, 100),
            options(),
            &mut generator,
        ),
        Err(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(generator.snapshot(), before);
}

#[derive(Default)]
struct Checkpoints {
    calls: usize,
    cancel: bool,
}
impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        self.checkpoint()
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.calls += 1;
        if self.cancel {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn generator_cancellation_leaves_the_verifier_untouched_and_verifier_cancellation_refuses() {
    let x = Symbol::new("cancel_x");
    let source = shifted(&x);
    for cancel_generator in [true, false] {
        let mut generator = Checkpoints {
            cancel: cancel_generator,
            ..Checkpoints::default()
        };
        let mut verifier = Checkpoints {
            cancel: !cancel_generator,
            ..Checkpoints::default()
        };
        assert!(matches!(
            metered_certified_scalar_root_bracket(
                &source,
                &x,
                &q(0, 1),
                &q(1, 1),
                &q(1, 100),
                options(),
                &mut generator,
                &mut verifier,
            ),
            Err(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        ));
        if cancel_generator {
            assert_eq!(verifier.calls, 0);
        } else {
            assert!(generator.calls > 1);
            assert!(verifier.calls > 0);
        }
    }
}

struct CancellationFlag {
    flag: Rc<Cell<bool>>,
    signal: bool,
}
impl BudgetMeter for CancellationFlag {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        self.checkpoint()
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.checkpoint()
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.signal {
            self.flag.set(true);
            Ok(())
        } else if self.flag.get() {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_observed_after_independent_verification_still_blocks_publication() {
    let x = Symbol::new("late_cancel_x");
    let flag = Rc::new(Cell::new(false));
    let mut generator = CancellationFlag {
        flag: flag.clone(),
        signal: false,
    };
    let mut verifier = CancellationFlag {
        flag: flag.clone(),
        signal: true,
    };
    assert!(matches!(
        metered_certified_scalar_root_bracket(
            &shifted(&x),
            &x,
            &q(0, 1),
            &q(1, 1),
            &q(1, 100),
            options(),
            &mut generator,
            &mut verifier,
        ),
        Err(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
            MeterError::Cancelled
        )))
    ));
    assert!(flag.get());
}
