//! Strip-dependent ordinary inverses, independent mode values and adversarial receipts.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_laplace::{LaplaceBlock, RationalLaplaceError};
use fsym_calculus::rational_mellin::{
    MellinStrip, MellinStripBound, MellinSupport, RationalMellinError, inverse_rational_mellin,
    metered_inverse_rational_mellin, verify_rational_mellin_certificate,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_polys::UnivariatePoly;
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn finite(value: i64) -> MellinStripBound {
    MellinStripBound::Finite(q(value, 1))
}

fn strip(lower: MellinStripBound, upper: MellinStripBound) -> MellinStrip {
    MellinStrip { lower, upper }
}

fn right_half_plane(value: i64) -> MellinStrip {
    strip(finite(value), MellinStripBound::PositiveInfinity)
}

fn left_half_plane(value: i64) -> MellinStrip {
    strip(MellinStripBound::NegativeInfinity, finite(value))
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn poly(values: &[i64], symbol: &Symbol) -> Expr {
    UnivariatePoly::new(
        symbol.clone(),
        values.iter().map(|&value| q(value, 1)).collect(),
    )
    .to_expr()
}

fn fraction(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, power(denominator, -1)])
}

/// A test-only real AST interpreter checks the rendered expression against
/// hand-derived functions. It does not call the production transform renderer,
/// symbolic simplifier, Laplace verifier or numerical compiler.
fn value(expression: &Expr, variable: &Symbol, point: f64) -> f64 {
    match expression {
        Expr::Integer(integer) => BigRational::from_integer(integer.clone()).to_f64().unwrap(),
        Expr::Rational(rational) => rational.to_f64().unwrap(),
        Expr::Sym(symbol) if symbol == variable => point,
        Expr::Add(terms) => terms.iter().map(|term| value(term, variable, point)).sum(),
        Expr::Mul(factors) => factors
            .iter()
            .map(|factor| value(factor, variable, point))
            .product(),
        Expr::Pow(base, exponent) => {
            value(base, variable, point).powf(value(exponent, variable, point))
        }
        Expr::Function(name, arguments) if arguments.len() == 1 => {
            let argument = value(&arguments[0], variable, point);
            match name.as_str() {
                "log" => argument.ln(),
                "exp" => argument.exp(),
                "sin" => argument.sin(),
                "cos" => argument.cos(),
                "sinh" => argument.sinh(),
                "cosh" => argument.cosh(),
                "Heaviside" => {
                    if argument > 0.0 {
                        1.0
                    } else if argument < 0.0 {
                        0.0
                    } else {
                        0.5
                    }
                }
                _ => panic!("unexpected function {name}"),
            }
        }
        _ => panic!("unexpected test expression {expression:?}"),
    }
}

fn assert_values(expression: &Expr, variable: &Symbol, expected: impl Fn(f64) -> f64) {
    for point in [0.125f64, 0.5, 2.0, 8.0] {
        let actual = value(expression, variable, point);
        let expected = expected(point);
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= 2e-12 * expected.abs().max(1.0),
            "x={point}: actual {actual}, expected {expected}; {expression:?}"
        );
    }
}

#[test]
fn the_same_simple_transform_has_distinct_left_and_right_inverses() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(Expr::Sym(s.clone()), -1);
    let left = inverse_rational_mellin(&source, &s, &x, &right_half_plane(0)).unwrap();
    let right = inverse_rational_mellin(&source, &s, &x, &left_half_plane(0)).unwrap();
    assert_eq!(left.certificate().sides, [MellinSupport::Left]);
    assert_eq!(right.certificate().sides, [MellinSupport::Right]);
    assert_values(left.function(), &x, |x| if x < 1.0 { 1.0 } else { 0.0 });
    assert_values(right.function(), &x, |x| if x > 1.0 { -1.0 } else { 0.0 });
    assert_eq!(value(left.function(), &x, 1.0), 0.5);
    assert_eq!(value(right.function(), &x, 1.0), -0.5);
    assert_eq!(left.source(), &source);
    assert_eq!(left.strip(), &right_half_plane(0));
    assert_eq!(left.original_guard_count(), 1);
}

#[test]
fn repeated_shifted_real_poles_preserve_logarithmic_weights_and_signs() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    for order in 1..=4i64 {
        let source = power(poly(&[-2, 1], &s), -order);
        let denominator = (1..order).product::<i64>() as f64;
        for (strip, left) in [(right_half_plane(2), true), (left_half_plane(2), false)] {
            let result = inverse_rational_mellin(&source, &s, &x, &strip).unwrap();
            assert_values(result.function(), &x, |x| {
                if (x < 1.0) == left {
                    let sign = if left { 1.0 } else { -1.0 };
                    sign * x.powi(-2) * (-x.ln()).powi((order - 1) as i32) / denominator
                } else {
                    0.0
                }
            });
        }
    }
}

#[test]
fn three_strips_of_a_two_pole_transform_give_three_correct_functions() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(poly(&[-1, 0, 1], &s), -1);
    let middle = inverse_rational_mellin(&source, &s, &x, &strip(finite(-1), finite(1))).unwrap();
    assert_values(middle.function(), &x, |x| {
        if x < 1.0 { -x / 2.0 } else { -1.0 / (2.0 * x) }
    });
    let left = inverse_rational_mellin(&source, &s, &x, &right_half_plane(1)).unwrap();
    assert_values(left.function(), &x, |x| {
        if x < 1.0 { (1.0 / x - x) / 2.0 } else { 0.0 }
    });
    let right = inverse_rational_mellin(&source, &s, &x, &left_half_plane(-1)).unwrap();
    assert_values(right.function(), &x, |x| {
        if x > 1.0 { (x - 1.0 / x) / 2.0 } else { 0.0 }
    });
}

#[test]
fn repeated_complex_quadratic_modes_use_normalized_sine_on_both_sides() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    // ((s+2)^2 + 5)^-2 has inverse exp(-2t)*(S_5(t)-t*C_5(t))/10.
    let source = power(poly(&[9, 4, 1], &s), -2);
    for (strip, left) in [(right_half_plane(-2), true), (left_half_plane(-2), false)] {
        let result = inverse_rational_mellin(&source, &s, &x, &strip).unwrap();
        assert_values(result.function(), &x, |x| {
            if (x < 1.0) != left {
                return 0.0;
            }
            let t = -x.ln();
            let r = 5.0f64.sqrt();
            let sign = if left { 1.0 } else { -1.0 };
            sign * x * x * ((r * t).sin() / r - t * (r * t).cos()) / 10.0
        });
    }
}

#[test]
fn irrational_real_quadratic_modes_split_with_independently_checked_coefficients() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(poly(&[-2, 0, 1], &s), -1);
    let middle = strip(finite(-1), finite(1));
    let result = inverse_rational_mellin(&source, &s, &x, &middle).unwrap();
    assert_eq!(result.certificate().sides, [MellinSupport::Split]);
    let r = 2.0f64.sqrt();
    assert_values(result.function(), &x, |x| {
        if x < 1.0 {
            -x.powf(r) / (2.0 * r)
        } else {
            -x.powf(-r) / (2.0 * r)
        }
    });
    let source = power(poly(&[-2, 0, 1], &s), -2);
    let result = inverse_rational_mellin(&source, &s, &x, &middle).unwrap();
    assert_values(result.function(), &x, |x| {
        let t = -x.ln();
        if x < 1.0 {
            x.powf(r) * (t + 1.0 / r) / 8.0
        } else {
            -x.powf(-r) * (t - 1.0 / r) / 8.0
        }
    });
}

#[test]
fn irrational_root_positions_use_exact_squared_comparisons() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(poly(&[-2, 0, 1], &s), -1);
    for strip in [
        strip(
            MellinStripBound::Finite(q(3, 2)),
            MellinStripBound::PositiveInfinity,
        ),
        strip(
            MellinStripBound::NegativeInfinity,
            MellinStripBound::Finite(q(-3, 2)),
        ),
        strip(
            MellinStripBound::Finite(q(-7, 5)),
            MellinStripBound::Finite(q(7, 5)),
        ),
    ] {
        inverse_rational_mellin(&source, &s, &x, &strip).unwrap();
    }
    for strip in [
        strip(
            MellinStripBound::Finite(q(7, 5)),
            MellinStripBound::PositiveInfinity,
        ),
        strip(
            MellinStripBound::NegativeInfinity,
            MellinStripBound::Finite(q(-7, 5)),
        ),
        strip(
            MellinStripBound::Finite(q(-3, 2)),
            MellinStripBound::Finite(q(3, 2)),
        ),
    ] {
        assert!(matches!(
            inverse_rational_mellin(&source, &s, &x, &strip),
            Err(RationalMellinError::InvalidStrip(_))
        ));
    }
    // The squared magnitude underflows binary64, but exact root positions remain distinct.
    let scale = BigInt::from(10).pow(200);
    let magnitude = BigRational::new(2.into(), &scale * &scale);
    let source = power(
        UnivariatePoly::new(s.clone(), vec![-magnitude, q(0, 1), q(1, 1)]).to_expr(),
        -1,
    );
    let small = BigRational::new(1.into(), scale.clone());
    let large = BigRational::new(2.into(), scale);
    let narrow = strip(
        MellinStripBound::Finite(-small.clone()),
        MellinStripBound::Finite(small),
    );
    let wide = strip(
        MellinStripBound::Finite(-large.clone()),
        MellinStripBound::Finite(large),
    );
    assert_eq!(
        inverse_rational_mellin(&source, &s, &x, &narrow)
            .unwrap()
            .certificate()
            .sides,
        [MellinSupport::Split]
    );
    assert!(matches!(
        inverse_rational_mellin(&source, &s, &x, &wide),
        Err(RationalMellinError::InvalidStrip(_))
    ));
}

#[test]
fn poles_inside_the_strip_and_empty_or_reversed_strips_refuse() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(Expr::Sym(s.clone()), -1);
    for strip in [
        strip(finite(-1), finite(1)),
        strip(finite(0), finite(0)),
        strip(finite(2), finite(1)),
        strip(
            MellinStripBound::PositiveInfinity,
            MellinStripBound::PositiveInfinity,
        ),
        strip(
            MellinStripBound::NegativeInfinity,
            MellinStripBound::NegativeInfinity,
        ),
    ] {
        assert!(matches!(
            inverse_rational_mellin(&source, &s, &x, &strip),
            Err(RationalMellinError::InvalidStrip(_))
        ));
    }
    let complex = power(poly(&[1, 0, 1], &s), -1);
    assert!(matches!(
        inverse_rational_mellin(&complex, &s, &x, &strip(finite(-1), finite(1))),
        Err(RationalMellinError::InvalidStrip(_))
    ));
}

#[test]
fn original_holes_survive_cancellation_zero_products_and_nested_powers() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let hole = poly(&[-1, 1], &s);
    let inverse = power(hole.clone(), -1);
    let originals = [
        Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
        Expr::Mul(vec![Expr::from_i64(0), power(inverse.clone(), -1)]),
        fraction(power(inverse, 0), Expr::Sym(s.clone())),
        fraction(power(hole.clone(), 0), Expr::Sym(s.clone())),
        fraction(hole.clone(), Expr::Mul(vec![hole, Expr::Sym(s.clone())])),
    ];
    for original in originals {
        assert!(matches!(
            inverse_rational_mellin(&original, &s, &x, &strip(finite(0), finite(2))),
            Err(RationalMellinError::InvalidStrip(_))
        ));
        let receipt = inverse_rational_mellin(&original, &s, &x, &right_half_plane(2)).unwrap();
        assert_eq!(receipt.source(), &original);
        assert!(receipt.original_guard_count() > 0);
        assert!(
            verify_rational_mellin_certificate(
                &original,
                &s,
                &x,
                &strip(finite(0), finite(2)),
                receipt.certificate(),
                &mut Unbounded
            )
            .is_err()
        );
    }
    for exponent in [-1, 0] {
        let original = Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), exponent)]);
        assert!(matches!(
            inverse_rational_mellin(&original, &s, &x, &right_half_plane(0)),
            Err(RationalMellinError::Laplace(
                RationalLaplaceError::InvalidSource(_)
            ))
        ));
    }
}

#[test]
fn zero_has_a_whole_plane_inverse_but_improper_and_unresolved_sources_refuse() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let whole = strip(
        MellinStripBound::NegativeInfinity,
        MellinStripBound::PositiveInfinity,
    );
    let zero = inverse_rational_mellin(&Expr::from_i64(0), &s, &x, &whole).unwrap();
    assert!(zero.function().is_zero());
    assert!(zero.certificate().sides.is_empty());
    for source in [
        Expr::from_i64(1),
        Expr::Sym(s.clone()),
        power(poly(&[1, 1, 0, 1], &s), -1),
        Expr::Function("gamma".into(), vec![Expr::Sym(s.clone())]),
        Expr::Mul(vec![Expr::from_i64(0), power(poly(&[1, 1, 0, 1], &s), -1)]),
    ] {
        assert!(
            inverse_rational_mellin(&source, &s, &x, &right_half_plane(0))
                .unwrap_err()
                .is_unsupported()
        );
    }
}

#[test]
fn certificate_replay_rejects_changed_modes_support_source_and_strip() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(poly(&[-2, 0, 1], &s), -2);
    let middle = strip(finite(-1), finite(1));
    let good = inverse_rational_mellin(&source, &s, &x, &middle).unwrap();
    let mut wrong_side = good.certificate().clone();
    wrong_side.sides[0] = MellinSupport::Left;
    let mut missing_side = good.certificate().clone();
    missing_side.sides.clear();
    let mut wrong_coefficient = good.certificate().clone();
    if let LaplaceBlock::Quadratic { cosine, .. } = &mut wrong_coefficient.laplace.blocks[0] {
        cosine[0] = q(1, 1);
    } else {
        panic!("expected quadratic");
    }
    let mut missing_mode = good.certificate().clone();
    missing_mode.laplace.blocks.clear();
    missing_mode.sides.clear();
    for mutant in [wrong_side, missing_side, wrong_coefficient, missing_mode] {
        assert!(
            verify_rational_mellin_certificate(&source, &s, &x, &middle, &mutant, &mut Unbounded)
                .is_err()
        );
    }
    assert!(
        verify_rational_mellin_certificate(
            &source,
            &s,
            &x,
            &right_half_plane(2),
            good.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_rational_mellin_certificate(
            &Expr::Mul(vec![Expr::from_i64(2), source]),
            &s,
            &x,
            &middle,
            good.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn typed_coordinate_owners_survive_and_foreign_owners_refuse() {
    let plain = Symbol::new("s");
    let s = Symbol::with_identity("s", SymbolIdentity::from_assumptions_digest([7; 32]));
    let x = Symbol::with_identity("s", SymbolIdentity::from_assumptions_digest([9; 32]));
    let source = power(poly(&[-1, 1], &s), -2);
    let result = inverse_rational_mellin(&source, &s, &x, &right_half_plane(1)).unwrap();
    assert_eq!(result.transform_variable(), &s);
    assert_eq!(result.argument_variable(), &x);
    assert_eq!(result.function().free_symbols(), vec![x.clone()]);
    assert!(
        inverse_rational_mellin(&source, &plain, &x, &right_half_plane(1))
            .unwrap_err()
            .is_unsupported()
    );
    assert!(matches!(
        inverse_rational_mellin(&source, &s, &s, &right_half_plane(1)),
        Err(RationalMellinError::InvalidSource(_))
    ));
}

#[test]
fn source_bound_and_certificate_limits_precede_expensive_work() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(Expr::Sym(s.clone()), -1);
    let huge = BigInt::one() << 4097u32;
    let bound = strip(
        MellinStripBound::Finite(BigRational::from_integer(huge.clone())),
        MellinStripBound::PositiveInfinity,
    );
    assert!(matches!(
        inverse_rational_mellin(&source, &s, &x, &bound),
        Err(RationalMellinError::ResourceLimit(_))
    ));
    for source in [
        Expr::Integer(huge),
        power(Expr::Sym(s.clone()), -33),
        Expr::Add(vec![Expr::from_i64(0); 4097]),
    ] {
        assert!(matches!(
            inverse_rational_mellin(&source, &s, &x, &right_half_plane(0)),
            Err(RationalMellinError::Laplace(
                RationalLaplaceError::ResourceLimit(_)
            ))
        ));
    }
    let good = inverse_rational_mellin(&source, &s, &x, &right_half_plane(0)).unwrap();
    let mut oversized = good.certificate().clone();
    oversized.sides = vec![MellinSupport::Left; 33];
    assert!(
        verify_rational_mellin_certificate(
            &source,
            &s,
            &x,
            &right_half_plane(0),
            &oversized,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    stopped: bool,
    charged: [u64; 3],
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "no work after cancellation");
        assert_ne!(amount, 0);
        let index = match dimension {
            Dimension::ComputeSteps => Some(0),
            Dimension::MemoryBytes => Some(1),
            Dimension::AllocationCount => Some(2),
            _ => None,
        };
        if let Some(index) = index {
            self.charged[index] += amount;
        }
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "no repeated callback after cancellation");
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_numerical_checkpoint_can_cancel_without_publishing_a_receipt() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(Expr::Sym(s.clone()), -1);
    let strip = right_half_plane(0);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_rational_mellin(&source, &s, &x, &strip, &mut generator, &mut verifier)
        .unwrap();
    assert!(generator.charged.iter().all(|&amount| amount > 0));
    assert!(verifier.charged.iter().all(|&amount| amount > 0));
    for checkpoint in 1..=generator.checkpoints {
        let mut cancelled = Meter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(
            matches!(
                metered_inverse_rational_mellin(
                    &source,
                    &s,
                    &x,
                    &strip,
                    &mut cancelled,
                    &mut Unbounded
                ),
                Err(RationalMellinError::Meter(MeterError::Cancelled))
            ),
            "generator checkpoint {checkpoint}"
        );
    }
    for checkpoint in 1..=verifier.checkpoints {
        let mut cancelled = Meter {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(
            matches!(
                metered_inverse_rational_mellin(
                    &source,
                    &s,
                    &x,
                    &strip,
                    &mut Unbounded,
                    &mut cancelled
                ),
                Err(RationalMellinError::Meter(MeterError::Cancelled))
            ),
            "verifier checkpoint {checkpoint}"
        );
    }
}

#[test]
fn cumulative_phase_budgets_include_strip_checks_rendering_and_publication() {
    let s = Symbol::new("s");
    let x = Symbol::new("x");
    let source = power(poly(&[-2, 0, 1], &s), -2);
    let strip = strip(finite(-1), finite(1));
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_rational_mellin(&source, &s, &x, &strip, &mut generator, &mut verifier)
        .unwrap();
    for (generation, steps) in [(true, generator.charged[0]), (false, verifier.charged[0])] {
        let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = steps - 1;
        let mut limited = Budget::new(limits);
        let result = if generation {
            metered_inverse_rational_mellin(&source, &s, &x, &strip, &mut limited, &mut Unbounded)
        } else {
            metered_inverse_rational_mellin(&source, &s, &x, &strip, &mut Unbounded, &mut limited)
        };
        assert!(matches!(
            result,
            Err(RationalMellinError::Meter(MeterError::Budget(_)))
        ));
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(matches!(
        metered_inverse_rational_mellin(&source, &s, &x, &strip, &mut tiny, &mut untouched),
        Err(RationalMellinError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(untouched.charged, [0; 3]);
    assert_eq!(untouched.checkpoints, 0);
}
