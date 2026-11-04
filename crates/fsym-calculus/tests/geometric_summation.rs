//! Direct finite sums and coefficient/owner mutations, independent of the
//! triangular antidifference generator.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::geometric_sum::{
    GeometricSummationError, PolynomialGeometricCertificate, metered_polynomial_geometric_sum,
    polynomial_geometric_sum, verify_polynomial_geometric_sum,
};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Constant, Expr, Symbol, SymbolIdentity};
use num_traits::Zero;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn e(n: i64) -> Expr {
    Expr::from_i64(n)
}

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}
fn polynomial(coefficients: &[BigRational], index: &Symbol) -> Expr {
    Expr::Add(
        coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i == 0 {
                    rational_expr(c.clone())
                } else {
                    Expr::Mul(vec![
                        rational_expr(c.clone()),
                        power(Expr::Sym(index.clone()), Expr::from_i64(i as i64)),
                    ])
                }
            })
            .collect(),
    )
}
fn source(coefficients: &[BigRational], ratio: &BigRational, index: &Symbol) -> Expr {
    Expr::Mul(vec![
        polynomial(coefficients, index),
        power(rational_expr(ratio.clone()), Expr::Sym(index.clone())),
    ])
}
fn number(expression: &Expr) -> BigRational {
    match fsym_simplify::simplify(expression) {
        Expr::Integer(n) => BigRational::from_integer(n),
        Expr::Rational(n) => n,
        other => panic!("expected exact number, got {other}"),
    }
}
fn direct(
    coefficients: &[BigRational],
    ratio: &BigRational,
    lower: i64,
    upper: i64,
) -> BigRational {
    if upper + 1 == lower {
        return BigRational::zero();
    }
    if upper < lower {
        return -direct(coefficients, ratio, upper + 1, lower - 1);
    }
    let mut sum = BigRational::zero();
    for k in lower..=upper {
        let point = q(k, 1);
        let weight = coefficients
            .iter()
            .enumerate()
            .fold(BigRational::zero(), |total, (i, c)| {
                total + c * point.pow(i as i32).unwrap()
            });
        sum += weight * ratio.pow(k as i32).unwrap();
    }
    sum
}

#[test]
fn finite_sums_match_direct_rational_evaluation_for_signed_ranges_and_ratios() {
    let k = Symbol::new("k");
    for degree in 0..=6 {
        let coefficients: Vec<_> = (0..=degree)
            .map(|i| q((i + 1) * if i % 2 == 0 { 1 } else { -1 }, i + 2))
            .collect();
        for ratio in [q(1, 2), q(-2, 3), q(2, 1), q(-1, 1), q(1, 1)] {
            let f = source(&coefficients, &ratio, &k);
            for (a, b) in [(-4, 7), (0, 0), (2, 9), (8, 1), (4, 3)] {
                let receipt =
                    polynomial_geometric_sum(&f, &k, &Expr::from_i64(a), &Expr::from_i64(b))
                        .unwrap()
                        .unwrap();
                assert_eq!(number(receipt.value()), direct(&coefficients, &ratio, a, b));
                assert_eq!(receipt.source(), &f);
                assert_eq!(receipt.index(), &k);
            }
        }
    }
}

#[test]
fn convergent_moments_and_alternating_series_have_exact_values() {
    let k = Symbol::new("k");
    for (degree, expected) in [(0, 2), (1, 2), (2, 6), (3, 26), (4, 150)] {
        let mut coefficients = vec![q(0, 1); degree + 1];
        coefficients[degree] = q(1, 1);
        let result = polynomial_geometric_sum(
            &source(&coefficients, &q(1, 2), &k),
            &k,
            &e(0),
            &Expr::Const(Constant::Infinity),
        )
        .unwrap()
        .unwrap();
        assert_eq!(number(result.value()), q(expected, 1));
    }
    let result = polynomial_geometric_sum(
        &source(&[q(0, 1), q(1, 1)], &q(-1, 2), &k),
        &k,
        &e(0),
        &Expr::Const(Constant::Infinity),
    )
    .unwrap()
    .unwrap();
    assert_eq!(number(result.value()), q(-2, 9));
    for ratio in [q(1, 1), q(-1, 1), q(2, 1)] {
        assert!(matches!(
            polynomial_geometric_sum(
                &source(&[q(0, 1), q(1, 1)], &ratio, &k),
                &k,
                &e(0),
                &Expr::Const(Constant::Infinity)
            ),
            Err(GeometricSummationError::Unsupported(_))
        ));
    }
}

#[test]
fn large_ranges_and_symbolic_integer_bounds_use_a_checked_closed_form() {
    let k = Symbol::new("k");
    let f = source(&[q(0, 1), q(1, 1)], &q(2, 1), &k);
    let result = polynomial_geometric_sum(&f, &k, &e(0), &e(10_000))
        .unwrap()
        .unwrap();
    assert_eq!(
        number(result.value()),
        q(9999, 1) * q(2, 1).pow(10001).unwrap() + q(2, 1)
    );
    let n = Symbol::new("n");
    let result = polynomial_geometric_sum(&f, &k, &e(0), &Expr::Sym(n.clone()))
        .unwrap()
        .unwrap();
    for end in [0, 1, 2, 9, 20] {
        let value = result
            .value()
            .subs(&HashMap::from([(n.clone(), Expr::from_i64(end))]));
        assert_eq!(
            number(&value),
            direct(&[q(0, 1), q(1, 1)], &q(2, 1), 0, end)
        );
    }
    assert!(fsym_calculus::summation::summation(&f, &k, &e(0), &Expr::Sym(n)).is_some());
}

#[test]
fn negative_slopes_offsets_and_multiple_geometric_factors_are_exact() {
    let k = Symbol::new("k");
    let exponent = Expr::Add(vec![Expr::Mul(vec![e(-1), Expr::Sym(k.clone())]), e(2)]);
    let f = Expr::Mul(vec![
        Expr::Sym(k.clone()),
        power(e(2), exponent),
        power(rational_expr(q(-1, 3)), Expr::Sym(k.clone())),
    ]);
    let result = polynomial_geometric_sum(&f, &k, &e(-2), &e(12))
        .unwrap()
        .unwrap();
    assert_eq!(
        number(result.value()),
        q(4, 1) * direct(&[q(0, 1), q(1, 1)], &q(-1, 6), -2, 12)
    );
}

#[test]
fn coefficient_mutations_and_same_spelling_typed_rebindings_do_not_verify() {
    let k = Symbol::new("k");
    let f = source(&[q(0, 1), q(1, 1)], &q(2, 1), &k);
    let receipt = polynomial_geometric_sum(&f, &k, &e(0), &e(7))
        .unwrap()
        .unwrap();
    for i in 0..receipt.certificate().antidifference.len() {
        let mut bad = receipt.certificate().clone();
        bad.antidifference[i] += q(1, 1_000_000_007);
        assert!(matches!(
            verify_polynomial_geometric_sum(&f, &k, &e(0), &e(7), bad, &mut Unbounded),
            Err(GeometricSummationError::InvalidCertificate)
        ));
    }
    let changed = source(&[q(0, 1), q(1, 1)], &q(3, 1), &k);
    assert!(
        verify_polynomial_geometric_sum(
            &changed,
            &k,
            &e(0),
            &e(7),
            receipt.certificate().clone(),
            &mut Unbounded
        )
        .is_err()
    );
    let owned = Symbol::with_identity("k", SymbolIdentity::from_assumptions_digest([73; 32]));
    assert!(
        verify_polynomial_geometric_sum(
            &f,
            &owned,
            &e(0),
            &e(7),
            receipt.certificate().clone(),
            &mut Unbounded
        )
        .is_err()
    );
    for bad in [vec![], vec![q(0, 1); 35], vec![q(-2, 1), q(1, 1), q(0, 1)]] {
        assert!(
            verify_polynomial_geometric_sum(
                &f,
                &k,
                &e(0),
                &e(7),
                PolynomialGeometricCertificate {
                    antidifference: bad
                },
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn zero_series_cannot_borrow_a_nonzero_constant_at_infinity() {
    let k = Symbol::new("k");
    let f = source(&[q(0, 1)], &q(1, 1), &k);
    let result = verify_polynomial_geometric_sum(
        &f,
        &k,
        &e(0),
        &Expr::Const(Constant::Infinity),
        PolynomialGeometricCertificate {
            antidifference: vec![q(7, 1)],
        },
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(number(result.value()), q(0, 1));
}

#[test]
fn held_undefined_children_limits_and_divergence_refuse_without_partial_results() {
    let k = Symbol::new("k");
    let geometric = power(e(2), Expr::Sym(k.clone()));
    for bad in [
        Expr::Mul(vec![e(0), power(e(0), e(-1)), geometric.clone()]),
        Expr::Mul(vec![power(power(e(0), e(-1)), e(0)), geometric.clone()]),
        Expr::Mul(vec![power(Expr::Sym(k.clone()), e(33)), geometric.clone()]),
    ] {
        assert!(polynomial_geometric_sum(&bad, &k, &e(0), &e(3)).is_err());
        assert!(fsym_calculus::summation::checked_summation(&bad, &k, &e(0), &e(3)).is_err());
    }
    assert!(polynomial_geometric_sum(&geometric, &k, &e(0), &e(100_000)).is_err());
    assert!(
        fsym_calculus::summation::checked_summation(&geometric, &k, &e(0), &e(100_000)).is_err()
    );
    assert!(matches!(
        polynomial_geometric_sum(&source(&[q(1, 1)], &q(0, 1), &k), &k, &e(0), &e(3)),
        Err(GeometricSummationError::Unsupported(_))
    ));
}

struct Stop {
    charges: usize,
    stop_after: usize,
}
impl BudgetMeter for Stop {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if self.charges >= self.stop_after {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
        self.checkpoint()?;
        self.charges += 1;
        Ok(())
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        for &(d, n) in charges {
            self.charge(d, n)?;
        }
        Ok(())
    }
}

#[test]
fn both_phases_observe_cancellation_at_entry_and_during_arithmetic() {
    let k = Symbol::new("k");
    let f = source(&[q(3, 1), q(2, 1), q(7, 1)], &q(1, 2), &k);
    for after in [0, 10, 100] {
        let mut stop = Stop {
            charges: 0,
            stop_after: after,
        };
        assert!(matches!(
            metered_polynomial_geometric_sum(&f, &k, &e(0), &e(20), &mut stop, &mut Unbounded),
            Err(GeometricSummationError::Meter(MeterError::Cancelled))
        ));
        let mut stop = Stop {
            charges: 0,
            stop_after: after,
        };
        assert!(matches!(
            metered_polynomial_geometric_sum(&f, &k, &e(0), &e(20), &mut Unbounded, &mut stop),
            Err(GeometricSummationError::Meter(MeterError::Cancelled))
        ));
    }
}

#[test]
fn both_phase_budgets_refuse_without_retrying_with_unbounded_work() {
    let k = Symbol::new("k");
    let f = source(&[q(3, 1), q(2, 1), q(7, 1)], &q(1, 2), &k);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 1;
        let mut generator = Budget::new(limits);
        assert!(matches!(
            metered_polynomial_geometric_sum(&f, &k, &e(0), &e(20), &mut generator, &mut Unbounded),
            Err(GeometricSummationError::Meter(MeterError::Budget(_)))
        ));
        let mut verifier = Budget::new(limits);
        assert!(matches!(
            metered_polynomial_geometric_sum(&f, &k, &e(0), &e(20), &mut Unbounded, &mut verifier),
            Err(GeometricSummationError::Meter(MeterError::Budget(_)))
        ));
    }
}

#[test]
fn ratios_arbitrarily_close_to_one_are_not_rounded_to_the_polynomial_case() {
    let k = Symbol::new("k");
    let coefficients = [q(2, 3), q(-7, 5), q(11, 13)];
    for ratio in [
        q(1_000_000_008, 1_000_000_007),
        q(1_000_000_006, 1_000_000_007),
    ] {
        let f = source(&coefficients, &ratio, &k);
        let result = polynomial_geometric_sum(&f, &k, &e(0), &e(8))
            .unwrap()
            .unwrap();
        assert_eq!(number(result.value()), direct(&coefficients, &ratio, 0, 8));
    }
}

#[test]
fn original_probe_counts_empty_product_nodes_before_normalization() {
    let k = Symbol::new("k");
    let empty = Expr::Mul(vec![]);
    let branch = Expr::Mul(vec![empty; 64]);
    let wide = Expr::Mul(vec![branch; 64]);
    assert!(matches!(
        polynomial_geometric_sum(&wide, &k, &e(0), &e(1)),
        Err(GeometricSummationError::Refused(_))
    ));
}

struct GenerationCancellation {
    flag: Arc<AtomicBool>,
    cancel: bool,
}
impl BudgetMeter for GenerationCancellation {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if self.cancel {
            self.flag.store(true, Ordering::SeqCst);
            Ok(())
        } else if self.flag.load(Ordering::SeqCst) {
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
fn cancellation_during_verification_prevents_generator_publication() {
    let k = Symbol::new("k");
    let f = source(&[q(0, 1), q(1, 1)], &q(1, 2), &k);
    let flag = Arc::new(AtomicBool::new(false));
    let mut generator = GenerationCancellation {
        flag: flag.clone(),
        cancel: false,
    };
    let mut verifier = GenerationCancellation { flag, cancel: true };
    assert!(matches!(
        metered_polynomial_geometric_sum(&f, &k, &e(0), &e(5), &mut generator, &mut verifier),
        Err(GeometricSummationError::Meter(MeterError::Cancelled))
    ));
}
