//! Independent finite enumeration, exact moment identities and certificate adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::binomial_sum::{
    BinomialSummationError, PolynomialBinomialCertificate, metered_polynomial_binomial_sum,
    polynomial_binomial_sum, verify_polynomial_binomial_sum,
};
use fsym_calculus::summation::checked_summation;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity};
use num_traits::Zero;
use std::sync::Arc;

fn e(value: i64) -> Expr {
    Expr::from_i64(value)
}
fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}
fn binomial(n: Expr, k: &Symbol) -> Expr {
    Expr::Function("binomial".into(), vec![n, Expr::Sym(k.clone())])
}
fn source(
    coefficients: &[BigRational],
    a: &BigRational,
    b: &BigRational,
    n: i64,
    k: &Symbol,
) -> Expr {
    let weight = Expr::Add(
        coefficients
            .iter()
            .enumerate()
            .map(|(j, c)| {
                Expr::Mul(vec![
                    rational_expr(c.clone()),
                    power(Expr::Sym(k.clone()), e(j as i64)),
                ])
            })
            .collect(),
    );
    Expr::Mul(vec![
        weight,
        binomial(e(n), k),
        power(rational_expr(a.clone()), Expr::Sym(k.clone())),
        power(
            rational_expr(b.clone()),
            Expr::Add(vec![e(n), Expr::Mul(vec![e(-1), Expr::Sym(k.clone())])]),
        ),
    ])
}

// Small finite enumeration with integer binomial coefficients; no basis
// transform, moment formula or candidate/verifier helper is reused.
fn direct(coefficients: &[BigRational], a: &BigRational, b: &BigRational, n: usize) -> BigRational {
    let mut choose = 1i64;
    let mut result = BigRational::zero();
    for k in 0..=n {
        let x = q(k as i64, 1);
        let weight = coefficients
            .iter()
            .rev()
            .fold(BigRational::zero(), |v, c| v * &x + c.clone());
        result += weight
            * q(choose, 1)
            * a.pow(k as i32).expect("nonnegative exact power")
            * b.pow((n - k) as i32).expect("nonnegative exact power");
        if k < n {
            choose = choose * (n - k) as i64 / (k + 1) as i64;
        }
    }
    result
}

#[test]
fn long_binomial_sums_and_first_two_moments_bypass_direct_enumeration() {
    let k = Symbol::new("k");
    let n = 3000;
    let expected = [
        BigInt::from(1) << 3000_u32,
        BigInt::from(n) * (BigInt::from(1) << 2999_u32),
        BigInt::from(n * (n + 1)) * (BigInt::from(1) << 2998_u32),
    ];
    for (degree, expected) in expected.into_iter().enumerate() {
        let mut coefficients = vec![q(0, 1); degree + 1];
        coefficients[degree] = q(1, 1);
        let summand = source(&coefficients, &q(1, 1), &q(1, 1), n, &k);
        let receipt = polynomial_binomial_sum(&summand, &k, &e(0), &e(n))
            .unwrap()
            .unwrap();
        assert_eq!(receipt.value(), &Expr::Integer(expected));
        assert_eq!(receipt.source(), &summand);
        assert_eq!(receipt.index(), &k);
        assert_eq!(receipt.lower(), &e(0));
        assert_eq!(receipt.upper(), &e(n));
        assert_eq!(
            checked_summation(&summand, &k, &e(0), &e(n))
                .unwrap()
                .as_ref(),
            Some(receipt.value())
        );
    }
    // Exact rational binomial probability mass and expectation.
    for (coefficients, expected) in [
        (vec![q(1, 1)], q(1, 1)),
        (vec![q(0, 1), q(1, 1)], q(1000, 1)),
    ] {
        let summand = source(&coefficients, &q(1, 3), &q(2, 3), n, &k);
        assert_eq!(
            checked_summation(&summand, &k, &e(0), &e(n)).unwrap(),
            Some(rational_expr(expected))
        );
    }
}

#[test]
fn rational_weighted_moments_match_independent_finite_enumeration() {
    let k = Symbol::new("k");
    let weights = [
        vec![q(0, 1)],
        vec![q(1, 1)],
        vec![q(5, 7), q(-3, 5), q(2, 3)],
        vec![
            q(0, 1),
            q(0, 1),
            q(0, 1),
            q(0, 1),
            q(0, 1),
            q(0, 1),
            q(1, 1),
        ],
    ];
    for n in 0..=8 {
        for (a, b) in [
            (q(2, 1), q(3, 1)),
            (q(-2, 1), q(2, 1)),
            (q(0, 1), q(0, 1)),
            (q(0, 1), q(3, 1)),
            (q(2, 1), q(0, 1)),
            (q(-1, 2), q(2, 3)),
        ] {
            for coefficients in &weights {
                let summand = source(coefficients, &a, &b, n as i64, &k);
                let answer = polynomial_binomial_sum(&summand, &k, &e(0), &e(n as i64))
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    answer.value(),
                    &rational_expr(direct(coefficients, &a, &b, n)),
                    "n={n}, a={a}, b={b}, p={coefficients:?}"
                );
            }
        }
    }
}

#[test]
fn alternating_sums_truncate_degree_above_n_before_zero_powers() {
    let k = Symbol::new("k");
    for (n, degree, expected) in [(5, 4, 0), (5, 5, 120), (3, 6, 540), (0, 0, 1), (0, 6, 0)] {
        let mut coefficients = vec![q(0, 1); degree + 1];
        coefficients[degree] = q(1, 1);
        let summand = source(&coefficients, &q(1, 1), &q(-1, 1), n, &k);
        assert_eq!(
            checked_summation(&summand, &k, &e(0), &e(n)).unwrap(),
            Some(e(expected))
        );
    }
}

#[test]
fn independent_verifier_rejects_value_coefficients_and_original_binding_changes() {
    let k = Symbol::with_identity("k", SymbolIdentity::from_assumptions_digest([4; 32]));
    let summand = source(&[q(5, 7), q(-3, 5), q(2, 3)], &q(2, 1), &q(3, 1), 7, &k);
    let receipt = polynomial_binomial_sum(&summand, &k, &e(0), &e(7))
        .unwrap()
        .unwrap();
    let valid = receipt.certificate();
    assert!(
        verify_polynomial_binomial_sum(&summand, &k, &e(0), &e(7), valid.clone(), &mut Unbounded)
            .is_ok()
    );
    let mut corruptions = Vec::new();
    let mut changed = valid.clone();
    changed.claimed_sum += q(1, 1_000_000_007);
    corruptions.push(changed);
    for j in 0..valid.falling_coefficients.len() {
        let mut changed = valid.clone();
        changed.falling_coefficients[j] += q(1, 1_000_000_007);
        corruptions.push(changed);
    }
    let mut changed = valid.clone();
    changed.source = Expr::Add(vec![changed.source, e(0)]);
    corruptions.push(changed);
    let mut changed = valid.clone();
    changed.index = Symbol::with_identity("k", SymbolIdentity::from_assumptions_digest([5; 32]));
    corruptions.push(changed);
    let mut changed = valid.clone();
    changed.upper = e(8);
    corruptions.push(changed);
    for mut changed in corruptions {
        assert!(matches!(
            verify_polynomial_binomial_sum(
                &summand,
                &k,
                &e(0),
                &e(7),
                changed.clone(),
                &mut Unbounded
            ),
            Err(BinomialSummationError::InvalidCertificate)
        ));
        changed.falling_coefficients.clear();
        assert!(
            verify_polynomial_binomial_sum(&summand, &k, &e(0), &e(7), changed, &mut Unbounded)
                .is_err()
        );
    }
    assert!(
        verify_polynomial_binomial_sum(&summand, &k, &e(1), &e(7), valid.clone(), &mut Unbounded)
            .is_err()
    );
    assert!(
        verify_polynomial_binomial_sum(
            &binomial(e(7), &k),
            &k,
            &e(0),
            &e(7),
            valid.clone(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn held_zero_cannot_erase_original_poles_degree_or_nonfinite_children() {
    let k = Symbol::new("k");
    for invalid in [
        power(e(0), e(-1)),
        power(Expr::Add(vec![e(1), e(-1)]), e(-1)),
        power(Expr::Sym(k.clone()), e(33)),
        Expr::Const(Constant::Infinity),
    ] {
        for erased in [Expr::Mul(vec![e(0), invalid.clone()]), power(invalid, e(0))] {
            let summand = Expr::Mul(vec![erased, binomial(e(3000), &k)]);
            assert!(checked_summation(&summand, &k, &e(0), &e(3000)).is_err());
        }
    }
}

#[test]
fn unsupported_symbolic_bounds_weights_and_typed_indices_remain_unaccepted() {
    let k = Symbol::with_identity("k", SymbolIdentity::from_assumptions_digest([11; 32]));
    let other = Symbol::with_identity("k", SymbolIdentity::from_assumptions_digest([12; 32]));
    let n = Expr::Sym(Symbol::with_identity(
        "n",
        SymbolIdentity::from_assumptions_digest([13; 32]),
    ));
    assert!(
        checked_summation(&binomial(n.clone(), &k), &k, &e(0), &n)
            .unwrap()
            .is_none()
    );
    assert!(
        polynomial_binomial_sum(&binomial(e(5), &other), &k, &e(0), &e(5))
            .unwrap()
            .is_none()
    );
    let constant = checked_summation(&binomial(e(5), &other), &k, &e(0), &e(5))
        .unwrap()
        .unwrap();
    assert!(constant.free_symbols().contains(&other));
    assert!(!constant.free_symbols().contains(&k));
    let unadmitted = power(
        Expr::Function("sin".into(), vec![Expr::Sym(k.clone())]),
        e(0),
    );
    let summand = Expr::Mul(vec![unadmitted, binomial(e(5), &k)]);
    assert!(
        checked_summation(&summand, &k, &e(0), &e(5))
            .unwrap()
            .is_none()
    );
    assert!(
        polynomial_binomial_sum(&binomial(e(5), &k), &k, &e(1), &e(5))
            .unwrap()
            .is_none()
    );
    assert!(polynomial_binomial_sum(&binomial(e(65_537), &k), &k, &e(0), &e(65_537)).is_err());
    let huge = Expr::Integer(BigInt::from(1) << 4096_u32);
    assert!(
        polynomial_binomial_sum(&Expr::Mul(vec![huge, binomial(e(5), &k)]), &k, &e(0), &e(5))
            .is_err()
    );
}

#[test]
fn short_partial_ranges_and_binomial_powers_keep_existing_exact_results() {
    let k = Symbol::new("k");
    let summand = binomial(e(5), &k);
    assert_eq!(
        checked_summation(&summand, &k, &e(1), &e(5)).unwrap(),
        Some(e(31))
    );
    assert_eq!(
        checked_summation(&summand, &k, &e(0), &e(3)).unwrap(),
        Some(e(26))
    );
    let squared = power(summand, e(2));
    assert!(
        polynomial_binomial_sum(&squared, &k, &e(0), &e(5))
            .unwrap()
            .is_none()
    );
    assert_eq!(
        checked_summation(&squared, &k, &e(0), &e(5)).unwrap(),
        Some(e(252))
    );
}

struct Stop {
    charges: usize,
    stop_after: usize,
}

#[test]
fn verifier_cannot_be_fooled_by_extra_degree_or_a_nonlinear_exponent() {
    let k = Symbol::new("k");
    let summand = binomial(e(5), &k);
    let valid = polynomial_binomial_sum(&summand, &k, &e(0), &e(5))
        .unwrap()
        .unwrap();
    // (k)_32 vanishes at 0..31, and even over the whole finite range0..5.
    // It still cannot be certified as the original constant weight.
    let mut extra_degree = valid.certificate().clone();
    extra_degree.falling_coefficients.resize(33, q(0, 1));
    extra_degree.falling_coefficients[32] = q(1, 1);
    assert!(matches!(
        verify_polynomial_binomial_sum(&summand, &k, &e(0), &e(5), extra_degree, &mut Unbounded),
        Err(BinomialSummationError::InvalidCertificate)
    ));
    // k+k*(k-1) agrees with k at zero and one but is quadratic. A degree
    // proof and a third exact evaluation must reject its exponential role.
    let ke = Expr::Sym(k.clone());
    let exponent = Expr::Add(vec![
        ke.clone(),
        Expr::Mul(vec![ke.clone(), Expr::Add(vec![ke, e(-1)])]),
    ]);
    let nonlinear = Expr::Mul(vec![summand.clone(), power(e(2), exponent)]);
    let forged = PolynomialBinomialCertificate {
        source: nonlinear.clone(),
        index: k.clone(),
        lower: e(0),
        upper: e(5),
        falling_coefficients: vec![q(1, 1)],
        claimed_sum: q(243, 1),
    };
    assert!(
        verify_polynomial_binomial_sum(&nonlinear, &k, &e(0), &e(5), forged, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn traversal_stack_capacity_is_charged_before_wide_source_admission() {
    let k = Symbol::new("k");
    let mut factors = vec![e(1); 3990];
    factors.push(binomial(e(5), &k));
    let summand = Expr::Mul(factors);
    let mut limits = BudgetLimits::uniform(1_000_000, 0);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024;
    let mut meter = Budget::new(limits);
    assert!(matches!(
        metered_polynomial_binomial_sum(&summand, &k, &e(0), &e(5), &mut meter),
        Err(BinomialSummationError::Meter(MeterError::Budget(
            fsym_budget::BudgetError::Exhausted {
                dimension: Dimension::MemoryBytes,
                ..
            }
        )))
    ));
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
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
}

#[test]
fn generation_checking_and_publication_share_terminal_cancellation_and_budget() {
    let k = Symbol::new("k");
    let summand = source(&[q(3, 1), q(-2, 1), q(7, 1)], &q(2, 1), &q(3, 1), 12, &k);
    let mut complete = Stop {
        charges: 0,
        stop_after: usize::MAX,
    };
    assert!(
        metered_polynomial_binomial_sum(&summand, &k, &e(0), &e(12), &mut complete)
            .unwrap()
            .is_some()
    );
    for stop_after in [
        0,
        10,
        complete.charges / 2,
        complete.charges - 1,
        complete.charges,
    ] {
        let mut meter = Stop {
            charges: 0,
            stop_after,
        };
        assert!(matches!(
            metered_polynomial_binomial_sum(&summand, &k, &e(0), &e(12), &mut meter),
            Err(BinomialSummationError::Meter(MeterError::Cancelled))
        ));
    }
    let mut exhausted = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        metered_polynomial_binomial_sum(&summand, &k, &e(0), &e(12), &mut exhausted),
        Err(BinomialSummationError::Meter(MeterError::Budget(_)))
    ));
}
