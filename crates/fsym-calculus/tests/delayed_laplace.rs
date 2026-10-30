//! Exact causal delay identities and independent source-coverage rejection.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::delayed_laplace::{
    DelayedLaplaceCertificate, DelayedLaplaceTermCertificate, inverse_delayed_laplace,
    metered_inverse_delayed_laplace, verify_delayed_laplace_certificate,
};
use fsym_calculus::rational_laplace::{
    LaplaceBlock, RationalLaplaceCertificate, RationalLaplaceError,
};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn analytic(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}
fn shifted(time: &Symbol, delay: i64) -> Expr {
    Expr::Add(vec![Expr::Sym(time.clone()), Expr::from_i64(-delay)])
}
fn delayed(source: Expr, variable: &Symbol, delay: BigRational) -> Expr {
    Expr::Mul(vec![
        analytic(
            "exp",
            Expr::Mul(vec![rational_expr(-delay), Expr::Sym(variable.clone())]),
        ),
        source,
    ])
}
fn quadratic(variable: &Symbol) -> Expr {
    Expr::Add(vec![
        power(Expr::Sym(variable.clone()), 2),
        Expr::from_i64(1),
    ])
}
fn repeated(variable: &Symbol, delay: BigRational) -> Expr {
    delayed(power(quadratic(variable), -2), variable, delay)
}
fn exponential(rate: i64, coefficients: Vec<BigRational>) -> RationalLaplaceCertificate {
    RationalLaplaceCertificate {
        blocks: vec![LaplaceBlock::Exponential {
            rate: q(rate, 1),
            coefficients,
        }],
    }
}
fn quadratic_certificate() -> RationalLaplaceCertificate {
    RationalLaplaceCertificate {
        blocks: vec![LaplaceBlock::Quadratic {
            rate: q(0, 1),
            squared_frequency: q(1, 1),
            cosine: vec![q(0, 1), q(-1, 2)],
            sine: vec![q(1, 2)],
        }],
    }
}
fn certificate(
    delay: BigRational,
    rational: RationalLaplaceCertificate,
) -> DelayedLaplaceCertificate {
    DelayedLaplaceCertificate {
        terms: vec![DelayedLaplaceTermCertificate { delay, rational }],
    }
}

#[test]
fn repeated_quadratic_inverse_is_shifted_by_the_exact_causal_delay() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = repeated(&s, q(1, 1));
    // Independently supplied classical transform identity:
    // L[(sin(t)-t*cos(t))/2] = 1/(s*s+1)^2.
    let expected = certificate(q(1, 1), quadratic_certificate());
    let reference =
        verify_delayed_laplace_certificate(&source, &s, &t, expected.clone(), &mut Unbounded)
            .unwrap();
    let result = inverse_delayed_laplace(&source, &s, &t).unwrap().unwrap();
    assert_eq!(result.source(), &source);
    assert_eq!(result.certificate(), &expected);
    assert_eq!(result.parts().len(), 1);
    assert_eq!(result.parts()[0].delay(), &q(1, 1));
    let u = shifted(&t, 1);
    let expected_function = Expr::Add(vec![
        Expr::Mul(vec![
            rational_expr(q(-1, 2)),
            u.clone(),
            analytic("cos", u.clone()),
        ]),
        Expr::Mul(vec![rational_expr(q(1, 2)), analytic("sin", u)]),
    ]);
    assert_eq!(
        fsym_simplify::simplify(result.parts()[0].shifted_function()),
        fsym_simplify::simplify(&expected_function),
    );
    assert_eq!(
        result.parts()[0].shifted_function(),
        reference.parts()[0].shifted_function()
    );
    assert_eq!(
        result.parts()[0].shifted_function().free_symbols(),
        vec![t.clone()]
    );
    assert_eq!(result.transform_variable(), &s);
    assert_eq!(result.time_variable(), &t);
}

#[test]
fn rational_delays_mixed_poles_and_zero_delay_keep_complete_coverage() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = Expr::Add(vec![
        power(Expr::Sym(s.clone()), -1),
        delayed(
            power(
                Expr::Add(vec![Expr::Sym(s.clone()), Expr::from_i64(-2)]),
                -3,
            ),
            &s,
            q(3, 2),
        ),
        repeated(&s, q(2, 1)),
    ]);
    let expected = DelayedLaplaceCertificate {
        terms: vec![
            DelayedLaplaceTermCertificate {
                delay: q(0, 1),
                rational: exponential(0, vec![q(1, 1)]),
            },
            DelayedLaplaceTermCertificate {
                delay: q(3, 2),
                rational: exponential(2, vec![q(0, 1), q(0, 1), q(1, 2)]),
            },
            DelayedLaplaceTermCertificate {
                delay: q(2, 1),
                rational: quadratic_certificate(),
            },
        ],
    };
    let result = inverse_delayed_laplace(&source, &s, &t).unwrap().unwrap();
    assert_eq!(result.certificate(), &expected);
    let checked =
        verify_delayed_laplace_certificate(&source, &s, &t, expected, &mut Unbounded).unwrap();
    assert_eq!(checked.parts().len(), 3);
    assert!(checked.sufficient_abscissa() > &q(2, 1));
    for part in checked.parts() {
        assert!(
            part.shifted_function()
                .free_symbols()
                .iter()
                .all(|symbol| symbol == &t)
        );
    }
}

#[test]
fn equal_delays_group_before_polynomial_parts_are_rejected() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let ratio = Expr::Mul(vec![
        Expr::Sym(s.clone()),
        power(Expr::Add(vec![Expr::Sym(s.clone()), Expr::from_i64(1)]), -1),
    ]);
    let sources = [
        delayed(
            Expr::Add(vec![ratio.clone(), Expr::from_i64(-1)]),
            &s,
            q(1, 1),
        ),
        Expr::Add(vec![
            delayed(ratio, &s, q(1, 1)),
            delayed(Expr::from_i64(-1), &s, q(1, 1)),
        ]),
    ];
    for source in sources {
        let result = inverse_delayed_laplace(&source, &s, &t).unwrap().unwrap();
        assert_eq!(
            result.certificate(),
            &certificate(q(1, 1), exponential(-1, vec![q(-1, 1)]))
        );
        assert_eq!(result.parts().len(), 1);
    }
}

#[test]
fn product_distribution_and_combined_exponentials_are_complete() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = Expr::Mul(vec![
        Expr::Add(vec![
            delayed(Expr::from_i64(1), &s, q(1, 1)),
            delayed(Expr::from_i64(1), &s, q(2, 1)),
        ]),
        Expr::Add(vec![
            delayed(Expr::from_i64(1), &s, q(0, 1)),
            delayed(Expr::from_i64(1), &s, q(1, 1)),
        ]),
        power(Expr::Sym(s.clone()), -1),
    ]);
    let result = inverse_delayed_laplace(&source, &s, &t).unwrap().unwrap();
    assert_eq!(
        result
            .certificate()
            .terms
            .iter()
            .map(|part| part.delay.clone())
            .collect::<Vec<_>>(),
        vec![q(1, 1), q(2, 1), q(3, 1)]
    );
    assert_eq!(
        result.certificate().terms[0].rational,
        exponential(0, vec![q(1, 1)])
    );
    assert_eq!(
        result.certificate().terms[1].rational,
        exponential(0, vec![q(2, 1)])
    );
    assert_eq!(
        result.certificate().terms[2].rational,
        exponential(0, vec![q(1, 1)])
    );
    let net = Expr::Mul(vec![
        analytic("exp", Expr::Sym(s.clone())),
        delayed(power(Expr::Sym(s.clone()), -1), &s, q(2, 1)),
    ]);
    let net = inverse_delayed_laplace(&net, &s, &t).unwrap().unwrap();
    assert_eq!(
        net.certificate(),
        &certificate(q(1, 1), exponential(0, vec![q(1, 1)]))
    );
}

#[test]
fn every_original_rational_guard_survives_delay_grouping_and_zero() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let pole = Expr::Add(vec![Expr::Sym(s.clone()), Expr::from_i64(-100)]);
    let held_one = power(power(pole, -1), 0);
    for rational in [
        Expr::Mul(vec![held_one.clone(), power(Expr::Sym(s.clone()), -1)]),
        Expr::Mul(vec![Expr::from_i64(0), held_one]),
    ] {
        let source = delayed(rational, &s, q(1, 1));
        let result = inverse_delayed_laplace(&source, &s, &t).unwrap().unwrap();
        assert!(result.sufficient_abscissa() > &q(100, 1));
        let checked = verify_delayed_laplace_certificate(
            &source,
            &s,
            &t,
            result.certificate().clone(),
            &mut Unbounded,
        )
        .unwrap();
        assert!(checked.sufficient_abscissa() > &q(100, 1));
    }
    let zero = Expr::Add(vec![
        Expr::Sym(s.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(s.clone())]),
    ]);
    for exponent in [-1, 0] {
        let invalid = Expr::Mul(vec![
            Expr::from_i64(0),
            power(power(zero.clone(), -1), exponent),
        ]);
        let source = delayed(invalid, &s, q(1, 1));
        assert!(matches!(
            inverse_delayed_laplace(&source, &s, &t),
            Err(RationalLaplaceError::InvalidSource(_))
        ));
    }
}

#[test]
fn reference_rejects_incomplete_reordered_mutated_and_rebound_claims() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = Expr::Add(vec![
        repeated(&s, q(1, 1)),
        delayed(power(Expr::Sym(s.clone()), -1), &s, q(2, 1)),
    ]);
    let valid = DelayedLaplaceCertificate {
        terms: vec![
            DelayedLaplaceTermCertificate {
                delay: q(1, 1),
                rational: quadratic_certificate(),
            },
            DelayedLaplaceTermCertificate {
                delay: q(2, 1),
                rational: exponential(0, vec![q(1, 1)]),
            },
        ],
    };
    verify_delayed_laplace_certificate(&source, &s, &t, valid.clone(), &mut Unbounded).unwrap();
    let mut bad = Vec::new();
    let mut omitted = valid.clone();
    omitted.terms.pop();
    bad.push(omitted);
    let mut duplicate = valid.clone();
    duplicate.terms.push(duplicate.terms[0].clone());
    bad.push(duplicate);
    let mut reordered = valid.clone();
    reordered.terms.reverse();
    bad.push(reordered);
    let mut wrong_delay = valid.clone();
    wrong_delay.terms[0].delay = q(3, 2);
    bad.push(wrong_delay);
    let mut negative_delay = valid.clone();
    negative_delay.terms[0].delay = q(-1, 1);
    bad.push(negative_delay);
    let mut missing_pole = valid.clone();
    missing_pole.terms[0].rational.blocks.clear();
    bad.push(missing_pole);
    let mut wrong_coefficient = valid.clone();
    wrong_coefficient.terms[1].rational = exponential(0, vec![q(2, 1)]);
    bad.push(wrong_coefficient);
    for candidate in bad {
        assert!(
            verify_delayed_laplace_certificate(&source, &s, &t, candidate, &mut Unbounded).is_err()
        );
    }
    let changed_source = Expr::Mul(vec![Expr::from_i64(2), source.clone()]);
    assert!(
        verify_delayed_laplace_certificate(&changed_source, &s, &t, valid.clone(), &mut Unbounded)
            .is_err()
    );
    let rebound = Symbol::with_identity("s", SymbolIdentity::from_assumptions_digest([9; 32]));
    assert!(
        verify_delayed_laplace_certificate(&source, &rebound, &t, valid, &mut Unbounded).is_err()
    );
}

#[test]
fn typed_same_printing_variables_remain_distinct_through_the_shift() {
    let s = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([7; 32]));
    let t = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([8; 32]));
    let source = repeated(&s, q(1, 1));
    let result = inverse_delayed_laplace(&source, &s, &t).unwrap().unwrap();
    assert_eq!(
        result.parts()[0].shifted_function().free_symbols(),
        vec![t.clone()]
    );
    assert!(matches!(
        inverse_delayed_laplace(&source, &s, &s),
        Err(RationalLaplaceError::InvalidSource(_))
    ));
    let plain = Symbol::new("same");
    assert!(
        inverse_delayed_laplace(&source, &plain, &t)
            .unwrap()
            .is_none()
    );
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    charged: [u64; 3],
}
impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
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
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn negative_unknown_nonaffine_and_large_delays_miss_before_rational_inversion() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let a = Symbol::new("a");
    let cases = [
        delayed(power(Expr::Sym(s.clone()), -1), &s, q(-1, 1)),
        Expr::Mul(vec![
            analytic("exp", Expr::Mul(vec![Expr::Sym(a), Expr::Sym(s.clone())])),
            power(Expr::Sym(s.clone()), -1),
        ]),
        Expr::Mul(vec![
            analytic("exp", power(Expr::Sym(s.clone()), 2)),
            power(Expr::Sym(s.clone()), -1),
        ]),
        delayed(power(Expr::Sym(s.clone()), -33), &s, q(1, 1)),
        Expr::Add(vec![
            repeated(&s, q(1, 1)),
            analytic("unhandled", Expr::Sym(s.clone())),
        ]),
    ];
    for source in cases {
        let mut generator = Meter::default();
        let mut verifier = Meter::default();
        assert!(
            metered_inverse_delayed_laplace(&source, &s, &t, &mut generator, &mut verifier)
                .unwrap()
                .is_none()
        );
        assert_eq!(verifier.charged, [0; 3]);
        assert_eq!(verifier.checkpoints, 0);
        assert!(
            generator.charged[0] < 10_000,
            "recognition must precede factorization"
        );
    }
}

#[test]
fn held_exponent_domains_cannot_be_erased_by_affine_normalization() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let pole = Expr::Add(vec![Expr::Sym(s.clone()), Expr::from_i64(-100)]);
    // This would become exp(0)/s if lowering silently removed (s-100)^0.
    let source = Expr::Mul(vec![
        analytic("exp", Expr::Add(vec![power(pole, 0), Expr::from_i64(-1)])),
        power(Expr::Sym(s.clone()), -1),
    ]);
    assert!(inverse_delayed_laplace(&source, &s, &t).unwrap().is_none());
    assert!(
        verify_delayed_laplace_certificate(
            &source,
            &s,
            &t,
            certificate(q(0, 1), exponential(0, vec![q(1, 1)])),
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn cumulative_generator_and_verifier_budgets_are_terminal() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = Expr::Add(vec![repeated(&s, q(1, 1)), repeated(&s, q(2, 1))]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_delayed_laplace(&source, &s, &t, &mut generator, &mut verifier)
        .unwrap()
        .unwrap();
    for (dimension, index) in [
        (Dimension::ComputeSteps, 0),
        (Dimension::MemoryBytes, 1),
        (Dimension::AllocationCount, 2),
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = generator.charged[index] - 1;
        let mut exhausted = Budget::new(limits);
        let mut unused = Meter::default();
        assert!(matches!(
            metered_inverse_delayed_laplace(&source, &s, &t, &mut exhausted, &mut unused),
            Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
        ));
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = verifier.charged[index] - 1;
        let mut exhausted = Budget::new(limits);
        assert!(matches!(
            metered_inverse_delayed_laplace(&source, &s, &t, &mut Unbounded, &mut exhausted),
            Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
        ));
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut unused = Meter::default();
    assert!(matches!(
        metered_inverse_delayed_laplace(&source, &s, &t, &mut tiny, &mut unused),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(unused.charged, [0; 3]);
}

#[test]
fn cancellation_reaches_each_lane_including_final_publication() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = Expr::Add(vec![repeated(&s, q(1, 1)), repeated(&s, q(2, 1))]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_delayed_laplace(&source, &s, &t, &mut generator, &mut verifier)
        .unwrap()
        .unwrap();
    for stop in [
        1,
        generator.checkpoints / 2,
        generator.checkpoints - 1,
        generator.checkpoints,
    ] {
        let mut interrupted = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert!(
            matches!(
                metered_inverse_delayed_laplace(&source, &s, &t, &mut interrupted, &mut Unbounded),
                Err(RationalLaplaceError::Meter(MeterError::Cancelled))
            ),
            "generator checkpoint {stop}"
        );
    }
    for stop in [
        1,
        verifier.checkpoints / 2,
        verifier.checkpoints - 1,
        verifier.checkpoints,
    ] {
        let mut interrupted = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert!(
            matches!(
                metered_inverse_delayed_laplace(&source, &s, &t, &mut Unbounded, &mut interrupted),
                Err(RationalLaplaceError::Meter(MeterError::Cancelled))
            ),
            "verifier checkpoint {stop}"
        );
    }
}

#[test]
fn term_expansion_and_shift_growth_have_intrinsic_limits_even_with_unbounded_meters() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let too_many = Expr::Add(
        (0..33)
            .map(|delay| delayed(power(Expr::Sym(s.clone()), -1), &s, q(delay, 1)))
            .collect(),
    );
    assert!(matches!(
        inverse_delayed_laplace(&too_many, &s, &t),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let delay = BigRational::from_integer(BigInt::one() << 2048u32);
    let source = delayed(power(Expr::Sym(s.clone()), -2), &s, delay);
    assert!(matches!(
        inverse_delayed_laplace(&source, &s, &t),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let source = delayed(power(Expr::Sym(s.clone()), -1), &s, q(1, 1));
    let oversized = DelayedLaplaceCertificate {
        terms: vec![
            DelayedLaplaceTermCertificate {
                delay: q(1, 1),
                rational: exponential(0, vec![q(1, 1)])
            };
            33
        ],
    };
    assert!(matches!(
        verify_delayed_laplace_certificate(&source, &s, &t, oversized, &mut Unbounded),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
}
#[test]
fn an_early_distributional_term_cannot_hide_a_later_invalid_domain() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let zero = Expr::Add(vec![
        Expr::Sym(s.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(s.clone())]),
    ]);
    let source = Expr::Add(vec![
        delayed(Expr::from_i64(1), &s, q(1, 1)),
        delayed(
            Expr::Mul(vec![Expr::from_i64(0), power(zero, -1)]),
            &s,
            q(2, 1),
        ),
    ]);
    assert!(matches!(
        inverse_delayed_laplace(&source, &s, &t),
        Err(RationalLaplaceError::InvalidSource(_))
    ));
}
