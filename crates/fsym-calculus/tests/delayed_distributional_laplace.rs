//! Complete shifted impulse/regular identities and adversarial coverage.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::delayed_laplace::distributional::{
    DelayedDistributionalLaplaceCertificate, DelayedDistributionalLaplaceTermCertificate,
    inverse_delayed_distributional_laplace, metered_inverse_delayed_distributional_laplace,
    verify_delayed_distributional_laplace_certificate,
};
use fsym_calculus::delayed_laplace::inverse_delayed_laplace;
use fsym_calculus::rational_laplace::distributional::DistributionalLaplaceCertificate;
use fsym_calculus::rational_laplace::{
    LaplaceBlock, RationalLaplaceCertificate, RationalLaplaceError,
};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_polys::UnivariatePoly;
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
fn polynomial(values: &[i64], variable: &Symbol) -> Expr {
    UnivariatePoly::new(variable.clone(), values.iter().map(|&n| q(n, 1)).collect()).to_expr()
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
fn displacement(time: &Symbol, delay: BigRational) -> Expr {
    Expr::Add(vec![Expr::Sym(time.clone()), rational_expr(-delay)])
}
fn delta(argument: Expr, order: i64) -> Expr {
    let mut arguments = vec![argument];
    if order != 0 {
        arguments.push(Expr::from_i64(order));
    }
    Expr::Function("DiracDelta".into(), arguments)
}
fn term(
    delay: BigRational,
    impulses: Vec<BigRational>,
    blocks: Vec<LaplaceBlock>,
) -> DelayedDistributionalLaplaceTermCertificate {
    DelayedDistributionalLaplaceTermCertificate {
        delay,
        distributional: DistributionalLaplaceCertificate {
            impulse_coefficients: impulses,
            regular: RationalLaplaceCertificate { blocks },
        },
    }
}
fn exponential(rate: i64, coefficients: Vec<BigRational>) -> LaplaceBlock {
    LaplaceBlock::Exponential {
        rate: q(rate, 1),
        coefficients,
    }
}
fn improper(variable: &Symbol) -> Expr {
    Expr::Mul(vec![
        power(Expr::Sym(variable.clone()), 2),
        power(polynomial(&[1, 1], variable), -1),
    ])
}
fn mixed_source(variable: &Symbol) -> Expr {
    Expr::Add(vec![
        Expr::Sym(variable.clone()),
        delayed(improper(variable), variable, q(1, 1)),
        delayed(
            power(polynomial(&[1, 0, 1], variable), -2),
            variable,
            q(2, 1),
        ),
    ])
}
fn mixed_certificate() -> DelayedDistributionalLaplaceCertificate {
    // s + exp(-s)*(s - 1 + 1/(s+1)) + exp(-2s)/(s*s+1)^2.
    DelayedDistributionalLaplaceCertificate {
        terms: vec![
            term(q(0, 1), vec![q(0, 1), q(1, 1)], vec![]),
            term(
                q(1, 1),
                vec![q(-1, 1), q(1, 1)],
                vec![exponential(-1, vec![q(1, 1)])],
            ),
            term(
                q(2, 1),
                vec![q(0, 1)],
                vec![LaplaceBlock::Quadratic {
                    rate: q(0, 1),
                    squared_frequency: q(1, 1),
                    cosine: vec![q(0, 1), q(-1, 2)],
                    sine: vec![q(1, 2)],
                }],
            ),
        ],
    }
}

#[test]
fn every_bounded_derivative_is_shifted_without_changing_its_order() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    for order in 0..=32 {
        let base = if order == 0 {
            Expr::from_i64(1)
        } else {
            power(Expr::Sym(s.clone()), order)
        };
        let source = delayed(base, &s, q(3, 2));
        let mut coefficients = vec![q(0, 1); order as usize + 1];
        coefficients[order as usize] = q(1, 1);
        let expected = DelayedDistributionalLaplaceCertificate {
            terms: vec![term(q(3, 2), coefficients, vec![])],
        };
        // Direct unilateral shift/derivative identity, independent of division.
        let checked = verify_delayed_distributional_laplace_certificate(
            &source,
            &s,
            &t,
            expected.clone(),
            &mut Unbounded,
        )
        .unwrap();
        let generated = inverse_delayed_distributional_laplace(&source, &s, &t)
            .unwrap()
            .unwrap();
        assert_eq!(generated.certificate(), &expected);
        assert_eq!(
            checked.parts()[0].shifted_regular_function(),
            &Expr::from_i64(0)
        );
        assert_eq!(
            checked.parts()[0].shifted_impulses(),
            &delta(displacement(&t, q(3, 2)), order)
        );
        assert_eq!(
            generated.parts()[0].shifted_impulses(),
            checked.parts()[0].shifted_impulses()
        );
        assert_eq!(checked.source(), &source);
        assert_eq!(checked.transform_variable(), &s);
        assert_eq!(checked.time_variable(), &t);
        // The established ordinary-only API keeps its existing promise.
        assert!(matches!(
            inverse_delayed_laplace(&source, &s, &t),
            Err(RationalLaplaceError::Unsupported(_))
        ));
    }
}

#[test]
fn mixed_improper_repeated_and_undelayed_terms_keep_both_components() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = mixed_source(&s);
    let expected = mixed_certificate();
    let checked = verify_delayed_distributional_laplace_certificate(
        &source,
        &s,
        &t,
        expected.clone(),
        &mut Unbounded,
    )
    .unwrap();
    let result = inverse_delayed_distributional_laplace(&source, &s, &t)
        .unwrap()
        .unwrap();
    assert_eq!(result.certificate(), &expected);
    assert_eq!(result.parts().len(), 3);
    assert_eq!(
        result.parts()[0].shifted_impulses(),
        &delta(Expr::Sym(t.clone()), 1)
    );
    let u = displacement(&t, q(1, 1));
    let impulses = Expr::Add(vec![
        Expr::Mul(vec![Expr::from_i64(-1), delta(u.clone(), 0)]),
        delta(u.clone(), 1),
    ]);
    assert_eq!(result.parts()[1].shifted_impulses(), &impulses);
    assert_eq!(
        fsym_simplify::simplify(result.parts()[1].shifted_regular_function()),
        fsym_simplify::simplify(&analytic("exp", Expr::Mul(vec![Expr::from_i64(-1), u]))),
    );
    let v = displacement(&t, q(2, 1));
    let regular = Expr::Add(vec![
        Expr::Mul(vec![
            rational_expr(q(-1, 2)),
            v.clone(),
            analytic("cos", v.clone()),
        ]),
        Expr::Mul(vec![rational_expr(q(1, 2)), analytic("sin", v)]),
    ]);
    assert_eq!(
        fsym_simplify::simplify(result.parts()[2].shifted_regular_function()),
        fsym_simplify::simplify(&regular)
    );
    assert_eq!(result.parts()[2].shifted_impulses(), &Expr::from_i64(0));
    for (generated, reference) in result.parts().iter().zip(checked.parts()) {
        assert_eq!(generated.delay(), reference.delay());
        assert_eq!(
            generated.shifted_regular_function(),
            reference.shifted_regular_function()
        );
        assert_eq!(generated.shifted_impulses(), reference.shifted_impulses());
    }
}

#[test]
fn products_and_equal_delays_combine_all_polynomial_coefficients() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = Expr::Mul(vec![
        Expr::Add(vec![
            delayed(Expr::from_i64(1), &s, q(1, 1)),
            delayed(Expr::from_i64(1), &s, q(2, 1)),
        ]),
        Expr::Add(vec![
            Expr::from_i64(1),
            delayed(Expr::from_i64(1), &s, q(1, 1)),
        ]),
        Expr::Sym(s.clone()),
    ]);
    let expected = DelayedDistributionalLaplaceCertificate {
        terms: vec![
            term(q(1, 1), vec![q(0, 1), q(1, 1)], vec![]),
            term(q(2, 1), vec![q(0, 1), q(2, 1)], vec![]),
            term(q(3, 1), vec![q(0, 1), q(1, 1)], vec![]),
        ],
    };
    assert_eq!(
        inverse_delayed_distributional_laplace(&source, &s, &t)
            .unwrap()
            .unwrap()
            .certificate(),
        &expected
    );
    verify_delayed_distributional_laplace_certificate(&source, &s, &t, expected, &mut Unbounded)
        .unwrap();
    let cancellation = Expr::Add(vec![
        delayed(
            Expr::Mul(vec![
                Expr::Sym(s.clone()),
                power(polynomial(&[1, 1], &s), -1),
            ]),
            &s,
            q(1, 1),
        ),
        delayed(Expr::from_i64(-1), &s, q(1, 1)),
    ]);
    let result = inverse_delayed_distributional_laplace(&cancellation, &s, &t)
        .unwrap()
        .unwrap();
    assert_eq!(
        result.certificate().terms,
        vec![term(
            q(1, 1),
            vec![q(0, 1)],
            vec![exponential(-1, vec![q(-1, 1)])]
        )]
    );
    assert_eq!(result.parts()[0].shifted_impulses(), &Expr::from_i64(0));
    let net_delay = Expr::Mul(vec![
        analytic("exp", Expr::Sym(s.clone())),
        delayed(Expr::Sym(s.clone()), &s, q(2, 1)),
    ]);
    assert_eq!(
        inverse_delayed_distributional_laplace(&net_delay, &s, &t)
            .unwrap()
            .unwrap()
            .certificate()
            .terms,
        vec![term(q(1, 1), vec![q(0, 1), q(1, 1)], vec![])]
    );
}

#[test]
fn reference_rejects_omissions_delay_changes_impulse_changes_and_rebound_sources() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = mixed_source(&s);
    let good = mixed_certificate();
    let mut mutants = Vec::new();
    let mut missing = good.clone();
    missing.terms.pop();
    mutants.push(missing);
    let mut duplicate = good.clone();
    duplicate.terms.push(duplicate.terms[0].clone());
    mutants.push(duplicate);
    let mut reversed = good.clone();
    reversed.terms.reverse();
    mutants.push(reversed);
    let mut wrong_delay = good.clone();
    wrong_delay.terms[1].delay = q(3, 2);
    mutants.push(wrong_delay);
    let mut negative = good.clone();
    negative.terms[1].delay = q(-1, 1);
    mutants.push(negative);
    let mut missing_impulse = good.clone();
    missing_impulse.terms[1].distributional.impulse_coefficients = vec![q(0, 1)];
    mutants.push(missing_impulse);
    let mut wrong_order = good.clone();
    wrong_order.terms[1].distributional.impulse_coefficients = vec![q(-1, 1), q(0, 1), q(1, 1)];
    mutants.push(wrong_order);
    let mut tiny_change = good.clone();
    tiny_change.terms[1].distributional.impulse_coefficients[0] +=
        BigRational::new(1.into(), BigInt::from(10).pow(80));
    mutants.push(tiny_change);
    let mut missing_regular = good.clone();
    missing_regular.terms[1]
        .distributional
        .regular
        .blocks
        .clear();
    mutants.push(missing_regular);
    let mut empty_polynomial = good.clone();
    empty_polynomial.terms[0]
        .distributional
        .impulse_coefficients
        .clear();
    mutants.push(empty_polynomial);
    let mut untrimmed = good.clone();
    untrimmed.terms[0]
        .distributional
        .impulse_coefficients
        .push(q(0, 1));
    mutants.push(untrimmed);
    for candidate in mutants {
        assert!(
            verify_delayed_distributional_laplace_certificate(
                &source,
                &s,
                &t,
                candidate,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let changed = Expr::Add(vec![source.clone(), Expr::from_i64(1)]);
    assert!(
        verify_delayed_distributional_laplace_certificate(
            &changed,
            &s,
            &t,
            good.clone(),
            &mut Unbounded
        )
        .is_err()
    );
    let rebound = Symbol::with_identity("s", SymbolIdentity::from_assumptions_digest([9; 32]));
    assert!(
        verify_delayed_distributional_laplace_certificate(
            &source,
            &rebound,
            &t,
            good,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn typed_time_identity_is_retained_in_regular_and_impulse_parts() {
    let s = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([7; 32]));
    let t = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([8; 32]));
    let source = delayed(improper(&s), &s, q(1, 1));
    let result = inverse_delayed_distributional_laplace(&source, &s, &t)
        .unwrap()
        .unwrap();
    assert_eq!(
        result.parts()[0].shifted_impulses().free_symbols(),
        vec![t.clone()]
    );
    assert_eq!(
        result.parts()[0].shifted_regular_function().free_symbols(),
        vec![t.clone()]
    );
    assert!(matches!(
        inverse_delayed_distributional_laplace(&source, &s, &s),
        Err(RationalLaplaceError::InvalidSource(_))
    ));
    assert!(
        inverse_delayed_distributional_laplace(&source, &Symbol::new("same"), &t)
            .unwrap()
            .is_none()
    );
}

#[test]
fn original_exclusions_survive_polynomial_cancellation_and_zero() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let held_one = power(power(polynomial(&[-100, 1], &s), -1), 0);
    for factor in [Expr::Sym(s.clone()), Expr::from_i64(0)] {
        let source = delayed(Expr::Mul(vec![factor, held_one.clone()]), &s, q(1, 1));
        let result = inverse_delayed_distributional_laplace(&source, &s, &t)
            .unwrap()
            .unwrap();
        assert!(result.sufficient_abscissa() > &q(100, 1));
        let reference = verify_delayed_distributional_laplace_certificate(
            &source,
            &s,
            &t,
            result.certificate().clone(),
            &mut Unbounded,
        )
        .unwrap();
        assert!(reference.sufficient_abscissa() > &q(100, 1));
    }
    let zero = Expr::Add(vec![
        Expr::Sym(s.clone()),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(s.clone())]),
    ]);
    for exponent in [-1, 0] {
        let invalid_tail = delayed(
            Expr::Mul(vec![
                Expr::from_i64(0),
                power(power(zero.clone(), -1), exponent),
            ]),
            &s,
            q(2, 1),
        );
        // The unsupported cubic tail in the first group cannot take precedence
        // over the undefined original hidden in the later group.
        let source = Expr::Add(vec![
            delayed(power(polynomial(&[1, -1, 0, 1], &s), -1), &s, q(1, 1)),
            invalid_tail,
        ]);
        assert!(matches!(
            inverse_delayed_distributional_laplace(&source, &s, &t),
            Err(RationalLaplaceError::InvalidSource(_))
        ));
        let proposal = DelayedDistributionalLaplaceCertificate {
            terms: vec![
                term(q(1, 1), vec![q(0, 1)], vec![]),
                term(q(2, 1), vec![q(0, 1)], vec![]),
            ],
        };
        assert!(matches!(
            verify_delayed_distributional_laplace_certificate(
                &source,
                &s,
                &t,
                proposal,
                &mut Unbounded
            ),
            Err(RationalLaplaceError::InvalidSource(_))
        ));
    }
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
fn unsupported_grammar_and_regular_tails_cannot_publish_partial_impulses() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let a = Symbol::new("a");
    for source in [
        delayed(Expr::Sym(s.clone()), &s, q(-1, 1)),
        Expr::Mul(vec![
            analytic("exp", Expr::Mul(vec![Expr::Sym(a), Expr::Sym(s.clone())])),
            Expr::Sym(s.clone()),
        ]),
        Expr::Mul(vec![
            analytic("exp", power(Expr::Sym(s.clone()), 2)),
            Expr::Sym(s.clone()),
        ]),
        delayed(power(Expr::Sym(s.clone()), 33), &s, q(1, 1)),
        Expr::Add(vec![
            delayed(Expr::Sym(s.clone()), &s, q(1, 1)),
            analytic("unhandled", Expr::Sym(s.clone())),
        ]),
    ] {
        let mut generator = Meter::default();
        let mut verifier = Meter::default();
        assert!(
            metered_inverse_delayed_distributional_laplace(
                &source,
                &s,
                &t,
                &mut generator,
                &mut verifier
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(verifier.charged, [0; 3]);
        assert_eq!(verifier.checkpoints, 0);
        assert!(generator.charged[0] < 10_000);
    }
    let unsupported = Expr::Add(vec![
        delayed(Expr::Sym(s.clone()), &s, q(1, 1)),
        delayed(
            Expr::Add(vec![
                Expr::from_i64(1),
                power(polynomial(&[1, -1, 0, 1], &s), -1),
            ]),
            &s,
            q(2, 1),
        ),
    ]);
    assert!(matches!(
        inverse_delayed_distributional_laplace(&unsupported, &s, &t),
        Err(RationalLaplaceError::Unsupported(_))
    ));
}

#[test]
fn held_exponent_domains_cannot_be_erased_to_obtain_an_impulse() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = analytic(
        "exp",
        Expr::Add(vec![
            power(polynomial(&[-100, 1], &s), 0),
            Expr::from_i64(-1),
        ]),
    );
    assert!(
        inverse_delayed_distributional_laplace(&source, &s, &t)
            .unwrap()
            .is_none()
    );
    let candidate = DelayedDistributionalLaplaceCertificate {
        terms: vec![term(q(0, 1), vec![q(1, 1)], vec![])],
    };
    assert!(
        verify_delayed_distributional_laplace_certificate(
            &source,
            &s,
            &t,
            candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn cumulative_budgets_cover_all_delays_and_both_output_components() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = mixed_source(&s);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_delayed_distributional_laplace(&source, &s, &t, &mut generator, &mut verifier)
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
        assert!(matches!(
            metered_inverse_delayed_distributional_laplace(
                &source,
                &s,
                &t,
                &mut exhausted,
                &mut Unbounded
            ),
            Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
        ));
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = verifier.charged[index] - 1;
        let mut exhausted = Budget::new(limits);
        assert!(matches!(
            metered_inverse_delayed_distributional_laplace(
                &source,
                &s,
                &t,
                &mut Unbounded,
                &mut exhausted
            ),
            Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
        ));
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut unused = Meter::default();
    assert!(matches!(
        metered_inverse_delayed_distributional_laplace(&source, &s, &t, &mut tiny, &mut unused),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(unused.charged, [0; 3]);
}

#[test]
fn cancellation_is_terminal_through_final_publication_in_each_lane() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = mixed_source(&s);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_delayed_distributional_laplace(&source, &s, &t, &mut generator, &mut verifier)
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
                metered_inverse_delayed_distributional_laplace(
                    &source,
                    &s,
                    &t,
                    &mut interrupted,
                    &mut Unbounded
                ),
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
                metered_inverse_delayed_distributional_laplace(
                    &source,
                    &s,
                    &t,
                    &mut Unbounded,
                    &mut interrupted
                ),
                Err(RationalLaplaceError::Meter(MeterError::Cancelled))
            ),
            "verifier checkpoint {stop}"
        );
    }
}

#[test]
fn intrinsic_limits_bound_term_count_shift_growth_and_complete_impulse_output() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let too_many = Expr::Add(
        (0..33)
            .map(|delay| delayed(Expr::from_i64(1), &s, q(delay, 1)))
            .collect(),
    );
    assert!(matches!(
        inverse_delayed_distributional_laplace(&too_many, &s, &t),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let huge_delay = BigRational::from_integer(BigInt::one() << 2048u32);
    assert!(matches!(
        inverse_delayed_distributional_laplace(
            &delayed(Expr::Sym(s.clone()), &s, huge_delay),
            &s,
            &t
        ),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let source = delayed(Expr::from_i64(1), &s, q(1, 1));
    let too_many_claims = DelayedDistributionalLaplaceCertificate {
        terms: vec![term(q(1, 1), vec![q(1, 1)], vec![]); 33],
    };
    assert!(matches!(
        verify_delayed_distributional_laplace_certificate(
            &source,
            &s,
            &t,
            too_many_claims,
            &mut Unbounded
        ),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    // Every group is individually admitted, but the complete shifted impulse
    // family exceeds the shared output bound. Resetting per group is unsound.
    let dense = polynomial(&[1; 33], &s);
    let source = Expr::Add(
        (1..=13)
            .map(|delay| delayed(dense.clone(), &s, q(delay, 1)))
            .collect(),
    );
    assert!(matches!(
        inverse_delayed_distributional_laplace(&source, &s, &t),
        Err(RationalLaplaceError::ResourceLimit(
            "complete delayed output nodes"
        ))
    ));
}
