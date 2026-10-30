//! Complete causal inverse identities, adversarial certificates and metering.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_laplace::{
    LaplaceBlock, RationalLaplaceCertificate, RationalLaplaceError, inverse_rational_laplace,
    metered_inverse_rational_laplace, verify_rational_laplace_certificate,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_polys::UnivariatePoly;
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn qv(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|&value| q(value, 1)).collect()
}

fn poly(values: &[i64], variable: &Symbol) -> Expr {
    UnivariatePoly::new(variable.clone(), qv(values)).to_expr()
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn fraction(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, power(denominator, -1)])
}

fn quadratic(
    rate: BigRational,
    d: BigRational,
    cosine: Vec<BigRational>,
    sine: Vec<BigRational>,
) -> LaplaceBlock {
    LaplaceBlock::Quadratic {
        rate,
        squared_frequency: d,
        cosine,
        sine,
    }
}

fn verify_expected(source: &Expr, expected: Vec<LaplaceBlock>) {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let result = inverse_rational_laplace(source, &s, &t).unwrap();
    assert_eq!(result.blocks().len(), expected.len());
    for block in &expected {
        assert!(
            result.blocks().contains(block),
            "{:?} missing from {:?}",
            block,
            result.blocks()
        );
    }
    let receipt = verify_rational_laplace_certificate(
        source,
        &s,
        &t,
        &RationalLaplaceCertificate { blocks: expected },
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.source(), source);
    assert_eq!(receipt.transform_variable(), &s);
    assert_eq!(receipt.time_variable(), &t);
    assert_eq!(
        fsym_simplify::simplify(result.function()),
        fsym_simplify::simplify(receipt.function()),
    );
}

#[test]
fn repeated_quadratic_poles_have_nonzero_complete_inverse() {
    let s = Symbol::new("s");
    let base = poly(&[1, 0, 1], &s);
    // L[(sin(t) - t*cos(t))/2] = 1/(s^2+1)^2.
    verify_expected(
        &power(base.clone(), -2),
        vec![quadratic(
            q(0, 1),
            q(1, 1),
            vec![q(0, 1), q(-1, 2)],
            vec![q(1, 2)],
        )],
    );
    // L[((3-t^2)*sin(t) - 3*t*cos(t))/8] = 1/(s^2+1)^3.
    verify_expected(
        &power(base.clone(), -3),
        vec![quadratic(
            q(0, 1),
            q(1, 1),
            vec![q(0, 1), q(-3, 8)],
            vec![q(3, 8), q(0, 1), q(-1, 8)],
        )],
    );
    verify_expected(
        &fraction(Expr::Sym(s.clone()), power(base, 3)),
        vec![quadratic(
            q(0, 1),
            q(1, 1),
            vec![q(0, 1), q(0, 1), q(-1, 8)],
            vec![q(0, 1), q(1, 8)],
        )],
    );
}

#[test]
fn shifted_irreducible_poles_and_rational_linear_scales_are_exact() {
    let s = Symbol::new("s");
    // z = s+2, d = 5. L[e^(-2t)*(t*S_5(t)-t^2*C_5(t))/40]
    // is z/(z^2+5)^3. No approximation of sqrt(5) enters the coefficients.
    verify_expected(
        &fraction(poly(&[2, 1], &s), power(poly(&[9, 4, 1], &s), 3)),
        vec![quadratic(
            q(-2, 1),
            q(5, 1),
            vec![q(0, 1), q(0, 1), q(-1, 40)],
            vec![q(0, 1), q(1, 40)],
        )],
    );
    verify_expected(
        &power(poly(&[3, 2], &s), -3),
        vec![LaplaceBlock::Exponential {
            rate: q(-3, 2),
            coefficients: vec![q(0, 1), q(0, 1), q(1, 16)],
        }],
    );
    verify_expected(
        &fraction(Expr::from_i64(3), power(poly(&[-2, 1], &s), 4)),
        vec![LaplaceBlock::Exponential {
            rate: q(2, 1),
            coefficients: vec![q(0, 1), q(0, 1), q(0, 1), q(1, 2)],
        }],
    );
}

#[test]
fn real_irrational_poles_use_the_same_exact_hyperbolic_basis() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let base = poly(&[-2, 0, 1], &s);
    verify_expected(
        &power(base.clone(), -1),
        vec![quadratic(q(0, 1), q(-2, 1), qv(&[0]), qv(&[1]))],
    );
    // General formula (S_d-t*C_d)/(2d), valid for d=-2 as well.
    let source = power(base, -2);
    verify_expected(
        &source,
        vec![quadratic(
            q(0, 1),
            q(-2, 1),
            vec![q(0, 1), q(1, 4)],
            vec![q(-1, 4)],
        )],
    );
    let receipt = inverse_rational_laplace(&source, &s, &t).unwrap();
    let names = function_names(receipt.function());
    assert!(names.iter().any(|name| name == "sinh"));
    assert!(names.iter().any(|name| name == "cosh"));
    assert!(!names.iter().any(|name| name == "sin" || name == "cos"));
    assert!(receipt.sufficient_abscissa() > &q(2, 1));
}

fn function_names(expression: &Expr) -> Vec<String> {
    let mut output = Vec::new();
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        match node {
            Expr::Function(name, children) => {
                output.push(name.clone());
                pending.extend(children);
            }
            Expr::Add(children) | Expr::Mul(children) => pending.extend(children),
            Expr::Pow(base, exponent) => {
                pending.push(base);
                pending.push(exponent);
            }
            _ => {}
        }
    }
    output
}

#[test]
fn mixed_factors_preserve_every_pole_and_polynomial_weight() {
    let s = Symbol::new("s");
    let source = Expr::Add(vec![
        power(Expr::Sym(s.clone()), -1),
        fraction(Expr::from_i64(2), power(poly(&[-1, 1], &s), 2)),
        fraction(poly(&[2, 1], &s), power(poly(&[5, 4, 1], &s), 2)),
    ]);
    verify_expected(
        &source,
        vec![
            LaplaceBlock::Exponential {
                rate: q(0, 1),
                coefficients: qv(&[1]),
            },
            LaplaceBlock::Exponential {
                rate: q(1, 1),
                coefficients: qv(&[0, 2]),
            },
            quadratic(q(-2, 1), q(1, 1), qv(&[0]), vec![q(0, 1), q(1, 2)]),
        ],
    );
}

#[test]
fn generated_shift_and_frequency_family_matches_independent_closed_inverses() {
    let s = Symbol::new("s");
    for a in [-2, 0, 1] {
        // Positive nonsquare and negative nonsquare frequencies exercise both
        // real bases. All expected coefficients come from the closed inverse,
        // rather than from the implementation's candidate or certificate.
        for d in [-3, 2, 5] {
            let base = poly(&[a * a + d, -2 * a, 1], &s);
            verify_expected(
                &power(base.clone(), -2),
                vec![quadratic(
                    q(a, 1),
                    q(d, 1),
                    vec![q(0, 1), q(-1, 2 * d)],
                    vec![q(1, 2 * d)],
                )],
            );
            verify_expected(
                &fraction(poly(&[-a, 1], &s), power(base, 3)),
                vec![quadratic(
                    q(a, 1),
                    q(d, 1),
                    vec![q(0, 1), q(0, 1), q(-1, 8 * d)],
                    vec![q(0, 1), q(1, 8 * d)],
                )],
            );
        }
    }
}

#[test]
fn source_domain_checks_precede_zero_and_cancellation_shortcuts() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    for exponent in [-1, 0] {
        let invalid = power(Expr::from_i64(0), exponent);
        for source in [
            invalid.clone(),
            Expr::Mul(vec![Expr::from_i64(0), invalid.clone()]),
            power(invalid, 0),
        ] {
            assert!(matches!(
                inverse_rational_laplace(&source, &s, &t),
                Err(RationalLaplaceError::InvalidSource(_))
            ));
            assert!(matches!(
                verify_rational_laplace_certificate(
                    &source,
                    &s,
                    &t,
                    &RationalLaplaceCertificate { blocks: vec![] },
                    &mut Unbounded,
                ),
                Err(RationalLaplaceError::InvalidSource(_))
            ));
        }
    }
    let original = Expr::Mul(vec![Expr::from_i64(0), power(poly(&[1, 1, 0, 1], &s), -1)]);
    let receipt = inverse_rational_laplace(&original, &s, &t).unwrap();
    assert_eq!(receipt.source(), &original);
    assert!(receipt.blocks().is_empty());
    assert!(receipt.function().is_zero());
    assert_eq!(receipt.sufficient_abscissa(), &q(2, 1));
    // The original removable pole remains outside the sufficient half-plane.
    let original = fraction(
        poly(&[-10, 1], &s),
        Expr::Mul(vec![poly(&[-10, 1], &s), poly(&[1, 1], &s)]),
    );
    let receipt = inverse_rational_laplace(&original, &s, &t).unwrap();
    assert_eq!(
        receipt.blocks(),
        &[LaplaceBlock::Exponential {
            rate: q(-1, 1),
            coefficients: qv(&[1])
        },]
    );
    assert!(receipt.sufficient_abscissa() > &q(10, 1));
}

#[test]
fn higher_factors_distributions_and_nonrational_sources_refuse_explicitly() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    for source in [
        power(poly(&[1, 1, 0, 1], &s), -1),
        Expr::from_i64(1),
        Expr::Sym(s.clone()),
        fraction(poly(&[1, 0, 1], &s), poly(&[1, 1], &s)),
        Expr::Function("exp".into(), vec![Expr::Sym(s.clone())]),
        Expr::Sym(Symbol::new("a")),
    ] {
        let error = inverse_rational_laplace(&source, &s, &t).unwrap_err();
        assert!(error.is_unsupported(), "{error:?}");
    }
    assert!(matches!(
        inverse_rational_laplace(&power(Expr::Sym(s.clone()), -1), &s, &s),
        Err(RationalLaplaceError::InvalidSource(_))
    ));
}

#[test]
fn certificates_cannot_omit_poles_mutate_coefficients_or_change_the_source() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = power(poly(&[1, 0, 1], &s), -2);
    let good = inverse_rational_laplace(&source, &s, &t).unwrap();
    let mutants = vec![
        RationalLaplaceCertificate { blocks: vec![] },
        RationalLaplaceCertificate {
            blocks: vec![quadratic(q(0, 1), q(1, 1), qv(&[0]), vec![q(1, 2)])],
        },
        RationalLaplaceCertificate {
            blocks: vec![quadratic(
                q(0, 1),
                q(1, 1),
                vec![q(0, 1), q(-1, 3)],
                vec![q(1, 2)],
            )],
        },
        RationalLaplaceCertificate {
            blocks: vec![quadratic(
                q(1, 1),
                q(1, 1),
                vec![q(0, 1), q(-1, 2)],
                vec![q(1, 2)],
            )],
        },
        RationalLaplaceCertificate {
            blocks: vec![quadratic(
                q(0, 1),
                q(2, 1),
                vec![q(0, 1), q(-1, 2)],
                vec![q(1, 2)],
            )],
        },
        RationalLaplaceCertificate {
            blocks: vec![quadratic(
                q(0, 1),
                q(0, 1),
                vec![q(0, 1), q(-1, 2)],
                vec![q(1, 2)],
            )],
        },
        RationalLaplaceCertificate {
            blocks: vec![quadratic(q(0, 1), q(1, 1), vec![], vec![q(1, 2)])],
        },
        RationalLaplaceCertificate {
            blocks: vec![quadratic(
                q(0, 1),
                q(1, 1),
                vec![q(0, 1), q(-1, 2), q(0, 1)],
                vec![q(1, 2)],
            )],
        },
    ];
    for mutant in mutants {
        assert!(
            verify_rational_laplace_certificate(&source, &s, &t, &mutant, &mut Unbounded,).is_err(),
            "{mutant:?}"
        );
    }
    let changed_source = Expr::Mul(vec![Expr::from_i64(2), source.clone()]);
    assert!(
        verify_rational_laplace_certificate(
            &changed_source,
            &s,
            &t,
            good.certificate(),
            &mut Unbounded,
        )
        .is_err()
    );
    let mixed = Expr::Add(vec![
        power(Expr::Sym(s.clone()), -1),
        power(poly(&[1, 0, 1], &s), -1),
    ]);
    let mut certificate = inverse_rational_laplace(&mixed, &s, &t)
        .unwrap()
        .certificate()
        .clone();
    certificate.blocks.pop();
    assert!(
        verify_rational_laplace_certificate(&mixed, &s, &t, &certificate, &mut Unbounded,).is_err()
    );
}

#[test]
fn typed_symbol_identity_survives_admission_and_output_lifting() {
    let plain = Symbol::new("s");
    let assumed = Symbol::with_identity("s", SymbolIdentity::from_assumptions_digest([7; 32]));
    let time = Symbol::with_identity("s", SymbolIdentity::from_assumptions_digest([9; 32]));
    let source = power(poly(&[-1, 1], &assumed), -1);
    let result = inverse_rational_laplace(&source, &assumed, &time).unwrap();
    assert_eq!(result.source(), &source);
    assert_eq!(result.function().free_symbols(), vec![time.clone()]);
    assert!(
        inverse_rational_laplace(&source, &plain, &time)
            .unwrap_err()
            .is_unsupported()
    );
    assert!(
        verify_rational_laplace_certificate(
            &source,
            &plain,
            &time,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap_err()
        .is_unsupported()
    );
}

#[test]
fn source_and_certificate_envelopes_refuse_before_unbounded_work() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let mut deep = Expr::from_i64(0);
    for _ in 0..66 {
        deep = Expr::Mul(vec![deep]);
    }
    let huge = Expr::Integer(BigInt::one() << 4096u32);
    for source in [
        deep,
        Expr::Mul(vec![Expr::from_i64(0), huge]),
        power(Expr::Sym(s.clone()), -33),
        Expr::Mul(vec![Expr::from_i64(0); 4097]),
    ] {
        assert!(matches!(
            inverse_rational_laplace(&source, &s, &t),
            Err(RationalLaplaceError::ResourceLimit(_))
        ));
    }
    let source = power(Expr::Sym(s.clone()), -1);
    let too_many = RationalLaplaceCertificate {
        blocks: vec![
            LaplaceBlock::Exponential {
                rate: q(0, 1),
                coefficients: qv(&[1])
            };
            33
        ],
    };
    assert!(matches!(
        verify_rational_laplace_certificate(&source, &s, &t, &too_many, &mut Unbounded),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let too_large = RationalLaplaceCertificate {
        blocks: vec![LaplaceBlock::Exponential {
            rate: q(0, 1),
            coefficients: vec![q(1, 1); 33],
        }],
    };
    assert!(
        verify_rational_laplace_certificate(&source, &s, &t, &too_large, &mut Unbounded,).is_err()
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
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn both_lanes_cancel_during_work_and_before_result_publication() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = power(poly(&[2, 0, 1], &s), -2);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_rational_laplace(&source, &s, &t, &mut generator, &mut verifier).unwrap();
    assert!(generator.charged.iter().all(|&value| value > 0));
    assert!(verifier.charged.iter().all(|&value| value > 0));
    for stop in [
        1,
        generator.checkpoints / 2,
        generator.checkpoints - 1,
        generator.checkpoints,
    ] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert!(
            matches!(
                metered_inverse_rational_laplace(&source, &s, &t, &mut cancelled, &mut Unbounded,),
                Err(RationalLaplaceError::Meter(MeterError::Cancelled))
            ),
            "generation checkpoint {stop}"
        );
    }
    for stop in [
        1,
        verifier.checkpoints / 2,
        verifier.checkpoints - 1,
        verifier.checkpoints,
    ] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert!(
            matches!(
                metered_inverse_rational_laplace(&source, &s, &t, &mut Unbounded, &mut cancelled,),
                Err(RationalLaplaceError::Meter(MeterError::Cancelled))
            ),
            "verification checkpoint {stop}"
        );
    }
}

#[test]
fn actual_budget_exhaustion_is_terminal_and_does_not_spend_the_other_lane() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = power(poly(&[1, 0, 1], &s), -2);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut verifier = Meter::default();
    assert!(matches!(
        metered_inverse_rational_laplace(&source, &s, &t, &mut tiny, &mut verifier),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(verifier.charged, [0; 3]);
    assert_eq!(verifier.checkpoints, 0);
    let mut generator = Meter::default();
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_inverse_rational_laplace(&source, &s, &t, &mut generator, &mut tiny),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
    assert!(generator.charged[0] > 0);
}

#[test]
fn outer_reciprocals_and_zeroth_powers_preserve_original_domain_exclusions() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let pole = poly(&[-100, 1], &s);
    let inner_reciprocal = power(pole.clone(), -1);
    let cases = [
        fraction(power(inner_reciprocal.clone(), 0), Expr::Sym(s.clone())),
        fraction(power(pole, 0), Expr::Sym(s.clone())),
        fraction(
            power(inner_reciprocal.clone(), -1),
            power(Expr::Sym(s.clone()), 2),
        ),
        Expr::Mul(vec![Expr::from_i64(0), power(inner_reciprocal, -1)]),
    ];
    for source in cases {
        let result = inverse_rational_laplace(&source, &s, &t).unwrap();
        assert_eq!(result.source(), &source);
        assert!(result.sufficient_abscissa() > &q(100, 1));
        let checked = verify_rational_laplace_certificate(
            &source,
            &s,
            &t,
            result.certificate(),
            &mut Unbounded,
        )
        .unwrap();
        assert!(checked.sufficient_abscissa() > &q(100, 1));
    }
}

fn has_numeric_square_root(expression: &Expr) -> bool {
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        match node {
            Expr::Pow(base, exponent) => {
                if matches!(base.as_ref(), Expr::Integer(_) | Expr::Rational(_))
                    && matches!(exponent.as_ref(),Expr::Rational(value) if value == &q(1,2))
                {
                    return true;
                }
                pending.push(base);
                pending.push(exponent);
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                pending.extend(children)
            }
            _ => {}
        }
    }
    false
}

#[test]
fn large_exact_and_held_radicals_lift_under_the_verifier_meter() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let square = BigInt::one() << 256u32;
    for (d, held) in [(square.clone(), false), (square + BigInt::one(), true)] {
        let d = BigRational::from_integer(d);
        let source = power(
            UnivariatePoly::new(s.clone(), vec![d.clone(), q(0, 1), q(1, 1)]).to_expr(),
            -1,
        );
        let certificate = RationalLaplaceCertificate {
            blocks: vec![quadratic(q(0, 1), d, qv(&[0]), qv(&[1]))],
        };
        let mut meter = Meter::default();
        let receipt =
            verify_rational_laplace_certificate(&source, &s, &t, &certificate, &mut meter).unwrap();
        assert_eq!(has_numeric_square_root(receipt.function()), held);
        for stop in [
            meter.checkpoints / 2,
            meter.checkpoints - 1,
            meter.checkpoints,
        ] {
            let mut cancelled = Meter {
                cancel_at: Some(stop),
                ..Default::default()
            };
            assert!(matches!(
                verify_rational_laplace_certificate(&source, &s, &t, &certificate, &mut cancelled,),
                Err(RationalLaplaceError::Meter(MeterError::Cancelled))
            ));
        }
    }
}

#[test]
fn zero_blocks_cannot_trigger_unnecessary_radical_work() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let huge = BigRational::from_integer((BigInt::one() << 4095u32) + BigInt::one());
    for block in [
        quadratic(q(0, 1), huge, qv(&[0]), qv(&[0])),
        LaplaceBlock::Exponential {
            rate: q(0, 1),
            coefficients: qv(&[0]),
        },
    ] {
        let certificate = RationalLaplaceCertificate {
            blocks: vec![block],
        };
        assert!(matches!(
            verify_rational_laplace_certificate(
                &Expr::from_i64(0),
                &s,
                &t,
                &certificate,
                &mut Unbounded,
            ),
            Err(RationalLaplaceError::Verification(_))
        ));
    }
}

#[test]
fn huge_integral_exponents_and_symbol_names_fail_borrowed_admission() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let exponent = Expr::Rational(BigRational::from_integer(BigInt::one() << 100_000u32));
    let source = Expr::Pow(Arc::new(Expr::Sym(s.clone())), Arc::new(exponent));
    assert!(matches!(
        inverse_rational_laplace(&source, &s, &t),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let long = Symbol::new("s".repeat(65_537));
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    assert!(matches!(
        metered_inverse_rational_laplace(
            &Expr::from_i64(0),
            &long,
            &long,
            &mut generator,
            &mut verifier,
        ),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    assert_eq!(generator.charged, [0; 3]);
    assert_eq!(verifier.charged, [0; 3]);
    assert!(matches!(
        inverse_rational_laplace(&Expr::Sym(long), &s, &t),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
}

fn analytic(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

/// Test-only rational AST fold, independent of both transform formulas.
fn rational_pair(expression: &Expr) -> (Expr, Expr) {
    match expression {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Sym(_) => {
            (expression.clone(), Expr::from_i64(1))
        }
        Expr::Add(children) => {
            let mut numerator = Expr::from_i64(0);
            let mut denominator = Expr::from_i64(1);
            for child in children {
                let (right_numerator, right_denominator) = rational_pair(child);
                numerator = Expr::Add(vec![
                    Expr::Mul(vec![numerator, right_denominator.clone()]),
                    Expr::Mul(vec![right_numerator, denominator.clone()]),
                ]);
                denominator = Expr::Mul(vec![denominator, right_denominator]);
            }
            (numerator, denominator)
        }
        Expr::Mul(children) => {
            let mut numerators = Vec::new();
            let mut denominators = Vec::new();
            for child in children {
                let (numerator, denominator) = rational_pair(child);
                numerators.push(numerator);
                denominators.push(denominator);
            }
            (Expr::Mul(numerators), Expr::Mul(denominators))
        }
        Expr::Pow(base, exponent) => {
            let exponent = match exponent.as_ref() {
                Expr::Integer(value) => value.to_i64().unwrap(),
                Expr::Rational(value) if value.is_integer() => value.numer().to_i64().unwrap(),
                _ => panic!("noninteger power in rational test expression"),
            };
            let (numerator, denominator) = rational_pair(base);
            if exponent < 0 {
                (power(denominator, -exponent), power(numerator, -exponent))
            } else {
                (power(numerator, exponent), power(denominator, exponent))
            }
        }
        _ => panic!("nonrational test expression"),
    }
}

fn verify_forward_expected(source: &Expr, expected: Expr, abscissa: Option<BigRational>) {
    use fsym_calculus::rational_laplace::{polynomial_laplace, verify_laplace_transform};
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let actual = polynomial_laplace(source, &t, &s).unwrap();
    assert_eq!(actual.source(), source);
    assert_eq!(actual.time_variable(), &t);
    assert_eq!(actual.transform_variable(), &s);
    assert_eq!(actual.sufficient_abscissa(), abscissa.as_ref());
    // Cross-multiply the independent hand-derived expressions, then prove
    // polynomial zero coefficient-by-coefficient in the general polynomial
    // engine. Generic simplify intentionally does not combine all these forms.
    let (actual_numerator, actual_denominator) = rational_pair(actual.transform());
    let (expected_numerator, expected_denominator) = rational_pair(&expected);
    for denominator in [&actual_denominator, &expected_denominator] {
        let polynomial = fsym_polys::bounded_polynomial_from_expr(
            denominator,
            std::slice::from_ref(&s),
            &mut Unbounded,
        )
        .unwrap();
        assert!(!polynomial.is_zero(), "undefined rational test expression");
    }
    let difference = Expr::Add(vec![
        Expr::Mul(vec![actual_numerator, expected_denominator]),
        Expr::Mul(vec![
            Expr::from_i64(-1),
            expected_numerator,
            actual_denominator,
        ]),
    ]);
    let polynomial = fsym_polys::bounded_polynomial_from_expr(
        &difference,
        std::slice::from_ref(&s),
        &mut Unbounded,
    )
    .unwrap();
    assert!(polynomial.is_zero(), "{polynomial:?}");
    let checked = verify_laplace_transform(source, &t, &s, &expected, &mut Unbounded).unwrap();
    assert_eq!(checked.sufficient_abscissa(), abscissa.as_ref());
}

#[test]
fn polynomial_weighted_trigonometric_transforms_use_full_derivative_orders() {
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let tv = Expr::Sym(t.clone());
    let base = poly(&[1, 0, 1], &s);
    verify_forward_expected(
        &Expr::Mul(vec![power(tv.clone(), 2), analytic("sin", tv.clone())]),
        fraction(poly(&[-2, 0, 6], &s), power(base.clone(), 3)),
        Some(q(0, 1)),
    );
    verify_forward_expected(
        &Expr::Mul(vec![power(tv.clone(), 3), analytic("cos", tv.clone())]),
        fraction(poly(&[6, 0, -36, 0, 6], &s), power(base, 4)),
        Some(q(0, 1)),
    );
    let shifted = Expr::Mul(vec![
        power(tv.clone(), 2),
        analytic("exp", Expr::Mul(vec![Expr::from_i64(-3), tv.clone()])),
        analytic("sin", Expr::Mul(vec![Expr::from_i64(2), tv])),
    ]);
    verify_forward_expected(
        &shifted,
        fraction(poly(&[92, 72, 12], &s), power(poly(&[13, 6, 1], &s), 3)),
        Some(q(-3, 1)),
    );
}

#[test]
fn distributed_time_polynomials_and_combined_exponentials_transform_exactly() {
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let tv = Expr::Sym(t.clone());
    let source = Expr::Mul(vec![
        Expr::Add(vec![tv.clone(), Expr::from_i64(1)]),
        Expr::Add(vec![tv.clone(), Expr::from_i64(2)]),
        analytic("sin", tv.clone()),
    ]);
    let base = poly(&[1, 0, 1], &s);
    let expected = Expr::Add(vec![
        fraction(poly(&[-2, 0, 6], &s), power(base.clone(), 3)),
        fraction(poly(&[0, 6], &s), power(base.clone(), 2)),
        fraction(Expr::from_i64(2), base),
    ]);
    verify_forward_expected(&source, expected, Some(q(0, 1)));
    let source = Expr::Mul(vec![
        power(tv.clone(), 4),
        analytic("exp", Expr::Mul(vec![Expr::from_i64(2), tv.clone()])),
        analytic("exp", Expr::Mul(vec![Expr::from_i64(-5), tv])),
    ]);
    verify_forward_expected(
        &source,
        fraction(Expr::from_i64(24), power(poly(&[3, 1], &s), 5)),
        Some(q(-3, 1)),
    );
}

#[test]
fn hyperbolic_expansion_preserves_cancellation_and_convergence_boundaries() {
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let tv = Expr::Sym(t.clone());
    let source = Expr::Mul(vec![
        power(tv.clone(), 2),
        analytic("cosh", Expr::Mul(vec![Expr::from_i64(3), tv.clone()])),
    ]);
    verify_forward_expected(
        &source,
        fraction(poly(&[0, 54, 0, 2], &s), power(poly(&[-9, 0, 1], &s), 3)),
        Some(q(3, 1)),
    );
    let argument = Expr::Mul(vec![Expr::from_i64(2), tv.clone()]);
    let source = Expr::Mul(vec![
        power(tv, 3),
        Expr::Add(vec![
            analytic("cosh", argument.clone()),
            Expr::Mul(vec![Expr::from_i64(-1), analytic("sinh", argument)]),
        ]),
    ]);
    verify_forward_expected(
        &source,
        fraction(Expr::from_i64(6), power(poly(&[2, 1], &s), 4)),
        Some(q(-2, 1)),
    );
}

#[test]
fn signed_and_fractional_frequencies_have_exact_normalizations() {
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let tv = Expr::Sym(t.clone());
    for frequency in [-3, -1, 1, 2] {
        let source = Expr::Mul(vec![
            power(tv.clone(), 2),
            analytic(
                "sin",
                Expr::Mul(vec![Expr::from_i64(frequency), tv.clone()]),
            ),
        ]);
        verify_forward_expected(
            &source,
            fraction(
                poly(
                    &[-2 * frequency * frequency * frequency, 0, 6 * frequency],
                    &s,
                ),
                power(poly(&[frequency * frequency, 0, 1], &s), 3),
            ),
            Some(q(0, 1)),
        );
    }
    let source = Expr::Mul(vec![
        power(tv.clone(), 2),
        analytic("sin", Expr::Mul(vec![Expr::Rational(q(1, 2)), tv])),
    ]);
    verify_forward_expected(
        &source,
        fraction(poly(&[-16, 0, 192], &s), power(poly(&[1, 0, 4], &s), 3)),
        Some(q(0, 1)),
    );
}

#[test]
fn forward_verifier_binds_original_time_source_and_rejects_extra_held_poles() {
    use fsym_calculus::rational_laplace::{polynomial_laplace, verify_laplace_transform};
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let tv = Expr::Sym(t.clone());
    let source = Expr::Mul(vec![power(tv.clone(), 2), analytic("sin", tv)]);
    let good = polynomial_laplace(&source, &t, &s).unwrap();
    for candidate in [
        Expr::from_i64(0),
        Expr::Mul(vec![Expr::from_i64(2), good.transform().clone()]),
        Expr::Add(vec![
            good.transform().clone(),
            power(Expr::Sym(s.clone()), -1),
        ]),
        Expr::Mul(vec![
            good.transform().clone(),
            power(power(poly(&[-100, 1], &s), -1), 0),
        ]),
    ] {
        assert!(verify_laplace_transform(&source, &t, &s, &candidate, &mut Unbounded).is_err());
    }
    let changed = Expr::Mul(vec![Expr::from_i64(2), source]);
    assert!(verify_laplace_transform(&changed, &t, &s, good.transform(), &mut Unbounded,).is_err());
}

#[test]
fn forward_zero_and_held_original_domains_are_distinct() {
    use fsym_calculus::rational_laplace::polynomial_laplace;
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let tv = Expr::Sym(t.clone());
    let sine = analytic("sin", tv.clone());
    let zero = Expr::Mul(vec![
        power(tv.clone(), 2),
        Expr::Add(vec![
            sine.clone(),
            Expr::Mul(vec![Expr::from_i64(-1), sine]),
        ]),
    ]);
    verify_forward_expected(&zero, Expr::from_i64(0), None);
    for invalid in [
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
        power(power(Expr::from_i64(0), -1), 0),
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), 0)]),
        analytic("sin", power(power(tv.clone(), -1), -1)),
        power(tv, 0),
    ] {
        assert!(matches!(
            polynomial_laplace(&invalid, &t, &s),
            Err(RationalLaplaceError::InvalidSource(_))
        ));
    }
}

#[test]
fn forward_refusals_preserve_typed_identity_and_the_declared_envelope() {
    use fsym_calculus::rational_laplace::polynomial_laplace;
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let other = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([4; 32]));
    let tv = Expr::Sym(t.clone());
    for unsupported in [
        Expr::Mul(vec![
            analytic("sin", tv.clone()),
            analytic("cos", tv.clone()),
        ]),
        analytic("exp", power(tv.clone(), 2)),
        Expr::Mul(vec![
            power(tv.clone(), 2),
            analytic("sin", Expr::Sym(other)),
        ]),
        analytic("log", tv.clone()),
    ] {
        assert!(
            polynomial_laplace(&unsupported, &t, &s)
                .unwrap_err()
                .is_unsupported()
        );
    }
    let too_high = Expr::Mul(vec![power(tv.clone(), 16), analytic("sin", tv)]);
    assert!(matches!(
        polynomial_laplace(&too_high, &t, &s),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let typed = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([9; 32]));
    let source = Expr::Mul(vec![
        power(Expr::Sym(typed.clone()), 2),
        analytic("sin", Expr::Sym(typed.clone())),
    ]);
    let result = polynomial_laplace(&source, &typed, &s).unwrap();
    assert_eq!(result.source(), &source);
    assert_eq!(result.transform().free_symbols(), vec![s]);
}

#[test]
fn weighted_forward_generation_and_reference_verification_share_no_budget_reset() {
    use fsym_calculus::rational_laplace::metered_laplace;
    let t = Symbol::new("t");
    let s = Symbol::new("s");
    let tv = Expr::Sym(t.clone());
    let source = Expr::Mul(vec![power(tv.clone(), 2), analytic("sin", tv)]);
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_laplace(&source, &t, &s, &mut generator, &mut verifier).unwrap();
    for stop in [1, generator.checkpoints / 2, generator.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert!(matches!(
            metered_laplace(&source, &t, &s, &mut cancelled, &mut Unbounded),
            Err(RationalLaplaceError::Meter(MeterError::Cancelled))
        ));
    }
    for stop in [1, verifier.checkpoints / 2, verifier.checkpoints] {
        let mut cancelled = Meter {
            cancel_at: Some(stop),
            ..Default::default()
        };
        assert!(matches!(
            metered_laplace(&source, &t, &s, &mut Unbounded, &mut cancelled),
            Err(RationalLaplaceError::Meter(MeterError::Cancelled))
        ));
    }
    let mut untouched = Meter::default();
    let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_laplace(&source, &t, &s, &mut budget, &mut untouched),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(untouched.charged, [0; 3]);
    let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_laplace(&source, &t, &s, &mut Unbounded, &mut budget),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
}
