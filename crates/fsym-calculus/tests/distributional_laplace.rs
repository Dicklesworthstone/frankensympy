//! Complete polynomial impulse parts and independently checked rational tails.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_laplace::distributional::{
    DistributionalLaplaceCertificate, inverse_distributional_laplace,
    metered_inverse_distributional_laplace, verify_distributional_laplace_certificate,
};
use fsym_calculus::rational_laplace::{
    LaplaceBlock, RationalLaplaceCertificate, RationalLaplaceError,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_polys::UnivariatePoly;
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn polynomial(values: &[i64], variable: &Symbol) -> Expr {
    UnivariatePoly::new(variable.clone(), values.iter().map(|&n| q(n, 1)).collect()).to_expr()
}
fn fraction(top: Expr, bottom: Expr) -> Expr {
    Expr::Mul(vec![top, power(bottom, -1)])
}
fn delta(time: &Symbol, order: i64) -> Expr {
    let mut arguments = vec![Expr::Sym(time.clone())];
    if order != 0 {
        arguments.push(Expr::from_i64(order));
    }
    Expr::Function("DiracDelta".into(), arguments)
}
fn empty_regular() -> RationalLaplaceCertificate {
    RationalLaplaceCertificate { blocks: vec![] }
}

#[test]
fn every_admitted_polynomial_degree_produces_the_corresponding_impulse_derivative() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    for order in 0..=32 {
        let source = if order == 0 {
            Expr::from_i64(1)
        } else {
            power(Expr::Sym(s.clone()), order)
        };
        let result = inverse_distributional_laplace(&source, &s, &t).unwrap();
        assert_eq!(result.regular_function(), &Expr::from_i64(0));
        assert_eq!(result.impulses(), &delta(&t, order));
        let mut coefficients = vec![q(0, 1); order as usize + 1];
        coefficients[order as usize] = q(1, 1);
        assert_eq!(result.certificate().impulse_coefficients, coefficients);
        // An independently supplied transform table entry is sufficient;
        // acceptance does not require the generator's division output.
        verify_distributional_laplace_certificate(
            &source,
            &s,
            &t,
            &DistributionalLaplaceCertificate {
                impulse_coefficients: coefficients,
                regular: empty_regular(),
            },
            &mut Unbounded,
        )
        .unwrap();
    }
}

#[test]
fn improper_rational_inverse_keeps_both_impulses_and_ordinary_function() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    // s/(s+1) = 1 - 1/(s+1).
    let source = fraction(Expr::Sym(s.clone()), polynomial(&[1, 1], &s));
    let expected = DistributionalLaplaceCertificate {
        impulse_coefficients: vec![q(1, 1)],
        regular: RationalLaplaceCertificate {
            blocks: vec![LaplaceBlock::Exponential {
                rate: q(-1, 1),
                coefficients: vec![q(-1, 1)],
            }],
        },
    };
    let checked =
        verify_distributional_laplace_certificate(&source, &s, &t, &expected, &mut Unbounded)
            .unwrap();
    let generated = inverse_distributional_laplace(&source, &s, &t).unwrap();
    assert_eq!(generated.certificate(), &expected);
    assert_eq!(checked.impulses(), &delta(&t, 0));
    assert_eq!(checked.regular_function(), generated.regular_function());
    assert_eq!(checked.source(), &source);
    // (s^3+2s+1)/(s^2+1) = s + (s+1)/(s^2+1).
    let source = fraction(polynomial(&[1, 2, 0, 1], &s), polynomial(&[1, 0, 1], &s));
    let expected = DistributionalLaplaceCertificate {
        impulse_coefficients: vec![q(0, 1), q(1, 1)],
        regular: RationalLaplaceCertificate {
            blocks: vec![LaplaceBlock::Quadratic {
                rate: q(0, 1),
                squared_frequency: q(1, 1),
                cosine: vec![q(1, 1)],
                sine: vec![q(1, 1)],
            }],
        },
    };
    let checked =
        verify_distributional_laplace_certificate(&source, &s, &t, &expected, &mut Unbounded)
            .unwrap();
    assert_eq!(checked.impulses(), &delta(&t, 1));
    assert_eq!(
        inverse_distributional_laplace(&source, &s, &t)
            .unwrap()
            .certificate(),
        &expected
    );
}

#[test]
fn polynomial_parts_compose_with_repeated_poles_and_exact_rational_scales() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    for rate in [-2, 0, 3] {
        for multiplicity in 1..=3 {
            for degree in 0..=4 {
                let coefficient = q(2 * degree + 1, 3);
                let polynomial_part = if degree == 0 {
                    Expr::Rational(coefficient.clone())
                } else {
                    Expr::Rational(coefficient.clone()) * power(Expr::Sym(s.clone()), degree)
                };
                let source = Expr::Add(vec![
                    polynomial_part,
                    power(polynomial(&[-rate, 1], &s), -multiplicity),
                ]);
                let mut impulse_coefficients = vec![q(0, 1); degree as usize + 1];
                impulse_coefficients[degree as usize] = coefficient;
                let mut coefficients = vec![q(0, 1); multiplicity as usize];
                // 1/(s-a)^m has inverse t^(m-1)*exp(a*t)/(m-1)!.
                let factorial: i64 = (1..multiplicity).product();
                coefficients[multiplicity as usize - 1] = q(1, factorial);
                let expected = DistributionalLaplaceCertificate {
                    impulse_coefficients,
                    regular: RationalLaplaceCertificate {
                        blocks: vec![LaplaceBlock::Exponential {
                            rate: q(rate, 1),
                            coefficients,
                        }],
                    },
                };
                let generated = inverse_distributional_laplace(&source, &s, &t).unwrap();
                assert_eq!(generated.certificate(), &expected);
                verify_distributional_laplace_certificate(
                    &source,
                    &s,
                    &t,
                    &expected,
                    &mut Unbounded,
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn neither_component_can_be_omitted_or_borrowed_from_a_different_source() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = fraction(Expr::Sym(s.clone()), polynomial(&[1, 1], &s));
    let good = inverse_distributional_laplace(&source, &s, &t).unwrap();
    let mut missing_impulse = good.certificate().clone();
    missing_impulse.impulse_coefficients = vec![q(0, 1)];
    let mut missing_regular = good.certificate().clone();
    missing_regular.regular.blocks.clear();
    let mut perturbation = good.certificate().clone();
    perturbation.impulse_coefficients[0] += BigRational::new(1.into(), BigInt::from(10).pow(80));
    let mut wrong_order = good.certificate().clone();
    wrong_order.impulse_coefficients = vec![q(0, 1), q(1, 1)];
    for mutant in [missing_impulse, missing_regular, perturbation, wrong_order] {
        assert!(
            verify_distributional_laplace_certificate(&source, &s, &t, &mutant, &mut Unbounded)
                .is_err()
        );
    }
    let changed = Expr::Add(vec![source, Expr::from_i64(1)]);
    assert!(
        verify_distributional_laplace_certificate(
            &changed,
            &s,
            &t,
            good.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn original_undefined_children_and_excluded_poles_survive_polynomial_division() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    for exponent in [-1, 0] {
        let invalid = power(Expr::from_i64(0), exponent);
        let source = Expr::Add(vec![
            Expr::from_i64(1),
            Expr::Mul(vec![Expr::from_i64(0), invalid]),
        ]);
        assert!(matches!(
            inverse_distributional_laplace(&source, &s, &t),
            Err(RationalLaplaceError::InvalidSource(_))
        ));
        let certificate = DistributionalLaplaceCertificate {
            impulse_coefficients: vec![q(1, 1)],
            regular: empty_regular(),
        };
        assert!(matches!(
            verify_distributional_laplace_certificate(
                &source,
                &s,
                &t,
                &certificate,
                &mut Unbounded
            ),
            Err(RationalLaplaceError::InvalidSource(_))
        ));
    }
    // The value is 1 but the original source is undefined at s=100.
    let source = power(power(polynomial(&[-100, 1], &s), -1), 0);
    let checked = inverse_distributional_laplace(&source, &s, &t).unwrap();
    assert_eq!(checked.impulses(), &delta(&t, 0));
    assert!(checked.sufficient_abscissa() > &q(100, 1));
    assert_eq!(checked.source(), &source);
}

#[test]
fn bounded_source_and_canonical_impulse_contracts_are_enforced() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    for coefficients in [
        vec![],
        vec![q(1, 1), q(0, 1)],
        vec![q(1, 1); 34],
        vec![BigRational::from_integer(BigInt::from(1) << 65_536u32)],
    ] {
        let certificate = DistributionalLaplaceCertificate {
            impulse_coefficients: coefficients,
            regular: empty_regular(),
        };
        assert!(
            verify_distributional_laplace_certificate(
                &Expr::from_i64(1),
                &s,
                &t,
                &certificate,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(matches!(
        inverse_distributional_laplace(&power(Expr::Sym(s.clone()), 33), &s, &t),
        Err(RationalLaplaceError::ResourceLimit(_))
    ));
    let irreducible = power(polynomial(&[1, -1, 0, 1], &s), -1);
    assert!(
        inverse_distributional_laplace(&Expr::Add(vec![Expr::from_i64(1), irreducible]), &s, &t)
            .unwrap_err()
            .is_unsupported()
    );
}

#[test]
fn zero_and_proper_results_and_typed_time_variables_remain_distinct() {
    let s = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([5; 32]));
    let t = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([7; 32]));
    let zero = inverse_distributional_laplace(&Expr::from_i64(0), &s, &t).unwrap();
    assert!(zero.regular_function().is_zero() && zero.impulses().is_zero());
    let ordinary =
        inverse_distributional_laplace(&power(Expr::Sym(s.clone()), -1), &s, &t).unwrap();
    assert_eq!(ordinary.regular_function(), &Expr::from_i64(1));
    assert!(ordinary.impulses().is_zero());
    let impulse = inverse_distributional_laplace(&Expr::Sym(s.clone()), &s, &t).unwrap();
    assert_eq!(impulse.impulses().free_symbols(), vec![t.clone()]);
    assert_eq!(impulse.transform_variable(), &s);
    assert_eq!(impulse.time_variable(), &t);
    assert!(
        inverse_distributional_laplace(&Expr::Sym(s.clone()), &Symbol::new("same"), &t)
            .unwrap_err()
            .is_unsupported()
    );
    assert!(matches!(
        inverse_distributional_laplace(&Expr::Sym(s.clone()), &s, &s),
        Err(RationalLaplaceError::InvalidSource(_))
    ));
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    charged: [u64; 3],
}
impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.charged[0] += amount,
            Dimension::MemoryBytes => self.charged[1] += amount,
            Dimension::AllocationCount => self.charged[2] += amount,
            _ => {}
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
fn generator_and_verifier_cancel_before_work_and_before_publication() {
    let s = Symbol::new("s");
    let t = Symbol::new("t");
    let source = fraction(Expr::Sym(s.clone()), polynomial(&[1, 1], &s));
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_inverse_distributional_laplace(&source, &s, &t, &mut generator, &mut verifier).unwrap();
    assert!(generator.charged.iter().all(|&n| n > 0));
    assert!(verifier.charged.iter().all(|&n| n > 0));
    for (is_generator, total) in [(true, generator.checkpoints), (false, verifier.checkpoints)] {
        for stop in [1, total / 2, total - 1, total] {
            let mut cancelled = Meter {
                cancel_at: Some(stop),
                ..Default::default()
            };
            let result = if is_generator {
                metered_inverse_distributional_laplace(
                    &source,
                    &s,
                    &t,
                    &mut cancelled,
                    &mut Unbounded,
                )
            } else {
                metered_inverse_distributional_laplace(
                    &source,
                    &s,
                    &t,
                    &mut Unbounded,
                    &mut cancelled,
                )
            };
            assert!(matches!(
                result,
                Err(RationalLaplaceError::Meter(MeterError::Cancelled))
            ));
        }
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(matches!(
        metered_inverse_distributional_laplace(&source, &s, &t, &mut tiny, &mut untouched),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(untouched.checkpoints, 0);
    assert_eq!(untouched.charged, [0; 3]);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_inverse_distributional_laplace(&source, &s, &t, &mut Unbounded, &mut tiny),
        Err(RationalLaplaceError::Meter(MeterError::Budget(_)))
    ));
}
