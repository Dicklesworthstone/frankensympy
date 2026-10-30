//! Exact ordering and equality of independently indexed real roots.
//! Expected orders follow rational inequalities or explicit common factors;
//! no approximate value is used to decide a test expectation.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::real_roots::{
    RealRootComparisonCertificate, RealRootError, RealRootInterval, RealRootIsolationCertificate,
    VerifiedRealRootComparison, compare_real_roots, isolate_real_roots,
    verify_real_root_comparison,
};
use fsym_core::{BigInt, BigRational};
use num_traits::{One, Zero};
use std::cmp::Ordering;

fn polynomial(coefficients: &[i64]) -> Vec<BigRational> {
    coefficients
        .iter()
        .copied()
        .map(BigRational::from)
        .collect()
}

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn product(left: &[BigRational], right: &[BigRational]) -> Vec<BigRational> {
    let mut result = vec![BigRational::zero(); left.len() + right.len() - 1];
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            result[i + j] += a * b;
        }
    }
    result
}

fn compare(
    left: &[BigRational],
    left_index: usize,
    right: &[BigRational],
    right_index: usize,
) -> VerifiedRealRootComparison {
    compare_real_roots(
        left,
        left_index,
        right,
        right_index,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn interval(lower: BigRational, upper: BigRational, multiplicity: usize) -> RealRootInterval {
    RealRootInterval {
        lower,
        upper,
        multiplicity,
    }
}

fn verify(
    left: &[BigRational],
    left_index: usize,
    right: &[BigRational],
    right_index: usize,
    certificate: &RealRootComparisonCertificate,
) -> Result<VerifiedRealRootComparison, RealRootError> {
    verify_real_root_comparison(
        left,
        left_index,
        right,
        right_index,
        certificate,
        &mut Unbounded,
    )
}

#[test]
fn overlapping_quadratic_isolations_refine_to_exact_order() {
    let p = polynomial(&[-2, 0, 1]);
    let q = polynomial(&[-3, 0, 1]);
    let initial_p = isolate_real_roots(&p, &mut Unbounded, &mut Unbounded).unwrap();
    let initial_q = isolate_real_roots(&q, &mut Unbounded, &mut Unbounded).unwrap();
    assert!(initial_p.intervals()[1].upper >= initial_q.intervals()[1].lower);
    assert!(initial_q.intervals()[1].upper >= initial_p.intervals()[1].lower);

    for (left_index, right_index, expected) in [
        (0, 0, Ordering::Greater),
        (0, 1, Ordering::Less),
        (1, 0, Ordering::Greater),
        (1, 1, Ordering::Less),
    ] {
        let result = compare(&p, left_index, &q, right_index);
        assert_eq!(result.ordering(), expected);
        assert_eq!(result.left_coefficients(), &p);
        assert_eq!(result.right_coefficients(), &q);
        assert_eq!(result.left_index(), left_index);
        assert_eq!(result.right_index(), right_index);
        if expected == Ordering::Less {
            assert!(result.left_interval().upper < result.right_interval().lower);
        } else {
            assert!(result.right_interval().upper < result.left_interval().lower);
        }
        assert_eq!(
            verify(&p, left_index, &q, right_index, result.certificate()).unwrap(),
            result
        );
        assert_eq!(
            compare(&q, right_index, &p, left_index).ordering(),
            expected.reverse()
        );
    }
}

#[test]
fn every_index_of_one_polynomial_has_the_expected_total_order() {
    // The three roots are exactly -1, 0, 1. Multiplicity never duplicates an
    // index in this native distinct-real-root API.
    let p = polynomial(&[0, -1, 0, 1]);
    for left in 0..3 {
        for right in 0..3 {
            assert_eq!(compare(&p, left, &p, right).ordering(), left.cmp(&right));
        }
    }
}

#[test]
fn common_roots_are_equal_across_different_reducible_definitions() {
    let p = polynomial(&[-2, 0, 1]);
    let q = product(&p, &polynomial(&[-3, 1]));
    for index in 0..2 {
        let result = compare(&p, index, &q, index);
        assert_eq!(result.ordering(), Ordering::Equal);
        assert_eq!(result.certificate().left_roots.intervals.len(), 2);
        assert_eq!(result.certificate().right_roots.intervals.len(), 3);
        assert_eq!(
            verify(&p, index, &q, index, result.certificate()).unwrap(),
            result
        );
    }
    assert_eq!(compare(&p, 1, &q, 0).ordering(), Ordering::Greater);
    assert_eq!(compare(&p, 1, &q, 2).ordering(), Ordering::Less);
}

#[test]
fn repeated_factors_preserve_all_original_multiplicities() {
    let primitive = polynomial(&[-2, 0, 1]);
    let p = product(&primitive, &primitive);
    let q = product(&product(&p, &primitive), &polynomial(&[-3, 1]));
    let equality = compare(&p, 1, &q, 1);
    assert_eq!(equality.ordering(), Ordering::Equal);
    assert_eq!(
        equality
            .certificate()
            .left_roots
            .intervals
            .iter()
            .map(|root| root.multiplicity)
            .collect::<Vec<_>>(),
        vec![2, 2]
    );
    assert_eq!(
        equality
            .certificate()
            .right_roots
            .intervals
            .iter()
            .map(|root| root.multiplicity)
            .collect::<Vec<_>>(),
        vec![3, 3, 1]
    );
    assert_eq!(compare(&p, 0, &q, 1).ordering(), Ordering::Less);
    assert_eq!(compare(&p, 1, &q, 2).ordering(), Ordering::Less);
}

#[test]
fn tiny_irrational_separation_is_not_an_equality_tolerance() {
    // sqrt(2 + 2^-256) > sqrt(2). Their separation is much smaller than any
    // binary64 distinction, while these exact coefficients stay modest.
    let epsilon = BigRational::new(BigInt::one(), BigInt::one() << 256u32);
    let p = polynomial(&[-2, 0, 1]);
    let q = vec![
        -BigRational::from(2_i64) - &epsilon,
        BigRational::zero(),
        BigRational::one(),
    ];
    let result = compare(&p, 1, &q, 1);
    assert_eq!(result.ordering(), Ordering::Less);
    let a = result.left_interval();
    let b = result.right_interval();
    assert!(a.lower > BigRational::zero());
    assert!(b.lower > BigRational::zero());
    assert!(a.upper < b.lower);
    assert!(&a.lower * &a.lower < BigRational::from(2_i64));
    assert!(&a.upper * &a.upper > BigRational::from(2_i64));
    assert!(&b.lower * &b.lower < BigRational::from(2_i64) + &epsilon);
    assert!(&b.upper * &b.upper > BigRational::from(2_i64) + &epsilon);
    assert_eq!(compare(&p, 0, &q, 0).ordering(), Ordering::Greater);
}

#[test]
fn tiny_rational_separation_and_zero_have_exact_signs() {
    let epsilon = BigRational::new(BigInt::one(), BigInt::one() << 300u32);
    let one = polynomial(&[-1, 1]);
    let larger = vec![-BigRational::one() - &epsilon, BigRational::one()];
    assert_eq!(compare(&one, 0, &larger, 0).ordering(), Ordering::Less);
    let zero = polynomial(&[0, 1]);
    let positive = vec![-epsilon.clone(), BigRational::one()];
    let negative = vec![epsilon, BigRational::one()];
    assert_eq!(compare(&zero, 0, &positive, 0).ordering(), Ordering::Less);
    assert_eq!(
        compare(&zero, 0, &negative, 0).ordering(),
        Ordering::Greater
    );
    assert_eq!(compare(&zero, 0, &zero, 0).ordering(), Ordering::Equal);
}

#[test]
fn coefficients_are_bound_with_scale_repetition_and_trailing_zeros() {
    let p = polynomial(&[-2, 0, 1]);
    let mut scaled: Vec<_> = product(&p, &p)
        .into_iter()
        .map(|coefficient| coefficient * rational(-7, 11))
        .collect();
    scaled.extend([BigRational::zero(), BigRational::zero()]);
    let result = compare(&p, 1, &scaled, 1);
    assert_eq!(result.ordering(), Ordering::Equal);
    assert_eq!(result.left_coefficients(), &p);
    assert_eq!(result.right_coefficients(), &scaled);
    assert_eq!(result.right_interval().multiplicity, 2);
}

#[test]
fn a_shared_rational_endpoint_proves_equality_but_never_strict_order() {
    let p = polynomial(&[-1, 1]);
    let q = polynomial(&[-1, 0, 1]);
    let certificate = RealRootComparisonCertificate {
        left_roots: RealRootIsolationCertificate {
            intervals: vec![interval(rational(1, 1), rational(1, 1), 1)],
        },
        right_roots: RealRootIsolationCertificate {
            intervals: vec![
                interval(rational(-2, 1), rational(0, 1), 1),
                interval(rational(1, 1), rational(2, 1), 1),
            ],
        },
        ordering: Ordering::Equal,
    };
    let accepted = verify(&p, 0, &q, 1, &certificate).unwrap();
    assert_eq!(accepted.ordering(), Ordering::Equal);
    for ordering in [Ordering::Less, Ordering::Greater] {
        let mut forged = certificate.clone();
        forged.ordering = ordering;
        assert!(verify(&p, 0, &q, 1, &forged).is_err());
    }
}

#[test]
fn shared_factor_membership_cannot_confuse_different_overlapping_embeddings() {
    // Both selected roots belong to gcd(p,p), and the selected intervals
    // overlap, but their intersection [5/4, 7/4] contains no common root.
    // A checker using membership in the gcd without the intersection is wrong.
    let p = polynomial(&[2, -3, 1]);
    let certificate = RealRootComparisonCertificate {
        left_roots: RealRootIsolationCertificate {
            intervals: vec![
                interval(rational(0, 1), rational(7, 4), 1),
                interval(rational(7, 4), rational(3, 1), 1),
            ],
        },
        right_roots: RealRootIsolationCertificate {
            intervals: vec![
                interval(rational(0, 1), rational(5, 4), 1),
                interval(rational(5, 4), rational(3, 1), 1),
            ],
        },
        ordering: Ordering::Equal,
    };
    fsym_core::real_roots::verify_real_root_isolation(&p, &certificate.left_roots, &mut Unbounded)
        .unwrap();
    fsym_core::real_roots::verify_real_root_isolation(&p, &certificate.right_roots, &mut Unbounded)
        .unwrap();
    assert!(verify(&p, 0, &p, 1, &certificate).is_err());
    assert!(verify(&p, 1, &p, 0, &certificate).is_err());
    assert_eq!(compare(&p, 0, &p, 1).ordering(), Ordering::Less);
}

#[test]
fn reference_checker_rejects_order_root_set_multiplicity_and_interval_mutations() {
    let p = polynomial(&[-2, 0, 1]);
    let q = polynomial(&[-3, 0, 1]);
    let accepted = compare(&p, 1, &q, 1);
    let source = accepted.certificate();
    let mut mutations = Vec::new();
    for ordering in [Ordering::Equal, Ordering::Greater] {
        let mut certificate = source.clone();
        certificate.ordering = ordering;
        mutations.push(certificate);
    }
    let mut certificate = source.clone();
    certificate.left_roots.intervals.remove(0);
    mutations.push(certificate);
    let mut certificate = source.clone();
    certificate.right_roots.intervals[1].multiplicity = 2;
    mutations.push(certificate);
    let mut certificate = source.clone();
    certificate.left_roots.intervals[1] = interval(rational(2, 1), rational(3, 1), 1);
    mutations.push(certificate);
    let mut certificate = source.clone();
    certificate.right_roots.intervals.reverse();
    mutations.push(certificate);
    let mut certificate = source.clone();
    certificate.left_roots.intervals[1].lower = rational(4, 1);
    mutations.push(certificate);
    for certificate in mutations {
        assert!(verify(&p, 1, &q, 1, &certificate).is_err());
    }

    let same_polynomial = compare(&p, 0, &p, 1);
    assert!(verify(&p, 1, &p, 1, same_polynomial.certificate()).is_err());
    assert!(verify(&p, 0, &p, 0, same_polynomial.certificate()).is_err());
    assert!(verify(&p, 0, &p, 2, same_polynomial.certificate()).is_err());
}

#[test]
fn equality_is_rechecked_against_both_coefficients_and_both_indices() {
    let p = polynomial(&[-2, 0, 1]);
    let q = product(&p, &polynomial(&[-3, 1]));
    let accepted = compare(&p, 1, &q, 1);
    let certificate = accepted.certificate();
    assert!(verify(&p, 0, &q, 1, certificate).is_err());
    assert!(verify(&p, 1, &q, 0, certificate).is_err());
    assert!(verify(&polynomial(&[-3, 0, 1]), 1, &q, 1, certificate).is_err());
    let changed = product(&polynomial(&[-3, 0, 1]), &polynomial(&[-3, 1]));
    assert!(verify(&p, 1, &changed, 1, certificate).is_err());
}

#[test]
fn absent_real_roots_invalid_indices_and_large_inputs_refuse() {
    let p = polynomial(&[-2, 0, 1]);
    for invalid in [Vec::new(), polynomial(&[0]), vec![BigRational::one(); 66]] {
        assert!(compare_real_roots(&invalid, 0, &p, 1, &mut Unbounded, &mut Unbounded).is_err());
    }
    for invalid in [polynomial(&[1]), polynomial(&[1, 0, 1])] {
        assert!(matches!(
            compare_real_roots(&invalid, 0, &p, 1, &mut Unbounded, &mut Unbounded),
            Err(RealRootError::InvalidIndex)
        ));
    }
    assert!(matches!(
        compare_real_roots(&p, 2, &p, 0, &mut Unbounded, &mut Unbounded),
        Err(RealRootError::InvalidIndex)
    ));
    assert!(matches!(
        compare_real_roots(&p, 0, &p, usize::MAX, &mut Unbounded, &mut Unbounded),
        Err(RealRootError::InvalidIndex)
    ));
    let oversized = vec![
        BigRational::from_integer(BigInt::one() << 4096u32),
        BigRational::one(),
    ];
    assert!(matches!(
        compare_real_roots(&p, 0, &oversized, 0, &mut Unbounded, &mut Unbounded),
        Err(RealRootError::ResourceLimit("input coefficient bits"))
    ));
    let accepted = compare(&p, 0, &p, 1);
    let mut certificate = accepted.certificate().clone();
    certificate.left_roots.intervals[0].lower =
        BigRational::from_integer(-(BigInt::one() << 32768u32));
    assert!(verify(&p, 0, &p, 1, &certificate).is_err());
}

#[derive(Default)]
struct CountingMeter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    compute: u64,
    memory: u64,
}

impl BudgetMeter for CountingMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
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
fn both_lanes_cancel_during_work_and_after_final_verification_before_publication() {
    let p = polynomial(&[-2, 0, 1]);
    for q in [p.clone(), polynomial(&[-3, 0, 1])] {
        let mut generator = CountingMeter::default();
        let mut verifier = CountingMeter::default();
        compare_real_roots(&p, 1, &q, 1, &mut generator, &mut verifier).unwrap();
        assert!(generator.compute > 0 && verifier.compute > 0);
        assert!(generator.memory > 0 && verifier.memory > 0);
        for checkpoint in [1, generator.checkpoints / 2, generator.checkpoints] {
            let mut stopped = CountingMeter {
                cancel_at: Some(checkpoint),
                ..CountingMeter::default()
            };
            assert_eq!(
                compare_real_roots(&p, 1, &q, 1, &mut stopped, &mut Unbounded).unwrap_err(),
                RealRootError::Meter(MeterError::Cancelled)
            );
        }
        for checkpoint in [1, verifier.checkpoints / 2, verifier.checkpoints] {
            let mut stopped = CountingMeter {
                cancel_at: Some(checkpoint),
                ..CountingMeter::default()
            };
            assert_eq!(
                compare_real_roots(&p, 1, &q, 1, &mut Unbounded, &mut stopped).unwrap_err(),
                RealRootError::Meter(MeterError::Cancelled)
            );
        }
    }
}

#[test]
fn direct_reference_verification_cancels_after_retaining_the_result() {
    let p = polynomial(&[-2, 0, 1]);
    let q = product(&p, &p);
    let accepted = compare(&p, 1, &q, 1);
    let mut meter = CountingMeter::default();
    verify_real_root_comparison(&p, 1, &q, 1, accepted.certificate(), &mut meter).unwrap();
    for checkpoint in [1, meter.checkpoints / 2, meter.checkpoints] {
        let mut stopped = CountingMeter {
            cancel_at: Some(checkpoint),
            ..CountingMeter::default()
        };
        assert_eq!(
            verify_real_root_comparison(&p, 1, &q, 1, accepted.certificate(), &mut stopped)
                .unwrap_err(),
            RealRootError::Meter(MeterError::Cancelled)
        );
    }
}

#[test]
fn exhausted_generator_and_verifier_budgets_never_publish_an_order() {
    let p = polynomial(&[-2, 0, 1]);
    let q = polynomial(&[-3, 0, 1]);
    let mut tiny_generator = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = CountingMeter::default();
    assert!(matches!(
        compare_real_roots(&p, 1, &q, 1, &mut tiny_generator, &mut untouched),
        Err(RealRootError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(untouched.checkpoints, 0);
    assert_eq!(untouched.compute, 0);
    let mut tiny_verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        compare_real_roots(&p, 1, &q, 1, &mut Unbounded, &mut tiny_verifier),
        Err(RealRootError::Meter(MeterError::Budget(_)))
    ));

    let mut measured = CountingMeter::default();
    compare_real_roots(&p, 1, &q, 1, &mut measured, &mut Unbounded).unwrap();
    let mut limits = BudgetLimits::uniform(512 * 1024 * 1024, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = 2 * measured.compute - 1;
    let mut cumulative = Budget::new(limits);
    compare_real_roots(&p, 1, &q, 1, &mut cumulative, &mut Unbounded).unwrap();
    // A second call retains the caller's already consumed allowance. Neither
    // a helper boundary nor a new outer request may reset that ledger.
    assert!(matches!(
        compare_real_roots(&p, 1, &q, 1, &mut cumulative, &mut Unbounded),
        Err(RealRootError::Meter(MeterError::Budget(_)))
    ));
}
