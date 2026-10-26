//! Exact complete root sets and adversarial acceptance boundaries.
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::real_roots::{
    RealRootError, RealRootInterval, RealRootIsolationCertificate, identify_real_root_factors,
    isolate_real_roots, real_root_decimal, verify_real_root_isolation,
};
use fsym_core::{BigInt, BigRational};
use num_traits::Zero;

fn q(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}
fn poly(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|&n| q(n)).collect()
}
fn product(a: &[BigRational], b: &[BigRational]) -> Vec<BigRational> {
    let mut result = vec![BigRational::zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            result[i + j] += x * y;
        }
    }
    result
}
fn point(value: BigRational, multiplicity: usize) -> RealRootInterval {
    RealRootInterval {
        lower: value.clone(),
        upper: value,
        multiplicity,
    }
}
fn within(root: &RealRootInterval, value: &BigRational) -> bool {
    &root.lower <= value && value <= &root.upper
}

#[test]
fn complete_real_roots_and_original_multiplicities() {
    // (x+2)^3 * x^2 * (x-1) * (x^2+1): three real roots, six with multiplicity.
    let mut p = poly(&[1]);
    for factor in [
        poly(&[2, 1]),
        poly(&[2, 1]),
        poly(&[2, 1]),
        poly(&[0, 1]),
        poly(&[0, 1]),
        poly(&[-1, 1]),
        poly(&[1, 0, 1]),
    ] {
        p = product(&p, &factor);
    }
    let roots = isolate_real_roots(&p, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(roots.intervals().len(), 3);
    assert_eq!(roots.count_with_multiplicity(), 6);
    for (root, (value, mult)) in roots.intervals().iter().zip([(-2, 3), (0, 2), (1, 1)]) {
        assert!(within(root, &q(value)));
        assert_eq!(root.multiplicity, mult);
    }
    let exact = RealRootIsolationCertificate {
        intervals: vec![point(q(-2), 3), point(q(0), 2), point(q(1), 1)],
    };
    verify_real_root_isolation(&p, &exact, &mut Unbounded).unwrap();
}

#[test]
fn rational_endpoints_zero_and_negative_leading_coefficient() {
    // The reference checker must count roots at either closed endpoint once.
    let p = poly(&[0, 1, 0, -1]);
    let cert = RealRootIsolationCertificate {
        intervals: vec![
            RealRootInterval {
                lower: q(-2),
                upper: q(-1),
                multiplicity: 1,
            },
            point(q(0), 1),
            RealRootInterval {
                lower: q(1),
                upper: q(2),
                multiplicity: 1,
            },
        ],
    };
    verify_real_root_isolation(&p, &cert, &mut Unbounded).unwrap();
    assert_eq!(
        real_root_decimal(&p, 1, 100, &mut Unbounded, &mut Unbounded).unwrap(),
        "0"
    );
}

#[test]
fn close_real_roots_are_distinct_and_near_axis_complex_roots_stay_nonreal() {
    let delta = BigRational::new(BigInt::one(), BigInt::from(10).pow(60));
    let a = q(1);
    let b = &a + &delta;
    let p = product(&[-a.clone(), q(1)], &[-b.clone(), q(1)]);
    let roots = isolate_real_roots(&p, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(roots.intervals().len(), 2);
    assert!(within(&roots.intervals()[0], &a));
    assert!(within(&roots.intervals()[1], &b));
    assert!(roots.intervals()[0].upper <= roots.intervals()[1].lower);
    let no_real = vec![q(1) + &delta * &delta, q(-2), q(1)];
    assert!(
        isolate_real_roots(&no_real, &mut Unbounded, &mut Unbounded)
            .unwrap()
            .intervals()
            .is_empty()
    );
}

#[test]
fn rounded_decimal_is_certified_beyond_binary64() {
    let p = poly(&[-2, 0, 1]);
    assert_eq!(
        real_root_decimal(&p, 1, 50, &mut Unbounded, &mut Unbounded).unwrap(),
        "1.4142135623730950488016887242096980785696718753769e0"
    );
    assert_eq!(
        real_root_decimal(&p, 0, 50, &mut Unbounded, &mut Unbounded).unwrap(),
        "-1.4142135623730950488016887242096980785696718753769e0"
    );
    // The same tie in a nonlinear polynomial must terminate without relying
    // on a fortunate dyadic bisection point.
    let nonlinear = product(
        &[BigRational::new((-5).into(), 4.into()), q(1)],
        &poly(&[-3, 1]),
    );
    assert_eq!(
        real_root_decimal(&nonlinear, 0, 2, &mut Unbounded, &mut Unbounded).unwrap(),
        "1.2e0"
    );
    // Exact rational tie: 1.25 rounded to two significant digits is 1.2.
    let rational = vec![BigRational::new((-5).into(), 4.into()), q(1)];
    assert_eq!(
        real_root_decimal(&rational, 0, 2, &mut Unbounded, &mut Unbounded).unwrap(),
        "1.2e0"
    );
}

#[test]
fn irreducible_quintic_has_one_real_root_and_four_nonreal_roots() {
    let p = poly(&[1, -1, 0, 0, 0, 1]);
    let roots = isolate_real_roots(&p, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(roots.intervals().len(), 1);
    assert_eq!(roots.count_with_multiplicity(), 1);
    let decimal = real_root_decimal(&p, 0, 30, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(decimal, "-1.16730397826141868425604589985e0");
}

#[test]
fn checker_rejects_omission_duplicates_order_overlap_multiplicity_and_tiny_mutants() {
    let p = poly(&[-1, 0, 1]);
    let good = RealRootIsolationCertificate {
        intervals: vec![point(q(-1), 1), point(q(1), 1)],
    };
    verify_real_root_isolation(&p, &good, &mut Unbounded).unwrap();
    let mut mutants = Vec::new();
    let mut omitted = good.clone();
    omitted.intervals.pop();
    mutants.push(omitted);
    let mut duplicate = good.clone();
    duplicate.intervals[1] = duplicate.intervals[0].clone();
    mutants.push(duplicate);
    let mut reordered = good.clone();
    reordered.intervals.reverse();
    mutants.push(reordered);
    let mut overlap = good.clone();
    overlap.intervals[0].upper = q(2);
    mutants.push(overlap);
    let mut multiplicity = good.clone();
    multiplicity.intervals[1].multiplicity = 2;
    mutants.push(multiplicity);
    let mut zero_mult = good.clone();
    zero_mult.intervals[1].multiplicity = 0;
    mutants.push(zero_mult);
    let mut tiny = good.clone();
    tiny.intervals[1] = point(
        q(1) + BigRational::new(BigInt::one(), BigInt::from(10).pow(100)),
        1,
    );
    mutants.push(tiny);
    for mutant in mutants {
        assert!(verify_real_root_isolation(&p, &mutant, &mut Unbounded).is_err());
    }
    assert!(verify_real_root_isolation(&poly(&[-2, 0, 1]), &good, &mut Unbounded).is_err());
}

#[test]
fn reference_checker_handles_repeated_roots_at_endpoints() {
    // An ordinary Sturm sequence evaluated at a repeated endpoint has a
    // vanishing gcd factor. The verifier uses square-free layers explicitly.
    let p = product(&product(&poly(&[0, 1]), &poly(&[0, 1])), &poly(&[-2, 1]));
    let cert = RealRootIsolationCertificate {
        intervals: vec![
            RealRootInterval {
                lower: q(0),
                upper: q(1),
                multiplicity: 2,
            },
            point(q(2), 1),
        ],
    };
    verify_real_root_isolation(&p, &cert, &mut Unbounded).unwrap();
}

#[test]
fn factor_indices_are_exact_and_product_bound() {
    let factors = vec![poly(&[-2, 0, 1]), poly(&[-1, 1]), poly(&[1, 0, 1])];
    let p = product(&product(&factors[0], &factors[1]), &factors[2]);
    let roots = isolate_real_roots(&p, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(
        identify_real_root_factors(&roots, &factors, &[1, 1, 1], &mut Unbounded).unwrap(),
        vec![(0, 0), (1, 0), (0, 1)]
    );
    assert!(identify_real_root_factors(&roots, &factors[..2], &[1, 1], &mut Unbounded).is_err());
    assert!(identify_real_root_factors(&roots, &factors, &[2, 1, 1], &mut Unbounded).is_err());
    let repeated = product(&poly(&[-1, 1]), &poly(&[-1, 1]));
    let repeated_roots = isolate_real_roots(&repeated, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(
        identify_real_root_factors(&repeated_roots, &[poly(&[-1, 1])], &[2], &mut Unbounded)
            .unwrap(),
        vec![(0, 0)]
    );
    assert!(
        identify_real_root_factors(
            &repeated_roots,
            &[poly(&[-1, 1]), poly(&[-1, 1])],
            &[1, 1],
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn finite_input_and_precision_envelopes_refuse_without_partial_results() {
    assert!(
        isolate_real_roots(&poly(&[7]), &mut Unbounded, &mut Unbounded)
            .unwrap()
            .intervals()
            .is_empty()
    );
    assert!(matches!(
        isolate_real_roots(&poly(&[0]), &mut Unbounded, &mut Unbounded),
        Err(RealRootError::ZeroPolynomial)
    ));
    assert!(isolate_real_roots(&vec![q(1); 66], &mut Unbounded, &mut Unbounded).is_err());
    let oversized = BigRational::from_integer(BigInt::one() << 4096u32);
    assert!(isolate_real_roots(&[oversized, q(1)], &mut Unbounded, &mut Unbounded).is_err());
    let p = poly(&[-2, 0, 1]);
    for precision in [0, 513, u32::MAX] {
        assert_eq!(
            real_root_decimal(&p, 0, precision, &mut Unbounded, &mut Unbounded),
            Err(RealRootError::InvalidPrecision)
        );
    }
    assert_eq!(
        real_root_decimal(&p, 2, 20, &mut Unbounded, &mut Unbounded),
        Err(RealRootError::InvalidIndex)
    );
    let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        isolate_real_roots(&p, &mut budget, &mut Unbounded),
        Err(RealRootError::Meter(_))
    ));
}

#[derive(Default)]
struct CancelMeter {
    count: usize,
    at: Option<usize>,
}
impl BudgetMeter for CancelMeter {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(amount > 0);
        Ok(())
    }
    fn charge_batch(&mut self, values: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in values {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.count += 1;
        if self.at == Some(self.count) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}
#[test]
fn generation_and_verification_cancel_independently_before_publication() {
    let p = poly(&[0, -2, 0, 1]);
    let (mut generator, mut verifier) = (CancelMeter::default(), CancelMeter::default());
    isolate_real_roots(&p, &mut generator, &mut verifier).unwrap();
    for at in [1, generator.count / 2, generator.count] {
        let mut cancel = CancelMeter {
            count: 0,
            at: Some(at),
        };
        assert!(matches!(
            isolate_real_roots(&p, &mut cancel, &mut Unbounded),
            Err(RealRootError::Meter(MeterError::Cancelled))
        ));
    }
    for at in [1, verifier.count / 2, verifier.count] {
        let mut cancel = CancelMeter {
            count: 0,
            at: Some(at),
        };
        assert!(matches!(
            isolate_real_roots(&p, &mut Unbounded, &mut cancel),
            Err(RealRootError::Meter(MeterError::Cancelled))
        ));
    }
}
