//! Exact domain-restricted root claims and adversarial acceptance tests.
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::real_roots::{
    RealRootError, RealRootInterval, RealRootIsolationCertificate, RealRootQuery,
    isolate_real_roots_in_interval, refine_real_root_interval, verify_real_root_query,
};
use fsym_core::{BigInt, BigRational};
use num_traits::Zero;

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}

fn polynomial(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|&value| q(value)).collect()
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

fn query(lower: Option<i64>, upper: Option<i64>) -> RealRootQuery {
    RealRootQuery {
        lower: lower.map(q),
        upper: upper.map(q),
        max_width: None,
        require_square_free: false,
    }
}

fn point(value: i64, multiplicity: usize) -> RealRootInterval {
    RealRootInterval {
        lower: q(value),
        upper: q(value),
        multiplicity,
    }
}

#[test]
fn complete_closed_queries_count_distinct_roots_and_keep_original_multiplicity() {
    let mut p = polynomial(&[-2, 0, 1]);
    for factor in [
        polynomial(&[2, 1]),
        polynomial(&[2, 1]),
        polynomial(&[-1, 1]),
        polynomial(&[-1, 1]),
        polynomial(&[-1, 1]),
    ] {
        p = product(&p, &factor);
    }
    let request = query(Some(-2), Some(1));
    let roots = isolate_real_roots_in_interval(&p, &request, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(roots.distinct_count(), 3);
    assert_eq!(roots.intervals()[0], point(-2, 2));
    assert_eq!(roots.intervals()[1].multiplicity, 1);
    assert_eq!(roots.intervals()[2], point(1, 3));
    assert_eq!(roots.query(), &request);
    assert_eq!(roots.coefficients(), p);
    let exact = isolate_real_roots_in_interval(
        &p,
        &query(Some(1), Some(1)),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(exact.intervals(), &[point(1, 3)]);
}

#[test]
fn half_lines_empty_windows_constants_and_nonreal_pairs() {
    let p = polynomial(&[-2, 0, 1]);
    for request in [query(None, Some(0)), query(Some(0), None)] {
        assert_eq!(
            isolate_real_roots_in_interval(&p, &request, &mut Unbounded, &mut Unbounded)
                .unwrap()
                .distinct_count(),
            1
        );
    }
    for (p, request) in [
        (p.clone(), query(Some(3), None)),
        (p, query(None, Some(-3))),
        (polynomial(&[7]), query(None, None)),
        (polynomial(&[1, 0, 1]), query(None, None)),
    ] {
        assert!(
            isolate_real_roots_in_interval(&p, &request, &mut Unbounded, &mut Unbounded)
                .unwrap()
                .intervals()
                .is_empty()
        );
    }
    assert!(matches!(
        isolate_real_roots_in_interval(
            &polynomial(&[0]),
            &query(None, None),
            &mut Unbounded,
            &mut Unbounded,
        ),
        Err(RealRootError::ZeroPolynomial)
    ));
}

#[test]
fn refinement_precision_is_exact_beyond_binary64() {
    let epsilon = BigRational::new(BigInt::one(), BigInt::from(10).pow(40));
    let request = RealRootQuery {
        max_width: Some(epsilon.clone()),
        ..query(None, None)
    };
    let roots = isolate_real_roots_in_interval(
        &polynomial(&[-2, 0, 1]),
        &request,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(roots.distinct_count(), 2);
    for root in roots.intervals() {
        assert!(&root.upper - &root.lower <= epsilon);
        let lower_square = &root.lower * &root.lower;
        let upper_square = &root.upper * &root.upper;
        assert!(lower_square.clone().min(upper_square.clone()) <= q(2));
        assert!(lower_square.max(upper_square) >= q(2));
    }
}

#[test]
fn close_roots_and_rational_boundary_are_not_merged() {
    let delta = BigRational::new(BigInt::one(), BigInt::from(10).pow(20));
    let other = q(1) + delta;
    let p = product(&polynomial(&[-1, 1]), &[-other, q(1)]);
    let all = isolate_real_roots_in_interval(
        &p,
        &query(None, None),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(all.distinct_count(), 2);
    let one = isolate_real_roots_in_interval(
        &p,
        &query(Some(1), Some(1)),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(one.intervals(), &[point(1, 1)]);
}

#[test]
fn square_free_obligation_covers_repeated_nonreal_roots_too() {
    let request = RealRootQuery {
        require_square_free: true,
        ..query(None, None)
    };
    let nonreal = polynomial(&[1, 0, 1]);
    assert!(
        isolate_real_roots_in_interval(
            &product(&nonreal, &nonreal),
            &request,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        isolate_real_roots_in_interval(&nonreal, &request, &mut Unbounded, &mut Unbounded).is_ok()
    );
}

#[test]
fn checker_rejects_missing_duplicate_outside_and_wrong_multiplicity_roots() {
    let p = polynomial(&[-1, 0, 1]);
    let request = query(Some(-2), Some(2));
    let good = RealRootIsolationCertificate {
        intervals: vec![point(-1, 1), point(1, 1)],
    };
    verify_real_root_query(&p, &request, &good, &mut Unbounded).unwrap();
    let mutants = [
        RealRootIsolationCertificate { intervals: vec![point(1, 1)] },
        RealRootIsolationCertificate { intervals: vec![point(1, 1), point(1, 1)] },
        RealRootIsolationCertificate { intervals: vec![point(-1, 2), point(1, 1)] },
        RealRootIsolationCertificate { intervals: vec![point(-2, 1), point(1, 1)] },
    ];
    for mutant in mutants {
        assert!(verify_real_root_query(&p, &request, &mutant, &mut Unbounded).is_err());
    }
    assert!(
        verify_real_root_query(&polynomial(&[-2, 0, 1]), &request, &good, &mut Unbounded).is_err()
    );
    assert!(
        verify_real_root_query(&p, &query(Some(0), Some(2)), &good, &mut Unbounded).is_err()
    );
}

#[test]
fn checker_enforces_interval_width_and_closed_domain() {
    let p = polynomial(&[-2, 0, 1]);
    let certificate = RealRootIsolationCertificate {
        intervals: vec![RealRootInterval { lower: q(1), upper: q(2), multiplicity: 1 }],
    };
    let mut request = query(Some(0), Some(2));
    verify_real_root_query(&p, &request, &certificate, &mut Unbounded).unwrap();
    request.max_width = Some(BigRational::new(1.into(), 10.into()));
    assert!(verify_real_root_query(&p, &request, &certificate, &mut Unbounded).is_err());
    request.max_width = None;
    request.lower = Some(BigRational::new(5.into(), 4.into()));
    assert!(verify_real_root_query(&p, &request, &certificate, &mut Unbounded).is_err());
}

#[test]
fn supplied_isolators_refine_by_steps_and_by_precision() {
    let p = polynomial(&[-2, 0, 1]);
    let request = query(Some(1), Some(2));
    let one = refine_real_root_interval(&p, &request, None, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(one.intervals()[0].lower, q(1));
    assert_eq!(one.intervals()[0].upper, BigRational::new(3.into(), 2.into()));
    let precise = RealRootQuery {
        max_width: Some(BigRational::new(1.into(), 1000.into())),
        ..request.clone()
    };
    let unchanged =
        refine_real_root_interval(&p, &precise, Some(0), &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(unchanged.intervals()[0].lower, q(1));
    assert_eq!(unchanged.intervals()[0].upper, q(2));
    assert_eq!(unchanged.query().max_width, None);
    let narrowed =
        refine_real_root_interval(&p, &precise, None, &mut Unbounded, &mut Unbounded).unwrap();
    assert!(
        &narrowed.intervals()[0].upper - &narrowed.intervals()[0].lower
            <= *precise.max_width.as_ref().unwrap()
    );
}

#[test]
fn supplied_isolator_must_contain_exactly_one_distinct_root() {
    let p = polynomial(&[-1, 0, 1]);
    for request in [
        query(Some(-2), Some(2)),
        query(Some(2), Some(3)),
        query(Some(0), Some(0)),
        query(Some(2), Some(1)),
        query(None, Some(2)),
    ] {
        assert!(
            refine_real_root_interval(&p, &request, None, &mut Unbounded, &mut Unbounded).is_err()
        );
    }
    let repeated = product(&polynomial(&[-1, 1]), &polynomial(&[-1, 1]));
    let root = refine_real_root_interval(
        &repeated,
        &query(Some(1), Some(1)),
        None,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(root.intervals(), &[point(1, 2)]);
}

#[test]
fn oversized_requests_and_nonpositive_precision_are_terminal() {
    let p = polynomial(&[-2, 0, 1]);
    for width in [q(0), q(-1)] {
        let request = RealRootQuery { max_width: Some(width), ..query(None, None) };
        assert!(
            isolate_real_roots_in_interval(&p, &request, &mut Unbounded, &mut Unbounded).is_err()
        );
    }
    assert!(
        refine_real_root_interval(
            &p,
            &query(Some(1), Some(2)),
            Some(32_769),
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let request = RealRootQuery {
        lower: Some(BigRational::from_integer(BigInt::one() << 4096u32)),
        ..query(None, None)
    };
    assert!(matches!(
        isolate_real_roots_in_interval(&p, &request, &mut Unbounded, &mut Unbounded),
        Err(RealRootError::ResourceLimit(_))
    ));
}

struct Cancelled;
impl BudgetMeter for Cancelled {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn exhaustion_and_both_phase_cancellations_publish_no_query() {
    let p = polynomial(&[-2, 0, 1]);
    let request = query(None, None);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        isolate_real_roots_in_interval(&p, &request, &mut tiny, &mut Unbounded),
        Err(RealRootError::Meter(_))
    ));
    assert!(matches!(
        isolate_real_roots_in_interval(&p, &request, &mut Cancelled, &mut Unbounded),
        Err(RealRootError::Meter(MeterError::Cancelled))
    ));
    assert!(matches!(
        isolate_real_roots_in_interval(&p, &request, &mut Unbounded, &mut Cancelled),
        Err(RealRootError::Meter(MeterError::Cancelled))
    ));
}
