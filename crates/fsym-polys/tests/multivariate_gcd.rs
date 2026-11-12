//! Hand-derived QQ GCD/LCM identities and adversarial composition receipts.
//! Complete principal intersections must be checked: an exact product or a
//! common divisor does not establish greatestness.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariateGcdCertificate as Certificate, MultivariatePoly as Polynomial,
    VerifiedMultivariateGcd as Checked, bounded_polynomial_from_expr,
    checked_ideal_intersection as intersection, checked_ideal_saturation as saturation,
    checked_multivariate_gcd as checked, verify_multivariate_gcd as verify,
};
use num_traits::One;
use std::collections::BTreeMap;
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn p(generators: &[Symbol], entries: &[(&[u32], i64, i64)]) -> Polynomial {
    Polynomial::new(
        generators.to_vec(),
        entries
            .iter()
            .map(|(powers, numerator, denominator)| {
                (powers.to_vec(), rational(*numerator, *denominator))
            })
            .collect(),
    )
    .unwrap()
}

fn expression(source: &str, generators: &[Symbol]) -> Polynomial {
    bounded_polynomial_from_expr(&parse(source).unwrap(), generators, &mut Unbounded).unwrap()
}

fn scalar(generators: &[Symbol], numerator: i64, denominator: i64) -> Polynomial {
    p(
        generators,
        &[(&vec![0; generators.len()], numerator, denominator)],
    )
}

fn solve(left: &Polynomial, right: &Polynomial) -> Checked {
    checked(left, right, &mut Unbounded, &mut Unbounded).unwrap()
}

// Independent small-fixture convolution. Expected factors and multiplicities
// are specified by each test, never inferred with a production GCD algorithm.
fn product(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let mut terms = BTreeMap::<Vec<u32>, BigRational>::new();
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            let powers: Vec<_> = a.iter().zip(b).map(|(a, b)| a + b).collect();
            *terms.entry(powers).or_default() += ca * cb;
        }
    }
    Polynomial::new(left.generators().to_vec(), terms).unwrap()
}

fn rejected(result: &Checked, certificate: &Certificate) {
    assert!(verify(result.left(), result.right(), certificate, &mut Unbounded).is_err());
}

#[test]
fn repeated_nonlinear_common_factors_retain_complete_multiplicities() {
    let generators = ring(&["x", "y"]);
    let h = expression("x*y+1", &generators);
    let a = expression("x+1", &generators);
    let b = expression("y+1", &generators);
    let h2 = product(&h, &h);
    let left = product(&h2, &a);
    let right = product(&h, &b);
    let result = solve(&left, &right);
    assert_eq!(result.gcd(), &h);
    assert_eq!(result.lcm(), &product(&product(&h2, &a), &b));
    assert_eq!(result.left(), &left);
    assert_eq!(result.right(), &right);
    let replay = verify(&left, &right, result.certificate(), &mut Unbounded).unwrap();
    assert_eq!(replay.gcd(), &h);
    let (gcd, lcm) = replay.into_parts();
    assert_eq!((&gcd, &lcm), (result.gcd(), result.lcm()));
    let reversed = solve(&right, &left);
    assert_eq!(reversed.gcd(), result.gcd());
    assert_eq!(reversed.lcm(), result.lcm());
}

#[test]
fn exact_rational_units_and_nonlinear_factors_are_normalized() {
    let generators = ring(&["x", "y"]);
    let h = expression("x*y+2/3", &generators);
    let a = expression("x+1/2", &generators);
    let b = expression("y-3/7", &generators);
    let expected_lcm = product(&product(&h, &a), &b);
    for (un, ud, vn, vd) in [(1, 1, 1, 1), (-2, 3, 5, 7), (13, 11, -17, 19)] {
        let left = product(&product(&h, &a), &scalar(&generators, un, ud));
        let right = product(&product(&h, &b), &scalar(&generators, vn, vd));
        let result = solve(&left, &right);
        assert_eq!(result.gcd(), &h);
        assert_eq!(result.lcm(), &expected_lcm);
    }
}

#[test]
fn gcd_does_not_require_factoring_unrelated_high_degree_cofactors() {
    let generators = ring(&["x", "y"]);
    // The cofactors differ by one, so they are coprime. Their Kronecker images
    // exceed the complete-factorization degree bound, which is irrelevant to
    // the original GCD problem and must not be a prerequisite for this lane.
    let h = expression("x+y", &generators);
    let a = expression("x**10+y**10+1", &generators);
    let b = expression("x**10+y**10+2", &generators);
    let left = product(&h, &a);
    let right = product(&h, &b);
    let result = solve(&left, &right);
    assert_eq!(result.gcd(), &h);
    assert_eq!(result.lcm(), &product(&product(&h, &a), &b));
}

#[test]
fn product_verification_need_not_materialize_a_twice_degree_polynomial() {
    let generators = ring(&["x", "y"]);
    let left = p(&generators, &[(&[200, 0], -3, 7)]);
    let right = p(&generators, &[(&[100, 0], 5, 11)]);
    let result = solve(&left, &right);
    assert_eq!(result.gcd(), &p(&generators, &[(&[100, 0], 1, 1)]));
    assert_eq!(result.lcm(), &p(&generators, &[(&[200, 0], 1, 1)]));
}

#[test]
fn identical_polynomials_preserve_all_factors_and_their_multiplicities() {
    let generators = ring(&["x", "y"]);
    let expected = expression("(x*y+1)**2", &generators);
    let source = product(&expected, &scalar(&generators, -13, 17));
    let result = solve(&source, &source);
    assert_eq!(result.gcd(), &expected);
    assert_eq!(result.lcm(), &expected);
    assert!(result.certificate().intersection.is_some());
}

#[test]
fn monomial_corpus_obeys_componentwise_minimum_and_maximum() {
    let generators = ring(&["x", "y"]);
    for a in [[0, 0], [1, 0], [0, 2], [2, 1], [3, 2]] {
        for b in [[0, 0], [0, 1], [2, 0], [1, 3]] {
            let left = p(&generators, &[(&a, -2, 3)]);
            let right = p(&generators, &[(&b, 5, 7)]);
            let result = solve(&left, &right);
            let minimum = [a[0].min(b[0]), a[1].min(b[1])];
            let maximum = [a[0].max(b[0]), a[1].max(b[1])];
            assert_eq!(result.gcd(), &p(&generators, &[(&minimum, 1, 1)]));
            assert_eq!(result.lcm(), &p(&generators, &[(&maximum, 1, 1)]));
        }
    }
}

#[test]
fn zero_and_scalars_preserve_empty_and_full_declared_rings() {
    let full: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    for generators in [vec![], ring(&["x", "y"]), full] {
        let zero = scalar(&generators, 0, 1);
        let unit = scalar(&generators, -5, 7);
        let monic = if generators.is_empty() {
            scalar(&generators, 1, 1)
        } else {
            let mut powers = vec![0; generators.len()];
            powers[generators.len() - 1] = 2;
            p(
                &generators,
                &[(&powers, 1, 1), (&vec![0; generators.len()], 2, 3)],
            )
        };
        let scaled = product(&monic, &unit);
        let both = solve(&zero, &zero);
        assert_eq!(both.gcd(), &zero);
        assert_eq!(both.lcm(), &zero);
        for (left, right) in [(&zero, &scaled), (&scaled, &zero)] {
            let result = solve(left, right);
            assert_eq!(result.gcd(), &monic);
            assert_eq!(result.lcm(), &zero);
            assert!(result.certificate().intersection.is_none());
        }
        for (left, right) in [(&unit, &scaled), (&scaled, &unit)] {
            let result = solve(left, right);
            assert_eq!(result.gcd(), &scalar(&generators, 1, 1));
            assert_eq!(result.lcm(), &monic);
            assert!(result.certificate().intersection.is_none());
        }
    }
}

#[test]
fn common_divisor_and_reconstructed_product_do_not_establish_greatestness() {
    let generators = ring(&["x", "y"]);
    let left = expression("x**2*y", &generators);
    let right = expression("x*y**2", &generators);
    let result = solve(&left, &right);
    let mut candidate = result.certificate().clone();
    candidate.gcd = scalar(&generators, 1, 1);
    candidate.lcm = product(&left, &right);
    rejected(&result, &candidate); // Reconstructs the exact input product.
    candidate.intersection = None;
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.gcd = scalar(&generators, 0, 1);
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.gcd = product(result.gcd(), &scalar(&generators, 2, 3));
    candidate.lcm = product(result.lcm(), &scalar(&generators, 3, 2));
    rejected(&result, &candidate); // Exact product, wrong normalizations.
    let mut candidate = result.certificate().clone();
    candidate.gcd = expression("x*y+1", &generators);
    rejected(&result, &candidate); // Correct leading term is insufficient.
}

#[test]
fn intersection_evidence_binds_whole_original_inputs_and_operation() {
    let generators = ring(&["x", "y"]);
    let left = expression("x**2*y", &generators);
    let right = expression("x*y**2", &generators);
    let result = solve(&left, &right);
    let unrelated = [
        intersection(
            std::slice::from_ref(&right),
            std::slice::from_ref(&left),
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        intersection(
            &[product(&left, &scalar(&generators, 2, 3))],
            std::slice::from_ref(&right),
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        intersection(
            &[left.clone(), left.clone()],
            std::slice::from_ref(&right),
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
        saturation(
            std::slice::from_ref(&left),
            &right,
            &generators,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap(),
    ];
    for receipt in unrelated {
        let mut candidate = result.certificate().clone();
        candidate.intersection = Some(Arc::new(receipt));
        rejected(&result, &candidate);
    }
}

#[test]
fn zero_and_scalar_mutants_cannot_erase_or_renormalize_nonzero_inputs() {
    let generators = ring(&["x", "y"]);
    let zero = scalar(&generators, 0, 1);
    let one = scalar(&generators, 1, 1);
    let f = expression("2*x/3+4*y/5", &generators);
    let result = solve(&zero, &f);
    let mut candidate = result.certificate().clone();
    candidate.gcd = f.clone();
    rejected(&result, &candidate);
    candidate.gcd = one.clone();
    rejected(&result, &candidate);
    candidate = result.certificate().clone();
    candidate.lcm = one.clone();
    rejected(&result, &candidate);
    let result = solve(&one, &f);
    let mut candidate = result.certificate().clone();
    candidate.gcd = zero.clone();
    rejected(&result, &candidate);
    candidate = result.certificate().clone();
    candidate.lcm = f;
    rejected(&result, &candidate);
    let result = solve(&zero, &zero);
    let mut candidate = result.certificate().clone();
    candidate.gcd = one;
    rejected(&result, &candidate);
    let nontrivial = solve(&expression("x", &generators), &expression("y", &generators));
    let mut candidate = result.certificate().clone();
    candidate.intersection = nontrivial.certificate().intersection.clone();
    rejected(&result, &candidate);
}

#[test]
fn typed_generator_order_and_same_name_identities_remain_distinct() {
    let generators = ring(&["x", "y"]);
    let left = expression("x*y", &generators);
    let right = expression("x", &generators);
    let result = solve(&left, &right);
    let mut dummy = generators.clone();
    dummy[0] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([101; 32]));
    for altered in [
        p(
            &[generators[1].clone(), generators[0].clone()],
            &[(&[1, 0], 1, 1)],
        ),
        p(&dummy, &[(&[1, 0], 1, 1)]),
    ] {
        assert!(checked(&left, &altered, &mut Unbounded, &mut Unbounded).is_err());
        assert!(verify(&left, &altered, result.certificate(), &mut Unbounded).is_err());
        let mut candidate = result.certificate().clone();
        candidate.gcd = altered;
        rejected(&result, &candidate);
    }
    // Same printed name with distinct identities is a valid two-axis ring.
    let generators = vec![generators[0].clone(), dummy[0].clone()];
    let left = p(&generators, &[(&[2, 1], 1, 1)]);
    let right = p(&generators, &[(&[1, 2], 1, 1)]);
    let result = solve(&left, &right);
    assert_eq!(result.gcd(), &p(&generators, &[(&[1, 1], 1, 1)]));
    assert_eq!(result.lcm(), &p(&generators, &[(&[2, 2], 1, 1)]));
}

#[test]
fn auxiliary_variables_and_ambient_degree_bounds_remain_explicit() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let a = p(&generators, &[(&[1, 0, 0, 0, 0, 0, 0, 0], 1, 1)]);
    let b = p(&generators, &[(&[0, 0, 0, 0, 0, 0, 0, 1], 1, 1)]);
    let error = checked(&a, &b, &mut Unbounded, &mut Unbounded).unwrap_err();
    assert!(error.to_string().contains("auxiliary"));
    let generators = &generators[..7];
    let a = p(generators, &[(&[1, 0, 0, 0, 0, 0, 0], 1, 1)]);
    let b = p(generators, &[(&[0, 0, 0, 0, 0, 0, 1], 1, 1)]);
    let result = solve(&a, &b);
    assert_eq!(result.gcd(), &scalar(generators, 1, 1));
    assert_eq!(result.lcm(), &product(&a, &b));
    let generators = ring(&["x", "y"]);
    let a = p(&generators, &[(&[256, 0], 1, 1)]);
    let b = p(&generators, &[(&[0, 1], 1, 1)]);
    assert!(checked(&a, &b, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn malformed_and_oversized_inputs_are_refused_even_for_trivial_results() {
    let generators = ring(&["x", "y"]);
    let zero = scalar(&generators, 0, 1);
    let oversized = Polynomial::new(
        generators.clone(),
        [(
            vec![1, 0],
            BigRational::from_integer(BigInt::one() << 4096u32),
        )]
        .into_iter()
        .collect(),
    )
    .unwrap();
    assert!(checked(&zero, &oversized, &mut Unbounded, &mut Unbounded).is_err());
    let degree = p(&generators, &[(&[257, 0], 1, 1)]);
    assert!(checked(&zero, &degree, &mut Unbounded, &mut Unbounded).is_err());
    // Public zero constructors can represent rings requiring admission.
    let duplicate = Polynomial::zero(vec![generators[0].clone(), generators[0].clone()]);
    assert!(checked(&duplicate, &duplicate, &mut Unbounded, &mut Unbounded).is_err());
    let wide = Polynomial::zero((0..9).map(|i| Symbol::new(format!("x{i}"))).collect());
    assert!(checked(&wide, &wide, &mut Unbounded, &mut Unbounded).is_err());
    let names = Polynomial::zero(vec![Symbol::new("x".repeat(65_537))]);
    assert!(checked(&names, &names, &mut Unbounded, &mut Unbounded).is_err());
    let terms: BTreeMap<_, _> = (0..4097)
        .map(|i| (vec![i / 65, i % 65], BigRational::one()))
        .collect();
    let large = Polynomial::new(generators, terms).unwrap();
    assert!(checked(&zero, &large, &mut Unbounded, &mut Unbounded).is_err());
}

#[derive(Default)]
struct Meter {
    steps: u64,
    memory: u64,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.steps += amount;
        } else if dimension == Dimension::MemoryBytes {
            self.memory += amount;
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
fn generation_verification_and_publication_respect_cancellation_and_separate_budgets() {
    let generators = ring(&["x", "y"]);
    let left = expression("x**2*y", &generators);
    let right = expression("x*y**2", &generators);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    let accepted = checked(&left, &right, &mut generation, &mut verification).unwrap();
    assert!(generation.steps > 0 && verification.steps > 0);
    assert!(generation.memory > 0 && verification.memory > 0);
    for phase in [false, true] {
        let count = if phase {
            verification.checkpoints
        } else {
            generation.checkpoints
        };
        for cancel_at in [1, count / 2, count] {
            let mut cancelled = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let error = if phase {
                checked(&left, &right, &mut Unbounded, &mut cancelled)
            } else {
                checked(&left, &right, &mut cancelled, &mut Unbounded)
            }
            .unwrap_err();
            assert!(error.to_string().to_lowercase().contains("cancel"));
            assert_eq!(cancelled.checkpoints, cancel_at);
        }
        for dimension in [Dimension::ComputeSteps, Dimension::MemoryBytes] {
            let consumed = match (phase, dimension) {
                (true, Dimension::ComputeSteps) => verification.steps,
                (true, _) => verification.memory,
                (false, Dimension::ComputeSteps) => generation.steps,
                (false, _) => generation.memory,
            };
            let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
            limits.dimensions[dimension.index()] = consumed - 1;
            let mut insufficient = Budget::new(limits);
            let result = if phase {
                checked(&left, &right, &mut Unbounded, &mut insufficient)
            } else {
                checked(&left, &right, &mut insufficient, &mut Unbounded)
            };
            assert!(result.is_err());
        }
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(verify(&left, &right, accepted.certificate(), &mut empty).is_err());
    // A second request cannot reset the caller's already consumed ledger.
    let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = generation.steps;
    let mut budget = Budget::new(limits);
    checked(&left, &right, &mut budget, &mut Unbounded).unwrap();
    assert!(checked(&left, &right, &mut budget, &mut Unbounded).is_err());
}

#[test]
fn replay_cancellation_is_terminal_at_every_observed_checkpoint() {
    let generators = ring(&["x", "y"]);
    let left = expression("x**2*y", &generators);
    let right = expression("x*y**2", &generators);
    let accepted = solve(&left, &right);
    let mut measurement = Meter::default();
    verify(&left, &right, accepted.certificate(), &mut measurement).unwrap();
    for cancel_at in 1..=measurement.checkpoints {
        let mut meter = Meter {
            cancel_at: Some(cancel_at),
            ..Meter::default()
        };
        let error = verify(&left, &right, accepted.certificate(), &mut meter).unwrap_err();
        assert!(error.to_string().to_lowercase().contains("cancel"));
        assert_eq!(meter.checkpoints, cancel_at);
    }
}
