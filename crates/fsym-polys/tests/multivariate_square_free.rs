//! Hand-derived square-free decompositions and adversarial composition proofs.
//! Exact products alone do not prove square-freeness or complete multiplicity.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly as Polynomial, MultivariateSquareFreeCertificate as Certificate,
    MultivariateSquareFreeFactor as Factor, VerifiedMultivariateSquareFree as Checked,
    bounded_polynomial_from_expr, checked_multivariate_gcd,
    checked_multivariate_square_free as checked, verify_multivariate_square_free as verify,
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

// A small independent convolution constructs fixtures from known factors.
// Expected multiplicities are supplied by each test, never production search.
fn product(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let mut terms = BTreeMap::<Vec<u32>, BigRational>::new();
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            let powers = a.iter().zip(b).map(|(a, b)| a + b).collect();
            *terms.entry(powers).or_default() += ca * cb;
        }
    }
    Polynomial::new(left.generators().to_vec(), terms).unwrap()
}

fn power(polynomial: &Polynomial, exponent: usize) -> Polynomial {
    let mut result = scalar(polynomial.generators(), 1, 1);
    for _ in 0..exponent {
        result = product(&result, polynomial);
    }
    result
}

fn factor(polynomial: Polynomial, multiplicity: usize) -> Factor {
    Factor {
        polynomial,
        multiplicity,
    }
}

fn solve(input: &Polynomial) -> Checked {
    checked(input, &mut Unbounded, &mut Unbounded).unwrap()
}

fn rejected(result: &Checked, certificate: &Certificate) {
    assert!(verify(result.input(), certificate, &mut Unbounded).is_err());
}

#[test]
fn nonlinear_factors_retain_exact_multiplicities_and_rational_scale() {
    let generators = ring(&["x", "y"]);
    let a = expression("x*y+1", &generators);
    let b = expression("x+y+1", &generators);
    let input = product(
        &product(&power(&a, 2), &power(&b, 3)),
        &scalar(&generators, -13, 17),
    );
    let result = solve(&input);
    assert_eq!(result.input(), &input);
    assert_eq!(result.scale(), &rational(-13, 17));
    assert_eq!(
        result.factors(),
        &[factor(a.clone(), 2), factor(b.clone(), 3)]
    );
    assert_eq!(result.square_free_part(), &product(&a, &b));
    let replay = verify(&input, result.certificate(), &mut Unbounded).unwrap();
    let (scale, factors, part) = replay.into_parts();
    assert_eq!(&scale, result.scale());
    assert_eq!(factors, result.factors());
    assert_eq!(&part, result.square_free_part());
}

#[test]
fn every_partial_derivative_is_required_including_unused_axes() {
    let generators = ring(&["x", "y", "z"]);
    let y = expression("y+1", &generators);
    let input = product(&power(&y, 4), &scalar(&generators, 5, 7));
    let result = solve(&input);
    assert_eq!(result.factors(), &[factor(y.clone(), 4)]);
    assert_eq!(result.square_free_part(), &y);
    let chain = &result.certificate().derivative_gcds;
    assert_eq!(chain.len(), 3);
    assert!(chain[0].right().is_zero());
    assert_eq!(chain[1].right(), &expression("20/7*(y+1)**3", &generators));
    assert!(chain[2].right().is_zero());

    let input = expression("y**4*(x+1)**2*(z-2)**3", &generators);
    let result = solve(&input);
    assert_eq!(
        result.factors(),
        &[
            factor(expression("x+1", &generators), 2),
            factor(expression("z-2", &generators), 3),
            factor(expression("y", &generators), 4),
        ]
    );
    assert_eq!(
        result.square_free_part(),
        &expression("y*(x+1)*(z-2)", &generators)
    );
}

#[test]
fn equal_multiplicities_are_grouped_without_irreducible_factorization() {
    let generators = ring(&["x", "y"]);
    let a = expression("x+1", &generators);
    let b = expression("y+1", &generators);
    let c = expression("x+y+2", &generators);
    let input = product(&power(&product(&a, &b), 2), &power(&c, 3));
    let result = solve(&input);
    assert_eq!(
        result.factors(),
        &[factor(product(&a, &b), 2), factor(c.clone(), 3)]
    );
    assert_eq!(result.square_free_part(), &product(&product(&a, &b), &c));
}

#[test]
fn high_degree_square_free_component_does_not_require_a_kronecker_factorization() {
    let generators = ring(&["x", "y"]);
    // Linear in y, so this is square-free, while its Kronecker image from the
    // squared input lies beyond the complete factorization degree envelope.
    let component = expression("x**20+y", &generators);
    let input = product(&power(&component, 2), &scalar(&generators, -2, 5));
    let result = solve(&input);
    assert_eq!(result.scale(), &rational(-2, 5));
    assert_eq!(result.factors(), &[factor(component.clone(), 2)]);
    assert_eq!(result.square_free_part(), &component);
}

#[test]
fn square_free_input_remains_one_group_even_when_it_is_reducible() {
    let generators = ring(&["x", "y"]);
    let input = expression("(x+1)*(y+1)", &generators);
    let result = solve(&input);
    assert_eq!(result.factors(), &[factor(input.clone(), 1)]);
    assert_eq!(result.square_free_part(), &input);
    assert_eq!(
        result.certificate().derivative_gcds[0].gcd(),
        &expression("y+1", &generators)
    );
    assert_eq!(
        result.certificate().derivative_gcds[1].gcd(),
        &scalar(&generators, 1, 1)
    );
}

#[test]
fn zero_and_constants_have_unscaled_parts_in_every_declared_ring() {
    let full: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    for generators in [vec![], ring(&["x"]), ring(&["x", "y"]), full] {
        for (numerator, denominator) in [(0, 1), (1, 1), (-5, 7), (13, 2)] {
            let input = scalar(&generators, numerator, denominator);
            let result = solve(&input);
            assert_eq!(result.scale(), &rational(numerator, denominator));
            assert!(result.factors().is_empty());
            assert!(result.certificate().derivative_gcds.is_empty());
            assert_eq!(
                result.square_free_part(),
                &scalar(&generators, i64::from(numerator != 0), 1)
            );
            verify(&input, result.certificate(), &mut Unbounded).unwrap();
        }
    }
}

#[test]
fn monomial_multiplicities_use_the_full_bounded_degree_and_ring() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("m{i}"))).collect();
    let input = p(&generators, &[(&[200, 0, 0, 0, 0, 0, 0, 0], -3, 7)]);
    let x = p(&generators, &[(&[1, 0, 0, 0, 0, 0, 0, 0], 1, 1)]);
    let result = solve(&input);
    assert_eq!(result.factors(), &[factor(x.clone(), 200)]);
    assert_eq!(result.square_free_part(), &x);
    assert_eq!(result.scale(), &rational(-3, 7));
    assert_eq!(result.certificate().derivative_gcds.len(), 8);
}

#[test]
fn exact_same_name_generator_identities_remain_distinct() {
    let a = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([1; 32]));
    let b = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([2; 32]));
    let generators = vec![a, b];
    let left = p(&generators, &[(&[1, 0], 1, 1), (&[0, 0], 1, 1)]);
    let right = p(&generators, &[(&[0, 1], 1, 1), (&[0, 0], 2, 1)]);
    let input = product(&power(&left, 2), &power(&right, 3));
    let result = solve(&input);
    assert_eq!(
        result.factors(),
        &[factor(left.clone(), 2), factor(right.clone(), 3)]
    );
    assert_eq!(result.square_free_part().generators(), &generators);
    let mut changed = result.certificate().clone();
    changed.square_free_part.generators.swap(0, 1);
    rejected(&result, &changed);
    let mut changed = result.certificate().clone();
    changed.factors[0].polynomial.generators[0] = Symbol::new("t");
    rejected(&result, &changed);
    let mut changed_input = input.clone();
    changed_input.generators.swap(0, 1);
    assert!(verify(&changed_input, result.certificate(), &mut Unbounded).is_err());
}

#[test]
fn product_identities_alone_cannot_certify_square_freeness_or_coprimality() {
    let generators = ring(&["x", "y"]);
    let input = expression("x**2*y**2", &generators);
    let result = solve(&input);
    let mut candidate = result.certificate().clone();
    // Both advertised products hold, but this proposed part is not radical.
    candidate.factors = vec![factor(input.clone(), 1)];
    candidate.square_free_part = input;
    rejected(&result, &candidate);

    let input = expression("x**3*y**2", &generators);
    let result = solve(&input);
    let mut candidate = result.certificate().clone();
    // x*(xy)^2 reconstructs f, but the two components share x.
    candidate.factors = vec![
        factor(expression("x", &generators), 1),
        factor(expression("x*y", &generators), 2),
    ];
    candidate.square_free_part = expression("x**2*y", &generators);
    rejected(&result, &candidate);
}

#[test]
fn factor_grouping_units_multiplicity_and_scale_mutations_are_rejected() {
    let generators = ring(&["x", "y"]);
    let a = expression("x+1", &generators);
    let b = expression("y+1", &generators);
    let input = power(&product(&a, &b), 2);
    let result = solve(&input);
    let mut candidate = result.certificate().clone();
    candidate.factors = vec![factor(a, 2), factor(b, 2)];
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate
        .factors
        .insert(0, factor(scalar(&generators, 1, 1), 1));
    rejected(&result, &candidate);
    for multiplicity in [0, 1, 3, usize::MAX] {
        let mut candidate = result.certificate().clone();
        candidate.factors[0].multiplicity = multiplicity;
        rejected(&result, &candidate);
    }
    let mut candidate = result.certificate().clone();
    candidate.factors.clear();
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.scale = rational(2, 1);
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.factors[0].polynomial =
        product(&candidate.factors[0].polynomial, &scalar(&generators, 2, 1));
    candidate.scale = rational(1, 4);
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.square_free_part = scalar(&generators, 0, 1);
    rejected(&result, &candidate);

    let result = solve(&expression("x**2*y**3", &generators));
    let mut candidate = result.certificate().clone();
    candidate.factors.reverse();
    rejected(&result, &candidate);
}

#[test]
fn derivative_chain_binds_exact_original_coefficients_and_every_edge() {
    let generators = ring(&["x", "y", "z"]);
    let input = expression("x**3*y**2", &generators);
    let result = solve(&input);
    let mut candidate = result.certificate().clone();
    candidate.derivative_gcds.pop();
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate
        .derivative_gcds
        .push(Arc::clone(&candidate.derivative_gcds[0]));
    rejected(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.derivative_gcds.swap(0, 1);
    rejected(&result, &candidate);

    // Multiplying a derivative by a unit leaves its GCD unchanged, but its
    // otherwise valid lower receipt must not stand for the original derivative.
    let incorrect = expression("6*x**2*y**2", &generators);
    let receipt =
        checked_multivariate_gcd(&input, &incorrect, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(receipt.gcd(), result.certificate().derivative_gcds[0].gcd());
    let mut candidate = result.certificate().clone();
    candidate.derivative_gcds[0] = Arc::new(receipt);
    rejected(&result, &candidate);

    let incorrect_left = product(&input, &scalar(&generators, 2, 1));
    let correct_right = expression("3*x**2*y**2", &generators);
    let receipt = checked_multivariate_gcd(
        &incorrect_left,
        &correct_right,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    let mut candidate = result.certificate().clone();
    candidate.derivative_gcds[0] = Arc::new(receipt);
    rejected(&result, &candidate);

    let left = result.certificate().derivative_gcds[1].left();
    let receipt = checked_multivariate_gcd(
        left,
        &scalar(&generators, 0, 1),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    let mut candidate = result.certificate().clone();
    candidate.derivative_gcds[1] = Arc::new(receipt);
    rejected(&result, &candidate);

    let altered_source = expression("x**3*y**2+1", &generators);
    assert!(verify(&altered_source, result.certificate(), &mut Unbounded).is_err());
}

#[test]
fn zero_and_scalar_certificates_cannot_claim_factors_or_change_normalization() {
    let generators = ring(&["x", "y"]);
    let zero = solve(&scalar(&generators, 0, 1));
    let mut candidate = zero.certificate().clone();
    candidate.square_free_part = scalar(&generators, 1, 1);
    rejected(&zero, &candidate);
    let mut candidate = zero.certificate().clone();
    candidate.scale = rational(1, 1);
    rejected(&zero, &candidate);
    let mut candidate = zero.certificate().clone();
    candidate
        .factors
        .push(factor(expression("x", &generators), 1));
    rejected(&zero, &candidate);

    let constant = solve(&scalar(&generators, -5, 7));
    let mut candidate = constant.certificate().clone();
    candidate.square_free_part = scalar(&generators, -5, 7);
    rejected(&constant, &candidate);
    let nonconstant = solve(&expression("x**2", &generators));
    let mut candidate = constant.certificate().clone();
    candidate.derivative_gcds = nonconstant.certificate().derivative_gcds.clone();
    rejected(&constant, &candidate);
}

#[test]
fn hand_derived_multiplicity_corpus_preserves_grouping_and_radicals() {
    let generators = ring(&["x", "y"]);
    let known = [
        expression("x", &generators),
        expression("y", &generators),
        expression("x+y+1", &generators),
    ];
    for exponents in [[1, 1, 1], [2, 1, 0], [2, 2, 3], [3, 1, 2], [0, 3, 2]] {
        let mut input = scalar(&generators, -2, 3);
        let mut expected_part = scalar(&generators, 1, 1);
        let mut groups = BTreeMap::new();
        for (polynomial, multiplicity) in known.iter().zip(exponents) {
            input = product(&input, &power(polynomial, multiplicity));
            if multiplicity != 0 {
                expected_part = product(&expected_part, polynomial);
                let group = groups
                    .entry(multiplicity)
                    .or_insert_with(|| scalar(&generators, 1, 1));
                *group = product(group, polynomial);
            }
        }
        let expected: Vec<_> = groups
            .into_iter()
            .map(|(multiplicity, polynomial)| factor(polynomial, multiplicity))
            .collect();
        let result = solve(&input);
        assert_eq!(result.factors(), expected, "multiplicities {exponents:?}");
        assert_eq!(result.square_free_part(), &expected_part);
        assert_eq!(result.scale(), &rational(-2, 3));
    }
}

#[test]
fn malformed_and_oversized_sources_are_refused_before_shortcuts() {
    let generators = ring(&["x", "y"]);
    let mut malformed = scalar(&generators, 0, 1);
    malformed.terms.insert(vec![0], BigRational::one());
    assert!(checked(&malformed, &mut Unbounded, &mut Unbounded).is_err());
    let mut duplicate = scalar(&generators, 0, 1);
    duplicate.generators[1] = duplicate.generators[0].clone();
    assert!(checked(&duplicate, &mut Unbounded, &mut Unbounded).is_err());
    let mut excessive = scalar(&generators, 0, 1);
    excessive.terms.insert(vec![257, 0], BigRational::one());
    assert!(checked(&excessive, &mut Unbounded, &mut Unbounded).is_err());
    let mut excessive = scalar(&generators, 0, 1);
    excessive.terms.insert(
        vec![0, 0],
        BigRational::from_integer(BigInt::one() << 4096u32),
    );
    assert!(checked(&excessive, &mut Unbounded, &mut Unbounded).is_err());
    let mut excessive = scalar(&generators, 0, 1);
    excessive.terms = (0..4097)
        .map(|i| (vec![i / 65, i % 65], BigRational::one()))
        .collect();
    assert!(checked(&excessive, &mut Unbounded, &mut Unbounded).is_err());
    let excessive = Polynomial::zero((0..9).map(|i| Symbol::new(format!("x{i}"))).collect());
    assert!(checked(&excessive, &mut Unbounded, &mut Unbounded).is_err());
    let long = Polynomial::zero(vec![Symbol::new("x".repeat(65_537))]);
    assert!(checked(&long, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn auxiliary_ring_limit_applies_only_when_nonconstant_gcd_search_needs_it() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let input = p(
        &generators,
        &[
            (&[1, 0, 0, 0, 0, 0, 0, 0], 3, 1),
            (&[0, 0, 0, 0, 0, 0, 0, 1], 5, 1),
        ],
    );
    let expected = p(
        &generators,
        &[
            (&[1, 0, 0, 0, 0, 0, 0, 0], 1, 1),
            (&[0, 0, 0, 0, 0, 0, 0, 1], 5, 3),
        ],
    );
    assert_eq!(solve(&input).factors(), &[factor(expected, 1)]);
    let input = p(
        &generators,
        &[
            (&[2, 0, 0, 0, 0, 0, 0, 0], 1, 1),
            (&[0, 0, 0, 0, 0, 0, 0, 1], 1, 1),
        ],
    );
    let error = checked(&input, &mut Unbounded, &mut Unbounded).unwrap_err();
    assert!(error.to_string().contains("auxiliary"));
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
fn generation_verification_and_publication_share_separate_cumulative_budgets() {
    let generators = ring(&["x", "y"]);
    let input = expression("x**3*y**2", &generators);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    let accepted = checked(&input, &mut generation, &mut verification).unwrap();
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
                checked(&input, &mut Unbounded, &mut cancelled)
            } else {
                checked(&input, &mut cancelled, &mut Unbounded)
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
                checked(&input, &mut Unbounded, &mut insufficient)
            } else {
                checked(&input, &mut insufficient, &mut Unbounded)
            };
            assert!(result.is_err());
        }
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(verify(&input, accepted.certificate(), &mut empty).is_err());
    let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = generation.steps;
    let mut cumulative = Budget::new(limits);
    checked(&input, &mut cumulative, &mut Unbounded).unwrap();
    assert!(checked(&input, &mut cumulative, &mut Unbounded).is_err());
}

#[test]
fn replay_cancellation_is_terminal_at_every_observed_checkpoint() {
    let generators = ring(&["x", "y"]);
    let input = expression("x**2*y", &generators);
    let result = solve(&input);
    let mut measurement = Meter::default();
    verify(&input, result.certificate(), &mut measurement).unwrap();
    for cancel_at in 1..=measurement.checkpoints {
        let mut cancelled = Meter {
            cancel_at: Some(cancel_at),
            ..Meter::default()
        };
        let error = verify(&input, result.certificate(), &mut cancelled).unwrap_err();
        assert!(error.to_string().to_lowercase().contains("cancel"));
        assert_eq!(cancelled.checkpoints, cancel_at);
    }
}
