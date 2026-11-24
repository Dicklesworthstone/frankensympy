//! Hand-derived prime-field square-free multiplicities and adversarial receipts.
//! Product equality does not certify square-free groups or their completeness.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateSquareFreeCertificate as Certificate,
    PrimeFieldMultivariateSquareFreeFactor as Factor, PrimeFieldMultivariateTerms as Terms,
    VerifiedPrimeFieldMultivariateSquareFree as Checked,
    checked_prime_field_multivariate_gcd as checked_gcd,
    checked_prime_field_multivariate_square_free as checked,
    verify_prime_field_multivariate_square_free as verify,
};
use std::collections::BTreeMap;
use std::sync::Arc;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn p(generators: &[Symbol], prime: u64, entries: &[(&[u32], u64)]) -> Polynomial {
    Polynomial::new(
        generators,
        prime,
        &entries
            .iter()
            .map(|(powers, value)| (powers.to_vec(), *value % prime))
            .collect(),
    )
    .unwrap()
}

fn scalar(generators: &[Symbol], prime: u64, value: u64) -> Polynomial {
    p(generators, prime, &[(&vec![0; generators.len()], value)])
}

// Fixture convolution is independent of production arithmetic and search.
// These tiny known products supply expected factors and exponents directly.
fn product(left: &Polynomial, right: &Polynomial) -> Polynomial {
    assert_eq!(left.generators(), right.generators());
    assert_eq!(left.prime(), right.prime());
    let mut columns = BTreeMap::<Vec<u32>, u128>::new();
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            let powers = a.iter().zip(b).map(|(a, b)| a + b).collect();
            *columns.entry(powers).or_default() += u128::from(*ca) * u128::from(*cb);
        }
    }
    let terms = columns
        .into_iter()
        .map(|(powers, value)| (powers, (value % u128::from(left.prime())) as u64))
        .collect();
    Polynomial::new(left.generators(), left.prime(), &terms).unwrap()
}

fn power(input: &Polynomial, exponent: usize) -> Polynomial {
    let mut result = scalar(input.generators(), input.prime(), 1);
    for _ in 0..exponent {
        result = product(&result, input);
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

fn rejected(input: &Polynomial, certificate: &Certificate) {
    assert!(verify(input, certificate, &mut Unbounded).is_err());
}

fn partial(input: &Polynomial, axis: usize) -> Polynomial {
    let mut terms = Terms::new();
    for (powers, value) in input.terms() {
        let coefficient =
            (u128::from(*value) * u128::from(powers[axis]) % u128::from(input.prime())) as u64;
        if coefficient != 0 {
            let mut powers = powers.clone();
            powers[axis] -= 1;
            terms.insert(powers, coefficient);
        }
    }
    Polynomial::new(input.generators(), input.prime(), &terms).unwrap()
}

// This builds only the lower evidence: the decomposition and claimed radical
// are supplied by the test, never obtained from the square-free generator.
fn hand_certificate(unit: u64, factors: Vec<Factor>, part: Polynomial) -> Certificate {
    let mut common = part.clone();
    let mut receipts = Vec::new();
    for axis in 0..part.generators().len() {
        let receipt = checked_gcd(
            &common,
            &partial(&part, axis),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        common = receipt.gcd().clone();
        receipts.push(Arc::new(receipt));
    }
    Certificate {
        unit,
        factors,
        square_free_part: part,
        radical_derivative_gcds: receipts,
    }
}

#[test]
fn nonlinear_components_preserve_units_and_exact_characteristic_multiplicities() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 65521] {
        let a = p(&generators, prime, &[(&[1, 1], 1), (&[0, 0], 1)]);
        let b = p(
            &generators,
            prime,
            &[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], 1)],
        );
        let input = product(
            &product(&power(&a, 2), &power(&b, 3)),
            &scalar(&generators, prime, prime - 1),
        );
        let result = solve(&input);
        assert_eq!(result.unit(), prime - 1, "prime {prime}");
        assert_eq!(result.input(), &input);
        assert_eq!(
            result.factors(),
            &[factor(a.clone(), 2), factor(b.clone(), 3)],
            "prime {prime}",
        );
        assert_eq!(result.square_free_part(), &product(&a, &b));
        let replay = verify(&input, result.certificate(), &mut Unbounded).unwrap();
        let (unit, groups, part) = replay.into_parts();
        assert_eq!(unit, result.unit());
        assert_eq!(groups, result.factors());
        assert_eq!(&part, result.square_free_part());
    }
}

#[test]
fn pure_frobenius_powers_recover_the_root_and_all_p_adic_multiplicity() {
    let generators = ring(&["x", "y"]);
    for (prime, exponents) in [(2, vec![2, 4, 8, 64]), (3, vec![3, 9, 27]), (5, vec![5, 25])] {
        let root = p(
            &generators,
            prime,
            &[(&[1, 0], 1), (&[0, 1], prime - 1), (&[0, 0], 1)],
        );
        for exponent in exponents {
            let input = product(
                &power(&root, exponent),
                &scalar(&generators, prime, prime - 1),
            );
            let result = solve(&input);
            assert_eq!(
                result.factors(),
                &[factor(root.clone(), exponent)],
                "prime {prime}, exponent {exponent}",
            );
            assert_eq!(result.unit(), prime - 1);
            assert_eq!(result.square_free_part(), &root);
        }
    }
}

#[test]
fn mixed_separable_and_inseparable_layers_preserve_sorted_distinct_groups() {
    let generators = ring(&["x", "y"]);
    for (prime, exponents) in [(2, [1, 2, 4]), (3, [2, 3, 9])] {
        let known = [
            p(&generators, prime, &[(&[1, 0], 1)]),
            p(&generators, prime, &[(&[0, 1], 1)]),
            p(
                &generators,
                prime,
                &[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], 1)],
            ),
        ];
        let mut input = scalar(&generators, prime, prime - 1);
        let mut part = scalar(&generators, prime, 1);
        for (polynomial, exponent) in known.iter().zip(exponents) {
            input = product(&input, &power(polynomial, exponent));
            part = product(&part, polynomial);
        }
        let expected: Vec<_> = known.into_iter().zip(exponents)
            .map(|(polynomial, exponent)| factor(polynomial, exponent))
            .collect();
        let result = solve(&input);
        assert_eq!(result.factors(), expected, "prime {prime}");
        assert_eq!(result.square_free_part(), &part);
    }
}

#[test]
fn equal_multiplicities_are_grouped_without_irreducible_factorization() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5] {
        let x = p(&generators, prime, &[(&[1, 0], 1)]);
        let y = p(&generators, prime, &[(&[0, 1], 1)]);
        let xy = product(&x, &y);
        for multiplicity in [1, 2, 3] {
            let result = solve(&power(&xy, multiplicity));
            assert_eq!(result.factors(), &[factor(xy.clone(), multiplicity)]);
            assert_eq!(result.square_free_part(), &xy);
        }
    }
}

#[test]
fn characteristic_two_changes_square_freeness_without_field_equations() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 7, 65521] {
        let source = p(&generators, prime, &[(&[2, 0], 1), (&[0, 2], 1)]);
        let result = solve(&source);
        if prime == 2 {
            let root = p(&generators, prime, &[(&[1, 0], 1), (&[0, 1], 1)]);
            assert_eq!(result.factors(), &[factor(root.clone(), 2)]);
            assert_eq!(result.square_free_part(), &root);
        } else {
            assert_eq!(result.factors(), &[factor(source.clone(), 1)]);
            assert_eq!(result.square_free_part(), &source);
        }
    }
    for prime in [2, 3, 5] {
        // x^p-x vanishes on F_p-valued points but is a square-free nonzero
        // polynomial in the formal ring. No field equations are adjoined.
        let input = p(
            &generators,
            prime,
            &[(&[prime as u32, 0], 1), (&[1, 0], prime - 1)],
        );
        let result = solve(&input);
        assert_eq!(result.factors(), &[factor(input.clone(), 1)]);
        assert_eq!(result.square_free_part(), &input);
    }
}

#[test]
fn every_original_axis_has_derivative_evidence_including_unused_axes() {
    let generators = ring(&["x", "y", "z"]);
    let root = p(&generators, 3, &[(&[0, 1, 0], 1), (&[0, 0, 0], 2)]);
    let input = power(&root, 6);
    let result = solve(&input);
    assert_eq!(result.factors(), &[factor(root.clone(), 6)]);
    let receipts = &result.certificate().radical_derivative_gcds;
    assert_eq!(receipts.len(), 3);
    assert_eq!(receipts[0].left(), &root);
    assert!(receipts[0].right().is_zero());
    assert_eq!(receipts[1].right(), &scalar(&generators, 3, 1));
    assert!(receipts[2].right().is_zero());
    assert_eq!(receipts[2].gcd(), &scalar(&generators, 3, 1));
}

#[test]
fn zero_and_constants_preserve_declared_rings_and_public_part_conventions() {
    let full: Vec<_> = (0..8).map(|axis| Symbol::new(format!("x{axis}"))).collect();
    for generators in [vec![], ring(&["x"]), ring(&["x", "y"]), full] {
        for prime in [2, 5, 65521] {
            for value in [0, 1, prime - 1] {
                let input = scalar(&generators, prime, value);
                let result = solve(&input);
                assert_eq!(result.input(), &input);
                assert_eq!(result.unit(), value);
                assert!(result.factors().is_empty());
                assert!(result.certificate().radical_derivative_gcds.is_empty());
                assert_eq!(
                    result.square_free_part(),
                    &scalar(&generators, prime, u64::from(value != 0)),
                );
                assert_eq!(result.square_free_part().generators(), generators.as_slice());
                verify(&input, result.certificate(), &mut Unbounded).unwrap();
            }
        }
    }
}

#[test]
fn eight_axes_keep_trivial_frobenius_support_and_general_gcd_refusals() {
    let generators: Vec<_> = (0..8).map(|axis| Symbol::new(format!("x{axis}"))).collect();
    let root = p(
        &generators,
        2,
        &[(&[1, 0, 0, 0, 0, 0, 0, 0], 1), (&[0, 0, 0, 0, 0, 0, 0, 1], 1)],
    );
    assert_eq!(solve(&power(&root, 64)).factors(), &[factor(root, 64)]);
    let needs_auxiliary = p(
        &generators,
        5,
        &[(&[2, 0, 0, 0, 0, 0, 0, 0], 1), (&[0, 0, 0, 0, 0, 0, 0, 1], 1)],
    );
    let error = checked(&needs_auxiliary, &mut Unbounded, &mut Unbounded).unwrap_err();
    assert!(error.to_string().contains("auxiliary"));
    let generators = ring(&["x", "y"]);
    let augmented_degree = p(&generators, 65521, &[(&[64, 0], 1), (&[0, 1], 1)]);
    assert!(checked(&augmented_degree, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn independently_supplied_decomposition_is_accepted_without_generator_replay() {
    let generators = ring(&["x", "y"]);
    let x = p(&generators, 3, &[(&[1, 0], 1)]);
    let y = p(&generators, 3, &[(&[0, 1], 1)]);
    let input = product(
        &product(&power(&x, 2), &power(&y, 3)),
        &scalar(&generators, 3, 2),
    );
    let certificate = hand_certificate(
        2,
        vec![factor(x.clone(), 2), factor(y.clone(), 3)],
        product(&x, &y),
    );
    let accepted = verify(&input, &certificate, &mut Unbounded).unwrap();
    assert_eq!(accepted.input(), &input);
    assert_eq!(accepted.factors(), certificate.factors.as_slice());
}

#[test]
fn correct_product_with_a_repeated_factor_is_not_a_square_free_certificate() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        let x = p(&generators, prime, &[(&[1, 0], 1)]);
        let input = power(&x, 2);
        let certificate = hand_certificate(
            1,
            vec![factor(input.clone(), 1)],
            input.clone(),
        );
        assert_eq!(certificate.factors[0].polynomial, input);
        assert_ne!(
            certificate.radical_derivative_gcds.last().unwrap().gcd(),
            &scalar(&generators, prime, 1),
        );
        rejected(&input, &certificate);
    }
}

#[test]
fn overlapping_components_in_different_multiplicity_groups_are_rejected() {
    let generators = ring(&["x", "y"]);
    let x = p(&generators, 5, &[(&[1, 0], 1)]);
    let xy = p(&generators, 5, &[(&[1, 1], 1)]);
    let input = product(&x, &power(&xy, 2));
    let false_part = product(&x, &xy);
    let certificate = hand_certificate(
        1,
        vec![factor(x, 1), factor(xy, 2)],
        false_part,
    );
    // Both exact identities and weighted degree are correct. The independent
    // complete gradient GCD detects the overlap hidden between the groups.
    rejected(&input, &certificate);
}

#[test]
fn square_free_part_identity_is_checked_separately_from_source_reconstruction() {
    let generators = ring(&["x", "y"]);
    let x = p(&generators, 5, &[(&[1, 0], 1)]);
    let y = p(&generators, 5, &[(&[0, 1], 1)]);
    let input = product(&power(&x, 2), &power(&y, 3));
    let certificate = hand_certificate(
        1,
        vec![factor(x.clone(), 2), factor(y, 3)],
        x,
    );
    // The forged part is square-free, and source reconstruction is correct.
    rejected(&input, &certificate);
}

#[test]
fn source_identity_is_checked_even_with_valid_radical_and_weighted_degree() {
    let generators = ring(&["x", "y"]);
    let input = p(&generators, 5, &[(&[2, 0], 1), (&[0, 2], 1)]);
    let xy = p(&generators, 5, &[(&[1, 1], 1)]);
    let certificate = hand_certificate(1, vec![factor(xy.clone(), 1)], xy);
    rejected(&input, &certificate);
}

#[test]
fn multiplicity_order_units_and_trivial_component_mutants_are_rejected() {
    let generators = ring(&["x", "y"]);
    let x = p(&generators, 5, &[(&[1, 0], 1)]);
    let y = p(&generators, 5, &[(&[0, 1], 1)]);
    let input = product(
        &product(&power(&x, 2), &power(&y, 3)),
        &scalar(&generators, 5, 2),
    );
    let result = solve(&input);
    for multiplicity in [0, 1, 3, usize::MAX] {
        let mut certificate = result.certificate().clone();
        certificate.factors[0].multiplicity = multiplicity;
        rejected(&input, &certificate);
    }
    for unit in [0, 1, 4, 5, u64::MAX] {
        let mut certificate = result.certificate().clone();
        certificate.unit = unit;
        rejected(&input, &certificate);
    }
    let mut certificate = result.certificate().clone();
    certificate.factors.reverse();
    rejected(&input, &certificate);
    let mut certificate = result.certificate().clone();
    certificate.factors.pop();
    rejected(&input, &certificate);
    for polynomial in [
        scalar(&generators, 5, 0),
        scalar(&generators, 5, 1),
        product(&x, &scalar(&generators, 5, 2)),
    ] {
        let mut certificate = result.certificate().clone();
        certificate.factors[0].polynomial = polynomial;
        rejected(&input, &certificate);
    }
    let mut certificate = result.certificate().clone();
    certificate.factors.push(factor(scalar(&generators, 5, 1), 4));
    rejected(&input, &certificate);
    let mut certificate = result.certificate().clone();
    certificate.factors = vec![factor(x, 1); 65];
    rejected(&input, &certificate);

    let square_free = product(
        &p(&generators, 5, &[(&[1, 0], 1)]),
        &y,
    );
    let mut certificate = solve(&square_free).certificate().clone();
    certificate.factors = vec![
        factor(p(&generators, 5, &[(&[1, 0], 1)]), 1),
        factor(y, 1),
    ];
    // Equal multiplicities must be represented by one canonical group even
    // when splitting that group would preserve both identities.
    rejected(&square_free, &certificate);
}

#[test]
fn omitted_reordered_or_foreign_gradient_evidence_is_rejected() {
    let generators = ring(&["x", "y", "z"]);
    let input = p(&generators, 5, &[(&[1, 1, 0], 1), (&[0, 0, 0], 1)]);
    let result = solve(&input);
    let mut certificate = result.certificate().clone();
    certificate.radical_derivative_gcds.pop();
    rejected(&input, &certificate);
    let mut certificate = result.certificate().clone();
    certificate.radical_derivative_gcds.swap(0, 1);
    rejected(&input, &certificate);
    let mut certificate = result.certificate().clone();
    certificate.radical_derivative_gcds[1] =
        Arc::clone(&certificate.radical_derivative_gcds[0]);
    rejected(&input, &certificate);
    let mut certificate = result.certificate().clone();
    certificate.radical_derivative_gcds =
        vec![Arc::clone(&certificate.radical_derivative_gcds[0]); 9];
    rejected(&input, &certificate);

    let first = &result.certificate().radical_derivative_gcds[0];
    let scaled_derivative = product(first.right(), &scalar(&generators, 5, 2));
    let altered_right = checked_gcd(
        first.left(),
        &scaled_derivative,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(altered_right.gcd(), first.gcd());
    let mut certificate = result.certificate().clone();
    certificate.radical_derivative_gcds[0] = Arc::new(altered_right);
    rejected(&input, &certificate); // Same GCD, wrong original derivative.

    let altered_left = checked_gcd(
        &scalar(&generators, 5, 1),
        first.right(),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(altered_left.gcd(), first.gcd());
    let mut certificate = result.certificate().clone();
    certificate.radical_derivative_gcds[0] = Arc::new(altered_left);
    rejected(&input, &certificate);
}

#[test]
fn derivatives_that_vanish_modulo_p_cannot_be_replaced_by_formal_integer_terms() {
    let generators = ring(&["x", "y"]);
    let input = p(&generators, 2, &[(&[2, 0], 1), (&[0, 1], 1)]);
    let result = solve(&input);
    assert!(result.certificate().radical_derivative_gcds[0].right().is_zero());
    let forged = checked_gcd(
        &input,
        &p(&generators, 2, &[(&[1, 0], 1)]),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    let mut certificate = result.certificate().clone();
    certificate.radical_derivative_gcds[0] = Arc::new(forged);
    rejected(&input, &certificate);
}

#[test]
fn typed_same_name_axes_field_and_original_coefficients_remain_bound() {
    let plain = Symbol::new("x");
    let distinct = Symbol::with_identity(
        "x",
        SymbolIdentity::from_assumptions_digest([101; 32]),
    );
    let generators = [plain.clone(), distinct.clone()];
    let x = p(&generators, 5, &[(&[1, 0], 1)]);
    let other_x = p(&generators, 5, &[(&[0, 1], 1)]);
    let input = product(&power(&x, 2), &power(&other_x, 3));
    let result = solve(&input);
    assert_eq!(
        result.factors(),
        &[factor(x.clone(), 2), factor(other_x, 3)],
    );
    for changed in [
        p(&generators, 7, &[(&[1, 0], 1)]),
        p(&[distinct, plain], 5, &[(&[1, 0], 1)]),
        p(&ring(&["x", "y"]), 5, &[(&[1, 0], 1)]),
    ] {
        let mut certificate = result.certificate().clone();
        certificate.factors[0].polynomial = changed;
        rejected(&input, &certificate);
    }
    let changed_source = p(&generators, 5, &[(&[3, 2], 1)]);
    rejected(&changed_source, result.certificate());
    let changed_source = product(&input, &scalar(&generators, 5, 2));
    rejected(&changed_source, result.certificate());
    let mut certificate = result.certificate().clone();
    certificate.square_free_part =
        p(&ring(&["x", "y"]), 5, &[(&[1, 1], 1)]);
    rejected(&input, &certificate);
}

#[test]
fn zero_and_constant_mutants_cannot_publish_noncanonical_radicals() {
    let generators = ring(&["x", "y"]);
    let zero = scalar(&generators, 5, 0);
    let one = scalar(&generators, 5, 1);
    let two = scalar(&generators, 5, 2);
    let x = p(&generators, 5, &[(&[1, 0], 1)]);
    let zero_result = solve(&zero);
    let constant_result = solve(&two);
    let nonconstant = solve(&x);

    let mut certificate = zero_result.certificate().clone();
    certificate.square_free_part = one.clone();
    rejected(&zero, &certificate);
    let mut certificate = zero_result.certificate().clone();
    certificate.unit = 1;
    rejected(&zero, &certificate);
    let mut certificate = zero_result.certificate().clone();
    certificate.factors.push(factor(x, 1));
    rejected(&zero, &certificate);
    for input in [&zero, &two] {
        let mut certificate = solve(input).certificate().clone();
        certificate.radical_derivative_gcds =
            nonconstant.certificate().radical_derivative_gcds.clone();
        rejected(input, &certificate);
    }
    for part in [zero, two.clone()] {
        let mut certificate = constant_result.certificate().clone();
        certificate.square_free_part = part;
        rejected(&two, &certificate);
    }
    let mut certificate = constant_result.certificate().clone();
    certificate.factors.push(factor(one, 1));
    rejected(&two, &certificate);
}

#[test]
fn canonical_source_envelope_remains_enforced_by_typed_construction() {
    let generators = ring(&["x", "y"]);
    for prime in [0, 1, 4, 65537] {
        assert!(Polynomial::new(&generators, prime, &Terms::new()).is_err());
    }
    for terms in [
        Terms::from([(vec![0], 0)]),
        Terms::from([(vec![65, 0], 0)]),
        Terms::from([(vec![0, 0], 5)]),
    ] {
        assert!(Polynomial::new(&generators, 5, &terms).is_err());
    }
    let nine: Vec<_> = (0..9).map(|axis| Symbol::new(format!("x{axis}"))).collect();
    assert!(Polynomial::new(&nine, 5, &Terms::new()).is_err());
    assert!(
        Polynomial::new(
            &[generators[0].clone(), generators[0].clone()],
            5,
            &Terms::new(),
        )
        .is_err()
    );
    assert!(
        Polynomial::new(&[Symbol::new("x".repeat(65_537))], 5, &Terms::new()).is_err()
    );
    let three = ring(&["x", "y", "z"]);
    let mut excessive = Terms::new();
    'outer: for a in 0..=64 {
        for b in 0..=64 - a {
            for c in 0..=64 - a - b {
                excessive.insert(vec![a, b, c], 1);
                if excessive.len() == 4097 {
                    break 'outer;
                }
            }
        }
    }
    assert!(Polynomial::new(&three, 5, &excessive).is_err());
}

#[derive(Default)]
struct Meter {
    dimensions: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(amount > 0);
        self.dimensions[dimension.index()] += amount;
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

fn measured_limits(meter: &Meter) -> BudgetLimits {
    BudgetLimits {
        dimensions: meter.dimensions,
        verifier_pool: 0,
    }
}

#[test]
fn both_phases_share_cumulative_exact_budgets_through_descent_and_publication() {
    let generators = ring(&["x", "y"]);
    let input = p(&generators, 2, &[(&[3, 2], 1)]);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    checked(&input, &mut generation, &mut verification).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(generation.dimensions[dimension.index()] > 0);
        assert!(verification.dimensions[dimension.index()] > 0);
    }
    let mut generation_budget = Budget::new(measured_limits(&generation));
    let mut verification_budget = Budget::new(measured_limits(&verification));
    checked(&input, &mut generation_budget, &mut verification_budget).unwrap();

    for phase in [false, true] {
        let recorded = if phase { &verification } else { &generation };
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut limits = measured_limits(recorded);
            limits.dimensions[dimension.index()] -= 1;
            let mut insufficient = Budget::new(limits);
            let result = if phase {
                checked(&input, &mut Unbounded, &mut insufficient)
            } else {
                checked(&input, &mut insufficient, &mut Unbounded)
            };
            assert!(result.is_err(), "phase {phase}, dimension {dimension:?}");
        }
        let mut cumulative = Budget::new(measured_limits(recorded));
        let first = if phase {
            checked(&input, &mut Unbounded, &mut cumulative)
        } else {
            checked(&input, &mut cumulative, &mut Unbounded)
        };
        first.unwrap();
        let second = if phase {
            checked(&input, &mut Unbounded, &mut cumulative)
        } else {
            checked(&input, &mut cumulative, &mut Unbounded)
        };
        assert!(second.is_err());
    }
}

#[test]
fn cancellation_in_generation_verification_and_final_publication_is_terminal() {
    let generators = ring(&["x", "y"]);
    for input in [
        p(&generators, 2, &[(&[3, 2], 1)]),
        scalar(&generators, 2, 0),
        scalar(&generators, 2, 1),
    ] {
        let (mut generation, mut verification) = (Meter::default(), Meter::default());
        checked(&input, &mut generation, &mut verification).unwrap();
        for phase in [false, true] {
            let recorded = if phase { &verification } else { &generation };
            for cancel_at in [1, recorded.checkpoints.div_ceil(2), recorded.checkpoints] {
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
        }
    }
}

#[test]
fn independent_replay_has_exact_budget_and_every_checkpoint_is_cancellable() {
    let generators = ring(&["x", "y"]);
    let input = p(&generators, 3, &[(&[2, 1], 1)]);
    let result = solve(&input);
    let mut recorded = Meter::default();
    verify(&input, result.certificate(), &mut recorded).unwrap();
    let mut exact = Budget::new(measured_limits(&recorded));
    verify(&input, result.certificate(), &mut exact).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = measured_limits(&recorded);
        limits.dimensions[dimension.index()] -= 1;
        let mut insufficient = Budget::new(limits);
        assert!(verify(&input, result.certificate(), &mut insufficient).is_err());
    }
    for cancel_at in 1..=recorded.checkpoints {
        let mut cancelled = Meter {
            cancel_at: Some(cancel_at),
            ..Meter::default()
        };
        let error = verify(&input, result.certificate(), &mut cancelled).unwrap_err();
        assert!(error.to_string().to_lowercase().contains("cancel"));
        assert_eq!(cancelled.checkpoints, cancel_at);
    }
}
