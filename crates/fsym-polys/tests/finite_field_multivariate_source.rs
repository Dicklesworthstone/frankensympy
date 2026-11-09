//! Original-source admission and independent ZZ[x_1,...,x_n] -> F_p checks.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    PrimeFieldElementSourceError, PrimeFieldMultivariatePolynomial, PrimeFieldMultivariateTerms,
    checked_prime_field_multivariate_source, verify_prime_field_multivariate_source,
};

fn integer(value: i64) -> Expr {
    Expr::Integer(BigInt::from(value))
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(base.into(), integer(exponent).into())
}

fn xy() -> Vec<Symbol> {
    vec![Symbol::new("x"), Symbol::new("y")]
}

fn candidate(
    generators: &[Symbol],
    prime: u64,
    terms: &PrimeFieldMultivariateTerms,
) -> PrimeFieldMultivariatePolynomial {
    PrimeFieldMultivariatePolynomial::metered_new(generators, prime, terms, &mut Unbounded).unwrap()
}

fn reduce(source: &Expr, generators: &[Symbol], prime: u64) -> PrimeFieldMultivariateTerms {
    checked_prime_field_multivariate_source(
        source,
        generators,
        prime,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
    .into_polynomial()
    .terms()
    .clone()
}

fn refused(source: &Expr, generators: &[Symbol]) {
    assert!(
        checked_prime_field_multivariate_source(
            source,
            generators,
            5,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let zero = candidate(generators, 5, &PrimeFieldMultivariateTerms::new());
    assert!(
        verify_prime_field_multivariate_source(source, generators, 5, &zero, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn multivariate_integer_images_keep_cross_terms_signs_and_characteristic() {
    let generators = xy();
    let cases = [
        (
            "7*x**2*y+5*y**2-11*x+14",
            5,
            PrimeFieldMultivariateTerms::from([(vec![2, 1], 2), (vec![1, 0], 4), (vec![0, 0], 4)]),
        ),
        (
            "(x+y)**2",
            2,
            PrimeFieldMultivariateTerms::from([(vec![2, 0], 1), (vec![0, 2], 1)]),
        ),
        (
            "(x+y)**2",
            5,
            PrimeFieldMultivariateTerms::from([(vec![2, 0], 1), (vec![1, 1], 2), (vec![0, 2], 1)]),
        ),
        ("10*x*y+15", 5, PrimeFieldMultivariateTerms::new()),
        (
            "-13",
            7,
            PrimeFieldMultivariateTerms::from([(vec![0, 0], 1)]),
        ),
        ("0", 3, PrimeFieldMultivariateTerms::new()),
    ];
    for (source, prime, expected) in cases {
        assert_eq!(
            reduce(&parse(source).unwrap(), &generators, prime),
            expected
        );
    }
}

#[test]
fn rational_intermediates_require_integral_final_coefficients() {
    let generators = xy();
    let (x, y) = (
        Expr::Sym(generators[0].clone()),
        Expr::Sym(generators[1].clone()),
    );
    let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
    let xy = Expr::Mul(vec![x.clone(), y]);
    let source = Expr::Mul(vec![integer(2), half.clone(), xy.clone()]);
    assert_eq!(
        reduce(&source, &generators, 2),
        PrimeFieldMultivariateTerms::from([(vec![1, 1], 1)])
    );
    assert_eq!(
        reduce(
            &Expr::Add(vec![
                Expr::Mul(vec![half.clone(), xy.clone()]),
                Expr::Mul(vec![half.clone(), xy.clone()]),
            ]),
            &generators,
            2,
        ),
        PrimeFieldMultivariateTerms::from([(vec![1, 1], 1)])
    );
    assert_eq!(
        reduce(
            &Expr::Mul(vec![integer(5), power(integer(5), -1), x]),
            &generators,
            5,
        ),
        PrimeFieldMultivariateTerms::from([(vec![1, 0], 1)])
    );
    for source in [half.clone(), Expr::Mul(vec![half, xy])] {
        let error = checked_prime_field_multivariate_source(
            &source,
            &generators,
            5,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap_err();
        assert_eq!(error, PrimeFieldElementSourceError::NonIntegralCoefficient);
        let zero = candidate(&generators, 5, &PrimeFieldMultivariateTerms::new());
        assert_eq!(
            verify_prime_field_multivariate_source(&source, &generators, 5, &zero, &mut Unbounded,)
                .unwrap_err(),
            PrimeFieldElementSourceError::NonIntegralCoefficient
        );
    }
}

#[test]
fn large_integer_magnitudes_are_reduced_by_the_independent_byte_checker() {
    let generators = xy();
    let value = (BigInt::from(1) << 2048u32) + BigInt::from(16);
    // 2**2048 == 4 (mod 7), so the positive and negative residues are 6 and 1.
    let source = Expr::Add(vec![
        Expr::Mul(vec![
            Expr::Integer(value.clone()),
            Expr::Sym(generators[0].clone()),
        ]),
        Expr::Mul(vec![
            Expr::Integer(-value),
            Expr::Sym(generators[1].clone()),
        ]),
    ]);
    assert_eq!(
        reduce(&source, &generators, 7),
        PrimeFieldMultivariateTerms::from([(vec![1, 0], 6), (vec![0, 1], 1)])
    );
    let maximum = (BigInt::from(1) << 4095u32) + BigInt::from(1);
    assert_eq!(
        reduce(&Expr::Integer(maximum), &generators, 2),
        PrimeFieldMultivariateTerms::from([(vec![0, 0], 1)])
    );
}

#[test]
fn accepted_receipt_retains_exact_held_source_and_replays_independently() {
    let generators = xy();
    let source = Expr::Add(vec![
        integer(0),
        Expr::Mul(vec![integer(1), Expr::Sym(generators[0].clone())]),
        Expr::Sym(generators[1].clone()),
    ]);
    let accepted = checked_prime_field_multivariate_source(
        &source,
        &generators,
        5,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(accepted.source(), &source);
    assert_eq!(accepted.polynomial().generators(), generators);
    let replay = verify_prime_field_multivariate_source(
        &source,
        &generators,
        5,
        accepted.polynomial(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(replay.source(), &source);
    assert_eq!(replay.into_polynomial(), accepted.into_polynomial());
}

#[test]
fn verifier_rejects_wrong_monomials_coefficients_omissions_and_extra_terms() {
    let generators = xy();
    let source = parse("3*x**2+2*x*y+y+1").unwrap();
    let expected = PrimeFieldMultivariateTerms::from([
        (vec![2, 0], 3),
        (vec![1, 1], 2),
        (vec![0, 1], 1),
        (vec![0, 0], 1),
    ]);
    let mut mutations = vec![PrimeFieldMultivariateTerms::new()];
    for key in expected.keys() {
        let mut missing = expected.clone();
        missing.remove(key);
        mutations.push(missing);
        let mut coefficient = expected.clone();
        coefficient.insert(key.clone(), (expected[key] + 1) % 5);
        mutations.push(coefficient);
    }
    let mut extra = expected.clone();
    extra.insert(vec![0, 2], 1);
    mutations.push(extra);
    let mut changed_coordinate = expected.clone();
    changed_coordinate.remove(&vec![2, 0]);
    changed_coordinate.insert(vec![0, 2], 3);
    mutations.push(changed_coordinate);
    for terms in mutations {
        let proposed = candidate(&generators, 5, &terms);
        assert!(
            verify_prime_field_multivariate_source(
                &source,
                &generators,
                5,
                &proposed,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn verifier_binds_characteristic_and_complete_ordered_typed_ring() {
    let generators = xy();
    let source = parse("x+2*y").unwrap();
    let terms = PrimeFieldMultivariateTerms::from([(vec![1, 0], 1), (vec![0, 1], 2)]);
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    for (ring, prime) in [
        (generators.clone(), 7),
        (vec![generators[1].clone(), generators[0].clone()], 5),
        (vec![typed, generators[1].clone()], 5),
    ] {
        let proposed = candidate(&ring, prime, &terms);
        assert!(
            verify_prime_field_multivariate_source(
                &source,
                &generators,
                5,
                &proposed,
                &mut Unbounded,
            )
            .is_err()
        );
    }
    let reversed = [generators[1].clone(), generators[0].clone()];
    assert_eq!(
        reduce(&source, &reversed, 5),
        PrimeFieldMultivariateTerms::from([(vec![0, 1], 1), (vec![1, 0], 2)])
    );
}

#[test]
fn identical_printed_names_with_distinct_identities_are_distinct_coordinates() {
    let a = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let b = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let source = Expr::Add(vec![
        power(Expr::Sym(a.clone()), 2),
        Expr::Mul(vec![integer(3), Expr::Sym(a.clone()), Expr::Sym(b.clone())]),
        Expr::Sym(b.clone()),
    ]);
    assert_eq!(
        reduce(&source, &[a.clone(), b], 5),
        PrimeFieldMultivariateTerms::from([(vec![2, 0], 1), (vec![1, 1], 3), (vec![0, 1], 1),])
    );
    refused(&source, &[a]);
    refused(&source, &[Symbol::new("x")]);
}

#[test]
fn every_original_child_is_checked_before_zero_and_unit_shortcuts() {
    let generators = xy();
    let bad = [
        Expr::Sym(Symbol::new("z")),
        Expr::Function("sin".into(), vec![Expr::Sym(generators[0].clone())]),
        Expr::Integer(BigInt::from(1) << 4096u32),
        power(Expr::Sym(generators[1].clone()), -1),
        power(integer(0), -1),
    ];
    for child in bad {
        refused(&Expr::Mul(vec![integer(0), child.clone()]), &generators);
        refused(&power(child, 0), &generators);
    }
    refused(&power(integer(0), 0), &generators);
    refused(
        &Expr::Pow(
            integer(1).into(),
            Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2))).into(),
        ),
        &generators,
    );
    let x = Expr::Sym(generators[0].clone());
    refused(&Expr::Mul(vec![x.clone(), power(x, -1)]), &generators);
}

#[test]
fn total_degree_bound_precedes_modular_annihilation() {
    let generators = xy();
    let x = Expr::Sym(generators[0].clone());
    let y = Expr::Sym(generators[1].clone());
    let admitted = Expr::Mul(vec![power(x.clone(), 32), power(y.clone(), 32)]);
    assert_eq!(
        reduce(&admitted, &generators, 5),
        PrimeFieldMultivariateTerms::from([(vec![32, 32], 1)])
    );
    let too_large = Expr::Mul(vec![integer(5), power(x, 33), power(y, 32)]);
    refused(&too_large, &generators);
}

#[test]
fn exact_qq_cancellation_can_reduce_original_total_degree_before_reduction() {
    let generators = xy();
    let high = parse("x**65+y**65").unwrap();
    let source = Expr::Add(vec![
        high.clone(),
        Expr::Mul(vec![integer(-1), high]),
        parse("x*y+1").unwrap(),
    ]);
    assert_eq!(
        reduce(&source, &generators, 3),
        PrimeFieldMultivariateTerms::from([(vec![1, 1], 1), (vec![0, 0], 1)])
    );
}

#[test]
fn source_node_depth_coefficient_and_name_limits_precede_owned_cloning() {
    let generators = xy();
    refused(&Expr::Add(vec![integer(0); 4096]), &generators);
    let mut deep = integer(0);
    for _ in 0..65 {
        deep = Expr::Add(vec![deep]);
    }
    refused(&deep, &generators);
    refused(&Expr::Sym(Symbol::new("z".repeat(65_537))), &generators);
    refused(&power(Expr::Sym(generators[0].clone()), 257), &generators);
    refused(&Expr::Integer(BigInt::from(1) << 4096u32), &generators);
}

#[test]
fn intermediate_degree_and_term_growth_fail_without_hiding_held_children() {
    let generators = xy();
    let high = power(parse("x*y").unwrap(), 129);
    refused(&high, &generators);
    refused(&Expr::Mul(vec![integer(0), high.clone()]), &generators);
    refused(&power(high, 0), &generators);
    let ring: Vec<_> = ["w", "x", "y", "z"].into_iter().map(Symbol::new).collect();
    // C(35,3)=6545 distinct QQ monomials; modular collapse cannot admit it.
    refused(&parse("(w+x+y+z)**32").unwrap(), &ring);
}

#[test]
fn ring_and_characteristic_validation_precede_constant_shortcuts() {
    let generators = xy();
    for prime in [0, 1, 4, 9, 65_523, u64::MAX] {
        assert!(
            checked_prime_field_multivariate_source(
                &integer(0),
                &generators,
                prime,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
    let many: Vec<_> = (0..9).map(|i| Symbol::new(format!("x{i}"))).collect();
    for ring in [
        vec![generators[0].clone(), generators[0].clone()],
        many,
        vec![Symbol::new("x".repeat(65_537))],
    ] {
        assert!(
            checked_prime_field_multivariate_source(
                &integer(0),
                &ring,
                5,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn explicit_constant_fields_preserve_original_sources_and_exact_residues() {
    // An empty ordered generator list denotes F_p, rather than an invalid
    // polynomial ring. Elimination and complete evaluation both reach it.
    for prime in [2, 3, 5, 7, 65_521] {
        for value in [0i64, 1, -1, 13, -13, 65_521] {
            let source = integer(value);
            let residue = value.rem_euclid(prime as i64) as u64;
            let expected = if residue == 0 {
                PrimeFieldMultivariateTerms::new()
            } else {
                PrimeFieldMultivariateTerms::from([(Vec::new(), residue)])
            };
            let accepted = checked_prime_field_multivariate_source(
                &source,
                &[],
                prime,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap();
            assert_eq!(accepted.source(), &source);
            assert!(accepted.polynomial().generators().is_empty());
            assert_eq!(accepted.polynomial().prime(), prime);
            assert_eq!(accepted.polynomial().terms(), &expected);
            // Independently supplied coefficient data, not generator output.
            let witness = candidate(&[], prime, &expected);
            assert!(
                verify_prime_field_multivariate_source(
                    &source,
                    &[],
                    prime,
                    &witness,
                    &mut Unbounded,
                )
                .is_ok()
            );
        }
    }
    let cancelled_denominator = Expr::Mul(vec![integer(5), power(integer(5), -1)]);
    assert_eq!(
        reduce(&cancelled_denominator, &[], 5),
        PrimeFieldMultivariateTerms::from([(Vec::new(), 1)]),
    );
}

#[test]
fn constant_field_admission_checks_hidden_children_and_nonintegral_coefficients() {
    let undeclared = Expr::Sym(Symbol::new("x"));
    for source in [
        undeclared.clone(),
        Expr::Mul(vec![integer(0), undeclared.clone()]),
        power(undeclared, 0),
        power(integer(0), -1),
        Expr::Mul(vec![integer(0), power(integer(0), -1)]),
        Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2))),
    ] {
        refused(&source, &[]);
    }
    let zero = candidate(&[], 5, &PrimeFieldMultivariateTerms::new());
    for prime in [0, 1, 4, 9, 65_523, u64::MAX] {
        assert!(
            checked_prime_field_multivariate_source(
                &integer(0),
                &[],
                prime,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
        assert!(
            verify_prime_field_multivariate_source(
                &integer(0),
                &[],
                prime,
                &zero,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn constant_field_verification_binds_every_coefficient_and_the_complete_ring() {
    let source = integer(7);
    for witness in [
        candidate(&[], 5, &PrimeFieldMultivariateTerms::new()),
        candidate(&[], 5, &PrimeFieldMultivariateTerms::from([(vec![], 3)])),
        candidate(&[], 7, &PrimeFieldMultivariateTerms::new()),
        candidate(&xy(), 5, &PrimeFieldMultivariateTerms::from([(vec![0, 0], 2)])),
    ] {
        assert!(
            verify_prime_field_multivariate_source(
                &source,
                &[],
                5,
                &witness,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn eight_generator_coordinate_positions_are_preserved() {
    let generators: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let source = Expr::Add(
        generators
            .iter()
            .enumerate()
            .map(|(index, symbol)| {
                Expr::Mul(vec![integer(index as i64 + 1), Expr::Sym(symbol.clone())])
            })
            .collect(),
    );
    let expected: PrimeFieldMultivariateTerms = (0..8)
        .map(|index| {
            let mut powers = vec![0; 8];
            powers[index] = 1;
            (powers, index as u64 + 1)
        })
        .collect();
    assert_eq!(reduce(&source, &generators, 11), expected);
}

#[test]
fn small_integer_bilinear_campaign_matches_direct_coefficient_formulas() {
    let generators = xy();
    for prime in [2, 3, 5] {
        for a in -3..=3 {
            for b in -2..=2 {
                let x = Expr::Sym(generators[0].clone());
                let y = Expr::Sym(generators[1].clone());
                let source = Expr::Mul(vec![
                    Expr::Add(vec![
                        Expr::Mul(vec![integer(a), x.clone()]),
                        y.clone(),
                        integer(1),
                    ]),
                    Expr::Add(vec![x, Expr::Mul(vec![integer(b), y]), integer(-2)]),
                ]);
                let expected: PrimeFieldMultivariateTerms = [
                    (vec![2, 0], a),
                    (vec![1, 1], a * b + 1),
                    (vec![0, 2], b),
                    (vec![1, 0], 1 - 2 * a),
                    (vec![0, 1], b - 2),
                    (vec![0, 0], -2),
                ]
                .into_iter()
                .filter_map(|(powers, value)| {
                    let residue = value.rem_euclid(prime as i64) as u64;
                    (residue != 0).then_some((powers, residue))
                })
                .collect();
                assert_eq!(reduce(&source, &generators, prime), expected);
            }
        }
    }
}

#[derive(Default)]
struct Count {
    steps: u64,
    bytes: u64,
    checkpoints: usize,
    cancel: Option<usize>,
}

impl BudgetMeter for Count {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.steps += amount,
            Dimension::MemoryBytes => self.bytes += amount,
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
        if self.cancel == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn cap(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn cancellation_in_generation_verification_and_final_publication_is_terminal() {
    let generators = xy();
    let source = parse("(x+y+1)**4-11*x*y+3").unwrap();
    let (mut generation, mut verification) = (Count::default(), Count::default());
    checked_prime_field_multivariate_source(
        &source,
        &generators,
        7,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for (is_generation, checkpoints) in [
        (true, generation.checkpoints),
        (false, verification.checkpoints),
    ] {
        for at in [1, checkpoints / 2, checkpoints] {
            let mut cancelled = Count {
                cancel: Some(at),
                ..Count::default()
            };
            let result = if is_generation {
                checked_prime_field_multivariate_source(
                    &source,
                    &generators,
                    7,
                    &mut cancelled,
                    &mut Unbounded,
                )
            } else {
                checked_prime_field_multivariate_source(
                    &source,
                    &generators,
                    7,
                    &mut Unbounded,
                    &mut cancelled,
                )
            };
            assert!(matches!(
                result,
                Err(PrimeFieldElementSourceError::Polynomial(_))
            ));
        }
    }
}

#[test]
fn phase_ledgers_cover_original_conversion_reference_work_and_source_cloning() {
    let generators = xy();
    let source = parse("(x+y+1)**5-13*x*y+1").unwrap();
    let (mut generation, mut verification) = (Count::default(), Count::default());
    checked_prime_field_multivariate_source(
        &source,
        &generators,
        11,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for (dimension, g, v) in [
        (
            Dimension::ComputeSteps,
            generation.steps,
            verification.steps,
        ),
        (Dimension::MemoryBytes, generation.bytes, verification.bytes),
    ] {
        checked_prime_field_multivariate_source(
            &source,
            &generators,
            11,
            &mut cap(dimension, g),
            &mut cap(dimension, v),
        )
        .unwrap();
        assert!(
            checked_prime_field_multivariate_source(
                &source,
                &generators,
                11,
                &mut cap(dimension, g - 1),
                &mut Unbounded,
            )
            .is_err()
        );
        assert!(
            checked_prime_field_multivariate_source(
                &source,
                &generators,
                11,
                &mut Unbounded,
                &mut cap(dimension, v - 1),
            )
            .is_err()
        );
    }
}
