//! Independent exact order-conversion fixtures and certificate adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldGroebnerCertificate, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder, VerifiedPrimeFieldGroebnerBasis,
    checked_prime_field_fglm as checked, checked_prime_field_groebner_basis as groebner,
    verify_prime_field_groebner_certificate as verify,
};

const ORDERS: [TermOrder; 3] = [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex];

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn poly(generators: &[Symbol], prime: u64, terms: &[(&[u32], u64)]) -> Polynomial {
    let terms: Terms = terms.iter().map(|(m, c)| (m.to_vec(), *c)).collect();
    Polynomial::new(generators, prime, &terms).unwrap()
}

fn coupled(generators: &[Symbol], prime: u64) -> Vec<Polynomial> {
    vec![
        poly(generators, prime, &[(&[1, 1], 1), (&[0, 0], prime - 1)]),
        poly(generators, prime, &[(&[0, 2], 1), (&[1, 0], prime - 1)]),
    ]
}

fn expected(generators: &[Symbol], prime: u64, order: TermOrder) -> Vec<Polynomial> {
    if order == TermOrder::Lex {
        vec![
            poly(generators, prime, &[(&[1, 0], 1), (&[0, 2], prime - 1)]),
            poly(generators, prime, &[(&[0, 3], 1), (&[0, 0], prime - 1)]),
        ]
    } else {
        vec![
            poly(generators, prime, &[(&[2, 0], 1), (&[0, 1], prime - 1)]),
            poly(generators, prime, &[(&[1, 1], 1), (&[0, 0], prime - 1)]),
            poly(generators, prime, &[(&[0, 2], 1), (&[1, 0], prime - 1)]),
        ]
    }
}

fn exact_basis(actual: &VerifiedPrimeFieldGroebnerBasis, wanted: &[Polynomial]) {
    assert_eq!(actual.basis().len(), wanted.len());
    for polynomial in wanted {
        assert!(
            actual.basis().contains(polynomial),
            "missing {polynomial:?}"
        );
    }
}

fn source(generators: &[Symbol], prime: u64, order: TermOrder) -> VerifiedPrimeFieldGroebnerBasis {
    groebner(
        generators,
        prime,
        &coupled(generators, prime),
        order,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn reject(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    order: TermOrder,
    candidate: &PrimeFieldGroebnerCertificate,
) {
    assert!(verify(generators, prime, inputs, order, candidate, &mut Unbounded).is_err());
}

#[test]
fn all_nine_order_pairs_over_small_primes_match_complete_closed_form_bases() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 3, 5, 7] {
        let inputs = coupled(&generators, prime);
        for from in ORDERS {
            let accepted = source(&generators, prime, from);
            for to in ORDERS {
                let result = checked(
                    &generators,
                    prime,
                    &inputs,
                    from,
                    to,
                    &mut Unbounded,
                    &mut Unbounded,
                )
                .unwrap();
                exact_basis(&result, &expected(&generators, prime, to));
                assert_eq!(result.generators(), generators);
                assert_eq!(result.prime(), prime);
                assert_eq!(result.order(), to);
                let monomials = result.standard_monomials(&mut Unbounded).unwrap();
                let wanted = if to == TermOrder::Lex {
                    vec![vec![0, 0], vec![0, 1], vec![0, 2]]
                } else {
                    vec![vec![0, 0], vec![0, 1], vec![1, 0]]
                };
                assert_eq!(monomials, wanted, "{prime}: {from:?}->{to:?}");
                let via_method = accepted.fglm(to, &mut Unbounded, &mut Unbounded).unwrap();
                assert_eq!(via_method.inputs(), accepted.basis());
                assert_eq!(via_method.basis(), result.basis());
            }
        }
    }
}

#[test]
fn three_variable_orders_keep_every_standard_monomial_and_distinguish_ties() {
    let generators = ring(&["x", "y", "z"]);
    let inputs = [
        poly(&generators, 5, &[(&[2, 0, 0], 1)]),
        poly(&generators, 5, &[(&[0, 3, 0], 1)]),
        poly(&generators, 5, &[(&[0, 0, 2], 1)]),
    ];
    let mut staircases = Vec::new();
    for to in ORDERS {
        let result = checked(
            &generators,
            5,
            &inputs,
            TermOrder::Lex,
            to,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        exact_basis(&result, &inputs);
        let actual = result.standard_monomials(&mut Unbounded).unwrap();
        assert_eq!(actual.len(), 12);
        for x in 0..2 {
            for y in 0..3 {
                for z in 0..2 {
                    assert!(actual.contains(&vec![x, y, z]));
                }
            }
        }
        assert_eq!(actual[0], vec![0, 0, 0]);
        staircases.push(actual);
    }
    let position = |basis: &[Vec<u32>], m: &[u32]| basis.iter().position(|v| v == m).unwrap();
    // At equal total degree lex puts y^2 before x*z, while reverse lex does the opposite.
    assert!(position(&staircases[1], &[0, 2, 0]) < position(&staircases[1], &[1, 0, 1]));
    assert!(position(&staircases[2], &[1, 0, 1]) < position(&staircases[2], &[0, 2, 0]));
}

#[test]
fn characteristic_three_retains_repeated_root_multiplicity_and_no_point_sampling() {
    let generators = ring(&["x", "y"]);
    let result = checked(
        &generators,
        3,
        &coupled(&generators, 3),
        TermOrder::DegRevLex,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    // y^3-1=(y-1)^3 in F_3, so the quotient has dimension three, not one.
    exact_basis(&result, &expected(&generators, 3, TermOrder::Lex));
    assert_eq!(result.standard_monomials(&mut Unbounded).unwrap().len(), 3);

    let univariate = ring(&["t"]);
    let no_rational_points = [poly(&univariate, 3, &[(&[2], 1), (&[0], 1)])];
    for to in ORDERS {
        let converted = checked(
            &univariate,
            3,
            &no_rational_points,
            TermOrder::DegLex,
            to,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        exact_basis(&converted, &no_rational_points);
        assert_eq!(
            converted.standard_monomials(&mut Unbounded).unwrap(),
            vec![vec![0], vec![1]]
        );
    }
}

#[test]
fn characteristic_changes_unit_ideal_without_adding_field_equations() {
    let generators = ring(&["x"]);
    for prime in [2, 3, 5] {
        let inputs = [
            poly(&generators, prime, &[(&[2], 1), (&[0], 1)]),
            poly(&generators, prime, &[(&[1], 1), (&[0], 1)]),
        ];
        let result = checked(
            &generators,
            prime,
            &inputs,
            TermOrder::DegLex,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        if prime == 2 {
            exact_basis(&result, &inputs[1..]);
            assert_eq!(
                result.standard_monomials(&mut Unbounded).unwrap(),
                vec![vec![0]]
            );
        } else {
            exact_basis(&result, &[poly(&generators, prime, &[(&[0], 1)])]);
            assert!(
                result
                    .standard_monomials(&mut Unbounded)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}

#[test]
fn zero_constant_field_and_unit_ideals_have_the_exact_empty_axis_semantics() {
    let empty: Vec<Symbol> = vec![];
    let zero = poly(&empty, 5, &[]);
    let two = poly(&empty, 5, &[(&[], 2)]);
    for inputs in [vec![], vec![zero.clone()], vec![two]] {
        for from in ORDERS {
            for to in ORDERS {
                let result =
                    checked(&empty, 5, &inputs, from, to, &mut Unbounded, &mut Unbounded).unwrap();
                if inputs.iter().all(Polynomial::is_zero) {
                    assert!(result.basis().is_empty());
                    assert_eq!(
                        result.standard_monomials(&mut Unbounded).unwrap(),
                        vec![Vec::<u32>::new()]
                    );
                } else {
                    exact_basis(&result, &[poly(&empty, 5, &[(&[], 1)])]);
                    assert!(
                        result
                            .standard_monomials(&mut Unbounded)
                            .unwrap()
                            .is_empty()
                    );
                }
            }
        }
    }
    let generators = ring(&["x", "y"]);
    let inputs = [poly(&generators, 5, &[(&[0, 0], 3)])];
    let result = checked(
        &generators,
        5,
        &inputs,
        TermOrder::Lex,
        TermOrder::DegRevLex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    exact_basis(&result, &[poly(&generators, 5, &[(&[0, 0], 1)])]);
    assert!(
        result
            .standard_monomials(&mut Unbounded)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn nonradical_monomial_ideal_keeps_all_nilpotent_directions() {
    let generators = ring(&["x", "y"]);
    for prime in [2, 5] {
        let inputs = [
            poly(&generators, prime, &[(&[2, 0], 1)]),
            poly(&generators, prime, &[(&[0, 2], 1)]),
        ];
        for from in ORDERS {
            for to in ORDERS {
                let result = checked(
                    &generators,
                    prime,
                    &inputs,
                    from,
                    to,
                    &mut Unbounded,
                    &mut Unbounded,
                )
                .unwrap();
                exact_basis(&result, &inputs);
                assert_eq!(
                    result.standard_monomials(&mut Unbounded).unwrap(),
                    vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]],
                );
            }
        }
    }
}

#[test]
fn different_order_requires_finite_dimension_but_same_order_still_verifies() {
    let generators = ring(&["x", "y"]);
    for inputs in [
        vec![],
        vec![poly(&generators, 5, &[(&[1, 1], 1)])],
        vec![poly(&generators, 5, &[(&[1, 0], 1)])],
    ] {
        for from in ORDERS {
            let source = groebner(
                &generators,
                5,
                &inputs,
                from,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap();
            assert!(source.standard_monomials(&mut Unbounded).is_err());
            let unchanged = source.fglm(from, &mut Unbounded, &mut Unbounded).unwrap();
            assert_eq!(unchanged.basis(), source.basis());
            for to in ORDERS {
                if from != to {
                    assert!(source.fglm(to, &mut Unbounded, &mut Unbounded).is_err());
                    assert!(
                        checked(
                            &generators,
                            5,
                            &inputs,
                            from,
                            to,
                            &mut Unbounded,
                            &mut Unbounded
                        )
                        .is_err()
                    );
                }
            }
        }
    }
}

#[test]
fn quotient_dimension_boundary_and_oversized_product_refuse_before_unbounded_enumeration() {
    let generators = ring(&["x"]);
    let boundary = [poly(&generators, 2, &[(&[64], 1)])];
    let result = checked(
        &generators,
        2,
        &boundary,
        TermOrder::DegLex,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    exact_basis(&result, &boundary);
    assert_eq!(
        result.standard_monomials(&mut Unbounded).unwrap(),
        (0..64).map(|k| vec![k]).collect::<Vec<_>>()
    );

    let generators = ring(&["x", "y"]);
    let oversized = [
        poly(&generators, 5, &[(&[8, 0], 1)]),
        poly(&generators, 5, &[(&[0, 9], 1)]),
    ];
    let source = groebner(
        &generators,
        5,
        &oversized,
        TermOrder::DegLex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(source.standard_monomials(&mut Unbounded).is_err());
    assert!(
        source
            .fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn typed_same_name_axes_preserve_identity_and_foreign_requests_refuse() {
    let generators = vec![
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([31; 32])),
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([32; 32])),
    ];
    let inputs = coupled(&generators, 5);
    let result = checked(
        &generators,
        5,
        &inputs,
        TermOrder::DegLex,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    exact_basis(&result, &expected(&generators, 5, TermOrder::Lex));
    assert_eq!(result.generators(), generators);
    let reversed = vec![generators[1].clone(), generators[0].clone()];
    let foreign = ring(&["x", "y"]);
    let duplicate = vec![generators[0].clone(), generators[0].clone()];
    for requested in [reversed, foreign, duplicate] {
        assert!(
            checked(
                &requested,
                5,
                &inputs,
                TermOrder::DegLex,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked(
            &generators,
            3,
            &inputs,
            TermOrder::DegLex,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn exact_prime_and_raw_request_size_are_admitted_before_fast_paths() {
    for prime in [0, 1, 4, 9, 65_537] {
        assert!(
            checked(
                &[],
                prime,
                &[],
                TermOrder::Lex,
                TermOrder::Lex,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let too_many: Vec<_> = (0..9).map(|i| Symbol::new(format!("x{i}"))).collect();
    assert!(
        checked(
            &too_many,
            5,
            &[],
            TermOrder::Lex,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let generators = ring(&["x"]);
    let zero = poly(&generators, 5, &[]);
    assert!(
        checked(
            &generators,
            5,
            &vec![zero; 33],
            TermOrder::Lex,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn fglm_candidate_is_replayable_but_binds_source_and_target_order() {
    let generators = ring(&["x", "y"]);
    let source = source(&generators, 5, TermOrder::DegRevLex);
    let certificate = source
        .fglm_with_certificate(TermOrder::Lex, &mut Unbounded)
        .unwrap();
    let replay = verify(
        &generators,
        5,
        source.basis(),
        TermOrder::Lex,
        &certificate,
        &mut Unbounded,
    )
    .unwrap();
    exact_basis(&replay, &expected(&generators, 5, TermOrder::Lex));
    assert_eq!(replay.inputs(), source.basis());
    reject(
        &generators,
        5,
        source.basis(),
        TermOrder::DegLex,
        &certificate,
    );
    reject(&generators, 3, source.basis(), TermOrder::Lex, &certificate);
    let mut changed_inputs = source.basis().to_vec();
    changed_inputs[0] = poly(&generators, 5, &[(&[2, 0], 1)]);
    reject(
        &generators,
        5,
        &changed_inputs,
        TermOrder::Lex,
        &certificate,
    );
    changed_inputs = source.basis().to_vec();
    changed_inputs.swap(0, 1);
    reject(
        &generators,
        5,
        &changed_inputs,
        TermOrder::Lex,
        &certificate,
    );
    let mut changed_ring = generators.clone();
    changed_ring.swap(0, 1);
    reject(
        &changed_ring,
        5,
        source.basis(),
        TermOrder::Lex,
        &certificate,
    );
}

#[test]
fn candidate_omissions_wrong_coefficients_and_incomplete_pair_evidence_are_rejected() {
    let generators = ring(&["x", "y"]);
    let source = source(&generators, 5, TermOrder::DegRevLex);
    let certificate = source
        .fglm_with_certificate(TermOrder::Lex, &mut Unbounded)
        .unwrap();
    assert_eq!(certificate.basis.len(), 2);
    assert_eq!(certificate.s_pairs.len(), 1);

    let mut mutant = certificate.clone();
    mutant.basis.pop();
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.basis[1].insert(vec![0, 0], 3);
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.basis_in_input.clear();
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.input_in_basis.pop();
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    for quotient in &mut mutant.basis_in_input[0] {
        quotient.clear();
    }
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.s_pairs.clear();
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.s_pairs.push(mutant.s_pairs[0].clone());
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.s_pairs[0].right = 0;
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    mutant = certificate.clone();
    mutant.s_pairs[0].quotients.clear();
    reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
}

#[test]
fn malformed_raw_monomials_and_noncanonical_residues_never_enter_receipts() {
    let generators = ring(&["x", "y"]);
    let source = source(&generators, 5, TermOrder::DegRevLex);
    let certificate = source
        .fglm_with_certificate(TermOrder::Lex, &mut Unbounded)
        .unwrap();
    for (monomial, coefficient) in [
        (vec![0], 1),
        (vec![0, 0, 0], 1),
        (vec![65, 0], 1),
        (vec![u32::MAX, 1], 1),
        (vec![0, 0], 5),
        (vec![0, 0], u64::MAX),
    ] {
        let mut mutant = certificate.clone();
        mutant.basis[0].insert(monomial.clone(), coefficient);
        reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
        mutant = certificate.clone();
        mutant.basis_in_input[0][0].insert(monomial.clone(), coefficient);
        reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
        mutant = certificate.clone();
        mutant.input_in_basis[0][0].insert(monomial, coefficient);
        reject(&generators, 5, source.basis(), TermOrder::Lex, &mutant);
    }
}

#[derive(Default)]
struct Count {
    checkpoints: usize,
    compute: u64,
    memory: u64,
    allocations: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Count {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
            Dimension::AllocationCount => self.allocations += amount,
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

fn cap(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn cumulative_generator_and_verifier_budgets_span_source_conversion_and_replay() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    for to in [TermOrder::Lex, TermOrder::DegRevLex] {
        let (mut generation, mut verification) = (Count::default(), Count::default());
        checked(
            &generators,
            5,
            &inputs,
            TermOrder::DegRevLex,
            to,
            &mut generation,
            &mut verification,
        )
        .unwrap();
        for (dimension, used_g, used_v) in [
            (
                Dimension::ComputeSteps,
                generation.compute,
                verification.compute,
            ),
            (
                Dimension::MemoryBytes,
                generation.memory,
                verification.memory,
            ),
            (
                Dimension::AllocationCount,
                generation.allocations,
                verification.allocations,
            ),
        ] {
            assert!(used_g > 0 && used_v > 0);
            checked(
                &generators,
                5,
                &inputs,
                TermOrder::DegRevLex,
                to,
                &mut cap(dimension, used_g),
                &mut cap(dimension, used_v),
            )
            .unwrap();
            assert!(
                checked(
                    &generators,
                    5,
                    &inputs,
                    TermOrder::DegRevLex,
                    to,
                    &mut cap(dimension, used_g - 1),
                    &mut Unbounded
                )
                .is_err()
            );
            assert!(
                checked(
                    &generators,
                    5,
                    &inputs,
                    TermOrder::DegRevLex,
                    to,
                    &mut Unbounded,
                    &mut cap(dimension, used_v - 1)
                )
                .is_err()
            );
            let mut spent = cap(dimension, used_g);
            BudgetMeter::charge(&mut spent, dimension, 1).unwrap();
            assert!(
                checked(
                    &generators,
                    5,
                    &inputs,
                    TermOrder::DegRevLex,
                    to,
                    &mut spent,
                    &mut Unbounded
                )
                .is_err()
            );
            let mut spent = cap(dimension, used_v);
            BudgetMeter::charge(&mut spent, dimension, 1).unwrap();
            assert!(
                checked(
                    &generators,
                    5,
                    &inputs,
                    TermOrder::DegRevLex,
                    to,
                    &mut Unbounded,
                    &mut spent
                )
                .is_err()
            );
        }
    }
}

#[test]
fn cancellation_during_generation_verification_and_final_publication_is_terminal() {
    let generators = ring(&["x", "y"]);
    let inputs = coupled(&generators, 5);
    let (mut generation, mut verification) = (Count::default(), Count::default());
    checked(
        &generators,
        5,
        &inputs,
        TermOrder::DegRevLex,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for at in [
        1,
        generation.checkpoints / 2,
        generation.checkpoints - 1,
        generation.checkpoints,
    ] {
        let mut cancelled = Count {
            cancel_at: Some(at),
            ..Count::default()
        };
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                TermOrder::DegRevLex,
                TermOrder::Lex,
                &mut cancelled,
                &mut Unbounded
            )
            .is_err()
        );
    }
    for at in [
        1,
        verification.checkpoints / 2,
        verification.checkpoints - 1,
        verification.checkpoints,
    ] {
        let mut cancelled = Count {
            cancel_at: Some(at),
            ..Count::default()
        };
        assert!(
            checked(
                &generators,
                5,
                &inputs,
                TermOrder::DegRevLex,
                TermOrder::Lex,
                &mut Unbounded,
                &mut cancelled
            )
            .is_err()
        );
    }
    let mut verification = Count::default();
    let mut cancelled = Count {
        cancel_at: Some(1),
        ..Count::default()
    };
    assert!(
        checked(
            &generators,
            5,
            &inputs,
            TermOrder::DegRevLex,
            TermOrder::Lex,
            &mut cancelled,
            &mut verification
        )
        .is_err()
    );
    assert_eq!(verification.checkpoints, 0);
}
