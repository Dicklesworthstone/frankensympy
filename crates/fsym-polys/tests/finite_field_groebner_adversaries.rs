//! Independently constructed witnesses exercise each complete-basis obligation.
//!
//! These fixtures do not ask the generator to create their expected evidence.
//! In particular, a tautological S-pair identity must not pass as a strict
//! representation, and neither direction of ideal inclusion is dispensable.

use fsym_budget::Unbounded;
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldGroebnerCertificate as Certificate, PrimeFieldGroebnerSPairCertificate as SPair,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    TermOrder, verify_prime_field_groebner_certificate as verify,
};

fn terms(values: &[(&[u32], u64)]) -> Terms {
    values.iter().map(|(m, c)| (m.to_vec(), *c)).collect()
}

fn ring() -> Vec<Symbol> {
    vec![Symbol::new("x"), Symbol::new("y")]
}

fn poly(prime: u64, source: &Terms) -> Polynomial {
    Polynomial::new(&ring(), prime, source).unwrap()
}

fn zero() -> Terms {
    Terms::new()
}

fn one() -> Terms {
    terms(&[(&[0, 0], 1)])
}

fn identity(width: usize, variables: usize) -> Vec<Vec<Terms>> {
    (0..width)
        .map(|i| {
            (0..width)
                .map(|j| {
                    if i == j {
                        [(vec![0; variables], 1)].into_iter().collect()
                    } else {
                        zero()
                    }
                })
                .collect()
        })
        .collect()
}

fn accepted(prime: u64, input: &[Polynomial], certificate: &Certificate) -> bool {
    verify(
        &ring(),
        prime,
        input,
        TermOrder::Lex,
        certificate,
        &mut Unbounded,
    )
    .is_ok()
}

// For I=<xy-1,y^2-1>, the lex basis is [x-y,y^2-1]. Every coefficient
// below is derived directly from these identities, independently of Buchberger.
fn complete_witness(prime: u64) -> (Vec<Polynomial>, Certificate) {
    let input = vec![
        poly(prime, &terms(&[(&[1, 1], 1), (&[0, 0], prime - 1)])),
        poly(prime, &terms(&[(&[0, 2], 1), (&[0, 0], prime - 1)])),
    ];
    let certificate = Certificate {
        basis: vec![
            terms(&[(&[1, 0], 1), (&[0, 1], prime - 1)]),
            input[1].terms().clone(),
        ],
        basis_in_input: vec![
            vec![terms(&[(&[0, 1], 1)]), terms(&[(&[1, 0], prime - 1)])],
            vec![zero(), one()],
        ],
        input_in_basis: vec![vec![terms(&[(&[0, 1], 1)]), one()], vec![zero(), one()]],
        s_pairs: vec![SPair {
            left: 0,
            right: 1,
            quotients: vec![one(), terms(&[(&[0, 1], prime - 1)])],
        }],
    };
    (input, certificate)
}

#[test]
fn hand_derived_complete_witnesses_are_accepted() {
    for prime in [2, 3, 5, 65521] {
        let (input, certificate) = complete_witness(prime);
        assert!(accepted(prime, &input, &certificate), "prime={prime}");
    }
}

#[test]
fn tautological_s_pair_identity_does_not_prove_a_groebner_basis() {
    for prime in [2, 5] {
        let (input, _) = complete_witness(prime);
        // S(xy-1,y^2-1)=x-y is represented by y*f1-x*f2. The
        // identity holds, but both products lead AT the LCM xy^2, not below.
        // This candidate is monic and reduced, yet is not a Groebner basis.
        let certificate = Certificate {
            basis: input.iter().map(|p| p.terms().clone()).collect(),
            basis_in_input: identity(2, 2),
            input_in_basis: identity(2, 2),
            s_pairs: vec![SPair {
                left: 0,
                right: 1,
                quotients: vec![terms(&[(&[0, 1], 1)]), terms(&[(&[1, 0], prime - 1)])],
            }],
        };
        assert!(!accepted(prime, &input, &certificate));
    }
}

#[test]
fn both_directions_of_original_ideal_inclusion_are_required() {
    let x = terms(&[(&[1, 0], 1)]);
    let input = vec![poly(5, &x)];
    // I is contained in <1>, but 1 is not in I.
    let larger = Certificate {
        basis: vec![one()],
        basis_in_input: vec![vec![zero()]],
        input_in_basis: vec![vec![x.clone()]],
        s_pairs: vec![],
    };
    assert!(!accepted(5, &input, &larger));
    // <x^2> is contained in I, but x is not in <x^2>.
    let smaller = Certificate {
        basis: vec![terms(&[(&[2, 0], 1)])],
        basis_in_input: vec![vec![x]],
        input_in_basis: vec![vec![zero()]],
        s_pairs: vec![],
    };
    assert!(!accepted(5, &input, &smaller));
}

#[test]
fn empty_basis_cannot_erase_a_nonzero_input() {
    let certificate = Certificate {
        basis: vec![],
        basis_in_input: vec![],
        input_in_basis: vec![vec![]],
        s_pairs: vec![],
    };
    assert!(accepted(5, &[poly(5, &zero())], &certificate));
    assert!(!accepted(
        5,
        &[poly(5, &terms(&[(&[1, 0], 1)]))],
        &certificate
    ));
}

#[test]
fn equivalent_ideals_do_not_excuse_nonmonic_or_unreduced_bases() {
    let x = terms(&[(&[1, 0], 1)]);
    let input = vec![poly(5, &x)];
    let nonmonic = Certificate {
        basis: vec![terms(&[(&[1, 0], 2)])],
        basis_in_input: vec![vec![terms(&[(&[0, 0], 2)])]],
        input_in_basis: vec![vec![terms(&[(&[0, 0], 3)])]],
        s_pairs: vec![],
    };
    assert!(!accepted(5, &input, &nonmonic));
    let redundant = Certificate {
        basis: vec![terms(&[(&[2, 0], 1)]), x.clone()],
        basis_in_input: vec![vec![x], vec![one()]],
        input_in_basis: vec![vec![zero(), one()]],
        s_pairs: vec![SPair {
            left: 0,
            right: 1,
            quotients: vec![zero(), zero()],
        }],
    };
    assert!(!accepted(5, &input, &redundant));
    // [x+y,y] is a Groebner basis with an exact strict S-pair witness, but
    // the y tail of the first polynomial is reducible by the second one.
    let tail = vec![terms(&[(&[1, 0], 1), (&[0, 1], 1)]), terms(&[(&[0, 1], 1)])];
    let input: Vec<_> = tail.iter().map(|t| poly(5, t)).collect();
    let certificate = Certificate {
        basis: tail,
        basis_in_input: identity(2, 2),
        input_in_basis: identity(2, 2),
        s_pairs: vec![SPair {
            left: 0,
            right: 1,
            quotients: vec![zero(), terms(&[(&[0, 1], 1)])],
        }],
    };
    assert!(!accepted(5, &input, &certificate));
}

#[test]
fn every_s_pair_is_required_once_in_canonical_orientation() {
    let generators = vec![Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let basis = vec![
        terms(&[(&[1, 0, 0], 1)]),
        terms(&[(&[0, 1, 0], 1)]),
        terms(&[(&[0, 0, 1], 1)]),
    ];
    let input: Vec<_> = basis
        .iter()
        .map(|t| Polynomial::new(&generators, 5, t).unwrap())
        .collect();
    let certificate = Certificate {
        basis,
        basis_in_input: identity(3, 3),
        input_in_basis: identity(3, 3),
        s_pairs: [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .map(|(left, right)| SPair {
                left,
                right,
                quotients: vec![zero(), zero(), zero()],
            })
            .collect(),
    };
    let check = |candidate: &Certificate| {
        verify(
            &generators,
            5,
            &input,
            TermOrder::Lex,
            candidate,
            &mut Unbounded,
        )
        .is_ok()
    };
    assert!(check(&certificate));
    let mut missing = certificate.clone();
    missing.s_pairs.pop();
    assert!(!check(&missing));
    for (left, right) in [(0, 1), (2, 0), (1, 1), (0, usize::MAX)] {
        let mut mutant = certificate.clone();
        mutant.s_pairs[1].left = left;
        mutant.s_pairs[1].right = right;
        assert!(!check(&mutant));
    }
    let mut reordered = certificate.clone();
    reordered.s_pairs.swap(0, 1);
    assert!(!check(&reordered));
    let mut short = certificate.clone();
    short.s_pairs[1].quotients.pop();
    assert!(!check(&short));
    let mut tautological_syzygy = certificate.clone();
    tautological_syzygy.s_pairs[0].quotients =
        vec![terms(&[(&[0, 1, 0], 1)]), terms(&[(&[1, 0, 0], 4)]), zero()];
    assert!(!check(&tautological_syzygy));
}

#[test]
fn witness_shapes_and_raw_terms_are_validated_before_cancellation() {
    let (input, certificate) = complete_witness(5);
    for invalid in [
        terms(&[(&[0, 0], 0)]),
        terms(&[(&[0, 0], 5)]),
        terms(&[(&[0], 0)]),
        terms(&[(&[65, 0], 0)]),
        terms(&[(&[u32::MAX, u32::MAX], 1)]),
    ] {
        // This slot is zero in the honest witness. All invalid variants are
        // either mathematically zero or beyond the admission boundary.
        let mut mutant = certificate.clone();
        mutant.basis_in_input[1][0] = invalid;
        assert!(!accepted(5, &input, &mutant));
    }
    let mut missing_row = certificate.clone();
    missing_row.input_in_basis.pop();
    assert!(!accepted(5, &input, &missing_row));
    let mut missing_coordinate = certificate.clone();
    missing_coordinate.basis_in_input[0].pop();
    assert!(!accepted(5, &input, &missing_coordinate));
    let mut altered_identity = certificate;
    altered_identity.s_pairs[0].quotients[0] = zero();
    assert!(!accepted(5, &input, &altered_identity));
}

#[test]
fn typed_ring_and_field_admission_cannot_be_skipped_for_zero_ideals() {
    let empty = Certificate {
        basis: vec![],
        basis_in_input: vec![],
        input_in_basis: vec![],
        s_pairs: vec![],
    };
    assert!(accepted(5, &[], &empty));
    assert!(verify(&[], 5, &[], TermOrder::Lex, &empty, &mut Unbounded).is_ok());
    for prime in [0, 1, 4, 65522, u64::MAX] {
        assert!(!accepted(prime, &[], &empty));
        assert!(verify(&[], prime, &[], TermOrder::Lex, &empty, &mut Unbounded).is_err());
    }
    let duplicate = vec![Symbol::new("x"), Symbol::new("x")];
    let too_many: Vec<_> = (0..9).map(|i| Symbol::new(format!("x{i}"))).collect();
    for invalid in [duplicate, too_many] {
        assert!(verify(&invalid, 5, &[], TermOrder::Lex, &empty, &mut Unbounded).is_err());
    }
    let zero_input = vec![poly(5, &zero())];
    let certificate = Certificate {
        input_in_basis: vec![vec![]],
        ..empty
    };
    assert!(accepted(5, &zero_input, &certificate));
    let shadow = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    for different in [
        vec![ring()[1].clone(), ring()[0].clone()],
        vec![shadow, ring()[1].clone()],
    ] {
        assert!(
            verify(
                &different,
                5,
                &zero_input,
                TermOrder::Lex,
                &certificate,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(!accepted(7, &zero_input, &certificate));
}

#[test]
fn explicit_constant_field_unit_witness_still_requires_exact_ideal_inclusions() {
    let scalar_one = terms(&[(&[], 1)]);
    let input = [Polynomial::new(&[], 3, &scalar_one).unwrap()];
    let honest = Certificate {
        basis: vec![scalar_one.clone()],
        basis_in_input: vec![vec![scalar_one.clone()]],
        input_in_basis: vec![vec![scalar_one]],
        s_pairs: vec![],
    };
    let accepted = verify(&[], 3, &input, TermOrder::Lex, &honest, &mut Unbounded).unwrap();
    assert_eq!(accepted.basis(), input);
    assert!(accepted.generators().is_empty());
    let mut bad = honest.clone();
    bad.basis_in_input[0][0].clear();
    assert!(verify(&[], 3, &input, TermOrder::Lex, &bad, &mut Unbounded).is_err());
    let mut bad = honest;
    bad.basis[0] = terms(&[(&[0], 1)]);
    assert!(verify(&[], 3, &input, TermOrder::Lex, &bad, &mut Unbounded).is_err());
}

#[test]
fn a_basis_verified_in_one_order_cannot_answer_for_another() {
    // [x-y^2] is monic in lex but has leading coefficient -1 in either
    // degree order. Same polynomial and same ideal do not bind the order.
    let source = terms(&[(&[1, 0], 1), (&[0, 2], 4)]);
    let input = vec![poly(5, &source)];
    let certificate = Certificate {
        basis: vec![source],
        basis_in_input: identity(1, 2),
        input_in_basis: identity(1, 2),
        s_pairs: vec![],
    };
    assert!(accepted(5, &input, &certificate));
    for order in [TermOrder::DegLex, TermOrder::DegRevLex] {
        assert!(verify(&ring(), 5, &input, order, &certificate, &mut Unbounded).is_err());
    }
}
