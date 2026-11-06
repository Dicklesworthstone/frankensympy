//! Constructive finite-field ideals, exact normal forms, and whole-phase budgets.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::Symbol;
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    TermOrder, bounded_prime_field_groebner_certificate as candidate,
    checked_prime_field_groebner_basis as groebner,
    checked_prime_field_groebner_reduction as reduce,
    checked_prime_field_multivariate_division as divide,
    verify_prime_field_groebner_certificate as verify,
};

const ORDERS: [TermOrder; 3] = [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex];

fn ring() -> Vec<Symbol> {
    vec![Symbol::new("x"), Symbol::new("y")]
}

fn terms(values: &[(&[u32], u64)]) -> Terms {
    values.iter().map(|(m, c)| (m.to_vec(), *c)).collect()
}

fn polynomial(generators: &[Symbol], prime: u64, values: &[(&[u32], u64)]) -> Polynomial {
    Polynomial::new(generators, prime, &terms(values)).unwrap()
}

fn coupled(prime: u64) -> Vec<Polynomial> {
    let variables = ring();
    vec![
        polynomial(&variables, prime, &[(&[1, 1], 1), (&[0, 0], prime - 1)]),
        polynomial(&variables, prime, &[(&[0, 2], 1), (&[1, 0], prime - 1)]),
    ]
}

#[test]
fn characteristic_changes_the_ideal_instead_of_reusing_a_rational_answer() {
    let variables = [Symbol::new("x")];
    for prime in [2, 3, 5, 65521] {
        let input = [
            polynomial(&variables, prime, &[(&[2], 1), (&[0], 1)]),
            polynomial(&variables, prime, &[(&[1], 1), (&[0], 1)]),
        ];
        for order in ORDERS {
            let accepted = groebner(
                &variables,
                prime,
                &input,
                order,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap();
            let expected = if prime == 2 {
                input[1].clone()
            } else {
                polynomial(&variables, prime, &[(&[0], 1)])
            };
            assert_eq!(
                accepted.basis(),
                [expected],
                "prime={prime}, order={order:?}"
            );
            assert_eq!(accepted.inputs(), input);
            assert_eq!(accepted.generators(), variables);
            assert_eq!(accepted.prime(), prime);
            assert_eq!(accepted.order(), order);
        }
    }
}

#[test]
fn coupled_ideal_has_exact_complete_bases_in_all_orders_and_characteristics() {
    let variables = ring();
    for prime in [2, 3, 5, 7, 65521] {
        let input = coupled(prime);
        for order in ORDERS {
            let accepted = groebner(
                &variables,
                prime,
                &input,
                order,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap();
            let expected = if order == TermOrder::Lex {
                vec![
                    polynomial(&variables, prime, &[(&[1, 0], 1), (&[0, 2], prime - 1)]),
                    polynomial(&variables, prime, &[(&[0, 3], 1), (&[0, 0], prime - 1)]),
                ]
            } else {
                vec![
                    polynomial(&variables, prime, &[(&[2, 0], 1), (&[0, 1], prime - 1)]),
                    input[0].clone(),
                    input[1].clone(),
                ]
            };
            assert_eq!(accepted.basis(), expected, "prime={prime}, order={order:?}");
            for source in &input {
                assert!(
                    reduce(
                        source,
                        accepted.basis(),
                        order,
                        &mut Unbounded,
                        &mut Unbounded
                    )
                    .unwrap()
                    .remainder()
                    .is_zero()
                );
            }
            let nonmember = polynomial(&variables, prime, &[(&[1, 0], 1)]);
            assert!(
                !reduce(
                    &nonmember,
                    accepted.basis(),
                    order,
                    &mut Unbounded,
                    &mut Unbounded
                )
                .unwrap()
                .remainder()
                .is_zero()
            );
        }
    }
}

#[test]
fn three_variable_orders_have_different_polynomial_sets() {
    let variables = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let input = [
        polynomial(&variables, 5, &[(&[2, 0, 0], 1), (&[0, 1, 1], 1)]),
        polynomial(&variables, 5, &[(&[1, 1, 0], 1), (&[0, 0, 2], 1)]),
    ];
    // y*f1-x*f2 = y^2*z-x*z^2. Its head changes between graded
    // lex and reverse lex; graded lex needs the additional y^3*z+z^4.
    // These full ordered fixtures also agree with isolated SymPy 1.14.0.
    let graded = groebner(
        &variables,
        5,
        &input,
        TermOrder::DegLex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        graded.basis(),
        [
            polynomial(&variables, 5, &[(&[0, 3, 1], 1), (&[0, 0, 4], 1)]),
            polynomial(&variables, 5, &[(&[1, 0, 2], 1), (&[0, 2, 1], 4)]),
            input[0].clone(),
            input[1].clone(),
        ]
    );
    let reverse = groebner(
        &variables,
        5,
        &input,
        TermOrder::DegRevLex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        reverse.basis(),
        [
            polynomial(&variables, 5, &[(&[0, 2, 1], 1), (&[1, 0, 2], 4)]),
            input[0].clone(),
            input[1].clone(),
        ]
    );
}

#[test]
fn original_zero_duplicate_and_nonmonic_inputs_survive_certificate_binding() {
    let variables = ring();
    let a = polynomial(&variables, 5, &[(&[1, 0], 2), (&[0, 1], 3), (&[0, 0], 1)]);
    let twice_a = polynomial(&variables, 5, &[(&[1, 0], 4), (&[0, 1], 1), (&[0, 0], 2)]);
    let zero = polynomial(&variables, 5, &[]);
    let input = vec![a.clone(), zero, twice_a, a];
    for order in ORDERS {
        let certificate = candidate(&variables, 5, &input, order, &mut Unbounded).unwrap();
        let accepted = verify(&variables, 5, &input, order, &certificate, &mut Unbounded).unwrap();
        assert_eq!(accepted.inputs(), input);
        assert_eq!(
            accepted.basis(),
            [polynomial(
                &variables,
                5,
                &[(&[1, 0], 1), (&[0, 1], 4), (&[0, 0], 3)]
            ),]
        );
        let mut permuted = input.clone();
        permuted.reverse();
        let changed = groebner(
            &variables,
            5,
            &permuted,
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(changed.basis(), accepted.basis());
        assert_eq!(changed.inputs(), permuted);
        assert_eq!(changed.into_basis(), accepted.basis());
    }
}

#[test]
fn zero_unit_and_nonradical_ideals_retain_their_distinct_meanings() {
    let variables = ring();
    let zero = polynomial(&variables, 5, &[]);
    let one = polynomial(&variables, 5, &[(&[0, 0], 1)]);
    let x = polynomial(&variables, 5, &[(&[1, 0], 1)]);
    for order in ORDERS {
        for input in [vec![], vec![zero.clone(), zero.clone()]] {
            let accepted =
                groebner(&variables, 5, &input, order, &mut Unbounded, &mut Unbounded).unwrap();
            assert!(accepted.basis().is_empty());
            assert_eq!(accepted.inputs(), input);
            let division =
                reduce(&x, accepted.basis(), order, &mut Unbounded, &mut Unbounded).unwrap();
            assert!(division.quotients().is_empty());
            assert_eq!(division.remainder(), &x);
        }
        let unit = groebner(
            &variables,
            5,
            &[zero.clone(), one.clone()],
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(unit.basis(), [one.clone()]);
        assert!(
            reduce(&x, unit.basis(), order, &mut Unbounded, &mut Unbounded)
                .unwrap()
                .remainder()
                .is_zero()
        );
        let nilpotent = [
            polynomial(&variables, 5, &[(&[2, 0], 1)]),
            polynomial(&variables, 5, &[(&[0, 3], 1)]),
        ];
        let accepted = groebner(
            &variables,
            5,
            &nilpotent,
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert!(
            !reduce(&x, accepted.basis(), order, &mut Unbounded, &mut Unbounded)
                .unwrap()
                .remainder()
                .is_zero()
        );
        for source in nilpotent {
            assert!(
                reduce(
                    &source,
                    accepted.basis(),
                    order,
                    &mut Unbounded,
                    &mut Unbounded
                )
                .unwrap()
                .remainder()
                .is_zero()
            );
        }
    }
    // Vanishing at every F_5 point does not make this the zero polynomial.
    let frobenius = polynomial(&variables, 5, &[(&[5, 0], 1), (&[1, 0], 4)]);
    assert_eq!(
        reduce(
            &frobenius,
            &[],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap()
        .remainder(),
        &frobenius
    );
}

#[test]
fn current_basis_validation_preserves_original_divisor_coordinates() {
    let variables = ring();
    let source = polynomial(&variables, 5, &[(&[1, 1], 1), (&[0, 0], 4)]);
    let a = polynomial(&variables, 5, &[(&[1, 0], 1), (&[0, 0], 4)]);
    let b = polynomial(&variables, 5, &[(&[0, 1], 1), (&[0, 0], 4)]);
    for order in ORDERS {
        let forward = reduce(
            &source,
            &[a.clone(), b.clone()],
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(forward.divisors(), [a.clone(), b.clone()]);
        assert_eq!(forward.quotients()[0].terms(), &terms(&[(&[0, 1], 1)]));
        assert_eq!(forward.quotients()[1].terms(), &terms(&[(&[0, 0], 1)]));
        assert!(forward.remainder().is_zero());
        let reverse = reduce(
            &source,
            &[b.clone(), a.clone()],
            order,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(reverse.divisors(), [b.clone(), a.clone()]);
        assert_eq!(reverse.quotients()[0].terms(), &terms(&[(&[1, 0], 1)]));
        assert_eq!(reverse.quotients()[1].terms(), &terms(&[(&[0, 0], 1)]));
        assert!(reverse.remainder().is_zero());
    }
    let zero = polynomial(&variables, 5, &[]);
    let incomplete = [
        source,
        polynomial(&variables, 5, &[(&[0, 2], 1), (&[0, 0], 4)]),
    ];
    assert!(
        reduce(
            &zero,
            &incomplete,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        reduce(
            &zero,
            &[a.clone(), a],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        reduce(
            &zero,
            &[polynomial(&variables, 5, &[(&[1, 0], 2)])],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        reduce(
            &zero,
            std::slice::from_ref(&zero),
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn degree_boundary_allows_exact_s_pair_cancellation_and_input_counts_are_bounded() {
    let variables = ring();
    let input = [
        polynomial(&variables, 5, &[(&[64, 0], 1)]),
        polynomial(&variables, 5, &[(&[0, 64], 1)]),
    ];
    for order in ORDERS {
        assert_eq!(
            groebner(&variables, 5, &input, order, &mut Unbounded, &mut Unbounded)
                .unwrap()
                .basis(),
            input
        );
    }
    let repeated = vec![input[0].clone(); 32];
    assert_eq!(
        groebner(
            &variables,
            5,
            &repeated,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .unwrap()
        .basis(),
        [input[0].clone()]
    );
    assert!(
        groebner(
            &variables,
            5,
            &vec![input[0].clone(); 33],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[derive(Default)]
struct Count {
    checkpoints: usize,
    steps: u64,
    bytes: u64,
    allocations: u64,
    cancel: Option<usize>,
}

impl BudgetMeter for Count {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.steps += amount,
            Dimension::MemoryBytes => self.bytes += amount,
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
fn complete_basis_and_receipt_publication_share_the_callers_phase_budgets() {
    let variables = ring();
    let input = coupled(5);
    let (mut g, mut v) = (Count::default(), Count::default());
    groebner(&variables, 5, &input, TermOrder::Lex, &mut g, &mut v).unwrap();
    for (dimension, generated, verified) in [
        (Dimension::ComputeSteps, g.steps, v.steps),
        (Dimension::MemoryBytes, g.bytes, v.bytes),
        (Dimension::AllocationCount, g.allocations, v.allocations),
    ] {
        assert!(generated > 1 && verified > 1);
        groebner(
            &variables,
            5,
            &input,
            TermOrder::Lex,
            &mut cap(dimension, generated),
            &mut cap(dimension, verified),
        )
        .unwrap();
        assert!(
            groebner(
                &variables,
                5,
                &input,
                TermOrder::Lex,
                &mut cap(dimension, generated - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            groebner(
                &variables,
                5,
                &input,
                TermOrder::Lex,
                &mut Unbounded,
                &mut cap(dimension, verified - 1)
            )
            .is_err()
        );
        let mut previously_used = cap(dimension, generated);
        BudgetMeter::charge(&mut previously_used, dimension, 1).unwrap();
        assert!(
            groebner(
                &variables,
                5,
                &input,
                TermOrder::Lex,
                &mut previously_used,
                &mut Unbounded
            )
            .is_err()
        );
    }
    for (generation, checkpoints) in [(true, g.checkpoints), (false, v.checkpoints)] {
        for at in [
            1,
            checkpoints / 4,
            checkpoints / 2,
            checkpoints * 3 / 4,
            checkpoints - 1,
            checkpoints,
        ] {
            let mut stopped = Count {
                cancel: Some(at),
                ..Count::default()
            };
            let result = if generation {
                groebner(
                    &variables,
                    5,
                    &input,
                    TermOrder::Lex,
                    &mut stopped,
                    &mut Unbounded,
                )
            } else {
                groebner(
                    &variables,
                    5,
                    &input,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut stopped,
                )
            };
            assert!(result.is_err(), "generation={generation}, checkpoint={at}");
        }
    }
}

#[test]
fn validating_a_current_basis_and_dividing_cannot_reset_either_phase_ledger() {
    let variables = ring();
    let basis = groebner(
        &variables,
        5,
        &coupled(5),
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    let source = polynomial(&variables, 5, &[(&[3, 2], 1), (&[0, 0], 3)]);
    let (mut g, mut v) = (Count::default(), Count::default());
    reduce(&source, basis.basis(), TermOrder::Lex, &mut g, &mut v).unwrap();
    let (mut plain_g, mut plain_v) = (Count::default(), Count::default());
    divide(
        &source,
        basis.basis(),
        TermOrder::Lex,
        &mut plain_g,
        &mut plain_v,
    )
    .unwrap();
    assert!(g.steps > plain_g.steps && v.steps > plain_v.steps);
    assert!(g.bytes > plain_g.bytes && v.bytes > plain_v.bytes);
    for (dimension, generated, verified) in [
        (Dimension::ComputeSteps, g.steps, v.steps),
        (Dimension::MemoryBytes, g.bytes, v.bytes),
        (Dimension::AllocationCount, g.allocations, v.allocations),
    ] {
        reduce(
            &source,
            basis.basis(),
            TermOrder::Lex,
            &mut cap(dimension, generated),
            &mut cap(dimension, verified),
        )
        .unwrap();
        assert!(
            reduce(
                &source,
                basis.basis(),
                TermOrder::Lex,
                &mut cap(dimension, generated - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            reduce(
                &source,
                basis.basis(),
                TermOrder::Lex,
                &mut Unbounded,
                &mut cap(dimension, verified - 1)
            )
            .is_err()
        );
    }
    assert!(
        reduce(
            &source,
            basis.basis(),
            TermOrder::Lex,
            &mut cap(Dimension::ComputeSteps, plain_g.steps),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        reduce(
            &source,
            basis.basis(),
            TermOrder::Lex,
            &mut Unbounded,
            &mut cap(Dimension::ComputeSteps, plain_v.steps)
        )
        .is_err()
    );
}

#[test]
fn every_observed_safe_point_can_cancel_without_publishing_an_accepted_result() {
    let variables = [Symbol::new("x")];
    let input = [polynomial(&variables, 5, &[(&[1], 2), (&[0], 1)])];
    let (mut g, mut v) = (Count::default(), Count::default());
    let accepted = groebner(&variables, 5, &input, TermOrder::Lex, &mut g, &mut v).unwrap();
    for (generation, checkpoints) in [(true, g.checkpoints), (false, v.checkpoints)] {
        for at in 1..=checkpoints {
            let mut stopped = Count {
                cancel: Some(at),
                ..Count::default()
            };
            let result = if generation {
                groebner(
                    &variables,
                    5,
                    &input,
                    TermOrder::Lex,
                    &mut stopped,
                    &mut Unbounded,
                )
            } else {
                groebner(
                    &variables,
                    5,
                    &input,
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut stopped,
                )
            };
            assert!(result.is_err(), "generation={generation}, checkpoint={at}");
        }
    }
    let source = polynomial(&variables, 5, &[(&[2], 1)]);
    let (mut g, mut v) = (Count::default(), Count::default());
    reduce(&source, accepted.basis(), TermOrder::Lex, &mut g, &mut v).unwrap();
    for (generation, checkpoints) in [(true, g.checkpoints), (false, v.checkpoints)] {
        for at in 1..=checkpoints {
            let mut stopped = Count {
                cancel: Some(at),
                ..Count::default()
            };
            let result = if generation {
                reduce(
                    &source,
                    accepted.basis(),
                    TermOrder::Lex,
                    &mut stopped,
                    &mut Unbounded,
                )
            } else {
                reduce(
                    &source,
                    accepted.basis(),
                    TermOrder::Lex,
                    &mut Unbounded,
                    &mut stopped,
                )
            };
            assert!(
                result.is_err(),
                "reduction generation={generation}, checkpoint={at}"
            );
        }
    }
}
