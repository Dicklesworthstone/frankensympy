//! Exact quotient algebras, complete order conversion and division adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity};
use fsym_polys::{
    MultivariatePoly, TermOrder, bounded_division, bounded_polynomial_from_expr,
    certificate_for_basis, certify_groebner_basis, checked_polynomial_division,
    verify_bounded_division, verify_complete_groebner_certificate,
};
use num_traits::One;
use std::collections::BTreeMap;

fn poly(gens: &[Symbol], terms: &[(Vec<u32>, i64, i64)]) -> MultivariatePoly {
    MultivariatePoly::new(
        gens.to_vec(),
        terms
            .iter()
            .map(|(m, n, d)| (m.clone(), BigRational::new((*n).into(), (*d).into())))
            .collect(),
    )
    .unwrap()
}

fn coupled() -> (Vec<Symbol>, Vec<MultivariatePoly>) {
    let gens = vec![Symbol::new("x"), Symbol::new("y")];
    let basis = vec![
        poly(&gens, &[(vec![2, 0], 1, 1), (vec![0, 1], -1, 1)]),
        poly(&gens, &[(vec![0, 2], 1, 1), (vec![1, 0], -1, 1)]),
    ];
    (gens, basis)
}

fn lex_coupled(gens: &[Symbol]) -> Vec<MultivariatePoly> {
    vec![
        poly(gens, &[(vec![1, 0], 1, 1), (vec![0, 2], -1, 1)]),
        poly(gens, &[(vec![0, 4], 1, 1), (vec![0, 1], -1, 1)]),
    ]
}

#[test]
fn graded_to_lex_fglm_matches_closed_form_and_retains_complete_evidence() {
    let (gens, input) = coupled();
    let source =
        certify_groebner_basis(&input, TermOrder::DegRevLex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert!(source.is_zero_dimensional());
    let monomials = source.standard_monomials(&mut Unbounded).unwrap();
    assert_eq!(
        monomials,
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );
    let certificate = source
        .fglm_with_certificate(TermOrder::Lex, &mut Unbounded)
        .unwrap();
    let result =
        verify_complete_groebner_certificate(source.basis(), &certificate, &mut Unbounded).unwrap();
    assert_eq!(result.basis(), lex_coupled(&gens));
    assert_eq!(result.receipt().s_pairs_verified(), 1);
    assert_eq!(
        result.standard_monomials(&mut Unbounded).unwrap(),
        vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![0, 3]]
    );
    assert_eq!(
        source
            .fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .unwrap()
            .basis(),
        result.basis()
    );
}

#[test]
fn all_order_pairs_preserve_the_same_ideal() {
    let (gens, graded) = coupled();
    for from in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let input = if from == TermOrder::Lex {
            lex_coupled(&gens)
        } else {
            graded.clone()
        };
        let source = certify_groebner_basis(&input, from, &mut Unbounded, &mut Unbounded).unwrap();
        for to in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
            let converted = source.fglm(to, &mut Unbounded, &mut Unbounded).unwrap();
            let expected = if to == TermOrder::Lex {
                lex_coupled(&gens)
            } else {
                graded.clone()
            };
            assert_eq!(converted.basis().len(), expected.len());
            for polynomial in &expected {
                assert!(converted.basis().contains(polynomial), "{from:?}->{to:?}");
            }
            assert_eq!(converted.order(), to);
            assert_eq!(
                converted.standard_monomials(&mut Unbounded).unwrap().len(),
                4
            );
        }
    }
}

#[test]
fn rational_nonmonic_source_is_checked_without_a_new_basis_search() {
    let g = vec![Symbol::new("x"), Symbol::new("y")];
    let supplied = vec![
        poly(&g, &[(vec![1, 0], 2, 1), (vec![0, 1], -3, 1)]),
        poly(&g, &[(vec![0, 2], 5, 1), (vec![0, 0], -20, 1)]),
    ];
    let source =
        certify_groebner_basis(&supplied, TermOrder::DegLex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(source.input_basis(), supplied);
    let result = source
        .fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
        .unwrap();
    assert_eq!(
        result.basis(),
        [
            poly(&g, &[(vec![1, 0], 1, 1), (vec![0, 1], -3, 2)]),
            poly(&g, &[(vec![0, 2], 1, 1), (vec![0, 0], -4, 1)]),
        ]
    );
}

#[test]
fn nonradical_quotients_keep_nilpotent_multiplicities() {
    let g = vec![Symbol::new("x"), Symbol::new("y")];
    let ideal = vec![
        poly(&g, &[(vec![2, 0], 1, 1)]),
        poly(&g, &[(vec![0, 3], 1, 1)]),
    ];
    let source =
        certify_groebner_basis(&ideal, TermOrder::DegRevLex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(source.standard_monomials(&mut Unbounded).unwrap().len(), 6);
    let result = source
        .fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
        .unwrap();
    assert_eq!(result.basis(), ideal);
    // The variety has one point, but its nonreduced quotient dimension is six.
    assert_eq!(result.standard_monomials(&mut Unbounded).unwrap().len(), 6);
}

#[test]
fn unit_zero_and_positive_dimensional_ideals_have_explicit_semantics() {
    let g = vec![Symbol::new("x"), Symbol::new("y")];
    let unit = certify_groebner_basis(
        &[MultivariatePoly::one(g.clone())],
        TermOrder::DegLex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(!unit.is_zero_dimensional());
    assert!(unit.standard_monomials(&mut Unbounded).unwrap().is_empty());
    assert_eq!(
        unit.fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .unwrap()
            .basis(),
        &[MultivariatePoly::one(g.clone())]
    );
    for input in [
        vec![MultivariatePoly::zero(g.clone())],
        vec![poly(&g, &[(vec![1, 1], 1, 1)])],
        vec![poly(&g, &[(vec![1, 0], 1, 1)])],
    ] {
        let source =
            certify_groebner_basis(&input, TermOrder::DegLex, &mut Unbounded, &mut Unbounded)
                .unwrap();
        assert!(!source.is_zero_dimensional());
        if source.basis().is_empty() {
            assert!(
                source
                    .fglm_with_certificate(TermOrder::DegLex, &mut Unbounded)
                    .is_err()
            );
        }
        assert!(source.standard_monomials(&mut Unbounded).is_err());
        assert!(
            source
                .fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
                .is_err()
        );
        assert_eq!(
            source
                .fglm(TermOrder::DegLex, &mut Unbounded, &mut Unbounded)
                .unwrap()
                .input_basis(),
            input
        );
    }
    let no_variables =
        certify_groebner_basis(&[], TermOrder::DegLex, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(
        no_variables.standard_monomials(&mut Unbounded).unwrap(),
        vec![Vec::<u32>::new()]
    );
    assert!(
        no_variables
            .fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .unwrap()
            .basis()
            .is_empty()
    );
}

#[test]
fn forged_wrong_order_nonreduced_and_non_groebner_sources_are_not_repaired() {
    let (g, graded) = coupled();
    assert!(
        certify_groebner_basis(&graded, TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err()
    );
    let not_groebner = vec![
        poly(&g, &[(vec![1, 1], 1, 1), (vec![0, 0], -1, 1)]),
        poly(&g, &[(vec![0, 2], 1, 1), (vec![0, 0], -1, 1)]),
    ];
    assert!(
        certify_groebner_basis(
            &not_groebner,
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let redundant = vec![
        poly(&g, &[(vec![1, 0], 1, 1)]),
        poly(&g, &[(vec![2, 0], 1, 1)]),
    ];
    assert!(
        certify_groebner_basis(&redundant, TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err()
    );
    let mut malformed = graded;
    malformed[0].terms.insert(vec![1], BigRational::one());
    assert!(
        certify_groebner_basis(
            &malformed,
            TermOrder::DegLex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn kernel_rejects_changed_order_omitted_pairs_and_tiny_forged_relations() {
    let (_, input) = coupled();
    let source =
        certify_groebner_basis(&input, TermOrder::DegRevLex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    let certificate = source
        .fglm_with_certificate(TermOrder::Lex, &mut Unbounded)
        .unwrap();
    let mut wrong_order = certificate.clone();
    wrong_order.candidate.order = TermOrder::DegLex;
    assert!(
        verify_complete_groebner_certificate(source.basis(), &wrong_order, &mut Unbounded).is_err()
    );
    let mut omitted_pair = certificate.clone();
    omitted_pair.s_pairs.clear();
    assert!(
        verify_complete_groebner_certificate(source.basis(), &omitted_pair, &mut Unbounded)
            .is_err()
    );
    let mut changed = certificate.clone();
    changed.candidate.basis[0].terms.insert(
        vec![0, 0],
        BigRational::new(BigInt::one(), BigInt::from(10).pow(80)),
    );
    assert!(
        verify_complete_groebner_certificate(source.basis(), &changed, &mut Unbounded).is_err()
    );
    let mut omitted_containment = certificate;
    omitted_containment.input_basis_witnesses.clear();
    assert!(
        verify_complete_groebner_certificate(source.basis(), &omitted_containment, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn quotient_dimension_degree_and_coefficient_limits_refuse_before_enumeration() {
    let g = vec![Symbol::new("x"), Symbol::new("y")];
    let large_quotient = vec![
        poly(&g, &[(vec![9, 0], 1, 1)]),
        poly(&g, &[(vec![0, 8], 1, 1)]),
    ];
    let source = certify_groebner_basis(
        &large_quotient,
        TermOrder::DegLex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert!(source.is_zero_dimensional());
    assert!(source.standard_monomials(&mut Unbounded).is_err());
    assert!(
        source
            .fglm(TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .is_err()
    );
    let boundary = vec![
        poly(&g, &[(vec![8, 0], 1, 1)]),
        poly(&g, &[(vec![0, 8], 1, 1)]),
    ];
    let boundary_source =
        certify_groebner_basis(&boundary, TermOrder::DegLex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(
        boundary_source
            .standard_monomials(&mut Unbounded)
            .unwrap()
            .len(),
        64
    );
    let degree = poly(&g, &[(vec![u32::MAX, 0], 1, 1)]);
    assert!(
        certify_groebner_basis(&[degree], TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err()
    );
    let coefficient = MultivariatePoly::new(
        g,
        BTreeMap::from([(
            vec![0, 0],
            BigRational::from_integer(BigInt::one() << 4096_u32),
        )]),
    )
    .unwrap();
    assert!(
        certify_groebner_basis(
            &[coefficient],
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn division_matches_a_hand_derived_multivariate_identity_in_every_order() {
    let g = vec![Symbol::new("x"), Symbol::new("y")];
    let f = poly(
        &g,
        &[(vec![2, 1], 1, 1), (vec![1, 2], 1, 1), (vec![0, 2], 1, 1)],
    );
    let divisors = vec![
        poly(&g, &[(vec![1, 1], 1, 1), (vec![0, 0], -1, 1)]),
        poly(&g, &[(vec![0, 2], 1, 1), (vec![0, 0], -1, 1)]),
    ];
    let expected_q = vec![
        poly(&g, &[(vec![1, 0], 1, 1), (vec![0, 1], 1, 1)]),
        MultivariatePoly::one(g.clone()),
    ];
    let expected_r = poly(
        &g,
        &[(vec![1, 0], 1, 1), (vec![0, 1], 1, 1), (vec![0, 0], 1, 1)],
    );
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let (q, r) =
            checked_polynomial_division(&f, &divisors, order, &mut Unbounded, &mut Unbounded)
                .unwrap();
        assert_eq!(q, expected_q);
        assert_eq!(r, expected_r);
    }
}

#[test]
fn division_checker_rejects_identity_preserving_unreduced_remainders_and_bad_coefficients() {
    let g = vec![Symbol::new("x")];
    let f = poly(&g, &[(vec![2], 1, 1), (vec![0], -1, 1)]);
    let divisor = vec![poly(&g, &[(vec![1], 1, 1), (vec![0], -1, 1)])];
    let (mut q, r) = bounded_division(&f, &divisor, TermOrder::Lex, &mut Unbounded).unwrap();
    assert!(r.is_zero());
    verify_bounded_division(&f, &divisor, &q, &r, TermOrder::Lex, &mut Unbounded).unwrap();
    assert!(
        verify_bounded_division(
            &f,
            &divisor,
            &[MultivariatePoly::zero(g.clone())],
            &f,
            TermOrder::Lex,
            &mut Unbounded
        )
        .is_err()
    );
    q[0].terms.insert(
        vec![0],
        BigRational::one() + BigRational::new(BigInt::one(), BigInt::from(10).pow(90)),
    );
    assert!(verify_bounded_division(&f, &divisor, &q, &r, TermOrder::Lex, &mut Unbounded).is_err());
    assert!(
        verify_bounded_division(&f, &divisor, &[], &r, TermOrder::Lex, &mut Unbounded).is_err()
    );
    let foreign = poly(&[Symbol::new("z")], &[(vec![0], 1, 1)]);
    assert!(bounded_division(&f, &[foreign], TermOrder::Lex, &mut Unbounded).is_err());
}

#[test]
fn zero_divisors_empty_lists_and_nonmembership_scope_stay_explicit() {
    let g = vec![Symbol::new("x"), Symbol::new("y")];
    let f = poly(&g, &[(vec![1, 0], 1, 1), (vec![0, 1], -1, 1)]);
    let (q, r) =
        checked_polynomial_division(&f, &[], TermOrder::Lex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert!(q.is_empty());
    assert_eq!(r, f);
    let (q, r) = checked_polynomial_division(
        &f,
        &[MultivariatePoly::zero(g.clone())],
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(q, vec![MultivariatePoly::zero(g.clone())]);
    assert_eq!(r, f);
    // x-y belongs to this ideal, although this non-GB divisor list cannot
    // reduce it. A division result alone must not certify non-membership.
    let divisors = vec![
        poly(&g, &[(vec![1, 1], 1, 1), (vec![0, 0], -1, 1)]),
        poly(&g, &[(vec![0, 2], 1, 1), (vec![0, 0], -1, 1)]),
    ];
    let (_, r) = checked_polynomial_division(
        &f,
        &divisors,
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(r, f);
    assert!(
        certify_groebner_basis(&divisors, TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err()
    );
}

#[derive(Default)]
struct Checkpoints {
    count: usize,
    cancel_at: Option<usize>,
}
impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.count += 1;
        if self.cancel_at == Some(self.count) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn generation_and_independent_verification_have_separate_cancellation_budgets() {
    let (_, input) = coupled();
    let source =
        certify_groebner_basis(&input, TermOrder::DegRevLex, &mut Unbounded, &mut Unbounded)
            .unwrap();
    let mut generation = Checkpoints::default();
    let mut checking = Checkpoints::default();
    source
        .fglm(TermOrder::Lex, &mut generation, &mut checking)
        .unwrap();
    for point in [1, generation.count / 2, generation.count] {
        let mut cancelled = Checkpoints {
            count: 0,
            cancel_at: Some(point),
        };
        assert!(
            source
                .fglm(TermOrder::Lex, &mut cancelled, &mut Unbounded)
                .is_err()
        );
    }
    for point in [1, checking.count / 2, checking.count] {
        let mut cancelled = Checkpoints {
            count: 0,
            cancel_at: Some(point),
        };
        assert!(
            source
                .fglm(TermOrder::Lex, &mut Unbounded, &mut cancelled)
                .is_err()
        );
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(
        source
            .fglm(TermOrder::Lex, &mut empty, &mut Unbounded)
            .is_err()
    );
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(
        source
            .fglm(TermOrder::Lex, &mut Unbounded, &mut empty)
            .is_err()
    );
    assert_eq!(source.basis(), input);
}

#[test]
fn direct_candidate_certification_checks_both_ideal_inclusions() {
    let g = vec![Symbol::new("x")];
    let source = vec![poly(&g, &[(vec![2], 1, 1)])];
    let larger_ideal = vec![poly(&g, &[(vec![1], 1, 1)])];
    assert!(certificate_for_basis(&source, &larger_ideal, TermOrder::Lex, &mut Unbounded).is_err());
    let smaller_ideal = vec![poly(&g, &[(vec![3], 1, 1)])];
    assert!(
        certificate_for_basis(&source, &smaller_ideal, TermOrder::Lex, &mut Unbounded).is_err()
    );
    let valid = certificate_for_basis(&source, &source, TermOrder::Lex, &mut Unbounded).unwrap();
    verify_complete_groebner_certificate(&source, &valid, &mut Unbounded).unwrap();
}

#[test]
fn bounded_expression_conversion_expands_exactly_and_preserves_typed_symbols() {
    let x = Symbol::new("x");
    let g = vec![x.clone()];
    let cubic = Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(1)]).pow(Expr::from_i64(3));
    assert_eq!(
        bounded_polynomial_from_expr(&cubic, &g, &mut Unbounded).unwrap(),
        poly(
            &g,
            &[
                (vec![3], 1, 1),
                (vec![2], 3, 1),
                (vec![1], 3, 1),
                (vec![0], 1, 1)
            ]
        )
    );
    let reciprocal = Expr::from_i64(2).pow(Expr::from_i64(-3));
    assert_eq!(
        bounded_polynomial_from_expr(&reciprocal, &g, &mut Unbounded).unwrap(),
        poly(&g, &[(vec![0], 1, 8)])
    );
    let distinct = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let generators = vec![x.clone(), distinct.clone()];
    let same_printed_names = Expr::Add(vec![Expr::Sym(x), Expr::Sym(distinct.clone())]);
    assert_eq!(
        bounded_polynomial_from_expr(&same_printed_names, &generators, &mut Unbounded).unwrap(),
        poly(&generators, &[(vec![1, 0], 1, 1), (vec![0, 1], 1, 1)])
    );
    assert!(bounded_polynomial_from_expr(&Expr::Sym(distinct), &g, &mut Unbounded).is_err());
    assert_eq!(
        bounded_polynomial_from_expr(&Expr::Mul(vec![]), &[], &mut Unbounded).unwrap(),
        MultivariatePoly::one(vec![])
    );
    assert!(
        bounded_polynomial_from_expr(&Expr::Add(vec![]), &g, &mut Unbounded)
            .unwrap()
            .is_zero()
    );
}

#[test]
fn bounded_conversion_checks_original_syntax_even_after_zero_and_cancellation() {
    let g = vec![Symbol::new("x")];
    let pole = Expr::from_i64(0).pow(Expr::from_i64(-1));
    for expression in [
        Expr::from_i64(0).pow(Expr::from_i64(0)),
        Expr::Add(vec![
            Expr::symbol("x"),
            Expr::Mul(vec![Expr::from_i64(-1), Expr::symbol("x")]),
        ])
        .pow(Expr::from_i64(0)),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::from_i64(0).pow(Expr::from_i64(0)),
        ]),
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        pole.pow(Expr::from_i64(0)),
        Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("foreign")]),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::symbol("x").pow(Expr::from_i64(-1)),
        ]),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function("sin".into(), vec![Expr::symbol("x")]),
        ]),
        Expr::from_i64(2).pow(Expr::rational(1, 2).unwrap()),
    ] {
        assert!(bounded_polynomial_from_expr(&expression, &g, &mut Unbounded).is_err());
    }
    let nonpolynomial = Expr::symbol("x").pow(Expr::from_i64(-1));
    let cancelled = Expr::Add(vec![
        nonpolynomial.clone(),
        Expr::Mul(vec![Expr::from_i64(-1), nonpolynomial]),
    ]);
    assert!(bounded_polynomial_from_expr(&cancelled, &g, &mut Unbounded).is_err());
}

#[test]
fn bounded_conversion_refuses_literal_and_expansion_bombs_before_large_growth() {
    let g = vec![Symbol::new("x")];
    let huge_power = Expr::from_i64(2).pow(Expr::from_i64(1_000_000_000));
    let mut meter = Checkpoints::default();
    assert!(bounded_polynomial_from_expr(&huge_power, &g, &mut meter).is_err());
    assert!(
        meter.count < 20,
        "large literal exponent must fail in syntax preflight"
    );
    for expression in [
        Expr::from_i64(2)
            .pow(Expr::from_i64(256))
            .pow(Expr::from_i64(256)),
        Expr::symbol("x")
            .pow(Expr::from_i64(256))
            .pow(Expr::from_i64(256)),
        Expr::Integer(BigInt::one() << 4096_u32),
        Expr::Rational(BigRational::new(BigInt::one(), BigInt::one() << 4096_u32)),
        Expr::Add(vec![Expr::from_i64(0); 4096]),
    ] {
        assert!(bounded_polynomial_from_expr(&expression, &g, &mut Unbounded).is_err());
    }
    let mut deep = Expr::symbol("x");
    for _ in 0..65 {
        deep = Expr::Add(vec![deep]);
    }
    assert!(bounded_polynomial_from_expr(&deep, &g, &mut Unbounded).is_err());
    let duplicate = vec![g[0].clone(), g[0].clone()];
    assert!(bounded_polynomial_from_expr(&Expr::from_i64(1), &duplicate, &mut Unbounded).is_err());
}

#[test]
fn bounded_conversion_observes_cancellation_before_during_and_after_expansion() {
    let g = vec![Symbol::new("x")];
    let expression = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]).pow(Expr::from_i64(8));
    let original = expression.clone();
    let mut completed = Checkpoints::default();
    bounded_polynomial_from_expr(&expression, &g, &mut completed).unwrap();
    for point in [1, completed.count / 2, completed.count] {
        let mut cancelled = Checkpoints {
            count: 0,
            cancel_at: Some(point),
        };
        assert!(bounded_polynomial_from_expr(&expression, &g, &mut cancelled).is_err());
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(bounded_polynomial_from_expr(&expression, &g, &mut empty).is_err());
    assert_eq!(expression, original);
}
