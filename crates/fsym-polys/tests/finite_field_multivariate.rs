//! Sparse field arithmetic fixtures, adversarial witnesses, ring binding,
//! independent finite-field evaluations, and whole-phase resource boundaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldMultivariateArithmeticCertificate as ArithmeticCertificate,
    PrimeFieldMultivariateDivisionCertificate as DivisionCertificate,
    PrimeFieldMultivariateOperation as Op,
    PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, TermOrder,
    checked_prime_field_multivariate_arithmetic as arithmetic,
    checked_prime_field_multivariate_division as division,
    verify_prime_field_multivariate_arithmetic as verify_arithmetic,
    verify_prime_field_multivariate_division as verify_division,
};

fn terms(values: &[(&[u32], u64)]) -> Terms {
    values.iter().map(|(m, c)| (m.to_vec(), *c)).collect()
}
fn ring() -> Vec<Symbol> { vec![Symbol::new("x"), Symbol::new("y")] }
fn poly(prime: u64, values: &[(&[u32], u64)]) -> Polynomial {
    Polynomial::new(&ring(), prime, &terms(values)).unwrap()
}
fn run(op: &Op, left: &Polynomial, right: Option<&Polynomial>) -> Polynomial {
    arithmetic(op, left, right, &mut Unbounded, &mut Unbounded).unwrap().into_result()
}
fn certificate(accepted: &fsym_polys::VerifiedPrimeFieldMultivariateDivision) -> DivisionCertificate {
    DivisionCertificate {
        quotients: accepted.quotients().iter().map(|q| q.terms().clone()).collect(),
        remainder: accepted.remainder().terms().clone(),
    }
}
fn evaluation(terms: &Terms, x: u64, y: u64, p: u64) -> u64 {
    terms.iter().fold(0, |sum, (m, c)| {
        let mut value = *c;
        for _ in 0..m[0] { value = value * x % p; }
        for _ in 0..m[1] { value = value * y % p; }
        (sum + value) % p
    })
}

#[test]
fn canonical_constructor_validates_raw_terms_before_trimming_zeros() {
    let accepted = poly(5, &[(&[1, 0], 3), (&[0, 0], 0)]);
    assert_eq!(accepted.terms(), &terms(&[(&[1, 0], 3)]));
    assert_eq!(accepted.generators(), ring());
    assert_eq!(accepted.prime(), 5);
    for value in [
        terms(&[(&[65, 0], 0)]),
        terms(&[(&[1], 0)]),
        terms(&[(&[0, 0], 5)]),
        terms(&[(&[u32::MAX, u32::MAX], 0)]),
    ] {
        assert!(Polynomial::new(&ring(), 5, &value).is_err());
    }
    for prime in [0, 1, 4, 65522, u64::MAX] {
        assert!(Polynomial::new(&ring(), prime, &Terms::new()).is_err());
    }
    assert!(Polynomial::new(&[], 5, &Terms::new()).is_err());
    assert!(Polynomial::new(&vec![Symbol::new("x"); 9], 5, &Terms::new()).is_err());
    assert!(Polynomial::new(&[Symbol::new("x"), Symbol::new("x")], 5, &Terms::new()).is_err());
}

#[test]
fn exact_sparse_arithmetic_and_characteristic_cancellation() {
    let a = poly(5, &[(&[1, 0], 2), (&[0, 1], 3), (&[0, 0], 4)]);
    let b = poly(5, &[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], 1)]);
    assert_eq!(run(&Op::Normalize, &a, None), a);
    assert_eq!(run(&Op::Negate, &a, None).terms(), &terms(&[(&[1, 0], 3), (&[0, 1], 2), (&[0, 0], 1)]));
    assert_eq!(run(&Op::Add, &a, Some(&b)).terms(), &terms(&[(&[1, 0], 3), (&[0, 1], 4)]));
    assert_eq!(run(&Op::Subtract, &a, Some(&b)).terms(), &terms(&[(&[1, 0], 1), (&[0, 1], 2), (&[0, 0], 3)]));
    assert_eq!(run(&Op::Multiply, &a, Some(&b)).terms(),
        &terms(&[(&[2, 0], 2), (&[0, 2], 3), (&[1, 0], 1), (&[0, 1], 2), (&[0, 0], 4)]));
    let f = poly(2, &[(&[1, 0], 1), (&[0, 1], 1)]);
    assert_eq!(run(&Op::Power(2), &f, None).terms(), &terms(&[(&[2, 0], 1), (&[0, 2], 1)]));
    assert!(run(&Op::Add, &f, Some(&f)).is_zero());
}

#[test]
fn monic_respects_selected_monomial_order() {
    let a = poly(5, &[(&[1, 0], 2), (&[0, 2], 3)]);
    assert_eq!(run(&Op::Monic(TermOrder::Lex), &a, None).terms(),
        &terms(&[(&[1, 0], 1), (&[0, 2], 4)]));
    for order in [TermOrder::DegLex, TermOrder::DegRevLex] {
        assert_eq!(run(&Op::Monic(order), &a, None).terms(),
            &terms(&[(&[1, 0], 4), (&[0, 2], 1)]));
    }
    assert!(run(&Op::Monic(TermOrder::Lex), &poly(5, &[]), None).is_zero());
}

#[test]
fn mixed_partial_derivatives_are_exact_in_positive_characteristic() {
    let a = poly(5, &[(&[3, 2], 2), (&[1, 1], 4), (&[0, 0], 1)]);
    assert_eq!(run(&Op::Derivative(vec![2, 1]), &a, None).terms(), &terms(&[(&[1, 1], 4)]));
    assert_eq!(run(&Op::Derivative(vec![0, 0]), &a, None), a);
    assert!(run(&Op::Derivative(vec![u32::MAX, 0]), &a, None).is_zero());
    let frobenius = poly(3, &[(&[3, 0], 1), (&[0, 3], 2)]);
    assert!(run(&Op::Derivative(vec![1, 0]), &frobenius, None).is_zero());
    assert!(arithmetic(&Op::Derivative(vec![1]), &a, None, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn power_zero_constants_and_degree_boundary_are_admitted_explicitly() {
    let zero = poly(5, &[]);
    let one = poly(5, &[(&[0, 0], 1)]);
    assert_eq!(run(&Op::Power(0), &zero, None), one);
    assert!(run(&Op::Power(u64::MAX), &zero, None).is_zero());
    let c = poly(5, &[(&[0, 0], 2)]);
    assert_eq!(run(&Op::Power(u64::MAX), &c, None).terms(), &terms(&[(&[0, 0], 3)]));
    let x = poly(5, &[(&[1, 0], 1)]);
    assert_eq!(run(&Op::Power(64), &x, None).terms(), &terms(&[(&[64, 0], 1)]));
    assert!(arithmetic(&Op::Power(65), &x, None, &mut Unbounded, &mut Unbounded).is_err());
    assert!(arithmetic(&Op::Power(u64::MAX), &x, None, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn immutable_receipts_retain_request_and_replay_independently() {
    let a = poly(7, &[(&[1, 0], 2), (&[0, 1], 3)]);
    let b = poly(7, &[(&[0, 1], 1), (&[0, 0], 4)]);
    let result = arithmetic(&Op::Multiply, &a, Some(&b), &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(result.operation(), &Op::Multiply);
    assert_eq!(result.left(), &a);
    assert_eq!(result.right(), Some(&b));
    let replay = verify_arithmetic(&Op::Multiply, &a, Some(&b),
        &ArithmeticCertificate { result: result.result().terms().clone() }, &mut Unbounded).unwrap();
    assert_eq!(replay, result);
}

#[test]
fn every_arithmetic_operation_rejects_modified_coefficients() {
    let a = poly(5, &[(&[2, 0], 2), (&[1, 1], 1), (&[0, 0], 3)]);
    let b = poly(5, &[(&[0, 1], 1), (&[0, 0], 1)]);
    for op in [Op::Normalize, Op::Negate, Op::Add, Op::Subtract, Op::Multiply,
        Op::Monic(TermOrder::Lex), Op::Power(3), Op::Derivative(vec![1, 0])] {
        let right = matches!(op, Op::Add | Op::Subtract | Op::Multiply).then_some(&b);
        let mut result = run(&op, &a, right).terms().clone();
        let constant = vec![0, 0];
        let value = (result.get(&constant).copied().unwrap_or(0) + 1) % 5;
        if value == 0 { result.remove(&constant); } else { result.insert(constant, value); }
        assert!(verify_arithmetic(&op, &a, right, &ArithmeticCertificate { result }, &mut Unbounded).is_err(), "{op:?}");
    }
}

#[test]
fn forged_shape_residue_source_and_operation_are_rejected() {
    let a = poly(5, &[(&[1, 0], 1)]);
    for result in [terms(&[(&[1], 1)]), terms(&[(&[65, 0], 1)]),
        terms(&[(&[1, 0], 6)]), terms(&[(&[1, 0], 1), (&[0, 0], 0)])] {
        assert!(verify_arithmetic(&Op::Normalize, &a, None, &ArithmeticCertificate { result }, &mut Unbounded).is_err());
    }
    let c = ArithmeticCertificate { result: run(&Op::Power(2), &a, None).terms().clone() };
    assert!(verify_arithmetic(&Op::Power(3), &a, None, &c, &mut Unbounded).is_err());
    assert!(verify_arithmetic(&Op::Power(2), &poly(5, &[(&[0, 1], 1)]), None, &c, &mut Unbounded).is_err());
    assert!(arithmetic(&Op::Normalize, &a, Some(&a), &mut Unbounded, &mut Unbounded).is_err());
    assert!(arithmetic(&Op::Add, &a, None, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn mismatched_characteristic_or_ordered_generators_never_shortcut_zero() {
    let zero = poly(5, &[]);
    let other_p = poly(7, &[]);
    let reversed = Polynomial::new(&[Symbol::new("y"), Symbol::new("x")], 5, &Terms::new()).unwrap();
    for bad in [other_p, reversed] {
        assert!(arithmetic(&Op::Multiply, &zero, Some(&bad), &mut Unbounded, &mut Unbounded).is_err());
        assert!(division(&zero, &[bad], TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err());
    }
}

#[test]
fn ordered_division_preserves_caller_coordinates() {
    let f = poly(5, &[(&[1, 2], 1), (&[0, 0], 1)]);
    let a = poly(5, &[(&[1, 1], 1), (&[0, 0], 4)]);
    let b = poly(5, &[(&[0, 2], 1), (&[0, 0], 4)]);
    let forward = division(&f, &[a.clone(), b.clone()], TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(forward.quotients()[0].terms(), &terms(&[(&[0, 1], 1)]));
    assert!(forward.quotients()[1].is_zero());
    assert_eq!(forward.remainder().terms(), &terms(&[(&[0, 1], 1), (&[0, 0], 1)]));
    let reverse = division(&f, &[b, a], TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(reverse.quotients()[0].terms(), &terms(&[(&[1, 0], 1)]));
    assert!(reverse.quotients()[1].is_zero());
    assert_eq!(reverse.remainder().terms(), &terms(&[(&[1, 0], 1), (&[0, 0], 1)]));
}

#[test]
fn division_handles_nonmonic_inputs_empty_lists_and_all_orders() {
    let f = poly(5, &[(&[2, 0], 2), (&[1, 1], 2), (&[1, 0], 2), (&[0, 1], 2), (&[0, 0], 3)]);
    let d = poly(5, &[(&[1, 0], 2), (&[0, 1], 2)]);
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let got = division(&f, std::slice::from_ref(&d), order, &mut Unbounded, &mut Unbounded).unwrap();
        assert_eq!(got.quotients()[0].terms(), &terms(&[(&[1, 0], 1), (&[0, 0], 1)]));
        assert_eq!(got.remainder().terms(), &terms(&[(&[0, 0], 3)]));
        assert_eq!(got.dividend(), &f);
        assert_eq!(got.divisors(), [d.clone()]);
        assert_eq!(got.order(), order);
        assert_eq!(verify_division(&f, std::slice::from_ref(&d), order, &certificate(&got), &mut Unbounded).unwrap(), got);
    }
    let got = division(&f, &[], TermOrder::Lex, &mut Unbounded, &mut Unbounded).unwrap();
    assert!(got.quotients().is_empty());
    assert_eq!(got.remainder(), &f);
    assert!(division(&poly(5, &[]), &[poly(5, &[])], TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err());
}

#[test]
fn division_requires_identity_reducedness_and_canonical_witnesses() {
    let x = poly(5, &[(&[1, 0], 1)]);
    let source = poly(5, &[(&[2, 0], 1)]);
    let valid = DivisionCertificate { quotients: vec![terms(&[(&[1, 0], 1)])], remainder: Terms::new() };
    assert!(verify_division(&source, std::slice::from_ref(&x), TermOrder::Lex, &valid, &mut Unbounded).is_ok());
    for bad in [
        DivisionCertificate { quotients: vec![Terms::new()], remainder: source.terms().clone() },
        DivisionCertificate { quotients: vec![], remainder: Terms::new() },
        DivisionCertificate { quotients: vec![terms(&[(&[1, 0], 2)])], remainder: Terms::new() },
        DivisionCertificate { quotients: valid.quotients.clone(), remainder: terms(&[(&[0, 0], 1)]) },
        DivisionCertificate { quotients: vec![terms(&[(&[1, 0], 6)])], remainder: Terms::new() },
    ] {
        assert!(verify_division(&source, std::slice::from_ref(&x), TermOrder::Lex, &bad, &mut Unbounded).is_err());
    }
}

#[test]
fn small_field_arithmetic_matches_independent_point_evaluation() {
    for p in [2, 3, 5] {
        for a in 0..p {
            for b in 0..p {
                let left = poly(p, &[(&[1, 0], a), (&[0, 1], b), (&[0, 0], 1)]);
                let right = poly(p, &[(&[1, 1], 1), (&[0, 0], b)]);
                let sum = run(&Op::Add, &left, Some(&right));
                let product = run(&Op::Multiply, &left, Some(&right));
                for x in 0..p {
                    for y in 0..p {
                        let u = evaluation(left.terms(), x, y, p);
                        let v = evaluation(right.terms(), x, y, p);
                        assert_eq!(evaluation(sum.terms(), x, y, p), (u + v) % p);
                        assert_eq!(evaluation(product.terms(), x, y, p), u * v % p);
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct Count { steps: u64, bytes: u64, checkpoints: usize, cancel: Option<usize> }
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
        for &(dimension, amount) in charges { self.charge(dimension, amount)?; }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel == Some(self.checkpoints) { Err(MeterError::Cancelled) } else { Ok(()) }
    }
}
fn cap(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn arithmetic_budgets_cover_full_generation_verification_and_publication() {
    let a = poly(7, &[(&[2, 0], 3), (&[1, 1], 1), (&[0, 0], 2)]);
    let (mut g, mut v) = (Count::default(), Count::default());
    arithmetic(&Op::Power(4), &a, None, &mut g, &mut v).unwrap();
    for (dimension, generated, verified) in [
        (Dimension::ComputeSteps, g.steps, v.steps), (Dimension::MemoryBytes, g.bytes, v.bytes),
    ] {
        assert!(generated > 0 && verified > 0);
        arithmetic(&Op::Power(4), &a, None, &mut cap(dimension, generated), &mut cap(dimension, verified)).unwrap();
        assert!(arithmetic(&Op::Power(4), &a, None, &mut cap(dimension, generated - 1), &mut Unbounded).is_err());
        assert!(arithmetic(&Op::Power(4), &a, None, &mut Unbounded, &mut cap(dimension, verified - 1)).is_err());
    }
    for (generation, checkpoints) in [(true, g.checkpoints), (false, v.checkpoints)] {
        for at in [1, checkpoints / 2, checkpoints] {
            let mut stopped = Count { cancel: Some(at), ..Count::default() };
            let result = if generation {
                arithmetic(&Op::Power(4), &a, None, &mut stopped, &mut Unbounded)
            } else { arithmetic(&Op::Power(4), &a, None, &mut Unbounded, &mut stopped) };
            assert!(result.is_err(), "phase={generation}, checkpoint={at}");
        }
    }
}

#[test]
fn division_budgets_cover_reconstruction_and_final_source_copies() {
    let f = poly(5, &[(&[3, 2], 1), (&[0, 0], 1)]);
    let divisors = [poly(5, &[(&[1, 1], 1), (&[0, 0], 4)]), poly(5, &[(&[0, 2], 1), (&[0, 0], 4)])];
    let (mut g, mut v) = (Count::default(), Count::default());
    division(&f, &divisors, TermOrder::Lex, &mut g, &mut v).unwrap();
    for (dimension, generated, verified) in [
        (Dimension::ComputeSteps, g.steps, v.steps), (Dimension::MemoryBytes, g.bytes, v.bytes),
    ] {
        division(&f, &divisors, TermOrder::Lex, &mut cap(dimension, generated), &mut cap(dimension, verified)).unwrap();
        assert!(division(&f, &divisors, TermOrder::Lex, &mut cap(dimension, generated - 1), &mut Unbounded).is_err());
        assert!(division(&f, &divisors, TermOrder::Lex, &mut Unbounded, &mut cap(dimension, verified - 1)).is_err());
    }
    for (generation, checkpoints) in [(true, g.checkpoints), (false, v.checkpoints)] {
        for at in [1, checkpoints / 2, checkpoints] {
            let mut stopped = Count { cancel: Some(at), ..Count::default() };
            let result = if generation {
                division(&f, &divisors, TermOrder::Lex, &mut stopped, &mut Unbounded)
            } else { division(&f, &divisors, TermOrder::Lex, &mut Unbounded, &mut stopped) };
            assert!(result.is_err(), "phase={generation}, checkpoint={at}");
        }
    }
}

#[test]
fn sparse_term_and_divisor_counts_are_bounded_before_copying() {
    let too_many: Terms = (0..4097u32).map(|i| (vec![i / 65, i % 65], 1)).collect();
    assert!(Polynomial::new(&ring(), 5, &too_many).is_err());
    let x = poly(5, &[(&[1, 0], 1)]);
    assert!(division(&x, &vec![x.clone(); 33], TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err());
    assert!(Polynomial::metered_new(&ring(), 5, x.terms(), &mut cap(Dimension::MemoryBytes, 0)).is_err());
    assert!(Polynomial::metered_new(&ring(), 5, x.terms(), &mut cap(Dimension::ComputeSteps, 0)).is_err());
}

#[test]
fn same_display_name_with_different_symbol_identity_is_a_different_ring() {
    let typed_x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([47; 32]));
    let changed = Polynomial::new(&[typed_x.clone(), Symbol::new("y")], 5, &Terms::new()).unwrap();
    let zero = poly(5, &[]);
    assert!(arithmetic(&Op::Multiply, &zero, Some(&changed), &mut Unbounded, &mut Unbounded).is_err());
    assert!(division(&zero, &[changed], TermOrder::Lex, &mut Unbounded, &mut Unbounded).is_err());
    // Distinct identities are valid distinct generators even with equal display.
    let separate = Polynomial::new(&[Symbol::new("x"), typed_x], 5, &terms(&[(&[1, 0], 1)])).unwrap();
    assert_eq!(separate.generators().len(), 2);
}

#[test]
fn three_variable_grevlex_uses_reverse_last_differing_exponent() {
    let generators = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let f = Polynomial::new(&generators, 5, &terms(&[(&[1, 0, 1], 2), (&[0, 2, 0], 3)])).unwrap();
    assert_eq!(run(&Op::Monic(TermOrder::DegLex), &f, None).terms(),
        &terms(&[(&[1, 0, 1], 1), (&[0, 2, 0], 4)]));
    assert_eq!(run(&Op::Monic(TermOrder::DegRevLex), &f, None).terms(),
        &terms(&[(&[1, 0, 1], 4), (&[0, 2, 0], 1)]));
}
