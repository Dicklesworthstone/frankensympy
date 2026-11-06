//! Exact field arithmetic, independent small-field enumeration, and forged
//! witnesses that a single product or Bezout identity would incorrectly accept.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PolyError, PrimeFieldArithmeticCertificate, PrimeFieldArithmeticOperation as Op,
    PrimeFieldGcdWitness, PrimeFieldPolynomial, VerifiedPrimeFieldArithmetic,
    checked_prime_field_arithmetic, verify_prime_field_arithmetic,
};

fn polynomial(prime: u64, values: &[u64]) -> PrimeFieldPolynomial {
    PrimeFieldPolynomial::new(Symbol::new("x"), prime, values.to_vec()).unwrap()
}
fn run(op: Op, prime: u64, left: &[u64], right: Option<&[u64]>) -> VerifiedPrimeFieldArithmetic {
    let left = polynomial(prime, left);
    let right = right.map(|values| polynomial(prime, values));
    checked_prime_field_arithmetic(op, &left, right.as_ref(), &mut Unbounded, &mut Unbounded)
        .unwrap()
}
fn result(op: Op, prime: u64, left: &[u64], right: Option<&[u64]>) -> Vec<Vec<u64>> {
    run(op, prime, left, right).into_results()
}
fn accepts(
    op: Op,
    prime: u64,
    left: &[u64],
    right: Option<&[u64]>,
    candidate: &PrimeFieldArithmeticCertificate,
) -> bool {
    verify_prime_field_arithmetic(
        op,
        &polynomial(prime, left),
        right.map(|values| polynomial(prime, values)).as_ref(),
        candidate,
        &mut Unbounded,
    )
    .is_ok()
}

#[test]
fn arithmetic_preserves_characteristic_and_canonical_coefficients() {
    assert_eq!(result(Op::Normalize, 5, &[4, 3, 0, 0], None), [vec![4, 3]]);
    assert_eq!(result(Op::Negate, 5, &[4, 3], None), [vec![1, 2]]);
    assert_eq!(result(Op::Add, 5, &[4, 3], Some(&[2, 4])), [vec![1, 2]]);
    assert_eq!(
        result(Op::Subtract, 5, &[4, 3], Some(&[2, 4])),
        [vec![2, 4]]
    );
    assert_eq!(
        result(Op::Multiply, 5, &[4, 3], Some(&[2, 4])),
        [vec![3, 2, 2]]
    );
    assert_eq!(
        result(Op::Multiply, 2, &[1, 1], Some(&[1, 1])),
        [vec![1, 0, 1]]
    );
    assert_eq!(result(Op::Monic, 5, &[4, 2], None), [vec![2, 1]]);
    assert_eq!(result(Op::Monic, 2, &[1, 0, 1], None), [vec![1, 0, 1]]);
}

#[test]
fn accepted_receipt_retains_the_complete_request_and_replays() {
    let a = polynomial(7, &[6, 2, 3]);
    let b = polynomial(7, &[4, 5]);
    let accepted =
        checked_prime_field_arithmetic(Op::Divide, &a, Some(&b), &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(accepted.operation(), Op::Divide);
    assert_eq!(accepted.left(), &a);
    assert_eq!(accepted.right(), Some(&b));
    assert_eq!(accepted.variable(), a.variable());
    assert_eq!(accepted.prime(), 7);
    let replay = verify_prime_field_arithmetic(
        Op::Divide,
        &a,
        Some(&b),
        accepted.certificate(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(accepted.results(), replay.results());
    let zero = run(Op::Normalize, 2, &[], None);
    assert!(zero.right().is_none());
    assert_eq!(zero.results(), [Vec::<u64>::new()]);
}

#[test]
fn division_uses_field_units_and_preserves_remainder_degree() {
    assert_eq!(
        result(Op::Divide, 5, &[1, 0, 1], Some(&[2, 2])),
        [vec![2, 3], vec![2]]
    );
    assert_eq!(
        result(Op::Divide, 2, &[1, 0, 1], Some(&[1, 1])),
        [vec![1, 1], vec![]]
    );
    assert_eq!(
        result(Op::Divide, 5, &[1, 2], Some(&[0, 0, 1])),
        [vec![], vec![1, 2]]
    );
    assert_eq!(
        result(Op::Divide, 7, &[3, 2, 6], Some(&[3])),
        [vec![1, 3, 2], vec![]]
    );
}

#[test]
fn gcd_extended_gcd_and_lcm_retain_characteristic_multiplicities() {
    let a = [1, 0, 1];
    let b = [1, 0, 0, 1];
    assert_eq!(result(Op::Gcd, 2, &a, Some(&b)), [vec![1, 1]]);
    assert_eq!(result(Op::Lcm, 2, &a, Some(&b)), [vec![1, 1, 0, 1, 1]]);
    let extended = run(Op::ExtendedGcd, 2, &a, Some(&b));
    assert_eq!(extended.results()[2], [1, 1]);
    assert!(accepts(
        Op::ExtendedGcd,
        2,
        &a,
        Some(&b),
        extended.certificate()
    ));
    assert_eq!(result(Op::Gcd, 5, &[3, 3], Some(&[2, 2])), [vec![1, 1]]);
    assert_eq!(result(Op::Lcm, 5, &[3, 3], Some(&[2, 2])), [vec![1, 1]]);
    assert_eq!(result(Op::Gcd, 3, &[1, 0, 1], Some(&[2, 1])), [vec![1]]);
}

#[test]
fn inverses_and_proved_nonunits_have_distinct_complete_results() {
    assert_eq!(
        result(Op::Invert, 5, &[0, 1], Some(&[3, 0, 1])),
        [vec![0, 3]]
    );
    assert_eq!(
        result(Op::Invert, 3, &[0, 1], Some(&[1, 0, 1])),
        [vec![0, 2]]
    );
    assert_eq!(result(Op::Invert, 5, &[2], Some(&[1, 0, 1])), [vec![3]]);
    assert!(result(Op::Invert, 5, &[0, 1], Some(&[0, 0, 1])).is_empty());
    assert!(result(Op::Invert, 2, &[], Some(&[1, 1])).is_empty());
    // A nonzero constant modulus gives the zero quotient ring, whose reduced
    // inverse is the zero polynomial; this is not an absent inverse.
    assert_eq!(
        result(Op::Invert, 7, &[1, 2], Some(&[4])),
        [Vec::<u64>::new()]
    );
    assert_eq!(result(Op::Invert, 7, &[], Some(&[4])), [Vec::<u64>::new()]);
}

#[test]
fn derivatives_include_frobenius_zero_and_whole_call_high_order() {
    assert_eq!(
        result(Op::Derivative(1), 2, &[1, 0, 1, 0, 1], None),
        [Vec::<u64>::new()]
    );
    assert_eq!(
        result(Op::Derivative(1), 3, &[1, 2, 1, 1, 2], None),
        [vec![2, 2, 0, 2]]
    );
    assert_eq!(
        result(Op::Derivative(2), 3, &[1, 2, 1, 1, 2], None),
        [vec![2]]
    );
    assert_eq!(
        result(Op::Derivative(0), 5, &[1, 2, 3], None),
        [vec![1, 2, 3]]
    );
    assert_eq!(
        result(Op::Derivative(u32::MAX), 5, &[1, 2, 3], None),
        [Vec::<u64>::new()]
    );
}

#[test]
fn powers_and_composition_are_polynomial_identities_not_point_samples() {
    for prime in [2, 3, 5] {
        let power = result(Op::Power(prime), prime, &[1, 1], None);
        let mut expected = vec![0; prime as usize + 1];
        expected[0] = 1;
        expected[prime as usize] = 1;
        assert_eq!(power, [expected]);
    }
    assert_eq!(result(Op::Power(u64::MAX), 3, &[2], None), [vec![2]]);
    assert_eq!(result(Op::Power(0), 3, &[], None), [vec![1]]);
    assert_eq!(
        result(Op::Power(u64::MAX), 3, &[], None),
        [Vec::<u64>::new()]
    );
    assert_eq!(
        result(Op::Compose, 5, &[3, 2, 1], Some(&[1, 0, 1])),
        [vec![1, 0, 4, 0, 1]]
    );
    // x^p-x vanishes at every F_p point but remains a nonzero polynomial.
    assert_eq!(
        result(Op::Compose, 2, &[0, 1, 1], Some(&[1, 1])),
        [vec![0, 1, 1]]
    );
    assert_eq!(result(Op::Compose, 5, &[3, 2, 1], Some(&[])), [vec![3]]);
}

#[test]
fn zero_and_constant_operations_follow_field_and_zero_ring_conventions() {
    for op in [
        Op::Normalize,
        Op::Negate,
        Op::Monic,
        Op::Derivative(0),
        Op::Derivative(3),
    ] {
        assert_eq!(result(op, 2, &[], None), [Vec::<u64>::new()]);
    }
    for op in [
        Op::Add,
        Op::Subtract,
        Op::Multiply,
        Op::Gcd,
        Op::Lcm,
        Op::Compose,
    ] {
        assert_eq!(result(op, 2, &[], Some(&[])), [Vec::<u64>::new()]);
    }
    assert_eq!(
        result(Op::ExtendedGcd, 2, &[], Some(&[])),
        vec![Vec::<u64>::new(), vec![], vec![]]
    );
    assert_eq!(
        result(Op::ExtendedGcd, 5, &[2], Some(&[])),
        [vec![3], vec![], vec![1]]
    );
    assert_eq!(
        result(Op::ExtendedGcd, 5, &[], Some(&[3])),
        [vec![], vec![2], vec![1]]
    );
    assert_eq!(result(Op::Monic, 5, &[3], None), [vec![1]]);
    assert_eq!(
        result(Op::Divide, 5, &[], Some(&[3])),
        [Vec::<u64>::new(), vec![]]
    );
    for op in [Op::Divide, Op::Invert] {
        assert_eq!(
            checked_prime_field_arithmetic(
                op,
                &polynomial(5, &[1]),
                Some(&polynomial(5, &[])),
                &mut Unbounded,
                &mut Unbounded
            )
            .unwrap_err(),
            PolyError::DivisionByZero
        );
    }
}

fn trimmed(mut coefficients: Vec<u64>) -> Vec<u64> {
    while coefficients.last() == Some(&0) {
        coefficients.pop();
    }
    coefficients
}
fn small_remainder(input: &[u64], divisor: &[u64], prime: u64) -> Vec<u64> {
    assert!(!divisor.is_empty());
    let inverse = (1..prime)
        .find(|value| value * divisor[divisor.len() - 1] % prime == 1)
        .unwrap();
    let mut remainder = input.to_vec();
    if remainder.len() >= divisor.len() {
        for power in (divisor.len() - 1..remainder.len()).rev() {
            let quotient = remainder[power] * inverse % prime;
            for (index, coefficient) in divisor.iter().enumerate() {
                let at = power + 1 - divisor.len() + index;
                remainder[at] = (remainder[at] + prime - quotient * coefficient % prime) % prime;
            }
        }
    }
    trimmed(remainder)
}
fn enumerate_polynomial(mut code: u64, length: usize, prime: u64) -> Vec<u64> {
    let mut values = vec![0; length];
    for value in &mut values {
        *value = code % prime;
        code /= prime;
    }
    trimmed(values)
}
// Complete enumeration of monic common divisors, independent of Euclid and
// of the certificate's Bezout data. Every field in this campaign is tiny.
fn exhaustive_gcd(left: &[u64], right: &[u64], prime: u64) -> Vec<u64> {
    if left.is_empty() && right.is_empty() {
        return Vec::new();
    }
    let degree = if left.is_empty() {
        right.len() - 1
    } else if right.is_empty() {
        left.len() - 1
    } else {
        left.len().min(right.len()) - 1
    };
    for d in (0..=degree).rev() {
        for mut code in 0..prime.pow(d as u32) {
            let mut divisor = vec![0; d + 1];
            for coefficient in &mut divisor[..d] {
                *coefficient = code % prime;
                code /= prime;
            }
            divisor[d] = 1;
            if small_remainder(left, &divisor, prime).is_empty()
                && small_remainder(right, &divisor, prime).is_empty()
            {
                return divisor;
            }
        }
    }
    unreachable!()
}

#[test]
fn exhaustive_small_fields_agree_with_complete_common_divisor_enumeration() {
    let mut pairs = 0;
    for (prime, length) in [(2u64, 3usize), (3, 2), (5, 2)] {
        for a in 0..prime.pow(length as u32) {
            for b in 0..prime.pow(length as u32) {
                let left = enumerate_polynomial(a, length, prime);
                let right = enumerate_polynomial(b, length, prime);
                let expected = exhaustive_gcd(&left, &right, prime);
                assert_eq!(
                    result(Op::Gcd, prime, &left, Some(&right)).as_slice(),
                    std::slice::from_ref(&expected)
                );
                let extended = result(Op::ExtendedGcd, prime, &left, Some(&right));
                assert_eq!(extended[2], expected);
                if !right.is_empty() {
                    let divided = result(Op::Divide, prime, &left, Some(&right));
                    assert_eq!(divided[1], small_remainder(&left, &right, prime));
                    let inverse = result(Op::Invert, prime, &left, Some(&right));
                    assert_eq!(!inverse.is_empty(), expected == [1]);
                }
                pairs += 1;
            }
        }
    }
    assert_eq!(pairs, 770);
}

#[test]
fn maximum_characteristic_and_degree_use_wide_exact_column_sums() {
    let input = vec![65_520; 33];
    let expected: Vec<_> = (0..65)
        .map(|power| (power.min(64 - power) + 1) as u64)
        .collect();
    assert_eq!(
        result(Op::Multiply, 65_521, &input, Some(&input)),
        [expected]
    );
    let mut degree64 = vec![0; 65];
    degree64[0] = 1;
    degree64[64] = 1;
    assert_eq!(result(Op::Power(64), 2, &[1, 1], None), [degree64.clone()]);
    assert_eq!(result(Op::Normalize, 2, &degree64, None), [degree64]);
    for op in [Op::Power(65), Op::Power(u64::MAX)] {
        assert!(
            checked_prime_field_arithmetic(
                op,
                &polynomial(2, &[1, 1]),
                None,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    let a = polynomial(2, &vec![1; 65]);
    let b = polynomial(2, &[0, 1]);
    assert!(
        checked_prime_field_arithmetic(Op::Multiply, &a, Some(&b), &mut Unbounded, &mut Unbounded)
            .is_err()
    );
    let cubic = polynomial(2, &[0, 0, 0, 1]);
    assert!(
        checked_prime_field_arithmetic(
            Op::Compose,
            &a,
            Some(&cubic),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn different_typed_fields_variables_and_wrong_arities_refuse_before_shortcuts() {
    let zero = polynomial(2, &[]);
    let other_prime = polynomial(3, &[]);
    let other_name = PrimeFieldPolynomial::new(Symbol::new("y"), 2, vec![]).unwrap();
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([73; 32]));
    let other_identity = PrimeFieldPolynomial::new(typed, 2, vec![]).unwrap();
    for other in [other_prime, other_name, other_identity] {
        assert!(
            checked_prime_field_arithmetic(
                Op::Multiply,
                &zero,
                Some(&other),
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked_prime_field_arithmetic(Op::Add, &zero, None, &mut Unbounded, &mut Unbounded)
            .is_err()
    );
    assert!(
        checked_prime_field_arithmetic(
            Op::Negate,
            &zero,
            Some(&zero),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn remainder_identity_without_reducedness_is_rejected() {
    let source = [1, 0, 1];
    let divisor = [1, 1];
    let mut certificate = run(Op::Divide, 5, &source, Some(&divisor))
        .certificate()
        .clone();
    certificate.results = vec![vec![], source.to_vec()];
    assert!(!accepts(
        Op::Divide,
        5,
        &source,
        Some(&divisor),
        &certificate
    ));
    certificate.results = vec![vec![2], vec![]];
    assert!(!accepts(
        Op::Divide,
        5,
        &source,
        Some(&divisor),
        &certificate
    ));
}

#[test]
fn common_divisibility_without_bezout_does_not_certify_a_greatest_divisor() {
    let a = [4, 0, 1];
    let b = [4, 1];
    let forged = PrimeFieldArithmeticCertificate {
        results: vec![vec![1]],
        gcd: Some(PrimeFieldGcdWitness {
            gcd: vec![1],
            left_quotient: a.to_vec(),
            right_quotient: b.to_vec(),
            bezout_left: vec![],
            bezout_right: vec![],
        }),
        inverse_quotient: None,
    };
    assert!(!accepts(Op::Gcd, 5, &a, Some(&b), &forged));
}

#[test]
fn bezout_without_common_divisibility_does_not_certify_a_gcd() {
    let a = [4, 0, 1];
    let b = [4, 1];
    let forged = PrimeFieldArithmeticCertificate {
        results: vec![a.to_vec()],
        gcd: Some(PrimeFieldGcdWitness {
            gcd: a.to_vec(),
            left_quotient: vec![1],
            right_quotient: vec![],
            bezout_left: vec![1],
            bezout_right: vec![],
        }),
        inverse_quotient: None,
    };
    assert!(!accepts(Op::Gcd, 5, &a, Some(&b), &forged));
}

#[test]
fn independent_handwritten_gcd_evidence_and_alternate_bezout_are_accepted() {
    let a = [4, 0, 1];
    let b = [4, 1];
    // a - x*b = b. This need not match the Euclidean generator's Bezout pair.
    let certificate = PrimeFieldArithmeticCertificate {
        results: vec![vec![1], vec![0, 4], b.to_vec()],
        gcd: Some(PrimeFieldGcdWitness {
            gcd: b.to_vec(),
            left_quotient: vec![1, 1],
            right_quotient: vec![1],
            bezout_left: vec![1],
            bezout_right: vec![0, 4],
        }),
        inverse_quotient: None,
    };
    assert!(accepts(Op::ExtendedGcd, 5, &a, Some(&b), &certificate));
}

#[test]
fn malformed_result_counts_residues_degrees_and_irrelevant_witnesses_are_rejected() {
    let a = [1, 2];
    let good = run(Op::Negate, 5, &a, None).certificate().clone();
    for results in [
        vec![],
        vec![vec![4, 3], vec![]],
        vec![vec![4, 3, 0]],
        vec![vec![4, 8]],
        vec![vec![1; 66]],
        vec![vec![]; 4],
    ] {
        let mut bad = good.clone();
        bad.results = results;
        assert!(!accepts(Op::Negate, 5, &a, None, &bad));
    }
    let mut bad = good.clone();
    bad.inverse_quotient = Some(vec![]);
    assert!(!accepts(Op::Negate, 5, &a, None, &bad));
    let mut bad = good;
    bad.gcd = run(Op::Gcd, 5, &a, Some(&[1])).certificate().gcd.clone();
    assert!(!accepts(Op::Negate, 5, &a, None, &bad));
}

#[test]
fn inverse_requires_reducedness_unit_status_and_the_exact_product_witness() {
    let a = [0, 1];
    let modulus = [3, 0, 1];
    let good = run(Op::Invert, 5, &a, Some(&modulus)).certificate().clone();
    let mut bad = good.clone();
    bad.results.clear();
    assert!(!accepts(Op::Invert, 5, &a, Some(&modulus), &bad));
    let mut bad = good.clone();
    bad.results[0][1] = 2;
    assert!(!accepts(Op::Invert, 5, &a, Some(&modulus), &bad));
    let mut bad = good.clone();
    bad.inverse_quotient = None;
    assert!(!accepts(Op::Invert, 5, &a, Some(&modulus), &bad));
    let mut bad = good.clone();
    bad.inverse_quotient = Some(vec![]);
    assert!(!accepts(Op::Invert, 5, &a, Some(&modulus), &bad));
    // inv + modulus is congruent to inv, but cannot be the reduced answer.
    let mut bad = good;
    bad.results[0] = vec![3, 3, 1];
    bad.inverse_quotient = Some(vec![3, 1]);
    assert!(!accepts(Op::Invert, 5, &a, Some(&modulus), &bad));
    let mut nonunit = run(Op::Invert, 5, &a, Some(&[0, 0, 1]))
        .certificate()
        .clone();
    nonunit.results = vec![vec![1]];
    nonunit.inverse_quotient = Some(vec![]);
    assert!(!accepts(Op::Invert, 5, &a, Some(&[0, 0, 1]), &nonunit));
}

#[test]
fn every_operation_rejects_changed_output_coefficients_or_a_changed_source() {
    let a = [3, 2, 1];
    let b = [1, 1];
    for op in [
        Op::Normalize,
        Op::Negate,
        Op::Add,
        Op::Subtract,
        Op::Multiply,
        Op::Divide,
        Op::Monic,
        Op::Gcd,
        Op::ExtendedGcd,
        Op::Lcm,
        Op::Invert,
        Op::Derivative(1),
        Op::Power(3),
        Op::Compose,
    ] {
        let right = matches!(
            op,
            Op::Add
                | Op::Subtract
                | Op::Multiply
                | Op::Divide
                | Op::Gcd
                | Op::ExtendedGcd
                | Op::Lcm
                | Op::Invert
                | Op::Compose
        )
        .then_some(b.as_slice());
        let mut bad = run(op, 5, &a, right).certificate().clone();
        assert!(!bad.results.is_empty());
        if bad.results[0].is_empty() {
            bad.results[0].push(1);
        } else {
            bad.results[0][0] = (bad.results[0][0] + 1) % 5;
            bad.results[0] = trimmed(bad.results[0].clone());
        }
        assert!(!accepts(op, 5, &a, right, &bad), "{op:?}");
    }
    let certificate = run(Op::Divide, 5, &a, Some(&b)).certificate().clone();
    assert!(!accepts(Op::Divide, 5, &[4, 2, 1], Some(&b), &certificate));
    let certificate = run(Op::Power(3), 5, &[1, 1], None).certificate().clone();
    assert!(!accepts(Op::Power(2), 5, &[1, 1], None, &certificate));
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
fn both_phases_cancel_before_during_and_at_final_publication() {
    let a = polynomial(5, &[0, 1]);
    for (op, values) in [
        (Op::Invert, vec![3, 0, 1]),
        (Op::Invert, vec![0, 0, 1]),
        (Op::ExtendedGcd, vec![4, 1]),
        (Op::Compose, vec![1, 1]),
    ] {
        let b = polynomial(5, &values);
        let (mut g, mut v) = (Count::default(), Count::default());
        checked_prime_field_arithmetic(op, &a, Some(&b), &mut g, &mut v).unwrap();
        for (generation, checkpoints) in [(true, g.checkpoints), (false, v.checkpoints)] {
            for at in [1, checkpoints / 2, checkpoints] {
                let mut stopped = Count {
                    cancel: Some(at),
                    ..Count::default()
                };
                let result = if generation {
                    checked_prime_field_arithmetic(op, &a, Some(&b), &mut stopped, &mut Unbounded)
                } else {
                    checked_prime_field_arithmetic(op, &a, Some(&b), &mut Unbounded, &mut stopped)
                };
                assert!(result.is_err(), "{op:?} phase={generation} checkpoint={at}");
            }
        }
    }
}

#[test]
fn cumulative_budgets_cover_euclid_verification_copies_and_publication() {
    let a = polynomial(7, &[3, 0, 2, 1]);
    for op in [Op::ExtendedGcd, Op::Invert, Op::Lcm] {
        let b = polynomial(7, &[2, 0, 1]);
        let (mut g, mut v) = (Count::default(), Count::default());
        checked_prime_field_arithmetic(op, &a, Some(&b), &mut g, &mut v).unwrap();
        for (dimension, g_used, v_used) in [
            (Dimension::ComputeSteps, g.steps, v.steps),
            (Dimension::MemoryBytes, g.bytes, v.bytes),
        ] {
            assert!(g_used > 0 && v_used > 0);
            checked_prime_field_arithmetic(
                op,
                &a,
                Some(&b),
                &mut cap(dimension, g_used),
                &mut cap(dimension, v_used),
            )
            .unwrap();
            assert!(
                checked_prime_field_arithmetic(
                    op,
                    &a,
                    Some(&b),
                    &mut cap(dimension, g_used - 1),
                    &mut Unbounded
                )
                .is_err()
            );
            assert!(
                checked_prime_field_arithmetic(
                    op,
                    &a,
                    Some(&b),
                    &mut Unbounded,
                    &mut cap(dimension, v_used - 1)
                )
                .is_err()
            );
        }
    }
}
