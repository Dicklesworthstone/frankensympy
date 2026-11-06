//! Resultants and universal discriminants in positive characteristic.
//!
//! Tiny exhaustive cases use integer permutation determinants, independent of
//! both the generator's polynomial Euclid and the verifier's modular Gaussian
//! elimination. Discriminants are formed over ZZ before reduction, exposing
//! characteristic-induced derivative degree drops that actual-degree formulas
//! can incorrectly cancel.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldArithmeticCertificate, PrimeFieldArithmeticOperation as Op, PrimeFieldPolynomial,
    checked_prime_field_arithmetic, verify_prime_field_arithmetic,
};

fn polynomial(prime: u64, values: &[u64]) -> PrimeFieldPolynomial {
    PrimeFieldPolynomial::new(Symbol::new("x"), prime, values.to_vec()).unwrap()
}

fn calculate(op: Op, prime: u64, left: &[u64], right: Option<&[u64]>) -> u64 {
    let left = polynomial(prime, left);
    let right = right.map(|values| polynomial(prime, values));
    let result =
        checked_prime_field_arithmetic(op, &left, right.as_ref(), &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(result.results().len(), 1);
    match result.results()[0].as_slice() {
        [] => 0,
        [scalar] => *scalar,
        _ => panic!("a resultant or discriminant is a field scalar"),
    }
}

fn certificate(value: u64) -> PrimeFieldArithmeticCertificate {
    PrimeFieldArithmeticCertificate {
        results: vec![if value == 0 { vec![] } else { vec![value] }],
        gcd: None,
        inverse_quotient: None,
    }
}

fn accepts(op: Op, prime: u64, left: &[u64], right: Option<&[u64]>, value: u64) -> bool {
    let left = polynomial(prime, left);
    let right = right.map(|values| polynomial(prime, values));
    verify_prime_field_arithmetic(
        op,
        &left,
        right.as_ref(),
        &certificate(value),
        &mut Unbounded,
    )
    .is_ok()
}

#[test]
fn resultant_signs_constant_powers_and_zero_conventions_are_exact() {
    assert_eq!(calculate(Op::Resultant, 5, &[1, 0, 1], Some(&[1, 1])), 2);
    assert_eq!(calculate(Op::Resultant, 7, &[3, 2], Some(&[1, 4])), 4);
    assert_eq!(calculate(Op::Resultant, 7, &[1, 4], Some(&[3, 2])), 3);
    assert_eq!(calculate(Op::Resultant, 5, &[3], Some(&[1, 0, 1])), 4);
    assert_eq!(calculate(Op::Resultant, 5, &[1, 0, 1], Some(&[3])), 4);
    assert_eq!(calculate(Op::Resultant, 7, &[4, 2], Some(&[2, 1])), 0);
    for (left, right) in [(&[][..], &[][..]), (&[], &[3]), (&[3], &[])] {
        assert_eq!(calculate(Op::Resultant, 5, left, Some(right)), 0);
    }
    for a in 1..5 {
        for b in 1..5 {
            assert_eq!(calculate(Op::Resultant, 5, &[a], Some(&[b])), 1);
        }
    }
}

#[test]
fn discriminants_have_the_universal_zero_linear_and_repeated_root_conventions() {
    for source in [&[][..], &[3]] {
        assert_eq!(calculate(Op::Discriminant, 5, source, None), 0);
    }
    for source in [&[0, 2][..], &[3, 4]] {
        assert_eq!(calculate(Op::Discriminant, 5, source, None), 1);
    }
    assert_eq!(calculate(Op::Discriminant, 5, &[4, 3, 2], None), 2); // 3^2-4*2*4
    assert_eq!(calculate(Op::Discriminant, 7, &[1, 4, 2, 3], None), 6); // -547 mod 7
    assert_eq!(calculate(Op::Discriminant, 7, &[3, 5, 0, 2, 6], None), 4); // 244724 mod 7
    assert_eq!(calculate(Op::Discriminant, 5, &[0, 0, 1], None), 0);
    assert_eq!(calculate(Op::Discriminant, 3, &[1, 0, 0, 2], None), 0);
    assert_eq!(calculate(Op::Discriminant, 2, &[1, 0, 1], None), 0);
    assert_eq!(calculate(Op::Discriminant, 2, &[1, 1, 0, 0, 1], None), 1);
}

#[test]
fn characteristic_degree_drops_preserve_the_universal_discriminant() {
    // Pinned SymPy 1.14.0 Poly(..., modulus=5).discriminant() returns 4
    // here, while its exact integer discriminant is 50216, hence 1 mod 5.
    // The actual derivative is 3*x^2; the formal derivative has degree 4.
    // This is an explicit known oracle disagreement, not a relaxed comparison.
    let quintic = [1, 0, 0, 1, 0, 2];
    assert_eq!(calculate(Op::Resultant, 5, &quintic, Some(&[0, 0, 3])), 3);
    assert_eq!(calculate(Op::Discriminant, 5, &quintic, None), 1);
    assert!(!accepts(Op::Discriminant, 5, &quintic, None, 4));

    // SymPy's field result is 1; its integer result is -112, hence 2 mod 3.
    // The derivative 2*x drops one degree. Padding restores the missing LC.
    let cubic = [1, 0, 1, 2];
    assert_eq!(calculate(Op::Resultant, 3, &cubic, Some(&[0, 2])), 1);
    assert_eq!(calculate(Op::Discriminant, 3, &cubic, None), 2);
    assert!(!accepts(Op::Discriminant, 3, &cubic, None, 1));

    // Derivative degree loss need not cause an oracle disagreement: its
    // missing leading-coefficient power may be one in the exact field.
    assert_eq!(calculate(Op::Discriminant, 5, &[1, 3, 0, 0, 0, 2], None), 4);
}

fn permutations(
    matrix: &[i128],
    size: usize,
    row: usize,
    occupied: u64,
    product: i128,
    negative: bool,
) -> i128 {
    if row == size {
        return if negative { -product } else { product };
    }
    let mut result = 0;
    for column in 0..size {
        if occupied & (1 << column) == 0 && matrix[row * size + column] != 0 {
            let inversions = (occupied >> (column + 1)).count_ones();
            result += permutations(
                matrix,
                size,
                row + 1,
                occupied | (1 << column),
                product * matrix[row * size + column],
                negative ^ !inversions.is_multiple_of(2),
            );
        }
    }
    result
}

fn integer_resultant(left: &[i128], m: usize, right: &[i128], n: usize) -> i128 {
    let size = m + n;
    assert!(size <= 6, "permutation oracle is deliberately tiny");
    let mut matrix = vec![0; size * size];
    for row in 0..size {
        let (coefficients, shift, degree) = if row < n {
            (left, row, m)
        } else {
            (right, row - n, n)
        };
        for (power, &value) in coefficients.iter().enumerate() {
            matrix[row * size + shift + degree - power] = value;
        }
    }
    permutations(&matrix, size, 0, 0, 1, false)
}

fn integer_discriminant(source: &[u64]) -> i128 {
    if source.len() <= 1 {
        return 0;
    }
    let n = source.len() - 1;
    let coefficients: Vec<_> = source.iter().map(|&value| i128::from(value)).collect();
    let derivative: Vec<_> = coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(power, &value)| power as i128 * value)
        .collect();
    let resultant = integer_resultant(&coefficients, n, &derivative, n - 1);
    assert_eq!(resultant % coefficients[n], 0);
    let sign = if (n * (n - 1) / 2).is_multiple_of(2) {
        1
    } else {
        -1
    };
    sign * resultant / coefficients[n]
}

fn enumerated(mut code: u64, length: usize, prime: u64) -> Vec<u64> {
    let mut values = vec![0; length];
    for value in &mut values {
        *value = code % prime;
        code /= prime;
    }
    while values.last() == Some(&0) {
        values.pop();
    }
    values
}

#[test]
fn exhaustive_small_fields_agree_with_integer_permutation_determinants() {
    let (mut pairs, mut discriminants) = (0, 0);
    for (prime, length) in [(2u64, 4usize), (3, 3), (5, 2)] {
        for a in 0..prime.pow(length as u32) {
            let left = enumerated(a, length, prime);
            let expected = integer_discriminant(&left).rem_euclid(i128::from(prime)) as u64;
            assert_eq!(calculate(Op::Discriminant, prime, &left, None), expected);
            discriminants += 1;
            for b in 0..prime.pow(length as u32) {
                let right = enumerated(b, length, prime);
                let expected = if left.is_empty() || right.is_empty() {
                    0
                } else {
                    let integers_left: Vec<_> =
                        left.iter().map(|&value| i128::from(value)).collect();
                    let integers_right: Vec<_> =
                        right.iter().map(|&value| i128::from(value)).collect();
                    integer_resultant(
                        &integers_left,
                        left.len() - 1,
                        &integers_right,
                        right.len() - 1,
                    )
                    .rem_euclid(i128::from(prime)) as u64
                };
                assert_eq!(
                    calculate(Op::Resultant, prime, &left, Some(&right)),
                    expected
                );
                pairs += 1;
            }
        }
    }
    assert_eq!((pairs, discriminants), (1610, 68));
}

#[test]
fn maximum_degree_and_characteristic_exercise_the_full_sylvester_bound() {
    let mut left = vec![0; 65];
    left[0] = 1;
    left[64] = 1;
    let mut right = left.clone();
    right[0] = 2;
    // Res(x^64+1,x^64+2)=1; both inputs occupy the degree-64 bound.
    assert_eq!(calculate(Op::Resultant, 65_521, &left, Some(&right)), 1);
    // Disc(x^64+1)=64^64=1787 mod 65521, a 127-row determinant.
    assert_eq!(calculate(Op::Discriminant, 65_521, &left, None), 1787);
    assert!(!accepts(Op::Resultant, 65_521, &left, Some(&right), 2));
}

#[test]
fn scalar_receipts_reject_wrong_values_nonconstant_results_and_changed_sources() {
    for (op, right, expected) in [
        (Op::Resultant, Some(&[1, 1][..]), 2),
        (Op::Discriminant, None, 1),
    ] {
        let left = polynomial(5, &[1, 0, 1]);
        let right_value = right.map(|values| polynomial(5, values));
        let original = certificate(expected);
        let accepted = verify_prime_field_arithmetic(
            op,
            &left,
            right_value.as_ref(),
            &original,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(accepted.operation(), op);
        assert_eq!(accepted.left(), &left);
        assert_eq!(accepted.right(), right_value.as_ref());
        assert_eq!(accepted.results(), original.results);
        assert!(!accepts(op, 5, &[1, 0, 1], right, (expected + 1) % 5));
        assert!(!accepts(op, 5, &[2, 0, 1], right, expected));
        for results in [
            vec![],
            vec![vec![expected], vec![]],
            vec![vec![expected, 1]],
            vec![vec![5]],
        ] {
            let mut forged = original.clone();
            forged.results = results;
            assert!(
                verify_prime_field_arithmetic(
                    op,
                    &left,
                    right_value.as_ref(),
                    &forged,
                    &mut Unbounded
                )
                .is_err()
            );
        }
        let mut forged = original;
        forged.inverse_quotient = Some(vec![]);
        assert!(
            verify_prime_field_arithmetic(op, &left, right_value.as_ref(), &forged, &mut Unbounded)
                .is_err()
        );
    }
    assert!(!accepts(Op::Resultant, 5, &[1, 1], Some(&[1, 1]), 1));
    assert!(!accepts(Op::Discriminant, 5, &[1, 2, 1], None, 1));
}

#[test]
fn typed_ring_and_operation_arity_are_admitted_before_zero_results() {
    let zero = polynomial(5, &[]);
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([91; 32]));
    for other in [
        polynomial(7, &[]),
        PrimeFieldPolynomial::new(typed, 5, vec![]).unwrap(),
    ] {
        assert!(
            checked_prime_field_arithmetic(
                Op::Resultant,
                &zero,
                Some(&other),
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(
        checked_prime_field_arithmetic(Op::Resultant, &zero, None, &mut Unbounded, &mut Unbounded)
            .is_err()
    );
    assert!(
        checked_prime_field_arithmetic(
            Op::Discriminant,
            &zero,
            Some(&zero),
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
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
fn complete_euclid_and_determinant_phases_observe_exact_resource_and_cancel_boundaries() {
    let left = polynomial(7, &[3, 5, 0, 2, 6]);
    let right = polynomial(7, &[1, 2, 3]);
    for (op, rhs) in [(Op::Resultant, Some(&right)), (Op::Discriminant, None)] {
        let (mut generation, mut verification) = (Count::default(), Count::default());
        checked_prime_field_arithmetic(op, &left, rhs, &mut generation, &mut verification).unwrap();
        for (dimension, g_used, v_used) in [
            (
                Dimension::ComputeSteps,
                generation.steps,
                verification.steps,
            ),
            (Dimension::MemoryBytes, generation.bytes, verification.bytes),
        ] {
            assert!(g_used > 0 && v_used > 0);
            checked_prime_field_arithmetic(
                op,
                &left,
                rhs,
                &mut cap(dimension, g_used),
                &mut cap(dimension, v_used),
            )
            .unwrap();
            assert!(
                checked_prime_field_arithmetic(
                    op,
                    &left,
                    rhs,
                    &mut cap(dimension, g_used - 1),
                    &mut Unbounded
                )
                .is_err()
            );
            assert!(
                checked_prime_field_arithmetic(
                    op,
                    &left,
                    rhs,
                    &mut Unbounded,
                    &mut cap(dimension, v_used - 1)
                )
                .is_err()
            );
        }
        for (generating, count) in [
            (true, generation.checkpoints),
            (false, verification.checkpoints),
        ] {
            for stop in [1, count / 2, count] {
                let mut cancelled = Count {
                    cancel: Some(stop),
                    ..Count::default()
                };
                let result = if generating {
                    checked_prime_field_arithmetic(op, &left, rhs, &mut cancelled, &mut Unbounded)
                } else {
                    checked_prime_field_arithmetic(op, &left, rhs, &mut Unbounded, &mut cancelled)
                };
                assert!(
                    result.is_err(),
                    "{op:?} generator={generating} checkpoint={stop}"
                );
            }
        }
    }
}
