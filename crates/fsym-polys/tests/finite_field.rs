//! Complete prime-field factorizations, independent small-field enumeration,
//! and mutations that a mere modular product identity would accept.

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldFactor, PrimeFieldPolynomial, VerifiedPrimeFieldFactorization,
    checked_prime_field_factorization, verify_prime_field_factorization,
};
use fsym_proof_kernel::{Claim, verify_derivation_independent};

fn polynomial(prime: u64, coefficients: &[u64]) -> PrimeFieldPolynomial {
    PrimeFieldPolynomial::new(Symbol::new("x"), prime, coefficients.to_vec()).unwrap()
}

fn factor(prime: u64, coefficients: &[u64]) -> VerifiedPrimeFieldFactorization {
    checked_prime_field_factorization(
        &polynomial(prime, coefficients),
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
}

fn expected(result: &VerifiedPrimeFieldFactorization, unit: u64, factors: &[(&[u64], u32)]) {
    assert_eq!(result.unit(), unit);
    assert_eq!(
        result.factors(),
        factors
            .iter()
            .map(|&(coefficients, multiplicity)| PrimeFieldFactor {
                coefficients: coefficients.to_vec(),
                multiplicity,
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(result.irreducibility().len(), factors.len());
    for (derivation, factor) in result.irreducibility().iter().zip(result.factors()) {
        let claim =
            verify_derivation_independent(derivation, &ImmutableAssumptionsSnapshot::empty())
                .unwrap();
        assert_eq!(
            claim,
            Claim::IrreducibleModPrime {
                coeffs: factor
                    .coefficients
                    .iter()
                    .copied()
                    .map(BigInt::from)
                    .collect(),
                prime: BigInt::from(result.prime()),
            }
        );
    }
}

#[test]
fn characteristic_two_and_odd_prime_factors_match_independent_known_results() {
    expected(&factor(2, &[1, 0, 1]), 1, &[(&[1, 1], 2)]);
    expected(
        &factor(3, &[1, 0, 0, 0, 1]),
        1,
        &[(&[2, 1, 1], 1), (&[2, 2, 1], 1)],
    );
    expected(
        &factor(3, &[0, 2, 0, 2]),
        2,
        &[(&[0, 1], 1), (&[1, 0, 1], 1)],
    );
    expected(
        &factor(5, &[1, 0, 2, 0, 1]),
        1,
        &[(&[2, 1], 2), (&[3, 1], 2)],
    );
    expected(&factor(65_521, &[65_520, 1]), 1, &[(&[65_520, 1], 1)]);
}

#[test]
fn inseparable_powers_and_mixed_multiplicities_are_complete() {
    // x^6+x^3+1=(x-1)^6 in F_3, although its derivative vanishes.
    expected(&factor(3, &[1, 0, 0, 1, 0, 0, 1]), 1, &[(&[2, 1], 6)]);
    // (x^2+x+1)^2 * (x^3+x+1)^3 in F_2; literal expanded oracle coefficients.
    expected(
        &factor(2, &[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]),
        1,
        &[(&[1, 1, 1], 2), (&[1, 1, 0, 1], 3)],
    );
    expected(
        &factor(3, &[1, 2, 1, 0, 0, 0, 1, 2, 1]),
        1,
        &[(&[1, 1], 2), (&[1, 0, 1], 3)],
    );
    let mut repeated = vec![0; 36];
    repeated[0] = 2;
    repeated[10] = 1;
    repeated[25] = 2;
    repeated[35] = 1;
    expected(&factor(5, &repeated), 1, &[(&[1, 1], 25), (&[2, 0, 1], 5)]);
    let mut boundary = vec![0; 65];
    boundary[0] = 1;
    boundary[64] = 1;
    expected(&factor(2, &boundary), 1, &[(&[1, 1], 64)]);
}

#[test]
fn zero_constants_and_canonical_input_normalization_remain_distinct() {
    expected(&factor(2, &[]), 0, &[]);
    expected(&factor(7, &[6]), 6, &[]);
    let input = polynomial(3, &[1, 0, 1, 0, 0]);
    assert_eq!(input.coefficients(), [1, 0, 1]);
    let result = checked_prime_field_factorization(&input, &mut Unbounded, &mut Unbounded).unwrap();
    expected(&result, 1, &[(&[1, 0, 1], 1)]);
    assert_eq!(result.source(), input.coefficients());
    assert_eq!(result.variable(), input.variable());
}

// A separate exhaustive irreducibility oracle: trial-divide by every monic
// polynomial of degree <=n/2. It uses neither Frobenius nor Berlekamp/Rabin.
fn has_proper_divisor(coefficients: &[u64], prime: u64) -> bool {
    for degree in 1..=(coefficients.len() - 1) / 2 {
        for mut code in 0..prime.pow(degree as u32) {
            let mut divisor = vec![0; degree + 1];
            for coefficient in &mut divisor[..degree] {
                *coefficient = code % prime;
                code /= prime;
            }
            divisor[degree] = 1;
            let mut remainder = coefficients.to_vec();
            for power in (degree..remainder.len()).rev() {
                let leading = remainder[power];
                for index in 0..=degree {
                    remainder[power - degree + index] = (remainder[power - degree + index] + prime
                        - leading * divisor[index] % prime)
                        % prime;
                }
            }
            if remainder.iter().all(|&coefficient| coefficient == 0) {
                return true;
            }
        }
    }
    false
}

#[test]
fn exhaustive_small_fields_agree_with_trial_division_irreducibility() {
    for (prime, maximum_degree) in [(2u64, 6u32), (3, 4), (5, 3)] {
        for degree in 1..=maximum_degree {
            for mut code in 0..prime.pow(degree) {
                let mut coefficients = vec![0; degree as usize + 1];
                for coefficient in &mut coefficients[..degree as usize] {
                    *coefficient = code % prime;
                    code /= prime;
                }
                coefficients[degree as usize] = 1;
                let result = factor(prime, &coefficients);
                assert_eq!(
                    result.factors().len() == 1 && result.factors()[0].multiplicity == 1,
                    !has_proper_divisor(&coefficients, prime),
                    "F_{prime}: {coefficients:?}"
                );
                for term in result.factors() {
                    assert!(!has_proper_divisor(&term.coefficients, prime));
                }
            }
        }
    }
}

#[test]
fn an_exact_product_of_reducible_factors_is_not_a_complete_factorization() {
    let input = polynomial(3, &[1, 0, 0, 0, 1]);
    let result = checked_prime_field_factorization(&input, &mut Unbounded, &mut Unbounded).unwrap();
    let mut candidate = result.certificate().clone();
    candidate.factors = vec![PrimeFieldFactor {
        coefficients: input.coefficients().to_vec(),
        multiplicity: 1,
    }];
    let error = verify_prime_field_factorization(&input, &candidate, &mut Unbounded).unwrap_err();
    assert!(error.to_string().contains("irreducib"), "{error}");
}

#[test]
fn source_field_typed_generator_unit_factor_and_multiplicity_mutations_refuse() {
    let input = polynomial(5, &[1, 0, 2, 0, 1]);
    let result = checked_prime_field_factorization(&input, &mut Unbounded, &mut Unbounded).unwrap();
    let reject = |candidate| {
        assert!(verify_prime_field_factorization(&input, &candidate, &mut Unbounded).is_err())
    };
    let mut changed = result.certificate().clone();
    changed.prime = 7;
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.variable =
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([13; 32]));
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.source[0] = 2;
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.unit = 2;
    reject(changed);
    for value in [0, 1, 65, u32::MAX] {
        let mut changed = result.certificate().clone();
        changed.factors[0].multiplicity = value;
        reject(changed);
    }
    for value in [0, 1, 5, u64::MAX] {
        let mut changed = result.certificate().clone();
        changed.factors[0].coefficients[0] = value;
        reject(changed);
    }
    let mut changed = result.certificate().clone();
    changed.factors[0].coefficients.push(0);
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.factors.swap(0, 1);
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.factors[0].coefficients = changed.factors[1].coefficients.clone();
    reject(changed);
    let mut changed = result.certificate().clone();
    changed.factors.pop();
    reject(changed);
}

#[test]
fn composite_moduli_out_of_range_coefficients_and_oversize_inputs_refuse() {
    for prime in [0, 1, 4, 6, 9, 341, 561, 65_523, u64::MAX] {
        assert!(PrimeFieldPolynomial::new(Symbol::new("x"), prime, vec![1, 1]).is_err());
    }
    assert!(PrimeFieldPolynomial::new(Symbol::new("x"), 3, vec![3, 1]).is_err());
    assert!(PrimeFieldPolynomial::new(Symbol::new("x"), 3, vec![0; 66]).is_err());
    assert!(PrimeFieldPolynomial::new(Symbol::new("x".repeat(65_537)), 3, vec![1]).is_err());
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
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
fn generation_and_independent_verification_cancel_or_exhaust_without_publication() {
    let input = polynomial(3, &[1, 2, 1, 0, 0, 0, 1, 2, 1]);
    let (mut generator, mut verifier) = (Meter::default(), Meter::default());
    checked_prime_field_factorization(&input, &mut generator, &mut verifier).unwrap();
    for (phase, count) in [(0, generator.checkpoints), (1, verifier.checkpoints)] {
        for cancel_at in [1, count / 2, count] {
            let mut meter = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let result = if phase == 0 {
                checked_prime_field_factorization(&input, &mut meter, &mut Unbounded)
            } else {
                checked_prime_field_factorization(&input, &mut Unbounded, &mut meter)
            };
            assert!(result.is_err(), "phase {phase}, checkpoint {cancel_at}");
        }
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        let result = if phase == 0 {
            checked_prime_field_factorization(&input, &mut budget, &mut Unbounded)
        } else {
            checked_prime_field_factorization(&input, &mut Unbounded, &mut budget)
        };
        assert!(result.is_err());
    }
}
