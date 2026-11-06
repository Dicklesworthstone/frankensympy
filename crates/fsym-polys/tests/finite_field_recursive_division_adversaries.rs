//! Independent witnesses for public Poly.div's recursive coefficient-ring rule.
//!
//! A valid polynomial identity and an obstructed final leading coefficient do
//! not prove that the quotient follows the required sequence of exact leading
//! coefficient cancellations. The fixtures below distinguish those claims.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{Symbol, SymbolIdentity};
use fsym_polys::{
    PrimeFieldMultivariateDivisionCertificate as Terminal,
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    PrimeFieldRecursiveDivisionCertificate as Certificate,
    checked_prime_field_recursive_division as checked,
    verify_prime_field_recursive_division as verify,
};

fn ring() -> Vec<Symbol> {
    vec![Symbol::new("x"), Symbol::new("y")]
}

fn terms(values: &[(&[u32], u64)]) -> Terms {
    values
        .iter()
        .map(|(powers, coefficient)| (powers.to_vec(), *coefficient))
        .collect()
}

fn polynomial(prime: u64, values: &[(&[u32], u64)]) -> Polynomial {
    Polynomial::new(&ring(), prime, &terms(values)).unwrap()
}

fn zero() -> Terms {
    Terms::new()
}

fn terminal(quotient: Terms, remainder: Terms) -> Terminal {
    Terminal {
        quotients: vec![quotient],
        remainder,
    }
}

fn accepted(dividend: &Polynomial, divisor: &Polynomial, witness: &Certificate) -> bool {
    verify(dividend, divisor, witness, &mut Unbounded).is_ok()
}

fn characteristic_seven() -> (Polynomial, Polynomial, Certificate) {
    let dividend = polynomial(7, &[(&[2, 1], 3), (&[1, 2], 2), (&[0, 1], 4), (&[0, 0], 5)]);
    let divisor = polynomial(7, &[(&[1, 1], 2), (&[0, 2], 3), (&[0, 0], 1)]);
    // First cancel 3*y*x^2 by (-2*x)*divisor. The next x coefficient
    // is y^2+2, which is not exactly divisible by 2*y in F_7[y].
    let witness = Certificate {
        quotient: terms(&[(&[1, 0], 5)]),
        remainder: terms(&[(&[1, 2], 1), (&[1, 0], 2), (&[0, 1], 4), (&[0, 0], 5)]),
        terminal: Some(terminal(terms(&[(&[0, 1], 4)]), terms(&[(&[0, 0], 2)]))),
    };
    (dividend, divisor, witness)
}

#[test]
fn hand_derived_recursive_stopping_witness_retains_the_complete_request() {
    let (dividend, divisor, witness) = characteristic_seven();
    let receipt = verify(&dividend, &divisor, &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.dividend(), &dividend);
    assert_eq!(receipt.divisor(), &divisor);
    assert_eq!(receipt.quotient().terms(), &witness.quotient);
    assert_eq!(receipt.remainder().terms(), &witness.remainder);
    assert_eq!(receipt.quotient().generators(), ring());
    assert_eq!(receipt.remainder().prime(), 7);
}

#[test]
fn a_correct_identity_and_terminal_obstruction_cannot_authorize_arbitrary_quotients() {
    for prime in [2, 5, 7] {
        let one = terms(&[(&[0, 0], 1)]);
        let dividend = polynomial(prime, &[(&[0, 0], 1)]);
        let divisor = polynomial(prime, &[(&[0, 1], 1)]);
        let honest = Certificate {
            quotient: zero(),
            remainder: one.clone(),
            terminal: Some(terminal(zero(), one.clone())),
        };
        assert!(accepted(&dividend, &divisor, &honest));
        // 1 = 1*y + (1-y), and (1-y) has nonzero remainder 1 modulo y.
        // Nevertheless the first coefficient 1 cannot be divided by y;
        // recursive division must stop before taking this invented step.
        let forged = Certificate {
            quotient: one.clone(),
            remainder: terms(&[(&[0, 0], 1), (&[0, 1], prime - 1)]),
            terminal: Some(terminal(terms(&[(&[0, 0], prime - 1)]), one)),
        };
        assert!(!accepted(&dividend, &divisor, &forged), "prime={prime}");
    }
}

#[test]
fn a_sparse_normal_form_is_not_the_recursive_poly_div_quotient() {
    let (dividend, divisor, honest) = characteristic_seven();
    assert!(accepted(&dividend, &divisor, &honest));
    // Sparse monomial reduction continues with -3*y and still leaves an
    // obstructed leading x coefficient 2. Both identities are exact, but
    // the attempted extra coefficient cancellation was not exact.
    let sparse_answer = Certificate {
        quotient: terms(&[(&[1, 0], 5), (&[0, 1], 4)]),
        remainder: terms(&[(&[1, 0], 2), (&[0, 3], 2), (&[0, 0], 5)]),
        terminal: Some(terminal(zero(), terms(&[(&[0, 0], 2)]))),
    };
    assert!(!accepted(&dividend, &divisor, &sparse_answer));
}

#[test]
fn exact_cancellation_stages_cannot_be_omitted() {
    let dividend = polynomial(5, &[(&[2, 0], 1), (&[1, 1], 1), (&[0, 0], 1)]);
    let divisor = polynomial(5, &[(&[1, 0], 1), (&[0, 0], 1)]);
    // x^2+x*y+1 = (x+y-1)*(x+1) + (2-y).
    let honest = Certificate {
        quotient: terms(&[(&[1, 0], 1), (&[0, 1], 1), (&[0, 0], 4)]),
        remainder: terms(&[(&[0, 0], 2), (&[0, 1], 4)]),
        terminal: None,
    };
    assert!(accepted(&dividend, &divisor, &honest));
    let premature = Certificate {
        quotient: terms(&[(&[1, 0], 1)]),
        remainder: terms(&[(&[1, 1], 1), (&[1, 0], 4), (&[0, 0], 1)]),
        // The remaining coefficient y-1 is exactly divisible by 1.
        terminal: Some(terminal(terms(&[(&[0, 1], 1), (&[0, 0], 4)]), zero())),
    };
    assert!(!accepted(&dividend, &divisor, &premature));
    let omitted_first = Certificate {
        quotient: terms(&[(&[0, 1], 1), (&[0, 0], 4)]),
        remainder: terms(&[(&[2, 0], 1), (&[1, 0], 1), (&[0, 1], 4), (&[0, 0], 2)]),
        terminal: Some(terminal(terms(&[(&[0, 0], 1)]), zero())),
    };
    assert!(!accepted(&dividend, &divisor, &omitted_first));
}

#[test]
fn coefficient_obstruction_requires_its_complete_and_irreducible_witness() {
    let (dividend, divisor, honest) = characteristic_seven();
    assert!(accepted(&dividend, &divisor, &honest));
    let mut missing = honest.clone();
    missing.terminal = None;
    assert!(!accepted(&dividend, &divisor, &missing));
    for malformed in [
        Terminal {
            quotients: vec![],
            remainder: terms(&[(&[0, 0], 2)]),
        },
        Terminal {
            quotients: vec![zero(), zero()],
            remainder: terms(&[(&[0, 0], 2)]),
        },
        terminal(terms(&[(&[0, 1], 4)]), zero()),
        terminal(terms(&[(&[0, 1], 4)]), terms(&[(&[0, 0], 3)])),
        // Correct coefficient identity, but its remainder is still reducible.
        terminal(zero(), terms(&[(&[0, 2], 1), (&[0, 0], 2)])),
    ] {
        let mut mutant = honest.clone();
        mutant.terminal = Some(malformed);
        assert!(!accepted(&dividend, &divisor, &mutant));
    }
}

#[test]
fn zero_constant_and_lower_degree_dividends_have_no_spurious_obstruction() {
    let x = polynomial(5, &[(&[1, 0], 1)]);
    let square = polynomial(5, &[(&[2, 0], 1)]);
    let lower = Certificate {
        quotient: zero(),
        remainder: x.terms().clone(),
        terminal: None,
    };
    assert!(accepted(&x, &square, &lower));
    let no_input = polynomial(5, &[]);
    let empty = Certificate {
        quotient: zero(),
        remainder: zero(),
        terminal: None,
    };
    assert!(accepted(&no_input, &square, &empty));
    let input = polynomial(5, &[(&[1, 0], 1), (&[0, 1], 1)]);
    let constant = polynomial(5, &[(&[0, 0], 2)]);
    let exact = Certificate {
        quotient: terms(&[(&[1, 0], 3), (&[0, 1], 3)]),
        remainder: zero(),
        terminal: None,
    };
    assert!(accepted(&input, &constant, &exact));
    let mut extraneous = exact;
    extraneous.terminal = Some(terminal(zero(), zero()));
    assert!(!accepted(&input, &constant, &extraneous));
    assert!(!accepted(&no_input, &no_input, &empty));
}

#[test]
fn raw_witness_bounds_are_not_erased_by_zero_coefficients() {
    let (dividend, divisor, honest) = characteristic_seven();
    for (powers, coefficient) in [
        (vec![65, 0], 0),
        (vec![0], 0),
        (vec![0, 0], 7),
        (vec![u32::MAX, u32::MAX], 0),
    ] {
        let mut mutant = honest.clone();
        mutant.quotient.insert(powers, coefficient);
        assert!(!accepted(&dividend, &divisor, &mutant));
    }
    let mut coefficient_witness = honest;
    coefficient_witness
        .terminal
        .as_mut()
        .unwrap()
        .remainder
        .insert(vec![0, 65], 0);
    assert!(!accepted(&dividend, &divisor, &coefficient_witness));
}

#[test]
fn zero_inputs_still_bind_ordered_typed_variables_and_the_prime() {
    let dividend = polynomial(5, &[]);
    let witness = Certificate {
        quotient: zero(),
        remainder: zero(),
        terminal: None,
    };
    let coefficient = terms(&[(&[0, 0], 1)]);
    let shadow = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([11; 32]));
    let wrong_rings = [
        vec![Symbol::new("y"), Symbol::new("x")],
        vec![shadow, Symbol::new("y")],
    ];
    for variables in wrong_rings {
        let divisor = Polynomial::new(&variables, 5, &coefficient).unwrap();
        assert!(!accepted(&dividend, &divisor, &witness));
    }
    assert!(!accepted(
        &dividend,
        &polynomial(7, &[(&[0, 0], 1)]),
        &witness
    ));
    assert!(accepted(
        &dividend,
        &polynomial(5, &[(&[0, 0], 1)]),
        &witness
    ));
}

#[derive(Default)]
struct Meter {
    steps: u64,
    bytes: u64,
    allocations: u64,
    checkpoints: usize,
    cancel: Option<usize>,
}

impl BudgetMeter for Meter {
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
fn recursive_stages_obstruction_and_receipt_share_each_callers_budget() {
    let (dividend, divisor, _) = characteristic_seven();
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    checked(&dividend, &divisor, &mut generation, &mut verification).unwrap();
    for (dimension, generated, verified) in [
        (
            Dimension::ComputeSteps,
            generation.steps,
            verification.steps,
        ),
        (Dimension::MemoryBytes, generation.bytes, verification.bytes),
        (
            Dimension::AllocationCount,
            generation.allocations,
            verification.allocations,
        ),
    ] {
        assert!(generated > 1 && verified > 1);
        checked(
            &dividend,
            &divisor,
            &mut cap(dimension, generated),
            &mut cap(dimension, verified),
        )
        .unwrap();
        assert!(
            checked(
                &dividend,
                &divisor,
                &mut cap(dimension, generated - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked(
                &dividend,
                &divisor,
                &mut Unbounded,
                &mut cap(dimension, verified - 1)
            )
            .is_err()
        );
        let mut partly_spent = cap(dimension, generated);
        BudgetMeter::charge(&mut partly_spent, dimension, 1).unwrap();
        assert!(checked(&dividend, &divisor, &mut partly_spent, &mut Unbounded).is_err());
    }
    for (is_generator, checkpoints) in [
        (true, generation.checkpoints),
        (false, verification.checkpoints),
    ] {
        for stop in [1, checkpoints / 2, checkpoints] {
            let mut meter = Meter {
                cancel: Some(stop),
                ..Meter::default()
            };
            let result = if is_generator {
                checked(&dividend, &divisor, &mut meter, &mut Unbounded)
            } else {
                checked(&dividend, &divisor, &mut Unbounded, &mut meter)
            };
            assert!(
                result.is_err(),
                "generator={is_generator}, checkpoint={stop}"
            );
        }
    }
}
