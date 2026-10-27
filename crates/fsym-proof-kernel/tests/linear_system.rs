//! Hand-authored linear certificates: no matrix generator is in this crate's
//! dependency tree. Mutants target completeness, invertibility and input binding.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational};
use fsym_proof_kernel::linear_system::{
    LINEAR_SYSTEM_SCHEMA_VERSION, LinearRowOperation as Row, LinearSystemVerificationError,
    LinearSystemWitness, MAX_LINEAR_SYSTEM_COEFFICIENT_BITS, MAX_LINEAR_SYSTEM_DIMENSION,
    RationalLinearSolution as Solution, RationalLinearSystem, verify_linear_system_witness,
};
use num_traits::Zero;

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}
fn rationals(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|&value| q(value)).collect()
}

fn affine_fixture() -> (RationalLinearSystem, LinearSystemWitness) {
    let system = RationalLinearSystem {
        rows: 2,
        columns: 3,
        coefficients: rationals(&[0, 2, 4, 1, 1, 1]),
        rhs: rationals(&[6, 3]),
    };
    let witness = LinearSystemWitness {
        schema_version: LINEAR_SYSTEM_SCHEMA_VERSION,
        operations: vec![
            Row::Swap { left: 0, right: 1 },
            Row::Scale {
                row: 1,
                factor: BigRational::new(1.into(), 2.into()),
            },
            Row::Add {
                target: 0,
                source: 1,
                factor: q(-1),
            },
        ],
        reduced: rationals(&[1, 0, -1, 0, 0, 1, 2, 3]),
        pivots: vec![0, 1],
        solution: Solution::Affine {
            particular: rationals(&[0, 3, 0]),
            basis: vec![rationals(&[1, -2, 1])],
            free_columns: vec![2],
        },
    };
    (system, witness)
}

#[test]
fn independent_certificate_checks_complete_affine_family_and_round_trip() {
    let (system, witness) = affine_fixture();
    let receipt = verify_linear_system_witness(&system, &witness, &mut Unbounded).unwrap();
    assert_eq!(receipt.rank(), 2);
    assert_eq!(receipt.nullity(), 1);
    assert!(receipt.is_consistent());
    assert_eq!(receipt.row_operations(), 3);
    let decoded: LinearSystemWitness =
        serde_json::from_str(&serde_json::to_string(&witness).unwrap()).unwrap();
    assert_eq!(decoded, witness);
    verify_linear_system_witness(&system, &decoded, &mut Unbounded).unwrap();
}

#[test]
fn independent_overdetermined_unique_and_inconsistent_claims_are_distinct() {
    let mut system = RationalLinearSystem {
        rows: 3,
        columns: 2,
        coefficients: rationals(&[1, 0, 0, 1, 1, 1]),
        rhs: rationals(&[2, 3, 5]),
    };
    let mut witness = LinearSystemWitness {
        schema_version: LINEAR_SYSTEM_SCHEMA_VERSION,
        operations: vec![
            Row::Add {
                target: 2,
                source: 0,
                factor: q(-1),
            },
            Row::Add {
                target: 2,
                source: 1,
                factor: q(-1),
            },
        ],
        reduced: rationals(&[1, 0, 2, 0, 1, 3, 0, 0, 0]),
        pivots: vec![0, 1],
        solution: Solution::Unique {
            values: rationals(&[2, 3]),
        },
    };
    verify_linear_system_witness(&system, &witness, &mut Unbounded).unwrap();
    system.rhs[2] = q(6);
    assert!(verify_linear_system_witness(&system, &witness, &mut Unbounded).is_err());
    witness.reduced[8] = q(1);
    witness.solution = Solution::Inconsistent { row: 2 };
    assert!(
        !verify_linear_system_witness(&system, &witness, &mut Unbounded)
            .unwrap()
            .is_consistent()
    );
    witness.solution = Solution::Unique {
        values: rationals(&[2, 3]),
    };
    assert!(verify_linear_system_witness(&system, &witness, &mut Unbounded).is_err());
}

#[test]
fn residual_satisfiers_cannot_claim_uniqueness_or_omit_free_directions() {
    let (system, valid) = affine_fixture();
    let mut witness = valid.clone();
    // [0,3,0] really satisfies the equations, but is not the unique solution.
    witness.solution = Solution::Unique {
        values: rationals(&[0, 3, 0]),
    };
    assert!(verify_linear_system_witness(&system, &witness, &mut Unbounded).is_err());
    for mode in 0..6 {
        let mut witness = valid.clone();
        let Solution::Affine {
            particular,
            basis,
            free_columns,
        } = &mut witness.solution
        else {
            unreachable!()
        };
        match mode {
            0 => {
                basis.clear();
                free_columns.clear();
            }
            1 => {
                basis.push(basis[0].clone());
                free_columns.push(2);
            }
            2 => basis[0][2] = q(0),
            3 => free_columns[0] = 0,
            4 => free_columns[0] = usize::MAX,
            _ => particular[0] = BigRational::new(1.into(), BigInt::one() << 200u32),
        }
        assert!(
            verify_linear_system_witness(&system, &witness, &mut Unbounded).is_err(),
            "mutant {mode}"
        );
    }
}

#[test]
fn row_operations_cannot_destroy_equations_or_forge_reduced_form() {
    let (system, valid) = affine_fixture();
    for mode in 0..9 {
        let mut witness = valid.clone();
        match mode {
            0 => {
                witness.operations[1] = Row::Scale {
                    row: 1,
                    factor: q(0),
                }
            }
            1 => {
                witness.operations[2] = Row::Add {
                    target: 0,
                    source: 0,
                    factor: q(-1),
                }
            }
            2 => witness.operations[0] = Row::Swap { left: 0, right: 2 },
            3 => witness.operations.clear(),
            4 => witness.reduced[0] = q(0),
            5 => witness.pivots.swap(0, 1),
            6 => witness.pivots.pop().map(|_| ()).unwrap(),
            7 => witness.schema_version += 1,
            _ => witness.solution = Solution::Inconsistent { row: 0 },
        }
        assert!(
            verify_linear_system_witness(&system, &witness, &mut Unbounded).is_err(),
            "mutant {mode}"
        );
    }
    let mut changed = system.clone();
    changed.coefficients[5] = q(2);
    assert!(verify_linear_system_witness(&changed, &valid, &mut Unbounded).is_err());
}

#[test]
fn unknown_fields_and_oversized_sequences_fail_closed() {
    let (_, witness) = affine_fixture();
    let mut value = serde_json::to_value(&witness).unwrap();
    value["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<LinearSystemWitness>(value).is_err());
    let mut value = serde_json::to_value(&witness).unwrap();
    value["solution"]["Affine"]["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<LinearSystemWitness>(value).is_err());
    let mut value = serde_json::to_value(&witness).unwrap();
    value["pivots"] = serde_json::json!(vec![0usize; MAX_LINEAR_SYSTEM_DIMENSION + 1]);
    assert!(serde_json::from_value::<LinearSystemWitness>(value).is_err());
    let mut value = serde_json::to_value(&witness).unwrap();
    value["solution"]["Affine"]["basis"][0] =
        serde_json::to_value(vec![q(0); MAX_LINEAR_SYSTEM_DIMENSION + 1]).unwrap();
    assert!(serde_json::from_value::<LinearSystemWitness>(value).is_err());
}

#[test]
fn wire_coefficients_and_operation_factors_are_bounded_before_the_next_entry() {
    let oversized =
        BigRational::from_integer(BigInt::one() << (MAX_LINEAR_SYSTEM_COEFFICIENT_BITS as u32));
    let encoded = serde_json::to_string(&oversized).unwrap();
    // The second entry is deliberately invalid JSON: height refusal must occur
    // at the first scalar, before either parsing or retaining another entry.
    let claim =
        format!("{{\"rows\":1,\"columns\":2,\"coefficients\":[{encoded},invalid],\"rhs\":[]}}");
    let error = serde_json::from_str::<RationalLinearSystem>(&claim).unwrap_err();
    assert!(error.to_string().contains("coefficient height"));
    let operation = format!("{{\"Scale\":{{\"row\":0,\"factor\":{encoded}}}}}");
    let error = serde_json::from_str::<Row>(&operation).unwrap_err();
    assert!(error.to_string().contains("coefficient height"));
    let operation = format!("{{\"Add\":{{\"target\":0,\"source\":1,\"factor\":{encoded}}}}}");
    assert!(
        serde_json::from_str::<Row>(&operation)
            .unwrap_err()
            .to_string()
            .contains("coefficient height")
    );

    let mut claim = RationalLinearSystem {
        rows: 1,
        columns: 1,
        coefficients: vec![BigRational::from_integer(
            BigInt::one() << (MAX_LINEAR_SYSTEM_COEFFICIENT_BITS as u32 - 1),
        )],
        rhs: rationals(&[0]),
    };
    assert_eq!(
        serde_json::from_str::<RationalLinearSystem>(&serde_json::to_string(&claim).unwrap())
            .unwrap(),
        claim
    );
    claim.rhs[0] = oversized;
    assert!(
        serde_json::from_str::<RationalLinearSystem>(&serde_json::to_string(&claim).unwrap())
            .is_err()
    );

    let (system, witness) = affine_fixture();
    let mut wire = serde_json::to_value(&system).unwrap();
    wire["coefficients"][0] = serde_json::json!([
        serde_json::to_value(BigInt::from(2)).unwrap(),
        serde_json::to_value(BigInt::from(2)).unwrap()
    ]);
    assert!(serde_json::from_value::<RationalLinearSystem>(wire).is_err());
    let mut wire = serde_json::to_value(&witness).unwrap();
    wire["solution"]["Affine"]["basis"][0][0] = serde_json::from_str(&encoded).unwrap();
    assert!(serde_json::from_value::<LinearSystemWitness>(wire).is_err());
}

#[test]
fn shape_and_coefficient_limits_are_checked_before_replay() {
    let (system, witness) = affine_fixture();
    let mut changed = system.clone();
    changed.rows = usize::MAX;
    assert!(matches!(
        verify_linear_system_witness(&changed, &witness, &mut Unbounded),
        Err(LinearSystemVerificationError::Limit(_))
    ));
    let mut changed = system.clone();
    changed.coefficients.pop();
    assert!(verify_linear_system_witness(&changed, &witness, &mut Unbounded).is_err());
    let mut changed = system;
    changed.coefficients[0] =
        BigRational::from_integer(BigInt::one() << (MAX_LINEAR_SYSTEM_COEFFICIENT_BITS as u32));
    assert!(matches!(
        verify_linear_system_witness(&changed, &witness, &mut Unbounded),
        Err(LinearSystemVerificationError::Limit(_))
    ));
}

#[derive(Default)]
struct SafePoints {
    seen: usize,
    cancel_at: Option<usize>,
}
impl BudgetMeter for SafePoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.seen += 1;
        if self.cancel_at == Some(self.seen) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_observed_verifier_safe_point_cancels_without_a_receipt() {
    let system = RationalLinearSystem {
        rows: 1,
        columns: 1,
        coefficients: rationals(&[2]),
        rhs: rationals(&[6]),
    };
    let witness = LinearSystemWitness {
        schema_version: LINEAR_SYSTEM_SCHEMA_VERSION,
        operations: vec![Row::Scale {
            row: 0,
            factor: BigRational::new(1.into(), 2.into()),
        }],
        reduced: rationals(&[1, 3]),
        pivots: vec![0],
        solution: Solution::Unique {
            values: rationals(&[3]),
        },
    };
    let mut count = SafePoints::default();
    verify_linear_system_witness(&system, &witness, &mut count).unwrap();
    assert!(count.seen > 10);
    for point in 1..=count.seen {
        let mut cancelled = SafePoints {
            seen: 0,
            cancel_at: Some(point),
        };
        assert!(
            matches!(
                verify_linear_system_witness(&system, &witness, &mut cancelled),
                Err(LinearSystemVerificationError::Meter(MeterError::Cancelled))
            ),
            "point {point}"
        );
    }
    let mut exhausted = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        verify_linear_system_witness(&system, &witness, &mut exhausted),
        Err(LinearSystemVerificationError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn zero_equations_still_require_a_complete_identity_basis() {
    let system = RationalLinearSystem {
        rows: 0,
        columns: 2,
        coefficients: vec![],
        rhs: vec![],
    };
    let mut witness = LinearSystemWitness {
        schema_version: LINEAR_SYSTEM_SCHEMA_VERSION,
        operations: vec![],
        reduced: vec![],
        pivots: vec![],
        solution: Solution::Affine {
            particular: rationals(&[0, 0]),
            basis: vec![rationals(&[1, 0]), rationals(&[0, 1])],
            free_columns: vec![0, 1],
        },
    };
    assert_eq!(
        verify_linear_system_witness(&system, &witness, &mut Unbounded)
            .unwrap()
            .nullity(),
        2
    );
    if let Solution::Affine { basis, .. } = &mut witness.solution {
        basis[1] = vec![BigRational::zero(), BigRational::zero()];
    }
    assert!(verify_linear_system_witness(&system, &witness, &mut Unbounded).is_err());
}
