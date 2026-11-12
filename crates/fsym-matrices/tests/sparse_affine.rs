//! Complete sparse affine families and native parameterized publication.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    MatrixError, SparseAffineSolutionCandidate, SparseMatrix,
    verify_sparse_affine_solution_candidate,
};
use fsym_proof_kernel::sparse_rref::{
    MAX_SPARSE_RREF_COEFFICIENT_BITS, MAX_SPARSE_RREF_DIMENSION, SparseAffineOutcome,
};
use std::collections::BTreeMap;

fn integers(rows: usize, columns: usize, values: &[i64]) -> SparseMatrix {
    assert_eq!(values.len(), rows * columns);
    SparseMatrix::new(
        rows,
        columns,
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0)
            .map(|(index, &value)| ((index / columns, index % columns), Expr::from_i64(value)))
            .collect(),
    )
    .unwrap()
}

fn candidate(matrix: &SparseMatrix, rhs: &SparseMatrix) -> SparseAffineSolutionCandidate {
    matrix
        .affine_solution_candidate_with_meter(rhs, &mut Unbounded)
        .unwrap()
}

fn assert_rejected(
    matrix: &SparseMatrix,
    rhs: &SparseMatrix,
    proposed: SparseAffineSolutionCandidate,
) {
    assert!(matches!(
        verify_sparse_affine_solution_candidate(matrix, rhs, proposed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn pivoted_wide_system_returns_one_particular_per_rhs_and_one_complete_kernel() {
    let matrix = integers(2, 4, &[0, 2, 0, 4, 0, 0, 1, 5]);
    let rhs = integers(2, 2, &[6, -2, 7, 3]);
    let original_a = matrix.clone();
    let original_b = rhs.clone();
    let accepted = matrix.solve_affine(&rhs).unwrap();
    assert_eq!(accepted.rank(), 2);
    assert_eq!(accepted.nullity(), 2);
    assert!(accepted.is_consistent());
    assert!(!accepted.is_unique());
    assert_eq!(accepted.free_columns(), &[0, 3]);
    assert!(accepted.receipt().particular_digest().is_some());
    assert!(accepted.receipt().kernel_digest().is_some());
    assert_eq!(accepted.witness().reduction.reduced.columns, 6);
    let (particular, kernel, free) = accepted.into_parts().unwrap();
    assert_eq!(particular, integers(4, 2, &[0, 0, 3, -1, 7, 3, 0, 0]));
    assert_eq!(kernel, integers(4, 2, &[1, 0, 0, -2, 0, -5, 0, 1]));
    assert_eq!(free, [0, 3]);
    assert_eq!(matrix.matmul(&particular).unwrap(), rhs);
    assert_eq!(matrix.matmul(&kernel).unwrap(), SparseMatrix::zeros(2, 2));
    assert_eq!(matrix, original_a);
    assert_eq!(rhs, original_b);
}

#[test]
fn all_729_small_systems_match_independent_determinants_and_consistency_minors() {
    // Every 2x2 matrix over {-1,0,1}, every first RHS over {-1,0,1},
    // and a second RHS known independently to be A*[2,-1].
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    for u in -1..=1 {
                        for v in -1..=1 {
                            let matrix = integers(2, 2, &[a, b, c, d]);
                            let rhs = integers(2, 2, &[u, 2 * a - b, v, 2 * c - d]);
                            let accepted = matrix.solve_affine(&rhs).unwrap();
                            let det = a * d - b * c;
                            let zero = a == 0 && b == 0 && c == 0 && d == 0;
                            let consistent = if det != 0 {
                                true
                            } else if zero {
                                u == 0 && v == 0
                            } else {
                                a * v - c * u == 0 && b * v - d * u == 0
                            };
                            assert_eq!(accepted.is_consistent(), consistent);
                            assert_eq!(
                                accepted.rank(),
                                if det != 0 {
                                    2
                                } else if zero {
                                    0
                                } else {
                                    1
                                }
                            );
                            if !consistent {
                                assert!(accepted.into_parts().is_none());
                                continue;
                            }
                            let (particular, kernel, free) = accepted.into_parts().unwrap();
                            assert_eq!(matrix.matmul(&particular).unwrap(), rhs);
                            assert_eq!(
                                matrix.matmul(&kernel).unwrap(),
                                SparseMatrix::zeros(2, free.len())
                            );
                            if det != 0 {
                                assert_eq!(
                                    particular.get(0, 0).unwrap(),
                                    Expr::rational(u * d - b * v, det).unwrap()
                                );
                                assert_eq!(
                                    particular.get(1, 0).unwrap(),
                                    Expr::rational(a * v - u * c, det).unwrap()
                                );
                                assert_eq!(particular.get(0, 1).unwrap(), Expr::from_i64(2));
                                assert_eq!(particular.get(1, 1).unwrap(), Expr::from_i64(-1));
                                assert!(free.is_empty());
                            } else {
                                let expected_free = if zero {
                                    vec![0, 1]
                                } else if a != 0 || c != 0 {
                                    vec![1]
                                } else {
                                    vec![0]
                                };
                                assert_eq!(free, expected_free);
                                for (index, &row) in free.iter().enumerate() {
                                    for column in 0..free.len() {
                                        assert_eq!(
                                            kernel.get(row, column).unwrap(),
                                            Expr::from_i64(i64::from(index == column))
                                        );
                                    }
                                    assert_eq!(particular.get(row, 0).unwrap(), Expr::from_i64(0));
                                    assert_eq!(particular.get(row, 1).unwrap(), Expr::from_i64(0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn one_inconsistent_rhs_rejects_the_joint_family_but_preserves_coefficient_rank() {
    let matrix = integers(3, 2, &[1, 2, 2, 4, 0, 0]);
    for rhs in [
        integers(3, 2, &[3, 0, 6, 0, 0, 1]),
        integers(3, 2, &[0, 3, 0, 6, 1, 0]),
    ] {
        let accepted = matrix.solve_affine(&rhs).unwrap();
        assert_eq!(accepted.rank(), 1);
        assert_eq!(accepted.receipt().augmented_rank(), 2);
        assert_eq!(accepted.free_columns(), &[1]);
        assert!(!accepted.is_consistent());
        assert!(!accepted.is_unique());
        assert!(accepted.receipt().contradiction_row().is_some());
        assert!(accepted.particular().is_none());
        assert!(accepted.kernel().is_none());
        assert!(accepted.into_parts().is_none());
        assert!(matrix.gauss_jordan_solve(&rhs).unwrap().is_none());
    }
}

#[test]
fn zero_shapes_and_empty_rhs_keep_the_kernel_without_inventing_parameters() {
    for rows in [0, 2] {
        for columns in [0, 3] {
            for rhs_columns in [0, 2] {
                let matrix = SparseMatrix::zeros(rows, columns);
                let rhs = SparseMatrix::zeros(rows, rhs_columns);
                let accepted = matrix.solve_affine(&rhs).unwrap();
                assert!(accepted.is_consistent());
                assert_eq!(accepted.rank(), 0);
                assert_eq!(accepted.nullity(), columns);
                assert_eq!(accepted.is_unique(), columns == 0 || rhs_columns == 0);
                let (particular, kernel, free) = accepted.into_parts().unwrap();
                assert_eq!(particular, SparseMatrix::zeros(columns, rhs_columns));
                assert_eq!(kernel, SparseMatrix::eye(columns).unwrap());
                assert_eq!(free, (0..columns).collect::<Vec<_>>());
                let (solution, parameters, rendered_free) =
                    matrix.gauss_jordan_solve(&rhs).unwrap().unwrap();
                assert_eq!((solution.rows(), solution.cols()), (columns, rhs_columns));
                assert_eq!(
                    (parameters.rows(), parameters.cols()),
                    (columns, rhs_columns)
                );
                assert_eq!(solution, parameters);
                assert_eq!(rendered_free, free);
            }
        }
    }
    let no_variables = SparseMatrix::zeros(2, 0);
    assert!(
        !no_variables
            .solve_affine(&integers(2, 1, &[0, 1]))
            .unwrap()
            .is_consistent()
    );
    let full_rank = integers(3, 2, &[1, 0, 0, 1, 1, 1]);
    let empty_rhs = SparseMatrix::zeros(3, 0);
    let accepted = full_rank.solve_affine(&empty_rhs).unwrap();
    assert_eq!(accepted.rank(), 2);
    assert!(accepted.is_unique());
    assert_eq!(
        accepted.into_parts().unwrap(),
        (SparseMatrix::zeros(2, 0), SparseMatrix::zeros(2, 0), vec![])
    );
}

#[test]
fn tiny_nonzero_coefficients_and_fractional_particulars_remain_exact() {
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(10).pow(80));
    let matrix = SparseMatrix::new(
        1,
        2,
        BTreeMap::from([((0, 0), Expr::Rational(tiny.clone()))]),
    )
    .unwrap();
    let rhs = SparseMatrix::new(
        1,
        2,
        BTreeMap::from([
            (
                (0, 0),
                Expr::Rational(tiny.clone() * BigRational::new(2.into(), 5.into())),
            ),
            (
                (0, 1),
                Expr::Rational(tiny * BigRational::from_integer((-7).into())),
            ),
        ]),
    )
    .unwrap();
    let accepted = matrix.solve_affine(&rhs).unwrap();
    assert_eq!(accepted.rank(), 1);
    let (particular, kernel, free) = accepted.into_parts().unwrap();
    assert_eq!(particular.get(0, 0).unwrap(), Expr::rational(2, 5).unwrap());
    assert_eq!(particular.get(0, 1).unwrap(), Expr::from_i64(-7));
    assert_eq!(kernel, integers(2, 1, &[0, 1]));
    assert_eq!(free, [1]);
}

#[test]
fn large_wide_input_complete_kernel_and_parameterized_result_stay_sparse() {
    let n = 600;
    let mut coefficients = BTreeMap::new();
    let mut rhs_entries = BTreeMap::new();
    let mut expected_particular = BTreeMap::new();
    let mut expected_kernel = BTreeMap::new();
    for row in 0..n {
        coefficients.insert((row, row), Expr::from_i64(2));
        coefficients.insert((row, n + row), Expr::from_i64(3));
        expected_kernel.insert((row, row), Expr::rational(-3, 2).unwrap());
        expected_kernel.insert((n + row, row), Expr::from_i64(1));
        if row % 7 == 0 {
            rhs_entries.insert((row, 0), Expr::from_i64(10));
            expected_particular.insert((row, 0), Expr::from_i64(5));
        }
        if row % 11 == 0 {
            rhs_entries.insert((row, 1), Expr::from_i64(-4));
            expected_particular.insert((row, 1), Expr::from_i64(-2));
        }
    }
    let matrix = SparseMatrix::new(n, 2 * n, coefficients).unwrap();
    let rhs = SparseMatrix::new(n, 2, rhs_entries).unwrap();
    assert!(matches!(
        matrix.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    let accepted = matrix.solve_affine(&rhs).unwrap();
    assert_eq!(accepted.rank(), n);
    let (particular, kernel, free) = accepted.into_parts().unwrap();
    assert_eq!(
        particular,
        SparseMatrix::new(2 * n, 2, expected_particular).unwrap()
    );
    assert_eq!(
        kernel,
        SparseMatrix::new(2 * n, n, expected_kernel).unwrap()
    );
    assert!(matches!(
        kernel.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    assert_eq!(free, (n..2 * n).collect::<Vec<_>>());
    let (solution, parameters, rendered_free) = matrix.gauss_jordan_solve(&rhs).unwrap().unwrap();
    assert_eq!(
        (solution.rows(), solution.cols(), solution.nnz()),
        (2 * n, 2, 4 * n)
    );
    assert_eq!(
        (parameters.rows(), parameters.cols(), parameters.nnz()),
        (n, 2, 2 * n)
    );
    assert_eq!(parameters.get(n - 1, 1).unwrap(), Expr::symbol("tau1199"));
    assert_eq!(solution.get(2 * n - 1, 1).unwrap(), Expr::symbol("tau1199"));
    assert_eq!(rendered_free, free);
}

#[test]
fn candidate_acceptance_binds_both_actual_original_matrices() {
    let matrix = integers(2, 3, &[1, 2, 0, 0, 0, 1]);
    let rhs = integers(2, 2, &[3, 7, 5, 11]);
    let changed_a = integers(2, 3, &[1, 3, 0, 0, 0, 1]);
    let changed_b = integers(2, 2, &[3, 7, 5, 12]);
    assert_rejected(&matrix, &rhs, candidate(&changed_a, &rhs));
    assert_rejected(&matrix, &rhs, candidate(&matrix, &changed_b));
    let mut proposed = candidate(&matrix, &rhs);
    proposed.coefficients.entries.swap(0, 1);
    assert_rejected(&matrix, &rhs, proposed);
    let mut proposed = candidate(&matrix, &rhs);
    proposed.rhs.columns += 1;
    assert_rejected(&matrix, &rhs, proposed);
}

#[test]
fn incorrect_incomplete_and_nearly_correct_families_never_get_an_accepted_wrapper() {
    let matrix = integers(2, 4, &[0, 2, 0, 4, 0, 0, 1, 5]);
    let rhs = integers(2, 2, &[6, -2, 7, 3]);
    let original = candidate(&matrix, &rhs);
    for mutation in 0..7 {
        let mut proposed = original.clone();
        let SparseAffineOutcome::Solvable {
            particular,
            kernel,
            free_columns,
        } = &mut proposed.witness.outcome
        else {
            panic!("expected a generated family")
        };
        match mutation {
            0 => particular.entries[0].value += BigRational::from_integer(1.into()),
            1 => {
                particular.entries.pop();
            }
            2 => kernel.entries[0].value += BigRational::new(1.into(), BigInt::from(10).pow(80)),
            3 => {
                kernel.entries.pop();
            }
            4 => free_columns.swap(0, 1),
            5 => kernel.columns -= 1,
            6 => {
                let _ = proposed.witness.reduction.pivots.pop();
            }
            _ => unreachable!(),
        }
        assert_rejected(&matrix, &rhs, proposed);
    }
    let mut proposed = original;
    proposed.witness.outcome = SparseAffineOutcome::Inconsistent { row: 0 };
    assert_rejected(&matrix, &rhs, proposed);
}

#[test]
fn original_domain_and_shape_preflight_applies_before_degenerate_shortcuts() {
    let literal = integers(2, 2, &[1, 0, 0, 0]);
    let symbolic = SparseMatrix::new(2, 2, BTreeMap::from([((1, 1), Expr::symbol("x"))])).unwrap();
    assert!(matches!(
        symbolic.solve_affine(&SparseMatrix::zeros(2, 0)),
        Err(MatrixError::UnsupportedCertificateDomain)
    ));
    assert!(matches!(
        literal.solve_affine(&symbolic),
        Err(MatrixError::UnsupportedCertificateDomain)
    ));
    let huge = Expr::Integer(BigInt::from(1) << MAX_SPARSE_RREF_COEFFICIENT_BITS as u32);
    let rhs = SparseMatrix::new(2, 1, BTreeMap::from([((1, 0), huge)])).unwrap();
    assert!(matches!(
        literal.solve_affine(&rhs),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        literal.solve_affine(&SparseMatrix::zeros(3, 1)),
        Err(MatrixError::ShapeMismatch(..))
    ));
    assert!(matches!(
        SparseMatrix::zeros(0, MAX_SPARSE_RREF_DIMENSION).solve_affine(&SparseMatrix::zeros(0, 1)),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        SparseMatrix::zeros(MAX_SPARSE_RREF_DIMENSION + 1, 0)
            .solve_affine(&SparseMatrix::zeros(MAX_SPARSE_RREF_DIMENSION + 1, 0)),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn native_parameters_are_row_major_and_shared_across_the_correct_free_directions() {
    let matrix = integers(2, 4, &[0, 2, 0, 4, 0, 0, 1, 5]);
    let rhs = integers(2, 2, &[6, -2, 7, 3]);
    let (solution, parameters, free) = matrix.gauss_jordan_solve(&rhs).unwrap().unwrap();
    assert_eq!(free, [0, 3]);
    for row in 0..2 {
        for column in 0..2 {
            assert_eq!(
                parameters.get(row, column).unwrap(),
                Expr::symbol(format!("tau{}", row * 2 + column))
            );
        }
    }
    assert_eq!(solution.get(0, 0).unwrap(), Expr::symbol("tau0"));
    assert_eq!(solution.get(0, 1).unwrap(), Expr::symbol("tau1"));
    assert_eq!(
        solution.get(1, 0).unwrap(),
        Expr::from_i64(3) - Expr::from_i64(2) * Expr::symbol("tau2")
    );
    assert_eq!(
        solution.get(1, 1).unwrap(),
        Expr::from_i64(-1) - Expr::from_i64(2) * Expr::symbol("tau3")
    );
    assert_eq!(
        solution.get(2, 0).unwrap(),
        Expr::from_i64(7) - Expr::from_i64(5) * Expr::symbol("tau2")
    );
    assert_eq!(
        solution.get(2, 1).unwrap(),
        Expr::from_i64(3) - Expr::from_i64(5) * Expr::symbol("tau3")
    );
    assert_eq!(solution.get(3, 0).unwrap(), Expr::symbol("tau2"));
    assert_eq!(solution.get(3, 1).unwrap(), Expr::symbol("tau3"));
    assert_eq!(matrix.matmul(&solution).unwrap(), rhs);
}

#[test]
fn parameter_and_term_amplification_refuses_before_constructing_large_outputs() {
    for (variables, rhs_columns, expected) in [
        (513, 513, "parameter stored-entry"),
        (512, 257, "aggregate affine symbolic stored-entry"),
    ] {
        let matrix = SparseMatrix::zeros(1, variables);
        let rhs = SparseMatrix::zeros(1, rhs_columns);
        assert!(matrix.solve_affine(&rhs).unwrap().is_consistent());
        let error = matrix.gauss_jordan_solve(&rhs).unwrap_err();
        assert!(
            matches!(&error, MatrixError::ResourceLimit(message) if message.contains(expected)),
            "{error}"
        );
    }
    let mut entries = BTreeMap::new();
    for row in 0..100 {
        entries.insert((row, row), Expr::from_i64(1));
        for column in 100..110 {
            entries.insert((row, column), Expr::from_i64(1));
        }
    }
    let matrix = SparseMatrix::new(100, 110, entries).unwrap();
    let rhs = SparseMatrix::zeros(100, 1000);
    let error = matrix.gauss_jordan_solve(&rhs).unwrap_err();
    assert!(matches!(
        error,
        MatrixError::ResourceLimit(message) if message.contains("symbolic term limit")
    ));
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    used: [u64; 3],
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        let index = match dimension {
            Dimension::ComputeSteps => Some(0),
            Dimension::MemoryBytes => Some(1),
            Dimension::AllocationCount => Some(2),
            _ => None,
        };
        if let Some(index) = index {
            self.used[index] += amount;
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

#[test]
fn canonical_and_parameterized_publication_each_charge_one_cumulative_ledger() {
    let matrix = integers(2, 3, &[1, 2, 0, 0, 0, 1]);
    let rhs = integers(2, 2, &[3, 7, 5, 11]);
    for rendered in [false, true] {
        let solve = |meter: &mut Budget| {
            if rendered {
                matrix.metered_gauss_jordan_solve(&rhs, meter)
            } else {
                matrix
                    .metered_solve_affine(&rhs, meter)
                    .map(|result| result.into_parts())
            }
        };
        let mut measured = Meter::default();
        let expected = if rendered {
            matrix
                .metered_gauss_jordan_solve(&rhs, &mut measured)
                .unwrap()
        } else {
            matrix
                .metered_solve_affine(&rhs, &mut measured)
                .unwrap()
                .into_parts()
        };
        for (dimension, index) in [
            (Dimension::ComputeSteps, 0),
            (Dimension::MemoryBytes, 1),
            (Dimension::AllocationCount, 2),
        ] {
            assert!(measured.used[index] > 0);
            let mut exact = BudgetLimits::uniform(u64::MAX, 0);
            exact.dimensions[dimension.index()] = measured.used[index];
            assert_eq!(solve(&mut Budget::new(exact)).unwrap(), expected);
            let mut short = BudgetLimits::uniform(u64::MAX, 0);
            short.dimensions[dimension.index()] = measured.used[index] - 1;
            assert!(matches!(
                solve(&mut Budget::new(short)),
                Err(MatrixError::ResourceLimit(_))
            ));
        }
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = measured.used[0] * 2 - 1;
        let mut shared = Budget::new(limits);
        assert_eq!(solve(&mut shared).unwrap(), expected);
        assert!(matches!(
            solve(&mut shared),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}

#[test]
fn cancellation_at_every_safe_point_prevents_parameterized_publication() {
    let matrix = integers(1, 2, &[1, 2]);
    let rhs = integers(1, 2, &[3, 7]);
    let mut measured = Meter::default();
    matrix
        .metered_gauss_jordan_solve(&rhs, &mut measured)
        .unwrap();
    for stop in 1..=measured.checkpoints {
        let mut meter = Meter {
            cancel_at: Some(stop),
            ..Meter::default()
        };
        assert!(
            matches!(
                matrix.metered_gauss_jordan_solve(&rhs, &mut meter),
                Err(MatrixError::Cancelled)
            ),
            "safe point {stop}"
        );
    }
}

#[test]
fn separate_generator_and_verifier_meters_never_reset_or_publish_a_candidate() {
    let matrix = integers(1, 2, &[1, 2]);
    let rhs = integers(1, 1, &[3]);
    let mut generator = Meter::default();
    let mut verifier = Meter {
        cancel_at: Some(1),
        ..Meter::default()
    };
    assert!(matches!(
        matrix.solve_affine_with_meters(&rhs, &mut generator, &mut verifier),
        Err(MatrixError::Cancelled)
    ));
    assert!(generator.used.iter().all(|&used| used > 0));
    assert_eq!(verifier.checkpoints, 1);
    let accepted = matrix
        .solve_affine_with_meters(&rhs, &mut Unbounded, &mut Unbounded)
        .unwrap();
    assert_eq!(accepted.free_columns(), &[1]);
    assert_eq!(accepted.particular().unwrap(), &integers(2, 1, &[3, 0]));
}
