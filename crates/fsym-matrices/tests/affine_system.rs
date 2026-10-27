use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_matrices::{
    Matrix, MatrixError, RationalLinearSolution as Solution, verify_affine_solution_candidate,
};
use num_traits::Zero;

fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        values.iter().map(|&value| Expr::from_i64(value)).collect(),
    )
    .unwrap()
}
fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}
fn values(data: &[i64]) -> Vec<BigRational> {
    data.iter().map(|&value| q(value)).collect()
}

#[test]
fn rectangular_affine_solve_returns_all_and_only_free_directions() {
    let a = matrix(2, 3, &[0, 2, 4, 1, 1, 1]);
    let b = matrix(2, 1, &[6, 3]);
    let result = a.solve_affine(&b).unwrap();
    assert_eq!(result.rank(), 2);
    assert_eq!(
        result.solution(),
        &Solution::Affine {
            particular: values(&[0, 3, 0]),
            basis: vec![values(&[1, -2, 1])],
            free_columns: vec![2],
        }
    );
    // Independently enumerate integer solutions to 2y+4z=6, x+y+z=3.
    for x in -4..=4 {
        for y in -4..=4 {
            for z in -4..=4 {
                assert_eq!(
                    2 * y + 4 * z == 6 && x + y + z == 3,
                    x == z && y == 3 - 2 * z
                );
            }
        }
    }
    assert_eq!(result.system().coefficients, values(&[0, 2, 4, 1, 1, 1]));
}

#[test]
fn all_small_two_by_two_systems_match_determinants_and_consistency_minors() {
    // 729 distinct input systems: exact determinant and rank criteria are
    // independent of the production elimination and certificate replay.
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    for u in -1..=1 {
                        for v in -1..=1 {
                            let coefficients = matrix(2, 2, &[a, b, c, d]);
                            let result = coefficients.solve_affine(&matrix(2, 1, &[u, v])).unwrap();
                            let det = a * d - b * c;
                            if det != 0 {
                                assert_eq!(
                                    result.solution(),
                                    &Solution::Unique {
                                        values: vec![
                                            BigRational::new((u * d - b * v).into(), det.into()),
                                            BigRational::new((a * v - u * c).into(), det.into())
                                        ]
                                    }
                                );
                                assert_eq!(result.rank(), 2);
                            } else {
                                let zero = a == 0 && b == 0 && c == 0 && d == 0;
                                let consistent = if zero {
                                    u == 0 && v == 0
                                } else {
                                    a * v - c * u == 0 && b * v - d * u == 0
                                };
                                assert_eq!(
                                    result.receipt().is_consistent(),
                                    consistent,
                                    "{a},{b};{c},{d} rhs {u},{v}"
                                );
                                assert_eq!(result.rank(), if zero { 0 } else { 1 });
                                assert_eq!(
                                    matches!(result.solution(), Solution::Inconsistent { .. }),
                                    !consistent
                                );
                                if let Solution::Affine {
                                    particular,
                                    basis,
                                    free_columns,
                                } = result.solution()
                                {
                                    assert_eq!(basis.len(), 2 - result.rank());
                                    assert_eq!(free_columns.len(), basis.len());
                                    for (row, expected) in [([a, b], u), ([c, d], v)] {
                                        assert_eq!(
                                            q(row[0]) * &particular[0] + q(row[1]) * &particular[1],
                                            q(expected)
                                        );
                                        for direction in basis {
                                            assert!(
                                                (q(row[0]) * &direction[0]
                                                    + q(row[1]) * &direction[1])
                                                    .is_zero()
                                            );
                                        }
                                    }
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
fn overdetermined_and_singular_systems_are_classified_exactly() {
    let a = matrix(3, 2, &[1, 0, 0, 1, 1, 1]);
    let result = a.solve_affine(&matrix(3, 1, &[2, 3, 5])).unwrap();
    assert_eq!(
        result.solution(),
        &Solution::Unique {
            values: values(&[2, 3])
        }
    );
    assert!(matches!(
        a.solve_affine(&matrix(3, 1, &[2, 3, 6]))
            .unwrap()
            .solution(),
        Solution::Inconsistent { .. }
    ));
    let a = matrix(2, 2, &[1, 2, 2, 4]);
    assert_eq!(
        a.solve_affine(&matrix(2, 1, &[3, 6])).unwrap().solution(),
        &Solution::Affine {
            particular: values(&[3, 0]),
            basis: vec![values(&[-2, 1])],
            free_columns: vec![1]
        }
    );
}

#[test]
fn free_columns_before_and_after_pivots_preserve_every_parameter() {
    let a = matrix(2, 4, &[0, 2, 0, 4, 0, 0, 1, 5]);
    assert_eq!(
        a.solve_affine(&matrix(2, 1, &[6, 7])).unwrap().solution(),
        &Solution::Affine {
            particular: values(&[0, 3, 7, 0]),
            basis: vec![values(&[1, 0, 0, 0]), values(&[0, -2, -5, 1])],
            free_columns: vec![0, 3],
        }
    );
}

#[test]
fn empty_dimensions_and_zero_rows_have_mathematical_solution_spaces() {
    assert_eq!(
        matrix(0, 0, &[])
            .solve_affine(&matrix(0, 1, &[]))
            .unwrap()
            .solution(),
        &Solution::Unique { values: vec![] }
    );
    assert_eq!(
        matrix(2, 0, &[])
            .solve_affine(&matrix(2, 1, &[0, 0]))
            .unwrap()
            .solution(),
        &Solution::Unique { values: vec![] }
    );
    assert!(matches!(
        matrix(2, 0, &[])
            .solve_affine(&matrix(2, 1, &[0, 1]))
            .unwrap()
            .solution(),
        Solution::Inconsistent { .. }
    ));
    assert_eq!(
        matrix(0, 3, &[])
            .solve_affine(&matrix(0, 1, &[]))
            .unwrap()
            .solution(),
        &Solution::Affine {
            particular: values(&[0, 0, 0]),
            basis: vec![values(&[1, 0, 0]), values(&[0, 1, 0]), values(&[0, 0, 1])],
            free_columns: vec![0, 1, 2]
        }
    );
    assert!(matches!(
        matrix(2, 3, &[0, 0, 0, 0, 0, 0])
            .solve_affine(&matrix(2, 1, &[0, 1]))
            .unwrap()
            .solution(),
        Solution::Inconsistent { .. }
    ));
}

#[test]
fn arbitrary_rationals_are_exact_and_no_small_value_is_treated_as_zero() {
    let tiny = BigRational::new(BigInt::one(), BigInt::one() << 300u32);
    let a = Matrix::new(1, 2, vec![Expr::Rational(tiny.clone()), Expr::from_i64(0)]).unwrap();
    let b = Matrix::new(1, 1, vec![Expr::Rational(tiny)]).unwrap();
    assert_eq!(
        a.solve_affine(&b).unwrap().solution(),
        &Solution::Affine {
            particular: values(&[1, 0]),
            basis: vec![values(&[0, 1])],
            free_columns: vec![1]
        }
    );
    let a = Matrix::new(
        2,
        2,
        vec![
            Expr::rational(1, 2).unwrap(),
            Expr::rational(1, 3).unwrap(),
            Expr::rational(2, 3).unwrap(),
            Expr::rational(-1, 4).unwrap(),
        ],
    )
    .unwrap();
    let b = Matrix::new(
        2,
        1,
        vec![
            Expr::rational(5, 6).unwrap(),
            Expr::rational(5, 12).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(
        a.solve_affine(&b).unwrap().solution(),
        &Solution::Unique {
            values: values(&[1, 1])
        }
    );
}

#[test]
fn mutated_candidate_is_not_publishable_as_a_verified_solution() {
    let a = matrix(2, 3, &[0, 2, 4, 1, 1, 1]);
    let mut candidate = a
        .affine_solution_candidate_with_meter(&matrix(2, 1, &[6, 3]), &mut Unbounded)
        .unwrap();
    if let Solution::Affine { basis, .. } = &mut candidate.witness.solution {
        basis[0][0] = q(2);
    }
    assert!(matches!(
        verify_affine_solution_candidate(candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
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
fn cancellation_in_either_phase_and_exhausted_verifier_prevent_publication() {
    let a = matrix(1, 1, &[2]);
    let b = matrix(1, 1, &[6]);
    let mut generator = SafePoints::default();
    let mut verifier = SafePoints::default();
    a.solve_affine_with_meters(&b, &mut generator, &mut verifier)
        .unwrap();
    for point in 1..=generator.seen {
        let mut cancel = SafePoints {
            seen: 0,
            cancel_at: Some(point),
        };
        assert_eq!(
            a.solve_affine_with_meters(&b, &mut cancel, &mut Unbounded),
            Err(MatrixError::Cancelled),
            "generation {point}"
        );
    }
    for point in 1..=verifier.seen {
        let mut cancel = SafePoints {
            seen: 0,
            cancel_at: Some(point),
        };
        assert_eq!(
            a.solve_affine_with_meters(&b, &mut Unbounded, &mut cancel),
            Err(MatrixError::Cancelled),
            "verification {point}"
        );
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        a.solve_affine_with_meters(&b, &mut Unbounded, &mut empty),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn all_inputs_and_shapes_are_checked_before_elimination() {
    assert!(matches!(
        matrix(129, 0, &[]).solve_affine(&matrix(129, 1, &vec![0; 129])),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        matrix(1, 1, &[1]).solve_affine(&matrix(1, 2, &[1, 2])),
        Err(MatrixError::ShapeMismatch(..))
    ));
    let symbolic = Matrix::new(1, 2, vec![Expr::from_i64(1), Expr::symbol("opaque")]).unwrap();
    assert_eq!(
        symbolic.solve_affine(&matrix(1, 1, &[0])),
        Err(MatrixError::UnsupportedCertificateDomain)
    );
    let huge = Matrix::new(
        1,
        2,
        vec![Expr::from_i64(0), Expr::Integer(BigInt::one() << 8192u32)],
    )
    .unwrap();
    assert!(matches!(
        huge.solve_affine(&matrix(1, 1, &[0])),
        Err(MatrixError::ResourceLimit(_))
    ));
}
