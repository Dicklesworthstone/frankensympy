//! Integer solvability, primitive lattices, multiple RHS and original binding.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_matrices::{
    IntegerMatrixRef, Matrix, MatrixError, generate_integer_linear_candidate,
    metered_solve_integer, verify_integer_linear_candidate,
};
use fsym_proof_kernel::integer_linear_system::{
    IntegerLinearOutcome, verify_integer_linear_system,
};
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry};

fn matrix(rows: usize, columns: usize, entries: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        entries.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn dense(source: &RationalSparseMatrix) -> Matrix {
    let mut entries = vec![Expr::from_i64(0); source.rows * source.columns];
    for entry in &source.entries {
        entries[entry.row * source.columns + entry.column] =
            Expr::Integer(entry.value.numer().clone());
    }
    Matrix::new(source.rows, source.columns, entries).unwrap()
}

#[test]
fn primitive_integer_kernel_represents_every_integer_parameter() {
    let a = matrix(1, 2, &[2, 4]);
    let b = matrix(1, 1, &[6]);
    let accepted = a.solve_integer(&b).unwrap();
    assert!(accepted.is_consistent());
    assert!(!accepted.is_unique());
    assert_eq!(accepted.rank(), 1);
    assert_eq!(accepted.nullity(), 1);
    assert_eq!(accepted.contradiction(), None);
    let p = dense(accepted.particular().unwrap());
    let k = dense(accepted.kernel().unwrap());
    assert_eq!(p, matrix(2, 1, &[3, 0]));
    assert_eq!(k, matrix(2, 1, &[-2, 1]));
    for t in -12..=12 {
        let x = p.add(&k.scalar_mul(&Expr::from_i64(t)).unwrap()).unwrap();
        assert_eq!(a.matmul(&x).unwrap(), b);
        assert_eq!(x, matrix(2, 1, &[3 - 2 * t, t]));
    }
    let replay = verify_integer_linear_system(
        accepted.source(),
        accepted.rhs(),
        accepted.witness(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(&replay, accepted.receipt());
}

#[test]
fn rationally_consistent_but_integer_impossible_is_a_checked_outcome() {
    let a = matrix(1, 2, &[2, 4]);
    let result = a.solve_integer(&matrix(1, 1, &[1])).unwrap();
    assert!(!result.is_consistent());
    assert!(!result.is_unique());
    assert_eq!(result.contradiction(), Some((0, 0)));
    assert!(result.particular().is_none());
    assert!(result.kernel().is_none());
    assert_eq!(result.receipt().particular_digest(), None);
    assert_eq!(result.receipt().kernel_digest(), None);
}

#[test]
fn transformed_zero_rows_and_all_rhs_columns_are_checked() {
    let a = matrix(2, 2, &[2, 4, 4, 8]);
    let result = a.solve_integer(&matrix(2, 1, &[6, 13])).unwrap();
    assert_eq!(result.contradiction(), Some((1, 0)));
    let b = matrix(2, 2, &[6, 10, 12, 20]);
    let accepted = a.solve_integer(&b).unwrap();
    let p = dense(accepted.particular().unwrap());
    let k = dense(accepted.kernel().unwrap());
    assert_eq!(p, matrix(2, 2, &[3, 5, 0, 0]));
    let t = matrix(1, 2, &[-7, 11]);
    let x = p.add(&k.matmul(&t).unwrap()).unwrap();
    assert_eq!(a.matmul(&x).unwrap(), b);
    let bad = a.solve_integer(&matrix(2, 2, &[6, 7, 12, 14])).unwrap();
    assert_eq!(bad.contradiction(), Some((0, 1)));
}

#[test]
fn full_column_rank_tall_sources_have_a_unique_integer_solution() {
    let a = matrix(3, 2, &[2, 0, 0, 3, 2, 3]);
    let b = matrix(3, 1, &[4, 9, 13]);
    let result = a.solve_integer(&b).unwrap();
    assert!(result.is_unique());
    assert_eq!(result.rank(), 2);
    assert_eq!(result.nullity(), 0);
    assert_eq!(dense(result.particular().unwrap()), matrix(2, 1, &[2, 3]));
    assert_eq!(
        dense(result.kernel().unwrap()),
        Matrix::zeros(2, 0).unwrap()
    );
}

#[test]
fn mixed_dense_sparse_inputs_have_identical_canonical_evidence() {
    let a = matrix(2, 3, &[2, 4, 4, 6, 6, 12]);
    let b = matrix(2, 2, &[10, 8, 24, 18]);
    let sparse_a = a.to_sparse().unwrap();
    let sparse_b = b.to_sparse().unwrap();
    let expected = a.solve_integer(&b).unwrap();
    assert!(expected.is_consistent());
    for source in [
        IntegerMatrixRef::Dense(&a),
        IntegerMatrixRef::Sparse(&sparse_a),
    ] {
        for rhs in [
            IntegerMatrixRef::Dense(&b),
            IntegerMatrixRef::Sparse(&sparse_b),
        ] {
            let result =
                metered_solve_integer(source, rhs, &mut Unbounded, &mut Unbounded).unwrap();
            assert_eq!(result.source(), expected.source());
            assert_eq!(result.rhs(), expected.rhs());
            assert_eq!(result.witness(), expected.witness());
            assert_eq!(result.receipt(), expected.receipt());
        }
    }
    assert_eq!(
        sparse_a.solve_integer(&sparse_b).unwrap().witness(),
        expected.witness()
    );
}

#[test]
fn empty_equations_variables_and_rhs_columns_keep_mathematical_shapes() {
    for variables in [0, 3] {
        for rhs_columns in [0, 2] {
            let a = Matrix::zeros(0, variables).unwrap();
            let b = Matrix::zeros(0, rhs_columns).unwrap();
            let result = a.solve_integer(&b).unwrap();
            assert!(result.is_consistent());
            assert_eq!(result.is_unique(), variables == 0 || rhs_columns == 0);
            assert_eq!(
                dense(result.particular().unwrap()),
                Matrix::zeros(variables, rhs_columns).unwrap()
            );
            assert_eq!(
                dense(result.kernel().unwrap()),
                Matrix::eye(variables).unwrap()
            );
        }
    }
    let a = Matrix::zeros(3, 0).unwrap();
    let result = a.solve_integer(&Matrix::zeros(3, 2).unwrap()).unwrap();
    assert!(result.is_unique());
    assert_eq!(
        dense(result.particular().unwrap()),
        Matrix::zeros(0, 2).unwrap()
    );
    assert_eq!(
        dense(result.kernel().unwrap()),
        Matrix::zeros(0, 0).unwrap()
    );
    let impossible = a.solve_integer(&matrix(3, 1, &[0, 1, 0])).unwrap();
    assert_eq!(impossible.contradiction(), Some((1, 0)));

    let a = matrix(2, 3, &[2, 4, 6, 4, 8, 12]);
    let empty_rhs = a.solve_integer(&Matrix::zeros(2, 0).unwrap()).unwrap();
    assert!(empty_rhs.is_unique());
    assert_eq!(empty_rhs.nullity(), 2);
    assert_eq!(
        dense(empty_rhs.particular().unwrap()),
        Matrix::zeros(3, 0).unwrap()
    );
    assert_eq!(
        (
            empty_rhs.kernel().unwrap().rows,
            empty_rhs.kernel().unwrap().columns
        ),
        (3, 2)
    );
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

#[test]
fn exhaustive_small_diophantine_rows_match_the_integer_gcd_criterion() {
    for a in -3..=3 {
        for b in -3..=3 {
            let source = matrix(1, 2, &[a, b]);
            let divisor = gcd(a, b);
            for c in -4..=4 {
                let rhs = matrix(1, 1, &[c]);
                let result = source.solve_integer(&rhs).unwrap();
                let possible = if divisor == 0 {
                    c == 0
                } else {
                    c % divisor == 0
                };
                assert_eq!(result.is_consistent(), possible, "{a}x+{b}y={c}");
                if possible {
                    assert_eq!(
                        source.matmul(&dense(result.particular().unwrap())).unwrap(),
                        rhs
                    );
                    let kernel = dense(result.kernel().unwrap());
                    assert_eq!(
                        source.matmul(&kernel).unwrap(),
                        Matrix::zeros(1, result.nullity()).unwrap()
                    );
                    if divisor != 0 {
                        let entries = kernel.data();
                        let values: Vec<i64> = entries
                            .iter()
                            .map(|entry| {
                                let Expr::Integer(value) = entry else {
                                    panic!("noninteger kernel")
                                };
                                value.to_string().parse().unwrap()
                            })
                            .collect();
                        assert_eq!(gcd(values[0], values[1]), 1);
                    }
                }
            }
        }
    }
}

#[test]
fn both_original_domains_are_admitted_before_empty_shortcuts() {
    let symbolic = Matrix::new(1, 1, vec![Expr::Sym(Symbol::new("x"))]).unwrap();
    assert!(matches!(
        symbolic.solve_integer(&Matrix::zeros(1, 0).unwrap()),
        Err(MatrixError::UnsupportedIntegerDomain)
    ));
    let no_variables = Matrix::zeros(1, 0).unwrap();
    assert!(matches!(
        no_variables.solve_integer(&symbolic),
        Err(MatrixError::UnsupportedIntegerDomain)
    ));
    let rational = Matrix::new(
        1,
        1,
        vec![Expr::Rational(BigRational::new(1.into(), 2.into()))],
    )
    .unwrap();
    assert!(matches!(
        no_variables.solve_integer(&rational),
        Err(MatrixError::UnsupportedIntegerDomain)
    ));
    let huge = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4096u32)]).unwrap();
    assert!(matches!(
        no_variables.solve_integer(&huge),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::zeros(0, 0)
            .unwrap()
            .solve_integer(&Matrix::zeros(0, 65).unwrap()),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::zeros(2, 2)
            .unwrap()
            .solve_integer(&Matrix::zeros(3, 1).unwrap()),
        Err(MatrixError::ShapeMismatch(_, _, _, _))
    ));
}

#[test]
fn sources_residual_only_lattices_and_false_contradictions_are_rejected() {
    let a = matrix(1, 2, &[2, 4]);
    let b = matrix(1, 1, &[6]);
    let candidate =
        generate_integer_linear_candidate((&a).into(), (&b).into(), &mut Unbounded).unwrap();
    let reject = |candidate| {
        assert!(
            verify_integer_linear_candidate((&a).into(), (&b).into(), candidate, &mut Unbounded)
                .is_err()
        );
    };
    let mut wrong_a = candidate.clone();
    wrong_a.source.entries[0].value = BigRational::from_integer(3.into());
    reject(wrong_a);
    let mut wrong_b = candidate.clone();
    wrong_b.rhs.entries[0].value = BigRational::from_integer(8.into());
    reject(wrong_b);
    let mut nonprimitive = candidate.clone();
    let IntegerLinearOutcome::Solvable { kernel, .. } = &mut nonprimitive.witness.outcome else {
        panic!()
    };
    for entry in &mut kernel.entries {
        entry.value *= BigRational::from_integer(2.into());
    }
    // This still has A*K=0 and the correct rational nullity, but omits valid
    // integer solutions whose second coordinate is odd.
    assert_eq!(a.matmul(&dense(kernel)).unwrap(), matrix(1, 1, &[0]));
    reject(nonprimitive);
    let mut shifted = candidate.clone();
    let IntegerLinearOutcome::Solvable { particular, .. } = &mut shifted.witness.outcome else {
        panic!()
    };
    particular.entries[0].value = BigRational::from_integer(1.into());
    particular.entries.push(SparseRationalEntry {
        row: 1,
        column: 0,
        value: BigRational::from_integer(1.into()),
    });
    assert_eq!(a.matmul(&dense(particular)).unwrap(), b);
    // The receiver binds a canonical representative to this exact witness.
    reject(shifted);
    let mut false_negative = candidate;
    false_negative.witness.outcome = IntegerLinearOutcome::Inconsistent {
        row: 0,
        rhs_column: 0,
    };
    reject(false_negative);
}

struct CancelAfter {
    used: usize,
    stop: usize,
}

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.used += 1;
        if self.used >= self.stop {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_across_the_complete_solve_never_becomes_inconsistency() {
    let a = matrix(2, 3, &[2, 4, 4, 6, 6, 12]);
    let b = matrix(2, 2, &[10, 8, 24, 18]);
    let mut generation = CancelAfter {
        used: 0,
        stop: usize::MAX,
    };
    let mut verification = CancelAfter {
        used: 0,
        stop: usize::MAX,
    };
    a.metered_solve_integer(&b, &mut generation, &mut verification)
        .unwrap();
    for stop in [1, generation.used / 2, generation.used] {
        assert!(matches!(
            a.metered_solve_integer(&b, &mut CancelAfter { used: 0, stop }, &mut Unbounded),
            Err(MatrixError::Cancelled)
        ));
    }
    for stop in [1, verification.used / 2, verification.used] {
        assert!(matches!(
            a.metered_solve_integer(&b, &mut Unbounded, &mut CancelAfter { used: 0, stop }),
            Err(MatrixError::Cancelled)
        ));
    }
}

#[test]
fn both_phases_refuse_compute_memory_and_allocation_exhaustion() {
    let a = matrix(1, 2, &[2, 4]);
    let b = matrix(1, 1, &[6]);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(
            a.metered_solve_integer(&b, &mut Budget::new(limits.clone()), &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            a.metered_solve_integer(&b, &mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}
