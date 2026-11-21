//! Canonical column-lattice bases, complete witnesses, admission and stops.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_matrices::{
    Matrix, MatrixError, generate_hermite_normal_form_candidate,
    verify_hermite_normal_form_candidate,
};
use fsym_proof_kernel::integer_hermite::verify_integer_hermite;
use fsym_proof_kernel::sparse_rref::{RationalSparseMatrix, SparseRationalEntry};

fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        values.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn dense(value: &RationalSparseMatrix) -> Matrix {
    let mut entries = vec![Expr::from_i64(0); value.rows * value.columns];
    for entry in &value.entries {
        assert!(entry.value.is_integer());
        entries[entry.row * value.columns + entry.column] =
            Expr::Integer(entry.value.numer().clone());
    }
    Matrix::new(value.rows, value.columns, entries).unwrap()
}

fn sparse(matrix: &Matrix) -> RationalSparseMatrix {
    let values = matrix.to_sparse().unwrap();
    RationalSparseMatrix {
        rows: values.rows(),
        columns: values.cols(),
        entries: values
            .entries()
            .iter()
            .map(|(&(row, column), value)| SparseRationalEntry {
                row,
                column,
                value: match value {
                    Expr::Integer(value) => BigRational::from_integer(value.clone()),
                    _ => panic!("literal integer fixture required"),
                },
            })
            .collect(),
    }
}

fn assert_form(source: &Matrix, expected: &Matrix) {
    let accepted = source.hermite_normal_form().unwrap();
    assert_eq!(dense(accepted.normal_form()), *expected);
    assert_eq!(accepted.rank(), expected.cols());
    assert_eq!(accepted.receipt().rows(), source.rows());
    assert_eq!(accepted.receipt().columns(), source.cols());
    let replay =
        verify_integer_hermite(accepted.source(), accepted.witness(), &mut Unbounded).unwrap();
    assert_eq!(accepted.receipt(), &replay);

    let mut full = vec![Expr::from_i64(0); source.rows() * source.cols()];
    for entry in &accepted.normal_form().entries {
        full[entry.row * source.cols() + source.cols() - expected.cols() + entry.column] =
            Expr::Integer(entry.value.numer().clone());
    }
    let right = dense(accepted.right());
    let inverse = dense(accepted.right_inverse());
    assert_eq!(
        source.matmul(&right).unwrap(),
        Matrix::new(source.rows(), source.cols(), full).unwrap()
    );
    assert_eq!(
        right.matmul(&inverse).unwrap(),
        Matrix::eye(source.cols()).unwrap()
    );
    assert_eq!(
        inverse.matmul(&right).unwrap(),
        Matrix::eye(source.cols()).unwrap()
    );
    let from_sparse = source.to_sparse().unwrap().hermite_normal_form().unwrap();
    assert_eq!(accepted.witness(), from_sparse.witness());
    assert_eq!(accepted.receipt(), from_sparse.receipt());
}

#[test]
fn zero_area_and_rank_zero_forms_have_no_basis_columns() {
    for (rows, columns) in [(0, 0), (0, 5), (4, 0), (3, 5), (5, 3)] {
        assert_form(
            &Matrix::zeros(rows, columns).unwrap(),
            &Matrix::zeros(rows, 0).unwrap(),
        );
    }
    let identity = Matrix::eye(64).unwrap();
    assert_form(&identity, &identity);
}

#[test]
fn fixed_upstream_114_forms_keep_coordinate_lattices_not_just_smith_factors() {
    // These exact expected forms were separately observed with the pinned
    // SymPy 1.14 oracle. Their canonical column convention is part of this API.
    assert_form(
        &matrix(3, 3, &[12, 6, 4, 3, 9, 6, 2, 16, 14]),
        &matrix(3, 3, &[10, 0, 2, 0, 15, 3, 0, 0, 2]),
    );
    assert_form(
        &matrix(2, 3, &[2, 4, 4, 6, 6, 12]),
        &matrix(2, 2, &[2, 0, 0, 6]),
    );
    assert_form(
        &matrix(3, 2, &[2, 6, 4, 6, 4, 12]),
        &matrix(3, 2, &[0, 2, 6, 4, 0, 4]),
    );
    // Unlike Smith form, changing the ambient row coordinates changes HNF.
    let a = matrix(2, 2, &[2, 0, 0, 3]);
    let b = matrix(2, 2, &[2, 0, 2, 3]);
    assert_ne!(
        a.hermite_normal_form().unwrap().normal_form(),
        b.hermite_normal_form().unwrap().normal_form(),
    );
    assert_eq!(
        a.smith_normal_form().unwrap().invariant_factors(),
        b.smith_normal_form().unwrap().invariant_factors(),
    );
}

#[test]
fn rank_deficient_forms_retain_zero_rows_and_all_lattice_directions() {
    assert_form(
        &matrix(3, 3, &[2, 4, 6, 4, 8, 12, 0, 0, 0]),
        &matrix(3, 1, &[2, 4, 0]),
    );
    assert_form(
        &matrix(3, 4, &[0, 2, 4, 8, 0, 0, 0, 0, 0, 6, 12, 18]),
        &matrix(3, 2, &[2, 0, 0, 0, 0, 6]),
    );
    let skipped_row = matrix(3, 2, &[2, 1, 0, 0, 0, 3]);
    assert_form(&skipped_row, &skipped_row);
    assert_form(&matrix(1, 4, &[0, -12, 18, 30]), &matrix(1, 1, &[6]));
    let column = matrix(4, 1, &[0, -12, 18, 30]);
    assert_form(&column, &column);
}

#[test]
fn pivot_sign_and_negative_residue_rounding_follow_euclidean_convention() {
    assert_form(&matrix(1, 1, &[-6]), &matrix(1, 1, &[6]));
    assert_form(&matrix(2, 2, &[3, -1, 0, 2]), &matrix(2, 2, &[3, 2, 0, 2]));
    assert_form(&matrix(2, 2, &[0, 2, 3, -1]), &matrix(2, 2, &[6, 4, 0, 1]));
    assert_form(
        &matrix(2, 2, &[-3, -8, 0, -2]),
        &matrix(2, 2, &[3, 2, 0, 2]),
    );
}

fn bezout(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut old_r, mut r) = (a.abs(), b.abs());
    let (mut old_s, mut s, mut old_t, mut t) = (1, 0, 0, 1);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
    }
    (old_r, old_s * a.signum(), old_t * b.signum())
}

#[test]
fn exhaustive_two_by_two_forms_match_gcd_and_determinant_formula() {
    // An independent closed formula uses the bottom-row image ideal and
    // determinant (lattice index), not the production column reducer.
    for a in -2..=2i64 {
        for b in -2..=2i64 {
            for c in -2..=2i64 {
                for d in -2..=2i64 {
                    let determinant = a * d - b * c;
                    let (g, u, v) = bezout(c, d);
                    let expected = if g == 0 {
                        let (top, _, _) = bezout(a, b);
                        if top == 0 {
                            Matrix::zeros(2, 0).unwrap()
                        } else {
                            matrix(2, 1, &[top, 0])
                        }
                    } else if determinant == 0 {
                        matrix(2, 1, &[a * u + b * v, g])
                    } else {
                        let first = determinant.abs() / g;
                        matrix(2, 2, &[first, (a * u + b * v).rem_euclid(first), 0, g])
                    };
                    let source = matrix(2, 2, &[a, b, c, d]);
                    let accepted = source.hermite_normal_form().unwrap();
                    assert_eq!(
                        dense(accepted.normal_form()),
                        expected,
                        "[{a}, {b}; {c}, {d}]",
                    );
                }
            }
        }
    }
}

#[test]
fn integer_unimodular_column_changes_preserve_the_unique_basis() {
    let expected = matrix(4, 3, &[2, 1, 0, 0, 1, 2, 0, 3, 2, 0, 0, 5]);
    for seed in 0..24i64 {
        let mut values = vec![0i64; 4 * 5];
        for row in 0..4 {
            for (column, value) in [[2, 1, 0], [0, 1, 2], [0, 3, 2], [0, 0, 5]][row]
                .iter()
                .enumerate()
            {
                values[row * 5 + column + 2] = *value;
            }
        }
        for target in 0..5 {
            let source = (target + 2) % 5;
            let multiple = (seed + target as i64 * 3) % 7 - 3;
            for row in 0..4 {
                let original = values[row * 5 + source];
                values[row * 5 + target] += multiple * original;
            }
        }
        for row in 0..4 {
            values.swap(row * 5, row * 5 + 4);
            values[row * 5 + 1] = -values[row * 5 + 1];
        }
        assert_form(&matrix(4, 5, &values), &expected);
    }
}

#[test]
fn coefficients_beyond_binary64_precision_keep_exact_nonnegative_residues() {
    let n = BigInt::one() << 80u32;
    let source = Matrix::new(
        2,
        2,
        vec![
            Expr::Integer(n.clone()),
            Expr::from_i64(-1),
            Expr::from_i64(0),
            Expr::Integer(&n + 1i64),
        ],
    )
    .unwrap();
    let expected = Matrix::new(
        2,
        2,
        vec![
            Expr::Integer(n.clone()),
            Expr::Integer(&n - 1i64),
            Expr::from_i64(0),
            Expr::Integer(&n + 1i64),
        ],
    )
    .unwrap();
    assert_form(&source, &expected);
}

#[test]
fn original_integer_domain_and_bounds_precede_rank_shortcuts() {
    for source in [
        Matrix::new(1, 1, vec![Expr::Sym(Symbol::new("x"))]).unwrap(),
        Matrix::new(
            1,
            1,
            vec![Expr::Rational(BigRational::new(1.into(), 2.into()))],
        )
        .unwrap(),
        Matrix::new(
            1,
            1,
            vec![Expr::Mul(vec![
                Expr::from_i64(0),
                Expr::Sym(Symbol::new("x")),
            ])],
        )
        .unwrap(),
    ] {
        assert!(matches!(
            source.hermite_normal_form(),
            Err(MatrixError::UnsupportedIntegerDomain)
        ));
        assert!(matches!(
            source.to_sparse().unwrap().hermite_normal_form(),
            Err(MatrixError::UnsupportedIntegerDomain)
        ));
    }
    let integral_rational = Matrix::new(
        1,
        1,
        vec![Expr::Rational(BigRational::from_integer((-12).into()))],
    )
    .unwrap();
    assert_form(&integral_rational, &matrix(1, 1, &[12]));
    for (rows, columns) in [(65, 0), (0, 65), (65, 1), (1, 65)] {
        let source = Matrix::zeros(rows, columns).unwrap();
        assert!(matches!(
            source.hermite_normal_form(),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            source.to_sparse().unwrap().hermite_normal_form(),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let huge = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4096u32)]).unwrap();
    assert!(matches!(
        huge.hermite_normal_form(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn source_substitution_lost_directions_and_noncanonical_equivalence_are_rejected() {
    let source = Matrix::eye(2).unwrap();
    let candidate = generate_hermite_normal_form_candidate(&source, &mut Unbounded).unwrap();
    verify_hermite_normal_form_candidate(&source, candidate.clone(), &mut Unbounded).unwrap();

    let mut substituted = candidate.clone();
    substituted.source.entries[0].value = BigRational::from_integer(2.into());
    assert!(matches!(
        verify_hermite_normal_form_candidate(&source, substituted, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));

    let mut omitted = candidate.clone();
    omitted.witness.normal_form = sparse(&matrix(2, 1, &[0, 1]));
    assert!(verify_hermite_normal_form_candidate(&source, omitted, &mut Unbounded).is_err());

    // This full-rank integer basis has the same lattice and valid inverse,
    // but its right residue is not the canonical representative modulo one.
    let mut noncanonical = candidate.clone();
    noncanonical.witness.normal_form = sparse(&matrix(2, 2, &[1, 1, 0, 1]));
    noncanonical.witness.right = noncanonical.witness.normal_form.clone();
    noncanonical.witness.right_inverse = sparse(&matrix(2, 2, &[1, -1, 0, 1]));
    assert!(verify_hermite_normal_form_candidate(&source, noncanonical, &mut Unbounded).is_err());

    // The product identity alone admits the proper sublattice 2Z x Z.
    let mut proper_sublattice = candidate;
    proper_sublattice.witness.normal_form = sparse(&matrix(2, 2, &[2, 0, 0, 1]));
    proper_sublattice.witness.right = proper_sublattice.witness.normal_form.clone();
    assert!(
        verify_hermite_normal_form_candidate(&source, proper_sublattice, &mut Unbounded).is_err()
    );
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
fn cancellation_during_either_phase_and_publication_never_returns_a_form() {
    let source = matrix(3, 3, &[12, 6, 4, 3, 9, 6, 2, 16, 14]);
    for sparse in [false, true] {
        let run = |generation: &mut CancelAfter, verification: &mut CancelAfter| {
            if sparse {
                source
                    .to_sparse()
                    .unwrap()
                    .metered_hermite_normal_form(generation, verification)
            } else {
                source.metered_hermite_normal_form(generation, verification)
            }
        };
        let unlimited = || CancelAfter {
            used: 0,
            stop: usize::MAX,
        };
        let mut generation = unlimited();
        let mut verification = unlimited();
        run(&mut generation, &mut verification).unwrap();
        for stop in [1, 2, generation.used / 2, generation.used] {
            assert!(matches!(
                run(&mut CancelAfter { used: 0, stop }, &mut unlimited()),
                Err(MatrixError::Cancelled)
            ));
        }
        for stop in [1, 2, verification.used / 2, verification.used] {
            assert!(matches!(
                run(&mut unlimited(), &mut CancelAfter { used: 0, stop }),
                Err(MatrixError::Cancelled)
            ));
        }
    }
}

#[test]
fn both_phases_enforce_independent_compute_memory_and_allocation_limits() {
    let source = matrix(2, 2, &[3, -1, 0, 2]);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(
            source.metered_hermite_normal_form(&mut Budget::new(limits), &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            source.metered_hermite_normal_form(&mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            source
                .to_sparse()
                .unwrap()
                .metered_hermite_normal_form(&mut Budget::new(limits), &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            source
                .to_sparse()
                .unwrap()
                .metered_hermite_normal_form(&mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}
