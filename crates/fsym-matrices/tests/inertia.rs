//! Exact inertia across scalar/paired pivots, singularity, sources and resources.
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_matrices::{Matrix, MatrixError};
use fsym_proof_kernel::rational_inertia::verify_rational_inertia;

fn matrix(n: usize, values: &[i64]) -> Matrix {
    Matrix::new(n, n, values.iter().copied().map(Expr::from_i64).collect()).unwrap()
}
fn counts(a: &Matrix) -> (usize, usize, usize) {
    let result = a.inertia().unwrap();
    let r = result.receipt();
    let replay =
        verify_rational_inertia(result.source(), result.witness(), &mut Unbounded).unwrap();
    assert_eq!(r, &replay);
    (r.positive(), r.negative(), r.zero())
}
#[test]
fn empty_zero_identity_singular_and_indefinite_forms() {
    assert_eq!(counts(&matrix(0, &[])), (0, 0, 0));
    assert_eq!(counts(&Matrix::zeros(4, 4).unwrap()), (0, 0, 4));
    assert_eq!(counts(&Matrix::eye(64).unwrap()), (64, 0, 0));
    assert_eq!(counts(&matrix(2, &[1, 2, 2, 4])), (1, 0, 1));
    assert_eq!(counts(&matrix(2, &[-1, -2, -2, -4])), (0, 1, 1));
    assert_eq!(counts(&matrix(3, &[0, 0, 1, 0, 0, 0, 1, 0, 0])), (1, 1, 1));
    // A 2x2 pivot is needed only after the first scalar Schur complement.
    assert_eq!(counts(&matrix(3, &[1, 1, 2, 1, 1, 1, 2, 1, 4])), (2, 1, 0));
}
#[test]
fn nonsymmetric_forms_are_not_classified_from_their_eigenvalues() {
    assert_eq!(counts(&matrix(2, &[1, 100, -100, 1])), (2, 0, 0));
    // Both eigenvalues of A are +1, but its quadratic form is indefinite.
    assert_eq!(counts(&matrix(2, &[1, 4, 0, 1])), (1, 1, 0));
    assert_eq!(counts(&matrix(2, &[0, 7, -7, 0])), (0, 0, 2));
}
#[test]
fn exhaustive_two_by_two_rational_forms_match_trace_determinant_classification() {
    for a in -2i64..=2 {
        for b in -2i64..=2 {
            for c in -2i64..=2 {
                for d in -2i64..=2 {
                    let determinant = 4 * a * d - (b + c) * (b + c);
                    let trace = a + d;
                    let expected = if determinant < 0 {
                        (1, 1, 0)
                    } else if determinant > 0 {
                        if trace > 0 { (2, 0, 0) } else { (0, 2, 0) }
                    } else if trace > 0 {
                        (1, 0, 1)
                    } else if trace < 0 {
                        (0, 1, 1)
                    } else {
                        (0, 0, 2)
                    };
                    assert_eq!(
                        counts(&matrix(2, &[a, b, c, d])),
                        expected,
                        "[{a},{b};{c},{d}]"
                    );
                }
            }
        }
    }
}
#[test]
fn known_congruences_retain_every_sign_after_permutations() {
    for seed in 0..16i64 {
        let n = 6;
        let signs = [1i64, -2, 0, 3, 0, -4];
        let mut lower = vec![0i64; n * n];
        for row in 0..n {
            for column in 0..=row {
                lower[row * n + column] = if row == column {
                    1
                } else {
                    (seed + row as i64 * 3 + column as i64) % 5 - 2
                };
            }
        }
        let mut source = vec![0i64; n * n];
        for row in 0..n {
            for column in 0..n {
                let value = (0..n)
                    .map(|i| lower[row * n + i] * signs[i] * lower[column * n + i])
                    .sum::<i64>();
                // Reversing coordinates forces pivots away from their original order.
                source[(n - row - 1) * n + n - column - 1] = value;
            }
        }
        let a = matrix(n, &source);
        assert_eq!(counts(&a), (2, 2, 2));
        let sparse = a.to_sparse().unwrap().inertia().unwrap();
        assert_eq!(
            (
                sparse.receipt().positive(),
                sparse.receipt().negative(),
                sparse.receipt().zero()
            ),
            (2, 2, 2)
        );
    }
}
#[test]
fn tiny_exact_positive_pivot_survives_binary64_rounding() {
    let denominator = BigInt::one() << 1024u32;
    let last = BigRational::new(&denominator + 1i64, denominator);
    let a = Matrix::new(
        2,
        2,
        vec![
            Expr::from_i64(1),
            Expr::from_i64(1),
            Expr::from_i64(1),
            Expr::Rational(last),
        ],
    )
    .unwrap();
    assert_eq!(counts(&a), (2, 0, 0));
}
#[test]
fn original_sources_shapes_and_height_are_admitted_before_shortcuts() {
    assert!(matches!(
        Matrix::zeros(0, 1).unwrap().inertia(),
        Err(MatrixError::NotSquare(0, 1))
    ));
    assert!(matches!(
        Matrix::eye(65).unwrap().inertia(),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        Matrix::eye(65).unwrap().to_sparse().unwrap().inertia(),
        Err(MatrixError::ResourceLimit(_))
    ));
    let unsupported = Matrix::new(
        2,
        2,
        vec![
            Expr::from_i64(0),
            Expr::Sym(Symbol::new("x")),
            Expr::from_i64(0),
            Expr::from_i64(0),
        ],
    )
    .unwrap();
    assert!(matches!(
        unsupported.inertia(),
        Err(MatrixError::UnsupportedCertificateDomain)
    ));
    let huge = Matrix::new(1, 1, vec![Expr::Integer(BigInt::one() << 4096u32)]).unwrap();
    assert!(matches!(huge.inertia(), Err(MatrixError::ResourceLimit(_))));
}
#[test]
fn certificate_mutations_cannot_relabel_a_quadratic_form() {
    let a = matrix(3, &[1, 1, 2, 1, 1, 1, 2, 1, 4]);
    let candidate = a.inertia_candidate_with_meter(&mut Unbounded).unwrap();
    let mut changed = candidate.witness.clone();
    changed.diagonal.entries[0].value = -changed.diagonal.entries[0].value.clone();
    assert!(verify_rational_inertia(&candidate.source, &changed, &mut Unbounded).is_err());
    let mut source = candidate.source.clone();
    source.entries[0].value = BigRational::from_integer(5.into());
    assert!(verify_rational_inertia(&source, &candidate.witness, &mut Unbounded).is_err());
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
fn stopped_generators_and_verifiers_do_not_publish_accepted_inertia() {
    let a = matrix(3, &[1, 1, 2, 1, 1, 1, 2, 1, 4]);
    let mut generator = CancelAfter {
        used: 0,
        stop: usize::MAX,
    };
    a.inertia_candidate_with_meter(&mut generator).unwrap();
    for stop in [1, 2, 100, generator.used] {
        assert!(matches!(
            a.inertia_with_meters(&mut CancelAfter { used: 0, stop }, &mut Unbounded),
            Err(MatrixError::Cancelled)
        ));
    }
    assert!(matches!(
        a.inertia_with_meters(&mut Unbounded, &mut CancelAfter { used: 0, stop: 1 }),
        Err(MatrixError::Cancelled)
    ));
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(matches!(
            a.inertia_with_meters(&mut Budget::new(limits.clone()), &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert!(matches!(
            a.inertia_with_meters(&mut Unbounded, &mut Budget::new(limits)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}
