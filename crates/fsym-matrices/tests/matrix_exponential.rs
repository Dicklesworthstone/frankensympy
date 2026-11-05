//! Exact matrix exponential consumers and cumulative publication boundaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol, parse};
use fsym_matrices::{Matrix, MatrixExponentialError, SparseMatrix};
use fsym_solvers::ode::system::general::{GeneralOdeSystemError, general_ode_system};
use std::collections::BTreeMap;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}
fn e(source: &str) -> Expr {
    parse(source).unwrap()
}
fn matrix(rows: &[&[i64]]) -> Matrix {
    Matrix::new(
        rows.len(),
        rows.first().map_or(0, |row| row.len()),
        rows.iter()
            .flat_map(|row| row.iter().map(|&value| Expr::from_i64(value)))
            .collect(),
    )
    .unwrap()
}
fn assert_equal(actual: &Expr, expected: &Expr) {
    let difference = actual.clone() - expected.clone();
    let expanded = fsym_simplify::try_expand(&difference).unwrap();
    let residual = fsym_simplify::try_simplify(&expanded).unwrap();
    assert!(
        residual.is_zero(),
        "{actual} != {expected}; residual {residual}"
    );
}

#[test]
fn nilpotent_chain_exponential_has_exact_factorial_entries() {
    let source = matrix(&[&[0, 1, 0], &[0, 0, 1], &[0, 0, 0]]);
    let actual = source.exp().unwrap();
    let expected = Matrix::new(
        3,
        3,
        vec![
            e("1"),
            e("1"),
            e("1/2"),
            e("0"),
            e("1"),
            e("1"),
            e("0"),
            e("0"),
            e("1"),
        ],
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(source.get(0, 2).unwrap(), &e("0"));
}

#[test]
fn empty_scalar_and_diagonal_exponentials_keep_literal_identity_values() {
    assert_eq!(
        Matrix::zeros(0, 0).unwrap().exp().unwrap(),
        Matrix::eye(0).unwrap()
    );
    assert_eq!(matrix(&[&[0]]).exp().unwrap(), matrix(&[&[1]]));
    assert_eq!(matrix(&[&[1]]).exp().unwrap().get(0, 0).unwrap(), &e("E"));
    let source = matrix(&[&[0, 0, 0], &[0, 1, 0], &[0, 0, -2]]);
    let actual = source.exp().unwrap();
    for (row, expected) in [e("1"), e("E"), e("exp(-2)")].iter().enumerate() {
        assert_equal(actual.get(row, row).unwrap(), expected);
    }
    assert_eq!(actual.get(0, 2).unwrap(), &e("0"));
}

#[test]
fn dense_oscillatory_similarity_is_evaluated_from_the_complete_matrix() {
    // S = [[1,1,1],[1,2,1],[1,1,2]], A = S diag(rotation,2) S^-1.
    let source = matrix(&[&[-6, 2, 3], &[-9, 3, 4], &[-8, 2, 5]]);
    let actual = source.exp().unwrap();
    let initial = [e("2"), e("2"), e("3")];
    let expected = [
        e("cos(1)-sin(1)+exp(2)"),
        e("cos(1)-2*sin(1)+exp(2)"),
        e("cos(1)-sin(1)+2*exp(2)"),
    ];
    for (row, wanted) in expected.iter().enumerate() {
        let value = initial
            .iter()
            .enumerate()
            .fold(e("0"), |sum, (column, initial)| {
                sum + actual.get(row, column).unwrap().clone() * initial.clone()
            });
        assert_equal(&value, wanted);
    }
}

#[test]
fn hyperbolic_normalization_and_shifted_jordan_families_are_exact() {
    let source = matrix(&[&[0, 1, 0], &[2, 0, 0], &[0, 0, 1]]);
    let actual = source.exp().unwrap();
    assert_equal(actual.get(0, 0).unwrap(), &e("cosh(sqrt(2))"));
    assert_equal(actual.get(0, 1).unwrap(), &e("sinh(sqrt(2))/sqrt(2)"));
    assert_equal(actual.get(1, 0).unwrap(), &e("sqrt(2)*sinh(sqrt(2))"));
    let source = matrix(&[&[1, 1, 0], &[0, 1, 1], &[0, 0, 1]]);
    let actual = source.exp().unwrap();
    assert_equal(actual.get(0, 0).unwrap(), &e("E"));
    assert_equal(actual.get(0, 1).unwrap(), &e("E"));
    assert_equal(actual.get(0, 2).unwrap(), &e("E/2"));
}

#[test]
fn sparse_exponential_handles_fill_and_omits_exact_zero_entries() {
    let source =
        SparseMatrix::new(3, 3, BTreeMap::from([((0, 1), e("1")), ((1, 2), e("1"))])).unwrap();
    let actual = source.exp().unwrap();
    assert_eq!(source.nnz(), 2);
    assert_eq!(actual.nnz(), 6);
    assert_eq!(actual.get(0, 2).unwrap(), e("1/2"));
    assert_eq!(actual.get(2, 0).unwrap(), e("0"));
    assert_eq!(
        actual.to_dense().unwrap(),
        source.to_dense().unwrap().exp().unwrap()
    );
    assert_eq!(SparseMatrix::zeros(0, 0).exp().unwrap().nnz(), 0);
    assert_eq!(
        SparseMatrix::zeros(3, 3).exp().unwrap(),
        SparseMatrix::eye(3).unwrap()
    );
}

#[test]
fn accepted_receipt_evaluation_uses_absolute_time_and_original_normalization() {
    let coefficients = vec![vec![q(1, 1), q(0, 1)], vec![q(0, 1), q(0, 1)]];
    let source = general_ode_system(
        &coefficients,
        &Symbol::new("t"),
        &q(2, 1),
        &[vec![q(0, 1)], vec![q(0, 1)]],
        &[e("0"), e("0")],
    )
    .unwrap();
    let at_initial = source.fundamental_at(&q(2, 1), &mut Unbounded).unwrap();
    assert_eq!(at_initial, vec![vec![e("1"), e("0")], vec![e("0"), e("1")]]);
    let at_three = source.fundamental_at(&q(3, 1), &mut Unbounded).unwrap();
    assert_eq!(at_three, vec![vec![e("E"), e("0")], vec![e("0"), e("1")]]);
    let huge = BigRational::from_integer(BigInt::from(1) << 1024usize);
    assert!(matches!(
        source.fundamental_at(&huge, &mut Unbounded),
        Err(GeneralOdeSystemError::ResourceLimit(_))
    ));
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        source.fundamental_at(&q(3, 1), &mut empty),
        Err(GeneralOdeSystemError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn nonliteral_and_original_undefined_cells_cannot_enter_the_checked_lane() {
    let held = Expr::Mul(vec![e("0"), Expr::Function("log".into(), vec![e("-1")])]);
    for value in [e("x"), e("sqrt(2)"), held] {
        let source = Matrix::new(1, 1, vec![value]).unwrap();
        assert!(matches!(
            source.exp(),
            Err(MatrixExponentialError::UnsupportedDomain)
        ));
    }
    let oversized = rational_expr(BigRational::from_integer(BigInt::from(1) << 1024usize));
    let source = Matrix::new(1, 1, vec![oversized]).unwrap();
    assert!(matches!(
        source.exp(),
        Err(MatrixExponentialError::ResourceLimit(_))
    ));
}

#[test]
fn square_and_dimension_bounds_apply_before_sparse_fill() {
    assert!(matches!(
        Matrix::zeros(0, 3).unwrap().exp(),
        Err(MatrixExponentialError::NotSquare(0, 3))
    ));
    assert!(matches!(
        Matrix::eye(9).unwrap().exp(),
        Err(MatrixExponentialError::ResourceLimit(_))
    ));
    let huge = SparseMatrix::zeros(1_000_000, 1_000_000);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    assert!(matches!(
        huge.metered_exp(&mut generation, &mut verification),
        Err(MatrixExponentialError::ResourceLimit(_))
    ));
    assert_eq!(generation.charges, [0; 3]);
    assert_eq!(verification.checkpoints, 0);
    assert_eq!(verification.charges, [0; 3]);
}

#[test]
fn unsupported_irreducible_spectra_do_not_publish_an_exponential() {
    let source = matrix(&[&[0, 1, 0], &[0, 0, 1], &[2, 0, 0]]);
    let error = source.exp().unwrap_err();
    assert!(error.is_unsupported(), "{error}");
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    stopped: bool,
    charges: [u64; 3],
}
impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after cancellation");
        let index = match dimension {
            Dimension::ComputeSteps => Some(0),
            Dimension::MemoryBytes => Some(1),
            Dimension::AllocationCount => Some(2),
            _ => None,
        };
        if let Some(index) = index {
            self.charges[index] += amount;
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
        assert!(!self.stopped, "checked after cancellation");
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn all_cumulative_consumer_phases_refuse_exhaustion_and_final_cancellation() {
    let source = matrix(&[&[0, 1, 0], &[0, 0, 1], &[0, 0, 0]]);
    let (mut generator, mut verifier) = (Meter::default(), Meter::default());
    source.metered_exp(&mut generator, &mut verifier).unwrap();
    for (in_generation, measured) in [(true, &generator), (false, &verifier)] {
        assert!(measured.charges.iter().all(|&value| value > 0));
        for stop in [
            1,
            measured.checkpoints / 2,
            measured.checkpoints - 1,
            measured.checkpoints,
        ] {
            let mut meter = Meter {
                cancel_at: Some(stop),
                ..Meter::default()
            };
            let result = if in_generation {
                source.metered_exp(&mut meter, &mut Unbounded)
            } else {
                source.metered_exp(&mut Unbounded, &mut meter)
            };
            assert!(matches!(
                result,
                Err(MatrixExponentialError::Meter(MeterError::Cancelled))
            ));
            assert_eq!(meter.checkpoints, stop);
        }
        for (index, dimension) in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ]
        .into_iter()
        .enumerate()
        {
            let mut limits = BudgetLimits::uniform(u64::MAX, 0);
            limits.dimensions[dimension.index()] = measured.charges[index] - 1;
            let mut budget = Budget::new(limits);
            let result = if in_generation {
                source.metered_exp(&mut budget, &mut Unbounded)
            } else {
                source.metered_exp(&mut Unbounded, &mut budget)
            };
            assert!(matches!(
                result,
                Err(MatrixExponentialError::Meter(MeterError::Budget(_)))
            ));
        }
    }
}
