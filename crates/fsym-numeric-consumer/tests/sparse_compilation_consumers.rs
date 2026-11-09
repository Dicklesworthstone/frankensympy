//! Large sparse artifacts reach real numerical consumers without dense programs.

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::EvalError;
use fsym_calculus::verified::{
    MAX_VERIFIED_JACOBIAN_ENTRIES, ResidualEvaluationError, VerifiedCompilationError,
};
use fsym_core::{Expr, Symbol};
use fsym_numeric_consumer::{
    LeastSquaresOptions, SystemConsumerError, sparse_least_squares, sparse_least_squares_with_meter,
};

fn variables(count: usize) -> Vec<Symbol> {
    (0..count)
        .map(|index| Symbol::new(format!("s{index}")))
        .collect()
}

fn target(column: usize) -> i64 {
    (column % 7) as i64 - 3
}

#[derive(Default)]
struct LargestBuffer {
    maximum: u64,
}

impl BudgetMeter for LargestBuffer {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::MemoryBytes {
            self.maximum = self.maximum.max(amount);
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
        Ok(())
    }
}

#[test]
fn all_sparse_shape_limits_can_be_used_together_for_an_inconsistent_fit() {
    let columns = 256;
    let rows = 4096;
    assert!(rows * columns > MAX_VERIFIED_JACOBIAN_ENTRIES);
    let coordinates = variables(columns);
    let residuals: Vec<_> = (0..rows)
        .map(|row| {
            let column = row % columns;
            let offset = if (row / columns).is_multiple_of(2) {
                -1
            } else {
                1
            };
            Expr::Add(vec![
                Expr::Sym(coordinates[column].clone()),
                Expr::from_i64(-target(column) - offset),
            ])
        })
        .collect();
    let program = VerifiedResidualSystem::try_compile(&residuals, &coordinates).unwrap();
    assert_eq!(program.jacobian_pattern().len(), rows);
    assert_eq!(program.residual_programs().len(), rows);
    assert!(program.stored_operation_count() <= 65_536);
    assert!(matches!(
        program.try_dense_program(),
        Err(VerifiedCompilationError::DenseJacobianLimit { .. })
    ));

    let mut meter = LargestBuffer::default();
    let result = sparse_least_squares_with_meter(
        &program,
        &vec![0.0; columns],
        &LeastSquaresOptions::default(),
        &mut meter,
    )
    .unwrap();
    for (column, &point) in result.point.iter().enumerate() {
        assert!((point - target(column) as f64).abs() < 1e-10);
    }
    for (row, &value) in result.residuals.iter().enumerate() {
        let expected = if (row / columns).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        assert!((value - expected).abs() < 1e-10);
    }
    assert!((result.cost - 2048.0).abs() < 1e-8);
    assert!(result.gradient_norm <= 1e-10);
    // Even one dense Jacobian of this shape would require 8 MiB. Measured
    // numerical buffer reservations stay within the vector/nonzero envelope.
    assert!(meter.maximum <= 128 * 1024);
}

#[test]
fn coupled_rank_deficient_sparse_artifacts_replay_and_fit_all_rows() {
    let columns = 256;
    let rows = 512;
    let coordinates = variables(columns);
    let residuals: Vec<_> = (0..rows)
        .map(|row| {
            let column = row % columns;
            let next = (column + 1) % columns;
            let offset = if row < columns { -1 } else { 1 };
            Expr::Add(vec![
                Expr::Sym(coordinates[column].clone()),
                Expr::Sym(coordinates[next].clone()),
                Expr::from_i64(-target(column) - target(next) - offset),
            ])
        })
        .collect();
    let program = VerifiedResidualSystem::try_compile(&residuals, &coordinates).unwrap();
    assert_eq!(program.jacobian_pattern().len(), 2 * rows);
    let bytes = program.to_json_bytes().unwrap();
    let restored = VerifiedResidualSystem::from_json_bytes(&bytes).unwrap();
    assert_eq!(restored.sparse_jacobian(), program.sparse_jacobian());
    assert_eq!(restored.residual_programs(), program.residual_programs());
    assert_eq!(restored.to_json_bytes().unwrap(), bytes);

    let result = sparse_least_squares(
        &restored,
        &vec![0.0; columns],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    // The even cycle has a one-dimensional nullspace. Assert the independently
    // determined adjacent sums and full cost, without inventing uniqueness.
    for column in 0..columns {
        let next = (column + 1) % columns;
        let expected = (target(column) + target(next)) as f64;
        assert!((result.point[column] + result.point[next] - expected).abs() < 1e-9);
    }
    assert!((result.cost - 256.0).abs() < 1e-8);
    assert!(result.gradient_norm <= 1e-10);
}

#[test]
fn empty_sparse_patterns_still_execute_every_original_large_system_source() {
    let coordinates = variables(256);
    let mut residuals = vec![Expr::from_i64(1); 4096];
    let constant = VerifiedResidualSystem::try_compile(&residuals, &coordinates).unwrap();
    assert!(constant.jacobian_pattern().is_empty());
    let result = sparse_least_squares(
        &constant,
        &vec![7.0; 256],
        &LeastSquaresOptions {
            max_iterations: 0,
            ..LeastSquaresOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.point, vec![7.0; 256]);
    assert_eq!(result.cost, 2048.0);
    assert_eq!(result.gradient_norm, 0.0);

    residuals[4095] = Expr::Function("log".into(), vec![Expr::from_i64(-1)]);
    let invalid = VerifiedResidualSystem::try_compile(&residuals, &coordinates).unwrap();
    assert!(invalid.jacobian_pattern().is_empty());
    assert!(matches!(
        sparse_least_squares(&invalid, &vec![7.0; 256], &LeastSquaresOptions::default()),
        Err(SystemConsumerError::SparseEvaluation(
            ResidualEvaluationError::Evaluation(EvalError::NonFiniteValue)
        ))
    ));
}
