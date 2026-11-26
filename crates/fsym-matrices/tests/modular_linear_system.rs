//! Public modular solving: exhaustive finite oracles, complete families,
//! original-source binding, and cumulative interruption/resource behavior.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_matrices::{
    Matrix, MatrixError, VerifiedIntegerModularSystem, generate_integer_modular_system_candidate,
    metered_solve_mod, verify_integer_modular_system_candidate,
};
use fsym_proof_kernel::integer_modular_system::{
    IntegerModularSystemOutcome, verify_integer_modular_system,
};
use fsym_proof_kernel::sparse_rref::RationalSparseMatrix;
use std::collections::BTreeSet;

fn matrix(rows: usize, columns: usize, values: &[i64]) -> Matrix {
    Matrix::new(
        rows,
        columns,
        values.iter().copied().map(Expr::from_i64).collect(),
    )
    .unwrap()
}

fn cell(matrix: &RationalSparseMatrix, row: usize, column: usize) -> i64 {
    matrix
        .entries
        .iter()
        .find(|entry| (entry.row, entry.column) == (row, column))
        .map_or(0, |entry| entry.value.numer().to_i64().unwrap())
}

fn original_cell(matrix: &Matrix, row: usize, column: usize) -> i64 {
    match matrix.get(row, column).unwrap() {
        Expr::Integer(value) => value.to_i64().unwrap(),
        Expr::Rational(value) if value.is_integer() => value.numer().to_i64().unwrap(),
        _ => panic!("literal integer expected"),
    }
}

/// Enumerate every residue matrix directly, without using Smith factors,
/// divisibility, modular inversion, or the claimed parameterization.
fn brute_force(source: &Matrix, rhs: &Matrix, modulus: i64) -> BTreeSet<Vec<i64>> {
    let count = modulus.pow((source.cols() * rhs.cols()) as u32);
    let mut solutions = BTreeSet::new();
    for mut index in 0..count {
        let values: Vec<_> = (0..source.cols() * rhs.cols())
            .map(|_| {
                let value = index % modulus;
                index /= modulus;
                value
            })
            .collect();
        if (0..source.rows()).all(|row| {
            (0..rhs.cols()).all(|column| {
                let sum: i64 = (0..source.cols())
                    .map(|inner| {
                        original_cell(source, row, inner) * values[inner * rhs.cols() + column]
                    })
                    .sum();
                (sum - original_cell(rhs, row, column)).rem_euclid(modulus) == 0
            })
        }) {
            solutions.insert(values);
        }
    }
    solutions
}

/// Enumerate the advertised independent finite parameter range for every RHS.
/// The insertion assertion tests injectivity, not just validity of residuals.
fn family(result: &VerifiedIntegerModularSystem, modulus: i64) -> BTreeSet<Vec<i64>> {
    let Some(particular) = result.particular() else {
        assert!(result.generators().is_none());
        assert!(result.periods().is_none());
        return BTreeSet::new();
    };
    let generators = result.generators().unwrap();
    let periods: Vec<_> = result
        .periods()
        .unwrap()
        .iter()
        .map(|period| period.to_i64().unwrap())
        .collect();
    assert_eq!(
        (generators.rows, generators.columns),
        (particular.rows, periods.len())
    );
    for period in &periods {
        assert!(*period >= 2 && *period <= modulus && modulus % period == 0);
    }
    for value in particular.entries.iter().chain(&generators.entries) {
        assert!(value.value.is_integer());
        let integer = value.value.numer().to_i64().unwrap();
        assert!((0..modulus).contains(&integer));
    }
    let count = periods
        .iter()
        .product::<i64>()
        .pow(particular.columns as u32);
    let mut values = BTreeSet::new();
    for mut index in 0..count {
        let mut solution: Vec<_> = (0..particular.rows)
            .flat_map(|row| {
                (0..particular.columns).map(move |column| cell(particular, row, column))
            })
            .collect();
        for rhs_column in 0..particular.columns {
            for (direction, period) in periods.iter().enumerate() {
                let parameter = index % period;
                index /= period;
                for row in 0..particular.rows {
                    let at = row * particular.columns + rhs_column;
                    solution[at] = (solution[at] + parameter * cell(generators, row, direction))
                        .rem_euclid(modulus);
                }
            }
        }
        assert!(
            values.insert(solution),
            "finite parameterization must be injective"
        );
    }
    assert_eq!(values.len() as i64, count);
    values
}

fn assert_complete(
    source: &Matrix,
    rhs: &Matrix,
    modulus: i64,
    result: &VerifiedIntegerModularSystem,
) -> usize {
    let expected = brute_force(source, rhs, modulus);
    assert_eq!(family(result, modulus), expected);
    assert_eq!(result.is_consistent(), !expected.is_empty());
    assert_eq!(result.is_unique(), expected.len() == 1);
    assert_eq!(result.contradiction().is_some(), expected.is_empty());
    assert_eq!(result.modulus(), &BigInt::from(modulus));
    assert_eq!(
        (
            result.receipt().rows(),
            result.receipt().columns(),
            result.receipt().rhs_columns()
        ),
        (source.rows(), source.cols(), rhs.cols()),
    );
    expected.len()
}

#[test]
fn composite_prime_power_signed_and_rectangular_families_match_all_residues() {
    let fixtures = [
        (matrix(2, 2, &[2, 4, 0, 6]), matrix(2, 1, &[4, 6]), 12, 12),
        (matrix(1, 1, &[6]), matrix(1, 1, &[0]), 6, 6),
        (matrix(1, 1, &[-3]), matrix(1, 1, &[9]), 12, 3),
        (matrix(1, 1, &[3]), matrix(1, 1, &[6]), 9, 3),
        (matrix(1, 1, &[5]), matrix(1, 1, &[10]), 25, 5),
        (matrix(2, 2, &[2, 0, 0, 4]), matrix(2, 1, &[2, 4]), 8, 8),
        (matrix(2, 2, &[2, 3, 3, 2]), matrix(2, 1, &[1, 5]), 6, 1),
        (matrix(1, 2, &[2, 0]), matrix(1, 1, &[4]), 6, 12),
        (matrix(2, 1, &[2, 4]), matrix(2, 1, &[4, 2]), 6, 2),
        (matrix(2, 1, &[2, 4]), matrix(2, 1, &[4, 3]), 6, 0),
        (matrix(2, 2, &[0, 0, 0, 0]), matrix(2, 1, &[0, 0]), 4, 16),
        (matrix(2, 2, &[2, 4, 4, 8]), matrix(2, 1, &[2, 5]), 6, 0),
        (matrix(1, 3, &[1, 2, 3]), matrix(1, 1, &[4]), 7, 49),
    ];
    for (source, rhs, modulus, expected_count) in fixtures {
        let original_source = source.clone();
        let original_rhs = rhs.clone();
        let result = source.solve_mod(&rhs, &modulus.into()).unwrap();
        assert_eq!(
            assert_complete(&source, &rhs, modulus, &result),
            expected_count
        );
        let replay = verify_integer_modular_system(
            result.source(),
            result.rhs(),
            result.modulus(),
            result.witness(),
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(&replay, result.receipt());

        let sparse_source = source.to_sparse().unwrap();
        let sparse_rhs = rhs.to_sparse().unwrap();
        let sparse = sparse_source
            .solve_mod(&sparse_rhs, &modulus.into())
            .unwrap();
        assert_eq!(sparse.witness(), result.witness());
        assert_eq!(sparse.receipt(), result.receipt());
        let mixed = metered_solve_mod(
            (&source).into(),
            (&sparse_rhs).into(),
            &modulus.into(),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(mixed.witness(), result.witness());
        assert_eq!(mixed.receipt(), result.receipt());
        assert_eq!(source, original_source);
        assert_eq!(rhs, original_rhs);
    }
}

#[test]
fn exhaustive_small_systems_check_completeness_and_uniqueness_without_smith_oracles() {
    let mut checked = 0;
    // Every 2x2 matrix with entries -1,0,1 and every RHS modulo2.
    for a in -1..=1 {
        for b in -1..=1 {
            for c in -1..=1 {
                for d in -1..=1 {
                    let source = matrix(2, 2, &[a, b, c, d]);
                    for first in 0..2 {
                        for second in 0..2 {
                            let rhs = matrix(2, 1, &[first, second]);
                            let result = source.solve_mod(&rhs, &2.into()).unwrap();
                            assert_complete(&source, &rhs, 2, &result);
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    // Every one-row matrix with entries -2..=2 and every RHS for these
    // moduli; nonunit pivots and rank-zero matrices are included.
    for a in -2..=2 {
        for b in -2..=2 {
            let source = matrix(1, 2, &[a, b]);
            for modulus in [2, 4, 6] {
                for value in 0..modulus {
                    let rhs = matrix(1, 1, &[value]);
                    let result = source.solve_mod(&rhs, &modulus.into()).unwrap();
                    assert_complete(&source, &rhs, modulus, &result);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 624);
}

#[test]
fn every_rhs_has_independent_parameters_and_a_later_contradiction_refuses_the_system() {
    let source = matrix(2, 2, &[2, 4, 0, 6]);
    let rhs = matrix(2, 2, &[4, 2, 6, 0]);
    let result = source.solve_mod(&rhs, &12.into()).unwrap();
    assert_eq!(assert_complete(&source, &rhs, 12, &result), 144);
    let single = source
        .solve_mod(&matrix(2, 1, &[4, 6]), &12.into())
        .unwrap();
    assert_eq!(result.generators(), single.generators());
    assert_eq!(result.periods(), single.periods());
    let bad_rhs = matrix(2, 2, &[4, 5, 6, 0]);
    let refused = source.solve_mod(&bad_rhs, &12.into()).unwrap();
    assert_eq!(assert_complete(&source, &bad_rhs, 12, &refused), 0);
    assert_eq!(refused.contradiction().unwrap().1, 1);
}

#[test]
fn zero_shapes_distinguish_unique_empty_matrices_from_free_residue_vectors() {
    let fixtures = [
        (matrix(0, 2, &[]), matrix(0, 1, &[]), 36),
        (matrix(2, 0, &[]), matrix(2, 1, &[0, 0]), 1),
        (matrix(2, 0, &[]), matrix(2, 1, &[0, 1]), 0),
        (matrix(0, 0, &[]), matrix(0, 2, &[]), 1),
        (matrix(1, 2, &[2, 4]), matrix(1, 0, &[]), 1),
        (matrix(0, 2, &[]), matrix(0, 0, &[]), 1),
    ];
    for (source, rhs, count) in fixtures {
        let result = source.solve_mod(&rhs, &6.into()).unwrap();
        assert_eq!(assert_complete(&source, &rhs, 6, &result), count);
        if rhs.cols() == 0 && source.cols() > 0 {
            assert!(result.is_unique());
            assert!(!result.periods().unwrap().is_empty());
            assert_eq!(result.particular().unwrap().columns, 0);
        }
        let sparse = source
            .to_sparse()
            .unwrap()
            .solve_mod(&rhs.to_sparse().unwrap(), &6.into())
            .unwrap();
        assert_eq!(sparse.witness(), result.witness());
    }
}

#[test]
fn large_composite_modulus_and_negative_rhs_retain_exact_nonunit_solutions() {
    let modulus = BigInt::one() << 256u32;
    let step = BigInt::one() << 255u32;
    let source = matrix(1, 1, &[6]);
    let rhs = matrix(1, 1, &[-12]);
    let result = source.solve_mod(&rhs, &modulus).unwrap();
    assert!(!result.is_unique());
    assert_eq!(result.periods(), Some([BigInt::from(2)].as_slice()));
    assert_eq!(
        result.particular().unwrap().entries[0].value.numer(),
        &(&step - 2i64)
    );
    assert_eq!(result.generators().unwrap().entries[0].value.numer(), &step);
    let (source, rhs, witness, receipt) = result.into_parts();
    assert_eq!(
        verify_integer_modular_system(&source, &rhs, &modulus, &witness, &mut Unbounded).unwrap(),
        receipt,
    );
}

#[test]
fn exact_original_matrices_and_modulus_cannot_be_rebound_even_to_congruent_inputs() {
    let source = matrix(2, 2, &[2, 4, 0, 6]);
    let rhs = matrix(2, 1, &[4, 6]);
    let candidate = generate_integer_modular_system_candidate(
        (&source).into(),
        (&rhs).into(),
        &12.into(),
        &mut Unbounded,
    )
    .unwrap();
    for (original, original_rhs, modulus) in [
        (matrix(2, 2, &[14, 4, 0, 6]), rhs.clone(), 12),
        (source.clone(), matrix(2, 1, &[16, 6]), 12),
        (source.clone(), rhs.clone(), 6),
    ] {
        assert!(matches!(
            verify_integer_modular_system_candidate(
                (&original).into(),
                (&original_rhs).into(),
                &modulus.into(),
                candidate.clone(),
                &mut Unbounded,
            ),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
    let mut altered = Vec::new();
    let mut bad = candidate.clone();
    bad.source.entries[0].value = BigRational::from_integer(14.into());
    altered.push(bad);
    let mut bad = candidate.clone();
    bad.rhs.entries[0].value = BigRational::from_integer(16.into());
    altered.push(bad);
    let mut bad = candidate.clone();
    bad.modulus = 24.into();
    altered.push(bad);
    let mut bad = candidate.clone();
    bad.witness.schema_version += 1;
    altered.push(bad);
    let mut bad = candidate.clone();
    bad.witness.outcome = IntegerModularSystemOutcome::Inconsistent {
        row: 0,
        rhs_column: 0,
    };
    altered.push(bad);
    let mut bad = candidate.clone();
    let IntegerModularSystemOutcome::Solvable {
        generators,
        periods,
        ..
    } = &mut bad.witness.outcome
    else {
        panic!("consistent fixture");
    };
    // A proper subfamily can retain valid residuals, but cannot be accepted
    // as the complete set of modular solutions.
    generators.columns -= 1;
    generators
        .entries
        .retain(|entry| entry.column < generators.columns);
    periods.pop();
    altered.push(bad);
    for bad in altered {
        assert!(matches!(
            verify_integer_modular_system_candidate(
                (&source).into(),
                (&rhs).into(),
                &12.into(),
                bad,
                &mut Unbounded,
            ),
            Err(MatrixError::InvalidCertificate(_))
        ));
    }
    let accepted = verify_integer_modular_system_candidate(
        (&source).into(),
        (&rhs).into(),
        &12.into(),
        candidate,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(assert_complete(&source, &rhs, 12, &accepted), 12);
}

#[test]
fn every_original_domain_is_admitted_before_zero_rhs_or_inconsistency_shortcuts() {
    for value in [
        Expr::Sym(Symbol::new("x")),
        Expr::Rational(BigRational::new(1.into(), 2.into())),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(Symbol::new("x"))]),
    ] {
        let bad_source = Matrix::new(
            2,
            2,
            vec![
                Expr::from_i64(0),
                Expr::from_i64(0),
                Expr::from_i64(0),
                value.clone(),
            ],
        )
        .unwrap();
        assert!(matches!(
            bad_source.solve_mod(&matrix(2, 0, &[]), &6.into()),
            Err(MatrixError::UnsupportedIntegerDomain)
        ));
        let bad_rhs = Matrix::new(
            2,
            2,
            vec![
                Expr::from_i64(1),
                Expr::from_i64(0),
                Expr::from_i64(0),
                value,
            ],
        )
        .unwrap();
        assert!(matches!(
            Matrix::zeros(2, 2).unwrap().solve_mod(&bad_rhs, &6.into()),
            Err(MatrixError::UnsupportedIntegerDomain)
        ));
    }
}

#[test]
fn modulus_shape_and_coefficient_limits_are_checked_before_generation() {
    let source = matrix(1, 1, &[2]);
    let rhs = matrix(1, 1, &[0]);
    for modulus in [-6, 0, 1] {
        assert!(matches!(
            source.solve_mod(&rhs, &modulus.into()),
            Err(MatrixError::InvalidModulus(_))
        ));
    }
    let large = BigInt::one() << 4096u32;
    assert!(matches!(
        source.solve_mod(&rhs, &large),
        Err(MatrixError::ResourceLimit(_))
    ));
    let oversized = Matrix::new(1, 1, vec![Expr::Integer(large)]).unwrap();
    assert!(matches!(
        oversized.solve_mod(&rhs, &6.into()),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert!(matches!(
        source.solve_mod(&oversized, &6.into()),
        Err(MatrixError::ResourceLimit(_))
    ));
    for (source, rhs) in [
        (matrix(0, 65, &[]), matrix(0, 1, &[])),
        (matrix(0, 0, &[]), matrix(0, 65, &[])),
        (Matrix::zeros(65, 0).unwrap(), Matrix::zeros(65, 1).unwrap()),
    ] {
        assert!(matches!(
            source.solve_mod(&rhs, &6.into()),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    assert!(matches!(
        source.solve_mod(&matrix(0, 1, &[]), &6.into()),
        Err(MatrixError::ShapeMismatch(_, _, _, _))
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
fn every_generation_verification_and_final_publication_checkpoint_is_terminal() {
    let source = matrix(1, 1, &[2]);
    for value in [1, 2] {
        let rhs = matrix(1, 1, &[value]);
        let mut generator = SafePoints::default();
        let mut verifier = SafePoints::default();
        metered_solve_mod(
            (&source).into(),
            (&rhs).into(),
            &6.into(),
            &mut generator,
            &mut verifier,
        )
        .unwrap();
        assert!(generator.seen > 20 && verifier.seen > 20);
        for point in 1..=generator.seen {
            let mut stopped = SafePoints {
                seen: 0,
                cancel_at: Some(point),
            };
            assert!(matches!(
                metered_solve_mod(
                    (&source).into(),
                    (&rhs).into(),
                    &6.into(),
                    &mut stopped,
                    &mut Unbounded,
                ),
                Err(MatrixError::Cancelled)
            ));
            assert_eq!(stopped.seen, point);
        }
        for point in 1..=verifier.seen {
            let mut stopped = SafePoints {
                seen: 0,
                cancel_at: Some(point),
            };
            assert!(matches!(
                metered_solve_mod(
                    (&source).into(),
                    (&rhs).into(),
                    &6.into(),
                    &mut Unbounded,
                    &mut stopped,
                ),
                Err(MatrixError::Cancelled)
            ));
            assert_eq!(stopped.seen, point);
        }
    }
}

#[test]
fn both_phases_enforce_exact_fit_one_short_and_protected_cumulative_allowances() {
    let source = matrix(2, 2, &[2, 4, 0, 6]);
    let rhs = matrix(2, 1, &[4, 6]);
    let initial = 100_000_000;
    let dimensions = [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ];
    let mut generator = Budget::new(BudgetLimits::uniform(initial, 43));
    let mut verifier = Budget::new(BudgetLimits::uniform(initial, 61));
    metered_solve_mod(
        (&source).into(),
        (&rhs).into(),
        &12.into(),
        &mut generator,
        &mut verifier,
    )
    .unwrap();
    assert_eq!(generator.snapshot().verifier_pool, 43);
    assert_eq!(verifier.snapshot().verifier_pool, 61);
    let mut generation_limits = BudgetLimits::uniform(initial, 43);
    let mut verification_limits = BudgetLimits::uniform(initial, 61);
    for dimension in dimensions {
        let generation_used = initial - generator.remaining(dimension);
        let verification_used = initial - verifier.remaining(dimension);
        assert!(generation_used > 0 && verification_used > 0);
        generation_limits.dimensions[dimension.index()] = generation_used;
        verification_limits.dimensions[dimension.index()] = verification_used;
    }
    let mut exact_generator = Budget::new(generation_limits);
    let mut exact_verifier = Budget::new(verification_limits);
    metered_solve_mod(
        (&source).into(),
        (&rhs).into(),
        &12.into(),
        &mut exact_generator,
        &mut exact_verifier,
    )
    .unwrap();
    for dimension in dimensions {
        assert_eq!(exact_generator.remaining(dimension), 0);
        assert_eq!(exact_verifier.remaining(dimension), 0);
        let mut short = generation_limits;
        short.dimensions[dimension.index()] -= 1;
        let mut generator = Budget::new(short);
        assert!(matches!(
            metered_solve_mod(
                (&source).into(),
                (&rhs).into(),
                &12.into(),
                &mut generator,
                &mut Unbounded,
            ),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert_eq!(generator.snapshot().verifier_pool, 43);
        let mut short = verification_limits;
        short.dimensions[dimension.index()] -= 1;
        let mut verifier = Budget::new(short);
        assert!(matches!(
            metered_solve_mod(
                (&source).into(),
                (&rhs).into(),
                &12.into(),
                &mut Unbounded,
                &mut verifier,
            ),
            Err(MatrixError::ResourceLimit(_))
        ));
        assert_eq!(verifier.snapshot().verifier_pool, 61);
    }
    assert!(matches!(
        metered_solve_mod(
            (&source).into(),
            (&rhs).into(),
            &12.into(),
            &mut exact_generator,
            &mut exact_verifier,
        ),
        Err(MatrixError::ResourceLimit(_))
    ));

    // A protected verifier lease can pay only for the independent phase.
    let compute = verification_limits.dimensions[Dimension::ComputeSteps.index()];
    let mut protected = BudgetLimits::uniform(initial, compute);
    protected.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut ledger = Budget::new(protected);
    let lease = ledger.verifier_lease().unwrap();
    metered_solve_mod(
        (&source).into(),
        (&rhs).into(),
        &12.into(),
        &mut Unbounded,
        &mut ledger.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(ledger.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(ledger.snapshot().verifier_pool, 0);
    let mut generation_forbidden = Budget::new(protected);
    assert!(matches!(
        generate_integer_modular_system_candidate(
            (&source).into(),
            (&rhs).into(),
            &12.into(),
            &mut generation_forbidden,
        ),
        Err(MatrixError::ResourceLimit(_))
    ));
    assert_eq!(generation_forbidden.snapshot().verifier_pool, compute);
}
