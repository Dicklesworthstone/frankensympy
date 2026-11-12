//! Independent checks for affine rendering with caller-owned typed coordinates.
//! Variable spelling is not identity, and one RHS must retain every independent
//! free direction without substituting newly generated look-alike parameters.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{Expr, Symbol, SymbolIdentity};
use fsym_matrices::{
    MAX_SPARSE_AFFINE_PARAMETER_NAME_BYTES, MAX_SPARSE_AFFINE_PARAMETER_TOTAL_BYTES, MatrixError,
    SparseMatrix,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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

fn typed(name: &str, tag: u8) -> Expr {
    Expr::Sym(Symbol::with_identity(
        name,
        SymbolIdentity::from_assumptions_digest([tag; 32]),
    ))
}

fn symbol(expression: &Expr) -> Symbol {
    let Expr::Sym(symbol) = expression else {
        panic!("test coordinate must be a symbol")
    };
    symbol.clone()
}

fn fixture() -> (SparseMatrix, SparseMatrix, SparseMatrix, Vec<Expr>) {
    let matrix = integers(2, 4, &[0, 2, 0, 4, 0, 0, 1, 5]);
    let rhs = integers(2, 1, &[6, 7]);
    let augmented = integers(2, 5, &[0, 2, 0, 4, 6, 0, 0, 1, 5, 7]);
    let variables = vec![
        typed("same", 1),
        typed("same", 2),
        Expr::symbol("same"),
        typed("same", 0),
    ];
    (matrix, rhs, augmented, variables)
}

#[test]
fn verified_free_columns_select_exact_caller_coordinates() {
    let (matrix, rhs, _, variables) = fixture();
    let originals = (matrix.clone(), rhs.clone(), variables.clone());
    let (solution, parameters, free) = matrix
        .solve_affine_parameters(&rhs, Some(&variables))
        .unwrap()
        .unwrap();
    assert_eq!(free, [0, 3]);
    assert_eq!((solution.rows(), solution.cols()), (4, 1));
    assert_eq!((parameters.rows(), parameters.cols()), (2, 1));
    assert_eq!(parameters.get(0, 0).unwrap(), variables[0]);
    assert_eq!(parameters.get(1, 0).unwrap(), variables[3]);
    assert_eq!(solution.get(0, 0).unwrap(), variables[0]);
    assert_eq!(solution.get(3, 0).unwrap(), variables[3]);
    let expected_symbols = BTreeSet::from([symbol(&variables[0]), symbol(&variables[3])]);
    let actual_symbols: BTreeSet<_> = (0..4)
        .flat_map(|row| solution.get(row, 0).unwrap().free_symbols())
        .collect();
    assert_eq!(actual_symbols, expected_symbols);
    let replacements = HashMap::from([
        (symbol(&variables[0]), Expr::from_i64(-4)),
        (symbol(&variables[3]), Expr::from_i64(5)),
    ]);
    for (row, expected) in [-4, -7, -18, 5].into_iter().enumerate() {
        assert_eq!(
            solution.get(row, 0).unwrap().subs(&replacements),
            Expr::from_i64(expected)
        );
    }
    assert_eq!((matrix, rhs, variables), originals);
}

#[test]
fn same_spelling_plain_and_declared_parameters_remain_independent_in_sums() {
    let matrix = integers(1, 4, &[1, 1, 1, 1]);
    let rhs = integers(1, 1, &[10]);
    let variables = vec![
        Expr::symbol("pivot"),
        Expr::symbol("x"),
        typed("x", 0),
        typed("x", 7),
    ];
    let (solution, parameters, free) = matrix
        .solve_affine_parameters(&rhs, Some(&variables))
        .unwrap()
        .unwrap();
    assert_eq!(free, [1, 2, 3]);
    for (index, coordinate) in variables[1..].iter().enumerate() {
        assert_eq!(parameters.get(index, 0).unwrap(), *coordinate);
        assert_eq!(solution.get(index + 1, 0).unwrap(), *coordinate);
    }
    let pivot = solution.get(0, 0).unwrap();
    let expected: BTreeSet<_> = variables[1..].iter().map(symbol).collect();
    assert_eq!(
        pivot.free_symbols().into_iter().collect::<BTreeSet<_>>(),
        expected
    );
    for values in [[1, 2, 4], [-3, 5, 11]] {
        let replacements = variables[1..]
            .iter()
            .zip(values)
            .map(|(variable, value)| (symbol(variable), Expr::from_i64(value)))
            .collect();
        assert_eq!(
            pivot.subs(&replacements),
            Expr::from_i64(10 - values.iter().sum::<i64>())
        );
    }
}

#[test]
fn dummy_encodings_unicode_empty_names_and_same_digest_keep_their_typed_atoms() {
    // Dummy spellings use the actual shell's internal encoding, retaining the
    // distinct indices although both objects have the public name "same".
    let variables = vec![
        Expr::symbol("__fsymDummy__11_same"),
        Expr::symbol("__fsymDummy__12_same"),
        typed("left", 3),
        typed("right", 3),
        Expr::symbol("λ_(x)"),
        Expr::symbol(""),
        Expr::symbol("tau0"),
    ];
    let matrix = SparseMatrix::zeros(0, variables.len());
    let rhs = SparseMatrix::zeros(0, 1);
    let (solution, parameters, free) = matrix
        .solve_affine_parameters(&rhs, Some(&variables))
        .unwrap()
        .unwrap();
    assert_eq!(free, (0..variables.len()).collect::<Vec<_>>());
    assert_eq!(solution, parameters);
    for (row, coordinate) in variables.iter().enumerate() {
        assert_eq!(solution.get(row, 0).unwrap(), *coordinate);
    }
}

#[test]
fn absent_coordinates_generate_one_parameter_per_actual_free_column() {
    let (matrix, rhs, _, _) = fixture();
    let (solution, parameters, free) = matrix.solve_affine_parameters(&rhs, None).unwrap().unwrap();
    assert_eq!(free, [0, 3]);
    assert_eq!(parameters.get(0, 0).unwrap(), Expr::symbol("tau0"));
    assert_eq!(parameters.get(1, 0).unwrap(), Expr::symbol("tau1"));
    assert_eq!(solution.get(0, 0).unwrap(), Expr::symbol("tau0"));
    assert_eq!(solution.get(3, 0).unwrap(), Expr::symbol("tau1"));
    assert_eq!(
        (solution, parameters, free),
        matrix.gauss_jordan_solve(&rhs).unwrap().unwrap()
    );
}

#[test]
fn augmented_dispatch_binds_the_actual_last_column() {
    let (matrix, rhs, augmented, variables) = fixture();
    let expected = matrix
        .solve_affine_parameters(&rhs, Some(&variables))
        .unwrap();
    assert_eq!(
        augmented
            .solve_augmented_affine_parameters(Some(&variables))
            .unwrap(),
        expected
    );
    let changed = integers(2, 5, &[0, 2, 0, 4, 6, 0, 0, 1, 5, 8]);
    let (solution, _, free) = changed
        .solve_augmented_affine_parameters(Some(&variables))
        .unwrap()
        .unwrap();
    assert_eq!(free, [0, 3]);
    let replacements = HashMap::from([(symbol(&variables[3]), Expr::from_i64(2))]);
    assert_eq!(
        solution.get(2, 0).unwrap().subs(&replacements),
        Expr::from_i64(-2)
    );
    let inconsistent = integers(2, 3, &[0, 0, 0, 0, 0, 1]);
    assert!(
        inconsistent
            .solve_augmented_affine_parameters(Some(&variables[..2]))
            .unwrap()
            .is_none()
    );
}

#[test]
fn all_coordinates_are_admitted_before_unique_or_inconsistent_shortcuts() {
    let unique = integers(2, 2, &[1, 0, 0, 1]);
    let rhs = integers(2, 1, &[3, 7]);
    let impossible = SparseMatrix::zeros(2, 2);
    let valid = vec![typed("x", 1), typed("x", 2)];
    let (solution, parameters, free) = unique
        .solve_affine_parameters(&rhs, Some(&valid))
        .unwrap()
        .unwrap();
    assert_eq!(solution, rhs);
    assert_eq!(parameters, SparseMatrix::zeros(0, 1));
    assert!(free.is_empty());
    assert!(
        impossible
            .solve_affine_parameters(&rhs, Some(&valid))
            .unwrap()
            .is_none()
    );
    for variables in [
        vec![valid[0].clone()],
        vec![valid[0].clone(), valid[0].clone()],
        vec![valid[0].clone(), Expr::from_i64(0)],
    ] {
        for matrix in [&unique, &impossible] {
            assert!(matches!(
                matrix.solve_affine_parameters(&rhs, Some(&variables)),
                Err(MatrixError::InvalidAffineParameters(_))
            ));
        }
    }
    let no_variables = SparseMatrix::zeros(2, 0);
    let (solution, parameters, free) = no_variables
        .solve_affine_parameters(&SparseMatrix::zeros(2, 1), Some(&[]))
        .unwrap()
        .unwrap();
    assert_eq!(solution, SparseMatrix::zeros(0, 1));
    assert_eq!(parameters, SparseMatrix::zeros(0, 1));
    assert!(free.is_empty());
    assert!(
        no_variables
            .solve_affine_parameters(&rhs, Some(&[]))
            .unwrap()
            .is_none()
    );
}

#[test]
fn nonsymbols_count_mismatches_and_exact_duplicates_never_become_free_coordinates() {
    let matrix = SparseMatrix::zeros(1, 3);
    let rhs = SparseMatrix::zeros(1, 1);
    let base = [Expr::symbol("a"), typed("b", 1), typed("c", 2)];
    for nonsymbol in [
        Expr::from_i64(1),
        Expr::rational(1, 2).unwrap(),
        Expr::Function("Symbol".into(), vec![Expr::symbol("a")]),
        Expr::Add(vec![Expr::symbol("a"), Expr::from_i64(0)]),
    ] {
        let variables = [base[0].clone(), base[1].clone(), nonsymbol];
        assert!(matches!(
            matrix.solve_affine_parameters(&rhs, Some(&variables)),
            Err(MatrixError::InvalidAffineParameters(_))
        ));
    }
    for length in [0, 1, 2, 4] {
        let variables = (0..length)
            .map(|i| Expr::symbol(format!("v{i}")))
            .collect::<Vec<_>>();
        assert!(matches!(
            matrix.solve_affine_parameters(&rhs, Some(&variables)),
            Err(MatrixError::InvalidAffineParameters(_))
        ));
    }
    for repeated in [Expr::symbol("x"), typed("x", 0), typed("x", 7)] {
        let variables = [base[0].clone(), repeated.clone(), repeated];
        assert!(matches!(
            matrix.solve_affine_parameters(&rhs, Some(&variables)),
            Err(MatrixError::InvalidAffineParameters(_))
        ));
    }
}

#[test]
fn a_single_coordinate_vector_cannot_couple_multiple_rhs_families() {
    let variables = [Expr::symbol("x"), Expr::symbol("y")];
    for rows in [0, 2] {
        let matrix = SparseMatrix::zeros(rows, 2);
        for columns in [0, 2, 5] {
            let rhs = SparseMatrix::zeros(rows, columns);
            for coordinates in [None, Some(variables.as_slice())] {
                assert!(matches!(
                    matrix.solve_affine_parameters(&rhs, coordinates),
                    Err(MatrixError::InvalidAffineParameters(_))
                ));
            }
        }
        assert!(matches!(
            SparseMatrix::zeros(rows, 0).solve_augmented_affine_parameters(None),
            Err(MatrixError::InvalidAffineParameters(_))
        ));
    }
    assert!(matches!(
        SparseMatrix::zeros(0, 0).solve_affine_parameters(&SparseMatrix::zeros(0, 2), Some(&[])),
        Err(MatrixError::InvalidAffineParameters(_))
    ));
}

#[test]
fn name_limits_count_utf8_and_apply_to_the_entire_coordinate_list() {
    assert_eq!(MAX_SPARSE_AFFINE_PARAMETER_NAME_BYTES, 4096);
    assert_eq!(MAX_SPARSE_AFFINE_PARAMETER_TOTAL_BYTES, 1_048_576);
    let matrix = SparseMatrix::zeros(0, 1);
    let rhs = SparseMatrix::zeros(0, 1);
    let edge = [Expr::symbol("λ".repeat(2048))];
    assert!(
        matrix
            .solve_affine_parameters(&rhs, Some(&edge))
            .unwrap()
            .is_some()
    );
    for oversized in ["a".repeat(4097), "λ".repeat(2049)] {
        assert!(matches!(
            matrix.solve_affine_parameters(&rhs, Some(&[Expr::symbol(oversized)])),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let variables = (0..257)
        .map(|index| Expr::symbol(format!("{index:04}{}", "a".repeat(4092))))
        .collect::<Vec<_>>();
    assert!(matches!(
        SparseMatrix::zeros(0, variables.len()).solve_affine_parameters(&rhs, Some(&variables)),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn large_augmented_sparse_system_preserves_a_typed_free_coordinate() {
    let rows = 700;
    let mut cells = BTreeMap::new();
    for row in 0..rows {
        cells.insert((row, row), Expr::from_i64(1));
        cells.insert((row, rows), Expr::from_i64(1));
        cells.insert((row, rows + 1), Expr::from_i64(row as i64 + 1));
    }
    let augmented = SparseMatrix::new(rows, rows + 2, cells).unwrap();
    assert!(matches!(
        augmented.to_dense(),
        Err(MatrixError::EntryLimitExceeded(_))
    ));
    let mut variables = (0..rows)
        .map(|i| Expr::symbol(format!("v{i}")))
        .collect::<Vec<_>>();
    variables.push(typed("same", 9));
    let (solution, parameters, free) = augmented
        .solve_augmented_affine_parameters(Some(&variables))
        .unwrap()
        .unwrap();
    assert_eq!(free, [rows]);
    assert_eq!(parameters.get(0, 0).unwrap(), variables[rows]);
    assert_eq!(solution.get(rows, 0).unwrap(), variables[rows]);
    let replacements = HashMap::from([(symbol(&variables[rows]), Expr::from_i64(13))]);
    for row in 0..rows {
        assert_eq!(
            solution.get(row, 0).unwrap().subs(&replacements),
            Expr::from_i64(row as i64 - 12)
        );
    }
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
fn cancellation_precedes_variable_admission_and_covers_every_publication_step() {
    let matrix = integers(1, 2, &[1, 2]);
    let rhs = integers(1, 1, &[3]);
    let augmented = integers(1, 3, &[1, 2, 3]);
    let variables = [typed("same", 1), typed("same", 2)];
    for use_augmented in [false, true] {
        let solve = |meter: &mut Meter| {
            if use_augmented {
                augmented.metered_solve_augmented_affine_parameters(Some(&variables), meter)
            } else {
                matrix.metered_solve_affine_parameters(&rhs, Some(&variables), meter)
            }
        };
        let mut observation = Meter::default();
        solve(&mut observation).unwrap();
        assert!(observation.checkpoints > 10);
        for point in 1..=observation.checkpoints {
            let mut meter = Meter {
                cancel_at: Some(point),
                ..Meter::default()
            };
            assert!(
                matches!(solve(&mut meter), Err(MatrixError::Cancelled)),
                "safe point {point}"
            );
        }
    }
    let malformed = [Expr::from_i64(0), Expr::from_i64(1)];
    let mut cancelled = Meter {
        cancel_at: Some(1),
        ..Meter::default()
    };
    assert!(matches!(
        matrix.metered_solve_affine_parameters(&rhs, Some(&malformed), &mut cancelled),
        Err(MatrixError::Cancelled)
    ));
}

#[test]
fn admission_replay_and_typed_rendering_share_exact_cumulative_budgets() {
    let matrix = integers(1, 3, &[1, 2, 3]);
    let rhs = integers(1, 1, &[7]);
    let augmented = integers(1, 4, &[1, 2, 3, 7]);
    let variables = [typed("x", 1), typed("x", 2), typed("x", 3)];
    for use_augmented in [false, true] {
        let mut observation = Meter::default();
        let expected = if use_augmented {
            augmented.metered_solve_augmented_affine_parameters(Some(&variables), &mut observation)
        } else {
            matrix.metered_solve_affine_parameters(&rhs, Some(&variables), &mut observation)
        }
        .unwrap();
        let solve = |meter: &mut Budget| {
            if use_augmented {
                augmented.metered_solve_augmented_affine_parameters(Some(&variables), meter)
            } else {
                matrix.metered_solve_affine_parameters(&rhs, Some(&variables), meter)
            }
        };
        for (dimension, index) in [
            (Dimension::ComputeSteps, 0),
            (Dimension::MemoryBytes, 1),
            (Dimension::AllocationCount, 2),
        ] {
            assert!(observation.used[index] > 0);
            let mut limits = BudgetLimits::uniform(u64::MAX, 0);
            limits.dimensions[dimension.index()] = observation.used[index];
            assert_eq!(solve(&mut Budget::new(limits)).unwrap(), expected);
            limits.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                solve(&mut Budget::new(limits)),
                Err(MatrixError::ResourceLimit(_))
            ));
            limits.dimensions[dimension.index()] = 0;
            assert!(matches!(
                solve(&mut Budget::new(limits)),
                Err(MatrixError::ResourceLimit(_))
            ));
        }
        let mut limits = BudgetLimits::uniform(u64::MAX, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = observation.used[0] * 2 - 1;
        let mut shared = Budget::new(limits);
        assert_eq!(solve(&mut shared).unwrap(), expected);
        assert!(matches!(
            solve(&mut shared),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
}

#[test]
fn long_parameter_names_charge_more_copying_and_comparison_work() {
    let matrix = integers(1, 3, &[1, 1, 1]);
    let rhs = integers(1, 1, &[10]);
    let short = [Expr::symbol("pivot"), Expr::symbol("a"), Expr::symbol("b")];
    let long = [
        Expr::symbol("pivot"),
        Expr::symbol(format!("{}a", "x".repeat(1023))),
        Expr::symbol(format!("{}b", "x".repeat(1023))),
    ];
    let mut short_work = Meter::default();
    let mut long_work = Meter::default();
    matrix
        .metered_solve_affine_parameters(&rhs, Some(&short), &mut short_work)
        .unwrap();
    matrix
        .metered_solve_affine_parameters(&rhs, Some(&long), &mut long_work)
        .unwrap();
    assert!(long_work.used[0] > short_work.used[0]);
    assert!(long_work.used[1] > short_work.used[1]);
}
