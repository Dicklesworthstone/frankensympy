//! Polynomial RHS solves: exact witnesses, original domains, typed variables,
//! full rectangular residuals and cumulative resource/cancellation failures.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_core::{BigInt, Constant, Expr, Symbol, SymbolIdentity};
use fsym_matrices::{Matrix, MatrixError, PolynomialLinearSystemCandidate, SparseMatrix};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn integer(value: i64) -> Expr {
    Expr::from_i64(value)
}

fn variable(name: &str) -> Expr {
    Expr::symbol(name)
}

fn add(values: Vec<Expr>) -> Expr {
    Expr::Add(values)
}

fn multiply(values: Vec<Expr>) -> Expr {
    Expr::Mul(values)
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(integer(exponent)))
}

fn imaginary() -> Expr {
    Expr::Const(Constant::I)
}

fn gaussian(real: i64, imag: i64) -> Expr {
    if imag == 0 {
        integer(real)
    } else if real == 0 {
        multiply(vec![integer(imag), imaginary()])
    } else {
        add(vec![
            integer(real),
            multiply(vec![integer(imag), imaginary()]),
        ])
    }
}

fn matrix(rows: usize, columns: usize, data: Vec<Expr>) -> Matrix {
    Matrix::new(rows, columns, data).unwrap()
}

fn assert_same(actual: &Matrix, expected: &Matrix) {
    assert_eq!(
        (actual.rows(), actual.cols()),
        (expected.rows(), expected.cols())
    );
    // Independent exact expansion is a second oracle, not the implementation's
    // polynomial interpreter or coefficient-dot certificate checker.
    for (actual, expected) in actual.data().iter().zip(expected.data()) {
        let difference = add(vec![
            actual.clone(),
            multiply(vec![integer(-1), expected.clone()]),
        ]);
        assert_eq!(
            fsym_simplify::simplify(&fsym_simplify::expand(&difference)),
            integer(0),
            "actual={actual}; expected={expected}"
        );
    }
}

fn witness() -> (Matrix, Matrix, PolynomialLinearSystemCandidate) {
    let x = variable("x");
    let y = variable("y");
    let first = add(vec![power(x.clone(), 2), y.clone()]);
    let second = add(vec![
        integer(1),
        multiply(vec![gaussian(1, 1), x.clone(), y.clone()]),
    ]);
    // A=[1;i] and L=[1,0]. This hand-written L is not A's pseudoinverse.
    let a = matrix(2, 1, vec![integer(1), imaginary()]);
    let b = matrix(
        2,
        2,
        vec![
            first.clone(),
            second.clone(),
            add(vec![
                multiply(vec![imaginary(), power(x.clone(), 2)]),
                multiply(vec![imaginary(), y.clone()]),
            ]),
            add(vec![imaginary(), multiply(vec![gaussian(-1, 1), x, y])]),
        ],
    );
    let candidate = PolynomialLinearSystemCandidate {
        left_inverse: matrix(1, 2, vec![integer(1), integer(0)]),
        solution: matrix(1, 2, vec![first, second]),
    };
    (a, b, candidate)
}

#[test]
fn independently_authored_non_penrose_witness_checks_all_polynomial_columns() {
    let (a, b, candidate) = witness();
    a.verify_polynomial_linear_system_candidate(&b, &candidate, &mut Unbounded)
        .unwrap();
    let original_a = a.clone();
    let original_b = b.clone();
    assert_same(&a.solve(&b).unwrap(), &candidate.solution);
    assert_same(
        &a.metered_polynomial_solve(&b, &mut Unbounded).unwrap(),
        &candidate.solution,
    );
    assert_eq!(a, original_a);
    assert_eq!(b, original_b);
}

#[test]
fn square_rational_coefficients_solve_multivariate_multi_rhs_systems() {
    let x = variable("x");
    let y = variable("y");
    let a = matrix(2, 2, vec![integer(2), integer(1), integer(0), integer(3)]);
    let b = matrix(
        2,
        2,
        vec![
            add(vec![
                multiply(vec![integer(2), power(x.clone(), 2)]),
                x.clone(),
                y.clone(),
            ]),
            add(vec![
                multiply(vec![integer(2), x.clone(), y.clone()]),
                integer(1),
            ]),
            add(vec![
                multiply(vec![integer(3), x.clone()]),
                multiply(vec![integer(-3), y.clone()]),
            ]),
            integer(3),
        ],
    );
    let expected = matrix(
        2,
        2,
        vec![
            add(vec![power(x.clone(), 2), y.clone()]),
            multiply(vec![x.clone(), y.clone()]),
            add(vec![x, multiply(vec![integer(-1), y])]),
            integer(1),
        ],
    );
    assert_same(&a.solve(&b).unwrap(), &expected);
}

#[test]
fn hand_derived_complex_triangular_corpus_covers_nonreal_pivots() {
    let x = variable("load_x");
    let y = variable("load_y");
    let expected = matrix(
        2,
        1,
        vec![
            add(vec![x.clone(), power(y.clone(), 2)]),
            multiply(vec![x.clone(), y.clone()]),
        ],
    );
    for real in [1, 2] {
        for imag in [-1, 0, 2] {
            let pivot = gaussian(real, imag);
            let a = matrix(
                2,
                2,
                vec![pivot.clone(), integer(1), integer(0), gaussian(1, 1)],
            );
            // RHS coefficients are derived by hand, without native matmul.
            let b = matrix(
                2,
                1,
                vec![
                    add(vec![
                        multiply(vec![pivot.clone(), x.clone()]),
                        multiply(vec![pivot, power(y.clone(), 2)]),
                        multiply(vec![x.clone(), y.clone()]),
                    ]),
                    multiply(vec![gaussian(1, 1), x.clone(), y.clone()]),
                ],
            );
            assert_same(&a.solve(&b).unwrap(), &expected);
        }
    }
}

#[test]
fn ground_gaussian_reciprocals_and_polynomial_powers_remain_exact() {
    let x = variable("x");
    let y = variable("y");
    let source = multiply(vec![
        power(add(vec![integer(1), imaginary()]), -1),
        power(add(vec![x.clone(), y.clone()]), 2),
    ]);
    let rhs = matrix(1, 1, vec![source]);
    let half = Expr::rational(1, 2).unwrap();
    let expected = matrix(
        1,
        1,
        vec![multiply(vec![
            add(vec![
                half.clone(),
                multiply(vec![integer(-1), half, imaginary()]),
            ]),
            add(vec![
                power(x.clone(), 2),
                multiply(vec![integer(2), x, y.clone()]),
                power(y, 2),
            ]),
        ])],
    );
    assert_same(&Matrix::eye(1).unwrap().solve(&rhs).unwrap(), &expected);
    let ground_zero_power = matrix(1, 1, vec![power(gaussian(1, 1), 0)]);
    assert_eq!(
        Matrix::eye(1)
            .unwrap()
            .metered_polynomial_solve(&ground_zero_power, &mut Unbounded)
            .unwrap(),
        Matrix::eye(1).unwrap()
    );
}

#[test]
fn every_rhs_column_must_be_consistent_as_a_polynomial_identity() {
    let x = variable("x");
    let a = matrix(2, 1, vec![integer(1), integer(1)]);
    let b = matrix(
        2,
        2,
        vec![
            x.clone(),
            integer(0),
            x.clone(),
            multiply(vec![x.clone(), add(vec![x, integer(-1)])]),
        ],
    );
    // The second residual vanishes at x=0 and x=1 but is not an identity.
    assert_eq!(a.solve(&b), Err(MatrixError::InconsistentLinearSystem));
    let candidate = a
        .polynomial_linear_system_candidate_with_meter(&b, &mut Unbounded)
        .unwrap();
    assert!(matches!(
        a.verify_polynomial_linear_system_candidate(&b, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn residual_zero_without_a_left_inverse_never_claims_uniqueness() {
    let x = variable("x");
    let a = matrix(1, 2, vec![integer(1), imaginary()]);
    let b = matrix(1, 1, vec![x.clone()]);
    let candidate = PolynomialLinearSystemCandidate {
        left_inverse: matrix(2, 1, vec![integer(1), integer(0)]),
        solution: matrix(2, 1, vec![x, integer(0)]),
    };
    assert!(matches!(
        a.verify_polynomial_linear_system_candidate(&b, &candidate, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
    assert_eq!(a.solve(&b), Err(MatrixError::SingularMatrix));
    assert_eq!(
        a.metered_polynomial_solve(&Matrix::zeros(1, 0).unwrap(), &mut Unbounded),
        Err(MatrixError::SingularMatrix)
    );
}

#[test]
fn original_sources_shapes_and_witnesses_are_not_replaceable() {
    let (a, b, candidate) = witness();
    let changed_a = matrix(2, 1, vec![integer(1), gaussian(0, 2)]);
    let mut changed_data = b.data().to_vec();
    changed_data[3] = add(vec![changed_data[3].clone(), variable("extra")]);
    let changed_b = matrix(2, 2, changed_data);
    let bad_solution = PolynomialLinearSystemCandidate {
        solution: Matrix::zeros(1, 2).unwrap(),
        ..candidate.clone()
    };
    let bad_inverse = PolynomialLinearSystemCandidate {
        left_inverse: Matrix::zeros(1, 2).unwrap(),
        ..candidate.clone()
    };
    let bad_shape = PolynomialLinearSystemCandidate {
        solution: Matrix::zeros(2, 1).unwrap(),
        ..candidate.clone()
    };
    for (left, right, proposed) in [
        (&changed_a, &b, &candidate),
        (&a, &changed_b, &candidate),
        (&a, &b, &bad_solution),
        (&a, &b, &bad_inverse),
        (&a, &b, &bad_shape),
    ] {
        assert!(
            left.verify_polynomial_linear_system_candidate(right, proposed, &mut Unbounded)
                .is_err()
        );
    }
    assert!(matches!(
        a.metered_polynomial_solve(&matrix(1, 1, vec![variable("x")]), &mut Unbounded),
        Err(MatrixError::ShapeMismatch(..))
    ));
}

fn symbols(expression: &Expr, output: &mut BTreeSet<Symbol>) {
    match expression {
        Expr::Sym(symbol) => {
            output.insert(symbol.clone());
        }
        Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
            for child in children {
                symbols(child, output);
            }
        }
        Expr::Pow(base, exponent) => {
            symbols(base, output);
            symbols(exponent, output);
        }
        _ => {}
    }
}

#[test]
fn same_printed_names_keep_distinct_typed_coordinates() {
    let first = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([1; 32]));
    let second = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([2; 32]));
    let rhs = matrix(
        1,
        1,
        vec![add(vec![
            Expr::Sym(first.clone()),
            multiply(vec![integer(2), Expr::Sym(second.clone())]),
        ])],
    );
    let a = matrix(1, 1, vec![integer(2)]);
    let result = a.solve(&rhs).unwrap();
    let mut observed = BTreeSet::new();
    symbols(result.get(0, 0).unwrap(), &mut observed);
    assert_eq!(observed, BTreeSet::from([first.clone(), second.clone()]));
    let candidate = PolynomialLinearSystemCandidate {
        left_inverse: matrix(1, 1, vec![Expr::rational(1, 2).unwrap()]),
        solution: matrix(
            1,
            1,
            vec![add(vec![
                multiply(vec![Expr::rational(1, 2).unwrap(), Expr::Sym(first)]),
                Expr::Sym(second),
            ])],
        ),
    };
    a.verify_polynomial_linear_system_candidate(&rhs, &candidate, &mut Unbounded)
        .unwrap();
    let changed = PolynomialLinearSystemCandidate {
        solution: matrix(1, 1, vec![variable("same")]),
        ..candidate
    };
    assert!(matches!(
        a.verify_polynomial_linear_system_candidate(&rhs, &changed, &mut Unbounded),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn zero_shapes_retain_uniqueness_and_full_polynomial_consistency() {
    let (a, _, _) = witness();
    assert_eq!(
        a.metered_polynomial_solve(&Matrix::zeros(2, 0).unwrap(), &mut Unbounded)
            .unwrap(),
        Matrix::zeros(1, 0).unwrap()
    );
    let no_unknowns = Matrix::zeros(2, 0).unwrap();
    let x = variable("x");
    let zero_rhs = matrix(
        2,
        1,
        vec![
            add(vec![x.clone(), multiply(vec![integer(-1), x.clone()])]),
            integer(0),
        ],
    );
    assert_eq!(
        no_unknowns.solve(&zero_rhs).unwrap(),
        Matrix::zeros(0, 1).unwrap()
    );
    assert_eq!(
        no_unknowns.solve(&matrix(2, 1, vec![integer(0), x])),
        Err(MatrixError::InconsistentLinearSystem)
    );
}

#[test]
fn original_poles_and_indeterminate_ground_powers_cannot_disappear() {
    let x = variable("x");
    let pole = power(integer(0), -1);
    let one = Matrix::eye(1).unwrap();
    for expression in [
        multiply(vec![integer(0), x.clone(), pole.clone()]),
        add(vec![x.clone(), power(pole.clone(), 0)]),
        multiply(vec![x.clone(), power(integer(0), 0)]),
        multiply(vec![
            integer(0),
            x,
            power(add(vec![integer(1), integer(-1)]), -1),
        ]),
    ] {
        let rhs = matrix(1, 1, vec![expression]);
        assert_eq!(
            one.metered_polynomial_solve(&rhs, &mut Unbounded),
            Err(MatrixError::DivisionByZero)
        );
        assert_eq!(one.solve(&rhs), Err(MatrixError::DivisionByZero));
    }
    let invalid_a = matrix(1, 1, vec![multiply(vec![integer(0), pole])]);
    assert_eq!(
        invalid_a.metered_polynomial_solve(&matrix(1, 1, vec![variable("x")]), &mut Unbounded),
        Err(MatrixError::DivisionByZero)
    );
}

#[test]
fn symbolic_nonpositive_powers_are_not_erased_by_source_reduction() {
    let x = variable("x");
    for expression in [
        power(x.clone(), -1),
        multiply(vec![integer(0), power(x.clone(), -1)]),
        power(
            add(vec![
                integer(1),
                x.clone(),
                multiply(vec![integer(-1), x.clone()]),
            ]),
            -1,
        ),
        power(x.clone(), 0),
        power(add(vec![x.clone(), multiply(vec![integer(-1), x])]), 0),
        power(Expr::Function("sin".into(), vec![variable("x")]), 0),
    ] {
        let rhs = matrix(1, 1, vec![expression]);
        assert!(
            !rhs.has_polynomial_rhs_syntax_with_meter(&mut Unbounded)
                .unwrap()
        );
        assert_eq!(
            rhs.validate_polynomial_rhs_source_with_meter(&mut Unbounded),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
        assert_eq!(
            Matrix::eye(1).unwrap().solve(&rhs),
            Err(MatrixError::UnsupportedCertificateDomain)
        );
    }
    let huge_unknown = matrix(
        1,
        1,
        vec![Expr::Function(
            "unsupported".into(),
            vec![Expr::Integer(BigInt::one() << 4097u32)],
        )],
    );
    assert!(matches!(
        huge_unknown.has_polynomial_rhs_syntax_with_meter(&mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn syntax_expansion_and_scalar_envelopes_refuse_before_publication() {
    let mut deep = variable("x");
    for _ in 0..66 {
        deep = add(vec![deep]);
    }
    let variables = (0..9).map(|index| variable(&format!("v{index}"))).collect();
    let sources = vec![
        deep,
        add(vec![variable("x"); 16_385]),
        add(variables),
        variable(&"x".repeat(1025)),
        power(variable("x"), 65),
        power(power(variable("x"), 2), 33),
        power(
            add(vec![
                integer(1),
                variable("a"),
                variable("b"),
                variable("c"),
                variable("d"),
            ]),
            8,
        ),
        multiply(vec![Expr::Integer(BigInt::one() << 4097u32), variable("x")]),
    ];
    for expression in sources {
        let rhs = matrix(1, 1, vec![expression]);
        assert!(matches!(
            Matrix::eye(1)
                .unwrap()
                .metered_polynomial_solve(&rhs, &mut Unbounded),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    let too_tall = matrix(33, 1, vec![variable("x"); 33]);
    assert!(matches!(
        too_tall.validate_polynomial_rhs_source_with_meter(&mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn sparse_originals_are_admitted_without_losing_implicit_zero_cells() {
    let sparse = SparseMatrix::new(
        3,
        2,
        BTreeMap::from([
            ((0, 1), power(variable("x"), 2)),
            ((2, 0), multiply(vec![imaginary(), variable("y")])),
        ]),
    )
    .unwrap();
    assert!(
        sparse
            .has_polynomial_rhs_syntax_with_meter(&mut Unbounded)
            .unwrap()
    );
    sparse
        .validate_polynomial_rhs_source_with_meter(&mut Unbounded)
        .unwrap();
    let invalid = SparseMatrix::new(
        1,
        1,
        BTreeMap::from([((0, 0), multiply(vec![variable("x"), power(integer(0), -1)]))]),
    )
    .unwrap();
    assert_eq!(
        invalid.validate_polynomial_rhs_source_with_meter(&mut Unbounded),
        Err(MatrixError::DivisionByZero)
    );
    let large = SparseMatrix::new(33, 1, BTreeMap::from([((0, 0), variable("x"))])).unwrap();
    assert!(matches!(
        large.has_polynomial_rhs_syntax_with_meter(&mut Unbounded),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[derive(Default)]
struct Recording {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Recording {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charges[dimension.index()] += amount;
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
fn complete_work_is_cumulative_and_cancellable_through_publication() {
    let a = matrix(1, 1, vec![gaussian(1, 1)]);
    let b = matrix(1, 1, vec![add(vec![variable("x"), integer(2)])]);
    let mut measured = Recording::default();
    let expected = a.metered_polynomial_solve(&b, &mut measured).unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    assert_eq!(
        a.metered_polynomial_solve(&b, &mut Budget::new(limits))
            .unwrap(),
        expected
    );
    for dimension in Dimension::ALL {
        if limits.dimensions[dimension.index()] == 0 {
            continue;
        }
        let mut short = limits;
        short.dimensions[dimension.index()] -= 1;
        assert!(matches!(
            a.metered_polynomial_solve(&b, &mut Budget::new(short)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    for cancel_at in [
        1,
        measured.checkpoints / 2,
        measured.checkpoints - 1,
        measured.checkpoints,
    ] {
        assert_eq!(
            a.metered_polynomial_solve(
                &b,
                &mut Recording {
                    cancel_at: Some(cancel_at),
                    ..Default::default()
                },
            ),
            Err(MatrixError::Cancelled)
        );
    }
}

#[test]
fn independent_checker_keeps_its_final_budget_and_cancellation_boundary() {
    let (a, b, candidate) = witness();
    let mut measured = Recording::default();
    a.verify_polynomial_linear_system_candidate(&b, &candidate, &mut measured)
        .unwrap();
    assert_eq!(
        a.verify_polynomial_linear_system_candidate(
            &b,
            &candidate,
            &mut Recording {
                cancel_at: Some(measured.checkpoints),
                ..Default::default()
            },
        ),
        Err(MatrixError::Cancelled)
    );
    let mut limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    limits.dimensions[Dimension::ComputeSteps.index()] -= 1;
    assert!(matches!(
        a.verify_polynomial_linear_system_candidate(&b, &candidate, &mut Budget::new(limits)),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn published_polynomials_have_the_canonical_expression_structure() {
    let x = variable("canonical_x");
    let y = variable("canonical_y");
    // Independently construct normal expressions through the core arithmetic
    // API. These assertions check public structural equality, not expansion.
    let expected = matrix(
        1,
        4,
        vec![
            x.clone() - integer(2) * y.clone(),
            integer(3) * imaginary() * power(x.clone(), 2) * y.clone(),
            x.clone() + integer(3) + integer(2) * imaginary(),
            (integer(1) + imaginary()) * power(x, 2) * y,
        ],
    );
    let identity = matrix(1, 1, vec![integer(1)]);
    let actual = identity.solve(&expected).unwrap();
    assert_eq!(actual, expected);
}
