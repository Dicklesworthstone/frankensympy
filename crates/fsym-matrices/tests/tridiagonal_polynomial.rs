use fsym_core::{BigRational, Expr, Symbol};
use fsym_matrices::{MatrixError, SparseMatrix};
use fsym_polys::UnivariatePoly as Poly;

#[test]
fn independently_rejects_a_wrong_candidate_even_if_first_points_agree() {
    // det [[x,1],[2,x]] = x^2 - 2. A planted error x(x+1) vanishes
    // at the first two verifier points, and only the final point detects it.
    let x = Symbol::new("x");
    let entries = [
        (0, 0, Expr::Sym(x.clone())),
        (0, 1, Expr::from_i64(1)),
        (1, 0, Expr::from_i64(2)),
        (1, 1, Expr::Sym(x.clone())),
    ];
    let matrix = SparseMatrix::new(
        2,
        2,
        entries.into_iter().map(|(r, c, p)| ((r, c), p)).collect(),
    )
    .unwrap();
    let polynomial = |coeffs: &[i64]| {
        Poly::new(
            x.clone(),
            coeffs
                .iter()
                .map(|&v| BigRational::from_integer(v.into()))
                .collect(),
        )
    };
    let expected = polynomial(&[-2, 0, 1]);
    assert_eq!(
        matrix.tridiagonal_polynomial_det().unwrap().unwrap(),
        expected.to_expr()
    );
    assert!(
        matrix
            .verify_tridiagonal_polynomial_determinant(&expected)
            .is_ok()
    );
    assert!(matches!(
        matrix.verify_tridiagonal_polynomial_determinant(&polynomial(&[-2, 1, 2])),
        Err(MatrixError::InvalidCertificate(_))
    ));
    assert!(matches!(
        matrix.verify_tridiagonal_polynomial_determinant(&polynomial(&[-2, 0, 0, 1])),
        Err(MatrixError::InvalidCertificate(_))
    ));
}

#[test]
fn unsupported_entries_are_not_erased_by_a_singular_row() {
    let x = Expr::symbol("x");
    let pole = Expr::Pow(x.clone().into(), Expr::from_i64(-1).into());
    let mixed = Expr::symbol("y");
    for cell in [pole, mixed] {
        let stored = [((0, 0), x.clone()), ((8, 8), cell)].into_iter().collect();
        let matrix = SparseMatrix::new(9, 9, stored).unwrap();
        assert_eq!(matrix.tridiagonal_polynomial_det().unwrap(), None);
    }
    let high_power = Expr::Pow(x.into(), Expr::from_i64(100_000_000).into());
    let matrix = SparseMatrix::new(9, 9, [((0, 0), high_power)].into_iter().collect()).unwrap();
    assert!(matches!(
        matrix.tridiagonal_polynomial_det(),
        Err(MatrixError::ResourceLimit(_))
    ));
}

#[test]
fn reference_verifier_requires_the_original_shape_band_and_typed_generator() {
    use fsym_core::SymbolIdentity;
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let candidate = Poly::monomial(y.clone(), BigRational::from_integer(1.into()), 1).unwrap();
    let diag = [((0, 0), Expr::Sym(x.clone())), ((1, 1), Expr::from_i64(1))];
    let matrix = SparseMatrix::new(2, 2, diag.into_iter().collect()).unwrap();
    assert!(matches!(
        matrix.verify_tridiagonal_polynomial_determinant(&candidate),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let rectangular =
        SparseMatrix::new(2, 3, [((0, 0), Expr::Sym(x.clone()))].into_iter().collect()).unwrap();
    assert!(matches!(
        rectangular.verify_tridiagonal_polynomial_determinant(&candidate),
        Err(MatrixError::NotSquare(2, 3))
    ));
    let off_band = SparseMatrix::new(
        3,
        3,
        [((0, 0), Expr::Sym(x.clone())), ((0, 2), Expr::from_i64(2))]
            .into_iter()
            .collect(),
    )
    .unwrap();
    assert_eq!(off_band.tridiagonal_polynomial_det().unwrap(), None);
    assert!(matches!(
        off_band.verify_tridiagonal_polynomial_determinant(&candidate),
        Err(MatrixError::InvalidCertificate(_))
    ));
    let typed_mismatch = SparseMatrix::new(
        2,
        2,
        [((0, 0), Expr::Sym(x)), ((1, 1), Expr::Sym(y))]
            .into_iter()
            .collect(),
    )
    .unwrap();
    assert_eq!(typed_mismatch.tridiagonal_polynomial_det().unwrap(), None);
}

#[test]
fn characteristic_polynomial_of_a_path_and_large_sparse_boundary() {
    use fsym_matrices::Matrix;
    let x = Symbol::new("x");
    let n: usize = 12;
    let cells = (0..n)
        .flat_map(|r| (0..n).map(move |c| (r, c)))
        .map(|(r, c)| {
            if r == c {
                Expr::Sym(x.clone())
            } else {
                Expr::from_i64(i64::from(r.abs_diff(c) == 1))
            }
        })
        .collect();
    let polynomial = Poly::new(
        x.clone(),
        [1, 0, -21, 0, 70, 0, -84, 0, 45, 0, -11, 0, 1]
            .into_iter()
            .map(|v| BigRational::from_integer(i64::from(v).into()))
            .collect(),
    );
    let determinant = Matrix::new(n, n, cells).unwrap().det().unwrap();
    assert_eq!(Poly::from_expr(&determinant, &x).unwrap(), polynomial);
    let Expr::Add(terms) = determinant else {
        panic!("expected the characteristic polynomial sum")
    };
    assert!(
        terms
            .windows(2)
            .all(|pair| fsym_core::cmp_add_args(&pair[0], &pair[1]).is_le())
    );
    let n = 600;
    let mut cells: std::collections::BTreeMap<_, _> =
        (0..n).map(|i| ((i, i), Expr::from_i64(1))).collect();
    cells.insert((0, 0), Expr::Sym(x.clone()));
    cells.insert((n - 1, n - 2), Expr::from_i64(-1));
    cells.insert((n - 2, n - 1), Expr::from_i64(2));
    let large = SparseMatrix::new(n, n, cells).unwrap();
    assert!(large.to_dense().is_err());
    let expected = Poly::monomial(x, BigRational::from_integer(3.into()), 1).unwrap();
    assert_eq!(
        large.tridiagonal_polynomial_det().unwrap(),
        Some(expected.to_expr())
    );
}

#[test]
fn standard_sparse_determinant_charges_the_caller_through_reference_verification() {
    use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
    let n = 12;
    let x = Expr::symbol("x");
    let cells = (0..n)
        .map(|i| ((i, i), x.clone()))
        .chain((1..n).flat_map(|i| {
            [
                ((i - 1, i), Expr::from_i64(1)),
                ((i, i - 1), Expr::from_i64(1)),
            ]
        }))
        .collect();
    let matrix = SparseMatrix::new(n, n, cells).unwrap();
    let answer = matrix.tridiagonal_polynomial_det().unwrap().unwrap();
    assert_eq!(matrix.det().unwrap(), answer);
    let limits = BudgetLimits::uniform(256 * 1024 * 1024, 0);
    let mut measured = Budget::new(limits);
    assert_eq!(matrix.metered_det(&mut measured).unwrap(), answer);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let spent = limits.dimensions[dimension.index()] - measured.remaining(dimension);
        assert!(spent > 100);
        let mut tight = limits;
        tight.dimensions[dimension.index()] = spent - 1;
        assert!(matches!(
            matrix.metered_det(&mut Budget::new(tight)),
            Err(MatrixError::ResourceLimit(_))
        ));
    }
    struct CancelAfter(usize);
    impl BudgetMeter for CancelAfter {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.0 == 0 {
                return Err(MeterError::Cancelled);
            }
            self.0 -= 1;
            Ok(())
        }
    }
    for stop in [0, 25, 200, 1000] {
        assert!(matches!(
            matrix.metered_det(&mut CancelAfter(stop)),
            Err(MatrixError::Cancelled)
        ));
    }
}
