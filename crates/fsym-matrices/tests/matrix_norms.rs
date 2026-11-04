//! Matrix norms must use the complex modulus, including for symbolic cells.

use fsym_core::{Constant, Expr, Symbol};
use fsym_matrices::Matrix;
use std::sync::Arc;

#[test]
fn imaginary_unit_has_unit_squared_norm() {
    let matrix = Matrix::new(1, 1, vec![Expr::Const(Constant::I)]).unwrap();
    assert_eq!(matrix.frobenius_norm_squared().unwrap(), Expr::from_i64(1));
}

#[test]
fn gaussian_rational_entries_add_their_real_and_imaginary_squares() {
    let i = Expr::Const(Constant::I);
    let entries = vec![
        Expr::from_i64(3) + Expr::from_i64(4) * i.clone(),
        Expr::from_i64(-2) * i.clone(),
        Expr::from_i64(1) + i,
        Expr::from_i64(-3),
    ];
    let matrix = Matrix::new(2, 2, entries).unwrap();
    assert_eq!(matrix.frobenius_norm_squared().unwrap(), Expr::from_i64(40));
}

#[test]
fn complex_phase_does_not_change_the_squared_norm() {
    let real = Matrix::new(
        2,
        2,
        vec![1, -2, 3, -4].into_iter().map(Expr::from_i64).collect(),
    )
    .unwrap();
    let imaginary = Matrix::new(
        2,
        2,
        vec![1, -2, 3, -4]
            .into_iter()
            .map(|n| Expr::from_i64(n) * Expr::Const(Constant::I))
            .collect(),
    )
    .unwrap();
    assert_eq!(real.frobenius_norm_squared().unwrap(), Expr::from_i64(30));
    assert_eq!(
        imaginary.frobenius_norm_squared().unwrap(),
        real.frobenius_norm_squared().unwrap()
    );
}

#[test]
fn unconstrained_symbol_keeps_its_absolute_value() {
    let symbol = Expr::Sym(Symbol::new("z"));
    let matrix = Matrix::new(1, 1, vec![symbol.clone()]).unwrap();
    let expected = Expr::Pow(
        Arc::new(Expr::Function("Abs".into(), vec![symbol])),
        Arc::new(Expr::from_i64(2)),
    );
    assert_eq!(matrix.frobenius_norm_squared().unwrap(), expected);
}

#[test]
fn empty_matrices_have_zero_frobenius_norm() {
    for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
        let matrix = Matrix::new(rows, cols, vec![]).unwrap();
        assert_eq!(matrix.frobenius_norm_squared().unwrap(), Expr::from_i64(0));
    }
}
