//! Independent checks of the fixed-parameter source projection.

use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{Symbol, parse_raw};
use fsym_numeric_consumer::{
    LeastSquaresOptions, NewtonGuards, sparse_least_squares_parametric,
    sparse_newton_refine_parametric,
};

fn system(sources: &[&str], axes: &[&str]) -> VerifiedResidualSystem {
    let expressions: Vec<_> = sources
        .iter()
        .map(|source| parse_raw(source).unwrap())
        .collect();
    let axes: Vec<_> = axes.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedResidualSystem::try_compile(&expressions, &axes).unwrap()
}

#[test]
fn the_fixed_parameter_suffix_preserves_negative_zero_in_original_evaluation() {
    let source = system(&["x-1", "p"], &["x", "p"]);
    let parameters = [-0.0_f64];
    let fit = sparse_least_squares_parametric(
        &source,
        &[0.0],
        &parameters,
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    assert_eq!(fit.residuals[1].to_bits(), (-0.0_f64).to_bits());
    let root =
        sparse_newton_refine_parametric(&source, &[0.0], &parameters, &NewtonGuards::default())
            .unwrap();
    assert_eq!(parameters[0].to_bits(), (-0.0_f64).to_bits());
    assert!((root.root[0] - 1.0).abs() <= 1e-10);
    assert!((fit.point[0] - 1.0).abs() <= 1e-10);
}

#[test]
fn rowwise_interleaving_of_unknown_and_parameter_columns_preserves_every_value() {
    let source = system(&["x+p", "y+q", "p*x+q*y+1"], &["x", "y", "p", "q"]);
    let fit = sparse_least_squares_parametric(
        &source,
        &[0.0, 0.0],
        &[2.0, 3.0],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    // The independently solved normal equations are
    // [5 6; 6 10] * [x; y] = [-4; -6].
    assert!((fit.point[0] + 2.0 / 7.0).abs() <= 1e-10);
    assert!((fit.point[1] + 3.0 / 7.0).abs() <= 1e-10);
    assert!((fit.cost - 36.0 / 7.0).abs() <= 1e-10);
    let expected = [12.0 / 7.0, 18.0 / 7.0, -6.0 / 7.0];
    for (&actual, target) in fit.residuals.iter().zip(expected) {
        assert!((actual - target).abs() <= 1e-10);
    }
    let fresh = source
        .evaluate_sparse(&[fit.point[0], fit.point[1], 2.0, 3.0])
        .unwrap();
    assert_eq!(fresh.residuals, fit.residuals);
    assert!(fit.gradient_norm <= 1e-10);
    assert!(
        sparse_newton_refine_parametric(
            &source,
            &[0.0, 0.0],
            &[2.0, 3.0],
            &NewtonGuards::default(),
        )
        .is_err()
    );
}
