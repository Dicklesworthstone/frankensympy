//! Independent constrained-fit regressions. The affine reference enumerates
//! box faces directly, without reusing the iterative solver or its projection.

use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{Expr, Symbol, parse_raw};
use fsym_numeric_consumer::{
    LeastSquaresOptions, SystemConsumerError, sparse_bounded_least_squares,
};

fn system(sources: &[&str]) -> VerifiedResidualSystem {
    let sources: Vec<_> = sources
        .iter()
        .map(|source| parse_raw(source).unwrap())
        .collect();
    VerifiedResidualSystem::try_compile(&sources, &[Symbol::new("x"), Symbol::new("y")])
        .unwrap()
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected}"
    );
}

#[test]
fn a_temporarily_active_coordinate_must_be_released_after_coupled_motion() {
    // Initially g_x = 1 points outside x >= 0, so the first free solve moves
    // only y. That motion changes g_x to a negative value: x must be released.
    // Keeping the initial active set forever would stop at (0, -3/5).
    let source = system(&["x+2*y+1", "y+1"]);
    let result = sparse_bounded_least_squares(
        &source,
        &[0.0, 0.0],
        &[0.0, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY],
        &LeastSquaresOptions::default(),
    )
    .unwrap();
    near(result.point[0], 1.0, 1e-9);
    near(result.point[1], -1.0, 1e-9);
    assert!(result.cost < 1e-18);
    assert!(result.projected_gradient_norm <= 1e-10);
}

#[test]
fn a_strictly_interior_point_cannot_borrow_a_nearby_bounds_kkt_condition() {
    let source = system(&["x+1"]);
    let options = LeastSquaresOptions {
        max_iterations: 0,
        gradient_tolerance: 0.0,
        ..LeastSquaresOptions::default()
    };
    let lower = [0.0, -1.0];
    let upper = [1.0, 1.0];
    assert!(matches!(
        sparse_bounded_least_squares(
            &source,
            &[2.0_f64.powi(-60), 0.0],
            &lower,
            &upper,
            &options,
        ),
        Err(SystemConsumerError::BoundedLeastSquaresNoConvergence {
            iterations: 0,
            ..
        })
    ));
    let boundary =
        sparse_bounded_least_squares(&source, &[0.0, 0.0], &lower, &upper, &options)
            .unwrap();
    assert_eq!(boundary.gradient_norm, 1.0);
    assert_eq!(boundary.projected_gradient_norm, 0.0);
    assert_eq!(boundary.cost, 0.5);
}

#[test]
fn fixed_coordinates_retain_original_derivative_and_held_source_domains() {
    for formula in ["sqrt(x)-1", "x+0*log(x)", "x+0*(1/x)"] {
        let source = system(&[formula]);
        assert!(
            sparse_bounded_least_squares(
                &source,
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &LeastSquaresOptions::default(),
            )
            .is_err(),
            "fixed coordinates concealed the original domain of {formula}"
        );
    }
}

fn affine_system(rows: &[[i64; 3]]) -> VerifiedResidualSystem {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let residuals: Vec<_> = rows
        .iter()
        .map(|&[a, b, c]| {
            Expr::Add(vec![
                Expr::Mul(vec![Expr::from_i64(a), Expr::Sym(x.clone())]),
                Expr::Mul(vec![Expr::from_i64(b), Expr::Sym(y.clone())]),
                Expr::from_i64(c),
            ])
        })
        .collect();
    VerifiedResidualSystem::try_compile(&residuals, &[x, y]).unwrap()
}

fn affine_cost(rows: &[[i64; 3]], point: [f64; 2]) -> f64 {
    rows.iter()
        .map(|&[a, b, c]| {
            let value = a as f64 * point[0] + b as f64 * point[1] + c as f64;
            value * value / 2.0
        })
        .sum()
}

/// A convex two-variable quadratic attains its box minimum in the interior,
/// on an edge, or at a corner. Enumerate those candidates in closed form.
fn face_reference(rows: &[[i64; 3]], lower: [f64; 2], upper: [f64; 2]) -> f64 {
    let (mut aa, mut ab, mut bb, mut ac, mut bc) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &[a, b, c] in rows {
        let (a, b, c) = (a as f64, b as f64, c as f64);
        aa += a * a;
        ab += a * b;
        bb += b * b;
        ac += a * c;
        bc += b * c;
    }
    let mut best = f64::INFINITY;
    let mut admit = |point: [f64; 2]| {
        if point
            .iter()
            .enumerate()
            .all(|(axis, &value)| lower[axis] <= value && value <= upper[axis])
        {
            best = best.min(affine_cost(rows, point));
        }
    };
    for x in [lower[0], upper[0]] {
        for y in [lower[1], upper[1]] {
            admit([x, y]);
        }
        if bb > 0.0 {
            admit([x, (-(ab * x + bc) / bb).clamp(lower[1], upper[1])]);
        }
    }
    for y in [lower[1], upper[1]] {
        if aa > 0.0 {
            admit([(-(ab * y + ac) / aa).clamp(lower[0], upper[0]), y]);
        }
    }
    let determinant = aa * bb - ab * ab;
    if determinant > 0.0 {
        admit([
            (ab * bc - bb * ac) / determinant,
            (ab * ac - aa * bc) / determinant,
        ]);
    }
    best
}

#[test]
fn coupled_affine_fits_match_independent_face_enumeration() {
    let cases = [
        [[1, 2, 1], [0, 1, -1], [0, 0, 0]],
        [[1, 2, 1], [0, 1, 1], [0, 0, 0]],
        [[1, 2, -3], [0, 1, -2], [1, -1, 1]],
        [[2, -3, -5], [1, 1, -1], [0, 0, 2]],
        [[1, -1, -1], [2, -2, -3], [0, 0, 1]],
        [[100, 1, -50], [1, 1, 2], [0, 0, 0]],
    ];
    let boxes = [([-1.0, -1.0], [1.0, 1.0]), ([0.0, -2.0], [2.0, 0.5])];
    let options = LeastSquaresOptions {
        max_iterations: 200,
        gradient_tolerance: 1e-8,
        ..LeastSquaresOptions::default()
    };
    for rows in cases {
        let source = affine_system(&rows);
        for (lower, upper) in boxes {
            let reference = face_reference(&rows, lower, upper);
            for initial in [lower, upper, [0.0, 0.0]] {
                let result = sparse_bounded_least_squares(
                    &source, &initial, &lower, &upper, &options,
                )
                .unwrap();
                for axis in 0..2 {
                    assert!(lower[axis] <= result.point[axis]);
                    assert!(result.point[axis] <= upper[axis]);
                }
                near(result.cost, reference, 1e-10 * (1.0 + reference));
                near(
                    result.cost,
                    affine_cost(&rows, [result.point[0], result.point[1]]),
                    1e-10 * (1.0 + reference),
                );
                assert!(result.projected_gradient_norm <= options.gradient_tolerance);
            }
        }
    }
}
