use fsym_core::elementary::eval_pow;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_solvers::{SolverError, solve, univariate::solve_univariate};
use std::collections::HashMap;

fn q(numerator: i64, denominator: i64) -> Expr {
    Expr::Rational(BigRational::new(numerator.into(), denominator.into()))
}

fn phase(numerator: i64, denominator: i64) -> Expr {
    eval_pow(Expr::from_i64(-1), q(numerator, denominator))
}

fn assert_same_roots(actual: &[Expr], expected: &[Expr]) {
    assert_eq!(actual.len(), expected.len(), "roots: {actual:?}");
    for root in expected {
        assert!(
            actual.iter().any(|candidate| {
                fsym_simplify::simplify(&fsym_simplify::expand(&(candidate.clone() - root.clone())))
                    .is_zero()
            }),
            "missing {root:?} from {actual:?}"
        );
    }
}

#[test]
fn negative_target_has_both_principal_three_halves_inverse_branches() {
    let x = Symbol::new("x");
    let equation = eval_pow(Expr::Sym(x.clone()), q(3, 2)) + Expr::from_i64(1);
    let roots = solve(&equation, &x).unwrap();
    assert_same_roots(&roots, &[phase(-2, 3), phase(2, 3)]);
}

#[test]
fn positive_target_keeps_complex_five_halves_and_seven_thirds_branches() {
    let x = Symbol::new("x");
    for (numerator, denominator, angle_numerator, angle_denominator) in [(5, 2, 4, 5), (7, 3, 6, 7)]
    {
        let equation =
            eval_pow(Expr::Sym(x.clone()), q(numerator, denominator)) - Expr::from_i64(1);
        let roots = solve(&equation, &x).unwrap();
        assert_same_roots(
            &roots,
            &[
                Expr::from_i64(1),
                phase(-angle_numerator, angle_denominator),
                phase(angle_numerator, angle_denominator),
            ],
        );
    }
}

#[test]
fn rational_magnitudes_and_affine_bases_reach_the_original_equation_checker() {
    let x = Symbol::new("x");
    let base = Expr::from_i64(2) * Expr::Sym(x.clone()) + Expr::from_i64(3);
    let equation = eval_pow(base, q(3, 2)) + Expr::from_i64(8);
    let roots = solve(&equation, &x).unwrap();
    let expected = [
        q(1, 2) * (Expr::from_i64(4) * phase(-2, 3) - Expr::from_i64(3)),
        q(1, 2) * (Expr::from_i64(4) * phase(2, 3) - Expr::from_i64(3)),
    ];
    assert_same_roots(&roots, &expected);
    for root in roots {
        let residual = equation.subs(&HashMap::from([(x.clone(), root)]));
        assert!(fsym_simplify::simplify(&residual).is_zero());
    }
}

#[test]
fn negative_exponents_keep_every_nonzero_branch() {
    let x = Symbol::new("x");
    let equation = eval_pow(Expr::Sym(x.clone()), q(-3, 2)) + Expr::from_i64(8);
    let roots = solve(&equation, &x).unwrap();
    assert_same_roots(&roots, &[q(1, 4) * phase(-2, 3), q(1, 4) * phase(2, 3)]);
    let equation = eval_pow(Expr::Sym(x.clone()), q(-5, 2)) - Expr::from_i64(1);
    let roots = solve(&equation, &x).unwrap();
    assert_same_roots(&roots, &[Expr::from_i64(1), phase(-4, 5), phase(4, 5)]);
}

#[test]
fn principal_range_excludes_algebraic_roots_on_other_sheets() {
    let x = Symbol::new("x");
    // Raising both sides to the denominator would give three algebraic
    // roots, but only one satisfies the original principal power.
    let equation = eval_pow(Expr::Sym(x.clone()), q(3, 2)) - Expr::from_i64(1);
    assert_eq!(solve(&equation, &x), Ok(vec![Expr::from_i64(1)]));
    for exponent in [q(1, 2), q(2, 3), q(-2, 3)] {
        let equation = eval_pow(Expr::Sym(x.clone()), exponent) + Expr::from_i64(1);
        assert_eq!(solve(&equation, &x), Ok(vec![]));
    }
}

#[test]
fn zero_target_distinguishes_positive_and_negative_powers() {
    let x = Symbol::new("x");
    let base = Expr::Sym(x.clone()) + Expr::from_i64(2);
    assert_eq!(
        solve(&eval_pow(base.clone(), q(3, 2)), &x),
        Ok(vec![Expr::from_i64(-2)])
    );
    assert_eq!(solve(&eval_pow(base, q(-3, 2)), &x), Ok(vec![]));
}

#[test]
fn unresolved_target_phase_is_incomplete_through_both_entry_points() {
    let x = Symbol::new("x");
    let equation = eval_pow(Expr::Sym(x.clone()), q(3, 2)) - Expr::symbol("parameter");
    let direct = solve_univariate(&equation, &x);
    assert!(matches!(direct, Err(SolverError::IncompleteSolutionSet(_))));
    assert_eq!(solve(&equation, &x), direct);
}

#[test]
fn bounded_branch_enumeration_refuses_without_returning_a_partial_list() {
    let x = Symbol::new("x");
    let equation = eval_pow(Expr::Sym(x.clone()), q(13, 2)) - Expr::from_i64(1);
    let direct = solve_univariate(&equation, &x);
    assert!(matches!(direct, Err(SolverError::IncompleteSolutionSet(_))));
    assert_eq!(solve(&equation, &x), direct);
}

#[test]
fn phase_counts_agree_with_closed_principal_sector_counts() {
    let x = Symbol::new("sector_coordinate");
    // For a reduced noninteger a=p/q, the positive real preimage contains
    // 1+2*floor(|a|/2) roots. The negative real preimage contains
    // 2*floor((|a|+1)/2). These closed counts are independent of enumerating k.
    for numerator in 1i64..=12 {
        for denominator in 2i64..=5 {
            let exponent = BigRational::new(numerator.into(), denominator.into());
            if exponent.is_integer() {
                continue;
            }
            for sign in [-1, 1] {
                for (target, expected) in [
                    (1, 1 + 2 * (numerator / (2 * denominator))),
                    (-1, 2 * ((numerator + denominator) / (2 * denominator))),
                ] {
                    let equation = eval_pow(Expr::Sym(x.clone()), q(sign * numerator, denominator))
                        - Expr::from_i64(target);
                    let roots = solve(&equation, &x).unwrap_or_else(|error| {
                        panic!("x^({}/{})={target}: {error}", sign * numerator, denominator)
                    });
                    assert_eq!(
                        roots.len(),
                        expected as usize,
                        "x^({}/{})={target}",
                        sign * numerator,
                        denominator
                    );
                    for root in roots {
                        let residual = equation.subs(&HashMap::from([(x.clone(), root)]));
                        assert!(fsym_simplify::simplify(&residual).is_zero());
                    }
                }
            }
        }
    }
}
