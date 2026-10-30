//! Complete finite systems that need a separating linear form.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::real_polysys::solve_real_shape_system;
use fsym_solvers::real_rur::{
    VerifiedRealRurSystem, metered_solve_real_rur_system, solve_real_rur_system,
};
use num_traits::Zero;
use std::sync::Arc;

fn vars(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn equations(sources: &[&str]) -> Vec<Expr> {
    sources
        .iter()
        .map(|source| parse(source).unwrap())
        .collect()
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn coeffs(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|&value| q(value, 1)).collect()
}

fn solve(sources: &[&str], names: &[&str]) -> VerifiedRealRurSystem {
    solve_real_rur_system(&equations(sources), &vars(names))
        .unwrap()
        .unwrap()
}

fn evaluate(coefficients: &[BigRational], point: &BigRational) -> BigRational {
    coefficients
        .iter()
        .rev()
        .fold(BigRational::zero(), |value, coefficient| {
            value * point + coefficient
        })
}

#[test]
fn independent_quadratic_coordinates_require_a_separating_parameter() {
    let input = equations(&["x**2-2", "y**2-3"]);
    let variables = vars(&["x", "y"]);
    assert!(
        solve_real_shape_system(&input, &variables)
            .unwrap()
            .is_none()
    );
    let result = solve_real_rur_system(&input, &variables).unwrap().unwrap();
    // t=x+y: (t^2-5)^2=24, x=(t^3-9t)/2, y=(11t-t^3)/2.
    assert_eq!(result.eliminant(), coeffs(&[1, 0, -10, 0, 1]));
    assert_eq!(
        result.coordinates(),
        [
            vec![q(0, 1), q(-9, 2), q(0, 1), q(1, 2)],
            vec![q(0, 1), q(11, 2), q(0, 1), q(-1, 2)],
        ]
    );
    assert_eq!(result.shear(), coeffs(&[1]));
    assert!(!result.variables().contains(result.parameter()));
    assert_eq!(result.variables(), variables);
    assert_eq!(result.equations().len(), input.len());
    assert_eq!(result.roots().intervals().len(), 4);
    assert_eq!(result.roots().count_with_multiplicity(), 4);
    assert_eq!(result.roots().coefficients(), result.eliminant());
    assert!(result.all_complex_roots_real());
}

#[test]
fn quintic_times_quadratic_has_ten_exact_real_solutions() {
    let input = equations(&["x**5+x**4-4*x**3-3*x**2+3*x+1", "y**2-2"]);
    let variables = vars(&["x", "y"]);
    assert!(
        solve_real_shape_system(&input, &variables)
            .unwrap()
            .is_none()
    );
    let result = solve_real_rur_system(&input, &variables).unwrap().unwrap();
    // The quintic alternates sign at -2,-3/2,-1,0,1,2, hence has five
    // distinct real roots. Each can be paired with either real sqrt(2).
    // Independently, p(t-y)=A(t)+B(t)y modulo y^2-2, where
    // A=t^5+t^4+16t^3+9t^2-t-1 and B=-5t^4-4t^3-8t^2-2t+1.
    // The literal expected eliminant is their exact norm A^2-2B^2.
    assert_eq!(
        result.eliminant(),
        coeffs(&[-1, 10, 7, -98, -93, 116, 80, -30, -17, 2, 1])
    );
    assert_eq!(result.shear(), coeffs(&[1]));
    assert_eq!(result.coordinates().len(), 2);
    assert_ne!(result.coordinates()[1], coeffs(&[0, 1]));
    assert_eq!(result.roots().intervals().len(), 10);
    assert_eq!(result.roots().count_with_multiplicity(), 10);
    assert!(result.all_complex_roots_real());
}

#[test]
fn nonseparating_first_forms_cannot_drop_colliding_points() {
    let result = solve(&["x**2-x", "y**2-y"], &["x", "y"]);
    // x+y and y-x identify two distinct points. The next form 2*x+y
    // separates all four: t=0,1,2,3 in the explicit expected order below.
    assert_eq!(result.shear(), coeffs(&[2]));
    assert_eq!(result.eliminant(), coeffs(&[0, -6, 11, -6, 1]));
    let expected = [(0, 0), (0, 1), (1, 0), (1, 1)];
    for (t, &(x, y)) in expected.iter().enumerate() {
        let point = q(t as i64, 1);
        assert_eq!(evaluate(&result.coordinates()[0], &point), q(x, 1));
        assert_eq!(evaluate(&result.coordinates()[1], &point), q(y, 1));
    }
    assert_eq!(result.roots().intervals().len(), 4);
    assert!(result.all_complex_roots_real());
}

#[test]
fn later_forms_recover_every_point_in_a_nine_point_grid() {
    let result = solve(&["x**3-x", "y**3-y"], &["x", "y"]);
    // Every slope with absolute value <=2 has collisions on this grid.
    // t=3*x+y takes each integer from -4 through 4 exactly once.
    assert_eq!(result.shear(), coeffs(&[3]));
    for x in -1..=1 {
        for y in -1..=1 {
            let point = q(3 * x + y, 1);
            assert!(evaluate(result.eliminant(), &point).is_zero());
            assert_eq!(evaluate(&result.coordinates()[0], &point), q(x, 1));
            assert_eq!(evaluate(&result.coordinates()[1], &point), q(y, 1));
        }
    }
    assert_eq!(result.eliminant().len(), 10);
    assert_eq!(result.roots().intervals().len(), 9);
    assert!(result.all_complex_roots_real());
}

#[test]
fn caller_order_and_redundant_original_constraints_are_preserved() {
    let ordinary = solve(&["x**2-2", "y**2-3"], &["x", "y"]);
    let redundant = solve(&["3*y**2-9", "0", "2*x**2/7-4/7", "x**2-2"], &["x", "y"]);
    assert_eq!(redundant.eliminant(), ordinary.eliminant());
    assert_eq!(redundant.coordinates(), ordinary.coordinates());
    assert_eq!(redundant.equations().len(), 4);
    let reversed = solve(&["x**2-2", "y**2-3"], &["y", "x"]);
    assert_eq!(reversed.variables(), vars(&["y", "x"]));
    assert_eq!(reversed.eliminant(), ordinary.eliminant());
    assert_eq!(reversed.coordinates()[0], ordinary.coordinates()[1]);
    assert_eq!(reversed.coordinates()[1], ordinary.coordinates()[0]);
}

#[test]
fn ordinary_shape_is_kept_without_a_coordinate_change() {
    let result = solve(&["x-y**2-1", "y**5+y**4-4*y**3-3*y**2+3*y+1"], &["x", "y"]);
    assert_eq!(result.parameter(), &Symbol::new("y"));
    assert_eq!(result.shear(), coeffs(&[0]));
    assert_eq!(result.coordinates(), [coeffs(&[1, 0, 1]), coeffs(&[0, 1])]);
    assert_eq!(result.eliminant(), coeffs(&[1, 3, -3, -4, 1, 1]));
    assert_eq!(result.roots().intervals().len(), 5);
    assert!(result.all_complex_roots_real());
    let univariate = solve(&["x**3-x-1"], &["x"]);
    assert_eq!(univariate.coordinates(), [coeffs(&[0, 1])]);
    assert!(univariate.shear().is_empty());
    assert_eq!(univariate.roots().intervals().len(), 1);
    assert!(!univariate.all_complex_roots_real());
}

#[test]
fn monogenic_nonradical_system_keeps_exact_multiplicities() {
    let result = solve(&["x**2", "y**2-3"], &["x", "y"]);
    // With t=x+y, (t^2-3)^2=0 and x=(t^3-3t)/6.
    assert_eq!(result.eliminant(), coeffs(&[9, 0, -6, 0, 1]));
    assert_eq!(
        result.coordinates(),
        [
            vec![q(0, 1), q(-1, 2), q(0, 1), q(1, 6)],
            vec![q(0, 1), q(3, 2), q(0, 1), q(-1, 6)],
        ]
    );
    assert_eq!(result.roots().intervals().len(), 2);
    assert_eq!(result.roots().count_with_multiplicity(), 4);
    assert!(
        result
            .roots()
            .intervals()
            .iter()
            .all(|root| root.multiplicity == 2)
    );
    assert!(result.all_complex_roots_real());
}

#[test]
fn nonreal_points_remain_visible_in_the_completeness_predicate() {
    let result = solve(&["x**2-2", "y**2+1"], &["x", "y"]);
    assert_eq!(result.eliminant(), coeffs(&[9, 0, -2, 0, 1]));
    assert!(result.roots().intervals().is_empty());
    assert!(!result.all_complex_roots_real());
    let mixed = solve(&["x**2-2", "y*(y-1)*(y**2+1)"], &["x", "y"]);
    assert_eq!(mixed.eliminant().len(), 9);
    assert_eq!(mixed.roots().intervals().len(), 4);
    assert!(!mixed.all_complex_roots_real());
}

#[test]
fn eight_variables_need_no_ninth_auxiliary_generator() {
    let names = ["a", "b", "c", "d", "e", "f", "x", "y"];
    let result = solve(&["a", "b", "c", "d", "e", "f", "x**2-2", "y**2-3"], &names);
    assert_eq!(result.variables(), vars(&names));
    assert_eq!(result.basis().input_basis()[0].generators().len(), 8);
    assert_eq!(result.coordinates().len(), 8);
    assert!(
        result.coordinates()[..6]
            .iter()
            .all(|coordinate| coordinate == &coeffs(&[0]))
    );
    assert_eq!(result.eliminant(), coeffs(&[1, 0, -10, 0, 1]));
    assert_eq!(result.roots().intervals().len(), 4);
    assert!(result.all_complex_roots_real());
}

#[test]
fn typed_identity_and_internal_parameter_freshness_survive_shearing() {
    let plain = Symbol::new("same");
    let assumed = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([91; 32]));
    let first = Expr::Sym(plain.clone());
    let second = Expr::Sym(assumed.clone());
    let result = solve_real_rur_system(
        &[
            first.clone() * first - Expr::from_i64(2),
            second.clone() * second - Expr::from_i64(3),
        ],
        &[plain.clone(), assumed.clone()],
    )
    .unwrap()
    .unwrap();
    assert_eq!(result.variables(), [plain, assumed]);
    assert_eq!(result.roots().intervals().len(), 4);
    let named = solve(
        &["_fsym_rur_parameter_0**2-2", "y**2-3"],
        &["_fsym_rur_parameter_0", "y"],
    );
    assert_eq!(named.parameter().name, "_fsym_rur_parameter_1");
    assert!(
        named
            .variables()
            .iter()
            .all(|variable| variable != named.parameter())
    );
}

#[test]
fn unit_ideal_empty_results_and_unsupported_ideals_stay_distinct() {
    let empty = solve(&["x**2-2", "y**2-3", "1"], &["x", "y"]);
    assert_eq!(empty.eliminant(), coeffs(&[1]));
    assert!(empty.roots().intervals().is_empty());
    assert!(empty.all_complex_roots_real());
    for input in [
        equations(&["x*y-1"]),
        equations(&["x-1"]),
        equations(&["0"]),
        vec![],
        // The three-dimensional local quotient has two independent square
        // zero directions, so no linear form generates the whole quotient.
        equations(&["x**2", "x*y", "y**2"]),
    ] {
        assert!(
            solve_real_rur_system(&input, &vars(&["x", "y"]))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn full_original_domain_admission_precedes_every_dispatch_decision() {
    let variables = vars(&["x", "y"]);
    for exponent in [-1, 0] {
        let invalid = Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Pow(
                Arc::new(Expr::from_i64(0)),
                Arc::new(Expr::from_i64(exponent)),
            ),
        ]);
        for prefix in [
            equations(&["1"]),
            equations(&["x*y"]),
            equations(&["x**2-2", "y**2-3"]),
        ] {
            let mut input = prefix;
            input.push(invalid.clone());
            assert!(solve_real_rur_system(&input, &variables).is_err());
        }
    }
    let unsupported = Expr::Function("opaque".into(), vec![Expr::Sym(variables[0].clone())]);
    assert!(solve_real_rur_system(&[Expr::from_i64(1), unsupported], &variables).is_err());
}

#[test]
fn original_size_and_final_eliminant_limits_are_real_refusals() {
    let x = Symbol::new("x");
    assert!(solve_real_rur_system(&equations(&["x"]), &[]).is_err());
    assert!(solve_real_rur_system(&equations(&["x"]), &[x.clone(), x.clone()]).is_err());
    assert!(solve_real_rur_system(&vec![Expr::from_i64(0); 65], std::slice::from_ref(&x)).is_err());
    let oversized: Vec<_> = (0..9)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    assert!(solve_real_rur_system(&[Expr::from_i64(1)], &oversized).is_err());
    assert!(solve_real_rur_system(&equations(&["x**65-1"]), std::slice::from_ref(&x)).is_err());
    let huge = Expr::Integer(BigInt::one() << 4097);
    assert!(solve_real_rur_system(&[Expr::from_i64(1), huge], std::slice::from_ref(&x)).is_err());
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    compute: u64,
    memory: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
            _ => {}
        }
        Ok(())
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_propagates_during_search_verification_and_final_publication() {
    let variables = vars(&["x", "y"]);
    for input in [
        equations(&["x**2-x", "y**2-y"]),
        equations(&["x+y-1", "x+y-2"]),
        equations(&["x*y-1"]),
    ] {
        let (mut generator, mut verifier) = (Meter::default(), Meter::default());
        metered_solve_real_rur_system(&input, &variables, &mut generator, &mut verifier).unwrap();
        assert!(generator.compute > 0 && generator.memory > 0);
        assert!(verifier.compute > 0 && verifier.memory > 0);
        for lane in 0..2 {
            let count = if lane == 0 {
                generator.checkpoints
            } else {
                verifier.checkpoints
            };
            for cancel_at in [1, count / 2, count] {
                let (mut generated, mut checked) = (Meter::default(), Meter::default());
                if lane == 0 {
                    generated.cancel_at = Some(cancel_at);
                } else {
                    checked.cancel_at = Some(cancel_at);
                }
                assert!(metered_solve_real_rur_system(
                    &input, &variables, &mut generated, &mut checked,
                ).is_err(), "lane {lane}, checkpoint {cancel_at}");
            }
        }
    }
}

#[test]
fn actual_budget_exhaustion_never_becomes_a_partial_or_empty_answer() {
    let input = equations(&["x**2-x", "y**2-y"]);
    let variables = vars(&["x", "y"]);
    for lane in 0..2 {
        let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
        let result = if lane == 0 {
            metered_solve_real_rur_system(&input, &variables, &mut budget, &mut Unbounded)
        } else {
            metered_solve_real_rur_system(&input, &variables, &mut Unbounded, &mut budget)
        };
        assert!(result.is_err());
    }
}
