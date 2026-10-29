//! Complete shape representations, exact real counting and source adversaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{TermOrder, bounded_polynomial_from_expr, checked_groebner_basis};
use fsym_solvers::real_polysys::{
    VerifiedRealShapeSystem, metered_real_shape_from_basis, metered_solve_real_shape_system,
    solve_real_shape_system, verify_shape_representation,
};
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

fn rational(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn coeffs(values: &[i64]) -> Vec<BigRational> {
    values.iter().map(|value| rational(*value, 1)).collect()
}

fn shape(sources: &[&str], names: &[&str]) -> VerifiedRealShapeSystem {
    solve_real_shape_system(&equations(sources), &vars(names))
        .unwrap()
        .unwrap()
}

#[test]
fn quintic_shape_has_every_exact_real_solution_without_radical_formulas() {
    // The quintic changes sign successively at -2, -3/2, -1, 0, 1, 2.
    // Degree five therefore proves five distinct real roots independently
    // of any numerical root approximation.
    let result = shape(
        &["x-y**2-1", "z-2*y+1", "y**5+y**4-4*y**3-3*y**2+3*y+1"],
        &["x", "z", "y"],
    );
    assert_eq!(result.eliminant(), coeffs(&[1, 3, -3, -4, 1, 1]));
    assert_eq!(
        result.coordinates(),
        [coeffs(&[1, 0, 1]), coeffs(&[-1, 2]), coeffs(&[0, 1])]
    );
    assert_eq!(result.roots().intervals().len(), 5);
    assert_eq!(result.roots().count_with_multiplicity(), 5);
    assert_eq!(result.roots().coefficients(), result.eliminant());
    assert!(result.all_complex_roots_real());
    assert_eq!(result.variables(), vars(&["x", "z", "y"]));
}

#[test]
fn source_order_and_rational_scaling_do_not_change_the_complete_shape() {
    let ordinary = shape(&["x-y**2-1", "y**3-3*y"], &["x", "y"]);
    let redundant = shape(
        &["7*y**3-21*y", "2*x/3-2*y**2/3-2/3", "x-y**2-1", "0"],
        &["x", "y"],
    );
    assert_eq!(ordinary.eliminant(), redundant.eliminant());
    assert_eq!(ordinary.coordinates(), redundant.coordinates());
    assert_eq!(ordinary.roots(), redundant.roots());
    assert!(ordinary.all_complex_roots_real());
}

#[test]
fn repeated_eliminant_roots_preserve_multiplicity_and_distinct_tuples() {
    let result = shape(&["x-y", "(y**2-2)**3"], &["x", "y"]);
    assert_eq!(result.roots().intervals().len(), 2);
    assert_eq!(result.roots().count_with_multiplicity(), 6);
    assert!(
        result
            .roots()
            .intervals()
            .iter()
            .all(|root| root.multiplicity == 3)
    );
    assert!(result.all_complex_roots_real());
    assert_eq!(result.coordinates(), [coeffs(&[0, 1]), coeffs(&[0, 1])]);
}

#[test]
fn nonreal_roots_never_become_a_complete_complex_answer() {
    let mixed = shape(&["x-y", "(y-2)*(y**2+1)"], &["x", "y"]);
    assert_eq!(mixed.roots().intervals().len(), 1);
    assert_eq!(mixed.roots().count_with_multiplicity(), 1);
    assert!(!mixed.all_complex_roots_real());
    let no_real = shape(&["x-y-1", "y**2+1"], &["x", "y"]);
    assert!(no_real.roots().intervals().is_empty());
    assert!(!no_real.all_complex_roots_real());
}

#[test]
fn unit_ideal_is_a_verified_empty_real_and_complex_solution_set() {
    let result = shape(&["x+y-1", "x+y-2"], &["x", "y"]);
    assert_eq!(result.eliminant(), coeffs(&[1]));
    assert_eq!(result.coordinates(), [coeffs(&[0]), coeffs(&[0, 1])]);
    assert!(result.roots().intervals().is_empty());
    assert!(result.all_complex_roots_real());
    verify_shape_representation(
        result.basis(),
        result.eliminant(),
        result.coordinates(),
        &mut Unbounded,
    )
    .unwrap();
}

#[test]
fn nonshape_and_positive_dimensional_ideals_remain_explicitly_unsupported() {
    for input in [
        equations(&["x**2-2", "y**2-3"]),
        equations(&["x*y"]),
        equations(&["x-1"]),
        equations(&["0"]),
        vec![],
    ] {
        assert!(
            solve_real_shape_system(&input, &vars(&["x", "y"]))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn typed_generator_identity_and_requested_order_are_preserved() {
    let first = Symbol::new("same");
    let second = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([73; 32]));
    let se = Expr::Sym(second.clone());
    let input = [
        Expr::Sym(first.clone()) - se.clone() * se.clone() - Expr::from_i64(1),
        se.clone() * se - Expr::from_i64(2),
    ];
    let result = solve_real_shape_system(&input, &[first.clone(), second.clone()])
        .unwrap()
        .unwrap();
    assert_eq!(result.variables(), [first, second]);
    assert_eq!(result.coordinates(), [coeffs(&[3]), coeffs(&[0, 1])]);
    let reordered = shape(&["x-2*y+1", "y**2-2"], &["y", "x"]);
    assert_eq!(reordered.eliminant(), coeffs(&[-7, 2, 1]));
    assert_eq!(
        reordered.coordinates(),
        [vec![rational(1, 2), rational(1, 2)], coeffs(&[0, 1])]
    );
    assert!(reordered.all_complex_roots_real());
}

#[test]
fn independent_reconstruction_rejects_coordinate_eliminant_and_source_mutations() {
    let result = shape(&["x-y**2-1", "y**3-3*y"], &["x", "y"]);
    let mut changed = result.coordinates().to_vec();
    changed[0][0] += BigRational::new(1.into(), BigInt::from(10).pow(80));
    assert!(
        verify_shape_representation(result.basis(), result.eliminant(), &changed, &mut Unbounded,)
            .is_err()
    );
    let mut changed_eliminant = result.eliminant().to_vec();
    changed_eliminant[0] += rational(1, 1);
    assert!(
        verify_shape_representation(
            result.basis(),
            &changed_eliminant,
            result.coordinates(),
            &mut Unbounded,
        )
        .is_err()
    );
    let mut swapped = result.coordinates().to_vec();
    swapped.swap(0, 1);
    assert!(
        verify_shape_representation(result.basis(), result.eliminant(), &swapped, &mut Unbounded,)
            .is_err()
    );
    let other = shape(&["x-y**2-2", "y**3-3*y"], &["x", "y"]);
    assert!(
        verify_shape_representation(
            other.basis(),
            result.eliminant(),
            result.coordinates(),
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn independent_reconstruction_rejects_noncanonical_and_oversized_proposals() {
    let result = shape(&["x-y", "y**2-2"], &["x", "y"]);
    for wrong in [vec![], coeffs(&[0]), coeffs(&[1]), coeffs(&[-4, 0, 2])] {
        assert!(
            verify_shape_representation(
                result.basis(),
                &wrong,
                result.coordinates(),
                &mut Unbounded,
            )
            .is_err()
        );
    }
    let mut padded = result.eliminant().to_vec();
    padded.push(rational(0, 1));
    assert!(
        verify_shape_representation(
            result.basis(),
            &padded,
            result.coordinates(),
            &mut Unbounded,
        )
        .is_err()
    );
    let mut huge = result.coordinates().to_vec();
    huge[0] = vec![rational(1, 1); 66];
    assert!(
        verify_shape_representation(result.basis(), result.eliminant(), &huge, &mut Unbounded)
            .is_err()
    );
    huge[0] = vec![BigRational::from_integer(BigInt::from(1) << 4097)];
    assert!(
        verify_shape_representation(result.basis(), result.eliminant(), &huge, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn every_original_is_admitted_before_unit_or_nonshape_dispatch() {
    let x = Symbol::new("x");
    let bad = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
    ]);
    for first in [Expr::from_i64(1), Expr::from_i64(0), Expr::Sym(x.clone())] {
        assert!(solve_real_shape_system(&[first, bad.clone()], &[x.clone()]).is_err());
    }
    let opaque = Expr::Function("opaque".into(), vec![Expr::Sym(x.clone())]);
    assert!(solve_real_shape_system(&[Expr::from_i64(1), opaque], &[x]).is_err());
}

#[test]
fn source_dimensions_and_eliminant_degree_are_bounded() {
    let x = Symbol::new("x");
    assert!(solve_real_shape_system(&equations(&["x**65-1"]), &[x.clone()]).is_err());
    assert!(solve_real_shape_system(&equations(&["x"]), &[]).is_err());
    assert!(solve_real_shape_system(&equations(&["x"]), &[x.clone(), x.clone()]).is_err());
    assert!(solve_real_shape_system(&vec![Expr::Sym(x.clone()); 65], &[x.clone()]).is_err());
    let too_many: Vec<_> = (0..9)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    assert!(solve_real_shape_system(&[Expr::from_i64(1)], &too_many).is_err());
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    compute: u64,
    memory: u64,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.compute += amount,
            Dimension::MemoryBytes => self.memory += amount,
            _ => {}
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
fn existing_checked_basis_is_reused_with_cumulative_work_and_typed_order() {
    let variables = vars(&["x", "y"]);
    let input = equations(&["x-y**2-1", "y**3-3*y"]);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    let polys = input
        .iter()
        .map(|expression| {
            bounded_polynomial_from_expr(expression, &variables, &mut generation).unwrap()
        })
        .collect::<Vec<_>>();
    let checked =
        checked_groebner_basis(&polys, TermOrder::Lex, &mut generation, &mut verification).unwrap();
    let generated_before = generation.compute;
    let verified_before = verification.compute;
    let result = metered_real_shape_from_basis(checked, &mut generation, &mut verification)
        .unwrap()
        .unwrap();
    assert_eq!(result.variables(), variables);
    assert_eq!(result.roots().intervals().len(), 3);
    assert!(generation.compute > generated_before);
    assert!(verification.compute > verified_before);
}

#[test]
fn separate_meters_cancel_during_both_phases_and_at_publication() {
    let input = equations(&["x-y", "y**2-2"]);
    let variables = vars(&["x", "y"]);
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    metered_solve_real_shape_system(&input, &variables, &mut generation, &mut verification)
        .unwrap()
        .unwrap();
    assert!(generation.compute > 0 && generation.memory > 0);
    assert!(verification.compute > 0 && verification.memory > 0);
    for point in [1, generation.checkpoints / 2, generation.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_solve_real_shape_system(&input, &variables, &mut stopped, &mut Unbounded)
                .is_err()
        );
    }
    for point in [1, verification.checkpoints / 2, verification.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(point),
            ..Meter::default()
        };
        assert!(
            metered_solve_real_shape_system(&input, &variables, &mut Unbounded, &mut stopped)
                .is_err()
        );
    }
}

#[test]
fn real_budget_exhaustion_does_not_publish_a_partial_representation() {
    let input = equations(&["x-y", "y**2-2"]);
    let variables = vars(&["x", "y"]);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        metered_solve_real_shape_system(&input, &variables, &mut tiny, &mut untouched).is_err()
    );
    assert_eq!(untouched.checkpoints, 0);
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_solve_real_shape_system(&input, &variables, &mut Unbounded, &mut tiny).is_err()
    );
}
