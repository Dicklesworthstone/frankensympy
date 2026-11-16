//! Complete original-domain rational solution sets through the public native facade.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::metered_localize_rational_system;
use fsym_solvers::real_rational::{VerifiedRealRationalSystem, metered_solve_real_rational_system};
use num_traits::Zero;
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn variables(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn expression(source: &str) -> Expr {
    parse(source).unwrap()
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn ratio(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, power(denominator, -1)])
}

fn solve(input: &[Expr], variables: &[Symbol]) -> VerifiedRealRationalSystem {
    metered_solve_real_rational_system(input, variables, &mut Unbounded, &mut Unbounded)
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

fn single_tuple(receipt: &VerifiedRealRationalSystem) -> Vec<BigRational> {
    assert_eq!(receipt.roots().intervals().len(), 1);
    assert_eq!(receipt.eliminant().len(), 2);
    let root = -&receipt.eliminant()[0] / &receipt.eliminant()[1];
    receipt
        .coordinates()
        .iter()
        .map(|coordinate| evaluate(coordinate, &root))
        .collect()
}

#[test]
fn cancelled_factors_and_multiple_denominators_remove_only_original_poles() {
    let vars = variables(&["x", "y"]);
    let input = [
        ratio(expression("x^2-1"), expression("x-1")),
        expression("y-x"),
    ];
    let receipt = solve(&input, &vars);
    assert_eq!(single_tuple(&receipt), [q(-1, 1), q(-1, 1)]);
    assert_eq!(receipt.variables(), vars);
    assert_eq!(receipt.coordinates().len(), 2);
    assert_eq!(receipt.representation().coordinates().len(), 3);
    assert_eq!(receipt.localization().unwrap().sources(), input);

    let input = [
        ratio(expression("(x-1)*(x+1)*(x-2)"), expression("x-1")),
        ratio(expression("y-x"), expression("x+1")),
    ];
    let receipt = solve(&input, &vars);
    assert_eq!(single_tuple(&receipt), [q(2, 1), q(2, 1)]);
    assert_eq!(receipt.localization().unwrap().guards().len(), 2);
}

#[test]
fn held_zero_and_zero_power_keep_hidden_inverse_domains() {
    let input = [
        Expr::Add(vec![
            expression("x"),
            Expr::Mul(vec![Expr::from_i64(0), power(expression("y"), -1)]),
        ]),
        expression("y*(y-1)"),
    ];
    let receipt = solve(&input, &variables(&["x", "y"]));
    assert_eq!(single_tuple(&receipt), [q(0, 1), q(1, 1)]);

    let input = [
        Expr::Add(vec![
            power(power(expression("x"), -1), 0),
            Expr::from_i64(-1),
        ]),
        expression("x*(x-1)"),
    ];
    assert_eq!(
        single_tuple(&solve(&input, &variables(&["x"]))),
        [q(1, 1)]
    );
}

#[test]
fn nested_inverse_and_excluded_only_root_have_complete_empty_sets() {
    for input in [
        vec![power(power(expression("x"), -1), -1)],
        vec![
            expression("x"),
            Expr::Mul(vec![Expr::from_i64(0), power(expression("x"), -1)]),
        ],
    ] {
        let receipt = solve(&input, &variables(&["x"]));
        assert!(receipt.roots().intervals().is_empty());
        assert_eq!(receipt.variables().len(), 1);
        assert_eq!(receipt.coordinates().len(), 1);
        assert!(receipt.localization().is_some());
    }
}

#[test]
fn irrational_branches_exclude_the_diagonal_without_losing_cross_points() {
    let input = [
        ratio(expression("x^2-2"), expression("x-y")),
        expression("y^2-2"),
    ];
    let receipt = solve(&input, &variables(&["x", "y"]));
    // The four polynomial grid points reduce to (sqrt(2),-sqrt(2))
    // and (-sqrt(2),sqrt(2)). With t=1/(x-y), t^2=1/8 and
    // x=4t, y=-4t. These literal maps establish the expected exact set.
    assert_eq!(receipt.eliminant(), [q(-1, 8), q(0, 1), q(1, 1)]);
    assert_eq!(
        receipt.coordinates(),
        [vec![q(0, 1), q(4, 1)], vec![q(0, 1), q(-4, 1)]]
    );
    assert_eq!(receipt.roots().intervals().len(), 2);
    assert!(receipt.all_complex_roots_real());
}

#[test]
fn all_five_quintic_roots_survive_projection_without_radical_formulas() {
    let input = [
        ratio(
            expression("x^5+x^4-4*x^3-3*x^2+3*x+1"),
            expression("x"),
        ),
        expression("y-x"),
    ];
    let receipt = solve(&input, &variables(&["x", "y"]));
    assert_eq!(receipt.roots().intervals().len(), 5);
    assert!(receipt.all_complex_roots_real());
    assert_eq!(receipt.coordinates().len(), 2);
    assert_eq!(receipt.coordinates()[0], receipt.coordinates()[1]);
    assert_eq!(
        receipt.eliminant(),
        [1, 1, -4, -3, 3, 1].map(BigRational::from)
    );
    let localization = receipt.localization().unwrap();
    assert_eq!(receipt.parameter(), &localization.certificate().auxiliary);
    assert!(!receipt.variables().contains(receipt.parameter()));
}

#[test]
fn no_real_roots_does_not_assert_the_complex_locus_is_empty() {
    let input = [
        ratio(expression("x^2+1"), expression("x")),
        expression("y-x"),
    ];
    let receipt = solve(&input, &variables(&["x", "y"]));
    assert!(receipt.roots().intervals().is_empty());
    assert!(!receipt.all_complex_roots_real());
    assert_eq!(receipt.coordinates().len(), 2);
}

#[test]
fn typed_identity_and_caller_order_survive_auxiliary_projection() {
    let plain = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([81; 32]));
    let input = [
        ratio(
            Expr::Sym(plain.clone()) - Expr::from_i64(1),
            Expr::Sym(other.clone()),
        ),
        Expr::Sym(other.clone()) - Expr::from_i64(2),
    ];
    let ordered = [other.clone(), plain.clone()];
    let receipt = solve(&input, &ordered);
    assert_eq!(receipt.variables(), ordered);
    assert_eq!(single_tuple(&receipt), [q(2, 1), q(1, 1)]);
    assert_eq!(receipt.localization().unwrap().original_variables(), ordered);

    let collision = Symbol::new("_fsym_rational_inverse_0");
    let input = [ratio(
        Expr::Sym(collision.clone()) - Expr::from_i64(2),
        Expr::Sym(collision.clone()),
    )];
    let receipt = solve(&input, std::slice::from_ref(&collision));
    assert_eq!(receipt.variables(), [collision]);
    assert_eq!(single_tuple(&receipt), [q(2, 1)]);
    assert_eq!(
        receipt.localization().unwrap().certificate().auxiliary.name,
        "_fsym_rational_inverse_1"
    );
}

#[test]
fn polynomial_inputs_keep_the_existing_complete_nonshape_path() {
    let input = [expression("x^2-2"), expression("y^2-3")];
    let receipt = solve(&input, &variables(&["x", "y"]));
    assert!(receipt.localization().is_none());
    assert_eq!(receipt.roots().intervals().len(), 4);
    assert_eq!(receipt.variables().len(), 2);
    assert_eq!(receipt.representation().variables(), receipt.variables());
    assert_eq!(
        receipt.eliminant(),
        [1, 0, -10, 0, 1].map(BigRational::from)
    );
}

#[test]
fn polynomial_bounds_are_not_reduced_by_the_rational_auxiliary_reservation() {
    let vars = (0..8)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect::<Vec<_>>();
    let input = vars
        .iter()
        .enumerate()
        .map(|(index, variable)| Expr::Sym(variable.clone()) - Expr::from_i64(index as i64))
        .collect::<Vec<_>>();
    let receipt = solve(&input, &vars);
    assert!(receipt.localization().is_none());
    assert_eq!(
        single_tuple(&receipt),
        (0..8).map(|i| q(i, 1)).collect::<Vec<_>>()
    );
    let input = vec![expression("x-1"); 64];
    assert_eq!(
        single_tuple(&solve(&input, &variables(&["x"]))),
        [q(1, 1)]
    );

    let rational = ratio(expression("x0-1"), expression("x0"));
    assert!(
        metered_solve_real_rational_system(
            &[rational.clone()],
            &vars,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    assert!(
        metered_solve_real_rational_system(
            &vec![rational; 64],
            &vars[..1],
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn positive_dimensional_punctured_families_never_become_partial_or_empty_sets() {
    for (input, vars) in [
        (
            vec![ratio(expression("x"), expression("y"))],
            variables(&["x", "y"]),
        ),
        (
            vec![Expr::Mul(vec![
                Expr::from_i64(0),
                power(expression("x"), -1),
            ])],
            variables(&["x"]),
        ),
        (
            vec![Expr::Add(vec![
                ratio(expression("x"), expression("x")),
                Expr::from_i64(-1),
            ])],
            variables(&["x"]),
        ),
    ] {
        assert!(
            metered_solve_real_rational_system(
                &input,
                &vars,
                &mut Unbounded,
                &mut Unbounded,
            )
            .unwrap()
            .is_none()
        );
    }
}

#[test]
fn invalid_originals_cannot_hide_behind_inconsistency_or_polynomial_routing() {
    let vars = variables(&["x"]);
    let zero = Expr::Add(vec![expression("x"), -expression("x")]);
    for invalid in [
        Expr::Mul(vec![Expr::from_i64(0), power(zero, -1)]),
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
        Expr::Mul(vec![Expr::from_i64(0), expression("sin(x)")]),
        expression("foreign"),
    ] {
        assert!(
            metered_solve_real_rational_system(
                &[Expr::from_i64(1), invalid],
                &vars,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
    assert!(
        metered_solve_real_rational_system(
            &[expression("x-1")],
            &[vars[0].clone(), vars[0].clone()],
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[derive(Default)]
struct CountingMeter {
    compute: u64,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CountingMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.compute += amount;
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
        if self.cancel_at == Some(self.checkpoints) {
            return Err(MeterError::Cancelled);
        }
        self.checkpoints += 1;
        Ok(())
    }
}

#[test]
fn localization_and_solving_cannot_reset_either_cumulative_meter() {
    let input = [ratio(expression("x^2-1"), expression("x-1"))];
    let vars = variables(&["x"]);
    let mut generator = CountingMeter::default();
    let mut verifier = CountingMeter::default();
    metered_localize_rational_system(&input, &vars, &mut generator, &mut verifier).unwrap();
    let localization_spent = [generator.compute, verifier.compute];
    for (lane, allowance) in localization_spent.into_iter().enumerate() {
        assert!(allowance > 0);
        let mut limits = BudgetLimits::uniform(1024 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = allowance;
        let mut limited = Budget::new(limits);
        let result = if lane == 0 {
            metered_solve_real_rational_system(&input, &vars, &mut limited, &mut Unbounded)
        } else {
            metered_solve_real_rational_system(&input, &vars, &mut Unbounded, &mut limited)
        };
        assert!(result.is_err());
        assert!(limited.snapshot().dimensions[Dimension::ComputeSteps.index()] < allowance);
    }
}

#[test]
fn cancellation_before_during_and_after_both_phases_never_publishes_a_set() {
    let input = [ratio(expression("x^2-1"), expression("x-1"))];
    let vars = variables(&["x"]);
    let mut generator = CountingMeter::default();
    let mut verifier = CountingMeter::default();
    metered_solve_real_rational_system(&input, &vars, &mut generator, &mut verifier).unwrap();
    for (lane, calls) in [generator.checkpoints, verifier.checkpoints]
        .into_iter()
        .enumerate()
    {
        for cancel_at in [0, 1, calls / 2, calls - 1] {
            let mut cancelled = CountingMeter {
                cancel_at: Some(cancel_at),
                ..CountingMeter::default()
            };
            let result = if lane == 0 {
                metered_solve_real_rational_system(&input, &vars, &mut cancelled, &mut Unbounded)
            } else {
                metered_solve_real_rational_system(&input, &vars, &mut Unbounded, &mut cancelled)
            };
            let error = result.unwrap_err().to_string();
            assert!(error.contains("cancel"), "{error}");
        }
    }
}
