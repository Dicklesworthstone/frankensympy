//! Global box claims are checked against fixed analytic inequalities, including
//! boundary optima, semidefinite curvature and deliberately false candidates.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, BoxExtremumError, BoxExtremumKind, BoxExtremumReceipt, Expr, RealBall,
    RootBoxError, Symbol, certify_box_extremum, verify_box_extremum,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn bounds(lo: i64, hi: i64) -> RealBall {
    RealBall::new(q(lo + hi, 2), q(hi - lo, 2)).unwrap()
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn minimum(
    source: &Expr,
    vars: &[Symbol],
    domain: &[RealBall],
    point: &[BigRational],
) -> Result<BoxExtremumReceipt, BoxExtremumError> {
    certify_box_extremum(source, vars, domain, point, BoxExtremumKind::Minimum, 24)
}

#[test]
fn interior_coupled_quadratic_has_unique_global_minimum_on_the_whole_box() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(vars[0].clone()) - Expr::from_i64(1);
    let y = Expr::Sym(vars[1].clone()) + Expr::from_i64(2);
    let source = power(x + Expr::from_i64(2) * y.clone(), 2) + power(y, 2);
    let domain = [bounds(-5, 6), bounds(-7, 8)];
    let receipt = minimum(&source, &vars, &domain, &[q(1, 1), q(-2, 1)]).unwrap();
    assert_eq!(receipt.objective(), &source);
    assert_eq!(receipt.variables(), vars);
    assert_eq!(receipt.domain(), domain);
    assert_eq!(receipt.point(), [q(1, 1), q(-2, 1)]);
    assert_eq!(receipt.value(), &RealBall::from_i64(0));
    assert_eq!(
        receipt.gradient(),
        [RealBall::from_i64(0), RealBall::from_i64(0)]
    );
    assert_eq!(receipt.kind(), BoxExtremumKind::Minimum);
    assert!(receipt.uniqueness_proved());
    assert_eq!(receipt.precision_digits(), 24);
}

#[test]
fn boundary_first_order_signs_handle_both_box_faces_and_extremum_directions() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = Expr::from_i64(2) * Expr::Sym(vars[0].clone())
        - Expr::from_i64(3) * Expr::Sym(vars[1].clone());
    let domain = [bounds(-1, 2), bounds(-2, 3)];
    let min = minimum(&source, &vars, &domain, &[q(-1, 1), q(3, 1)]).unwrap();
    assert_eq!(min.value(), &RealBall::from_i64(-11));
    assert!(min.uniqueness_proved());
    let max = certify_box_extremum(
        &source,
        &vars,
        &domain,
        &[q(2, 1), q(-2, 1)],
        BoxExtremumKind::Maximum,
        24,
    )
    .unwrap();
    assert_eq!(max.value(), &RealBall::from_i64(10));
    assert!(max.uniqueness_proved());
    for point in [[q(2, 1), q(3, 1)], [q(-1, 1), q(-2, 1)], [q(0, 1), q(0, 1)]] {
        assert!(matches!(
            minimum(&source, &vars, &domain, &point),
            Err(BoxExtremumError::FirstOrderNotProved { .. })
        ));
    }
}

#[test]
fn mixed_boundary_and_interior_optimum_is_not_required_to_be_stationary() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = power(Expr::Sym(vars[0].clone()) + Expr::from_i64(1), 2)
        + power(Expr::Sym(vars[1].clone()) - Expr::from_i64(2), 2);
    let receipt = minimum(
        &source,
        &vars,
        &[bounds(0, 3), bounds(0, 4)],
        &[q(0, 1), q(2, 1)],
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(1));
    assert_eq!(
        receipt.gradient(),
        [RealBall::from_i64(2), RealBall::from_i64(0)]
    );
    assert!(receipt.uniqueness_proved());
}

#[test]
fn flat_convex_objectives_are_accepted_without_inventing_uniqueness() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(vars[0].clone());
    let domain = [bounds(-1, 1), bounds(-1, 1)];
    for source in [Expr::from_i64(7), power(x.clone(), 2), power(x, 4)] {
        let receipt = minimum(&source, &vars, &domain, &[q(0, 1), q(0, 1)]).unwrap();
        assert!(!receipt.uniqueness_proved());
    }
}

#[test]
fn singular_coupled_psd_hessian_has_global_but_not_uniqueness_receipt() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = power(Expr::Sym(vars[0].clone()) + Expr::Sym(vars[1].clone()), 2);
    let receipt = minimum(
        &source,
        &vars,
        &[bounds(-2, 2), bounds(-2, 2)],
        &[q(0, 1), q(0, 1)],
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(0));
    assert!(!receipt.uniqueness_proved());
}

#[test]
fn exact_hessian_grid_matches_independent_two_by_two_principal_minor_criterion() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                let source = Expr::from_i64(a) * power(x.clone(), 2)
                    + Expr::from_i64(b) * x.clone() * y.clone()
                    + Expr::from_i64(c) * power(y.clone(), 2);
                let result = minimum(
                    &source,
                    &vars,
                    &[bounds(-1, 1), bounds(-1, 1)],
                    &[q(0, 1), q(0, 1)],
                );
                let psd = a >= 0 && c >= 0 && 4 * a * c - b * b >= 0;
                assert_eq!(result.is_ok(), psd, "a={a}, b={b}, c={c}");
                if let Ok(receipt) = result {
                    assert_eq!(receipt.uniqueness_proved(), a > 0 && 4 * a * c - b * b > 0);
                }
            }
        }
    }
}

#[test]
fn negative_curvature_saddles_and_deceptive_stationary_candidates_refuse() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    for source in [
        x.clone() * y,
        Expr::from_i64(-1) * power(x.clone(), 2),
        power(power(x, 2) - Expr::from_i64(1), 2),
    ] {
        assert_eq!(
            minimum(
                &source,
                &vars,
                &[bounds(-2, 2), bounds(-2, 2)],
                &[q(0, 1), q(0, 1)]
            ),
            Err(BoxExtremumError::CurvatureNotProved)
        );
    }
}

#[test]
fn concavity_certifies_a_maximum_and_rejects_the_reversed_claim() {
    let vars = [Symbol::new("x")];
    let source = Expr::from_i64(5) - power(Expr::Sym(vars[0].clone()) - Expr::from_i64(1), 2);
    let receipt = certify_box_extremum(
        &source,
        &vars,
        &[bounds(-3, 4)],
        &[q(1, 1)],
        BoxExtremumKind::Maximum,
        24,
    )
    .unwrap();
    assert_eq!(receipt.kind(), BoxExtremumKind::Maximum);
    assert_eq!(receipt.value(), &RealBall::from_i64(5));
    assert!(receipt.uniqueness_proved());
    assert_eq!(
        minimum(&source, &vars, &[bounds(-3, 4)], &[q(1, 1)]),
        Err(BoxExtremumError::CurvatureNotProved)
    );
}

#[test]
fn transcendental_boundary_and_sub_binary64_curvature_remain_exact() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let source = Expr::Function("exp".into(), vec![x.clone()]);
    let receipt = minimum(&source, &vars, &[bounds(0, 1)], &[q(0, 1)]).unwrap();
    assert!(receipt.value().contains(&q(1, 1)));
    assert!(receipt.uniqueness_proved());
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(10).pow(400));
    let source = Expr::Rational(tiny.clone()) * power(x - Expr::Rational(tiny.clone()), 2);
    let receipt = minimum(&source, &vars, &[bounds(-1, 1)], &[tiny]).unwrap();
    assert!(receipt.uniqueness_proved());
    assert_eq!(receipt.value(), &RealBall::from_i64(0));
}

#[test]
fn original_holes_cannot_be_erased_by_candidate_or_zero_factors() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    for child in [
        power(x.clone(), -1),
        Expr::Function("log".into(), vec![x.clone()]),
    ] {
        for hidden in [
            Expr::Mul(vec![Expr::from_i64(0), child.clone()]),
            power(child, 0),
        ] {
            let source = Expr::Add(vec![power(x.clone() - Expr::from_i64(1), 2), hidden]);
            assert!(minimum(&source, &vars, &[bounds(-1, 2)], &[q(1, 1)]).is_err());
        }
    }
}

#[test]
fn exact_point_box_dimensions_and_typed_source_binding_are_checked() {
    let vars = [Symbol::new("x")];
    let source = Expr::Sym(vars[0].clone());
    let fixed = minimum(&source, &vars, &[bounds(2, 2)], &[q(2, 1)]).unwrap();
    assert!(fixed.uniqueness_proved());
    assert_eq!(fixed.value(), &RealBall::from_i64(2));
    assert!(minimum(&source, &vars, &[bounds(0, 1)], &[]).is_err());
    assert!(minimum(&source, &vars, &[bounds(0, 1)], &[q(2, 1)]).is_err());
    assert!(minimum(&source, &[Symbol::new("y")], &[bounds(0, 1)], &[q(0, 1)]).is_err());
    for precision in [7, 129] {
        assert!(
            certify_box_extremum(
                &source,
                &vars,
                &[bounds(0, 1)],
                &[q(0, 1)],
                BoxExtremumKind::Minimum,
                precision
            )
            .is_err()
        );
    }
}

#[derive(Default)]
struct Cancel {
    visits: usize,
    stop: Option<usize>,
}
impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let here = self.visits;
        self.visits += 1;
        if self.stop == Some(here) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_safe_point_prevents_extremum_publication() {
    let vars = [Symbol::new("x")];
    let source = power(Expr::Sym(vars[0].clone()), 2);
    let domain = [bounds(-1, 1)];
    let point = [q(0, 1)];
    let mut count = Cancel::default();
    verify_box_extremum(
        &source,
        &vars,
        &domain,
        &point,
        BoxExtremumKind::Minimum,
        24,
        &mut count,
    )
    .unwrap();
    for stop in 0..count.visits {
        let mut meter = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            verify_box_extremum(
                &source,
                &vars,
                &domain,
                &point,
                BoxExtremumKind::Minimum,
                24,
                &mut meter
            ),
            Err(BoxExtremumError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            ))),
            "safe point {stop}"
        );
    }
}

#[test]
fn failure_retains_cumulative_budget_and_never_consumes_protected_resources() {
    let vars = [Symbol::new("x")];
    let source = power(Expr::Sym(vars[0].clone()), 2);
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    let before = budget.snapshot();
    assert!(
        verify_box_extremum(
            &source,
            &vars,
            &[bounds(-1, 1)],
            &[q(1, 1)],
            BoxExtremumKind::Minimum,
            24,
            &mut budget
        )
        .is_err()
    );
    let consumed = budget.snapshot();
    assert_ne!(before, consumed);
    assert_eq!(budget.verifier_remaining(), 10_000);
    assert!(matches!(
        verify_box_extremum(
            &source,
            &vars,
            &[bounds(-1, 1)],
            &[q(0, 1)],
            BoxExtremumKind::Minimum,
            24,
            &mut budget
        ),
        Err(BoxExtremumError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.snapshot(), consumed);
}

#[test]
fn curvature_is_restricted_to_coordinates_that_can_move() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = fsym_core::parse("x^2-y^2+3*x*y").unwrap();
    let domain = [bounds(-3, 3), bounds(1, 1)];
    let point = [q(-3, 2), q(1, 1)];
    let receipt = minimum(&source, &vars, &domain, &point).unwrap();
    assert!(receipt.uniqueness_proved());
    assert_eq!(receipt.value(), &RealBall::exact(q(-13, 4)));
    assert_eq!(receipt.hessian()[1][1], RealBall::from_i64(-2));
    assert!(minimum(&source, &vars, &[bounds(-3, 3), bounds(0, 2)], &point).is_err());

    let singleton = [bounds(0, 0), bounds(1, 1)];
    for kind in [BoxExtremumKind::Minimum, BoxExtremumKind::Maximum] {
        let receipt =
            certify_box_extremum(&source, &vars, &singleton, &[q(0, 1), q(1, 1)], kind, 24)
                .unwrap();
        assert!(receipt.uniqueness_proved());
    }
}

#[test]
fn whole_box_derivative_signs_reduce_a_nonconvex_problem_to_a_convex_face() {
    // f(0,y)=y^2, while f_x=5-2x+y^2 stays strictly positive on the box.
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = fsym_core::parse("5*x-x^2+(1+x)*y^2").unwrap();
    let domain = [bounds(0, 2), bounds(-1, 1)];
    let origin = [q(0, 1), q(0, 1)];
    let minimum_receipt = minimum(&source, &vars, &domain, &origin).unwrap();
    assert!(minimum_receipt.uniqueness_proved());
    assert_eq!(minimum_receipt.value(), &RealBall::from_i64(0));
    let maximum = certify_box_extremum(
        &(-source.clone()),
        &vars,
        &domain,
        &origin,
        BoxExtremumKind::Maximum,
        24,
    )
    .unwrap();
    assert!(maximum.uniqueness_proved());
    assert!(minimum(&source, &vars, &domain, &[q(2, 1), q(0, 1)]).is_err());
    assert!(minimum(&source, &vars, &domain, &[q(0, 1), q(1, 1)]).is_err());
    // The gradient at (0,0) alone gives the same signs; f(6,0)=-6.
    assert!(minimum(&source, &vars, &[bounds(0, 6), bounds(-1, 1)], &origin).is_err());
    let flat_face = fsym_core::parse("5*x-x^2").unwrap();
    assert!(
        !minimum(&flat_face, &vars, &domain, &origin)
            .unwrap()
            .uniqueness_proved()
    );
}

#[test]
fn weak_monotonicity_can_prove_an_optimum_shared_by_entire_faces() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = fsym_core::parse("x*y").unwrap();
    let domain = [bounds(0, 1), bounds(0, 1)];
    for point in [[q(0, 1), q(0, 1)], [q(0, 1), q(1, 2)], [q(1, 2), q(0, 1)]] {
        let receipt = minimum(&source, &vars, &domain, &point).unwrap();
        assert_eq!(receipt.value(), &RealBall::from_i64(0));
        assert!(!receipt.uniqueness_proved());
    }
    assert!(minimum(&source, &vars, &domain, &[q(1, 2), q(1, 2)]).is_err());
}

#[test]
fn strict_active_bounds_and_relative_curvature_jointly_prove_uniqueness() {
    let vars = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let domain = [bounds(0, 2), bounds(-2, 2), bounds(-2, 2)];
    let point = [q(0, 1), q(0, 1), q(0, 1)];
    // For an equal-cost point the supporting inequality forces x=0,
    // and the coupled y,z quadratic is positive definite on that plane.
    let source = fsym_core::parse("x+(y-z)^2+z^2").unwrap();
    assert!(
        minimum(&source, &vars, &domain, &point)
            .unwrap()
            .uniqueness_proved()
    );
    let maximum = certify_box_extremum(
        &(-source),
        &vars,
        &domain,
        &point,
        BoxExtremumKind::Maximum,
        24,
    )
    .unwrap();
    assert!(maximum.uniqueness_proved());
    // Dropping z^2 leaves all feasible (0,t,t) as minimizers.
    let flat = fsym_core::parse("x+(y-z)^2").unwrap();
    assert!(
        !minimum(&flat, &vars, &domain, &point)
            .unwrap()
            .uniqueness_proved()
    );
}

#[test]
fn a_nonconvex_objective_attains_its_exact_range_at_distinct_global_extrema() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let wells = power(power(x, 2) - Expr::from_i64(1), 2);
    let box_ = [bounds(-2, 2)];
    for (source, kind) in [
        (Expr::from_i64(7) + wells.clone(), BoxExtremumKind::Minimum),
        (Expr::from_i64(7) - wells.clone(), BoxExtremumKind::Maximum),
    ] {
        for point in [-1, 1] {
            let receipt =
                certify_box_extremum(&source, &vars, &box_, &[q(point, 1)], kind, 24).unwrap();
            assert_eq!(receipt.value(), &RealBall::from_i64(7));
            assert!(!receipt.uniqueness_proved());
        }
        assert!(certify_box_extremum(&source, &vars, &box_, &[q(0, 1)], kind, 24).is_err());
    }
    let hidden = Expr::Mul(vec![
        Expr::from_i64(0),
        power(Expr::Sym(vars[0].clone()), -1),
    ]);
    assert!(minimum(&Expr::Add(vec![wells, hidden]), &vars, &box_, &[q(1, 1)]).is_err());
}
