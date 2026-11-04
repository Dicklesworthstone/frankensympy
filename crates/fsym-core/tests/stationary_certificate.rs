use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness, StationaryPointKind, Symbol,
    certify_stationary_point, verify_stationary_point,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn witness(center: Vec<BigRational>, radius: BigRational) -> RootBoxWitness {
    let n = center.len();
    RootBoxWitness {
        domain: center
            .iter()
            .map(|value| RealBall::new(value.clone(), radius.clone()).unwrap())
            .collect(),
        center,
        preconditioner: (0..n)
            .map(|i| (0..n).map(|j| q(i64::from(i == j), 1)).collect())
            .collect(),
        precision_digits: 24,
    }
}

#[test]
fn coupled_nondiagonally_dominant_quadratic_has_checked_minimum() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(vars[0].clone()) - Expr::from_i64(1);
    let y = Expr::Sym(vars[1].clone()) + Expr::from_i64(2);
    // H = [[2,4],[4,10]], det(H)=4. The first row is not diagonally dominant.
    let source = power(x + Expr::from_i64(2) * y.clone(), 2) + power(y, 2);
    let mut proposal = witness(vec![q(1, 1), q(-2, 1)], q(1, 4));
    proposal.preconditioner = vec![vec![q(5, 2), q(-1, 1)], vec![q(-1, 1), q(1, 2)]];
    let accepted = certify_stationary_point(&source, &vars, &proposal).unwrap();
    assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    assert_eq!(
        accepted.stationary_enclosure(),
        [RealBall::exact(q(1, 1)), RealBall::exact(q(-2, 1))]
    );
    assert_eq!(accepted.objective(), &source);
    assert_eq!(accepted.variables(), vars);
    assert_eq!(accepted.witness(), &proposal);
    assert!(accepted.objective_enclosure().contains(&q(0, 1)));
    assert_eq!(
        accepted.hessian_enclosure(),
        [
            vec![RealBall::from_i64(2), RealBall::from_i64(4)],
            vec![RealBall::from_i64(4), RealBall::from_i64(10)]
        ]
    );
    // Caller mutation cannot rewrite the accepted source or domain.
    proposal.center[0] = q(100, 1);
    assert_eq!(accepted.witness().center[0], q(1, 1));
}

#[test]
fn maximum_and_saddle_are_never_promoted_to_minima() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let x2 = power(Expr::Sym(vars[0].clone()), 2);
    let y2 = power(Expr::Sym(vars[1].clone()), 2);
    let mut proposed = witness(vec![q(0, 1); 2], q(1, 4));
    proposed.preconditioner = vec![vec![q(-1, 2), q(0, 1)], vec![q(0, 1), q(-1, 2)]];
    let maximum = Expr::from_i64(-1) * (x2.clone() + y2.clone());
    assert_eq!(
        certify_stationary_point(&maximum, &vars, &proposed)
            .unwrap()
            .kind(),
        StationaryPointKind::StrictMaximum
    );
    proposed.preconditioner[0][0] = q(1, 2);
    assert_eq!(
        certify_stationary_point(&(x2 - y2), &vars, &proposed)
            .unwrap()
            .kind(),
        StationaryPointKind::Saddle
    );
}

#[test]
fn inconclusive_curvature_is_explicit_even_when_stationarity_is_proved() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let source = Expr::Sym(vars[0].clone()) * Expr::Sym(vars[1].clone());
    let mut proposed = witness(vec![q(0, 1); 2], q(1, 4));
    proposed.preconditioner = vec![vec![q(0, 1), q(1, 1)], vec![q(1, 1), q(0, 1)]];
    // No-pivot LDL cannot start on H=[[0,1],[1,0]]. The stationary-point
    // theorem still holds; it must not silently assert positive curvature.
    let accepted = certify_stationary_point(&source, &vars, &proposed).unwrap();
    assert_eq!(accepted.kind(), StationaryPointKind::Unclassified);
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
}

#[test]
fn nonlinear_rosenbrock_basin_is_certified_from_original_source() {
    let vars = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let source =
        power(Expr::from_i64(1) - x.clone(), 2) + Expr::from_i64(100) * power(y - power(x, 2), 2);
    let mut proposed = witness(vec![q(1, 1); 2], q(1, 1_000_000));
    // H(1,1)=[[802,-400],[-400,200]], exact inverse below.
    proposed.preconditioner = vec![vec![q(1, 2), q(1, 1)], vec![q(1, 1), q(401, 200)]];
    let accepted = certify_stationary_point(&source, &vars, &proposed).unwrap();
    assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
    for coordinate in accepted.stationary_enclosure() {
        assert!(coordinate.contains(&q(1, 1)));
    }
    assert!(accepted.contraction_bound() < &q(1, 1));
}

#[test]
fn transcendental_stationary_point_uses_independent_interval_rules() {
    let vars = [Symbol::new("x")];
    let source = Expr::Function("cosh".into(), vec![Expr::Sym(vars[0].clone())]);
    let proposed = witness(vec![q(0, 1)], q(1, 100));
    let accepted = certify_stationary_point(&source, &vars, &proposed).unwrap();
    assert_eq!(accepted.kind(), StationaryPointKind::StrictMinimum);
    assert!(accepted.stationary_enclosure()[0].contains(&q(0, 1)));
    assert!(accepted.objective_enclosure().contains(&q(1, 1)));
}

#[test]
fn no_stationary_point_flat_points_and_multiple_points_refuse() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let proposed = witness(vec![q(0, 1)], q(1, 4));
    for source in [x.clone(), power(x.clone(), 4), Expr::from_i64(1)] {
        assert!(matches!(
            certify_stationary_point(&source, &vars, &proposed),
            Err(RootBoxError::NotContractive)
        ));
    }
    let source = x.clone() + Expr::Rational(q(1, 2)) * power(x.clone(), 2);
    assert!(matches!(
        certify_stationary_point(&source, &vars, &proposed),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    ));
    let double_well = power(power(x, 2) - Expr::from_i64(1), 2);
    assert!(
        certify_stationary_point(&double_well, &vars, &witness(vec![q(0, 1)], q(2, 1))).is_err()
    );
}

#[test]
fn source_and_witness_mutations_cannot_reuse_an_extremum_claim() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let source = Expr::Rational(q(1, 2)) * power(x.clone(), 2);
    let original = witness(vec![q(0, 1)], q(1, 4));
    certify_stationary_point(&source, &vars, &original).unwrap();
    let moved_source = source.clone() + Expr::from_i64(10) * x;
    assert!(certify_stationary_point(&moved_source, &vars, &original).is_err());
    for coefficient in [q(0, 1), q(-1, 1), q(3, 1)] {
        let mut changed = original.clone();
        changed.preconditioner[0][0] = coefficient;
        assert_eq!(
            certify_stationary_point(&source, &vars, &changed),
            Err(RootBoxError::NotContractive)
        );
    }
    let moved_box = witness(vec![q(2, 1)], q(1, 4));
    assert!(certify_stationary_point(&source, &vars, &moved_box).is_err());
    let mut outside = original.clone();
    outside.center[0] = q(1, 1);
    assert_eq!(
        certify_stationary_point(&source, &vars, &outside),
        Err(RootBoxError::CenterOutsideBox { coordinate: 0 })
    );
    let boundary = witness(vec![q(1, 1)], q(1, 1));
    assert_eq!(
        certify_stationary_point(&source, &vars, &boundary),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    );
    let mut bad_shape = original;
    bad_shape.preconditioner[0].push(q(1, 1));
    assert_eq!(
        certify_stationary_point(&source, &vars, &bad_shape),
        Err(RootBoxError::DimensionMismatch)
    );
}

#[test]
fn original_domain_obligations_survive_zero_factors_and_zero_powers() {
    let vars = [Symbol::new("x")];
    let x = Expr::Sym(vars[0].clone());
    let smooth = Expr::Rational(q(1, 2)) * power(x.clone(), 2);
    let bad_children = [
        power(x.clone(), -1),
        Expr::Function("log".into(), vec![x.clone()]),
        Expr::Function("sqrt".into(), vec![x]),
    ];
    let proposed = witness(vec![q(0, 1)], q(1, 4));
    for child in bad_children {
        for hidden in [
            Expr::Mul(vec![Expr::from_i64(0), child.clone()]),
            power(child, 0),
        ] {
            let source = Expr::Add(vec![smooth.clone(), hidden]);
            assert!(certify_stationary_point(&source, &vars, &proposed).is_err());
        }
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
fn cancellation_at_every_stage_returns_no_stationary_receipt() {
    let vars = [Symbol::new("x")];
    let source = Expr::Rational(q(1, 2)) * power(Expr::Sym(vars[0].clone()), 2);
    let proposed = witness(vec![q(0, 1)], q(1, 4));
    let mut count = Cancel::default();
    verify_stationary_point(&source, &vars, &proposed, &mut count).unwrap();
    for stop in 0..count.visits {
        let mut meter = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            verify_stationary_point(&source, &vars, &proposed, &mut meter),
            Err(RootBoxError::Meter(MeterError::Cancelled)),
            "safe point {stop}"
        );
    }
}

#[test]
fn real_budgets_remain_cumulative_and_protect_the_verifier_pool() {
    let vars = [Symbol::new("x")];
    let source = Expr::Rational(q(1, 2)) * power(Expr::Sym(vars[0].clone()), 2);
    let proposed = witness(vec![q(0, 1)], q(1, 4));
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        verify_stationary_point(&source, &vars, &proposed, &mut empty),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    let mut limited = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    let before = limited.snapshot();
    verify_stationary_point(&source, &vars, &proposed, &mut limited).unwrap();
    let used = limited.snapshot();
    assert_ne!(before, used);
    assert_eq!(limited.verifier_remaining(), 10_000);
    // The second call cannot regain the first call's reserved interval memory.
    assert!(matches!(
        verify_stationary_point(&source, &vars, &proposed, &mut limited),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(limited.snapshot(), used);
}
