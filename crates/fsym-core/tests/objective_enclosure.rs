//! Independent second-order enclosure acceptance and refusal regressions.
//! Analytic derivatives here are fixed test oracles, never verifier inputs.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Constant, Expr, MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_LIMBS,
    MAX_ROOT_BOX_NODES, MAX_ROOT_BOX_VARIABLES, RealBall, RootBoxError, Symbol, SymbolIdentity,
    enclose_objective, metered_enclose_objective,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn variables(n: usize) -> Vec<Symbol> {
    (0..n).map(|i| Symbol::new(format!("x{i}"))).collect()
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}
fn exact(values: &[i64]) -> Vec<RealBall> {
    values.iter().copied().map(RealBall::from_i64).collect()
}

#[test]
fn exact_dense_quadratic_retains_both_mixed_product_terms() {
    let vars = variables(2);
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let mut objective = Expr::from_i64(3) * power(x.clone(), 2)
        + Expr::from_i64(2) * x.clone() * y.clone()
        + Expr::from_i64(5) * power(y.clone(), 2)
        + Expr::from_i64(7) * x
        - Expr::from_i64(11) * y
        + Expr::from_i64(13);
    let original = objective.clone();
    let domain = exact(&[2, -1]);
    let enclosed = enclose_objective(&objective, &vars, &domain, 24).unwrap();
    assert_eq!(enclosed.value(), &RealBall::from_i64(51));
    assert_eq!(enclosed.gradient(), exact(&[17, -17]));
    assert_eq!(enclosed.hessian(), vec![exact(&[6, 2]), exact(&[2, 10])]);
    assert_eq!(enclosed.variables(), vars);
    assert_eq!(enclosed.domain(), domain);
    assert_eq!(enclosed.precision_digits(), 24);
    objective = Expr::from_i64(0);
    assert_ne!(enclosed.objective(), &objective);
    assert_eq!(enclosed.objective(), &original);
}

#[test]
fn nested_quartic_keeps_inner_hessian_and_outer_gradient_terms() {
    let vars = variables(2);
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let objective = power(power(x, 2) + y, 2);
    let enclosed = enclose_objective(&objective, &vars, &exact(&[2, 3]), 24).unwrap();
    assert_eq!(enclosed.value(), &RealBall::from_i64(49));
    assert_eq!(enclosed.gradient(), exact(&[56, 14]));
    assert_eq!(enclosed.hessian(), vec![exact(&[60, 8]), exact(&[8, 2])]);
}

#[test]
fn whole_box_polynomial_bounds_cover_fixed_analytic_values() {
    let vars = variables(2);
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let objective = x * power(y, 2);
    let domain = vec![
        RealBall::new(q(1, 1), q(1, 2)).unwrap(),
        RealBall::new(q(2, 1), q(1, 3)).unwrap(),
    ];
    let enclosed = enclose_objective(&objective, &vars, &domain, 24).unwrap();
    assert_eq!(enclosed.hessian()[0][0], RealBall::from_i64(0));
    assert_eq!(enclosed.hessian()[0][1], enclosed.hessian()[1][0]);
    for a in [q(1, 2), q(1, 1), q(3, 2)] {
        for b in [q(5, 3), q(2, 1), q(7, 3)] {
            assert!(enclosed.value().contains(&(&a * &b * &b)));
            assert!(enclosed.gradient()[0].contains(&(&b * &b)));
            assert!(enclosed.gradient()[1].contains(&(q(2, 1) * &a * &b)));
            assert!(enclosed.hessian()[0][1].contains(&(q(2, 1) * &b)));
            assert!(enclosed.hessian()[1][1].contains(&(q(2, 1) * &a)));
        }
    }
}

#[test]
fn reciprocal_curvature_and_affine_cross_terms_are_exact() {
    let vars = variables(2);
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let objective = power(x + y, -1);
    let enclosed = enclose_objective(&objective, &vars, &exact(&[2, 1]), 24).unwrap();
    assert_eq!(enclosed.value(), &RealBall::exact(q(1, 3)));
    assert_eq!(enclosed.gradient(), vec![RealBall::exact(q(-1, 9)); 2]);
    assert_eq!(
        enclosed.hessian(),
        vec![vec![RealBall::exact(q(2, 27)); 2]; 2]
    );
}

#[test]
fn all_admitted_elementary_rules_contain_known_derivatives() {
    let vars = variables(1);
    let x = Expr::Sym(vars[0].clone());
    // (function, point, rational value if available, first and second derivatives)
    for (name, point, value, first, second) in [
        ("sin", 0, Some(q(0, 1)), q(1, 1), q(0, 1)),
        ("cos", 0, Some(q(1, 1)), q(0, 1), q(-1, 1)),
        ("exp", 0, Some(q(1, 1)), q(1, 1), q(1, 1)),
        ("log", 2, None, q(1, 2), q(-1, 4)),
        ("ln", 1, Some(q(0, 1)), q(1, 1), q(-1, 1)),
        ("sqrt", 4, Some(q(2, 1)), q(1, 4), q(-1, 32)),
        ("atan", 0, Some(q(0, 1)), q(1, 1), q(0, 1)),
        ("tan", 0, Some(q(0, 1)), q(1, 1), q(0, 1)),
        ("sinh", 0, Some(q(0, 1)), q(1, 1), q(0, 1)),
        ("cosh", 0, Some(q(1, 1)), q(0, 1), q(1, 1)),
        ("tanh", 0, Some(q(0, 1)), q(1, 1), q(0, 1)),
    ] {
        let enclosed = enclose_objective(&function(name, x.clone()), &vars, &exact(&[point]), 24)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        if let Some(value) = value {
            assert!(enclosed.value().contains(&value), "{name}: value");
        }
        assert!(enclosed.gradient()[0].contains(&first), "{name}: gradient");
        assert!(
            enclosed.hessian()[0][0].contains(&second),
            "{name}: Hessian"
        );
    }
    for (objective, second) in [
        (function("exp", function("sin", x.clone())), q(1, 1)),
        (function("exp", power(x, 2)), q(2, 1)),
    ] {
        let enclosed = enclose_objective(&objective, &vars, &exact(&[0]), 24).unwrap();
        assert!(enclosed.value().contains(&q(1, 1)));
        assert!(enclosed.hessian()[0][0].contains(&second));
    }
}

#[test]
fn positive_dynamic_power_keeps_exponent_and_mixed_derivatives() {
    let vars = variables(2);
    let objective = Expr::Pow(
        Arc::new(Expr::Sym(vars[0].clone())),
        Arc::new(Expr::Sym(vars[1].clone())),
    );
    let enclosed = enclose_objective(&objective, &vars, &exact(&[1, 3]), 24).unwrap();
    assert!(enclosed.value().contains(&q(1, 1)));
    for (entry, expected) in enclosed.gradient().iter().zip([q(3, 1), q(0, 1)]) {
        assert!(entry.contains(&expected));
    }
    for (entry, expected) in
        enclosed
            .hessian()
            .iter()
            .flatten()
            .zip([q(6, 1), q(1, 1), q(1, 1), q(0, 1)])
    {
        assert!(entry.contains(&expected));
    }
    let rational_power = Expr::Pow(
        Arc::new(Expr::Sym(vars[0].clone())),
        Arc::new(Expr::Rational(q(1, 2))),
    );
    let enclosed = enclose_objective(&rational_power, &vars[..1], &exact(&[4]), 24).unwrap();
    assert!(enclosed.value().contains(&q(2, 1)));
    assert!(enclosed.gradient()[0].contains(&q(1, 4)));
    assert!(enclosed.hessian()[0][0].contains(&q(-1, 32)));
}

#[test]
fn constant_and_identity_powers_do_not_invent_zero_denominators() {
    let vars = variables(1);
    let x = Expr::Sym(vars[0].clone());
    for (exponent, point, value, first, second) in [
        (0, 2, 1, 0, 0),
        (1, 0, 0, 1, 0),
        (2, 0, 0, 0, 2),
        (3, 0, 0, 0, 0),
    ] {
        let enclosed =
            enclose_objective(&power(x.clone(), exponent), &vars, &exact(&[point]), 24).unwrap();
        assert_eq!(enclosed.value(), &RealBall::from_i64(value));
        assert_eq!(enclosed.gradient(), exact(&[first]));
        assert_eq!(enclosed.hessian(), vec![exact(&[second])]);
    }
}

#[test]
fn tiny_exact_curvature_and_huge_coordinates_are_not_rounded_to_binary64() {
    let vars = variables(1);
    let x = Expr::Sym(vars[0].clone());
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(10).pow(400));
    let huge = BigRational::from_integer(BigInt::from(10).pow(200));
    let objective = Expr::Rational(tiny.clone()) * power(x, 2);
    let enclosed =
        enclose_objective(&objective, &vars, &[RealBall::exact(huge.clone())], 24).unwrap();
    assert_eq!(enclosed.value(), &RealBall::from_i64(1));
    assert_eq!(
        enclosed.gradient()[0],
        RealBall::exact(q(2, 1) * &tiny * huge)
    );
    assert_eq!(enclosed.hessian()[0][0], RealBall::exact(q(2, 1) * tiny));
    assert!(enclosed.hessian()[0][0].is_positive());
}

#[test]
fn typed_binding_identity_is_used_even_when_names_match() {
    let x = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([13; 32]));
    let vars = vec![x.clone(), other.clone()];
    let objective = power(Expr::Sym(x), 2) * Expr::Sym(other);
    let enclosed = enclose_objective(&objective, &vars, &exact(&[2, 3]), 24).unwrap();
    assert_eq!(enclosed.gradient(), exact(&[12, 4]));
    assert_eq!(enclosed.hessian(), vec![exact(&[6, 4]), exact(&[4, 0])]);
    assert!(matches!(
        enclose_objective(&objective, &vars[..1], &exact(&[2]), 24),
        Err(RootBoxError::Unsupported(_))
    ));
    assert_eq!(
        enclose_objective(
            &objective,
            &[vars[0].clone(), vars[0].clone()],
            &exact(&[2, 3]),
            24
        ),
        Err(RootBoxError::DuplicateVariable)
    );
}

#[test]
fn original_domain_checks_survive_zero_products_and_held_zero_powers() {
    let vars = variables(1);
    let x = Expr::Sym(vars[0].clone());
    let box_touching_zero = vec![RealBall::new(q(1, 1), q(1, 1)).unwrap()];
    for bad in [
        function("log", x.clone()),
        function("sqrt", x.clone()),
        power(x.clone(), -1),
        Expr::Pow(Arc::new(x.clone()), Arc::new(Expr::Rational(q(1, 2)))),
        power(Expr::from_i64(0), 0),
    ] {
        for held in [
            Expr::Mul(vec![Expr::from_i64(0), bad.clone()]),
            power(bad.clone(), 0),
        ] {
            assert!(enclose_objective(&held, &vars, &box_touching_zero, 24).is_err());
        }
    }
    assert!(
        enclose_objective(
            &function("tan", x),
            &vars,
            &[RealBall::new(q(8, 5), q(1, 10)).unwrap()],
            24,
        )
        .is_err()
    );
}

#[test]
fn unsupported_shapes_precision_and_intrinsic_source_limits_fail_closed() {
    let vars = variables(1);
    let x = Expr::Sym(vars[0].clone());
    assert_eq!(
        enclose_objective(&x, &[], &[], 24),
        Err(RootBoxError::DimensionMismatch)
    );
    assert_eq!(
        enclose_objective(&x, &vars, &[], 24),
        Err(RootBoxError::DimensionMismatch)
    );
    assert_eq!(
        enclose_objective(
            &x,
            &variables(MAX_ROOT_BOX_VARIABLES + 1),
            &vec![RealBall::from_i64(0); MAX_ROOT_BOX_VARIABLES + 1],
            24,
        ),
        Err(RootBoxError::DimensionMismatch)
    );
    for precision in [0, 7, 129, u32::MAX] {
        assert_eq!(
            enclose_objective(&x, &vars, &exact(&[1]), precision),
            Err(RootBoxError::InvalidPrecision)
        );
    }
    for unsupported in [
        function("abs", x.clone()),
        Expr::Function("sin".into(), vec![x.clone(), x.clone()]),
        Expr::Const(Constant::I),
        power(x.clone(), 65),
    ] {
        assert!(enclose_objective(&unsupported, &vars, &exact(&[1]), 24).is_err());
    }
    let mut deep = x.clone();
    for _ in 0..=MAX_ROOT_BOX_DEPTH {
        deep = Expr::Add(vec![deep]);
    }
    let huge = BigInt::from(1) << (MAX_ROOT_BOX_LIMBS * 64) as u32;
    for large in [
        deep,
        Expr::Add(vec![x; MAX_ROOT_BOX_NODES]),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Integer(huge)]),
    ] {
        assert!(matches!(
            enclose_objective(&large, &vars, &exact(&[1]), 24),
            Err(RootBoxError::ResourceLimit(_))
        ));
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
fn cancellation_at_every_checkpoint_publishes_no_partial_objective_receipt() {
    let vars = variables(1);
    let objective = power(Expr::Sym(vars[0].clone()), 2);
    let mut count = Cancel::default();
    metered_enclose_objective(&objective, &vars, &exact(&[2]), 24, &mut count).unwrap();
    for stop in 0..count.visits {
        let mut cancel = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            metered_enclose_objective(&objective, &vars, &exact(&[2]), 24, &mut cancel),
            Err(RootBoxError::Meter(MeterError::Cancelled)),
            "checkpoint {stop}"
        );
    }
}

#[test]
fn one_real_budget_is_cumulative_and_failed_reservations_are_atomic() {
    let vars = variables(2);
    let objective = power(Expr::Sym(vars[0].clone()), 2) * Expr::Sym(vars[1].clone());
    let domain = exact(&[2, 3]);
    let limit = 512 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    metered_enclose_objective(&objective, &vars, &domain, 24, &mut budget).unwrap();
    let dimensions = [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ];
    let after_first: Vec<_> = dimensions
        .into_iter()
        .map(|dimension| budget.remaining(dimension))
        .collect();
    metered_enclose_objective(&objective, &vars, &domain, 24, &mut budget).unwrap();
    for (dimension, first) in dimensions.into_iter().zip(after_first) {
        assert!(first < limit);
        assert!(budget.remaining(dimension) < first);
    }
    let mut empty = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_enclose_objective(&objective, &vars, &domain, 24, &mut empty),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    for dimension in dimensions {
        assert_eq!(empty.remaining(dimension), 1);
    }
}
