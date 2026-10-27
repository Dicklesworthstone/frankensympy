use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_NODES,
    MAX_ROOT_BOX_VARIABLES, RealBall, RootBoxError, RootBoxWitness, Symbol, SymbolIdentity,
    certify_root_box, enclose_residual_jacobian, metered_enclose_residual_jacobian,
    verify_root_box,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn symbols(n: usize) -> Vec<Symbol> {
    (0..n).map(|i| Symbol::new(format!("x{i}"))).collect()
}
fn witness(center: Vec<BigRational>, radius: BigRational) -> RootBoxWitness {
    let n = center.len();
    RootBoxWitness {
        domain: center
            .iter()
            .map(|x| RealBall::new(x.clone(), radius.clone()).unwrap())
            .collect(),
        center,
        preconditioner: (0..n)
            .map(|i| (0..n).map(|j| q(i64::from(i == j), 1)).collect())
            .collect(),
        precision_digits: 24,
    }
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn scalar_irrational_root_gets_exact_strict_bounds() {
    let vars = symbols(1);
    let source = vec![power(Expr::Sym(vars[0].clone()), 2) - Expr::from_i64(2)];
    let mut proposed = witness(vec![q(7, 5)], q(1, 10));
    proposed.preconditioner[0][0] = q(5, 14);
    let accepted = certify_root_box(&source, &vars, &proposed).unwrap();
    let root = &accepted.root_enclosure()[0];
    assert!(root.lower() > proposed.domain[0].lower());
    assert!(root.upper() < proposed.domain[0].upper());
    assert!(&root.lower() * &root.lower() < q(2, 1));
    assert!(&root.upper() * &root.upper() > q(2, 1));
    assert!(accepted.contraction_bound() < &q(1, 1));
}

#[test]
fn coupled_three_dimensional_nonlinear_system_is_certified() {
    let vars = symbols(3);
    let xyz: Vec<_> = vars.iter().cloned().map(Expr::Sym).collect();
    let source = vec![
        power(xyz[0].clone(), 2) - Expr::from_i64(2),
        xyz[1].clone() - xyz[0].clone(),
        xyz[2].clone() - xyz[1].clone(),
    ];
    let mut proposed = witness(vec![q(7, 5); 3], q(1, 10));
    proposed.preconditioner = vec![
        vec![q(5, 14), q(0, 1), q(0, 1)],
        vec![q(5, 14), q(1, 1), q(0, 1)],
        vec![q(5, 14), q(1, 1), q(1, 1)],
    ];
    let accepted = certify_root_box(&source, &vars, &proposed).unwrap();
    for root in accepted.root_enclosure() {
        assert!(root.lower() > q(0, 1));
        assert!(&root.lower() * &root.lower() < q(2, 1));
        assert!(&root.upper() * &root.upper() > q(2, 1));
    }
    assert_eq!(accepted.residuals(), source);
    assert_eq!(accepted.variables(), vars);
    assert_eq!(accepted.witness(), &proposed);
}

#[test]
fn nonsymmetric_preconditioner_orientation_is_checked() {
    let vars = symbols(3);
    let xyz: Vec<_> = vars.iter().cloned().map(Expr::Sym).collect();
    let source = vec![
        xyz[0].clone() + Expr::from_i64(2) * xyz[1].clone() - Expr::from_i64(5),
        xyz[1].clone() + Expr::from_i64(3) * xyz[2].clone() - Expr::from_i64(11),
        xyz[2].clone() - Expr::from_i64(3),
    ];
    let mut proposed = witness(vec![q(1, 1), q(2, 1), q(3, 1)], q(1, 4));
    proposed.preconditioner = vec![
        vec![q(1, 1), q(-2, 1), q(6, 1)],
        vec![q(0, 1), q(1, 1), q(-3, 1)],
        vec![q(0, 1), q(0, 1), q(1, 1)],
    ];
    let accepted = certify_root_box(&source, &vars, &proposed).unwrap();
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    assert_eq!(
        accepted.root_enclosure(),
        &[
            RealBall::exact(q(1, 1)),
            RealBall::exact(q(2, 1)),
            RealBall::exact(q(3, 1))
        ]
    );
    let original = proposed.preconditioner.clone();
    for (i, row) in proposed.preconditioner.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = original[j][i].clone();
        }
    }
    assert!(matches!(
        certify_root_box(&source, &vars, &proposed),
        Err(RootBoxError::NotContractive)
    ));
}

#[test]
fn maximum_dimension_coupled_triangular_system_fits_the_fixed_budget() {
    let vars = symbols(MAX_ROOT_BOX_VARIABLES);
    let n = vars.len();
    let exact: Vec<_> = (1..=n).map(|i| q(i as i64, 1)).collect();
    let mut source = Vec::new();
    let mut proposed = witness(exact.iter().map(|r| r + q(1, 20)).collect(), q(1, 4));
    for (i, variable) in vars.iter().enumerate() {
        let mut equation = Expr::Sym(variable.clone()) - Expr::Rational(exact[i].clone());
        if i + 1 < n {
            equation = equation
                + Expr::Rational(q(1, 2))
                    * (Expr::Sym(vars[i + 1].clone()) - Expr::Rational(exact[i + 1].clone()));
        }
        source.push(equation);
        let mut coefficient = q(1, 1);
        for entry in &mut proposed.preconditioner[i][i..] {
            *entry = coefficient.clone();
            coefficient *= q(-1, 2);
        }
    }
    let accepted = certify_root_box(&source, &vars, &proposed).unwrap();
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    assert_eq!(
        accepted.root_enclosure(),
        exact.into_iter().map(RealBall::exact).collect::<Vec<_>>()
    );
}

#[test]
fn rectangular_enclosures_bind_exact_source_and_all_partial_derivatives() {
    let vars = symbols(2);
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let mut source = vec![
        power(x.clone(), 2) + y.clone(),
        x.clone() * y.clone(),
        Expr::Function("exp".into(), vec![x]) - Expr::Function("sin".into(), vec![y]),
    ];
    let domain = vec![
        RealBall::new(q(3, 2), q(1, 2)).unwrap(),
        RealBall::new(q(0, 1), q(1, 1)).unwrap(),
    ];
    let accepted = enclose_residual_jacobian(&source, &vars, &domain, 32).unwrap();
    assert_eq!(accepted.domain(), domain);
    assert_eq!(accepted.variables(), vars);
    assert_eq!(accepted.precision_digits(), 32);
    assert_eq!(accepted.jacobian().len(), 3);
    assert_eq!(
        accepted.jacobian()[0],
        vec![
            RealBall::new(q(3, 1), q(1, 1)).unwrap(),
            RealBall::from_i64(1)
        ]
    );
    assert_eq!(
        accepted.jacobian()[1],
        vec![domain[1].clone(), domain[0].clone()]
    );
    assert!(accepted.residual_values()[0].contains(&q(0, 1)));
    assert!(accepted.residual_values()[0].contains(&q(5, 1)));
    assert!(accepted.jacobian()[2][0].is_positive());
    assert!(accepted.jacobian()[2][1].contains(&q(-1, 1)));
    let original = source.clone();
    source[0] = Expr::from_i64(99);
    assert_eq!(accepted.residuals(), original);
}

#[test]
fn point_and_large_exact_coordinates_do_not_go_through_binary64() {
    let vars = symbols(1);
    let x = Expr::Sym(vars[0].clone());
    let huge = BigRational::from_integer(BigInt::from(10).pow(400));
    let source = vec![x.clone() - Expr::Rational(huge.clone()), power(x, 2)];
    let accepted =
        enclose_residual_jacobian(&source, &vars, &[RealBall::exact(huge.clone())], 24).unwrap();
    assert_eq!(accepted.residual_values()[0], RealBall::from_i64(0));
    assert_eq!(
        accepted.residual_values()[1],
        RealBall::exact(&huge * &huge)
    );
    assert_eq!(accepted.jacobian()[0][0], RealBall::from_i64(1));
    assert_eq!(accepted.jacobian()[1][0], RealBall::exact(huge * q(2, 1)));
}

#[test]
fn typed_variable_identity_and_witness_mutations_are_never_trusted() {
    let x = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let vars = vec![x.clone(), other.clone()];
    let source = vec![
        Expr::Sym(x) - Expr::from_i64(1),
        Expr::Sym(other) - Expr::from_i64(2),
    ];
    let mut proposed = witness(vec![q(1, 1), q(2, 1)], q(1, 10));
    let accepted = certify_root_box(&source, &vars, &proposed).unwrap();
    assert_eq!(accepted.variables(), vars);
    assert!(
        enclose_residual_jacobian(&source, &[vars[0].clone()], &proposed.domain[..1], 24).is_err()
    );
    assert_eq!(
        certify_root_box(&source, &[vars[0].clone(), vars[0].clone()], &proposed),
        Err(RootBoxError::DuplicateVariable)
    );
    let original = proposed.clone();
    proposed.preconditioner[0][0] = q(0, 1);
    assert!(matches!(
        certify_root_box(&source, &vars, &proposed),
        Err(RootBoxError::NotContractive)
    ));
    proposed = original.clone();
    proposed.center[0] = q(3, 1);
    assert_eq!(
        certify_root_box(&source, &vars, &proposed),
        Err(RootBoxError::CenterOutsideBox { coordinate: 0 })
    );
    let changed = vec![source[0].clone() + Expr::from_i64(9), source[1].clone()];
    assert!(certify_root_box(&changed, &vars, &original).is_err());
    assert_eq!(accepted.witness(), &original);
}

#[test]
fn whole_box_domains_survive_held_zeros_powers_and_singularities() {
    let vars = symbols(1);
    let x = Expr::Sym(vars[0].clone());
    let log = Expr::Function("log".into(), vec![x.clone()]);
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), log.clone()]),
        power(log, 0),
        Expr::Mul(vec![Expr::from_i64(0), power(x.clone(), -1)]),
    ] {
        let source = vec![Expr::Add(vec![x.clone() - Expr::from_i64(1), hidden])];
        assert!(
            enclose_residual_jacobian(
                &source,
                &vars,
                &[RealBall::new(q(1, 1), q(1, 1)).unwrap()],
                24
            )
            .is_err()
        );
    }
    let sqrt = Expr::Pow(Arc::new(x), Arc::new(Expr::Rational(q(1, 2))));
    assert!(
        enclose_residual_jacobian(
            &[sqrt],
            &vars,
            &[RealBall::new(q(1, 1), q(1, 1)).unwrap()],
            24
        )
        .is_err()
    );
}

#[test]
fn contraction_and_strict_interior_refuse_false_or_singular_root_claims() {
    let vars = symbols(1);
    let x = Expr::Sym(vars[0].clone());
    let proposed = witness(vec![q(0, 1)], q(2, 1));
    for source in [
        power(x.clone(), 2),
        power(x.clone(), 2) - Expr::from_i64(1),
        power(x.clone(), 2) + Expr::Rational(q(1, 1_000_000)),
    ] {
        assert!(matches!(
            certify_root_box(&[source], &vars, &proposed),
            Err(RootBoxError::NotContractive)
        ));
    }
    let boundary = witness(vec![q(1, 1)], q(1, 1));
    assert_eq!(
        certify_root_box(&[x], &vars, &boundary),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    );
}

#[test]
fn shape_precision_and_source_bounds_refuse_before_numeric_work() {
    let vars = symbols(1);
    let x = Expr::Sym(vars[0].clone());
    let proposed = witness(vec![q(0, 1)], q(1, 1));
    assert_eq!(
        certify_root_box(&[], &[], &witness(vec![], q(1, 1))),
        Err(RootBoxError::DimensionMismatch)
    );
    assert_eq!(
        certify_root_box(&[x.clone(), x.clone()], &vars, &proposed),
        Err(RootBoxError::DimensionMismatch)
    );
    assert_eq!(
        enclose_residual_jacobian(&vec![x.clone(); 65], &vars, &proposed.domain, 24),
        Err(RootBoxError::DimensionMismatch)
    );
    for precision in [0, 7, 129, u32::MAX] {
        assert_eq!(
            enclose_residual_jacobian(std::slice::from_ref(&x), &vars, &proposed.domain, precision),
            Err(RootBoxError::InvalidPrecision)
        );
    }
    let wide = Expr::Add(vec![x.clone(); MAX_ROOT_BOX_NODES]);
    assert!(matches!(
        enclose_residual_jacobian(&[wide], &vars, &proposed.domain, 24),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let mut deep = x;
    for _ in 0..=MAX_ROOT_BOX_DEPTH {
        deep = Expr::Add(vec![deep]);
    }
    assert!(matches!(
        enclose_residual_jacobian(&[deep], &vars, &proposed.domain, 24),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let huge = BigInt::from(1) << (MAX_ROOT_BOX_LIMBS * 64) as u32;
    let hidden = Expr::Mul(vec![Expr::from_i64(0), Expr::Integer(huge)]);
    assert!(matches!(
        enclose_residual_jacobian(&[hidden], &vars, &proposed.domain, 24),
        Err(RootBoxError::ResourceLimit(_))
    ));
}

#[test]
fn bounded_transport_rejects_excess_rows_limbs_and_noncanonical_scalars() {
    let original = witness(vec![q(1, 1)], q(1, 10));
    let wire = serde_json::to_value(&original).unwrap();
    let decoded: RootBoxWitness = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(decoded, original);
    let mut bad = wire.clone();
    bad["extra"] = serde_json::json!(true);
    assert!(serde_json::from_value::<RootBoxWitness>(bad).is_err());
    let mut bad = wire.clone();
    bad["center"] = serde_json::json!(vec![q(1, 1); 17]);
    assert!(serde_json::from_value::<RootBoxWitness>(bad).is_err());
    let mut bad = wire.clone();
    bad["preconditioner"] = serde_json::json!(vec![vec![q(1, 1); 17]]);
    assert!(serde_json::from_value::<RootBoxWitness>(bad).is_err());
    let mut bad = wire.clone();
    bad["center"][0][0][1] = serde_json::json!(vec![1u32; MAX_ROOT_BOX_LIMBS as usize * 2 + 1]);
    assert!(serde_json::from_value::<RootBoxWitness>(bad).is_err());
    for malformed in [
        serde_json::json!([[1, [2]], [1, [2]]]),
        serde_json::json!([[1, [1]], [-1, [1]]]),
        serde_json::json!([[0, [1]], [1, [1]]]),
    ] {
        let mut bad = wire.clone();
        bad["center"][0] = malformed;
        assert!(serde_json::from_value::<RootBoxWitness>(bad).is_err());
    }
    // The 17th element must be refused without traversing its malformed body.
    let center = format!(
        "[{},{{\"unclosed\":",
        vec![serde_json::to_string(&q(1, 1)).unwrap(); 16].join(",")
    );
    let hostile = format!("{{\"center\":{center}");
    let error = serde_json::from_str::<RootBoxWitness>(&hostile).unwrap_err();
    assert!(error.to_string().contains("sequence length"), "{error}");
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
fn cancellation_at_every_checkpoint_publishes_no_generic_receipt() {
    let vars = symbols(1);
    let source = vec![Expr::Sym(vars[0].clone()) - Expr::from_i64(1)];
    let proposed = witness(vec![q(1, 1)], q(1, 4));
    let mut count = Cancel::default();
    verify_root_box(&source, &vars, &proposed, &mut count).unwrap();
    for stop in 0..count.visits {
        let mut meter = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            verify_root_box(&source, &vars, &proposed, &mut meter),
            Err(RootBoxError::Meter(MeterError::Cancelled))
        );
    }
    let mut count = Cancel::default();
    metered_enclose_residual_jacobian(&source, &vars, &proposed.domain, 24, &mut count).unwrap();
    for stop in 0..count.visits {
        let mut meter = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            metered_enclose_residual_jacobian(&source, &vars, &proposed.domain, 24, &mut meter),
            Err(RootBoxError::Meter(MeterError::Cancelled))
        );
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        verify_root_box(&source, &vars, &proposed, &mut empty),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn real_budget_accepts_rectangular_enclosures_and_preserves_atomic_refusal() {
    let vars = symbols(2);
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let source = vec![x.clone() + y.clone(), x.clone() * y.clone(), x - y];
    let domain = vec![
        RealBall::new(q(2, 1), q(1, 4)).unwrap(),
        RealBall::new(q(3, 1), q(1, 4)).unwrap(),
    ];
    let limit = 512 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    let accepted =
        metered_enclose_residual_jacobian(&source, &vars, &domain, 24, &mut budget).unwrap();
    assert_eq!(accepted.jacobian()[0], vec![RealBall::from_i64(1); 2]);
    assert_eq!(
        accepted.jacobian()[2],
        vec![RealBall::from_i64(1), RealBall::from_i64(-1)]
    );
    assert_eq!(
        accepted.jacobian()[1],
        vec![domain[1].clone(), domain[0].clone()]
    );
    assert!(budget.remaining(Dimension::ComputeSteps) < limit);
    assert!(budget.remaining(Dimension::MemoryBytes) < limit);
    assert!(budget.remaining(Dimension::AllocationCount) < limit);

    let mut insufficient = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_enclose_residual_jacobian(&source, &vars, &domain, 24, &mut insufficient),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    // The initial multidimensional reservation must remain all-or-nothing.
    assert_eq!(insufficient.remaining(Dimension::ComputeSteps), 1);
    assert_eq!(insufficient.remaining(Dimension::MemoryBytes), 1);
    assert_eq!(insufficient.remaining(Dimension::AllocationCount), 1);
}

#[test]
fn real_budget_accepts_a_nonlinear_root_certificate() {
    let vars = symbols(1);
    let source = vec![power(Expr::Sym(vars[0].clone()), 2) - Expr::from_i64(2)];
    let mut proposed = witness(vec![q(7, 5)], q(1, 10));
    proposed.preconditioner[0][0] = q(5, 14);
    let limit = 512 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    let accepted = verify_root_box(&source, &vars, &proposed, &mut budget).unwrap();
    let root = &accepted.root_enclosure()[0];
    assert!(&root.lower() * &root.lower() < q(2, 1));
    assert!(&root.upper() * &root.upper() > q(2, 1));
    assert!(accepted.contraction_bound() < &q(1, 1));
    assert!(budget.remaining(Dimension::ComputeSteps) < limit);
    assert!(budget.remaining(Dimension::MemoryBytes) < limit);
    assert!(budget.remaining(Dimension::AllocationCount) < limit);
}

#[test]
fn intrinsic_arithmetic_budget_applies_even_without_a_caller_meter() {
    let vars = symbols(16);
    // Source shape remains admitted and exact point arithmetic stays small,
    // but the dense product-rule work exceeds the independent cumulative cap.
    let source = vec![Expr::Mul(vec![Expr::Sym(vars[0].clone()); 200]); 64];
    let domain = vec![RealBall::from_i64(1); 16];
    assert!(matches!(
        enclose_residual_jacobian(&source, &vars, &domain, 24),
        Err(RootBoxError::ResourceLimit(_))
    ));
}
