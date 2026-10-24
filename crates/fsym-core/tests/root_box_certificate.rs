use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_NODES,
    RealBall, RootBoxError, RootBoxWitness2, Symbol, certify_root_box_2x2, verify_root_box_2x2,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn symbols() -> [Symbol; 2] {
    [Symbol::new("x"), Symbol::new("y")]
}
fn affine() -> [Expr; 2] {
    let [x, y] = symbols().map(Expr::Sym);
    [
        Expr::from_i64(2) * x.clone() + y.clone() - Expr::from_i64(5),
        x - y - Expr::from_i64(1),
    ]
}
fn affine_witness() -> RootBoxWitness2 {
    let center = [q(21, 10), q(1, 1)];
    RootBoxWitness2 {
        domain: center.clone().map(|x| RealBall::new(x, q(1, 4)).unwrap()),
        center,
        preconditioner: [[q(1, 3), q(1, 3)], [q(1, 3), q(-2, 3)]],
        precision_digits: 24,
    }
}
fn identity_witness(center: [BigRational; 2], radius: BigRational) -> RootBoxWitness2 {
    RootBoxWitness2 {
        domain: center
            .clone()
            .map(|x| RealBall::new(x, radius.clone()).unwrap()),
        center,
        preconditioner: [[q(1, 1), q(0, 1)], [q(0, 1), q(1, 1)]],
        precision_digits: 24,
    }
}

#[test]
fn hand_supplied_affine_witness_proves_the_exact_root_without_a_generator() {
    let witness = affine_witness();
    let receipt = certify_root_box_2x2(&affine(), &symbols(), &witness).unwrap();
    assert_eq!(
        receipt.root_enclosure(),
        &[RealBall::exact(q(2, 1)), RealBall::exact(q(1, 1))]
    );
    assert_eq!(receipt.contraction_bound(), &q(0, 1));
    assert_eq!(receipt.residuals(), &affine());
    assert_eq!(receipt.variables(), &symbols());
    assert_eq!(receipt.witness(), &witness);
    let encoded = serde_json::to_vec(&witness).unwrap();
    let decoded = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        certify_root_box_2x2(&affine(), &symbols(), &decoded).unwrap(),
        receipt
    );
}

#[test]
fn nonlinear_irrational_roots_receive_strict_rational_enclosures() {
    let [x, y] = symbols().map(Expr::Sym);
    let equations = [
        x.pow(Expr::from_i64(2)) - Expr::from_i64(2),
        y.pow(Expr::from_i64(2)) - Expr::from_i64(3),
    ];
    let mut witness = identity_witness([q(7, 5), q(17, 10)], q(1, 10));
    witness.preconditioner = [[q(5, 14), q(0, 1)], [q(0, 1), q(5, 17)]];
    let receipt = certify_root_box_2x2(&equations, &symbols(), &witness).unwrap();
    for (i, target) in [2, 3].into_iter().enumerate() {
        let enclosure = &receipt.root_enclosure()[i];
        assert!(enclosure.lower() > q(0, 1));
        assert!(&enclosure.lower() * &enclosure.lower() < q(target, 1));
        assert!(&enclosure.upper() * &enclosure.upper() > q(target, 1));
        assert!(enclosure.lower() > witness.domain[i].lower());
        assert!(enclosure.upper() < witness.domain[i].upper());
    }
    assert!(receipt.contraction_bound() < &q(1, 1));
}

#[test]
fn small_residuals_multiple_roots_and_singular_roots_do_not_imply_certification() {
    let [x, y] = symbols().map(Expr::Sym);
    let witness = identity_witness([q(0, 1), q(0, 1)], q(2, 1));
    for first in [
        x.clone().pow(Expr::from_i64(2)) + Expr::Rational(q(1, 1_000_000)),
        x.clone().pow(Expr::from_i64(2)) - Expr::from_i64(1),
        x.pow(Expr::from_i64(2)),
    ] {
        assert!(matches!(
            certify_root_box_2x2(&[first, y.clone()], &symbols(), &witness),
            Err(RootBoxError::NotContractive)
        ));
    }
}

#[test]
fn forged_preconditioners_centers_boxes_and_systems_are_recomputed() {
    let original = affine_witness();
    let mut bad = original.clone();
    bad.preconditioner = [[q(0, 1), q(0, 1)], [q(0, 1), q(0, 1)]];
    assert_eq!(
        certify_root_box_2x2(&affine(), &symbols(), &bad),
        Err(RootBoxError::SingularPreconditioner)
    );
    let mut bad = original.clone();
    bad.center[0] = q(5, 1);
    assert_eq!(
        certify_root_box_2x2(&affine(), &symbols(), &bad),
        Err(RootBoxError::CenterOutsideBox { coordinate: 0 })
    );
    let mut bad = original.clone();
    bad.domain[0] = RealBall::exact(bad.center[0].clone());
    assert_eq!(
        certify_root_box_2x2(&affine(), &symbols(), &bad),
        Err(RootBoxError::DegenerateBox { coordinate: 0 })
    );
    let mut bad = original.clone();
    bad.center[0] = q(5, 1);
    bad.domain[0] = RealBall::new(q(5, 1), q(1, 4)).unwrap();
    assert_eq!(
        certify_root_box_2x2(&affine(), &symbols(), &bad),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    );
    let mut altered = affine();
    altered[0] = altered[0].clone() + Expr::from_i64(100);
    assert!(certify_root_box_2x2(&altered, &symbols(), &original).is_err());
    assert_eq!(
        certify_root_box_2x2(&affine(), &[Symbol::new("x"), Symbol::new("x")], &original),
        Err(RootBoxError::DuplicateVariable)
    );
    for precision in [0, 7, 129, u32::MAX] {
        let mut bad = original.clone();
        bad.precision_digits = precision;
        assert_eq!(
            certify_root_box_2x2(&affine(), &symbols(), &bad),
            Err(RootBoxError::InvalidPrecision)
        );
    }
}

#[test]
fn original_domain_is_checked_through_zero_multipliers_and_held_powers() {
    let [x, y] = symbols().map(Expr::Sym);
    let log_x = Expr::Function("log".into(), vec![x.clone()]);
    let hidden = Expr::Mul(vec![Expr::from_i64(0), log_x.clone()]);
    let equations = [
        Expr::Add(vec![hidden, x.clone() - Expr::from_i64(1)]),
        y.clone(),
    ];
    let wide = identity_witness([q(1, 1), q(0, 1)], q(1, 1));
    assert!(matches!(
        certify_root_box_2x2(&equations, &symbols(), &wide),
        Err(RootBoxError::Domain(_))
    ));
    let narrow = identity_witness([q(1, 1), q(0, 1)], q(1, 10));
    certify_root_box_2x2(&equations, &symbols(), &narrow).unwrap();
    let held = Expr::Pow(Arc::new(log_x), Arc::new(Expr::from_i64(0)));
    let equations = [Expr::Add(vec![held, x - Expr::from_i64(2)]), y];
    assert!(matches!(
        certify_root_box_2x2(&equations, &symbols(), &wide),
        Err(RootBoxError::Domain(_))
    ));
}

#[test]
fn transcendental_enclosures_and_chain_rules_are_checked_on_the_entire_box() {
    let [x, y] = symbols().map(Expr::Sym);
    let witness = identity_witness([q(0, 1), q(0, 1)], q(1, 10));
    for first in [
        Expr::Function("sin".into(), vec![x.clone()]),
        Expr::Function("exp".into(), vec![x.clone()]) - Expr::from_i64(1),
        Expr::Function("atan".into(), vec![x.clone()]),
    ] {
        let receipt = certify_root_box_2x2(&[first, y.clone()], &symbols(), &witness).unwrap();
        assert!(receipt.root_enclosure()[0].contains(&q(0, 1)));
    }
    let mut positive = identity_witness([q(1, 1), q(0, 1)], q(1, 100));
    positive.preconditioner[0][0] = q(2, 1);
    let sqrt_x = Expr::Pow(Arc::new(x), Arc::new(Expr::Rational(q(1, 2))));
    let receipt =
        certify_root_box_2x2(&[sqrt_x - Expr::from_i64(1), y], &symbols(), &positive).unwrap();
    assert!(receipt.root_enclosure()[0].contains(&q(1, 1)));
}

#[test]
fn strict_inclusion_does_not_accept_a_root_only_on_the_box_boundary() {
    let equations = symbols().map(Expr::Sym);
    let witness = identity_witness([q(1, 1), q(0, 1)], q(1, 1));
    assert_eq!(
        certify_root_box_2x2(&equations, &symbols(), &witness),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    );
}

#[derive(Default)]
struct CancellingMeter {
    checkpoints: usize,
    stop_at: Option<usize>,
}
impl BudgetMeter for CancellingMeter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let current = self.checkpoints;
        self.checkpoints += 1;
        if self.stop_at == Some(current) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_safe_point_and_exhausted_budgets_publish_no_receipt() {
    let mut counter = CancellingMeter::default();
    verify_root_box_2x2(&affine(), &symbols(), &affine_witness(), &mut counter).unwrap();
    for stop_at in 0..counter.checkpoints {
        let mut meter = CancellingMeter {
            checkpoints: 0,
            stop_at: Some(stop_at),
        };
        assert_eq!(
            verify_root_box_2x2(&affine(), &symbols(), &affine_witness(), &mut meter),
            Err(RootBoxError::Meter(MeterError::Cancelled))
        );
    }
    let mut budget = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        verify_root_box_2x2(&affine(), &symbols(), &affine_witness(), &mut budget),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
}

#[test]
fn hostile_expressions_and_untrusted_wire_fields_refuse() {
    let [x, y] = symbols().map(Expr::Sym);
    let witness = identity_witness([q(1, 1), q(0, 1)], q(1, 10));
    let mut admitted = x.clone() - Expr::from_i64(1);
    for _ in 0..(MAX_ROOT_BOX_DEPTH - 2) {
        admitted = Expr::Add(vec![admitted]);
    }
    certify_root_box_2x2(&[admitted, y.clone()], &symbols(), &witness).unwrap();
    let mut deep = x.clone();
    for _ in 0..=MAX_ROOT_BOX_DEPTH {
        deep = Expr::Add(vec![deep]);
    }
    assert!(matches!(
        certify_root_box_2x2(&[deep, y.clone()], &symbols(), &witness),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let wide = Expr::Add(vec![Expr::from_i64(1); MAX_ROOT_BOX_NODES + 1]);
    assert!(matches!(
        certify_root_box_2x2(&[wide, y.clone()], &symbols(), &witness),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let huge = BigInt::from(1) << (MAX_ROOT_BOX_LIMBS * 64) as u32;
    assert!(matches!(
        certify_root_box_2x2(&[Expr::Integer(huge), y.clone()], &symbols(), &witness),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let unknown = Expr::Function("uninterpreted".into(), vec![x]);
    assert!(matches!(
        certify_root_box_2x2(&[unknown, y], &symbols(), &witness),
        Err(RootBoxError::Unsupported(_))
    ));
    let large_name = Symbol::new("s".repeat(4097));
    assert!(matches!(
        certify_root_box_2x2(&affine(), &[large_name.clone(), large_name], &witness),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let mut forged = serde_json::to_value(affine_witness()).unwrap();
    forged["verified"] = serde_json::json!(true);
    assert!(serde_json::from_value::<RootBoxWitness2>(forged).is_err());
}
