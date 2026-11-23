//! Numerical parameter samples propose; independent full-box checking accepts.

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_calculus::VerifiedResidualSystem;
use fsym_core::{
    BigInt, BigRational, ParametricRootBoxWitness, RealBall, RootBoxError, Symbol, parse_raw,
};
use fsym_numeric_consumer::{SystemConsumerError, propose_parametric_root_box_with_meter};
use std::error::Error;

const AMPLE: u64 = 1 << 40;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn ball(midpoint: BigRational, radius: BigRational) -> RealBall {
    RealBall::new(midpoint, radius).unwrap()
}

fn system(residuals: &[&str], variables: &[&str]) -> VerifiedResidualSystem {
    let residuals: Vec<_> = residuals
        .iter()
        .map(|source| parse_raw(source).unwrap())
        .collect();
    let variables: Vec<_> = variables.iter().map(|name| Symbol::new(*name)).collect();
    VerifiedResidualSystem::try_compile(&residuals, &variables).unwrap()
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(AMPLE, AMPLE))
}

fn verify(
    source: &VerifiedResidualSystem,
    witness: &ParametricRootBoxWitness,
) -> Result<fsym_core::ParametricRootBoxReceipt, RootBoxError> {
    let unknowns = source.num_residuals();
    fsym_core::certify_parametric_root_box(
        source.residuals(),
        &source.vars()[..unknowns],
        &source.vars()[unknowns..],
        witness,
    )
}

fn meter_cause(error: &(dyn Error + 'static)) -> Option<MeterError> {
    let mut cause = Some(error);
    while let Some(current) = cause {
        if let Some(meter) = current.downcast_ref::<MeterError>() {
            return Some(*meter);
        }
        cause = current.source();
    }
    None
}

#[test]
fn nonlinear_witness_retains_exact_parameter_variation_for_separate_protected_checking() {
    let source = system(&["x^2-p"], &["x", "p"]);
    let domain = [ball(q(4, 1), q(1, 1))];
    let mut meter = budget();
    let witness = propose_parametric_root_box_with_meter(
        &source,
        &[2.0],
        &[4.0],
        &[q(1, 1)],
        &domain,
        24,
        &mut meter,
    )
    .unwrap();
    assert_eq!(witness.center, [q(2, 1)]);
    assert_eq!(witness.parameter_domain, domain);
    assert_eq!(witness.unknown_domain[0].radius(), &q(1, 1));
    assert_eq!(witness.preconditioner, [vec![q(1, 4)]]);
    assert_eq!(meter.verifier_remaining(), AMPLE);
    let lease = meter.verifier_lease().unwrap();
    let receipt = fsym_core::verify_parametric_root_box(
        source.residuals(),
        &source.vars()[..1],
        &source.vars()[1..],
        &witness,
        &mut meter.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    let root = &receipt.root_enclosure()[0];
    assert!(&root.lower() * &root.lower() < q(3, 1));
    assert!(&root.upper() * &root.upper() > q(5, 1));
    assert!(meter.verifier_remaining() < AMPLE);
    assert_eq!(receipt.witness().parameter_domain, domain);
    assert_eq!(verify(&source, &witness).unwrap(), receipt);
}

#[test]
fn nonsymmetric_unknown_inverse_orientation_and_mutations_remain_observable() {
    let source = system(&["x+2*y-p", "y-p"], &["x", "y", "p"]);
    let witness = propose_parametric_root_box_with_meter(
        &source,
        &[0.0, 0.0],
        &[0.0],
        &[q(4, 1), q(2, 1)],
        &[ball(q(0, 1), q(1, 1))],
        24,
        &mut budget(),
    )
    .unwrap();
    assert_eq!(
        witness.preconditioner,
        [vec![q(1, 1), q(-2, 1)], vec![q(0, 1), q(1, 1)]]
    );
    let receipt = verify(&source, &witness).unwrap();
    for root in receipt.root_enclosure() {
        assert!(root.contains(&q(-1, 1)));
        assert!(root.contains(&q(1, 1)));
    }
    let mut forged = witness;
    forged.preconditioner[0][1] = q(0, 1);
    forged.preconditioner[1][0] = q(-2, 1);
    assert_eq!(verify(&source, &forged), Err(RootBoxError::NotContractive));
}

#[test]
fn fixed_exact_parameter_points_are_not_replaced_by_rounded_binary64_samples() {
    let source = system(&["x-p"], &["x", "p"]);
    let exact_parameter = RealBall::exact(q(1, 10));
    let witness = propose_parametric_root_box_with_meter(
        &source,
        &[0.1],
        &[0.1],
        &[q(1, 1000)],
        std::slice::from_ref(&exact_parameter),
        24,
        &mut budget(),
    )
    .unwrap();
    assert_ne!(witness.center[0], q(1, 10));
    assert_eq!(witness.parameter_domain, [exact_parameter]);
    let receipt = verify(&source, &witness).unwrap();
    assert!(receipt.root_enclosure()[0].contains(&q(1, 10)));
}

#[test]
fn pointwise_success_cannot_hide_missing_family_roots_parameter_poles_or_singularities() {
    for (source, approximation, parameters, radius, domain) in [
        ("x-p^2", 0.0, 0.0, q(1, 4), ball(q(0, 1), q(1, 1))),
        ("p*x-1", 1.0, 1.0, q(1, 2), ball(q(1, 1), q(1, 1))),
        ("x-log(p)", 0.0, 1.0, q(1, 2), ball(q(1, 1), q(2, 1))),
        ("x+0*(1/p)", 0.0, 1.0, q(1, 2), ball(q(1, 1), q(2, 1))),
    ] {
        let source = system(&[source], &["x", "p"]);
        let mut meter = budget();
        let witness = propose_parametric_root_box_with_meter(
            &source,
            &[approximation],
            &[parameters],
            &[radius],
            &[domain],
            24,
            &mut meter,
        )
        .unwrap();
        assert!(verify(&source, &witness).is_err());
        assert_eq!(meter.verifier_remaining(), AMPLE);
    }
    let singular = system(&["x^2-p"], &["x", "p"]);
    assert!(matches!(
        propose_parametric_root_box_with_meter(
            &singular,
            &[0.0],
            &[0.0],
            &[q(1, 10)],
            &[RealBall::exact(q(0, 1))],
            24,
            &mut budget(),
        ),
        Err(SystemConsumerError::LinearSolve(_))
    ));
}

#[test]
fn malformed_box_controls_and_unbounded_exact_payloads_refuse_before_solver_storage() {
    let source = system(&["x-p"], &["x", "p"]);
    for (radii, domain, precision) in [
        (vec![], vec![RealBall::exact(q(1, 1))], 24),
        (vec![q(0, 1)], vec![RealBall::exact(q(1, 1))], 24),
        (vec![q(-1, 1)], vec![RealBall::exact(q(1, 1))], 24),
        (vec![q(1, 10)], vec![], 24),
        (vec![q(1, 10)], vec![RealBall::exact(q(1, 1)); 2], 24),
        (vec![q(1, 10)], vec![RealBall::exact(q(1, 1))], 7),
        (vec![q(1, 10)], vec![RealBall::exact(q(1, 1))], 129),
    ] {
        let mut meter = budget();
        assert!(
            propose_parametric_root_box_with_meter(
                &source,
                &[1.0],
                &[1.0],
                &radii,
                &domain,
                precision,
                &mut meter,
            )
            .is_err()
        );
        assert_eq!(meter.remaining(Dimension::MemoryBytes), AMPLE);
        assert_eq!(meter.verifier_remaining(), AMPLE);
    }
    let huge = BigRational::from_integer(
        BigInt::from(2).pow(u32::try_from(64 * fsym_core::MAX_ROOT_BOX_LIMBS).unwrap()),
    );
    for (radii, domain) in [
        (vec![huge.clone()], vec![RealBall::exact(q(1, 1))]),
        (vec![q(1, 10)], vec![RealBall::exact(huge.clone())]),
        (vec![q(1, 10)], vec![ball(q(1, 1), huge)]),
    ] {
        let mut meter = budget();
        assert!(matches!(
            propose_parametric_root_box_with_meter(
                &source,
                &[1.0],
                &[1.0],
                &radii,
                &domain,
                24,
                &mut meter,
            ),
            Err(SystemConsumerError::Certification(
                RootBoxError::ResourceLimit(_)
            ))
        ));
        assert_eq!(meter.remaining(Dimension::MemoryBytes), AMPLE);
        assert_eq!(meter.verifier_remaining(), AMPLE);
    }
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    cancel_at: usize,
}

impl Interrupt {
    fn new(cancel_at: usize) -> Self {
        Self {
            budget: budget(),
            checkpoints: 0,
            cancel_at,
        }
    }
}

impl BudgetMeter for Interrupt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let index = self.checkpoints;
        self.checkpoints += 1;
        if index >= self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_witness_safe_point_and_last_resource_unit_can_refuse_without_publication() {
    let source = system(&["x-p"], &["x", "p"]);
    let make = |meter: &mut Budget| {
        propose_parametric_root_box_with_meter(
            &source,
            &[1.0],
            &[1.0],
            &[q(1, 10)],
            &[RealBall::exact(q(1, 1))],
            24,
            meter,
        )
    };
    let mut measured = Interrupt::new(usize::MAX);
    propose_parametric_root_box_with_meter(
        &source,
        &[1.0],
        &[1.0],
        &[q(1, 10)],
        &[RealBall::exact(q(1, 1))],
        24,
        &mut measured,
    )
    .unwrap();
    assert!(measured.checkpoints > 10);
    for cancel_at in 0..measured.checkpoints {
        let mut meter = Interrupt::new(cancel_at);
        let error = propose_parametric_root_box_with_meter(
            &source,
            &[1.0],
            &[1.0],
            &[q(1, 10)],
            &[RealBall::exact(q(1, 1))],
            24,
            &mut meter,
        )
        .unwrap_err();
        assert_eq!(
            meter_cause(&error),
            Some(MeterError::Cancelled),
            "safe point {cancel_at}"
        );
        assert_eq!(meter.budget.verifier_remaining(), AMPLE);
    }
    let mut measured = budget();
    make(&mut measured).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let spent = AMPLE - measured.remaining(dimension);
        for allowance in [0, spent - 1] {
            let mut limits = BudgetLimits::uniform(AMPLE, AMPLE);
            limits.dimensions[dimension.index()] = allowance;
            let mut meter = Budget::new(limits);
            let error = make(&mut meter).unwrap_err();
            assert!(matches!(
                meter_cause(&error),
                Some(MeterError::Budget(BudgetError::Exhausted { dimension: found, .. }))
                    if found == dimension
            ));
            assert_eq!(meter.verifier_remaining(), AMPLE);
        }
        let mut limits = BudgetLimits::uniform(AMPLE, AMPLE);
        limits.dimensions[dimension.index()] = spent;
        let mut meter = Budget::new(limits);
        make(&mut meter).unwrap();
        assert_eq!(meter.remaining(dimension), 0);
        let error = make(&mut meter).unwrap_err();
        assert!(matches!(
            meter_cause(&error),
            Some(MeterError::Budget(BudgetError::Exhausted { dimension: found, .. }))
                if found == dimension
        ));
        assert_eq!(meter.verifier_remaining(), AMPLE);
    }
}
