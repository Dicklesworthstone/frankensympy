//! Global box optimization through the immutable public objective API.

use frankensympy::CertificationError;
use frankensympy::objective::ObjectiveProblem;
use frankensympy::optimization::OptimizationError;
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::BoxOptimizationOptions;
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationReceipt,
    BoxOptimizationWitness, BoxPartitionStep, Expr, RealBall, RootBoxError, Symbol, SymbolIdentity,
};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn problem(source: &str, axes: &[&str]) -> ObjectiveProblem {
    ObjectiveProblem::parse_source(
        source,
        &axes.iter().map(|name| (*name).into()).collect::<Vec<_>>(),
        &mut Unbounded,
    )
    .unwrap()
}

fn options() -> BoxOptimizationOptions {
    BoxOptimizationOptions {
        precision_digits: 16,
        max_leaves: 128,
    }
}

fn checked(
    problem: &ObjectiveProblem,
    domain: &[RealBall],
    kind: BoxExtremumKind,
    tolerance: &BigRational,
) -> BoxOptimizationReceipt {
    let witness = problem
        .propose_box_optimization(domain, kind, tolerance, options(), &mut Unbounded)
        .unwrap();
    let receipt = problem
        .verify_box_optimization(domain, kind, tolerance, &witness, &mut Unbounded)
        .unwrap();
    assert_eq!(receipt.objective(), problem.objective());
    assert_eq!(receipt.variables(), problem.variables());
    assert_eq!(receipt.domain(), domain);
    assert_eq!(receipt.kind(), kind);
    assert_eq!(receipt.absolute_tolerance(), tolerance);
    assert!(receipt.gap() >= &q(0, 1));
    assert!(receipt.gap() <= tolerance);
    receipt
}

#[test]
fn interior_quadratic_and_flat_quartic_have_global_value_certificates() {
    for source in ["(x-1)^2", "(x-1)^4"] {
        let result = checked(
            &problem(source, &["x"]),
            &[interval(0, 2)],
            BoxExtremumKind::Minimum,
            &q(1, 1000),
        );
        assert_eq!(result.point(), &[q(1, 1)]);
        assert_eq!(result.value(), &RealBall::from_i64(0));
        assert_eq!(result.point_value(), &RealBall::from_i64(0));
        assert_eq!(result.gap(), &q(0, 1));
    }
}

#[test]
fn boundary_minimum_and_maximum_cover_the_same_closed_box() {
    let objective = problem("x-2*y", &["x", "y"]);
    let domain = [interval(-1, 2), interval(-3, 4)];
    for (kind, point, value) in [
        (BoxExtremumKind::Minimum, vec![q(-1, 1), q(4, 1)], -9),
        (BoxExtremumKind::Maximum, vec![q(2, 1), q(-3, 1)], 8),
    ] {
        let result = checked(&objective, &domain, kind, &q(1, 1000));
        assert_eq!(result.point(), point);
        assert_eq!(result.value(), &RealBall::from_i64(value));
        assert_eq!(result.gap(), &q(0, 1));
    }
}

#[test]
fn irrational_nonconvex_minima_produce_a_tolerance_bound_not_an_exact_optimizer() {
    let result = checked(
        &problem("(x^2-2)^2", &["x"]),
        &[interval(-2, 2)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
    );
    // The global minimum is zero at +/-sqrt(2), and no rational proposal can
    // equal either optimizer. The certificate must preserve that distinction.
    assert!(result.value().contains(&q(0, 1)));
    assert!(result.point_value().lower() > q(0, 1));
    assert!(result.gap() > &q(0, 1));
    assert!(result.leaf_count() > 1);
    let delta = &result.point()[0] * &result.point()[0] - q(2, 1);
    assert_eq!(result.point_value(), &RealBall::exact(&delta * &delta));
}

#[test]
fn fixed_coordinates_and_typed_same_name_axes_preserve_the_original_problem() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let objective = ObjectiveProblem::new(
        Expr::Add(vec![
            Expr::Sym(x.clone()),
            Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(other.clone())]),
        ]),
        vec![other, x],
        &mut Unbounded,
    )
    .unwrap();
    let result = checked(
        &objective,
        &[RealBall::from_i64(3), interval(-1, 1)],
        BoxExtremumKind::Minimum,
        &q(1, 1000),
    );
    assert_eq!(result.point(), &[q(3, 1), q(-1, 1)]);
    assert_eq!(result.value(), &RealBall::from_i64(5));
}

#[test]
fn untrusted_partition_point_and_accuracy_claims_are_independently_rejected() {
    let objective = problem("x^2", &["x"]);
    let domain = [interval(-1, 1)];
    let tolerance = q(1, 1000);
    let mut witness = BoxOptimizationWitness {
        partition: vec![BoxPartitionStep::Leaf],
        point: vec![q(0, 1)],
        precision_digits: 16,
    };
    objective
        .verify_box_optimization(
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            &witness,
            &mut Unbounded,
        )
        .unwrap();
    for partition in [
        vec![BoxPartitionStep::Split { axis: 0 }, BoxPartitionStep::Leaf],
        vec![BoxPartitionStep::Leaf, BoxPartitionStep::Leaf],
        vec![
            BoxPartitionStep::Split { axis: 1 },
            BoxPartitionStep::Leaf,
            BoxPartitionStep::Leaf,
        ],
    ] {
        witness.partition = partition;
        assert!(
            objective
                .verify_box_optimization(
                    &domain,
                    BoxExtremumKind::Minimum,
                    &tolerance,
                    &witness,
                    &mut Unbounded,
                )
                .is_err()
        );
    }
    witness.partition = vec![BoxPartitionStep::Leaf];
    witness.point[0] = q(2, 1);
    assert!(matches!(
        objective.verify_box_optimization(
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            &witness,
            &mut Unbounded,
        ),
        Err(OptimizationError::Optimization(
            BoxOptimizationError::Verification(RootBoxError::CenterOutsideBox { .. })
        ))
    ));
    witness.point[0] = q(1, 1);
    assert!(matches!(
        objective.verify_box_optimization(
            &domain,
            BoxExtremumKind::Minimum,
            &tolerance,
            &witness,
            &mut Unbounded,
        ),
        Err(OptimizationError::Optimization(
            BoxOptimizationError::AccuracyNotReached
        ))
    ));
}

#[test]
fn original_poles_and_nonsmooth_sources_cannot_be_discarded() {
    for source in ["0*(1/x)", "log(x)-log(x)", "(1/x)^0", "sqrt(x)", "abs(x)"] {
        let objective = problem(source, &["x"]);
        let domain = [interval(-1, 1)];
        let tolerance = q(1, 1000);
        assert!(
            objective
                .propose_box_optimization(
                    &domain,
                    BoxExtremumKind::Minimum,
                    &tolerance,
                    BoxOptimizationOptions {
                        max_leaves: 4,
                        ..options()
                    },
                    &mut Unbounded,
                )
                .is_err(),
            "{source}"
        );
        assert!(
            objective
                .verify_box_optimization(
                    &domain,
                    BoxExtremumKind::Minimum,
                    &tolerance,
                    &BoxOptimizationWitness {
                        partition: vec![BoxPartitionStep::Leaf],
                        point: vec![q(0, 1)],
                        precision_digits: 16,
                    },
                    &mut Unbounded,
                )
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn invalid_tolerances_and_exhausted_search_bounds_return_no_receipt() {
    let objective = problem("(x^2-2)^2", &["x"]);
    for tolerance in [q(0, 1), q(-1, 1)] {
        assert!(matches!(
            objective.propose_box_optimization(
                &[interval(-2, 2)],
                BoxExtremumKind::Minimum,
                &tolerance,
                options(),
                &mut Unbounded,
            ),
            Err(OptimizationError::Optimization(
                BoxOptimizationError::InvalidTolerance
            ))
        ));
    }
    for cap in [0, 1, 1025] {
        assert!(
            objective
                .propose_box_optimization(
                    &[interval(-2, 2)],
                    BoxExtremumKind::Minimum,
                    &q(1, 1000),
                    BoxOptimizationOptions {
                        max_leaves: cap,
                        ..options()
                    },
                    &mut Unbounded,
                )
                .is_err()
        );
    }
}

#[derive(Default)]
struct CancelAt {
    checkpoints: usize,
    at: usize,
    stopped: bool,
    charged: u64,
}

impl BudgetMeter for CancelAt {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.charged += amount;
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.checkpoints += 1;
        if self.checkpoints == self.at {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_is_terminal_in_both_generation_and_independent_verification() {
    let objective = problem("x^2", &["x"]);
    let domain = [interval(-1, 1)];
    let tolerance = q(1, 1000);
    let witness = BoxOptimizationWitness {
        partition: vec![BoxPartitionStep::Leaf],
        point: vec![q(0, 1)],
        precision_digits: 16,
    };
    for verifying in [false, true] {
        let run = |meter: &mut CancelAt| {
            if verifying {
                objective
                    .verify_box_optimization(
                        &domain,
                        BoxExtremumKind::Minimum,
                        &tolerance,
                        &witness,
                        meter,
                    )
                    .map(|_| ())
            } else {
                objective
                    .propose_box_optimization(
                        &domain,
                        BoxExtremumKind::Minimum,
                        &tolerance,
                        options(),
                        meter,
                    )
                    .map(|_| ())
            }
        };
        let mut measured = CancelAt::default();
        run(&mut measured).unwrap();
        for at in [1, 2, measured.checkpoints / 2, measured.checkpoints] {
            let mut stopped = CancelAt {
                at,
                ..CancelAt::default()
            };
            assert!(matches!(
                run(&mut stopped),
                Err(OptimizationError::Certification(CertificationError::Meter(
                    MeterError::Cancelled
                ))) | Err(OptimizationError::Optimization(
                    BoxOptimizationError::Verification(RootBoxError::Meter(MeterError::Cancelled))
                ))
            ));
            assert!(stopped.stopped);
            assert!(stopped.charged <= measured.charged);
        }
    }
}
