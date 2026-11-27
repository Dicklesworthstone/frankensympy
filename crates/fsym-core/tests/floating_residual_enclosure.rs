//! Source-bound exports keep the exact input domain, typed axes, and all rows.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, Expr, FloatingEnclosureError, RealBall, ResidualJacobianEnclosure, Symbol,
    enclose_residual_jacobian,
};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn rectangle() -> ResidualJacobianEnclosure {
    let x = Expr::symbol("x");
    let y = Expr::symbol("y");
    enclose_residual_jacobian(
        &[
            x.clone() - y.clone(),
            Expr::from_i64(2) * x + Expr::from_i64(3) * y,
            Expr::from_i64(0),
        ],
        &[Symbol::new("y"), Symbol::new("x")],
        &[
            RealBall::new(q(-1, 7), q(1, 21)).unwrap(),
            RealBall::exact(q(1, 10)),
        ],
        40,
    )
    .unwrap()
}

#[test]
fn rectangular_outputs_bind_the_exact_source_and_preserve_typed_axis_order() {
    let source = rectangle();
    let narrow = source.metered_to_f32_bounds(&mut Unbounded).unwrap();
    let wide = source.metered_to_f64_bounds(&mut Unbounded).unwrap();
    assert!(std::ptr::eq(narrow.source(), &source));
    assert!(std::ptr::eq(wide.source(), &source));
    assert_eq!(narrow.domain().len(), 2);
    assert_eq!(wide.residual_values().len(), 3);
    let ten = 0.1_f32;
    assert_eq!(narrow.domain()[1], (f32::from_bits(ten.to_bits() - 1), ten));
    let ten = 0.1_f64;
    assert_eq!(wide.domain()[1], (f64::from_bits(ten.to_bits() - 1), ten));
    let derivatives = [[-1, 1], [3, 2], [0, 0]];
    for (i, row) in derivatives.iter().enumerate() {
        assert_eq!(wide.jacobian()[i].len(), 2);
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(narrow.jacobian()[i][j], (v as f32, v as f32));
            assert_eq!(wide.jacobian()[i][j], (f64::from(v), f64::from(v)));
        }
        assert_eq!(
            narrow.residual_values()[i],
            source.residual_values()[i]
                .metered_to_f32_bounds(&mut Unbounded)
                .unwrap()
                .unwrap()
        );
        assert_eq!(
            wide.residual_values()[i],
            source.residual_values()[i]
                .metered_to_f64_bounds(&mut Unbounded)
                .unwrap()
                .unwrap()
        );
    }
    assert_eq!(source.domain()[1].radius(), &q(0, 1));
}

#[test]
fn infinite_bounds_and_negative_zero_survive_without_narrowing_the_exact_box() {
    let huge = BigInt::from(1_i64) << 160_u32;
    let point = BigRational::from_integer(huge.clone());
    let tiny = BigRational::new((-1).into(), huge);
    let source = enclose_residual_jacobian(
        &[Expr::symbol("x"), Expr::symbol("y")],
        &[Symbol::new("x"), Symbol::new("y")],
        &[RealBall::exact(point), RealBall::exact(tiny)],
        64,
    )
    .unwrap();
    let narrow = source.metered_to_f32_bounds(&mut Unbounded).unwrap();
    assert_eq!(narrow.domain()[0], (f32::MAX, f32::INFINITY));
    let (lo, hi) = narrow.domain()[1];
    assert_eq!(lo.to_bits(), (-f32::from_bits(1)).to_bits());
    assert_eq!(hi.to_bits(), (-0.0_f32).to_bits());
    let wide = source.metered_to_f64_bounds(&mut Unbounded).unwrap();
    assert_eq!(wide.domain()[0], (2_f64.powi(160), 2_f64.powi(160)));
    assert_eq!(wide.domain()[1], (-2_f64.powi(-160), -2_f64.powi(-160)));
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    cancel_at: usize,
}

impl BudgetMeter for Interrupt {
    fn charge(&mut self, d: Dimension, n: u64) -> Result<(), MeterError> {
        self.budget.charge(d, n)
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.budget.charge_batch(charges)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.checkpoints == self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn conversion_and_publication_are_atomic_on_a_single_ordinary_ledger() {
    let source = rectangle();
    let limits = BudgetLimits::uniform(u64::MAX, 91);
    let mut complete = Interrupt {
        budget: Budget::new(limits),
        checkpoints: 0,
        cancel_at: 0,
    };
    source.metered_to_f64_bounds(&mut complete).unwrap();
    for stop in [
        1,
        complete.checkpoints / 2,
        complete.checkpoints - 1,
        complete.checkpoints,
    ] {
        let mut meter = Interrupt {
            budget: Budget::new(limits),
            checkpoints: 0,
            cancel_at: stop,
        };
        assert!(matches!(
            source.metered_to_f64_bounds(&mut meter),
            Err(FloatingEnclosureError::Meter(MeterError::Cancelled))
        ));
        assert_eq!(meter.checkpoints, stop);
        assert_eq!(meter.budget.verifier_remaining(), 91);
    }
    for dim in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut short = limits;
        short.dimensions[dim.index()] -= complete.budget.remaining(dim) + 1;
        let mut budget = Budget::new(short);
        assert!(matches!(
            source.metered_to_f64_bounds(&mut budget),
            Err(FloatingEnclosureError::Meter(_))
        ));
        assert_eq!(budget.verifier_remaining(), 91);
    }
}

#[test]
fn residual_and_derivative_endpoints_are_rounded_directly_in_the_target_format() {
    let huge = BigInt::from(1_i64) << 1100_u32;
    let denominator = BigInt::from(1_i64) << 80_u32;
    let above_one = BigRational::new(&denominator + &BigInt::from(1_i64), denominator);
    let source = enclose_residual_jacobian(
        &[
            Expr::Rational(above_one),
            Expr::Rational(BigRational::from_integer(huge)) * Expr::symbol("x"),
        ],
        &[Symbol::new("x")],
        &[RealBall::from_i64(0)],
        100,
    )
    .unwrap();
    let narrow = source.metered_to_f32_bounds(&mut Unbounded).unwrap();
    assert_eq!(
        narrow.residual_values()[0],
        (1.0, f32::from_bits(1.0_f32.to_bits() + 1))
    );
    let wide = source.metered_to_f64_bounds(&mut Unbounded).unwrap();
    assert_eq!(wide.residual_values()[1], (0.0, 0.0));
    assert_eq!(wide.jacobian()[1][0], (f64::MAX, f64::INFINITY));
}
