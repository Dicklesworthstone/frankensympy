//! Bounded binary64 Gaussian error functions for approximate bytecode.
//!
//! The central Maclaurin series and the positive-tail Mills continued fraction
//! avoid cancellation in erfc. These are numerical approximations, not proof
//! receipts; certified values come from fsym-core's independent rational balls.
//! Every iteration charges the caller's existing evaluation ledger.

use super::{EvalError, charge_evaluation_steps};
use fsym_budget::BudgetMeter;

pub(super) fn evaluate_error_function<M: BudgetMeter + ?Sized>(
    value: f64,
    complementary: bool,
    meter: &mut M,
) -> Result<f64, EvalError> {
    charge_evaluation_steps(meter, 1)?;
    if value.is_nan() {
        return Ok(f64::NAN);
    }
    if value == 0.0 {
        return Ok(if complementary { 1.0 } else { value });
    }
    let x = value.abs();
    let root_pi = std::f64::consts::PI.sqrt();
    let (erf, erfc) = if x <= 1.0 {
        // (-1)^n x^(2n+1)/n!, divided by 2n+1 in the sum. In this
        // interval all terms decrease and cancellation stays well conditioned.
        let square = x * x;
        let mut term = x;
        let mut sum = x;
        let mut converged = false;
        for n in 1..=128u32 {
            charge_evaluation_steps(meter, 12)?;
            term *= -square / f64::from(n);
            let contribution = term / f64::from(2 * n + 1);
            sum += contribution;
            if contribution.abs() <= f64::EPSILON * sum.abs() / 4.0 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(EvalError::ErrorFunctionConvergenceLimit);
        }
        let erf = 2.0 * sum / root_pi;
        (erf, 1.0 - erf)
    } else if x >= 28.0 {
        // erfc(28) is already below the smallest binary64 subnormal.
        // Finite underflow is part of the approximate evaluator's contract.
        (1.0, 0.0)
    } else {
        // sqrt(pi)*exp(x*x)*erfc(x) =
        // 1 / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))).
        // Successive numerator/denominator continuants use bounded storage.
        let mut previous_a = 0.0_f64;
        let mut a = 1.0_f64;
        let mut previous_b = 1.0_f64;
        let mut b = x;
        let mut ratio = a / b;
        let mut converged = false;
        for n in 1..=512u32 {
            charge_evaluation_steps(meter, 20)?;
            let largest = a
                .abs()
                .max(previous_a.abs())
                .max(b.abs())
                .max(previous_b.abs());
            if largest > 1e150 {
                a *= 1e-150;
                previous_a *= 1e-150;
                b *= 1e-150;
                previous_b *= 1e-150;
            }
            let weight = f64::from(n) / 2.0;
            let next_a = x * a + weight * previous_a;
            let next_b = x * b + weight * previous_b;
            let next_ratio = next_a / next_b;
            if !next_ratio.is_finite() || next_ratio <= 0.0 {
                return Err(EvalError::NonFiniteValue);
            }
            if (next_ratio - ratio).abs() <= 2.0 * f64::EPSILON * next_ratio {
                ratio = next_ratio;
                converged = true;
                break;
            }
            previous_a = a;
            a = next_a;
            previous_b = b;
            b = next_b;
            ratio = next_ratio;
        }
        if !converged {
            return Err(EvalError::ErrorFunctionConvergenceLimit);
        }
        let erfc = (-(x * x)).exp() * ratio / root_pi;
        (1.0 - erfc, erfc)
    };
    let result = if complementary {
        if value.is_sign_negative() {
            2.0 - erfc
        } else {
            erfc
        }
    } else if value.is_sign_negative() {
        -erf
    } else {
        erf
    };
    charge_evaluation_steps(meter, 1)?;
    Ok(result)
}
