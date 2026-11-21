//! Bounded binary64 evaluation of the entire, unnormalized sinc and its jets.
//!
//! Near zero, exact Taylor coefficients avoid both the removable singularity
//! and cancellation in the first two derivatives. Outside that interval the
//! quotient identities divide before multiplying, so finite large arguments
//! do not produce artificial overflow. These are approximate numeric values;
//! independently directed rational balls supply certified enclosures.

use super::{EvalError, charge_evaluation_steps};
use fsym_budget::BudgetMeter;

#[derive(Clone, Copy)]
pub(super) enum SincOrder {
    Value,
    First,
    Second,
}

pub(super) fn evaluate_sinc<M: BudgetMeter + ?Sized>(
    value: f64,
    order: SincOrder,
    meter: &mut M,
) -> Result<f64, EvalError> {
    charge_evaluation_steps(meter, 1)?;
    if !value.is_finite() {
        return Ok(f64::NAN);
    }
    let result = if value.abs() <= 1.0 {
        // sinc^(k)(x) = sum_{2n>=k} (-1)^n x^(2n-k)
        //                              / ((2n+1) (2n-k)!).
        // The first derivative is evaluated after factoring out x. This
        // retains tiny and subnormal slopes without dividing by x or x*x.
        let (derivative, start, mut term) = match order {
            SincOrder::Value => (0u32, 0u32, 1.0),
            SincOrder::First => (1, 1, -1.0 / 3.0),
            SincOrder::Second => (2, 1, -1.0 / 3.0),
        };
        let square = value * value;
        let mut sum = term;
        // At |x|<=1 the omitted alternating tail after these sixteen
        // updates is below 1e-36 for every order. Binary64 rounding is the
        // dominant error; no convergence guess grants proof authority.
        for n in start..start + 16 {
            charge_evaluation_steps(meter, 12)?;
            let numerator = f64::from(2 * n + 1);
            let denominator = f64::from(2 * n + 3)
                * f64::from(2 * n + 2 - derivative)
                * f64::from(2 * n + 1 - derivative);
            term *= -square * numerator / denominator;
            sum += term;
        }
        if matches!(order, SincOrder::First) {
            value * sum
        } else {
            sum
        }
    } else {
        charge_evaluation_steps(meter, 12)?;
        let sine = value.sin();
        let sinc = sine / value;
        match order {
            SincOrder::Value => sinc,
            SincOrder::First => (value.cos() - sinc) / value,
            SincOrder::Second => {
                let first = (value.cos() - sinc) / value;
                -sinc - 2.0 * (first / value)
            }
        }
    };
    charge_evaluation_steps(meter, 1)?;
    Ok(result)
}
