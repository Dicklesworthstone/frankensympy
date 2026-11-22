//! Private value and coordinate-slope bounds for continuous Abs compositions.
//!
//! Each admitted coordinate slice is absolutely continuous. `slopes[i]`
//! encloses its derivative wherever that derivative exists, so integrating
//! along coordinate segments proves a centered value bound and monotonicity.
//! At an absolute-value corner this is a Lipschitz slope enclosure, not a C1
//! derivative, a subgradient certificate, or a Hessian. This distinct type is
//! consumed only by global value-gap optimization; it never constructs `Jet`,
//! `ObjectiveJet`, root receipts, or derivative enclosures.
//!
//! All ordinary smooth primitives keep their strict original domains. Visited
//! Piecewise guards retain the independent neighborhood-stability policy.
//! Every original child is checked before a zero product or power can cancel
//! it. No symbolic differentiation or candidate-produced bounds are used.

use super::{
    MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_NODES, RootBoxError, Work, check_ball,
    checked,
};
use crate::{BigRational, Constant, Expr, RealBall, Symbol};
use fsym_budget::BudgetMeter;
use num_traits::{One, Signed, ToPrimitive, Zero};

pub(super) struct LipschitzEnclosure {
    pub(super) value: RealBall,
    pub(super) slopes: Vec<RealBall>,
}

impl LipschitzEnclosure {
    // The evaluator charges each node and its coordinate storage first.
    fn constant(value: RealBall, dimensions: usize) -> Self {
        Self {
            value,
            slopes: vec![RealBall::from_i64(0); dimensions],
        }
    }

    fn checked(self) -> Result<Self, RootBoxError> {
        check_ball(&self.value)?;
        for slope in &self.slopes {
            check_ball(slope)?;
        }
        Ok(self)
    }
}

fn add(
    a: LipschitzEnclosure,
    b: LipschitzEnclosure,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<LipschitzEnclosure, RootBoxError> {
    let value = work.add(&a.value, &b.value)?;
    let mut slopes = Vec::with_capacity(a.slopes.len());
    for (left, right) in a.slopes.iter().zip(&b.slopes) {
        slopes.push(work.add(left, right)?);
    }
    LipschitzEnclosure { value, slopes }.checked()
}

fn multiply(
    a: LipschitzEnclosure,
    b: LipschitzEnclosure,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<LipschitzEnclosure, RootBoxError> {
    let value = work.mul(&a.value, &b.value)?;
    let mut slopes = Vec::with_capacity(a.slopes.len());
    for (left, right) in a.slopes.iter().zip(&b.slopes) {
        let first = work.mul(left, &b.value)?;
        let second = work.mul(&a.value, right)?;
        slopes.push(work.add(&first, &second)?);
    }
    LipschitzEnclosure { value, slopes }.checked()
}

fn chain(
    argument: LipschitzEnclosure,
    value: RealBall,
    slope: RealBall,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<LipschitzEnclosure, RootBoxError> {
    check_ball(&value)?;
    check_ball(&slope)?;
    let mut slopes = Vec::with_capacity(argument.slopes.len());
    for inner in &argument.slopes {
        slopes.push(work.mul(inner, &slope)?);
    }
    LipschitzEnclosure { value, slopes }.checked()
}

/// The caller has admitted the whole original source, ordered coordinates and
/// box. Work and allocations remain cumulative across the earlier C2 attempt.
pub(super) fn evaluate_lipschitz(
    expression: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<LipschitzEnclosure, RootBoxError> {
    evaluate_inner(expression, variables, domain, 0, work)
}

fn evaluate_inner(
    expression: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    depth: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<LipschitzEnclosure, RootBoxError> {
    work.step()?;
    if depth > MAX_ROOT_BOX_DEPTH || work.nodes >= MAX_ROOT_BOX_NODES * 2 {
        return Err(RootBoxError::ResourceLimit(
            "Lipschitz expression depth or node count",
        ));
    }
    work.charge(1, 512 + variables.len() as u64 * 128, 16)?;
    work.nodes += 1;
    let dimensions = variables.len();
    let result = match expression {
        Expr::Integer(value) => {
            if value.limb_count() > MAX_ROOT_BOX_LIMBS {
                return Err(RootBoxError::ResourceLimit("integer height"));
            }
            work.charge(1, 128 + value.limb_count() * 16, 2)?;
            LipschitzEnclosure::constant(
                RealBall::exact(BigRational::from_integer(value.clone())),
                dimensions,
            )
        }
        Expr::Rational(value) => {
            work.arithmetic(&[], &[value])?;
            LipschitzEnclosure::constant(RealBall::exact(value.clone()), dimensions)
        }
        Expr::Const(Constant::Pi) => {
            work.elementary(&RealBall::from_i64(1))?;
            LipschitzEnclosure::constant(RealBall::pi(work.precision)?, dimensions)
        }
        Expr::Const(Constant::E) => {
            LipschitzEnclosure::constant(work.exp(&RealBall::from_i64(1))?, dimensions)
        }
        Expr::Const(_) => return Err(RootBoxError::Domain("non-real or non-finite constant")),
        Expr::Sym(symbol) => {
            let index = variables
                .iter()
                .position(|variable| variable == symbol)
                .ok_or(RootBoxError::Unsupported("unbound symbol identity"))?;
            work.arithmetic(&[&domain[index]], &[])?;
            let mut result = LipschitzEnclosure::constant(domain[index].clone(), dimensions);
            result.slopes[index] = RealBall::from_i64(1);
            result
        }
        Expr::Add(children) | Expr::Mul(children) => {
            if children.len() > MAX_ROOT_BOX_NODES {
                return Err(RootBoxError::ResourceLimit("variadic expression width"));
            }
            let sum = matches!(expression, Expr::Add(_));
            let mut result = LipschitzEnclosure::constant(
                RealBall::from_i64(if sum { 0 } else { 1 }),
                dimensions,
            );
            for child in children {
                let next = evaluate_inner(child, variables, domain, depth + 1, work)?;
                result = if sum {
                    add(result, next, work)?
                } else {
                    multiply(result, next, work)?
                };
            }
            result
        }
        Expr::Pow(base, exponent) => {
            let base_value = evaluate_inner(base, variables, domain, depth + 1, work)?;
            // Retain exponent-domain obligations even when the base is one.
            let exponent_value = evaluate_inner(exponent, variables, domain, depth + 1, work)?;
            let integer = match exponent.as_ref() {
                Expr::Integer(value) => Some(value),
                Expr::Rational(value) if value.denom().is_one() => Some(value.numer()),
                _ => None,
            };
            if let Some(integer) = integer {
                let power = integer
                    .to_i32()
                    .filter(|power| (-64..=64).contains(power))
                    .ok_or(RootBoxError::ResourceLimit("integer exponent"))?;
                let value = work.power(&base_value.value, power)?;
                let slope = match power {
                    0 => RealBall::from_i64(0),
                    1 => RealBall::from_i64(1),
                    _ => {
                        let factor = work.power(&base_value.value, power - 1)?;
                        work.mul(&factor, &RealBall::from_i64(i64::from(power)))?
                    }
                };
                chain(base_value, value, slope, work)?
            } else {
                if !base_value.value.is_positive() {
                    return Err(RootBoxError::Domain(
                        "noninteger power base is not strictly positive",
                    ));
                }
                work.elementary(&base_value.value)?;
                let logarithm = checked(base_value.value.ln(work.precision)?)?;
                let exponent = work.mul(&exponent_value.value, &logarithm)?;
                let value = work.exp(&exponent)?;
                let mut slopes = Vec::with_capacity(dimensions);
                for i in 0..dimensions {
                    let left = work.mul(&exponent_value.slopes[i], &logarithm)?;
                    let quotient = work.div(&base_value.slopes[i], &base_value.value)?;
                    let right = work.mul(&exponent_value.value, &quotient)?;
                    let total = work.add(&left, &right)?;
                    slopes.push(work.mul(&value, &total)?);
                }
                LipschitzEnclosure { value, slopes }
            }
        }
        Expr::Function(name, arguments) => {
            if name == "Piecewise" {
                // Guards deliberately keep the strict C1 neighborhood check.
                // Only the selected value can use the Lipschitz lane.
                let selected = work.select_stable_piecewise(arguments, variables, domain, depth)?;
                return evaluate_inner(selected, variables, domain, depth + 2, work);
            }
            if name == "atan2" {
                let [y, x] = arguments.as_slice() else {
                    return Err(RootBoxError::Unsupported("atan2 requires two arguments"));
                };
                let y = evaluate_inner(y, variables, domain, depth + 1, work)?;
                let x = evaluate_inner(x, variables, domain, depth + 1, work)?;
                let (value, dy, dx) = work.atan2(&y.value, &x.value)?;
                let mut slopes = Vec::with_capacity(dimensions);
                for (y_slope, x_slope) in y.slopes.iter().zip(&x.slopes) {
                    let y_term = work.mul(&dy, y_slope)?;
                    let x_term = work.mul(&dx, x_slope)?;
                    slopes.push(work.add(&y_term, &x_term)?);
                }
                return LipschitzEnclosure { value, slopes }.checked();
            }
            let [argument] = arguments.as_slice() else {
                return Err(RootBoxError::Unsupported("non-unary function"));
            };
            let argument = evaluate_inner(argument, variables, domain, depth + 1, work)?;
            let (value, slope) = if matches!(name.as_str(), "abs" | "Abs") {
                work.arithmetic(&[&argument.value], &[])?;
                let lower = argument.value.lower();
                let upper = argument.value.upper();
                // Non-strict signs are sufficient on this closed box: Abs
                // equals one affine branch there, even on a boundary face.
                // No smooth extension outside the box is claimed.
                let slope = if !lower.is_negative() {
                    RealBall::from_i64(1)
                } else if !upper.is_positive() {
                    RealBall::from_i64(-1)
                } else {
                    RealBall::new(BigRational::zero(), BigRational::one())?
                };
                (checked(argument.value.abs())?, slope)
            } else {
                work.smooth_unary_rule(name, &argument.value)?
            };
            chain(argument, value, slope, work)?
        }
    };
    result.checked()
}
