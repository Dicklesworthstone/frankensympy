//! One-sided analytic continuations of principal inverse functions.
//!
//! Taylor coefficients evaluated on a cut belong to its defined principal
//! boundary value. The complete source path can approach the other boundary:
//! reflection changes every derivative, whereas a logarithmic jump changes
//! only the constant. These identities follow the principal logarithmic forms
//! in DLMF 4.23(iv) and 4.37(iv), with SymPy's acot/acoth boundary conventions.

use super::{SeriesError, branch, leading_term};
use crate::gruntz::Direction;
use fsym_budget::{Budget, BudgetLimits};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::elementary::eval_function;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use num_traits::Signed;
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum Region {
    Regular,
    Outer(i32),
    Inner(i32),
    AcoshInner,
    AcoshLeft,
    Endpoint(i32),
}

fn undecided() -> SeriesError {
    SeriesError::Unsupported("cannot establish the principal inverse-function branch".into())
}

fn sign(value: &Expr) -> Result<i32, SeriesError> {
    crate::gruntz::constant_sign(value).ok_or_else(undecided)
}

fn imaginary_axis(name: &str) -> bool {
    matches!(name, "atan" | "acot" | "asinh")
}

fn is_inverse(name: &str) -> bool {
    matches!(
        name,
        "asin" | "acos" | "atan" | "acot" | "asinh" | "acosh" | "atanh" | "acoth"
    )
}

pub(crate) fn validate_inputs(
    source: &Expr,
    variable: &Symbol,
    point: &Expr,
    direction: Direction,
) -> Result<(), SeriesError> {
    super::modulus::validate_function_inputs(
        source,
        variable,
        point,
        direction,
        &[
            "asin", "acos", "atan", "acot", "asinh", "acosh", "atanh", "acoth",
        ],
    )
}

/// Exact real and imaginary projections, with borrowed-source preflight before
/// constructing even a small wrapper. Unknown complex parameters stay unknown.
fn parts(value: &Expr) -> Result<(Expr, Expr), SeriesError> {
    branch::preflight(value)?;
    let imaginary = branch::imaginary_part(value)?;
    let real = branch::imaginary_part(&(Expr::Const(Constant::I) * value.clone()))?;
    Ok((real, imaginary))
}

fn region(name: &str, value: &Expr) -> Result<Option<Region>, SeriesError> {
    if !is_inverse(name) {
        return Ok(None);
    }
    let (real, imaginary) = parts(value)?;
    let (axial, perpendicular) = if imaginary_axis(name) {
        (imaginary, real)
    } else {
        (real, imaginary)
    };
    match sign(&perpendicular)? {
        -1 | 1 => return Ok(Some(Region::Regular)),
        0 => {}
        _ => return Err(undecided()),
    }
    let left = sign(&(axial.clone() + Expr::from_i64(1)))?;
    let right = sign(&(axial.clone() - Expr::from_i64(1)))?;
    if left == 0 {
        return Ok(Some(Region::Endpoint(-1)));
    }
    if right == 0 {
        return Ok(Some(Region::Endpoint(1)));
    }
    let result = if name == "acosh" {
        if right == 1 {
            Region::Regular
        } else if left == -1 {
            Region::AcoshLeft
        } else if left == 1 && right == -1 {
            Region::AcoshInner
        } else {
            return Err(undecided());
        }
    } else if matches!(name, "acot" | "acoth") {
        if left == -1 || right == 1 {
            Region::Regular
        } else if left == 1 && right == -1 {
            Region::Inner(sign(&axial)?)
        } else {
            return Err(undecided());
        }
    } else if left == -1 {
        Region::Outer(-1)
    } else if right == 1 {
        Region::Outer(1)
    } else if left == 1 && right == -1 {
        Region::Regular
    } else {
        return Err(undecided());
    };
    Ok(Some(result))
}

/// Sufficient continuity admission for finite direct substitution. A principal
/// value on a cut does not establish continuity from an arbitrary source path.
pub(crate) fn continuous_at(name: &str, value: &Expr) -> bool {
    match region(name, value) {
        Ok(None | Some(Region::Regular)) => true,
        // These finite square-root branch points are continuous, although
        // they require Puiseux rather than ordinary Taylor expansions.
        Ok(Some(Region::Endpoint(endpoint))) => {
            matches!(name, "asin" | "acos" | "asinh") || (name == "acosh" && endpoint == 1)
        }
        _ => false,
    }
}

fn principal_value(name: &str, center: &Expr) -> Expr {
    if name == "acoth" && center.is_zero() {
        return Expr::Rational(BigRational::new(1.into(), 2.into()))
            * Expr::Const(Constant::I)
            * Expr::Const(Constant::Pi);
    }
    eval_function(name, std::slice::from_ref(center))
        .unwrap_or_else(|| Expr::Function(name.into(), vec![center.clone()]))
}

/// A finite inverse-function limit needs the finite argument's boundary value,
/// not a proof that the resulting complex constant is nonzero. The generic
/// leading-term engine deliberately refuses constants outside its enclosure
/// grammar; this exact path does not relax that proof obligation.
pub(crate) fn finite_limit(
    expression: &Expr,
    variable: &Symbol,
    point: &Expr,
    direction: Direction,
) -> Option<Result<Expr, SeriesError>> {
    let Expr::Function(name, arguments) = expression else {
        return None;
    };
    if arguments.len() != 1
        || !is_inverse(name)
        || matches!(
            point,
            Expr::Const(Constant::Infinity | Constant::NegativeInfinity)
        )
    {
        return None;
    }
    let attempt = || -> Result<Option<Expr>, SeriesError> {
        branch::preflight(expression)?;
        branch::preflight(point)?;
        let local = super::fresh_symbol("_fsym_inverse_t", &[expression, point], &[variable])?;
        let logarithm = super::fresh_symbol(
            "_fsym_inverse_log",
            &[expression, point],
            &[variable, &local],
        )?;
        let mut checks = 0;
        let mut one_side = |from_left: bool| -> Result<Option<Expr>, SeriesError> {
            let coordinate = Expr::Sym(local.clone());
            let replacement = if from_left {
                Expr::Add(vec![
                    point.clone(),
                    Expr::Mul(vec![Expr::from_i64(-1), coordinate]),
                ])
            } else {
                Expr::Add(vec![point.clone(), coordinate])
            };
            let source =
                super::modulus::checked_substitute_source(&arguments[0], variable, &replacement)?;
            with_symbol_facts(
                HashMap::from([(local.clone(), Facts::from_names(&["positive"]))]),
                || {
                    super::modulus::validate_source(&source, &local, &mut checks)?;
                    let series = match super::series_in(
                        &source,
                        &local,
                        &super::q(1),
                        &Expr::Sym(logarithm.clone()),
                    ) {
                        Ok(series) => series,
                        Err(SeriesError::Essential(_)) => return Ok(None),
                        Err(error) => return Err(error),
                    };
                    if series
                        .order
                        .as_ref()
                        .is_some_and(|order| !order.is_positive())
                        || series.terms.iter().any(|(power, coefficient)| {
                            power.is_negative()
                                || !super::is_free_of(coefficient, &local)
                                || !super::is_free_of(coefficient, &logarithm)
                        })
                    {
                        return Ok(None);
                    }
                    for coefficient in series.terms.values() {
                        super::modulus::validate_source(coefficient, &local, &mut checks)?;
                    }
                    let center = series
                        .terms
                        .get(&super::q(0))
                        .cloned()
                        .unwrap_or_else(|| Expr::from_i64(0));
                    let correction = if continuous_at(name, &center) {
                        (1, Expr::from_i64(0))
                    } else {
                        continuation(name, &source, &center, &local)?
                    };
                    let value = Expr::from_i64(i64::from(correction.0))
                        * principal_value(name, &center)
                        + correction.1;
                    branch::preflight(&value)?;
                    let mut meter = Budget::new(BudgetLimits::uniform(65_536, 0));
                    fsym_simplify::simplify_with(&value, &mut meter)
                        .map(Some)
                        .map_err(|_| undecided())
                },
            )
        };
        match direction {
            Direction::Plus => one_side(false),
            Direction::Minus => one_side(true),
            Direction::Both => {
                let right = one_side(false)?;
                let left = one_side(true)?;
                match (right, left) {
                    (Some(right), Some(left)) if right == left => Ok(Some(right)),
                    (Some(_), Some(_)) => Err(SeriesError::Unsupported(
                        "inverse-function one-sided limits do not agree".into(),
                    )),
                    _ => Ok(None),
                }
            }
        }
    };
    match attempt() {
        Ok(Some(value)) => Some(Ok(value)),
        Ok(None) => None,
        Err(error) => Some(Err(error)),
    }
}

fn eventual_sign(value: &Expr, variable: &Symbol) -> Result<i32, SeriesError> {
    branch::preflight(value)?;
    if let Some(result) = crate::gruntz::constant_sign(value) {
        return Ok(result);
    }
    let logarithm = Expr::Function("log".into(), vec![Expr::Sym(variable.clone())]);
    let (coefficient, _) = leading_term(value, variable, &logarithm)?;
    sign(&coefficient)
}

/// Affine correction `actual = multiplier * Taylor + offset` to the
/// principal-value Taylor germ. The local coordinate is positive and typed;
/// no facts are attached to other same-named source symbols.
pub(super) fn continuation(
    name: &str,
    source: &Expr,
    center: &Expr,
    variable: &Symbol,
) -> Result<(i32, Expr), SeriesError> {
    if !is_inverse(name) {
        return Ok((1, Expr::from_i64(0)));
    }
    branch::preflight(source)?;
    with_symbol_facts(
        HashMap::from([(variable.clone(), Facts::from_names(&["positive"]))]),
        || {
            let mut checks = 0;
            super::modulus::validate_source(source, variable, &mut checks)?;
            let region = region(name, center)?.ok_or_else(undecided)?;
            if matches!(region, Region::Regular) {
                return Ok((1, Expr::from_i64(0)));
            }
            if matches!(region, Region::Endpoint(_)) {
                return Err(SeriesError::Unsupported(
                    "inverse-function branch point needs a Puiseux expansion".into(),
                ));
            }
            let (real, imaginary) = parts(source)?;
            let (axial, perpendicular) = if imaginary_axis(name) {
                (imaginary, real)
            } else {
                (real, imaginary)
            };
            let side = eventual_sign(&perpendicular, variable)?;
            let pi = Expr::Const(Constant::Pi);
            let imaginary_pi = Expr::Const(Constant::I) * pi.clone();
            let unchanged = (1, Expr::from_i64(0));
            match (name, region) {
                ("asin", Region::Outer(axis)) if side * axis > 0 => {
                    Ok((-1, Expr::from_i64(i64::from(axis)) * pi))
                }
                ("acos", Region::Outer(axis)) if side * axis > 0 => {
                    Ok((-1, Expr::from_i64(i64::from(1 - axis)) * pi))
                }
                ("atan", Region::Outer(axis)) if side * axis < 0 => {
                    Ok((1, Expr::from_i64(i64::from(-axis)) * pi))
                }
                ("asinh", Region::Outer(axis)) if side * axis < 0 => {
                    Ok((-1, Expr::from_i64(i64::from(axis)) * imaginary_pi))
                }
                ("atanh", Region::Outer(axis)) if side * axis > 0 => {
                    Ok((1, Expr::from_i64(i64::from(axis)) * imaginary_pi))
                }
                ("acosh", Region::AcoshLeft) if side < 0 => {
                    Ok((1, Expr::from_i64(-2) * imaginary_pi))
                }
                ("acosh", Region::AcoshInner) if side < 0 => Ok((-1, Expr::from_i64(0))),
                ("acot", Region::Inner(0)) => {
                    let opposite = side < 0 || (side == 0 && eventual_sign(&axial, variable)? > 0);
                    Ok((1, if opposite { -pi } else { Expr::from_i64(0) }))
                }
                ("acoth", Region::Inner(0)) => {
                    let opposite = side > 0 || (side == 0 && eventual_sign(&axial, variable)? > 0);
                    Ok((
                        1,
                        if opposite {
                            -imaginary_pi
                        } else {
                            Expr::from_i64(0)
                        },
                    ))
                }
                ("acot", Region::Inner(axis)) if side * axis > 0 => {
                    Ok((1, Expr::from_i64(i64::from(axis)) * pi))
                }
                ("acoth", Region::Inner(axis)) if side * axis < 0 => {
                    Ok((1, Expr::from_i64(i64::from(axis)) * imaginary_pi))
                }
                _ => Ok(unchanged),
            }
        },
    )
}
