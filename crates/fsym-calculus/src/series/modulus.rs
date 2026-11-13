//! Original-source definedness for modulus series and limit shortcuts.
//!
//! Every partial operation has its own local nonzero obligation. A zero
//! multiplier or a later norm-square cancellation cannot erase that premise.

use super::{SeriesError, branch, fresh_symbol, leading_term};
use crate::gruntz::{Direction, substitute_limit_source};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::{Constant, Expr, Symbol};
use num_traits::Signed;
use std::collections::HashMap;
use std::sync::Arc;

const MAX_DENOMINATOR_CHECKS: usize = 16;
const MAX_NUMERIC_LIMBS: u64 = 1_024;

fn unsupported() -> SeriesError {
    SeriesError::Unsupported("modulus source has an unresolved finite-domain obligation".into())
}

/// Preflight the expanded raw tree before copying a replacement at each
/// occurrence. Individually bounded inputs can otherwise have an unbounded
/// product of source occurrences and replacement size.
pub(super) fn checked_substitute_source(
    source: &Expr,
    variable: &Symbol,
    replacement: &Expr,
) -> Result<Expr, SeriesError> {
    let mut pending = vec![(source, 0usize, true)];
    let mut visited = 0usize;
    while let Some((node, depth, substitute)) = pending.pop() {
        if depth > branch::MAX_CONJUGATE_DEPTH || visited == branch::MAX_CONJUGATE_NODES {
            return Err(unsupported());
        }
        if substitute && matches!(node, Expr::Sym(symbol) if symbol == variable) {
            pending.push((replacement, depth, false));
            continue;
        }
        visited += 1;
        match node {
            Expr::Function(name, _)
                if matches!(
                    name.as_str(),
                    "Lambda" | "Sum" | "Product" | "Integral" | "Subs" | "Derivative" | "diff"
                ) =>
            {
                // The shared scalar substitution delegates these to an
                // evaluating binder lane, outside this finite-source slice.
                return Err(unsupported());
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                if visited + pending.len() + args.len() > branch::MAX_CONJUGATE_NODES {
                    return Err(unsupported());
                }
                pending.extend(args.iter().map(|arg| (arg, depth + 1, substitute)));
            }
            Expr::Pow(base, exponent) => {
                if visited + pending.len() + 2 > branch::MAX_CONJUGATE_NODES {
                    return Err(unsupported());
                }
                pending.push((base, depth + 1, substitute));
                pending.push((exponent, depth + 1, substitute));
            }
            _ => {}
        }
    }
    Ok(substitute_limit_source(source, variable, replacement))
}

fn denominator_is_nonzero(
    denominator: &Expr,
    variable: &Symbol,
    checks: &mut usize,
) -> Result<(), SeriesError> {
    if fsym_core::assume::is_nonzero(denominator) {
        return Ok(());
    }
    if let Expr::Function(name, args) = denominator
        && name == "cosh"
        && args.len() == 1
        && fsym_core::assume::is_real(&args[0])
    {
        // cosh(r) >= 1 for every finite real r.
        return Ok(());
    }
    if *checks == MAX_DENOMINATOR_CHECKS {
        return Err(unsupported());
    }
    *checks += 1;
    branch::preflight(denominator).map_err(|_| unsupported())?;
    // A nested modulus or inverse function can re-enter this validator through
    // leading_term. Such obligations need a separate bounded proof rather
    // than recursive retries that reset the denominator-work counter.
    let mut pending = vec![denominator];
    while let Some(node) = pending.pop() {
        match node {
            Expr::Function(name, _)
                if matches!(
                    name.as_str(),
                    "Abs"
                        | "abs"
                        | "asin"
                        | "acos"
                        | "atan"
                        | "acot"
                        | "asinh"
                        | "acosh"
                        | "atanh"
                        | "acoth"
                ) =>
            {
                return Err(unsupported());
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                pending.extend(args);
            }
            Expr::Pow(base, exponent) => {
                pending.push(base);
                pending.push(exponent);
            }
            _ => {}
        }
    }
    let log_t = Expr::Function("log".into(), vec![Expr::Sym(variable.clone())]);
    let (coefficient, _) =
        leading_term(denominator, variable, &log_t).map_err(|_| unsupported())?;
    if crate::gruntz::is_zero_coefficient(&coefficient) == Some(false) {
        Ok(())
    } else {
        Err(unsupported())
    }
}

/// Validate every raw child before collecting any algebraic cancellation.
/// The caller supplies the positive local coordinate facts and one counter
/// shared by all denominator proofs in the requested source batch.
pub(super) fn validate_source(
    source: &Expr,
    variable: &Symbol,
    checks: &mut usize,
) -> Result<(), SeriesError> {
    branch::preflight(source).map_err(|_| unsupported())?;
    if crate::unsafe_direct_substitution(source) {
        return Err(unsupported());
    }
    let mut pending = vec![source];
    // Borrow all premises before constructing denominator expressions.
    // Reversing preorder discharges child premises before their parents.
    let mut denominators: Vec<(&Expr, Option<&str>)> = Vec::new();
    while let Some(node) = pending.pop() {
        match node {
            Expr::Integer(n) => {
                if n.limb_count() > MAX_NUMERIC_LIMBS {
                    return Err(unsupported());
                }
            }
            Expr::Rational(q) => {
                if q.numer().limb_count() > MAX_NUMERIC_LIMBS
                    || q.denom().limb_count() > MAX_NUMERIC_LIMBS
                {
                    return Err(unsupported());
                }
            }
            Expr::Const(Constant::Pi | Constant::E | Constant::I) => {}
            Expr::Const(_) => return Err(unsupported()),
            Expr::Sym(_) if !fsym_core::assume::is_real(node) => return Err(unsupported()),
            Expr::Sym(_) => {}
            Expr::Add(args) | Expr::Mul(args) => pending.extend(args),
            Expr::Pow(base, exponent) => {
                let nonnegative = match exponent.as_ref() {
                    Expr::Integer(n) => !n.is_negative(),
                    Expr::Rational(q) => !q.is_negative(),
                    _ => false,
                };
                if !nonnegative {
                    denominators.push((base, None));
                }
                pending.push(base);
                pending.push(exponent);
            }
            Expr::Function(name, args) => {
                let [arg] = args.as_slice() else {
                    return Err(unsupported());
                };
                let denominator = match name.as_str() {
                    "tan" | "sec" => Some("cos"),
                    "cot" | "csc" => Some("sin"),
                    "tanh" | "sech" => Some("cosh"),
                    "coth" | "csch" => Some("sinh"),
                    "atan" | "acot" if fsym_core::assume::is_real(arg) => None,
                    "atan" | "acot" => Some("inverse_circular"),
                    "atanh" | "acoth" => Some("inverse_hyperbolic"),
                    _ => None,
                };
                if let Some(denominator) = denominator {
                    denominators.push((arg, Some(denominator)));
                } else {
                    match name.as_str() {
                        "log" | "ln" => denominators.push((arg, None)),
                        "exp" | "sin" | "cos" | "sinh" | "cosh" | "erf" | "erfc" | "sinc"
                        | "Abs" | "abs" | "sqrt" | "re" | "im" | "conjugate" | "asin" | "acos"
                        | "asinh" | "acosh" => {}
                        "atan" | "acot" if fsym_core::assume::is_real(arg) => {}
                        _ if fsym_core::assume::is_real(node) => {}
                        _ => return Err(unsupported()),
                    }
                }
                pending.push(arg);
            }
        }
    }
    for (argument, function) in denominators.into_iter().rev() {
        if let Some(name) = function {
            let denominator = match name {
                "inverse_circular" | "inverse_hyperbolic" => {
                    let square = Expr::Pow(Arc::new(argument.clone()), Arc::new(Expr::from_i64(2)));
                    let signed = if name == "inverse_hyperbolic" {
                        Expr::Mul(vec![Expr::from_i64(-1), square])
                    } else {
                        square
                    };
                    Expr::Add(vec![Expr::from_i64(1), signed])
                }
                _ => Expr::Function(name.into(), vec![argument.clone()]),
            };
            denominator_is_nonzero(&denominator, variable, checks)?;
        } else {
            denominator_is_nonzero(argument, variable, checks)?;
        }
    }
    Ok(())
}

/// Inspect original modulus arguments before public coordinate substitution
/// or direct-limit evaluation can erase held children. Other expression
/// families retain their own existing admission paths.
pub(crate) fn validate_inputs(
    source: &Expr,
    variable: &Symbol,
    point: &Expr,
    direction: Direction,
) -> Result<(), SeriesError> {
    validate_function_inputs(source, variable, point, direction, &["Abs", "abs"])
}

/// Retain complete function arguments and the original point before public
/// substitution can cancel held children or expand a small tree excessively.
pub(super) fn validate_function_inputs(
    source: &Expr,
    variable: &Symbol,
    point: &Expr,
    direction: Direction,
    names: &[&str],
) -> Result<(), SeriesError> {
    if crate::unsafe_direct_substitution(source) || crate::unsafe_direct_substitution(point) {
        return Err(unsupported());
    }
    let mut pending = vec![source];
    let mut arguments = Vec::new();
    while let Some(node) = pending.pop() {
        match node {
            Expr::Function(name, args) if names.contains(&name.as_str()) => {
                let [_arg] = args.as_slice() else {
                    return Err(unsupported());
                };
                if arguments.len() == MAX_DENOMINATOR_CHECKS {
                    return Err(unsupported());
                }
                arguments.push(node);
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => pending.extend(args),
            Expr::Pow(base, exponent) => {
                pending.push(base);
                pending.push(exponent);
            }
            _ => {}
        }
    }
    if arguments.is_empty() {
        return Ok(());
    }
    let coordinate = fresh_symbol("_fsym_modulus_t", &[source, point], &[variable])?;
    let local = Expr::Sym(coordinate.clone());
    let inverse = || Expr::Pow(Arc::new(local.clone()), Arc::new(Expr::from_i64(-1)));
    let negative = |value: Expr| Expr::Mul(vec![Expr::from_i64(-1), value]);
    let replacements = match point {
        Expr::Const(Constant::Infinity) => vec![inverse()],
        Expr::Const(Constant::NegativeInfinity) => vec![negative(inverse())],
        _ => {
            let plus = || Expr::Add(vec![point.clone(), local.clone()]);
            let minus = || Expr::Add(vec![point.clone(), negative(local.clone())]);
            match direction {
                Direction::Plus => vec![plus()],
                Direction::Minus => vec![minus()],
                Direction::Both => vec![plus(), minus()],
            }
        }
    };
    let mut checks = 0;
    with_symbol_facts(
        HashMap::from([(coordinate.clone(), Facts::from_names(&["positive"]))]),
        || {
            for replacement in replacements {
                for argument in &arguments {
                    branch::preflight(argument).map_err(|_| unsupported())?;
                    let source = checked_substitute_source(argument, variable, &replacement)?;
                    validate_source(&source, &coordinate, &mut checks)?;
                }
            }
            Ok(())
        },
    )
}
