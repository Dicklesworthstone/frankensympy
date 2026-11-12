//! One-sided principal-logarithm continuation across the negative real cut.
//!
//! A truncated series cannot decide which side of a cut the source occupies:
//! its first imaginary term may lie beyond the requested expansion order.
//! Recover the imaginary part of the complete source by exact conjugation,
//! then establish its eventual sign on the positive local coordinate.

use super::{SeriesError, leading_term};
use fsym_budget::{Budget, BudgetLimits};
use fsym_core::assume::{Facts, with_symbol_facts};
use fsym_core::{BigRational, Constant, Expr, Symbol};
use std::collections::HashMap;
use std::sync::Arc;

const MAX_CONJUGATE_NODES: usize = 2_048;
const MAX_CONJUGATE_DEPTH: usize = 32;

fn undecided() -> SeriesError {
    SeriesError::Unsupported("cannot establish the side of the principal branch cut".into())
}

/// Check the entire borrowed tree before recursive assumptions, expansion or
/// cloning. A real-valued subtree still consumes the same node/depth limits.
fn preflight(e: &Expr) -> Result<(), SeriesError> {
    let mut pending = vec![(e, 0usize)];
    let mut visited = 0usize;
    while let Some((node, depth)) = pending.pop() {
        if depth > MAX_CONJUGATE_DEPTH || visited == MAX_CONJUGATE_NODES {
            return Err(undecided());
        }
        visited += 1;
        match node {
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if children.len()
                    > MAX_CONJUGATE_NODES.saturating_sub(visited + pending.len())
                {
                    return Err(undecided());
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if visited + pending.len() + 2 > MAX_CONJUGATE_NODES {
                    return Err(undecided());
                }
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
        }
    }
    Ok(())
}

/// Conjugation identities over the declared real symbols and supported
/// arithmetic. Unknown complex symbols/functions remain an obligation.
fn conjugate(e: &Expr, remaining: &mut usize, depth: usize) -> Result<Expr, SeriesError> {
    if depth > MAX_CONJUGATE_DEPTH || *remaining == 0 {
        return Err(undecided());
    }
    *remaining -= 1;
    if fsym_core::assume::is_real(e) {
        return Ok(e.clone());
    }
    match e {
        Expr::Const(Constant::I) => Ok(-e.clone()),
        Expr::Add(children) | Expr::Mul(children) => {
            if children.len() > *remaining {
                return Err(undecided());
            }
            let children = children
                .iter()
                .map(|child| conjugate(child, remaining, depth + 1))
                .collect::<Result<_, _>>()?;
            Ok(if matches!(e, Expr::Add(_)) {
                Expr::Add(children)
            } else {
                Expr::Mul(children)
            })
        }
        Expr::Pow(base, exponent) if matches!(exponent.as_ref(), Expr::Integer(_)) => {
            Ok(Expr::Pow(
                Arc::new(conjugate(base, remaining, depth + 1)?),
                exponent.clone(),
            ))
        }
        Expr::Function(name, args)
            if args.len() == 1
                && matches!(name.as_str(), "log" | "ln")
                && fsym_core::assume::sign(&args[0]) == Some(1) =>
        {
            Ok(e.clone())
        }
        Expr::Function(name, args)
            if args.len() == 1
                && matches!(
                    name.as_str(),
                    "exp" | "sin" | "cos" | "sinh" | "cosh" | "erf" | "erfc"
                ) =>
        {
            Ok(Expr::Function(
                name.clone(),
                vec![conjugate(&args[0], remaining, depth + 1)?],
            ))
        }
        _ => Err(undecided()),
    }
}

fn imaginary_part(e: &Expr) -> Result<Expr, SeriesError> {
    preflight(e)?;
    let mut remaining = MAX_CONJUGATE_NODES;
    let conjugated = conjugate(e, &mut remaining, 0)?;
    let difference = Expr::Add(vec![e.clone(), -conjugated]);
    let imaginary = Expr::Mul(vec![
        Expr::Rational(BigRational::new((-1).into(), 2.into())),
        Expr::Const(Constant::I),
        difference,
    ]);
    let mut meter = Budget::new(BudgetLimits::uniform(65_536, 0));
    let expanded =
        fsym_simplify::expand_with(&imaginary, &mut meter).map_err(|_| undecided())?;
    fsym_simplify::simplify_with(&expanded, &mut meter).map_err(|_| undecided())
}

/// A finite value is in the open domain of the principal logarithm when it
/// is positive real or has an independently separated nonzero imaginary part.
/// Unknown, zero and negative-real values do not establish continuity.
pub(crate) fn avoids_principal_cut(value: &Expr) -> bool {
    if preflight(value).is_err() {
        return false;
    }
    if crate::gruntz::constant_sign(value) == Some(1) {
        return true;
    }
    if let Ok(imaginary) = imaginary_part(value)
        && matches!(crate::gruntz::constant_sign(&imaginary), Some(-1 | 1))
    {
        return true;
    }
    if value.free_symbols().is_empty() {
        for precision in [24, 48, 96] {
            if let Ok(ball) = value.eval_complex_ball(precision)
                && (ball.imag().is_positive() || ball.imag().is_negative())
            {
                return true;
            }
        }
    }
    false
}

/// Correction to Log(leading) when expanding source = leading*t^v*(1+u).
/// The principal argument at a negative real value is +pi, while a path
/// approaching from below has argument tending to -pi.
pub(super) fn logarithm_correction(
    source: &Expr,
    leading: &Expr,
    variable: &Symbol,
) -> Result<Expr, SeriesError> {
    preflight(leading)?;
    if avoids_principal_cut(leading) {
        return Ok(Expr::from_i64(0));
    }
    with_symbol_facts(
        HashMap::from([(variable.clone(), Facts::from_names(&["positive"]))]),
        || {
            let imaginary = imaginary_part(source)?;
            if imaginary.is_zero() {
                // A real path stays on the cut and retains its defined +pi
                // convention; it does not approach through the lower plane.
                return Ok(Expr::from_i64(0));
            }
            if crate::gruntz::constant_sign(leading) != Some(-1) {
                return Err(undecided());
            }
            let sign = match crate::gruntz::constant_sign(&imaginary) {
                Some(sign) => sign,
                None => {
                    let log_t = Expr::Function("log".into(), vec![Expr::Sym(variable.clone())]);
                    let (coefficient, _) = leading_term(&imaginary, variable, &log_t)?;
                    crate::gruntz::constant_sign(&coefficient).ok_or_else(undecided)?
                }
            };
            match sign {
                -1 => Ok(Expr::from_i64(-2)
                    * Expr::Const(Constant::I)
                    * Expr::Const(Constant::Pi)),
                0 | 1 => Ok(Expr::from_i64(0)),
                _ => Err(undecided()),
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_real_subtrees_refuse_before_recursive_assumptions_or_cloning() {
        let mut deep = Expr::from_i64(1);
        for _ in 0..=MAX_CONJUGATE_DEPTH {
            deep = Expr::Function("exp".into(), vec![deep]);
        }
        assert!(imaginary_part(&deep).is_err());
        assert!(!avoids_principal_cut(&deep));

        let wide = Expr::Add(vec![Expr::from_i64(1); MAX_CONJUGATE_NODES]);
        assert!(imaginary_part(&wide).is_err());
        assert!(!avoids_principal_cut(&wide));

        let x = Symbol::new("x");
        let source = Expr::Add(vec![
            Expr::from_i64(-1),
            Expr::Mul(vec![Expr::Const(Constant::I), deep]),
            Expr::Sym(x.clone()),
        ]);
        assert!(logarithm_correction(&source, &Expr::from_i64(-1), &x).is_err());
    }
}
