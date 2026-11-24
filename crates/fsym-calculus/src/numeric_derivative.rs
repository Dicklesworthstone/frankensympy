//! Bounded execution of independently supplied numerical derivative rules.
//!
//! A raw rule asks for simplified derivatives of source children. Missing
//! children suspend that attempt and enter an explicit work list; the rule is
//! retried with completed children. No raw derivative frame recursively owns
//! another raw derivative frame. The cached values retain the recipe's exact
//! normal forms, and all attempts spend the same caller/private ledger.

use crate::compile::{CompileError, charge_compilation_work, compilation_finish};
use crate::sparse_jacobian::charge_expression_copy;
use fsym_budget::BudgetMeter;
use fsym_core::{Expr, Symbol};
use fsym_simplify::SimplifyError;
use std::collections::HashMap;
use std::mem::size_of;

pub(crate) enum DerivativeStepError {
    Operational(CompileError),
    NeedChild(usize),
}

impl From<CompileError> for DerivativeStepError {
    fn from(error: CompileError) -> Self {
        Self::Operational(error)
    }
}

fn invariant(reason: &'static str) -> CompileError {
    CompileError::DerivativeNormalization(SimplifyError::ArithmeticInvariant(reason))
}

fn reserve<T, M: BudgetMeter + ?Sized>(
    values: &mut Vec<T>,
    count: usize,
    meter: &mut M,
) -> Result<(), CompileError> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or(CompileError::DimensionOverflow)?;
    charge_compilation_work(meter, count, bytes, usize::from(count != 0))?;
    values
        .try_reserve_exact(count)
        .map_err(|_| CompileError::AllocationFailure)
}

/// Resource control only: the generator and verifier supply distinct raw
/// rules, constancy checks and zero cleanup. This module never chooses a
/// mathematical derivative or accepts a proof.
pub(crate) fn evaluate<M, R, C>(
    source: &Expr,
    variable: &Symbol,
    normalize_root: bool,
    meter: &mut M,
    mut raw_rule: R,
    mut constant: C,
    cleanup: fn(&Expr) -> Expr,
) -> Result<Expr, CompileError>
where
    M: BudgetMeter + ?Sized,
    R: FnMut(
        &Expr,
        &mut dyn FnMut(&Expr, &Symbol) -> Result<Expr, DerivativeStepError>,
    ) -> Result<Expr, DerivativeStepError>,
    C: FnMut(&Expr) -> bool,
{
    let (count, _) = charge_expression_copy(source, meter)?;
    let mut pending = Vec::new();
    let mut nodes = Vec::new();
    let mut values: Vec<Option<Expr>> = Vec::new();
    let mut work = Vec::new();
    reserve(&mut pending, count, meter)?;
    reserve(&mut nodes, count, meter)?;
    reserve(&mut values, count, meter)?;
    reserve(&mut work, count, meter)?;
    charge_compilation_work(
        meter,
        count,
        count
            .checked_mul(size_of::<(*const Expr, usize)>())
            .ok_or(CompileError::DimensionOverflow)?,
        usize::from(count != 0),
    )?;
    let mut indices = HashMap::new();
    indices
        .try_reserve(count)
        .map_err(|_| CompileError::AllocationFailure)?;
    pending.push(source);
    while let Some(expression) = pending.pop() {
        charge_compilation_work(meter, 1, 0, 0)?;
        // These addresses are ephemeral lookup keys within this immutable
        // borrow. They are never serialized, ordered into a result, or used
        // as symbolic identity. Every callback must name an actual retained
        // source node; an unrelated temporary cannot alias a cached entry.
        let key = expression as *const Expr;
        if indices.contains_key(&key) {
            continue;
        }
        let index = nodes.len();
        indices.insert(key, index);
        nodes.push(expression);
        values.push(None);
        match expression {
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                pending.extend(children);
            }
            Expr::Pow(base, exponent) => {
                pending.push(base.as_ref());
                pending.push(exponent.as_ref());
            }
            _ => {}
        }
    }
    work.push(0);
    while let Some(&index) = work.last() {
        charge_compilation_work(meter, 1, 0, 0)?;
        let expression = nodes[index];
        let normalize = index != 0 || normalize_root;
        // The structural copy envelope also bounds the reference recipe's
        // source-only free-symbol query before it can skip a constant subtree.
        charge_expression_copy(expression, meter)?;
        let result = if normalize && constant(expression) {
            Ok(Expr::from_i64(0))
        } else {
            raw_rule(expression, &mut |child, requested_variable| {
                charge_compilation_work(meter, variable.name.len() + 1, 0, 0)?;
                if requested_variable != variable {
                    return Err(invariant("numerical rule changed its differentiation axis").into());
                }
                let child_index = indices
                    .get(&(child as *const Expr))
                    .copied()
                    .ok_or_else(|| invariant("numerical rule requested a non-source child"))?;
                match &values[child_index] {
                    Some(value) => {
                        charge_expression_copy(value, meter)?;
                        Ok(value.clone())
                    }
                    None => Err(DerivativeStepError::NeedChild(child_index)),
                }
            })
        };
        match result {
            Err(DerivativeStepError::Operational(error)) => return Err(error),
            Err(DerivativeStepError::NeedChild(child)) => {
                charge_compilation_work(meter, work.len() + 1, 0, 0)?;
                if work.contains(&child) {
                    return Err(invariant("numerical rule requested a cyclic derivative"));
                }
                work.push(child);
            }
            Ok(raw) => {
                let derivative = if normalize {
                    charge_expression_copy(&raw, meter)?;
                    let cleaned = cleanup(&raw);
                    fsym_simplify::simplify_numeric_with(&cleaned, meter)?
                } else {
                    raw
                };
                values[index] = Some(derivative);
                work.pop();
            }
        }
    }
    let derivative = values[0]
        .take()
        .ok_or_else(|| invariant("numerical derivative work list finished without a result"))?;
    compilation_finish(Ok(derivative), meter)
}
