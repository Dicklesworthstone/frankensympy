//! Source-only branch stability for the independent C1/C2 interval checkers.
//!
//! A selected branch must be unchanged on an open neighborhood of the whole
//! closed box. Strict interval separation and identical defined operands can
//! establish this; truth only at a point or at an included boundary cannot.
//! No symbolic derivative, compiled predicate, or sampled value is consulted.

use super::{BudgetMeter, Expr, MAX_ROOT_BOX_DEPTH, RealBall, RootBoxError, Symbol, Work};

fn branch_parts(branch: &Expr) -> Result<(&Expr, &Expr), RootBoxError> {
    if let Expr::Function(name, arguments) = branch
        && name == "Tuple"
        && let [value, condition] = arguments.as_slice()
    {
        return Ok((value, condition));
    }
    Err(RootBoxError::Unsupported(
        "Piecewise requires Tuple(value, condition) branches",
    ))
}

impl<M: BudgetMeter> Work<'_, M> {
    pub(super) fn select_stable_piecewise<'source>(
        &mut self,
        branches: &'source [Expr],
        variables: &[Symbol],
        domain: &[RealBall],
        depth: usize,
    ) -> Result<&'source Expr, RootBoxError> {
        if branches.is_empty() {
            return Err(RootBoxError::Unsupported(
                "Piecewise requires a nonempty branch list",
            ));
        }
        // Whole-source preflight already bounds every child, including dead
        // branches. Check pair shape before selecting an early true branch.
        for branch in branches {
            self.step()?;
            branch_parts(branch)?;
        }
        for branch in branches {
            self.step()?;
            let (value, condition) = branch_parts(branch)?;
            if self.stable_condition(condition, variables, domain, depth + 2)? {
                return Ok(value);
            }
        }
        Err(RootBoxError::Domain(
            "Piecewise has no matching branch on this box",
        ))
    }

    fn stable_condition(
        &mut self,
        condition: &Expr,
        variables: &[Symbol],
        domain: &[RealBall],
        depth: usize,
    ) -> Result<bool, RootBoxError> {
        self.step()?;
        if depth > MAX_ROOT_BOX_DEPTH {
            return Err(RootBoxError::ResourceLimit("Piecewise condition depth"));
        }
        if let Expr::Sym(symbol) = condition
            && symbol.is_plain()
        {
            match symbol.name.as_str() {
                "True" => return Ok(true),
                "False" => return Ok(false),
                _ => {}
            }
        }
        if let Expr::Function(name, arguments) = condition {
            match (name.as_str(), arguments.as_slice()) {
                ("Not", [argument]) => {
                    return Ok(!self.stable_condition(argument, variables, domain, depth + 1)?);
                }
                ("And" | "Or", _) => {
                    let conjunction = name == "And";
                    for argument in arguments {
                        let value =
                            self.stable_condition(argument, variables, domain, depth + 1)?;
                        if value != conjunction {
                            return Ok(value);
                        }
                    }
                    return Ok(conjunction);
                }
                (
                    "Equality" | "Unequality" | "StrictLessThan" | "LessThan" | "StrictGreaterThan"
                    | "GreaterThan",
                    [left, right],
                ) => {
                    // Continuity on a neighborhood is part of each operand's
                    // C1 source check, even when their syntax is identical.
                    // Thus Eq(log(0), log(0)) cannot conceal an undefined term.
                    let left_value = self.eval(left, variables, domain, depth + 1)?.value;
                    let right_value = self.eval(right, variables, domain, depth + 1)?.value;
                    if self.same_guard_source(left, right, depth + 1)? {
                        return Ok(matches!(
                            name.as_str(),
                            "Equality" | "LessThan" | "GreaterThan"
                        ));
                    }
                    let difference = self.sub(&left_value, &right_value)?;
                    self.arithmetic(&[&difference], &[])?;
                    let negative = difference.is_negative();
                    let positive = difference.is_positive();
                    if negative || positive {
                        return Ok(match name.as_str() {
                            "Equality" => false,
                            "Unequality" => true,
                            "StrictLessThan" | "LessThan" => negative,
                            _ => positive,
                        });
                    }
                    return Err(RootBoxError::Domain(
                        "Piecewise condition is not stable on an open neighborhood of the box",
                    ));
                }
                _ => {}
            }
        }
        Err(RootBoxError::Unsupported(
            "Piecewise condition requires a boolean, comparison, And, Or or Not",
        ))
    }

    /// Structural identity, with each borrowed scalar/text comparison metered.
    /// Equal interval values alone are not an identity proof: x == 0 on [0,0]
    /// changes truth immediately off the box and must remain inconclusive.
    fn same_guard_source(
        &mut self,
        left: &Expr,
        right: &Expr,
        depth: usize,
    ) -> Result<bool, RootBoxError> {
        self.step()?;
        if depth > MAX_ROOT_BOX_DEPTH {
            return Err(RootBoxError::ResourceLimit("Piecewise identity depth"));
        }
        let children = match (left, right) {
            (Expr::Integer(a), Expr::Integer(b)) => {
                self.charge(1 + a.limb_count() + b.limb_count(), 0, 0)?;
                return Ok(a == b);
            }
            (Expr::Rational(a), Expr::Rational(b)) => {
                self.arithmetic(&[], &[a, b])?;
                return Ok(a == b);
            }
            (Expr::Const(a), Expr::Const(b)) => return Ok(a == b),
            (Expr::Sym(a), Expr::Sym(b)) => {
                self.charge(1 + (a.name.len() + b.name.len()) as u64, 0, 0)?;
                return Ok(a == b);
            }
            (Expr::Pow(a, b), Expr::Pow(c, d)) => {
                return Ok(self.same_guard_source(a, c, depth + 1)?
                    && self.same_guard_source(b, d, depth + 1)?);
            }
            (Expr::Add(a), Expr::Add(b)) | (Expr::Mul(a), Expr::Mul(b)) => (a, b),
            (Expr::Function(a, aa), Expr::Function(b, ba)) => {
                self.charge(1 + (a.len() + b.len()) as u64, 0, 0)?;
                if a != b {
                    return Ok(false);
                }
                (aa, ba)
            }
            _ => return Ok(false),
        };
        if children.0.len() != children.1.len() {
            return Ok(false);
        }
        for (a, b) in children.0.iter().zip(children.1) {
            if !self.same_guard_source(a, b, depth + 1)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
