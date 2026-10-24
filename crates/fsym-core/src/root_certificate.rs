//! Independent real 2x2 root-box verification by interval automatic
//! differentiation and the Krawczyk fixed-point test.
//!
//! The checker uses the original typed expressions, never a symbolic
//! differentiator, compiled program, candidate solver, or sampled derivative.
//! It checks C1 domain obligations on the entire closed box, strict interior
//! inclusion of the Krawczyk image, and an exact rational contraction bound.
//! These establish one unique nonsingular real root in that box.
//!
//! Reference: Burr, Lee and Leykin (2019), Theorem 3, Remark 4 and equation (5):
//! https://arxiv.org/html/1901.10384v2#S2.SS2
//!
//! This is a bounded native checker, not a compatibility/profile promotion.

use crate::{BallError, BigRational, Constant, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAX_ROOT_BOX_NODES: usize = 16_384;
pub const MAX_ROOT_BOX_DEPTH: usize = 32;
pub const MAX_ROOT_BOX_LIMBS: u64 = 1_024;
pub const MAX_ROOT_BOX_PRECISION: u32 = 128;

/// Untrusted proposed box, center and rational preconditioner. Deserialization
/// does not confer verified status; the independent checker must be called.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootBoxWitness2 {
    pub domain: [RealBall; 2],
    pub center: [BigRational; 2],
    pub preconditioner: [[BigRational; 2]; 2],
    pub precision_digits: u32,
}

/// Immutable acceptance bound to the exact equations and variable identities.
/// The enclosure contains the unique nonsingular real root in witness.domain.
/// There is no unchecked decoder or public mutable field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootBoxReceipt2 {
    residuals: [Expr; 2],
    variables: [Symbol; 2],
    witness: RootBoxWitness2,
    root_enclosure: [RealBall; 2],
    contraction_bound: BigRational,
}

impl RootBoxReceipt2 {
    pub fn residuals(&self) -> &[Expr; 2] {
        &self.residuals
    }
    pub fn variables(&self) -> &[Symbol; 2] {
        &self.variables
    }
    pub fn witness(&self) -> &RootBoxWitness2 {
        &self.witness
    }
    pub fn root_enclosure(&self) -> &[RealBall; 2] {
        &self.root_enclosure
    }
    pub fn contraction_bound(&self) -> &BigRational {
        &self.contraction_bound
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RootBoxError {
    #[error("root-box variables must be distinct")]
    DuplicateVariable,
    #[error("root-box coordinate {coordinate} must have strictly positive radius")]
    DegenerateBox { coordinate: usize },
    #[error("center coordinate {coordinate} is outside the proposed box")]
    CenterOutsideBox { coordinate: usize },
    #[error("root-box precision must be between 8 and 128 decimal digits")]
    InvalidPrecision,
    #[error("rational preconditioner is singular")]
    SingularPreconditioner,
    #[error("expression lies outside the root-box checker subset: {0}")]
    Unsupported(&'static str),
    #[error("the original expression is not certified C1 throughout the box: {0}")]
    Domain(&'static str),
    #[error("Krawczyk image is not strictly inside coordinate {coordinate}")]
    NoInteriorInclusion { coordinate: usize },
    #[error("the exact Jacobian contraction bound is not strictly below one")]
    NotContractive,
    #[error("root-box resource limit exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("root-box arithmetic refused: {0}")]
    Ball(#[from] BallError),
    #[error("root-box work stopped: {0}")]
    Meter(#[from] MeterError),
}

/// Convenience checker with explicit fixed resource caps and no caller meter.
pub fn certify_root_box_2x2(
    residuals: &[Expr; 2],
    variables: &[Symbol; 2],
    witness: &RootBoxWitness2,
) -> Result<RootBoxReceipt2, RootBoxError> {
    verify_root_box_2x2(residuals, variables, witness, &mut Unbounded)
}

/// Independently recompute all domain, enclosure and contraction obligations.
/// Cancellation is checked at every expression node and matrix work unit;
/// existing ball kernels also enforce their own fixed numeric envelopes.
pub fn verify_root_box_2x2(
    residuals: &[Expr; 2],
    variables: &[Symbol; 2],
    witness: &RootBoxWitness2,
    meter: &mut impl BudgetMeter,
) -> Result<RootBoxReceipt2, RootBoxError> {
    meter.checkpoint()?;
    // Charge a conservative bounded scratch envelope before recursive or
    // numeric work. Per-node allocations are charged separately below.
    meter.charge_batch(&[
        (Dimension::MemoryBytes, 64 * 1024 * 1024),
        (Dimension::AllocationCount, 1),
    ])?;
    if variables.iter().any(|variable| variable.name.len() > 4096) {
        return Err(RootBoxError::ResourceLimit("symbol length"));
    }
    if variables[0] == variables[1] {
        return Err(RootBoxError::DuplicateVariable);
    }
    if !(8..=MAX_ROOT_BOX_PRECISION).contains(&witness.precision_digits) {
        return Err(RootBoxError::InvalidPrecision);
    }
    for i in 0..2 {
        check_ball(&witness.domain[i])?;
        check_rational(&witness.center[i])?;
        if witness.domain[i].radius().is_zero() {
            return Err(RootBoxError::DegenerateBox { coordinate: i });
        }
        if !witness.domain[i].contains(&witness.center[i]) {
            return Err(RootBoxError::CenterOutsideBox { coordinate: i });
        }
        for coefficient in &witness.preconditioner[i] {
            check_rational(coefficient)?;
        }
    }
    preflight_expressions(residuals, meter)?;
    let c = &witness.preconditioner;
    if &c[0][0] * &c[1][1] - &c[0][1] * &c[1][0] == BigRational::zero() {
        return Err(RootBoxError::SingularPreconditioner);
    }
    let mut work = Work {
        meter,
        nodes: 0,
        precision: witness.precision_digits,
    };
    let box_jets = [
        work.eval(&residuals[0], variables, &witness.domain, 0)?,
        work.eval(&residuals[1], variables, &witness.domain, 0)?,
    ];
    let point = witness.center.clone().map(RealBall::exact);
    let point_values = [
        work.eval(&residuals[0], variables, &point, 0)?.value,
        work.eval(&residuals[1], variables, &point, 0)?.value,
    ];
    let mut defect = std::array::from_fn::<_, 2, _>(|_| {
        std::array::from_fn::<_, 2, _>(|_| RealBall::from_i64(0))
    });
    let mut norm = BigRational::zero();
    for (i, row) in defect.iter_mut().enumerate() {
        work.step()?;
        let mut row_sum = BigRational::zero();
        for (j, entry) in row.iter_mut().enumerate() {
            let mut product = RealBall::from_i64(0);
            for k in 0..2 {
                product = checked(
                    product.add(&RealBall::exact(c[i][k].clone()).mul(&box_jets[k].gradient[j])),
                )?;
            }
            *entry = checked(RealBall::from_i64(i64::from(i == j)).sub(&product))?;
            row_sum += entry.midpoint().abs() + entry.radius();
            check_rational(&row_sum)?;
        }
        norm = norm.max(row_sum);
    }
    if norm >= BigRational::one() {
        return Err(RootBoxError::NotContractive);
    }
    let delta = [
        witness.domain[0].sub(&point[0]),
        witness.domain[1].sub(&point[1]),
    ];
    let mut image = point.clone();
    for i in 0..2 {
        work.step()?;
        for j in 0..2 {
            image[i] =
                checked(image[i].sub(&RealBall::exact(c[i][j].clone()).mul(&point_values[j])))?;
            image[i] = checked(image[i].add(&defect[i][j].mul(&delta[j])))?;
        }
        if image[i].lower() <= witness.domain[i].lower()
            || image[i].upper() >= witness.domain[i].upper()
        {
            return Err(RootBoxError::NoInteriorInclusion { coordinate: i });
        }
    }
    // No accepted object is published until every obligation and final safe
    // point has succeeded.
    work.step()?;
    work.meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            (work.nodes as u64).saturating_mul(128).max(1),
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    Ok(RootBoxReceipt2 {
        residuals: residuals.clone(),
        variables: variables.clone(),
        witness: witness.clone(),
        root_enclosure: image,
        contraction_bound: norm,
    })
}

#[derive(Clone)]
struct Jet {
    value: RealBall,
    gradient: [RealBall; 2],
}

impl Jet {
    fn constant(value: RealBall) -> Self {
        Self {
            value,
            gradient: [RealBall::from_i64(0), RealBall::from_i64(0)],
        }
    }
    fn checked(self) -> Result<Self, RootBoxError> {
        check_ball(&self.value)?;
        for derivative in &self.gradient {
            check_ball(derivative)?;
        }
        Ok(self)
    }
    fn add(self, rhs: Self) -> Result<Self, RootBoxError> {
        Self {
            value: self.value.add(&rhs.value),
            gradient: std::array::from_fn(|i| self.gradient[i].add(&rhs.gradient[i])),
        }
        .checked()
    }
    fn mul(self, rhs: Self) -> Result<Self, RootBoxError> {
        Self {
            value: self.value.mul(&rhs.value),
            gradient: std::array::from_fn(|i| {
                self.gradient[i]
                    .mul(&rhs.value)
                    .add(&self.value.mul(&rhs.gradient[i]))
            }),
        }
        .checked()
    }
    fn chain(self, value: RealBall, slope: RealBall) -> Result<Self, RootBoxError> {
        Self {
            value,
            gradient: self.gradient.map(|g| g.mul(&slope)),
        }
        .checked()
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    nodes: usize,
    precision: u32,
}

impl<M: BudgetMeter> Work<'_, M> {
    fn step(&mut self) -> Result<(), RootBoxError> {
        self.meter.checkpoint()?;
        self.meter.charge(Dimension::ComputeSteps, 1)?;
        Ok(())
    }

    fn eval(
        &mut self,
        expression: &Expr,
        variables: &[Symbol; 2],
        domain: &[RealBall; 2],
        depth: usize,
    ) -> Result<Jet, RootBoxError> {
        self.step()?;
        if depth > MAX_ROOT_BOX_DEPTH || self.nodes >= MAX_ROOT_BOX_NODES {
            return Err(RootBoxError::ResourceLimit(
                "expression depth or node count",
            ));
        }
        self.meter.charge_batch(&[
            (Dimension::MemoryBytes, MAX_ROOT_BOX_LIMBS * 32 + 512),
            (Dimension::AllocationCount, 16),
        ])?;
        self.nodes += 1;
        let digits = self.precision;
        let result = match expression {
            Expr::Integer(n) => {
                if n.limb_count() > MAX_ROOT_BOX_LIMBS {
                    return Err(RootBoxError::ResourceLimit("integer height"));
                }
                Jet::constant(RealBall::exact(BigRational::from_integer(n.clone())))
            }
            Expr::Rational(q) => {
                check_rational(q)?;
                Jet::constant(RealBall::exact(q.clone()))
            }
            Expr::Const(Constant::Pi) => Jet::constant(RealBall::pi(digits)?),
            Expr::Const(Constant::E) => Jet::constant(RealBall::from_i64(1).exp(digits)?),
            Expr::Const(_) => return Err(RootBoxError::Domain("non-real or non-finite constant")),
            Expr::Sym(s) => {
                if s.name.len() > 4096 {
                    return Err(RootBoxError::ResourceLimit("symbol length"));
                }
                let index = variables
                    .iter()
                    .position(|v| v == s)
                    .ok_or(RootBoxError::Unsupported("unbound symbol identity"))?;
                let mut result = Jet::constant(domain[index].clone());
                result.gradient[index] = RealBall::from_i64(1);
                result
            }
            Expr::Add(args) | Expr::Mul(args) => {
                if args.len() > MAX_ROOT_BOX_NODES.saturating_sub(self.nodes) {
                    return Err(RootBoxError::ResourceLimit("variadic expression width"));
                }
                let is_sum = matches!(expression, Expr::Add(_));
                let mut result = Jet::constant(RealBall::from_i64(if is_sum { 0 } else { 1 }));
                for arg in args {
                    // Never simplify away a child's original domain obligation.
                    let next = self.eval(arg, variables, domain, depth + 1)?;
                    result = if is_sum {
                        result.add(next)?
                    } else {
                        result.mul(next)?
                    };
                }
                result
            }
            Expr::Pow(base, exponent) => {
                let b = self.eval(base, variables, domain, depth + 1)?;
                let e = self.eval(exponent, variables, domain, depth + 1)?;
                let integer = match exponent.as_ref() {
                    Expr::Integer(n) => Some(n),
                    Expr::Rational(q) if q.denom().is_one() => Some(q.numer()),
                    _ => None,
                };
                if let Some(integer) = integer {
                    let k = integer
                        .to_i32()
                        .filter(|k| (-64..=64).contains(k))
                        .ok_or(RootBoxError::ResourceLimit("integer exponent"))?;
                    let value = b.value.pow(k)?;
                    let slope = match k {
                        0 => RealBall::from_i64(0),
                        1 => RealBall::from_i64(1),
                        _ => b.value.pow(k - 1)?.mul(&RealBall::from_i64(i64::from(k))),
                    };
                    b.chain(value, slope)?
                } else {
                    // Real principal power is C1 on a neighborhood of the
                    // complete box only after this strict domain obligation.
                    if !b.value.is_positive() {
                        return Err(RootBoxError::Domain(
                            "noninteger power base is not strictly positive",
                        ));
                    }
                    let logarithm = b.value.ln(digits)?;
                    let value = interval_exp(&checked(e.value.mul(&logarithm))?, digits)?;
                    let mut gradient = [RealBall::from_i64(0), RealBall::from_i64(0)];
                    for (i, derivative) in gradient.iter_mut().enumerate() {
                        let chain = e.gradient[i]
                            .mul(&logarithm)
                            .add(&e.value.mul(&b.gradient[i].div(&b.value)?));
                        *derivative = checked(value.mul(&checked(chain)?))?;
                    }
                    Jet { value, gradient }
                }
            }
            Expr::Function(name, args) => {
                let [arg] = args.as_slice() else {
                    return Err(RootBoxError::Unsupported("non-unary function"));
                };
                let a = self.eval(arg, variables, domain, depth + 1)?;
                let (value, slope) = match name.as_str() {
                    "sin" => (a.value.sin(digits)?, a.value.cos(digits)?),
                    "cos" => (a.value.cos(digits)?, a.value.sin(digits)?.neg()),
                    "exp" => {
                        let value = interval_exp(&a.value, digits)?;
                        (value.clone(), value)
                    }
                    "log" | "ln" => {
                        if !a.value.is_positive() {
                            return Err(RootBoxError::Domain(
                                "logarithm argument is not strictly positive",
                            ));
                        }
                        (a.value.ln(digits)?, a.value.inv()?)
                    }
                    "sqrt" => {
                        if !a.value.is_positive() {
                            return Err(RootBoxError::Domain("square root is not C1 on this box"));
                        }
                        let value = a.value.sqrt(digits.saturating_mul(4).saturating_add(64))?;
                        let slope = value.mul(&RealBall::from_i64(2)).inv()?;
                        (value, slope)
                    }
                    "atan" => {
                        let slope = RealBall::from_i64(1).add(&a.value.pow(2)?).inv()?;
                        (a.value.atan(digits)?, slope)
                    }
                    "tan" => {
                        let cosine = a.value.cos(digits)?;
                        let inverse = cosine.inv()?;
                        (a.value.sin(digits)?.mul(&inverse), inverse.pow(2)?)
                    }
                    "sinh" | "cosh" | "tanh" => {
                        let plus = interval_exp(&a.value, digits)?;
                        let minus = interval_exp(&a.value.neg(), digits)?;
                        let sinh = plus.sub(&minus).div(&RealBall::from_i64(2))?;
                        let cosh = plus.add(&minus).div(&RealBall::from_i64(2))?;
                        match name.as_str() {
                            "sinh" => (sinh, cosh),
                            "cosh" => (cosh, sinh),
                            _ => (sinh.div(&cosh)?, cosh.inv()?.pow(2)?),
                        }
                    }
                    _ => {
                        return Err(RootBoxError::Unsupported(
                            "function has no C1 interval rule",
                        ));
                    }
                };
                a.chain(value, slope)?
            }
        };
        result.checked()
    }
}

/// Check shape iteratively before the recursive interval evaluator can consume
/// stack space. The admitted depth is tested on the normal thread stack.
fn preflight_expressions(
    residuals: &[Expr; 2],
    meter: &mut impl BudgetMeter,
) -> Result<(), RootBoxError> {
    meter.charge_batch(&[
        (
            Dimension::MemoryBytes,
            (MAX_ROOT_BOX_NODES * std::mem::size_of::<(&Expr, usize)>()) as u64,
        ),
        (Dimension::AllocationCount, 1),
    ])?;
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(MAX_ROOT_BOX_NODES)
        .map_err(|_| RootBoxError::ResourceLimit("preflight allocation"))?;
    stack.extend(residuals.iter().map(|expression| (expression, 0usize)));
    let mut visited = 0usize;
    while let Some((expression, depth)) = stack.pop() {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        if depth > MAX_ROOT_BOX_DEPTH || visited >= MAX_ROOT_BOX_NODES {
            return Err(RootBoxError::ResourceLimit(
                "expression depth or node count",
            ));
        }
        visited += 1;
        match expression {
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                if args.len() > MAX_ROOT_BOX_NODES.saturating_sub(visited + stack.len()) {
                    return Err(RootBoxError::ResourceLimit("variadic expression width"));
                }
                stack.extend(args.iter().map(|arg| (arg, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if visited + stack.len() + 2 > MAX_ROOT_BOX_NODES {
                    return Err(RootBoxError::ResourceLimit("expression node count"));
                }
                stack.push((base.as_ref(), depth + 1));
                stack.push((exponent.as_ref(), depth + 1));
            }
            Expr::Integer(integer) if integer.limb_count() > MAX_ROOT_BOX_LIMBS => {
                return Err(RootBoxError::ResourceLimit("integer height"));
            }
            Expr::Rational(rational) => check_rational(rational)?,
            _ => {}
        }
    }
    Ok(())
}

/// RealBall::exp promises an absolute output-radius target and therefore
/// refuses wide input balls. The interval extension instead uses monotonicity
/// and certified exact-endpoint evaluations; it retains the full range width.
fn interval_exp(argument: &RealBall, digits: u32) -> Result<RealBall, RootBoxError> {
    if argument.radius().is_zero() {
        return checked(argument.exp(digits)?);
    }
    let lower = RealBall::exact(argument.lower()).exp(digits)?.lower();
    let upper = RealBall::exact(argument.upper()).exp(digits)?.upper();
    let two = BigRational::from_integer(2.into());
    checked(RealBall::new(
        (&lower + &upper) / &two,
        (upper - lower) / two,
    )?)
}

fn check_rational(value: &BigRational) -> Result<(), RootBoxError> {
    if value.numer().limb_count() > MAX_ROOT_BOX_LIMBS
        || value.denom().limb_count() > MAX_ROOT_BOX_LIMBS
    {
        return Err(RootBoxError::ResourceLimit("rational height"));
    }
    Ok(())
}

fn check_ball(value: &RealBall) -> Result<(), RootBoxError> {
    check_rational(value.midpoint())?;
    check_rational(value.radius())
}

fn checked(value: RealBall) -> Result<RealBall, RootBoxError> {
    check_ball(&value)?;
    Ok(value)
}
