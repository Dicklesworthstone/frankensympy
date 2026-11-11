//! Global extrema on a declared finite real box, checked from the original
//! source. The candidate point has no authority: interval second-order AD,
//! a sufficient semidefinite LDL check, and box first-order conditions decide.
//! No claim is made outside the box or about arbitrary nonconvex objectives.

use super::objective::evaluate_objective;
use super::{RootBoxError, Work, binding_bytes, check_rational, rational_limbs, validate_inputs};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::{Signed, Zero};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Which global inequality is checked on the complete declared box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoxExtremumKind {
    Minimum,
    Maximum,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BoxExtremumError {
    #[error("box-extremum verification refused: {0}")]
    Verification(#[from] RootBoxError),
    #[error("the required Hessian semidefiniteness is not proved throughout the box")]
    CurvatureNotProved,
    #[error("the box first-order condition is not proved at coordinate {coordinate}")]
    FirstOrderNotProved { coordinate: usize },
}

/// An exact candidate is a global minimum/maximum of the original objective
/// on this closed box. A false uniqueness_proved means only that uniqueness
/// was not established; it does not assert that other optimizers exist.
/// Fields are private, with no unchecked constructor or deserializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxExtremumReceipt {
    objective: Expr,
    variables: Vec<Symbol>,
    domain: Vec<RealBall>,
    point: Vec<BigRational>,
    precision_digits: u32,
    kind: BoxExtremumKind,
    value: RealBall,
    gradient: Vec<RealBall>,
    hessian: Vec<Vec<RealBall>>,
    uniqueness_proved: bool,
}

impl BoxExtremumReceipt {
    pub fn objective(&self) -> &Expr {
        &self.objective
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }
    pub fn domain(&self) -> &[RealBall] {
        &self.domain
    }
    pub fn point(&self) -> &[BigRational] {
        &self.point
    }
    pub fn precision_digits(&self) -> u32 {
        self.precision_digits
    }
    pub fn kind(&self) -> BoxExtremumKind {
        self.kind
    }
    /// Enclosure of the objective at the proved optimizer, hence the optimum
    /// value on the box. This is not the range of all objective values.
    pub fn value(&self) -> &RealBall {
        &self.value
    }
    /// Original-source gradient enclosure at the exact candidate point.
    pub fn gradient(&self) -> &[RealBall] {
        &self.gradient
    }
    /// Original-source Hessian enclosure throughout the complete box.
    pub fn hessian(&self) -> &[Vec<RealBall>] {
        &self.hessian
    }
    pub fn uniqueness_proved(&self) -> bool {
        self.uniqueness_proved
    }
}

/// Check an exact candidate without a caller-owned work ledger. Intrinsic
/// bounds are the same as the original-source objective enclosure checker.
pub fn certify_box_extremum(
    objective: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    point: &[BigRational],
    kind: BoxExtremumKind,
    precision_digits: u32,
) -> Result<BoxExtremumReceipt, BoxExtremumError> {
    verify_box_extremum(
        objective,
        variables,
        domain,
        point,
        kind,
        precision_digits,
        &mut Unbounded,
    )
}

/// Verify the convex first-order inequality for a minimum, or its concave
/// counterpart for a maximum. Original-source C2 admission covers every box
/// point, including endpoints and children hidden by zero factors/powers.
/// All preparation, arithmetic and publication share the supplied ledger.
pub fn verify_box_extremum(
    objective: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    point: &[BigRational],
    kind: BoxExtremumKind,
    precision_digits: u32,
    meter: &mut impl BudgetMeter,
) -> Result<BoxExtremumReceipt, BoxExtremumError> {
    let mut work = Work::new(meter, precision_digits)?;
    validate_inputs(
        std::slice::from_ref(objective),
        variables,
        domain,
        &mut work,
    )?;
    let n = variables.len();
    if point.len() != n {
        return Err(RootBoxError::DimensionMismatch.into());
    }
    for (coordinate, (interval, value)) in domain.iter().zip(point).enumerate() {
        check_rational(value)?;
        work.arithmetic(&[interval], &[value])?;
        if !interval.contains(value) {
            return Err(RootBoxError::CenterOutsideBox { coordinate }.into());
        }
    }
    let whole = evaluate_objective(objective, variables, domain, &mut work)?;
    let definite = semidefinite_hessian(&whole.hessian, kind, &mut work)?;

    work.charge(1, (n * 256) as u64, (n + 1) as u64)?;
    let at: Vec<_> = point.iter().cloned().map(RealBall::exact).collect();
    let candidate = evaluate_objective(objective, variables, &at, &mut work)?;
    // A strict supporting linear form on every movable coordinate also proves
    // uniqueness, including an affine objective minimized at a strict corner.
    let mut strict_first_order = true;
    for (coordinate, ((interval, value), gradient)) in domain
        .iter()
        .zip(point)
        .zip(&candidate.gradient)
        .enumerate()
    {
        work.arithmetic(&[interval, gradient], &[value])?;
        if interval.radius().is_zero() {
            continue;
        }
        let gradient = orient(gradient, kind);
        let (admitted, strict) = if value == &interval.lower() {
            (!gradient.lower().is_negative(), gradient.is_positive())
        } else if value == &interval.upper() {
            (!gradient.upper().is_positive(), gradient.is_negative())
        } else {
            (is_exact_zero(&gradient), false)
        };
        if !admitted {
            return Err(BoxExtremumError::FirstOrderNotProved { coordinate });
        }
        strict_first_order &= strict;
    }
    let point_bytes = point
        .iter()
        .map(|q| 128 + 16 * rational_limbs(q))
        .sum::<u64>();
    work.charge(
        1,
        work.source_bytes + binding_bytes(variables, domain) + point_bytes,
        (n + 6) as u64,
    )?;
    let receipt = BoxExtremumReceipt {
        objective: objective.clone(),
        variables: variables.to_vec(),
        domain: domain.to_vec(),
        point: point.to_vec(),
        precision_digits,
        kind,
        value: candidate.value,
        gradient: candidate.gradient,
        hessian: whole.hessian,
        uniqueness_proved: definite || strict_first_order,
    };
    work.step()?;
    Ok(receipt)
}

fn is_exact_zero(value: &RealBall) -> bool {
    value.midpoint().is_zero() && value.radius().is_zero()
}

fn orient(value: &RealBall, kind: BoxExtremumKind) -> RealBall {
    match kind {
        BoxExtremumKind::Minimum => value.clone(),
        BoxExtremumKind::Maximum => value.neg(),
    }
}

/// Sufficient interval LDL test for positive semidefiniteness of H or -H.
/// Every exact Schur complement is enclosed. A pivot reaching zero is safe
/// only if its whole remaining Schur column is identically zero; otherwise
/// division/acceptance is refused. In particular, zero diagonal entries do
/// not excuse an indefinite off-diagonal block such as [[0,1],[1,0]].
fn semidefinite_hessian(
    hessian: &[Vec<RealBall>],
    kind: BoxExtremumKind,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool, BoxExtremumError> {
    let n = hessian.len();
    work.charge(1, (n * n * 512) as u64, (n + 3) as u64)?;
    let mut lower = vec![vec![RealBall::from_i64(0); n]; n];
    let mut pivots = Vec::<RealBall>::with_capacity(n);
    let mut definite = true;
    for i in 0..n {
        work.arithmetic(&[&hessian[i][i]], &[])?;
        let mut diagonal = orient(&hessian[i][i], kind);
        for (k, pivot) in pivots.iter().enumerate() {
            let square = work.power(&lower[i][k], 2)?;
            let contribution = work.mul(&square, pivot)?;
            diagonal = work.sub(&diagonal, &contribution)?;
        }
        work.arithmetic(&[&diagonal], &[])?;
        if diagonal.lower().is_negative() {
            return Err(BoxExtremumError::CurvatureNotProved);
        }
        let positive = diagonal.is_positive();
        definite &= positive;
        for j in i + 1..n {
            work.arithmetic(&[&hessian[j][i]], &[])?;
            let mut entry = orient(&hessian[j][i], kind);
            for (k, pivot) in pivots.iter().enumerate() {
                let product = work.mul(&lower[j][k], &lower[i][k])?;
                let contribution = work.mul(&product, pivot)?;
                entry = work.sub(&entry, &contribution)?;
            }
            if positive {
                lower[j][i] = work.div(&entry, &diagonal)?;
            } else if !is_exact_zero(&entry) {
                return Err(BoxExtremumError::CurvatureNotProved);
            }
        }
        pivots.push(diagonal);
    }
    Ok(definite)
}
