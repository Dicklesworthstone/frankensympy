//! Stationary-point certificates from second-order interval evaluation of the
//! original objective. Symbolic derivatives and numerical optimizers are not
//! part of this checker's dependency or trust boundary.

use super::objective::evaluate_objective;
use super::{
    RootBoxError, RootBoxWitness, Work, binding_bytes, check_rational, rational_limbs,
    validate_inputs,
};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::{One, Signed, Zero};

/// A proved local classification. `Unclassified` does not assert the absence
/// of an extremum: interval LDL may be inconclusive even for a nonsingular
/// Hessian. No variant makes a claim about points outside the supplied box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationaryPointKind {
    StrictMinimum,
    StrictMaximum,
    Saddle,
    Unclassified,
}

/// Source-bound existence and uniqueness of a stationary point in one closed
/// box, with an optional proved local classification. Construction always
/// checks the original source's C2 domain, strict Krawczyk inclusion and an
/// exact contraction bound. There is no unchecked constructor/deserializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StationaryPointReceipt {
    objective: Expr,
    variables: Vec<Symbol>,
    witness: RootBoxWitness,
    stationary_enclosure: Vec<RealBall>,
    objective_enclosure: RealBall,
    gradient_enclosure: Vec<RealBall>,
    hessian_enclosure: Vec<Vec<RealBall>>,
    contraction_bound: BigRational,
    kind: StationaryPointKind,
}

impl StationaryPointReceipt {
    pub fn objective(&self) -> &Expr {
        &self.objective
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }
    pub fn witness(&self) -> &RootBoxWitness {
        &self.witness
    }
    /// A strict interior enclosure of the unique stationary point.
    pub fn stationary_enclosure(&self) -> &[RealBall] {
        &self.stationary_enclosure
    }
    /// Objective range on the full supplied box, hence also an enclosure of
    /// the value at the stationary point. It need not be a tight value bound.
    pub fn objective_enclosure(&self) -> &RealBall {
        &self.objective_enclosure
    }
    pub fn gradient_enclosure(&self) -> &[RealBall] {
        &self.gradient_enclosure
    }
    pub fn hessian_enclosure(&self) -> &[Vec<RealBall>] {
        &self.hessian_enclosure
    }
    pub fn contraction_bound(&self) -> &BigRational {
        &self.contraction_bound
    }
    pub fn kind(&self) -> StationaryPointKind {
        self.kind
    }
}

/// Convenience form with fixed intrinsic bounds and no caller-owned meter.
pub fn certify_stationary_point(
    objective: &Expr,
    variables: &[Symbol],
    witness: &RootBoxWitness,
) -> Result<StationaryPointReceipt, RootBoxError> {
    verify_stationary_point(objective, variables, witness, &mut Unbounded)
}

/// Independently check one proposed stationary-point box. The proposed
/// preconditioner is an arbitrary exact rational matrix, never a trusted
/// inverse. The same work ledger covers source validation, both interval
/// evaluations, Krawczyk contraction, curvature checks and publication.
pub fn verify_stationary_point(
    objective: &Expr,
    variables: &[Symbol],
    witness: &RootBoxWitness,
    meter: &mut impl BudgetMeter,
) -> Result<StationaryPointReceipt, RootBoxError> {
    let mut work = Work::new(meter, witness.precision_digits)?;
    validate_inputs(
        std::slice::from_ref(objective),
        variables,
        &witness.domain,
        &mut work,
    )?;
    let n = variables.len();
    if witness.center.len() != n
        || witness.preconditioner.len() != n
        || witness.preconditioner.iter().any(|row| row.len() != n)
    {
        return Err(RootBoxError::DimensionMismatch);
    }
    for value in witness
        .center
        .iter()
        .chain(witness.preconditioner.iter().flatten())
    {
        check_rational(value)?;
    }
    for (index, (coordinate, center)) in witness.domain.iter().zip(&witness.center).enumerate() {
        work.arithmetic(&[coordinate], &[center])?;
        if coordinate.radius().is_zero() {
            return Err(RootBoxError::DegenerateBox { coordinate: index });
        }
        if !coordinate.contains(center) {
            return Err(RootBoxError::CenterOutsideBox { coordinate: index });
        }
    }
    // Second-order AD sees every original child, including children whose
    // symbolic value or derivative could otherwise cancel to zero.
    let whole = evaluate_objective(objective, variables, &witness.domain, &mut work)?;
    work.charge(1, (n * n * 1024) as u64, (n * n + 8) as u64)?;
    for coordinate in &witness.center {
        work.arithmetic(&[], &[coordinate])?;
    }
    let point: Vec<_> = witness
        .center
        .iter()
        .cloned()
        .map(RealBall::exact)
        .collect();
    let at_center = evaluate_objective(objective, variables, &point, &mut work)?;

    // Apply the fixed-point test to the independently evaluated gradient.
    // ||I-C H(X)||_infinity < 1 also proves C and every H(x) nonsingular.
    let mut defect = vec![vec![RealBall::from_i64(0); n]; n];
    let mut norm = BigRational::zero();
    for (i, row) in defect.iter_mut().enumerate() {
        let mut row_sum = BigRational::zero();
        for (j, entry) in row.iter_mut().enumerate() {
            let mut product = RealBall::from_i64(0);
            for (coefficient, hessian_row) in witness.preconditioner[i].iter().zip(&whole.hessian) {
                let term = work.scale(&hessian_row[j], coefficient)?;
                product = work.add(&product, &term)?;
            }
            *entry = work.sub(&RealBall::from_i64(i64::from(i == j)), &product)?;
            work.arithmetic(&[entry], &[&row_sum])?;
            row_sum += entry.midpoint().abs() + entry.radius();
            check_rational(&row_sum)?;
        }
        work.arithmetic(&[], &[&norm, &row_sum])?;
        norm = norm.max(row_sum);
    }
    if norm >= BigRational::one() {
        return Err(RootBoxError::NotContractive);
    }
    let mut delta = Vec::with_capacity(n);
    for (coordinate, center) in witness.domain.iter().zip(&point) {
        delta.push(work.sub(coordinate, center)?);
    }
    let mut image = point;
    for (i, coordinate) in image.iter_mut().enumerate() {
        for (j, displacement) in delta.iter().enumerate() {
            let term = work.scale(&at_center.gradient[j], &witness.preconditioner[i][j])?;
            *coordinate = work.sub(coordinate, &term)?;
            let term = work.mul(&defect[i][j], displacement)?;
            *coordinate = work.add(coordinate, &term)?;
        }
        work.arithmetic(&[coordinate, &witness.domain[i]], &[])?;
        if coordinate.lower() <= witness.domain[i].lower()
            || coordinate.upper() >= witness.domain[i].upper()
        {
            return Err(RootBoxError::NoInteriorInclusion { coordinate: i });
        }
    }
    let kind = classify_hessian(&whole.hessian, &mut work)?;
    let witness_bytes = witness
        .center
        .iter()
        .chain(witness.preconditioner.iter().flatten())
        .map(|q| 128 + 16 * rational_limbs(q))
        .sum::<u64>();
    work.charge(
        1,
        work.source_bytes + witness_bytes + binding_bytes(variables, &witness.domain),
        (n + 6) as u64,
    )?;
    let receipt = StationaryPointReceipt {
        objective: objective.clone(),
        variables: variables.to_vec(),
        witness: witness.clone(),
        stationary_enclosure: image,
        objective_enclosure: whole.value,
        gradient_enclosure: whole.gradient,
        hessian_enclosure: whole.hessian,
        contraction_bound: norm,
        kind,
    };
    work.step()?;
    Ok(receipt)
}

/// Interval LDL is a sufficient inertia check, not an eigenvalue estimate.
/// For every symmetric matrix in the enclosure, each exact elimination lies
/// in these intervals. Nonzero pivots establish a congruence to D; their signs
/// therefore prove positive/negative definiteness or indefiniteness. A pivot
/// crossing zero is inconclusive and never becomes an extremum assertion.
fn classify_hessian(
    hessian: &[Vec<RealBall>],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<StationaryPointKind, RootBoxError> {
    let n = hessian.len();
    work.charge(1, (n * n * 256) as u64, (n + 2) as u64)?;
    let mut lower = vec![vec![RealBall::from_i64(0); n]; n];
    let mut pivots = Vec::<RealBall>::with_capacity(n);
    let mut positive = 0usize;
    for i in 0..n {
        work.arithmetic(&[&hessian[i][i]], &[])?;
        let mut diagonal = hessian[i][i].clone();
        for (k, pivot) in pivots.iter().enumerate() {
            let square = work.power(&lower[i][k], 2)?;
            let contribution = work.mul(&square, pivot)?;
            diagonal = work.sub(&diagonal, &contribution)?;
        }
        if diagonal.contains_zero() {
            return Ok(StationaryPointKind::Unclassified);
        }
        if diagonal.is_positive() {
            positive += 1;
        }
        for j in i + 1..n {
            work.arithmetic(&[&hessian[j][i]], &[])?;
            let mut entry = hessian[j][i].clone();
            for (k, pivot) in pivots.iter().enumerate() {
                let product = work.mul(&lower[j][k], &lower[i][k])?;
                let contribution = work.mul(&product, pivot)?;
                entry = work.sub(&entry, &contribution)?;
            }
            lower[j][i] = work.div(&entry, &diagonal)?;
        }
        pivots.push(diagonal);
    }
    Ok(if positive == n {
        StationaryPointKind::StrictMinimum
    } else if positive == 0 {
        StationaryPointKind::StrictMaximum
    } else {
        StationaryPointKind::Saddle
    })
}
