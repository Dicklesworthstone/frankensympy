//! Uniform original-source root certification for a family F(x, p) = 0.
//!
//! For every fixed parameter p in its declared closed box P, the same
//! rational preconditioner and unknown center define a contraction of X
//! into its strict interior. This proves exactly one nonsingular root in X
//! for each p. The image encloses the union of those parameter-dependent
//! roots; it is not one fixed root, an exact symbolic solution, an assertion
//! outside X, or a global continuation/path certificate.
//!
//! The original source is checked C1 over the full product X × P. Both
//! F(c, P) and J_x(X × P) retain every parameter interval. Only derivatives
//! with respect to the explicitly ordered unknowns enter the Krawczyk map.

use super::{
    MAX_ROOT_BOX_PRECISION, MAX_ROOT_BOX_VARIABLES, RootBoxError, Work, binding_bytes, check_ball,
    check_rational, rational_limbs, validate_inputs,
};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::{One, Signed, Zero};

/// Untrusted uniform Krawczyk proposal. Unknowns and parameters are bound
/// separately by the verifier. Parameter intervals may have zero width;
/// each unknown coordinate must have strictly positive width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricRootBoxWitness {
    pub unknown_domain: Vec<RealBall>,
    pub parameter_domain: Vec<RealBall>,
    pub center: Vec<BigRational>,
    pub preconditioner: Vec<Vec<BigRational>>,
    pub precision_digits: u32,
}

/// For each parameter in parameter_domain there exists exactly one
/// nonsingular real unknown root in unknown_domain. The root_enclosure
/// contains every such root. All fields are private; no unchecked
/// constructor or deserializer can manufacture uniform acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricRootBoxReceipt {
    residuals: Vec<Expr>,
    unknowns: Vec<Symbol>,
    parameters: Vec<Symbol>,
    witness: ParametricRootBoxWitness,
    root_enclosure: Vec<RealBall>,
    contraction_bound: BigRational,
}

impl ParametricRootBoxReceipt {
    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }
    pub fn unknowns(&self) -> &[Symbol] {
        &self.unknowns
    }
    pub fn parameters(&self) -> &[Symbol] {
        &self.parameters
    }
    pub fn witness(&self) -> &ParametricRootBoxWitness {
        &self.witness
    }
    pub fn root_enclosure(&self) -> &[RealBall] {
        &self.root_enclosure
    }
    pub fn contraction_bound(&self) -> &BigRational {
        &self.contraction_bound
    }
}

/// Independent uniform certification on a closed parameter box.
///
/// All candidate data is rechecked against the original expressions, never
/// a symbolic derivative, compiled program or sampled parameter value. One
/// Work phase retains arithmetic, allocation and cancellation accounting;
/// only the node counter restarts for the second complete source traversal.
pub fn verify_parametric_root_box(
    residuals: &[Expr],
    unknowns: &[Symbol],
    parameters: &[Symbol],
    witness: &ParametricRootBoxWitness,
    meter: &mut impl BudgetMeter,
) -> Result<ParametricRootBoxReceipt, RootBoxError> {
    let mut work = Work::new(meter, witness.precision_digits)?;
    let n = unknowns.len();
    let p = parameters.len();
    let total = n.checked_add(p).ok_or(RootBoxError::DimensionMismatch)?;
    if n == 0
        || p == 0
        || total > MAX_ROOT_BOX_VARIABLES
        || residuals.len() != n
        || witness.unknown_domain.len() != n
        || witness.parameter_domain.len() != p
        || witness.center.len() != n
        || witness.preconditioner.len() != n
        || witness.preconditioner.iter().any(|row| row.len() != n)
    {
        return Err(RootBoxError::DimensionMismatch);
    }
    if !(8..=MAX_ROOT_BOX_PRECISION).contains(&witness.precision_digits) {
        return Err(RootBoxError::InvalidPrecision);
    }
    // Bound all metadata and scalar payloads before concatenating or cloning
    // the caller's two coordinate groups. Identical printed names do not
    // replace typed identity; duplicates across either group are rejected.
    for variable in unknowns.iter().chain(parameters) {
        if variable.name.len() > 4096 {
            return Err(RootBoxError::ResourceLimit("symbol length"));
        }
    }
    for (index, variable) in unknowns.iter().chain(parameters).enumerate() {
        if unknowns
            .iter()
            .chain(parameters)
            .take(index)
            .any(|v| v == variable)
        {
            return Err(RootBoxError::DuplicateVariable);
        }
    }
    for interval in witness
        .unknown_domain
        .iter()
        .chain(&witness.parameter_domain)
    {
        check_ball(interval)?;
    }
    for value in witness
        .center
        .iter()
        .chain(witness.preconditioner.iter().flatten())
    {
        check_rational(value)?;
    }
    for (coordinate, (interval, center)) in witness
        .unknown_domain
        .iter()
        .zip(&witness.center)
        .enumerate()
    {
        work.arithmetic(&[interval], &[center])?;
        if interval.radius().is_zero() {
            return Err(RootBoxError::DegenerateBox { coordinate });
        }
        if !interval.contains(center) {
            return Err(RootBoxError::CenterOutsideBox { coordinate });
        }
    }
    let bindings = binding_bytes(unknowns, &witness.unknown_domain)
        + binding_bytes(parameters, &witness.parameter_domain);
    work.charge(1, bindings, (2 * total + 2) as u64)?;
    let mut variables = Vec::new();
    variables
        .try_reserve_exact(total)
        .map_err(|_| RootBoxError::ResourceLimit("parametric variable allocation"))?;
    variables.extend_from_slice(unknowns);
    variables.extend_from_slice(parameters);
    let mut product_domain = Vec::new();
    product_domain
        .try_reserve_exact(total)
        .map_err(|_| RootBoxError::ResourceLimit("parametric domain allocation"))?;
    product_domain.extend_from_slice(&witness.unknown_domain);
    product_domain.extend_from_slice(&witness.parameter_domain);
    validate_inputs(residuals, &variables, &product_domain, &mut work)?;

    work.charge(
        1,
        (n * total * 512 + total * 256) as u64 + bindings,
        (n + total + 4) as u64,
    )?;
    let mut jets = Vec::with_capacity(n);
    for residual in residuals {
        jets.push(work.eval(residual, &variables, &product_domain, 0)?);
    }
    let center_boxes: Vec<_> = witness
        .center
        .iter()
        .cloned()
        .map(RealBall::exact)
        .collect();
    let mut centered_domain = Vec::with_capacity(total);
    centered_domain.extend_from_slice(&center_boxes);
    // These are the FULL parameter intervals, not their midpoints.
    centered_domain.extend_from_slice(&witness.parameter_domain);
    let mut values = Vec::with_capacity(n);
    work.nodes = 0;
    for residual in residuals {
        values.push(work.eval(residual, &variables, &centered_domain, 0)?.value);
    }

    work.charge(1, (n * n * 512) as u64, (n * n + 4) as u64)?;
    let mut defect = vec![vec![RealBall::from_i64(0); n]; n];
    let mut norm = BigRational::zero();
    for (i, row) in defect.iter_mut().enumerate() {
        let mut row_sum = BigRational::zero();
        for (j, entry) in row.iter_mut().enumerate() {
            let mut product = RealBall::from_i64(0);
            for (coefficient, jet) in witness.preconditioner[i].iter().zip(&jets) {
                // Variables are ordered [unknowns, parameters]. The j loop
                // ranges over unknowns only; parameter partials do not
                // belong to the square fixed-point map on X.
                let term = work.scale(&jet.gradient[j], coefficient)?;
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
    for (interval, center) in witness.unknown_domain.iter().zip(&center_boxes) {
        delta.push(work.sub(interval, center)?);
    }
    let mut image = center_boxes;
    for (i, coordinate) in image.iter_mut().enumerate() {
        for j in 0..n {
            let displacement = work.scale(&values[j], &witness.preconditioner[i][j])?;
            *coordinate = work.sub(coordinate, &displacement)?;
            let remainder = work.mul(&defect[i][j], &delta[j])?;
            *coordinate = work.add(coordinate, &remainder)?;
        }
        work.arithmetic(&[coordinate, &witness.unknown_domain[i]], &[])?;
        if coordinate.lower() <= witness.unknown_domain[i].lower()
            || coordinate.upper() >= witness.unknown_domain[i].upper()
        {
            return Err(RootBoxError::NoInteriorInclusion { coordinate: i });
        }
    }
    let witness_bytes = witness
        .center
        .iter()
        .chain(witness.preconditioner.iter().flatten())
        .map(|value| 128 + 16 * rational_limbs(value))
        .sum::<u64>();
    work.charge(
        1,
        work.source_bytes + bindings + witness_bytes,
        (n + p + 8) as u64,
    )?;
    let receipt = ParametricRootBoxReceipt {
        residuals: residuals.to_vec(),
        unknowns: unknowns.to_vec(),
        parameters: parameters.to_vec(),
        witness: witness.clone(),
        root_enclosure: image,
        contraction_bound: norm,
    };
    work.step()?;
    Ok(receipt)
}

/// Convenience checking preserves the intrinsic phase bounds even without
/// an external ledger. The preconditioner remains an untrusted input.
pub fn certify_parametric_root_box(
    residuals: &[Expr],
    unknowns: &[Symbol],
    parameters: &[Symbol],
    witness: &ParametricRootBoxWitness,
) -> Result<ParametricRootBoxReceipt, RootBoxError> {
    verify_parametric_root_box(residuals, unknowns, parameters, witness, &mut Unbounded)
}
