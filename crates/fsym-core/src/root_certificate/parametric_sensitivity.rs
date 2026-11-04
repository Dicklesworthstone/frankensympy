//! Original-source implicit derivatives of already accepted root families.
//!
//! The private root receipt supplies existence and uniqueness. This operation
//! independently recomputes both Jacobian blocks on every complete parameter
//! cell. With D = I - C*J_x, B = -C*J_p and q = ||D||_infinity < 1, the true
//! solution derivative S satisfies S = B + D*S. For column j, let M_j bound
//! every entry of B_j in absolute value. Then ||S_j|| <= M_j/(1-q), and each
//! S_ij is in B_ij + [-q*M_j/(1-q), q*M_j/(1-q)]. No sampled derivative or
//! user-supplied sensitivity matrix is accepted as evidence.
//!
//! Original-source C1 neighborhood checks and the accepted nonsingular roots
//! justify the implicit derivatives. On a closed parameter face these are the
//! derivatives of the local implicit extension, restricted to the box. A
//! complete fixed-X cover binds the same function at shared faces; its global
//! derivative enclosure is the hull of all accepted cells. No exact formula,
//! requested accuracy, or existence outside the original boxes is claimed.

use super::{
    MAX_PARAMETRIC_ROOT_COVER_LEAVES, MAX_ROOT_BOX_VARIABLES, ParametricRootBoxReceipt,
    ParametricRootCoverReceipt, RootBoxError, Work, binding_bytes, check_rational, checked,
    rational_limbs, validate_inputs,
};
use crate::{BigRational, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Unbounded};
use num_traits::{One, Signed, Zero};

/// Source-bound bounds on dx_i/dp_j for an accepted unique root family.
/// There is no unchecked constructor, public mutable field or deserializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParametricSensitivityReceipt {
    residuals: Vec<Expr>,
    unknowns: Vec<Symbol>,
    parameters: Vec<Symbol>,
    unknown_domain: Vec<RealBall>,
    parameter_domain: Vec<RealBall>,
    root_enclosure: Vec<RealBall>,
    solution_jacobian: Vec<Vec<RealBall>>,
    contraction_bound: BigRational,
    precision_digits: u32,
    leaf_count: usize,
}

impl ParametricSensitivityReceipt {
    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }
    pub fn unknowns(&self) -> &[Symbol] {
        &self.unknowns
    }
    pub fn parameters(&self) -> &[Symbol] {
        &self.parameters
    }
    pub fn unknown_domain(&self) -> &[RealBall] {
        &self.unknown_domain
    }
    pub fn parameter_domain(&self) -> &[RealBall] {
        &self.parameter_domain
    }
    pub fn root_enclosure(&self) -> &[RealBall] {
        &self.root_enclosure
    }
    /// Row i corresponds to unknown i; column j corresponds to parameter j.
    pub fn solution_jacobian(&self) -> &[Vec<RealBall>] {
        &self.solution_jacobian
    }
    /// Maximum independently recomputed contraction bound over every cell.
    pub fn contraction_bound(&self) -> &BigRational {
        &self.contraction_bound
    }
    pub fn precision_digits(&self) -> u32 {
        self.precision_digits
    }
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }
}

/// Only existing immutable acceptance types can construct this borrowed view.
/// The cover verifier has already bound every leaf to the same source and X,
/// with complete coverage of P. No new user-supplied leaf list enters here.
struct AcceptedFamily<'a> {
    residuals: &'a [Expr],
    unknowns: &'a [Symbol],
    parameters: &'a [Symbol],
    unknown_domain: &'a [RealBall],
    parameter_domain: &'a [RealBall],
    root_enclosure: &'a [RealBall],
    precision_digits: u32,
    leaves: &'a [ParametricRootBoxReceipt],
}

/// Derive interval sensitivities without rerunning the root search. Accepted
/// input supplies root existence; the original source supplies all derivatives.
pub fn metered_enclose_parametric_root_sensitivity(
    family: &ParametricRootBoxReceipt,
    meter: &mut impl BudgetMeter,
) -> Result<ParametricSensitivityReceipt, RootBoxError> {
    enclose_accepted_family(
        AcceptedFamily {
            residuals: family.residuals(),
            unknowns: family.unknowns(),
            parameters: family.parameters(),
            unknown_domain: &family.witness().unknown_domain,
            parameter_domain: &family.witness().parameter_domain,
            root_enclosure: family.root_enclosure(),
            precision_digits: family.witness().precision_digits,
            leaves: std::slice::from_ref(family),
        },
        meter,
    )
}

/// Derive bounds on the entire accepted parameter cover, retaining every leaf.
/// One source preflight and scratch reservation serve the complete operation.
pub fn metered_enclose_parametric_cover_sensitivity(
    family: &ParametricRootCoverReceipt,
    meter: &mut impl BudgetMeter,
) -> Result<ParametricSensitivityReceipt, RootBoxError> {
    enclose_accepted_family(
        AcceptedFamily {
            residuals: family.residuals(),
            unknowns: family.unknowns(),
            parameters: family.parameters(),
            unknown_domain: family.unknown_domain(),
            parameter_domain: family.parameter_domain(),
            root_enclosure: family.root_enclosure(),
            precision_digits: family.precision_digits(),
            leaves: family.leaves(),
        },
        meter,
    )
}

pub fn enclose_parametric_root_sensitivity(
    family: &ParametricRootBoxReceipt,
) -> Result<ParametricSensitivityReceipt, RootBoxError> {
    metered_enclose_parametric_root_sensitivity(family, &mut Unbounded)
}

pub fn enclose_parametric_cover_sensitivity(
    family: &ParametricRootCoverReceipt,
) -> Result<ParametricSensitivityReceipt, RootBoxError> {
    metered_enclose_parametric_cover_sensitivity(family, &mut Unbounded)
}

fn enclose_accepted_family(
    family: AcceptedFamily<'_>,
    meter: &mut impl BudgetMeter,
) -> Result<ParametricSensitivityReceipt, RootBoxError> {
    let mut work = Work::new(meter, family.precision_digits)?;
    let n = family.unknowns.len();
    let p = family.parameters.len();
    let total = n.checked_add(p).ok_or(RootBoxError::DimensionMismatch)?;
    if n == 0
        || p == 0
        || total > MAX_ROOT_BOX_VARIABLES
        || family.residuals.len() != n
        || family.unknown_domain.len() != n
        || family.parameter_domain.len() != p
        || family.root_enclosure.len() != n
        || !(1..=MAX_PARAMETRIC_ROOT_COVER_LEAVES).contains(&family.leaves.len())
    {
        return Err(RootBoxError::DimensionMismatch);
    }
    // All scalars, names and source nodes are bounded by the accepted input.
    // Account before any input clone; recheck the original source once below.
    let bindings = binding_bytes(family.unknowns, family.unknown_domain)
        + binding_bytes(family.parameters, family.parameter_domain);
    work.charge(1, bindings, (2 * total + 4) as u64)?;
    let variables: Vec<_> = family
        .unknowns
        .iter()
        .chain(family.parameters)
        .cloned()
        .collect();
    let product: Vec<_> = family
        .unknown_domain
        .iter()
        .chain(family.parameter_domain)
        .cloned()
        .collect();
    validate_inputs(family.residuals, &variables, &product, &mut work)?;
    let mut solution_jacobian: Option<Vec<Vec<RealBall>>> = None;
    let mut contraction_bound = BigRational::zero();
    for leaf in family.leaves {
        work.charge(
            1,
            (n * total * 512 + total * 256 + n * p * 512) as u64
                + binding_bytes(family.unknowns, family.unknown_domain)
                + binding_bytes(family.parameters, &leaf.witness().parameter_domain),
            (n * 2 + total + 8) as u64,
        )?;
        let domain: Vec<_> = family
            .unknown_domain
            .iter()
            .chain(&leaf.witness().parameter_domain)
            .cloned()
            .collect();
        // Reset only the per-forest traversal counter; all arithmetic,
        // allocation, source bindings and caller cancellation remain cumulative.
        work.nodes = 0;
        let mut jets = Vec::with_capacity(n);
        for residual in family.residuals {
            jets.push(work.eval(residual, &variables, &domain, 0)?);
        }
        let mut q = BigRational::zero();
        for (i, coefficients) in leaf.witness().preconditioner.iter().enumerate() {
            let mut row_sum = BigRational::zero();
            for j in 0..n {
                let mut product = RealBall::from_i64(0);
                for (coefficient, jet) in coefficients.iter().zip(&jets) {
                    let term = work.scale(&jet.gradient[j], coefficient)?;
                    product = work.add(&product, &term)?;
                }
                let defect = work.sub(&RealBall::from_i64(i64::from(i == j)), &product)?;
                work.arithmetic(&[&defect], &[&row_sum])?;
                row_sum += defect.midpoint().abs() + defect.radius();
                check_rational(&row_sum)?;
            }
            work.arithmetic(&[], &[&q, &row_sum])?;
            q = q.max(row_sum);
        }
        work.arithmetic(&[], &[&q])?;
        if q >= BigRational::one() {
            return Err(RootBoxError::NotContractive);
        }
        work.arithmetic(&[], &[&q, &contraction_bound])?;
        contraction_bound = contraction_bound.max(q.clone());
        let mut rhs = vec![vec![RealBall::from_i64(0); p]; n];
        for (row, coefficients) in rhs.iter_mut().zip(&leaf.witness().preconditioner) {
            for (j, entry) in row.iter_mut().enumerate() {
                for (coefficient, jet) in coefficients.iter().zip(&jets) {
                    let term = work.scale(&jet.gradient[n + j], coefficient)?;
                    *entry = work.sub(entry, &term)?;
                }
            }
        }
        work.arithmetic(&[], &[&q])?;
        let margin = BigRational::one() - &q;
        check_rational(&margin)?;
        for j in 0..p {
            let mut magnitude = BigRational::zero();
            for row in &rhs {
                work.arithmetic(&[&row[j]], &[&magnitude])?;
                magnitude = magnitude.max(row[j].midpoint().abs() + row[j].radius());
                check_rational(&magnitude)?;
            }
            work.arithmetic(&[], &[&q, &magnitude])?;
            let numerator = &q * &magnitude;
            check_rational(&numerator)?;
            work.arithmetic(&[], &[&numerator, &margin])?;
            let radius = numerator / &margin;
            check_rational(&radius)?;
            let remainder = checked(RealBall::new(BigRational::zero(), radius)?)?;
            for row in &mut rhs {
                row[j] = work.add(&row[j], &remainder)?;
            }
        }
        if let Some(hull) = &mut solution_jacobian {
            for (row, next) in hull.iter_mut().zip(rhs) {
                for (entry, next) in row.iter_mut().zip(next) {
                    work.arithmetic(&[entry, &next], &[])?;
                    let lower = entry.lower().min(next.lower());
                    let upper = entry.upper().max(next.upper());
                    work.arithmetic(&[], &[&lower, &upper])?;
                    *entry = checked(RealBall::new(
                        (&lower + &upper) / BigRational::from_integer(2.into()),
                        (upper - lower) / BigRational::from_integer(2.into()),
                    )?)?;
                }
            }
        } else {
            solution_jacobian = Some(rhs);
        }
    }
    let solution_jacobian = solution_jacobian.ok_or(RootBoxError::DimensionMismatch)?;
    let root_bytes = family
        .root_enclosure
        .iter()
        .map(|value| 256 + 16 * (rational_limbs(value.midpoint()) + rational_limbs(value.radius())))
        .sum::<u64>();
    work.charge(
        1,
        work.source_bytes + bindings + root_bytes,
        (n + p + 8) as u64,
    )?;
    let receipt = ParametricSensitivityReceipt {
        residuals: family.residuals.to_vec(),
        unknowns: family.unknowns.to_vec(),
        parameters: family.parameters.to_vec(),
        unknown_domain: family.unknown_domain.to_vec(),
        parameter_domain: family.parameter_domain.to_vec(),
        root_enclosure: family.root_enclosure.to_vec(),
        solution_jacobian,
        contraction_bound,
        precision_digits: family.precision_digits,
        leaf_count: family.leaves.len(),
    };
    work.step()?;
    Ok(receipt)
}
