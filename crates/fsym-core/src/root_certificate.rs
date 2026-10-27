//! Independent bounded real root-box verification by interval automatic
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

use crate::{BallError, BigInt, BigRational, Constant, Expr, RealBall, Symbol};
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::{fmt, marker::PhantomData};
use thiserror::Error;

pub const MAX_ROOT_BOX_NODES: usize = 16_384;
pub const MAX_ROOT_BOX_DEPTH: usize = 32;
pub const MAX_ROOT_BOX_LIMBS: u64 = 1_024;
pub const MAX_ROOT_BOX_PRECISION: u32 = 128;
pub const MAX_ROOT_BOX_VARIABLES: usize = 16;
pub const MAX_ROOT_BOX_RESIDUALS: usize = 64;
pub const MAX_ROOT_BOX_JACOBIAN_ENTRIES: usize = 1024;
pub const MAX_ROOT_BOX_STEPS: u64 = 100_000_000;
pub const MAX_ROOT_BOX_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;

/// Untrusted square-system witness. The matrix is row-major and multiplies
/// the residual vector on the left. No numerical solver or matrix inverse is
/// trusted: acceptance recomputes the original source on the entire box.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RootBoxWitness {
    pub domain: Vec<RealBall>,
    pub center: Vec<BigRational>,
    pub preconditioner: Vec<Vec<BigRational>>,
    pub precision_digits: u32,
}

/// Source-bound acceptance of a unique nonsingular real root in one closed
/// box. This makes no assertion about roots outside the box. It deliberately
/// has no public fields, unchecked constructor, or Deserialize implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootBoxReceipt {
    residuals: Vec<Expr>,
    variables: Vec<Symbol>,
    witness: RootBoxWitness,
    root_enclosure: Vec<RealBall>,
    contraction_bound: BigRational,
}

impl RootBoxReceipt {
    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }
    pub fn witness(&self) -> &RootBoxWitness {
        &self.witness
    }
    pub fn root_enclosure(&self) -> &[RealBall] {
        &self.root_enclosure
    }
    pub fn contraction_bound(&self) -> &BigRational {
        &self.contraction_bound
    }
}

/// Independently checked residual and Jacobian enclosures on a closed real
/// coordinate box. Rectangular systems and point coordinates are admitted.
/// This object asserts whole-box C1 bounds, not existence of a root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualJacobianEnclosure {
    residuals: Vec<Expr>,
    variables: Vec<Symbol>,
    domain: Vec<RealBall>,
    precision_digits: u32,
    residual_values: Vec<RealBall>,
    jacobian: Vec<Vec<RealBall>>,
}

impl ResidualJacobianEnclosure {
    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }
    pub fn domain(&self) -> &[RealBall] {
        &self.domain
    }
    pub fn precision_digits(&self) -> u32 {
        self.precision_digits
    }
    pub fn residual_values(&self) -> &[RealBall] {
        &self.residual_values
    }
    pub fn jacobian(&self) -> &[Vec<RealBall>] {
        &self.jacobian
    }
}

// Decode each sequence before reserving or decoding an excess element. A
// hostile size_hint is not an allocation instruction. The scalar wrappers
// impose this check on limbs before constructing BigInts or taking a gcd.
struct Bounded<T, const N: usize>(Vec<T>);
struct RejectExtra;
impl<'de> Deserialize<'de> for RejectExtra {
    fn deserialize<D: Deserializer<'de>>(_: D) -> Result<Self, D::Error> {
        Err(D::Error::custom(
            "root-box sequence length exceeds its bound",
        ))
    }
}
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Bounded<T, N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BoundedVisitor<T, const N: usize>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for BoundedVisitor<T, N> {
            type Value = Bounded<T, N>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a sequence with at most {N} entries")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                if seq.size_hint().is_some_and(|size| size > N) {
                    return Err(A::Error::custom(
                        "root-box sequence length exceeds its bound",
                    ));
                }
                let mut values = Vec::new();
                while values.len() < N {
                    let Some(value) = seq.next_element()? else {
                        return Ok(Bounded(values));
                    };
                    values
                        .try_reserve(1)
                        .map_err(|_| A::Error::custom("root-box decoding allocation"))?;
                    values.push(value);
                }
                let _ = seq.next_element::<RejectExtra>()?;
                Ok(Bounded(values))
            }
        }
        deserializer.deserialize_seq(BoundedVisitor::<T, N>(PhantomData))
    }
}

struct BoundedInteger(BigInt);
impl<'de> Deserialize<'de> for BoundedInteger {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct IntegerVisitor;
        impl<'de> Visitor<'de> for IntegerVisitor {
            type Value = BoundedInteger;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a canonical bounded sign/u32-magnitude integer")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let sign: i8 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::custom("missing integer sign"))?;
                if !(-1..=1).contains(&sign) {
                    return Err(A::Error::custom("invalid integer sign"));
                }
                let digits = if sign == 0 {
                    seq.next_element::<Bounded<u32, 0>>()?.map(|x| x.0)
                } else {
                    seq.next_element::<Bounded<u32, { MAX_ROOT_BOX_LIMBS as usize * 2 }>>()?
                        .map(|x| x.0)
                }
                .ok_or_else(|| A::Error::custom("missing integer magnitude"))?;
                if digits.last() == Some(&0) || (sign != 0 && digits.is_empty()) {
                    return Err(A::Error::custom("noncanonical integer magnitude"));
                }
                let _ = seq.next_element::<RejectExtra>()?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(digits.len() * 4)
                    .map_err(|_| A::Error::custom("root-box decoding allocation"))?;
                for digit in digits {
                    bytes.extend_from_slice(&digit.to_le_bytes());
                }
                let value = BigInt::from_bytes_le(&bytes);
                Ok(BoundedInteger(if sign < 0 { -value } else { value }))
            }
        }
        deserializer.deserialize_tuple(2, IntegerVisitor)
    }
}

struct BoundedRational(BigRational);
impl<'de> Deserialize<'de> for BoundedRational {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (BoundedInteger(n), BoundedInteger(d)) = Deserialize::deserialize(deserializer)?;
        if !d.is_positive() || !n.gcd(&d).is_one() {
            return Err(D::Error::custom(
                "root-box rational must have positive coprime denominator",
            ));
        }
        Ok(Self(BigRational::new(n, d)))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundedBall {
    midpoint: BoundedRational,
    radius: BoundedRational,
}
impl BoundedBall {
    fn into_ball<E: serde::de::Error>(self) -> Result<RealBall, E> {
        RealBall::new(self.midpoint.0, self.radius.0).map_err(E::custom)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WitnessWire {
    domain: Bounded<BoundedBall, MAX_ROOT_BOX_VARIABLES>,
    center: Bounded<BoundedRational, MAX_ROOT_BOX_VARIABLES>,
    preconditioner:
        Bounded<Bounded<BoundedRational, MAX_ROOT_BOX_VARIABLES>, MAX_ROOT_BOX_VARIABLES>,
    precision_digits: u32,
}
impl<'de> Deserialize<'de> for RootBoxWitness {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = WitnessWire::deserialize(deserializer)?;
        Ok(Self {
            domain: wire
                .domain
                .0
                .into_iter()
                .map(BoundedBall::into_ball)
                .collect::<Result<_, D::Error>>()?,
            center: wire.center.0.into_iter().map(|x| x.0).collect(),
            preconditioner: wire
                .preconditioner
                .0
                .into_iter()
                .map(|row| row.0.into_iter().map(|x| x.0).collect())
                .collect(),
            precision_digits: wire.precision_digits,
        })
    }
}

/// Untrusted proposed box, center and rational preconditioner. Deserialization
/// does not confer verified status; the independent checker must be called.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RootBoxWitness2 {
    pub domain: [RealBall; 2],
    pub center: [BigRational; 2],
    pub preconditioner: [[BigRational; 2]; 2],
    pub precision_digits: u32,
}

impl<'de> Deserialize<'de> for RootBoxWitness2 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let witness = RootBoxWitness::deserialize(deserializer)?;
        let dimension_error = || D::Error::custom("legacy root box requires dimension two");
        let preconditioner: Vec<[BigRational; 2]> = witness
            .preconditioner
            .into_iter()
            .map(|row| row.try_into().map_err(|_| dimension_error()))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            domain: witness.domain.try_into().map_err(|_| dimension_error())?,
            center: witness.center.try_into().map_err(|_| dimension_error())?,
            preconditioner: preconditioner.try_into().map_err(|_| dimension_error())?,
            precision_digits: witness.precision_digits,
        })
    }
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
    #[error("root-box dimensions are inconsistent or exceed the supported bounds")]
    DimensionMismatch,
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
    for ball in &witness.domain {
        check_ball(ball)?;
    }
    for q in witness
        .center
        .iter()
        .chain(witness.preconditioner.iter().flatten())
    {
        check_rational(q)?;
    }
    meter.charge_batch(&[
        (Dimension::MemoryBytes, 1024 * 1024),
        (Dimension::AllocationCount, 8),
    ])?;
    let generic = RootBoxWitness {
        domain: witness.domain.to_vec(),
        center: witness.center.to_vec(),
        preconditioner: witness
            .preconditioner
            .iter()
            .map(|row| row.to_vec())
            .collect(),
        precision_digits: witness.precision_digits,
    };
    let receipt = verify_root_box_impl(residuals, variables, &generic, meter, true)?;
    let root_enclosure = receipt
        .root_enclosure
        .try_into()
        .map_err(|_| RootBoxError::DimensionMismatch)?;
    let result = RootBoxReceipt2 {
        residuals: residuals.clone(),
        variables: variables.clone(),
        witness: witness.clone(),
        root_enclosure,
        contraction_bound: receipt.contraction_bound,
    };
    meter.checkpoint()?;
    Ok(result)
}

/// Check a 1..16 dimensional square system with explicit intrinsic limits.
pub fn certify_root_box(
    residuals: &[Expr],
    variables: &[Symbol],
    witness: &RootBoxWitness,
) -> Result<RootBoxReceipt, RootBoxError> {
    verify_root_box(residuals, variables, witness, &mut Unbounded)
}

/// Independently check the original source, whole-box C1 domain, strict
/// Krawczyk inclusion and contraction. Rational preconditioner entries are
/// untrusted. The norm bound itself proves the square C and every J(x)
/// nonsingular, since ||I-CJ(x)||<1 implies invertibility by the Neumann series.
pub fn verify_root_box(
    residuals: &[Expr],
    variables: &[Symbol],
    witness: &RootBoxWitness,
    meter: &mut impl BudgetMeter,
) -> Result<RootBoxReceipt, RootBoxError> {
    verify_root_box_impl(residuals, variables, witness, meter, false)
}

fn validate_inputs(
    residuals: &[Expr],
    variables: &[Symbol],
    domain: &[RealBall],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(), RootBoxError> {
    let n = variables.len();
    if !(1..=MAX_ROOT_BOX_VARIABLES).contains(&n)
        || !(1..=MAX_ROOT_BOX_RESIDUALS).contains(&residuals.len())
        || residuals.len() * n > MAX_ROOT_BOX_JACOBIAN_ENTRIES
        || domain.len() != n
    {
        return Err(RootBoxError::DimensionMismatch);
    }
    if !(8..=MAX_ROOT_BOX_PRECISION).contains(&work.precision) {
        return Err(RootBoxError::InvalidPrecision);
    }
    for (i, variable) in variables.iter().enumerate() {
        if variable.name.len() > 4096 {
            return Err(RootBoxError::ResourceLimit("symbol length"));
        }
        if variables[..i].contains(variable) {
            return Err(RootBoxError::DuplicateVariable);
        }
    }
    for ball in domain {
        check_ball(ball)?;
    }
    preflight_expressions(residuals, work)?;
    work.charge(
        1,
        (n * n * 256) as u64 + binding_bytes(variables, domain),
        8,
    )?;
    Ok(())
}

/// Reusable complete-box enclosures for rectangular residual/Jacobian systems.
pub fn enclose_residual_jacobian(
    residuals: &[Expr],
    variables: &[Symbol],
    domain: &[RealBall],
    precision_digits: u32,
) -> Result<ResidualJacobianEnclosure, RootBoxError> {
    metered_enclose_residual_jacobian(
        residuals,
        variables,
        domain,
        precision_digits,
        &mut Unbounded,
    )
}

pub fn metered_enclose_residual_jacobian(
    residuals: &[Expr],
    variables: &[Symbol],
    domain: &[RealBall],
    precision_digits: u32,
    meter: &mut impl BudgetMeter,
) -> Result<ResidualJacobianEnclosure, RootBoxError> {
    let mut work = Work::new(meter, precision_digits)?;
    validate_inputs(residuals, variables, domain, &mut work)?;
    let mut residual_values = Vec::with_capacity(residuals.len());
    let mut jacobian = Vec::with_capacity(residuals.len());
    for residual in residuals {
        let jet = work.eval(residual, variables, domain, 0)?;
        residual_values.push(jet.value);
        jacobian.push(jet.gradient);
    }
    work.charge(1, work.source_bytes + binding_bytes(variables, domain), 4)?;
    let result = ResidualJacobianEnclosure {
        residuals: residuals.to_vec(),
        variables: variables.to_vec(),
        domain: domain.to_vec(),
        precision_digits,
        residual_values,
        jacobian,
    };
    work.step()?;
    Ok(result)
}

fn verify_root_box_impl(
    residuals: &[Expr],
    variables: &[Symbol],
    witness: &RootBoxWitness,
    meter: &mut impl BudgetMeter,
    legacy_determinant: bool,
) -> Result<RootBoxReceipt, RootBoxError> {
    let mut work = Work::new(meter, witness.precision_digits)?;
    validate_inputs(residuals, variables, &witness.domain, &mut work)?;
    let n = variables.len();
    if residuals.len() != n
        || witness.center.len() != n
        || witness.preconditioner.len() != n
        || witness.preconditioner.iter().any(|row| row.len() != n)
    {
        return Err(RootBoxError::DimensionMismatch);
    }
    for q in witness
        .center
        .iter()
        .chain(witness.preconditioner.iter().flatten())
    {
        check_rational(q)?;
    }
    for (i, (ball, center)) in witness.domain.iter().zip(&witness.center).enumerate() {
        work.arithmetic(&[ball], &[center])?;
        if ball.radius().is_zero() {
            return Err(RootBoxError::DegenerateBox { coordinate: i });
        }
        if !ball.contains(center) {
            return Err(RootBoxError::CenterOutsideBox { coordinate: i });
        }
    }
    if legacy_determinant {
        let c = &witness.preconditioner;
        work.arithmetic(&[], &[&c[0][0], &c[0][1], &c[1][0], &c[1][1]])?;
        if &c[0][0] * &c[1][1] - &c[0][1] * &c[1][0] == BigRational::zero() {
            return Err(RootBoxError::SingularPreconditioner);
        }
    }
    let mut jets = Vec::with_capacity(n);
    for residual in residuals {
        jets.push(work.eval(residual, variables, &witness.domain, 0)?);
    }
    work.charge(1, (n * n * 512) as u64, (n * n + 4) as u64)?;
    let point: Vec<_> = witness
        .center
        .iter()
        .cloned()
        .map(RealBall::exact)
        .collect();
    let mut values = Vec::with_capacity(n);
    for residual in residuals {
        values.push(work.eval(residual, variables, &point, 0)?.value);
    }
    let mut defect = vec![vec![RealBall::from_i64(0); n]; n];
    let mut norm = BigRational::zero();
    for (i, row) in defect.iter_mut().enumerate() {
        let mut row_sum = BigRational::zero();
        for (j, entry) in row.iter_mut().enumerate() {
            let mut product = RealBall::from_i64(0);
            for (coefficient, jet) in witness.preconditioner[i].iter().zip(&jets) {
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
    for (ball, center) in witness.domain.iter().zip(&point) {
        delta.push(work.sub(ball, center)?);
    }
    let mut image = point;
    for (i, coordinate) in image.iter_mut().enumerate() {
        for j in 0..n {
            let term = work.scale(&values[j], &witness.preconditioner[i][j])?;
            *coordinate = work.sub(coordinate, &term)?;
            let term = work.mul(&defect[i][j], &delta[j])?;
            *coordinate = work.add(coordinate, &term)?;
        }
        work.arithmetic(&[coordinate, &witness.domain[i]], &[])?;
        if coordinate.lower() <= witness.domain[i].lower()
            || coordinate.upper() >= witness.domain[i].upper()
        {
            return Err(RootBoxError::NoInteriorInclusion { coordinate: i });
        }
    }
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
    let result = RootBoxReceipt {
        residuals: residuals.to_vec(),
        variables: variables.to_vec(),
        witness: witness.clone(),
        root_enclosure: image,
        contraction_bound: norm,
    };
    work.step()?;
    Ok(result)
}

#[derive(Clone)]
struct Jet {
    value: RealBall,
    gradient: Vec<RealBall>,
}

impl Jet {
    fn constant(value: RealBall, dimension: usize) -> Self {
        Self {
            value,
            gradient: vec![RealBall::from_i64(0); dimension],
        }
    }
    fn checked(self) -> Result<Self, RootBoxError> {
        check_ball(&self.value)?;
        for derivative in &self.gradient {
            check_ball(derivative)?;
        }
        Ok(self)
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    nodes: usize,
    precision: u32,
    steps: u64,
    allocation_bytes: u64,
    source_bytes: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M, precision: u32) -> Result<Self, RootBoxError> {
        let mut work = Self {
            meter,
            nodes: 0,
            precision,
            steps: 0,
            allocation_bytes: 0,
            source_bytes: 0,
        };
        // This is a conservative reservation for the bounded elementary ball
        // kernels. Their fixed internal envelopes remain in force; no claim
        // of a universal cancellation latency inside one bigint call is made.
        work.charge(1, 64 * 1024 * 1024, 1)?;
        Ok(work)
    }
    fn charge(&mut self, steps: u64, bytes: u64, allocations: u64) -> Result<(), RootBoxError> {
        self.meter.checkpoint()?;
        let total_steps = self
            .steps
            .checked_add(steps)
            .filter(|n| *n <= MAX_ROOT_BOX_STEPS)
            .ok_or(RootBoxError::ResourceLimit("cumulative arithmetic work"))?;
        let total_bytes = self
            .allocation_bytes
            .checked_add(bytes)
            .filter(|n| *n <= MAX_ROOT_BOX_ALLOCATION_BYTES)
            .ok_or(RootBoxError::ResourceLimit("cumulative allocation"))?;
        self.meter.charge_batch(&[
            (Dimension::ComputeSteps, steps),
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, allocations),
        ])?;
        self.steps = total_steps;
        self.allocation_bytes = total_bytes;
        Ok(())
    }
    fn step(&mut self) -> Result<(), RootBoxError> {
        self.charge(1, 0, 0)
    }
    fn arithmetic(
        &mut self,
        balls: &[&RealBall],
        rationals: &[&BigRational],
    ) -> Result<(), RootBoxError> {
        let mut limbs = 1;
        for q in rationals.iter().copied().chain(
            balls
                .iter()
                .flat_map(|ball| [ball.midpoint(), ball.radius()]),
        ) {
            check_rational(q)?;
            limbs = limbs.max(rational_limbs(q));
        }
        // Limb-weighted conservative bounds cover rational products, sums and
        // reductions, rather than charging a constant for arbitrary widths.
        self.charge(64 * (limbs + 1).pow(2), 128 * (limbs + 1), 16)
    }
    fn elementary(&mut self, ball: &RealBall) -> Result<(), RootBoxError> {
        check_ball(ball)?;
        let width = rational_limbs(ball.midpoint()).max(rational_limbs(ball.radius()))
            + u64::from(self.precision)
            + 1;
        self.charge(64 * width.pow(2), 4096 * width, 64)
    }
    fn add(&mut self, a: &RealBall, b: &RealBall) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[a, b], &[])?;
        checked(a.add(b))
    }
    fn sub(&mut self, a: &RealBall, b: &RealBall) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[a, b], &[])?;
        checked(a.sub(b))
    }
    fn mul(&mut self, a: &RealBall, b: &RealBall) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[a, b], &[])?;
        checked(a.mul(b))
    }
    fn div(&mut self, a: &RealBall, b: &RealBall) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[a, b], &[])?;
        checked(a.div(b)?)
    }
    fn inv(&mut self, a: &RealBall) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[a], &[])?;
        checked(a.inv()?)
    }
    fn scale(&mut self, a: &RealBall, q: &BigRational) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[a], &[q])?;
        checked(a.mul(&RealBall::exact(q.clone())))
    }
    fn power(&mut self, a: &RealBall, k: i32) -> Result<RealBall, RootBoxError> {
        self.arithmetic(&[a], &[])?;
        let width = rational_limbs(a.midpoint()).max(rational_limbs(a.radius())) + 1;
        self.charge(
            (u64::from(k.unsigned_abs()) + 1) * 32 * width.pow(2),
            128 * width * (u64::from(k.unsigned_abs()) + 1),
            16,
        )?;
        checked(a.pow(k)?)
    }
    fn exp(&mut self, a: &RealBall) -> Result<RealBall, RootBoxError> {
        self.elementary(a)?;
        interval_exp(a, self.precision)
    }
    fn jet_add(&mut self, a: Jet, b: Jet) -> Result<Jet, RootBoxError> {
        let value = self.add(&a.value, &b.value)?;
        let mut gradient = Vec::with_capacity(a.gradient.len());
        for (left, right) in a.gradient.iter().zip(&b.gradient) {
            gradient.push(self.add(left, right)?);
        }
        Jet { value, gradient }.checked()
    }
    fn jet_mul(&mut self, a: Jet, b: Jet) -> Result<Jet, RootBoxError> {
        let value = self.mul(&a.value, &b.value)?;
        let mut gradient = Vec::with_capacity(a.gradient.len());
        for (left, right) in a.gradient.iter().zip(&b.gradient) {
            let first = self.mul(left, &b.value)?;
            let second = self.mul(&a.value, right)?;
            gradient.push(self.add(&first, &second)?);
        }
        Jet { value, gradient }.checked()
    }
    fn chain(&mut self, a: Jet, value: RealBall, slope: RealBall) -> Result<Jet, RootBoxError> {
        let mut gradient = Vec::with_capacity(a.gradient.len());
        for derivative in &a.gradient {
            gradient.push(self.mul(derivative, &slope)?);
        }
        Jet { value, gradient }.checked()
    }

    fn eval(
        &mut self,
        expression: &Expr,
        variables: &[Symbol],
        domain: &[RealBall],
        depth: usize,
    ) -> Result<Jet, RootBoxError> {
        self.step()?;
        if depth > MAX_ROOT_BOX_DEPTH || self.nodes >= MAX_ROOT_BOX_NODES * 2 {
            return Err(RootBoxError::ResourceLimit(
                "expression depth or node count",
            ));
        }
        self.charge(1, 512 + (variables.len() as u64) * 128, 16)?;
        self.nodes += 1;
        let digits = self.precision;
        let result = match expression {
            Expr::Integer(n) => {
                if n.limb_count() > MAX_ROOT_BOX_LIMBS {
                    return Err(RootBoxError::ResourceLimit("integer height"));
                }
                self.charge(1, 128 + n.limb_count() * 16, 2)?;
                Jet::constant(
                    RealBall::exact(BigRational::from_integer(n.clone())),
                    variables.len(),
                )
            }
            Expr::Rational(q) => {
                self.arithmetic(&[], &[q])?;
                Jet::constant(RealBall::exact(q.clone()), variables.len())
            }
            Expr::Const(Constant::Pi) => {
                self.elementary(&RealBall::from_i64(1))?;
                Jet::constant(RealBall::pi(digits)?, variables.len())
            }
            Expr::Const(Constant::E) => {
                Jet::constant(self.exp(&RealBall::from_i64(1))?, variables.len())
            }
            Expr::Const(_) => return Err(RootBoxError::Domain("non-real or non-finite constant")),
            Expr::Sym(s) => {
                if s.name.len() > 4096 {
                    return Err(RootBoxError::ResourceLimit("symbol length"));
                }
                let index = variables
                    .iter()
                    .position(|v| v == s)
                    .ok_or(RootBoxError::Unsupported("unbound symbol identity"))?;
                self.arithmetic(&[&domain[index]], &[])?;
                let mut result = Jet::constant(domain[index].clone(), variables.len());
                result.gradient[index] = RealBall::from_i64(1);
                result
            }
            Expr::Add(args) | Expr::Mul(args) => {
                if args.len() > MAX_ROOT_BOX_NODES {
                    return Err(RootBoxError::ResourceLimit("variadic expression width"));
                }
                let is_sum = matches!(expression, Expr::Add(_));
                let mut result = Jet::constant(
                    RealBall::from_i64(if is_sum { 0 } else { 1 }),
                    variables.len(),
                );
                for arg in args {
                    // Never simplify away a child's original domain obligation.
                    let next = self.eval(arg, variables, domain, depth + 1)?;
                    result = if is_sum {
                        self.jet_add(result, next)?
                    } else {
                        self.jet_mul(result, next)?
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
                    let value = self.power(&b.value, k)?;
                    let slope = match k {
                        0 => RealBall::from_i64(0),
                        1 => RealBall::from_i64(1),
                        _ => {
                            let power = self.power(&b.value, k - 1)?;
                            self.mul(&power, &RealBall::from_i64(i64::from(k)))?
                        }
                    };
                    self.chain(b, value, slope)?
                } else {
                    // Real principal power is C1 on a neighborhood of the
                    // complete box only after this strict domain obligation.
                    if !b.value.is_positive() {
                        return Err(RootBoxError::Domain(
                            "noninteger power base is not strictly positive",
                        ));
                    }
                    self.elementary(&b.value)?;
                    let logarithm = checked(b.value.ln(digits)?)?;
                    let exponent = self.mul(&e.value, &logarithm)?;
                    let value = self.exp(&exponent)?;
                    let mut gradient = vec![RealBall::from_i64(0); variables.len()];
                    for (i, derivative) in gradient.iter_mut().enumerate() {
                        let left = self.mul(&e.gradient[i], &logarithm)?;
                        let quotient = self.div(&b.gradient[i], &b.value)?;
                        let right = self.mul(&e.value, &quotient)?;
                        let chain = self.add(&left, &right)?;
                        *derivative = self.mul(&value, &chain)?;
                    }
                    Jet { value, gradient }
                }
            }
            Expr::Function(name, args) => {
                let [arg] = args.as_slice() else {
                    return Err(RootBoxError::Unsupported("non-unary function"));
                };
                let a = self.eval(arg, variables, domain, depth + 1)?;
                self.elementary(&a.value)?;
                let (value, slope) = match name.as_str() {
                    "sin" => (a.value.sin(digits)?, a.value.cos(digits)?),
                    "cos" => (a.value.cos(digits)?, a.value.sin(digits)?.neg()),
                    "exp" => {
                        let value = self.exp(&a.value)?;
                        (value.clone(), value)
                    }
                    "log" | "ln" => {
                        if !a.value.is_positive() {
                            return Err(RootBoxError::Domain(
                                "logarithm argument is not strictly positive",
                            ));
                        }
                        (a.value.ln(digits)?, self.inv(&a.value)?)
                    }
                    "sqrt" => {
                        if !a.value.is_positive() {
                            return Err(RootBoxError::Domain("square root is not C1 on this box"));
                        }
                        let value = a.value.sqrt(digits.saturating_mul(4).saturating_add(64))?;
                        let twice = self.mul(&value, &RealBall::from_i64(2))?;
                        let slope = self.inv(&twice)?;
                        (value, slope)
                    }
                    "atan" => {
                        let square = self.power(&a.value, 2)?;
                        let denominator = self.add(&RealBall::from_i64(1), &square)?;
                        let slope = self.inv(&denominator)?;
                        (a.value.atan(digits)?, slope)
                    }
                    "tan" => {
                        let cosine = a.value.cos(digits)?;
                        let inverse = self.inv(&cosine)?;
                        (
                            self.mul(&a.value.sin(digits)?, &inverse)?,
                            self.power(&inverse, 2)?,
                        )
                    }
                    "sinh" | "cosh" | "tanh" => {
                        let plus = self.exp(&a.value)?;
                        let minus = self.exp(&a.value.neg())?;
                        let difference = self.sub(&plus, &minus)?;
                        let sum = self.add(&plus, &minus)?;
                        let sinh = self.div(&difference, &RealBall::from_i64(2))?;
                        let cosh = self.div(&sum, &RealBall::from_i64(2))?;
                        match name.as_str() {
                            "sinh" => (sinh, cosh),
                            "cosh" => (cosh, sinh),
                            _ => {
                                let inverse = self.inv(&cosh)?;
                                (self.div(&sinh, &cosh)?, self.power(&inverse, 2)?)
                            }
                        }
                    }
                    _ => {
                        return Err(RootBoxError::Unsupported(
                            "function has no C1 interval rule",
                        ));
                    }
                };
                self.chain(a, value, slope)?
            }
        };
        result.checked()
    }
}

/// Check shape iteratively before the recursive interval evaluator can consume
/// stack space. The admitted depth is tested on the normal thread stack.
fn preflight_expressions(
    residuals: &[Expr],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(), RootBoxError> {
    work.charge(
        1,
        (MAX_ROOT_BOX_NODES * std::mem::size_of::<(&Expr, usize)>()) as u64,
        1,
    )?;
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(MAX_ROOT_BOX_NODES)
        .map_err(|_| RootBoxError::ResourceLimit("preflight allocation"))?;
    stack.extend(residuals.iter().map(|expression| (expression, 0usize)));
    let mut visited = 0usize;
    while let Some((expression, depth)) = stack.pop() {
        work.step()?;
        if depth > MAX_ROOT_BOX_DEPTH || visited >= MAX_ROOT_BOX_NODES {
            return Err(RootBoxError::ResourceLimit(
                "expression depth or node count",
            ));
        }
        visited += 1;
        // Scalar/identity bounds precede any cloned source or arithmetic.
        let bytes = match expression {
            Expr::Integer(integer) => {
                if integer.limb_count() > MAX_ROOT_BOX_LIMBS {
                    return Err(RootBoxError::ResourceLimit("integer height"));
                }
                128 + 16 * integer.limb_count()
            }
            Expr::Rational(rational) => {
                check_rational(rational)?;
                128 + 16 * rational_limbs(rational)
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > 4096 {
                    return Err(RootBoxError::ResourceLimit("symbol length"));
                }
                128 + symbol.name.len() as u64
            }
            Expr::Function(name, _) => {
                if name.len() > 4096 {
                    return Err(RootBoxError::ResourceLimit("function name length"));
                }
                128 + name.len() as u64
            }
            _ => 128,
        };
        work.charge(1, bytes, 1)?;
        work.source_bytes += bytes;
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

fn rational_limbs(value: &BigRational) -> u64 {
    value.numer().limb_count() + value.denom().limb_count()
}

fn binding_bytes(variables: &[Symbol], domain: &[RealBall]) -> u64 {
    variables
        .iter()
        .map(|v| 256 + 2 * v.name.len() as u64)
        .sum::<u64>()
        + domain
            .iter()
            .map(|b| 256 + 16 * (rational_limbs(b.midpoint()) + rational_limbs(b.radius())))
            .sum::<u64>()
}

fn check_ball(value: &RealBall) -> Result<(), RootBoxError> {
    check_rational(value.midpoint())?;
    check_rational(value.radius())
}

fn checked(value: RealBall) -> Result<RealBall, RootBoxError> {
    check_ball(&value)?;
    Ok(value)
}
