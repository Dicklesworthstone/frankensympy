//! Exact comparison of two independently indexed real algebraic roots.
//!
//! Equality is a common root of the two original polynomials in the
//! intersection of their checked isolating intervals. Strict ordering is a
//! pair of disjoint rational intervals. The reference verifier checks both
//! complete original root sets, all multiplicities and the requested indices
//! before accepting either claim; it does not rerun the refinement search.
//!
//! The polynomials need not be square-free or irreducible, and may differ.
//! There is no floating approximation, resultant, or minimal-polynomial claim.

use super::*;
use fsym_budget::BudgetError;
use std::cmp::Ordering;

/// Keep both intrinsic limits cumulative across all nested isolation,
/// refinement and verification calls, even with an unbounded caller meter.
struct Capped<'a, M> {
    inner: &'a mut M,
    compute: u64,
    memory: u64,
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        let (mut compute, mut memory) = (self.compute, self.memory);
        for &(dimension, amount) in charges {
            let (used, cap) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_WORK),
                Dimension::MemoryBytes => (&mut memory, MAX_ALLOCATION),
                _ => continue,
            };
            let remaining = cap.saturating_sub(*used);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *used += amount;
        }
        self.inner.charge_batch(charges)?;
        self.compute = compute;
        self.memory = memory;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.inner.checkpoint()
    }
}

/// An untrusted comparison proposal. Input polynomials and indices are
/// explicit verifier arguments, rather than mutable identity assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealRootComparisonCertificate {
    pub left_roots: RealRootIsolationCertificate,
    pub right_roots: RealRootIsolationCertificate,
    pub ordering: Ordering,
}

/// An independently checked order, bound to both original coefficient lists
/// and distinct-real-root indices. There is no unchecked constructor or
/// deserializer. Scaling, repeated factors and trailing zero coefficients
/// remain visible in the retained original inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRealRootComparison {
    left_coefficients: Poly,
    left_index: usize,
    right_coefficients: Poly,
    right_index: usize,
    certificate: RealRootComparisonCertificate,
}

impl VerifiedRealRootComparison {
    pub fn ordering(&self) -> Ordering {
        self.certificate.ordering
    }
    pub fn left_coefficients(&self) -> &[BigRational] {
        &self.left_coefficients
    }
    pub fn left_index(&self) -> usize {
        self.left_index
    }
    pub fn right_coefficients(&self) -> &[BigRational] {
        &self.right_coefficients
    }
    pub fn right_index(&self) -> usize {
        self.right_index
    }
    pub fn left_interval(&self) -> &RealRootInterval {
        &self.certificate.left_roots.intervals[self.left_index]
    }
    pub fn right_interval(&self) -> &RealRootInterval {
        &self.certificate.right_roots.intervals[self.right_index]
    }
    pub fn certificate(&self) -> &RealRootComparisonCertificate {
        &self.certificate
    }
}

fn selected(roots: &VerifiedRealRoots, index: usize) -> Result<&RealRootInterval, RealRootError> {
    roots
        .intervals()
        .get(index)
        .ok_or(RealRootError::InvalidIndex)
}

fn separated<M: BudgetMeter>(
    left: &RealRootInterval,
    right: &RealRootInterval,
    work: &mut Work<'_, M>,
) -> Result<Option<Ordering>, RealRootError> {
    work.binary_preflight(&left.upper, &right.lower)?;
    if left.upper < right.lower {
        return Ok(Some(Ordering::Less));
    }
    work.binary_preflight(&left.lower, &right.upper)?;
    if left.lower > right.upper {
        return Ok(Some(Ordering::Greater));
    }
    Ok(None)
}

fn common_root<M: BudgetMeter>(
    left: &VerifiedRealRoots,
    left_index: usize,
    right: &VerifiedRealRoots,
    right_index: usize,
    work: &mut Work<'_, M>,
) -> Result<bool, RealRootError> {
    let a = selected(left, left_index)?;
    let b = selected(right, right_index)?;
    work.binary_preflight(&a.lower, &b.lower)?;
    let lower = if a.lower >= b.lower {
        &a.lower
    } else {
        &b.lower
    };
    work.binary_preflight(&a.upper, &b.upper)?;
    let upper = if a.upper <= b.upper {
        &a.upper
    } else {
        &b.upper
    };
    work.binary_preflight(lower, upper)?;
    if lower > upper {
        return Ok(false);
    }
    let common = work.gcd(left.coefficients(), right.coefficients())?;
    if common.len() == 1 {
        return Ok(false);
    }
    let square_free = work.square_free(&common)?;
    let sequence = work.sturm(&square_free)?;
    match work.count(&sequence, lower, upper)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(RealRootError::Arithmetic(
            "common-root intersection lost uniqueness",
        )),
    }
}

/// Independently check a root comparison without refining or searching.
///
/// Both complete isolations are checked against the supplied original
/// polynomials. Equality additionally requires a gcd root in the intersection
/// of the two selected intervals; merely belonging to the same gcd is not
/// enough. Strict comparisons require separated closed intervals.
pub fn verify_real_root_comparison(
    left_coefficients: &[BigRational],
    left_index: usize,
    right_coefficients: &[BigRational],
    right_index: usize,
    certificate: &RealRootComparisonCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRootComparison, RealRootError> {
    let mut capped = Capped {
        inner: meter,
        compute: 0,
        memory: 0,
    };
    let left = verify_real_root_isolation(left_coefficients, &certificate.left_roots, &mut capped)?;
    let right =
        verify_real_root_isolation(right_coefficients, &certificate.right_roots, &mut capped)?;
    let a = selected(&left, left_index)?;
    let b = selected(&right, right_index)?;
    let mut work = Work::new(&mut capped)?;
    let accepted = match certificate.ordering {
        Ordering::Equal => common_root(&left, left_index, &right, right_index, &mut work)?,
        ordering => separated(a, b, &mut work)? == Some(ordering),
    };
    if !accepted {
        return Err(RealRootError::InvalidCertificate(
            "selected intervals do not prove the claimed comparison",
        ));
    }

    // Retain the charged copies of the complete original inputs. Move the
    // already checked certificates instead of cloning their intervals again.
    let left_coefficients = work.copy_poly(left_coefficients)?;
    let right_coefficients = work.copy_poly(right_coefficients)?;
    let result = VerifiedRealRootComparison {
        left_coefficients,
        left_index,
        right_coefficients,
        right_index,
        certificate: RealRootComparisonCertificate {
            left_roots: left.certificate,
            right_roots: right.certificate,
            ordering: certificate.ordering,
        },
    };
    work.meter.checkpoint()?;
    Ok(result)
}

/// Compare two distinct-real-root indices of rational polynomials exactly.
///
/// Each polynomial has degree at most 64 and input coefficients of at most
/// 4096 bits. Both caller lanes have cumulative intrinsic limits of 100 million
/// compute steps and 256 MiB allocation, including every nested helper. At most
/// 32768 additional bisections are requested across the two selected roots.
/// Exhaustion, cancellation, invalid indices and absent real roots refuse;
/// none of them becomes an approximate order or an equality decision.
pub fn compare_real_roots(
    left_coefficients: &[BigRational],
    left_index: usize,
    right_coefficients: &[BigRational],
    right_index: usize,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRootComparison, RealRootError> {
    let mut generator_cap = Capped {
        inner: generator_meter,
        compute: 0,
        memory: 0,
    };
    let mut verifier_cap = Capped {
        inner: verifier_meter,
        compute: 0,
        memory: 0,
    };
    let mut generator = Work::new(&mut generator_cap)?;
    let mut left = isolate_real_roots(left_coefficients, generator.meter, &mut verifier_cap)?;
    selected(&left, left_index)?;
    let mut right = isolate_real_roots(right_coefficients, generator.meter, &mut verifier_cap)?;
    selected(&right, right_index)?;

    let initial_order = separated(
        selected(&left, left_index)?,
        selected(&right, right_index)?,
        &mut generator,
    )?;
    let ordering = if let Some(ordering) = initial_order {
        ordering
    } else if common_root(&left, left_index, &right, right_index, &mut generator)? {
        Ordering::Equal
    } else {
        let mut bisections = 0usize;
        let mut batch = 32usize;
        loop {
            let mut refined = false;
            for (roots, index) in [(&mut left, left_index), (&mut right, right_index)] {
                let root = selected(roots, index)?;
                generator.binary_preflight(&root.lower, &root.upper)?;
                if root.lower != root.upper {
                    let steps = batch.min(MAX_BISECTIONS - bisections);
                    if steps == 0 {
                        return Err(RealRootError::ResourceLimit("comparison bisections"));
                    }
                    *roots = refine_real_root_isolation(
                        roots,
                        index,
                        steps,
                        generator.meter,
                        &mut verifier_cap,
                    )?;
                    bisections += steps;
                    refined = true;
                }
            }
            if let Some(ordering) = separated(
                selected(&left, left_index)?,
                selected(&right, right_index)?,
                &mut generator,
            )? {
                break ordering;
            }
            if !refined {
                return Err(RealRootError::Arithmetic(
                    "distinct exact point roots were not separated",
                ));
            }
            batch = (batch * 2).min(1024);
        }
    };

    let certificate = RealRootComparisonCertificate {
        left_roots: left.certificate,
        right_roots: right.certificate,
        ordering,
    };
    let result = verify_real_root_comparison(
        left_coefficients,
        left_index,
        right_coefficients,
        right_index,
        &certificate,
        &mut verifier_cap,
    )?;
    generator.meter.checkpoint()?;
    verifier_cap.checkpoint()?;
    Ok(result)
}
