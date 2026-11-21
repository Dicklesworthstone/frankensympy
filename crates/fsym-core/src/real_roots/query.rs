//! Closed-interval queries on exact real roots.
//!
//! The generator selects and refines candidates from a complete isolation.
//! The checker independently counts the original polynomial's distinct roots
//! in the requested domain, checks each interval and derives multiplicities.
//! A restricted query does not claim completeness over the entire real line.

use super::*;

/// Exact domain and optional precision obligation of a real-root query.
/// Missing endpoints denote negative/positive infinity respectively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealRootQuery {
    pub lower: Option<BigRational>,
    pub upper: Option<BigRational>,
    pub max_width: Option<BigRational>,
    pub require_square_free: bool,
}

/// Complete distinct roots in the stated closed domain, with multiplicity
/// in the original polynomial. Only the reference checker constructs this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRealRootQuery {
    coefficients: Poly,
    query: RealRootQuery,
    certificate: RealRootIsolationCertificate,
}

impl VerifiedRealRootQuery {
    pub fn coefficients(&self) -> &[BigRational] {
        &self.coefficients
    }

    pub fn query(&self) -> &RealRootQuery {
        &self.query
    }

    pub fn intervals(&self) -> &[RealRootInterval] {
        &self.certificate.intervals
    }

    pub fn certificate(&self) -> &RealRootIsolationCertificate {
        &self.certificate
    }

    pub fn distinct_count(&self) -> usize {
        self.intervals().len()
    }
}

fn admit_query<M: BudgetMeter>(
    p: &[BigRational],
    query: &RealRootQuery,
    work: &mut Work<'_, M>,
) -> Result<(), RealRootError> {
    for value in [&query.lower, &query.upper, &query.max_width]
        .into_iter()
        .flatten()
    {
        if bits(value) > MAX_REAL_ROOT_INPUT_BITS {
            return Err(RealRootError::ResourceLimit("root-query rational bits"));
        }
        work.scalar(value)?;
    }
    if let (Some(lower), Some(upper)) = (&query.lower, &query.upper)
        && lower > upper
    {
        return Err(RealRootError::InvalidCertificate("reversed query interval"));
    }
    if query
        .max_width
        .as_ref()
        .is_some_and(|width| !width.is_positive())
    {
        return Err(RealRootError::InvalidCertificate(
            "root-query precision must be positive",
        ));
    }
    if query.require_square_free && work.square_free(p)?.len() != p.len() {
        return Err(RealRootError::InvalidCertificate(
            "root query requires a square-free polynomial",
        ));
    }
    Ok(())
}

fn bounded_domain<M: BudgetMeter>(
    p: &[BigRational],
    query: &RealRootQuery,
    work: &mut Work<'_, M>,
) -> Result<(BigRational, BigRational), RealRootError> {
    // A strict Cauchy bound contains every root. Clipping also avoids
    // evaluating the polynomial at irrelevant enormous endpoints.
    let bound = work.bound(p)?;
    let lower = query.lower.clone().unwrap_or_else(|| -bound.clone());
    let upper = query.upper.clone().unwrap_or_else(|| bound.clone());
    Ok((lower.max(-bound.clone()), upper.min(bound)))
}

/// Independently check completeness, closed endpoints, original multiplicity,
/// square-freeness when requested, and the exact rational precision bound.
/// This checker does not call the interval generator or refinement search.
pub fn verify_real_root_query(
    coefficients: &[BigRational],
    query: &RealRootQuery,
    certificate: &RealRootIsolationCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRootQuery, RealRootError> {
    let mut work = Work::new(meter)?;
    let p = work.input(coefficients)?;
    admit_query(&p, query, &mut work)?;
    if certificate.intervals.len() > p.len() - 1 {
        return Err(RealRootError::InvalidCertificate("too many query roots"));
    }
    let layers = work.layers(&p)?;
    let (lower, upper) = bounded_domain(&p, query, &mut work)?;
    let count = if layers.is_empty() || lower > upper {
        0
    } else {
        work.count(&layers[0], &lower, &upper)?
    };
    if count != certificate.intervals.len() {
        return Err(RealRootError::InvalidCertificate(
            "incomplete distinct roots in the requested interval",
        ));
    }
    let mut previous: Option<&RealRootInterval> = None;
    for interval in &certificate.intervals {
        work.scalar(&interval.lower)?;
        work.scalar(&interval.upper)?;
        if query.lower.as_ref().is_some_and(|lo| &interval.lower < lo)
            || query.upper.as_ref().is_some_and(|hi| &interval.upper > hi)
        {
            return Err(RealRootError::InvalidCertificate(
                "root interval leaves the requested domain",
            ));
        }
        if let Some(width) = &query.max_width
            && work.sub(&interval.upper, &interval.lower)? > *width
        {
            return Err(RealRootError::InvalidCertificate(
                "root interval misses the requested precision",
            ));
        }
        if let Some(before) = previous
            && (before.upper > interval.lower
                || (before.upper == interval.lower
                    && work.eval(&layers[0][0], &interval.lower)?.is_zero()))
        {
            return Err(RealRootError::InvalidCertificate(
                "overlapping or unsorted query roots",
            ));
        }
        if work.count(&layers[0], &interval.lower, &interval.upper)? != 1 {
            return Err(RealRootError::InvalidCertificate(
                "query interval must contain exactly one distinct root",
            ));
        }
        let mut multiplicity = 0;
        for sequence in &layers {
            multiplicity += work.count(sequence, &interval.lower, &interval.upper)?;
        }
        if interval.multiplicity != multiplicity {
            return Err(RealRootError::InvalidCertificate(
                "wrong original query-root multiplicity",
            ));
        }
        previous = Some(interval);
    }
    work.charge(1, (certificate.intervals.len() as u64 + 1) * 96)?;
    work.meter.checkpoint()?;
    Ok(VerifiedRealRootQuery {
        coefficients: p,
        query: query.clone(),
        certificate: certificate.clone(),
    })
}

fn collapse_endpoint<M: BudgetMeter>(
    root: &mut RealRootInterval,
    sequence: &[Poly],
    work: &mut Work<'_, M>,
) -> Result<(), RealRootError> {
    if work.eval(&sequence[0], &root.lower)?.is_zero() {
        root.upper = root.lower.clone();
    } else if work.eval(&sequence[0], &root.upper)?.is_zero() {
        root.lower = root.upper.clone();
    }
    Ok(())
}

/// Select a complete closed-domain subset of a checked global isolation, then
/// independently verify the restricted claim. Both phase meters are cumulative
/// through isolation, clipping, refinement, and final acceptance.
pub fn isolate_real_roots_in_interval(
    coefficients: &[BigRational],
    query: &RealRootQuery,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRootQuery, RealRootError> {
    let mut work = Work::new(generator_meter)?;
    let p = work.input(coefficients)?;
    admit_query(&p, query, &mut work)?;
    // Keep the intrinsic generation envelope cumulative as well as the
    // caller meter; selecting a smaller domain cannot restart either one.
    let global = generate(&p, &mut work)?;
    let roots = verify_real_root_isolation(&p, &global, verifier_meter)?;
    let mut certificate = RealRootIsolationCertificate {
        intervals: Vec::new(),
    };
    if !roots.intervals().is_empty() {
        let square_free = work.square_free(roots.coefficients())?;
        let sequence = work.sturm(&square_free)?;
        let (lower, upper) = bounded_domain(roots.coefficients(), query, &mut work)?;
        let mut bisections = 0;
        for root in roots.intervals() {
            work.charge(1, 96)?;
            let lo = root.lower.clone().max(lower.clone());
            let hi = root.upper.clone().min(upper.clone());
            if lo > hi || work.count(&sequence, &lo, &hi)? == 0 {
                continue;
            }
            let mut root = RealRootInterval {
                lower: lo,
                upper: hi,
                multiplicity: root.multiplicity,
            };
            collapse_endpoint(&mut root, &sequence, &mut work)?;
            if let Some(width) = &query.max_width {
                while work.sub(&root.upper, &root.lower)? > *width {
                    bisections += 1;
                    if bisections > MAX_BISECTIONS {
                        return Err(RealRootError::ResourceLimit("query refinement bisections"));
                    }
                    refine_once(&mut root, &sequence, &mut work)?;
                }
            }
            certificate.intervals.push(root);
        }
    }
    let result = verify_real_root_query(coefficients, query, &certificate, verifier_meter)?;
    work.meter.checkpoint()?;
    verifier_meter.checkpoint()?;
    Ok(result)
}

/// Refine a rational interval containing exactly one distinct real root.
/// With an explicit step cap, precision is a stopping target, not a promise
/// to exceed that cap. Without one, a requested width must be met or the
/// whole request fails. No partial interval is published on exhaustion.
pub fn refine_real_root_interval(
    coefficients: &[BigRational],
    query: &RealRootQuery,
    steps: Option<usize>,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRealRootQuery, RealRootError> {
    let mut work = Work::new(generator_meter)?;
    let p = work.input(coefficients)?;
    admit_query(&p, query, &mut work)?;
    let (Some(lower), Some(upper)) = (&query.lower, &query.upper) else {
        return Err(RealRootError::InvalidCertificate(
            "root refinement requires two finite rational endpoints",
        ));
    };
    let limit = steps.unwrap_or(if query.max_width.is_some() {
        MAX_BISECTIONS
    } else {
        1
    });
    if limit > MAX_BISECTIONS {
        return Err(RealRootError::ResourceLimit("refinement bisections"));
    }
    let layers = work.layers(&p)?;
    if layers.is_empty() || work.count(&layers[0], lower, upper)? != 1 {
        return Err(RealRootError::InvalidCertificate(
            "refinement interval must contain exactly one distinct root",
        ));
    }
    let mut multiplicity = 0;
    for sequence in &layers {
        multiplicity += work.count(sequence, lower, upper)?;
    }
    let mut root = RealRootInterval {
        lower: lower.clone(),
        upper: upper.clone(),
        multiplicity,
    };
    collapse_endpoint(&mut root, &layers[0], &mut work)?;
    for _ in 0..limit {
        if root.lower == root.upper {
            break;
        }
        if let Some(width) = &query.max_width
            && work.sub(&root.upper, &root.lower)? <= *width
        {
            break;
        }
        refine_once(&mut root, &layers[0], &mut work)?;
    }
    let mut accepted_query = query.clone();
    if steps.is_some() {
        accepted_query.max_width = None;
    } else if let Some(width) = &query.max_width
        && work.sub(&root.upper, &root.lower)? > *width
    {
        return Err(RealRootError::ResourceLimit("refinement precision"));
    }
    let certificate = RealRootIsolationCertificate {
        intervals: vec![root],
    };
    let result = verify_real_root_query(coefficients, &accepted_query, &certificate, verifier_meter)?;
    work.meter.checkpoint()?;
    verifier_meter.checkpoint()?;
    Ok(result)
}
