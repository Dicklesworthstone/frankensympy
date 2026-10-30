//! Exact rational functions at one indexed real algebraic embedding.
//!
//! Original sources use the same bounded trusted rational recurrence as the
//! real-equation lane. Every original pole is retained before cancellation.
//! A candidate generator reduces modulo the square-free defining polynomial
//! and refines a checked root interval. A separate reference acceptance lane
//! recomputes the original source, checks complete root isolation and every
//! original guard, and evaluates a term-wise rational interval or checks an
//! exact rational value by gcd/Sturm membership. A rounded decimal is accepted
//! only when both accepted enclosure endpoints round identically.
//!
//! This does not admit several independent algebraic generators, nonreal
//! embeddings, transcendental expressions, or a portable source-reduction
//! certificate. Both caller meters remain cumulative across all refinements.

#![forbid(unsafe_code)]

use super::*;
use fsym_budget::BudgetError;
use fsym_core::real_roots::{
    MAX_REAL_ROOT_DIGITS, RealRootIsolationCertificate, metered_round_rational,
    real_root_polynomial_zero, refine_real_root_isolation, verify_real_root_isolation,
};

const MAX_REFINEMENT_BATCHES: usize = 512;
const MAX_REFINEMENT_STEPS: usize = 32_768;

/// Keep intrinsic limits cumulative across nested core refinement and
/// reference checks, including when the caller supplies `Unbounded`.
struct Capped<'a, M> {
    inner: &'a mut M,
    compute: u64,
    memory: u64,
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
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

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.inner.checkpoint()
    }
}

/// An untrusted proposal: constructing or deserializing its fields does not
/// create acceptance. The source, root index and precision are explicit inputs
/// to the verifier, rather than mutable fields that can rebind a checked claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraicValueCertificate {
    pub roots: RealRootIsolationCertificate,
    pub lower: BigRational,
    pub upper: BigRational,
    pub exact_value: Option<BigRational>,
    pub decimal: String,
}

/// A source-bound exact enclosure and correctly rounded decimal value.
/// The decimal is a rounded approximation, not itself an enclosing interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAlgebraicValue {
    expression: Expr,
    variable: Symbol,
    defining_polynomial: Poly,
    index: usize,
    precision: u32,
    certificate: AlgebraicValueCertificate,
}

impl VerifiedAlgebraicValue {
    pub fn expression(&self) -> &Expr {
        &self.expression
    }
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn defining_polynomial(&self) -> &[BigRational] {
        &self.defining_polynomial
    }
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn precision(&self) -> u32 {
        self.precision
    }
    pub fn lower(&self) -> &BigRational {
        &self.certificate.lower
    }
    pub fn upper(&self) -> &BigRational {
        &self.certificate.upper
    }
    pub fn decimal(&self) -> &str {
        &self.certificate.decimal
    }
    pub fn certificate(&self) -> &AlgebraicValueCertificate {
        &self.certificate
    }
}

#[derive(Debug)]
struct Interval {
    lower: BigRational,
    upper: BigRational,
}

fn point<M: BudgetMeter>(value: &BigRational, work: &mut Work<'_, M>) -> Result<Interval> {
    work.scalar(value)?;
    work.scalar(value)?;
    Ok(Interval {
        lower: value.clone(),
        upper: value.clone(),
    })
}

fn interval_add<M: BudgetMeter>(
    a: &Interval,
    b: &Interval,
    work: &mut Work<'_, M>,
) -> Result<Interval> {
    Ok(Interval {
        lower: work.add(&a.lower, &b.lower)?,
        upper: work.add(&a.upper, &b.upper)?,
    })
}

fn interval_mul<M: BudgetMeter>(
    a: &Interval,
    b: &Interval,
    work: &mut Work<'_, M>,
) -> Result<Interval> {
    let products = [
        work.mul(&a.lower, &b.lower)?,
        work.mul(&a.lower, &b.upper)?,
        work.mul(&a.upper, &b.lower)?,
        work.mul(&a.upper, &b.upper)?,
    ];
    let lower = products.iter().min().unwrap();
    let upper = products.iter().max().unwrap();
    work.scalar(lower)?;
    work.scalar(upper)?;
    Ok(Interval {
        lower: lower.clone(),
        upper: upper.clone(),
    })
}

fn interval_div<M: BudgetMeter>(
    numerator: &Interval,
    denominator: &Interval,
    work: &mut Work<'_, M>,
) -> Result<Option<Interval>> {
    if denominator.lower <= BigRational::zero() && denominator.upper >= BigRational::zero() {
        return Ok(None);
    }
    let quotients = [
        work.div(&numerator.lower, &denominator.lower)?,
        work.div(&numerator.lower, &denominator.upper)?,
        work.div(&numerator.upper, &denominator.lower)?,
        work.div(&numerator.upper, &denominator.upper)?,
    ];
    let lower = quotients.iter().min().unwrap();
    let upper = quotients.iter().max().unwrap();
    work.scalar(lower)?;
    work.scalar(upper)?;
    Ok(Some(Interval {
        lower: lower.clone(),
        upper: upper.clone(),
    }))
}

fn scalar_power<M: BudgetMeter>(
    value: &BigRational,
    exponent: usize,
    work: &mut Work<'_, M>,
) -> Result<BigRational> {
    work.scalar(value)?;
    let mut base = value.clone();
    let mut result = BigRational::one();
    let mut exponent = exponent;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = work.mul(&result, &base)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = work.mul(&base, &base)?;
        }
    }
    Ok(result)
}

fn interval_power<M: BudgetMeter>(
    value: &Interval,
    exponent: usize,
    work: &mut Work<'_, M>,
) -> Result<Interval> {
    if exponent == 0 {
        return point(&BigRational::one(), work);
    }
    let a = scalar_power(&value.lower, exponent, work)?;
    let b = scalar_power(&value.upper, exponent, work)?;
    if exponent & 1 != 0 {
        return Ok(Interval { lower: a, upper: b });
    }
    let crosses_zero = value.lower <= BigRational::zero() && value.upper >= BigRational::zero();
    let lower = if crosses_zero {
        BigRational::zero()
    } else {
        work.scalar(&a)?;
        work.scalar(&b)?;
        a.clone().min(b.clone())
    };
    Ok(Interval {
        lower,
        upper: a.max(b),
    })
}

/// Generator range computation uses Horner's recurrence.
fn horner<M: BudgetMeter>(
    polynomial: &[BigRational],
    variable: &Interval,
    work: &mut Work<'_, M>,
) -> Result<Interval> {
    let mut value = point(&BigRational::zero(), work)?;
    for coefficient in polynomial.iter().rev() {
        let product = interval_mul(&value, variable, work)?;
        let constant = point(coefficient, work)?;
        value = interval_add(&product, &constant, work)?;
    }
    Ok(value)
}

/// Independent reference range: sum separately enclosed monomials. Even
/// powers crossing zero use their true nonnegative range, not Horner's tree.
fn reference_polynomial<M: BudgetMeter>(
    polynomial: &[BigRational],
    variable: &Interval,
    work: &mut Work<'_, M>,
) -> Result<Interval> {
    let mut sum = point(&BigRational::zero(), work)?;
    for (degree, coefficient) in polynomial.iter().enumerate() {
        work.charge(1, 1)?;
        if coefficient.is_zero() {
            continue;
        }
        let power = interval_power(variable, degree, work)?;
        let constant = point(coefficient, work)?;
        let term = interval_mul(&constant, &power, work)?;
        sum = interval_add(&sum, &term, work)?;
    }
    Ok(sum)
}

fn root_interval<M: BudgetMeter>(
    roots: &VerifiedRealRoots,
    index: usize,
    work: &mut Work<'_, M>,
) -> Result<Interval> {
    let root = roots
        .intervals()
        .get(index)
        .ok_or(RealRootError::InvalidIndex)?;
    work.scalar(&root.lower)?;
    work.scalar(&root.upper)?;
    Ok(Interval {
        lower: root.lower.clone(),
        upper: root.upper.clone(),
    })
}

/// The original conversion already retains every inverse's numerator. Add
/// the nonzero-base obligations of held zeroth powers as well: an undefined
/// child or 0**0 must not be erased by an outer zero product or exponent.
fn source_fraction<M: BudgetMeter>(
    expression: &Expr,
    variable: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<(Fraction, Vec<Poly>)> {
    let mut guards = Vec::new();
    let fraction = convert(expression, variable, &mut guards, work)?;
    work.charge(1, 32)?;
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        work.charge(1, 32)?;
        match node {
            Expr::Add(children) | Expr::Mul(children) => {
                work.charge(1, children.len() as u64 * 16)?;
                pending.extend(children);
            }
            Expr::Pow(base, exponent) => {
                if matches!(exponent.as_ref(), Expr::Integer(n) if n.is_zero()) {
                    if guards.len() == MAX_GUARDS {
                        return Err(limit("original algebraic domain guards"));
                    }
                    let part = convert(base, variable, &mut Vec::new(), work)?;
                    guards.push(part.numerator);
                }
                pending.push(base);
            }
            _ => {}
        }
    }
    work.charge(1, 32)?;
    guards.push(work.copy(&fraction.denominator)?);
    Ok((fraction, guards))
}

fn check_domain<M: BudgetMeter>(
    roots: &VerifiedRealRoots,
    index: usize,
    guards: &[Poly],
    work: &mut Work<'_, M>,
) -> Result<()> {
    for guard in guards {
        work.charge(1, 1)?;
        if real_root_polynomial_zero(roots, index, guard, work.meter)? {
            return Err(invalid("algebraic value is undefined at the selected root"));
        }
    }
    Ok(())
}

fn checked_remainder<G: BudgetMeter, V: BudgetMeter>(
    polynomial: &[BigRational],
    modulus: &[BigRational],
    generator: &mut Work<'_, G>,
    verifier: &mut Work<'_, V>,
) -> Result<Poly> {
    let (quotient, remainder) = generator.division(polynomial, modulus)?;
    if !zero(&remainder) && remainder.len() >= modulus.len() {
        return Err(invalid("algebraic remainder degree"));
    }
    let product = verifier.product(&quotient, modulus)?;
    if verifier.sum(&product, &remainder, false)? != polynomial {
        return Err(invalid("algebraic remainder identity"));
    }
    Ok(remainder)
}

fn rounded<M: BudgetMeter>(
    value: &BigRational,
    digits: u32,
    work: &mut Work<'_, M>,
) -> Result<(String, i64)> {
    Ok(metered_round_rational(value, digits, work.meter)?)
}

fn decimal_text(rounded: &(String, i64)) -> String {
    if rounded.0 == "0" {
        rounded.0.clone()
    } else {
        format!("{}e{}", rounded.0, rounded.1)
    }
}

/// Decode only a bounded decimal emitted by the exact rounding primitive.
/// This is numeric serialization, never a printed symbolic expression.
fn rounded_rational<M: BudgetMeter>(
    rounded: &(String, i64),
    work: &mut Work<'_, M>,
) -> Result<BigRational> {
    if rounded.0 == "0" {
        return Ok(BigRational::zero());
    }
    let fractional = rounded.0.split_once('.').map_or(0, |(_, tail)| tail.len());
    let exponent = rounded.1 - fractional as i64;
    let power = u32::try_from(exponent.unsigned_abs()).map_err(|_| limit("decimal exponent"))?;
    if u64::from(power) * 4 > MAX_WORK_BITS {
        return Err(limit("decimal exponent bits"));
    }
    work.charge(
        (u64::from(power) + 1) * 8,
        (u64::from(power) + 1) * 8 + rounded.0.len() as u64 * 2,
    )?;
    let mantissa = BigInt::parse_bytes(rounded.0.replace('.', "").as_bytes(), 10)
        .ok_or(invalid("generated decimal mantissa"))?;
    let scale = BigInt::from(10).pow(power);
    let value = if exponent >= 0 {
        BigRational::from_integer(mantissa * scale)
    } else {
        BigRational::new(mantissa, scale)
    };
    work.scalar(&value)?;
    Ok(value)
}

fn exact_value_relation<M: BudgetMeter>(
    fraction: &Fraction,
    value: &BigRational,
    work: &mut Work<'_, M>,
) -> Result<Poly> {
    let scaled = work.scale(&fraction.denominator, value)?;
    work.sum(&fraction.numerator, &scaled, true)
}

fn certificate<M: BudgetMeter>(
    roots: &VerifiedRealRoots,
    interval: Interval,
    exact_value: Option<BigRational>,
    decimal: String,
    work: &mut Work<'_, M>,
) -> Result<AlgebraicValueCertificate> {
    work.charge(
        1,
        roots.intervals().len() as u64 * 96 + decimal.len() as u64,
    )?;
    for root in roots.intervals() {
        work.scalar(&root.lower)?;
        work.scalar(&root.upper)?;
    }
    Ok(AlgebraicValueCertificate {
        roots: roots.certificate().clone(),
        lower: interval.lower,
        upper: interval.upper,
        exact_value,
        decimal,
    })
}

/// Independently check the original source, chosen embedding, domain,
/// enclosure and final decimal. No candidate search or bisection is repeated.
pub fn verify_real_algebraic_value(
    expression: &Expr,
    variable: &Symbol,
    defining_coefficients: &[BigRational],
    index: usize,
    digits: u32,
    certificate: &AlgebraicValueCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedAlgebraicValue> {
    let mut capped = Capped {
        inner: meter,
        compute: 0,
        memory: 0,
    };
    let mut work = Work::new(&mut capped)?;
    if !(1..=MAX_REAL_ROOT_DIGITS).contains(&digits) {
        return Err(RealRootError::InvalidPrecision.into());
    }
    if !admit(expression, variable, &mut work)? {
        return Err(invalid("unsupported algebraic value source"));
    }
    if certificate.decimal.len() > digits as usize + 64 {
        return Err(limit("algebraic decimal bytes"));
    }
    work.scalar(&certificate.lower)?;
    work.scalar(&certificate.upper)?;
    if certificate.lower > certificate.upper {
        return Err(invalid("reversed algebraic value enclosure"));
    }
    let roots = verify_real_root_isolation(defining_coefficients, &certificate.roots, work.meter)?;
    let interval = root_interval(&roots, index, &mut work)?;
    let (fraction, guards) = source_fraction(expression, variable, &mut work)?;
    check_domain(&roots, index, &guards, &mut work)?;
    if let Some(value) = &certificate.exact_value {
        work.scalar(value)?;
        if certificate.lower != *value || certificate.upper != *value {
            return Err(invalid("exact algebraic value must have a point enclosure"));
        }
        let relation = exact_value_relation(&fraction, value, &mut work)?;
        if !real_root_polynomial_zero(&roots, index, &relation, work.meter)? {
            return Err(invalid("exact algebraic value relation"));
        }
    } else {
        let numerator = reference_polynomial(&fraction.numerator, &interval, &mut work)?;
        let denominator = reference_polynomial(&fraction.denominator, &interval, &mut work)?;
        let Some(reference) = interval_div(&numerator, &denominator, &mut work)? else {
            return Err(invalid("algebraic reference denominator includes zero"));
        };
        if certificate.lower > reference.lower || certificate.upper < reference.upper {
            return Err(invalid("algebraic enclosure omits the reference range"));
        }
    }
    let lower = rounded(&certificate.lower, digits, &mut work)?;
    let upper = rounded(&certificate.upper, digits, &mut work)?;
    if lower != upper || certificate.decimal != decimal_text(&lower) {
        return Err(invalid(
            "algebraic decimal is not determined by its enclosure",
        ));
    }
    // Preflight every owned retained field, not just the final decimal text.
    let defining_polynomial = work.copy(roots.coefficients())?;
    work.charge(
        1,
        variable.name.len() as u64 + certificate.decimal.len() as u64 + 256,
    )?;
    for interval in &certificate.roots.intervals {
        work.scalar(&interval.lower)?;
        work.scalar(&interval.upper)?;
    }
    work.scalar(&certificate.lower)?;
    work.scalar(&certificate.upper)?;
    if let Some(value) = &certificate.exact_value {
        work.scalar(value)?;
    }
    let result = VerifiedAlgebraicValue {
        expression: expression.clone(),
        variable: variable.clone(),
        defining_polynomial,
        index,
        precision: digits,
        certificate: certificate.clone(),
    };
    work.meter.checkpoint()?;
    Ok(result)
}

/// Evaluate one exact rational expression at an indexed real algebraic root.
/// `None` means the complete original grammar is unsupported. Undefined values,
/// invalid indices, exhausted budgets and cancellations are explicit errors.
pub fn metered_evaluate_real_algebraic(
    expression: &Expr,
    variable: &Symbol,
    defining_coefficients: &[BigRational],
    index: usize,
    digits: u32,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedAlgebraicValue>> {
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
    if !(1..=MAX_REAL_ROOT_DIGITS).contains(&digits) {
        return Err(RealRootError::InvalidPrecision.into());
    }
    if !admit(expression, variable, &mut generator)? {
        generator.meter.checkpoint()?;
        verifier_cap.checkpoint()?;
        return Ok(None);
    }
    let mut verifier = Work::new(&mut verifier_cap)?;
    let (original, guards) = source_fraction(expression, variable, &mut generator)?;
    let mut roots = isolate_real_roots(defining_coefficients, generator.meter, verifier.meter)?;
    root_interval(&roots, index, &mut generator)?;
    check_domain(&roots, index, &guards, &mut generator)?;

    // Legal cancellation happens only after the original guards were checked.
    let gcd = checked_gcd(
        &original.numerator,
        &original.denominator,
        variable,
        &mut generator,
        &mut verifier,
    )?;
    let numerator = checked_quotient(&original.numerator, &gcd, &mut generator, &mut verifier)?;
    let denominator = checked_quotient(&original.denominator, &gcd, &mut generator, &mut verifier)?;
    let modulus = square_free(
        roots.coefficients(),
        variable,
        &mut generator,
        &mut verifier,
    )?;
    let fraction = Fraction {
        numerator: checked_remainder(&numerator, &modulus, &mut generator, &mut verifier)?,
        denominator: checked_remainder(&denominator, &modulus, &mut generator, &mut verifier)?,
    };
    let exact = if real_root_polynomial_zero(&roots, index, &fraction.numerator, generator.meter)? {
        Some(BigRational::zero())
    } else if fraction.numerator.len() == 1 && fraction.denominator.len() == 1 {
        Some(generator.div(&fraction.numerator[0], &fraction.denominator[0])?)
    } else {
        None
    };
    let mut candidate = None;
    if let Some(exact) = exact {
        let interval = point(&exact, &mut generator)?;
        let text = decimal_text(&rounded(&exact, digits, &mut generator)?);
        candidate = Some(certificate(
            &roots,
            interval,
            Some(exact),
            text,
            &mut generator,
        )?);
    }

    // 3322/1000 is a strict upper bound on log2(10). Start near the
    // requested decimal precision instead of repeating high-precision
    // certificate checks after every small handful of bisections. Further
    // batches handle source conditioning under the same cumulative caps.
    let refinement_batch = (digits as usize * 3322).div_ceil(1000) + 16;
    let mut refinement_steps = 0;
    for _ in 0..MAX_REFINEMENT_BATCHES {
        if candidate.is_some() {
            break;
        }
        let interval = root_interval(&roots, index, &mut generator)?;
        let n = horner(&fraction.numerator, &interval, &mut generator)?;
        let d = horner(&fraction.denominator, &interval, &mut generator)?;
        let n_reference = reference_polynomial(&original.numerator, &interval, &mut generator)?;
        let d_reference = reference_polynomial(&original.denominator, &interval, &mut generator)?;
        let value = interval_div(&n, &d, &mut generator)?;
        let reference = interval_div(&n_reference, &d_reference, &mut generator)?;
        if let (Some(value), Some(reference)) = (value, reference) {
            // The reference representation may have a wider dependency
            // enclosure than the reduced generator. Publish their hull.
            let enclosure = Interval {
                lower: value.lower.min(reference.lower),
                upper: value.upper.max(reference.upper),
            };
            let lower = rounded(&enclosure.lower, digits, &mut generator)?;
            let upper = rounded(&enclosure.upper, digits, &mut generator)?;
            if lower == upper {
                candidate = Some(certificate(
                    &roots,
                    enclosure,
                    None,
                    decimal_text(&lower),
                    &mut generator,
                )?);
                break;
            }
            // Exact rounding ties need algebraic recognition: an arbitrarily
            // narrow interval can straddle the tie forever. An exact candidate
            // is accepted only if the original numerator-c*denominator vanishes
            // at this precise embedding, after its original poles were checked.
            let lower_value = rounded_rational(&lower, &mut generator)?;
            let upper_value = rounded_rational(&upper, &mut generator)?;
            let sum = generator.add(&lower_value, &upper_value)?;
            let boundary = generator.div(&sum, &BigRational::from_integer(2.into()))?;
            if boundary >= enclosure.lower && boundary <= enclosure.upper {
                let relation = exact_value_relation(&original, &boundary, &mut generator)?;
                if real_root_polynomial_zero(&roots, index, &relation, generator.meter)? {
                    let interval = point(&boundary, &mut generator)?;
                    let text = decimal_text(&rounded(&boundary, digits, &mut generator)?);
                    candidate = Some(certificate(
                        &roots,
                        interval,
                        Some(boundary),
                        text,
                        &mut generator,
                    )?);
                    break;
                }
            }
        }
        let steps = refinement_batch.min(MAX_REFINEMENT_STEPS - refinement_steps);
        if steps == 0 {
            break;
        }
        refinement_steps += steps;
        roots = refine_real_root_isolation(&roots, index, steps, generator.meter, verifier.meter)?;
    }
    let candidate = candidate.ok_or(limit("algebraic value refinement"))?;
    let result = verify_real_algebraic_value(
        expression,
        variable,
        defining_coefficients,
        index,
        digits,
        &candidate,
        verifier.meter,
    )?;
    generator.meter.checkpoint()?;
    verifier.meter.checkpoint()?;
    Ok(Some(result))
}

/// Convenience entry point with intrinsic bounds in addition to caller meters
/// available through [`metered_evaluate_real_algebraic`].
pub fn evaluate_real_algebraic(
    expression: &Expr,
    variable: &Symbol,
    defining_coefficients: &[BigRational],
    index: usize,
    digits: u32,
) -> Result<Option<VerifiedAlgebraicValue>> {
    metered_evaluate_real_algebraic(
        expression,
        variable,
        defining_coefficients,
        index,
        digits,
        &mut Unbounded,
        &mut Unbounded,
    )
}
