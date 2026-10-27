//! Complete bounded sign decomposition of an original rational expression.
//!
//! The source-to-fraction recurrence is shared with exact real equations.
//! Original reciprocal exclusions are retained before any cancellation.
//! Independently checked polynomial identities and complete real-root counts
//! establish the entire critical set. A separate exact source evaluator checks
//! every proposed gap sign; endpoint ownership distinguishes zeros from holes.
//! These are synchronous reference checks, not a portable proof-kernel claim.

use super::*;
use crate::inequality::IneqOp;
use fsym_budget::BudgetError;
use fsym_core::real_roots::identify_real_root_factors;
use std::cmp::Ordering;

/// Complete real sign-chart result tied to the original typed source.
///
/// If the critical polynomial has `n` real roots, the gap mask contains `n+1`
/// entries in increasing real order, and the point mask contains `n` entries.
/// Gaps are open. A point included by a weak inequality is never an original
/// denominator hole. Constant one is the critical polynomial for no endpoints.
/// No unchecked constructor, mutable field, or deserializer grants acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRealInequality {
    expression: Expr,
    variable: Symbol,
    operation: IneqOp,
    polynomial: UnivariatePoly,
    roots: VerifiedRealRoots,
    gap_inclusions: Vec<bool>,
    point_inclusions: Vec<bool>,
}

impl VerifiedRealInequality {
    pub fn expression(&self) -> &Expr {
        &self.expression
    }
    pub fn variable(&self) -> &Symbol {
        &self.variable
    }
    pub fn operation(&self) -> IneqOp {
        self.operation
    }
    pub fn polynomial(&self) -> &UnivariatePoly {
        &self.polynomial
    }
    pub fn roots(&self) -> &VerifiedRealRoots {
        &self.roots
    }
    pub fn gap_inclusions(&self) -> &[bool] {
        &self.gap_inclusions
    }
    pub fn point_inclusions(&self) -> &[bool] {
        &self.point_inclusions
    }
}

/// All stages, including core isolation and root-factor verification, consume
/// one fixed allowance per lane. Nested reference kernels cannot reset it.
struct CappedMeter<'a, M> {
    inner: &'a mut M,
    used: [u64; 3],
}

impl<M: BudgetMeter> BudgetMeter for CappedMeter<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let mut amounts = [0u64; 3];
        for &(dimension, amount) in charges {
            let slot = match dimension {
                Dimension::ComputeSteps => Some(0),
                Dimension::MemoryBytes => Some(1),
                Dimension::AllocationCount => Some(2),
                _ => None,
            };
            if let Some(slot) = slot {
                amounts[slot] = amounts[slot].checked_add(amount).ok_or(MeterError::Budget(
                    BudgetError::ChargeOverflow { dimension },
                ))?;
            }
        }
        for (slot, dimension, cap) in [
            (0, Dimension::ComputeSteps, MAX_WORK),
            (1, Dimension::MemoryBytes, MAX_ALLOCATION),
            (2, Dimension::AllocationCount, MAX_WORK),
        ] {
            if amounts[slot] > cap - self.used[slot] {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amounts[slot],
                    remaining: cap - self.used[slot],
                }));
            }
        }
        self.inner.charge_batch(charges)?;
        for (used, amount) in self.used.iter_mut().zip(amounts) {
            *used += amount;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.inner.checkpoint()
    }
}

fn accepts(operation: IneqOp, sign: Ordering) -> bool {
    match operation {
        IneqOp::Lt => sign == Ordering::Less,
        IneqOp::Le => sign != Ordering::Greater,
        IneqOp::Gt => sign == Ordering::Greater,
        IneqOp::Ge => sign != Ordering::Less,
    }
}

fn evaluate_polynomial<M: BudgetMeter>(
    polynomial: &[BigRational],
    point: &BigRational,
    work: &mut Work<'_, M>,
) -> Result<BigRational> {
    let mut result = BigRational::zero();
    for coefficient in polynomial.iter().rev() {
        result = work.mul(&result, point)?;
        result = work.add(&result, coefficient)?;
    }
    Ok(result)
}

/// Evaluate the original tree directly, without conversion, cancellation,
/// symbolic simplification, approximate arithmetic, or short-circuiting zero
/// products/zeroth powers around undefined children.
fn evaluate_source<M: BudgetMeter>(
    expression: &Expr,
    variable: &Symbol,
    point: &BigRational,
    work: &mut Work<'_, M>,
) -> Result<Option<BigRational>> {
    work.charge(1, 64)?;
    match expression {
        Expr::Integer(n) => {
            work.charge(1, 32 + n.bits().div_ceil(8))?;
            Ok(Some(BigRational::from_integer(n.clone())))
        }
        Expr::Rational(q) => {
            work.scalar(q)?;
            Ok(Some(q.clone()))
        }
        Expr::Sym(symbol) if symbol == variable => {
            work.scalar(point)?;
            Ok(Some(point.clone()))
        }
        Expr::Add(terms) | Expr::Mul(terms) => {
            let adding = matches!(expression, Expr::Add(_));
            let mut result = if adding {
                BigRational::zero()
            } else {
                BigRational::one()
            };
            for term in terms {
                let Some(value) = evaluate_source(term, variable, point, work)? else {
                    return Ok(None);
                };
                result = if adding {
                    work.add(&result, &value)?
                } else {
                    work.mul(&result, &value)?
                };
            }
            Ok(Some(result))
        }
        Expr::Pow(base, exponent) => {
            let Expr::Integer(exponent) = exponent.as_ref() else {
                return Err(invalid("unadmitted sign-chart exponent"));
            };
            let exponent = exponent.to_i64().ok_or(limit("integer exponent"))?;
            let Some(mut base) = evaluate_source(base, variable, point, work)? else {
                return Ok(None);
            };
            if exponent < 0 {
                if base.is_zero() {
                    return Ok(None);
                }
                base = work.div(&BigRational::one(), &base)?;
            }
            let mut exponent = exponent.unsigned_abs();
            let mut result = BigRational::one();
            while exponent > 0 {
                if exponent & 1 == 1 {
                    result = work.mul(&result, &base)?;
                }
                exponent >>= 1;
                if exponent > 0 {
                    base = work.mul(&base, &base)?;
                }
            }
            Ok(Some(result))
        }
        _ => Err(invalid("unadmitted sign-chart expression")),
    }
}

fn samples<M: BudgetMeter>(
    roots: &VerifiedRealRoots,
    work: &mut Work<'_, M>,
) -> Result<Vec<BigRational>> {
    let intervals = roots.intervals();
    work.charge(1, (intervals.len() as u64 + 1) * 64)?;
    let mut result = Vec::with_capacity(intervals.len() + 1);
    if intervals.is_empty() {
        result.push(BigRational::zero());
        return Ok(result);
    }
    result.push(work.sub(&intervals[0].lower, &BigRational::one())?);
    for pair in intervals.windows(2) {
        let sum = work.add(&pair[0].upper, &pair[1].lower)?;
        result.push(work.div(&sum, &BigRational::from_integer(2.into()))?);
    }
    result.push(work.add(&intervals.last().unwrap().upper, &BigRational::one())?);
    Ok(result)
}

struct ChartCandidate {
    samples: Vec<BigRational>,
    gaps: Vec<bool>,
    points: Vec<bool>,
}

struct ChartStatement<'a> {
    expression: &'a Expr,
    variable: &'a Symbol,
    operation: IneqOp,
    roots: &'a VerifiedRealRoots,
    zero_points: &'a [bool],
    nowhere_defined: bool,
}

/// Check the sample position in every certified open gap, then evaluate the
/// original expression independently. Complete critical roots establish that
/// the continuous rational function has constant sign in each entire gap.
fn check_chart<M: BudgetMeter>(
    statement: &ChartStatement<'_>,
    candidate: &ChartCandidate,
    work: &mut Work<'_, M>,
) -> Result<()> {
    let intervals = statement.roots.intervals();
    if candidate.samples.len() != intervals.len() + 1
        || candidate.gaps.len() != intervals.len() + 1
        || candidate.points.len() != intervals.len()
        || statement.zero_points.len() != intervals.len()
    {
        return Err(invalid("sign-chart shape"));
    }
    for (index, point) in candidate.samples.iter().enumerate() {
        work.scalar(point)?;
        if index > 0 && point < &intervals[index - 1].upper
            || index < intervals.len() && point > &intervals[index].lower
            || evaluate_polynomial(statement.roots.coefficients(), point, work)?.is_zero()
        {
            return Err(invalid("sample is not in its certified open gap"));
        }
        let value = evaluate_source(statement.expression, statement.variable, point, work)?;
        let expected = if statement.nowhere_defined {
            if value.is_some() {
                return Err(invalid("nowhere-defined source evaluates"));
            }
            false
        } else {
            let value = value.ok_or(invalid("source undefined away from critical points"))?;
            accepts(statement.operation, value.cmp(&BigRational::zero()))
        };
        if candidate.gaps[index] != expected {
            return Err(invalid(
                "sign-chart gap decision differs from original source",
            ));
        }
    }
    let weak = accepts(statement.operation, Ordering::Equal);
    for (included, is_zero) in candidate.points.iter().zip(statement.zero_points) {
        work.charge(1, 1)?;
        if *included != (*is_zero && weak) {
            return Err(invalid("sign-chart endpoint decision"));
        }
    }
    work.meter.checkpoint()?;
    Ok(())
}

/// Complete solution of `expression operation 0` over the reals.
///
/// Unsupported original grammar returns `Ok(None)`. Resource, cancellation,
/// and reduction errors never become an empty set. The combined critical
/// polynomial is bounded at degree 64 and 4096 coefficient bits; original
/// expression bounds and guarded division semantics match exact real solving.
pub fn metered_solve_real_inequality(
    expression: &Expr,
    variable: &Symbol,
    operation: IneqOp,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealInequality>> {
    let mut generator_meter = CappedMeter {
        inner: generator_meter,
        used: [0; 3],
    };
    let mut verifier_meter = CappedMeter {
        inner: verifier_meter,
        used: [0; 3],
    };
    let mut generator = Work::new(&mut generator_meter)?;
    if !admit(expression, variable, &mut generator)? {
        return Ok(None);
    }
    let mut verifier = Work::new(&mut verifier_meter)?;
    let mut guards = Vec::new();
    let fraction = convert(expression, variable, &mut guards, &mut generator)?;
    generator.charge(1, 32)?;
    guards.push(generator.copy(&fraction.denominator)?);
    let nowhere_defined = guards.iter().any(|p| zero(p));

    let mut exclusions = generator.constant(1)?;
    let mut zeros = generator.constant(1)?;
    if !nowhere_defined {
        for guard in guards {
            if guard.len() == 1 {
                continue;
            }
            let guard = square_free(&guard, variable, &mut generator, &mut verifier)?;
            let gcd = checked_gcd(&exclusions, &guard, variable, &mut generator, &mut verifier)?;
            let remaining = checked_quotient(&guard, &gcd, &mut generator, &mut verifier)?;
            exclusions = verifier.product(&exclusions, &remaining)?;
            if exclusions.len() > MAX_REAL_ROOT_DEGREE + 1 {
                return Err(limit("sign-chart exclusion degree"));
            }
        }
        if !zero(&fraction.numerator) {
            zeros = square_free(&fraction.numerator, variable, &mut generator, &mut verifier)?;
            let gcd = checked_gcd(&zeros, &exclusions, variable, &mut generator, &mut verifier)?;
            zeros = checked_quotient(&zeros, &gcd, &mut generator, &mut verifier)?;
        }
    }
    let polynomial = verifier.product(&zeros, &exclusions)?;
    if polynomial.len() > MAX_REAL_ROOT_DEGREE + 1
        || polynomial
            .iter()
            .any(|q| bits(q) > MAX_REAL_ROOT_INPUT_BITS)
    {
        return Err(limit("sign-chart critical degree/coefficient bits"));
    }
    let roots = isolate_real_roots(&polynomial, generator.meter, verifier.meter)?;
    if roots.intervals().iter().any(|root| root.multiplicity != 1) {
        return Err(invalid("sign-chart critical polynomial is not square-free"));
    }
    verifier.charge(1, 256)?;
    let mut factors = Vec::with_capacity(2);
    let mut zero_factor = None;
    if zeros.len() > 1 {
        zero_factor = Some(factors.len());
        factors.push(zeros);
    }
    if exclusions.len() > 1 {
        factors.push(exclusions);
    }
    let owners =
        identify_real_root_factors(&roots, &factors, &vec![1; factors.len()], verifier.meter)?;
    generator.charge(1, roots.intervals().len() as u64 * 4 + 64)?;
    let zero_points: Vec<bool> = owners
        .iter()
        .map(|&(owner, _)| Some(owner) == zero_factor)
        .collect();
    let mut candidate = ChartCandidate {
        samples: samples(&roots, &mut generator)?,
        gaps: Vec::with_capacity(roots.intervals().len() + 1),
        points: zero_points
            .iter()
            .map(|&is_zero| is_zero && accepts(operation, Ordering::Equal))
            .collect(),
    };
    for point in &candidate.samples {
        let included = if nowhere_defined {
            false
        } else {
            let numerator = evaluate_polynomial(&fraction.numerator, point, &mut generator)?;
            let denominator = evaluate_polynomial(&fraction.denominator, point, &mut generator)?;
            let value = generator.div(&numerator, &denominator)?;
            accepts(operation, value.cmp(&BigRational::zero()))
        };
        candidate.gaps.push(included);
    }
    check_chart(
        &ChartStatement {
            expression,
            variable,
            operation,
            roots: &roots,
            zero_points: &zero_points,
            nowhere_defined,
        },
        &candidate,
        &mut verifier,
    )?;
    generator.charge(1, 256 + variable.name.len() as u64 * 2)?;
    verifier.meter.checkpoint()?;
    Ok(Some(VerifiedRealInequality {
        expression: expression.clone(),
        variable: variable.clone(),
        operation,
        polynomial: UnivariatePoly::new(variable.clone(), polynomial),
        roots,
        gap_inclusions: candidate.gaps,
        point_inclusions: candidate.points,
    }))
}

/// Convenience entry point retaining intrinsic bounds without a caller budget.
pub fn solve_real_inequality(
    expression: &Expr,
    variable: &Symbol,
    operation: IneqOp,
) -> Result<Option<VerifiedRealInequality>> {
    metered_solve_real_inequality(
        expression,
        variable,
        operation,
        &mut Unbounded,
        &mut Unbounded,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_source_check_rejects_gap_point_and_sample_mutants() {
        let x = Symbol::new("x");
        let expression = Expr::Sym(x.clone());
        let roots =
            isolate_real_roots(&[0.into(), 1.into()], &mut Unbounded, &mut Unbounded).unwrap();
        let statement = ChartStatement {
            expression: &expression,
            variable: &x,
            operation: IneqOp::Ge,
            roots: &roots,
            zero_points: &[true],
            nowhere_defined: false,
        };
        let mut candidate = ChartCandidate {
            samples: samples(&roots, &mut Work::new(&mut Unbounded).unwrap()).unwrap(),
            gaps: vec![false, true],
            points: vec![true],
        };
        let check = |candidate: &ChartCandidate| {
            check_chart(
                &statement,
                candidate,
                &mut Work::new(&mut Unbounded).unwrap(),
            )
        };
        assert!(check(&candidate).is_ok());
        candidate.gaps[0] = true;
        assert!(check(&candidate).is_err());
        candidate.gaps[0] = false;
        candidate.points[0] = false;
        assert!(check(&candidate).is_err());
        candidate.points[0] = true;
        candidate.samples[0] = 0.into();
        assert!(check(&candidate).is_err());
        candidate.samples[0] = 1.into();
        assert!(check(&candidate).is_err());
        candidate.points.push(true);
        assert!(check(&candidate).is_err());
    }

    #[test]
    fn cumulative_caps_and_underlying_budget_refusal_do_not_advance_counters() {
        let mut unlimited = Unbounded;
        let mut meter = CappedMeter {
            inner: &mut unlimited,
            used: [0; 3],
        };
        meter.charge(Dimension::ComputeSteps, MAX_WORK).unwrap();
        assert!(meter.charge(Dimension::ComputeSteps, 1).is_err());
        assert_eq!(meter.used, [MAX_WORK, 0, 0]);
        let mut tiny = fsym_budget::Budget::new(fsym_budget::BudgetLimits::uniform(1, 0));
        let mut meter = CappedMeter {
            inner: &mut tiny,
            used: [0; 3],
        };
        assert!(
            meter
                .charge_batch(&[(Dimension::ComputeSteps, 1), (Dimension::MemoryBytes, 2)])
                .is_err()
        );
        assert_eq!(meter.used, [0; 3]);
    }
}
