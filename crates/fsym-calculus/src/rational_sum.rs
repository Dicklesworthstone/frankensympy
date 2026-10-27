//! Bounded rational telescoping over `Q(k)`.
//!
//! Linear rational poles are grouped by integer shifts, separately at every
//! pole order. Vanishing total residues in each orbit give a rational
//! antidifference, including repeated poles and nonintegral offsets. The
//! independent proof-kernel checks the exact difference identity and the
//! original integer-range domain before a definite sum is published.

use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::factorization::metered_complete_factorization;
use fsym_polys::{PolyError, UnivariatePoly};
use fsym_proof_kernel::rational_sum::{
    RationalAntidifference, RationalSumPole, RationalSumUpper, RationalSumVerificationError,
    verify_rational_antidifference, verify_rational_sum,
};
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use thiserror::Error;

const MAX_INPUT_DEGREE: usize = 32;
const MAX_OUTPUT_DEGREE: usize = 128;
const MAX_INPUT_BITS: u64 = 2048;
const MAX_WORK_BITS: u64 = 16_384;
const MAX_SHIFT_SPAN: i64 = 64;
const MAX_PARTIAL_FRACTIONS: usize = 128;

#[derive(Debug, Error)]
pub enum RationalSummationError {
    #[error("rational summation refused: {0}")]
    Refused(&'static str),
    #[error(transparent)]
    Polynomial(#[from] PolyError),
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Verification(#[from] RationalSumVerificationError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for RationalSummationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, RationalSummationError>;

/// A rational sum and its recheckable antidifference certificate. Its value
/// is produced only after independent identity, domain and endpoint checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalSummation {
    pub value: BigRational,
    pub certificate: RationalAntidifference,
}

fn refuse(reason: &'static str) -> RationalSummationError {
    RationalSummationError::Refused(reason)
}

fn coefficient_bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}

fn validate(poly: &UnivariatePoly, degree: usize, bits: u64) -> Result<()> {
    poly.validate_shape()?;
    if poly.degree().unwrap_or(0) > degree || poly.coeffs.iter().any(|c| coefficient_bits(c) > bits)
    {
        return Err(refuse("polynomial degree or coefficient-height limit"));
    }
    Ok(())
}

/// Charges bounded synchronous polynomial batches before allocation. The
/// scalar elimination steps below use the arithmetic substrate's own meters.
fn batch(meter: &mut impl BudgetMeter, polys: &[&UnivariatePoly]) -> Result<()> {
    meter.checkpoint()?;
    let mut width = 1u64;
    let mut bits = 1u64;
    for poly in polys {
        validate(poly, MAX_OUTPUT_DEGREE, MAX_WORK_BITS)?;
        width = width.max(poly.coeffs.len() as u64);
        bits = bits.max(poly.coeffs.iter().map(coefficient_bits).max().unwrap_or(1));
    }
    let work = width.saturating_pow(3).saturating_mul(8);
    meter.charge_batch(&[
        (Dimension::ComputeSteps, work),
        (Dimension::AllocationCount, work),
        (
            Dimension::MemoryBytes,
            work.saturating_mul((bits + 16).div_ceil(8)),
        ),
    ])?;
    Ok(())
}

fn check_value(value: BigRational) -> Result<BigRational> {
    if coefficient_bits(&value) > MAX_WORK_BITS {
        Err(refuse("rational coefficient growth limit"))
    } else {
        Ok(value)
    }
}

fn scaled(poly: &UnivariatePoly, coefficient: &BigRational) -> UnivariatePoly {
    UnivariatePoly::new(
        poly.gen_sym.clone(),
        poly.coeffs.iter().map(|c| c * coefficient).collect(),
    )
}

fn exact_quotient(a: &UnivariatePoly, b: &UnivariatePoly) -> Result<UnivariatePoly> {
    let (q, r) = a.div_rem(b)?;
    if !r.is_zero() {
        return Err(refuse("nonexact partial-fraction division"));
    }
    Ok(q)
}

struct ForwardMeter<'a, M> {
    inner: &'a mut M,
    stopped: Option<MeterError>,
}

impl<M: BudgetMeter> BudgetMeter for ForwardMeter<'_, M> {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        let result = self.inner.charge(dimension, amount);
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let result = self.inner.charge_batch(charges);
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

fn denominator_poles(
    denominator: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<RationalSumPole>> {
    let mut forwarding = ForwardMeter {
        inner: meter,
        stopped: None,
    };
    let result = metered_complete_factorization(denominator, &mut forwarding);
    if let Some(error) = forwarding.stopped {
        return Err(error.into());
    }
    let factorization = result?;
    let mut poles = Vec::new();
    for factor in factorization.factors {
        if factor.poly.degree() != Some(1) {
            return Err(refuse(
                "denominator must split into rational linear factors",
            ));
        }
        poles.push(RationalSumPole {
            root: -factor.poly.coeffs[0].clone() / factor.poly.coeffs[1].clone(),
            multiplicity: u32::try_from(factor.multiplicity)
                .map_err(|_| refuse("pole multiplicity limit"))?,
        });
    }
    poles.sort_by(|a, b| a.root.cmp(&b.root));
    Ok(poles)
}

/// Solve the exact coefficient system for the partial fractions. Columns
/// are D/(k-r)^j, so repeated poles are handled without numerical residues.
fn partial_fractions(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    poles: &[RationalSumPole],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<(BigRational, u32, BigRational)>> {
    let n = denominator.degree().unwrap_or(0);
    batch(meter, &[numerator, denominator])?;
    let mut labels = Vec::new();
    let mut columns = Vec::new();
    for pole in poles {
        let factor = UnivariatePoly::new(
            denominator.gen_sym.clone(),
            vec![-pole.root.clone(), BigRational::one()],
        );
        let mut power = UnivariatePoly::one(denominator.gen_sym.clone());
        for order in 1..=pole.multiplicity {
            batch(meter, &[denominator, &power])?;
            power = power.mul(&factor)?;
            columns.push(exact_quotient(denominator, &power)?);
            labels.push((-pole.root.clone(), order));
        }
    }
    if columns.len() != n {
        return Err(refuse("denominator factor coverage mismatch"));
    }
    let mut rows = vec![vec![BigRational::zero(); n + 1]; n];
    for (row_index, row) in rows.iter_mut().enumerate() {
        for (column_index, column) in columns.iter().enumerate() {
            row[column_index] = column
                .coeffs
                .get(row_index)
                .cloned()
                .unwrap_or_else(BigRational::zero);
        }
        row[n] = numerator
            .coeffs
            .get(row_index)
            .cloned()
            .unwrap_or_else(BigRational::zero);
    }
    for column in 0..n {
        meter.checkpoint()?;
        let pivot = (column..n)
            .find(|&row| !rows[row][column].is_zero())
            .ok_or_else(|| refuse("singular partial-fraction coefficient system"))?;
        rows.swap(column, pivot);
        let divisor = rows[column][column].clone();
        for entry in &mut rows[column][column..=n] {
            *entry = check_value(entry.metered_div(&divisor, meter)?)?;
        }
        let pivot_row = rows[column].clone();
        for (row_index, row) in rows.iter_mut().enumerate() {
            if row_index == column || row[column].is_zero() {
                continue;
            }
            let coefficient = row[column].clone();
            for (entry, pivot_value) in row[column..=n].iter_mut().zip(&pivot_row[column..=n]) {
                let product = check_value(coefficient.metered_mul(pivot_value, meter)?)?;
                *entry = check_value(entry.metered_sub(&product, meter)?)?;
            }
        }
    }
    Ok(labels
        .into_iter()
        .enumerate()
        .filter_map(|(i, (shift, order))| {
            let coefficient = rows[i][n].clone();
            (!coefficient.is_zero()).then_some((shift, order, coefficient))
        })
        .collect())
}

fn polynomial_antidifference(
    polynomial: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<UnivariatePoly> {
    let mut remainder = polynomial.clone();
    let mut output = UnivariatePoly::zero(polynomial.gen_sym.clone());
    while !remainder.is_zero() {
        batch(meter, &[&remainder, &output])?;
        let degree = remainder.degree().expect("nonzero polynomial") + 1;
        let coefficient = remainder.leading_coeff().clone()
            / BigRational::from_integer(BigInt::from(degree as u64));
        let mut coefficients = vec![BigRational::zero(); degree + 1];
        coefficients[degree] = coefficient;
        let term = UnivariatePoly::new(polynomial.gen_sym.clone(), coefficients);
        let difference = term.shift(&BigRational::one())?.sub(&term)?;
        remainder = remainder.sub(&difference)?;
        if remainder.degree().is_some_and(|next| next + 1 >= degree) {
            return Err(refuse(
                "polynomial antidifference did not reduce the degree",
            ));
        }
        output = output.add(&term)?;
    }
    Ok(output)
}

fn orbit_antidifference(
    mut fractions: Vec<(BigRational, u32, BigRational)>,
    meter: &mut impl BudgetMeter,
) -> Result<BTreeMap<(BigRational, u32), BigRational>> {
    fractions.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut output = BTreeMap::new();
    while let Some((base, order, first)) = fractions.first().cloned() {
        meter.checkpoint()?;
        let mut orbit = BTreeMap::from([(0i64, first)]);
        let mut remaining = Vec::new();
        for (shift, power, coefficient) in fractions.into_iter().skip(1) {
            let difference = shift - base.clone();
            if power == order && difference.is_integer() {
                let offset = difference
                    .to_integer()
                    .to_i64()
                    .ok_or_else(|| refuse("integer-shift span limit"))?;
                if !(0..=MAX_SHIFT_SPAN).contains(&offset) {
                    return Err(refuse("integer-shift span limit"));
                }
                orbit.insert(offset, coefficient);
            } else {
                remaining.push((difference + base.clone(), power, coefficient));
            }
        }
        let last = *orbit.last_key_value().expect("nonempty orbit").0;
        let mut running = BigRational::zero();
        for offset in 0..=last {
            meter.checkpoint()?;
            if let Some(coefficient) = orbit.get(&offset) {
                running = check_value(running.metered_add(coefficient, meter)?)?;
            }
            if offset < last && !running.is_zero() {
                if output.len() >= MAX_PARTIAL_FRACTIONS {
                    return Err(refuse("antidifference term limit"));
                }
                meter.charge_batch(&[
                    (Dimension::ComputeSteps, 1),
                    (Dimension::AllocationCount, 8),
                    (Dimension::MemoryBytes, 128 + 4 * MAX_WORK_BITS.div_ceil(8)),
                ])?;
                output.insert(
                    (
                        base.clone() + BigRational::from_integer(BigInt::from(offset)),
                        order,
                    ),
                    -running.clone(),
                );
            }
        }
        if !running.is_zero() {
            return Err(refuse(
                "nonzero discrete residue; no rational antidifference",
            ));
        }
        fractions = remaining;
    }
    Ok(output)
}

/// Generate and independently check a rational antidifference. This does
/// not by itself assert that a sum crosses no original input poles.
pub fn rational_antidifference(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
) -> Result<RationalAntidifference> {
    metered_rational_antidifference(numerator, denominator, &mut Unbounded, &mut Unbounded)
}

pub fn metered_rational_antidifference(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<RationalAntidifference> {
    validate(numerator, MAX_INPUT_DEGREE, MAX_INPUT_BITS)?;
    validate(denominator, MAX_INPUT_DEGREE, MAX_INPUT_BITS)?;
    if numerator.gen_sym != denominator.gen_sym || denominator.is_zero() {
        return Err(refuse("different generators or zero denominator"));
    }
    batch(generator_meter, &[numerator, denominator])?;
    let poles = denominator_poles(denominator, generator_meter)?;
    let (polynomial, remainder) = numerator.div_rem(denominator)?;
    let polynomial = polynomial_antidifference(&polynomial, generator_meter)?;
    let fractions = partial_fractions(&remainder, denominator, &poles, generator_meter)?;
    let fractions = orbit_antidifference(fractions, generator_meter)?;
    let x = numerator.gen_sym.clone();
    let mut orders: BTreeMap<BigRational, u32> = BTreeMap::new();
    for (shift, order) in fractions.keys() {
        generator_meter.checkpoint()?;
        generator_meter.charge_batch(&[
            (Dimension::AllocationCount, 4),
            (Dimension::MemoryBytes, 64 + 2 * MAX_WORK_BITS.div_ceil(8)),
        ])?;
        let existing = orders.entry(shift.clone()).or_default();
        *existing = (*existing).max(*order);
    }
    let degree: usize = orders.values().map(|&value| value as usize).sum();
    if degree + polynomial.degree().unwrap_or(0) > MAX_OUTPUT_DEGREE {
        return Err(refuse("antidifference denominator degree limit"));
    }
    let mut bottom = UnivariatePoly::one(x.clone());
    for (shift, order) in &orders {
        let factor = UnivariatePoly::new(x.clone(), vec![shift.clone(), BigRational::one()]);
        batch(generator_meter, &[&bottom, &factor])?;
        bottom = bottom.mul(&factor.pow(*order)?)?;
    }
    batch(generator_meter, &[&polynomial, &bottom])?;
    let mut top = polynomial.mul(&bottom)?;
    for ((shift, order), coefficient) in fractions {
        let factor = UnivariatePoly::new(x.clone(), vec![shift, BigRational::one()]);
        batch(generator_meter, &[&top, &bottom])?;
        let quotient = exact_quotient(&bottom, &factor.pow(order)?)?;
        top = top.add(&scaled(&quotient, &coefficient))?;
    }
    validate(&top, MAX_OUTPUT_DEGREE, MAX_WORK_BITS)?;
    validate(&bottom, MAX_OUTPUT_DEGREE, MAX_WORK_BITS)?;
    let certificate = RationalAntidifference {
        numerator: top.coeffs,
        denominator: bottom.coeffs,
        input_poles: poles,
        antidifference_poles: orders
            .into_iter()
            .map(|(shift, multiplicity)| RationalSumPole {
                root: -shift,
                multiplicity,
            })
            .collect(),
    };
    verify_rational_antidifference(
        &numerator.coeffs,
        &denominator.coeffs,
        &certificate,
        verifier_meter,
    )?;
    Ok(certificate)
}

pub fn rational_sum(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &BigInt,
    upper: &RationalSumUpper,
) -> Result<RationalSummation> {
    metered_rational_sum(
        numerator,
        denominator,
        lower,
        upper,
        &mut Unbounded,
        &mut Unbounded,
    )
}

/// Separate generator and verifier meters preserve the verifier's resource
/// reservation. Interruption is returned without an unbounded retry.
pub fn metered_rational_sum(
    numerator: &UnivariatePoly,
    denominator: &UnivariatePoly,
    lower: &BigInt,
    upper: &RationalSumUpper,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<RationalSummation> {
    let certificate =
        metered_rational_antidifference(numerator, denominator, generator_meter, verifier_meter)?;
    let value = verify_rational_sum(
        &numerator.coeffs,
        &denominator.coeffs,
        &certificate,
        lower,
        upper,
        verifier_meter,
    )?;
    Ok(RationalSummation { value, certificate })
}

/// Recognizes the bounded rational quotient grammar before lowering. A
/// refused rational denominator must not fall through to expression folding,
/// where a zero numerator or nested inverse could hide an original pole.
pub(crate) fn is_rational_quotient(e: &Expr, x: &Symbol) -> Option<bool> {
    let mut pending = vec![(e, 0usize)];
    let mut visited = 0usize;
    let mut inverse = false;
    while let Some((node, depth)) = pending.pop() {
        visited += 1;
        if depth > 64 || visited + pending.len() > 2048 {
            return None;
        }
        match node {
            Expr::Integer(_) | Expr::Rational(_) => {}
            Expr::Sym(symbol) if symbol == x => {}
            Expr::Add(children) | Expr::Mul(children) => {
                if visited + pending.len() + children.len() > 2048 {
                    return None;
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                let power = match exponent.as_ref() {
                    Expr::Integer(value) => value,
                    Expr::Rational(value) if value.is_integer() => value.numer(),
                    _ => return Some(false),
                };
                inverse |= power.is_negative();
                pending.push((base, depth + 1));
            }
            _ => return Some(false),
        }
    }
    Some(inverse)
}

/// Bounded exact rational lowering that retains original undefined points.
/// In particular an inverse of N/D keeps D as a denominator guard, even
/// though that guard algebraically cancels in the resulting rational function.
pub(crate) fn original_rational_parts(
    e: &Expr,
    x: &Symbol,
) -> Option<(UnivariatePoly, UnivariatePoly)> {
    fn convert(
        e: &Expr,
        x: &Symbol,
        remaining: &mut usize,
        depth: usize,
    ) -> Option<(UnivariatePoly, UnivariatePoly)> {
        *remaining = remaining.checked_sub(1)?;
        if depth > 64 {
            return None;
        }
        let one = || UnivariatePoly::one(x.clone());
        let admit = |p: &UnivariatePoly| validate(p, MAX_INPUT_DEGREE, MAX_INPUT_BITS).ok();
        let pair = match e {
            Expr::Integer(n) => (
                UnivariatePoly::new(x.clone(), vec![BigRational::from_integer(n.clone())]),
                one(),
            ),
            Expr::Rational(r) => (UnivariatePoly::new(x.clone(), vec![r.clone()]), one()),
            Expr::Sym(s) if s == x => (
                UnivariatePoly::new(x.clone(), vec![BigRational::zero(), BigRational::one()]),
                one(),
            ),
            Expr::Add(terms) => {
                let mut top = UnivariatePoly::zero(x.clone());
                let mut bottom = one();
                for term in terms {
                    let (n, d) = convert(term, x, remaining, depth + 1)?;
                    if top.degree().unwrap_or(0) + d.degree().unwrap_or(0) > MAX_INPUT_DEGREE
                        || n.degree().unwrap_or(0) + bottom.degree().unwrap_or(0) > MAX_INPUT_DEGREE
                        || bottom.degree().unwrap_or(0) + d.degree().unwrap_or(0) > MAX_INPUT_DEGREE
                    {
                        return None;
                    }
                    top = top.mul(&d).ok()?.add(&n.mul(&bottom).ok()?).ok()?;
                    bottom = bottom.mul(&d).ok()?;
                    admit(&top)?;
                    admit(&bottom)?;
                }
                (top, bottom)
            }
            Expr::Mul(factors) => {
                let mut top = one();
                let mut bottom = one();
                for factor in factors {
                    let (n, d) = convert(factor, x, remaining, depth + 1)?;
                    if top.degree().unwrap_or(0) + n.degree().unwrap_or(0) > MAX_INPUT_DEGREE
                        || bottom.degree().unwrap_or(0) + d.degree().unwrap_or(0) > MAX_INPUT_DEGREE
                    {
                        return None;
                    }
                    top = top.mul(&n).ok()?;
                    bottom = bottom.mul(&d).ok()?;
                    admit(&top)?;
                    admit(&bottom)?;
                }
                (top, bottom)
            }
            Expr::Pow(base, exponent) => {
                let power = match exponent.as_ref() {
                    Expr::Integer(n) => n.to_i64()?,
                    Expr::Rational(r) if r.is_integer() => r.to_integer().to_i64()?,
                    _ => return None,
                };
                if power.unsigned_abs() > 32 {
                    return None;
                }
                let (n, d) = convert(base, x, remaining, depth + 1)?;
                let degree = n.degree().unwrap_or(0).max(d.degree().unwrap_or(0));
                if degree * (power.unsigned_abs() as usize + 1) > MAX_INPUT_DEGREE {
                    return None;
                }
                let magnitude = power.unsigned_abs() as u32;
                if power > 0 {
                    (n.pow(magnitude).ok()?, d.pow(magnitude).ok()?)
                } else if power == 0 {
                    (d.clone(), d)
                } else {
                    (
                        d.pow(magnitude + 1).ok()?,
                        n.pow(magnitude).ok()?.mul(&d).ok()?,
                    )
                }
            }
            _ => return None,
        };
        admit(&pair.0)?;
        admit(&pair.1)?;
        (!pair.1.is_zero()).then_some(pair)
    }
    convert(e, x, &mut 2048, 0)
}
