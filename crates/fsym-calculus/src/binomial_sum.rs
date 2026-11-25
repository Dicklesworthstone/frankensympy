//! Checked exact polynomial-weighted finite binomial sums.
//!
//! The admitted family is `sum(k=0..n) p(k)*binomial(n,k)*a^k*b^(n-k)`,
//! with `p in QQ[k]`, rational `a,b`, and a literal nonnegative integer `n`.
//! Generation uses forward differences and consecutive binomial moments.
//! Verification independently admits the original weight grammar and degree,
//! reconstructs the falling-factorial polynomial, and proves equality at
//! enough exact integer points to exceed BOTH degree bounds. Original source
//! evaluation and exponential-factor classification do not use the generator's
//! polynomial expansion. Each finite binomial moment and the claimed final
//! exact value are then checked independently.
//!
//! For `j <= n`, `(k)_j*binomial(n,k) = (n)_j*binomial(n-j,k-j)`;
//! shifting the finite range proves the moment
//! `(n)_j*a^j*(a+b)^(n-j)`. Terms with `j > n` vanish identically and are
//! removed BEFORE powers are constructed, including when `a+b == 0`.
//! The discrete convention is `0^0 = 1`, as for a polynomial's constant term.
//!
//! One cumulative meter covers generation, checking and publication. Source
//! children under held zero products/powers are admitted before arithmetic.
//! This is an in-process receipt, not a portable proof-kernel certificate;
//! symbolic upper bounds are unsupported because an identity digest alone
//! does not establish the required nonnegative-integer assumptions.
//! Only an exact full-range binomial factor selects this lane. Partial ranges,
//! squared binomials and other families retain the dispatcher's existing rules
//! without receiving this receipt; failures AFTER selection remain terminal.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::arith::RationalArithmeticError;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use num_traits::{One, Zero};
use thiserror::Error;

const MAX_DEGREE: usize = 32;
const MAX_UPPER: usize = 65_536;
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_INPUT_BITS: u64 = 4096;
const MAX_WORK_BITS: u64 = 65_536;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum BinomialSummationError {
    #[error("polynomial-binomial summation does not support {0}")]
    Unsupported(&'static str),
    #[error("polynomial-binomial summation refused: {0}")]
    Refused(&'static str),
    #[error("polynomial-binomial source, coefficient identity or claimed value does not match")]
    InvalidCertificate,
    #[error(transparent)]
    Meter(#[from] MeterError),
    #[error(transparent)]
    Arithmetic(RationalArithmeticError),
}

impl From<RationalArithmeticError> for BinomialSummationError {
    fn from(error: RationalArithmeticError) -> Self {
        match error {
            RationalArithmeticError::Meter(error) => Self::Meter(error),
            error => Self::Arithmetic(error),
        }
    }
}

type Result<T> = std::result::Result<T, BinomialSummationError>;
type Coefficients = Vec<BigRational>;

/// Untrusted, original-source-bound proposal, including the exact claimed sum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialBinomialCertificate {
    pub source: Expr,
    pub index: Symbol,
    pub lower: Expr,
    pub upper: Expr,
    pub falling_coefficients: Coefficients,
    pub claimed_sum: BigRational,
}

/// Acceptance is available only after source, identity and value verification.
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialBinomialSum {
    certificate: PolynomialBinomialCertificate,
    value: Expr,
}

impl VerifiedPolynomialBinomialSum {
    pub fn source(&self) -> &Expr {
        &self.certificate.source
    }
    pub fn index(&self) -> &Symbol {
        &self.certificate.index
    }
    pub fn lower(&self) -> &Expr {
        &self.certificate.lower
    }
    pub fn upper(&self) -> &Expr {
        &self.certificate.upper
    }
    pub fn certificate(&self) -> &PolynomialBinomialCertificate {
        &self.certificate
    }
    pub fn value(&self) -> &Expr {
        &self.value
    }
    pub fn into_value(self) -> Expr {
        self.value
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    compute: u64,
    memory: u64,
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            compute: 0,
            memory: 0,
        }
    }
    fn allocate(&mut self, count: usize, bytes: u64) -> Result<()> {
        self.checkpoint()?;
        self.charge_batch(&[
            (Dimension::ComputeSteps, count.max(1) as u64),
            (Dimension::AllocationCount, count.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1)),
        ])?;
        Ok(())
    }
    fn copy(&mut self, value: &BigRational) -> Result<BigRational> {
        self.allocate(1, 64 + bits(value).div_ceil(8) * 2)?;
        Ok(value.clone())
    }
    fn preflight(&mut self, a: &BigRational, b: &BigRational) -> Result<()> {
        self.checkpoint()?;
        if bits(a).saturating_add(bits(b)).saturating_add(2) > MAX_WORK_BITS {
            return Err(BinomialSummationError::Refused(
                "intermediate scalar growth bound",
            ));
        }
        Ok(())
    }
    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_add(b, self)?)
    }
    fn sub(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_sub(b, self)?)
    }
    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        Ok(a.metered_mul(b, self)?)
    }
    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        self.preflight(a, b)?;
        if b.is_zero() {
            return Err(BinomialSummationError::Refused("division by zero"));
        }
        Ok(a.metered_div(b, self)?)
    }
    fn pow(&mut self, base: &BigRational, exponent: i32) -> Result<BigRational> {
        self.checkpoint()?;
        if base.is_zero() && exponent < 0 {
            return Err(BinomialSummationError::Refused("undefined zero power"));
        }
        if exponent == 0 {
            return Ok(BigRational::one());
        }
        if base.is_zero() || base.is_one() {
            return self.copy(base);
        }
        if exponent.unsigned_abs() as usize > MAX_UPPER
            || bits(base).saturating_mul(u64::from(exponent.unsigned_abs())) > MAX_WORK_BITS / 2
        {
            return Err(BinomialSummationError::Refused("scalar power growth bound"));
        }
        Ok(base.metered_pow(exponent, self)?)
    }
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        self.meter.checkpoint()
    }
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        let (mut compute, mut memory) = (self.compute, self.memory);
        for &(dimension, amount) in charges {
            let (used, limit) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_COMPUTE),
                Dimension::MemoryBytes => (&mut memory, MAX_MEMORY),
                _ => continue,
            };
            let remaining = limit.saturating_sub(*used);
            if amount > remaining {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                }));
            }
            *used += amount;
        }
        self.meter.charge_batch(charges)?;
        self.compute = compute;
        self.memory = memory;
        Ok(())
    }
}

fn bits(value: &BigRational) -> u64 {
    value.numer().bits().max(value.denom().bits()).max(1)
}
fn integer(value: usize) -> BigRational {
    BigRational::from_integer(BigInt::from(value as u64))
}
fn number(expression: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<Option<BigRational>> {
    match expression {
        Expr::Integer(value) => {
            work.allocate(1, 64 + value.bits().div_ceil(8))?;
            Ok(Some(BigRational::from_integer(value.clone())))
        }
        Expr::Rational(value) => Ok(Some(work.copy(value)?)),
        _ => Ok(None),
    }
}
fn trim(poly: &mut Coefficients) {
    while poly.len() > 1 && poly.last().is_some_and(Zero::is_zero) {
        poly.pop();
    }
}

/// Bound the complete borrowed tree, even children under held zero factors.
/// `seen` is a family probe only; mathematical grammar is checked separately.
fn admit(source: &Expr, index: &Symbol, work: &mut Work<'_, impl BudgetMeter>) -> Result<bool> {
    // Reserve the entire bounded stack before any push/extend can allocate.
    work.allocate(
        1,
        (MAX_NODES * std::mem::size_of::<(&Expr, usize)>()) as u64,
    )?;
    let mut pending = Vec::with_capacity(MAX_NODES);
    pending.push((source, 0usize));
    let (mut visited, mut seen) = (0usize, false);
    let mut refusal = (index.name.len() > 4096).then_some("index name bound");
    while let Some((node, depth)) = pending.pop() {
        work.checkpoint()?;
        if visited == MAX_NODES || depth > MAX_DEPTH {
            return Err(BinomialSummationError::Refused("source shape bound"));
        }
        visited += 1;
        work.allocate(1, 128)?;
        let children = match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_INPUT_BITS {
                    refusal = Some("input scalar bound");
                } else {
                    work.allocate(1, value.bits().div_ceil(8))?;
                }
                None
            }
            Expr::Rational(value) => {
                if bits(value) > MAX_INPUT_BITS {
                    refusal = Some("input scalar bound");
                } else {
                    work.allocate(1, bits(value).div_ceil(8) * 2)?;
                }
                None
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > 4096 {
                    refusal = Some("symbol name bound");
                } else {
                    work.allocate(1, symbol.name.len() as u64)?;
                }
                None
            }
            Expr::Add(children) | Expr::Mul(children) => Some(children),
            Expr::Function(name, children) => {
                seen |= name == "binomial";
                if name.len() > 4096 {
                    refusal = Some("function name bound");
                } else {
                    work.allocate(1, name.len() as u64)?;
                }
                Some(children)
            }
            Expr::Pow(base, exponent) => {
                if base.is_zero()
                    && matches!(exponent.as_ref(), Expr::Integer(value) if value.is_negative())
                {
                    refusal = Some("original zero denominator");
                }
                if visited + pending.len() + 2 > MAX_NODES {
                    return Err(BinomialSummationError::Refused("source shape bound"));
                }
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
                None
            }
            Expr::Const(
                Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity
                | Constant::NaN,
            ) => {
                refusal = Some("nonfinite original summand");
                None
            }
            Expr::Const(_) => None,
        };
        if let Some(children) = children {
            if children.len() > MAX_NODES.saturating_sub(visited + pending.len()) {
                return Err(BinomialSummationError::Refused("source shape bound"));
            }
            pending.extend(children.iter().map(|child| (child, depth + 1)));
        }
    }
    if seen && let Some(reason) = refusal {
        return Err(BinomialSummationError::Refused(reason));
    }
    Ok(seen)
}

fn add_poly(
    a: &[BigRational],
    b: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    let size = a.len().max(b.len());
    work.allocate(size, (size * 64) as u64)?;
    let zero = BigRational::zero();
    let mut result = Vec::with_capacity(size);
    for i in 0..size {
        result.push(work.add(a.get(i).unwrap_or(&zero), b.get(i).unwrap_or(&zero))?);
    }
    trim(&mut result);
    Ok(result)
}
fn mul_poly(
    a: &[BigRational],
    b: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    if a.len() + b.len() - 2 > MAX_DEGREE {
        return Err(BinomialSummationError::Refused("polynomial degree bound"));
    }
    let size = a.len() + b.len() - 1;
    work.allocate(size, (size * 64) as u64)?;
    let mut result = vec![BigRational::zero(); size];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let term = work.mul(x, y)?;
            result[i + j] = work.add(&result[i + j], &term)?;
        }
    }
    trim(&mut result);
    Ok(result)
}
fn polynomial(
    source: &Expr,
    index: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    work.allocate(1, 64)?;
    match source {
        Expr::Integer(_) | Expr::Rational(_) => {
            Ok(vec![number(source, work)?.expect("numeric atom")])
        }
        Expr::Sym(symbol) if symbol == index => Ok(vec![BigRational::zero(), BigRational::one()]),
        Expr::Add(children) | Expr::Mul(children) => {
            let multiply = matches!(source, Expr::Mul(_));
            let mut result = vec![if multiply {
                BigRational::one()
            } else {
                BigRational::zero()
            }];
            for child in children {
                let next = polynomial(child, index, work)?;
                result = if multiply {
                    mul_poly(&result, &next, work)?
                } else {
                    add_poly(&result, &next, work)?
                };
            }
            Ok(result)
        }
        Expr::Pow(base, exponent) => {
            // Parsing the base first prevents a held **0 from hiding a pole.
            let mut base = polynomial(base, index, work)?;
            let power = match exponent.as_ref() {
                Expr::Integer(value) => value.to_i64().and_then(|v| i32::try_from(v).ok()),
                _ => None,
            }
            .ok_or(BinomialSummationError::Unsupported(
                "noninteger polynomial powers",
            ))?;
            if base.len() == 1 {
                return Ok(vec![work.pow(&base[0], power)?]);
            }
            if power < 0 {
                return Err(BinomialSummationError::Unsupported(
                    "rational weights with possible poles",
                ));
            }
            if (base.len() - 1).saturating_mul(power as usize) > MAX_DEGREE {
                return Err(BinomialSummationError::Refused("polynomial degree bound"));
            }
            let mut result = vec![BigRational::one()];
            let mut remaining = power as u32;
            while remaining != 0 {
                if remaining & 1 != 0 {
                    result = mul_poly(&result, &base, work)?;
                }
                remaining >>= 1;
                if remaining != 0 {
                    base = mul_poly(&base, &base, work)?;
                }
            }
            Ok(result)
        }
        _ => Err(BinomialSummationError::Unsupported(
            "coefficients outside QQ or nonpolynomial weights",
        )),
    }
}

/// Selection only, after complete borrowed shape admission. Inspect root
/// multiplicative factors, without looking through powers, sums or functions.
/// Existing short sums of binomial powers and partial ranges stay outside
/// this checked family rather than becoming failed candidates.
fn selects_full_range_factor(
    source: &Expr,
    index: &Symbol,
    upper: &Expr,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<bool> {
    work.allocate(1, (MAX_NODES * std::mem::size_of::<&Expr>()) as u64)?;
    let mut pending = Vec::with_capacity(MAX_NODES);
    pending.push(source);
    let (mut factors, mut matches) = (0usize, false);
    while let Some(part) = pending.pop() {
        work.allocate(1, 64)?;
        match part {
            Expr::Mul(children) => pending.extend(children.iter()),
            Expr::Function(name, args) if name == "binomial" => {
                factors += 1;
                matches |= matches!(args.as_slice(), [Expr::Integer(top), Expr::Sym(variable)]
                    if matches!(upper, Expr::Integer(bound) if top == bound) && variable == index);
            }
            _ => {}
        }
    }
    Ok(factors == 1 && matches)
}

fn upper_endpoint(lower: &Expr, upper: &Expr) -> Result<usize> {
    if !matches!(lower, Expr::Integer(value) if value.is_zero()) {
        return Err(BinomialSummationError::Unsupported(
            "a lower endpoint other than zero",
        ));
    }
    match upper {
        Expr::Integer(value) => value
            .to_i64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v <= MAX_UPPER)
            .ok_or(BinomialSummationError::Refused(
                "nonnegative upper endpoint bound",
            )),
        _ => Err(BinomialSummationError::Unsupported(
            "a symbolic or noninteger upper endpoint",
        )),
    }
}

struct Normalized {
    polynomial: Coefficients,
    a: BigRational,
    b: BigRational,
}

fn normalize(
    source: &Expr,
    index: &Symbol,
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Normalized> {
    work.allocate(1, (MAX_NODES * std::mem::size_of::<&Expr>()) as u64)?;
    let mut pending = Vec::with_capacity(MAX_NODES);
    pending.push(source);
    let mut binomials = 0usize;
    work.allocate(1, 64)?;
    let mut weight = vec![BigRational::one()];
    let (mut a, mut b) = (BigRational::one(), BigRational::one());
    while let Some(part) = pending.pop() {
        work.allocate(1, 64)?;
        if let Expr::Mul(children) = part {
            pending.extend(children.iter());
            continue;
        }
        if let Expr::Function(name, args) = part
            && name == "binomial"
        {
            if args.len() != 2
                || !matches!(&args[0], Expr::Integer(value) if value.to_i64() == Some(n as i64))
                || !matches!(&args[1], Expr::Sym(symbol) if symbol == index)
            {
                return Err(BinomialSummationError::Unsupported(
                    "a binomial with different typed index or upper argument",
                ));
            }
            binomials += 1;
            continue;
        }
        if let Expr::Pow(base, exponent) = part
            && let Some(ratio) = number(base, work)?
        {
            let powers = polynomial(exponent, index, work)?;
            if powers == [BigRational::zero(), BigRational::one()] {
                a = work.mul(&a, &ratio)?;
                continue;
            }
            if powers == [integer(n), BigRational::from_integer(BigInt::from(-1))] {
                b = work.mul(&b, &ratio)?;
                continue;
            }
        }
        let next = polynomial(part, index, work)?;
        weight = mul_poly(&weight, &next, work)?;
    }
    if binomials != 1 {
        return Err(BinomialSummationError::Unsupported(
            "a single binomial factor",
        ));
    }
    Ok(Normalized {
        polynomial: weight,
        a,
        b,
    })
}

fn evaluate(
    poly: &[BigRational],
    point: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let mut result = BigRational::zero();
    for coefficient in poly.iter().rev() {
        let product = work.mul(&result, point)?;
        result = work.add(&product, coefficient)?;
    }
    Ok(result)
}

/// Candidate generation only: Newton forward differences divided by j!.
fn generate_coefficients(
    poly: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    work.allocate(poly.len() * 2, (poly.len() * 128) as u64)?;
    let mut differences = (0..poly.len())
        .map(|j| evaluate(poly, &integer(j), work))
        .collect::<Result<Vec<_>>>()?;
    let mut factorial = BigRational::one();
    let mut result = Vec::with_capacity(poly.len());
    for order in 0..poly.len() {
        result.push(work.div(&differences[0], &factorial)?);
        for j in 0..differences.len() - 1 {
            differences[j] = work.sub(&differences[j + 1], &differences[j])?;
        }
        differences.pop();
        factorial = work.mul(&factorial, &integer(order + 1))?;
    }
    trim(&mut result);
    Ok(result)
}

/// Candidate value by a recurrence between consecutive nonzero-sum moments.
fn generate_value(
    coefficients: &[BigRational],
    a: &BigRational,
    b: &BigRational,
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let sum = work.add(a, b)?;
    if sum.is_zero() {
        if n >= coefficients.len() {
            return Ok(BigRational::zero());
        }
        let mut factorial = BigRational::one();
        for factor in 1..=n {
            factorial = work.mul(&factorial, &integer(factor))?;
        }
        let power = work.pow(a, n as i32)?;
        let moment = work.mul(&factorial, &power)?;
        return work.mul(&coefficients[n], &moment);
    }
    let count = coefficients.len().min(n + 1);
    let mut moment = work.pow(&sum, n as i32)?;
    let mut value = BigRational::zero();
    for (j, coefficient) in coefficients.iter().take(count).enumerate() {
        let term = work.mul(coefficient, &moment)?;
        value = work.add(&value, &term)?;
        if j + 1 < count {
            let next = work.mul(&moment, &integer(n - j))?;
            let next = work.mul(&next, a)?;
            moment = work.div(&next, &sum)?;
        }
    }
    Ok(value)
}

/// Checker only: expand each falling-factorial product coefficientwise.
fn reconstruct_falling_polynomial(
    coefficients: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Coefficients> {
    if coefficients.is_empty()
        || coefficients.len() > MAX_DEGREE + 1
        || (coefficients.len() > 1 && coefficients.last().is_some_and(Zero::is_zero))
        || coefficients.iter().any(|v| bits(v) > MAX_WORK_BITS)
    {
        return Err(BinomialSummationError::InvalidCertificate);
    }
    work.allocate(coefficients.len() * 2, (coefficients.len() * 128) as u64)?;
    let mut reconstructed = vec![BigRational::zero(); coefficients.len()];
    let mut falling = vec![BigRational::one()];
    for (order, coefficient) in coefficients.iter().enumerate() {
        for (power, factor) in falling.iter().enumerate() {
            let term = work.mul(coefficient, factor)?;
            reconstructed[power] = work.add(&reconstructed[power], &term)?;
        }
        if order + 1 < coefficients.len() {
            work.allocate(falling.len() + 1, ((falling.len() + 1) * 64) as u64)?;
            let mut next = vec![BigRational::zero(); falling.len() + 1];
            for (power, factor) in falling.iter().enumerate() {
                let offset = work.mul(&integer(order), factor)?;
                next[power] = work.sub(&next[power], &offset)?;
                next[power + 1] = work.add(&next[power + 1], factor)?;
            }
            falling = next;
        }
    }
    trim(&mut reconstructed);
    Ok(reconstructed)
}

/// Verifier-only grammar and syntactic degree proof. Every held child is
/// visited, even if a sibling is zero or an outer power is zero. Cancellation
/// never lowers this independently proved bound.
fn checked_source_degree(
    source: &Expr,
    index: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<usize> {
    work.allocate(1, 64)?;
    let degree = match source {
        Expr::Integer(_) | Expr::Rational(_) => 0,
        Expr::Sym(symbol) if symbol == index => 1,
        Expr::Add(children) => {
            let mut maximum = 0;
            for child in children {
                maximum = maximum.max(checked_source_degree(child, index, work)?);
            }
            maximum
        }
        Expr::Mul(children) => {
            let mut total = 0usize;
            for child in children {
                total = total.saturating_add(checked_source_degree(child, index, work)?);
            }
            total
        }
        Expr::Pow(base, exponent) => {
            let base_degree = checked_source_degree(base, index, work)?;
            let power = match exponent.as_ref() {
                Expr::Integer(value) => value.to_i64().and_then(|v| i32::try_from(v).ok()),
                _ => None,
            }
            .ok_or(BinomialSummationError::Unsupported(
                "noninteger original weight power",
            ))?;
            if base_degree == 0 {
                // A negative constant power is permitted only after its
                // original constant expression has been checked for zero.
                let value = evaluate_original_weight(base, index, &BigRational::zero(), work)?;
                work.pow(&value, power)?;
                0
            } else {
                if power < 0 {
                    return Err(BinomialSummationError::Unsupported(
                        "original weight may have poles",
                    ));
                }
                base_degree.saturating_mul(power as usize)
            }
        }
        _ => {
            return Err(BinomialSummationError::Unsupported(
                "original weight is not in QQ[k]",
            ));
        }
    };
    if degree > MAX_DEGREE {
        return Err(BinomialSummationError::Refused(
            "original polynomial degree bound",
        ));
    }
    Ok(degree)
}

/// Verifier-only evaluation of the original expression tree. This does not
/// call the generator's coefficient conversion or polynomial evaluation.
fn evaluate_original_weight(
    source: &Expr,
    index: &Symbol,
    point: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    work.allocate(1, 64)?;
    match source {
        Expr::Integer(value) => {
            work.allocate(1, value.bits().div_ceil(8) + 64)?;
            Ok(BigRational::from_integer(value.clone()))
        }
        Expr::Rational(value) => work.copy(value),
        Expr::Sym(symbol) if symbol == index => work.copy(point),
        Expr::Add(children) => {
            let mut value = BigRational::zero();
            for child in children {
                let next = evaluate_original_weight(child, index, point, work)?;
                value = work.add(&value, &next)?;
            }
            Ok(value)
        }
        Expr::Mul(children) => {
            let mut value = BigRational::one();
            for child in children {
                let next = evaluate_original_weight(child, index, point, work)?;
                value = work.mul(&value, &next)?;
            }
            Ok(value)
        }
        Expr::Pow(base, exponent) => {
            let base = evaluate_original_weight(base, index, point, work)?;
            let power = match exponent.as_ref() {
                Expr::Integer(value) => value.to_i64().and_then(|v| i32::try_from(v).ok()),
                _ => None,
            }
            .ok_or(BinomialSummationError::Unsupported(
                "noninteger original weight power",
            ))?;
            work.pow(&base, power)
        }
        _ => Err(BinomialSummationError::Unsupported(
            "original weight is not in QQ[k]",
        )),
    }
}

/// Certify that an original exponent is exactly k or n-k. Degree-proof plus
/// max(degree,1)+1 evaluations prevents a higher-degree impostor matching at
/// just zero and one. Only the EXPONENT is sampled: zero weighted bases are
/// never raised to out-of-range negative powers.
fn checked_exponent_kind(
    exponent: &Expr,
    index: &Symbol,
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Option<bool>> {
    let degree = checked_source_degree(exponent, index, work)?;
    let (mut forward, mut reverse) = (true, true);
    for point in 0..=degree.max(1) {
        let position = integer(point);
        let value = evaluate_original_weight(exponent, index, &position, work)?;
        let opposite = work.sub(&integer(n), &position)?;
        forward &= value == position;
        reverse &= value == opposite;
    }
    Ok(if forward {
        Some(true)
    } else if reverse {
        Some(false)
    } else {
        None
    })
}

/// Independent source classification and exact polynomial identity proof.
/// The generator's normalize(), polynomial(), and evaluate() are never used.
fn verify_original_source(
    source: &Expr,
    index: &Symbol,
    n: usize,
    coefficients: &[BigRational],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(BigRational, BigRational)> {
    let reconstructed = reconstruct_falling_polynomial(coefficients, work)?;
    work.allocate(2, (2 * MAX_NODES * std::mem::size_of::<&Expr>()) as u64)?;
    let mut pending = Vec::with_capacity(MAX_NODES);
    let mut weights = Vec::with_capacity(MAX_NODES);
    pending.push(source);
    let (mut a, mut b) = (BigRational::one(), BigRational::one());
    let (mut count, mut degree) = (0usize, 0usize);
    while let Some(node) = pending.pop() {
        work.allocate(1, 64)?;
        match node {
            Expr::Mul(children) => {
                pending.extend(children.iter());
            }
            Expr::Function(name, args) if name == "binomial" => {
                match args.as_slice() {
                    [Expr::Integer(top), Expr::Sym(variable)]
                        if top.to_i64() == Some(n as i64) && variable == index => {}
                    _ => {
                        return Err(BinomialSummationError::Unsupported(
                            "original binomial arguments differ from the range",
                        ));
                    }
                }
                count += 1;
            }
            _ => {
                if let Expr::Pow(base, exponent) = node
                    && matches!(base.as_ref(), Expr::Integer(_) | Expr::Rational(_))
                    && let Some(forward) = checked_exponent_kind(exponent, index, n, work)?
                {
                    let ratio = evaluate_original_weight(base, index, &BigRational::zero(), work)?;
                    if forward {
                        a = work.mul(&a, &ratio)?;
                    } else {
                        b = work.mul(&b, &ratio)?;
                    }
                    continue;
                }
                degree = degree.saturating_add(checked_source_degree(node, index, work)?);
                if degree > MAX_DEGREE {
                    return Err(BinomialSummationError::Refused(
                        "original polynomial degree bound",
                    ));
                }
                weights.push(node);
            }
        }
    }
    if count != 1 {
        return Err(BinomialSummationError::Unsupported(
            "a single original binomial factor",
        ));
    }
    // A polynomial of degree <=d is uniquely determined by d+1 distinct
    // rational points. Include the candidate's degree even if the source is
    // constant, so a forged high-degree vanishing factor cannot pass.
    let comparison_degree = degree.max(reconstructed.len() - 1);
    for sample in 0..=comparison_degree {
        let point = integer(sample);
        let mut original = BigRational::one();
        for weight in &weights {
            let value = evaluate_original_weight(weight, index, &point, work)?;
            original = work.mul(&original, &value)?;
        }
        let (mut expected, mut monomial) = (BigRational::zero(), BigRational::one());
        for coefficient in &reconstructed {
            let term = work.mul(coefficient, &monomial)?;
            expected = work.add(&expected, &term)?;
            monomial = work.mul(&monomial, &point)?;
        }
        if original != expected {
            return Err(BinomialSummationError::InvalidCertificate);
        }
    }
    Ok((a, b))
}

/// Checker only: evaluate each exact finite binomial-theorem moment directly.
fn verified_value(
    coefficients: &[BigRational],
    a: &BigRational,
    b: &BigRational,
    n: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigRational> {
    let sum = work.add(a, b)?;
    let mut value = BigRational::zero();
    let mut falling = BigRational::one();
    for (j, coefficient) in coefficients.iter().take(n + 1).enumerate() {
        if j > 0 {
            falling = work.mul(&falling, &integer(n + 1 - j))?;
        }
        if coefficient.is_zero() {
            continue;
        }
        let left = work.pow(a, j as i32)?;
        let right = work.pow(&sum, (n - j) as i32)?;
        let moment = work.mul(&falling, &left)?;
        let moment = work.mul(&moment, &right)?;
        let term = work.mul(coefficient, &moment)?;
        value = work.add(&value, &term)?;
    }
    Ok(value)
}

fn verify_with_work(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
    certificate: PolynomialBinomialCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedPolynomialBinomialSum> {
    work.checkpoint()?;
    if !admit(source, index, work)? || !admit(&certificate.source, &certificate.index, work)? {
        return Err(BinomialSummationError::InvalidCertificate);
    }
    let n = upper_endpoint(lower, upper)?;
    let bound_n = upper_endpoint(&certificate.lower, &certificate.upper)?;
    if n != bound_n
        || certificate.source != *source
        || certificate.index != *index
        || certificate.lower != *lower
        || certificate.upper != *upper
        || bits(&certificate.claimed_sum) > MAX_WORK_BITS
    {
        return Err(BinomialSummationError::InvalidCertificate);
    }
    let (a, b) = verify_original_source(source, index, n, &certificate.falling_coefficients, work)?;
    let value = verified_value(&certificate.falling_coefficients, &a, &b, n, work)?;
    if value != certificate.claimed_sum {
        return Err(BinomialSummationError::InvalidCertificate);
    }
    let value = rational_expr(work.copy(&value)?);
    work.checkpoint()?;
    Ok(VerifiedPolynomialBinomialSum { certificate, value })
}

/// Verify a caller's untrusted certificate without invoking its generator.
pub fn verify_polynomial_binomial_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
    certificate: PolynomialBinomialCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialBinomialSum> {
    verify_with_work(
        source,
        index,
        lower,
        upper,
        certificate,
        &mut Work::new(meter),
    )
}

/// Generate and verify on one cumulative caller meter, with no fallback reset.
pub fn metered_polynomial_binomial_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
    meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedPolynomialBinomialSum>> {
    let mut work = Work::new(meter);
    work.checkpoint()?;
    if !matches!(lower, Expr::Integer(value) if value.is_zero())
        || !matches!(upper, Expr::Integer(value) if !value.is_negative())
    {
        return Ok(None);
    }
    if !admit(source, index, &mut work)? {
        return Ok(None);
    }
    if !selects_full_range_factor(source, index, upper, &mut work)? {
        return Ok(None);
    }
    let n = upper_endpoint(lower, upper)?;
    let normalized = normalize(source, index, n, &mut work)?;
    let falling_coefficients = generate_coefficients(&normalized.polynomial, &mut work)?;
    let claimed_sum = generate_value(
        &falling_coefficients,
        &normalized.a,
        &normalized.b,
        n,
        &mut work,
    )?;
    // Charge the complete bounded ownership copy before certificate creation.
    admit(source, index, &mut work)?;
    work.allocate(3, index.name.len() as u64 + 384)?;
    work.checkpoint()?;
    let certificate = PolynomialBinomialCertificate {
        source: source.clone(),
        index: index.clone(),
        lower: lower.clone(),
        upper: upper.clone(),
        falling_coefficients,
        claimed_sum,
    };
    let verified = verify_with_work(source, index, lower, upper, certificate, &mut work)?;
    work.checkpoint()?;
    Ok(Some(verified))
}

pub fn polynomial_binomial_sum(
    source: &Expr,
    index: &Symbol,
    lower: &Expr,
    upper: &Expr,
) -> Result<Option<VerifiedPolynomialBinomialSum>> {
    metered_polynomial_binomial_sum(source, index, lower, upper, &mut Unbounded)
}
