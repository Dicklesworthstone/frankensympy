//! Normalized half-line sine and cosine integrals.
//!
//! Both directions use sqrt(2/pi) and the angular kernel sin(k*x) or cos(k*x),
//! matching the executable SymPy 1.14 classes (their prose contains a 2*pi
//! typo). This bounded catalog shares original-source admission, exact
//! arithmetic and one cumulative work ledger with the ordinary Fourier lane.
//! It does not produce a portable mathematical certificate.

use super::{
    FourierDirection, FourierError, Polynomial, Result, Work, number, polynomial,
    polynomial_product, preflight,
};
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigRational, Constant, Expr, Symbol};
use num_traits::{One, Signed, Zero};

/// The kernel of the normalized half-line integral; its inverse uses the same kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfLineKernel {
    Sine,
    Cosine,
}

/// A bounded direct integral identity, with its output-domain requirement.
#[derive(Debug)]
pub struct OrdinaryHalfLineTransform {
    pub value: Expr,
    pub kernel: HalfLineKernel,
    /// Exponential tails need a real output coordinate here. Complex strips
    /// are outside this catalog; Gaussian outputs are entire.
    pub requires_real_frequency: bool,
}

/// Compute sqrt(2/pi) times the integral on [0, infinity).
///
/// Supported sources are finite linear combinations of p(x)*exp(c-a*x),
/// with rational a>0, c and degree(p)<=2, plus even quadratic weights for
/// exp(c-a*x^2) under cosine and linear odd weights under sine. Multipliers
/// may contain bounded positive rational powers of positive constants and pi.
/// All original children are admitted before zero coefficients are removed.
/// Unsupported sources are distinct from exhaustion, cancellation and faults.
pub fn metered_half_line_transform<M: BudgetMeter>(
    source: &Expr,
    variable: &Symbol,
    frequency: &Symbol,
    kernel: HalfLineKernel,
    meter: &mut M,
) -> Result<OrdinaryHalfLineTransform> {
    let mut work = Work::new(meter, FourierDirection::Forward);
    preflight(source, variable, frequency, &mut work)?;
    let mut requires_real_frequency = false;
    let value = transform(
        source,
        variable,
        frequency,
        kernel,
        &mut requires_real_frequency,
        &mut work,
    )?;
    work.checkpoint()?;
    Ok(OrdinaryHalfLineTransform {
        value,
        kernel,
        requires_real_frequency,
    })
}

/// Rebuild only manifestly positive, finite constants. This permits Gaussian
/// round trips without admitting symbolic signs or complex power branches.
fn positive_scalar<M: BudgetMeter>(
    source: &Expr,
    work: &mut Work<'_, M>,
) -> Result<Option<Expr>> {
    if let Some(value) = number(source, work)? {
        return if value.is_positive() {
            Ok(Some(work.scalar(&value)?))
        } else {
            Ok(None)
        };
    }
    match source {
        Expr::Const(Constant::Pi) => Ok(Some(work.constant(Constant::Pi)?)),
        Expr::Add(children) | Expr::Mul(children) if !children.is_empty() => {
            work.allocate(children.len(), 128 * children.len() as u64)?;
            let mut values = Vec::with_capacity(children.len());
            for child in children {
                let Some(value) = positive_scalar(child, work)? else {
                    return Ok(None);
                };
                values.push(value);
            }
            Ok(Some(if matches!(source, Expr::Add(_)) {
                work.sum(values)?
            } else {
                work.product(values)?
            }))
        }
        Expr::Pow(base, exponent) => {
            let Some(exponent) = number(exponent, work)? else {
                return Ok(None);
            };
            if exponent
                .numer()
                .to_i64()
                .is_none_or(|value| value.unsigned_abs() > 16)
                || exponent.denom().to_u64().is_none_or(|value| value > 16)
            {
                return Err(FourierError::Refused("constant power bound"));
            }
            let Some(base) = positive_scalar(base, work)? else {
                return Ok(None);
            };
            Ok(Some(work.power(base, exponent)?))
        }
        _ => Ok(None),
    }
}

fn frequency_polynomial<M: BudgetMeter>(
    coefficients: &[BigRational],
    frequency: &Symbol,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    work.allocate(coefficients.len(), 128 * coefficients.len() as u64)?;
    let mut terms = Vec::with_capacity(coefficients.len());
    for (degree, coefficient) in coefficients.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let coefficient = work.scalar(coefficient)?;
        if degree == 0 {
            terms.push(coefficient);
        } else {
            let coordinate = work.symbol(frequency)?;
            let power = work.power(coordinate, BigRational::from(degree as i64))?;
            terms.push(work.product(vec![coefficient, power])?);
        }
    }
    work.sum(terms)
}

fn exponential<M: BudgetMeter>(
    exponent: &[BigRational],
    weight: &[BigRational],
    frequency: &Symbol,
    kernel: HalfLineKernel,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    let a = -work.copy(&exponent[1])?;
    let a_squared = work.mul(&a, &a)?;
    let a_cubed = work.mul(&a_squared, &a)?;
    work.allocate(weight.len(), 128 * weight.len() as u64)?;
    let mut terms = Vec::with_capacity(weight.len());
    for (degree, coefficient) in weight.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        // Real/imaginary parts of n!/(a-I*k)^(n+1), n=0,1,2.
        let numerator = match (kernel, degree) {
            (HalfLineKernel::Cosine, 0) => vec![work.copy(&a)?],
            (HalfLineKernel::Sine, 0) => vec![BigRational::zero(), BigRational::one()],
            (HalfLineKernel::Cosine, 1) => {
                vec![work.copy(&a_squared)?, BigRational::zero(), BigRational::from(-1)]
            }
            (HalfLineKernel::Sine, 1) => {
                vec![BigRational::zero(), work.mul(&BigRational::from(2), &a)?]
            }
            (HalfLineKernel::Cosine, 2) => vec![
                work.mul(&BigRational::from(2), &a_cubed)?,
                BigRational::zero(),
                work.mul(&BigRational::from(-6), &a)?,
            ],
            (HalfLineKernel::Sine, 2) => vec![
                BigRational::zero(),
                work.mul(&BigRational::from(6), &a_squared)?,
                BigRational::zero(),
                BigRational::from(-2),
            ],
            _ => return Err(FourierError::Unsupported("half-line weight degree above two")),
        };
        let numerator = frequency_polynomial(&numerator, frequency, work)?;
        let denominator = frequency_polynomial(
            &[work.copy(&a_squared)?, BigRational::zero(), BigRational::one()],
            frequency,
            work,
        )?;
        let reciprocal = work.power(denominator, BigRational::from(-(degree as i64) - 1))?;
        let coefficient = work.scalar(coefficient)?;
        terms.push(work.product(vec![coefficient, numerator, reciprocal])?);
    }
    let rational = work.sum(terms)?;
    let two = work.scalar(&BigRational::from(2))?;
    let sqrt_two = work.power(two, BigRational::new(1.into(), 2.into()))?;
    let pi = work.constant(Constant::Pi)?;
    let inverse_sqrt_pi = work.power(pi, BigRational::new((-1).into(), 2.into()))?;
    let height = work.scalar(&exponent[0])?;
    let amplitude = work.exp(height)?;
    work.product(vec![sqrt_two, inverse_sqrt_pi, amplitude, rational])
}

fn gaussian<M: BudgetMeter>(
    exponent: &[BigRational],
    weight: &[BigRational],
    frequency: &Symbol,
    kernel: HalfLineKernel,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    let mut coefficients = [
        BigRational::zero(),
        BigRational::zero(),
        BigRational::zero(),
    ];
    for (target, source) in coefficients.iter_mut().zip(weight) {
        *target = work.copy(source)?;
    }
    let a = -work.copy(&exponent[2])?;
    let two_a = work.mul(&BigRational::from(2), &a)?;
    let four_a = work.mul(&BigRational::from(4), &a)?;
    let four_a_squared = work.mul(&four_a, &a)?;
    let output = match kernel {
        HalfLineKernel::Cosine if coefficients[1].is_zero() => {
            let shift = work.div(&coefficients[2], &two_a)?;
            vec![
                work.add(&coefficients[0], &shift)?,
                BigRational::zero(),
                -work.div(&coefficients[2], &four_a_squared)?,
            ]
        }
        HalfLineKernel::Sine if coefficients[0].is_zero() && coefficients[2].is_zero() => {
            vec![BigRational::zero(), work.div(&coefficients[1], &two_a)?]
        }
        _ => return Err(FourierError::Unsupported("Gaussian weight has unsupported parity")),
    };
    let output = frequency_polynomial(&output, frequency, work)?;
    let amplitude = work.scalar(&two_a)?;
    let amplitude = work.power(amplitude, BigRational::new((-1).into(), 2.into()))?;
    let decay = -work.div(&BigRational::one(), &four_a)?;
    let argument = frequency_polynomial(
        &[work.copy(&exponent[0])?, BigRational::zero(), decay],
        frequency,
        work,
    )?;
    let envelope = work.exp(argument)?;
    work.product(vec![amplitude, output, envelope])
}

fn weighted_exponential<M: BudgetMeter>(
    exponent: &Expr,
    weight: &[BigRational],
    variable: &Symbol,
    frequency: &Symbol,
    kernel: HalfLineKernel,
    requires_real: &mut bool,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    work.catalog_term()?;
    let exponent = polynomial(exponent, variable, work)?;
    match exponent.as_slice() {
        [_, slope] if slope.is_negative() => {
            *requires_real = true;
            exponential(&exponent, weight, frequency, kernel, work)
        }
        [_, linear, quadratic] if linear.is_zero() && quadratic.is_negative() => {
            gaussian(&exponent, weight, frequency, kernel, work)
        }
        _ => Err(FourierError::Unsupported(
            "half-line exponential requires positive rational decay and an admitted quadratic shape",
        )),
    }
}

fn transform<M: BudgetMeter>(
    source: &Expr,
    variable: &Symbol,
    frequency: &Symbol,
    kernel: HalfLineKernel,
    requires_real: &mut bool,
    work: &mut Work<'_, M>,
) -> Result<Expr> {
    work.charge(Dimension::ComputeSteps, 1)?;
    if let Some(value) = number(source, work)? {
        return if value.is_zero() {
            work.scalar(&value)
        } else {
            Err(FourierError::Unsupported("a nonzero constant half-line integrand"))
        };
    }
    match source {
        Expr::Add(children) => {
            work.allocate(children.len(), 128 * children.len() as u64)?;
            let mut values = Vec::with_capacity(children.len());
            for child in children {
                values.push(transform(child, variable, frequency, kernel, requires_real, work)?);
            }
            work.sum(values)
        }
        Expr::Function(name, arguments) if name == "exp" && arguments.len() == 1 => {
            weighted_exponential(
                &arguments[0],
                &[BigRational::one()],
                variable,
                frequency,
                kernel,
                requires_real,
                work,
            )
        }
        Expr::Mul(children) => {
            work.allocate(children.len() + 1, 256 * (children.len() + 1) as u64)?;
            let mut scalar = Vec::with_capacity(children.len() + 1);
            let mut dependent = Vec::with_capacity(children.len());
            for child in children {
                if let Some(value) = number(child, work)? {
                    scalar.push(work.scalar(&value)?);
                } else if let Some(value) = positive_scalar(child, work)? {
                    scalar.push(value);
                } else {
                    dependent.push(child);
                }
            }
            let value = if dependent.len() == 1 {
                transform(dependent[0], variable, frequency, kernel, requires_real, work)?
            } else {
                let mut exponent = None;
                let mut weight: Polynomial = vec![BigRational::one()];
                for factor in dependent {
                    if let Expr::Function(name, arguments) = factor
                        && name == "exp"
                        && arguments.len() == 1
                    {
                        if exponent.is_some() {
                            return Err(FourierError::Unsupported("multiple exponential factors"));
                        }
                        exponent = Some(&arguments[0]);
                    } else {
                        let factor = polynomial(factor, variable, work)?;
                        weight = polynomial_product(&weight, &factor, work)?;
                    }
                }
                let exponent = exponent.ok_or(FourierError::Unsupported(
                    "half-line products require one exponential factor",
                ))?;
                weighted_exponential(
                    exponent, &weight, variable, frequency, kernel, requires_real, work,
                )?
            };
            // Every original dependent factor has been checked even when one
            // of the scalar factors is zero.
            scalar.push(value);
            work.product(scalar)
        }
        _ => Err(FourierError::Unsupported("this half-line source grammar")),
    }
}
