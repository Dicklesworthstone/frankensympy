//! Polynomial forcing for complete rational two-component ODE systems.
//!
//! Generation treats invertible, rank-one, and nilpotent coefficient matrices
//! separately. Acceptance does not repeat those formulas: every coefficient
//! of p' - A*p must equal the original forcing coefficient. The final family
//! is h + p - Phi*p(t0), where h and the normalized Phi already passed the
//! homogeneous system verifier. This exact construction preserves every
//! original initial coordinate and adds every homogeneous solution.
//!
//! The forcing has degree at most 16 and rational coefficients of at most
//! 1024 bits. A particular solution may have degree 18. Generator and verifier
//! use the parent's independent cumulative caps, including all composition
//! work. Final expressions are assembled directly from accepted components;
//! no unchecked final simplification is used. This is an in-process receipt.

use super::{
    Capped, LinearOdeSystemError, MAX_DERIVED_BITS, MAX_INPUT_BITS, MAX_TEXT, RationalMatrix2,
    Result, VerifiedLinearOdeSystem, admit_inputs, bits, charge, metered_linear_ode_system,
    product, rational_expr, scalar, sum, validate_number,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::Zero;
use std::sync::Arc;

const MAX_FORCING_DEGREE: usize = 16;
const MAX_PARTICULAR_DEGREE: usize = MAX_FORCING_DEGREE + 2;
const MAX_OUTPUT_NODES: usize = 1024;

/// Ascending rational coefficients of the two forcing coordinates.
/// Each coordinate is nonempty and has no trailing zero unless it is [0].
pub type PolynomialVector2 = [Vec<BigRational>; 2];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialOdeParticularCertificate {
    pub particular: PolynomialVector2,
}

/// An accepted coefficient identity in the universal univariate ring Q[t].
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialOdeParticular {
    coefficients: RationalMatrix2,
    forcing: PolynomialVector2,
    particular: PolynomialVector2,
}

impl VerifiedPolynomialOdeParticular {
    pub fn coefficients(&self) -> &RationalMatrix2 {
        &self.coefficients
    }
    pub fn forcing(&self) -> &PolynomialVector2 {
        &self.forcing
    }
    pub fn particular(&self) -> &PolynomialVector2 {
        &self.particular
    }
}

/// A complete affine family, including exact full or partial rational IVPs.
#[derive(Debug, Clone)]
pub struct VerifiedPolynomialOdeSystem {
    homogeneous: VerifiedLinearOdeSystem,
    particular: VerifiedPolynomialOdeParticular,
    solutions: [Expr; 2],
}

impl VerifiedPolynomialOdeSystem {
    pub fn homogeneous(&self) -> &VerifiedLinearOdeSystem {
        &self.homogeneous
    }
    pub fn particular(&self) -> &VerifiedPolynomialOdeParticular {
        &self.particular
    }
    pub fn solutions(&self) -> &[Expr; 2] {
        &self.solutions
    }
    pub fn into_solutions(self) -> [Expr; 2] {
        self.solutions
    }
}

fn admit_polynomials(
    values: &PolynomialVector2,
    degree: usize,
    maximum_bits: u64,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    meter.checkpoint()?;
    for coordinate in values {
        if coordinate.is_empty()
            || coordinate.len() > degree + 1
            || (coordinate.len() > 1 && coordinate.last().is_some_and(Zero::is_zero))
        {
            return Err(LinearOdeSystemError::InvalidSource(
                "polynomial forcing/particular degree or canonical coefficient shape",
            ));
        }
        for value in coordinate {
            validate_number(value, maximum_bits)?;
        }
    }
    charge(
        meter,
        values.iter().map(Vec::len).sum(),
        values
            .iter()
            .flatten()
            .map(|value| 128 + 2 * bits(value).div_ceil(8) as usize)
            .sum(),
    )
}

fn preflight(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<()> {
    meter.checkpoint()?;
    if bits(a).saturating_add(bits(b)).saturating_add(2) > MAX_DERIVED_BITS {
        return Err(LinearOdeSystemError::ResourceLimit(
            "polynomial forcing intermediate rational growth",
        ));
    }
    Ok(())
}

fn add(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    preflight(a, b, meter)?;
    scalar(a.metered_add(b, meter)?, meter)
}
fn sub(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    preflight(a, b, meter)?;
    scalar(a.metered_sub(b, meter)?, meter)
}
fn mul(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    preflight(a, b, meter)?;
    scalar(a.metered_mul(b, meter)?, meter)
}
fn div(a: &BigRational, b: &BigRational, meter: &mut impl BudgetMeter) -> Result<BigRational> {
    preflight(a, b, meter)?;
    scalar(a.metered_div(b, meter)?, meter)
}

fn zero_vector(size: usize, meter: &mut impl BudgetMeter) -> Result<PolynomialVector2> {
    charge(meter, 2 * size, 256 * size)?;
    Ok(std::array::from_fn(|_| vec![BigRational::zero(); size]))
}

fn coefficient(values: &PolynomialVector2, row: usize, degree: usize) -> &BigRational {
    static ZERO: std::sync::LazyLock<BigRational> = std::sync::LazyLock::new(BigRational::zero);
    values[row].get(degree).unwrap_or(&ZERO)
}

fn dot(
    row: &[BigRational; 2],
    values: [&BigRational; 2],
    meter: &mut impl BudgetMeter,
) -> Result<BigRational> {
    let first = mul(&row[0], values[0], meter)?;
    let second = mul(&row[1], values[1], meter)?;
    add(&first, &second, meter)
}

fn generate(
    matrix: &RationalMatrix2,
    forcing: &PolynomialVector2,
    meter: &mut impl BudgetMeter,
) -> Result<PolynomialOdeParticularCertificate> {
    let degree = forcing.iter().map(Vec::len).max().expect("two rows") - 1;
    let ad = mul(&matrix[0][0], &matrix[1][1], meter)?;
    let bc = mul(&matrix[0][1], &matrix[1][0], meter)?;
    let determinant = sub(&ad, &bc, meter)?;
    let trace = add(&matrix[0][0], &matrix[1][1], meter)?;
    let mut particular;
    if !determinant.is_zero() {
        let negative_b = sub(&BigRational::zero(), &matrix[0][1], meter)?;
        let negative_c = sub(&BigRational::zero(), &matrix[1][0], meter)?;
        let inverse = [
            [
                div(&matrix[1][1], &determinant, meter)?,
                div(&negative_b, &determinant, meter)?,
            ],
            [
                div(&negative_c, &determinant, meter)?,
                div(&matrix[0][0], &determinant, meter)?,
            ],
        ];
        particular = zero_vector(degree + 1, meter)?;
        for power in (0..=degree).rev() {
            let factor = BigRational::from((power + 1) as i64);
            let mut residual = [BigRational::zero(), BigRational::zero()];
            for (row, value) in residual.iter_mut().enumerate() {
                let derivative = mul(&factor, coefficient(&particular, row, power + 1), meter)?;
                *value = sub(&derivative, coefficient(forcing, row, power), meter)?;
            }
            for (row, inverse_row) in inverse.iter().enumerate() {
                particular[row][power] = dot(inverse_row, [&residual[0], &residual[1]], meter)?;
            }
        }
    } else if !trace.is_zero() {
        // Cayley-Hamilton gives A^2=trace*A. The projectors A/trace and
        // I-A/trace split the nonzero eigenvalue from the kernel.
        let projection = [
            [
                div(&matrix[0][0], &trace, meter)?,
                div(&matrix[0][1], &trace, meter)?,
            ],
            [
                div(&matrix[1][0], &trace, meter)?,
                div(&matrix[1][1], &trace, meter)?,
            ],
        ];
        let mut projected = zero_vector(degree + 1, meter)?;
        for (row, coordinate) in projected.iter_mut().enumerate() {
            for (power, value) in coordinate.iter_mut().enumerate() {
                *value = dot(
                    &projection[row],
                    [
                        coefficient(forcing, 0, power),
                        coefficient(forcing, 1, power),
                    ],
                    meter,
                )?;
            }
        }
        particular = zero_vector(degree + 2, meter)?;
        for power in (0..=degree).rev() {
            let factor = BigRational::from((power + 1) as i64);
            for (row, coordinate) in particular.iter_mut().enumerate() {
                let derivative = mul(&factor, &coordinate[power + 1], meter)?;
                let residual = sub(&derivative, &projected[row][power], meter)?;
                coordinate[power] = div(&residual, &trace, meter)?;
            }
        }
        for power in 0..=degree {
            let divisor = BigRational::from((power + 1) as i64);
            for (row, coordinate) in particular.iter_mut().enumerate() {
                let kernel = sub(
                    coefficient(forcing, row, power),
                    &projected[row][power],
                    meter,
                )?;
                let integral = div(&kernel, &divisor, meter)?;
                coordinate[power + 1] = add(&coordinate[power + 1], &integral, meter)?;
            }
        }
    } else {
        // A^2=0, including A=0: integrate b once and A*b twice.
        particular = zero_vector(degree + 3, meter)?;
        for power in 0..=degree {
            let once = BigRational::from((power + 1) as i64);
            let twice = BigRational::from(((power + 1) * (power + 2)) as i64);
            for (row, matrix_row) in matrix.iter().enumerate() {
                let integral = div(coefficient(forcing, row, power), &once, meter)?;
                particular[row][power + 1] = add(&particular[row][power + 1], &integral, meter)?;
                let transformed = dot(
                    matrix_row,
                    [
                        coefficient(forcing, 0, power),
                        coefficient(forcing, 1, power),
                    ],
                    meter,
                )?;
                let integral = div(&transformed, &twice, meter)?;
                particular[row][power + 2] = add(&particular[row][power + 2], &integral, meter)?;
            }
        }
    }
    for coordinate in &mut particular {
        while coordinate.len() > 1 && coordinate.last().is_some_and(Zero::is_zero) {
            coordinate.pop();
        }
    }
    meter.checkpoint()?;
    Ok(PolynomialOdeParticularCertificate { particular })
}

/// Check every original forcing coefficient without invoking the generator.
pub fn verify_polynomial_ode_particular(
    matrix: &RationalMatrix2,
    forcing: &PolynomialVector2,
    candidate: PolynomialOdeParticularCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialOdeParticular> {
    let mut meter = Capped {
        inner: verifier,
        used: [0; 3],
    };
    for value in matrix.iter().flatten() {
        validate_number(value, MAX_INPUT_BITS)?;
    }
    admit_polynomials(forcing, MAX_FORCING_DEGREE, MAX_INPUT_BITS, &mut meter)?;
    admit_polynomials(
        &candidate.particular,
        MAX_PARTICULAR_DEGREE,
        MAX_DERIVED_BITS,
        &mut meter,
    )?;
    let size = forcing
        .iter()
        .chain(candidate.particular.iter())
        .map(Vec::len)
        .max()
        .expect("four rows");
    for power in 0..size {
        let factor = BigRational::from((power + 1) as i64);
        for (row, matrix_row) in matrix.iter().enumerate() {
            let derivative = mul(
                &factor,
                coefficient(&candidate.particular, row, power + 1),
                &mut meter,
            )?;
            let transformed = dot(
                matrix_row,
                [
                    coefficient(&candidate.particular, 0, power),
                    coefficient(&candidate.particular, 1, power),
                ],
                &mut meter,
            )?;
            if sub(&derivative, &transformed, &mut meter)? != *coefficient(forcing, row, power) {
                return Err(LinearOdeSystemError::Verification(
                    "polynomial particular coefficient identity",
                ));
            }
        }
    }
    charge(&mut meter, 8, 8192)?;
    let accepted = VerifiedPolynomialOdeParticular {
        coefficients: matrix.clone(),
        forcing: forcing.clone(),
        particular: candidate.particular,
    };
    meter.checkpoint()?;
    Ok(accepted)
}

fn evaluate(
    poly: &[BigRational],
    point: &BigRational,
    meter: &mut impl BudgetMeter,
) -> Result<BigRational> {
    let mut value = BigRational::zero();
    for coefficient in poly.iter().rev() {
        value = mul(&value, point, meter)?;
        value = add(&value, coefficient, meter)?;
    }
    Ok(value)
}

fn expression(poly: &[BigRational], time: &Symbol, meter: &mut impl BudgetMeter) -> Result<Expr> {
    charge(meter, 6 * poly.len(), poly.len() * (4096 + time.name.len()))?;
    let mut terms = Vec::new();
    for (power, coefficient) in poly.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let value = rational_expr(coefficient.clone());
        terms.push(if power == 0 {
            value
        } else {
            let monomial = if power == 1 {
                Expr::Sym(time.clone())
            } else {
                Expr::Pow(
                    Arc::new(Expr::Sym(time.clone())),
                    Arc::new(Expr::from_i64(power as i64)),
                )
            };
            product(vec![value, monomial])
        });
    }
    Ok(sum(terms))
}

/// The output is constructed only from accepted components and positive
/// integer monomials. This borrowed pass bounds publication of both coordinates.
fn prepay_output(
    expression: &Expr,
    remaining: &mut usize,
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let mut pending = vec![(expression, 0usize)];
    let mut bytes = 0usize;
    let mut text = 0usize;
    let mut nodes = 0usize;
    while let Some((node, depth)) = pending.pop() {
        if *remaining == 0 || depth > 64 {
            return Err(LinearOdeSystemError::ResourceLimit(
                "complete forced-system output shape",
            ));
        }
        *remaining -= 1;
        nodes += 1;
        bytes += 128;
        match node {
            Expr::Integer(value) => {
                if value.bits() > MAX_DERIVED_BITS {
                    return Err(LinearOdeSystemError::ResourceLimit(
                        "forced-system output coefficient",
                    ));
                }
                bytes += value.bits().div_ceil(8) as usize;
            }
            Expr::Rational(value) => {
                validate_number(value, MAX_DERIVED_BITS)?;
                bytes += 2 * bits(value).div_ceil(8) as usize;
            }
            Expr::Sym(symbol) => text += symbol.name.len(),
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if children.len() > *remaining || children.len() > 32 {
                    return Err(LinearOdeSystemError::ResourceLimit(
                        "forced-system output arity",
                    ));
                }
                if let Expr::Function(name, _) = node {
                    text += name.len();
                }
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Const(_) => {
                return Err(LinearOdeSystemError::Verification(
                    "unexpected forced-system output constant",
                ));
            }
        }
        if text > MAX_TEXT {
            return Err(LinearOdeSystemError::ResourceLimit(
                "forced-system output text",
            ));
        }
    }
    charge(meter, nodes, bytes + text)
}

/// Solve bounded polynomial forcing with complete general/partial/full data.
#[allow(clippy::too_many_arguments)]
pub fn metered_polynomial_ode_system(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    forcing: &PolynomialVector2,
    initial: [&Expr; 2],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPolynomialOdeSystem> {
    let mut generation = Capped {
        inner: generator,
        used: [0; 3],
    };
    let mut verification = Capped {
        inner: verifier,
        used: [0; 3],
    };
    admit_inputs(matrix, time, point, &mut generation)?;
    admit_polynomials(forcing, MAX_FORCING_DEGREE, MAX_INPUT_BITS, &mut generation)?;
    let homogeneous = metered_linear_ode_system(
        matrix,
        time,
        point,
        initial,
        &mut generation,
        &mut verification,
    )?;
    let candidate = generate(matrix, forcing, &mut generation)?;
    let particular =
        verify_polynomial_ode_particular(matrix, forcing, candidate, &mut verification)?;
    let values_at_point = [
        evaluate(&particular.particular[0], point, &mut verification)?,
        evaluate(&particular.particular[1], point, &mut verification)?,
    ];
    let fundamental = homogeneous.fundamental().fundamental();
    let mut solutions = [Expr::from_i64(0), Expr::from_i64(0)];
    let mut remaining = MAX_OUTPUT_NODES;
    for (row, solution) in solutions.iter_mut().enumerate() {
        // All clones and arithmetic nodes are prepaid. The literal source
        // coefficients already passed the polynomial identity checker.
        charge(&mut verification, 256, 512 * 1024)?;
        let polynomial = expression(&particular.particular[row], time, &mut verification)?;
        let correction = sum((0..2)
            .map(|column| {
                product(vec![
                    fundamental[row][column].clone(),
                    rational_expr(values_at_point[column].clone()),
                ])
            })
            .collect());
        *solution = sum(vec![
            homogeneous.solutions()[row].clone(),
            polynomial,
            product(vec![Expr::from_i64(-1), correction]),
        ]);
        prepay_output(solution, &mut remaining, &mut verification)?;
    }
    let result = VerifiedPolynomialOdeSystem {
        homogeneous,
        particular,
        solutions,
    };
    generation.checkpoint()?;
    verification.checkpoint()?;
    Ok(result)
}

pub fn polynomial_ode_system(
    matrix: &RationalMatrix2,
    time: &Symbol,
    point: &BigRational,
    forcing: &PolynomialVector2,
    initial: [&Expr; 2],
) -> Result<VerifiedPolynomialOdeSystem> {
    metered_polynomial_ode_system(
        matrix,
        time,
        point,
        forcing,
        initial,
        &mut Unbounded,
        &mut Unbounded,
    )
}
