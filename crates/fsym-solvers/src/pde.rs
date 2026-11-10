//! Checked characteristic families for a bounded first-order linear PDE.
//!
//! For rational constants a, b, c with (a,b) != (0,0), this lane solves
//! a*u_x + b*u_y + c*u = f(x,y) with rational polynomial forcing. Its output
//! describes exp(-c*s)*F(t) + q for an arbitrary differentiable function F.
//! The separate checker establishes Ds=1, Dt=0, an invertible affine
//! coordinate map, and every coefficient of the original residual Dq+cq-f.
//! This is a verified solution family, not a general PDE-completeness,
//! boundary-value, portable-proof, or asynchronous-runtime certificate.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{PolyError, bounded_polynomial_from_expr};
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;

const MAX_FORCING_DEGREE: u32 = 16;
const MAX_PARTICULAR_DEGREE: u32 = 17;
const MAX_TERMS: usize = 192;
const MAX_BITS: u64 = 4096;
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_ARITY: usize = 256;
const MAX_TEXT: usize = 16_384;
const MAX_POWER: i64 = 64;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

type Polynomial = BTreeMap<[u32; 2], BigRational>;
type Result<T> = std::result::Result<T, PdeError>;

#[derive(Debug, Error)]
pub enum PdeError {
    #[error("linear first-order PDE source refused: {0}")]
    InvalidSource(&'static str),
    #[error("linear first-order PDE bound exceeded: {0}")]
    ResourceLimit(&'static str),
    #[error("linear first-order PDE verification failed: {0}")]
    Verification(&'static str),
    #[error("linear first-order PDE arithmetic failed: {0}")]
    Arithmetic(String),
    #[error(transparent)]
    Polynomial(#[from] PolyError),
    #[error(transparent)]
    Meter(#[from] MeterError),
}

/// A family identity is checked; no broader PDE completeness is asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdeCompleteness {
    VerifiedSolutionFamily,
}

/// Untrusted polynomial expressions in the original two coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearFirstOrderPdeCertificate {
    pub time_coordinate: Expr,
    pub invariant_coordinate: Expr,
    pub particular: Expr,
}

/// An immutable source-bound family exp(-c*s)*F(t) + q.
#[derive(Debug, Clone)]
pub struct VerifiedLinearPde {
    coefficients: [BigRational; 3],
    variables: [Symbol; 2],
    forcing: Expr,
    certificate: LinearFirstOrderPdeCertificate,
    homogeneous_coefficient: Expr,
}

impl VerifiedLinearPde {
    pub fn coefficients(&self) -> &[BigRational; 3] {
        &self.coefficients
    }
    pub fn variables(&self) -> &[Symbol; 2] {
        &self.variables
    }
    pub fn forcing(&self) -> &Expr {
        &self.forcing
    }
    pub fn certificate(&self) -> &LinearFirstOrderPdeCertificate {
        &self.certificate
    }
    pub fn time_coordinate(&self) -> &Expr {
        &self.certificate.time_coordinate
    }
    pub fn invariant_coordinate(&self) -> &Expr {
        &self.certificate.invariant_coordinate
    }
    pub fn particular(&self) -> &Expr {
        &self.certificate.particular
    }
    pub fn homogeneous_coefficient(&self) -> &Expr {
        &self.homogeneous_coefficient
    }
    pub fn completeness(&self) -> PdeCompleteness {
        PdeCompleteness::VerifiedSolutionFamily
    }
    /// Transfer the accepted (s, t, q, exp(-c*s)) expressions to a shell.
    pub fn into_parts(self) -> (Expr, Expr, Expr, Expr) {
        (
            self.certificate.time_coordinate,
            self.certificate.invariant_coordinate,
            self.certificate.particular,
            self.homogeneous_coefficient,
        )
    }
}

struct Work<'a, M> {
    meter: &'a mut M,
    used: [u64; 3],
    stopped: Option<MeterError>,
}

impl<M: BudgetMeter> BudgetMeter for Work<'_, M> {
    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = self.meter.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        let result = (|| {
            self.checkpoint()?;
            let mut next = self.used;
            for &(dimension, amount) in charges {
                let (index, cap) = match dimension {
                    Dimension::ComputeSteps => (0, MAX_COMPUTE),
                    Dimension::MemoryBytes => (1, MAX_MEMORY),
                    Dimension::AllocationCount => (2, MAX_COMPUTE),
                    _ => continue,
                };
                let remaining = cap.saturating_sub(next[index]);
                if amount > remaining {
                    return Err(MeterError::Budget(BudgetError::Exhausted {
                        dimension,
                        requested: amount,
                        remaining,
                    }));
                }
                next[index] += amount;
            }
            self.meter.charge_batch(charges)?;
            self.used = next;
            Ok(())
        })();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

fn scalar_ok(value: &BigRational) -> Result<()> {
    if value.denom().is_zero() || value.denom().is_negative() {
        return Err(PdeError::InvalidSource("invalid rational coefficient"));
    }
    if value.numer().bits() > MAX_BITS || value.denom().bits() > MAX_BITS {
        return Err(PdeError::ResourceLimit(
            "rational coefficients exceed 4096 bits",
        ));
    }
    Ok(())
}

fn known_nonzero_literal(expression: &Expr) -> bool {
    match expression {
        Expr::Integer(value) => !value.is_zero(),
        Expr::Rational(value) => !value.is_zero(),
        _ => false,
    }
}

impl<'a, M: BudgetMeter> Work<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            used: [0; 3],
            stopped: None,
        }
    }

    fn charge_storage(&mut self, units: usize, bytes: usize) -> Result<()> {
        self.charge_batch(&[
            (Dimension::ComputeSteps, units.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])?;
        Ok(())
    }

    fn arithmetic_error(&self, error: impl std::fmt::Display) -> PdeError {
        self.stopped.map_or_else(
            || PdeError::Arithmetic(error.to_string()),
            PdeError::Meter,
        )
    }

    fn add(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        scalar_ok(a)?;
        scalar_ok(b)?;
        let value = a
            .metered_add(b, self)
            .map_err(|e| self.arithmetic_error(e))?;
        scalar_ok(&value)?;
        Ok(value)
    }

    fn mul(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        scalar_ok(a)?;
        scalar_ok(b)?;
        let value = a
            .metered_mul(b, self)
            .map_err(|e| self.arithmetic_error(e))?;
        scalar_ok(&value)?;
        Ok(value)
    }

    fn div(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational> {
        scalar_ok(a)?;
        scalar_ok(b)?;
        let value = a
            .metered_div(b, self)
            .map_err(|e| self.arithmetic_error(e))?;
        scalar_ok(&value)?;
        Ok(value)
    }

    fn admit_source(
        &mut self,
        coefficients: [&BigRational; 3],
        variables: [&Symbol; 2],
    ) -> Result<()> {
        self.checkpoint()?;
        // Inspect scalars and identifiers before cloning any source payload.
        for value in coefficients {
            scalar_ok(value)?;
        }
        if coefficients[0].is_zero() && coefficients[1].is_zero() {
            return Err(PdeError::InvalidSource(
                "at least one derivative coefficient must be nonzero",
            ));
        }
        if variables[0] == variables[1] {
            return Err(PdeError::InvalidSource(
                "the two coordinates must have distinct typed identities",
            ));
        }
        if variables[0]
            .name
            .len()
            .saturating_add(variables[1].name.len())
            > MAX_TEXT
        {
            return Err(PdeError::ResourceLimit(
                "coordinate identifiers exceed the text bound",
            ));
        }
        Ok(())
    }

    fn preflight(
        &mut self,
        expression: &Expr,
        variables: [&Symbol; 2],
        max_degree: u32,
        depth: usize,
        observed: &mut [usize; 2],
    ) -> Result<u32> {
        self.charge(Dimension::ComputeSteps, 1)?;
        observed[0] += 1;
        if depth > MAX_DEPTH || observed[0] > MAX_NODES {
            return Err(PdeError::ResourceLimit(
                "expression exceeds its node or depth envelope",
            ));
        }
        let degree = match expression {
            Expr::Integer(value) => {
                if value.bits() > MAX_BITS {
                    return Err(PdeError::ResourceLimit(
                        "integer literal exceeds 4096 bits",
                    ));
                }
                0
            }
            Expr::Rational(value) => {
                scalar_ok(value)?;
                0
            }
            Expr::Sym(symbol) if variables.contains(&symbol) => {
                observed[1] = observed[1].saturating_add(symbol.name.len());
                if observed[1] > MAX_TEXT {
                    return Err(PdeError::ResourceLimit(
                        "expression exceeds the text envelope",
                    ));
                }
                1
            }
            Expr::Add(children) | Expr::Mul(children) => {
                if children.is_empty() || children.len() > MAX_ARITY {
                    return Err(PdeError::InvalidSource(
                        "unsupported polynomial operation arity",
                    ));
                }
                let mut degree = 0;
                for child in children {
                    // Every original child is inspected, including after zero.
                    let next =
                        self.preflight(child, variables, max_degree, depth + 1, observed)?;
                    degree = if matches!(expression, Expr::Add(_)) {
                        degree.max(next)
                    } else {
                        degree + next
                    };
                    if degree > max_degree {
                        return Err(PdeError::ResourceLimit(
                            "polynomial degree exceeds the supported bound",
                        ));
                    }
                }
                degree
            }
            Expr::Pow(base, exponent) => {
                let base_degree =
                    self.preflight(base, variables, max_degree, depth + 1, observed)?;
                self.preflight(exponent, variables, max_degree, depth + 1, observed)?;
                let Expr::Integer(power) = exponent.as_ref() else {
                    return Err(PdeError::InvalidSource(
                        "polynomial exponents must be literal integers",
                    ));
                };
                let power = power
                    .to_i64()
                    .filter(|n| (-MAX_POWER..=MAX_POWER).contains(n))
                    .ok_or(PdeError::ResourceLimit("literal exponent exceeds 64"))?;
                if power <= 0 && !known_nonzero_literal(base) {
                    return Err(PdeError::InvalidSource(
                        "a nonpositive held power needs a nonzero rational literal base",
                    ));
                }
                base_degree
                    * u32::try_from(power.max(0)).expect("bounded nonnegative exponent")
            }
            _ => {
                return Err(PdeError::InvalidSource(
                    "forcing and candidates must be rational polynomials in the two coordinates",
                ));
            }
        };
        if degree > max_degree {
            return Err(PdeError::ResourceLimit(
                "polynomial degree exceeds the supported bound",
            ));
        }
        Ok(degree)
    }

    fn parse(
        &mut self,
        expression: &Expr,
        variables: [&Symbol; 2],
        max_degree: u32,
    ) -> Result<Polynomial> {
        self.preflight(expression, variables, max_degree, 0, &mut [0, 0])?;
        self.charge_storage(1, MAX_TEXT + 256)?;
        let parsed = bounded_polynomial_from_expr(
            expression,
            &[variables[0].clone(), variables[1].clone()],
            self,
        );
        let parsed = parsed.map_err(|error| {
            self.stopped
                .map_or_else(|| PdeError::Polynomial(error), PdeError::Meter)
        })?;
        if parsed.terms.len() > MAX_TERMS {
            return Err(PdeError::ResourceLimit(
                "expanded polynomial exceeds 192 terms",
            ));
        }
        for coefficient in parsed.terms.values() {
            scalar_ok(coefficient)?;
        }
        self.charge_storage(parsed.terms.len(), parsed.terms.len() * 96)?;
        Ok(parsed
            .terms
            .into_iter()
            .map(|(powers, value)| ([powers[0], powers[1]], value))
            .collect())
    }

    fn insert(
        &mut self,
        output: &mut Polynomial,
        powers: [u32; 2],
        value: BigRational,
    ) -> Result<()> {
        if powers[0] + powers[1] > MAX_PARTICULAR_DEGREE {
            return Err(PdeError::ResourceLimit(
                "derived polynomial degree exceeds seventeen",
            ));
        }
        scalar_ok(&value)?;
        let value = match output.get(&powers) {
            Some(previous) => self.add(previous, &value)?,
            None => value,
        };
        self.charge_storage(1, 128)?;
        if value.is_zero() {
            output.remove(&powers);
        } else {
            output.insert(powers, value);
        }
        if output.len() > MAX_TERMS {
            return Err(PdeError::ResourceLimit(
                "derived polynomial exceeds 192 terms",
            ));
        }
        Ok(())
    }

    fn scaled(
        &mut self,
        polynomial: &Polynomial,
        scale: &BigRational,
    ) -> Result<Polynomial> {
        let mut output = Polynomial::new();
        for (&powers, coefficient) in polynomial {
            let value = self.mul(coefficient, scale)?;
            self.insert(&mut output, powers, value)?;
        }
        Ok(output)
    }

    fn directional(
        &mut self,
        polynomial: &Polynomial,
        a: &BigRational,
        b: &BigRational,
    ) -> Result<Polynomial> {
        let mut output = Polynomial::new();
        for (&powers, coefficient) in polynomial {
            for (axis, speed) in [a, b].into_iter().enumerate() {
                if powers[axis] != 0 {
                    let factor = self.mul(
                        speed,
                        &BigRational::from_integer(powers[axis].into()),
                    )?;
                    let value = self.mul(coefficient, &factor)?;
                    let mut derivative = powers;
                    derivative[axis] -= 1;
                    self.insert(&mut output, derivative, value)?;
                }
            }
        }
        Ok(output)
    }

    fn integrate(
        &mut self,
        polynomial: &Polynomial,
        axis: usize,
        speed: &BigRational,
    ) -> Result<Polynomial> {
        let mut output = Polynomial::new();
        for (&powers, coefficient) in polynomial {
            let mut integrated = powers;
            integrated[axis] += 1;
            let divisor = self.mul(
                speed,
                &BigRational::from_integer(integrated[axis].into()),
            )?;
            let value = self.div(coefficient, &divisor)?;
            self.insert(&mut output, integrated, value)?;
        }
        Ok(output)
    }

    fn expression(
        &mut self,
        polynomial: &Polynomial,
        variables: [&Symbol; 2],
    ) -> Result<Expr> {
        self.charge_storage(
            polynomial.len(),
            polynomial.len() * (MAX_TEXT + 2048),
        )?;
        let mut terms = Vec::with_capacity(polynomial.len());
        for (powers, coefficient) in polynomial {
            let mut factors = vec![rational_expr(coefficient.clone())];
            for (axis, &power) in powers.iter().enumerate() {
                if power != 0 {
                    let variable = Expr::Sym(variables[axis].clone());
                    factors.push(if power == 1 {
                        variable
                    } else {
                        Expr::Pow(
                            Arc::new(variable),
                            Arc::new(Expr::from_i64(i64::from(power))),
                        )
                    });
                }
            }
            terms.push(if factors.len() == 1 {
                factors.remove(0)
            } else {
                Expr::Mul(factors)
            });
        }
        Ok(match terms.len() {
            0 => Expr::from_i64(0),
            1 => terms.remove(0),
            _ => Expr::Add(terms),
        })
    }
}

fn generate<M: BudgetMeter>(
    coefficients: [&BigRational; 3],
    forcing: &Expr,
    variables: [&Symbol; 2],
    work: &mut Work<'_, M>,
) -> Result<LinearFirstOrderPdeCertificate> {
    work.admit_source(coefficients, variables)?;
    let [a, b, c] = coefficients;
    let source = work.parse(forcing, variables, MAX_FORCING_DEGREE)?;
    let axis = usize::from(a.is_zero());
    let speed = coefficients[axis];
    let other = coefficients[1 - axis];
    let inverse = work.div(&BigRational::one(), speed)?;
    let slope = -work.div(other, speed)?;
    let mut time = Polynomial::new();
    let mut invariant = Polynomial::new();
    let mut unit = [0, 0];
    unit[axis] = 1;
    work.insert(&mut time, unit, inverse)?;
    work.insert(&mut invariant, unit, slope)?;
    unit = [0, 0];
    unit[1 - axis] = 1;
    work.insert(&mut invariant, unit, BigRational::one())?;

    let mut particular = Polynomial::new();
    if c.is_zero() {
        // Triangular integration is the finite characteristic-coordinate
        // antiderivative, expressed directly in the original coordinates.
        let mut remainder = source;
        for _ in 0..=MAX_FORCING_DEGREE {
            let integrated = work.integrate(&remainder, axis, speed)?;
            for (&powers, coefficient) in &integrated {
                work.insert(&mut particular, powers, coefficient.clone())?;
            }
            let (dx, dy) = if axis == 0 {
                (BigRational::zero(), -other)
            } else {
                (-other, BigRational::zero())
            };
            remainder = work.directional(&integrated, &dx, &dy)?;
            if remainder.is_empty() {
                break;
            }
        }
        if !remainder.is_empty() {
            return Err(PdeError::ResourceLimit(
                "characteristic integration did not terminate",
            ));
        }
    } else {
        // On degree-d polynomials, D^(d+1)=0. The finite geometric inverse
        // of c+D is sum_k (-D)^k/c^(k+1).
        let inverse = work.div(&BigRational::one(), c)?;
        let mut term = work.scaled(&source, &inverse)?;
        for _ in 0..=MAX_FORCING_DEGREE {
            for (&powers, coefficient) in &term {
                work.insert(&mut particular, powers, coefficient.clone())?;
            }
            let derivative = work.directional(&term, a, b)?;
            term = work.scaled(&derivative, &(-&inverse))?;
            if term.is_empty() {
                break;
            }
        }
        if !term.is_empty() {
            return Err(PdeError::ResourceLimit(
                "polynomial directional inverse did not terminate",
            ));
        }
    }
    Ok(LinearFirstOrderPdeCertificate {
        time_coordinate: work.expression(&time, variables)?,
        invariant_coordinate: work.expression(&invariant, variables)?,
        particular: work.expression(&particular, variables)?,
    })
}

fn check<M: BudgetMeter>(
    coefficients: [&BigRational; 3],
    forcing: &Expr,
    variables: [&Symbol; 2],
    candidate: &LinearFirstOrderPdeCertificate,
    work: &mut Work<'_, M>,
) -> Result<VerifiedLinearPde> {
    work.admit_source(coefficients, variables)?;
    let [a, b, c] = coefficients;
    let source = work.parse(forcing, variables, MAX_FORCING_DEGREE)?;
    let time = work.parse(&candidate.time_coordinate, variables, 1)?;
    let invariant = work.parse(&candidate.invariant_coordinate, variables, 1)?;
    let particular = work.parse(
        &candidate.particular,
        variables,
        MAX_PARTICULAR_DEGREE,
    )?;
    let zero = BigRational::zero();
    let sx = time.get(&[1, 0]).unwrap_or(&zero);
    let sy = time.get(&[0, 1]).unwrap_or(&zero);
    let tx = invariant.get(&[1, 0]).unwrap_or(&zero);
    let ty = invariant.get(&[0, 1]).unwrap_or(&zero);
    let asx = work.mul(a, sx)?;
    let bsy = work.mul(b, sy)?;
    let atx = work.mul(a, tx)?;
    let bty = work.mul(b, ty)?;
    if work.add(&asx, &bsy)? != BigRational::one()
        || !work.add(&atx, &bty)?.is_zero()
    {
        return Err(PdeError::Verification(
            "characteristic directional identities failed",
        ));
    }
    let diagonal = work.mul(sx, ty)?;
    let off_diagonal = work.mul(sy, tx)?;
    if work.add(&diagonal, &(-off_diagonal))?.is_zero() {
        return Err(PdeError::Verification(
            "characteristic coordinate map is singular",
        ));
    }

    // Independent scalar coefficient recurrence. This does not call the
    // generator's directional derivative, integration, or polynomial inverse.
    // Checking the full degree envelope also detects spurious high-degree q.
    for i in 0..=MAX_PARTICULAR_DEGREE {
        for j in 0..=MAX_PARTICULAR_DEGREE - i {
            let qx = particular.get(&[i + 1, j]).unwrap_or(&zero);
            let qy = particular.get(&[i, j + 1]).unwrap_or(&zero);
            let q0 = particular.get(&[i, j]).unwrap_or(&zero);
            let dx = work.mul(qx, &BigRational::from_integer((i + 1).into()))?;
            let dy = work.mul(qy, &BigRational::from_integer((j + 1).into()))?;
            let dx = work.mul(a, &dx)?;
            let dy = work.mul(b, &dy)?;
            let reaction = work.mul(c, q0)?;
            let transport = work.add(&dx, &dy)?;
            let residual = work.add(&transport, &reaction)?;
            if &residual != source.get(&[i, j]).unwrap_or(&zero) {
                return Err(PdeError::Verification(
                    "particular solution fails the original PDE coefficient identity",
                ));
            }
        }
    }

    let time_coordinate = work.expression(&time, variables)?;
    let invariant_coordinate = work.expression(&invariant, variables)?;
    let particular = work.expression(&particular, variables)?;
    work.charge_storage(MAX_NODES, MAX_NODES * 2048 + MAX_TEXT)?;
    let homogeneous_coefficient = if c.is_zero() {
        Expr::from_i64(1)
    } else {
        Expr::Function(
            "exp".into(),
            vec![rational_expr(-c) * time_coordinate.clone()],
        )
    };
    let accepted = VerifiedLinearPde {
        coefficients: [a.clone(), b.clone(), c.clone()],
        variables: [variables[0].clone(), variables[1].clone()],
        forcing: forcing.clone(),
        certificate: LinearFirstOrderPdeCertificate {
            time_coordinate,
            invariant_coordinate,
            particular,
        },
        homogeneous_coefficient,
    };
    work.checkpoint()?;
    Ok(accepted)
}

/// Verify a proposal against the original coefficients, forcing, and axes.
///
/// The polynomial coefficient recurrence and affine determinant are exact;
/// no numerical observations, simplifier zero guesses, or generator calls
/// participate in acceptance. The caller's meter is cumulative throughout.
pub fn verify_linear_first_order_pde(
    coefficients: [&BigRational; 3],
    forcing: &Expr,
    variables: [&Symbol; 2],
    candidate: &LinearFirstOrderPdeCertificate,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedLinearPde> {
    check(
        coefficients,
        forcing,
        variables,
        candidate,
        &mut Work::new(meter),
    )
}

/// Construct and check a rational constant-coefficient characteristic family.
///
/// Forcing has total degree at most sixteen; the particular polynomial has
/// degree at most seventeen. Every original node is admitted before any zero
/// cancellation. All source and derived scalars are bounded to 4096 bits.
/// Generation and independent verification share the same cumulative caller
/// meter and a fixed twenty-million-step/128-MiB logical allocation envelope.
/// This synchronous API does not reserve a protected runtime verifier pool.
pub fn metered_linear_first_order_pde(
    a: &BigRational,
    b: &BigRational,
    c: &BigRational,
    forcing: &Expr,
    x: &Symbol,
    y: &Symbol,
    meter: &mut impl BudgetMeter,
) -> Result<VerifiedLinearPde> {
    let mut work = Work::new(meter);
    let candidate = generate([a, b, c], forcing, [x, y], &mut work)?;
    check([a, b, c], forcing, [x, y], &candidate, &mut work)
}

/// Convenience entry with the same fixed internal work/allocation envelope.
pub fn linear_first_order_pde(
    a: &BigRational,
    b: &BigRational,
    c: &BigRational,
    forcing: &Expr,
    x: &Symbol,
    y: &Symbol,
) -> Result<VerifiedLinearPde> {
    metered_linear_first_order_pde(a, b, c, forcing, x, y, &mut Unbounded)
}
