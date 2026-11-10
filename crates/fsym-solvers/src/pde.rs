//! Checked characteristic families for a bounded first-order linear PDE.
//!
//! For rational constants a, b, c with (a,b) != (0,0), this lane solves
//! a*u_x + b*u_y + c*u = f(x,y) with bounded elementary forcing. Each
//! forcing mode is exp(E)*(P*cos(T)+Q*sin(T)), where E and T are rational
//! affine forms and P and Q are rational polynomials. Its output describes
//! exp(-c*s)*F(t) + q for an arbitrary differentiable function F.
//! The separate checker establishes Ds=1, Dt=0, an invertible affine
//! coordinate map, and every coefficient of each original residual mode.
//! This is a verified solution family, not a general PDE-completeness,
//! boundary-value, portable-proof, or asynchronous-runtime certificate.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_polys::{PolyError, bounded_polynomial_from_expr};
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use thiserror::Error;

const MAX_FORCING_DEGREE: u32 = 16;
const MAX_PARTICULAR_DEGREE: u32 = 17;
const MAX_TERMS: usize = 192;
const MAX_MODES: usize = 16;
const MAX_BITS: u64 = 4096;
const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_ARITY: usize = 256;
const MAX_TEXT: usize = 16_384;
const MAX_POWER: i64 = 64;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

type Polynomial = BTreeMap<[u32; 2], BigRational>;

// Affine triples are [x coefficient, y coefficient, constant]. A zero phase
// denotes a plain exponential (or polynomial when the exponential is zero).
// For a nonzero phase its first nonzero entry is positive; changing its sign
// changes the sine amplitude's sign and leaves the cosine amplitude intact.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Mode {
    exponential: [BigRational; 3],
    phase: [BigRational; 3],
}

impl Mode {
    fn polynomial() -> Self {
        Self {
            exponential: std::array::from_fn(|_| BigRational::zero()),
            phase: std::array::from_fn(|_| BigRational::zero()),
        }
    }

    fn has_phase(&self) -> bool {
        self.phase.iter().any(|value| !value.is_zero())
    }
}

type Amplitudes = [Polynomial; 2];
type Modes = BTreeMap<Mode, Amplitudes>;
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

/// Untrusted affine coordinates and an elementary particular expression.
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

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
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
        self.stopped
            .map_or_else(|| PdeError::Arithmetic(error.to_string()), PdeError::Meter)
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
        elementary: bool,
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
                    return Err(PdeError::ResourceLimit("integer literal exceeds 4096 bits"));
                }
                0
            }
            Expr::Rational(value) => {
                scalar_ok(value)?;
                0
            }
            Expr::Const(Constant::E) if elementary => 0,
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
                    let next = self.preflight(
                        child,
                        variables,
                        max_degree,
                        elementary,
                        depth + 1,
                        observed,
                    )?;
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
                    self.preflight(base, variables, max_degree, false, depth + 1, observed)?;
                self.preflight(exponent, variables, max_degree, false, depth + 1, observed)?;
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
                base_degree * u32::try_from(power.max(0)).expect("bounded nonnegative exponent")
            }
            Expr::Function(name, arguments)
                if elementary && matches!(name.as_str(), "exp" | "sin" | "cos") =>
            {
                observed[1] = observed[1].saturating_add(name.len());
                if observed[1] > MAX_TEXT {
                    return Err(PdeError::ResourceLimit(
                        "expression exceeds the text envelope",
                    ));
                }
                if arguments.len() != 1 {
                    return Err(PdeError::InvalidSource(
                        "elementary forcing functions must have exactly one argument",
                    ));
                }
                // Function arguments have their own affine bound. Polynomial
                // powers never recurse with elementary=true, even at power zero.
                self.preflight(&arguments[0], variables, 1, false, depth + 1, observed)?;
                0
            }
            _ => {
                return Err(PdeError::InvalidSource(
                    "expected rational polynomials and exponentials or trigonometric functions of rational affine forms",
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
        self.preflight(expression, variables, max_degree, false, 0, &mut [0, 0])?;
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

    fn parse_modes(
        &mut self,
        expression: &Expr,
        variables: [&Symbol; 2],
        max_degree: u32,
    ) -> Result<Modes> {
        // This complete original-graph pass precedes all arithmetic, expansion,
        // mode merging, and removal of zero amplitudes.
        self.preflight(expression, variables, max_degree, true, 0, &mut [0, 0])?;
        self.trigonometric_factors(expression)?;
        self.parse_modes_inner(expression, variables, max_degree)
    }

    fn trigonometric_factors(&mut self, expression: &Expr) -> Result<usize> {
        self.charge(Dimension::ComputeSteps, 1)?;
        let mut count = 0;
        match expression {
            Expr::Add(children) | Expr::Mul(children) => {
                for child in children {
                    let next = self.trigonometric_factors(child)?;
                    count = if matches!(expression, Expr::Add(_)) {
                        count.max(next)
                    } else {
                        count + next
                    };
                    if count > 1 {
                        return Err(PdeError::InvalidSource(
                            "each elementary term supports at most one trigonometric factor",
                        ));
                    }
                }
            }
            Expr::Function(name, _) if matches!(name.as_str(), "sin" | "cos") => {
                count = 1;
            }
            _ => {}
        }
        Ok(count)
    }

    fn parse_modes_inner(
        &mut self,
        expression: &Expr,
        variables: [&Symbol; 2],
        max_degree: u32,
    ) -> Result<Modes> {
        self.charge_storage(1, 256)?;
        let mut output = Modes::new();
        match expression {
            Expr::Const(Constant::E) => {
                let mut mode = Mode::polynomial();
                mode.exponential[2] = BigRational::one();
                let one = BTreeMap::from([([0, 0], BigRational::one())]);
                self.insert_mode(&mut output, mode, [one, Polynomial::new()])?;
            }
            Expr::Add(children) => {
                for child in children {
                    for (mode, amplitudes) in
                        self.parse_modes_inner(child, variables, max_degree)?
                    {
                        self.insert_mode(&mut output, mode, amplitudes)?;
                    }
                }
            }
            Expr::Mul(children) => {
                let one = BTreeMap::from([([0, 0], BigRational::one())]);
                self.insert_mode(&mut output, Mode::polynomial(), [one, Polynomial::new()])?;
                for child in children {
                    let factor = self.parse_modes_inner(child, variables, max_degree)?;
                    output = self.multiply_modes(&output, &factor)?;
                }
            }
            Expr::Function(name, arguments) => {
                let argument = self.parse(&arguments[0], variables, 1)?;
                self.charge_storage(1, 8192)?;
                let mut affine = [
                    argument
                        .get(&[1, 0])
                        .cloned()
                        .unwrap_or_else(BigRational::zero),
                    argument
                        .get(&[0, 1])
                        .cloned()
                        .unwrap_or_else(BigRational::zero),
                    argument
                        .get(&[0, 0])
                        .cloned()
                        .unwrap_or_else(BigRational::zero),
                ];
                let mut mode = Mode::polynomial();
                let mut amplitudes = [Polynomial::new(), Polynomial::new()];
                match name.as_str() {
                    "exp" => {
                        mode.exponential = affine;
                        self.insert(&mut amplitudes[0], [0, 0], BigRational::one())?;
                    }
                    "sin" | "cos" => {
                        let negative = affine
                            .iter()
                            .find(|value| !value.is_zero())
                            .is_some_and(|value| value < &BigRational::zero());
                        if negative {
                            affine = affine.map(|value| -value);
                        }
                        mode.phase = affine;
                        let channel = usize::from(name == "sin");
                        if channel == 0 || mode.has_phase() {
                            let unit = if channel == 1 && negative {
                                -BigRational::one()
                            } else {
                                BigRational::one()
                            };
                            self.insert(&mut amplitudes[channel], [0, 0], unit)?;
                        }
                    }
                    _ => {
                        return Err(PdeError::InvalidSource(
                            "unsupported elementary forcing function",
                        ));
                    }
                }
                self.insert_mode(&mut output, mode, amplitudes)?;
            }
            _ => {
                let polynomial = self.parse(expression, variables, max_degree)?;
                self.insert_mode(
                    &mut output,
                    Mode::polynomial(),
                    [polynomial, Polynomial::new()],
                )?;
            }
        }
        Ok(output)
    }

    fn insert_mode(
        &mut self,
        output: &mut Modes,
        mode: Mode,
        amplitudes: Amplitudes,
    ) -> Result<()> {
        self.charge_storage(1, 8192)?;
        let target = output
            .entry(mode)
            .or_insert_with(|| [Polynomial::new(), Polynomial::new()]);
        for (channel, polynomial) in amplitudes.into_iter().enumerate() {
            for (powers, coefficient) in polynomial {
                self.insert(&mut target[channel], powers, coefficient)?;
            }
        }
        output.retain(|_, pair| !pair[0].is_empty() || !pair[1].is_empty());
        if output.len() > MAX_MODES {
            return Err(PdeError::ResourceLimit(
                "elementary forcing exceeds sixteen distinct modes",
            ));
        }
        Ok(())
    }

    fn multiply_polynomials(
        &mut self,
        left: &Polynomial,
        right: &Polynomial,
    ) -> Result<Polynomial> {
        let mut output = Polynomial::new();
        for (lp, lc) in left {
            for (rp, rc) in right {
                let coefficient = self.mul(lc, rc)?;
                self.insert(&mut output, [lp[0] + rp[0], lp[1] + rp[1]], coefficient)?;
            }
        }
        Ok(output)
    }

    fn multiply_modes(&mut self, left: &Modes, right: &Modes) -> Result<Modes> {
        let mut output = Modes::new();
        for (lm, la) in left {
            for (rm, ra) in right {
                if lm.has_phase() && rm.has_phase() {
                    return Err(PdeError::InvalidSource(
                        "each elementary term supports at most one trigonometric factor",
                    ));
                }
                self.charge_storage(1, 8192)?;
                let mode = Mode {
                    exponential: [
                        self.add(&lm.exponential[0], &rm.exponential[0])?,
                        self.add(&lm.exponential[1], &rm.exponential[1])?,
                        self.add(&lm.exponential[2], &rm.exponential[2])?,
                    ],
                    phase: if lm.has_phase() {
                        lm.phase.clone()
                    } else {
                        rm.phase.clone()
                    },
                };
                let pair = if lm.has_phase() {
                    [
                        self.multiply_polynomials(&la[0], &ra[0])?,
                        self.multiply_polynomials(&la[1], &ra[0])?,
                    ]
                } else {
                    [
                        self.multiply_polynomials(&la[0], &ra[0])?,
                        self.multiply_polynomials(&la[0], &ra[1])?,
                    ]
                };
                self.insert_mode(&mut output, mode, pair)?;
            }
        }
        Ok(output)
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

    fn scaled(&mut self, polynomial: &Polynomial, scale: &BigRational) -> Result<Polynomial> {
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
                    let factor =
                        self.mul(speed, &BigRational::from_integer(powers[axis].into()))?;
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
            let divisor = self.mul(speed, &BigRational::from_integer(integrated[axis].into()))?;
            let value = self.div(coefficient, &divisor)?;
            self.insert(&mut output, integrated, value)?;
        }
        Ok(output)
    }

    fn expression(&mut self, polynomial: &Polynomial, variables: [&Symbol; 2]) -> Result<Expr> {
        self.charge_storage(polynomial.len(), polynomial.len() * (MAX_TEXT + 2048))?;
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

    fn affine_expression(
        &mut self,
        affine: &[BigRational; 3],
        variables: [&Symbol; 2],
    ) -> Result<Expr> {
        self.charge_storage(3, 4096)?;
        let mut polynomial = Polynomial::new();
        for (powers, coefficient) in [[1, 0], [0, 1], [0, 0]].into_iter().zip(affine) {
            self.insert(&mut polynomial, powers, coefficient.clone())?;
        }
        self.expression(&polynomial, variables)
    }

    fn modes_expression(&mut self, modes: &Modes, variables: [&Symbol; 2]) -> Result<Expr> {
        self.charge_storage(modes.len(), modes.len() * 16_384)?;
        let mut terms = Vec::with_capacity(modes.len());
        for (mode, amplitudes) in modes {
            let wave = if mode.has_phase() {
                let phase = self.affine_expression(&mode.phase, variables)?;
                let mut pair = Vec::with_capacity(2);
                for (channel, polynomial) in amplitudes.iter().enumerate() {
                    if !polynomial.is_empty() {
                        pair.push(Expr::Mul(vec![
                            self.expression(polynomial, variables)?,
                            Expr::Function(
                                if channel == 0 { "cos" } else { "sin" }.into(),
                                vec![phase.clone()],
                            ),
                        ]));
                    }
                }
                match pair.len() {
                    0 => continue,
                    1 => pair.remove(0),
                    _ => Expr::Add(pair),
                }
            } else {
                if !amplitudes[1].is_empty() {
                    return Err(PdeError::Verification(
                        "a zero phase has an invalid sine amplitude",
                    ));
                }
                self.expression(&amplitudes[0], variables)?
            };
            terms.push(if mode.exponential.iter().all(BigRational::is_zero) {
                wave
            } else {
                Expr::Mul(vec![
                    Expr::Function(
                        "exp".into(),
                        vec![self.affine_expression(&mode.exponential, variables)?],
                    ),
                    wave,
                ])
            });
        }
        Ok(match terms.len() {
            0 => Expr::from_i64(0),
            1 => terms.remove(0),
            _ => Expr::Add(terms),
        })
    }

    fn paired_scale(
        &mut self,
        amplitudes: &Amplitudes,
        real: &BigRational,
        imaginary: &BigRational,
    ) -> Result<Amplitudes> {
        // [[real,-imaginary],[imaginary,real]] times [cosine,sine].
        let mut cosine = self.scaled(&amplitudes[0], real)?;
        for (powers, coefficient) in self.scaled(&amplitudes[1], &(-imaginary))? {
            self.insert(&mut cosine, powers, coefficient)?;
        }
        let mut sine = self.scaled(&amplitudes[0], imaginary)?;
        for (powers, coefficient) in self.scaled(&amplitudes[1], real)? {
            self.insert(&mut sine, powers, coefficient)?;
        }
        Ok([cosine, sine])
    }
}

fn transport_integral<M: BudgetMeter>(
    source: Polynomial,
    a: &BigRational,
    b: &BigRational,
    work: &mut Work<'_, M>,
) -> Result<Polynomial> {
    let axis = usize::from(a.is_zero());
    let (speed, other) = if axis == 0 { (a, b) } else { (b, a) };
    let (dx, dy) = if axis == 0 {
        (BigRational::zero(), -other)
    } else {
        (-other, BigRational::zero())
    };
    let mut remainder = source;
    let mut particular = Polynomial::new();
    // Triangular integration is the finite characteristic antiderivative,
    // expressed directly in the original coordinates.
    for _ in 0..=MAX_FORCING_DEGREE {
        let integrated = work.integrate(&remainder, axis, speed)?;
        remainder = work.directional(&integrated, &dx, &dy)?;
        for (powers, coefficient) in integrated {
            work.insert(&mut particular, powers, coefficient)?;
        }
        if remainder.is_empty() {
            return Ok(particular);
        }
    }
    Err(PdeError::ResourceLimit(
        "characteristic integration did not terminate",
    ))
}

fn mode_particular<M: BudgetMeter>(
    source: Amplitudes,
    mode: &Mode,
    coefficients: [&BigRational; 3],
    work: &mut Work<'_, M>,
) -> Result<Amplitudes> {
    let [a, b, c] = coefficients;
    let ex = work.mul(a, &mode.exponential[0])?;
    let ey = work.mul(b, &mode.exponential[1])?;
    let rate = work.add(&ex, &ey)?;
    let lambda = work.add(c, &rate)?;
    let px = work.mul(a, &mode.phase[0])?;
    let py = work.mul(b, &mode.phase[1])?;
    let omega = work.add(&px, &py)?;
    if lambda.is_zero() && omega.is_zero() {
        let [cosine, sine] = source;
        return Ok([
            transport_integral(cosine, a, b, work)?,
            transport_integral(sine, a, b, work)?,
        ]);
    }
    let lambda_squared = work.mul(&lambda, &lambda)?;
    let omega_squared = work.mul(&omega, &omega)?;
    let determinant = work.add(&lambda_squared, &omega_squared)?;
    // All rates are rational. This determinant is nonzero exactly when one
    // of lambda or omega is nonzero; no branch or numerical decision occurs.
    let real = work.div(&lambda, &determinant)?;
    let imaginary = work.div(&omega, &determinant)?;
    let mut term = work.paired_scale(&source, &real, &imaginary)?;
    let mut particular = [Polynomial::new(), Polynomial::new()];
    // D is nilpotent on the admitted polynomial degree envelope. A^{-1}
    // commutes with D, so (D+A)^{-1}=sum_k (-A^{-1}D)^k A^{-1}.
    for _ in 0..=MAX_FORCING_DEGREE {
        let derivative = [
            work.directional(&term[0], a, b)?,
            work.directional(&term[1], a, b)?,
        ];
        for (channel, polynomial) in term.into_iter().enumerate() {
            for (powers, coefficient) in polynomial {
                work.insert(&mut particular[channel], powers, coefficient)?;
            }
        }
        if derivative.iter().all(Polynomial::is_empty) {
            return Ok(particular);
        }
        term = work.paired_scale(&derivative, &(-&real), &(-&imaginary))?;
    }
    Err(PdeError::ResourceLimit(
        "elementary directional inverse did not terminate",
    ))
}

fn generate<M: BudgetMeter>(
    coefficients: [&BigRational; 3],
    forcing: &Expr,
    variables: [&Symbol; 2],
    work: &mut Work<'_, M>,
) -> Result<LinearFirstOrderPdeCertificate> {
    work.admit_source(coefficients, variables)?;
    let source = work.parse_modes(forcing, variables, MAX_FORCING_DEGREE)?;
    let axis = usize::from(coefficients[0].is_zero());
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

    let mut particular = Modes::new();
    for (mode, amplitudes) in source {
        let solved = mode_particular(amplitudes, &mode, coefficients, work)?;
        work.insert_mode(&mut particular, mode, solved)?;
    }
    Ok(LinearFirstOrderPdeCertificate {
        time_coordinate: work.expression(&time, variables)?,
        invariant_coordinate: work.expression(&invariant, variables)?,
        particular: work.modes_expression(&particular, variables)?,
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
    let source = work.parse_modes(forcing, variables, MAX_FORCING_DEGREE)?;
    let time = work.parse(&candidate.time_coordinate, variables, 1)?;
    let invariant = work.parse(&candidate.invariant_coordinate, variables, 1)?;
    let particular = work.parse_modes(&candidate.particular, variables, MAX_PARTICULAR_DEGREE)?;
    let zero = BigRational::zero();
    let sx = time.get(&[1, 0]).unwrap_or(&zero);
    let sy = time.get(&[0, 1]).unwrap_or(&zero);
    let tx = invariant.get(&[1, 0]).unwrap_or(&zero);
    let ty = invariant.get(&[0, 1]).unwrap_or(&zero);
    let asx = work.mul(a, sx)?;
    let bsy = work.mul(b, sy)?;
    let atx = work.mul(a, tx)?;
    let bty = work.mul(b, ty)?;
    if work.add(&asx, &bsy)? != BigRational::one() || !work.add(&atx, &bty)?.is_zero() {
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

    // A separate scalar recurrence checks both amplitudes of every original
    // or candidate mode. It does not call the generator's directional
    // derivative, paired inverse, integration, or mode-particular helpers.
    // Candidate-only modes are permitted only when their original residual
    // is exactly zero (for example, an additional homogeneous solution).
    work.charge_storage(source.len() + particular.len(), 32 * 1024)?;
    let modes: BTreeSet<_> = source.keys().chain(particular.keys()).collect();
    let empty = [Polynomial::new(), Polynomial::new()];
    for mode in modes {
        let original = source.get(mode).unwrap_or(&empty);
        let proposed = particular.get(mode).unwrap_or(&empty);
        let ex = work.mul(a, &mode.exponential[0])?;
        let ey = work.mul(b, &mode.exponential[1])?;
        let rate = work.add(&ex, &ey)?;
        let reaction_rate = work.add(c, &rate)?;
        let tx = work.mul(a, &mode.phase[0])?;
        let ty = work.mul(b, &mode.phase[1])?;
        let frequency = work.add(&tx, &ty)?;
        for channel in 0..2 {
            for i in 0..=MAX_PARTICULAR_DEGREE {
                for j in 0..=MAX_PARTICULAR_DEGREE - i {
                    let qx = proposed[channel].get(&[i + 1, j]).unwrap_or(&zero);
                    let qy = proposed[channel].get(&[i, j + 1]).unwrap_or(&zero);
                    let q0 = proposed[channel].get(&[i, j]).unwrap_or(&zero);
                    let paired = proposed[1 - channel].get(&[i, j]).unwrap_or(&zero);
                    let dx = work.mul(qx, &BigRational::from_integer((i + 1).into()))?;
                    let dy = work.mul(qy, &BigRational::from_integer((j + 1).into()))?;
                    let dx = work.mul(a, &dx)?;
                    let dy = work.mul(b, &dy)?;
                    let reaction = work.mul(&reaction_rate, q0)?;
                    let coupling = work.mul(&frequency, paired)?;
                    let coupling = if channel == 0 { coupling } else { -coupling };
                    let transport = work.add(&dx, &dy)?;
                    let diagonal = work.add(&transport, &reaction)?;
                    let residual = work.add(&diagonal, &coupling)?;
                    if &residual != original[channel].get(&[i, j]).unwrap_or(&zero) {
                        return Err(PdeError::Verification(
                            "particular solution fails an original PDE mode coefficient identity",
                        ));
                    }
                }
            }
        }
    }

    let time_coordinate = work.expression(&time, variables)?;
    let invariant_coordinate = work.expression(&invariant, variables)?;
    let particular = work.modes_expression(&particular, variables)?;
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
/// The two mode coefficient recurrences and affine determinant are exact;
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
/// Forcing is a sum of at most sixteen exp(E)*(P*cos(T)+Q*sin(T)) modes,
/// including pure polynomials, exponentials, and trigonometric terms. E and T
/// are affine in the original axes with exact rational rates and offsets.
/// The built-in constant E is admitted as the canonical form of exp(1).
/// P and Q have degree at most sixteen; particular amplitudes have degree at
/// most seventeen. Add/Mul may distribute these forms and combine exponentials,
/// but two trigonometric factors in a product and elementary-function powers
/// are refused. Every original node is admitted before zero cancellation.
/// Each amplitude has at most 192 terms, and every scalar at most 4096 bits.
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
