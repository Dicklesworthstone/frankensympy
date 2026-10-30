//! Checked localization of exact rational systems without losing source poles.
//!
//! The generator lowers the held source into numerator/denominator pairs.
//! A separate reference recurrence reconstructs every pair and every original
//! inverse guard. It then checks every coefficient of the numerator equations
//! and u*product(guards)-1. Thus projection of the augmented polynomial variety
//! is exactly the original rational zero set, with a unique auxiliary value.
//!
//! This receipt is a construction theorem, not a finiteness certificate.
//! A caller must separately prove that the augmented system is finite before
//! publishing a finite solution set. In particular saturation alone cannot
//! represent excluded points of a positive-dimensional family.
//!
//! Both construction lanes have cumulative twenty-million-step / 128 MiB
//! envelopes. All source nodes are checked before zero/constant shortcuts;
//! cancellation and resource failures remain errors.

use crate::division::{
    MAX_BASIS, MAX_VARIABLES, PolynomialWork, Terms, accumulate, copy_poly, monomial_poly,
    multiply_polynomials, refused, ring_storage, same_ring, validate_poly,
};
use crate::{MultivariatePoly, PolyError, bounded_polynomial_from_expr};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::{One, Zero};
use std::mem::size_of;

const MAX_ORIGINAL_VARIABLES: usize = MAX_VARIABLES - 1;
const MAX_ORIGINAL_EQUATIONS: usize = MAX_BASIS - 1;
const MAX_SOURCE_NODES: usize = 4096;
const MAX_SOURCE_DEPTH: usize = 64;
const MAX_GUARDS: usize = 128;
const MAX_BITS: u64 = 4096;

/// An untrusted rational source pair. It is deliberately not cancelled:
/// source-domain obligations are independent of algebraic simplification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalSourceFraction {
    pub numerator: MultivariatePoly,
    pub denominator: MultivariatePoly,
}

/// Untrusted exact source reduction. Every retained guard must be nonzero;
/// these obligations survive cancellation, zero products and outer powers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalSourceCertificate {
    pub fraction: RationalSourceFraction,
    pub guards: Vec<MultivariatePoly>,
}

/// An exact rational function bound to its original held source and typed
/// ring. This establishes equality only where every original guard is
/// nonzero; callers must separately discharge their own domain obligations.
/// It is an in-process construction receipt, not a portable proof artifact.
#[derive(Debug, Clone)]
pub struct VerifiedRationalSource {
    source: Expr,
    variables: Vec<Symbol>,
    certificate: RationalSourceCertificate,
}

impl VerifiedRationalSource {
    pub fn source(&self) -> &Expr {
        &self.source
    }

    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    pub fn fraction(&self) -> &RationalSourceFraction {
        &self.certificate.fraction
    }

    pub fn guards(&self) -> &[MultivariatePoly] {
        &self.certificate.guards
    }

    pub fn certificate(&self) -> &RationalSourceCertificate {
        &self.certificate
    }
}

/// Untrusted proposal for an exact localization construction. Fractions and
/// guards use the original ring. Equations append the auxiliary variable last.
#[derive(Debug, Clone)]
pub struct RationalLocalizationCertificate {
    pub auxiliary: Symbol,
    pub fractions: Vec<RationalSourceFraction>,
    pub guards: Vec<MultivariatePoly>,
    pub equations: Vec<MultivariatePoly>,
}

/// Original-source-bound polynomial construction with exact source domains.
/// No public constructor can create this accepted receipt without checking.
#[derive(Debug, Clone)]
pub struct VerifiedRationalLocalization {
    sources: Vec<Expr>,
    variables: Vec<Symbol>,
    equations: Vec<Expr>,
    certificate: RationalLocalizationCertificate,
}

impl VerifiedRationalLocalization {
    pub fn sources(&self) -> &[Expr] {
        &self.sources
    }

    /// Original variables followed by exactly one private auxiliary variable.
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    pub fn original_variables(&self) -> &[Symbol] {
        &self.variables[..self.original_width()]
    }

    pub fn original_width(&self) -> usize {
        self.variables.len() - 1
    }

    /// Checked, bounded expression lifting of all augmented equations.
    pub fn equations(&self) -> &[Expr] {
        &self.equations
    }

    pub fn polynomials(&self) -> &[MultivariatePoly] {
        &self.certificate.equations
    }

    pub fn guards(&self) -> &[MultivariatePoly] {
        &self.certificate.guards
    }

    pub fn certificate(&self) -> &RationalLocalizationCertificate {
        &self.certificate
    }
}

/// Polynomial is only a routing hint, not source admission. The caller must
/// still admit every original through its ordinary polynomial path.
#[derive(Debug, Clone)]
pub enum RationalSystemAdmission {
    Polynomial,
    Localized(Box<VerifiedRationalLocalization>),
}

fn invalid(message: &'static str) -> PolyError {
    PolyError::NonPolynomialExpression(message.into())
}

fn check_ring(
    variables: &[Symbol],
    maximum: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if variables.is_empty() || variables.len() > maximum {
        return Err(refused(
            "rational-system variable count exceeds its envelope",
        ));
    }
    let mut bytes = 0usize;
    for (index, variable) in variables.iter().enumerate() {
        work.tick(index + 1)?;
        bytes = bytes
            .checked_add(variable.name.len())
            .filter(|&bytes| bytes <= 65_536)
            .ok_or_else(|| refused("rational-system generator names exceed 65536 bytes"))?;
        if variables[..index].contains(variable) {
            return Err(refused("duplicate typed rational-system variable"));
        }
    }
    Ok(())
}

/// Borrowed bounded recognition. A Symbol below a negative integer power
/// requires rational source admission; literal and compound constant inverses
/// remain on the established polynomial route.
fn needs_localization(
    sources: &[Expr],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    if sources.len() > MAX_BASIS {
        return Err(refused("rational-system input count exceeds 64"));
    }
    let mut needed = false;
    for source in sources {
        work.allocate(128)?;
        let mut stack = vec![(source, 0usize, false)];
        let mut visited = 0usize;
        while let Some((node, depth, inverse)) = stack.pop() {
            work.tick(1)?;
            visited += 1;
            if visited > MAX_SOURCE_NODES || depth > MAX_SOURCE_DEPTH {
                return Err(refused(
                    "rational-system source exceeds its node/depth envelope",
                ));
            }
            match node {
                Expr::Sym(_) => needed |= inverse,
                Expr::Pow(base, exponent) => {
                    if visited + stack.len() + 2 > MAX_SOURCE_NODES {
                        return Err(refused("rational-system source exceeds its node envelope"));
                    }
                    let negative = matches!(exponent.as_ref(),
                        Expr::Integer(value) if value.to_i64().is_some_and(|n| n < 0));
                    work.allocate(256)?;
                    stack.push((exponent.as_ref(), depth + 1, inverse));
                    stack.push((base.as_ref(), depth + 1, inverse || negative));
                }
                Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                    if children.len() > MAX_SOURCE_NODES.saturating_sub(visited + stack.len()) {
                        return Err(refused("rational-system source exceeds its node envelope"));
                    }
                    work.allocate(children.len() * 128)?;
                    stack.extend(children.iter().map(|child| (child, depth + 1, inverse)));
                }
                _ => {}
            }
        }
    }
    Ok(needed)
}

/// Admit every held original before conversion. The returned byte count
/// prepays the eventual retained-source copy without copying during admission.
fn preflight_sources(
    sources: &[Expr],
    variables: &[Symbol],
    maximum_variables: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<usize, PolyError> {
    check_ring(variables, maximum_variables, work)?;
    if sources.len() > MAX_ORIGINAL_EQUATIONS {
        return Err(refused(
            "localization supports at most 63 original equations",
        ));
    }
    let mut copy_bytes = 0usize;
    for source in sources {
        work.allocate(128)?;
        let mut stack = vec![(source, 0usize)];
        let mut visited = 0usize;
        while let Some((node, depth)) = stack.pop() {
            work.tick(1)?;
            visited += 1;
            if visited > MAX_SOURCE_NODES || depth > MAX_SOURCE_DEPTH {
                return Err(refused(
                    "rational-system source exceeds its node/depth envelope",
                ));
            }
            let extra = match node {
                Expr::Integer(value) => {
                    if value.bits() > MAX_BITS {
                        return Err(refused("rational-system integer exceeds 4096 bits"));
                    }
                    value.limb_count() as usize * 8
                }
                Expr::Rational(value) => {
                    if value.numer().bits().max(value.denom().bits()) > MAX_BITS {
                        return Err(refused("rational-system coefficient exceeds 4096 bits"));
                    }
                    (value.numer().limb_count() + value.denom().limb_count()) as usize * 8
                }
                Expr::Sym(symbol) => {
                    if !variables.contains(symbol) {
                        return Err(invalid("undeclared typed rational-system variable"));
                    }
                    symbol.name.len()
                }
                Expr::Add(children) | Expr::Mul(children) => {
                    if children.len() > MAX_SOURCE_NODES.saturating_sub(visited + stack.len()) {
                        return Err(refused("rational-system source exceeds its node envelope"));
                    }
                    work.allocate(children.len() * 128)?;
                    stack.extend(children.iter().map(|child| (child, depth + 1)));
                    0
                }
                Expr::Pow(base, exponent) => {
                    if !matches!(exponent.as_ref(),
                        Expr::Integer(value) if value.to_i64().is_some_and(|n| (-256..=256).contains(&n)))
                    {
                        return Err(invalid(
                            "rational systems require literal integer powers between -256 and 256",
                        ));
                    }
                    if visited + stack.len() + 2 > MAX_SOURCE_NODES {
                        return Err(refused("rational-system source exceeds its node envelope"));
                    }
                    work.allocate(256)?;
                    stack.push((exponent.as_ref(), depth + 1));
                    stack.push((base.as_ref(), depth + 1));
                    0
                }
                _ => {
                    return Err(invalid(
                        "unsupported original syntax in exact rational system",
                    ));
                }
            };
            copy_bytes = copy_bytes
                .checked_add(256 + extra)
                .ok_or_else(|| refused("retained rational source size overflow"))?;
        }
    }
    Ok(copy_bytes)
}

fn constant(
    variables: &[Symbol],
    value: &BigRational,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    work.allocate(
        size_of::<MultivariatePoly>() + ring_storage(variables) + 256 + variables.len() * 4,
    )?;
    let mut terms = Terms::new();
    if !value.is_zero() {
        terms.insert(vec![0; variables.len()], work.clone_value(value)?);
    }
    Ok(MultivariatePoly {
        generators: variables.to_vec(),
        terms,
    })
}

fn one(
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    constant(variables, &BigRational::one(), work)
}

fn combine(
    left: &MultivariatePoly,
    right: &MultivariatePoly,
    subtract: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    same_ring(&left.generators, &right.generators)?;
    let mut output = copy_poly(left, work)?;
    for (powers, value) in &right.terms {
        work.allocate(powers.len() * 4)?;
        let value = if subtract {
            work.subtract(&BigRational::zero(), value)?
        } else {
            work.clone_value(value)?
        };
        accumulate(&mut output.terms, powers.clone(), value, work)?;
    }
    Ok(output)
}

fn power(
    polynomial: &MultivariatePoly,
    mut exponent: u32,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let mut value = one(polynomial.generators(), work)?;
    let mut factor = copy_poly(polynomial, work)?;
    while exponent != 0 {
        work.tick(1)?;
        if exponent & 1 != 0 {
            value = multiply_polynomials(&value, &factor, work)?;
        }
        exponent >>= 1;
        if exponent != 0 {
            factor = multiply_polynomials(&factor, &factor, work)?;
        }
    }
    Ok(value)
}

fn guard(
    polynomial: &MultivariatePoly,
    guards: &mut Vec<MultivariatePoly>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if guards.len() == MAX_GUARDS {
        return Err(refused(
            "rational-system source has more than 128 inverse guards",
        ));
    }
    work.allocate(2 * size_of::<MultivariatePoly>())?;
    guards.push(copy_poly(polynomial, work)?);
    Ok(())
}

fn atom(
    source: &Expr,
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    match source {
        Expr::Integer(value) => {
            work.allocate(128 + value.limb_count() as usize * 8)?;
            constant(variables, &BigRational::from_integer(value.clone()), work)
        }
        Expr::Rational(value) => constant(variables, value, work),
        Expr::Sym(symbol) => {
            work.allocate(variables.len() * 4)?;
            let mut powers = vec![0; variables.len()];
            let index = variables
                .iter()
                .position(|variable| variable == symbol)
                .ok_or_else(|| invalid("undeclared typed rational-system variable"))?;
            powers[index] = 1;
            monomial_poly(variables, powers, work)
        }
        _ => Err(invalid("unadmitted rational-system atom")),
    }
}

fn exponent(source: &Expr) -> Result<i64, PolyError> {
    match source {
        Expr::Integer(value) => value
            .to_i64()
            .filter(|n| (-256..=256).contains(n))
            .ok_or_else(|| invalid("unadmitted rational-system exponent")),
        _ => Err(invalid("unadmitted rational-system exponent")),
    }
}

/// Optimizing proposal: left-fold sums and exponentiation by squaring.
fn propose_fraction(
    source: &Expr,
    variables: &[Symbol],
    guards: &mut Vec<MultivariatePoly>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RationalSourceFraction, PolyError> {
    work.tick(1)?;
    let (numerator, denominator) = match source {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Sym(_) => {
            (atom(source, variables, work)?, one(variables, work)?)
        }
        Expr::Add(children) | Expr::Mul(children) => {
            let adding = matches!(source, Expr::Add(_));
            let mut numerator = constant(
                variables,
                &BigRational::from(if adding { 0_i64 } else { 1_i64 }),
                work,
            )?;
            let mut denominator = one(variables, work)?;
            for child in children {
                // Do not skip any source child when the accumulated value is zero.
                let part = propose_fraction(child, variables, guards, work)?;
                numerator = if adding {
                    let left = multiply_polynomials(&numerator, &part.denominator, work)?;
                    let right = multiply_polynomials(&part.numerator, &denominator, work)?;
                    combine(&left, &right, false, work)?
                } else {
                    multiply_polynomials(&numerator, &part.numerator, work)?
                };
                denominator = multiply_polynomials(&denominator, &part.denominator, work)?;
            }
            (numerator, denominator)
        }
        Expr::Pow(base, exp) => {
            let exp = exponent(exp)?;
            let part = propose_fraction(base, variables, guards, work)?;
            if exp <= 0 && part.numerator.is_zero() {
                return Err(invalid(
                    "zero rational base with a nonpositive exponent is undefined",
                ));
            }
            if exp < 0 {
                guard(&part.numerator, guards, work)?;
            }
            let numerator = power(&part.numerator, exp.unsigned_abs() as u32, work)?;
            let denominator = power(&part.denominator, exp.unsigned_abs() as u32, work)?;
            if exp < 0 {
                (denominator, numerator)
            } else {
                (numerator, denominator)
            }
        }
        _ => return Err(invalid("unadmitted rational-system source")),
    };
    Ok(RationalSourceFraction {
        numerator,
        denominator,
    })
}

/// Reference recurrence: sums expand each numerator against every other
/// denominator, and powers use repeated multiplication. No generator pair,
/// trace or claimed guard participates in this computation.
fn reference_fraction(
    source: &Expr,
    variables: &[Symbol],
    guards: &mut Vec<MultivariatePoly>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RationalSourceFraction, PolyError> {
    work.tick(1)?;
    let (numerator, denominator) = match source {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Sym(_) => {
            (atom(source, variables, work)?, one(variables, work)?)
        }
        Expr::Add(children) | Expr::Mul(children) => {
            work.allocate(children.len() * size_of::<RationalSourceFraction>())?;
            let parts = children
                .iter()
                .map(|child| reference_fraction(child, variables, guards, work))
                .collect::<Result<Vec<_>, _>>()?;
            let mut denominator = one(variables, work)?;
            for part in parts.iter().rev() {
                denominator = multiply_polynomials(&denominator, &part.denominator, work)?;
            }
            let numerator = if matches!(source, Expr::Add(_)) {
                let mut sum = constant(variables, &BigRational::zero(), work)?;
                for (index, part) in parts.iter().enumerate() {
                    let mut term = copy_poly(&part.numerator, work)?;
                    for (other, factor) in parts.iter().enumerate().rev() {
                        work.tick(1)?;
                        if other != index {
                            term = multiply_polynomials(&term, &factor.denominator, work)?;
                        }
                    }
                    sum = combine(&sum, &term, false, work)?;
                }
                sum
            } else {
                let mut product = one(variables, work)?;
                for part in parts.iter().rev() {
                    product = multiply_polynomials(&product, &part.numerator, work)?;
                }
                product
            };
            (numerator, denominator)
        }
        Expr::Pow(base, exp) => {
            let exp = exponent(exp)?;
            let part = reference_fraction(base, variables, guards, work)?;
            if part.numerator.is_zero() && exp <= 0 {
                return Err(invalid("undefined original rational power"));
            }
            if exp < 0 {
                guard(&part.numerator, guards, work)?;
            }
            let mut numerator = one(variables, work)?;
            let mut denominator = one(variables, work)?;
            for _ in 0..exp.unsigned_abs() {
                numerator = multiply_polynomials(&numerator, &part.numerator, work)?;
                denominator = multiply_polynomials(&denominator, &part.denominator, work)?;
            }
            if exp < 0 {
                (denominator, numerator)
            } else {
                (numerator, denominator)
            }
        }
        _ => {
            return Err(invalid(
                "unsupported original source in localization reference check",
            ));
        }
    };
    Ok(RationalSourceFraction {
        numerator,
        denominator,
    })
}

fn fresh_auxiliary(
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Symbol, PolyError> {
    for index in 0..=MAX_ORIGINAL_VARIABLES {
        work.tick(variables.len() + 1)?;
        work.allocate(128)?;
        let name = format!("_fsym_rational_inverse_{index}");
        if variables.iter().all(|variable| variable.name != name) {
            return Ok(Symbol::new(name));
        }
    }
    Err(refused("no fresh rational-system auxiliary variable"))
}

fn augmented_ring(
    original: &[Symbol],
    auxiliary: &Symbol,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Symbol>, PolyError> {
    if original
        .iter()
        .any(|variable| variable.name == auxiliary.name)
    {
        return Err(refused(
            "localization auxiliary aliases an original variable",
        ));
    }
    if auxiliary.name.len() > 65_536 {
        return Err(refused("localization auxiliary name exceeds its bound"));
    }
    work.allocate(ring_storage(original) + auxiliary.name.len() + 256)?;
    let mut augmented = original.to_vec();
    augmented.push(auxiliary.clone());
    check_ring(&augmented, MAX_VARIABLES, work)?;
    Ok(augmented)
}

fn embed(
    polynomial: &MultivariatePoly,
    augmented: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    same_ring(polynomial.generators(), &augmented[..augmented.len() - 1])?;
    work.allocate(ring_storage(augmented) + size_of::<MultivariatePoly>())?;
    let mut terms = Terms::new();
    for (powers, coefficient) in polynomial.terms() {
        work.tick(powers.len() + 1)?;
        work.allocate((powers.len() + 1) * 4)?;
        let mut extended = powers.clone();
        extended.push(0);
        let coefficient = work.clone_value(coefficient)?;
        accumulate(&mut terms, extended, coefficient, work)?;
    }
    Ok(MultivariatePoly {
        generators: augmented.to_vec(),
        terms,
    })
}

fn propose_equations(
    fractions: &[RationalSourceFraction],
    guards: &[MultivariatePoly],
    augmented: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    work.allocate((fractions.len() + 1) * size_of::<MultivariatePoly>())?;
    let mut equations = fractions
        .iter()
        .map(|part| embed(&part.numerator, augmented, work))
        .collect::<Result<Vec<_>, _>>()?;
    let original = &augmented[..augmented.len() - 1];
    let mut product = one(original, work)?;
    for condition in guards {
        product = multiply_polynomials(&product, condition, work)?;
    }
    let product = embed(&product, augmented, work)?;
    work.allocate(augmented.len() * 4)?;
    let mut powers = vec![0; augmented.len()];
    *powers
        .last_mut()
        .ok_or_else(|| refused("empty localization ring"))? = 1;
    let auxiliary = monomial_poly(augmented, powers, work)?;
    let inverted = multiply_polynomials(&product, &auxiliary, work)?;
    let unit = one(augmented, work)?;
    equations.push(combine(&inverted, &unit, true, work)?);
    Ok(equations)
}

/// Check coefficient correspondence directly, independently of the generator's
/// embedding and augmented-equation construction routines.
fn check_equation_binding(
    fractions: &[RationalSourceFraction],
    guards: &[MultivariatePoly],
    equations: &[MultivariatePoly],
    augmented: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let width = augmented.len() - 1;
    for (fraction, equation) in fractions.iter().zip(equations) {
        if equation.terms().len() != fraction.numerator.terms().len() {
            return Err(refused(
                "localized numerator lost or gained an original term",
            ));
        }
        for (powers, coefficient) in equation.terms() {
            work.tick(powers.len() + 1)?;
            if powers[width] != 0
                || fraction.numerator.terms().get(&powers[..width]) != Some(coefficient)
            {
                return Err(refused(
                    "localized numerator is not the exact original numerator",
                ));
            }
        }
    }
    let mut product = one(&augmented[..width], work)?;
    for condition in guards.iter().rev() {
        product = multiply_polynomials(condition, &product, work)?;
    }
    let equation = equations
        .last()
        .ok_or_else(|| refused("missing localization equation"))?;
    if equation.terms().len() != product.terms().len() + 1 {
        return Err(refused("inverse equation omits an original source guard"));
    }
    work.allocate(128)?;
    let minus_one = BigRational::from(-1_i64);
    let mut has_constant = false;
    for (powers, coefficient) in equation.terms() {
        work.tick(powers.len() + 1)?;
        match powers[width] {
            0 if powers[..width].iter().all(|&power| power == 0) && coefficient == &minus_one => {
                has_constant = true;
            }
            1 if product.terms().get(&powers[..width]) == Some(coefficient) => {}
            _ => {
                return Err(refused(
                    "inverse equation changed its exact source guard product",
                ));
            }
        }
    }
    if !has_constant {
        return Err(refused("inverse equation lost its nonzero unit condition"));
    }
    Ok(())
}

fn clone_certificate(
    certificate: &RationalLocalizationCertificate,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RationalLocalizationCertificate, PolyError> {
    work.allocate(certificate.auxiliary.name.len() + size_of::<Symbol>())?;
    work.allocate(certificate.fractions.len() * size_of::<RationalSourceFraction>())?;
    let fractions = certificate
        .fractions
        .iter()
        .map(|fraction| {
            Ok(RationalSourceFraction {
                numerator: copy_poly(&fraction.numerator, work)?,
                denominator: copy_poly(&fraction.denominator, work)?,
            })
        })
        .collect::<Result<Vec<_>, PolyError>>()?;
    work.allocate(
        (certificate.guards.len() + certificate.equations.len()) * size_of::<MultivariatePoly>(),
    )?;
    let guards = certificate
        .guards
        .iter()
        .map(|guard| copy_poly(guard, work))
        .collect::<Result<Vec<_>, _>>()?;
    let equations = certificate
        .equations
        .iter()
        .map(|equation| copy_poly(equation, work))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RationalLocalizationCertificate {
        auxiliary: certificate.auxiliary.clone(),
        fractions,
        guards,
        equations,
    })
}

fn lift_equations(
    equations: &[MultivariatePoly],
    variables: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Expr>, PolyError> {
    work.allocate(equations.len() * size_of::<Expr>())?;
    let mut output = Vec::with_capacity(equations.len());
    let mut total_nodes = 0usize;
    for polynomial in equations {
        let nodes = 1 + polynomial
            .terms()
            .keys()
            .map(|powers| 2 + 4 * powers.iter().filter(|&&power| power != 0).count())
            .sum::<usize>();
        total_nodes = total_nodes
            .checked_add(nodes)
            .filter(|&count| count <= 65_536)
            .ok_or_else(|| refused("localization output node envelope exceeded"))?;
        if nodes > MAX_SOURCE_NODES {
            return Err(refused("localization equation exceeds 4096 output nodes"));
        }
        work.tick(nodes)?;
        work.allocate(nodes * (2048 + ring_storage(variables)))?;
        let expression = polynomial.to_expr()?;
        let recovered = bounded_polynomial_from_expr(&expression, variables, work)?;
        if recovered != *polynomial {
            return Err(refused(
                "localized polynomial expression changed during lifting",
            ));
        }
        output.push(expression);
    }
    Ok(output)
}

/// Independently reconstruct every original rational fraction and inverse
/// guard, then check the complete augmented polynomial construction.
pub fn verify_rational_localization(
    sources: &[Expr],
    variables: &[Symbol],
    certificate: &RationalLocalizationCertificate,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRationalLocalization, PolyError> {
    let mut work = PolynomialWork::new(verifier_meter);
    work.tick(1)?;
    let source_bytes = preflight_sources(sources, variables, MAX_ORIGINAL_VARIABLES, &mut work)?;
    if certificate.fractions.len() != sources.len()
        || certificate.guards.len() > MAX_GUARDS
        || certificate.equations.len() != sources.len() + 1
    {
        return Err(refused(
            "localization certificate dimensions differ from the source",
        ));
    }
    let augmented = augmented_ring(variables, &certificate.auxiliary, &mut work)?;
    for fraction in &certificate.fractions {
        for polynomial in [&fraction.numerator, &fraction.denominator] {
            validate_poly(polynomial, &mut work)?;
            same_ring(variables, polynomial.generators())?;
        }
        if fraction.denominator.is_zero() {
            return Err(refused("zero denominator in rational-source certificate"));
        }
    }
    for guard in &certificate.guards {
        validate_poly(guard, &mut work)?;
        same_ring(variables, guard.generators())?;
    }
    for equation in &certificate.equations {
        validate_poly(equation, &mut work)?;
        same_ring(&augmented, equation.generators())?;
    }
    work.allocate(sources.len() * size_of::<RationalSourceFraction>())?;
    let mut guards = Vec::new();
    let fractions = sources
        .iter()
        .map(|source| reference_fraction(source, variables, &mut guards, &mut work))
        .collect::<Result<Vec<_>, _>>()?;
    if fractions != certificate.fractions || guards != certificate.guards {
        return Err(refused(
            "rational localization is not bound to every original source and guard",
        ));
    }
    check_equation_binding(
        &fractions,
        &guards,
        &certificate.equations,
        &augmented,
        &mut work,
    )?;
    let equations = lift_equations(&certificate.equations, &augmented, &mut work)?;
    let certificate = clone_certificate(certificate, &mut work)?;
    work.allocate(source_bytes + std::mem::size_of_val(sources))?;
    let sources = sources.to_vec();
    work.tick(1)?;
    Ok(VerifiedRationalLocalization {
        sources,
        variables: augmented,
        equations,
        certificate,
    })
}

/// Recognize rational source poles and construct a checked exact localization.
/// The auxiliary is appended last, so a finite solver can project its last
/// coordinate away. Positive-dimensional solution publication is not supplied
/// by this API: callers must retain that explicit unsupported outcome.
pub fn metered_localize_rational_system(
    sources: &[Expr],
    variables: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<RationalSystemAdmission, PolyError> {
    let mut generator = PolynomialWork::new(generator_meter);
    check_ring(variables, MAX_VARIABLES, &mut generator)?;
    if !needs_localization(sources, &mut generator)? {
        generator.tick(1)?;
        return Ok(RationalSystemAdmission::Polynomial);
    }
    preflight_sources(sources, variables, MAX_ORIGINAL_VARIABLES, &mut generator)?;
    generator.allocate(sources.len() * size_of::<RationalSourceFraction>())?;
    let mut guards = Vec::new();
    let fractions = sources
        .iter()
        .map(|source| propose_fraction(source, variables, &mut guards, &mut generator))
        .collect::<Result<Vec<_>, _>>()?;
    if guards
        .iter()
        .all(|guard| guard.total_degree().is_none_or(|degree| degree == 0))
    {
        // Symbolic cancellation may expose a nonzero constant inverse. The
        // caller's original polynomial admission remains authoritative.
        generator.tick(1)?;
        return Ok(RationalSystemAdmission::Polynomial);
    }
    let auxiliary = fresh_auxiliary(variables, &mut generator)?;
    let augmented = augmented_ring(variables, &auxiliary, &mut generator)?;
    let equations = propose_equations(&fractions, &guards, &augmented, &mut generator)?;
    let certificate = RationalLocalizationCertificate {
        auxiliary,
        fractions,
        guards,
        equations,
    };
    let result = verify_rational_localization(sources, variables, &certificate, verifier_meter)?;
    generator.allocate(size_of::<VerifiedRationalLocalization>())?;
    generator.tick(1)?;
    verifier_meter.checkpoint().map_err(refused)?;
    Ok(RationalSystemAdmission::Localized(Box::new(result)))
}

/// Independently reconstruct an original rational source without introducing
/// a localization variable or equation. The reference uses common-denominator
/// sums and repeated multiplication, independently of the generator's folds
/// and exponentiation by squaring. No claimed fraction or guard is trusted.
pub fn verify_rational_source(
    source: &Expr,
    variables: &[Symbol],
    certificate: &RationalSourceCertificate,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRationalSource, PolyError> {
    let mut work = PolynomialWork::new(verifier_meter);
    let source_bytes = preflight_sources(
        std::slice::from_ref(source),
        variables,
        MAX_VARIABLES,
        &mut work,
    )?;
    if certificate.guards.len() > MAX_GUARDS {
        return Err(refused("rational source certificate has too many guards"));
    }
    for polynomial in [
        &certificate.fraction.numerator,
        &certificate.fraction.denominator,
    ]
    .into_iter()
    .chain(&certificate.guards)
    {
        validate_poly(polynomial, &mut work)?;
        same_ring(variables, polynomial.generators())?;
    }
    if certificate.fraction.denominator.is_zero() {
        return Err(refused("zero denominator in rational-source certificate"));
    }
    let mut guards = Vec::new();
    let fraction = reference_fraction(source, variables, &mut guards, &mut work)?;
    if fraction != certificate.fraction || guards != certificate.guards {
        return Err(refused(
            "rational fraction or guards do not match the complete original source",
        ));
    }
    // Retain the independently reconstructed parts, rather than another
    // copy of the untrusted certificate. All source/name storage is prepaid.
    work.allocate(source_bytes + ring_storage(variables) + size_of::<VerifiedRationalSource>())?;
    let accepted = VerifiedRationalSource {
        source: source.clone(),
        variables: variables.to_vec(),
        certificate: RationalSourceCertificate { fraction, guards },
    };
    work.tick(1)?;
    Ok(accepted)
}

/// Construct and check an exact rational source over at most eight typed
/// variables. Unlike localization this also admits polynomial and constant
/// sources, and never adds an auxiliary variable. Both phases retain their
/// cumulative 20M-compute/128MiB caps and the caller's cancellation ledger.
pub fn metered_rational_source(
    source: &Expr,
    variables: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedRationalSource, PolyError> {
    let mut generator = PolynomialWork::new(generator_meter);
    preflight_sources(
        std::slice::from_ref(source),
        variables,
        MAX_VARIABLES,
        &mut generator,
    )?;
    let mut guards = Vec::new();
    let fraction = propose_fraction(source, variables, &mut guards, &mut generator)?;
    let certificate = RationalSourceCertificate { fraction, guards };
    let accepted = verify_rational_source(source, variables, &certificate, verifier_meter)?;
    generator.tick(1)?;
    verifier_meter.checkpoint().map_err(refused)?;
    Ok(accepted)
}
