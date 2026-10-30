//! Complete elimination, intersection and principal saturation over QQ.
//!
//! Candidates are reduced Lex bases of an exactly constructed ambient ideal.
//! A separate reference checker binds every ambient input coefficient to the
//! original typed request and checks the *entire* elimination subset, not just
//! membership of the reported polynomials. The elimination theorem then gives
//! ideal equality. Intersection uses `t I + (1-t) J`; saturation uses
//! `I + <1-t f>`. Saturation returns `I : f^infinity`, a closure operation on
//! ideals, not a solution set that excludes all points where `f` vanishes.
//!
//! The ambient Groebner receipt is immutable and already independently checked
//! by the lower-layer kernel. These are in-process composition receipts, not a
//! new portable proof-kernel certificate. Each whole operation keeps the
//! existing 20M-compute/128MiB cumulative cap separately for generator and
//! verifier, including input construction, Groebner work and publication.

#![forbid(unsafe_code)]

use crate::division::{
    MAX_BASIS, MAX_TERMS, MAX_VARIABLES, PolynomialWork, Terms, accumulate, copy_poly, refused,
    ring_storage, same_ring, validate_poly,
};
use crate::{
    MultivariatePoly, PolyError, TermOrder, VerifiedGroebnerBasis, checked_groebner_basis,
};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdealOperation {
    Elimination,
    Intersection,
    Saturation,
}

/// Untrusted projection data around an immutable accepted ambient Lex basis.
/// Its exact original ideal binding and complete projection must be verified.
#[derive(Debug, Clone)]
pub struct IdealOperationCertificate {
    pub ambient_basis: Arc<VerifiedGroebnerBasis>,
    pub projected_basis: Vec<MultivariatePoly>,
}

/// A complete reduced Lex basis of the requested ideal in `generators()`.
/// Elimination retains the relative order of every non-eliminated generator;
/// eliminating all variables returns the constant ring QQ, with basis [] or
/// [1]. Inputs and eliminated symbols retain their original typed identity.
#[derive(Debug, Clone)]
pub struct VerifiedIdealOperation {
    operation: IdealOperation,
    input_generators: Vec<Symbol>,
    generators: Vec<Symbol>,
    inputs: Vec<MultivariatePoly>,
    other_inputs: Vec<MultivariatePoly>,
    saturating_polynomial: Option<MultivariatePoly>,
    eliminated_variables: Vec<Symbol>,
    certificate: IdealOperationCertificate,
}

impl VerifiedIdealOperation {
    pub fn operation(&self) -> IdealOperation {
        self.operation
    }
    pub fn input_generators(&self) -> &[Symbol] {
        &self.input_generators
    }
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }
    pub fn inputs(&self) -> &[MultivariatePoly] {
        &self.inputs
    }
    pub fn other_inputs(&self) -> &[MultivariatePoly] {
        &self.other_inputs
    }
    pub fn saturating_polynomial(&self) -> Option<&MultivariatePoly> {
        self.saturating_polynomial.as_ref()
    }
    pub fn eliminated_variables(&self) -> &[Symbol] {
        &self.eliminated_variables
    }
    pub fn basis(&self) -> &[MultivariatePoly] {
        &self.certificate.projected_basis
    }
    pub fn into_basis(self) -> Vec<MultivariatePoly> {
        self.certificate.projected_basis
    }
    pub fn certificate(&self) -> &IdealOperationCertificate {
        &self.certificate
    }
}

#[derive(Clone, Copy)]
enum Request<'a> {
    Eliminate(&'a [MultivariatePoly], &'a [Symbol]),
    Intersect(&'a [MultivariatePoly], &'a [MultivariatePoly]),
    Saturate(&'a [MultivariatePoly], &'a MultivariatePoly),
}

impl<'a> Request<'a> {
    fn inputs(self) -> &'a [MultivariatePoly] {
        match self {
            Self::Eliminate(inputs, _) | Self::Intersect(inputs, _) | Self::Saturate(inputs, _) => {
                inputs
            }
        }
    }
    fn other_inputs(self) -> &'a [MultivariatePoly] {
        match self {
            Self::Intersect(_, other) => other,
            _ => &[],
        }
    }
    fn operation(self) -> IdealOperation {
        match self {
            Self::Eliminate(_, _) => IdealOperation::Elimination,
            Self::Intersect(_, _) => IdealOperation::Intersection,
            Self::Saturate(_, _) => IdealOperation::Saturation,
        }
    }
}

struct RingPlan {
    ambient: Vec<Symbol>,
    output: Vec<Symbol>,
    /// Position in the ambient ring of each original generator.
    positions: Vec<usize>,
    eliminated_count: usize,
}

fn check_ring(
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(1)?;
    if generators.len() > MAX_VARIABLES {
        return Err(refused(
            "ideal operations admit at most eight ambient variables",
        ));
    }
    let mut name_bytes = 0usize;
    for (index, generator) in generators.iter().enumerate() {
        work.tick(index + 1)?;
        name_bytes = name_bytes
            .checked_add(generator.name.len())
            .filter(|&length| length <= 65_536)
            .ok_or_else(|| refused("ideal generator names exceed 65536 bytes"))?;
        if generators[..index].contains(generator) {
            return Err(refused("duplicate typed ideal generator"));
        }
    }
    Ok(())
}

fn admit(
    request: Request<'_>,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    check_ring(generators, work)?;
    let constructed_count = request
        .inputs()
        .len()
        .checked_add(request.other_inputs().len())
        .and_then(|count| count.checked_add(usize::from(matches!(request, Request::Saturate(..)))))
        .ok_or_else(|| refused("ideal input dimension overflow"))?;
    if constructed_count > MAX_BASIS {
        return Err(refused("constructed ideal exceeds 64 input equations"));
    }
    if !matches!(request, Request::Eliminate(..)) && generators.len() == MAX_VARIABLES {
        return Err(refused(
            "intersection or saturation requires one auxiliary variable",
        ));
    }
    for polynomial in request.inputs().iter().chain(request.other_inputs()) {
        validate_poly(polynomial, work)?;
        same_ring(generators, polynomial.generators())?;
    }
    if let Request::Saturate(_, factor) = request {
        validate_poly(factor, work)?;
        same_ring(generators, factor.generators())?;
    }
    if let Request::Eliminate(_, eliminated) = request {
        if eliminated.len() > generators.len() {
            return Err(refused("too many elimination variables"));
        }
        for (index, variable) in eliminated.iter().enumerate() {
            work.tick(generators.len() + index + 1)?;
            if !generators.contains(variable) || eliminated[..index].contains(variable) {
                return Err(refused(
                    "elimination variables must be distinct typed ring generators",
                ));
            }
        }
    }
    Ok(())
}

fn ring_plan(
    request: Request<'_>,
    generators: &[Symbol],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<RingPlan, PolyError> {
    work.allocate(2 * ring_storage(generators) + 1024)?;
    let (ambient, output, eliminated_count): (Vec<Symbol>, Vec<Symbol>, usize) =
        if let Request::Eliminate(_, eliminated) = request {
            let retained: Vec<_> = generators
                .iter()
                .filter(|symbol| !eliminated.contains(symbol))
                .cloned()
                .collect();
            let ambient = eliminated.iter().chain(&retained).cloned().collect();
            (ambient, retained, eliminated.len())
        } else {
            let mut auxiliary = None;
            for index in 0..=MAX_VARIABLES {
                work.tick(generators.len() + 1)?;
                let name = format!("_fsym_ideal_parameter_{index}");
                if generators.iter().all(|symbol| symbol.name != name) {
                    auxiliary = Some(Symbol::new(name));
                    break;
                }
            }
            let auxiliary = auxiliary.ok_or_else(|| refused("no fresh auxiliary generator"))?;
            let ambient = std::iter::once(auxiliary)
                .chain(generators.iter().cloned())
                .collect();
            (ambient, generators.to_vec(), 1)
        };
    let positions = generators
        .iter()
        .map(|symbol| {
            ambient
                .iter()
                .position(|candidate| candidate == symbol)
                .ok_or_else(|| refused("lost original ideal generator"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RingPlan {
        ambient,
        output,
        positions,
        eliminated_count,
    })
}

#[derive(Clone, Copy)]
enum Multiplier {
    One,
    Auxiliary,
    OneMinusAuxiliary,
}

/// Generator transformation. Reference verification below checks coefficients
/// directly and does not invoke this function or regenerate a Groebner basis.
fn transform(
    source: &MultivariatePoly,
    plan: &RingPlan,
    multiplier: Multiplier,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<MultivariatePoly, PolyError> {
    let copies = if matches!(multiplier, Multiplier::OneMinusAuxiliary) {
        2
    } else {
        1
    };
    if source.terms().len() * copies > MAX_TERMS {
        return Err(refused("auxiliary construction exceeds 4096 terms"));
    }
    let mut terms = Terms::new();
    for (powers, coefficient) in source.terms() {
        work.tick(powers.len() + 1)?;
        work.allocate(plan.ambient.len() * 4)?;
        let mut mapped = vec![0; plan.ambient.len()];
        for (&power, &position) in powers.iter().zip(&plan.positions) {
            mapped[position] = power;
        }
        if matches!(multiplier, Multiplier::Auxiliary) {
            mapped[0] = 1;
        }
        let copied = work.clone_value(coefficient)?;
        if matches!(multiplier, Multiplier::OneMinusAuxiliary) {
            work.allocate(mapped.len() * 4)?;
            accumulate(&mut terms, mapped.clone(), copied, work)?;
            mapped[0] = 1;
            let negative = work.subtract(&BigRational::zero(), coefficient)?;
            accumulate(&mut terms, mapped, negative, work)?;
        } else {
            accumulate(&mut terms, mapped, copied, work)?;
        }
    }
    work.allocate(ring_storage(&plan.ambient) + std::mem::size_of::<MultivariatePoly>())?;
    Ok(MultivariatePoly {
        generators: plan.ambient.clone(),
        terms,
    })
}

fn construct(
    request: Request<'_>,
    plan: &RingPlan,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    let count = request.inputs().len()
        + request.other_inputs().len()
        + usize::from(matches!(request, Request::Saturate(..)));
    work.allocate(count.max(1) * std::mem::size_of::<MultivariatePoly>())?;
    let mut inputs = Vec::with_capacity(count.max(1));
    let left_multiplier = if matches!(request, Request::Intersect(..)) {
        Multiplier::Auxiliary
    } else {
        Multiplier::One
    };
    for source in request.inputs() {
        inputs.push(transform(source, plan, left_multiplier, work)?);
    }
    for source in request.other_inputs() {
        inputs.push(transform(
            source,
            plan,
            Multiplier::OneMinusAuxiliary,
            work,
        )?);
    }
    if let Request::Saturate(_, factor) = request {
        if factor.terms().len() >= MAX_TERMS {
            return Err(refused("saturation relation exceeds 4096 terms"));
        }
        let mut relation = transform(factor, plan, Multiplier::Auxiliary, work)?;
        for coefficient in relation.terms.values_mut() {
            *coefficient = work.subtract(&BigRational::zero(), coefficient)?;
        }
        work.allocate(plan.ambient.len() * 4)?;
        accumulate(
            &mut relation.terms,
            vec![0; plan.ambient.len()],
            BigRational::one(),
            work,
        )?;
        inputs.push(relation);
    }
    if inputs.is_empty() {
        // Keep the ambient typed ring even for the zero ideal.
        work.allocate(ring_storage(&plan.ambient) + std::mem::size_of::<MultivariatePoly>())?;
        inputs.push(MultivariatePoly::zero(plan.ambient.clone()));
    }
    Ok(inputs)
}

fn free_of_prefix(
    polynomial: &MultivariatePoly,
    count: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<bool, PolyError> {
    work.tick(polynomial.terms().len().max(1) * count.max(1))?;
    Ok(polynomial
        .terms()
        .keys()
        .all(|powers| powers[..count].iter().all(|&power| power == 0)))
}

fn project(
    basis: &VerifiedGroebnerBasis,
    plan: &RingPlan,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    work.allocate(std::mem::size_of_val(basis.basis()))?;
    let mut result = Vec::with_capacity(basis.basis().len());
    for polynomial in basis.basis() {
        if !free_of_prefix(polynomial, plan.eliminated_count, work)? {
            continue;
        }
        let mut terms = Terms::new();
        for (powers, coefficient) in polynomial.terms() {
            work.allocate(plan.output.len() * 4)?;
            let projected = powers[plan.eliminated_count..].to_vec();
            let value = work.clone_value(coefficient)?;
            accumulate(&mut terms, projected, value, work)?;
        }
        work.allocate(ring_storage(&plan.output) + std::mem::size_of::<MultivariatePoly>())?;
        result.push(MultivariatePoly {
            generators: plan.output.clone(),
            terms,
        });
    }
    Ok(result)
}

/// Reference coefficient matcher for an original polynomial embedded in the
/// exactly requested ring, optionally multiplied by t or (1-t). The generator
/// transform is not reused, and unexpected terms cannot pass a length check.
fn verify_transformed(
    source: &MultivariatePoly,
    actual: &MultivariatePoly,
    plan: &RingPlan,
    multiplier: Multiplier,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    validate_poly(actual, work)?;
    same_ring(actual.generators(), &plan.ambient)?;
    let copies = if matches!(multiplier, Multiplier::OneMinusAuxiliary) {
        2
    } else {
        1
    };
    if actual.terms().len() != source.terms().len() * copies {
        return Err(refused("constructed ideal input changed its term coverage"));
    }
    for (powers, value) in actual.terms() {
        work.tick(powers.len() + 1)?;
        let auxiliary_power = if matches!(multiplier, Multiplier::One)
            && plan.ambient.len() == source.generators().len()
        {
            None
        } else {
            Some(powers[0])
        };
        let negative = match (multiplier, auxiliary_power) {
            (Multiplier::One, None | Some(0))
            | (Multiplier::Auxiliary, Some(1))
            | (Multiplier::OneMinusAuxiliary, Some(0)) => false,
            (Multiplier::OneMinusAuxiliary, Some(1)) => true,
            _ => {
                return Err(refused(
                    "constructed ideal input changed its auxiliary factor",
                ));
            }
        };
        work.allocate(source.generators().len() * 4)?;
        let original: Vec<_> = plan
            .positions
            .iter()
            .map(|&position| powers[position])
            .collect();
        let expected = source
            .terms()
            .get(&original)
            .ok_or_else(|| refused("constructed input added an original monomial"))?;
        if negative {
            if work.add(value, expected)? != BigRational::zero() {
                return Err(refused("constructed input changed an original coefficient"));
            }
        } else if value != expected {
            return Err(refused("constructed input changed an original coefficient"));
        }
    }
    Ok(())
}

fn verify_saturation_relation(
    factor: &MultivariatePoly,
    actual: &MultivariatePoly,
    plan: &RingPlan,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    validate_poly(actual, work)?;
    same_ring(actual.generators(), &plan.ambient)?;
    if actual.terms().len() != factor.terms().len() + 1 {
        return Err(refused("saturation relation omitted or added terms"));
    }
    let mut unit_seen = false;
    for (powers, value) in actual.terms() {
        work.tick(powers.len() + 1)?;
        if powers.iter().all(|&power| power == 0) {
            if !value.is_one() {
                return Err(refused("saturation relation lost its unit constant"));
            }
            unit_seen = true;
            continue;
        }
        if powers[0] != 1 {
            return Err(refused(
                "saturation relation is not linear in its fresh auxiliary",
            ));
        }
        let expected = factor
            .terms()
            .get(&powers[1..])
            .ok_or_else(|| refused("saturation factor changed its monomials"))?;
        if !work.add(value, expected)?.is_zero() {
            return Err(refused("saturation factor changed its coefficients"));
        }
    }
    if !unit_seen {
        return Err(refused("saturation relation has no unit constant"));
    }
    Ok(())
}

fn verify_projection(
    basis: &VerifiedGroebnerBasis,
    projected: &[MultivariatePoly],
    plan: &RingPlan,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if projected.len() > MAX_BASIS || basis.basis().len() > MAX_BASIS {
        return Err(refused("projected basis exceeds 64 elements"));
    }
    for polynomial in projected {
        validate_poly(polynomial, work)?;
        same_ring(polynomial.generators(), &plan.output)?;
    }
    let mut next = 0;
    for polynomial in basis.basis() {
        validate_poly(polynomial, work)?;
        same_ring(polynomial.generators(), &plan.ambient)?;
        if !free_of_prefix(polynomial, plan.eliminated_count, work)? {
            continue;
        }
        let actual = projected
            .get(next)
            .ok_or_else(|| refused("elimination basis omitted a required generator"))?;
        if actual.terms().len() != polynomial.terms().len() {
            return Err(refused("elimination changed the projected polynomial"));
        }
        for (powers, value) in polynomial.terms() {
            work.tick(powers.len() + 1)?;
            if actual.terms().get(&powers[plan.eliminated_count..]) != Some(value) {
                return Err(refused(
                    "elimination changed a projected monomial or coefficient",
                ));
            }
        }
        next += 1;
    }
    if next != projected.len() {
        return Err(refused("elimination basis added unrelated generators"));
    }
    Ok(())
}

fn copy_polynomials(
    polynomials: &[MultivariatePoly],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    work.allocate(std::mem::size_of_val(polynomials))?;
    let mut copied = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        copied.push(copy_poly(polynomial, work)?);
    }
    Ok(copied)
}

fn verify(
    request: Request<'_>,
    generators: &[Symbol],
    certificate: &IdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(request, generators, &mut work)?;
    let plan = ring_plan(request, generators, &mut work)?;
    let basis = certificate.ambient_basis.as_ref();
    if basis.order() != TermOrder::Lex {
        return Err(refused("ideal projection requires an accepted Lex basis"));
    }
    let expected_count = request.inputs().len()
        + request.other_inputs().len()
        + usize::from(matches!(request, Request::Saturate(..)));
    let actual = basis.input_basis();
    if actual.len() != expected_count.max(1) {
        return Err(refused(
            "ambient basis is not bound to every original input",
        ));
    }
    if expected_count == 0 {
        validate_poly(&actual[0], &mut work)?;
        same_ring(actual[0].generators(), &plan.ambient)?;
        if !actual[0].is_zero() {
            return Err(refused(
                "empty original ideal was rebound to a nonzero equation",
            ));
        }
    }
    let multiplier = if matches!(request, Request::Intersect(..)) {
        Multiplier::Auxiliary
    } else {
        Multiplier::One
    };
    for (source, actual) in request.inputs().iter().zip(actual) {
        verify_transformed(source, actual, &plan, multiplier, &mut work)?;
    }
    for (source, actual) in request
        .other_inputs()
        .iter()
        .zip(&actual[request.inputs().len()..])
    {
        verify_transformed(
            source,
            actual,
            &plan,
            Multiplier::OneMinusAuxiliary,
            &mut work,
        )?;
    }
    if let Request::Saturate(_, factor) = request {
        verify_saturation_relation(
            factor,
            actual
                .last()
                .ok_or_else(|| refused("missing saturation relation"))?,
            &plan,
            &mut work,
        )?;
    }
    verify_projection(basis, &certificate.projected_basis, &plan, &mut work)?;
    // Retain the exact request and proposed projection after bounded copies.
    // The potentially large accepted ambient receipt is shared through Arc.
    work.allocate(std::mem::size_of::<VerifiedIdealOperation>() + ring_storage(generators))?;
    let inputs = copy_polynomials(request.inputs(), &mut work)?;
    let other_inputs = copy_polynomials(request.other_inputs(), &mut work)?;
    let saturating_polynomial = if let Request::Saturate(_, polynomial) = request {
        Some(copy_poly(polynomial, &mut work)?)
    } else {
        None
    };
    let eliminated_variables = if let Request::Eliminate(_, eliminated) = request {
        work.allocate(ring_storage(eliminated))?;
        eliminated.to_vec()
    } else {
        Vec::new()
    };
    let projected_basis = copy_polynomials(&certificate.projected_basis, &mut work)?;
    let accepted = VerifiedIdealOperation {
        operation: request.operation(),
        input_generators: generators.to_vec(),
        generators: plan.output,
        inputs,
        other_inputs,
        saturating_polynomial,
        eliminated_variables,
        certificate: IdealOperationCertificate {
            ambient_basis: Arc::clone(&certificate.ambient_basis),
            projected_basis,
        },
    };
    work.checkpoint().map_err(refused)?;
    Ok(accepted)
}

fn generate(
    request: Request<'_>,
    generators: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    // The outer work meters are also passed to every nested operation, so
    // nested helper envelopes cannot reset either cumulative phase.
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(request, generators, &mut generation)?;
    let plan = ring_plan(request, generators, &mut generation)?;
    let inputs = construct(request, &plan, &mut generation)?;
    let ambient_basis =
        checked_groebner_basis(&inputs, TermOrder::Lex, &mut generation, &mut verification)?;
    let projected_basis = project(&ambient_basis, &plan, &mut generation)?;
    generation.allocate(
        std::mem::size_of::<VerifiedGroebnerBasis>()
            + std::mem::size_of::<IdealOperationCertificate>()
            + 32,
    )?;
    let certificate = IdealOperationCertificate {
        ambient_basis: Arc::new(ambient_basis),
        projected_basis,
    };
    let accepted = verify(request, generators, &certificate, &mut verification)?;
    generation.checkpoint().map_err(refused)?;
    verification.checkpoint().map_err(refused)?;
    Ok(accepted)
}

/// Compute the complete elimination ideal for any distinct typed subset of
/// the ring generators. All retained variables keep their original order.
pub fn checked_elimination_ideal(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    eliminated: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    generate(
        Request::Eliminate(inputs, eliminated),
        generators,
        generator,
        verifier,
    )
}

pub fn verify_elimination_ideal(
    inputs: &[MultivariatePoly],
    generators: &[Symbol],
    eliminated: &[Symbol],
    certificate: &IdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    verify(
        Request::Eliminate(inputs, eliminated),
        generators,
        certificate,
        verifier,
    )
}

/// Compute I intersect J as the complete elimination of t from t I+(1-t) J.
/// Both input ideals must use the same typed QQ ring in the same order.
pub fn checked_ideal_intersection(
    left: &[MultivariatePoly],
    right: &[MultivariatePoly],
    generators: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    generate(
        Request::Intersect(left, right),
        generators,
        generator,
        verifier,
    )
}

pub fn verify_ideal_intersection(
    left: &[MultivariatePoly],
    right: &[MultivariatePoly],
    generators: &[Symbol],
    certificate: &IdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    verify(
        Request::Intersect(left, right),
        generators,
        certificate,
        verifier,
    )
}

/// Compute the principal saturation I:f^infinity. This is an ideal closure,
/// not a domain-filtered point set; points with f=0 may remain in its variety.
pub fn checked_ideal_saturation(
    inputs: &[MultivariatePoly],
    factor: &MultivariatePoly,
    generators: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    generate(
        Request::Saturate(inputs, factor),
        generators,
        generator,
        verifier,
    )
}

pub fn verify_ideal_saturation(
    inputs: &[MultivariatePoly],
    factor: &MultivariatePoly,
    generators: &[Symbol],
    certificate: &IdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIdealOperation, PolyError> {
    verify(
        Request::Saturate(inputs, factor),
        generators,
        certificate,
        verifier,
    )
}
