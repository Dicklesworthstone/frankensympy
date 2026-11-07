//! Complete elimination, intersection and principal saturation over F_p.
//!
//! Generation constructs an ambient ideal and obtains an independently checked
//! reduced Lex basis. A separate composition checker binds that immutable
//! receipt to every coefficient of the original typed request and verifies the
//! entire projected basis. The elimination theorem establishes ideal equality,
//! including multiplicities; these are not merely polynomial consequences.
//!
//! Intersection uses t I + (1-t) J; saturation uses I + <1-t f>. The latter is
//! the ideal closure I:f^infinity, not a point set excluding f=0. No equations
//! restricting variables to F_p-rational points are introduced. All variables
//! eliminated yields the explicit constant field F_p, with basis [] or [1].
//!
//! At most eight ambient variables, 32 constructed inputs, degree 64 and 4096
//! terms per polynomial are admitted. The auxiliary variable counts toward
//! those existing bounds. Separate cumulative 20M-step / 128MiB generator and
//! verifier envelopes cover construction, nested Groebner work and publication.
//! These are immutable in-process composition receipts, not portable proofs.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    MAX_BASIS, MAX_DEGREE, MAX_TERMS, MAX_VARIABLES, accumulate, admit_ring, admit_terms,
    copied_polynomial, copy_generators, empty, monomial_copy, same_ring,
};
use crate::{
    IdealOperation, PolyError, PrimeFieldMultivariatePolynomial as Polynomial, TermOrder,
    VerifiedPrimeFieldGroebnerBasis, checked_prime_field_groebner_basis,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;
use std::sync::Arc;

/// Untrusted projection around an immutable, independently accepted ambient
/// basis. The original ideal binding and complete projection remain to check.
#[derive(Debug, Clone)]
pub struct PrimeFieldIdealOperationCertificate {
    pub ambient_basis: Arc<VerifiedPrimeFieldGroebnerBasis>,
    pub projected_basis: Vec<Polynomial>,
}

/// Complete reduced Lex basis of the requested ideal over the retained prime.
/// The retained variables preserve their original typed identity and order.
#[derive(Debug, Clone)]
pub struct VerifiedPrimeFieldIdealOperation {
    operation: IdealOperation,
    input_generators: Vec<Symbol>,
    generators: Vec<Symbol>,
    prime: u64,
    inputs: Vec<Polynomial>,
    other_inputs: Vec<Polynomial>,
    saturating_polynomial: Option<Polynomial>,
    eliminated_variables: Vec<Symbol>,
    certificate: PrimeFieldIdealOperationCertificate,
}

impl VerifiedPrimeFieldIdealOperation {
    pub fn operation(&self) -> IdealOperation {
        self.operation
    }
    pub fn input_generators(&self) -> &[Symbol] {
        &self.input_generators
    }
    pub fn generators(&self) -> &[Symbol] {
        &self.generators
    }
    pub fn prime(&self) -> u64 {
        self.prime
    }
    pub fn inputs(&self) -> &[Polynomial] {
        &self.inputs
    }
    pub fn other_inputs(&self) -> &[Polynomial] {
        &self.other_inputs
    }
    pub fn saturating_polynomial(&self) -> Option<&Polynomial> {
        self.saturating_polynomial.as_ref()
    }
    pub fn eliminated_variables(&self) -> &[Symbol] {
        &self.eliminated_variables
    }
    pub fn basis(&self) -> &[Polynomial] {
        &self.certificate.projected_basis
    }
    pub fn into_basis(self) -> Vec<Polynomial> {
        self.certificate.projected_basis
    }
    pub fn certificate(&self) -> &PrimeFieldIdealOperationCertificate {
        &self.certificate
    }
}

#[derive(Clone, Copy)]
enum Request<'a> {
    Eliminate(&'a [Polynomial], &'a [Symbol]),
    Intersect(&'a [Polynomial], &'a [Polynomial]),
    Saturate(&'a [Polynomial], &'a Polynomial),
}

impl<'a> Request<'a> {
    fn inputs(self) -> &'a [Polynomial] {
        match self {
            Self::Eliminate(inputs, _) | Self::Intersect(inputs, _) | Self::Saturate(inputs, _) => {
                inputs
            }
        }
    }
    fn other_inputs(self) -> &'a [Polynomial] {
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
    fn constructed_count(self) -> Option<usize> {
        self.inputs()
            .len()
            .checked_add(self.other_inputs().len())?
            .checked_add(usize::from(matches!(self, Self::Saturate(..))))
    }
}

struct RingPlan {
    ambient: Vec<Symbol>,
    output: Vec<Symbol>,
    /// Ambient position of each original typed generator.
    positions: Vec<usize>,
    eliminated_count: usize,
}

fn admit(
    request: Request<'_>,
    generators: &[Symbol],
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(generators, prime, work)?;
    if request
        .constructed_count()
        .is_none_or(|count| count > MAX_BASIS)
    {
        return Err(refused(
            "prime-field ideal construction exceeds 32 input equations",
        ));
    }
    if !matches!(request, Request::Eliminate(..)) && generators.len() == MAX_VARIABLES {
        return Err(refused(
            "prime-field ideal operation needs one auxiliary variable",
        ));
    }
    for polynomial in request.inputs().iter().chain(request.other_inputs()) {
        same_ring(polynomial, generators, prime, work)?;
    }
    if let Request::Saturate(_, factor) = request {
        same_ring(factor, generators, prime, work)?;
    }
    if let Request::Eliminate(_, eliminated) = request {
        if eliminated.len() > generators.len() {
            return Err(refused("too many prime-field elimination variables"));
        }
        for (index, variable) in eliminated.iter().enumerate() {
            work.tick(generators.len() + index + 1)?;
            if !generators.contains(variable) || eliminated[..index].contains(variable) {
                return Err(refused(
                    "elimination needs distinct original typed generators",
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
    // Every source has already passed the bounded name/identity admission.
    let mut ambient = copy_generators(generators, work)?;
    let (output, eliminated_count) = if let Request::Eliminate(_, eliminated) = request {
        work.tick((generators.len() + 1) * (eliminated.len() + 1))?;
        let mut retained = copy_generators(generators, work)?;
        retained.retain(|symbol| !eliminated.contains(symbol));
        // Move owned symbols in place: no second uncharged name clone, and the
        // relative order of the remaining original generators is preserved.
        for (index, symbol) in eliminated.iter().enumerate() {
            let position = ambient
                .iter()
                .position(|candidate| candidate == symbol)
                .ok_or_else(|| refused("lost original elimination generator"))?;
            let variable = ambient.remove(position);
            ambient.insert(index, variable);
        }
        (retained, eliminated.len())
    } else {
        let mut auxiliary = None;
        for index in 0..=MAX_VARIABLES {
            work.tick(generators.len() + 1)?;
            work.allocate(128)?;
            let name = format!("_fsym_prime_field_ideal_parameter_{index}");
            if generators.iter().all(|symbol| symbol.name != name) {
                auxiliary = Some(Symbol::new(name));
                break;
            }
        }
        let auxiliary = auxiliary.ok_or_else(|| refused("no fresh ideal auxiliary"))?;
        work.allocate((ambient.len() + 1) * std::mem::size_of::<Symbol>())?;
        ambient
            .try_reserve_exact(1)
            .map_err(|_| refused("ideal ambient ring allocation"))?;
        ambient.insert(0, auxiliary);
        (copy_generators(generators, work)?, 1)
    };
    work.tick((generators.len() + 1) * (ambient.len() + 1))?;
    work.allocate(std::mem::size_of_val(generators) + 1)?;
    let positions = generators
        .iter()
        .map(|symbol| {
            ambient
                .iter()
                .position(|candidate| candidate == symbol)
                .ok_or_else(|| refused("lost original typed ideal generator"))
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

/// Independently establish the exact elimination theorem's variable partition.
/// Source exponent matching below likewise does not trust `plan.positions`.
fn verify_ring_plan(
    request: Request<'_>,
    generators: &[Symbol],
    plan: &RingPlan,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick((generators.len() + 1) * (generators.len() + 1))?;
    if let Request::Eliminate(_, eliminated) = request {
        if plan.eliminated_count != eliminated.len()
            || plan.ambient.len() != generators.len()
            || plan.ambient.get(..eliminated.len()) != Some(eliminated)
        {
            return Err(refused(
                "ambient prefix changed the requested eliminated variables",
            ));
        }
        let mut next = 0;
        for symbol in generators {
            if eliminated.contains(symbol) {
                continue;
            }
            if plan.output.get(next) != Some(symbol)
                || plan.ambient.get(eliminated.len() + next) != Some(symbol)
            {
                return Err(refused(
                    "elimination changed the retained typed variable order",
                ));
            }
            next += 1;
        }
        if next != plan.output.len() {
            return Err(refused("elimination added unrelated output variables"));
        }
    } else if plan.eliminated_count != 1
        || plan.ambient.len() != generators.len() + 1
        || plan.ambient.get(1..) != Some(generators)
        || plan.output != generators
        || plan.ambient.first().is_none_or(|auxiliary| {
            generators
                .iter()
                .any(|source| source.name == auxiliary.name)
        })
    {
        return Err(refused(
            "augmented ideal lost the fresh auxiliary or original typed ring",
        ));
    }
    Ok(())
}

/// Candidate construction; the reference coefficient matcher does not call it.
fn transform(
    source: &Polynomial,
    plan: &RingPlan,
    multiplier: Multiplier,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Polynomial, PolyError> {
    let copies = if matches!(multiplier, Multiplier::OneMinusAuxiliary) {
        2
    } else {
        1
    };
    if source.terms().len() * copies > MAX_TERMS {
        return Err(refused(
            "prime-field auxiliary construction exceeds 4096 terms",
        ));
    }
    let prime = source.prime();
    let mut terms = empty(work)?;
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
        if matches!(multiplier, Multiplier::OneMinusAuxiliary) {
            accumulate(
                &mut terms,
                monomial_copy(&mapped, work)?,
                *coefficient,
                prime,
                work,
            )?;
            mapped[0] = 1;
            accumulate(&mut terms, mapped, prime - coefficient, prime, work)?;
        } else {
            accumulate(&mut terms, mapped, *coefficient, prime, work)?;
        }
    }
    admit_terms(&terms, plan.ambient.len(), prime, MAX_DEGREE, false, work)?;
    copied_polynomial(&plan.ambient, prime, &terms, work)
}

fn construct(
    request: Request<'_>,
    plan: &RingPlan,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Polynomial>, PolyError> {
    let count = request
        .constructed_count()
        .ok_or_else(|| refused("ideal input count overflow"))?;
    work.allocate(count * std::mem::size_of::<Polynomial>())?;
    let mut inputs = Vec::with_capacity(count);
    let multiplier = if matches!(request, Request::Intersect(..)) {
        Multiplier::Auxiliary
    } else {
        Multiplier::One
    };
    for source in request.inputs() {
        inputs.push(transform(source, plan, multiplier, work)?);
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
            return Err(refused(
                "prime-field saturation relation exceeds 4096 terms",
            ));
        }
        let mut terms = empty(work)?;
        work.allocate(plan.ambient.len() * 4)?;
        accumulate(&mut terms, vec![0; plan.ambient.len()], 1, prime, work)?;
        for (powers, coefficient) in factor.terms() {
            work.tick(powers.len() + 1)?;
            work.allocate(plan.ambient.len() * 4)?;
            let mut mapped = Vec::with_capacity(plan.ambient.len());
            mapped.push(1);
            mapped.extend_from_slice(powers);
            accumulate(&mut terms, mapped, prime - coefficient, prime, work)?;
        }
        admit_terms(&terms, plan.ambient.len(), prime, MAX_DEGREE, false, work)?;
        inputs.push(copied_polynomial(&plan.ambient, prime, &terms, work)?);
    }
    // Unlike an inferred ring, the Groebner API retains the explicit generators
    // and prime even for empty input; the zero ideal needs no sentinel equation.
    Ok(inputs)
}

fn project(
    basis: &VerifiedPrimeFieldGroebnerBasis,
    plan: &RingPlan,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Polynomial>, PolyError> {
    work.allocate(std::mem::size_of_val(basis.basis()))?;
    let mut result = Vec::with_capacity(basis.basis().len());
    for polynomial in basis.basis() {
        work.tick(polynomial.terms().len().max(1) * plan.eliminated_count.max(1))?;
        if polynomial.terms().keys().any(|powers| {
            powers[..plan.eliminated_count]
                .iter()
                .any(|&power| power != 0)
        }) {
            continue;
        }
        let mut terms = empty(work)?;
        for (powers, coefficient) in polynomial.terms() {
            let monomial = monomial_copy(&powers[plan.eliminated_count..], work)?;
            accumulate(&mut terms, monomial, *coefficient, basis.prime(), work)?;
        }
        result.push(copied_polynomial(
            &plan.output,
            basis.prime(),
            &terms,
            work,
        )?);
    }
    Ok(result)
}

/// Check actual ambient coefficients directly against the original request;
/// no generator transformation, multiplication or Groebner search is replayed.
fn verify_transformed(
    source: &Polynomial,
    actual: &Polynomial,
    plan: &RingPlan,
    multiplier: Multiplier,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    same_ring(actual, &plan.ambient, prime, work)?;
    let copies = if matches!(multiplier, Multiplier::OneMinusAuxiliary) {
        2
    } else {
        1
    };
    if actual.terms().len() != source.terms().len() * copies {
        return Err(refused("ambient ideal changed original term coverage"));
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
            _ => return Err(refused("ambient ideal changed its auxiliary multiplier")),
        };
        work.tick((source.generators().len() + 1) * (actual.generators().len() + 1))?;
        work.allocate(source.generators().len() * 4)?;
        let original = source
            .generators()
            .iter()
            .map(|symbol| {
                actual
                    .generators()
                    .iter()
                    .position(|candidate| candidate == symbol)
                    .map(|position| powers[position])
                    .ok_or_else(|| refused("ambient input lost an original typed coordinate"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected = source
            .terms()
            .get(&original)
            .ok_or_else(|| refused("ambient ideal added an original monomial"))?;
        if (negative && !(value + expected).is_multiple_of(prime))
            || (!negative && value != expected)
        {
            return Err(refused("ambient ideal changed an original coefficient"));
        }
    }
    Ok(())
}

fn verify_saturation_relation(
    factor: &Polynomial,
    actual: &Polynomial,
    plan: &RingPlan,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    same_ring(actual, &plan.ambient, prime, work)?;
    if actual.terms().len() != factor.terms().len() + 1 {
        return Err(refused("saturation relation omitted or added terms"));
    }
    let mut unit_seen = false;
    for (powers, value) in actual.terms() {
        work.tick(powers.len() + 1)?;
        if powers.iter().all(|&power| power == 0) {
            if *value != 1 {
                return Err(refused("saturation relation lost its unit constant"));
            }
            unit_seen = true;
            continue;
        }
        if powers[0] != 1 {
            return Err(refused(
                "saturation relation is not linear in the fresh auxiliary",
            ));
        }
        let expected = factor
            .terms()
            .get(&powers[1..])
            .ok_or_else(|| refused("saturation factor changed its monomials"))?;
        if !(value + expected).is_multiple_of(prime) {
            return Err(refused("saturation factor changed its coefficients"));
        }
    }
    if !unit_seen {
        return Err(refused("saturation relation has no unit constant"));
    }
    Ok(())
}

fn verify_projection(
    basis: &VerifiedPrimeFieldGroebnerBasis,
    projected: &[Polynomial],
    plan: &RingPlan,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    if projected.len() > MAX_BASIS || basis.basis().len() > MAX_BASIS {
        return Err(refused(
            "projected prime-field ideal exceeds 32 basis elements",
        ));
    }
    for polynomial in projected {
        same_ring(polynomial, &plan.output, prime, work)?;
    }
    let mut next = 0;
    for polynomial in basis.basis() {
        same_ring(polynomial, &plan.ambient, prime, work)?;
        let mut selected = true;
        for monomial in polynomial.terms().keys() {
            for exponent in &monomial[..plan.eliminated_count] {
                work.tick(1)?;
                selected &= *exponent == 0;
            }
        }
        if !selected {
            continue;
        }
        let actual = projected
            .get(next)
            .ok_or_else(|| refused("elimination omitted a required basis polynomial"))?;
        if actual.terms().len() != polynomial.terms().len() {
            return Err(refused("elimination changed projected term coverage"));
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
        return Err(refused("elimination added unrelated basis polynomials"));
    }
    Ok(())
}

fn copy_polynomials(
    polynomials: &[Polynomial],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Polynomial>, PolyError> {
    work.allocate(std::mem::size_of_val(polynomials))?;
    let mut copied = Vec::with_capacity(polynomials.len());
    for polynomial in polynomials {
        copied.push(copied_polynomial(
            polynomial.generators(),
            polynomial.prime(),
            polynomial.terms(),
            work,
        )?);
    }
    Ok(copied)
}

fn verify(
    request: Request<'_>,
    generators: &[Symbol],
    prime: u64,
    certificate: &PrimeFieldIdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    admit(request, generators, prime, &mut work)?;
    let plan = ring_plan(request, generators, &mut work)?;
    verify_ring_plan(request, generators, &plan, &mut work)?;
    let basis = certificate.ambient_basis.as_ref();
    if basis.order() != TermOrder::Lex
        || basis.prime() != prime
        || basis.generators() != plan.ambient
    {
        return Err(refused(
            "ambient ideal requires the exact typed prime-field Lex ring",
        ));
    }
    let expected_count = request
        .constructed_count()
        .ok_or_else(|| refused("ideal input count overflow"))?;
    let actual = basis.inputs();
    if actual.len() != expected_count {
        return Err(refused(
            "ambient ideal is not bound to every original input",
        ));
    }
    let multiplier = if matches!(request, Request::Intersect(..)) {
        Multiplier::Auxiliary
    } else {
        Multiplier::One
    };
    for (source, actual) in request.inputs().iter().zip(actual) {
        verify_transformed(source, actual, &plan, multiplier, prime, &mut work)?;
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
            prime,
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
            prime,
            &mut work,
        )?;
    }
    verify_projection(basis, &certificate.projected_basis, &plan, prime, &mut work)?;
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldIdealOperation>())?;
    let inputs = copy_polynomials(request.inputs(), &mut work)?;
    let other_inputs = copy_polynomials(request.other_inputs(), &mut work)?;
    let saturating_polynomial = if let Request::Saturate(_, factor) = request {
        Some(copied_polynomial(
            factor.generators(),
            prime,
            factor.terms(),
            &mut work,
        )?)
    } else {
        None
    };
    let eliminated_variables = if let Request::Eliminate(_, eliminated) = request {
        copy_generators(eliminated, &mut work)?
    } else {
        Vec::new()
    };
    let accepted = VerifiedPrimeFieldIdealOperation {
        operation: request.operation(),
        input_generators: copy_generators(generators, &mut work)?,
        generators: plan.output,
        prime,
        inputs,
        other_inputs,
        saturating_polynomial,
        eliminated_variables,
        certificate: PrimeFieldIdealOperationCertificate {
            ambient_basis: Arc::clone(&certificate.ambient_basis),
            projected_basis: copy_polynomials(&certificate.projected_basis, &mut work)?,
        },
    };
    work.tick(1)?;
    Ok(accepted)
}

fn generate(
    request: Request<'_>,
    generators: &[Symbol],
    prime: u64,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    // Nested helpers charge these same outer phase meters; no stage resets the
    // cumulative construction, Groebner, projection or acceptance accounting.
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    admit(request, generators, prime, &mut generation)?;
    let plan = ring_plan(request, generators, &mut generation)?;
    let inputs = construct(request, &plan, prime, &mut generation)?;
    let ambient_basis = checked_prime_field_groebner_basis(
        &plan.ambient,
        prime,
        &inputs,
        TermOrder::Lex,
        &mut generation,
        &mut verification,
    )?;
    let projected_basis = project(&ambient_basis, &plan, &mut generation)?;
    generation.allocate(
        std::mem::size_of::<VerifiedPrimeFieldGroebnerBasis>()
            + std::mem::size_of::<PrimeFieldIdealOperationCertificate>()
            + 32,
    )?;
    let certificate = PrimeFieldIdealOperationCertificate {
        ambient_basis: Arc::new(ambient_basis),
        projected_basis,
    };
    let accepted = verify(request, generators, prime, &certificate, &mut verification)?;
    generation.checkpoint().map_err(refused)?;
    verification.checkpoint().map_err(refused)?;
    Ok(accepted)
}

/// Compute the complete elimination ideal of any distinct typed subset.
/// The remaining variables keep their original order; all eliminated yields F_p.
pub fn checked_prime_field_elimination_ideal(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    eliminated: &[Symbol],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    generate(
        Request::Eliminate(inputs, eliminated),
        generators,
        prime,
        generator,
        verifier,
    )
}

pub fn verify_prime_field_elimination_ideal(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    eliminated: &[Symbol],
    certificate: &PrimeFieldIdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    verify(
        Request::Eliminate(inputs, eliminated),
        generators,
        prime,
        certificate,
        verifier,
    )
}

/// Compute I intersect J in the same ordered typed prime-field ring.
pub fn checked_prime_field_ideal_intersection(
    generators: &[Symbol],
    prime: u64,
    left: &[Polynomial],
    right: &[Polynomial],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    generate(
        Request::Intersect(left, right),
        generators,
        prime,
        generator,
        verifier,
    )
}

pub fn verify_prime_field_ideal_intersection(
    generators: &[Symbol],
    prime: u64,
    left: &[Polynomial],
    right: &[Polynomial],
    certificate: &PrimeFieldIdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    verify(
        Request::Intersect(left, right),
        generators,
        prime,
        certificate,
        verifier,
    )
}

/// Compute I:f^infinity as an ideal closure, preserving all remaining nilpotents.
/// This does not return a solution set with points satisfying f=0 excluded.
pub fn checked_prime_field_ideal_saturation(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    factor: &Polynomial,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    generate(
        Request::Saturate(inputs, factor),
        generators,
        prime,
        generator,
        verifier,
    )
}

pub fn verify_prime_field_ideal_saturation(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    factor: &Polynomial,
    certificate: &PrimeFieldIdealOperationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldIdealOperation, PolyError> {
    verify(
        Request::Saturate(inputs, factor),
        generators,
        prime,
        certificate,
        verifier,
    )
}
