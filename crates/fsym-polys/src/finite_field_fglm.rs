//! Exact finite-quotient FGLM order conversion over a retained prime field.
//!
//! A complete source staircase supplies quotient coordinates. A deterministic
//! target-order heap enumerates monomials, and incremental modular elimination
//! finds dependencies among their source normal forms. Only independent
//! monomials extend the frontier. No target-order Buchberger search, field-point
//! enumeration, numerical rank or radicalization is used.
//!
//! Every relation carries source-ideal cofactors. The complete target witness
//! also contains the reverse inclusion and every strict S-pair identity; the
//! existing independent signed-convolution verifier alone publishes a basis.
//! Its inputs are the accepted source basis, establishing ideal equality
//! transitively with the source receipt's original generators.
//!
//! The finite quotient has dimension at most 64. Existing prime-field bounds
//! (eight variables, 32 basis elements, degree 64 and 4096 terms) are retained,
//! with separate cumulative 20M-step / 128MiB allocation caps for generation
//! and verification. Same-order requests still undergo complete verification,
//! but need no finite-quotient enumeration and allow positive-dimensional ideals.

#![forbid(unsafe_code)]

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use crate::division::{PolynomialWork, refused};
use crate::finite_field::scalar_power;
use crate::finite_field_groebner::complete_candidate;
use crate::finite_field_multivariate::{
    MAX_BASIS, MAX_DEGREE, MAX_VARIABLES, accumulate, admit_ring, admit_terms, copy_terms, divide,
    divides, empty, monomial_copy, same_ring,
};
use crate::{
    PolyError, PrimeFieldGroebnerCertificate, PrimeFieldMultivariatePolynomial,
    PrimeFieldMultivariateTerms, TermOrder, VerifiedPrimeFieldGroebnerBasis,
    checked_prime_field_groebner_basis, verify_prime_field_groebner_certificate,
};
use fsym_budget::BudgetMeter;
use fsym_core::Symbol;

type Terms = PrimeFieldMultivariateTerms;
type Polynomial = PrimeFieldMultivariatePolynomial;
const MAX_QUOTIENT_DIMENSION: usize = 64;
const MAX_MONOMIAL_STATES: usize = 1 + MAX_QUOTIENT_DIMENSION * MAX_VARIABLES;

fn reserve<T>(
    count: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<T>, PolyError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| refused("prime-field FGLM allocation size overflow"))?;
    work.allocate(bytes)?;
    let mut result = Vec::new();
    result.try_reserve_exact(count).map_err(refused)?;
    Ok(result)
}

fn validate_source(
    source: &VerifiedPrimeFieldGroebnerBasis,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    admit_ring(source.generators(), source.prime(), work)?;
    if source.basis().len() > MAX_BASIS || source.inputs().len() > MAX_BASIS {
        return Err(refused("prime-field FGLM source exceeds 32 polynomials"));
    }
    for polynomial in source.inputs().iter().chain(source.basis()) {
        same_ring(polynomial, source.generators(), source.prime(), work)?;
    }
    work.tick(1)
}

#[derive(PartialEq, Eq)]
struct QueuedMonomial {
    powers: Vec<u32>,
    order: TermOrder,
}

fn order_rank(order: TermOrder) -> u8 {
    match order {
        TermOrder::Lex => 0,
        TermOrder::DegLex => 1,
        TermOrder::DegRevLex => 2,
    }
}

impl Ord for QueuedMonomial {
    fn cmp(&self, other: &Self) -> Ordering {
        // Every heap has one fixed order. Including its rank also makes the
        // private key's total-order contract valid across distinct heaps.
        order_rank(self.order)
            .cmp(&order_rank(other.order))
            .then_with(|| self.order.compare_monomials(&self.powers, &other.powers))
    }
}

impl PartialOrd for QueuedMonomial {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn comparison_charge(count: usize, width: usize) -> usize {
    // More than the worst-case comparisons of a bounded B-tree lookup or heap
    // push/pop, including both degree sums and the ordered exponent scan.
    let levels = (count + 1).next_power_of_two().trailing_zeros() as usize + 1;
    16 * (width + 1) * levels
}

fn enqueue(
    powers: Vec<u32>,
    order: TermOrder,
    frontier: &mut BinaryHeap<Reverse<QueuedMonomial>>,
    seen: &mut BTreeSet<Vec<u32>>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(powers.len() + 1)?;
    if powers.iter().map(|&power| u64::from(power)).sum::<u64>() > MAX_DEGREE {
        return Err(refused("prime-field FGLM monomial exceeds degree 64"));
    }
    work.tick(comparison_charge(seen.len(), powers.len()))?;
    if seen.contains(&powers) {
        return Ok(());
    }
    if seen.len() == MAX_MONOMIAL_STATES {
        return Err(refused("prime-field FGLM frontier exceeds its state bound"));
    }
    work.allocate(128)?;
    let retained = monomial_copy(&powers, work)?;
    work.tick(comparison_charge(seen.len(), powers.len()))?;
    seen.insert(retained);
    work.tick(comparison_charge(frontier.len(), powers.len()))?;
    frontier.push(Reverse(QueuedMonomial { powers, order }));
    Ok(())
}

struct LinearRow {
    pivot: usize,
    vector: Vec<u64>,
    // This normalized vector as a combination of independent normal forms.
    combination: Vec<u64>,
}

fn zero_vector(
    dimension: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<u64>, PolyError> {
    let mut result = reserve(dimension, work)?;
    work.tick(dimension)?;
    result.resize(dimension, 0);
    Ok(result)
}

fn convert(
    source: &VerifiedPrimeFieldGroebnerBasis,
    target: TermOrder,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Terms>, PolyError> {
    let standard = source.standard_monomials(work)?;
    let dimension = standard.len();
    if dimension > MAX_QUOTIENT_DIMENSION {
        return Err(refused("prime-field FGLM quotient dimension exceeds 64"));
    }
    let width = source.generators().len();
    let prime = source.prime();
    let mut coordinates = BTreeMap::new();
    for (index, monomial) in standard.iter().enumerate() {
        work.allocate(128)?;
        let powers = monomial_copy(monomial, work)?;
        work.tick(comparison_charge(coordinates.len(), width))?;
        coordinates.insert(powers, index);
    }
    let mut views = reserve(source.basis().len(), work)?;
    for polynomial in source.basis() {
        work.tick(1)?;
        views.push(polynomial.terms());
    }
    work.allocate(MAX_MONOMIAL_STATES * std::mem::size_of::<Reverse<QueuedMonomial>>())?;
    let mut frontier = BinaryHeap::new();
    frontier
        .try_reserve_exact(MAX_MONOMIAL_STATES)
        .map_err(refused)?;
    let mut seen = BTreeSet::new();
    let mut one = reserve(width, work)?;
    one.resize(width, 0);
    enqueue(one, target, &mut frontier, &mut seen, work)?;
    let mut independent: Vec<Vec<u32>> = reserve(dimension, work)?;
    let mut rows: Vec<LinearRow> = reserve(dimension, work)?;
    let mut output: Vec<(Vec<u32>, Terms)> = reserve(MAX_BASIS, work)?;

    while !frontier.is_empty() {
        work.tick(comparison_charge(frontier.len(), width))?;
        let Reverse(queued) = frontier
            .pop()
            .ok_or_else(|| refused("prime-field FGLM lost a queued monomial"))?;
        let monomial = queued.powers;
        work.tick(output.len().max(1) * (width + 1))?;
        if output.iter().any(|(head, _)| divides(head, &monomial)) {
            continue;
        }
        let mut polynomial = empty(work)?;
        let powers = monomial_copy(&monomial, work)?;
        accumulate(&mut polynomial, powers, 1, prime, work)?;
        let normal_form = divide(&polynomial, &views, source.order(), prime, work)?.remainder;
        let mut vector = zero_vector(dimension, work)?;
        for (powers, coefficient) in &normal_form {
            work.tick(comparison_charge(coordinates.len(), width))?;
            let index = coordinates.get(powers).ok_or_else(|| {
                refused("prime-field normal form lies outside the complete source staircase")
            })?;
            vector[*index] = *coefficient;
        }
        let mut combination = zero_vector(dimension, work)?;
        for row in &rows {
            work.tick(1)?;
            let factor = vector[row.pivot];
            if factor == 0 {
                continue;
            }
            work.tick(8 * dimension)?;
            for index in 0..dimension {
                vector[index] =
                    (vector[index] + prime - factor * row.vector[index] % prime) % prime;
                combination[index] = (combination[index] + factor * row.combination[index]) % prime;
            }
        }
        work.tick(dimension)?;
        if let Some(pivot) = vector.iter().position(|&value| value != 0) {
            if independent.len() == dimension {
                return Err(refused("prime-field FGLM rank exceeded quotient dimension"));
            }
            let inverse = scalar_power(vector[pivot], prime - 2, prime, work)?;
            work.tick(7 * dimension)?;
            for index in 0..dimension {
                vector[index] = vector[index] * inverse % prime;
                let unit = u64::from(index == independent.len());
                combination[index] = (unit + prime - combination[index]) * inverse % prime;
            }
            rows.push(LinearRow {
                pivot,
                vector,
                combination,
            });
            for variable in 0..width {
                let mut child = monomial_copy(&monomial, work)?;
                child[variable] = child[variable]
                    .checked_add(1)
                    .ok_or_else(|| refused("prime-field FGLM monomial exponent overflow"))?;
                enqueue(child, target, &mut frontier, &mut seen, work)?;
            }
            independent.push(monomial);
        } else {
            if output.len() == MAX_BASIS {
                return Err(refused("prime-field FGLM target exceeds 32 basis elements"));
            }
            for (powers, &coefficient) in independent.iter().zip(&combination) {
                work.tick(1)?;
                if coefficient != 0 {
                    let powers = monomial_copy(powers, work)?;
                    accumulate(&mut polynomial, powers, prime - coefficient, prime, work)?;
                }
            }
            admit_terms(&polynomial, width, prime, MAX_DEGREE, false, work)?;
            output.push((monomial, polynomial));
        }
    }
    if independent.len() != dimension {
        return Err(refused(
            "prime-field FGLM monomials did not span the source quotient",
        ));
    }
    // Precharge a quadratic upper bound before the bounded in-place sort.
    work.tick(output.len().max(1).pow(2) * 3 * (width + 1))?;
    output.sort_unstable_by(|(left, _), (right, _)| target.compare_monomials(right, left));
    let mut result = reserve(output.len(), work)?;
    for (_, polynomial) in output {
        work.tick(1)?;
        result.push(polynomial);
    }
    work.tick(1)?;
    Ok(result)
}

impl VerifiedPrimeFieldGroebnerBasis {
    /// Generate an untrusted complete target-order witness bound to this
    /// accepted source's `basis()`. The ring and prime remain explicit even
    /// when the zero ideal has no generators. Acceptance requires the separate
    /// complete prime-field Groebner verifier.
    pub fn fglm_with_certificate(
        &self,
        target: TermOrder,
        generator: &mut impl BudgetMeter,
    ) -> Result<PrimeFieldGroebnerCertificate, PolyError> {
        let mut work = PolynomialWork::new(generator);
        validate_source(self, &mut work)?;
        let candidate = if target == self.order() {
            let mut copied = reserve(self.basis().len(), &mut work)?;
            for polynomial in self.basis() {
                copied.push(copy_terms(polynomial.terms(), &mut work)?);
            }
            copied
        } else {
            convert(self, target, &mut work)?
        };
        let mut views = reserve(self.basis().len(), &mut work)?;
        for polynomial in self.basis() {
            work.tick(1)?;
            views.push(polynomial.terms());
        }
        let mut inclusion = reserve(candidate.len(), &mut work)?;
        for polynomial in &candidate {
            let division = divide(polynomial, &views, self.order(), self.prime(), &mut work)?;
            if !division.remainder.is_empty() {
                return Err(refused(
                    "prime-field FGLM relation is not in the source ideal",
                ));
            }
            inclusion.push(division.quotients);
        }
        let certificate = complete_candidate(
            self.basis(),
            candidate,
            inclusion,
            target,
            self.prime(),
            self.generators().len(),
            &mut work,
        )?;
        work.tick(1)?;
        Ok(certificate)
    }

    /// Convert to a complete reduced basis of the same ideal in `target` order.
    /// All arithmetic is in the retained F_p, including characteristic-dependent
    /// rank. Same-order calls also verify a complete witness before publication.
    pub fn fglm(
        &self,
        target: TermOrder,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedPrimeFieldGroebnerBasis, PolyError> {
        let mut generation = PolynomialWork::new(generator);
        let mut verification = PolynomialWork::new(verifier);
        let certificate = self.fglm_with_certificate(target, &mut generation)?;
        let accepted = verify_prime_field_groebner_certificate(
            self.generators(),
            self.prime(),
            self.basis(),
            target,
            &certificate,
            &mut verification,
        )?;
        generation.tick(1)?;
        verification.tick(1)?;
        Ok(accepted)
    }
}

/// Check the source ideal, convert through its finite quotient, and independently
/// verify the complete target basis. Both phase meters remain cumulative across
/// source construction, conversion and target verification. The returned inputs
/// are the independently accepted intermediate source basis.
pub fn checked_prime_field_fglm(
    generators: &[Symbol],
    prime: u64,
    inputs: &[Polynomial],
    source_order: TermOrder,
    target_order: TermOrder,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldGroebnerBasis, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let source = checked_prime_field_groebner_basis(
        generators,
        prime,
        inputs,
        source_order,
        &mut generation,
        &mut verification,
    )?;
    let accepted = source.fglm(target_order, &mut generation, &mut verification)?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
