//! Checked change of monomial order for finite rational quotient algebras.
//!
//! This normal-form variant of FGLM enumerates target-order monomials and
//! detects exact linear dependencies in the source quotient algebra. It does
//! not run Buchberger in the target order. Publication requires the existing
//! independent proof-kernel checker to accept both ideal inclusions and every
//! strict S-pair representation in the target order.
//!
//! Bounds: at most 64 standard monomials, eight variables, 64 basis elements,
//! and the degree/coefficient/work/allocation envelope of `division`.

use crate::division::{
    MAX_BASIS, PolynomialWork, Terms, accumulate, charge_copy, check_monomial, copy_poly,
    divide_with_work, divides, monomial_poly, refused, ring_storage, same_ring, validate_poly,
};
use crate::groebner::{
    CompleteGroebnerBasisCertificate, GroebnerBasisCertificate, GroebnerSPairCertificate,
    VerifiedGroebnerBasis, verify_complete_groebner_certificate,
};
use crate::{MultivariatePoly, PolyError, TermOrder};
use fsym_budget::BudgetMeter;
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};
use std::collections::{BTreeMap, BTreeSet};

const MAX_QUOTIENT_DIMENSION: usize = 64;
const MAX_WITNESS_POLYNOMIALS: usize = 16_384;
const MAX_MONOMIAL_STATES: usize = 1 + MAX_QUOTIENT_DIMENSION * crate::division::MAX_VARIABLES;

fn generators(source: &VerifiedGroebnerBasis) -> &[Symbol] {
    source
        .input_basis()
        .first()
        .or_else(|| source.basis().first())
        .map_or(&[], |poly| poly.generators.as_slice())
}

fn leading_powers(source: &VerifiedGroebnerBasis) -> Vec<bool> {
    let mut bounded = vec![false; generators(source).len()];
    for polynomial in source.basis() {
        if let Some((powers, _)) = polynomial.leading_term(source.order()) {
            let mut variables = powers.iter().enumerate().filter(|(_, power)| **power != 0);
            if let Some((variable, _)) = variables.next() {
                if variables.next().is_none() {
                    bounded[variable] = true;
                }
            }
        }
    }
    bounded
}

fn validate_source<M: BudgetMeter>(
    source: &VerifiedGroebnerBasis,
    work: &mut PolynomialWork<'_, M>,
) -> Result<(), PolyError> {
    if source.basis().len() > MAX_BASIS {
        return Err(refused("source basis exceeds 64 elements"));
    }
    for poly in source.input_basis().iter().chain(source.basis()) {
        validate_poly(poly, work)?;
    }
    Ok(())
}

fn standard_monomials<M: BudgetMeter>(
    source: &VerifiedGroebnerBasis,
    work: &mut PolynomialWork<'_, M>,
) -> Result<Vec<Vec<u32>>, PolyError> {
    validate_source(source, work)?;
    if source.basis().iter().any(MultivariatePoly::is_one) {
        return Ok(Vec::new());
    }
    if !source.is_zero_dimensional() {
        return Err(refused(
            "FGLM requires a finite-dimensional quotient; source ideal has positive dimension",
        ));
    }
    let width = generators(source).len();
    work.allocate(source.basis().len() * 32 + width * 4 + 256)?;
    let leading: Vec<_> = source
        .basis()
        .iter()
        .filter_map(|p| p.leading_monomial(source.order()))
        .collect();
    let mut queue = BTreeSet::from([vec![0; width]]);
    let mut visited = BTreeSet::new();
    let mut result = Vec::new();
    while let Some(monomial) = queue.pop_first() {
        work.tick(1)?;
        if queue.len() > MAX_MONOMIAL_STATES || visited.len() > MAX_MONOMIAL_STATES {
            return Err(refused(
                "standard monomial frontier exceeds its state bound",
            ));
        }
        if !visited.insert(monomial.clone()) {
            continue;
        }
        if result.len() >= MAX_QUOTIENT_DIMENSION {
            return Err(refused("quotient dimension exceeds 64"));
        }
        work.allocate(256 + width * 8)?;
        result.push(monomial.clone());
        for variable in 0..width {
            work.tick(leading.len().max(1) * width.max(1))?;
            work.allocate(256 + width * 4)?;
            let mut child = monomial.clone();
            child[variable] += 1;
            check_monomial(&child)?;
            if !visited.contains(&child) && !leading.iter().any(|head| divides(head, &child)) {
                queue.insert(child);
            }
        }
    }
    result.sort_by(|a, b| source.order().compare_monomials(a, b));
    work.tick(result.len().max(1) * width.max(1))?;
    Ok(result)
}

fn monic<M: BudgetMeter>(
    poly: &MultivariatePoly,
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    let mut out = copy_poly(poly, work)?;
    if let Some(coefficient) = poly.leading_coeff(order) {
        for value in out.terms.values_mut() {
            *value = work.divide(value, &coefficient)?;
        }
    }
    Ok(out)
}

fn s_pair<M: BudgetMeter>(
    a: &MultivariatePoly,
    b: &MultivariatePoly,
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<MultivariatePoly, PolyError> {
    let (a_head, a_value) = a
        .leading_term(order)
        .ok_or_else(|| refused("zero target basis element"))?;
    let (b_head, b_value) = b
        .leading_term(order)
        .ok_or_else(|| refused("zero target basis element"))?;
    work.allocate(a_head.len() * 4)?;
    let common: Vec<u32> = a_head.iter().zip(b_head).map(|(a, b)| *a.max(b)).collect();
    check_monomial(&common)?;
    let mut terms = Terms::new();
    for (poly, head, coefficient, negative) in
        [(a, a_head, a_value, false), (b, b_head, b_value, true)]
    {
        for (powers, value) in &poly.terms {
            work.allocate(powers.len() * 4)?;
            let shifted: Vec<u32> = powers
                .iter()
                .zip(head)
                .zip(&common)
                .map(|((p, h), c)| p + c - h)
                .collect();
            check_monomial(&shifted)?;
            let scaled = work.divide(value, coefficient)?;
            let scaled = if negative {
                work.subtract(&BigRational::zero(), &scaled)?
            } else {
                scaled
            };
            accumulate(&mut terms, shifted, scaled, work)?;
        }
    }
    work.allocate(ring_storage(&a.generators))?;
    Ok(MultivariatePoly {
        generators: a.generators.clone(),
        terms,
    })
}

fn certificate_with_work<M: BudgetMeter>(
    initial: &[MultivariatePoly],
    candidate: &[MultivariatePoly],
    order: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<CompleteGroebnerBasisCertificate, PolyError> {
    let inputs = initial.len();
    let outputs = candidate.len();
    if inputs > MAX_BASIS || outputs > MAX_BASIS {
        return Err(refused("basis size exceeds 64"));
    }
    let pairs = outputs * outputs.saturating_sub(1) / 2;
    let objects = 2 * inputs * outputs + pairs * outputs + inputs + outputs;
    if objects > MAX_WITNESS_POLYNOMIALS {
        return Err(refused(
            "certificate witness count exceeds the complete-basis envelope",
        ));
    }
    let ring: &[Symbol] = initial
        .first()
        .or_else(|| candidate.first())
        .map_or(&[], |p| p.generators.as_slice());
    for poly in initial.iter().chain(candidate) {
        validate_poly(poly, work)?;
        same_ring(ring, &poly.generators)?;
    }
    work.allocate(
        objects.max(1) * std::mem::size_of::<MultivariatePoly>()
            + pairs * std::mem::size_of::<GroebnerSPairCertificate>(),
    )?;
    let mut basis_in_input = Vec::with_capacity(outputs);
    for polynomial in candidate {
        // The input list is a source Groebner basis for order conversion or
        // scalar multiples of the supplied candidate in direct certification.
        let (coefficients, remainder) = divide_with_work(polynomial, initial, order, work)?;
        if !remainder.is_zero() {
            return Err(refused(
                "candidate containment needs a supplied source-order witness",
            ));
        }
        basis_in_input.push(coefficients);
    }
    certificate_with_inclusion(initial, candidate, order, basis_in_input, work)
}

fn certificate_with_inclusion<M: BudgetMeter>(
    initial: &[MultivariatePoly],
    candidate: &[MultivariatePoly],
    order: TermOrder,
    basis_in_input: Vec<Vec<MultivariatePoly>>,
    work: &mut PolynomialWork<'_, M>,
) -> Result<CompleteGroebnerBasisCertificate, PolyError> {
    let mut input_in_basis = Vec::with_capacity(initial.len());
    for polynomial in initial {
        let (coefficients, remainder) = divide_with_work(polynomial, candidate, order, work)?;
        if !remainder.is_zero() {
            return Err(refused("candidate does not contain every source generator"));
        }
        input_in_basis.push(coefficients);
    }
    let mut pairs = Vec::new();
    for left in 0..candidate.len() {
        for right in (left + 1)..candidate.len() {
            let polynomial = s_pair(&candidate[left], &candidate[right], order, work)?;
            let (coefficients, remainder) = divide_with_work(&polynomial, candidate, order, work)?;
            if !remainder.is_zero() {
                return Err(refused("candidate fails an S-pair reduction"));
            }
            pairs.push(GroebnerSPairCertificate {
                left,
                right,
                basis_coefficients: coefficients,
            });
        }
    }
    let basis = candidate
        .iter()
        .map(|p| copy_poly(p, work))
        .collect::<Result<_, _>>()?;
    work.tick(1)?;
    Ok(CompleteGroebnerBasisCertificate {
        candidate: GroebnerBasisCertificate {
            order,
            basis,
            input_ideal_witnesses: basis_in_input,
        },
        input_basis_witnesses: input_in_basis,
        s_pairs: pairs,
    })
}

/// Build complete-basis evidence for an existing candidate without a
/// Buchberger search. Candidate-to-input membership is reduced in `order`;
/// callers with a source basis in another order should use `fglm` instead.
pub fn certificate_for_basis(
    initial: &[MultivariatePoly],
    candidate: &[MultivariatePoly],
    order: TermOrder,
    generator_meter: &mut impl BudgetMeter,
) -> Result<CompleteGroebnerBasisCertificate, PolyError> {
    certificate_with_work(
        initial,
        candidate,
        order,
        &mut PolynomialWork::new(generator_meter),
    )
}

/// Check a supplied basis directly, allowing its nonzero members to be
/// nonmonic scalar multiples (e.g. primitive integer-domain output). This
/// does not repair a non-Groebner, redundant or nonreduced candidate by search.
pub fn certify_groebner_basis(
    basis: &[MultivariatePoly],
    order: TermOrder,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<VerifiedGroebnerBasis, PolyError> {
    let mut work = PolynomialWork::new(generator_meter);
    if basis.len() > MAX_BASIS {
        return Err(refused("source basis exceeds 64 elements"));
    }
    let candidate = basis
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| monic(p, order, &mut work))
        .collect::<Result<Vec<_>, _>>()?;
    let certificate = certificate_with_work(basis, &candidate, order, &mut work)?;
    verify_complete_groebner_certificate(basis, &certificate, verifier_meter)
}

struct LinearRow {
    pivot: usize,
    vector: Vec<BigRational>,
    // This normalized vector as a combination of independent normal forms.
    combination: Vec<BigRational>,
}

fn zero_vector<M: BudgetMeter>(
    dimension: usize,
    work: &mut PolynomialWork<'_, M>,
) -> Result<Vec<BigRational>, PolyError> {
    work.allocate(dimension.max(1) * (std::mem::size_of::<BigRational>() + 32))?;
    Ok(vec![BigRational::zero(); dimension])
}

fn convert<M: BudgetMeter>(
    source: &VerifiedGroebnerBasis,
    target: TermOrder,
    work: &mut PolynomialWork<'_, M>,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    let standard = standard_monomials(source, work)?;
    let dimension = standard.len();
    let ring = generators(source);
    let width = ring.len();
    work.allocate(dimension.max(1) * (width * 4 + 256))?;
    let coordinates: BTreeMap<_, _> = standard
        .iter()
        .enumerate()
        .map(|(i, m)| (m.clone(), i))
        .collect();
    let mut frontier = BTreeSet::from([vec![0; width]]);
    let mut visited = BTreeSet::new();
    let mut independent: Vec<Vec<u32>> = Vec::new();
    let mut rows: Vec<LinearRow> = Vec::new();
    let mut output: Vec<MultivariatePoly> = Vec::new();
    while !frontier.is_empty() {
        if frontier.len() > MAX_MONOMIAL_STATES || visited.len() > MAX_MONOMIAL_STATES {
            return Err(refused("FGLM frontier exceeds its state bound"));
        }
        work.tick(frontier.len() * width.max(1))?;
        work.allocate(width * 4 + 256)?;
        let monomial = frontier
            .iter()
            .min_by(|a, b| target.compare_monomials(a, b))
            .expect("nonempty frontier")
            .clone();
        frontier.remove(&monomial);
        if !visited.insert(monomial.clone()) {
            continue;
        }
        if output.iter().any(|g| {
            g.leading_term(target)
                .is_some_and(|(head, _)| divides(head, &monomial))
        }) {
            continue;
        }
        let polynomial = monomial_poly(ring, monomial.clone(), work)?;
        let (_, normal_form) = divide_with_work(&polynomial, source.basis(), source.order(), work)?;
        let mut vector = zero_vector(dimension, work)?;
        for (powers, coefficient) in &normal_form.terms {
            let index = coordinates
                .get(powers)
                .ok_or_else(|| refused("normal form lies outside the source quotient basis"))?;
            vector[*index] = work.clone_value(coefficient)?;
        }
        let mut combination = zero_vector(dimension, work)?;
        for row in &rows {
            let factor = work.clone_value(&vector[row.pivot])?;
            if factor.is_zero() {
                continue;
            }
            for i in 0..dimension {
                let term = work.multiply(&factor, &row.vector[i])?;
                vector[i] = work.subtract(&vector[i], &term)?;
                let term = work.multiply(&factor, &row.combination[i])?;
                combination[i] = work.add(&combination[i], &term)?;
            }
        }
        if let Some(pivot) = vector.iter().position(|value| !value.is_zero()) {
            if independent.len() >= dimension {
                return Err(refused(
                    "quotient linear-algebra rank exceeded its dimension",
                ));
            }
            let scale = work.clone_value(&vector[pivot])?;
            for i in 0..dimension {
                vector[i] = work.divide(&vector[i], &scale)?;
                let unit = if i == independent.len() {
                    BigRational::one()
                } else {
                    BigRational::zero()
                };
                let difference = work.subtract(&unit, &combination[i])?;
                combination[i] = work.divide(&difference, &scale)?;
            }
            rows.push(LinearRow {
                pivot,
                vector,
                combination,
            });
            independent.push(monomial.clone());
            for variable in 0..width {
                work.allocate(width * 4 + 256)?;
                let mut child = monomial.clone();
                child[variable] += 1;
                check_monomial(&child)?;
                if !visited.contains(&child) {
                    frontier.insert(child);
                }
            }
        } else {
            if output.len() >= MAX_BASIS {
                return Err(refused("target basis exceeds 64 elements"));
            }
            let mut relation = polynomial;
            for (powers, coefficient) in independent.iter().zip(&combination) {
                let negative = work.subtract(&BigRational::zero(), coefficient)?;
                accumulate(&mut relation.terms, powers.clone(), negative, work)?;
            }
            output.push(relation);
        }
    }
    if independent.len() != dimension {
        return Err(refused(
            "target standard monomials did not span the source quotient",
        ));
    }
    output.sort_by(|a, b| {
        target.compare_monomials(
            &b.leading_monomial(target).expect("nonzero relation"),
            &a.leading_monomial(target).expect("nonzero relation"),
        )
    });
    work.tick(1)?;
    Ok(output)
}

impl VerifiedGroebnerBasis {
    /// Pure-leading-power criterion on an already independently checked basis.
    /// For nonempty generator lists the unit ideal reports false, matching the
    /// existing SymPy-facing criterion; its zero-dimensional vector quotient
    /// is nevertheless supported by `standard_monomials` and native conversion.
    pub fn is_zero_dimensional(&self) -> bool {
        leading_powers(self).into_iter().all(|bounded| bounded)
    }

    /// A finite basis of Q[x]/I, or an explicit positive-dimension/size refusal.
    /// The unit ideal has an empty quotient basis. No numerical rank is used.
    pub fn standard_monomials(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<Vec<Vec<u32>>, PolyError> {
        standard_monomials(self, &mut PolynomialWork::new(meter))
    }

    /// Produce an untrusted order-conversion candidate bound to `self.basis()`.
    /// Complete verification is required before accepting or publishing it.
    pub fn fglm_with_certificate(
        &self,
        target: TermOrder,
        generator_meter: &mut impl BudgetMeter,
    ) -> Result<CompleteGroebnerBasisCertificate, PolyError> {
        let mut work = PolynomialWork::new(generator_meter);
        validate_source(self, &mut work)?;
        if self.basis().is_empty() && !generators(self).is_empty() {
            return Err(refused(
                "empty candidate certificate would erase the source ring",
            ));
        }
        let candidate = if target == self.order() {
            self.basis()
                .iter()
                .map(|p| copy_poly(p, &mut work))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            convert(self, target, &mut work)?
        };
        let inputs = self.basis().len();
        let outputs = candidate.len();
        let pairs = outputs * outputs.saturating_sub(1) / 2;
        let objects = 2 * inputs * outputs + pairs * outputs + inputs + outputs;
        if objects > MAX_WITNESS_POLYNOMIALS {
            return Err(refused(
                "order-conversion certificate exceeds the witness envelope",
            ));
        }
        work.allocate(objects.max(1) * std::mem::size_of::<MultivariatePoly>())?;
        let mut basis_in_input = Vec::with_capacity(outputs);
        for polynomial in &candidate {
            let (coefficients, remainder) =
                divide_with_work(polynomial, self.basis(), self.order(), &mut work)?;
            if !remainder.is_zero() {
                return Err(refused("FGLM relation is not in the source ideal"));
            }
            basis_in_input.push(coefficients);
        }
        certificate_with_inclusion(self.basis(), &candidate, target, basis_in_input, &mut work)
    }

    /// Change monomial order and independently verify the complete result.
    /// Source and target describe the same ideal; no target-order Buchberger
    /// search or sampled numerical equality is used.
    pub fn fglm(
        &self,
        target: TermOrder,
        generator_meter: &mut impl BudgetMeter,
        verifier_meter: &mut impl BudgetMeter,
    ) -> Result<VerifiedGroebnerBasis, PolyError> {
        if target == self.order() {
            let mut work = PolynomialWork::new(generator_meter);
            validate_source(self, &mut work)?;
            for polynomial in self.input_basis().iter().chain(self.basis()) {
                charge_copy(polynomial, &mut work)?;
            }
            work.tick(1)?;
            return Ok(self.clone());
        }
        let certificate = self.fglm_with_certificate(target, generator_meter)?;
        verify_complete_groebner_certificate(self.basis(), &certificate, verifier_meter)
    }
}
