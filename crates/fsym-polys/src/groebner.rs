//! Groebner bases, Buchberger algorithm, ideal membership, and elimination (WS17).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::multivariate::{MultivariatePoly, TermOrder};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{BudgetMeter, Dimension};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_proof_kernel::{
    Claim, DerivationTree, ProofKernel, derivation_verification_units,
    verify_derivation_independent,
};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const MAX_GROEBNER_BASIS_POLYNOMIALS: usize = 1_024;
const MAX_GROEBNER_PENDING_PAIRS: usize = 1_000_000;
const MAX_ELIMINATION_WITNESS_EXPR_NODES: usize = 16_384;
const MAX_ELIMINATION_WITNESS_COEFFICIENT_LIMBS: u64 = 4_096;

impl TermOrder {
    /// Compares two exponent vectors according to the term ordering.
    pub fn compare_monomials(&self, a: &[u32], b: &[u32]) -> Ordering {
        match self {
            TermOrder::Lex => a.cmp(b),
            TermOrder::DegLex => {
                let sum_a: u128 = a.iter().map(|&value| u128::from(value)).sum();
                let sum_b: u128 = b.iter().map(|&value| u128::from(value)).sum();
                match sum_a.cmp(&sum_b) {
                    Ordering::Equal => TermOrder::Lex.compare_monomials(a, b),
                    other => other,
                }
            }
            TermOrder::DegRevLex => {
                let sum_a: u128 = a.iter().map(|&value| u128::from(value)).sum();
                let sum_b: u128 = b.iter().map(|&value| u128::from(value)).sum();
                match sum_a.cmp(&sum_b) {
                    Ordering::Equal => {
                        for (x, y) in a.iter().rev().zip(b.iter().rev()) {
                            match y.cmp(x) {
                                Ordering::Equal => continue,
                                other => return other,
                            }
                        }
                        a.len().cmp(&b.len())
                    }
                    other => other,
                }
            }
        }
    }
}

impl MultivariatePoly {
    /// Returns the leading term `(exponent_vector, coefficient)` under the specified term ordering.
    pub fn leading_term(&self, order: TermOrder) -> Option<(&Vec<u32>, &BigRational)> {
        self.terms
            .iter()
            .max_by(|(exp_a, _), (exp_b, _)| order.compare_monomials(exp_a, exp_b))
    }

    /// Returns the leading monomial exponent vector.
    pub fn leading_monomial(&self, order: TermOrder) -> Option<Vec<u32>> {
        self.leading_term(order).map(|(exp, _)| exp.clone())
    }

    /// Returns the leading coefficient.
    pub fn leading_coeff(&self, order: TermOrder) -> Option<BigRational> {
        self.leading_term(order).map(|(_, c)| c.clone())
    }

    /// Exact multivariate polynomial division with remainder: $f = \sum q_i g_i + r$.
    pub fn div_rem(
        &self,
        divisors: &[MultivariatePoly],
        order: TermOrder,
    ) -> Result<(Vec<MultivariatePoly>, MultivariatePoly), PolyError> {
        self.validate_shape()?;
        for divisor in divisors {
            divisor.validate_shape()?;
            if divisor.generators != self.generators {
                return Err(incompatible_rings(&self.generators, &divisor.generators));
            }
        }

        let n_divs = divisors.len();
        let mut quotients = vec![MultivariatePoly::zero(self.generators.clone()); n_divs];
        let mut remainder_terms = BTreeMap::new();
        let mut p = self.clone();

        while !p.is_zero() {
            let (lt_exp, lt_coeff) = match p.leading_term(order) {
                Some((e, c)) => (e.clone(), c.clone()),
                None => break,
            };

            let mut divided = false;
            for (i, g) in divisors.iter().enumerate() {
                if g.is_zero() {
                    continue;
                }
                let Some((g_lt_exp, g_lt_coeff)) = g.leading_term(order) else {
                    continue;
                };

                // Check if g's leading monomial divides p's leading monomial: g_lt_exp <= lt_exp component-wise
                if divides(g_lt_exp, &lt_exp) {
                    let diff_exp = monomial_sub(&lt_exp, g_lt_exp);
                    let q_coeff = &lt_coeff / g_lt_coeff;

                    // Add term to quotient q_i
                    let mut term_map = BTreeMap::new();
                    term_map.insert(diff_exp.clone(), q_coeff.clone());
                    let term_poly = MultivariatePoly::new(self.generators.clone(), term_map)?;

                    quotients[i] = quotients[i].add(&term_poly)?;

                    // p = p - term_poly * g
                    let prod = term_poly.mul(g)?;
                    p = p.sub(&prod)?;

                    divided = true;
                    break;
                }
            }

            if !divided {
                // Move leading term of p to remainder
                let entry = remainder_terms
                    .entry(lt_exp.clone())
                    .or_insert_with(BigRational::zero);
                *entry += &lt_coeff;
                p.terms.remove(&lt_exp);
            }
        }

        Ok((
            quotients,
            MultivariatePoly::new(self.generators.clone(), remainder_terms)?,
        ))
    }

    /// Makes the polynomial monic (leading coefficient = 1).
    pub fn to_monic(&self, order: TermOrder) -> Result<Self, PolyError> {
        self.validate_shape()?;
        if self.is_zero() {
            return Ok(self.clone());
        }
        let lc = self.leading_coeff(order).ok_or_else(|| {
            PolyError::General("non-zero polynomial has no leading coefficient".to_string())
        })?;
        let mut monic_terms = BTreeMap::new();
        for (exp, coeff) in &self.terms {
            monic_terms.insert(exp.clone(), coeff / &lc);
        }
        Self::new(self.generators.clone(), monic_terms)
    }
}

fn incompatible_rings(expected: &[Symbol], actual: &[Symbol]) -> PolyError {
    PolyError::IncompatibleGenerators(format!("{expected:?}"), format!("{actual:?}"))
}

fn validate_common_ring(polys: &[MultivariatePoly]) -> Result<(), PolyError> {
    let Some(first) = polys.first() else {
        return Ok(());
    };
    first.validate_shape()?;
    for poly in &polys[1..] {
        poly.validate_shape()?;
        if poly.generators != first.generators {
            return Err(incompatible_rings(&first.generators, &poly.generators));
        }
    }
    Ok(())
}

/// Helper checking if monomial `a` divides monomial `b`.
fn divides(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x <= y)
}

/// Subtracts monomial `b` from `a` assuming `b <= a`.
fn monomial_sub(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Component-wise LCM of two monomials: $\max(a_i, b_i)$.
fn monomial_lcm(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter().zip(b.iter()).map(|(x, y)| (*x).max(*y)).collect()
}

/// S-polynomial: $S(f, g) = \frac{\text{lcm}(\text{LM}(f), \text{LM}(g))}{\text{LT}(f)} f - \frac{\text{lcm}(\text{LM}(f), \text{LM}(g))}{\text{LT}(g)} g$.
pub fn s_polynomial(
    f: &MultivariatePoly,
    g: &MultivariatePoly,
    order: TermOrder,
) -> Result<MultivariatePoly, PolyError> {
    f.validate_shape()?;
    g.validate_shape()?;
    if f.generators != g.generators {
        return Err(incompatible_rings(&f.generators, &g.generators));
    }
    if f.is_zero() || g.is_zero() {
        return Ok(MultivariatePoly::zero(f.generators.clone()));
    }
    let (f_exp, f_c) = f.leading_term(order).unwrap();
    let (g_exp, g_c) = g.leading_term(order).unwrap();

    let lcm_exp = monomial_lcm(f_exp, g_exp);
    let f_shift = monomial_sub(&lcm_exp, f_exp);
    let g_shift = monomial_sub(&lcm_exp, g_exp);

    let mut term_f = BTreeMap::new();
    term_f.insert(f_shift, BigRational::one() / f_c);
    let poly_f = MultivariatePoly::new(f.generators.clone(), term_f)?;

    let mut term_g = BTreeMap::new();
    term_g.insert(g_shift, BigRational::one() / g_c);
    let poly_g = MultivariatePoly::new(g.generators.clone(), term_g)?;

    let t1 = poly_f.mul(f)?;
    let t2 = poly_g.mul(g)?;
    t1.sub(&t2)
}

/// Computes the minimal reduced Groebner basis using the Buchberger algorithm.
pub fn groebner_basis(
    initial_basis: &[MultivariatePoly],
    order: TermOrder,
) -> Result<Vec<MultivariatePoly>, PolyError> {
    validate_common_ring(initial_basis)?;
    if initial_basis.is_empty() {
        return Ok(Vec::new());
    }
    if initial_basis.len() > MAX_GROEBNER_BASIS_POLYNOMIALS {
        return Err(PolyError::General(format!(
            "Groebner basis input exceeds the polynomial limit of {MAX_GROEBNER_BASIS_POLYNOMIALS}"
        )));
    }
    let mut g: Vec<MultivariatePoly> = initial_basis
        .iter()
        .filter(|poly| !poly.is_zero())
        .cloned()
        .collect();

    let mut pairs = Vec::new();
    for i in 0..g.len() {
        for j in (i + 1)..g.len() {
            if pairs.len() == MAX_GROEBNER_PENDING_PAIRS {
                return Err(PolyError::General(format!(
                    "Groebner computation exceeds the pending-pair limit of {MAX_GROEBNER_PENDING_PAIRS}"
                )));
            }
            pairs.push((i, j));
        }
    }

    while let Some((i, j)) = pairs.pop() {
        let s = s_polynomial(&g[i], &g[j], order)?;
        let (_, remainder) = s.div_rem(&g, order)?;
        if !remainder.is_zero() {
            if g.len() == MAX_GROEBNER_BASIS_POLYNOMIALS {
                return Err(PolyError::General(format!(
                    "Groebner computation exceeds the basis limit of {MAX_GROEBNER_BASIS_POLYNOMIALS}"
                )));
            }
            let new_idx = g.len();
            for k in 0..g.len() {
                if pairs.len() == MAX_GROEBNER_PENDING_PAIRS {
                    return Err(PolyError::General(format!(
                        "Groebner computation exceeds the pending-pair limit of {MAX_GROEBNER_PENDING_PAIRS}"
                    )));
                }
                pairs.push((k, new_idx));
            }
            g.push(remainder);
        }
    }

    let reduced: Vec<MultivariatePoly> = g
        .into_iter()
        .map(|poly| poly.to_monic(order))
        .collect::<Result<_, _>>()?;

    let mut minimal = Vec::new();
    for (i, poly) in reduced.iter().enumerate() {
        if poly.is_zero() {
            continue;
        }
        let leading_monomial = poly.leading_monomial(order).unwrap();
        let is_redundant = reduced.iter().enumerate().any(|(j, other)| {
            if i == j || other.is_zero() {
                return false;
            }
            let other_leading_monomial = other.leading_monomial(order).unwrap();
            if other_leading_monomial == leading_monomial {
                j < i
            } else {
                divides(&other_leading_monomial, &leading_monomial)
            }
        });
        if !is_redundant {
            minimal.push(poly.clone());
        }
    }

    let mut fully_reduced = Vec::new();
    for i in 0..minimal.len() {
        let poly = minimal[i].clone();
        let other_divisors: Vec<MultivariatePoly> = minimal
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, other)| other.clone())
            .collect();
        let (_, remainder) = poly.div_rem(&other_divisors, order)?;
        if !remainder.is_zero() {
            fully_reduced.push(remainder.to_monic(order)?);
        }
    }

    Ok(fully_reduced)
}

#[derive(Clone)]
struct TrackedPolynomial {
    poly: MultivariatePoly,
    /// Coefficients expressing `poly` as a linear combination of the input basis.
    input_coefficients: Vec<MultivariatePoly>,
}

fn scale_poly(
    poly: &MultivariatePoly,
    scalar: &BigRational,
) -> Result<MultivariatePoly, PolyError> {
    let terms = poly
        .terms
        .iter()
        .map(|(exponents, coefficient)| (exponents.clone(), coefficient * scalar))
        .collect();
    MultivariatePoly::new(poly.generators.clone(), terms)
}

fn make_tracked_monic(
    tracked: TrackedPolynomial,
    order: TermOrder,
) -> Result<TrackedPolynomial, PolyError> {
    let leading_coefficient = tracked.poly.leading_coeff(order).ok_or_else(|| {
        PolyError::General("non-zero tracked polynomial has no leading coefficient".to_string())
    })?;
    let scale = BigRational::one() / leading_coefficient;
    Ok(TrackedPolynomial {
        poly: tracked.poly.to_monic(order)?,
        input_coefficients: tracked
            .input_coefficients
            .iter()
            .map(|coefficient| scale_poly(coefficient, &scale))
            .collect::<Result<_, _>>()?,
    })
}

fn reduce_tracked(
    dividend: TrackedPolynomial,
    divisors: &[TrackedPolynomial],
    order: TermOrder,
) -> Result<TrackedPolynomial, PolyError> {
    let divisor_polys: Vec<_> = divisors
        .iter()
        .map(|divisor| divisor.poly.clone())
        .collect();
    let (quotients, remainder) = dividend.poly.div_rem(&divisor_polys, order)?;
    let mut input_coefficients = dividend.input_coefficients;

    for (quotient, divisor) in quotients.iter().zip(divisors) {
        for (coefficient, divisor_coefficient) in input_coefficients
            .iter_mut()
            .zip(&divisor.input_coefficients)
        {
            *coefficient = coefficient.sub(&quotient.mul(divisor_coefficient)?)?;
        }
    }

    Ok(TrackedPolynomial {
        poly: remainder,
        input_coefficients,
    })
}

fn tracked_s_polynomial(
    f: &TrackedPolynomial,
    g: &TrackedPolynomial,
    order: TermOrder,
) -> Result<TrackedPolynomial, PolyError> {
    let (f_exp, f_coefficient) = f.poly.leading_term(order).ok_or_else(|| {
        PolyError::General("zero polynomial reached a tracked S-pair".to_string())
    })?;
    let (g_exp, g_coefficient) = g.poly.leading_term(order).ok_or_else(|| {
        PolyError::General("zero polynomial reached a tracked S-pair".to_string())
    })?;
    let lcm_exp = monomial_lcm(f_exp, g_exp);

    let mut f_factor_terms = BTreeMap::new();
    f_factor_terms.insert(
        monomial_sub(&lcm_exp, f_exp),
        BigRational::one() / f_coefficient,
    );
    let f_factor = MultivariatePoly::new(f.poly.generators.clone(), f_factor_terms)?;

    let mut g_factor_terms = BTreeMap::new();
    g_factor_terms.insert(
        monomial_sub(&lcm_exp, g_exp),
        BigRational::one() / g_coefficient,
    );
    let g_factor = MultivariatePoly::new(g.poly.generators.clone(), g_factor_terms)?;

    let poly = f_factor.mul(&f.poly)?.sub(&g_factor.mul(&g.poly)?)?;
    let mut input_coefficients = Vec::with_capacity(f.input_coefficients.len());
    for (f_input, g_input) in f.input_coefficients.iter().zip(&g.input_coefficients) {
        input_coefficients.push(f_factor.mul(f_input)?.sub(&g_factor.mul(g_input)?)?);
    }

    Ok(TrackedPolynomial {
        poly,
        input_coefficients,
    })
}

fn tracked_groebner_basis(
    initial_basis: &[MultivariatePoly],
    order: TermOrder,
) -> Result<Vec<TrackedPolynomial>, PolyError> {
    validate_common_ring(initial_basis)?;
    if initial_basis.is_empty() {
        return Ok(Vec::new());
    }
    if initial_basis.len() > MAX_GROEBNER_BASIS_POLYNOMIALS {
        return Err(PolyError::General(format!(
            "Groebner basis input exceeds the polynomial limit of {MAX_GROEBNER_BASIS_POLYNOMIALS}"
        )));
    }
    let generators = initial_basis[0].generators.clone();
    let mut g: Vec<TrackedPolynomial> = initial_basis
        .iter()
        .enumerate()
        .filter(|(_, poly)| !poly.is_zero())
        .map(|(index, poly)| {
            let mut input_coefficients =
                vec![MultivariatePoly::zero(generators.clone()); initial_basis.len()];
            input_coefficients[index] = MultivariatePoly::one(generators.clone());
            TrackedPolynomial {
                poly: poly.clone(),
                input_coefficients,
            }
        })
        .collect();

    let mut pairs = Vec::new();
    for i in 0..g.len() {
        for j in (i + 1)..g.len() {
            if pairs.len() == MAX_GROEBNER_PENDING_PAIRS {
                return Err(PolyError::General(format!(
                    "Groebner computation exceeds the pending-pair limit of {MAX_GROEBNER_PENDING_PAIRS}"
                )));
            }
            pairs.push((i, j));
        }
    }

    while let Some((i, j)) = pairs.pop() {
        let s = tracked_s_polynomial(&g[i], &g[j], order)?;
        let remainder = reduce_tracked(s, &g, order)?;
        if !remainder.poly.is_zero() {
            if g.len() == MAX_GROEBNER_BASIS_POLYNOMIALS {
                return Err(PolyError::General(format!(
                    "Groebner computation exceeds the basis limit of {MAX_GROEBNER_BASIS_POLYNOMIALS}"
                )));
            }
            let new_idx = g.len();
            for k in 0..g.len() {
                if pairs.len() == MAX_GROEBNER_PENDING_PAIRS {
                    return Err(PolyError::General(format!(
                        "Groebner computation exceeds the pending-pair limit of {MAX_GROEBNER_PENDING_PAIRS}"
                    )));
                }
                pairs.push((k, new_idx));
            }
            g.push(remainder);
        }
    }

    // Auto-reduction: make monic and reduce each element against all others
    let reduced: Vec<TrackedPolynomial> = g
        .into_iter()
        .map(|tracked| make_tracked_monic(tracked, order))
        .collect::<Result<_, _>>()?;

    // Remove redundant elements whose leading monomial is divisible by another
    let mut minimal = Vec::new();
    for (i, p) in reduced.iter().enumerate() {
        if p.poly.is_zero() {
            continue;
        }
        let lm_p = p.poly.leading_monomial(order).unwrap();
        let is_redundant = reduced.iter().enumerate().any(|(j, other)| {
            if i == j || other.poly.is_zero() {
                return false;
            }
            let lm_other = other.poly.leading_monomial(order).unwrap();
            if lm_other == lm_p {
                j < i // keep the first one with this leading monomial
            } else {
                divides(&lm_other, &lm_p)
            }
        });
        if !is_redundant {
            minimal.push(p.clone());
        }
    }

    // Fully reduce remaining elements
    let mut fully_reduced = Vec::new();
    for i in 0..minimal.len() {
        let p = minimal[i].clone();
        let other_divisors: Vec<TrackedPolynomial> = minimal
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, tracked)| tracked.clone())
            .collect();
        let remainder = reduce_tracked(p, &other_divisors, order)?;
        if !remainder.poly.is_zero() {
            fully_reduced.push(make_tracked_monic(remainder, order)?);
        }
    }

    Ok(fully_reduced)
}

/// Checks if polynomial `f` belongs to the ideal $\langle G \rangle$: $f \in I(G) \iff f \xrightarrow{G} 0$.
pub fn ideal_membership(
    poly: &MultivariatePoly,
    groebner_basis: &[MultivariatePoly],
    order: TermOrder,
) -> Result<bool, PolyError> {
    let (_, rem) = poly.div_rem(groebner_basis, order)?;
    Ok(rem.is_zero())
}

/// Variable elimination via a lexicographic Groebner basis.
///
/// The requested variables are placed before all retained variables for the computation,
/// preserving their relative order from the input ring. The output is mapped back to the
/// original generator order. This is essential when eliminating variables that are not an
/// initial block of the input ring: filtering a basis in the original order is not elimination.
///
/// An empty elimination list returns the lexicographic basis. Eliminating every variable
/// returns the constant part (empty for a proper ideal, `[1]` for the unit ideal). An empty
/// input list denotes the zero ideal and returns no polynomials; it supplies no ambient ring
/// against which to validate variable names.
pub fn eliminate(
    initial_basis: &[MultivariatePoly],
    eliminated_vars: &[Symbol],
) -> Result<Vec<MultivariatePoly>, PolyError> {
    if initial_basis.is_empty() {
        return Ok(Vec::new());
    }
    let (reordered, inverse) = prepare_elimination(initial_basis, eliminated_vars)?;
    let gb = groebner_basis(&reordered, TermOrder::Lex)?;
    gb.into_iter()
        .filter(|poly| (0..eliminated_vars.len()).all(|index| poly.degree_in(index) == 0))
        .map(|poly| permute_generators(&poly, &inverse))
        .collect()
}

fn elimination_permutation(
    generators: &[Symbol],
    eliminated_vars: &[Symbol],
) -> Result<Vec<usize>, PolyError> {
    if eliminated_vars.len() > generators.len() {
        return Err(PolyError::General(
            "elimination variable count exceeds the ring generator count".to_string(),
        ));
    }
    let mut eliminated = vec![false; generators.len()];
    for symbol in eliminated_vars {
        let index = generators
            .iter()
            .position(|generator| generator == symbol)
            .ok_or_else(|| {
                PolyError::IncompatibleGenerators(
                    format!("elimination variable {}", symbol.name),
                    "missing from polynomial generators".to_string(),
                )
            })?;
        if eliminated[index] {
            return Err(PolyError::General(
                "elimination variable list contains a duplicate".to_string(),
            ));
        }
        eliminated[index] = true;
    }
    Ok((0..generators.len())
        .filter(|&index| eliminated[index])
        .chain((0..generators.len()).filter(|&index| !eliminated[index]))
        .collect())
}

/// Reindexes a polynomial using an internally constructed permutation, without routing through
/// expression parsing or changing any symbol identity/coefficient.
fn permute_generators(
    poly: &MultivariatePoly,
    permutation: &[usize],
) -> Result<MultivariatePoly, PolyError> {
    let generators = permutation
        .iter()
        .map(|&index| poly.generators[index].clone())
        .collect();
    let terms = poly
        .terms
        .iter()
        .map(|(exponents, coefficient)| {
            (
                permutation.iter().map(|&index| exponents[index]).collect(),
                coefficient.clone(),
            )
        })
        .collect();
    MultivariatePoly::new(generators, terms)
}

fn prepare_elimination(
    initial_basis: &[MultivariatePoly],
    eliminated_vars: &[Symbol],
) -> Result<(Vec<MultivariatePoly>, Vec<usize>), PolyError> {
    if initial_basis.len() > MAX_GROEBNER_BASIS_POLYNOMIALS {
        return Err(PolyError::General(format!(
            "elimination input exceeds the polynomial limit of {MAX_GROEBNER_BASIS_POLYNOMIALS}"
        )));
    }
    validate_common_ring(initial_basis)?;
    let permutation = elimination_permutation(&initial_basis[0].generators, eliminated_vars)?;
    let mut inverse = vec![0; permutation.len()];
    for (new_index, &old_index) in permutation.iter().enumerate() {
        inverse[old_index] = new_index;
    }
    let reordered = initial_basis
        .iter()
        .map(|poly| permute_generators(poly, &permutation))
        .collect::<Result<_, _>>()?;
    Ok((reordered, inverse))
}

/// An elimination consequence with explicit coefficients against the original input generators:
/// `polynomial = sum(input_coefficients[i] * initial_basis[i])`.
///
/// The witness establishes membership and independence of the eliminated variables. It does
/// **not** certify that a list of consequences is a complete or reduced elimination basis.
/// Its public fields are untrusted until [`verify_elimination_witness`] accepts them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EliminationWitness {
    pub polynomial: MultivariatePoly,
    pub input_coefficients: Vec<MultivariatePoly>,
}

/// Computes elimination consequences while retaining input-ideal witnesses through every
/// S-pair, reduction, normalization, and generator permutation.
///
/// Like [`groebner_basis_with_certificate`], this synchronous generator uses the bounded
/// polynomial/basis/pair limits and returns an error if the search cannot finish within them.
/// It never returns a partial basis on refusal. Verification is a separate, budgeted operation.
pub fn eliminate_with_witnesses(
    initial_basis: &[MultivariatePoly],
    eliminated_vars: &[Symbol],
) -> Result<Vec<EliminationWitness>, PolyError> {
    if initial_basis.is_empty() {
        return Ok(Vec::new());
    }
    let (reordered, inverse) = prepare_elimination(initial_basis, eliminated_vars)?;
    tracked_groebner_basis(&reordered, TermOrder::Lex)?
        .into_iter()
        .filter(|tracked| {
            (0..eliminated_vars.len()).all(|index| tracked.poly.degree_in(index) == 0)
        })
        .map(|tracked| {
            Ok(EliminationWitness {
                polynomial: permute_generators(&tracked.poly, &inverse)?,
                input_coefficients: tracked
                    .input_coefficients
                    .iter()
                    .map(|coefficient| permute_generators(coefficient, &inverse))
                    .collect::<Result<_, _>>()?,
            })
        })
        .collect()
}

fn witness_error(error: impl std::fmt::Display) -> PolyError {
    PolyError::General(format!("elimination witness refused: {error}"))
}

/// Bound the expression representation before `to_expr` copies coefficients and symbol names.
/// The estimates include expression/vector storage and coefficient/name payloads; the separate
/// proof-kernel preflight and normalization bounds still apply to arithmetic scratch work.
fn preflight_witness_polynomial(
    poly: &MultivariatePoly,
    expression_nodes: &mut usize,
    meter: &mut impl BudgetMeter,
) -> Result<(), PolyError> {
    meter.checkpoint().map_err(witness_error)?;
    poly.validate_shape()?;
    let mut nodes = 1usize; // outer sum, or canonical zero
    let mut payload_bytes = 0u64;
    for (exponents, coefficient) in &poly.terms {
        meter.checkpoint().map_err(witness_error)?;
        meter
            .charge(Dimension::ComputeSteps, (exponents.len() as u64) + 1)
            .map_err(witness_error)?;
        if coefficient.numer().limb_count() > MAX_ELIMINATION_WITNESS_COEFFICIENT_LIMBS
            || coefficient.denom().limb_count() > MAX_ELIMINATION_WITNESS_COEFFICIENT_LIMBS
        {
            return Err(witness_error("coefficient limb bound exceeded"));
        }
        nodes += 2; // coefficient and monomial product
        payload_bytes +=
            coefficient.numer().bits().div_ceil(8) + coefficient.denom().bits().div_ceil(8);
        for (index, &exponent) in exponents.iter().enumerate() {
            if exponent != 0 {
                nodes += if exponent == 1 { 1 } else { 3 };
                payload_bytes += poly.generators[index].name.len() as u64;
            }
        }
        if nodes > MAX_ELIMINATION_WITNESS_EXPR_NODES {
            return Err(witness_error("expression node bound exceeded"));
        }
    }
    *expression_nodes = expression_nodes
        .checked_add(nodes)
        .ok_or_else(|| witness_error("expression node count overflowed"))?;
    if *expression_nodes > MAX_ELIMINATION_WITNESS_EXPR_NODES {
        return Err(witness_error("expression node bound exceeded"));
    }
    meter
        .charge_batch(&[
            (
                Dimension::MemoryBytes,
                payload_bytes + (nodes as u64) * (std::mem::size_of::<Expr>() as u64),
            ),
            (Dimension::AllocationCount, nodes as u64),
        ])
        .map_err(witness_error)
}

/// Independently checks one elimination consequence and returns a portable kernel derivation.
///
/// The request supplies the original input ideal and the variables to eliminate. The verifier
/// validates ring/width/shape, checks the forbidden variables are absent, and passes the raw
/// coefficient products to `fsym-proof-kernel`'s independent polynomial normalizer. It does not
/// trust a generator product, a stored verdict, or a Groebner search trace. An empty elimination
/// list makes this a standalone input-ideal membership verifier.
///
/// Preflight and derivation work consume the caller's meter. Cancellation is checked before
/// expression construction, before each bounded kernel call, and before returning the proof;
/// no accepted state is published on cancellation, resource refusal, or verification failure.
pub fn verify_elimination_witness(
    initial_basis: &[MultivariatePoly],
    eliminated_vars: &[Symbol],
    witness: &EliminationWitness,
    meter: &mut impl BudgetMeter,
) -> Result<DerivationTree, PolyError> {
    meter.checkpoint().map_err(witness_error)?;
    meter
        .charge(Dimension::ComputeSteps, 1)
        .map_err(witness_error)?;
    if initial_basis.len() > MAX_GROEBNER_BASIS_POLYNOMIALS
        || witness.input_coefficients.len() != initial_basis.len()
    {
        return Err(witness_error("input-ideal witness width is invalid"));
    }
    witness.polynomial.validate_shape()?;
    let permutation = elimination_permutation(&witness.polynomial.generators, eliminated_vars)?;
    if permutation[..eliminated_vars.len()]
        .iter()
        .any(|&index| witness.polynomial.degree_in(index) != 0)
    {
        return Err(witness_error(
            "result still contains an eliminated variable",
        ));
    }

    let mut expression_nodes = initial_basis.len() + 1; // product wrappers and sum
    preflight_witness_polynomial(&witness.polynomial, &mut expression_nodes, meter)?;
    for poly in initial_basis.iter().chain(&witness.input_coefficients) {
        if poly.generators != witness.polynomial.generators {
            return Err(incompatible_rings(
                &witness.polynomial.generators,
                &poly.generators,
            ));
        }
        preflight_witness_polynomial(poly, &mut expression_nodes, meter)?;
    }

    meter.checkpoint().map_err(witness_error)?;
    let lhs = witness.polynomial.to_expr()?;
    let mut products = Vec::with_capacity(initial_basis.len());
    for (coefficient, input) in witness.input_coefficients.iter().zip(initial_basis) {
        meter.checkpoint().map_err(witness_error)?;
        products.push(Expr::Mul(vec![coefficient.to_expr()?, input.to_expr()?]));
    }
    let rhs = if products.is_empty() {
        Expr::from_i64(0)
    } else {
        Expr::Add(products)
    };
    let expected = Claim::AlgebraicIdentity {
        lhs: lhs.clone(),
        rhs: rhs.clone(),
    };
    let context = ImmutableAssumptionsSnapshot::empty();
    let mut kernel = ProofKernel::new((*context).clone());
    let root = kernel
        .prove_definitional_reduction(lhs, rhs, "polynomial_ring_equivalence", meter)
        .map_err(witness_error)?;
    let derivation = kernel.export_derivation(root).map_err(witness_error)?;
    let verification_units = derivation_verification_units(&derivation).map_err(witness_error)?;
    meter.checkpoint().map_err(witness_error)?;
    meter
        .charge(Dimension::ComputeSteps, verification_units)
        .map_err(witness_error)?;
    let established =
        verify_derivation_independent(&derivation, &context).map_err(witness_error)?;
    if established != expected {
        return Err(witness_error(
            "independent verifier established a different claim",
        ));
    }
    meter.checkpoint().map_err(witness_error)?;

    Ok(derivation)
}

/// Certificate for a minimal reduced Groebner basis under a specified term ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroebnerBasisCertificate {
    pub order: TermOrder,
    pub basis: Vec<MultivariatePoly>,
    /// For each basis polynomial, coefficients expressing it as a linear combination of the
    /// original input basis. The verifier checks these exact identities independently.
    pub input_ideal_witnesses: Vec<Vec<MultivariatePoly>>,
}

/// Computes the minimal reduced Groebner basis along with a typed verification certificate.
pub fn groebner_basis_with_certificate(
    initial_basis: &[MultivariatePoly],
    order: TermOrder,
) -> Result<GroebnerBasisCertificate, PolyError> {
    let tracked_basis = tracked_groebner_basis(initial_basis, order)?;
    let mut basis = Vec::with_capacity(tracked_basis.len());
    let mut input_ideal_witnesses = Vec::with_capacity(tracked_basis.len());
    for tracked in tracked_basis {
        basis.push(tracked.poly);
        input_ideal_witnesses.push(tracked.input_coefficients);
    }
    let cert = GroebnerBasisCertificate {
        order,
        basis,
        input_ideal_witnesses,
    };
    verify_groebner_certificate(initial_basis, &cert)?;
    Ok(cert)
}

/// Independent verifier for a minimal reduced Groebner basis certificate.
///
/// Verifies without search:
/// 1. Ring and generator compatibility between input generators and basis polynomials.
/// 2. Every reported basis polynomial is an exact linear combination of the input generators.
/// 3. Every input generator $f_i$ belongs to the ideal $\langle G \rangle$: $f_i \xrightarrow{G} 0$.
/// 4. Every basis polynomial $g_j$ is monic ($\text{LC}(g_j) = 1$).
/// 5. Buchberger's S-polynomial criterion: for all $i < j$, $S(g_i, g_j) \xrightarrow{G} 0$.
/// 6. Minimal reduced property: for all $i \neq j$, no monomial in $\text{supp}(g_i)$ is divisible by $\text{LM}(g_j)$.
pub fn verify_groebner_certificate(
    initial_basis: &[MultivariatePoly],
    cert: &GroebnerBasisCertificate,
) -> Result<(), PolyError> {
    validate_common_ring(initial_basis)?;
    validate_common_ring(&cert.basis)?;

    if initial_basis.len() > MAX_GROEBNER_BASIS_POLYNOMIALS
        || cert.basis.len() > MAX_GROEBNER_BASIS_POLYNOMIALS
    {
        return Err(PolyError::General(format!(
            "Groebner certificate exceeds the polynomial limit of {MAX_GROEBNER_BASIS_POLYNOMIALS}"
        )));
    }
    if cert.input_ideal_witnesses.len() != cert.basis.len() {
        return Err(PolyError::General(
            "Groebner certificate witness count does not match the basis".to_string(),
        ));
    }

    if let (Some(first_in), Some(first_gb)) = (initial_basis.first(), cert.basis.first())
        && first_in.generators != first_gb.generators
    {
        return Err(incompatible_rings(
            &first_in.generators,
            &first_gb.generators,
        ));
    }

    // Reverse ideal containment: every g_j must be proven to belong to the input ideal.
    for (basis_poly, witness) in cert.basis.iter().zip(&cert.input_ideal_witnesses) {
        if witness.len() != initial_basis.len() {
            return Err(PolyError::General(
                "Groebner certificate witness width does not match the input basis".to_string(),
            ));
        }
        let mut reconstructed = MultivariatePoly::zero(basis_poly.generators.clone());
        for (coefficient, input_poly) in witness.iter().zip(initial_basis) {
            coefficient.validate_shape()?;
            if coefficient.generators != basis_poly.generators {
                return Err(incompatible_rings(
                    &basis_poly.generators,
                    &coefficient.generators,
                ));
            }
            reconstructed = reconstructed.add(&coefficient.mul(input_poly)?)?;
        }
        if reconstructed != *basis_poly {
            return Err(PolyError::General(
                "Groebner basis polynomial is not established as a member of the input ideal"
                    .to_string(),
            ));
        }
    }

    let order = cert.order;

    // 1. Monic check and non-zero check
    for g in &cert.basis {
        if g.is_zero() {
            return Err(PolyError::General(
                "Groebner basis certificate contains non-canonical zero polynomial".to_string(),
            ));
        }
        let lc = g.leading_coeff(order).ok_or_else(|| {
            PolyError::General("Groebner basis polynomial has no leading coefficient".to_string())
        })?;
        if !lc.is_one() {
            return Err(PolyError::General(
                "Groebner basis polynomial is not monic".to_string(),
            ));
        }
    }

    // 2. Input ideal containment: every f_i in initial_basis reduces to 0 modulo G
    for f in initial_basis {
        if f.is_zero() {
            continue;
        }
        let (_, rem) = f.div_rem(&cert.basis, order)?;
        if !rem.is_zero() {
            return Err(PolyError::General(
                "Input ideal polynomial does not reduce to zero modulo Groebner basis".to_string(),
            ));
        }
    }

    // 3. S-pair Buchberger criterion: S(g_i, g_j) reduces to 0 modulo G for all pairs
    for i in 0..cert.basis.len() {
        for j in (i + 1)..cert.basis.len() {
            let s = s_polynomial(&cert.basis[i], &cert.basis[j], order)?;
            let (_, rem) = s.div_rem(&cert.basis, order)?;
            if !rem.is_zero() {
                return Err(PolyError::General(format!(
                    "S-polynomial S(g_{i}, g_{j}) does not reduce to zero modulo Groebner basis"
                )));
            }
        }
    }

    // 4. Reducedness: for all i != j, no monomial in supp(g_i) is divisible by LM(g_j)
    for (i, gi) in cert.basis.iter().enumerate() {
        for (j, gj) in cert.basis.iter().enumerate() {
            if i == j {
                continue;
            }
            let lm_j = gj.leading_monomial(order).unwrap();
            for exp_i in gi.terms.keys() {
                if divides(&lm_j, exp_i) {
                    return Err(PolyError::General(format!(
                        "Groebner basis is not reduced: term in g_{i} is divisible by leading monomial of g_{j}"
                    )));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod elimination_tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, MeterError, Unbounded};
    use fsym_core::BigInt;

    fn poly(generators: &[Symbol], terms: &[(Vec<u32>, i64, i64)]) -> MultivariatePoly {
        MultivariatePoly::new(
            generators.to_vec(),
            terms
                .iter()
                .map(|(exponents, numerator, denominator)| {
                    (
                        exponents.clone(),
                        BigRational::new((*numerator).into(), (*denominator).into()),
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    fn quadratic_system() -> (Vec<Symbol>, Vec<MultivariatePoly>) {
        let generators = vec![Symbol::new("x"), Symbol::new("y")];
        let input = vec![
            poly(&generators, &[(vec![1, 0], 1, 1), (vec![0, 1], -1, 1)]),
            poly(&generators, &[(vec![0, 2], 1, 1), (vec![0, 0], -1, 1)]),
        ];
        (generators, input)
    }

    #[test]
    fn eliminates_nonleading_variable_without_losing_the_elimination_ideal() {
        // <x-y, y^2-1> intersect Q[x] = <x^2-1>. Filtering the basis in x>y
        // incorrectly returns the empty list; elimination needs y>x.
        let (generators, input) = quadratic_system();
        let expected = poly(&generators, &[(vec![2, 0], 1, 1), (vec![0, 0], -1, 1)]);
        assert_eq!(
            eliminate(&input, &generators[1..]).unwrap(),
            vec![expected.clone()]
        );
        let witnesses = eliminate_with_witnesses(&input, &generators[1..]).unwrap();
        assert_eq!(witnesses.len(), 1);
        assert_eq!(witnesses[0].polynomial, expected);
        let proof =
            verify_elimination_witness(&input, &generators[1..], &witnesses[0], &mut Unbounded)
                .unwrap();
        let claim =
            verify_derivation_independent(&proof, &ImmutableAssumptionsSnapshot::empty()).unwrap();
        assert!(
            matches!(claim, Claim::AlgebraicIdentity { lhs, .. } if lhs == expected.to_expr().unwrap())
        );

        // Independent, hand-derived cofactor identity, rather than a generated oracle:
        // x^2-1 = (x+y)(x-y) + (y^2-1).
        let direct = EliminationWitness {
            polynomial: expected,
            input_coefficients: vec![
                poly(&generators, &[(vec![1, 0], 1, 1), (vec![0, 1], 1, 1)]),
                MultivariatePoly::one(generators.clone()),
            ],
        };
        verify_elimination_witness(&input, &generators[1..], &direct, &mut Unbounded).unwrap();
    }

    #[test]
    fn noncontiguous_elimination_is_independent_of_request_order() {
        // y=x^2, w=yz, w=1, eliminate y,w -> x^2*z-1.
        let generators = ["x", "y", "z", "w"].map(Symbol::new).to_vec();
        let input = vec![
            poly(
                &generators,
                &[(vec![0, 1, 0, 0], 1, 1), (vec![2, 0, 0, 0], -1, 1)],
            ),
            poly(
                &generators,
                &[(vec![0, 0, 0, 1], 1, 1), (vec![0, 1, 1, 0], -1, 1)],
            ),
            poly(
                &generators,
                &[(vec![0, 0, 0, 1], 1, 1), (vec![0, 0, 0, 0], -1, 1)],
            ),
        ];
        let expected = poly(
            &generators,
            &[(vec![2, 0, 1, 0], 1, 1), (vec![0, 0, 0, 0], -1, 1)],
        );
        let eliminated = vec![generators[3].clone(), generators[1].clone()];
        assert_eq!(
            eliminate(&input, &eliminated).unwrap(),
            vec![expected.clone()]
        );
        let witnesses = eliminate_with_witnesses(&input, &eliminated).unwrap();
        assert_eq!(witnesses.len(), 1);
        assert_eq!(witnesses[0].polynomial, expected);
        verify_elimination_witness(&input, &eliminated, &witnesses[0], &mut Unbounded).unwrap();
        let reverse = vec![generators[1].clone(), generators[3].clone()];
        assert_eq!(
            eliminate_with_witnesses(&input, &reverse).unwrap(),
            witnesses
        );
        let reordered = input
            .iter()
            .map(|p| permute_generators(p, &[2, 0, 3, 1]).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            eliminate(&reordered, &eliminated).unwrap(),
            vec![permute_generators(&expected, &[2, 0, 3, 1]).unwrap()]
        );
    }

    #[test]
    fn rational_nonmonic_inputs_keep_witnesses_in_the_original_ring() {
        let generators = vec![Symbol::new("x"), Symbol::new("y")];
        let input = vec![
            poly(&generators, &[(vec![1, 0], 3, 2), (vec![0, 1], -1, 1)]),
            poly(&generators, &[(vec![0, 2], 1, 2), (vec![0, 0], -2, 1)]),
        ];
        let expected = poly(&generators, &[(vec![2, 0], 1, 1), (vec![0, 0], -16, 9)]);
        let witnesses = eliminate_with_witnesses(&input, &generators[1..]).unwrap();
        assert_eq!(witnesses.len(), 1);
        assert_eq!(witnesses[0].polynomial, expected);
        assert!(
            witnesses[0]
                .input_coefficients
                .iter()
                .all(|p| p.generators == generators)
        );
        verify_elimination_witness(&input, &generators[1..], &witnesses[0], &mut Unbounded)
            .unwrap();
    }

    #[test]
    fn empty_all_and_zero_ideal_elimination_have_explicit_semantics() {
        let (generators, input) = quadratic_system();
        assert_eq!(
            eliminate(&input, &[]).unwrap(),
            groebner_basis(&input, TermOrder::Lex).unwrap()
        );
        assert!(eliminate(&input, &generators).unwrap().is_empty());
        assert!(eliminate(&[], &generators).unwrap().is_empty());
        assert!(
            eliminate_with_witnesses(&[], &generators)
                .unwrap()
                .is_empty()
        );
        assert!(
            eliminate(&[MultivariatePoly::zero(generators.clone())], &generators)
                .unwrap()
                .is_empty()
        );
        let unit_input = vec![
            poly(&generators, &[(vec![1, 0], 1, 1)]),
            poly(&generators, &[(vec![1, 0], 1, 1), (vec![0, 0], -1, 1)]),
        ];
        let witnesses = eliminate_with_witnesses(&unit_input, &generators).unwrap();
        assert_eq!(witnesses.len(), 1);
        assert_eq!(
            witnesses[0].polynomial,
            MultivariatePoly::one(generators.clone())
        );
        verify_elimination_witness(&unit_input, &generators, &witnesses[0], &mut Unbounded)
            .unwrap();
        let zero = EliminationWitness {
            polynomial: MultivariatePoly::zero(generators.clone()),
            input_coefficients: vec![],
        };
        verify_elimination_witness(&[], &generators, &zero, &mut Unbounded).unwrap();
        let false_unit = EliminationWitness {
            polynomial: MultivariatePoly::one(generators.clone()),
            input_coefficients: vec![],
        };
        assert!(verify_elimination_witness(&[], &generators, &false_unit, &mut Unbounded).is_err());
        assert_eq!(
            eliminate(&[MultivariatePoly::one(vec![])], &[]).unwrap(),
            vec![MultivariatePoly::one(vec![])]
        );
    }

    #[test]
    fn invalid_variables_and_ring_shapes_are_refused_before_reindexing() {
        let (generators, input) = quadratic_system();
        for eliminated in [
            vec![Symbol::new("missing")],
            vec![generators[0].clone(), generators[0].clone()],
            vec![
                generators[0].clone(),
                generators[1].clone(),
                Symbol::new("missing"),
            ],
        ] {
            assert!(eliminate(&input, &eliminated).is_err());
            assert!(eliminate_with_witnesses(&input, &eliminated).is_err());
        }
        let malformed = MultivariatePoly {
            generators: generators.clone(),
            terms: BTreeMap::from([(vec![1], BigRational::one())]),
        };
        assert!(eliminate(std::slice::from_ref(&malformed), &generators[1..]).is_err());
        assert!(eliminate_with_witnesses(&[malformed], &generators[1..]).is_err());
        let mut mixed = input;
        mixed.push(MultivariatePoly::one(vec![Symbol::new("z")]));
        assert!(eliminate(&mixed, &[]).is_err());
    }

    #[test]
    fn independent_verifier_rejects_forged_consequences_and_cofactors() {
        let (generators, input) = quadratic_system();
        let witness = eliminate_with_witnesses(&input, &generators[1..])
            .unwrap()
            .remove(0);
        let mut wrong_result = witness.clone();
        wrong_result.polynomial = MultivariatePoly::one(generators.clone());
        assert!(
            verify_elimination_witness(&input, &generators[1..], &wrong_result, &mut Unbounded)
                .is_err()
        );
        let mut wrong_cofactor = witness.clone();
        wrong_cofactor.input_coefficients[0] = wrong_cofactor.input_coefficients[0]
            .add(&MultivariatePoly::one(generators.clone()))
            .unwrap();
        assert!(
            verify_elimination_witness(&input, &generators[1..], &wrong_cofactor, &mut Unbounded)
                .is_err()
        );
        let mut missing_cofactor = witness.clone();
        missing_cofactor.input_coefficients.pop();
        assert!(
            verify_elimination_witness(&input, &generators[1..], &missing_cofactor, &mut Unbounded)
                .is_err()
        );
        // Valid input-ideal membership does not imply an elimination consequence.
        let contains_eliminated = EliminationWitness {
            polynomial: input[0].clone(),
            input_coefficients: vec![
                MultivariatePoly::one(generators.clone()),
                MultivariatePoly::zero(generators.clone()),
            ],
        };
        verify_elimination_witness(&input, &[], &contains_eliminated, &mut Unbounded).unwrap();
        assert!(
            verify_elimination_witness(
                &input,
                &generators[1..],
                &contains_eliminated,
                &mut Unbounded
            )
            .is_err()
        );
        let mut wrong_ring = witness.clone();
        wrong_ring.input_coefficients[0] = MultivariatePoly::one(vec![Symbol::new("z")]);
        assert!(
            verify_elimination_witness(&input, &generators[1..], &wrong_ring, &mut Unbounded)
                .is_err()
        );
        let mut malformed = witness.clone();
        malformed.input_coefficients[0].terms = BTreeMap::from([(vec![1], BigRational::one())]);
        assert!(
            verify_elimination_witness(&input, &generators[1..], &malformed, &mut Unbounded)
                .is_err()
        );
        let mut wrong_input = input.clone();
        wrong_input[1] = poly(&generators, &[(vec![0, 2], 1, 1), (vec![0, 0], -2, 1)]);
        assert!(
            verify_elimination_witness(&wrong_input, &generators[1..], &witness, &mut Unbounded)
                .is_err()
        );
    }

    #[derive(Default)]
    struct WitnessMeter {
        checkpoints: usize,
        cancel_at: Option<usize>,
    }

    impl BudgetMeter for WitnessMeter {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints += 1;
            if self.cancel_at == Some(self.checkpoints) {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn witness_verification_cancellation_and_resource_refusal_never_return_a_proof() {
        let (generators, input) = quadratic_system();
        let witness = eliminate_with_witnesses(&input, &generators[1..])
            .unwrap()
            .remove(0);
        let mut counting = WitnessMeter::default();
        verify_elimination_witness(&input, &generators[1..], &witness, &mut counting).unwrap();
        assert!(counting.checkpoints > 3);
        for checkpoint in 1..=counting.checkpoints {
            let mut meter = WitnessMeter {
                cancel_at: Some(checkpoint),
                ..WitnessMeter::default()
            };
            assert!(matches!(
                verify_elimination_witness(&input, &generators[1..], &witness, &mut meter),
                Err(PolyError::General(message)) if message.contains("cancelled")
            ));
        }
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut limits = BudgetLimits::uniform(1_000_000, 0);
            limits.dimensions[dimension.index()] = 0;
            let mut budget = Budget::new(limits);
            assert!(
                verify_elimination_witness(&input, &generators[1..], &witness, &mut budget)
                    .is_err()
            );
        }
    }

    #[test]
    fn oversized_witness_is_refused_before_expression_construction() {
        let (generators, input) = quadratic_system();
        let mut witness = eliminate_with_witnesses(&input, &generators[1..])
            .unwrap()
            .remove(0);
        witness.input_coefficients[0].terms = BTreeMap::from([(
            vec![0, 0],
            BigRational::from_integer(
                BigInt::one() << (MAX_ELIMINATION_WITNESS_COEFFICIENT_LIMBS * 64) as usize,
            ),
        )]);
        assert!(matches!(
            verify_elimination_witness(&input, &generators[1..], &witness, &mut Unbounded),
            Err(PolyError::General(message)) if message.contains("coefficient limb bound")
        ));
    }
}
