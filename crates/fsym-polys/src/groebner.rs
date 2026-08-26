//! Groebner bases, Buchberger algorithm, ideal membership, and elimination (WS17).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::multivariate::{MultivariatePoly, TermOrder};
use fsym_core::{BigRational, Symbol};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const MAX_GROEBNER_BASIS_POLYNOMIALS: usize = 1_024;
const MAX_GROEBNER_PENDING_PAIRS: usize = 1_000_000;

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
                    let term_poly = MultivariatePoly::new(self.generators.clone(), term_map);

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
            MultivariatePoly::new(self.generators.clone(), remainder_terms),
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
        Ok(Self::new(self.generators.clone(), monic_terms))
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
    let poly_f = MultivariatePoly::new(f.generators.clone(), term_f);

    let mut term_g = BTreeMap::new();
    term_g.insert(g_shift, BigRational::one() / g_c);
    let poly_g = MultivariatePoly::new(g.generators.clone(), term_g);

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
        .filter(|p| !p.is_zero())
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
        let (_, rem) = s.div_rem(&g, order)?;
        if !rem.is_zero() {
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
            g.push(rem);
        }
    }

    // Auto-reduction: make monic and reduce each element against all others
    let reduced: Vec<MultivariatePoly> = g
        .into_iter()
        .map(|p| p.to_monic(order))
        .collect::<Result<_, _>>()?;

    // Remove redundant elements whose leading monomial is divisible by another
    let mut minimal = Vec::new();
    for (i, p) in reduced.iter().enumerate() {
        if p.is_zero() {
            continue;
        }
        let lm_p = p.leading_monomial(order).unwrap();
        let is_redundant = reduced.iter().enumerate().any(|(j, other)| {
            if i == j || other.is_zero() {
                return false;
            }
            let lm_other = other.leading_monomial(order).unwrap();
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
        let other_divisors: Vec<MultivariatePoly> = minimal
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, q)| q.clone())
            .collect();
        let (_, rem) = p.div_rem(&other_divisors, order)?;
        if !rem.is_zero() {
            fully_reduced.push(rem.to_monic(order)?);
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

/// Variable elimination via Lexicographic Groebner basis.
///
/// Returns polynomials in the Groebner basis that do not contain any of `eliminated_vars`.
pub fn eliminate(
    initial_basis: &[MultivariatePoly],
    eliminated_vars: &[Symbol],
) -> Result<Vec<MultivariatePoly>, PolyError> {
    if initial_basis.is_empty() {
        return Ok(Vec::new());
    }
    let gb = groebner_basis(initial_basis, TermOrder::Lex)?;
    let gens = &initial_basis[0].generators;

    let elim_indices: Vec<usize> = eliminated_vars
        .iter()
        .map(|symbol| {
            gens.iter()
                .position(|generator| generator == symbol)
                .ok_or_else(|| {
                    PolyError::IncompatibleGenerators(
                        format!("elimination variable {}", symbol.name),
                        "missing from polynomial generators".to_string(),
                    )
                })
        })
        .collect::<Result<_, _>>()?;

    let eliminated_basis = gb
        .into_iter()
        .filter(|p| elim_indices.iter().all(|&idx| p.degree_in(idx) == 0))
        .collect();

    Ok(eliminated_basis)
}

/// Certificate for a minimal reduced Groebner basis under a specified term ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroebnerBasisCertificate {
    pub order: TermOrder,
    pub basis: Vec<MultivariatePoly>,
}

/// Computes the minimal reduced Groebner basis along with a typed verification certificate.
pub fn groebner_basis_with_certificate(
    initial_basis: &[MultivariatePoly],
    order: TermOrder,
) -> Result<GroebnerBasisCertificate, PolyError> {
    let basis = groebner_basis(initial_basis, order)?;
    let cert = GroebnerBasisCertificate { order, basis };
    verify_groebner_certificate(initial_basis, &cert)?;
    Ok(cert)
}

/// Independent verifier for a minimal reduced Groebner basis certificate.
///
/// Verifies without search:
/// 1. Ring and generator compatibility between input generators and basis polynomials.
/// 2. Every input generator $f_i$ belongs to the ideal $\langle G \rangle$: $f_i \xrightarrow{G} 0$.
/// 3. Every basis polynomial $g_j$ is monic ($\text{LC}(g_j) = 1$).
/// 4. Buchberger's S-polynomial criterion: for all $i < j$, $S(g_i, g_j) \xrightarrow{G} 0$.
/// 5. Minimal reduced property: for all $i \neq j$, no monomial in $\text{supp}(g_i)$ is divisible by $\text{LM}(g_j)$.
pub fn verify_groebner_certificate(
    initial_basis: &[MultivariatePoly],
    cert: &GroebnerBasisCertificate,
) -> Result<(), PolyError> {
    validate_common_ring(initial_basis)?;
    validate_common_ring(&cert.basis)?;

    if let (Some(first_in), Some(first_gb)) = (initial_basis.first(), cert.basis.first()) {
        if first_in.generators != first_gb.generators {
            return Err(incompatible_rings(
                &first_in.generators,
                &first_gb.generators,
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
            for (exp_i, _) in &gi.terms {
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
