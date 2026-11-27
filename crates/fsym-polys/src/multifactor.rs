//! Multivariate factorization over QQ (WS09).
//!
//! `p = scale * prod(f_i ^ m_i)` with primitive integer factors whose
//! lex-leading coefficient is positive. The pipeline:
//!
//! 1. numeric content and monomial content come out first;
//! 2. a univariate remainder goes to [`complete_factorization`]; a
//!    bivariate affine remainder splits into coefficient content and a
//!    primitive affine factor over the other coordinate;
//! 3. a homogeneous remainder is dehomogenized in its last variable,
//!    factored, and every factor rehomogenized (a bijection on factors once
//!    the monomial content is gone);
//! 4. otherwise Kronecker substitution `x_{i+1} -> t^(B_i)` maps the
//!    polynomial to a univariate one; its irreducible factors are
//!    recombined and every candidate is mapped back and accepted only after
//!    exact multivariate division of the remaining polynomial. If the
//!    mixed-radix image exceeds the degree limit, independent groups of coordinates
//!    can instead be identified exactly on the sparse coefficients.
//!
//! Every accepted factor is therefore an exact divisor, and the product of
//! the result is re-checked against the input before returning. Inputs
//! outside the bounded regime (Kronecker image degree or recombination
//! count) refuse with an error rather than returning a partial answer.

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::factorization::complete_factorization;
use crate::multivariate::{MultivariatePoly, TermOrder};
use crate::univariate::UnivariatePoly;
use fsym_core::{BigInt, BigRational, Symbol};
use num_traits::{One, Signed, Zero};
use std::collections::BTreeMap;

/// Recombination budget: the number of candidate subsets examined.
const MAX_RECOMBINATIONS: usize = 1 << 14;
/// Non-singleton partitions are tried in rings with four to eight active variables.
const MAX_PARTITION_AXES: usize = 8;
/// Shared by all recursive branches. Each candidate prepays one support scan
/// and at most one exact coefficient-table check per source term.
const MAX_PARTITION_TERM_VISITS: usize = 262_144;

/// `scale * prod(factor ^ multiplicity)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultivariateFactorization {
    pub scale: BigRational,
    pub factors: Vec<(MultivariatePoly, usize)>,
}

fn gcd_int(a: &BigInt, b: &BigInt) -> BigInt {
    fsym_core::arith::gcd(a, b)
}

fn lcm_int(a: &BigInt, b: &BigInt) -> BigInt {
    let g = gcd_int(a, b);
    if g.is_zero() {
        return BigInt::zero();
    }
    (a.clone() * b.clone() / g).abs()
}

/// Leading coefficient in lex order (the largest exponent vector).
fn lex_lead(p: &MultivariatePoly) -> BigRational {
    p.terms
        .iter()
        .next_back()
        .map(|(_, c)| c.clone())
        .unwrap_or_else(BigRational::zero)
}

/// `(c, q)` with `p = c * q`, `q` integral, primitive and lex-leading positive.
fn primitive(p: &MultivariatePoly) -> Result<(BigRational, MultivariatePoly), PolyError> {
    let mut den = BigInt::one();
    for c in p.terms.values() {
        den = lcm_int(&den, c.denom());
    }
    let mut num = BigInt::zero();
    for c in p.terms.values() {
        let v = (c.clone() * BigRational::from_integer(den.clone())).to_integer();
        num = gcd_int(&num, &v);
    }
    if num.is_zero() {
        return Ok((BigRational::zero(), p.clone()));
    }
    let mut content = BigRational::new(num, den);
    if lex_lead(p).is_negative() {
        content = -content;
    }
    let inv = BigRational::one() / content.clone();
    let terms = p
        .terms
        .iter()
        .map(|(e, c)| (e.clone(), c.clone() * inv.clone()))
        .collect();
    Ok((content, MultivariatePoly::new(p.generators.clone(), terms)?))
}

fn exact_div(
    p: &MultivariatePoly,
    d: &MultivariatePoly,
) -> Result<Option<MultivariatePoly>, PolyError> {
    // {d} is a Groebner basis of (d), so a zero remainder is exactly
    // divisibility.
    let (q, r) = p.div_rem(std::slice::from_ref(d), TermOrder::Lex)?;
    if !r.is_zero() {
        return Ok(None);
    }
    Ok(q.into_iter().next())
}

fn variable(p: &MultivariatePoly, i: usize) -> Result<MultivariatePoly, PolyError> {
    MultivariatePoly::var(p.generators.clone(), &p.generators[i])
}

fn is_constant(p: &MultivariatePoly) -> bool {
    p.terms.keys().all(|e| e.iter().all(|&k| k == 0))
}

fn present_vars(p: &MultivariatePoly) -> Vec<usize> {
    (0..p.generators.len())
        .filter(|&i| p.terms.keys().any(|e| e[i] > 0))
        .collect()
}

fn is_homogeneous(p: &MultivariatePoly) -> bool {
    let mut degs = p
        .terms
        .keys()
        .map(|e| e.iter().map(|&k| u64::from(k)).sum::<u64>());
    match degs.next() {
        Some(d) => degs.all(|k| k == d),
        None => true,
    }
}

/// Univariate factorization of a polynomial in exactly one variable.
fn factor_univariate(
    p: &MultivariatePoly,
    var: usize,
) -> Result<Vec<(MultivariatePoly, usize)>, PolyError> {
    let sym = p.generators[var].clone();
    let deg = p.degree_in(var) as usize;
    if deg > crate::factorization::MAX_FACTOR_DEGREE {
        return Err(PolyError::General(format!(
            "refused: univariate factor degree {deg} exceeds the complete-factorization bound"
        )));
    }
    let mut coeffs = vec![BigRational::zero(); deg + 1];
    for (e, c) in &p.terms {
        coeffs[e[var] as usize] += c.clone();
    }
    factor_univariate_in_ring(p, var, &UnivariatePoly::new(sym, coeffs))
}

/// Factor a coefficient polynomial and lift each factor into the original
/// ordered ambient ring, preserving its exact coordinate and multiplicity.
fn factor_univariate_in_ring(
    p: &MultivariatePoly,
    var: usize,
    polynomial: &UnivariatePoly,
) -> Result<Vec<(MultivariatePoly, usize)>, PolyError> {
    let cf = complete_factorization(polynomial)?;
    let mut out = Vec::new();
    for term in cf.factors {
        let mut terms = BTreeMap::new();
        for (k, c) in term.poly.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let mut e = vec![0u32; p.generators.len()];
            e[var] = k as u32;
            terms.insert(e, c.clone());
        }
        let (_, f) = primitive(&MultivariatePoly::new(p.generators.clone(), terms)?)?;
        out.push((f, term.multiplicity));
    }
    Ok(out)
}

/// Factors of a primitive polynomial without monomial content.
fn factor_core(
    p: &MultivariatePoly,
    partition_work: &mut usize,
) -> Result<Vec<(MultivariatePoly, usize)>, PolyError> {
    let present = present_vars(p);
    match present.len() {
        0 => return Ok(Vec::new()),
        1 => return factor_univariate(p, present[0]),
        _ => {}
    }
    for &axis in &present {
        if p.degree_in(axis) != 1 {
            continue;
        }
        // In R[v], a*v+b with a unit coefficient is primitive. Since its
        // degree over Frac(R) is one, Gauss's lemma makes it irreducible.
        // Here R=QQ[other coordinates] and its units are nonzero scalars.
        if (0..=1).any(|power| {
            let mut row = p.terms.keys().filter(|e| e[axis] == power);
            row.next()
                .is_some_and(|e| e.iter().enumerate().all(|(i, &k)| i == axis || k == 0))
                && row.next().is_none()
        }) {
            return Ok(vec![(p.clone(), 1)]);
        }
        if let [i, j] = present.as_slice() {
            let other = if *i == axis { *j } else { *i };
            if p.degree_in(other) <= crate::factorization::MAX_FACTOR_DEGREE as u32
                && p.terms
                    .values()
                    .all(|c| c.numer().bits() <= crate::factorization::MAX_FACTOR_COEFF_BITS as u64)
            {
                return factor_bivariate_affine(p, axis, other);
            }
        }
    }
    if is_homogeneous(p) {
        return factor_homogeneous(
            p,
            *present.last().expect("two or more variables"),
            partition_work,
        );
    }
    factor_kronecker(p, &present, partition_work)
}

/// In QQ[u][v], the only factors of a*v+b independent of v divide both
/// coefficient polynomials. Extract their monic GCD; the primitive quotient
/// has degree one over QQ(u) and hence cannot split further in QQ[u,v].
fn factor_bivariate_affine(
    p: &MultivariatePoly,
    axis: usize,
    other: usize,
) -> Result<Vec<(MultivariatePoly, usize)>, PolyError> {
    let width = p.degree_in(other) as usize + 1;
    let symbol = &p.generators[other];
    let mut rows = [
        vec![BigRational::zero(); width],
        vec![BigRational::zero(); width],
    ];
    for (e, c) in &p.terms {
        rows[e[axis] as usize][e[other] as usize] = c.clone();
    }
    let [constant, linear] = rows.map(|row| UnivariatePoly::new(symbol.clone(), row));
    let certificate = constant.extended_gcd(&linear)?;
    crate::gcd::verify_bezout_certificate(&constant, &linear, &certificate)?;
    let content = certificate.gcd;
    if content.is_one() {
        return Ok(vec![(p.clone(), 1)]);
    }
    let mut affine = BTreeMap::new();
    for (power, row) in [constant, linear].iter().enumerate() {
        let (quotient, remainder) = row.div_rem(&content)?;
        if !remainder.is_zero() {
            return Err(PolyError::IdentityCheckFailed(
                "affine coefficient GCD does not divide the source rows".to_string(),
            ));
        }
        for (degree, c) in quotient.coeffs.into_iter().enumerate() {
            if !c.is_zero() {
                let mut e = vec![0; p.generators.len()];
                e[axis] = power as u32;
                e[other] = degree as u32;
                affine.insert(e, c);
            }
        }
    }
    let (_, affine) = primitive(&MultivariatePoly::new(p.generators.clone(), affine)?)?;
    let mut factors = factor_univariate_in_ring(p, other, &content)?;
    factors.push((affine, 1));
    Ok(factors)
}

fn factor_homogeneous(
    p: &MultivariatePoly,
    v: usize,
    partition_work: &mut usize,
) -> Result<Vec<(MultivariatePoly, usize)>, PolyError> {
    let mut terms = BTreeMap::new();
    for (e, c) in &p.terms {
        let mut e2 = e.clone();
        e2[v] = 0;
        terms.insert(e2, c.clone());
    }
    let dehom = MultivariatePoly::new(p.generators.clone(), terms)?;
    let (_, dehom) = primitive(&dehom)?;
    let mut out = Vec::new();
    for (f, m) in factor_core(&dehom, partition_work)? {
        let d = f
            .terms
            .keys()
            .map(|e| e.iter().map(|&k| u64::from(k)).sum::<u64>())
            .max()
            .unwrap_or(0);
        let mut terms = BTreeMap::new();
        for (e, c) in &f.terms {
            let mut e2 = e.clone();
            let degree: u64 = e.iter().map(|&k| u64::from(k)).sum();
            e2[v] = u32::try_from(d - degree).map_err(|_| {
                PolyError::General("refused: rehomogenized exponent exceeds u32".to_string())
            })?;
            terms.insert(e2, c.clone());
        }
        let (_, f) = primitive(&MultivariatePoly::new(p.generators.clone(), terms)?)?;
        out.push((f, m));
    }
    Ok(out)
}

/// Mixed-radix decoding of a Kronecker exponent.
fn decode(mut k: u64, present: &[usize], radix: &[u64], width: usize) -> Vec<u32> {
    let mut e = vec![0u32; width];
    for (i, &v) in present.iter().enumerate() {
        e[v] = (k % radix[i]) as u32;
        k /= radix[i];
    }
    e
}

type CoefficientFiber<'a> = Vec<(&'a Vec<u32>, &'a BigRational)>;

/// Confirm rank one for the proposed partition, including every implicit
/// zero. The pivot row and column have unique projections. Their Cartesian
/// size must equal the support size, and every crossed entry must exist
/// with p[i,j]*p[pivot] == p[i,pivot]*p[pivot,j].
fn split_fibers(
    p: &MultivariatePoly,
    axes: &[usize],
    pivot_value: &BigRational,
    row: CoefficientFiber<'_>,
    column: CoefficientFiber<'_>,
) -> Result<Option<(MultivariatePoly, MultivariatePoly)>, PolyError> {
    if row.len() < 2
        || column.len() < 2
        || row.len().checked_mul(column.len()) != Some(p.terms.len())
    {
        return Ok(None);
    }
    for &(other, b) in &column {
        let mut exponent = other.clone();
        for &(entry, a) in &row {
            for &axis in axes {
                exponent[axis] = entry[axis];
            }
            if !p
                .terms
                .get(&exponent)
                .is_some_and(|value| value * pivot_value == a * b)
            {
                return Ok(None);
            }
        }
    }
    let left = row
        .into_iter()
        .map(|(e, c)| {
            let mut powers = vec![0; p.generators.len()];
            for &axis in axes {
                powers[axis] = e[axis];
            }
            (powers, c.clone())
        })
        .collect();
    let right = column
        .into_iter()
        .map(|(e, c)| {
            let mut powers = e.clone();
            for &axis in axes {
                powers[axis] = 0;
            }
            (powers, c.clone())
        })
        .collect();
    let (_, left) = primitive(&MultivariatePoly::new(p.generators.clone(), left)?)?;
    let (_, right) = primitive(&MultivariatePoly::new(p.generators.clone(), right)?)?;
    Ok(Some((left, right)))
}

/// Extract independent QQ factors on disjoint coordinates. The singleton lane
/// keeps its original selection order and bounds for every ring size. Only
/// when it cannot split an oversized image do we search non-singleton groups.
fn factor_separated_coordinates(
    p: &MultivariatePoly,
    present: &[usize],
    partition_work: &mut usize,
) -> Result<Option<Vec<(MultivariatePoly, usize)>>, PolyError> {
    if p.terms
        .values()
        .any(|c| c.numer().bits() > crate::factorization::MAX_FACTOR_COEFF_BITS as u64)
    {
        return Ok(None);
    }
    let Some((pivot, pivot_value)) = p.terms.first_key_value() else {
        return Ok(None);
    };
    for &axis in present {
        if p.degree_in(axis) > crate::factorization::MAX_FACTOR_DEGREE as u32 {
            continue;
        }
        let row = p
            .terms
            .iter()
            .filter(|(e, _)| e[..axis] == pivot[..axis] && e[axis + 1..] == pivot[axis + 1..])
            .collect();
        let column = p
            .terms
            .iter()
            .filter(|(e, _)| e[axis] == pivot[axis])
            .collect();
        if let Some((coordinate, complement)) = split_fibers(p, &[axis], pivot_value, row, column)?
        {
            let mut factors = factor_univariate(&coordinate, axis)?;
            factors.extend(factor_core(&complement, partition_work)?);
            return Ok(Some(factors));
        }
    }
    if !(4..=MAX_PARTITION_AXES).contains(&present.len()) {
        return Ok(None);
    }
    // Keep the first active coordinate on the left so a partition and its
    // complement are examined exactly once, regardless of ambient order.
    for mask in (1usize..(1 << present.len())).step_by(2) {
        let count = mask.count_ones() as usize;
        if count < 2 || present.len() - count < 2 {
            continue;
        }
        *partition_work = partition_work
            .checked_sub(2 * p.terms.len())
            .ok_or_else(|| {
                PolyError::General(
                    "refused: coordinate partition term-work budget exhausted".into(),
                )
            })?;
        let mut left_axes = Vec::with_capacity(count);
        let mut right_axes = Vec::with_capacity(present.len() - count);
        for (bit, &axis) in present.iter().enumerate() {
            if mask & (1 << bit) == 0 {
                right_axes.push(axis);
            } else {
                left_axes.push(axis);
            }
        }
        let mut row = Vec::new();
        let mut column = Vec::new();
        for (e, c) in &p.terms {
            if right_axes.iter().all(|&i| e[i] == pivot[i]) {
                row.push((e, c));
            }
            if left_axes.iter().all(|&i| e[i] == pivot[i]) {
                column.push((e, c));
            }
        }
        if let Some((left, right)) = split_fibers(p, &left_axes, pivot_value, row, column)? {
            let mut factors = factor_core(&left, partition_work)?;
            factors.extend(factor_core(&right, partition_work)?);
            return Ok(Some(factors));
        }
    }
    Ok(None)
}

fn factor_kronecker(
    p: &MultivariatePoly,
    present: &[usize],
    partition_work: &mut usize,
) -> Result<Vec<(MultivariatePoly, usize)>, PolyError> {
    let width = p.generators.len();
    let radix: Vec<u64> = present
        .iter()
        .map(|&v| u64::from(p.degree_in(v)) + 1)
        .collect();
    let mut weights = Vec::with_capacity(present.len());
    let mut w = 1u64;
    for r in &radix {
        weights.push(w);
        let Some(next) = w.checked_mul(*r) else {
            return factor_separated_coordinates(p, present, partition_work)?.ok_or_else(|| {
                PolyError::General("refused: Kronecker weight overflow".to_string())
            });
        };
        w = next;
    }
    // Only occupied monomials enter the dense image. The corner of the
    // enclosing degree box need not be present, especially for sparse sums.
    // The checked product of radices above bounds every encoded exponent.
    let encode = |powers: &[u32]| -> u64 {
        present
            .iter()
            .zip(&weights)
            .map(|(&v, &wt)| u64::from(powers[v]) * wt)
            .sum()
    };
    let image_degree = p.terms.keys().map(|e| encode(e)).max().unwrap_or(0);
    if image_degree > crate::factorization::MAX_FACTOR_DEGREE as u64 {
        return factor_separated_coordinates(p, present, partition_work)?.ok_or_else(|| PolyError::General(format!(
            "refused: Kronecker image degree {image_degree} exceeds the complete-factorization bound"
        )));
    }
    let t = Symbol::new("t_kronecker");
    let mut coeffs = vec![BigRational::zero(); image_degree as usize + 1];
    for (e, c) in &p.terms {
        coeffs[encode(e) as usize] = c.clone();
    }
    let image = UnivariatePoly::new(t, coeffs);
    let cf = complete_factorization(&image)?;
    let mut pieces: Vec<UnivariatePoly> = Vec::new();
    for term in &cf.factors {
        for _ in 0..term.multiplicity {
            pieces.push(term.poly.clone());
        }
    }

    let mut remaining = p.clone();
    let mut out: Vec<(MultivariatePoly, usize)> = Vec::new();
    let mut tried = 0usize;
    let mut size = 1usize;
    // In every proper factorization one side uses at most half the remaining
    // univariate factors (with multiplicity). Its mixed-radix decoding is
    // unique since a true divisor cannot exceed the source degree in any axis.
    while size <= pieces.len() / 2 && !is_constant(&remaining) {
        let mut hit: Option<(Vec<usize>, MultivariatePoly, usize)> = None;
        let mut idx: Vec<usize> = (0..size).collect();
        loop {
            tried += 1;
            if tried > MAX_RECOMBINATIONS {
                return Err(PolyError::General(
                    "refused: multivariate recombination budget exhausted".to_string(),
                ));
            }
            let mut prod = UnivariatePoly::new(image.gen_sym.clone(), vec![BigRational::one()]);
            for &i in &idx {
                prod = prod.mul(&pieces[i])?;
            }
            let mut terms = BTreeMap::new();
            for (k, c) in prod.coeffs.iter().enumerate() {
                if !c.is_zero() {
                    terms.insert(decode(k as u64, present, &radix, width), c.clone());
                }
            }
            let cand = MultivariatePoly::new(p.generators.clone(), terms)?;
            if !is_constant(&cand) {
                let (_, cand) = primitive(&cand)?;
                if let Some(mut q) = exact_div(&remaining, &cand)? {
                    let mut mult = 1usize;
                    while let Some(q2) = exact_div(&q, &cand)? {
                        q = q2;
                        mult += 1;
                    }
                    hit = Some((idx.clone(), cand, mult));
                    remaining = q;
                    break;
                }
            }
            if !advance_distinct_subset(&mut idx, &pieces) {
                break;
            }
        }
        match hit {
            Some((used, cand, mult)) => {
                // Remove the used pieces, and for extra multiplicity the
                // matching copies of the same pieces.
                let used_polys: Vec<UnivariatePoly> =
                    used.iter().map(|&i| pieces[i].clone()).collect();
                for _ in 0..mult {
                    for up in &used_polys {
                        if let Some(pos) = pieces.iter().position(|q| q == up) {
                            pieces.remove(pos);
                        }
                    }
                }
                out.push((cand, mult));
            }
            None => size += 1,
        }
    }
    if !is_constant(&remaining) {
        let (_, rest) = primitive(&remaining)?;
        out.push((rest, 1));
    }
    Ok(out)
}

/// Advance a combination of indices to the next distinct multiset. Equal
/// factors occupy consecutive positions, including after accepted factors
/// are removed. Exchanging two copies gives exactly the same candidate.
fn advance_distinct_subset(indices: &mut [usize], pieces: &[UnivariatePoly]) -> bool {
    for pivot in (0..indices.len()).rev() {
        let last = pieces.len() - (indices.len() - pivot);
        let previous = indices[pivot];
        if let Some(next) = (previous + 1..=last).find(|&i| pieces[i] != pieces[previous]) {
            for (offset, index) in indices[pivot..].iter_mut().enumerate() {
                *index = next + offset;
            }
            return true;
        }
    }
    false
}

/// Complete factorization of a multivariate polynomial over QQ.
pub fn factor_multivariate(p: &MultivariatePoly) -> Result<MultivariateFactorization, PolyError> {
    p.validate_shape()?;
    if p.is_zero() {
        return Ok(MultivariateFactorization {
            scale: BigRational::zero(),
            factors: Vec::new(),
        });
    }
    let (_, prim) = primitive(p)?;
    let width = p.generators.len();
    let mut mono = vec![u32::MAX; width];
    for e in prim.terms.keys() {
        for (m, &k) in mono.iter_mut().zip(e) {
            *m = (*m).min(k);
        }
    }
    let mut factors: Vec<(MultivariatePoly, usize)> = Vec::new();
    let mut rest_terms = BTreeMap::new();
    for (e, c) in &prim.terms {
        let e2: Vec<u32> = e.iter().zip(&mono).map(|(k, m)| k - m).collect();
        rest_terms.insert(e2, c.clone());
    }
    let rest = MultivariatePoly::new(p.generators.clone(), rest_terms)?;
    for (i, &m) in mono.iter().enumerate() {
        if m > 0 {
            factors.push((variable(p, i)?, m as usize));
        }
    }
    let mut partition_work = MAX_PARTITION_TERM_VISITS;
    for (f, m) in factor_core(&rest, &mut partition_work)? {
        if let Some(existing) = factors.iter_mut().find(|(g, _)| *g == f) {
            existing.1 += m;
        } else {
            factors.push((f, m));
        }
    }
    // Scale and independent product check.
    let mut product = MultivariatePoly::one(p.generators.clone());
    for (f, m) in &factors {
        product = product.mul(&f.pow(*m as u32)?)?;
    }
    let scale = lex_lead(p) / lex_lead(&product);
    let scaled_terms = product
        .terms
        .iter()
        .map(|(e, c)| (e.clone(), c.clone() * scale.clone()))
        .collect();
    let check = MultivariatePoly::new(p.generators.clone(), scaled_terms)?;
    if check != *p {
        return Err(PolyError::IdentityCheckFailed(
            "multivariate factorization product does not reproduce the input".to_string(),
        ));
    }
    Ok(MultivariateFactorization { scale, factors })
}

/// Independent verification lane for a multivariate factorization claim.
///
/// Checks, without reusing the generator's search: every factor is
/// non-constant, integral, primitive (coefficient gcd 1) with a positive
/// lex-leading coefficient; multiplicities are positive; no factor repeats;
/// distinct factors are not associates; and `scale * prod(f_i^m_i)` equals
/// the input exactly. Irreducibility is NOT claimed by this lane.
pub fn verify_multivariate_factorization(
    p: &MultivariatePoly,
    claim: &MultivariateFactorization,
) -> Result<(), PolyError> {
    let reject = |why: &str| {
        Err(PolyError::IdentityCheckFailed(format!(
            "factorization rejected: {why}"
        )))
    };
    if p.is_zero() {
        return if claim.scale.is_zero() && claim.factors.is_empty() {
            Ok(())
        } else {
            reject("zero polynomial must have scale 0 and no factors")
        };
    }
    let mut product = MultivariatePoly::one(p.generators.clone());
    for (idx, (f, m)) in claim.factors.iter().enumerate() {
        if f.generators != p.generators {
            return reject("factor generators differ from the input");
        }
        if *m == 0 {
            return reject("zero multiplicity");
        }
        if f.terms.keys().all(|e| e.iter().all(|&k| k == 0)) {
            return reject("constant factor");
        }
        let mut g = BigInt::zero();
        for c in f.terms.values() {
            if !c.is_integer() {
                return reject("non-integral factor coefficient");
            }
            g = gcd_int(&g, &c.to_integer());
        }
        if g != BigInt::one() {
            return reject("factor is not primitive");
        }
        if lex_lead(f).is_negative() {
            return reject("factor lex-leading coefficient is negative");
        }
        for (other, _) in claim.factors.iter().skip(idx + 1) {
            if other == f {
                return reject("repeated factor (multiplicities must be merged)");
            }
        }
        product = product.mul(&f.pow(u32::try_from(*m).map_err(|_| {
            PolyError::IdentityCheckFailed("multiplicity out of range".to_string())
        })?)?)?;
    }
    let scaled: BTreeMap<Vec<u32>, BigRational> = product
        .terms
        .iter()
        .map(|(e, c)| (e.clone(), c.clone() * claim.scale.clone()))
        .collect();
    let check = MultivariatePoly::new(p.generators.clone(), scaled)?;
    if check != *p {
        return reject("scale * product does not reproduce the input");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_core::Expr;

    fn poly(src: &str, gens: &[&str]) -> MultivariatePoly {
        let e = fsym_core::parse(src).expect("parse");
        let g: Vec<Symbol> = gens.iter().map(|s| Symbol::new(*s)).collect();
        MultivariatePoly::from_expr(&e, &g).expect("poly")
    }

    fn factor_strings(src: &str, gens: &[&str]) -> (String, Vec<(String, usize)>) {
        let f = factor_multivariate(&poly(src, gens)).expect("factor");
        let mut v: Vec<(String, usize)> = f
            .factors
            .iter()
            .map(|(p, m)| (p.to_expr().map(|e: Expr| e.to_string()).expect("expr"), *m))
            .collect();
        v.sort();
        (f.scale.to_string(), v)
    }

    #[test]
    fn factors_bivariate_non_homogeneous() {
        let (s, f) = factor_strings("x**2*y + x*y**2 + x + y", &["x", "y"]);
        assert_eq!(s, "1");
        assert_eq!(f.len(), 2);
        assert!(f.iter().all(|(_, m)| *m == 1));
    }

    #[test]
    fn sparse_image_and_repeated_image_factors_preserve_multivariate_factors() {
        // The degree-box corner has image degree 65; the actual image is
        // t^31 (1+t^2) (1+t). Choosing copies of t by position would also
        // exhaust the recombination budget before admitting the remainder.
        let gens = ["x", "unused", "y"];
        let input = poly("(x + 1)*(x**31 + y)", &gens);
        let mut work = MAX_PARTITION_TERM_VISITS;
        let found = MultivariateFactorization {
            scale: BigRational::one(),
            factors: factor_kronecker(&input, &[0, 2], &mut work).expect("sparse image factors"),
        };
        assert_eq!(found.scale, BigRational::one());
        assert_eq!(found.factors.len(), 2);
        for expected in ["x + 1", "x**31 + y"] {
            assert!(found.factors.contains(&(poly(expected, &gens), 1)));
        }
        verify_multivariate_factorization(&input, &found).expect("product verifies");
    }

    #[test]
    fn unique_subsets_match_expanded_enumeration() {
        use std::collections::BTreeSet;
        let factors: Vec<_> = [1, 1, 1, 2, 2, 3, 4, 4]
            .map(|c| {
                UnivariatePoly::new(Symbol::new("x"), vec![BigRational::from_integer(c.into())])
            })
            .into();
        for size in 1..=factors.len() {
            let mut indices: Vec<_> = (0..size).collect();
            let mut seen = BTreeSet::new();
            loop {
                let subset: Vec<_> = indices
                    .iter()
                    .map(|&i| factors[i].coeffs[0].clone())
                    .collect();
                assert!(seen.insert(subset));
                if !advance_distinct_subset(&mut indices, &factors) {
                    break;
                }
            }
            let expected = (0..1usize << factors.len())
                .filter(|mask| mask.count_ones() as usize == size)
                .map(|mask| {
                    factors
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, f)| f.coeffs[0].clone())
                        .collect::<Vec<_>>()
                })
                .collect();
            assert_eq!(seen, expected);
        }
    }

    #[test]
    fn genuine_image_and_univariate_degree_limits_still_refuse() {
        let gens = ["x", "y"];
        let oversized = poly("x**32*y + x + 1", &gens);
        let mut work = MAX_PARTITION_TERM_VISITS;
        assert!(factor_kronecker(&oversized, &[0, 1], &mut work).is_err());
        let enormous = MultivariatePoly::new(
            vec![Symbol::new("x")],
            [
                (vec![u32::MAX], BigRational::one()),
                (vec![0], BigRational::one()),
            ]
            .into(),
        )
        .unwrap();
        assert!(factor_multivariate(&enormous).is_err());
    }

    #[test]
    fn affine_coefficient_content_reaches_factors_beyond_the_image_limit() {
        for gens in [["x", "unused", "y"], ["y", "unused", "x"]] {
            let input = poly("-3/2*(2*x+1)**3*(x-2)*(x**38*y+x+1)", &gens);
            let found = factor_multivariate(&input).expect("affine content factors");
            assert_eq!(found.scale, BigRational::new((-3).into(), 2.into()));
            assert_eq!(found.factors.len(), 3);
            for (factor, multiplicity) in [("2*x+1", 3), ("x-2", 1), ("x**38*y+x+1", 1)] {
                assert!(found.factors.contains(&(poly(factor, &gens), multiplicity)));
            }
            verify_multivariate_factorization(&input, &found).expect("independent product");
        }
    }

    #[test]
    fn coprime_affine_rows_and_scalar_coefficients_remain_whole() {
        for (src, gens) in [
            ("(x**48+1)*y+(x**48+2)", vec!["x", "y"]),
            ("x*y**90*z**91+3", vec!["z", "y", "x"]),
            ("7*x+y**90*z**91-1", vec!["z", "y", "x"]),
        ] {
            let input = poly(src, &gens);
            let found = factor_multivariate(&input).unwrap();
            assert_eq!(found.scale, BigRational::one());
            assert_eq!(found.factors, vec![(input.clone(), 1)]);
            verify_multivariate_factorization(&input, &found).unwrap();
        }
    }

    #[test]
    fn independent_parameter_blocks_factor_when_the_dense_image_exceeds_its_limit() {
        let gens = ["w", "x", "y", "z", "unused"];
        let input = poly("(-7/3)*(w+1)**2*(x**2+1)*(y**2+2)*(z**2+z+1)", &gens);
        let found = factor_multivariate(&input).unwrap();
        assert_eq!(found.scale, BigRational::new((-7).into(), 3.into()));
        for (factor, multiplicity) in [("w+1", 2), ("x**2+1", 1), ("y**2+2", 1), ("z**2+z+1", 1)] {
            assert!(found.factors.contains(&(poly(factor, &gens), multiplicity)));
        }
        assert_eq!(found.factors.len(), 4);
        verify_multivariate_factorization(&input, &found).unwrap();
    }

    #[test]
    fn factors_disjoint_multivariate_blocks_in_interleaved_ambient_axes() {
        for gens in [
            ["x", "z", "y", "unused", "w"],
            ["w", "x", "unused", "y", "z"],
        ] {
            let p = poly("(1+x**2*y**2)*(1+z**2*w**2)", &gens);
            let found = factor_multivariate(&p).unwrap();
            assert_eq!(found.scale, BigRational::one());
            assert_eq!(found.factors.len(), 2);
            for factor in ["1+x**2*y**2", "1+z**2*w**2"] {
                assert!(found.factors.contains(&(poly(factor, &gens), 1)));
            }
            verify_multivariate_factorization(&p, &found).unwrap();
        }
    }

    #[test]
    fn grouped_factors_keep_rational_scale_multiplicity_and_nested_splits() {
        let gens = ["z", "y", "w", "x", "v", "u"];
        let p = poly("(-5/7)*(1+x*y)**3*(1+z*w)**2*(1+u**2*v**2)", &gens);
        let found = factor_multivariate(&p).unwrap();
        assert_eq!(found.scale, BigRational::new((-5).into(), 7.into()));
        assert_eq!(found.factors.len(), 3);
        for (expr, m) in [("1+x*y", 3), ("1+z*w", 2), ("1+u**2*v**2", 1)] {
            assert!(found.factors.contains(&(poly(expr, &gens), m)));
        }
        verify_multivariate_factorization(&p, &found).unwrap();
    }

    #[test]
    fn grouped_split_checks_missing_support_and_coefficients_outside_pivot() {
        let gens = ["x", "z", "y", "w"];
        for src in [
            "1+x**2*y**2+x**2*y**2*z**2*w**2",
            "1+x**2*y**2+z**2*w**2+2*x**2*y**2*z**2*w**2",
        ] {
            let p = poly(src, &gens);
            let mut work = MAX_PARTITION_TERM_VISITS;
            assert!(
                factor_separated_coordinates(&p, &[0, 1, 2, 3], &mut work)
                    .unwrap()
                    .is_none()
            );
            assert!(factor_multivariate(&p).is_err());
        }
    }

    #[test]
    fn grouped_partition_allowance_is_shared_and_refuses_before_publication() {
        let gens = ["x", "z", "u", "y", "w", "v"];
        let p = poly("(1+x**2*y**2)*(1+z**2*w**2)*(1+u**2*v**2)", &gens);
        let mut full = MAX_PARTITION_TERM_VISITS;
        let expected = factor_core(&p, &mut full).unwrap();
        let cost = MAX_PARTITION_TERM_VISITS - full;
        assert!(cost > 2 * p.terms.len());
        let mut exact = cost;
        assert_eq!(factor_core(&p, &mut exact).unwrap(), expected);
        assert_eq!(exact, 0);
        let mut short = cost - 1;
        let err = factor_core(&p, &mut short).unwrap_err();
        assert!(err.to_string().contains("partition term-work"));
        let mut singleton_work = 0;
        let singleton = poly("(1+x**2)*(1+y**2)*(1+z**2)*(1+w**2)", &gens);
        assert_eq!(
            factor_core(&singleton, &mut singleton_work).unwrap().len(),
            4
        );
    }

    #[test]
    fn sparse_separation_checks_nonpivot_coefficients_and_missing_monomials() {
        let gens = ["x", "y"];
        for source in [
            "(x**20+1)*(y**20+1)+x**2*y**2",
            "x**20*y**20+x**20+y**20+2",
            "x**20*y**20+x**20+y**20",
        ] {
            let p = poly(source, &gens);
            let mut work = MAX_PARTITION_TERM_VISITS;
            assert!(
                factor_separated_coordinates(&p, &[0, 1], &mut work)
                    .unwrap()
                    .is_none()
            );
            assert!(factor_multivariate(&p).is_err());
        }
    }

    #[test]
    fn affine_content_matches_constructed_coprime_rows() {
        let gens = ["x", "y"];
        // A and k*A+1 are coprime for every k, independently of the GCD algorithm.
        for degree in [2, 7, 18, 33] {
            for k in [-3, 1, 4] {
                let residual = format!("(x**{degree}+2)*y+({k})*(x**{degree}+2)+1");
                let input = poly(&format!("(x-1)**2*({residual})"), &gens);
                let found = factor_multivariate(&input).unwrap();
                assert_eq!(found.factors.len(), 2);
                assert!(found.factors.contains(&(poly("x-1", &gens), 2)));
                assert!(found.factors.contains(&(poly(&residual, &gens), 1)));
                verify_multivariate_factorization(&input, &found).unwrap();
            }
        }
    }

    #[test]
    fn factors_with_content_and_multiplicity() {
        let (s, f) = factor_strings("6*x**2*z + 24*x*y*z + 24*y**2*z", &["x", "y", "z"]);
        assert_eq!(s, "6");
        assert_eq!(f.len(), 2);
        assert!(f.iter().any(|(_, m)| *m == 2));
    }

    #[test]
    fn irreducible_input_stays_whole() {
        let (_, f) = factor_strings("x**2 + y**2 + 1", &["x", "y"]);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn three_variable_difference_of_squares() {
        let (_, f) = factor_strings("x**2 + 2*x*y + y**2 - z**2", &["x", "y", "z"]);
        assert_eq!(f.len(), 2);
    }

    fn claim_of(src: &str, gens: &[&str]) -> (MultivariatePoly, MultivariateFactorization) {
        let p = poly(src, gens);
        let f = factor_multivariate(&p).expect("factor");
        (p, f)
    }

    #[test]
    fn verifier_accepts_generated_factorizations() {
        for (src, gens) in [
            ("x**2*y + x*y**2 + x + y", vec!["x", "y"]),
            ("6*x**2*z + 24*x*y*z + 24*y**2*z", vec!["x", "y", "z"]),
            ("x**6 - y**6", vec!["x", "y"]),
            ("x**3 + y**3 + z**3 - 3*x*y*z", vec!["x", "y", "z"]),
        ] {
            let (p, f) = claim_of(src, &gens);
            verify_multivariate_factorization(&p, &f).expect("generated claim verifies");
        }
    }

    #[test]
    fn verifier_rejects_planted_wrong_claims() {
        let (p, f) = claim_of("x**2*y + x*y**2 + x + y", &["x", "y"]);
        // wrong scale
        let mut bad = f.clone();
        bad.scale = bad.scale.clone() * BigRational::from_integer(BigInt::from(2));
        assert!(verify_multivariate_factorization(&p, &bad).is_err());
        // wrong multiplicity
        let mut bad = f.clone();
        bad.factors[0].1 += 1;
        assert!(verify_multivariate_factorization(&p, &bad).is_err());
        // perturbed factor coefficient
        let mut bad = f.clone();
        let key = bad.factors[0].0.terms.keys().next().cloned().expect("term");
        let c = bad.factors[0].0.terms[&key].clone();
        bad.factors[0].0.terms.insert(key, c + BigRational::one());
        assert!(verify_multivariate_factorization(&p, &bad).is_err());
        // non-primitive factor with compensating scale (product still equal)
        let mut bad = f.clone();
        let doubled: BTreeMap<Vec<u32>, BigRational> = bad.factors[0]
            .0
            .terms
            .iter()
            .map(|(e, c)| {
                (
                    e.clone(),
                    c.clone() * BigRational::from_integer(BigInt::from(2)),
                )
            })
            .collect();
        bad.factors[0].0 =
            MultivariatePoly::new(bad.factors[0].0.generators.clone(), doubled).expect("poly");
        bad.scale = bad.scale.clone() / BigRational::from_integer(BigInt::from(2));
        assert!(verify_multivariate_factorization(&p, &bad).is_err());
        // dropped factor
        let mut bad = f.clone();
        bad.factors.pop();
        assert!(verify_multivariate_factorization(&p, &bad).is_err());
    }
}
