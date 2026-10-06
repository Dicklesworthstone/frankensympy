//! Multivariate factorization over QQ (WS09).
//!
//! `p = scale * prod(f_i ^ m_i)` with primitive integer factors whose
//! lex-leading coefficient is positive. The pipeline:
//!
//! 1. numeric content and monomial content come out first;
//! 2. a univariate remainder goes to [`complete_factorization`];
//! 3. a homogeneous remainder is dehomogenized in its last variable,
//!    factored, and every factor rehomogenized (a bijection on factors once
//!    the monomial content is gone);
//! 4. otherwise Kronecker substitution `x_{i+1} -> t^(B_i)` maps the
//!    polynomial to a univariate one; its irreducible factors are
//!    recombined and every candidate is mapped back and accepted only after
//!    exact multivariate division of the remaining polynomial.
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
    let mut coeffs = vec![BigRational::zero(); deg + 1];
    for (e, c) in &p.terms {
        coeffs[e[var] as usize] += c.clone();
    }
    let up = UnivariatePoly::new(sym, coeffs);
    let cf = complete_factorization(&up)?;
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
fn factor_core(p: &MultivariatePoly) -> Result<Vec<(MultivariatePoly, usize)>, PolyError> {
    let present = present_vars(p);
    match present.len() {
        0 => return Ok(Vec::new()),
        1 => return factor_univariate(p, present[0]),
        _ => {}
    }
    if is_homogeneous(p) {
        return factor_homogeneous(p, *present.last().expect("two or more variables"));
    }
    factor_kronecker(p, &present)
}

fn factor_homogeneous(
    p: &MultivariatePoly,
    v: usize,
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
    for (f, m) in factor_core(&dehom)? {
        let d = f
            .terms
            .keys()
            .map(|e| e.iter().sum::<u32>())
            .max()
            .unwrap_or(0);
        let mut terms = BTreeMap::new();
        for (e, c) in &f.terms {
            let mut e2 = e.clone();
            e2[v] = d - e.iter().sum::<u32>();
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

fn factor_kronecker(
    p: &MultivariatePoly,
    present: &[usize],
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
        w = w
            .checked_mul(*r)
            .ok_or_else(|| PolyError::General("refused: Kronecker weight overflow".to_string()))?;
    }
    let image_degree: u64 = present
        .iter()
        .zip(&weights)
        .map(|(&v, &wt)| u64::from(p.degree_in(v)) * wt)
        .sum();
    if image_degree as usize > crate::factorization::MAX_FACTOR_DEGREE {
        return Err(PolyError::General(format!(
            "refused: Kronecker image degree {image_degree} exceeds the complete-factorization bound"
        )));
    }
    let t = Symbol::new("t_kronecker");
    let mut coeffs = vec![BigRational::zero(); image_degree as usize + 1];
    for (e, c) in &p.terms {
        let k: u64 = present
            .iter()
            .zip(&weights)
            .map(|(&v, &wt)| u64::from(e[v]) * wt)
            .sum();
        coeffs[k as usize] += c.clone();
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
    while size <= pieces.len() && !is_constant(&remaining) {
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
            // Next combination of `size` indices.
            let n = pieces.len();
            let mut j = size;
            while j > 0 && idx[j - 1] == n - size + j - 1 {
                j -= 1;
            }
            if j == 0 {
                break;
            }
            idx[j - 1] += 1;
            for k in j..size {
                idx[k] = idx[k - 1] + 1;
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

/// Complete factorization of a multivariate polynomial over QQ.
pub fn factor_multivariate(p: &MultivariatePoly) -> Result<MultivariateFactorization, PolyError> {
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
    for (f, m) in factor_core(&rest)? {
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
}
