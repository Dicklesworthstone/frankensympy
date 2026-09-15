//! Polynomial factorization and square-free decomposition (WS09).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::univariate::UnivariatePoly;
use fsym_core::{BigInt, BigRational, Symbol};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};

/// Factor with multiplicity: $(f(x), k)$.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactorTerm {
    pub poly: UnivariatePoly,
    pub multiplicity: usize,
}

/// Exact product decomposition: $P(x) = \text{scale} \cdot \prod f_i(x)^{e_i}$.
///
/// The factors are not implicitly irreducible. Callers must not promote this representation to a
/// complete factorization without separate irreducibility evidence for every factor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactorizationResult {
    pub scale: BigRational,
    pub factors: Vec<FactorTerm>,
}

impl FactorizationResult {
    /// Reconstructs the expanded product polynomial from the factored representation.
    pub fn expand(&self, sym: Symbol) -> Result<UnivariatePoly, PolyError> {
        let mut prod = UnivariatePoly::new(sym.clone(), vec![self.scale.clone()]);
        for factor in &self.factors {
            factor.poly.validate_shape()?;
            if factor.poly.gen_sym != sym {
                return Err(PolyError::IncompatibleGenerators(
                    sym.name.clone(),
                    factor.poly.gen_sym.name.clone(),
                ));
            }
            if factor.multiplicity == 0 {
                return Err(PolyError::IdentityCheckFailed(
                    "factor multiplicity must be positive".to_string(),
                ));
            }
            let exponent = u32::try_from(factor.multiplicity).map_err(|_| {
                PolyError::IdentityCheckFailed(format!(
                    "factor multiplicity {} exceeds the supported exponent range",
                    factor.multiplicity
                ))
            })?;
            let factor_pow = factor.poly.pow(exponent)?;
            prod = prod.mul(&factor_pow)?;
        }
        Ok(prod)
    }
}

/// Computes the square-free decomposition of a univariate polynomial using Yun's algorithm:
/// $P(x) = c \cdot f_1^1 \cdot f_2^2 \cdots f_k^k$ where each $f_i$ is square-free and pairwise coprime.
pub fn square_free_decomposition(poly: &UnivariatePoly) -> Result<FactorizationResult, PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        return Ok(FactorizationResult {
            scale: BigRational::zero(),
            factors: Vec::new(),
        });
    }

    let lc = poly.leading_coeff().clone();
    let monic_p = poly.to_monic();
    if monic_p.degree() == Some(0) {
        return Ok(FactorizationResult {
            scale: lc,
            factors: Vec::new(),
        });
    }

    let p_prime = monic_p.derivative();
    let c = monic_p.gcd(&p_prime)?;

    if c.degree() == Some(0) {
        // Polynomial is already square-free
        return Ok(FactorizationResult {
            scale: lc,
            factors: vec![FactorTerm {
                poly: monic_p,
                multiplicity: 1,
            }],
        });
    }

    let (mut w, _) = monic_p.div_rem(&c)?;
    let (mut y, _) = p_prime.div_rem(&c)?;

    let mut factors = Vec::new();
    let mut i = 1;
    let max_iterations = poly.degree().unwrap_or(0);

    while !w.is_one() {
        if i > max_iterations || w.is_zero() {
            return Err(PolyError::General(format!(
                "Yun square-free decomposition exceeded degree iteration bound {max_iterations}"
            )));
        }
        let y_sub_w_prime = y.sub(&w.derivative())?;
        let a_i = w.gcd(&y_sub_w_prime)?;

        if !a_i.is_one() {
            factors.push(FactorTerm {
                poly: a_i.clone(),
                multiplicity: i,
            });
        }

        let (next_w, _) = w.div_rem(&a_i)?;
        let (next_y, _) = y_sub_w_prime.div_rem(&a_i)?;

        w = next_w;
        y = next_y;
        i += 1;
    }

    Ok(FactorizationResult { scale: lc, factors })
}

/// Checks an exact square-free product-decomposition witness.
///
/// Acceptance criteria:
/// 1. The product of factors $\text{scale} \cdot \prod f_i(x)^{e_i}$ equals $P(x)$ exactly.
/// 2. Each factor $f_i(x)$ is monic and square-free ($\gcd(f_i, f_i') = 1$).
/// 3. All factors $f_i(x), f_j(x)$ are pairwise coprime ($\gcd(f_i, f_j) = 1$ for $i \neq j$).
///
/// Acceptance does not establish that any factor is irreducible or that the decomposition is a
/// complete factorization into irreducibles.
pub fn verify_square_free_product_decomposition(
    poly: &UnivariatePoly,
    factorization: &FactorizationResult,
) -> Result<(), PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        if !factorization.scale.is_zero() || !factorization.factors.is_empty() {
            return Err(PolyError::IdentityCheckFailed(
                "the zero polynomial requires canonical scale zero with no factors".to_string(),
            ));
        }
    } else if factorization.scale.is_zero() {
        return Err(PolyError::IdentityCheckFailed(
            "a nonzero polynomial cannot have factorization scale zero".to_string(),
        ));
    }

    // Admit the certificate shape before exponentiation. This prevents a
    // malformed multiplicity from triggering disproportionate dense work and
    // enforces the canonical monic, positive-degree factor convention.
    let target_degree = poly.degree().unwrap_or(0);
    let mut reconstructed_degree = 0usize;
    for factor in &factorization.factors {
        factor.poly.validate_shape()?;
        if factor.poly.gen_sym != poly.gen_sym {
            return Err(PolyError::IncompatibleGenerators(
                poly.gen_sym.name.clone(),
                factor.poly.gen_sym.name.clone(),
            ));
        }
        if factor.multiplicity == 0 {
            return Err(PolyError::IdentityCheckFailed(
                "factor multiplicity must be positive".to_string(),
            ));
        }
        u32::try_from(factor.multiplicity).map_err(|_| {
            PolyError::IdentityCheckFailed(format!(
                "factor multiplicity {} exceeds the supported exponent range",
                factor.multiplicity
            ))
        })?;
        let factor_degree = factor.poly.degree().ok_or_else(|| {
            PolyError::IdentityCheckFailed("zero polynomial cannot be a factor".to_string())
        })?;
        if factor_degree == 0 {
            return Err(PolyError::IdentityCheckFailed(
                "constant factors must be absorbed into the scale".to_string(),
            ));
        }
        if !factor.poly.leading_coeff().is_one() {
            return Err(PolyError::IdentityCheckFailed(format!(
                "factor `{}` is not monic",
                factor.poly
            )));
        }
        let degree_contribution =
            factor_degree
                .checked_mul(factor.multiplicity)
                .ok_or_else(|| {
                    PolyError::IdentityCheckFailed(
                        "factorization degree calculation overflowed".to_string(),
                    )
                })?;
        reconstructed_degree = reconstructed_degree
            .checked_add(degree_contribution)
            .ok_or_else(|| {
                PolyError::IdentityCheckFailed(
                    "factorization degree calculation overflowed".to_string(),
                )
            })?;
        if reconstructed_degree > target_degree {
            return Err(PolyError::IdentityCheckFailed(format!(
                "factorization degree {reconstructed_degree} exceeds polynomial degree {target_degree}"
            )));
        }
    }
    if reconstructed_degree != target_degree {
        return Err(PolyError::IdentityCheckFailed(format!(
            "factorization degree {reconstructed_degree} does not match polynomial degree {target_degree}"
        )));
    }

    // 1. Reconstruct product and check equality
    let reconstructed = factorization.expand(poly.gen_sym.clone())?;
    if &reconstructed != poly {
        return Err(PolyError::IdentityCheckFailed(format!(
            "Factorization product `{reconstructed}` does not match original polynomial `{poly}`"
        )));
    }

    // 2. Verify square-freeness of each factor
    for factor in &factorization.factors {
        let f_prime = factor.poly.derivative();
        let g = factor.poly.gcd(&f_prime)?;
        if !g.is_one() {
            return Err(PolyError::IdentityCheckFailed(format!(
                "Factor `{}` is not square-free: gcd with derivative is `{g}`",
                factor.poly
            )));
        }
    }

    // 3. Verify pairwise coprimality
    for (i, f_i) in factorization.factors.iter().enumerate() {
        for (_j, f_j) in factorization.factors.iter().enumerate().skip(i + 1) {
            let g = f_i.poly.gcd(&f_j.poly)?;
            if !g.is_one() {
                return Err(PolyError::IdentityCheckFailed(format!(
                    "Factors `{}` and `{}` are not coprime: gcd is `{g}`",
                    f_i.poly, f_j.poly
                )));
            }
        }
    }

    Ok(())
}

fn bounded_integer_divisors(n: &BigInt, max_trial_divisors: usize) -> Vec<BigInt> {
    let abs_n = if n < &BigInt::zero() {
        -n.clone()
    } else {
        n.clone()
    };
    if abs_n.is_zero() {
        return Vec::new();
    }
    if abs_n.is_one() {
        return vec![BigInt::one()];
    }
    if let Ok(val) = u64::try_from(abs_n.clone()) {
        let mut divs = Vec::new();
        let mut d = 1u64;
        let mut trials = 0usize;
        while d <= val / d && trials < max_trial_divisors {
            if val % d == 0 {
                divs.push(BigInt::from(d));
                if d * d != val {
                    divs.push(BigInt::from(val / d));
                }
            }
            d += 1;
            trials += 1;
        }
        divs.sort();
        divs
    } else {
        vec![BigInt::one(), abs_n]
    }
}

fn find_bounded_rational_roots(poly: &UnivariatePoly) -> Result<Vec<BigRational>, PolyError> {
    if poly.degree() == Some(0) || poly.is_zero() {
        return Ok(Vec::new());
    }
    let mut roots = Vec::new();
    let mut current = poly.clone();
    while current.coeffs.first().is_some_and(Zero::is_zero) && current.degree() > Some(0) {
        if !roots.contains(&BigRational::zero()) {
            roots.push(BigRational::zero());
        }
        let monomial = UnivariatePoly::monomial(current.gen_sym.clone(), BigRational::one(), 1)?;
        let (quotient, remainder) = current.div_rem(&monomial)?;
        if !remainder.is_zero() {
            return Err(PolyError::IdentityCheckFailed(
                "zero constant coefficient did not divide exactly by the generator".to_string(),
            ));
        }
        current = quotient;
    }
    if current.degree() == Some(0) {
        return Ok(roots);
    }
    let mut denom_lcm = BigInt::one();
    for c in &current.coeffs {
        let d = c.denom();
        let gcd_d = denom_lcm.gcd(d);
        if !gcd_d.is_zero() {
            denom_lcm = (&denom_lcm * d) / gcd_d;
        }
    }
    let mut int_coeffs: Vec<BigInt> = current
        .coeffs
        .iter()
        .map(|c| (c * BigRational::from_integer(denom_lcm.clone())).to_integer())
        .collect();
    while int_coeffs.len() > 1 && int_coeffs.last().is_some_and(|c| c.is_zero()) {
        int_coeffs.pop();
    }
    if int_coeffs.len() <= 1 {
        return Ok(roots);
    }
    let (Some(a0), Some(an)) = (int_coeffs.first(), int_coeffs.last()) else {
        return Ok(roots);
    };
    let p_divs = bounded_integer_divisors(a0, 500);
    let q_divs = bounded_integer_divisors(an, 100);

    for p in &p_divs {
        for q in &q_divs {
            if q.is_zero() {
                continue;
            }
            for sign in &[1i64, -1i64] {
                let candidate_p = if *sign == -1 { -p.clone() } else { p.clone() };
                let candidate = BigRational::new(candidate_p, q.clone());
                let val = current.eval(&candidate);
                if val.is_zero() && !roots.contains(&candidate) {
                    roots.push(candidate);
                }
            }
        }
    }
    Ok(roots)
}

fn split_bounded_rational_roots(poly: &UnivariatePoly) -> Result<Vec<UnivariatePoly>, PolyError> {
    poly.validate_shape()?;
    if poly.degree() == Some(0) || poly.is_zero() {
        return Ok(Vec::new());
    }
    let mut factors = Vec::new();
    let mut rem = poly.clone();
    let roots = find_bounded_rational_roots(&rem)?;
    for r in roots {
        let linear = UnivariatePoly::new(rem.gen_sym.clone(), vec![-r, BigRational::one()]);
        loop {
            let (q, remainder) = rem.div_rem(&linear)?;
            if remainder.is_zero() {
                factors.push(linear.clone());
                rem = q;
                if rem.degree() == Some(0) {
                    break;
                }
            } else {
                break;
            }
        }
    }
    if rem.degree() > Some(0) {
        if rem.degree() == Some(2) {
            let b = rem.coeffs.get(1).ok_or_else(|| {
                PolyError::General(
                    "quadratic decomposition is missing its linear coefficient".to_string(),
                )
            })?;
            let c = rem.coeffs.first().ok_or_else(|| {
                PolyError::General(
                    "quadratic decomposition is missing its constant coefficient".to_string(),
                )
            })?;
            let four = BigRational::from_integer(BigInt::from(4));
            let discr = b * b - four * c;
            if discr >= BigRational::zero()
                && let (Some(num_sqrt), Some(den_sqrt)) =
                    (discr.numer().sqrt(), discr.denom().sqrt())
                && &num_sqrt * &num_sqrt == *discr.numer()
                && &den_sqrt * &den_sqrt == *discr.denom()
            {
                let d = BigRational::new(num_sqrt, den_sqrt);
                let two = BigRational::from_integer(BigInt::from(2));
                let r1 = (-b + &d) / &two;
                let r2 = (-b - &d) / &two;
                factors.push(UnivariatePoly::new(
                    rem.gen_sym.clone(),
                    vec![-r1, BigRational::one()],
                ));
                factors.push(UnivariatePoly::new(
                    rem.gen_sym.clone(),
                    vec![-r2, BigRational::one()],
                ));
                return Ok(factors);
            }
        }
        if rem != UnivariatePoly::one(rem.gen_sym.clone()) {
            factors.push(rem);
        }
    }
    Ok(factors)
}

/// Computes a bounded rational-root refinement of a square-free decomposition over
/// $\mathbb{Q}[x]$.
///
/// The generator extracts rational linear factors found by its bounded divisor search and splits
/// a remaining quadratic when its discriminant is a rational square. Any remaining square-free
/// component is preserved as one factor. The result is therefore an exact product decomposition,
/// but it is not a complete factorization into irreducibles.
pub fn bounded_rational_root_decomposition(
    poly: &UnivariatePoly,
) -> Result<FactorizationResult, PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        return Ok(FactorizationResult {
            scale: BigRational::zero(),
            factors: Vec::new(),
        });
    }
    let sqf = square_free_decomposition(poly)?;
    let mut factors_vec: Vec<FactorTerm> = Vec::new();

    for sqf_term in sqf.factors {
        let components = split_bounded_rational_roots(&sqf_term.poly)?;
        for component in components {
            if let Some(existing) = factors_vec.iter_mut().find(|f| f.poly == component) {
                existing.multiplicity += sqf_term.multiplicity;
            } else {
                factors_vec.push(FactorTerm {
                    poly: component,
                    multiplicity: sqf_term.multiplicity,
                });
            }
        }
    }

    let res = FactorizationResult {
        scale: sqf.scale,
        factors: factors_vec,
    };
    verify_square_free_product_decomposition(poly, &res)?;
    Ok(res)
}

// ============================================================================
// Bounded complete factorization over ZZ with irreducibility evidence
// (WS09, bead fra-rc-factor-lt4).
//
// Declared regime — outside it the entry points return a typed refusal and
// never present a partial factorization as complete:
//   * square-free parts of degree <= MAX_FACTOR_DEGREE,
//   * primitive integer coefficient height <= MAX_FACTOR_COEFF_BITS,
//   * at most MAX_MODP_FACTORS irreducible factors over the lifting prime.
// ============================================================================

/// Upper bound on the degree of a square-free part admitted to the modular
/// factoring regime.
pub const MAX_FACTOR_DEGREE: usize = 64;
/// Upper bound on the bit length of any primitive integer coefficient.
pub const MAX_FACTOR_COEFF_BITS: usize = 2048;
/// Upper bound on the number of irreducible factors over the lifting prime.
const MAX_MODP_FACTORS: usize = 10;
/// Prime candidates tried per irreducibility witness before the claim is
/// recorded as unproven (the factorization itself stays exact).
const PRIME_ATTEMPTS: usize = 24;
/// Deterministic Cantor-Zassenhaus seed attempts per equal-degree class.
const EDF_SEED_BUDGET: u32 = 64;

/// Independent certificate that `poly` is irreducible over ZZ: `poly` reduced
/// modulo `prime` is irreducible over GF(prime), which excludes every
/// nontrivial integer factorization (Gauss). The prime is re-verified by the
/// independent verifier before any certificate is trusted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IrreducibilityWitness {
    pub prime: BigInt,
}

/// One factor of a complete factorization, carrying its optional
/// irreducibility certificate. `irreducibility: None` records the weaker
/// (still exact) decomposition claim required where no witness was found
/// inside the declared prime budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompleteFactorTerm {
    pub poly: UnivariatePoly,
    pub multiplicity: usize,
    pub irreducibility: Option<IrreducibilityWitness>,
}

/// Exact complete factorization over QQ/ZZ:
/// `scale * prod(poly_i ^ multiplicity_i)` with square-free, pairwise coprime
/// monic factors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompleteFactorization {
    pub scale: BigRational,
    pub factors: Vec<CompleteFactorTerm>,
}

// ---------------------------------------------------------------------------
// GF(p) polynomial layer (ascending dense coefficients, p < 2^31).
// ---------------------------------------------------------------------------

fn fp_trim(v: &mut Vec<u64>) {
    while v.last().is_some_and(|c| *c == 0) {
        v.pop();
    }
}

fn fp_norm(mut v: Vec<u64>, p: u64) -> Vec<u64> {
    for c in &mut v {
        *c %= p;
    }
    fp_trim(&mut v);
    v
}

fn fp_add(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let mut out = vec![0u64; a.len().max(b.len())];
    for (i, c) in a.iter().enumerate() {
        out[i] = *c;
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = (out[i] + c) % p;
    }
    fp_trim(&mut out);
    out
}

fn fp_sub(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let mut out = vec![0u64; a.len().max(b.len())];
    for (i, c) in a.iter().enumerate() {
        out[i] = *c;
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = (out[i] + p - c % p) % p;
    }
    fp_trim(&mut out);
    out
}

fn fp_mul(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u64; a.len() + b.len() - 1];
    for (i, left) in a.iter().enumerate() {
        if *left == 0 {
            continue;
        }
        for (j, right) in b.iter().enumerate() {
            out[i + j] = (out[i + j] + (*left as u128) * (*right as u128) % (p as u128)) as u64;
        }
    }
    fp_trim(&mut out);
    out
}

/// Remainder of `a` modulo the monic polynomial `f` over GF(p).
fn fp_rem_monic(a: &[u64], f: &[u64], p: u64) -> Vec<u64> {
    let mut out = a.to_vec();
    let flen = f.len();
    if flen == 0 {
        return out;
    }
    while out.len() >= flen {
        let shift = out.len() - flen;
        let lead = out[out.len() - 1];
        if lead == 0 {
            out.pop();
            continue;
        }
        for (k, fc) in f.iter().enumerate() {
            let idx = shift + k;
            out[idx] = (out[idx] + p - (lead * fc % p)) % p;
        }
        debug_assert_eq!(out[out.len() - 1], 0);
        out.pop();
    }
    fp_trim(&mut out);
    out
}

fn fp_pow_mod(base: &[u64], mut exp: u64, f: &[u64], p: u64) -> Vec<u64> {
    let one = vec![1u64];
    let mut result = fp_rem_monic(&one, f, p);
    let mut squares = fp_rem_monic(base, f, p).to_vec();
    while exp > 0 {
        if exp & 1 == 1 {
            result = fp_rem_monic(&fp_mul(&result, &squares, p), f, p);
        }
        squares = fp_rem_monic(&fp_mul(&squares, &squares, p), f, p);
        exp >>= 1;
    }
    result
}

fn fp_gcd(mut a: Vec<u64>, mut b: Vec<u64>, p: u64) -> Vec<u64> {
    while !b.is_empty() {
        let r = fp_rem_monic(&a, &b, p);
        a = b;
        b = r;
    }
    // Normalize the leading coefficient to 1.
    if let Some(last) = a.last().copied() {
        if last != 1 {
            let inv = fp_scalar_inverse(last, p);
            for c in &mut a {
                *c = *c * inv % p;
            }
        }
    }
    a
}

fn fp_scalar_inverse(a: u64, p: u64) -> u64 {
    debug_assert!(a % p != 0, "inverse of zero over a prime field");
    fp_scalar_pow(a, p - 2, p)
}

fn fp_scalar_pow(mut base: u64, mut exp: u64, p: u64) -> u64 {
    base %= p;
    let mut acc = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = (acc as u128 * base as u128 % p as u128) as u64;
        }
        base = (base as u128 * base as u128 % p as u128) as u64;
        exp >>= 1;
    }
    acc
}

fn fp_derivative(a: &[u64], p: u64) -> Vec<u64> {
    if a.len() <= 1 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(a.len() - 1);
    for (degree, c) in a.iter().enumerate().skip(1) {
        out.push((*c as u128 * (degree as u64) % (p as u128)) as u64);
    }
    fp_trim(&mut out);
    out
}

fn fp_is_squarefree(f: &[u64], p: u64) -> bool {
    if f.len() <= 2 {
        return true; // constants and nonconstant linear polys are square-free
    }
    fp_gcd(f.to_vec(), fp_derivative(f, p), p).len() <= 1
}

/// Rabin irreducibility test over GF(p): `x^(p^d) == x (mod f)` and
/// `gcd(x^(p^(d/ell)) - x, f) == 1` for every prime divisor `ell` of `d`.
fn fp_rabin_irreducible(f: &[u64], p: u64) -> bool {
    let d = match f.len().checked_sub(1) {
        Some(d) if d >= 1 => d,
        _ => return false,
    };
    // Frobenius chain: x^(p^k) mod f for k = 0..=d.
    let one = vec![1u64];
    let x = if d == 1 { fp_rem_monic(&[0, 1], f, p) } else { vec![0u64, 1] };
    let mut frob = vec![0u64, 1];
    let mut powers: Vec<Vec<u64>> = Vec::with_capacity(d + 1);
    powers.push(fp_rem_monic(&one, f, p));
    let mut xp = fp_rem_monic(&x, f, p);
    powers.push(xp.clone());
    for _ in 1..=d {
        frob = fp_pow_mod(&frob, p, f, p);
        powers.push(frob.clone());
    }
    // powers[k] = x^(p^k) mod f; x^(p^d) must equal x.
    if frob != xp {
        return false;
    }
    for ell in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61] {
        if ell > d || d % ell != 0 {
            continue;
        }
        let sub = powers[(d / ell) as usize].clone();
        let diff = fp_sub(&sub, &[0, 1], p);
        let g = fp_gcd(diff, f.to_vec(), p);
        if g.len() > 1 {
            return false;
        }
    }
    true
}

/// Distinct-degree factorization: returns `(degree, product)` pairs whose
/// products multiply to `f`, each product a monic product of irreducible
/// polynomials of exactly that degree.
fn fp_distinct_degree(
    f: &[u64],
    p: u64,
) -> Vec<(usize, Vec<u64>)> {
    let mut remaining = f.to_vec();
    let mut x = vec![0u64, 1];
    let mut frob = x.clone();
    let mut classes: Vec<(usize, Vec<u64>)> = Vec::new();
    let mut degree = 0usize;
    while remaining.len() > 2 {
        degree += 1;
        frob = fp_pow_mod(&frob, p, &remaining, p);
        let diff = fp_sub(&frob, &x, p);
        let g = fp_gcd(diff, remaining.clone(), p);
        if g.len() > 1 {
            remaining = fp_div_monic(&remaining, &g, p);
            frob = fp_rem_monic(&frob, &remaining, p);
        }
        if degree * 2 > remaining.len() - 1 {
            break;
        }
    }
    if remaining.len() > 1 {
        classes.push((remaining.len() - 1, remaining));
    }
    classes
}

/// Exact division of `a` by the monic polynomial `b` over GF(p).
fn fp_div_monic(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let mut q = vec![0u64; a.len().saturating_sub(b.len()) + 1];
    let mut rem = a.to_vec();
    while rem.len() >= b.len() && !rem.is_empty() {
        let shift = rem.len() - b.len();
        let lead = rem[rem.len() - 1];
        if lead == 0 {
            rem.pop();
            continue;
        }
        q[shift] = lead;
        for (k, bc) in b.iter().enumerate() {
            let idx = shift + k;
            rem[idx] = (rem[idx] + p - (lead * bc % p)) % p;
        }
        rem.pop();
    }
    fp_trim(&mut q);
    q
}

/// Cantor-Zassenhaus equal-degree splitting with a deterministic seed.
/// `h` is a monic product of irreducible polynomials each of degree
/// `class_degree`; returns the two coprime split halves or `None` when the
/// seed budget is exhausted for this seed stream.
fn fp_equal_degree_split(
    h: &[u64],
    class_degree: usize,
    p: u64,
    seed: u64,
) -> Option<(Vec<u64>, Vec<u64>)> {
    let total_degree = h.len() - 1;
    if total_degree == class_degree {
        return Some((h.to_vec(), vec![1u64]));
    }
    // Deterministic LCG; the sequence is part of the pinned behavior.
    let mut state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
        | 1;
    let mut draw = |bound: usize, state: &mut u64| -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33) % (bound as u64)
    };
    let half_scalar = (p - 1) / 2;
    for _ in 0..EDF_SEED_BUDGET {
        // Random nonconstant polynomial of degree < deg(h).
        let mut r = vec![0u64; total_degree];
        for slot in r.iter_mut().take(total_degree) {
            *slot = draw(p as usize, &mut state);
        }
        fp_trim(&mut r);
        if r.is_empty() {
            continue;
        }
        // A = prod_{j=0}^{d-1} r^(p^j);  A^((p-1)/2) == r^((p^d - 1)/2).
        let mut product = r.clone();
        let mut frob = r.clone();
        for _ in 1..class_degree {
            frob = fp_pow_mod(&frob, p, h, p);
            product = fp_rem_monic(&fp_mul(&product, &frob, p), h, p);
        }
        let a = fp_scalar_pow_poly(&product, half_scalar, h, p);
        let diff = fp_sub(&a, &[1u64], p);
        let g = fp_gcd(diff, h.to_vec(), p);
        if g.len() > 1 && g.len() < h.len() {
            return Some((g.clone(), fp_div_monic(h, &g, p)));
        }
    }
    None
}

fn fp_scalar_pow_poly(base: &[u64], mut exp: u64, f: &[u64], p: u64) -> Vec<u64> {
    let one = fp_rem_monic(&[1u64], f, p);
    let mut acc = one;
    let mut squares = base.to_vec();
    while exp > 0 {
        if exp & 1 == 1 {
            acc = fp_rem_monic(&fp_mul(&acc, &squares, p), f, p);
        }
        squares = fp_rem_monic(&fp_mul(&squares, &squares, p), f, p);
        exp >>= 1;
    }
    acc
}

/// Full factorization of a monic square-free polynomial over GF(p).
/// Returns monic irreducible factors in deterministic order.
fn fp_factor_squarefree(
    f: &[u64],
    p: u64,
) -> Result<Vec<Vec<u64>>, &'static str> {
    if f.len() <= 1 {
        return Ok(Vec::new());
    }
    if !fp_is_squarefree(f, p) {
        return Err("reduction mod prime is not square-free");
    }
    let mut out: Vec<Vec<u64>> = Vec::new();
    for (class_degree, product) in fp_distinct_degree(f, p) {
        let mut queue = vec![product];
        while let Some(current) = queue.pop() {
            if current.len() <= 1 {
                continue;
            }
            if current.len() - 1 == class_degree {
                out.push(current);
                continue;
            }
            let seed = (p as u64)
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(class_degree as u64)
                .wrapping_mul(31);
            match fp_equal_degree_split(&current, class_degree, p, seed) {
                Some((left, right)) => {
                    queue.push(left);
                    queue.push(right);
                }
                None => {
                    return Err("equal-degree split seed budget exhausted");
                }
            }
        }
    }
    out.sort();
    Ok(out)
}
