//! Polynomial factorization and square-free decomposition (WS09).

#![forbid(unsafe_code)]

use crate::PolyError;
use crate::univariate::UnivariatePoly;
use fsym_budget::{BudgetMeter, Dimension, Unbounded};
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
    metered_square_free_decomposition(poly, &mut Unbounded)
}

/// Metered Yun core with the exact mathematics of
/// [`square_free_decomposition`]: initialization and each loop iteration charge
/// their arithmetic batch and take a cancellation safe point before allocation.
pub fn metered_square_free_decomposition(
    poly: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<FactorizationResult, PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        return Ok(FactorizationResult {
            scale: BigRational::zero(),
            factors: Vec::new(),
        });
    }

    // Prepay normalization, derivative, Euclidean GCD and both initial exact
    // divisions, even when the square-free fast path returns without a loop.
    // Cubic dense work and quadratic coefficient-height growth are conservative
    // logical envelopes for the rational remainder sequence, not allocator metrics.
    let width = u64::try_from(poly.coeffs.len()).unwrap_or(u64::MAX);
    let input_bits = poly
        .coeffs
        .iter()
        .map(|c| c.numer().bits().max(c.denom().bits()))
        .max()
        .unwrap_or(1);
    let initial_work = width.saturating_pow(3).saturating_mul(16);
    let initial_bits = input_bits
        .saturating_add(u64::from(width.ilog2()).saturating_add(1))
        .saturating_mul(width.saturating_pow(2))
        .saturating_mul(8);
    factor_work(meter, initial_work, initial_work, initial_bits)?;

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
        // Preflight: charge this iteration's derivative/sub/GCD/division batch
        // before its allocations.
        factor_work(
            meter,
            8 * w.coeffs.len() as u64,
            8 * w.coeffs.len() as u64,
            128,
        )?;
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

/// Maximum degree admitted by the independent Kronecker generator.
pub const MAX_KRONECKER_DEGREE: usize = 8;
/// Maximum absolute integer coefficient bit length, including recursive factors.
pub const MAX_KRONECKER_COEFF_BITS: u64 = 16;
/// Nonzero evaluations above this magnitude are not used for interpolation.
pub const MAX_KRONECKER_EVALUATION: u64 = 1_000_000;
/// Total interpolation tuples across every recursive split, not per factor.
pub const MAX_KRONECKER_CANDIDATES: usize = 100_000;

// Charge before each bounded arithmetic batch, including discarded candidates and
// scratch storage. These are conservative logical work/storage units, not measured
// allocator bytes. There are no refunds for transient work. A rational slot includes
// both numerator and denominator plus their allocation headers.
fn factor_work(
    meter: &mut impl BudgetMeter,
    steps: u64,
    slots: u64,
    bits: u64,
) -> Result<(), PolyError> {
    meter
        .checkpoint()
        .map_err(|e| PolyError::General(e.to_string()))?;
    let limbs = bits.div_ceil(64).max(1);
    meter
        .charge_batch(&[
            (
                Dimension::ComputeSteps,
                steps.max(1).saturating_mul(limbs.saturating_mul(limbs)),
            ),
            (
                Dimension::MemoryBytes,
                slots.max(1).saturating_mul(64 + 16 * limbs),
            ),
            (Dimension::AllocationCount, slots.max(1).saturating_mul(2)),
        ])
        .map_err(|e| PolyError::General(e.to_string()))
}

/// Independent evaluation/divisor/interpolation Kronecker factorization.
///
/// Admits canonical monic ZZ polynomials of degree <= 8 with <= 16-bit
/// coefficients; integer constants (including zero) retain their exact scalar.
/// Nonconstant nonmonic or rational input is explicitly refused. Recursive factors
/// must satisfy the same coefficient bound. Evaluation points are deterministically
/// 0, 1, -1, ..., 8, -8; only nonzero values of magnitude <= 1,000,000 are used.
/// Every signed divisor is enumerated by trial division through the integer square
/// root. Each possible factor degree uses d+1 points and exact rational Lagrange
/// interpolation, admits only monic integer coefficients, then exact-divides.
/// Failure to obtain enough bounded points or exhausting the global 100,000-tuple
/// search bound returns an error, never a partial result advertised as complete.
///
/// Output is sorted by degree then ascending coefficients and merges multiplicities.
/// It is an exact product decomposition only: no irreducibility evidence is issued.
/// Every evaluation, divisor trial, interpolation and division has a charged safe
/// point; caller cancellation/budget errors propagate without an unbounded fallback.
pub fn metered_kronecker_factorization(
    poly: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<FactorizationResult, PolyError> {
    meter
        .checkpoint()
        .map_err(|e| PolyError::General(e.to_string()))?;
    poly.validate_shape()?;
    kronecker_admit(poly)?;
    factor_work(
        meter,
        poly.coeffs.len() as u64,
        poly.coeffs.len() as u64,
        32,
    )?;
    if poly.degree().unwrap_or(0) == 0 {
        return Ok(FactorizationResult {
            scale: poly.coeffs[0].clone(),
            factors: Vec::new(),
        });
    }
    let mut remaining_candidates = MAX_KRONECKER_CANDIDATES;
    let mut pieces = Vec::new();
    kronecker_split(poly.clone(), &mut pieces, &mut remaining_candidates, meter)?;
    factor_work(meter, 128, 32, 32)?;
    pieces.sort_by(|a, b| {
        a.degree()
            .cmp(&b.degree())
            .then_with(|| a.coeffs.cmp(&b.coeffs))
    });
    let mut factors: Vec<FactorTerm> = Vec::new();
    for poly in pieces {
        if let Some(last) = factors.last_mut().filter(|last| last.poly == poly) {
            last.multiplicity += 1;
        } else {
            factors.push(FactorTerm {
                poly,
                multiplicity: 1,
            });
        }
    }
    meter
        .checkpoint()
        .map_err(|e| PolyError::General(e.to_string()))?;
    Ok(FactorizationResult {
        scale: BigRational::one(),
        factors,
    })
}

fn kronecker_admit(poly: &UnivariatePoly) -> Result<(), PolyError> {
    if poly.degree().unwrap_or(0) > MAX_KRONECKER_DEGREE
        || poly
            .coeffs
            .iter()
            .any(|c| !c.is_integer() || c.numer().bits() > MAX_KRONECKER_COEFF_BITS)
        || (poly.degree().unwrap_or(0) > 0 && !poly.is_monic())
    {
        return Err(PolyError::General(
            "refused: Kronecker requires monic ZZ, degree <= 8 and coefficient height <= 16 bits (or an integer constant)".into(),
        ));
    }
    Ok(())
}

fn kronecker_divisors(n: u64, meter: &mut impl BudgetMeter) -> Result<Vec<BigRational>, PolyError> {
    let mut divisors = Vec::new();
    let mut d = 1;
    while d <= n / d {
        factor_work(meter, 1, 4, 32)?;
        if n.is_multiple_of(d) {
            for value in [d, n / d] {
                // A square-root divisor is pushed twice; sort+dedup collapses it.
                let value = BigRational::from_integer(BigInt::from_u64(value));
                divisors.push(-value.clone());
                divisors.push(value);
            }
        }
        d += 1;
    }
    divisors.sort();
    divisors.dedup();
    Ok(divisors)
}

fn kronecker_split(
    poly: UnivariatePoly,
    out: &mut Vec<UnivariatePoly>,
    remaining_candidates: &mut usize,
    meter: &mut impl BudgetMeter,
) -> Result<(), PolyError> {
    factor_work(meter, 1, poly.coeffs.len() as u64, 64)?;
    kronecker_admit(&poly)?;
    let degree = poly.degree().unwrap_or(0);
    if degree <= 1 {
        if degree == 1 {
            out.push(poly);
        }
        return Ok(());
    }
    let mut points = Vec::new();
    let mut divisors = Vec::new();
    for index in 0..=2 * MAX_KRONECKER_DEGREE {
        let integer = if index % 2 == 1 {
            index.div_ceil(2) as i64
        } else {
            -(index as i64 / 2)
        };
        let point = BigRational::from_integer(BigInt::from(integer));
        factor_work(
            meter,
            2 * poly.coeffs.len() as u64,
            2 * poly.coeffs.len() as u64,
            64,
        )?;
        let value = poly.eval(&point);
        if value.is_zero() {
            let linear =
                UnivariatePoly::new(poly.gen_sym.clone(), vec![-point, BigRational::one()]);
            factor_work(meter, 2 * degree as u64, 4 * degree as u64, 64)?;
            let (quotient, remainder) = poly.div_rem(&linear)?;
            if !remainder.is_zero() {
                return Err(PolyError::IdentityCheckFailed(
                    "Kronecker root division failed".into(),
                ));
            }
            out.push(linear);
            return kronecker_split(quotient, out, remaining_candidates, meter);
        }
        let magnitude = value.numer().abs();
        if magnitude > BigInt::from_u64(MAX_KRONECKER_EVALUATION) {
            continue;
        }
        let n = magnitude
            .to_u64()
            .ok_or_else(|| PolyError::General("Kronecker evaluation conversion failed".into()))?;
        points.push(point);
        divisors.push(kronecker_divisors(n, meter)?);
        if points.len() == degree / 2 + 1 {
            break;
        }
    }
    if points.len() < degree / 2 + 1 {
        return Err(PolyError::General(
            "refused: Kronecker has too few bounded nonzero evaluation points".into(),
        ));
    }
    for factor_degree in 1..=degree / 2 {
        let count = factor_degree + 1;
        // Lagrange bases are shared across all divisor tuples at this degree.
        let mut bases = Vec::with_capacity(count);
        for i in 0..count {
            factor_work(
                meter,
                (count * count * count) as u64,
                (count * count * count) as u64,
                128,
            )?;
            let mut basis = UnivariatePoly::one(poly.gen_sym.clone());
            let mut denominator = BigRational::one();
            for j in 0..count {
                if i != j {
                    basis = basis.mul(&UnivariatePoly::new(
                        poly.gen_sym.clone(),
                        vec![-points[j].clone(), BigRational::one()],
                    ))?;
                    denominator *= &points[i] - &points[j];
                }
            }
            for coefficient in &mut basis.coeffs {
                *coefficient = &*coefficient / &denominator;
            }
            bases.push(basis);
        }
        let mut indices = vec![0; count];
        loop {
            if *remaining_candidates == 0 {
                return Err(PolyError::General(
                    "refused: Kronecker interpolation search bound exhausted".into(),
                ));
            }
            factor_work(
                meter,
                (2 * count * count) as u64,
                (3 * count * count) as u64,
                128,
            )?;
            *remaining_candidates -= 1;
            let mut coeffs = vec![BigRational::zero(); count];
            for i in 0..count {
                for (coefficient, basis) in coeffs.iter_mut().zip(&bases[i].coeffs) {
                    *coefficient += basis * &divisors[i][indices[i]];
                }
            }
            if coeffs.last().is_some_and(One::is_one) && coeffs.iter().all(BigRational::is_integer)
            {
                let candidate = UnivariatePoly::new(poly.gen_sym.clone(), coeffs);
                factor_work(
                    meter,
                    (2 * degree * count) as u64,
                    (4 * degree * count) as u64,
                    256,
                )?;
                let (quotient, remainder) = poly.div_rem(&candidate)?;
                if remainder.is_zero() {
                    kronecker_split(candidate, out, remaining_candidates, meter)?;
                    return kronecker_split(quotient, out, remaining_candidates, meter);
                }
            }
            let mut position = 0;
            while position < count {
                indices[position] += 1;
                if indices[position] < divisors[position].len() {
                    break;
                }
                indices[position] = 0;
                position += 1;
            }
            if position == count {
                break;
            }
        }
    }
    out.push(poly);
    Ok(())
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
/// Prime candidates tried for the Hensel lifting prime selection.
const LIFT_PRIME_ATTEMPTS: usize = 16;
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
            out[i + j] =
                ((out[i + j] as u128 + (*left as u128) * (*right as u128)) % (p as u128)) as u64;
        }
    }
    fp_trim(&mut out);
    out
}

/// Remainder of `a` modulo the nonzero polynomial `f` over GF(p). The
/// divisor need not be monic: the leading coefficient is inverted so a
/// primitive integer associate reduced mod p is accepted unchanged.
fn fp_rem_monic(a: &[u64], f: &[u64], p: u64) -> Vec<u64> {
    let mut out = a.to_vec();
    let flen = f.len();
    if flen == 0 {
        return out;
    }
    let inv_lead = fp_scalar_inverse(f[flen - 1], p);
    while out.len() >= flen && out.iter().any(|c| *c != 0) {
        let shift = out.len() - flen;
        let lead = out[out.len() - 1];
        if lead == 0 {
            out.pop();
            continue;
        }
        let factor = lead * inv_lead % p;
        for (k, fc) in f.iter().enumerate() {
            let idx = shift + k;
            out[idx] = (out[idx] + p - (factor * fc % p)) % p;
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
    if let Some(last) = a.last().copied()
        && last != 1
    {
        let inv = fp_scalar_inverse(last, p);
        for c in &mut a {
            *c = *c * inv % p;
        }
    }
    a
}

fn fp_scalar_inverse(a: u64, p: u64) -> u64 {
    debug_assert!(!a.is_multiple_of(p), "inverse of zero over a prime field");
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
        out.push((*c as u128 * (degree as u128) % (p as u128)) as u64);
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
    let d: u64 = match f.len().checked_sub(1) {
        Some(d) if d >= 1 => d as u64,
        _ => return false,
    };
    if d == 1 {
        return true; // every linear polynomial over a field is irreducible
    }
    // powers[k] = x^(p^k) mod f, built by d Frobenius steps.
    let mut powers: Vec<Vec<u64>> = Vec::with_capacity(d as usize + 1);
    powers.push(vec![0u64, 1]);
    for k in 1..=d {
        let next = fp_pow_mod(&powers[(k - 1) as usize], p, f, p);
        powers.push(next);
    }
    // Rabin: x^(p^d) == x (mod f) ...
    if powers[d as usize] != vec![0u64, 1] {
        return false;
    }
    // ... and gcd(x^(p^(d/ell)) - x, f) == 1 for every prime divisor ell of d.
    for ell in [
        2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    ] {
        if ell > d || !d.is_multiple_of(ell) {
            continue;
        }
        let sub = fp_sub(&powers[(d / ell) as usize], &[0, 1], p);
        let g = fp_gcd(sub, f.to_vec(), p);
        if g.len() > 1 {
            return false;
        }
    }
    true
}

/// Distinct-degree factorization: returns `(degree, product)` pairs whose
/// products multiply to `f`, each product a monic product of irreducible
/// polynomials of exactly that degree.
fn fp_distinct_degree(f: &[u64], p: u64) -> Vec<(usize, Vec<u64>)> {
    let mut remaining = f.to_vec();
    let x = vec![0u64, 1];
    let mut frob = x.clone();
    let mut classes: Vec<(usize, Vec<u64>)> = Vec::new();
    let mut degree = 0usize;
    while remaining.len() > 2 {
        degree += 1;
        frob = fp_pow_mod(&frob, p, &remaining, p);
        let diff = fp_sub(&frob, &x, p);
        let g = {
            let g = fp_gcd(diff, remaining.clone(), p);
            #[cfg(any(test, debug_assertions))]
            eprintln!(
                "[ddf] degree={degree} g_len={} remaining_len={}",
                g.len(),
                remaining.len()
            );
            g
        };
        if g.len() > 1 {
            classes.push((degree, g.clone()));
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
    let draw = |bound: usize, state: &mut u64| -> u64 {
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

/// Square-free modular factorization: preflighted charge before the
/// distinct-degree pass and before every equal-degree split attempt; each
/// queue iteration is a cancellation safe point.
fn fp_factor_squarefree(
    f: &[u64],
    p: u64,
    meter: &mut impl BudgetMeter,
) -> Result<Result<Vec<Vec<u64>>, &'static str>, PolyError> {
    if f.len() <= 1 {
        return Ok(Ok(Vec::new()));
    }
    if !fp_is_squarefree(f, p) {
        return Ok(Err("reduction mod prime is not square-free"));
    }
    // Preflight: the whole distinct-degree pass (Frobenius chain + gcd per
    // degree class), charged before its allocations.
    factor_work(
        meter,
        31 * (f.len() * f.len()) as u64,
        4 * f.len() as u64,
        32,
    )?;
    let mut out: Vec<Vec<u64>> = Vec::new();
    let classes = fp_distinct_degree(f, p);
    #[cfg(any(test, debug_assertions))]
    eprintln!("[ddf] p={p} deg={} classes={}", f.len() - 1, classes.len());
    for (class_degree, product) in classes {
        #[cfg(any(test, debug_assertions))]
        eprintln!(
            "[ddf]   class_degree={class_degree} product_len={}",
            product.len()
        );
        let mut queue = vec![product];
        while let Some(current) = queue.pop() {
            if current.len() <= 1 {
                continue;
            }
            if current.len() - 1 == class_degree {
                out.push(current);
                continue;
            }
            // Preflight: seed draw, Frobenius product chain, and the (p-1)/2
            // exponentiation for this split attempt, charged before allocation.
            factor_work(
                meter,
                (class_degree as u64 + 31) * (current.len() * current.len()) as u64,
                4 * current.len() as u64,
                32,
            )?;
            let seed = p
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(class_degree as u64)
                .wrapping_mul(31);
            match fp_equal_degree_split(&current, class_degree, p, seed) {
                Some((left, right)) => {
                    queue.push(left);
                    queue.push(right);
                }
                None => {
                    return Ok(Err("equal-degree split seed budget exhausted"));
                }
            }
        }
    }
    out.sort();
    Ok(Ok(out))
}

// ---------------------------------------------------------------------------
// ZZ integer-polynomial helpers (ascending dense coefficients).
// ---------------------------------------------------------------------------

fn z_trim(v: &mut Vec<BigInt>) {
    while v.len() > 1 && v.last().is_some_and(|c| c.is_zero()) {
        v.pop();
    }
    if v.is_empty() {
        v.push(BigInt::from_u64(0));
    }
}

fn z_add(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let mut out = vec![BigInt::from_u64(0); a.len().max(b.len())];
    for (i, c) in a.iter().enumerate() {
        out[i] = out[i].clone() + c;
    }
    for (i, c) in b.iter().enumerate() {
        out[i] = out[i].clone() + c;
    }
    z_trim(&mut out);
    out
}

fn z_mul(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    if a == [BigInt::from_u64(0)] || b == [BigInt::from_u64(0)] {
        return vec![BigInt::from_u64(0)];
    }
    let mut out = vec![BigInt::from_u64(0); a.len() + b.len() - 1];
    for (i, left) in a.iter().enumerate() {
        for (j, right) in b.iter().enumerate() {
            out[i + j] = out[i + j].clone() + left * right;
        }
    }
    z_trim(&mut out);
    out
}

/// Symmetric representative of each coefficient modulo `m` in
/// `(-m/2, m/2]`.
fn z_symmetric_reduce(v: &[BigInt], m: &BigInt) -> Vec<BigInt> {
    let half = m / BigInt::from_u64(2);
    v.iter()
        .map(|c| {
            let rem = c.div_rem(m).1;
            if rem > half { rem - m } else { rem }
        })
        .collect()
}

/// Exact division of an integer polynomial by a monic integer polynomial;
/// `None` when the remainder is nonzero.
fn z_div_monic(a: &[BigInt], monic: &[BigInt]) -> Option<Vec<BigInt>> {
    let mut rem = a.to_vec();
    let dlen = monic.len();
    if dlen == 0 || rem.len() < dlen {
        return None;
    }
    let mut q = vec![BigInt::from_u64(0); rem.len() - dlen + 1];
    while rem.len() >= dlen {
        let lead = rem[rem.len() - 1].clone();
        let shift = rem.len() - dlen;
        q[shift] = lead.clone();
        for (k, bc) in monic.iter().enumerate() {
            rem[shift + k] = rem[shift + k].clone() - (lead.clone() * bc);
        }
        rem.pop();
        z_trim(&mut rem);
        if rem.len() < dlen {
            break;
        }
    }
    z_trim(&mut rem);
    if rem == [BigInt::from_u64(0)] {
        z_trim(&mut q);
        Some(q)
    } else {
        None
    }
}

fn z_coeff_max_bits(v: &[BigInt]) -> u64 {
    v.iter().map(|c| c.bits()).max().unwrap_or(0)
}

/// Reduces an integer polynomial's coefficients modulo the prime `p`
/// into the GF(p) representation.
fn z_to_fp(v: &[BigInt], p: u64) -> Vec<u64> {
    v.iter()
        .map(|c| {
            let (_, rem) = c.div_rem(&BigInt::from_u64(p));
            let mut r = rem;
            if r < BigInt::from_u64(0) {
                r += BigInt::from_u64(p);
            }
            // p < 2^31 in the declared regime, so the reduced value is u64.
            let bytes = r.to_bytes_le();
            let mut out = 0u64;
            for (i, b) in bytes.iter().enumerate().take(8) {
                out |= (*b as u64) << (8 * i);
            }
            out % p
        })
        .collect()
}

/// Converts a GF(p) polynomial's coefficients into ZZ coefficients in
/// `[0, p)`.
fn fp_to_z(v: &[u64]) -> Vec<BigInt> {
    v.iter().map(|c| BigInt::from_u64(*c)).collect()
}

fn fp_bezout(a: &[u64], b: &[u64], p: u64) -> (Vec<u64>, Vec<u64>) {
    let mut r0 = a.to_vec();
    let mut r1 = b.to_vec();
    let mut s0 = vec![1u64];
    let mut s1: Vec<u64> = Vec::new();
    let mut t0: Vec<u64> = Vec::new();
    let mut t1 = vec![1u64];
    while !r1.is_empty() {
        // Euclid requires a monic divisor: scale r1 (and its Bezout
        // coefficients by the same unit) so the invariants `s_k*a + t_k*b
        // == r_k` are preserved.
        let lead = *r1.last().expect("nonempty by the loop guard");
        if lead != 1 {
            let inv = fp_scalar_inverse(lead, p);
            for c in &mut r1 {
                *c = *c * inv % p;
            }
            for c in &mut s1 {
                *c = *c * inv % p;
            }
            for c in &mut t1 {
                *c = *c * inv % p;
            }
        }
        let q = fp_div_monic(&r0, &r1, p);
        let q = if q.is_empty() { vec![0u64] } else { q };
        let rem = fp_rem_monic(&r0, &r1, p);
        let s2 = fp_sub(&s0, &fp_mul(&q, &s1, p), p);
        let t2 = fp_sub(&t0, &fp_mul(&q, &t1, p), p);
        r0 = std::mem::replace(&mut r1, rem);
        s0 = std::mem::replace(&mut s1, s2);
        t0 = std::mem::replace(&mut t1, t2);
    }
    // r0 is the (monic-normalized) gcd; normalize the final unit.
    if let Some(last) = r0.last().copied()
        && last != 1
    {
        let inv = fp_scalar_inverse(last, p);
        for c in &mut s0 {
            *c = *c * inv % p;
        }
        for c in &mut t0 {
            *c = *c * inv % p;
        }
    }
    (s0, t0)
}

/// Classical two-factor Hensel lift: given `base ≡ u*v (mod p)` with
/// `gcd(u, v) = 1` over GF(p), returns `(U, V)` over ZZ with
/// `base ≡ U*V (mod m)` for `m` = the first power of `p` that is `>= target`,
/// `U ≡ u (mod p)`, `V ≡ v (mod p)`.
fn hensel_lift_pair(
    base: &[BigInt],
    u: &[u64],
    v: &[u64],
    p: u64,
    target: &BigInt,
) -> (Vec<BigInt>, Vec<BigInt>) {
    let (bez_s, bez_t) = fp_bezout(u, v, p);
    let mut big_u = fp_to_z(u);
    let mut big_v = fp_to_z(v);
    let mut m = BigInt::from_u64(p);
    let p_big = BigInt::from_u64(p);
    while &m < target {
        // c = (base - U*V) / m, exact by the invariant.
        let product = z_mul(&big_u, &big_v);
        let c: Vec<BigInt> = base
            .iter()
            .zip(product.iter())
            .map(|(b, uv)| {
                let diff = b.clone() - uv;
                let (q, r) = diff.div_rem(&m);
                debug_assert!(
                    r == BigInt::from_u64(0),
                    "hensel invariant violated at m={m}: correction {diff} leaves remainder {r}"
                );
                q
            })
            .collect();
        let c_fp = z_to_fp(&c, p);
        let alpha_raw = fp_mul(&c_fp, &bez_t, p);
        let alpha = fp_rem_monic(&alpha_raw, u, p);
        let k = fp_div_monic(&alpha_raw, u, p);
        let beta = fp_add(&fp_mul(&c_fp, &bez_s, p), &fp_mul(&k, v, p), p);
        let lift_u: Vec<BigInt> = alpha
            .iter()
            .map(|c| m.clone() * BigInt::from_u64(*c))
            .collect();
        let lift_v: Vec<BigInt> = beta
            .iter()
            .map(|c| m.clone() * BigInt::from_u64(*c))
            .collect();
        big_u = z_add(&big_u, &lift_u);
        big_v = z_add(&big_v, &lift_v);
        m = m.clone() * p_big.clone();
    }
    (big_u, big_v)
}

/// Mignotte-type coefficient bound for a factor of a monic integer
/// polynomial: `binom(d, d/2) * (d+1) * max|coeff|` is a safe integer upper
/// bound for every coefficient of every integer factor.
fn mignotte_bound(z: &[BigInt]) -> BigInt {
    let d = z.len() - 1;
    let half = d / 2;
    let mut binom = BigInt::from_u64(1);
    for k in 0..half {
        binom = binom * BigInt::from_u64((d - k) as u64) / BigInt::from_u64((k + 1) as u64);
    }
    let max_abs = z
        .iter()
        .map(|c| c.abs())
        .max()
        .unwrap_or_else(|| BigInt::from_u64(0));
    binom * BigInt::from_u64((d + 1) as u64) * max_abs
}

/// Independent Rabin check used by the verifier lane (deliberately coded
/// separately from the generator's `fp_rabin_irreducible`).
fn verify_irreducible_mod_prime(int_poly: &[BigInt], prime: &BigInt) -> bool {
    // The verifier regime caps primes below 2^32 so GF(p) coefficients fit u64.
    if prime.bits() > 32 {
        return false;
    }
    let bytes = prime.to_bytes_le();
    let mut p = 0u64;
    for (i, b) in bytes.iter().enumerate().take(8) {
        p |= (*b as u64) << (8 * i);
    }
    if p < 2 {
        return false;
    }
    let f = z_to_fp(int_poly, p);
    if f.len() <= 1 {
        return false;
    }
    // Square-free check.
    if fp_gcd(f.clone(), fp_derivative(&f, p), p).len() > 1 {
        return false;
    }
    let d: u64 = (f.len() - 1) as u64;
    let mut frob = vec![0u64, 1];
    for _ in 0..d {
        frob = fp_pow_mod(&frob, p, &f, p);
    }
    let x = fp_rem_monic(&[0u64, 1], &f, p);
    if frob != x {
        return false;
    }
    // For each prime divisor ell of d: gcd(x^(p^(d/ell)) - x, f) == 1.
    // powers[k] = x^(p^k) mod f, built independently by d Frobenius steps.
    let mut powers = Vec::with_capacity(d as usize + 1);
    powers.push(vec![0u64, 1]);
    for _ in 0..d {
        let next = fp_pow_mod(&powers[powers.len() - 1], p, &f, p);
        powers.push(next);
    }
    for ell in [
        2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    ] {
        if ell <= d && d.is_multiple_of(ell) {
            let diff = fp_sub(&powers[(d / ell) as usize], &[0, 1], p);
            if fp_gcd(diff, f.clone(), p).len() > 1 {
                return false;
            }
        }
    }
    true
}

/// Searches a fresh prime budget for a Rabin-verified irreducibility
/// witness of a primitive integer polynomial.
fn prime_to_u64(prime: &BigInt) -> u64 {
    let bytes = prime.to_bytes_le();
    let mut v = 0u64;
    for (i, b) in bytes.iter().enumerate().take(8) {
        v |= (*b as u64) << (8 * i);
    }
    v
}

/// Searches a fresh prime budget for a Rabin-verified irreducibility
/// witness of a primitive integer polynomial. Generator-side selection uses
/// the generator's own Rabin test; the verifier later re-checks the recorded
/// prime through its independently coded lane.
fn irreducibility_witness(
    int_poly: &[BigInt],
    meter: &mut impl BudgetMeter,
) -> Result<Option<IrreducibilityWitness>, PolyError> {
    let mut stream = fsym_modular::PrimeStream::new();
    for _ in 0..PRIME_ATTEMPTS {
        factor_work(
            meter,
            (int_poly.len() * int_poly.len()) as u64,
            4 * int_poly.len() as u64,
            32,
        )?;
        let prime = match stream.try_next() {
            Ok(prime) => prime,
            Err(_) => return Ok(None),
        };
        let (_, rem) = int_poly.last().expect("nonempty").div_rem(&prime);
        if rem.is_zero() {
            continue; // the prime divides the leading coefficient: skip it
        }
        if prime.bits() > 31 {
            continue;
        }
        let p = prime_to_u64(&prime);
        let fp = z_to_fp(int_poly, p);
        if fp_rabin_irreducible(&fp, p) {
            return Ok(Some(IrreducibilityWitness { prime }));
        }
    }
    Ok(None)
}

fn combinations_of(pool: &[usize], size: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    fn walk(
        pool: &[usize],
        start: usize,
        size: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == size {
            out.push(current.clone());
            return;
        }
        for (offset, item) in pool.iter().enumerate().skip(start) {
            current.push(*item);
            walk(pool, offset + 1, size, current, out);
            current.pop();
        }
    }
    walk(pool, 0, size, &mut current, &mut out);
    out
}

/// Factors a monic square-free primitive integer polynomial completely.
/// Every returned factor is monic, and every factor is irreducible over ZZ:
/// accepted recombination candidates are proven irreducible by the ascending
/// subset argument, and the trailing remainder is proven irreducible because
/// no proper subset product divides it.
fn zassenhaus_monic(
    z: &[BigInt],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<Vec<BigInt>>, PolyError> {
    let degree = z.len() - 1;
    if degree == 0 {
        return Ok(Vec::new());
    }
    if degree == 1 {
        return Ok(vec![z.to_vec()]);
    }
    // Prime search: odd, the reduction must stay square-free with at most
    // MAX_MODP_FACTORS irreducible factors. Prepay reduction and both square-free
    // checks (including the modular factor core's defensive recheck); that core
    // separately charges its distinct/equal-degree factoring work.
    let mut stream = fsym_modular::PrimeStream::new();
    let mut chosen: Option<(u64, Vec<Vec<u64>>)> = None;
    let width = u64::try_from(z.len()).unwrap_or(u64::MAX);
    let prime_work = width
        .saturating_pow(3)
        .saturating_mul(8)
        .saturating_add(width.saturating_mul(256));
    let prime_bits = z_coeff_max_bits(z).max(64);
    for _ in 0..LIFT_PRIME_ATTEMPTS {
        factor_work(meter, prime_work, prime_work, prime_bits)?;
        let prime = match stream.try_next() {
            Ok(prime) => prime,
            Err(_) => continue,
        };
        if prime.bits() > 31 {
            continue; // keep GF(p) coefficients inside u64 arithmetic
        }
        let p = {
            let bytes = prime.to_bytes_le();
            let mut v = 0u64;
            for (i, b) in bytes.iter().enumerate().take(8) {
                v |= (*b as u64) << (8 * i);
            }
            v
        };
        if p < 3 {
            continue;
        }
        let fp = z_to_fp(z, p);
        if fp.len() - 1 != degree {
            continue; // prime divides the leading coefficient (defensive)
        }
        if !fp_is_squarefree(&fp, p) {
            continue;
        }
        let factors = match fp_factor_squarefree(&fp, p, meter) {
            Ok(Ok(factors)) => factors,
            Ok(Err(_)) => continue,
            Err(e) => return Err(e),
        };
        if factors.len() > MAX_MODP_FACTORS {
            continue;
        }
        chosen = Some((p, factors));
        break;
    }
    let (p, modp_factors) = chosen.ok_or_else(|| {
        PolyError::General(format!(
            "refused: no usable lifting prime found within {LIFT_PRIME_ATTEMPTS} attempts \
             for degree-{degree} square-free part"
        ))
    })?;

    // Mignotte target: smallest power p^e strictly above 2 * bound.
    let bound = mignotte_bound(z);
    let mut target = BigInt::from_u64(p);
    let twice = bound.clone() * BigInt::from_u64(2) + BigInt::from_u64(1);
    while target <= twice {
        target = target.clone() * BigInt::from_u64(p);
    }
    #[cfg(any(test, debug_assertions))]
    eprintln!(
        "[zassenhaus] degree={} prime={} modp_factors={} target={}",
        degree,
        p,
        modp_factors.len(),
        target
    );
    #[cfg(any(test, debug_assertions))]
    for (idx, fp_factor) in modp_factors.iter().enumerate() {
        eprintln!("[zassenhaus]   u_{idx} = {fp_factor:?}");
    }

    // Nested binary Hensel lifting: each step lifts one mod-p factor of the
    // current remainder to the full target modulus.
    let modp_z: Vec<Vec<BigInt>> = modp_factors.iter().map(|f| fp_to_z(f)).collect();
    let mut lifted: Vec<Vec<BigInt>> = Vec::with_capacity(modp_z.len());
    let mut base = z.to_vec();
    for u_fp_full in modp_factors.iter().take(modp_z.len().saturating_sub(1)) {
        meter
            .checkpoint()
            .map_err(|e| PolyError::General(format!("cancelled: {e}")))?;
        let u_fp = u_fp_full.clone();
        let base_fp = z_to_fp(&base, p);
        let w_fp = fp_div_monic(&base_fp, &u_fp, p);
        factor_work(
            meter,
            2 * (base.len() as u64)
                .saturating_mul(target.bits().div_ceil(64))
                .max(1),
            (4 * base.len() as u64).saturating_mul(2),
            target.bits().max(64),
        )?;
        let (lifted_u, lifted_v) = hensel_lift_pair(&base, &u_fp, &w_fp, p, &target);
        lifted.push(lifted_u);
        base = lifted_v;
    }
    z_trim(&mut base);
    lifted.push(base);

    let pe = {
        // target is itself a power of p >= the needed modulus.
        target
    };
    let pool: Vec<usize> = (0..lifted.len()).collect();
    let mut found: Vec<Vec<BigInt>> = Vec::new();
    recombine_monic(z.to_vec(), &pool, &lifted, &pe, &mut found, meter)?;
    Ok(found)
}

/// Zassenhaus recombination: ascending subset sizes against the remaining
/// polynomial. Accepted candidates and the trailing remainder are each
/// irreducible over ZZ (a proper factor of an accepted candidate would have
/// been found at a strictly smaller subset size against the same remaining
/// polynomial).
fn recombine_monic(
    remaining: Vec<BigInt>,
    pool: &[usize],
    lifted: &[Vec<BigInt>],
    pe: &BigInt,
    out: &mut Vec<Vec<BigInt>>,
    meter: &mut impl BudgetMeter,
) -> Result<(), PolyError> {
    if remaining.len() <= 1 {
        return Ok(());
    }
    for size in 1..=pool.len() / 2 {
        // The recursive enumerator also visits incomplete subsets. All 2^n
        // subsets bound those visits and their cloned index vectors. Overflow
        // saturates the charge instead of admitting a wrapped cheap batch.
        let subsets = u32::try_from(pool.len())
            .ok()
            .and_then(|n| 1u64.checked_shl(n))
            .unwrap_or(u64::MAX);
        let combination_work = subsets
            .saturating_mul(u64::try_from(size).unwrap_or(u64::MAX).saturating_add(1))
            .saturating_mul(2);
        factor_work(meter, combination_work, combination_work, 64)?;
        for combo in combinations_of(pool, size) {
            // Estimate from existing coefficient metadata before constructing
            // the product. Sum log2(l1 norms) bounds coefficient growth through
            // every convolution; division adds at most one modulus-height
            // coefficient per eliminated degree. Include rejected candidates,
            // scratch coefficients, accepted-output growth and next_pool.
            let mut product_width = 1u64;
            let mut product_bits = 1u64;
            let mut candidate_work = 1u64;
            for &i in &combo {
                let factor_width = u64::try_from(lifted[i].len()).unwrap_or(u64::MAX);
                candidate_work = candidate_work
                    .saturating_add(product_width.saturating_mul(factor_width).saturating_mul(4));
                product_width = product_width.saturating_add(factor_width.saturating_sub(1));
                product_bits = product_bits
                    .saturating_add(z_coeff_max_bits(&lifted[i]))
                    .saturating_add(u64::from(factor_width.max(1).ilog2()).saturating_add(1));
            }
            let remaining_width = u64::try_from(remaining.len()).unwrap_or(u64::MAX);
            let division_bits = z_coeff_max_bits(&remaining)
                .saturating_add(remaining_width.saturating_mul(pe.bits().saturating_add(1)));
            candidate_work = candidate_work
                .saturating_add(
                    remaining_width
                        .saturating_mul(product_width)
                        .saturating_mul(4),
                )
                .saturating_add(product_width.saturating_mul(4))
                .saturating_add(
                    u64::try_from(pool.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(4),
                );
            factor_work(
                meter,
                candidate_work,
                candidate_work,
                product_bits.max(division_bits).max(pe.bits()),
            )?;
            let mut prod = vec![BigInt::from_u64(1)];
            for &i in &combo {
                prod = z_mul(&prod, &lifted[i]);
            }
            let mut cand = z_symmetric_reduce(&prod, pe);
            z_trim(&mut cand);
            if cand.len() <= 1 {
                continue;
            }
            if let Some(quotient) = z_div_monic(&remaining, &cand) {
                out.push(cand);
                let next_pool: Vec<usize> = pool
                    .iter()
                    .copied()
                    .filter(|i| !combo.contains(i))
                    .collect();
                return recombine_monic(quotient, &next_pool, lifted, pe, out, meter);
            }
        }
    }
    // No proper subset product divides: the remainder is irreducible over ZZ.
    out.push(remaining);
    Ok(())
}

type PartFactorization = (
    BigRational,
    Vec<(UnivariatePoly, Option<IrreducibilityWitness>)>,
);

/// Factors one square-free QQ part completely. Returns the part's scalar and
/// its monic irreducible factors.
fn factor_squarefree_part(
    part: &UnivariatePoly,
    meter: &mut impl BudgetMeter,
) -> Result<PartFactorization, PolyError> {
    let degree = part.degree().unwrap_or(0);
    if degree == 0 {
        return Ok((part.coeffs[0].clone(), Vec::new()));
    }
    // Make monic over QQ; the leading coefficient moves into the scale.
    let lc = part.leading_coeff().clone();
    let monic = part.make_monic()?;
    let mut scale = lc;

    // Clear denominators: Z = D * monic is a primitive integer polynomial
    // whose leading coefficient is D.
    let mut lcm_den = BigInt::from_u64(1);
    for c in &monic.coeffs {
        let den = c.denom().clone();
        lcm_den = lcm_den.clone() * den.clone() / den.gcd(&lcm_den);
    }
    let mut z_int: Vec<BigInt> = monic
        .coeffs
        .iter()
        .map(|c| {
            let (num, _) = (c.numer().clone(), c.denom().clone());
            num * (lcm_den.clone() / c.denom().clone())
        })
        .collect();
    // Integer content back into the scale.
    let gamma = z_int
        .iter()
        .fold(None::<BigInt>, |acc, c| {
            Some(match acc {
                Some(a) => a.gcd(c),
                None => c.clone(),
            })
        })
        .unwrap_or_else(|| BigInt::from_u64(1));
    if !gamma.is_zero() && gamma != BigInt::from_u64(1) {
        for c in &mut z_int {
            let (q, r) = c.clone().div_rem(&gamma);
            debug_assert!(r.is_zero());
            *c = q;
        }
    }
    scale = scale * BigRational::from_integer(gamma) / BigRational::from_integer(lcm_den.clone());
    if z_coeff_max_bits(&z_int) > MAX_FACTOR_COEFF_BITS as u64 {
        return Err(PolyError::General(format!(
            "refused: primitive coefficient height exceeds the declared \
             complete-factorization bound of {MAX_FACTOR_COEFF_BITS} bits"
        )));
    }

    // Monic transform when the primitive leading coefficient is not 1:
    // M(x) = a^(d-1) * Z(x/a) is monic integer and factors correspond
    // one-to-one through x -> x/a (Gauss).
    let lead_a = z_int[z_int.len() - 1].clone();
    let transformed = lead_a != BigInt::from_u64(1);
    let monic_int: Vec<BigInt> = if transformed {
        let a = lead_a.clone();
        (0..=degree)
            .map(|k| {
                if k == degree {
                    BigInt::from_u64(1) // z_d = a, so a^(d-1-d) * z_d = 1
                } else {
                    z_int[k].clone() * a.pow((degree - 1 - k) as u32)
                }
            })
            .collect()
    } else {
        z_int.clone()
    };
    // Z' = a^(1-d) * prod(H_i(a*x)) after the monic transform: the a^(1-d)
    // denominator is absorbed into the part scale.
    if transformed {
        scale *= BigRational::from_integer(lead_a.clone())
            .pow(1 - degree as i32)
            .map_err(|e| PolyError::General(format!("scale power failed: {e}")))?;
    }
    let int_factors = zassenhaus_monic(&monic_int, meter)?;
    let mut out = Vec::new();
    for int_factor in int_factors {
        let mapped_int: Vec<BigInt> = if transformed {
            let a = lead_a.clone();
            int_factor
                .iter()
                .enumerate()
                .map(|(k, c)| c.clone() * a.pow(k as u32))
                .collect()
        } else {
            int_factor.clone()
        };
        // Primitive associate carries the content into the scale.
        let content = mapped_int
            .iter()
            .fold(None::<BigInt>, |acc, c| {
                Some(match acc {
                    Some(a) => a.gcd(c),
                    None => c.clone(),
                })
            })
            .unwrap_or_else(|| BigInt::from_u64(1));
        let primitive: Vec<BigInt> = if content != BigInt::from_u64(1) {
            mapped_int
                .iter()
                .map(|c| c.clone() / content.clone())
                .collect()
        } else {
            mapped_int.clone()
        };
        scale *= BigRational::from_integer(content);
        let pre_monic_lc = primitive
            .last()
            .cloned()
            .unwrap_or_else(|| BigInt::from_u64(1));
        let coeffs: Vec<BigRational> = primitive
            .iter()
            .map(|c| BigRational::from_integer(c.clone()))
            .collect();
        let factor_poly = UnivariatePoly::new(part.gen_sym.clone(), coeffs).make_monic()?;
        scale *= BigRational::from_integer(pre_monic_lc);
        factor_work(meter, 8, 4 * primitive.len() as u64, 64)?;
        let witness = irreducibility_witness(&primitive, meter)?;
        out.push((factor_poly, witness));
    }
    Ok((scale, out))
}

/// Complete bounded factorization over QQ/ZZ with per-factor irreducibility
/// evidence. Outside the declared regime this refuses explicitly; it never
/// returns a partial factorization presented as complete.
pub fn metered_complete_factorization<M: BudgetMeter>(
    poly: &UnivariatePoly,
    meter: &mut M,
) -> Result<CompleteFactorization, PolyError> {
    poly.validate_shape()?;
    if poly.is_zero() {
        return Ok(CompleteFactorization {
            scale: BigRational::zero(),
            factors: Vec::new(),
        });
    }
    let degree = poly.degree().unwrap_or(0);
    if degree > MAX_FACTOR_DEGREE {
        return Err(PolyError::General(format!(
            "refused: degree {degree} exceeds the declared complete-factorization \
             bound of {MAX_FACTOR_DEGREE}"
        )));
    }
    if degree == 0 {
        return Ok(CompleteFactorization {
            scale: poly.coeffs[0].clone(),
            factors: Vec::new(),
        });
    }
    meter
        .checkpoint()
        .map_err(|e| PolyError::General(format!("cancelled: {e}")))?;
    let square_free = metered_square_free_decomposition(poly, meter)?;
    let mut scale = square_free.scale.clone();
    let mut factors: Vec<CompleteFactorTerm> = Vec::new();
    for term in &square_free.factors {
        let (part_scale, part_factors) = factor_squarefree_part(&term.poly, meter)?;
        scale *= part_scale
            .pow(term.multiplicity as i32)
            .map_err(|e| PolyError::General(format!("factorization scale power failed: {e}")))?;
        for (factor_poly, witness) in part_factors {
            if let Some(existing) = factors
                .iter_mut()
                .find(|f| f.poly == factor_poly && f.irreducibility == witness)
            {
                existing.multiplicity += term.multiplicity;
            } else {
                factors.push(CompleteFactorTerm {
                    poly: factor_poly,
                    multiplicity: term.multiplicity,
                    irreducibility: witness,
                });
            }
        }
    }
    let result = CompleteFactorization { scale, factors };
    factor_work(meter, 32, 4 * poly.coeffs.len() as u64, 64)?;
    verify_complete_factorization(poly, &result)?;
    Ok(result)
}

/// Convenience wrapper over [`metered_complete_factorization`] with no
/// budget.
pub fn complete_factorization(poly: &UnivariatePoly) -> Result<CompleteFactorization, PolyError> {
    metered_complete_factorization(poly, &mut Unbounded)
}

impl CompleteFactorization {
    /// Reconstructs the expanded product polynomial.
    pub fn expand(&self, sym: Symbol) -> Result<UnivariatePoly, PolyError> {
        let mut prod = UnivariatePoly::new(sym.clone(), vec![self.scale.clone()]);
        for term in &self.factors {
            let power = term.poly.pow(term.multiplicity as u32)?;
            prod = prod.mul(&power)?;
        }
        Ok(prod)
    }
}

/// Independent verification lane for a complete factorization: product
/// identity, canonical factor shape, pairwise coprimality via resultants,
/// and — for every claimed witness — an independently coded Rabin check.
/// The generator's decisions are never trusted as authority.
pub fn verify_complete_factorization(
    poly: &UnivariatePoly,
    factorization: &CompleteFactorization,
) -> Result<(), PolyError> {
    poly.validate_shape()?;
    let expanded = factorization.expand(poly.gen_sym.clone())?;
    if expanded != *poly {
        return Err(PolyError::IdentityCheckFailed(
            "complete factorization product does not reproduce the input polynomial".to_string(),
        ));
    }
    for term in &factorization.factors {
        term.poly.validate_shape()?;
        if term.multiplicity == 0 {
            return Err(PolyError::IdentityCheckFailed(
                "factor multiplicity must be positive".to_string(),
            ));
        }
        if !term.poly.is_monic() {
            return Err(PolyError::IdentityCheckFailed(
                "complete-factorization factors must be monic".to_string(),
            ));
        }
        if let Some(witness) = &term.irreducibility {
            // Independent re-check: clear denominators and run the
            // verifier's own Rabin implementation.
            let mut lcm_den = BigInt::from_u64(1);
            for c in &term.poly.coeffs {
                let den = c.denom().clone();
                lcm_den = lcm_den.clone() * den.clone() / den.gcd(&lcm_den);
            }
            let int_poly: Vec<BigInt> = term
                .poly
                .coeffs
                .iter()
                .map(|c| c.numer().clone() * (lcm_den.clone() / c.denom().clone()))
                .collect();
            if !verify_irreducible_mod_prime(&int_poly, &witness.prime) {
                return Err(PolyError::IdentityCheckFailed(format!(
                    "irreducibility witness {} failed independent verification",
                    witness.prime
                )));
            }
        }
    }
    // Pairwise coprimality: distinct monic irreducible factors have a
    // nonzero resultant.
    for i in 0..factorization.factors.len() {
        for j in (i + 1)..factorization.factors.len() {
            let res = factorization.factors[i]
                .poly
                .resultant(&factorization.factors[j].poly)?;
            if res.is_zero() {
                return Err(PolyError::IdentityCheckFailed(
                    "complete-factorization factors must be pairwise coprime".to_string(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod complete_factorization_tests {
    use super::*;
    use fsym_budget::{Dimension, MeterError};

    fn sym() -> Symbol {
        Symbol::new("x")
    }

    /// Ascending integer-coefficient polynomial.
    fn ipoly(coeffs: &[i64]) -> UnivariatePoly {
        let coeffs = coeffs
            .iter()
            .map(|c| BigRational::from_integer(BigInt::from(*c)))
            .collect();
        UnivariatePoly::new(sym(), coeffs)
    }

    fn witness_count(result: &CompleteFactorization) -> usize {
        result
            .factors
            .iter()
            .filter(|f| f.irreducibility.is_some())
            .count()
    }

    #[test]
    fn quartic_without_rational_roots_factors_with_witnesses() {
        // x^4 + 4 = (x^2 - 2x + 2)(x^2 + 2x + 2): the reality-check repro.
        let p = ipoly(&[4, 0, 0, 0, 1]);
        let result = complete_factorization(&p).expect("factors within bounds");
        assert_eq!(result.scale, BigRational::from_integer(BigInt::from(1)));
        assert_eq!(result.factors.len(), 2, "both quadratic factors found");
        for term in &result.factors {
            assert_eq!(term.multiplicity, 1);
            assert_eq!(term.poly.degree(), Some(2));
        }
        // Exact product identity plus independent certificate verification.
        verify_complete_factorization(&p, &result).expect("verifier accepts");
        // The factors are provably irreducible; the declared prime budget is
        // generous enough that both witnesses are found for this input.
        assert_eq!(witness_count(&result), 2, "both quadratics carry witnesses");
    }

    #[test]
    fn irreducible_quartic_stays_whole_with_a_witness() {
        // x^4 + 1 is irreducible over ZZ.
        let p = ipoly(&[1, 0, 0, 0, 1]);
        let result = complete_factorization(&p).expect("factors within bounds");
        assert_eq!(result.factors.len(), 1);
        assert_eq!(result.factors[0].multiplicity, 1);
        assert_eq!(result.factors[0].poly.degree(), Some(4));
        // NOTE: x^4 + 1 is the canonical polynomial that is irreducible over
        // ZZ yet reducible modulo every prime, so the weaker claim (no Rabin
        // witness) is the correct outcome here per the bead's deliverable.
        assert!(result.factors[0].irreducibility.is_none());
        verify_complete_factorization(&p, &result).expect("verifier accepts");
    }

    #[test]
    fn multiplicities_and_linear_factors_survive_the_complete_path() {
        // (x - 3) * (x^2 + 1)^2
        let p = ipoly(&[-3, 1, -6, 2, -3, 1]);
        let result = complete_factorization(&p).expect("factors within bounds");
        let squares: Vec<_> = result
            .factors
            .iter()
            .filter(|f| f.poly.degree() == Some(2))
            .collect();
        assert_eq!(squares.len(), 1, "x^2 + 1 appears once with multiplicity");
        assert_eq!(squares[0].multiplicity, 2);
        let linear: Vec<_> = result
            .factors
            .iter()
            .filter(|f| f.poly.degree() == Some(1))
            .collect();
        assert_eq!(linear.len(), 1);
        assert_eq!(linear[0].multiplicity, 1);
        verify_complete_factorization(&p, &result).expect("verifier accepts");
    }

    #[test]
    fn non_monic_primitive_part_factors_through_the_monic_transform() {
        // 2x^2 + x - 6 = (x + 2)(2x - 3)
        let p = crate::factorization::complete_factorization_tests::ipoly(&[-6, 1, 2]);
        let result = complete_factorization(&p).expect("factors within bounds");
        verify_complete_factorization(&p, &result).expect("verifier accepts");
        assert_eq!(result.factors.len(), 2);
        let expanded = result.expand(Symbol::new("x")).expect("expands");
        assert_eq!(expanded, p, "product identity after the monic transform");
    }

    #[test]
    fn degree_bound_is_a_typed_refusal() {
        // Degree 65 > declared cap 64.
        let coeffs: Vec<i64> = std::iter::once(1)
            .chain(std::iter::repeat_n(0, 64))
            .chain(std::iter::once(1))
            .collect();
        let p = ipoly(&coeffs);
        let err = complete_factorization(&p).unwrap_err();
        assert!(
            matches!(err, PolyError::General(ref m) if m.contains("refused")),
            "expected a typed refusal, got {err}"
        );
    }

    #[test]
    fn coefficient_height_bound_is_a_typed_refusal() {
        // A primitive coefficient exceeding the declared 2048-bit height.
        let big = BigInt::from_u64(1) << 2100u32;
        let p = UnivariatePoly::new(
            sym(),
            vec![
                BigRational::from_integer(big),
                BigRational::zero(),
                BigRational::one(),
            ],
        );
        let err = complete_factorization(&p).unwrap_err();
        assert!(
            matches!(err, PolyError::General(ref m) if m.contains("refused")),
            "expected a typed refusal, got {err}"
        );
    }

    struct CancellingMeter {
        checkpoints_left: u32,
    }

    impl BudgetMeter for CancellingMeter {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.checkpoints_left == 0 {
                return Err(MeterError::Cancelled);
            }
            self.checkpoints_left -= 1;
            Ok(())
        }
    }

    struct RefusingWorkMeter;

    impl BudgetMeter for RefusingWorkMeter {
        fn charge(&mut self, dimension: Dimension, requested: u64) -> Result<(), MeterError> {
            Err(MeterError::Budget(fsym_budget::BudgetError::Exhausted {
                dimension,
                requested,
                remaining: 0,
            }))
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            let (dimension, requested) = charges[0];
            self.charge(dimension, requested)
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            Ok(())
        }
    }

    #[test]
    fn yun_square_free_fast_path_cannot_bypass_work_refusal() {
        let p = ipoly(&[2, 2]);
        let err = metered_square_free_decomposition(&p, &mut RefusingWorkMeter).unwrap_err();
        assert!(matches!(&err, PolyError::General(m) if m.contains("budget exhausted")));
        let err = metered_square_free_decomposition(
            &p,
            &mut CancellingMeter {
                checkpoints_left: 0,
            },
        )
        .unwrap_err();
        assert!(matches!(&err, PolyError::General(m) if m.contains("cancelled")));
    }

    #[test]
    fn unusable_lifting_primes_cannot_bypass_work_refusal() {
        // This repeated factor is rejected as non-square-free at every prime,
        // so the modular factor core never provides a later charging point.
        let repeated = [BigInt::from(1), BigInt::from(-2), BigInt::from(1)];
        let err = zassenhaus_monic(&repeated, &mut RefusingWorkMeter).unwrap_err();
        assert!(matches!(&err, PolyError::General(m) if m.contains("budget exhausted")));
    }

    #[test]
    fn cancellation_at_a_safe_point_surfaces_as_cancelled() {
        let p = ipoly(&[4, 0, 0, 0, 1]);
        let mut meter = CancellingMeter {
            checkpoints_left: 0,
        };
        let err = metered_complete_factorization(&p, &mut meter).unwrap_err();
        assert!(
            matches!(err, PolyError::General(ref m) if m.contains("cancelled")),
            "expected the cancellation to surface, got {err}"
        );
        // Cancellation after generation has begun must also propagate.
        let mut meter = CancellingMeter {
            checkpoints_left: 8,
        };
        let err = metered_complete_factorization(&p, &mut meter).unwrap_err();
        assert!(matches!(err, PolyError::General(ref m) if m.contains("cancelled")));
    }
}

#[cfg(test)]
mod complete_factorization_proptests {
    use super::*;

    fn sym() -> Symbol {
        Symbol::new("x")
    }

    fn ipoly(coeffs: &[i64]) -> UnivariatePoly {
        let parsed: Vec<BigRational> = coeffs
            .iter()
            .map(|c| BigRational::from_integer(BigInt::from(*c)))
            .collect();
        UnivariatePoly::new(sym(), parsed)
    }

    fn mul_all(polys: &[UnivariatePoly]) -> UnivariatePoly {
        polys.iter().fold(UnivariatePoly::one(sym()), |acc, p| {
            acc.mul(p).expect("product in regime")
        })
    }

    /// Builds `content * (x-r1)(x-r2)(x-r3) * q^m_quad` with roots drawn
    /// from a small range (so multiplicities collide) and one of two fixed
    /// irreducible quadratics.
    fn gen_constructed_poly(
        (seed_a, seed_b, seed_c, mult_seed, content_seed, quad_pick, _pad): (
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
        ),
    ) -> (UnivariatePoly, Vec<(i64, i64)>) {
        let r = |s: u64| (s % 19) as i64 - 9;
        let (r1, r2, r3) = (r(seed_a), r(seed_b), r(seed_c));
        let m_lin = (mult_seed % 3 + 1) as i64;
        let m_quad = (mult_seed % 3) as i64;
        let content = (content_seed % 7 + 1) as i64;
        let quad = if quad_pick % 2 == 0 {
            ipoly(&[1, 0, 1])
        } else {
            ipoly(&[2, 2, 1])
        };

        let mut factors = vec![ipoly(&[content])];
        let mut expected: Vec<(i64, i64)> = Vec::new();
        for root in [r1, r2, r3] {
            // Each root occurrence contributes its linear factor m_lin
            // times, so the expected multiplicity is occurrences * m_lin.
            for _ in 0..m_lin {
                factors.push(ipoly(&[-root, 1]));
            }
            match expected.iter_mut().find(|(existing, _)| *existing == root) {
                Some((_, count)) => *count += 1,
                None => expected.push((root, 1)),
            }
        }
        for (_, count) in expected.iter_mut() {
            *count = count.saturating_mul(m_lin);
        }
        for _ in 0..m_quad {
            factors.push(quad.clone());
        }
        (mul_all(&factors), expected)
    }

    fn observed_multiplicity(result: &CompleteFactorization, root: i64) -> i64 {
        result
            .factors
            .iter()
            .filter(|f| {
                f.poly.degree() == Some(1)
                    && f.poly.coeffs[0] == BigRational::from_integer(BigInt::from(-root))
                    && f.poly.coeffs[1] == BigRational::one()
            })
            .map(|f| f.multiplicity as i64)
            .sum()
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(192))]

        /// Soundness (verifier accepts) + completeness (every constructed
        /// root's linear factor appears with exactly the constructed
        /// multiplicity) on deterministically generated products.
        #[test]
        fn constructed_products_factor_back_exactly(
            input in (0u64..=u64::MAX, 0u64..=u64::MAX, 0u64..=u64::MAX,
                      0u64..=u64::MAX, 0u64..=u64::MAX, 0u64..=u64::MAX,
                      0u64..=u64::MAX),
        ) {
            let (poly, expected) = gen_constructed_poly(input);
            let result = complete_factorization(&poly).expect("in-regime factorization");
            verify_complete_factorization(&poly, &result).expect("verifier accepts");
            for (root, expected_mult) in expected {
                assert_eq!(
                    observed_multiplicity(&result, root),
                    expected_mult,
                    "root {root}: multiplicity must match the constructed product"
                );
            }
            // Degree conservation: sum(deg * mult) over all factors equals
            // the input degree.
            let input_degree = poly.degree().unwrap_or(0);
            let factor_degree: i64 = result
                .factors
                .iter()
                .map(|f| f.poly.degree().unwrap_or(0) as i64 * f.multiplicity as i64)
                .sum();
            assert_eq!(factor_degree, input_degree as i64);
        }
    }
}
#[cfg(test)]
mod kronecker_factorization_tests {
    use super::*;
    use fsym_budget::{BudgetError, Dimension, MeterError};

    fn sym() -> Symbol {
        Symbol::new("x")
    }

    /// Ascending integer-coefficient polynomial.
    fn ipoly(coeffs: &[i64]) -> UnivariatePoly {
        let coeffs = coeffs
            .iter()
            .map(|c| BigRational::from_integer(BigInt::from(*c)))
            .collect();
        UnivariatePoly::new(sym(), coeffs)
    }

    fn int_rat(value: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(value))
    }

    fn assert_exact_product(poly: &UnivariatePoly, result: &FactorizationResult) {
        let expanded = result.expand(poly.gen_sym.clone()).expect("expands");
        assert_eq!(&expanded, poly, "exact product decomposition");
    }

    #[test]
    fn sophie_germain_quartic_factors_into_two_quadratics() {
        // x^4 + 4 = (x^2 - 2x + 2)(x^2 + 2x + 2): the reality-check repro.
        let p = ipoly(&[4, 0, 0, 0, 1]);
        let result = metered_kronecker_factorization(&p, &mut Unbounded).expect("factors");
        assert_eq!(result.scale, BigRational::one());
        assert_eq!(result.factors.len(), 2);
        assert_eq!(result.factors[0].poly.degree(), Some(2));
        assert_eq!(result.factors[1].poly.degree(), Some(2));
        for term in &result.factors {
            assert_eq!(term.multiplicity, 1);
            assert!(term.poly.is_monic());
        }
        assert_eq!(
            result.factors[0].poly.coeffs,
            vec![int_rat(2), int_rat(-2), int_rat(1)]
        );
        assert_eq!(
            result.factors[1].poly.coeffs,
            vec![int_rat(2), int_rat(2), int_rat(1)]
        );
        assert_exact_product(&p, &result);
        verify_square_free_product_decomposition(&p, &result).expect("verifier accepts");
    }

    #[test]
    fn mixed_degrees_product_recovers_all_factors() {
        // (x - 1)(x + 2)(x^2 + x + 1)(x^2 - 3x + 3)
        let p = ipoly(&[-1, 1])
            .mul(&ipoly(&[2, 1]))
            .unwrap()
            .mul(&ipoly(&[1, 1, 1]))
            .unwrap()
            .mul(&ipoly(&[3, -3, 1]))
            .unwrap();
        let result = metered_kronecker_factorization(&p, &mut Unbounded).expect("factors");
        assert_eq!(result.factors.len(), 4);
        assert!(result.factors.iter().all(|f| f.multiplicity == 1));
        assert_eq!(
            result
                .factors
                .iter()
                .map(|f| f.poly.degree().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 1, 2, 2]
        );
        assert_exact_product(&p, &result);
    }

    #[test]
    fn repeated_factor_merges_multiplicity() {
        // (x + 1)^2 (x^2 + 1) = x^4 + 2x^3 + 2x^2 + 2x + 1
        let p = ipoly(&[1, 2, 2, 2, 1]);
        let result = metered_kronecker_factorization(&p, &mut Unbounded).expect("factors");
        assert_eq!(result.factors.len(), 2);
        let linear = result
            .factors
            .iter()
            .find(|f| f.poly.degree() == Some(1))
            .expect("linear factor present");
        assert_eq!(linear.multiplicity, 2);
        assert_exact_product(&p, &result);
    }

    #[test]
    fn rational_coefficient_input_is_refused() {
        let p = UnivariatePoly::new(
            sym(),
            vec![
                BigRational::new(BigInt::from(1), BigInt::from(2)),
                BigRational::one(),
            ],
        );
        let err = metered_kronecker_factorization(&p, &mut Unbounded).unwrap_err();
        assert!(
            matches!(&err, PolyError::General(m) if m.contains("refused")),
            "expected typed refusal, got {err}"
        );
    }

    #[test]
    fn nonmonic_input_is_refused() {
        let p = ipoly(&[-6, 1, 2]);
        let err = metered_kronecker_factorization(&p, &mut Unbounded).unwrap_err();
        assert!(
            matches!(&err, PolyError::General(m) if m.contains("refused")),
            "expected typed refusal, got {err}"
        );
    }

    #[test]
    fn oversize_degree_is_refused() {
        let p = ipoly(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        let err = metered_kronecker_factorization(&p, &mut Unbounded).unwrap_err();
        assert!(
            matches!(&err, PolyError::General(m) if m.contains("refused")),
            "expected typed refusal, got {err}"
        );
    }

    struct CancellingMeter {
        checkpoints_left: u32,
    }

    impl BudgetMeter for CancellingMeter {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            Ok(())
        }
        fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            if self.checkpoints_left == 0 {
                return Err(MeterError::Cancelled);
            }
            self.checkpoints_left -= 1;
            Ok(())
        }
    }

    struct ExhaustingMeter {
        charges_left: u32,
    }

    impl BudgetMeter for ExhaustingMeter {
        fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
            if self.charges_left == 0 {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension: Dimension::ComputeSteps,
                    requested: 1,
                    remaining: 0,
                }));
            }
            self.charges_left = 0;
            Ok(())
        }
        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            if charges.is_empty() {
                return Ok(());
            }
            if self.charges_left == 0 {
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension: charges[0].0,
                    requested: charges[0].1,
                    remaining: 0,
                }));
            }
            self.charges_left = 0;
            Ok(())
        }
        fn checkpoint(&mut self) -> Result<(), MeterError> {
            Ok(())
        }
    }

    #[test]
    fn cancellation_before_work_propagates() {
        let p = ipoly(&[4, 0, 0, 0, 1]);
        let mut meter = CancellingMeter {
            checkpoints_left: 0,
        };
        let err = metered_kronecker_factorization(&p, &mut meter).unwrap_err();
        assert!(
            matches!(&err, PolyError::General(m) if m.contains("cancelled")),
            "expected cancellation to propagate, got {err}"
        );
    }

    #[test]
    fn budget_exhaustion_during_search_surfaces() {
        let p = ipoly(&[4, 0, 0, 0, 1]);
        let mut meter = ExhaustingMeter { charges_left: 3 };
        let err = metered_kronecker_factorization(&p, &mut meter).unwrap_err();
        assert!(
            matches!(&err, PolyError::General(m) if m.contains("budget exhausted")),
            "expected exhaustion to surface, got {err}"
        );
    }
}
