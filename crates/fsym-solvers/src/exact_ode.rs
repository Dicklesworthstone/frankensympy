//! Bounded polynomial potentials for exact first-order differential forms.
//!
//! The result is a potential F with dF = M dx + N dy. Its level sets are
//! implicit relations; this module does not select graphs, impose initial
//! conditions, or claim uniqueness at singular points.

#![forbid(unsafe_code)]

use crate::SolverError;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_polys::multivariate::MultivariatePoly;
use std::collections::BTreeMap;

const MAX_INPUT_DEGREE: u32 = 16;
const MAX_POTENTIAL_DEGREE: u32 = MAX_INPUT_DEGREE + 1;
const MAX_LITERAL_POWER: u64 = 64;
const MAX_POLYNOMIAL_TERMS: usize = 128;
const MAX_HEIGHT_BITS: u64 = 4_096;

fn refusal(reason: &str) -> SolverError {
    SolverError::IncompleteSolutionSet(format!("bounded exact polynomial ODE: {reason}"))
}

fn polynomial_error(error: fsym_polys::PolyError) -> SolverError {
    SolverError::InvalidSystem(format!("exact polynomial ODE: {error}"))
}

fn validate_variables(x: &Symbol, y: &Symbol) -> Result<(), SolverError> {
    if x == y {
        return Err(SolverError::InvalidSystem(
            "exact ODE variables must be distinct".into(),
        ));
    }
    if !matches!(
        x.name.len().checked_add(y.name.len()),
        Some(bytes) if bytes <= crate::MAX_VERIFIER_INPUT_TEXT_BYTES
    ) {
        return Err(refusal("variable identifiers exceed the text bound"));
    }
    Ok(())
}

/// A polynomial can be represented as P/D with a shared integer D.
/// These bounds control degree, the L1 norm of P's coefficients, and D,
/// before conversion performs any polynomial powers or large arithmetic.
#[derive(Clone, Copy)]
struct PolynomialBounds {
    degree: u32,
    numerator_bits: u64,
    denominator_bits: u64,
}

impl PolynomialBounds {
    fn checked(self, max_degree: u32) -> Result<Self, SolverError> {
        if self.degree > max_degree {
            return Err(refusal("polynomial degree exceeds the supported bound"));
        }
        if self.numerator_bits > MAX_HEIGHT_BITS || self.denominator_bits > MAX_HEIGHT_BITS {
            return Err(refusal(
                "polynomial coefficient growth exceeds the bit bound",
            ));
        }
        Ok(self)
    }
}

fn checked_sum(a: u64, b: u64) -> Result<u64, SolverError> {
    a.checked_add(b)
        .ok_or_else(|| refusal("coefficient-growth accounting overflowed"))
}

fn polynomial_bounds(
    expression: &Expr,
    x: &Symbol,
    y: &Symbol,
    max_degree: u32,
) -> Result<PolynomialBounds, SolverError> {
    let bounds = match expression {
        Expr::Integer(value) => PolynomialBounds {
            degree: 0,
            numerator_bits: value.bits(),
            denominator_bits: 0,
        },
        Expr::Rational(value) => PolynomialBounds {
            degree: 0,
            numerator_bits: value.numer().bits(),
            denominator_bits: if value.denom().is_one() {
                0
            } else {
                value.denom().bits()
            },
        },
        Expr::Sym(symbol) if symbol == x || symbol == y => PolynomialBounds {
            degree: 1,
            numerator_bits: 1,
            denominator_bits: 0,
        },
        Expr::Add(terms) | Expr::Mul(terms) => {
            let (first, rest) = terms
                .split_first()
                .ok_or_else(|| refusal("empty polynomial operations are unsupported"))?;
            let mut accumulated = polynomial_bounds(first, x, y, max_degree)?;
            for term in rest {
                // Inspect every original child, including factors behind a
                // zero, before polynomial cancellation can erase it.
                let next = polynomial_bounds(term, x, y, max_degree)?;
                accumulated = if matches!(expression, Expr::Add(_)) {
                    PolynomialBounds {
                        degree: accumulated.degree.max(next.degree),
                        numerator_bits: checked_sum(
                            checked_sum(accumulated.numerator_bits, next.denominator_bits)?.max(
                                checked_sum(next.numerator_bits, accumulated.denominator_bits)?,
                            ),
                            1,
                        )?,
                        denominator_bits: checked_sum(
                            accumulated.denominator_bits,
                            next.denominator_bits,
                        )?,
                    }
                } else {
                    PolynomialBounds {
                        degree: accumulated
                            .degree
                            .checked_add(next.degree)
                            .ok_or_else(|| refusal("polynomial degree accounting overflowed"))?,
                        numerator_bits: checked_sum(
                            accumulated.numerator_bits,
                            next.numerator_bits,
                        )?,
                        denominator_bits: checked_sum(
                            accumulated.denominator_bits,
                            next.denominator_bits,
                        )?,
                    }
                }
                .checked(max_degree)?;
            }
            accumulated
        }
        Expr::Pow(base, exponent) => {
            let Expr::Integer(power) = exponent.as_ref() else {
                return Err(refusal(
                    "only nonnegative literal integer powers are supported",
                ));
            };
            let power = power
                .to_u64()
                .filter(|power| *power <= MAX_LITERAL_POWER)
                .ok_or_else(|| refusal("literal polynomial power exceeds the supported bound"))?;
            let base = polynomial_bounds(base, x, y, max_degree)?;
            if power == 0 {
                PolynomialBounds {
                    degree: 0,
                    numerator_bits: 1,
                    denominator_bits: 0,
                }
            } else {
                let degree = u64::from(base.degree)
                    .checked_mul(power)
                    .and_then(|degree| u32::try_from(degree).ok())
                    .ok_or_else(|| refusal("polynomial degree accounting overflowed"))?;
                PolynomialBounds {
                    degree,
                    numerator_bits: base
                        .numerator_bits
                        .checked_mul(power)
                        .ok_or_else(|| refusal("coefficient-growth accounting overflowed"))?,
                    denominator_bits: base
                        .denominator_bits
                        .checked_mul(power)
                        .ok_or_else(|| refusal("coefficient-growth accounting overflowed"))?,
                }
            }
        }
        _ => {
            return Err(refusal(
                "coefficients and potentials must be polynomials over Q in the two supplied variables",
            ));
        }
    };
    bounds.checked(max_degree)
}

fn check_polynomial(poly: &MultivariatePoly, max_degree: u32) -> Result<(), SolverError> {
    if poly.terms().len() > MAX_POLYNOMIAL_TERMS
        || poly
            .total_degree()
            .is_some_and(|degree| degree > max_degree)
    {
        return Err(refusal(
            "expanded polynomial exceeds the degree or term bound",
        ));
    }
    if poly.terms().values().any(|coefficient| {
        coefficient.numer().bits() > MAX_HEIGHT_BITS || coefficient.denom().bits() > MAX_HEIGHT_BITS
    }) {
        return Err(refusal(
            "expanded polynomial coefficient exceeds the bit bound",
        ));
    }
    Ok(())
}

fn parse_polynomial(
    expression: &Expr,
    x: &Symbol,
    y: &Symbol,
    max_degree: u32,
) -> Result<MultivariatePoly, SolverError> {
    polynomial_bounds(expression, x, y, max_degree)?;
    let poly = MultivariatePoly::from_expr(expression, &[x.clone(), y.clone()])
        .map_err(polynomial_error)?;
    check_polynomial(&poly, max_degree)?;
    Ok(poly)
}

fn coefficients(
    m: &Expr,
    n: &Expr,
    x: &Symbol,
    y: &Symbol,
) -> Result<(MultivariatePoly, MultivariatePoly), SolverError> {
    validate_variables(x, y)?;
    if !crate::verifier_inputs_within_bounds([m, n]) {
        return Err(refusal("input exceeds the expression envelope"));
    }
    let m = parse_polynomial(m, x, y, MAX_INPUT_DEGREE)?;
    let n = parse_polynomial(n, x, y, MAX_INPUT_DEGREE)?;
    if m.is_zero() && n.is_zero() {
        return Err(SolverError::InfiniteSolutions);
    }
    Ok((m, n))
}

fn integrate_terms(
    poly: &MultivariatePoly,
    variable: usize,
) -> Result<MultivariatePoly, SolverError> {
    let mut terms = BTreeMap::new();
    for (exponents, coefficient) in poly.terms() {
        let mut integrated_exponents = exponents.clone();
        let power = integrated_exponents[variable]
            .checked_add(1)
            .filter(|power| *power <= MAX_POTENTIAL_DEGREE)
            .ok_or_else(|| refusal("integrated polynomial degree exceeds the supported bound"))?;
        integrated_exponents[variable] = power;
        let divisor = BigRational::from_integer(BigInt::from(i64::from(power)));
        terms.insert(integrated_exponents, coefficient / &divisor);
    }
    let integrated =
        MultivariatePoly::new(poly.generators().to_vec(), terms).map_err(polynomial_error)?;
    check_polynomial(&integrated, MAX_POTENTIAL_DEGREE)?;
    Ok(integrated)
}

/// Find a polynomial potential F with F_x = M and F_y = N.
///
/// This bounded lane admits nonzero exact one-forms M dx + N dy over
/// Q[x,y], with coefficient degree at most 16, at most 128 expanded terms,
/// and conservative 4096-bit coefficient-growth and expression bounds.
/// The variables must be distinct. Other symbols, non-polynomial inputs,
/// non-exact forms and exhausted bounds are explicit errors.
///
/// The return value is the potential itself, without an integration
/// constant. F = C describes implicit level sets; no explicit graph,
/// initial-condition solution, singular-branch completeness, or uniqueness
/// is asserted. A zero one-form is refused as underdetermined.
pub fn exact_ode_potential(
    m: &Expr,
    n: &Expr,
    x: &Symbol,
    y: &Symbol,
) -> Result<Expr, SolverError> {
    let (m_poly, n_poly) = coefficients(m, n, x, y)?;
    if m_poly.derivative(1).map_err(polynomial_error)?
        != n_poly.derivative(0).map_err(polynomial_error)?
    {
        return Err(SolverError::InvalidSystem(
            "the polynomial differential form is not exact: M_y differs from N_x".into(),
        ));
    }

    let integrated_m = integrate_terms(&m_poly, 0)?;
    let correction = n_poly
        .sub(&integrated_m.derivative(1).map_err(polynomial_error)?)
        .map_err(polynomial_error)?;
    check_polynomial(&correction, MAX_INPUT_DEGREE)?;
    if correction.terms().keys().any(|exponents| exponents[0] != 0) {
        return Err(refusal("the exact-form correction still depends on x"));
    }
    let potential_poly = integrated_m
        .add(&integrate_terms(&correction, 1)?)
        .map_err(polynomial_error)?;
    check_polynomial(&potential_poly, MAX_POTENTIAL_DEGREE)?;
    let potential = potential_poly.to_expr().map_err(polynomial_error)?;

    if verify_exact_ode_potential(&potential, m, n, x, y) {
        Ok(potential)
    } else {
        Err(refusal(
            "the candidate did not pass independent gradient verification",
        ))
    }
}

/// Independently check F_x = M and F_y = N by expression differentiation
/// and exact polynomial equality. This checker never calls the generator
/// or its coefficient-map integration routine, and never accepts a numeric
/// approximation to zero. A rational additive constant in F is permitted.
/// Unsupported inputs, the zero one-form, or exceeded bounds return false.
pub fn verify_exact_ode_potential(
    potential: &Expr,
    m: &Expr,
    n: &Expr,
    x: &Symbol,
    y: &Symbol,
) -> bool {
    let check = || -> Result<bool, SolverError> {
        if !crate::verifier_inputs_within_bounds([potential, m, n]) {
            return Err(refusal(
                "verification input exceeds the expression envelope",
            ));
        }
        let (m_poly, n_poly) = coefficients(m, n, x, y)?;
        let potential_poly = parse_polynomial(potential, x, y, MAX_POTENTIAL_DEGREE)?;
        let canonical = potential_poly.to_expr().map_err(polynomial_error)?;
        if !crate::verifier_inputs_within_bounds([&canonical]) {
            return Err(refusal(
                "expanded potential exceeds the expression envelope",
            ));
        }
        // The independent calculus path sees only the checked polynomial
        // expression. Canonicalizing first bounds product-rule expansion.
        let dx = fsym_calculus::diff(&canonical, x);
        let dy = fsym_calculus::diff(&canonical, y);
        if !crate::verifier_inputs_within_bounds([&dx, &dy]) {
            return Err(refusal("gradient exceeds the expression envelope"));
        }
        Ok(parse_polynomial(&dx, x, y, MAX_INPUT_DEGREE)? == m_poly
            && parse_polynomial(&dy, x, y, MAX_INPUT_DEGREE)? == n_poly)
    };
    check() == Ok(true)
}
