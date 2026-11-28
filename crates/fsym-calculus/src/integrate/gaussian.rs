//! Exact reduction of P(x) exp(q(x)) for rational P and a decaying quadratic q.

use super::{func, number, pow, q, recip, rq};
use fsym_core::{BigRational, Constant, Expr, Symbol};
use fsym_polys::UnivariatePoly;
use num_traits::{Signed, ToPrimitive, Zero};

const MAX_DEGREE: usize = 32;
const MAX_SOURCE_BITS: u64 = 512;

/// Bound the degree and the bit size of a common denominator and of the sum
/// of absolute numerators before dense arithmetic expands the source. A sum
/// with unrelated denominators can require the product of those denominators.
fn polynomial_shape(
    expr: &Expr,
    x: &Symbol,
    degree_limit: usize,
    nodes: &mut usize,
    depth: usize,
) -> Option<(usize, u64)> {
    *nodes = nodes.checked_sub(1)?;
    if depth > 32 {
        return None;
    }
    let (degree, bits) = match expr {
        Expr::Integer(n) => (0, n.bits().max(1)),
        Expr::Rational(r) => (0, r.numer().bits().max(r.denom().bits())),
        Expr::Sym(symbol) if symbol == x => (1, 1),
        Expr::Add(children) | Expr::Mul(children) => {
            let mut degree: usize = 0;
            let mut bits: u64 = 1;
            for child in children {
                let (d, b) = polynomial_shape(child, x, degree_limit, nodes, depth + 1)?;
                degree = if matches!(expr, Expr::Add(_)) {
                    degree.max(d)
                } else {
                    degree.checked_add(d)?
                };
                bits = bits.checked_add(b)?.checked_add(1)?;
                if degree > degree_limit || bits > MAX_SOURCE_BITS {
                    return None;
                }
            }
            (degree, bits)
        }
        Expr::Pow(base, exponent) => {
            let Expr::Integer(power) = exponent.as_ref() else {
                return None;
            };
            let power = power.to_u32()?;
            if power > MAX_DEGREE as u32 {
                return None;
            }
            // Visit the base even for a zero exponent: a held pole cannot
            // become an entire polynomial simply because its exponent is zero.
            let (d, b) = polynomial_shape(base, x, degree_limit, nodes, depth + 1)?;
            (
                d.checked_mul(power as usize)?,
                b.checked_mul(u64::from(power).max(1))?,
            )
        }
        _ => return None,
    };
    (degree <= degree_limit && bits <= MAX_SOURCE_BITS).then_some((degree, bits))
}

pub(super) struct Primitive {
    exponent: UnivariatePoly,
    polynomial: UnivariatePoly,
    residual: BigRational,
}

impl Primitive {
    pub(super) fn from_source(source: &Expr, x: &Symbol) -> Option<Self> {
        let factors = if let Expr::Mul(factors) = source {
            factors.as_slice()
        } else {
            std::slice::from_ref(source)
        };
        if factors.len() > 256 {
            return None;
        }
        let mut exponent = None;
        for (i, factor) in factors.iter().enumerate() {
            if let Expr::Function(name, args) = factor
                && name == "exp"
                && args.len() == 1
                && exponent.replace((i, &args[0])).is_some()
            {
                return None;
            }
        }
        let (exponential, argument) = exponent?;
        let mut nodes = 256;
        polynomial_shape(argument, x, 2, &mut nodes, 0)?;
        let exponent = UnivariatePoly::from_expr(argument, x).ok()?;
        if exponent.degree() != Some(2) || !exponent.coeffs[2].is_negative() {
            return None;
        }
        let mut weight = UnivariatePoly::one(x.clone());
        let (mut degree, mut bits) = (0usize, 0u64);
        for (i, factor) in factors.iter().enumerate() {
            if i == exponential {
                continue;
            }
            let (d, b) = polynomial_shape(factor, x, MAX_DEGREE, &mut nodes, 0)?;
            degree += d;
            bits += b;
            if degree > MAX_DEGREE || bits > MAX_SOURCE_BITS {
                return None;
            }
            weight = weight
                .mul(&UnivariatePoly::from_expr(factor, x).ok()?)
                .ok()?;
        }

        // Solve P = H' + q' H + c by degree descent. q' has degree one
        // with a proven nonzero leading coefficient, so H and c are unique.
        let slope = exponent.derivative();
        let mut rest = weight.coeffs.clone();
        let mut h = vec![BigRational::zero(); rest.len().saturating_sub(1)];
        for k in (1..rest.len()).rev() {
            let coefficient = &rest[k] / &slope.coeffs[1];
            rest[k - 1] -= &coefficient * &slope.coeffs[0];
            if k > 1 {
                rest[k - 2] -= &coefficient * BigRational::from((k - 1) as i64);
            }
            h[k - 1] = coefficient;
        }
        let polynomial = UnivariatePoly::new(x.clone(), h);
        let constant = UnivariatePoly::new(x.clone(), vec![rest[0].clone()]);
        // Reconstruct the complete input polynomial by ordinary polynomial
        // arithmetic. No numerical derivative or sample is used for admission.
        if polynomial
            .derivative()
            .add(&slope.mul(&polynomial).ok()?)
            .ok()?
            .add(&constant)
            .ok()?
            != weight
        {
            return None;
        }
        Some(Self {
            exponent,
            polynomial,
            residual: rest[0].clone(),
        })
    }

    fn mass_and_argument(&self) -> (Expr, BigRational, Expr) {
        // q(x) = height - a*(x-center)^2, with a strictly positive rational.
        let a = -&self.exponent.coeffs[2];
        let center = &self.exponent.coeffs[1] / (&a * BigRational::from(2));
        let height = self.exponent.eval(&center);
        let root = pow(rq(a), q(1, 2));
        let half_mass = rq(self.residual.clone())
            * func("exp", rq(height))
            * pow(Expr::Const(Constant::Pi), q(1, 2))
            * q(1, 2)
            * recip(root.clone());
        (half_mass, center, root)
    }

    pub(super) fn expression(&self) -> Expr {
        let head = self.polynomial.to_expr() * func("exp", self.exponent.to_expr());
        if self.residual.is_zero() {
            return head;
        }
        let (half_mass, center, root) = self.mass_and_argument();
        let x = Expr::Sym(self.exponent.gen_sym.clone());
        head + half_mass * func("erf", root * (x - rq(center)))
    }

    fn endpoint(&self, point: &Expr) -> Option<Expr> {
        if matches!(
            point,
            Expr::Const(Constant::Infinity | Constant::NegativeInfinity)
        ) {
            // Every polynomial times this decaying Gaussian tends to zero
            // at either end. The real erf limits are +1 and -1 respectively.
            if self.residual.is_zero() {
                return Some(Expr::from_i64(0));
            }
            let (half_mass, _, _) = self.mass_and_argument();
            return Some(if matches!(point, Expr::Const(Constant::Infinity)) {
                half_mass
            } else {
                -half_mass
            });
        }
        let value = number(point)?;
        if value.numer().bits().max(value.denom().bits()) > MAX_SOURCE_BITS {
            return None;
        }
        let head = rq(self.polynomial.eval(&value)) * func("exp", rq(self.exponent.eval(&value)));
        if self.residual.is_zero() {
            return Some(head);
        }
        let (half_mass, center, root) = self.mass_and_argument();
        Some(head + half_mass * func("erf", root * rq(value - center)))
    }

    pub(super) fn definite(&self, lower: &Expr, upper: &Expr) -> Option<Expr> {
        Some(self.endpoint(upper)? - self.endpoint(lower)?)
    }
}
