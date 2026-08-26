//! Exact ordinary differential equation (ODE) solvers and solution certificate verification (WS19).

#![forbid(unsafe_code)]

use crate::SolverError;
use fsym_calculus::{diff, integrate};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_simplify::{simplify, try_expand, try_simplify};
use std::sync::Arc;

fn square_root_if_exact(value: &BigInt) -> Option<BigInt> {
    let root = value.sqrt();
    (&root * &root == value.clone()).then_some(root)
}

fn half_power(value: BigInt) -> Expr {
    Expr::Pow(
        Arc::new(Expr::Integer(value)),
        Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
    )
}

/// Solves first-order linear ODE: $y'(x) + P(x) y(x) = Q(x)$.
///
/// Solution: $y(x) = \frac{1}{\mu(x)} \left( \int \mu(x) Q(x) dx + C_1 \right)$ where $\mu(x) = \exp(\int P(x) dx)$.
pub fn dsolve_linear_first_order(
    p_expr: &Expr,
    q_expr: &Expr,
    x: &Symbol,
    c1: &Symbol,
) -> Result<Expr, SolverError> {
    // \int P(x) dx
    let int_p = integrate(p_expr, x).map_err(|error| {
        SolverError::IncompleteSolutionSet(format!(
            "integrating the first-order ODE coefficient failed: {error}"
        ))
    })?;
    let mu = Expr::Function("exp".into(), vec![int_p]);

    // \int mu(x) * Q(x) dx
    let mu_q = simplify(&Expr::Mul(vec![mu.clone(), q_expr.clone()]));
    let int_mu_q = integrate(&mu_q, x).map_err(|error| {
        SolverError::IncompleteSolutionSet(format!(
            "integrating the first-order ODE forcing term failed: {error}"
        ))
    })?;

    let numerator = Expr::Add(vec![int_mu_q, Expr::Sym(c1.clone())]);
    let inv_mu = Expr::Pow(Arc::new(mu), Arc::new(Expr::from_i64(-1)));

    let y_sol = simplify(&Expr::Mul(vec![numerator, inv_mu]));
    Ok(y_sol)
}

/// Solves second-order linear homogeneous ODE with constant coefficients: $a y''(x) + b y'(x) + c y(x) = 0$.
pub fn dsolve_const_coeff_second_order(
    a: i64,
    b: i64,
    c: i64,
    x: &Symbol,
    c1: &Symbol,
    c2: &Symbol,
) -> Result<Expr, SolverError> {
    if a == 0 {
        return Err(SolverError::InvalidSystem(
            "second-order ODE leading coefficient must be nonzero".to_string(),
        ));
    }
    // Characteristic equation: a*r^2 + b*r + c = 0
    // r = (-b ± sqrt(b^2 - 4*a*c)) / (2*a)
    let a = BigInt::from(a);
    let b = BigInt::from(b);
    let c = BigInt::from(c);
    let disc = &b * &b - BigInt::from(4) * &a * &c;
    let neg_b = -&b;
    let two_a = BigInt::from(2) * &a;
    let x_sym = Expr::Sym(x.clone());
    let c1_sym = Expr::Sym(c1.clone());
    let c2_sym = Expr::Sym(c2.clone());

    if disc.is_zero() {
        // Repeated real root r = -b / (2*a)
        let r = Expr::Rational(BigRational::new(neg_b.clone(), two_a.clone()));
        let exp_rx = Expr::Function("exp".into(), vec![Expr::Mul(vec![r, x_sym.clone()])]);
        // y(x) = (C1 + C2 * x) * exp(r*x)
        let term = Expr::Add(vec![c1_sym, Expr::Mul(vec![c2_sym, x_sym])]);
        Ok(simplify(&Expr::Mul(vec![term, exp_rx])))
    } else if disc.is_positive() {
        // Two distinct real roots
        if let Some(sqrt_disc) = square_root_if_exact(&disc) {
            let r1 = Expr::Rational(BigRational::new(&neg_b + &sqrt_disc, two_a.clone()));
            let r2 = Expr::Rational(BigRational::new(&neg_b - &sqrt_disc, two_a.clone()));
            let exp1 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r1, x_sym.clone()])]);
            let exp2 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r2, x_sym])]);
            let sol = Expr::Add(vec![
                Expr::Mul(vec![c1_sym, exp1]),
                Expr::Mul(vec![c2_sym, exp2]),
            ]);
            Ok(simplify(&sol))
        } else {
            // General real roots with sqrt
            let r1_num = Expr::Add(vec![Expr::Integer(neg_b.clone()), half_power(disc.clone())]);
            let two_a_inv = Expr::Pow(Arc::new(Expr::Integer(two_a)), Arc::new(Expr::from_i64(-1)));
            let r1 = Expr::Mul(vec![r1_num, two_a_inv.clone()]);
            let r2_num = Expr::Add(vec![
                Expr::Integer(neg_b),
                Expr::Mul(vec![Expr::from_i64(-1), half_power(disc)]),
            ]);
            let r2 = Expr::Mul(vec![r2_num, two_a_inv]);

            let exp1 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r1, x_sym.clone()])]);
            let exp2 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r2, x_sym])]);
            let sol = Expr::Add(vec![
                Expr::Mul(vec![c1_sym, exp1]),
                Expr::Mul(vec![c2_sym, exp2]),
            ]);
            Ok(simplify(&sol))
        }
    } else {
        // Complex conjugate roots: alpha ± i*beta
        // alpha = -b / (2*a), beta = sqrt(-disc) / (2*a)
        let alpha = Expr::Rational(BigRational::new(neg_b, two_a.clone()));
        let pos_disc = -disc;
        let beta = if let Some(sqrt_disc) = square_root_if_exact(&pos_disc) {
            Expr::Rational(BigRational::new(sqrt_disc, two_a.clone()))
        } else {
            Expr::Mul(vec![
                half_power(pos_disc),
                Expr::Pow(Arc::new(Expr::Integer(two_a)), Arc::new(Expr::from_i64(-1))),
            ])
        };

        let exp_ax = Expr::Function("exp".into(), vec![Expr::Mul(vec![alpha, x_sym.clone()])]);
        let cos_bx = Expr::Function(
            "cos".into(),
            vec![Expr::Mul(vec![beta.clone(), x_sym.clone()])],
        );
        let sin_bx = Expr::Function("sin".into(), vec![Expr::Mul(vec![beta, x_sym])]);

        let trig_part = Expr::Add(vec![
            Expr::Mul(vec![c1_sym, cos_bx]),
            Expr::Mul(vec![c2_sym, sin_bx]),
        ]);

        Ok(simplify(&Expr::Mul(vec![exp_ax, trig_part])))
    }
}

/// Exact residual checker for a candidate solution of $y'(x) + P(x) y(x) = Q(x)$.
pub fn verify_first_order_linear_solution(
    sol: &Expr,
    p_expr: &Expr,
    q_expr: &Expr,
    x: &Symbol,
) -> bool {
    let dy = diff(sol, x);
    let py = Expr::Mul(vec![p_expr.clone(), sol.clone()]);
    let residual = Expr::Add(vec![
        dy,
        py,
        Expr::Mul(vec![Expr::from_i64(-1), q_expr.clone()]),
    ]);
    let expanded = match try_expand(&residual) {
        Ok(exp) => exp,
        Err(_) => return false,
    };
    try_simplify(&expanded).is_ok_and(|simplified| simplified.is_zero())
}

/// Solves separable ODE of the form $y'(x) = f(x) \cdot y(x)$:
/// Solution: $y(x) = C_1 \cdot \exp(\int f(x) dx)$.
pub fn dsolve_separable_linear(
    f_expr: &Expr,
    x: &Symbol,
    c1: &Symbol,
) -> Result<Expr, SolverError> {
    let int_f = integrate(f_expr, x).map_err(|error| {
        SolverError::IncompleteSolutionSet(format!(
            "integrating separable ODE coefficient failed: {error}"
        ))
    })?;
    let exp_int = Expr::Function("exp".into(), vec![int_f]);
    let sol = Expr::Mul(vec![Expr::Sym(c1.clone()), exp_int]);
    Ok(simplify(&sol))
}

/// Exact residual checker for a candidate solution of $a y'' + b y' + c y = 0$.
///
/// This establishes only that the native differentiator and simplifier reduce
/// the residual to exact zero; it is not an independent completeness proof.
/// Unsupported or resource-limited residual reduction returns `false`.
pub fn verify_const_coeff_second_order_solution(
    sol: &Expr,
    a: i64,
    b: i64,
    c: i64,
    x: &Symbol,
) -> bool {
    let mut terms = Vec::with_capacity(3);
    if a != 0 {
        let dy = diff(sol, x);
        let d2y = diff(&dy, x);
        terms.push(Expr::Mul(vec![Expr::from_i64(a), d2y]));
        if b != 0 {
            terms.push(Expr::Mul(vec![Expr::from_i64(b), dy]));
        }
    } else if b != 0 {
        terms.push(Expr::Mul(vec![Expr::from_i64(b), diff(sol, x)]));
    }
    if c != 0 {
        terms.push(Expr::Mul(vec![Expr::from_i64(c), sol.clone()]));
    }

    let expanded = match try_expand(&Expr::Add(terms)) {
        Ok(expanded) => expanded,
        Err(_) => return false,
    };
    try_simplify(&expanded).is_ok_and(|simplified| simplified.is_zero())
}
