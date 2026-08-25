//! Exact ordinary differential equation (ODE) solvers and solution certificate verification (WS19).

#![forbid(unsafe_code)]

use crate::SolverError;
use fsym_calculus::{diff, integrate};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_simplify::simplify;
use std::sync::Arc;

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
    let int_p = integrate(p_expr, x).map_err(|_e| SolverError::NonLinear)?;
    let mu = Expr::Function("exp".into(), vec![int_p]);

    // \int mu(x) * Q(x) dx
    let mu_q = simplify(&Expr::Mul(vec![mu.clone(), q_expr.clone()]));
    let int_mu_q = integrate(&mu_q, x).map_err(|_e| SolverError::NonLinear)?;

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
        return Err(SolverError::NonLinear);
    }
    // Characteristic equation: a*r^2 + b*r + c = 0
    // r = (-b ± sqrt(b^2 - 4*a*c)) / (2*a)
    let disc = b * b - 4 * a * c;
    let x_sym = Expr::Sym(x.clone());
    let c1_sym = Expr::Sym(c1.clone());
    let c2_sym = Expr::Sym(c2.clone());

    if disc == 0 {
        // Repeated real root r = -b / (2*a)
        let r = Expr::Rational(BigRational::new((-b).into(), (2 * a).into()));
        let exp_rx = Expr::Function("exp".into(), vec![Expr::Mul(vec![r, x_sym.clone()])]);
        // y(x) = (C1 + C2 * x) * exp(r*x)
        let term = Expr::Add(vec![c1_sym, Expr::Mul(vec![c2_sym, x_sym])]);
        Ok(simplify(&Expr::Mul(vec![term, exp_rx])))
    } else if disc > 0 {
        // Two distinct real roots
        // For simplicity when sqrt(disc) is integer
        let isqrt_disc = (disc as f64).sqrt().round() as i64;
        if isqrt_disc * isqrt_disc == disc {
            let r1 = Expr::Rational(BigRational::new((-b + isqrt_disc).into(), (2 * a).into()));
            let r2 = Expr::Rational(BigRational::new((-b - isqrt_disc).into(), (2 * a).into()));
            let exp1 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r1, x_sym.clone()])]);
            let exp2 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r2, x_sym])]);
            let sol = Expr::Add(vec![
                Expr::Mul(vec![c1_sym, exp1]),
                Expr::Mul(vec![c2_sym, exp2]),
            ]);
            Ok(simplify(&sol))
        } else {
            // General real roots with sqrt
            let r1_num = Expr::Add(vec![
                Expr::from_i64(-b),
                Expr::Pow(
                    Arc::new(Expr::from_i64(disc)),
                    Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
                ),
            ]);
            let two_a_inv = Expr::Pow(
                Arc::new(Expr::from_i64(2 * a)),
                Arc::new(Expr::from_i64(-1)),
            );
            let r1 = Expr::Mul(vec![r1_num, two_a_inv.clone()]);
            let r2_num = Expr::Add(vec![
                Expr::from_i64(-b),
                Expr::Mul(vec![
                    Expr::from_i64(-1),
                    Expr::Pow(
                        Arc::new(Expr::from_i64(disc)),
                        Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
                    ),
                ]),
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
        let alpha = Expr::Rational(BigRational::new((-b).into(), (2 * a).into()));
        let pos_disc = -disc;
        let isqrt = (pos_disc as f64).sqrt().round() as i64;
        let beta = if isqrt * isqrt == pos_disc {
            Expr::Rational(BigRational::new(isqrt.into(), (2 * a).into()))
        } else {
            Expr::Mul(vec![
                Expr::Pow(
                    Arc::new(Expr::from_i64(pos_disc)),
                    Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
                ),
                Expr::Pow(
                    Arc::new(Expr::from_i64(2 * a)),
                    Arc::new(Expr::from_i64(-1)),
                ),
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

/// Independent verifier checking that candidate $y(x)$ satisfies linear ODE $a y'' + b y' + c y = 0$.
pub fn verify_const_coeff_second_order_solution(
    sol: &Expr,
    a: i64,
    b: i64,
    c: i64,
    x: &Symbol,
) -> bool {
    let dy = diff(sol, x);
    let d2y = diff(&dy, x);

    let lhs = Expr::Add(vec![
        Expr::Mul(vec![Expr::from_i64(a), d2y.clone()]),
        Expr::Mul(vec![Expr::from_i64(b), dy.clone()]),
        Expr::Mul(vec![Expr::from_i64(c), sol.clone()]),
    ]);

    let expanded = fsym_simplify::expand(&lhs);
    let simplified = simplify(&expanded);
    simplified.is_zero()
}
