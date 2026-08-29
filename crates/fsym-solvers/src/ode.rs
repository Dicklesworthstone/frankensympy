//! Bounded ODE candidate generators and conservative exact residual checks (WS19).

#![forbid(unsafe_code)]

use crate::SolverError;
use fsym_calculus::{diff, integrate};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol};
use fsym_simplify::{simplify, try_expand, try_simplify};
use num_traits::Zero;
use std::sync::Arc;

fn square_root_if_exact(value: &BigInt) -> Option<BigInt> {
    let root = value.sqrt()?;
    (&root * &root == value.clone()).then_some(root)
}

fn require_fresh_integration_constants(
    x: &Symbol,
    constants: &[&Symbol],
    inputs: &[&Expr],
) -> Result<(), SolverError> {
    let identifier_bytes = constants.iter().try_fold(x.name.len(), |total, constant| {
        total.checked_add(constant.name.len())
    });
    if !matches!(
        identifier_bytes,
        Some(bytes) if bytes <= crate::MAX_VERIFIER_INPUT_TEXT_BYTES
    ) {
        return Err(SolverError::InvalidSystem(
            "ODE identifiers exceed the supported text bound".to_string(),
        ));
    }
    if !crate::verifier_inputs_within_bounds(inputs.iter().copied()) {
        return Err(SolverError::InvalidSystem(
            "ODE input exceeds the supported expression bounds".to_string(),
        ));
    }
    for (index, constant) in constants.iter().enumerate() {
        if *constant == x || constants[..index].contains(constant) {
            return Err(SolverError::InvalidSystem(
                "integration constants must be pairwise distinct from the independent variable"
                    .to_string(),
            ));
        }
        if inputs
            .iter()
            .any(|input| expression_contains_symbol(input, constant))
        {
            return Err(SolverError::InvalidSystem(
                "an integration constant already occurs in an ODE coefficient or forcing term"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

fn expression_contains_symbol(expr: &Expr, symbol: &Symbol) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match current {
            Expr::Sym(candidate) if candidate == symbol => return true,
            Expr::Add(terms) | Expr::Mul(terms) | Expr::Function(_, terms) => stack.extend(terms),
            Expr::Pow(base, exponent) => {
                stack.push(exponent);
                stack.push(base);
            }
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
        }
    }
    false
}

fn ode_expression_is_total(expr: &Expr) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match current {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Sym(_) => {}
            Expr::Const(Constant::Pi | Constant::E | Constant::I) => {}
            Expr::Const(
                Constant::Infinity
                | Constant::NegativeInfinity
                | Constant::ComplexInfinity
                | Constant::NaN,
            ) => return false,
            Expr::Add(terms) | Expr::Mul(terms) => {
                if terms.is_empty() {
                    return false;
                }
                stack.extend(terms);
            }
            Expr::Pow(base, exponent) => match exponent.as_ref() {
                Expr::Integer(value) if value > &BigInt::zero() => stack.push(base),
                _ => return false,
            },
            Expr::Function(name, args)
                if matches!(name.as_str(), "exp" | "sin" | "cos" | "sinh" | "cosh")
                    && args.len() == 1 =>
            {
                stack.extend(args);
            }
            Expr::Function(_, _) => return false,
        }
    }
    true
}

fn ode_inputs_are_total(inputs: &[&Expr]) -> bool {
    crate::verifier_inputs_within_bounds(inputs.iter().copied())
        && inputs.iter().all(|input| ode_expression_is_total(input))
}

fn residual_is_exact_zero(residual: &Expr) -> bool {
    if !ode_inputs_are_total(&[residual]) {
        return false;
    }
    let expanded = match try_expand(residual) {
        Ok(expanded) if ode_inputs_are_total(&[&expanded]) => expanded,
        _ => return false,
    };
    let simplified = match try_simplify(&expanded) {
        Ok(simplified) if ode_inputs_are_total(&[&simplified]) => simplified,
        _ => return false,
    };
    simplified.is_zero()
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
    require_fresh_integration_constants(x, &[c1], &[p_expr, q_expr])?;
    // \int P(x) dx
    let int_p = integrate(p_expr, x).map_err(|error| {
        SolverError::IncompleteSolutionSet(format!(
            "integrating the first-order ODE coefficient failed: {error}"
        ))
    })?;
    let mu = Expr::Function("exp".into(), vec![int_p.clone()]);

    // \int mu(x) * Q(x) dx
    let mu_q = simplify(&Expr::Mul(vec![mu.clone(), q_expr.clone()]));
    let int_mu_q = integrate(&mu_q, x).map_err(|error| {
        SolverError::IncompleteSolutionSet(format!(
            "integrating the first-order ODE forcing term failed: {error}"
        ))
    })?;

    let numerator = Expr::Add(vec![int_mu_q, Expr::Sym(c1.clone())]);
    let neg_int_p = simplify(&Expr::Mul(vec![Expr::from_i64(-1), int_p]));
    let inv_mu = Expr::Function("exp".into(), vec![neg_int_p]);

    let raw_sol = Expr::Mul(vec![numerator, inv_mu]);
    let expanded = try_expand(&raw_sol).unwrap_or(raw_sol);
    let y_sol = simplify(&expanded);
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
    require_fresh_integration_constants(x, &[c1, c2], &[])?;
    if a == 0 {
        return Err(SolverError::InvalidSystem(
            "second-order ODE leading coefficient must be nonzero".to_string(),
        ));
    }
    let coefficients = (a, b, c);
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

    let solution = if disc.is_zero() {
        // Repeated real root r = -b / (2*a)
        let r = Expr::Rational(BigRational::new(neg_b.clone(), two_a.clone()));
        let exp_rx = Expr::Function("exp".into(), vec![Expr::Mul(vec![r, x_sym.clone()])]);
        // y(x) = (C1 + C2 * x) * exp(r*x)
        let term = Expr::Add(vec![c1_sym, Expr::Mul(vec![c2_sym, x_sym])]);
        simplify(&Expr::Mul(vec![term, exp_rx]))
    } else if disc.is_positive() {
        // Two distinct real roots
        let sqrt_disc = square_root_if_exact(&disc).ok_or_else(|| {
            SolverError::IncompleteSolutionSet(
                "non-square characteristic radicals are not supported by the exact ODE verifier"
                    .to_string(),
            )
        })?;
        let r1 = Expr::Rational(BigRational::new(&neg_b + &sqrt_disc, two_a.clone()));
        let r2 = Expr::Rational(BigRational::new(&neg_b - &sqrt_disc, two_a.clone()));
        let exp1 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r1, x_sym.clone()])]);
        let exp2 = Expr::Function("exp".into(), vec![Expr::Mul(vec![r2, x_sym])]);
        let sol = Expr::Add(vec![
            Expr::Mul(vec![c1_sym, exp1]),
            Expr::Mul(vec![c2_sym, exp2]),
        ]);
        simplify(&sol)
    } else {
        // Complex conjugate roots: alpha ± i*beta
        // alpha = -b / (2*a), beta = sqrt(-disc) / (2*a)
        let alpha = Expr::Rational(BigRational::new(neg_b, two_a.clone()));
        let pos_disc = -disc;
        let sqrt_disc = square_root_if_exact(&pos_disc).ok_or_else(|| {
            SolverError::IncompleteSolutionSet(
                "non-square characteristic radicals are not supported by the exact ODE verifier"
                    .to_string(),
            )
        })?;
        let beta = Expr::Rational(BigRational::new(sqrt_disc, two_a.clone()));

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

        simplify(&Expr::Mul(vec![exp_ax, trig_part]))
    };

    if verify_const_coeff_second_order_solution(
        &solution,
        coefficients.0,
        coefficients.1,
        coefficients.2,
        x,
    ) {
        Ok(solution)
    } else {
        Err(SolverError::IncompleteSolutionSet(
            "the generated ODE family did not pass exact residual verification".to_string(),
        ))
    }
}

/// Exact residual checker for a candidate solution of $y'(x) + P(x) y(x) = Q(x)$.
pub fn verify_first_order_linear_solution(
    sol: &Expr,
    p_expr: &Expr,
    q_expr: &Expr,
    x: &Symbol,
) -> bool {
    verify_linear_first_order_solution(sol, p_expr, q_expr, x)
}

/// Solves separable ODE of the form $y'(x) = f(x) \cdot y(x)$:
/// Solution: $y(x) = C_1 \cdot \exp(\int f(x) dx)$.
pub fn dsolve_separable_linear(
    f_expr: &Expr,
    x: &Symbol,
    c1: &Symbol,
) -> Result<Expr, SolverError> {
    require_fresh_integration_constants(x, &[c1], &[f_expr])?;
    let int_f = integrate(f_expr, x).map_err(|error| {
        SolverError::IncompleteSolutionSet(format!(
            "integrating separable ODE coefficient failed: {error}"
        ))
    })?;
    let exp_int = Expr::Function("exp".into(), vec![int_f]);
    let sol = Expr::Mul(vec![Expr::Sym(c1.clone()), exp_int]);
    Ok(simplify(&sol))
}

fn parse_rational_scalar(expr: &Expr) -> Option<BigRational> {
    match expr {
        Expr::Integer(n) => Some(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

fn parse_linear_arg(expr: &Expr, x: &Symbol) -> Option<BigRational> {
    match expr {
        Expr::Sym(s) if s == x => Some(BigRational::from_integer(1.into())),
        Expr::Mul(factors) => {
            let mut k = BigRational::from_integer(1.into());
            let mut found_x = false;
            for f in factors {
                if let Expr::Sym(s) = f {
                    if s == x && !found_x {
                        found_x = true;
                    } else {
                        return None;
                    }
                } else {
                    let r = parse_rational_scalar(f)?;
                    k *= r;
                }
            }
            if found_x { Some(k) } else { None }
        }
        _ => None,
    }
}

fn parse_exponential(expr: &Expr, x: &Symbol) -> Option<(BigRational, BigRational)> {
    match expr {
        Expr::Function(name, args) if name == "exp" && args.len() == 1 => {
            let gamma = parse_linear_arg(&args[0], x)?;
            Some((BigRational::from_integer(1.into()), gamma))
        }
        Expr::Mul(factors) => {
            let mut k = BigRational::from_integer(1.into());
            let mut exp_gamma = None;
            for f in factors {
                if let Expr::Function(name, args) = f {
                    if name == "exp" && args.len() == 1 && exp_gamma.is_none() {
                        exp_gamma = parse_linear_arg(&args[0], x);
                    } else {
                        return None;
                    }
                } else {
                    let r = parse_rational_scalar(f)?;
                    k *= r;
                }
            }
            exp_gamma.map(|g| (k, g))
        }
        _ => None,
    }
}

fn parse_trig(expr: &Expr, x: &Symbol) -> Option<(BigRational, BigRational, bool)> {
    match expr {
        Expr::Function(name, args) if args.len() == 1 => {
            let is_cos = match name.as_str() {
                "cos" => true,
                "sin" => false,
                _ => return None,
            };
            let omega = parse_linear_arg(&args[0], x)?;
            Some((BigRational::from_integer(1.into()), omega, is_cos))
        }
        Expr::Mul(factors) => {
            let mut k = BigRational::from_integer(1.into());
            let mut trig_data = None;
            for f in factors {
                if let Expr::Function(name, args) = f {
                    if args.len() == 1 && trig_data.is_none() {
                        let is_cos = match name.as_str() {
                            "cos" => true,
                            "sin" => false,
                            _ => return None,
                        };
                        let omega = parse_linear_arg(&args[0], x)?;
                        trig_data = Some((omega, is_cos));
                    } else {
                        return None;
                    }
                } else {
                    let r = parse_rational_scalar(f)?;
                    k *= r;
                }
            }
            trig_data.map(|(w, is_cos)| (k, w, is_cos))
        }
        _ => None,
    }
}

fn solve_particular_term(
    a: &BigRational,
    b: &BigRational,
    c: &BigRational,
    term: &Expr,
    x: &Symbol,
) -> Result<Expr, SolverError> {
    let x_sym = Expr::Sym(x.clone());

    // 1. Constant term
    if let Some(k) = parse_rational_scalar(term) {
        if k.is_zero() {
            return Ok(Expr::from_i64(0));
        }
        if !c.is_zero() {
            let yp_scalar = k / c;
            return Ok(Expr::Rational(yp_scalar));
        } else if !b.is_zero() {
            let yp_scalar = k / b;
            return Ok(Expr::Mul(vec![Expr::Rational(yp_scalar), x_sym]));
        } else if !a.is_zero() {
            let yp_scalar = k / (a * BigRational::from_integer(2.into()));
            return Ok(Expr::Mul(vec![
                Expr::Rational(yp_scalar),
                Expr::Pow(Arc::new(x_sym), Arc::new(Expr::from_i64(2))),
            ]));
        } else {
            return Err(SolverError::InvalidSystem(
                "all ODE coefficients are zero".into(),
            ));
        }
    }

    // 2. Exponential term: k * exp(gamma * x)
    if let Some((k, gamma)) = parse_exponential(term, x) {
        let p_gamma = a * &gamma * &gamma + b * &gamma + c;
        let p_prime_gamma = a * BigRational::from_integer(2.into()) * &gamma + b;
        let exp_gamma_x = Expr::Function(
            "exp".into(),
            vec![Expr::Mul(vec![Expr::Rational(gamma), x_sym.clone()])],
        );

        if !p_gamma.is_zero() {
            let coeff = k / p_gamma;
            return Ok(Expr::Mul(vec![Expr::Rational(coeff), exp_gamma_x]));
        } else if !p_prime_gamma.is_zero() {
            let coeff = k / p_prime_gamma;
            return Ok(Expr::Mul(vec![Expr::Rational(coeff), x_sym, exp_gamma_x]));
        } else if !a.is_zero() {
            let coeff = k / (a * BigRational::from_integer(2.into()));
            return Ok(Expr::Mul(vec![
                Expr::Rational(coeff),
                Expr::Pow(Arc::new(x_sym), Arc::new(Expr::from_i64(2))),
                exp_gamma_x,
            ]));
        }
    }

    // 3. Trigonometric term: k * cos(omega * x) or k * sin(omega * x)
    if let Some((k, omega, is_cos)) = parse_trig(term, x) {
        if omega.is_zero() {
            return if is_cos {
                solve_particular_term(a, b, c, &Expr::Rational(k), x)
            } else {
                Ok(Expr::from_i64(0))
            };
        }
        let omega_sq = &omega * &omega;
        let d = c - a * &omega_sq;
        let e = b * &omega;
        let denom = &d * &d + &e * &e;

        let cos_wx = Expr::Function(
            "cos".into(),
            vec![Expr::Mul(vec![
                Expr::Rational(omega.clone()),
                x_sym.clone(),
            ])],
        );
        let sin_wx = Expr::Function(
            "sin".into(),
            vec![Expr::Mul(vec![
                Expr::Rational(omega.clone()),
                x_sym.clone(),
            ])],
        );

        if !denom.is_zero() {
            if is_cos {
                let c_cos = &k * &d / &denom;
                let c_sin = &k * &e / &denom;
                let mut add_terms = Vec::new();
                if !c_cos.is_zero() {
                    add_terms.push(Expr::Mul(vec![Expr::Rational(c_cos), cos_wx]));
                }
                if !c_sin.is_zero() {
                    add_terms.push(Expr::Mul(vec![Expr::Rational(c_sin), sin_wx]));
                }
                return Ok(match add_terms.len() {
                    0 => Expr::from_i64(0),
                    1 => add_terms.pop().unwrap(),
                    _ => Expr::Add(add_terms),
                });
            } else {
                let c_cos = -&k * &e / &denom;
                let c_sin = &k * &d / &denom;
                let mut add_terms = Vec::new();
                if !c_cos.is_zero() {
                    add_terms.push(Expr::Mul(vec![Expr::Rational(c_cos), cos_wx]));
                }
                if !c_sin.is_zero() {
                    add_terms.push(Expr::Mul(vec![Expr::Rational(c_sin), sin_wx]));
                }
                return Ok(match add_terms.len() {
                    0 => Expr::from_i64(0),
                    1 => add_terms.pop().unwrap(),
                    _ => Expr::Add(add_terms),
                });
            }
        } else {
            let two_a_w = a * BigRational::from_integer(2.into()) * &omega;
            if !two_a_w.is_zero() {
                if is_cos {
                    let coeff = k / two_a_w;
                    return Ok(Expr::Mul(vec![Expr::Rational(coeff), x_sym, sin_wx]));
                } else {
                    let coeff = -k / two_a_w;
                    return Ok(Expr::Mul(vec![Expr::Rational(coeff), x_sym, cos_wx]));
                }
            }
        }
    }

    Err(SolverError::IncompleteSolutionSet(format!(
        "unsupported forcing term in nonhomogeneous ODE: {term}"
    )))
}

/// Solves second-order linear non-homogeneous ODE with constant coefficients:
/// $a y''(x) + b y'(x) + c y(x) = f(x)$.
pub fn dsolve_const_coeff_second_order_nonhomogeneous(
    a: i64,
    b: i64,
    c: i64,
    f_expr: &Expr,
    x: &Symbol,
    c1: &Symbol,
    c2: &Symbol,
) -> Result<Expr, SolverError> {
    require_fresh_integration_constants(x, &[c1, c2], &[f_expr])?;
    let yh = dsolve_const_coeff_second_order(a, b, c, x, c1, c2)?;
    if f_expr.is_zero() {
        return Ok(yh);
    }

    let a_rat = BigRational::from_integer(BigInt::from(a));
    let b_rat = BigRational::from_integer(BigInt::from(b));
    let c_rat = BigRational::from_integer(BigInt::from(c));

    let terms = match f_expr {
        Expr::Add(terms) => terms.as_slice(),
        other => std::slice::from_ref(other),
    };

    let mut particular_terms = Vec::with_capacity(terms.len());
    for t in terms {
        let yp_t = solve_particular_term(&a_rat, &b_rat, &c_rat, t, x)?;
        if !yp_t.is_zero() {
            particular_terms.push(yp_t);
        }
    }

    let yp = match particular_terms.len() {
        0 => Expr::from_i64(0),
        1 => particular_terms.pop().unwrap(),
        _ => Expr::Add(particular_terms),
    };

    Ok(simplify(&Expr::Add(vec![yh, yp])))
}

/// Exact residual checker for a candidate solution of $a y'' + b y' + c y = 0$.
pub fn verify_const_coeff_second_order_solution(
    sol: &Expr,
    a: i64,
    b: i64,
    c: i64,
    x: &Symbol,
) -> bool {
    verify_const_coeff_second_order_nonhomogeneous_solution(sol, a, b, c, &Expr::from_i64(0), x)
}

/// Exact residual checker for a candidate solution of $a y'' + b y' + c y = f(x)$.
pub fn verify_const_coeff_second_order_nonhomogeneous_solution(
    sol: &Expr,
    a: i64,
    b: i64,
    c: i64,
    f_expr: &Expr,
    x: &Symbol,
) -> bool {
    if !ode_inputs_are_total(&[sol, f_expr]) {
        return false;
    }
    let mut terms = Vec::with_capacity(4);
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
    if !f_expr.is_zero() {
        terms.push(Expr::Mul(vec![Expr::from_i64(-1), f_expr.clone()]));
    }

    residual_is_exact_zero(&Expr::Add(terms))
}

/// Exact residual checker for a candidate solution of a first-order linear ODE: $y'(x) + P(x) y(x) = Q(x)$.
pub fn verify_linear_first_order_solution(
    sol: &Expr,
    p_expr: &Expr,
    q_expr: &Expr,
    x: &Symbol,
) -> bool {
    if !ode_inputs_are_total(&[sol, p_expr, q_expr]) {
        return false;
    }
    let dy = diff(sol, x);
    let py = Expr::Mul(vec![p_expr.clone(), sol.clone()]);
    let mut terms = vec![dy, py];
    if !q_expr.is_zero() {
        terms.push(Expr::Mul(vec![Expr::from_i64(-1), q_expr.clone()]));
    }
    residual_is_exact_zero(&Expr::Add(terms))
}
