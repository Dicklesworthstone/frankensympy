//! Elementary PDE modes, exact resonance, original residuals, and mutants.

use fsym_budget::Unbounded;
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::pde::{
    LinearFirstOrderPdeCertificate, PdeCompleteness, PdeError, linear_first_order_pde,
    verify_linear_first_order_pde,
};
use std::sync::Arc;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn normalize_circular_parity(expression: &Expr) -> Expr {
    // Construction deliberately leaves balanced sums such as x-y and y-x
    // unoriented. For this independent residual check, use only the exact
    // identities sin(-u)=-sin(u), cos(-u)=cos(u). Structural ordering chooses
    // a representative; it asserts nothing about the mathematical sign of u.
    match expression {
        Expr::Add(children) => Expr::Add(children.iter().map(normalize_circular_parity).collect()),
        Expr::Mul(children) => Expr::Mul(children.iter().map(normalize_circular_parity).collect()),
        Expr::Pow(base, exponent) => Expr::Pow(
            Arc::new(normalize_circular_parity(base)),
            Arc::new(normalize_circular_parity(exponent)),
        ),
        Expr::Function(name, arguments) => {
            let arguments: Vec<_> = arguments.iter().map(normalize_circular_parity).collect();
            if matches!(name.as_str(), "sin" | "cos") && arguments.len() == 1 {
                let argument =
                    fsym_simplify::try_simplify(&fsym_simplify::try_expand(&arguments[0]).unwrap())
                        .unwrap();
                let opposite = fsym_core::elementary::negate(&argument);
                let opposite =
                    fsym_simplify::try_simplify(&fsym_simplify::try_expand(&opposite).unwrap())
                        .unwrap();
                if opposite < argument {
                    let opposite_function = Expr::Function(name.clone(), vec![opposite]);
                    return if name == "sin" {
                        -opposite_function
                    } else {
                        opposite_function
                    };
                }
                return Expr::Function(name.clone(), vec![argument]);
            }
            Expr::Function(name.clone(), arguments)
        }
        _ => expression.clone(),
    }
}

fn exact_equal(actual: &Expr, expected: &Expr) {
    let difference = actual.clone() - expected.clone();
    let expanded = fsym_simplify::try_expand(&difference).unwrap();
    let mut reduced = fsym_simplify::try_simplify(&expanded).unwrap();
    if !reduced.is_zero() {
        let normalized = normalize_circular_parity(&reduced);
        reduced =
            fsym_simplify::try_simplify(&fsym_simplify::try_expand(&normalized).unwrap()).unwrap();
    }
    assert!(
        reduced.is_zero(),
        "{actual} != {expected}; residual {reduced}"
    );
}

fn original_residual(
    solution: &Expr,
    coefficients: [&BigRational; 3],
    forcing: &Expr,
    x: &Symbol,
    y: &Symbol,
) {
    // This differentiation/simplification path is independent of the PDE
    // mode generator and the scalar recurrence used by its verifier.
    let residual = rational_expr(coefficients[0].clone()) * fsym_calculus::diff(solution, x)
        + rational_expr(coefficients[1].clone()) * fsym_calculus::diff(solution, y)
        + rational_expr(coefficients[2].clone()) * solution.clone()
        - forcing.clone();
    exact_equal(&residual, &Expr::from_i64(0));
}

fn check_case(coefficients: [BigRational; 3], source: &str, expected: &str) {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let forcing = e(source);
    let [a, b, c] = coefficients;
    let accepted = linear_first_order_pde(&a, &b, &c, &forcing, &x, &y).unwrap();
    exact_equal(accepted.particular(), &e(expected));
    original_residual(accepted.particular(), [&a, &b, &c], &forcing, &x, &y);
    assert_eq!(accepted.forcing(), &forcing);
    assert_eq!(
        accepted.completeness(),
        PdeCompleteness::VerifiedSolutionFamily
    );
}

#[test]
fn coupled_exponential_trigonometric_amplitudes_recover_a_hand_solution() {
    // D=dx+2dy, exponential directional rate 0, frequency 4, reaction 3.
    // The chosen amplitudes (x,y) give (1+3x+4y, 2+3y-4x).
    check_case(
        [q(1, 1), q(2, 1), q(3, 1)],
        "exp(2*x-y+1)*((1+3*x+4*y)*cos(2*x+y+2)+(2+3*y-4*x)*sin(2*x+y+2))",
        "exp(2*x-y+1)*(x*cos(2*x+y+2)+y*sin(2*x+y+2))",
    );
}

#[test]
fn pure_exponentials_and_pure_trigonometric_forcing_are_supported() {
    // exp(1) canonically becomes the built-in constant E in public objects.
    check_case([q(1, 1), q(0, 1), q(0, 1)], "(1+E)*cos(x)", "(1+E)*sin(x)");
    check_case(
        [q(2, 1), q(1, 1), q(2, 1)],
        "x*exp(2*x-3*y+4)",
        "(x/3-2/9)*exp(2*x-3*y+4)",
    );
    check_case([q(1, 1), q(0, 1), q(0, 1)], "cos(x)", "sin(x)");
    check_case([q(1, 1), q(0, 1), q(0, 1)], "sin(x)", "-cos(x)");
    check_case(
        [q(1, 1), q(0, 1), q(2, 1)],
        "cos(x)+sin(x)",
        "(cos(x)+3*sin(x))/5",
    );
}

#[test]
fn resonant_exponential_and_invariant_trigonometric_modes_integrate() {
    // Both effective reaction and phase frequency vanish. A matrix inverse
    // would be singular; the two polynomial transport integrals are needed.
    check_case(
        [q(1, 1), q(1, 1), q(2, 1)],
        "exp(-x-y+1)*(x*cos(x-y+3)+y*sin(x-y+3))",
        "exp(-x-y+1)*(x**2*cos(x-y+3)/2+(x*y-x**2/2)*sin(x-y+3))",
    );
    check_case([q(1, 1), q(1, 1), q(2, 1)], "exp(-x-y)", "x*exp(-x-y)");
}

#[test]
fn single_axis_resonance_and_rational_rates_keep_exact_coefficients() {
    check_case(
        [q(0, 1), q(2, 1), q(-1, 1)],
        "x*y*exp(x+y/2+2)*cos(3*x+1)",
        "x*y**2*exp(x+y/2+2)*cos(3*x+1)/4",
    );
    // With D=dx/2-2dy/3, E=x/3+y/4 and T=2x+3y, the effective
    // reaction is 3/5 and the frequency is -1. Amplitudes (x,y) give
    // (1/2+3x/5-y, -2/3+3y/5+x).
    check_case(
        [q(1, 2), q(-2, 3), q(3, 5)],
        "exp(x/3+y/4)*((1/2+3*x/5-y)*cos(2*x+3*y)+(-2/3+3*y/5+x)*sin(2*x+3*y))",
        "exp(x/3+y/4)*(x*cos(2*x+3*y)+y*sin(2*x+3*y))",
    );
}

#[test]
fn mixed_modes_product_exponentials_and_signed_phases_combine() {
    check_case(
        [q(1, 1), q(1, 1), q(1, 1)],
        "x+y+x*exp(x)*exp(-y)+2*sin(y-x)",
        "x+y-2+(x-1)*exp(x-y)-2*sin(x-y)",
    );
    check_case(
        [q(1, 1), q(0, 1), q(0, 1)],
        "exp(2)*(x*cos(-3)+y*sin(-3))",
        "exp(2)*(x**2*cos(3)/2-x*y*sin(3))",
    );
    check_case([q(1, 1), q(0, 1), q(1, 1)], "exp(0)*cos(0)+sin(0)", "1");
}

#[test]
fn original_exponential_rates_phases_offsets_and_pairing_are_binding() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b, c) = (q(1, 1), q(2, 1), q(3, 1));
    let forcing = e("exp(2*x-y+1)*((1+3*x+4*y)*cos(2*x+y+2)+(2+3*y-4*x)*sin(2*x+y+2))");
    let accepted = linear_first_order_pde(&a, &b, &c, &forcing, &x, &y).unwrap();
    let epsilon = rational_expr(BigRational::new(BigInt::one(), BigInt::from(10).pow(90)));
    for mutation in [
        e("exp(3*x-y+1)*(x*cos(2*x+y+2)+y*sin(2*x+y+2))"),
        e("exp(2*x-y+2)*(x*cos(2*x+y+2)+y*sin(2*x+y+2))"),
        e("exp(2*x-y+1)*(x*cos(3*x+y+2)+y*sin(3*x+y+2))"),
        e("exp(2*x-y+1)*(x*cos(2*x+y+3)+y*sin(2*x+y+3))"),
        e("exp(2*x-y+1)*(x*cos(2*x+y+2)-y*sin(2*x+y+2))"),
        accepted.particular().clone() + epsilon * e("exp(2*x-y+1)*cos(2*x+y+2)"),
    ] {
        let mut certificate = accepted.certificate().clone();
        certificate.particular = mutation;
        assert!(
            verify_linear_first_order_pde(
                [&a, &b, &c],
                &forcing,
                [&x, &y],
                &certificate,
                &mut Unbounded,
            )
            .is_err(),
        );
    }
    let altered_source = e("exp(2*x-y+2)*((1+3*x+4*y)*cos(2*x+y+2)+(2+3*y-4*x)*sin(2*x+y+2))");
    assert!(
        verify_linear_first_order_pde(
            [&a, &b, &c],
            &altered_source,
            [&x, &y],
            accepted.certificate(),
            &mut Unbounded,
        )
        .is_err(),
    );
}

#[test]
fn checker_accepts_independently_supplied_homogeneous_elementary_additions() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let (a, b, c) = (q(1, 1), q(1, 1), q(2, 1));
    let forcing = e("exp(x-y)");
    let supplied = LinearFirstOrderPdeCertificate {
        time_coordinate: e("x"),
        invariant_coordinate: e("y-x"),
        particular: e("exp(x-y)/2+exp(-x-y)*(cos(x-y)+sin(x-y))"),
    };
    let accepted =
        verify_linear_first_order_pde([&a, &b, &c], &forcing, [&x, &y], &supplied, &mut Unbounded)
            .unwrap();
    original_residual(accepted.particular(), [&a, &b, &c], &forcing, &x, &y);
    let mut invalid = supplied;
    invalid.particular = e("exp(x-y)/2+exp(-x-2*y)*(cos(x-y)+sin(x-y))");
    assert!(
        verify_linear_first_order_pde([&a, &b, &c], &forcing, [&x, &y], &invalid, &mut Unbounded,)
            .is_err(),
    );
}

#[test]
fn original_held_elementary_domains_and_grammar_are_admitted_before_cancellation() {
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let zero = Expr::from_i64(0);
    let bad_phase = function("sin", power(e("x"), 2));
    let certificate = LinearFirstOrderPdeCertificate {
        time_coordinate: e("x"),
        invariant_coordinate: e("y"),
        particular: zero.clone(),
    };
    for invalid in [
        bad_phase.clone(),
        Expr::Add(vec![bad_phase.clone(), Expr::Mul(vec![e("-1"), bad_phase])]),
        function("exp", power(e("x"), -1)),
        function("cos", Expr::Sym(Symbol::new("foreign"))),
        function("sin", function("exp", e("x"))),
        function("exp", e("E*x")),
        function("tan", e("x")),
        Expr::Function("exp".into(), Vec::new()),
        Expr::Function("sin".into(), vec![e("x"), e("y")]),
        power(function("exp", e("x")), 0),
        power(function("cos", e("x")), 2),
        power(Expr::Add(vec![e("1"), function("sin", e("x"))]), 0),
        Expr::Mul(vec![zero.clone(), function("exp", power(e("x"), -1))]),
        Expr::Mul(vec![zero, function("sin", e("x")), function("sin", e("y"))]),
        Expr::Mul(vec![function("cos", e("x")), function("sin", e("y"))]),
    ] {
        assert!(
            linear_first_order_pde(&q(1, 1), &q(0, 1), &q(0, 1), &invalid, &x, &y,).is_err(),
            "admitted unsupported original forcing: {invalid}",
        );
        assert!(
            verify_linear_first_order_pde(
                [&q(1, 1), &q(0, 1), &q(0, 1)],
                &invalid,
                [&x, &y],
                &certificate,
                &mut Unbounded,
            )
            .is_err(),
        );
    }
}

#[test]
fn degree_and_mode_count_bounds_cover_the_boundary_and_refuse_excess() {
    check_case(
        [q(1, 1), q(1, 1), q(0, 1)],
        "x**16*exp(x-y)*cos(x-y)",
        "x**17*exp(x-y)*cos(x-y)/17",
    );
    let (x, y) = (Symbol::new("x"), Symbol::new("y"));
    let forcing = Expr::Add((1..=16).map(|rate| e(&format!("exp({rate}*x)"))).collect());
    let accepted = linear_first_order_pde(&q(1, 1), &q(0, 1), &q(1, 1), &forcing, &x, &y).unwrap();
    let expected = Expr::Add(
        (1..=16)
            .map(|rate| e(&format!("exp({rate}*x)/{}", rate + 1)))
            .collect(),
    );
    exact_equal(accepted.particular(), &expected);
    let too_many = Expr::Add((1..=17).map(|rate| e(&format!("exp({rate}*x)"))).collect());
    for invalid in [e("x**17*exp(x-y)"), too_many] {
        assert!(matches!(
            linear_first_order_pde(&q(1, 1), &q(0, 1), &q(1, 1), &invalid, &x, &y,),
            Err(PdeError::ResourceLimit(_)),
        ));
    }
    let huge = Expr::Integer(BigInt::one() << 4097u32);
    let invalid = function("exp", Expr::Mul(vec![huge, e("x")]));
    assert!(matches!(
        linear_first_order_pde(&q(1, 1), &q(0, 1), &q(1, 1), &invalid, &x, &y,),
        Err(PdeError::ResourceLimit(_)),
    ));
}

#[test]
fn same_name_typed_axes_are_preserved_inside_elementary_arguments() {
    let x = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([2; 32]));
    let (xe, ye) = (Expr::Sym(x.clone()), Expr::Sym(y.clone()));
    let common = Expr::Mul(vec![function("exp", ye.clone()), function("sin", ye)]);
    let forcing = Expr::Mul(vec![xe.clone(), common.clone()]);
    let accepted = linear_first_order_pde(&q(1, 1), &q(0, 1), &q(0, 1), &forcing, &x, &y).unwrap();
    let expected = rational_expr(q(1, 2)) * power(xe, 2) * common;
    exact_equal(accepted.particular(), &expected);
    original_residual(
        accepted.particular(),
        [&q(1, 1), &q(0, 1), &q(0, 1)],
        &forcing,
        &x,
        &y,
    );
    assert_eq!(accepted.variables(), &[x, y]);
}
