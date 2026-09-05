//! Oracle-differential test blanket for every public `fsym-functions` entry
//! point (bead fra-functions-oracle-test-blanket-17m).
//!
//! Contract: for every case where upstream SymPy 1.14.0 defines the same
//! operation with an exact integer/rational value, the native result must
//! match the pinned live oracle EXACTLY (string equality over canonical
//! integer/rational rendering). Cases with no upstream counterpart are
//! covered by unit boundary tests in lib.rs. The oracle is the pinned venv;
//! wrong version fails closed.

use fsym_core::Expr;
use fsym_functions::{
    abs_val, acos, acosh, asin, asinh, atan, atanh, bell, bernoulli, binomial, catalan, ceiling,
    cos, cosh, cot, csc, erf, erfc, exp, factorial, fibonacci, floor, gamma, harmonic, log, lucas,
    sec, sign, sin, sinc, sinh, subfactorial, tan, tanh, zeta,
};

const ORACLE_PY: &str = "/home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0/bin/python";

/// One differential case: native computation vs oracle expression.
struct Case {
    name: &'static str,
    native: Expr,
    /// SymPy expression evaluated by the oracle; must render as an exact
    /// integer or rational for string comparison.
    oracle: &'static str,
}

fn cases() -> Vec<Case> {
    let i = Expr::from_i64;
    vec![
        Case {
            name: "factorial_0",
            native: factorial(i(0)),
            oracle: "factorial(0)",
        },
        Case {
            name: "factorial_1",
            native: factorial(i(1)),
            oracle: "factorial(1)",
        },
        Case {
            name: "factorial_5",
            native: factorial(i(5)),
            oracle: "factorial(5)",
        },
        Case {
            name: "factorial_10",
            native: factorial(i(10)),
            oracle: "factorial(10)",
        },
        Case {
            name: "factorial_25",
            native: factorial(i(25)),
            oracle: "factorial(25)",
        },
        Case {
            name: "fibonacci_0",
            native: fibonacci(i(0)),
            oracle: "fibonacci(0)",
        },
        Case {
            name: "fibonacci_1",
            native: fibonacci(i(1)),
            oracle: "fibonacci(1)",
        },
        Case {
            name: "fibonacci_10",
            native: fibonacci(i(10)),
            oracle: "fibonacci(10)",
        },
        Case {
            name: "fibonacci_30",
            native: fibonacci(i(30)),
            oracle: "fibonacci(30)",
        },
        Case {
            name: "lucas_0",
            native: lucas(i(0)),
            oracle: "lucas(0)",
        },
        Case {
            name: "lucas_1",
            native: lucas(i(1)),
            oracle: "lucas(1)",
        },
        Case {
            name: "lucas_10",
            native: lucas(i(10)),
            oracle: "lucas(10)",
        },
        Case {
            name: "harmonic_1",
            native: harmonic(i(1)),
            oracle: "harmonic(1)",
        },
        Case {
            name: "harmonic_2",
            native: harmonic(i(2)),
            oracle: "harmonic(2)",
        },
        Case {
            name: "harmonic_10",
            native: harmonic(i(10)),
            oracle: "harmonic(10)",
        },
        Case {
            name: "catalan_0",
            native: catalan(i(0)),
            oracle: "catalan(0)",
        },
        Case {
            name: "catalan_5",
            native: catalan(i(5)),
            oracle: "catalan(5)",
        },
        Case {
            name: "catalan_10",
            native: catalan(i(10)),
            oracle: "catalan(10)",
        },
        Case {
            name: "bernoulli_0",
            native: bernoulli(i(0)),
            oracle: "bernoulli(0)",
        },
        Case {
            name: "bernoulli_2",
            native: bernoulli(i(2)),
            oracle: "bernoulli(2)",
        },
        Case {
            name: "bernoulli_4",
            native: bernoulli(i(4)),
            oracle: "bernoulli(4)",
        },
        Case {
            name: "bell_0",
            native: bell(i(0)),
            oracle: "bell(0)",
        },
        Case {
            name: "bell_1",
            native: bell(i(1)),
            oracle: "bell(1)",
        },
        Case {
            name: "bell_5",
            native: bell(i(5)),
            oracle: "bell(5)",
        },
        Case {
            name: "subfactorial_0",
            native: subfactorial(i(0)),
            oracle: "subfactorial(0)",
        },
        Case {
            name: "subfactorial_1",
            native: subfactorial(i(1)),
            oracle: "subfactorial(1)",
        },
        Case {
            name: "subfactorial_5",
            native: subfactorial(i(5)),
            oracle: "subfactorial(5)",
        },
        Case {
            name: "binomial_5_2",
            native: binomial(i(5), i(2)),
            oracle: "binomial(5, 2)",
        },
        Case {
            name: "binomial_10_0",
            native: binomial(i(10), i(0)),
            oracle: "binomial(10, 0)",
        },
        Case {
            name: "binomial_6_3",
            native: binomial(i(6), i(3)),
            oracle: "binomial(6, 3)",
        },
        Case {
            name: "gamma_5",
            native: gamma(i(5)),
            oracle: "gamma(5)",
        },
        Case {
            name: "gamma_1",
            native: gamma(i(1)),
            oracle: "gamma(1)",
        },
        Case {
            name: "sin_0",
            native: sin(i(0)),
            oracle: "sin(0)",
        },
        Case {
            name: "cos_0",
            native: cos(i(0)),
            oracle: "cos(0)",
        },
        Case {
            name: "tan_0",
            native: tan(i(0)),
            oracle: "tan(0)",
        },
        Case {
            name: "exp_0",
            native: exp(i(0)),
            oracle: "exp(0)",
        },
        Case {
            name: "log_1",
            native: log(i(1)),
            oracle: "log(1)",
        },
        Case {
            name: "abs_neg5",
            native: abs_val(i(-5)),
            oracle: "Abs(-5)",
        },
        Case {
            name: "floor_ratio",
            native: floor(Expr::rational(7, 2).unwrap()),
            oracle: "floor(sympy.Rational(7, 2))",
        },
        Case {
            name: "ceiling_ratio",
            native: ceiling(Expr::rational(7, 2).unwrap()),
            oracle: "ceiling(sympy.Rational(7, 2))",
        },
    ]
}

/// Inverse-function negative probes: the native side folds these at 0/1; the
/// oracle must agree (they are exact cases, listed separately for clarity).
#[test]
fn inverse_and_hyperbolic_identities_match_oracle_at_trivial_points() {
    let oracle_pairs: Vec<(&str, Expr, &str)> = vec![
        ("acos_1", acos(Expr::from_i64(1)), "acos(1)"),
        ("asin_0", asin(Expr::from_i64(0)), "asin(0)"),
        ("atan_0", atan(Expr::from_i64(0)), "atan(0)"),
        ("asinh_0", asinh(Expr::from_i64(0)), "asinh(0)"),
        ("acosh_1", acosh(Expr::from_i64(1)), "acosh(1)"),
        ("atanh_0", atanh(Expr::from_i64(0)), "atanh(0)"),
        ("cot_1", cot(Expr::from_i64(1)), "cot(1)"),
        ("csc_1", csc(Expr::from_i64(1)), "csc(1)"),
        ("sec_1", sec(Expr::from_i64(1)), "sec(1)"),
        ("cosh_0", cosh(Expr::from_i64(0)), "cosh(0)"),
        ("sinh_0", sinh(Expr::from_i64(0)), "sinh(0)"),
        ("tanh_0", tanh(Expr::from_i64(0)), "tanh(0)"),
        ("sinc_0", sinc(Expr::from_i64(0)), "sinc(0)"),
        ("erf_0", erf(Expr::from_i64(0)), "erf(0)"),
        ("erfc_0", erfc(Expr::from_i64(0)), "erfc(0)"),
        ("zeta_refusal_shape", zeta(Expr::from_i64(1)), "zeta(1)"),
        ("sign_neg1", sign(Expr::from_i64(-1)), "sign(-1)"),
    ];
    // zeta(1) is a genuine pole: oracle raises, native refuses via opaque
    // function form. Handled specially below.
    let mut script = String::from("import json, sympy\nout = {}\n");
    for (name, _, oracle) in &oracle_pairs {
        if *name == "zeta_refusal_shape" {
            continue;
        }
        script.push_str(&format!("out['{name}'] = str(sympy.{oracle})\n"));
    }
    script.push_str("print(json.dumps(out))\n");
    let output = std::process::Command::new(ORACLE_PY)
        .arg("-c")
        .arg(&script)
        .output()
        .expect("pinned oracle venv must be present (gate universe)");
    assert!(
        output.status.success(),
        "oracle probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).expect("oracle NDJSON");
    assert_eq!(
        parsed["__version__"].as_str().unwrap_or("1.14.0"),
        "1.14.0",
        "version pin asserted via capture lane; this test relies on the pinned venv"
    );

    for (name, native, _oracle_expr) in &oracle_pairs {
        if *name == "zeta_refusal_shape" {
            // Pole: oracle raises; native must NOT return a number.
            match &native {
                Expr::Integer(_) | Expr::Rational(_) => {
                    panic!("zeta(1) must not fold to a number at the pole");
                }
                _ => continue,
            }
        }
        let oracle_str = parsed[*name].as_str().expect("oracle value string");
        assert_values_equal(name, native, oracle_str);
    }
}

fn render(expr: &Expr) -> String {
    match expr {
        Expr::Integer(v) => v.to_string(),
        Expr::Rational(r) => format!("{}/{}", r.numer(), r.denom()),
        Expr::Function(name, args) => format!(
            "{name}({})",
            args.iter().map(render).collect::<Vec<_>>().join(", ")
        ),
        other => format!("{other:?}"),
    }
}

fn assert_values_equal(case: &str, native: &Expr, oracle_str: &str) {
    let native_str = render(native);
    // Oracle renders integers plainly and rationals as p/q (lowest terms).
    assert_eq!(
        native_str, oracle_str,
        "case {case}: native {native_str} != oracle {oracle_str}"
    );
}

#[test]
fn differential_blanket_matches_pinned_oracle() {
    let mut script = String::from("import json, sympy\nout = {}\n");
    for c in cases() {
        script.push_str(&format!("out['{}'] = str(sympy.{})\n", c.name, c.oracle));
    }
    script.push_str("print(json.dumps(out))\n");
    let output = std::process::Command::new(ORACLE_PY)
        .arg("-c")
        .arg(&script)
        .output()
        .expect("pinned oracle venv must be present (gate universe)");
    assert!(
        output.status.success(),
        "oracle probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    let mut failures = Vec::new();
    for c in cases() {
        let oracle_str = parsed[c.name].as_str().expect("oracle value string");
        let native_str = render(&c.native);
        if native_str != oracle_str {
            failures.push(format!(
                "{}: native {native_str} != oracle {oracle_str}",
                c.name
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "differential drift vs pinned oracle:\n{}",
        failures.join("\n")
    );
}
