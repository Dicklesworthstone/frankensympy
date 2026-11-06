//! Exact convention, convergence, source-domain, identity and budget checks.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::fourier::{FourierError, fourier_transform_l1, metered_fourier_transform_l1};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_simplify::simplify;
use std::collections::HashMap;
use std::sync::Arc;

fn raw_power(base: Expr, power: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(power)))
}

fn transform(source: &str) -> Expr {
    fourier_transform_l1(
        &parse(source).unwrap(),
        &Symbol::new("x"),
        &Symbol::new("k"),
    )
    .unwrap()
}

fn at_frequency(value: &Expr, frequency: &str) -> Expr {
    simplify(&value.subs(&HashMap::from([(
        Symbol::new("k"),
        parse(frequency).unwrap(),
    )])))
}

fn equal(left: Expr, right: Expr) {
    assert_eq!(simplify(&(left - right)), Expr::from_i64(0));
}

#[test]
fn bilateral_decay_uses_cycles_per_unit_and_exact_rational_rates() {
    // k=1/(2*pi) has angular frequency exactly one. These independent
    // rational values distinguish this convention from the angular API.
    for (source, mass, unit_frequency) in [
        ("exp(-abs(x))", "2", "1"),
        ("exp(-2*abs(x))", "1", "4/5"),
        ("exp(-abs(x)/3)", "6", "3/5"),
        ("3*exp(-abs(2*x))", "3", "12/5"),
        ("exp(-2*abs(-3*x))", "1/3", "12/37"),
    ] {
        let result = transform(source);
        equal(at_frequency(&result, "0"), parse(mass).unwrap());
        equal(
            at_frequency(&result, "1/(2*pi)"),
            parse(unit_frequency).unwrap(),
        );
        equal(
            at_frequency(&result, "-1/(2*pi)"),
            parse(unit_frequency).unwrap(),
        );
    }
}

#[test]
fn gaussian_mass_is_invariant_under_exact_translations() {
    for source in ["exp(-x^2)", "exp(-(x-3)^2)", "exp(-(x+2/3)^2)"] {
        equal(
            at_frequency(&transform(source), "0"),
            parse("pi^(1/2)").unwrap(),
        );
    }
    equal(
        at_frequency(&transform("exp(-4*x^2)"), "0"),
        parse("pi^(1/2)/2").unwrap(),
    );
    equal(
        at_frequency(&transform("exp(-x^2/4)"), "0"),
        parse("2*pi^(1/2)").unwrap(),
    );
}

#[test]
fn translated_gaussians_and_decays_have_the_negative_fourier_phase() {
    for (source, unshifted, phase, positive_phase, negative_phase) in [
        (
            "exp(-(x-2)^2)",
            "exp(-x^2)",
            "exp(-4*pi*I*k)",
            "exp(-2*I)",
            "exp(2*I)",
        ),
        (
            "exp(-3*abs(x-2))",
            "exp(-3*abs(x))",
            "exp(-4*pi*I*k)",
            "exp(-2*I)",
            "exp(2*I)",
        ),
        (
            "exp(-(x+2/3)^2)",
            "exp(-x^2)",
            "exp(4*pi*I*k/3)",
            "exp(2*I/3)",
            "exp(-2*I/3)",
        ),
        (
            "exp(-3*abs(x+2/3))",
            "exp(-3*abs(x))",
            "exp(4*pi*I*k/3)",
            "exp(2*I/3)",
            "exp(-2*I/3)",
        ),
    ] {
        let shifted = transform(source);
        let unshifted = transform(unshifted);
        equal(shifted.clone(), parse(phase).unwrap() * unshifted.clone());
        // These are exactly the original angular-frequency points +/-1.
        // Factoring 1/(2*pi) as pi^(-1)/2 lets the arithmetic simplifier
        // cancel pi without requiring it to distribute a product inverse
        // inside an exponential. The expected phase remains unchanged.
        for (frequency, phase) in [
            ("pi^(-1)/2", positive_phase),
            ("-pi^(-1)/2", negative_phase),
        ] {
            equal(
                at_frequency(&shifted, frequency),
                parse(phase).unwrap() * at_frequency(&unshifted, frequency),
            );
        }
    }
}

#[test]
fn gaussian_frequency_decay_satisfies_its_differential_identity() {
    let frequency = Symbol::new("k");
    for (source, slope) in [("exp(-x^2)", "2*pi^2*k"), ("exp(-2*x^2)", "pi^2*k")] {
        let value = transform(source);
        let residual = fsym_calculus::diff(&value, &frequency) + parse(slope).unwrap() * value;
        equal(residual, Expr::from_i64(0));
    }
}

#[test]
fn finite_linearity_keeps_all_summands_and_scalar_signs() {
    let actual = transform("3*exp(-x^2)-2*exp(-abs(x))");
    equal(at_frequency(&actual, "0"), parse("3*pi^(1/2)-4").unwrap());
    equal(
        transform("2*(exp(-x^2)+exp(-abs(x)))"),
        Expr::from_i64(2) * (transform("exp(-x^2)") + transform("exp(-abs(x))")),
    );
}

#[test]
fn constants_do_not_invent_an_ordinary_integral_or_a_distribution() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    assert_eq!(
        fourier_transform_l1(&Expr::from_i64(0), &x, &k).unwrap(),
        Expr::from_i64(0),
    );
    for n in [-5, 1, 5] {
        assert!(matches!(
            fourier_transform_l1(&Expr::from_i64(n), &x, &k),
            Err(FourierError::Unsupported(_)),
        ));
    }
    // The explicitly angular native catalog is a separate unchanged API.
    assert!(
        fsym_calculus::fourier_transform(&Expr::from_i64(5), &x, &k)
            .unwrap()
            .to_string()
            .contains("dirac"),
    );
}

#[test]
fn exact_convergence_premises_cannot_be_replaced_by_candidate_shape() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    for source in [
        "exp(x^2)",
        "exp(x)",
        "exp(abs(x))",
        "exp(-a*x^2)",
        "exp(-a*abs(x))",
        "exp(-abs(x^2))",
        "exp(-x^4)",
        "exp(-abs(x)-abs(x-1))",
        "exp(-x^2)*sin(x)",
        "exp(-x^2)+sin(x)",
        "exp(-x^2)+1",
    ] {
        assert!(
            matches!(
                fourier_transform_l1(&parse(source).unwrap(), &x, &k),
                Err(FourierError::Unsupported(_)),
            ),
            "source {source}",
        );
    }
}

#[test]
fn original_zero_products_and_powers_do_not_hide_invalid_children() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    for power in [-1, 0] {
        let invalid = Expr::Mul(vec![Expr::from_i64(0), raw_power(Expr::from_i64(0), power)]);
        assert!(matches!(
            fourier_transform_l1(&invalid, &x, &k),
            Err(FourierError::Refused(_)),
        ));
    }
    for hidden in [
        parse("sin(x)").unwrap(),
        parse("exp(x^2)").unwrap(),
        raw_power(Expr::Sym(x.clone()), -1),
    ] {
        let source = Expr::Mul(vec![Expr::from_i64(0), hidden]);
        assert!(matches!(
            fourier_transform_l1(&source, &x, &k),
            Err(FourierError::Unsupported(_)),
        ));
    }
    let held = Expr::Function(
        "exp".into(),
        vec![Expr::Add(vec![
            parse("-x^2").unwrap(),
            Expr::Mul(vec![
                Expr::from_i64(0),
                raw_power(raw_power(Expr::from_i64(0), -1), 0),
            ]),
        ])],
    );
    assert!(matches!(
        fourier_transform_l1(&held, &x, &k),
        Err(FourierError::Refused(_)),
    ));
}

#[test]
fn typed_symbols_can_print_identically_without_becoming_the_same_variable() {
    let x = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([1; 32]));
    let k = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = Expr::Function(
        "exp".into(),
        vec![Expr::Mul(vec![
            Expr::from_i64(-1),
            raw_power(Expr::Sym(x.clone()), 2),
        ])],
    );
    let value = fourier_transform_l1(&source, &x, &k).unwrap();
    let symbols = value.free_symbols();
    assert_eq!(symbols.len(), 1);
    assert!(symbols.contains(&k));
    assert!(matches!(
        fourier_transform_l1(&source, &x, &x),
        Err(FourierError::Refused(_)),
    ));
    let contaminated = Expr::Add(vec![
        source,
        Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(k.clone())]),
    ]);
    assert!(matches!(
        fourier_transform_l1(&contaminated, &x, &k),
        Err(FourierError::Refused(_)),
    ));
}

#[test]
fn original_size_and_scalar_limits_are_terminal() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let large = Expr::Integer(BigInt::from(1) << 4097u32);
    let deep = (0..66).fold(Expr::Sym(x.clone()), |node, _| {
        Expr::Function("exp".into(), vec![node])
    });
    let wide = Expr::Add(vec![Expr::from_i64(0); 4097]);
    let power = raw_power(Expr::Sym(x.clone()), 1_000_000);
    for source in [large, deep, wide, power, Expr::Const(Constant::NaN)] {
        assert!(matches!(
            fourier_transform_l1(&source, &x, &k),
            Err(FourierError::Refused(_)),
        ));
    }
    let many = Expr::Add(vec![parse("exp(-x^2)").unwrap(); 65]);
    assert!(matches!(
        fourier_transform_l1(&many, &x, &k),
        Err(FourierError::Refused(_)),
    ));
}

struct CancelAfter(usize);

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> std::result::Result<(), MeterError> {
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> std::result::Result<(), MeterError> {
        Ok(())
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if self.0 == 0 {
            return Err(MeterError::Cancelled);
        }
        self.0 -= 1;
        Ok(())
    }
}

#[test]
fn caller_exhaustion_and_cancellation_never_become_unsupported_results() {
    let source = parse("exp(-2*x^2+3*x+1)+exp(-2*abs(3*x+1)+4)").unwrap();
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_fourier_transform_l1(&source, &x, &k, &mut budget),
        Err(FourierError::Meter(_)),
    ));
    for checkpoint in [0, 1, 10, 100] {
        assert!(matches!(
            metered_fourier_transform_l1(&source, &x, &k, &mut CancelAfter(checkpoint)),
            Err(FourierError::Meter(MeterError::Cancelled)),
        ));
    }
    let result = metered_fourier_transform_l1(&source, &x, &k, &mut Unbounded).unwrap();
    assert!(result.used_bilateral_identity);
    assert!(
        !metered_fourier_transform_l1(&parse("exp(-x^2)").unwrap(), &x, &k, &mut Unbounded,)
            .unwrap()
            .used_bilateral_identity,
    );
}

#[test]
fn exact_small_rates_do_not_get_rounded_to_nondecay() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let denominator = BigInt::from(1) << 200u32;
    let rate = BigRational::new(1.into(), denominator.clone());
    let source = Expr::Function(
        "exp".into(),
        vec![Expr::Mul(vec![
            Expr::Rational(-rate),
            raw_power(Expr::Sym(x.clone()), 2),
        ])],
    );
    let value = fourier_transform_l1(&source, &x, &k).unwrap();
    let mass = at_frequency(&value, "0");
    let expected_squared = Expr::Integer(denominator) * Expr::Const(Constant::Pi);
    equal(
        fsym_core::elementary::eval_pow(mass, Expr::from_i64(2)),
        expected_squared,
    );
}
