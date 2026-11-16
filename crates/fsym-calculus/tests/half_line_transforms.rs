//! Native normalized half-line convention and original-source regressions.
//!
//! Exponential values use exact Laplace moments and their frequency
//! derivatives; Gaussian values use mass/differential identities. These
//! tests do not promote direct catalog identities to portable certificates.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::fourier::{FourierError, HalfLineKernel, metered_half_line_transform};
use fsym_core::{BigInt, BigRational, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_simplify::simplify;
use std::collections::HashMap;
use std::sync::Arc;

fn checked(source: &Expr, kernel: HalfLineKernel) -> Result<Expr, FourierError> {
    Ok(metered_half_line_transform(
        source,
        &Symbol::new("x"),
        &Symbol::new("k"),
        kernel,
        &mut Unbounded,
    )?
    .value)
}

fn transform(source: &str, kernel: HalfLineKernel) -> Expr {
    checked(&parse(source).unwrap(), kernel).unwrap()
}

fn equal(left: Expr, right: Expr) {
    assert_eq!(simplify(&(left - right)), Expr::from_i64(0));
}

fn at(value: &Expr, point: &str) -> Expr {
    simplify(&value.subs(&HashMap::from([(Symbol::new("k"), parse(point).unwrap())])))
}

fn raw_power(base: Expr, power: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(power)))
}

#[test]
fn exponential_moments_have_the_pinned_normalization_and_angular_frequency() {
    let normalization = parse("2^(1/2)*pi^(-1/2)").unwrap();
    for (source, cosine, sine, mass) in [
        ("exp(-2*x)", "2/5", "1/5", "1/2"),
        ("x*exp(-2*x)", "3/25", "4/25", "1/4"),
        ("x^2*exp(-2*x)", "4/125", "22/125", "1/4"),
    ] {
        let c = transform(source, HalfLineKernel::Cosine);
        let s = transform(source, HalfLineKernel::Sine);
        equal(at(&c, "1"), normalization.clone() * parse(cosine).unwrap());
        equal(at(&s, "1"), normalization.clone() * parse(sine).unwrap());
        equal(at(&c, "0"), normalization.clone() * parse(mass).unwrap());
        equal(at(&s, "0"), Expr::from_i64(0));
        equal(at(&c, "-1"), at(&c, "1"));
        equal(at(&s, "-1"), -at(&s, "1"));
    }
}

#[test]
fn frequency_derivatives_agree_with_independently_transformed_moments() {
    let k = Symbol::new("k");
    for (base, weighted) in [
        ("exp(-3*x)", "x*exp(-3*x)"),
        ("x*exp(-3*x)", "x^2*exp(-3*x)"),
    ] {
        equal(
            fsym_calculus::diff(&transform(base, HalfLineKernel::Cosine), &k),
            -transform(weighted, HalfLineKernel::Sine),
        );
        equal(
            fsym_calculus::diff(&transform(base, HalfLineKernel::Sine), &k),
            transform(weighted, HalfLineKernel::Cosine),
        );
    }
}

#[test]
fn gaussian_mass_and_odd_moment_fix_both_normalizations() {
    let c = transform("exp(-2*x^2)", HalfLineKernel::Cosine);
    let c2 = transform("x^2*exp(-2*x^2)", HalfLineKernel::Cosine);
    let s1 = transform("x*exp(-2*x^2)", HalfLineKernel::Sine);
    equal(at(&c, "0"), parse("1/2").unwrap());
    equal(at(&c2, "0"), parse("1/8").unwrap());
    equal(at(&c, "2"), parse("exp(-1/2)/2").unwrap());
    equal(at(&c2, "2"), Expr::from_i64(0));
    equal(at(&s1, "2"), parse("exp(-1/2)/4").unwrap());
    equal(fsym_calculus::diff(&c, &Symbol::new("k")), -s1);
}

#[test]
fn normalized_gaussian_transforms_are_involutions_on_the_admitted_parities() {
    for (source, kernel) in [
        ("exp(-x^2)", HalfLineKernel::Cosine),
        ("(2-3*x^2)*exp(-2*x^2)", HalfLineKernel::Cosine),
        ("x*exp(-2*x^2)", HalfLineKernel::Sine),
        ("3*x*exp(-x^2/2)", HalfLineKernel::Sine),
    ] {
        let source = parse(source).unwrap();
        let forward = checked(&source, kernel).unwrap();
        let inverse = metered_half_line_transform(
            &simplify(&forward),
            &Symbol::new("k"),
            &Symbol::new("x"),
            kernel,
            &mut Unbounded,
        )
        .unwrap();
        equal(inverse.value, source);
    }
}

#[test]
fn finite_linearity_includes_factored_sums_and_exact_constant_radicals() {
    let source = "2^(1/2)*(exp(-x^2)+3*exp(-x))";
    let result = transform(source, HalfLineKernel::Cosine);
    let expected = parse("2^(1/2)").unwrap()
        * (transform("exp(-x^2)", HalfLineKernel::Cosine)
            + Expr::from_i64(3) * transform("exp(-x)", HalfLineKernel::Cosine));
    equal(at(&result, "0"), at(&expected, "0"));
    equal(at(&result, "1"), at(&expected, "1"));
    equal(
        transform("pi^(-1/2)*exp(-x)", HalfLineKernel::Sine),
        parse("pi^(-1/2)").unwrap() * transform("exp(-x)", HalfLineKernel::Sine),
    );
}

#[test]
fn unsupported_shapes_do_not_publish_partial_or_distributional_answers() {
    for kernel in [HalfLineKernel::Sine, HalfLineKernel::Cosine] {
        for source in [
            "1",
            "exp(x)",
            "exp(x^2)",
            "exp(-a*x)",
            "exp(-x^2+x)",
            "x^3*exp(-x)",
            "exp(-x)+sin(x)",
            "exp(-x)*sin(x)",
            "1/(1+x^2)",
            "exp(-x)*exp(-2*x)",
        ] {
            assert!(
                matches!(
                    checked(&parse(source).unwrap(), kernel),
                    Err(FourierError::Unsupported(_))
                ),
                "source {source}, kernel {kernel:?}",
            );
        }
    }
    assert!(matches!(
        checked(&parse("exp(-x^2)").unwrap(), HalfLineKernel::Sine),
        Err(FourierError::Unsupported(_)),
    ));
    assert!(matches!(
        checked(&parse("x*exp(-x^2)").unwrap(), HalfLineKernel::Cosine),
        Err(FourierError::Unsupported(_)),
    ));
}

#[test]
fn original_poles_and_zeroth_powers_survive_zero_factors() {
    for kernel in [HalfLineKernel::Sine, HalfLineKernel::Cosine] {
        for power in [-1, 0] {
            let source = Expr::Mul(vec![
                Expr::from_i64(0),
                raw_power(Expr::from_i64(0), power),
                parse("exp(-x)").unwrap(),
            ]);
            assert!(matches!(
                checked(&source, kernel),
                Err(FourierError::Refused(_)),
            ));
        }
        let zero_polynomial = Expr::Mul(vec![Expr::from_i64(0), Expr::symbol("x")]);
        let source = Expr::Mul(vec![
            raw_power(zero_polynomial, 0),
            parse("exp(-x)").unwrap(),
        ]);
        assert!(matches!(
            checked(&source, kernel),
            Err(FourierError::Refused(_)),
        ));
        let hidden = Expr::Mul(vec![
            Expr::from_i64(0),
            raw_power(Expr::symbol("x"), -1),
            parse("exp(-x)").unwrap(),
        ]);
        assert!(matches!(
            checked(&hidden, kernel),
            Err(FourierError::Unsupported(_)),
        ));
    }
}

#[test]
fn output_identity_and_real_domain_requirements_are_explicit() {
    let x = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([1; 32]));
    let k = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = Expr::Function(
        "exp".into(),
        vec![Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(x.clone())])],
    );
    let result =
        metered_half_line_transform(&source, &x, &k, HalfLineKernel::Sine, &mut Unbounded).unwrap();
    assert_eq!(result.kernel, HalfLineKernel::Sine);
    assert!(result.requires_real_frequency);
    assert_eq!(result.value.free_symbols(), vec![k.clone()]);
    assert!(matches!(
        metered_half_line_transform(&source, &x, &x, HalfLineKernel::Sine, &mut Unbounded),
        Err(FourierError::Refused(_)),
    ));
    let contaminated = Expr::Add(vec![
        source,
        Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(k.clone())]),
    ]);
    assert!(matches!(
        metered_half_line_transform(&contaminated, &x, &k, HalfLineKernel::Sine, &mut Unbounded,),
        Err(FourierError::Refused(_)),
    ));
    let gaussian = metered_half_line_transform(
        &parse("exp(-x^2)").unwrap(),
        &Symbol::new("x"),
        &Symbol::new("k"),
        HalfLineKernel::Cosine,
        &mut Unbounded,
    )
    .unwrap();
    assert!(!gaussian.requires_real_frequency);
}

#[test]
fn tiny_exact_positive_decay_does_not_underflow_to_a_divergent_case() {
    let denominator = BigInt::from(1) << 200u32;
    let source = Expr::Function(
        "exp".into(),
        vec![Expr::Mul(vec![
            Expr::Rational(BigRational::new((-1).into(), denominator.clone())),
            Expr::symbol("x"),
        ])],
    );
    let result = checked(&source, HalfLineKernel::Cosine).unwrap();
    equal(
        at(&result, "0"),
        Expr::Integer(denominator) * parse("2^(1/2)*pi^(-1/2)").unwrap(),
    );
}

struct CancelAfter(usize);

impl BudgetMeter for CancelAfter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }

    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.0 == 0 {
            return Err(MeterError::Cancelled);
        }
        self.0 -= 1;
        Ok(())
    }
}

#[test]
fn resource_limits_and_cancellation_are_terminal() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let source = parse("(1+x+x^2)*exp(-2*x)").unwrap();
    let mut budget = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(matches!(
        metered_half_line_transform(&source, &x, &k, HalfLineKernel::Cosine, &mut budget),
        Err(FourierError::Meter(_)),
    ));
    for checkpoint in [0, 1, 10] {
        assert!(matches!(
            metered_half_line_transform(
                &source,
                &x,
                &k,
                HalfLineKernel::Cosine,
                &mut CancelAfter(checkpoint),
            ),
            Err(FourierError::Meter(MeterError::Cancelled)),
        ));
    }
    for source in [
        Expr::Add(vec![parse("exp(-x)").unwrap(); 65]),
        Expr::Add(vec![Expr::from_i64(0); 4097]),
        Expr::Integer(BigInt::from(1) << 4097u32),
        Expr::Const(Constant::NaN),
    ] {
        assert!(matches!(
            checked(&source, HalfLineKernel::Cosine),
            Err(FourierError::Refused(_)),
        ));
    }
}
