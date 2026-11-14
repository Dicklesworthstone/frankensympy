//! Independent convention, source-domain and operational checks for the
//! bounded inverse integral and the native real-axis Cauchy kernel.

use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_calculus::fourier::{
    FourierDirection, FourierError, fourier_transform_l1, inverse_fourier_transform_l1,
    metered_fourier_transform_l1, metered_inverse_fourier_transform_l1,
};
use fsym_core::{BigInt, Expr, Symbol, SymbolIdentity, parse};
use fsym_simplify::simplify;
use std::collections::HashMap;
use std::sync::Arc;

fn transform(source: &str, inverse: bool) -> Expr {
    let source = parse(source).unwrap();
    let variable = Symbol::new("x");
    let coordinate = Symbol::new("k");
    if inverse {
        inverse_fourier_transform_l1(&source, &variable, &coordinate).unwrap()
    } else {
        fourier_transform_l1(&source, &variable, &coordinate).unwrap()
    }
}

fn equal(left: Expr, right: Expr) {
    assert_eq!(simplify(&(left - right)), Expr::from_i64(0));
}

fn at(value: &Expr, coordinate: i64) -> Expr {
    simplify(&value.subs(&HashMap::from([(
        Symbol::new("k"),
        Expr::from_i64(coordinate),
    )])))
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn inverse_gaussian_and_bilateral_translation_use_the_positive_phase() {
    for (source, centered, phase) in [
        ("exp(-(x-2)^2)", "exp(-x^2)", "exp(4*pi*I*k)"),
        ("exp(-(x+2/3)^2)", "exp(-x^2)", "exp(-4*pi*I*k/3)"),
        ("exp(-3*abs(x-2))", "exp(-3*abs(x))", "exp(4*pi*I*k)"),
        ("exp(-3*abs(x+2/3))", "exp(-3*abs(x))", "exp(-4*pi*I*k/3)"),
    ] {
        let actual = transform(source, true);
        equal(
            actual.clone(),
            parse(phase).unwrap() * transform(centered, true),
        );
        // Reflection of the output reverses the kernel independently of the
        // native direction tag; a copied forward phase cannot pass this.
        let reflected_forward = transform(source, false)
            .subs(&HashMap::from([(Symbol::new("k"), parse("-k").unwrap())]));
        equal(actual, reflected_forward);
    }
}

#[test]
fn inverse_gaussian_mass_and_frequency_identity_are_exact() {
    let value = transform("exp(-2*x^2+3*x+1)", true);
    equal(
        at(&value, 0),
        parse("pi^(1/2)*2^(1/2)*exp(17/8)/2").unwrap(),
    );
    let derivative = fsym_calculus::diff(&value, &Symbol::new("k"));
    // Completing the square predicts the mass and this first-order ODE,
    // independently constraining both the Gaussian width and inverse phase.
    let residual = derivative + parse("pi^2*k-3*pi*I/2").unwrap() * value;
    equal(fsym_simplify::expand(&residual), Expr::from_i64(0));
}

#[test]
fn native_direction_metadata_cannot_conflate_the_two_integrals() {
    let source = parse("exp(-(x-1)^2)").unwrap();
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let forward = metered_fourier_transform_l1(&source, &x, &k, &mut Unbounded).unwrap();
    let inverse = metered_inverse_fourier_transform_l1(&source, &x, &k, &mut Unbounded).unwrap();
    assert_eq!(forward.direction, FourierDirection::Forward);
    assert_eq!(inverse.direction, FourierDirection::Inverse);
    assert_ne!(forward.value, inverse.value);
    assert!(!forward.used_cauchy_identity);
    assert!(!inverse.used_bilateral_identity);
}

#[test]
fn cauchy_kernels_use_both_real_half_axes_and_cycles_normalization() {
    for (source, mass, at_one) in [
        ("1/(x^2+1)", "pi", "pi*exp(-2*pi)"),
        ("3/(4*x^2+9)", "pi/2", "pi*exp(-3*pi)/2"),
        ("1/(x^2+1/9)", "3*pi", "3*pi*exp(-2*pi/3)"),
    ] {
        for inverse in [false, true] {
            let value = transform(source, inverse);
            equal(at(&value, 0), parse(mass).unwrap());
            equal(at(&value, 1), parse(at_one).unwrap());
            equal(at(&value, -1), parse(at_one).unwrap());
        }
    }
}

#[test]
fn cauchy_completion_handles_irreducible_quadratics_and_rational_centers() {
    let expected = parse("pi*2^(-1/2)*exp(-2*pi*(1/2)^(1/2)*Abs(k))*exp(2*pi*I*k)").unwrap();
    equal(transform("1/(2*x^2+4*x+3)", false), expected);
    for inverse in [false, true] {
        equal(
            transform("1/(2*x^2+4*x+3)", inverse),
            transform("1/(2*(x+1)^2+1)", inverse),
        );
        let translated = transform("1/(9*(x-2/3)^2+4)", inverse);
        let center = transform("1/(9*x^2+4)", inverse);
        let phase = if inverse {
            "exp(4*pi*I*k/3)"
        } else {
            "exp(-4*pi*I*k/3)"
        };
        equal(translated, parse(phase).unwrap() * center);
    }
}

#[test]
fn cauchy_catalog_flags_and_finite_linearity_preserve_every_term() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let source = parse("2/(x^2+1)-3*exp(-abs(x))+exp(-x^2)").unwrap();
    let result = metered_inverse_fourier_transform_l1(&source, &x, &k, &mut Unbounded).unwrap();
    assert!(result.used_cauchy_identity);
    assert!(result.used_bilateral_identity);
    equal(at(&result.value, 0), parse("2*pi-6+pi^(1/2)").unwrap());
}

#[test]
fn real_poles_and_unsupported_tails_do_not_become_ordinary_transforms() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    for source in [
        "1/x",
        "1/x^2",
        "1/(x^2-1)",
        "1/(x-1)^2",
        "1/(-x^2-1)",
        "1/(x^4+1)",
        "1/(a*x^2+1)",
        "x/(x^2+1)",
        "1/(x^2+1)^2",
        "1/(x^2+1)+1",
        "exp(x^2)",
        "exp(abs(x))",
    ] {
        for operation in [fourier_transform_l1, inverse_fourier_transform_l1] {
            assert!(
                matches!(
                    operation(&parse(source).unwrap(), &x, &k),
                    Err(FourierError::Unsupported(_))
                ),
                "source {source}",
            );
        }
    }
}

#[test]
fn inverse_zero_products_cannot_erase_original_poles_or_undefined_powers() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let pole = power(Expr::Sym(x.clone()), -1);
    let zero_pole = Expr::Mul(vec![Expr::from_i64(0), pole]);
    let denominator = Expr::Add(vec![parse("x^2+1").unwrap(), zero_pole.clone()]);
    for source in [zero_pole, power(denominator, -1)] {
        assert!(matches!(
            inverse_fourier_transform_l1(&source, &x, &k),
            Err(FourierError::Unsupported(_)),
        ));
    }
    let undefined = Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]);
    assert!(matches!(
        inverse_fourier_transform_l1(&undefined, &x, &k),
        Err(FourierError::Refused(_)),
    ));
}

#[test]
fn inverse_integrals_preserve_distinct_same_name_typed_coordinates() {
    let x = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([3; 32]));
    let k = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([4; 32]));
    let source = power(
        Expr::Add(vec![power(Expr::Sym(x.clone()), 2), Expr::from_i64(1)]),
        -1,
    );
    let result = inverse_fourier_transform_l1(&source, &x, &k).unwrap();
    assert_eq!(result.free_symbols().len(), 1);
    assert!(result.free_symbols().contains(&k));
    assert!(matches!(
        inverse_fourier_transform_l1(&source, &x, &x),
        Err(FourierError::Refused(_))
    ));
    let polluted = Expr::Add(vec![
        source,
        Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(k.clone())]),
    ]);
    assert!(matches!(
        inverse_fourier_transform_l1(&polluted, &x, &k),
        Err(FourierError::Refused(_))
    ));
}

#[derive(Default)]
struct Recording {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Recording {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charges[dimension.index()] += amount;
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn inverse_and_cauchy_resource_exhaustion_and_cancellation_are_terminal() {
    let source = parse("1/(2*x^2+4*x+3)+exp(-(x-2)^2)").unwrap();
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let mut measured = Recording::default();
    let value = metered_inverse_fourier_transform_l1(&source, &x, &k, &mut measured).unwrap();
    let limits = BudgetLimits {
        dimensions: measured.charges,
        verifier_pool: 0,
    };
    let replay =
        metered_inverse_fourier_transform_l1(&source, &x, &k, &mut Budget::new(limits)).unwrap();
    assert_eq!(value.value, replay.value);
    for dimension in Dimension::ALL {
        if measured.charges[dimension.index()] != 0 {
            let mut short = limits;
            short.dimensions[dimension.index()] -= 1;
            assert!(matches!(
                metered_inverse_fourier_transform_l1(&source, &x, &k, &mut Budget::new(short)),
                Err(FourierError::Meter(_)),
            ));
        }
    }
    for checkpoint in 1..=measured.checkpoints {
        let mut cancelled = Recording {
            cancel_at: Some(checkpoint),
            ..Default::default()
        };
        assert!(matches!(
            metered_inverse_fourier_transform_l1(&source, &x, &k, &mut cancelled),
            Err(FourierError::Meter(MeterError::Cancelled)),
        ));
    }
}

#[test]
fn native_cauchy_terms_share_the_whole_request_bounds() {
    let x = Symbol::new("x");
    let k = Symbol::new("k");
    let term = parse("1/(x^2+1)").unwrap();
    for source in [
        Expr::Add(vec![term; 65]),
        Expr::Integer(BigInt::from(1) << 4097u32),
        power(parse("x^2+1").unwrap(), -1_000_000),
    ] {
        assert!(matches!(
            inverse_fourier_transform_l1(&source, &x, &k),
            Err(FourierError::Refused(_)),
        ));
    }
}
