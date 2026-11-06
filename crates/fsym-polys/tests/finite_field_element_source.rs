//! The ZZ -> F_p source boundary must precede modular simplification.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    PrimeFieldElementSourceError, PrimeFieldPolynomial, checked_prime_field_element_source,
    verify_prime_field_element_source,
};

fn integer(value: i64) -> Expr {
    Expr::Integer(BigInt::from(value))
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(base.into(), integer(exponent).into())
}
fn reduce(expression: &Expr, prime: u64) -> Vec<u64> {
    checked_prime_field_element_source(
        expression,
        &Symbol::new("x"),
        prime,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap()
    .into_polynomial()
    .coefficients()
    .to_vec()
}
fn refused(expression: &Expr) {
    let variable = Symbol::new("x");
    assert!(
        checked_prime_field_element_source(
            expression,
            &variable,
            5,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let candidate = PrimeFieldPolynomial::new(variable.clone(), 5, vec![]).unwrap();
    assert!(
        verify_prime_field_element_source(expression, &variable, 5, &candidate, &mut Unbounded,)
            .is_err()
    );
}

#[test]
fn integer_images_preserve_negative_coefficients_and_degree_drop() {
    for (source, prime, expected) in [
        ("-11*x**3+5*x**2-6*x+14", 5, vec![4, 4, 0, 4]),
        ("5*x**2+10*x+3", 5, vec![3]),
        ("10*x+15", 5, vec![]),
        ("(x+1)**5", 5, vec![1, 0, 0, 0, 0, 1]),
        ("(x+1)**2", 2, vec![1, 0, 1]),
        ("0", 2, vec![]),
        ("-13", 7, vec![1]),
    ] {
        assert_eq!(reduce(&parse(source).unwrap(), prime), expected, "{source}");
    }
}

#[test]
fn rational_intermediates_must_finish_as_integer_coefficients() {
    let x = Expr::Sym(Symbol::new("x"));
    let half = Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)));
    let source = Expr::Mul(vec![integer(2), half.clone(), x.clone()]);
    assert_eq!(reduce(&source, 2), [0, 1]);
    assert_eq!(reduce(&Expr::Add(vec![half.clone(), half.clone()]), 2), [1]);
    assert_eq!(
        reduce(&Expr::Mul(vec![integer(5), power(integer(5), -1), x]), 5),
        [0, 1]
    );
    for source in [
        half.clone(),
        Expr::Add(vec![Expr::Sym(Symbol::new("x")), half]),
    ] {
        let error = checked_prime_field_element_source(
            &source,
            &Symbol::new("x"),
            5,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap_err();
        assert_eq!(error, PrimeFieldElementSourceError::NonIntegralCoefficient);
        let candidate = PrimeFieldPolynomial::new(Symbol::new("x"), 5, vec![]).unwrap();
        assert_eq!(
            verify_prime_field_element_source(
                &source,
                &Symbol::new("x"),
                5,
                &candidate,
                &mut Unbounded,
            )
            .unwrap_err(),
            PrimeFieldElementSourceError::NonIntegralCoefficient
        );
    }
}

#[test]
fn arbitrarily_large_admitted_integers_have_independent_bytewise_images() {
    let value = (BigInt::from(1) << 2048u32) + BigInt::from(17);
    // 2^2048 == 4 (mod 7), hence both signs are nontrivial.
    assert!(reduce(&Expr::Integer(value.clone()), 7).is_empty());
    // Use a nonzero residue as well, to exercise sign adjustment.
    assert_eq!(reduce(&Expr::Integer(-value + BigInt::from(3)), 7), [3]);
    let maximum = (BigInt::from(1) << 4095u32) + BigInt::from(1);
    assert_eq!(reduce(&Expr::Integer(maximum), 2), [1]);
}

#[test]
fn accepted_source_is_original_owned_syntax_and_replays_independently() {
    let variable = Symbol::new("x");
    let source = Expr::Add(vec![
        integer(0),
        Expr::Mul(vec![integer(1), Expr::Sym(variable.clone())]),
    ]);
    let result =
        checked_prime_field_element_source(&source, &variable, 5, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(result.source(), &source);
    assert_eq!(result.polynomial().coefficients(), [0, 1]);
    let replay = verify_prime_field_element_source(
        &source,
        &variable,
        5,
        result.polynomial(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(replay.source(), &source);
    assert_eq!(replay.into_polynomial(), result.into_polynomial());
}

#[test]
fn verifier_checks_every_coefficient_and_exact_typed_field() {
    let source = parse("3*x**2+2*x+1").unwrap();
    let variable = Symbol::new("x");
    for values in [
        vec![],
        vec![2, 2, 3],
        vec![1, 3, 3],
        vec![1, 2, 4],
        vec![1, 2],
    ] {
        let candidate = PrimeFieldPolynomial::new(variable.clone(), 5, values).unwrap();
        assert!(
            verify_prime_field_element_source(&source, &variable, 5, &candidate, &mut Unbounded,)
                .is_err()
        );
    }
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    for (symbol, prime) in [(variable.clone(), 7), (Symbol::new("y"), 5), (typed, 5)] {
        let candidate = PrimeFieldPolynomial::new(symbol, prime, vec![1, 2, 3]).unwrap();
        assert!(
            verify_prime_field_element_source(&source, &variable, 5, &candidate, &mut Unbounded,)
                .is_err()
        );
    }
}

#[test]
fn typed_generator_identity_is_retained_without_printed_name_matching() {
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = Expr::Sym(typed.clone());
    let accepted =
        checked_prime_field_element_source(&source, &typed, 3, &mut Unbounded, &mut Unbounded)
            .unwrap();
    assert_eq!(accepted.polynomial().variable(), &typed);
    refused(&source);
}

#[test]
fn all_original_children_are_admitted_before_zero_shortcuts() {
    let bad = [
        Expr::Sym(Symbol::new("y")),
        Expr::Function("sin".into(), vec![Expr::Sym(Symbol::new("x"))]),
        Expr::Integer(BigInt::from(1) << 4096u32),
        power(Expr::Sym(Symbol::new("x")), -1),
        power(integer(0), -1),
    ];
    for child in bad {
        refused(&Expr::Mul(vec![integer(0), child.clone()]));
        refused(&power(child, 0));
    }
    refused(&power(integer(0), 0));
    refused(&Expr::Pow(
        integer(1).into(),
        Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2))).into(),
    ));
}

#[test]
fn final_degree_bound_is_checked_before_modular_annihilation() {
    let x = Expr::Sym(Symbol::new("x"));
    let admitted = power(x.clone(), 64);
    let values = reduce(&admitted, 2);
    assert_eq!(values.len(), 65);
    assert_eq!(values[64], 1);
    refused(&Expr::Mul(vec![integer(5), power(x.clone(), 65)]));
    refused(&power(x, 257));
}

#[test]
fn exact_cancellation_may_reduce_the_final_integer_polynomial_degree() {
    let high = power(Expr::Sym(Symbol::new("x")), 65);
    let source = Expr::Add(vec![
        high.clone(),
        Expr::Mul(vec![integer(-1), high]),
        integer(1),
    ]);
    assert_eq!(reduce(&source, 2), [1]);
}

#[test]
fn full_original_node_depth_and_symbol_limits_are_enforced() {
    refused(&Expr::Add(vec![integer(0); 4096]));
    let mut deep = integer(0);
    for _ in 0..65 {
        deep = Expr::Add(vec![deep]);
    }
    refused(&deep);
    refused(&Expr::Sym(Symbol::new("x".repeat(65_537))));
}

#[test]
fn invalid_characteristics_refuse_before_constant_shortcuts() {
    for prime in [0, 1, 4, 9, 65_523, u64::MAX] {
        assert!(
            checked_prime_field_element_source(
                &integer(0),
                &Symbol::new("x"),
                prime,
                &mut Unbounded,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[derive(Default)]
struct Count {
    steps: u64,
    bytes: u64,
    checkpoints: usize,
    cancel: Option<usize>,
}
impl BudgetMeter for Count {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        match dimension {
            Dimension::ComputeSteps => self.steps += amount,
            Dimension::MemoryBytes => self.bytes += amount,
            _ => {}
        }
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
        if self.cancel == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}
fn cap(dimension: Dimension, amount: u64) -> Budget {
    let mut limits = BudgetLimits::uniform(u64::MAX, 0);
    limits.dimensions[dimension.index()] = amount;
    Budget::new(limits)
}

#[test]
fn cancellation_in_either_phase_including_publication_is_terminal() {
    let source = parse("(x+1)**7-11*x+3").unwrap();
    let variable = Symbol::new("x");
    let (mut generation, mut verification) = (Count::default(), Count::default());
    checked_prime_field_element_source(&source, &variable, 7, &mut generation, &mut verification)
        .unwrap();
    for (is_generation, checkpoints) in [
        (true, generation.checkpoints),
        (false, verification.checkpoints),
    ] {
        for at in [1, checkpoints / 2, checkpoints] {
            let mut cancelled = Count {
                cancel: Some(at),
                ..Count::default()
            };
            let result = if is_generation {
                checked_prime_field_element_source(
                    &source,
                    &variable,
                    7,
                    &mut cancelled,
                    &mut Unbounded,
                )
            } else {
                checked_prime_field_element_source(
                    &source,
                    &variable,
                    7,
                    &mut Unbounded,
                    &mut cancelled,
                )
            };
            assert!(matches!(
                result,
                Err(PrimeFieldElementSourceError::Polynomial(_))
            ));
        }
    }
}

#[test]
fn shared_ledgers_cover_conversion_reduction_and_accepted_source_cloning() {
    let source = parse("(x+1)**9-13*x+1").unwrap();
    let variable = Symbol::new("x");
    let (mut generation, mut verification) = (Count::default(), Count::default());
    checked_prime_field_element_source(&source, &variable, 11, &mut generation, &mut verification)
        .unwrap();
    for (dimension, g, v) in [
        (
            Dimension::ComputeSteps,
            generation.steps,
            verification.steps,
        ),
        (Dimension::MemoryBytes, generation.bytes, verification.bytes),
    ] {
        checked_prime_field_element_source(
            &source,
            &variable,
            11,
            &mut cap(dimension, g),
            &mut cap(dimension, v),
        )
        .unwrap();
        assert!(
            checked_prime_field_element_source(
                &source,
                &variable,
                11,
                &mut cap(dimension, g - 1),
                &mut Unbounded
            )
            .is_err()
        );
        assert!(
            checked_prime_field_element_source(
                &source,
                &variable,
                11,
                &mut Unbounded,
                &mut cap(dimension, v - 1)
            )
            .is_err()
        );
    }
}
