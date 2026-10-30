//! Exact one-generator algebraic evaluation, including original poles and
//! independent acceptance of enclosures, indexed embeddings and decimal ties.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::real_roots::{
    isolate_real_roots, real_root_polynomial_zero, refine_real_root_isolation,
};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::real_equation::{
    VerifiedAlgebraicValue, evaluate_real_algebraic, metered_evaluate_real_algebraic,
    verify_real_algebraic_value,
};
use std::sync::Arc;

fn polynomial(coefficients: &[i64]) -> Vec<BigRational> {
    coefficients
        .iter()
        .copied()
        .map(BigRational::from)
        .collect()
}

fn value(
    expression: &str,
    coefficients: &[i64],
    index: usize,
    digits: u32,
) -> VerifiedAlgebraicValue {
    evaluate_real_algebraic(
        &parse(expression).unwrap(),
        &Symbol::new("x"),
        &polynomial(coefficients),
        index,
        digits,
    )
    .unwrap()
    .expect("the exact rational source is admitted")
}

fn power(expression: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(expression), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn eighty_digit_values_and_transforms_are_not_binary64_extensions() {
    let root = value("x", &[-2, 0, 1], 1, 80);
    assert_eq!(
        root.decimal(),
        "1.4142135623730950488016887242096980785696718753769480731766797379907324784621070e0"
    );
    let shifted = value("x+1", &[-2, 0, 1], 1, 80);
    assert_eq!(
        shifted.decimal(),
        "2.4142135623730950488016887242096980785696718753769480731766797379907324784621070e0"
    );
    let inverse = value("1/(x+1)", &[-2, 0, 1], 1, 80);
    assert_eq!(
        inverse.decimal(),
        "4.1421356237309504880168872420969807856967187537694807317667973799073247846210704e-1"
    );
    // Independent polynomial signs certify the three selected positive
    // embeddings, without consulting a native root or rounding routine.
    let two = BigRational::from(2_i64);
    assert!(root.lower() > &BigRational::from(0_i64));
    assert!(root.lower() * root.lower() < two);
    assert!(root.upper() * root.upper() > two);
    for endpoint in [shifted.lower(), shifted.upper()] {
        assert!(endpoint > &BigRational::from(2_i64));
    }
    let one = BigRational::from(1_i64);
    let inverse_polynomial = |x: &BigRational| x * x + &two * x - &one;
    assert!(inverse_polynomial(inverse.lower()) < BigRational::from(0_i64));
    assert!(inverse_polynomial(inverse.upper()) > BigRational::from(0_i64));
    let width = root.upper() - root.lower();
    assert!(width < BigRational::new(1.into(), BigInt::from(10).pow(79)));
}

#[test]
fn maximum_precision_refines_an_irrational_coordinate_with_exact_bounds() {
    let result = value("x+1", &[-2, 0, 1], 1, 512);
    assert!(result.decimal().starts_with(
        "2.414213562373095048801688724209698078569671875376948073176679737990732478462107"
    ));
    let mantissa = result.decimal().strip_suffix("e0").unwrap();
    assert_eq!(
        mantissa
            .chars()
            .filter(|character| character.is_ascii_digit())
            .count(),
        512
    );
    let low = result.lower() - BigRational::from(1_i64);
    let high = result.upper() - BigRational::from(1_i64);
    assert!(low > BigRational::from(0_i64));
    assert!(&low * &low < BigRational::from(2_i64));
    assert!(&high * &high > BigRational::from(2_i64));
    assert!(high - low < BigRational::new(1.into(), BigInt::from(10).pow(511)));
}

#[test]
fn algebraic_relations_and_rational_functions_reduce_exactly_before_rounding() {
    for (expression, expected) in [
        ("x**2-2", 0_i64),
        ("x**4-4", 0),
        ("x**2+1", 3),
        ("(x**3+2*x)/(2*x)", 2),
        ("(x**2+3)/(x**2+3)", 1),
    ] {
        for index in [0, 1] {
            let result = value(expression, &[-2, 0, 1], index, 100);
            let expected = BigRational::from(expected);
            assert_eq!(result.lower(), &expected);
            assert_eq!(result.upper(), &expected);
            assert_eq!(result.certificate().exact_value.as_ref(), Some(&expected));
        }
    }
    assert_eq!(value("x**2-2", &[-2, 0, 1], 1, 512).decimal(), "0");
}

#[test]
fn every_original_pole_survives_zero_products_and_zeroth_powers() {
    let defining = polynomial(&[-2, 0, 1]);
    let variable = Symbol::new("x");
    let zero_at_root = parse("x**2-2").unwrap();
    for source in [
        power(zero_at_root.clone(), -1),
        Expr::Mul(vec![zero_at_root.clone(), power(zero_at_root.clone(), -1)]),
        Expr::Mul(vec![Expr::from_i64(0), power(zero_at_root.clone(), -1)]),
        power(zero_at_root.clone(), 0),
        power(power(zero_at_root, -1), 0),
        power(Expr::from_i64(0), 0),
        Expr::Mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), 0)]),
    ] {
        for index in [0, 1] {
            assert!(evaluate_real_algebraic(&source, &variable, &defining, index, 40).is_err());
        }
    }
}

#[test]
fn cancellation_is_legal_only_at_the_selected_defined_embedding() {
    let variable = Symbol::new("x");
    let factor = parse("x-1").unwrap();
    let source = Expr::Mul(vec![factor.clone(), power(factor, -1)]);
    let defining = polynomial(&[2, -3, 1]); // roots 1 and 2
    assert!(evaluate_real_algebraic(&source, &variable, &defining, 0, 80).is_err());
    let result = evaluate_real_algebraic(&source, &variable, &defining, 1, 80)
        .unwrap()
        .unwrap();
    assert_eq!(result.lower(), &BigRational::from(1_i64));
    assert_eq!(result.upper(), result.lower());
    assert_eq!(result.expression(), &source);
    assert_eq!(result.variable(), &variable);
    assert_eq!(result.defining_polynomial(), defining);
    assert_eq!(result.index(), 1);
    assert_eq!(result.precision(), 80);
}

#[test]
fn exact_ties_use_half_even_at_non_dyadic_isolation_boundaries() {
    // Roots 2/3 and 5/4. The initial Cauchy bracket has denominator 12,
    // and repeated halving need not ever land on the second root itself.
    let tie = value("x", &[10, -23, 12], 1, 2);
    assert_eq!(tie.decimal(), "1.2e0");
    assert_eq!(tie.lower(), &BigRational::new(5.into(), 4.into()));
    assert_eq!(tie.upper(), tie.lower());
    assert_eq!(value("x", &[10, 23, 12], 0, 2).decimal(), "-1.2e0");
    assert_eq!(value("x", &[10, -23, 12], 0, 2).decimal(), "6.7e-1");
    assert_eq!(value("x+1/10", &[10, -23, 12], 1, 2).decimal(), "1.4e0");
}

#[test]
fn repeated_and_reducible_definitions_preserve_exact_indexed_membership() {
    let repeated = value("x**2+1", &[4, 0, -4, 0, 1], 1, 50);
    assert_eq!(repeated.lower(), &BigRational::from(3_i64));
    assert!(
        repeated
            .certificate()
            .roots
            .intervals
            .iter()
            .all(|root| root.multiplicity == 2)
    );
    // (x^2-2)(x-3): the first two roots satisfy x^2-2=0, the last does not.
    for index in [0, 1] {
        assert_eq!(value("x**2-2", &[6, -2, -3, 1], index, 50).decimal(), "0");
    }
    let third = value("x**2-2", &[6, -2, -3, 1], 2, 50);
    assert!(third.lower() <= &BigRational::from(7_i64));
    assert!(third.upper() >= &BigRational::from(7_i64));
    assert!(
        third
            .decimal()
            .starts_with("7.0000000000000000000000000000000000000000000000000e0")
    );
    let scaled = value("x+1", &[-14, 0, 7], 1, 80);
    assert_eq!(scaled.decimal(), value("x+1", &[-2, 0, 1], 1, 80).decimal());
}

#[test]
fn branch_local_rational_images_require_exact_membership_before_collapse() {
    // (x^2-2)(x^2-3): x^2 is not globally constant in the quotient, but
    // each selected real embedding has the exact rational image 2 or 3.
    for (index, expected) in [(0, 3_i64), (1, 2), (2, 2), (3, 3)] {
        let result = value("x**2", &[6, 0, -5, 0, 1], index, 25);
        assert_eq!(result.lower(), &BigRational::from(expected));
        assert_eq!(result.upper(), result.lower());
        assert_eq!(
            result.certificate().exact_value.as_ref(),
            Some(result.lower())
        );
    }

    // This irrational root rounds to 2 at twelve digits but is strictly
    // greater. A rounded candidate is not an algebraic identity.
    let epsilon = BigRational::new(1.into(), BigInt::from(1) << 200);
    let radicand = BigRational::from(4_i64) + epsilon;
    let defining = vec![
        -radicand.clone(),
        BigRational::from(0_i64),
        BigRational::from(1_i64),
    ];
    let variable = Symbol::new("x");
    let source = Expr::Sym(variable.clone());
    let result = evaluate_real_algebraic(&source, &variable, &defining, 1, 12)
        .unwrap()
        .unwrap();
    assert_eq!(result.decimal(), "2.00000000000e0");
    assert!(result.certificate().exact_value.is_none());
    assert!(result.lower() < result.upper());
    assert!(result.upper() > &BigRational::from(2_i64));
    assert!(result.lower() * result.lower() <= radicand);
    assert!(result.upper() * result.upper() >= radicand);
    let mut forged = result.certificate().clone();
    forged.lower = BigRational::from(2_i64);
    forged.upper = forged.lower.clone();
    forged.exact_value = Some(forged.lower.clone());
    assert!(
        verify_real_algebraic_value(
            &source,
            &variable,
            &defining,
            1,
            12,
            &forged,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn a_nonconstant_quintic_coordinate_gets_a_source_bound_enclosure() {
    // p has five distinct real roots, independently bracketed by alternating
    // signs at -2,-3/2,-1,0,1,2. The largest lies between 1 and 2.
    let p = [1, 3, -3, -4, 1, 1];
    let result = value("1+x**2", &p, 4, 50);
    assert!(result.lower() > &BigRational::from(2_i64));
    assert!(result.upper() < &BigRational::from(5_i64));
    assert!(result.upper() - result.lower() < BigRational::new(1.into(), BigInt::from(10).pow(48)));
    let verified = verify_real_algebraic_value(
        result.expression(),
        result.variable(),
        result.defining_polynomial(),
        result.index(),
        result.precision(),
        result.certificate(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(verified, result);
}

#[test]
fn complete_isolation_and_decimal_mutations_do_not_inherit_acceptance() {
    let expression = parse("x+1").unwrap();
    let variable = Symbol::new("x");
    let p = polynomial(&[-2, 0, 1]);
    let accepted = evaluate_real_algebraic(&expression, &variable, &p, 1, 40)
        .unwrap()
        .unwrap();
    let reject = |certificate| {
        assert!(
            verify_real_algebraic_value(
                &expression,
                &variable,
                &p,
                1,
                40,
                &certificate,
                &mut Unbounded
            )
            .is_err()
        );
    };
    let mut altered = accepted.certificate().clone();
    altered.roots.intervals.pop();
    reject(altered);
    let mut altered = accepted.certificate().clone();
    altered.roots.intervals[0].multiplicity = 2;
    reject(altered);
    let mut altered = accepted.certificate().clone();
    altered.lower = altered.upper.clone();
    reject(altered);
    let mut altered = accepted.certificate().clone();
    altered.decimal = "2.000000000000000000000000000000000000000e0".into();
    reject(altered);
    let mut altered = accepted.certificate().clone();
    altered.lower = BigRational::from(1_i64);
    altered.upper = altered.lower.clone();
    altered.exact_value = Some(altered.lower.clone());
    altered.decimal = "1.000000000000000000000000000000000000000e0".into();
    reject(altered);
    for (source, index, precision) in [
        (parse("x+2").unwrap(), 1, 40),
        (expression.clone(), 0, 40),
        (expression.clone(), 1, 20),
    ] {
        assert!(
            verify_real_algebraic_value(
                &source,
                &variable,
                &p,
                index,
                precision,
                accepted.certificate(),
                &mut Unbounded
            )
            .is_err()
        );
    }
    let fake_variable =
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([61; 32]));
    assert!(
        verify_real_algebraic_value(
            &expression,
            &fake_variable,
            &p,
            1,
            40,
            accepted.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn an_exact_value_claim_cannot_hide_an_original_domain_exclusion() {
    let p = polynomial(&[-2, 0, 1]);
    let variable = Symbol::new("x");
    let result = value("1", &[-2, 0, 1], 1, 40);
    let factor = parse("x**2-2").unwrap();
    let guarded = Expr::Mul(vec![factor.clone(), power(factor, -1)]);
    assert!(
        verify_real_algebraic_value(
            &guarded,
            &variable,
            &p,
            1,
            40,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    let mut false_point = result.certificate().clone();
    false_point.upper = BigRational::from(2_i64);
    assert!(
        verify_real_algebraic_value(
            &Expr::from_i64(1),
            &variable,
            &p,
            1,
            40,
            &false_point,
            &mut Unbounded
        )
        .is_err()
    );
}

#[test]
fn unsupported_original_nodes_and_typed_symbols_are_not_erased() {
    let variable = Symbol::new("x");
    let p = polynomial(&[-2, 0, 1]);
    let alias = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([47; 32]));
    for source in [
        Expr::Sym(alias),
        parse("sin(x)").unwrap(),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function("opaque".into(), vec![Expr::Sym(variable.clone())]),
        ]),
        Expr::Pow(
            Arc::new(Expr::Sym(variable.clone())),
            Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
        ),
    ] {
        assert!(
            evaluate_real_algebraic(&source, &variable, &p, 1, 40)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn input_precision_index_degree_and_numeric_height_limits_fail_explicitly() {
    let variable = Symbol::new("x");
    let p = polynomial(&[-2, 0, 1]);
    for digits in [0, 513, u32::MAX] {
        assert!(
            evaluate_real_algebraic(&Expr::Sym(variable.clone()), &variable, &p, 1, digits)
                .is_err()
        );
    }
    for index in [2, usize::MAX] {
        assert!(
            evaluate_real_algebraic(&Expr::Sym(variable.clone()), &variable, &p, index, 20)
                .is_err()
        );
    }
    for bad in [
        vec![],
        polynomial(&[0]),
        vec![BigRational::from(1_i64); 66],
        vec![
            BigRational::from_integer(BigInt::from(1) << 4097),
            BigRational::from(1_i64),
        ],
    ] {
        assert!(
            evaluate_real_algebraic(&Expr::Sym(variable.clone()), &variable, &bad, 0, 20).is_err()
        );
    }
    for source in [
        power(Expr::Sym(variable.clone()), 65),
        Expr::Integer(BigInt::from(1) << 4097),
    ] {
        assert!(evaluate_real_algebraic(&source, &variable, &p, 1, 20).is_err());
    }
}

#[test]
fn indexed_refinement_and_polynomial_membership_keep_all_original_roots() {
    let p = polynomial(&[4, 0, -4, 0, 1]);
    let roots = isolate_real_roots(&p, &mut Unbounded, &mut Unbounded).unwrap();
    let before = roots.intervals()[1].upper.clone() - roots.intervals()[1].lower.clone();
    let refined =
        refine_real_root_isolation(&roots, 1, 40, &mut Unbounded, &mut Unbounded).unwrap();
    let after = refined.intervals()[1].upper.clone() - refined.intervals()[1].lower.clone();
    assert!(after < before);
    assert_eq!(refined.intervals()[0], roots.intervals()[0]);
    assert_eq!(refined.count_with_multiplicity(), 4);
    for index in [0, 1] {
        assert!(
            real_root_polynomial_zero(&refined, index, &polynomial(&[-2, 0, 1]), &mut Unbounded)
                .unwrap()
        );
        assert!(
            real_root_polynomial_zero(&refined, index, &polynomial(&[0]), &mut Unbounded).unwrap()
        );
        assert!(
            !real_root_polynomial_zero(&refined, index, &polynomial(&[-1, 1]), &mut Unbounded)
                .unwrap()
        );
    }
    assert!(real_root_polynomial_zero(&refined, 2, &polynomial(&[0]), &mut Unbounded).is_err());
    assert!(
        refine_real_root_isolation(&roots, 1, usize::MAX, &mut Unbounded, &mut Unbounded).is_err()
    );
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    compute: u64,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.compute += amount;
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
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn generator_and_reference_checker_cancel_during_work_and_before_publication() {
    let variable = Symbol::new("x");
    let p = polynomial(&[-2, 0, 1]);
    for expression in [parse("x+1").unwrap(), parse("x**2-2").unwrap()] {
        let mut generator = Meter::default();
        let mut verifier = Meter::default();
        metered_evaluate_real_algebraic(
            &expression,
            &variable,
            &p,
            1,
            20,
            &mut generator,
            &mut verifier,
        )
        .unwrap()
        .unwrap();
        assert!(generator.compute > 0 && verifier.compute > 0);
        for checkpoint in [1, generator.checkpoints / 2, generator.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(checkpoint),
                ..Meter::default()
            };
            assert!(
                metered_evaluate_real_algebraic(
                    &expression,
                    &variable,
                    &p,
                    1,
                    20,
                    &mut stopped,
                    &mut Unbounded
                )
                .is_err()
            );
        }
        for checkpoint in [1, verifier.checkpoints / 2, verifier.checkpoints] {
            let mut stopped = Meter {
                cancel_at: Some(checkpoint),
                ..Meter::default()
            };
            assert!(
                metered_evaluate_real_algebraic(
                    &expression,
                    &variable,
                    &p,
                    1,
                    20,
                    &mut Unbounded,
                    &mut stopped
                )
                .is_err()
            );
        }
    }
}

#[test]
fn actual_budget_refusal_never_publishes_an_algebraic_value() {
    let variable = Symbol::new("x");
    let p = polynomial(&[-2, 0, 1]);
    let expression = parse("x+1").unwrap();
    let mut generator = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        metered_evaluate_real_algebraic(
            &expression,
            &variable,
            &p,
            1,
            30,
            &mut generator,
            &mut untouched
        )
        .is_err()
    );
    assert_eq!(untouched.checkpoints, 0);
    let mut verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_evaluate_real_algebraic(
            &expression,
            &variable,
            &p,
            1,
            30,
            &mut Unbounded,
            &mut verifier
        )
        .is_err()
    );
    let mut generator = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let mut verifier = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    assert!(
        metered_evaluate_real_algebraic(
            &expression,
            &variable,
            &p,
            1,
            50,
            &mut generator,
            &mut verifier
        )
        .unwrap()
        .is_some()
    );
}
