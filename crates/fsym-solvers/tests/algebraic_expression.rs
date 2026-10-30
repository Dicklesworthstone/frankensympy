//! Selected real embeddings, exact multi-generator arithmetic, original
//! domains, and adversarial composition certificates.

use std::sync::Arc;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::algebraic_expression::{
    RealAlgebraicBinding, VerifiedAlgebraicExpression, evaluate_real_algebraic_expression,
    metered_evaluate_real_algebraic_expression, verify_real_algebraic_expression,
};

fn binding(variable: &str, coefficients: &[i64], index: usize) -> RealAlgebraicBinding {
    RealAlgebraicBinding {
        variable: Symbol::new(variable),
        coefficients: coefficients
            .iter()
            .copied()
            .map(BigRational::from)
            .collect(),
        index,
    }
}

fn pair(i: usize, j: usize) -> Vec<RealAlgebraicBinding> {
    vec![binding("x", &[-2, 0, 1], i), binding("y", &[-3, 0, 1], j)]
}

fn value(
    source: &str,
    bindings: &[RealAlgebraicBinding],
    digits: u32,
) -> VerifiedAlgebraicExpression {
    evaluate_real_algebraic_expression(&parse(source).unwrap(), bindings, digits)
        .unwrap()
        .expect("the bounded Cartesian algebraic source is supported")
}

fn pow(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

#[test]
fn sum_of_independent_square_roots_has_an_exact_high_precision_enclosure() {
    let result = value("x+y", &pair(1, 1), 60);
    assert!(
        result
            .decimal()
            .starts_with("3.1462643699419723423291350657155704455124771291873")
    );
    let p = |x: &BigRational| {
        x * x * x * x - BigRational::from(10_i64) * x * x + BigRational::from(1_i64)
    };
    // p is strictly increasing above 3. Independent endpoint signs prove
    // containment of its unique root there; no root generator is the oracle.
    assert!(result.lower() > &BigRational::from(3_i64));
    assert!(result.upper() < &BigRational::from(4_i64));
    assert!(p(result.lower()) < BigRational::from(0_i64));
    assert!(p(result.upper()) > BigRational::from(0_i64));
    assert!(result.upper() - result.lower() < BigRational::new(1.into(), BigInt::from(10).pow(58)));
    assert_eq!(result.bindings(), pair(1, 1));
    assert_eq!(result.precision(), 60);
}

#[test]
fn all_four_selected_embedding_pairs_preserve_their_signs() {
    for (i, j, low, high) in [(0, 0, -4, -3), (0, 1, 0, 1), (1, 0, -1, 0), (1, 1, 3, 4)] {
        let result = value("x+y", &pair(i, j), 12);
        assert!(result.lower() > &BigRational::from(low));
        assert!(result.upper() < &BigRational::from(high));
    }
}

#[test]
fn three_independent_quadratic_embeddings_share_one_checked_parameter() {
    let inputs = [
        binding("x", &[-2, 0, 1], 1),
        binding("y", &[-3, 0, 1], 1),
        binding("z", &[-5, 0, 1], 1),
    ];
    let result = value("x+y+z", &inputs, 15);
    assert!(result.lower() > &BigRational::from(5_i64));
    assert!(result.upper() < &BigRational::from(6_i64));
    assert_eq!(result.bindings(), inputs);
    assert_eq!(result.certificate().representation.eliminant().len(), 9);
}

#[test]
fn multiplication_reciprocals_and_exact_relations_share_the_selected_field() {
    let inputs = pair(1, 1);
    let product = value("x*y", &inputs, 30);
    assert!(product.lower() > &BigRational::from(2_i64));
    assert!(product.lower() * product.lower() < BigRational::from(6_i64));
    assert!(product.upper() * product.upper() > BigRational::from(6_i64));
    let inverse = value("1/(x+y)", &inputs, 30);
    assert!(inverse.lower() > &BigRational::new(3.into(), 10.into()));
    assert!(inverse.upper() < &BigRational::new(4.into(), 10.into()));
    for source in [
        "x**2+y**2-5",
        "(x*y)**2-6",
        "1/(x+y)-(y-x)",
        "(x+y)**4-10*(x+y)**2+1",
    ] {
        let result = value(source, &inputs, 30);
        assert_eq!(result.lower(), &BigRational::from(0_i64), "{source}");
        assert_eq!(result.upper(), result.lower(), "{source}");
        assert_eq!(result.decimal(), "0");
    }
}

#[test]
fn equal_and_unequal_roots_of_the_same_polynomial_are_distinguished() {
    for (j, expected) in [(1, 0_i64), (0, 8_i64)] {
        let inputs = [binding("x", &[-2, 0, 1], 1), binding("y", &[-2, 0, 1], j)];
        let result = value("(x-y)**2", &inputs, 25);
        assert_eq!(result.lower(), &BigRational::from(expected));
        assert_eq!(result.upper(), result.lower());
    }
    let inputs = [binding("x", &[-2, 0, 1], 1), binding("y", &[-2, 0, 1], 1)];
    assert!(evaluate_real_algebraic_expression(&parse("1/(x-y)").unwrap(), &inputs, 20).is_err());
}

#[test]
fn distinct_reducible_polynomials_can_name_the_same_selected_number() {
    // (y^2-2)(y-3) has real indices -sqrt(2), sqrt(2), 3.
    let inputs = [
        binding("x", &[-2, 0, 1], 1),
        binding("y", &[6, -2, -3, 1], 1),
    ];
    let result = value("x-y", &inputs, 30);
    assert_eq!(result.decimal(), "0");
    let inputs = [inputs[0].clone(), binding("y", &[6, -2, -3, 1], 2)];
    let result = value("x-y", &inputs, 20);
    assert!(result.upper() < &BigRational::from(-1_i64));
}

#[test]
fn real_embeddings_do_not_require_all_conjugates_to_be_real() {
    let inputs = [
        binding("x", &[-2, 0, 0, 1], 0),
        binding("y", &[-3, 0, 1], 1),
    ];
    let result = value("x+y", &inputs, 25);
    assert!(result.lower() > &BigRational::from(2_i64));
    assert!(result.upper() < &BigRational::from(3_i64));
    assert!(!result.certificate().representation.all_complex_roots_real());
}

#[test]
fn repeated_scaled_definitions_and_trailing_zeros_retain_original_binding() {
    let inputs = [
        binding("x", &[28, 0, -28, 0, 7, 0], 1),
        binding("y", &[-1, 1], 0),
    ];
    let result = value("x**2+y", &inputs, 20);
    assert_eq!(result.lower(), &BigRational::from(3_i64));
    assert_eq!(result.upper(), result.lower());
    assert_eq!(result.bindings(), inputs);
    assert!(
        result.certificate().input_roots[0]
            .intervals
            .iter()
            .all(|root| root.multiplicity == 2)
    );
}

#[test]
fn typed_variables_with_the_same_printed_name_remain_distinct() {
    let x = Symbol::new("x");
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([39; 32]));
    let inputs = [
        RealAlgebraicBinding {
            variable: x.clone(),
            ..binding("x", &[-2, 0, 1], 1)
        },
        RealAlgebraicBinding {
            variable: other.clone(),
            ..binding("y", &[-3, 0, 1], 1)
        },
    ];
    let source = Expr::Add(vec![Expr::Sym(x), Expr::Sym(other)]);
    let result = evaluate_real_algebraic_expression(&source, &inputs, 20)
        .unwrap()
        .unwrap();
    assert!(result.lower() > &BigRational::from(3_i64));
    assert_eq!(result.expression(), &source);
    assert_eq!(result.bindings(), inputs);
}

#[test]
fn original_held_poles_and_zero_powers_survive_substitution() {
    let inputs = pair(1, 1);
    let zero = parse("x**2+y**2-5").unwrap();
    for source in [
        pow(zero.clone(), -1),
        Expr::Mul(vec![zero.clone(), pow(zero.clone(), -1)]),
        Expr::Mul(vec![Expr::from_i64(0), pow(zero.clone(), -1)]),
        pow(zero.clone(), 0),
        pow(pow(zero, -1), 0),
        Expr::Mul(vec![parse("x+y").unwrap(), pow(Expr::from_i64(0), 0)]),
    ] {
        assert!(evaluate_real_algebraic_expression(&source, &inputs, 20).is_err());
    }
    let factor = parse("x+y").unwrap();
    let legal = Expr::Mul(vec![factor.clone(), pow(factor, -1)]);
    let result = evaluate_real_algebraic_expression(&legal, &inputs, 20)
        .unwrap()
        .unwrap();
    assert_eq!(result.lower(), &BigRational::from(1_i64));
    assert_eq!(result.expression(), &legal);
}

#[test]
fn composition_mutations_cannot_rebind_the_source_embeddings_or_proof_chain() {
    let inputs = pair(1, 1);
    let source = parse("x+y").unwrap();
    let result = value("x+y", &inputs, 20);
    let check = |candidate| {
        assert!(
            verify_real_algebraic_expression(&source, &inputs, 20, &candidate, &mut Unbounded)
                .is_err()
        );
    };
    let mut altered = result.certificate().clone();
    altered.primitive_index = 0;
    check(altered);
    let mut altered = result.certificate().clone();
    altered.input_roots[0].intervals.pop();
    check(altered);
    let mut altered = result.certificate().clone();
    altered.input_roots[0].intervals[1].multiplicity += 1;
    check(altered);
    let mut altered = result.certificate().clone();
    altered.embedding_checks.pop();
    check(altered);
    let mut altered = result.certificate().clone();
    altered.embedding_checks[0].0.lower = BigRational::from(1_i64);
    altered.embedding_checks[0].0.upper = BigRational::from(1_i64);
    altered.embedding_checks[0].0.exact_value = Some(BigRational::from(1_i64));
    altered.embedding_checks[0].0.decimal = "1e0".into();
    check(altered);
    let mut altered = result.certificate().clone();
    altered.value.decimal = "9.0000000000000000000e0".into();
    check(altered);
    let mut altered = result.certificate().clone();
    altered.representation = value(
        "x+y",
        &[inputs[0].clone(), binding("y", &[-5, 0, 1], 1)],
        20,
    )
    .certificate()
    .representation
    .clone();
    check(altered);
    for (target, bindings, precision) in [
        (parse("x-y").unwrap(), inputs.clone(), 20),
        (source.clone(), pair(0, 1), 20),
        (source.clone(), inputs.clone(), 10),
    ] {
        assert!(
            verify_real_algebraic_expression(
                &target,
                &bindings,
                precision,
                result.certificate(),
                &mut Unbounded
            )
            .is_err()
        );
    }
    let mut rebound = inputs.clone();
    rebound[0].variable =
        Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([91; 32]));
    assert!(
        verify_real_algebraic_expression(
            &source,
            &rebound,
            20,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
    );
    verify_real_algebraic_expression(&source, &inputs, 20, result.certificate(), &mut Unbounded)
        .unwrap();
}

#[test]
fn an_exact_zero_certificate_cannot_erase_original_domain_failure() {
    let inputs = pair(1, 1);
    let result = value("x**2+y**2-5", &inputs, 20);
    let zero = parse("x**2+y**2-5").unwrap();
    let source = Expr::Mul(vec![Expr::from_i64(0), pow(zero, -1)]);
    assert!(
        verify_real_algebraic_expression(
            &source,
            &inputs,
            20,
            result.certificate(),
            &mut Unbounded
        )
        .is_err()
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
fn unsupported_source_is_admitted_before_any_cartesian_search() {
    let inputs = pair(1, 1);
    for source in [
        parse("sin(x)+y").unwrap(),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function("custom".into(), vec![parse("x+y").unwrap()]),
        ]),
        Expr::Sym(Symbol::with_identity(
            "x",
            SymbolIdentity::from_assumptions_digest([41; 32]),
        )),
        Expr::Pow(
            Arc::new(parse("x+y").unwrap()),
            Arc::new(Expr::Rational(BigRational::new(1.into(), 2.into()))),
        ),
    ] {
        let mut generator = Meter::default();
        let mut verifier = Meter::default();
        assert!(
            metered_evaluate_real_algebraic_expression(
                &source,
                &inputs,
                20,
                &mut generator,
                &mut verifier
            )
            .unwrap()
            .is_none()
        );
        assert!(generator.compute > 0);
        assert_eq!(verifier.compute, 0);
    }
}

#[test]
fn malformed_or_oversized_inputs_refuse_before_unbounded_work() {
    let source = parse("x+y").unwrap();
    for inputs in [
        vec![],
        vec![binding("x", &[-2, 0, 1], 1); 9],
        vec![binding("x", &[-2, 0, 1], 1), binding("x", &[-3, 0, 1], 1)],
        vec![
            binding("x", &[-2, 0, 1], usize::MAX),
            binding("y", &[-3, 0, 1], 1),
        ],
        vec![binding("x", &[0], 0), binding("y", &[-3, 0, 1], 1)],
        vec![
            binding("x", &[-1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 0),
            binding("y", &[-1, 0, 0, 0, 0, 0, 0, 0, 1], 0),
        ],
    ] {
        assert!(evaluate_real_algebraic_expression(&source, &inputs, 20).is_err());
    }
    for digits in [0, 513, u32::MAX] {
        assert!(evaluate_real_algebraic_expression(&source, &pair(1, 1), digits).is_err());
    }
    let oversized = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Integer(BigInt::from(1) << 4097),
    ]);
    assert!(evaluate_real_algebraic_expression(&oversized, &pair(1, 1), 20).is_err());
    assert!(evaluate_real_algebraic_expression(&pow(source, 65), &pair(1, 1), 20).is_err());
}

#[test]
fn generator_and_verifier_cancel_during_search_and_before_publication() {
    let inputs = pair(1, 1);
    let source = parse("x+y").unwrap();
    let mut generator = Meter::default();
    let mut verifier = Meter::default();
    metered_evaluate_real_algebraic_expression(&source, &inputs, 12, &mut generator, &mut verifier)
        .unwrap()
        .unwrap();
    assert!(generator.compute > 0 && verifier.compute > 0);
    for checkpoint in [1, generator.checkpoints / 2, generator.checkpoints] {
        let mut stopped = Meter {
            cancel_at: Some(checkpoint),
            ..Meter::default()
        };
        assert!(
            metered_evaluate_real_algebraic_expression(
                &source,
                &inputs,
                12,
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
            metered_evaluate_real_algebraic_expression(
                &source,
                &inputs,
                12,
                &mut Unbounded,
                &mut stopped
            )
            .is_err()
        );
    }
}

#[test]
fn actual_exhausted_budgets_never_publish_a_partial_value() {
    let source = parse("x+y").unwrap();
    let inputs = pair(1, 1);
    let mut generator = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        metered_evaluate_real_algebraic_expression(
            &source,
            &inputs,
            20,
            &mut generator,
            &mut untouched
        )
        .is_err()
    );
    assert_eq!(untouched.compute, 0);
    let mut verifier = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        metered_evaluate_real_algebraic_expression(
            &source,
            &inputs,
            20,
            &mut Unbounded,
            &mut verifier
        )
        .is_err()
    );
    let accepted = value("x+y", &inputs, 12);
    let mut checker = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(
        verify_real_algebraic_expression(
            &source,
            &inputs,
            12,
            accepted.certificate(),
            &mut checker
        )
        .is_err()
    );
}
