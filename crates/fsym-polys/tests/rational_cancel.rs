//! Complete QQ cancellation, exact source domains, and forged composition data.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly as Polynomial, RationalCancellationCertificate as Certificate,
    RationalLocalizationCertificate, VerifiedRationalCancellation as Checked,
    bounded_polynomial_from_expr, checked_multivariate_gcd, checked_rational_cancel as checked,
    metered_localize_rational_system, metered_rational_source, verify_rational_cancel as verify,
    verify_rational_localization,
};
use std::collections::BTreeMap;
use std::sync::Arc;

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn ratio(numerator: Expr, denominator: Expr) -> Expr {
    Expr::Mul(vec![numerator, power(denominator, -1)])
}

fn polynomial(source: &str, variables: &[Symbol]) -> Polynomial {
    bounded_polynomial_from_expr(&e(source), variables, &mut Unbounded).unwrap()
}

fn scalar(value: BigRational, variables: &[Symbol]) -> Polynomial {
    Polynomial::new(
        variables.to_vec(),
        BTreeMap::from([(vec![0; variables.len()], value)]),
    )
    .unwrap()
}

fn solve(source: &Expr, variables: &[Symbol]) -> Checked {
    checked(source, variables, &mut Unbounded, &mut Unbounded).unwrap()
}

fn rejects(accepted: &Checked, candidate: &Certificate) {
    assert!(
        verify(
            accepted.source(),
            accepted.variables(),
            candidate,
            &mut Unbounded
        )
        .is_err()
    );
}

// Tiny independent fixture multiplication; expected factors are hand supplied,
// never inferred by the production cancellation, division or GCD routines.
fn multiply(left: &Polynomial, right: &Polynomial) -> Polynomial {
    let mut result = BTreeMap::<Vec<u32>, BigRational>::new();
    for (a, ca) in left.terms() {
        for (b, cb) in right.terms() {
            let powers = a.iter().zip(b).map(|(a, b)| a + b).collect();
            *result.entry(powers).or_default() += ca * cb;
        }
    }
    Polynomial::new(left.generators().to_vec(), result).unwrap()
}

#[test]
fn multivariate_nonlinear_cancellation_preserves_exact_units_and_guard() {
    let variables = ring(&["x", "y", "z"]);
    let source = ratio(e("-6/5*(x*y+1)**2*(x+1)"), e("9/7*(x*y+1)*(z-1)"));
    let result = solve(&source, &variables);
    assert_eq!(result.source(), &source);
    assert_eq!(result.variables(), &variables);
    assert_eq!(
        result.numerator(),
        &polynomial("-14/15*(x*y+1)*(x+1)", &variables)
    );
    assert_eq!(result.denominator(), &polynomial("z-1", &variables));
    assert_eq!(
        result.guards(),
        &[polynomial("9/7*(x*y+1)*(z-1)", &variables)]
    );
    assert_eq!(
        result.certificate().gcd.gcd(),
        &polynomial("x*y+1", &variables)
    );
    let replay = verify(&source, &variables, result.certificate(), &mut Unbounded).unwrap();
    let (numerator, denominator, original) = replay.into_parts();
    assert_eq!(&numerator, result.numerator());
    assert_eq!(&denominator, result.denominator());
    assert_eq!(original.source(), &source);
    assert_eq!(original.guards(), result.guards());
}

#[test]
fn degree65_cancellation_bypasses_the_unrelated_expand_power64_limit() {
    let variables = ring(&["x", "y"]);
    let source = ratio(power(e("x+y"), 65), e("2*x+2*y"));
    let h = polynomial("x+y", &variables);
    let mut expected = scalar(q(1, 2), &variables);
    for _ in 0..64 {
        expected = multiply(&expected, &h);
    }
    let result = solve(&source, &variables);
    assert_eq!(result.numerator(), &expected);
    assert_eq!(result.numerator().terms().len(), 65);
    assert_eq!(result.denominator(), &polynomial("1", &variables));
    assert_eq!(result.guards(), &[polynomial("2*x+2*y", &variables)]);
    let lifted = result
        .metered_expression(&mut Unbounded, &mut Unbounded)
        .unwrap();
    assert_eq!(
        bounded_polynomial_from_expr(&lifted, &variables, &mut Unbounded).unwrap(),
        expected
    );
}

#[test]
fn common_denominators_and_cancelling_sums_are_reduced_as_one_source() {
    let variables = ring(&["x", "y"]);
    let source = Expr::Add(vec![ratio(e("1"), e("x+1")), ratio(e("1"), e("x-1"))]);
    let result = solve(&source, &variables);
    assert_eq!(result.numerator(), &polynomial("2*x", &variables));
    assert_eq!(result.denominator(), &polynomial("x**2-1", &variables));
    assert_eq!(
        result.guards(),
        &[polynomial("x+1", &variables), polynomial("x-1", &variables)]
    );

    let source = Expr::Add(vec![ratio(e("x"), e("x+y")), ratio(e("y"), e("x+y"))]);
    let result = solve(&source, &variables);
    assert_eq!(result.numerator(), &polynomial("1", &variables));
    assert_eq!(result.denominator(), &polynomial("1", &variables));
    // Duplicate exact guards may coalesce; their zero set is still excluded.
    assert_eq!(result.guards(), &[polynomial("x+y", &variables)]);
}

#[test]
fn nested_reciprocals_keep_poles_absent_from_the_reduced_denominator() {
    let variables = ring(&["x", "y"]);
    let source = power(Expr::Add(vec![power(e("x"), -1), power(e("y"), -1)]), -1);
    let result = solve(&source, &variables);
    assert_eq!(result.numerator(), &polynomial("x*y", &variables));
    assert_eq!(result.denominator(), &polynomial("x+y", &variables));
    assert_eq!(
        result.guards(),
        &[
            polynomial("x", &variables),
            polynomial("y", &variables),
            polynomial("x+y", &variables),
        ]
    );
    let source = power(power(e("x"), -1), -1);
    let result = solve(&source, &variables);
    assert_eq!(result.numerator(), &polynomial("x", &variables));
    assert_eq!(result.denominator(), &polynomial("1", &variables));
    assert_eq!(
        result.guards(),
        &[polynomial("x", &variables), polynomial("1", &variables)]
    );
}

#[test]
fn held_zero_products_and_zero_powers_retain_original_domain_obligations() {
    let variables = ring(&["x", "y"]);
    let inverse = power(e("x-y"), -1);
    for (source, expected) in [
        (Expr::Mul(vec![e("0"), inverse.clone()]), "0"),
        (power(inverse.clone(), 0), "1"),
    ] {
        let result = solve(&source, &variables);
        assert_eq!(result.numerator(), &polynomial(expected, &variables));
        assert_eq!(result.denominator(), &polynomial("1", &variables));
        assert_eq!(result.guards(), &[polynomial("x-y", &variables)]);
    }
    for invalid in [
        power(e("0"), -1),
        power(Expr::Add(vec![e("x"), e("-x")]), -1),
        power(Expr::Mul(vec![e("0"), inverse]), 0),
        Expr::Function("sin".into(), vec![e("x")]),
    ] {
        for source in [Expr::Mul(vec![e("0"), invalid.clone()]), power(invalid, 0)] {
            assert!(checked(&source, &variables, &mut Unbounded, &mut Unbounded).is_err());
        }
    }
}

#[test]
fn empty_ring_scalars_use_source_verification_while_localization_stays_nonempty() {
    let full: Vec<_> = (0..8).map(|i| Symbol::new(format!("v{i}"))).collect();
    for variables in [vec![], ring(&["x"]), full] {
        for (source, expected) in [
            (e("0"), q(0, 1)),
            (e("-7/3"), q(-7, 3)),
            (ratio(e("5"), e("6")), q(5, 6)),
        ] {
            let result = solve(&source, &variables);
            assert_eq!(result.numerator(), &scalar(expected, &variables));
            assert_eq!(result.denominator(), &scalar(q(1, 1), &variables));
            assert_eq!(result.variables(), &variables);
            verify(&source, &variables, result.certificate(), &mut Unbounded).unwrap();
        }
    }
    let source = ratio(e("5"), e("6"));
    let result = solve(&source, &[]);
    assert_eq!(result.guards(), &[scalar(q(6, 1), &[])]);
    assert!(checked(&e("x"), &[], &mut Unbounded, &mut Unbounded).is_err());
    assert!(
        metered_localize_rational_system(
            std::slice::from_ref(&source),
            &[],
            &mut Unbounded,
            &mut Unbounded
        )
        .is_err()
    );
    let certificate = RationalLocalizationCertificate {
        auxiliary: Symbol::new("u"),
        fractions: vec![],
        guards: vec![],
        equations: vec![],
    };
    assert!(verify_rational_localization(&[], &[], &certificate, &mut Unbounded).is_err());
}

#[test]
fn polynomial_and_already_coprime_inputs_keep_the_full_declared_ring() {
    let variables = ring(&["x", "y", "z"]);
    let polynomial_source = e("(x+y)**3");
    let result = solve(&polynomial_source, &variables);
    assert_eq!(result.numerator(), &polynomial("(x+y)**3", &variables));
    assert_eq!(result.denominator(), &polynomial("1", &variables));
    assert!(result.guards().is_empty());
    let source = ratio(e("2*x+3*y"), e("-5*z-10"));
    let result = solve(&source, &variables);
    assert_eq!(result.numerator(), &polynomial("-2/5*x-3/5*y", &variables));
    assert_eq!(result.denominator(), &polynomial("z+2", &variables));
}

#[test]
fn full_eight_axis_monomials_retain_the_bounded_high_degree_lane() {
    let variables: Vec<_> = (0..8).map(|i| Symbol::new(format!("a{i}"))).collect();
    let source = ratio(e("3/7*a0**200*a7**2"), e("-5/11*a0**100*a7"));
    let result = solve(&source, &variables);
    assert_eq!(
        result.numerator(),
        &polynomial("-33/35*a0**100*a7", &variables)
    );
    assert_eq!(result.denominator(), &polynomial("1", &variables));
    assert_eq!(result.variables(), &variables);
    assert!(
        result
            .certificate()
            .gcd
            .certificate()
            .intersection
            .is_none()
    );
}

#[test]
fn typed_same_name_axes_and_their_order_cannot_be_rebound() {
    let a = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([1; 32]));
    let b = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([2; 32]));
    let variables = vec![a.clone(), b.clone()];
    let minus_b = Expr::Mul(vec![e("-1"), Expr::Sym(b.clone())]);
    let source = ratio(
        Expr::Add(vec![
            power(Expr::Sym(a.clone()), 2),
            Expr::Mul(vec![e("-1"), power(Expr::Sym(b.clone()), 2)]),
        ]),
        Expr::Add(vec![Expr::Sym(a.clone()), minus_b]),
    );
    let expected = Polynomial::new(
        variables.clone(),
        BTreeMap::from([(vec![1, 0], q(1, 1)), (vec![0, 1], q(1, 1))]),
    )
    .unwrap();
    let result = solve(&source, &variables);
    assert_eq!(result.numerator(), &expected);
    assert_eq!(result.denominator(), &scalar(q(1, 1), &variables));
    assert!(verify(&source, &[b, a], result.certificate(), &mut Unbounded).is_err());
    let mut candidate = result.certificate().clone();
    candidate.numerator.generators[0] = Symbol::new("t");
    rejects(&result, &candidate);
}

#[test]
fn an_equivalent_but_unreduced_pair_cannot_be_accepted() {
    let variables = ring(&["x", "y"]);
    let source = ratio(e("x**2-y**2"), e("x-y"));
    let result = solve(&source, &variables);
    let extra = polynomial("x+1", &variables);
    let mut candidate = result.certificate().clone();
    candidate.numerator = multiply(result.numerator(), &extra);
    candidate.denominator = multiply(result.denominator(), &extra);
    // The two ratios are identical and the forged denominator is monic.
    // Only the complete-GCD/quotient obligations rule out this extra factor.
    rejects(&result, &candidate);
    for scale in [q(-1, 1), q(2, 3), q(7, 1)] {
        let mut candidate = result.certificate().clone();
        let scale = scalar(scale, &variables);
        candidate.numerator = multiply(result.numerator(), &scale);
        candidate.denominator = multiply(result.denominator(), &scale);
        rejects(&result, &candidate);
    }
}

#[test]
fn exact_quotient_support_and_coefficients_cannot_be_forged() {
    let variables = ring(&["x", "y"]);
    let source = ratio(e("(x+y)*(x+1)"), e("(x+y)*(y+1)"));
    let result = solve(&source, &variables);
    let mut candidate = result.certificate().clone();
    *candidate.numerator.terms.values_mut().next().unwrap() += q(1, 1_000_000_007);
    rejects(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.numerator.terms.pop_first();
    rejects(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.denominator = polynomial("y+2", &variables);
    rejects(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.denominator = Polynomial::zero(variables.clone());
    rejects(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.numerator = Polynomial::zero(variables);
    rejects(&result, &candidate);
}

#[test]
fn complete_gcd_receipt_binds_both_uncancelled_original_polynomials() {
    let variables = ring(&["x", "y"]);
    let source = ratio(e("(x+y)*(x+1)"), e("(x+y)*(y+1)"));
    let result = solve(&source, &variables);
    let original = &result.certificate().original;
    let numerator = &original.fraction().numerator;
    let denominator = &original.fraction().denominator;
    let twice = scalar(q(2, 1), &variables);
    for (left, right) in [
        (multiply(numerator, &twice), denominator.clone()),
        (numerator.clone(), multiply(denominator, &twice)),
        (denominator.clone(), numerator.clone()),
        (result.numerator().clone(), result.denominator().clone()),
    ] {
        let evidence =
            checked_multivariate_gcd(&left, &right, &mut Unbounded, &mut Unbounded).unwrap();
        let mut candidate = result.certificate().clone();
        candidate.gcd = Arc::new(evidence);
        rejects(&result, &candidate);
    }
}

#[test]
fn algebraically_equal_sources_do_not_replace_held_source_or_guard_binding() {
    let variables = ring(&["x"]);
    let source = power(power(e("x"), -1), -1);
    let result = solve(&source, &variables);
    // The value x alone omits the original inverse guard x!=0.
    let replacement =
        metered_rational_source(&e("x"), &variables, &mut Unbounded, &mut Unbounded).unwrap();
    let mut candidate = result.certificate().clone();
    candidate.original = Arc::new(replacement);
    rejects(&result, &candidate);
    // Adding a held zero preserves the exact fraction and every guard, but is
    // still a different original expression from the immutable receipt.
    let changed = Expr::Add(vec![source.clone(), e("0")]);
    let replacement =
        metered_rational_source(&changed, &variables, &mut Unbounded, &mut Unbounded).unwrap();
    let mut candidate = result.certificate().clone();
    candidate.original = Arc::new(replacement);
    rejects(&result, &candidate);
    assert!(verify(&changed, &variables, result.certificate(), &mut Unbounded).is_err());
}

#[test]
fn source_and_untrusted_output_bounds_precede_shortcuts_and_copying() {
    let variables = ring(&["x", "y"]);
    let huge = Expr::Integer(BigInt::one() << 4096u32);
    let deep = (0..66).fold(e("x"), |source, _| Expr::Add(vec![source]));
    for invalid in [
        huge,
        power(e("x+y"), 257),
        deep,
        Expr::Add(vec![e("1"); 4097]),
    ] {
        let source = Expr::Mul(vec![e("0"), invalid]);
        assert!(checked(&source, &variables, &mut Unbounded, &mut Unbounded).is_err());
    }
    for bad_ring in [
        vec![variables[0].clone(), variables[0].clone()],
        (0..9).map(|i| Symbol::new(format!("a{i}"))).collect(),
        vec![Symbol::new("a".repeat(65_537))],
    ] {
        assert!(checked(&e("0"), &bad_ring, &mut Unbounded, &mut Unbounded).is_err());
    }
    let result = solve(&ratio(e("x**2-y**2"), e("x-y")), &variables);
    let mut candidate = result.certificate().clone();
    candidate.numerator.terms.insert(vec![257, 0], q(1, 1));
    rejects(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.numerator.generators[0] = Symbol::new("x".repeat(65_537));
    rejects(&result, &candidate);
    let mut candidate = result.certificate().clone();
    candidate.numerator.terms.insert(vec![0], q(1, 1));
    rejects(&result, &candidate);
}

#[derive(Default)]
struct Meter {
    steps: u64,
    memory: u64,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.steps += amount;
        }
        if dimension == Dimension::MemoryBytes {
            self.memory += amount;
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
fn generation_verification_and_publication_keep_separate_cumulative_budgets() {
    let variables = ring(&["x", "y"]);
    let source = ratio(e("x**3*y"), e("x*y**2"));
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    checked(&source, &variables, &mut generation, &mut verification).unwrap();
    assert!(
        generation.steps > 0
            && verification.steps > 0
            && generation.memory > 0
            && verification.memory > 0
    );
    for phase in [false, true] {
        let observed = if phase { &verification } else { &generation };
        for cancel_at in [1, observed.checkpoints / 2, observed.checkpoints] {
            let mut interrupted = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let error = if phase {
                checked(&source, &variables, &mut Unbounded, &mut interrupted)
            } else {
                checked(&source, &variables, &mut interrupted, &mut Unbounded)
            }
            .unwrap_err();
            assert!(error.to_string().to_lowercase().contains("cancel"));
            assert_eq!(interrupted.checkpoints, cancel_at);
        }
        for dimension in [Dimension::ComputeSteps, Dimension::MemoryBytes] {
            let consumed = if dimension == Dimension::ComputeSteps {
                observed.steps
            } else {
                observed.memory
            };
            let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
            limits.dimensions[dimension.index()] = consumed - 1;
            let mut budget = Budget::new(limits);
            let result = if phase {
                checked(&source, &variables, &mut Unbounded, &mut budget)
            } else {
                checked(&source, &variables, &mut budget, &mut Unbounded)
            };
            assert!(result.is_err());
        }
    }
    let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = generation.steps;
    let mut budget = Budget::new(limits);
    checked(&source, &variables, &mut budget, &mut Unbounded).unwrap();
    assert!(checked(&source, &variables, &mut budget, &mut Unbounded).is_err());
}

#[test]
fn independent_replay_cancellation_never_publishes_a_partial_receipt() {
    let variables = ring(&["x"]);
    let source = ratio(e("x"), e("x"));
    let result = solve(&source, &variables);
    let mut observed = Meter::default();
    verify(&source, &variables, result.certificate(), &mut observed).unwrap();
    for cancel_at in 1..=observed.checkpoints {
        let mut meter = Meter {
            cancel_at: Some(cancel_at),
            ..Meter::default()
        };
        let error = verify(&source, &variables, result.certificate(), &mut meter).unwrap_err();
        assert!(error.to_string().to_lowercase().contains("cancel"));
        assert_eq!(meter.checkpoints, cancel_at);
    }
}

#[test]
fn accepted_expression_lifting_is_coefficient_checked_and_cancellable() {
    let variables = ring(&["x", "y"]);
    let source = ratio(e("(x+y)*(x+1)"), e("(x+y)*(y+1)"));
    let result = solve(&source, &variables);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    let expression = result
        .metered_expression(&mut generation, &mut verification)
        .unwrap();
    let lifted =
        metered_rational_source(&expression, &variables, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(&lifted.fraction().numerator, result.numerator());
    assert_eq!(&lifted.fraction().denominator, result.denominator());
    for phase in [false, true] {
        let checkpoints = if phase {
            verification.checkpoints
        } else {
            generation.checkpoints
        };
        for cancel_at in [1, checkpoints / 2, checkpoints] {
            let mut meter = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let error = if phase {
                result.metered_expression(&mut Unbounded, &mut meter)
            } else {
                result.metered_expression(&mut meter, &mut Unbounded)
            }
            .unwrap_err();
            assert!(error.to_string().to_lowercase().contains("cancel"));
        }
        let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
        assert!(
            if phase {
                result.metered_expression(&mut Unbounded, &mut empty)
            } else {
                result.metered_expression(&mut empty, &mut Unbounded)
            }
            .is_err()
        );
    }
}
