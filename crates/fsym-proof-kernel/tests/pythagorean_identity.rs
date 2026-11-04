//! A Pythagorean cancellation must consume each input monomial at most once.

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Unbounded;
use fsym_core::{BigRational, Constant, Expr};
use fsym_proof_kernel::{
    Claim, DerivationStep, DerivationTree, ProofKernel, ProofRule, StepId,
    verify_derivation_independent,
};
use std::sync::Arc;

fn square(name: &str, argument: &Expr) -> Expr {
    Expr::Pow(
        Arc::new(Expr::Function(name.into(), vec![argument.clone()])),
        Arc::new(Expr::from_i64(2)),
    )
}

fn product(factors: &[Expr]) -> Expr {
    Expr::Mul(factors.to_vec())
}

fn sin_cos_pair(argument: &Expr) -> Expr {
    Expr::Add(vec![square("sin", argument), square("cos", argument)])
}

fn raw_power(base: Expr, exponent: Expr) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(exponent))
}

fn rule(lhs: Expr, rhs: Expr) -> ProofRule {
    ProofRule::DefinitionalReduction {
        lhs,
        rhs,
        rule_name: "pythagorean_identity".into(),
    }
}

fn reject_direct_and_replay(lhs: Expr, rhs: Expr) {
    let context = ImmutableAssumptionsSnapshot::empty();
    let proof_rule = rule(lhs.clone(), rhs.clone());
    let mut kernel = ProofKernel::new((*context).clone());
    assert!(
        kernel.add_step(proof_rule.clone(), &mut Unbounded).is_err(),
        "accepted a false equality: {lhs} = {rhs}"
    );
    assert_eq!(kernel.step_count(), 0, "a refusal must not publish a step");
    let tree = DerivationTree {
        steps: vec![DerivationStep {
            id: StepId(0),
            rule: proof_rule,
            claim: Claim::equality(lhs, rhs),
        }],
        root: StepId(0),
    };
    assert!(verify_derivation_independent(&tree, &context).is_err());
}

fn accept_direct_and_replay(lhs: Expr, rhs: Expr) {
    let context = ImmutableAssumptionsSnapshot::empty();
    let mut kernel = ProofKernel::new((*context).clone());
    let root = kernel
        .add_step(rule(lhs.clone(), rhs.clone()), &mut Unbounded)
        .expect("valid Pythagorean equality");
    let tree = kernel.export_derivation(root).unwrap();
    assert_eq!(
        verify_derivation_independent(&tree, &context).unwrap(),
        Claim::equality(lhs, rhs)
    );
}

#[test]
fn overlapping_pairs_cannot_double_count_a_shared_monomial() {
    let x = Expr::symbol("x");
    let y = Expr::symbol("y");
    let cx = square("cos", &x);
    let sx = square("sin", &x);
    let cy = square("cos", &y);
    let sy = square("sin", &y);

    // At x = y = 0 the LHS is 1 and the proposed RHS is 2. The two
    // candidate cancellations share cx*cy; subtracting it twice is unsound.
    let terms = [
        product(&[cx.clone(), cy.clone()]),
        product(&[sx, cy.clone()]),
        product(&[cx.clone(), sy]),
    ];
    let rhs = Expr::Add(vec![cx, cy]);
    for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
        reject_direct_and_replay(
            Expr::Add(order.map(|index| terms[index].clone()).to_vec()),
            rhs.clone(),
        );
    }
}

#[test]
fn overlapping_pairs_from_different_families_cannot_reuse_a_monomial() {
    let x = Expr::symbol("x");
    let y = Expr::symbol("y");
    let cx = square("cos", &x);
    let sx = square("sin", &x);
    let chy = square("cosh", &y);
    let shy = square("sinh", &y);
    // This false equality also gives 1 = 2 at the origin. Pair ownership
    // must span different families, including their signed coefficients.
    reject_direct_and_replay(
        Expr::Add(vec![
            product(&[cx.clone(), chy.clone()]),
            product(&[sx, chy.clone()]),
            product(&[Expr::from_i64(-1), cx.clone(), shy]),
        ]),
        Expr::Add(vec![cx, chy]),
    );
}

#[test]
fn valid_disjoint_pairs_and_residual_factors_still_verify() {
    let x = Expr::symbol("x");
    let y = Expr::symbol("y");
    let z = Expr::symbol("z");
    let sx = square("sin", &x);
    let cx = square("cos", &x);
    let chy = square("cosh", &y);
    let shy = square("sinh", &y);
    accept_direct_and_replay(Expr::Add(vec![sx.clone(), cx.clone()]), Expr::from_i64(1));
    accept_direct_and_replay(
        Expr::Add(vec![
            chy.clone(),
            product(&[Expr::from_i64(-1), shy.clone()]),
        ]),
        Expr::from_i64(1),
    );
    accept_direct_and_replay(
        Expr::Add(vec![
            product(&[z.clone(), sx.clone()]),
            product(&[z.clone(), cx.clone()]),
            product(&[Expr::from_i64(2), chy]),
            product(&[Expr::from_i64(-2), shy]),
        ]),
        Expr::Add(vec![z, Expr::from_i64(2)]),
    );

    // A three-term expression with an overlapping *possible* second pair
    // still admits one sound cancellation, leaving the other term intact.
    let cy = square("cos", &y);
    let sy = square("sin", &y);
    accept_direct_and_replay(
        Expr::Add(vec![
            product(&[cx.clone(), cy.clone()]),
            product(&[sx, cy.clone()]),
            product(&[cx.clone(), sy.clone()]),
        ]),
        Expr::Add(vec![cy, product(&[cx, sy])]),
    );
}

#[test]
fn partial_functions_cannot_be_erased_or_hidden_inside_entire_functions() {
    let x = Expr::symbol("x");
    let log_zero = Expr::Function("log".into(), vec![Expr::from_i64(0)]);
    let log_x = Expr::Function("log".into(), vec![x.clone()]);
    for partial in [
        log_zero.clone(),
        log_x.clone(),
        Expr::Function("unknown".into(), vec![x.clone()]),
        Expr::Function("sin".into(), vec![]),
        Expr::Function("cos".into(), vec![x.clone(), x.clone()]),
    ] {
        reject_direct_and_replay(
            product(&[Expr::from_i64(0), partial.clone()]),
            Expr::from_i64(0),
        );
        reject_direct_and_replay(sin_cos_pair(&partial), Expr::from_i64(1));
    }
    // Finding an actual Pythagorean pair must not hide an unrelated domain loss.
    reject_direct_and_replay(
        Expr::Add(vec![
            product(&[Expr::from_i64(0), log_zero.clone()]),
            square("sin", &x),
            square("cos", &x),
        ]),
        Expr::from_i64(1),
    );
    reject_direct_and_replay(
        product(&[
            Expr::from_i64(0),
            Expr::Function("exp".into(), vec![log_zero]),
        ]),
        Expr::from_i64(0),
    );
    // Domain obligations bind the complete expression, including its argument.
    reject_direct_and_replay(
        product(&[Expr::from_i64(0), log_x]),
        product(&[
            Expr::from_i64(0),
            Expr::Function("log".into(), vec![Expr::symbol("y")]),
        ]),
    );
}

#[test]
fn meromorphic_pairs_cannot_erase_their_poles() {
    for (positive, negative) in [("sec", "tan"), ("csc", "cot")] {
        reject_direct_and_replay(
            Expr::Add(vec![
                square(positive, &Expr::symbol("x")),
                product(&[Expr::from_i64(-1), square(negative, &Expr::symbol("x"))]),
            ]),
            Expr::from_i64(1),
        );
    }
    // csc(0) and cot(0) are undefined: a ground counterexample, not merely
    // an unavailable symbolic assumption.
    reject_direct_and_replay(
        Expr::Add(vec![
            square("csc", &Expr::from_i64(0)),
            product(&[Expr::from_i64(-1), square("cot", &Expr::from_i64(0))]),
        ]),
        Expr::from_i64(1),
    );
}

#[test]
fn partial_powers_in_function_arguments_keep_their_obligations() {
    let x = Expr::symbol("x");
    for partial in [
        raw_power(Expr::from_i64(0), Expr::from_i64(0)),
        raw_power(Expr::from_i64(0), Expr::from_i64(-1)),
        raw_power(x.clone(), Expr::from_i64(0)),
        raw_power(x.clone(), Expr::from_i64(-1)),
        raw_power(x, Expr::Rational(BigRational::new(1.into(), 2.into()))),
        // Even a positive power cannot hide a partial exponent or base.
        raw_power(
            Expr::Function("log".into(), vec![Expr::from_i64(0)]),
            Expr::from_i64(2),
        ),
        raw_power(
            Expr::from_i64(2),
            Expr::Function("log".into(), vec![Expr::from_i64(0)]),
        ),
    ] {
        reject_direct_and_replay(sin_cos_pair(&partial), Expr::from_i64(1));
    }
}

#[test]
fn nonfinite_values_are_rejected_inside_function_atoms() {
    for value in [
        Constant::Infinity,
        Constant::NegativeInfinity,
        Constant::ComplexInfinity,
        Constant::NaN,
    ] {
        let nonfinite = Expr::Const(value);
        reject_direct_and_replay(sin_cos_pair(&nonfinite), Expr::from_i64(1));
        let hidden = Expr::Function("unknown".into(), vec![nonfinite]);
        reject_direct_and_replay(
            Expr::Add(vec![hidden.clone(), sin_cos_pair(&Expr::symbol("x"))]),
            Expr::Add(vec![hidden, Expr::from_i64(1)]),
        );
    }
}

#[test]
fn preserved_partial_domains_and_exact_ground_discharges_still_verify() {
    let log_x = Expr::Function("log".into(), vec![Expr::symbol("x")]);
    accept_direct_and_replay(
        Expr::Add(vec![log_x.clone(), sin_cos_pair(&log_x)]),
        Expr::Add(vec![log_x, Expr::from_i64(1)]),
    );
    let reciprocal = raw_power(Expr::symbol("x"), Expr::from_i64(-1));
    let residual = Expr::Function("unknown".into(), vec![reciprocal.clone()]);
    accept_direct_and_replay(
        Expr::Add(vec![residual.clone(), sin_cos_pair(&reciprocal)]),
        Expr::Add(vec![residual, Expr::from_i64(1)]),
    );
    for total in [
        raw_power(Expr::symbol("x"), Expr::from_i64(2)),
        raw_power(Expr::from_i64(2), Expr::from_i64(-1)),
        raw_power(Expr::Const(Constant::I), Expr::from_i64(0)),
        Expr::Function("log".into(), vec![Expr::from_i64(1)]),
    ] {
        accept_direct_and_replay(sin_cos_pair(&total), Expr::from_i64(1));
    }
    // The existing exact ground reductions sec(0)=1 and tan(0)=0 discharge
    // these particular pole obligations without trusting an optimizing simplifier.
    accept_direct_and_replay(
        Expr::Add(vec![
            square("sec", &Expr::from_i64(0)),
            product(&[Expr::from_i64(-1), square("tan", &Expr::from_i64(0))]),
        ]),
        Expr::from_i64(1),
    );
}

#[test]
fn raw_function_arguments_are_bounded_before_normalization() {
    let argument = Expr::Add(vec![Expr::from_i64(0); 16_384]);
    reject_direct_and_replay(sin_cos_pair(&argument), Expr::from_i64(1));
}
