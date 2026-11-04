//! A Pythagorean cancellation must consume each input monomial at most once.

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Unbounded;
use fsym_core::Expr;
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
