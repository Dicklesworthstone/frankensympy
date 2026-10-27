//! Derived terms must fit before a rule clones its referenced operands.

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Unbounded;
use fsym_core::{BigInt, Expr, Symbol};
use fsym_proof_kernel::{
    Claim, DerivationStep, KernelError, ProofKernel, ProofRule, StepId,
    derivation_verification_units, verify_derivation_independent,
};
use std::sync::Arc;

fn kernel() -> ProofKernel {
    ProofKernel::new((*ImmutableAssumptionsSnapshot::empty()).clone())
}

fn nested(mut expr: Expr, depth: usize) -> Expr {
    for _ in 0..depth {
        expr = Expr::Function("f".into(), vec![expr]);
    }
    expr
}

fn assert_limit(error: KernelError, resource: &str) {
    assert!(
        matches!(error, KernelError::DerivationLimitExceeded { resource: found, .. } if found == resource),
        "unexpected refusal: {error:?}"
    );
}

// A forged claim stays tiny while its rule would copy large prior operands.
fn reject_both(mut kernel: ProofKernel, root: StepId, rule: ProofRule, resource: &str) {
    let mut tree = kernel.export_derivation(root).unwrap();
    let id = StepId(tree.steps.len() as u32);
    tree.steps.push(DerivationStep {
        id,
        rule: rule.clone(),
        claim: Claim::equality(Expr::from_i64(0), Expr::from_i64(0)),
    });
    tree.root = id;
    assert!(
        derivation_verification_units(&tree).is_ok(),
        "encoded proof fits"
    );
    assert_limit(
        verify_derivation_independent(&tree, &ImmutableAssumptionsSnapshot::empty()).unwrap_err(),
        resource,
    );
    let before = kernel.step_count();
    assert_limit(kernel.add_step(rule, &mut Unbounded).unwrap_err(), resource);
    assert_eq!(
        kernel.step_count(),
        before,
        "refused steps must not publish"
    );
}

#[test]
fn repeated_congruence_operands_refuse_before_expanding_small_claims() {
    for rule in [
        ProofRule::CongruenceAdd(vec![StepId(0); 128]),
        ProofRule::CongruenceMul(vec![StepId(0); 128]),
        ProofRule::CongruenceFunction {
            name: "g".into(),
            args: vec![StepId(0); 128],
        },
    ] {
        let mut kernel = kernel();
        let base = Expr::Add(vec![Expr::from_i64(1); 1024]);
        let root = kernel.prove_reflexivity(base, &mut Unbounded).unwrap();
        reject_both(kernel, root, rule, "constructed expression nodes");
    }
}

#[test]
fn repeated_numeric_and_text_payloads_are_bounded_separately() {
    let mut numeric = kernel();
    let root = numeric
        .prove_reflexivity(Expr::Integer(BigInt::from(2).pow(4096)), &mut Unbounded)
        .unwrap();
    reject_both(
        numeric,
        root,
        ProofRule::CongruenceAdd(vec![root; 2048]),
        "constructed numeric limbs",
    );
    let mut text = kernel();
    let root = text
        .prove_reflexivity(Expr::symbol("s".repeat(2048)), &mut Unbounded)
        .unwrap();
    reject_both(
        text,
        root,
        ProofRule::CongruenceFunction {
            name: "g".into(),
            args: vec![root; 512],
        },
        "constructed text bytes",
    );
}

#[test]
fn constructors_cannot_push_a_valid_operand_past_the_depth_limit() {
    for rule in [
        ProofRule::CongruenceFunction {
            name: "g".into(),
            args: vec![StepId(0)],
        },
        ProofRule::CongruencePow {
            base: StepId(0),
            exp: StepId(0),
        },
    ] {
        let mut kernel = kernel();
        let root = kernel
            .prove_reflexivity(nested(Expr::from_i64(1), 256), &mut Unbounded)
            .unwrap();
        reject_both(kernel, root, rule, "constructed expression depth");
    }
}

#[test]
fn substitution_counts_repeated_targets_before_copying_the_replacement() {
    let mut kernel = kernel();
    let root = kernel
        .prove_reflexivity(
            Expr::Function("g".into(), vec![Expr::from_i64(1); 1024]),
            &mut Unbounded,
        )
        .unwrap();
    reject_both(
        kernel,
        root,
        ProofRule::Substitution {
            template: Expr::Add(vec![Expr::symbol("x"); 512]),
            var: Symbol::new("x"),
            step: root,
        },
        "constructed expression nodes",
    );
}

#[test]
fn substitution_projects_the_combined_depth_before_construction() {
    let mut kernel = kernel();
    let root = kernel
        .prove_reflexivity(nested(Expr::from_i64(1), 100), &mut Unbounded)
        .unwrap();
    reject_both(
        kernel,
        root,
        ProofRule::Substitution {
            template: nested(Expr::symbol("x"), 200),
            var: Symbol::new("x"),
            step: root,
        },
        "constructed expression depth",
    );
}

#[test]
fn normal_congruence_substitution_and_capture_avoidance_still_replay() {
    let mut kernel = kernel();
    let one = kernel
        .prove_reflexivity(Expr::from_i64(1), &mut Unbounded)
        .unwrap();
    let y = kernel
        .prove_reflexivity(Expr::symbol("y"), &mut Unbounded)
        .unwrap();
    let mut roots = vec![
        kernel
            .prove_congruence_add(vec![one, y, one], &mut Unbounded)
            .unwrap(),
        kernel
            .prove_congruence_mul(vec![one, y, one], &mut Unbounded)
            .unwrap(),
        kernel.prove_congruence_pow(y, one, &mut Unbounded).unwrap(),
        kernel
            .prove_congruence_function("f", vec![one, y], &mut Unbounded)
            .unwrap(),
    ];
    let template = Expr::Function(
        "Lambda".into(),
        vec![
            Expr::symbol("y"),
            Expr::Add(vec![Expr::symbol("x"), Expr::symbol("y")]),
        ],
    );
    let substitution = kernel
        .prove_substitution(template, Symbol::new("x"), y, &mut Unbounded)
        .unwrap();
    let renamed = Expr::Function(
        "Lambda".into(),
        vec![
            Expr::symbol("y_1"),
            Expr::Add(vec![Expr::symbol("y"), Expr::symbol("y_1")]),
        ],
    );
    assert_eq!(
        kernel.get_claim(substitution),
        Some(&Claim::equality(renamed.clone(), renamed))
    );
    roots.push(substitution);
    for root in roots {
        let derivation = kernel.export_derivation(root).unwrap();
        assert_eq!(
            verify_derivation_independent(&derivation, &ImmutableAssumptionsSnapshot::empty())
                .unwrap(),
            *kernel.get_claim(root).unwrap()
        );
    }
}

#[test]
fn an_absent_substitution_target_does_not_multiply_operand_sizes() {
    let mut kernel = kernel();
    let large = kernel
        .prove_reflexivity(Expr::Add(vec![Expr::from_i64(1); 60_000]), &mut Unbounded)
        .unwrap();
    let template = Expr::Pow(
        Arc::new(Expr::from_i64(2)),
        Arc::new(Expr::from_i64(100_000)),
    );
    let root = kernel
        .prove_substitution(
            template.clone(),
            Symbol::new("absent"),
            large,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(
        kernel.get_claim(root),
        Some(&Claim::equality(template.clone(), template))
    );
    let tree = kernel.export_derivation(root).unwrap();
    assert!(verify_derivation_independent(&tree, &ImmutableAssumptionsSnapshot::empty()).is_ok());
}
