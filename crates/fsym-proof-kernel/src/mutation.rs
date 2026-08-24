//! Mutation testing harness and adversarial negative test suite for fsym-proof-kernel (WS06).
//!
//! Layer: L2 (claims and proof kernel).
//! Ensures that registered weakening mutants, forged claims, invalid transitivity links,
//! and context violations are strictly caught and rejected.

#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    use crate::claim::Claim;
    use crate::kernel::{
        DerivationStep, DerivationTree, KernelError, ProofKernel, verify_derivation_independent,
    };
    use crate::rule::{ProofRule, StepId};
    use fsym_assumptions::{AssumptionsContext, Predicate};
    use fsym_budget::Unbounded;
    use fsym_core::{Expr, Symbol};

    fn empty_context() -> fsym_assumptions::ImmutableAssumptionsSnapshot {
        AssumptionsContext::default().snapshot()
    }

    #[test]
    fn valid_derivation_chain_succeeds_and_verifies_independently() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx.clone());
        let mut meter = Unbounded;

        // Step 0: x + 0 -> x (DefinitionalReduction)
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let x_plus_0 = Expr::Add(vec![x.clone(), zero]);

        let s0 = kernel
            .prove_definitional_reduction(
                x_plus_0.clone(),
                x.clone(),
                "add_zero_identity",
                &mut meter,
            )
            .expect("valid step 0");

        // Step 1: x -> x (Reflexivity)
        let s1 = kernel
            .prove_reflexivity(x.clone(), &mut meter)
            .expect("valid step 1");

        // Step 2: x + 0 = x by Transitivity(s0, s1)
        let s2 = kernel
            .prove_transitivity(s0, s1, &mut meter)
            .expect("valid step 2");

        let derivation = kernel.export_derivation(s2).expect("export derivation");
        let claim =
            verify_derivation_independent(&derivation, &ctx).expect("independent verification");

        assert_eq!(claim, Claim::equality(x_plus_0, x));
    }

    #[test]
    fn mutant_transitivity_mismatch_killed() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx.clone());
        let mut meter = Unbounded;

        let a = Expr::symbol("a");
        let b = Expr::symbol("b");

        // Step 0: a = a
        let s0 = kernel.prove_reflexivity(a.clone(), &mut meter).unwrap();
        // Step 1: b = b
        let s1 = kernel.prove_reflexivity(b.clone(), &mut meter).unwrap();

        // Mutant attempt: Transitivity(a = a, b = b) -> left RHS 'a' != right LHS 'b'
        let err = kernel.prove_transitivity(s0, s1, &mut meter).unwrap_err();
        assert!(matches!(err, KernelError::TransitivityMismatch { .. }));
    }

    #[test]
    fn mutant_forward_or_self_reference_killed() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;

        // Attempting to self-reference step 0 before it exists
        let err = kernel
            .add_step(ProofRule::Symmetry(StepId(0)), &mut meter)
            .unwrap_err();
        assert!(matches!(err, KernelError::InvalidStepReference(StepId(0))));

        // Attempting to forward-reference step 42
        let err2 = kernel
            .add_step(ProofRule::Symmetry(StepId(42)), &mut meter)
            .unwrap_err();
        assert!(matches!(
            err2,
            KernelError::InvalidStepReference(StepId(42))
        ));
    }

    #[test]
    fn mutant_forged_context_predicate_killed() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;

        let x = Expr::symbol("x");
        // Empty context has no information on 'x' -> TruthValue::Unknown
        let err = kernel
            .prove_context_predicate(x, Predicate::Positive, &mut meter)
            .unwrap_err();
        assert!(matches!(err, KernelError::PredicateNotEntailed { .. }));
    }

    #[test]
    fn mutant_forged_definitional_arithmetic_killed() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;

        let two = Expr::from_i64(2);
        let three = Expr::from_i64(3);
        let seven = Expr::from_i64(7); // Wrong! 2 + 3 = 5 != 7

        let err = kernel
            .prove_definitional_reduction(
                Expr::Add(vec![two, three]),
                seven,
                "constant_eval_add",
                &mut meter,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            KernelError::InvalidDefinitionalReduction { .. }
        ));
    }

    #[test]
    fn mutant_claim_tampering_in_derivation_tree_killed() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx.clone());
        let mut meter = Unbounded;

        let x = Expr::symbol("x");
        let s0 = kernel.prove_reflexivity(x.clone(), &mut meter).unwrap();

        let mut derivation = kernel.export_derivation(s0).unwrap();

        // Adversarial tampering: alter the claimed outcome of the step
        let y = Expr::symbol("y");
        derivation.steps[0].claim = Claim::equality(x, y);

        let err = verify_derivation_independent(&derivation, &ctx).unwrap_err();
        assert!(matches!(err, KernelError::ClaimDiscrepancy { .. }));
    }

    #[test]
    fn mutant_unchecked_certificate_lemma_killed() {
        let ctx = empty_context();
        let forged_claim = Claim::equality(Expr::symbol("x"), Expr::symbol("y"));
        let derivation = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::CertificateLemma {
                    family: "unregistered-forged-family".to_string(),
                    claim: forged_claim.clone(),
                    receipt_digest: [0x42; 32],
                },
                claim: forged_claim,
            }],
            root: StepId(0),
        };

        let error = verify_derivation_independent(&derivation, &ctx).unwrap_err();
        assert!(matches!(
            error,
            KernelError::UnverifiedCertificateLemma { .. }
        ));
    }

    #[test]
    fn mutant_broad_normal_form_claim_killed() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;

        for rule_name in ["simplify_normal_form", "polynomial_ring_equivalence"] {
            let error = kernel
                .prove_definitional_reduction(
                    Expr::symbol("x"),
                    Expr::symbol("y"),
                    rule_name,
                    &mut meter,
                )
                .unwrap_err();
            assert!(matches!(
                error,
                KernelError::InvalidDefinitionalReduction { .. }
            ));
        }
    }

    #[test]
    fn bounded_polynomial_normal_form_proves_only_matching_identity() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx.clone());
        let mut meter = Unbounded;
        let x = Expr::symbol("x");
        let lhs = Expr::Add(vec![x.clone(), x.clone()]);
        let rhs = Expr::Mul(vec![Expr::from_i64(2), x.clone()]);

        let step = kernel
            .prove_definitional_reduction(
                lhs.clone(),
                rhs.clone(),
                "polynomial_ring_equivalence",
                &mut meter,
            )
            .unwrap();
        let derivation = kernel.export_derivation(step).unwrap();

        assert_eq!(
            verify_derivation_independent(&derivation, &ctx).unwrap(),
            Claim::AlgebraicIdentity { lhs, rhs }
        );
    }

    #[test]
    fn substitution_congruence_verified() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx.clone());
        let mut meter = Unbounded;

        // Step 0: 2 + 3 -> 5 (constant_eval_add)
        let s0 = kernel
            .prove_definitional_reduction(
                Expr::Add(vec![Expr::from_i64(2), Expr::from_i64(3)]),
                Expr::from_i64(5),
                "constant_eval_add",
                &mut meter,
            )
            .unwrap();

        // Template: x^2 + 1
        let x = Symbol::new("x");
        let template = Expr::Add(vec![
            Expr::Pow(
                std::sync::Arc::new(Expr::Sym(x.clone())),
                std::sync::Arc::new(Expr::from_i64(2)),
            ),
            Expr::from_i64(1),
        ]);

        // Step 1: (2+3)^2 + 1 = 5^2 + 1 via Substitution
        let s1 = kernel
            .prove_substitution(template, x, s0, &mut meter)
            .unwrap();

        let derivation = kernel.export_derivation(s1).unwrap();
        let claim = verify_derivation_independent(&derivation, &ctx).unwrap();

        let expected_lhs = Expr::Add(vec![
            Expr::Pow(
                std::sync::Arc::new(Expr::Add(vec![Expr::from_i64(2), Expr::from_i64(3)])),
                std::sync::Arc::new(Expr::from_i64(2)),
            ),
            Expr::from_i64(1),
        ]);
        let expected_rhs = Expr::Add(vec![
            Expr::Pow(
                std::sync::Arc::new(Expr::from_i64(5)),
                std::sync::Arc::new(Expr::from_i64(2)),
            ),
            Expr::from_i64(1),
        ]);

        assert_eq!(claim, Claim::equality(expected_lhs, expected_rhs));
    }
}
