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
        DerivationStep, DerivationTree, KernelError, MAX_DERIVATION_NUMERIC_LIMBS,
        MAX_DERIVATION_STEPS, ProofKernel, derivation_verification_units,
        verify_derivation_independent,
    };
    use crate::rule::{ProofRule, StepId};
    use fsym_assumptions::{AssumptionsContext, BinderNode, Domain, Predicate};
    use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
    use fsym_core::{Constant, Expr, Symbol};

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
    fn substitution_refuses_non_alpha_derivative_variable_renaming() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;

        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let premise = kernel
            .prove_reflexivity(Expr::Sym(x.clone()), &mut meter)
            .expect("reflexivity premise");
        let template = BinderNode::derivative(
            x.clone(),
            Expr::Mul(vec![Expr::Sym(x), Expr::Sym(y.clone())]),
        )
        .to_expr();

        let error = kernel
            .prove_substitution(template, y, premise, &mut meter)
            .unwrap_err();
        assert!(matches!(error, KernelError::InvalidSubstitution(message)
                if message.contains("non-alpha-equivalent variable rename")));
    }

    #[test]
    fn substitution_renames_derivative_variable_when_symbolic() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;

        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let premise = kernel
            .prove_reflexivity(Expr::Sym(y.clone()), &mut meter)
            .expect("reflexivity premise");
        let step = kernel
            .prove_substitution(
                BinderNode::derivative(x.clone(), Expr::pow(Expr::Sym(x), Expr::from_i64(2)))
                    .to_expr(),
                Symbol::new("x"),
                premise,
                &mut meter,
            )
            .expect("symbolic operator-variable substitution is representable");
        let renamed =
            BinderNode::derivative(y.clone(), Expr::pow(Expr::Sym(y), Expr::from_i64(2))).to_expr();
        assert_eq!(
            kernel.get_claim(step),
            Some(&Claim::equality(renamed.clone(), renamed))
        );
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
    fn valid_real_ball_certificate_lemma_proves_ground_claims() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx.clone());
        let mut meter = Unbounded;

        // Positive test 1: Prove Predicate::Positive on ground expression 2 + 3
        let expr = Expr::Add(vec![Expr::from_i64(2), Expr::from_i64(3)]);
        let ball = fsym_core::RealBall::exact(fsym_core::BigRational::from_integer(
            fsym_core::BigInt::from(5),
        ));
        let claim = Claim::predicate(expr, Predicate::Positive);
        let s0 = kernel
            .prove_certificate_lemma(
                "RealBall",
                claim.clone(),
                crate::rule::CertificatePayload::RealBall(ball),
                &mut meter,
            )
            .expect("valid RealBall certificate lemma step");

        let derivation = kernel.export_derivation(s0).expect("export derivation");
        let verified =
            verify_derivation_independent(&derivation, &ctx).expect("independent verification");
        assert_eq!(verified, claim);

        // Positive test 2: Prove Domain::RR membership on ground value 42
        let expr42 = Expr::from_i64(42);
        let ball42 = fsym_core::RealBall::from_i64(42);
        let claim42 = Claim::domain_membership(expr42, Domain::RR);
        let s1 = kernel
            .prove_certificate_lemma(
                "RealBall",
                claim42.clone(),
                crate::rule::CertificatePayload::RealBall(ball42),
                &mut meter,
            )
            .expect("valid RealBall domain step");
        assert_eq!(kernel.get_claim(s1), Some(&claim42));
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
                    certificate: crate::rule::CertificatePayload::Opaque {
                        receipt_digest: [0x42; 32],
                    },
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
    fn mutant_opaque_payload_for_real_ball_rejected() {
        let ctx = empty_context();
        let claim = Claim::predicate(Expr::from_i64(5), Predicate::Positive);
        let derivation = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::CertificateLemma {
                    family: "RealBall".to_string(),
                    claim: claim.clone(),
                    certificate: crate::rule::CertificatePayload::Opaque {
                        receipt_digest: [0x42; 32],
                    },
                },
                claim,
            }],
            root: StepId(0),
        };

        let error = verify_derivation_independent(&derivation, &ctx).unwrap_err();
        assert!(matches!(
            error,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
        ));
    }

    #[test]
    fn registered_family_cannot_authorize_unrelated_symbol() {
        let ctx = empty_context();
        let ball = fsym_core::RealBall::new(
            fsym_core::BigRational::from_integer(fsym_core::BigInt::from(3)),
            fsym_core::BigRational::from_integer(fsym_core::BigInt::from(1)),
        )
        .unwrap();
        let forged_claim = Claim::domain_membership(Expr::symbol("x"), Domain::RR);
        let derivation = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::CertificateLemma {
                    family: "RealBall".to_string(),
                    claim: forged_claim.clone(),
                    certificate: crate::rule::CertificatePayload::RealBall(ball),
                },
                claim: forged_claim,
            }],
            root: StepId(0),
        };

        let error = verify_derivation_independent(&derivation, &ctx).unwrap_err();
        assert!(matches!(
            error,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
        ));
    }

    #[test]
    fn mutant_inconsistent_real_ball_claims_rejected() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;

        // Inconsistent claim 1: Negative number claimed to be Positive
        let neg5 = Expr::from_i64(-5);
        let ball_neg5 = fsym_core::RealBall::from_i64(-5);
        let false_pos_claim = Claim::predicate(neg5, Predicate::Positive);
        let err1 = kernel
            .prove_certificate_lemma(
                "RealBall",
                false_pos_claim,
                crate::rule::CertificatePayload::RealBall(ball_neg5),
                &mut meter,
            )
            .unwrap_err();
        assert!(matches!(
            err1,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
        ));

        // Inconsistent claim 2: Zero claimed to be NonZero
        let zero = Expr::from_i64(0);
        let ball_zero = fsym_core::RealBall::from_i64(0);
        let false_nz_claim = Claim::non_zero(zero);
        let err2 = kernel
            .prove_certificate_lemma(
                "RealBall",
                false_nz_claim,
                crate::rule::CertificatePayload::RealBall(ball_zero),
                &mut meter,
            )
            .unwrap_err();
        assert!(matches!(
            err2,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
        ));
    }

    #[test]
    fn mutant_real_ball_enclosure_check_deletion_killed() {
        let mut kernel = ProofKernel::new(empty_context());
        let mut meter = Unbounded;
        let false_positive = Claim::predicate(Expr::from_i64(-5), Predicate::Positive);

        let error = kernel
            .prove_certificate_lemma(
                "RealBall",
                false_positive,
                crate::rule::CertificatePayload::RealBall(fsym_core::RealBall::from_i64(5)),
                &mut meter,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
        ));
        assert_eq!(kernel.step_count(), 0);
    }

    #[test]
    fn real_ball_refuses_equality_without_equality_specific_payload() {
        let mut kernel = ProofKernel::new(empty_context());
        let mut meter = Unbounded;
        let true_equality = Claim::equality(Expr::from_i64(1), Expr::from_i64(1));

        let error = kernel
            .prove_certificate_lemma(
                "RealBall",
                true_equality,
                crate::rule::CertificatePayload::RealBall(fsym_core::RealBall::from_i64(999)),
                &mut meter,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
        ));
        assert_eq!(kernel.step_count(), 0);
    }

    #[test]
    fn oversized_real_ball_certificate_is_refused_by_preflight() {
        let bit_count = u32::try_from(MAX_DERIVATION_NUMERIC_LIMBS * 64)
            .expect("configured limb limit fits the bigint shift API");
        let huge_midpoint = fsym_core::BigInt::from(1) << bit_count;
        let certificate =
            fsym_core::RealBall::exact(fsym_core::BigRational::from_integer(huge_midpoint));
        let claim = Claim::predicate(Expr::from_i64(1), Predicate::Positive);
        let derivation = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::CertificateLemma {
                    family: "RealBall".to_string(),
                    claim: claim.clone(),
                    certificate: crate::rule::CertificatePayload::RealBall(certificate),
                },
                claim,
            }],
            root: StepId(0),
        };

        assert!(matches!(
            derivation_verification_units(&derivation),
            Err(KernelError::DerivationLimitExceeded {
                resource: "numeric limbs",
                ..
            })
        ));
    }

    #[test]
    fn real_ball_power_growth_is_refused_before_oversized_multiplication() {
        let base_limb_count = MAX_DERIVATION_NUMERIC_LIMBS / 24;
        let bit_count = u32::try_from(base_limb_count * 64)
            .expect("configured limb limit fits the bigint shift API");
        let denominator = fsym_core::BigInt::from(1) << bit_count;
        let base = Expr::Rational(fsym_core::BigRational::new(
            fsym_core::BigInt::from(1),
            denominator,
        ));
        let claim = Claim::predicate(base.pow(Expr::from_i64(3)), Predicate::Positive);
        let mut kernel = ProofKernel::new(empty_context());
        let mut meter = Unbounded;

        let error = kernel
            .prove_certificate_lemma(
                "RealBall",
                claim,
                crate::rule::CertificatePayload::RealBall(fsym_core::RealBall::from_i64(1)),
                &mut meter,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            KernelError::DerivationLimitExceeded {
                resource: "RealBall intermediate numeric limbs",
                ..
            }
        ));
        assert_eq!(kernel.step_count(), 0);
    }

    #[test]
    fn mutant_real_ball_zero_to_zero_admission_killed() {
        let ctx = empty_context();
        let undefined = Expr::from_i64(0).pow(Expr::from_i64(0));
        let forged_claim = Claim::non_zero(undefined);
        let certificate = fsym_core::RealBall::from_i64(1);

        let mut kernel = ProofKernel::new(ctx.clone());
        let mut meter = Unbounded;
        let online_error = kernel
            .prove_certificate_lemma(
                "RealBall",
                forged_claim.clone(),
                crate::rule::CertificatePayload::RealBall(certificate.clone()),
                &mut meter,
            )
            .unwrap_err();
        assert!(matches!(
            online_error,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
        ));

        let derivation = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::CertificateLemma {
                    family: "RealBall".to_string(),
                    claim: forged_claim.clone(),
                    certificate: crate::rule::CertificatePayload::RealBall(certificate),
                },
                claim: forged_claim,
            }],
            root: StepId(0),
        };
        let independent_error = verify_derivation_independent(&derivation, &ctx).unwrap_err();
        assert!(matches!(
            independent_error,
            KernelError::InvalidCertificateLemma { family, .. } if family == "RealBall"
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
    fn mutant_partial_values_cannot_be_erased_by_polynomial_zero() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;
        let x = Expr::symbol("x");
        let zero = Expr::from_i64(0);
        let negative_one = Expr::from_i64(-1);
        let partial_inputs = [
            x.clone().pow(negative_one),
            Expr::Const(Constant::Infinity),
            Expr::Function("log".to_string(), vec![x]),
        ];

        for partial in partial_inputs {
            let lhs = Expr::Mul(vec![zero.clone(), partial]);
            for rule_name in ["mul_zero_annihilator", "polynomial_ring_equivalence"] {
                let error = kernel
                    .prove_definitional_reduction(lhs.clone(), zero.clone(), rule_name, &mut meter)
                    .unwrap_err();
                assert!(matches!(
                    error,
                    KernelError::InvalidDefinitionalReduction { .. }
                ));
            }
        }
    }

    #[test]
    fn mutant_polynomial_proof_refuses_excessive_depth_at_trust_boundary() {
        let ctx = empty_context();
        let mut kernel = ProofKernel::new(ctx);
        let mut meter = Unbounded;
        let mut deep = Expr::symbol("x");
        for _ in 0..300 {
            deep = Expr::Add(vec![deep]);
        }

        let error = kernel
            .prove_definitional_reduction(
                deep.clone(),
                deep,
                "polynomial_ring_equivalence",
                &mut meter,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            KernelError::DerivationLimitExceeded {
                resource: "expression depth",
                ..
            }
        ));
    }

    #[test]
    fn mutant_deep_reflexivity_derivation_is_refused_before_replay() {
        let ctx = empty_context();
        let mut deep = Expr::symbol("x");
        for _ in 0..300 {
            deep = Expr::Add(vec![deep]);
        }
        let claim = Claim::equality(deep.clone(), deep.clone());
        let derivation = DerivationTree {
            steps: vec![DerivationStep {
                id: StepId(0),
                rule: ProofRule::Reflexivity(deep),
                claim,
            }],
            root: StepId(0),
        };

        assert!(matches!(
            verify_derivation_independent(&derivation, &ctx),
            Err(KernelError::DerivationLimitExceeded {
                resource: "expression depth",
                ..
            })
        ));
    }

    #[test]
    fn mutant_derivation_step_flood_is_refused() {
        let x = Expr::symbol("x");
        let claim = Claim::equality(x.clone(), x.clone());
        let step = DerivationStep {
            id: StepId(0),
            rule: ProofRule::Reflexivity(x),
            claim,
        };
        let derivation = DerivationTree {
            steps: vec![step; MAX_DERIVATION_STEPS + 1],
            root: StepId(0),
        };

        assert!(matches!(
            derivation_verification_units(&derivation),
            Err(KernelError::DerivationLimitExceeded {
                resource: "steps",
                ..
            })
        ));
    }

    #[test]
    fn online_step_charge_is_atomic_across_coupled_dimensions() {
        let mut limits = BudgetLimits::uniform(1, 0);
        limits.dimensions[Dimension::AllocationCount.index()] = 0;
        let mut meter = Budget::new(limits);
        let mut kernel = ProofKernel::new(empty_context());

        assert!(
            kernel
                .prove_reflexivity(Expr::symbol("x"), &mut meter)
                .is_err()
        );
        assert_eq!(meter.remaining(Dimension::ComputeSteps), 1);
        assert_eq!(meter.remaining(Dimension::AllocationCount), 0);
        assert_eq!(kernel.step_count(), 0);
    }

    #[test]
    fn cancellation_before_publication_does_not_leave_a_verified_step() {
        struct CancelBeforePublication {
            checkpoints: usize,
        }

        impl BudgetMeter for CancelBeforePublication {
            fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
                Ok(())
            }

            fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
                Ok(())
            }

            fn checkpoint(&mut self) -> Result<(), MeterError> {
                self.checkpoints += 1;
                if self.checkpoints == 2 {
                    Err(MeterError::Cancelled)
                } else {
                    Ok(())
                }
            }
        }

        let mut meter = CancelBeforePublication { checkpoints: 0 };
        let mut kernel = ProofKernel::new(empty_context());

        assert!(matches!(
            kernel.prove_reflexivity(Expr::symbol("x"), &mut meter),
            Err(KernelError::Budget(_))
        ));
        assert_eq!(kernel.step_count(), 0);
    }

    #[test]
    fn online_step_refuses_before_publication_when_preflight_work_exceeds_budget() {
        let limits = BudgetLimits::uniform(1, 0);
        let mut meter = Budget::new(limits);
        let mut kernel = ProofKernel::new(empty_context());
        let wide_name = "x".repeat(4_096);

        assert!(matches!(
            kernel.prove_reflexivity(Expr::symbol(wide_name), &mut meter),
            Err(KernelError::Budget(_))
        ));
        assert_eq!(kernel.step_count(), 0);
        assert_eq!(meter.remaining(Dimension::ComputeSteps), 0);
    }

    #[test]
    fn domain_generator_flood_is_refused_by_claim_preflight() {
        let generators = (0..4_097)
            .map(|index| Symbol::new(format!("x{index}")))
            .collect();
        let claim =
            Claim::domain_membership(Expr::symbol("x"), Domain::poly_ring(Domain::ZZ, generators));

        assert!(matches!(
            crate::claim_verification_units(&claim),
            Err(KernelError::DerivationLimitExceeded {
                resource: "domain generators",
                ..
            })
        ));
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
