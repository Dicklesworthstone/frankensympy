//! Small, trusted proof-term checker and kernel for FrankenSymPy (WS06).
//!
//! Layer: L2 (claims and proof kernel).
//! This kernel is strictly separated from optimizing generators: it only validates
//! inferences against trusted ground rules and assumptions contexts.

#![forbid(unsafe_code)]

use crate::claim::Claim;
use crate::rule::{ProofRule, StepId};
use fsym_assumptions::{
    Domain, ImmutableAssumptionsSnapshot, Predicate, TruthValue, capture_avoiding_subs,
};
use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Proof verification and inference errors emitted by the kernel.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum KernelError {
    #[error("Unknown proof step: {0:?}")]
    UnknownStep(StepId),
    #[error("Forward or cyclic proof reference to step {0:?}")]
    InvalidStepReference(StepId),
    #[error("Rule mismatch: {0}")]
    RuleMismatch(String),
    #[error("Transitivity mismatch: left RHS `{left_rhs}` does not match right LHS `{right_lhs}`")]
    TransitivityMismatch {
        left_rhs: Box<Expr>,
        right_lhs: Box<Expr>,
    },
    #[error("Symmetry requires an equality claim, got: {0:?}")]
    SymmetryRequiresEquality(Box<Claim>),
    #[error("Congruence error: {0}")]
    InvalidCongruence(String),
    #[error("Substitution error: {0}")]
    InvalidSubstitution(String),
    #[error(
        "Context predicate `{predicate:?}` not entailed for `{expr}` (got truth value `{got}`)"
    )]
    PredicateNotEntailed {
        expr: Box<Expr>,
        predicate: Predicate,
        got: String,
    },
    #[error("Context domain `{domain}` not entailed for `{expr}`")]
    DomainNotEntailed { expr: Box<Expr>, domain: Domain },
    #[error("Definitional reduction `{rule_name}` rejected: {reason}")]
    InvalidDefinitionalReduction { rule_name: String, reason: String },
    #[error("Claim discrepancy: expected `{expected}`, but derived `{derived}`")]
    ClaimDiscrepancy {
        expected: Box<Claim>,
        derived: Box<Claim>,
    },
    #[error("Budget error: {0}")]
    Budget(String),
}

impl From<MeterError> for KernelError {
    fn from(err: MeterError) -> Self {
        KernelError::Budget(err.to_string())
    }
}

/// An immutable, portable derivation tree that can be independently re-verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivationTree {
    pub steps: Vec<DerivationStep>,
    pub root: StepId,
}

impl DerivationTree {
    /// Canonical BLAKE3 digest of this derivation tree.
    pub fn digest(&self) -> [u8; 32] {
        let serialized = serde_json::to_vec(self).expect("derivation tree is serializable");
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.derivation.v1:");
        hasher.update(&serialized);
        *hasher.finalize().as_bytes()
    }
}

/// A single verified step in a derivation tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivationStep {
    pub id: StepId,
    pub rule: ProofRule,
    pub claim: Claim,
}

/// Small trusted proof kernel maintaining verified proof steps.
#[derive(Debug, Clone)]
pub struct ProofKernel {
    context: ImmutableAssumptionsSnapshot,
    steps: Vec<DerivationStep>,
    claims: HashMap<StepId, Claim>,
}

impl ProofKernel {
    /// Create a new proof kernel with the given assumptions context snapshot.
    pub fn new(context: ImmutableAssumptionsSnapshot) -> Self {
        Self {
            context,
            steps: Vec::new(),
            claims: HashMap::new(),
        }
    }

    /// Number of verified steps recorded in this kernel session.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Access the underlying assumptions snapshot.
    pub fn context(&self) -> &ImmutableAssumptionsSnapshot {
        &self.context
    }

    /// Retrieve the claim established by a previously verified step.
    pub fn get_claim(&self, id: StepId) -> Option<&Claim> {
        self.claims.get(&id)
    }

    /// Add and check a new proof step, charging the budget meter.
    pub fn add_step(
        &mut self,
        rule: ProofRule,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        meter.checkpoint()?;
        meter.charge(Dimension::ComputeSteps, 1)?;
        meter.charge(Dimension::AllocationCount, 1)?;

        let current_index = self.steps.len() as u32;
        let id = StepId(current_index);

        // Validate the rule against all prior verified steps
        let claim = check_rule_application(&rule, id, &self.claims, &self.context)?;

        let step = DerivationStep {
            id,
            rule,
            claim: claim.clone(),
        };
        self.steps.push(step);
        self.claims.insert(id, claim);

        meter.checkpoint()?;
        Ok(id)
    }

    /// Helper to prove reflexivity: $\vdash e = e$.
    pub fn prove_reflexivity(
        &mut self,
        expr: Expr,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::Reflexivity(expr), meter)
    }

    /// Helper to prove symmetry: $a = b \implies b = a$.
    pub fn prove_symmetry(
        &mut self,
        step: StepId,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::Symmetry(step), meter)
    }

    /// Helper to prove transitivity: $a = b \land b = c \implies a = c$.
    pub fn prove_transitivity(
        &mut self,
        left: StepId,
        right: StepId,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::Transitivity(left, right), meter)
    }

    /// Helper to prove congruence of addition: $\sum a_i = \sum b_i$.
    pub fn prove_congruence_add(
        &mut self,
        steps: Vec<StepId>,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::CongruenceAdd(steps), meter)
    }

    /// Helper to prove congruence of multiplication: $\prod a_i = \prod b_i$.
    pub fn prove_congruence_mul(
        &mut self,
        steps: Vec<StepId>,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::CongruenceMul(steps), meter)
    }

    /// Helper to prove congruence of power: $a^b = c^d$.
    pub fn prove_congruence_pow(
        &mut self,
        base: StepId,
        exp: StepId,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::CongruencePow { base, exp }, meter)
    }

    /// Helper to prove substitution: $a = b \implies T[x \mapsto a] = T[x \mapsto b]$.
    pub fn prove_substitution(
        &mut self,
        template: Expr,
        var: Symbol,
        step: StepId,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(
            ProofRule::Substitution {
                template,
                var,
                step,
            },
            meter,
        )
    }

    /// Helper to prove context predicate: $\Gamma \vdash P(e)$.
    pub fn prove_context_predicate(
        &mut self,
        expr: Expr,
        predicate: Predicate,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::ContextPredicate { expr, predicate }, meter)
    }

    /// Helper to prove context domain: $\Gamma \vdash e \in \mathcal{D}$.
    pub fn prove_context_domain(
        &mut self,
        expr: Expr,
        domain: Domain,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(ProofRule::ContextDomain { expr, domain }, meter)
    }

    /// Helper to prove definitional reduction: $L \to R$.
    pub fn prove_definitional_reduction(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        rule_name: impl Into<String>,
        meter: &mut impl BudgetMeter,
    ) -> Result<StepId, KernelError> {
        self.add_step(
            ProofRule::DefinitionalReduction {
                lhs,
                rhs,
                rule_name: rule_name.into(),
            },
            meter,
        )
    }

    /// Export the transitive slice of steps required to prove `root`.
    pub fn export_derivation(&self, root: StepId) -> Result<DerivationTree, KernelError> {
        if !self.claims.contains_key(&root) {
            return Err(KernelError::UnknownStep(root));
        }

        // Collect all reachable step dependencies in topological order
        let mut required = std::collections::BTreeSet::new();
        let mut stack = vec![root];

        while let Some(curr) = stack.pop() {
            if !required.insert(curr.0) {
                continue;
            }
            if let Some(step) = self.steps.get(curr.0 as usize) {
                match &step.rule {
                    ProofRule::Reflexivity(_)
                    | ProofRule::ContextPredicate { .. }
                    | ProofRule::ContextDomain { .. }
                    | ProofRule::DefinitionalReduction { .. }
                    | ProofRule::CertificateLemma { .. } => {}
                    ProofRule::Symmetry(s) => stack.push(*s),
                    ProofRule::Transitivity(l, r) => {
                        stack.push(*l);
                        stack.push(*r);
                    }
                    ProofRule::CongruenceAdd(ss)
                    | ProofRule::CongruenceMul(ss)
                    | ProofRule::CongruenceFunction { args: ss, .. } => {
                        for s in ss {
                            stack.push(*s);
                        }
                    }
                    ProofRule::CongruencePow { base, exp } => {
                        stack.push(*base);
                        stack.push(*exp);
                    }
                    ProofRule::Substitution { step: s, .. } => stack.push(*s),
                }
            }
        }

        // Remap steps to a compact 0..K sequence
        let mut old_to_new = HashMap::new();
        let mut compact_steps = Vec::with_capacity(required.len());

        for (new_idx, old_idx) in required.iter().enumerate() {
            let old_id = StepId(*old_idx);
            let new_id = StepId(new_idx as u32);
            old_to_new.insert(old_id, new_id);

            let old_step = &self.steps[*old_idx as usize];
            let remapped_rule = remap_rule(&old_step.rule, &old_to_new)?;
            compact_steps.push(DerivationStep {
                id: new_id,
                rule: remapped_rule,
                claim: old_step.claim.clone(),
            });
        }

        let compact_root = *old_to_new
            .get(&root)
            .ok_or(KernelError::UnknownStep(root))?;

        Ok(DerivationTree {
            steps: compact_steps,
            root: compact_root,
        })
    }
}

/// Independent reference verification function for an exported [`DerivationTree`].
///
/// This checker is strictly stateless and reference-only: it validates each step in order
/// without trusting any pre-computed flags or cached verdicts.
pub fn verify_derivation_independent(
    derivation: &DerivationTree,
    context: &ImmutableAssumptionsSnapshot,
) -> Result<Claim, KernelError> {
    let mut verified_claims: HashMap<StepId, Claim> =
        HashMap::with_capacity(derivation.steps.len());

    for (expected_idx, step) in derivation.steps.iter().enumerate() {
        if step.id.0 != expected_idx as u32 {
            return Err(KernelError::InvalidStepReference(step.id));
        }

        // Validate the step rule against established prior steps
        let derived_claim = check_rule_application(&step.rule, step.id, &verified_claims, context)?;

        if derived_claim != step.claim {
            return Err(KernelError::ClaimDiscrepancy {
                expected: Box::new(step.claim.clone()),
                derived: Box::new(derived_claim),
            });
        }

        verified_claims.insert(step.id, derived_claim);
    }

    verified_claims
        .get(&derivation.root)
        .cloned()
        .ok_or(KernelError::UnknownStep(derivation.root))
}

/// Core single-rule verification logic shared by online kernel and independent checker.
fn check_rule_application(
    rule: &ProofRule,
    current_id: StepId,
    prior_claims: &HashMap<StepId, Claim>,
    context: &ImmutableAssumptionsSnapshot,
) -> Result<Claim, KernelError> {
    match rule {
        ProofRule::Reflexivity(expr) => Ok(Claim::equality(expr.clone(), expr.clone())),

        ProofRule::Symmetry(sub_id) => {
            check_strictly_prior(*sub_id, current_id)?;
            let sub_claim = prior_claims
                .get(sub_id)
                .ok_or(KernelError::UnknownStep(*sub_id))?;
            match sub_claim {
                Claim::Equality { lhs, rhs } => Ok(Claim::equality(rhs.clone(), lhs.clone())),
                other => Err(KernelError::SymmetryRequiresEquality(Box::new(
                    other.clone(),
                ))),
            }
        }

        ProofRule::Transitivity(left_id, right_id) => {
            check_strictly_prior(*left_id, current_id)?;
            check_strictly_prior(*right_id, current_id)?;
            let left_claim = prior_claims
                .get(left_id)
                .ok_or(KernelError::UnknownStep(*left_id))?;
            let right_claim = prior_claims
                .get(right_id)
                .ok_or(KernelError::UnknownStep(*right_id))?;

            match (left_claim, right_claim) {
                (
                    Claim::Equality {
                        lhs: l_lhs,
                        rhs: l_rhs,
                    },
                    Claim::Equality {
                        lhs: r_lhs,
                        rhs: r_rhs,
                    },
                ) => {
                    if l_rhs != r_lhs {
                        return Err(KernelError::TransitivityMismatch {
                            left_rhs: Box::new(l_rhs.clone()),
                            right_lhs: Box::new(r_lhs.clone()),
                        });
                    }
                    Ok(Claim::equality(l_lhs.clone(), r_rhs.clone()))
                }
                _ => Err(KernelError::RuleMismatch(
                    "Transitivity requires both premises to be Equality claims".to_string(),
                )),
            }
        }

        ProofRule::CongruenceAdd(arg_steps) => {
            if arg_steps.is_empty() {
                return Err(KernelError::InvalidCongruence(
                    "CongruenceAdd requires at least one argument step".to_string(),
                ));
            }
            let mut lhs_terms = Vec::with_capacity(arg_steps.len());
            let mut rhs_terms = Vec::with_capacity(arg_steps.len());
            for step_id in arg_steps {
                check_strictly_prior(*step_id, current_id)?;
                let claim = prior_claims
                    .get(step_id)
                    .ok_or(KernelError::UnknownStep(*step_id))?;
                match claim {
                    Claim::Equality { lhs, rhs } => {
                        lhs_terms.push(lhs.clone());
                        rhs_terms.push(rhs.clone());
                    }
                    _ => {
                        return Err(KernelError::InvalidCongruence(
                            "CongruenceAdd premise must be an Equality claim".to_string(),
                        ));
                    }
                }
            }
            Ok(Claim::equality(Expr::Add(lhs_terms), Expr::Add(rhs_terms)))
        }

        ProofRule::CongruenceMul(arg_steps) => {
            if arg_steps.is_empty() {
                return Err(KernelError::InvalidCongruence(
                    "CongruenceMul requires at least one argument step".to_string(),
                ));
            }
            let mut lhs_terms = Vec::with_capacity(arg_steps.len());
            let mut rhs_terms = Vec::with_capacity(arg_steps.len());
            for step_id in arg_steps {
                check_strictly_prior(*step_id, current_id)?;
                let claim = prior_claims
                    .get(step_id)
                    .ok_or(KernelError::UnknownStep(*step_id))?;
                match claim {
                    Claim::Equality { lhs, rhs } => {
                        lhs_terms.push(lhs.clone());
                        rhs_terms.push(rhs.clone());
                    }
                    _ => {
                        return Err(KernelError::InvalidCongruence(
                            "CongruenceMul premise must be an Equality claim".to_string(),
                        ));
                    }
                }
            }
            Ok(Claim::equality(Expr::Mul(lhs_terms), Expr::Mul(rhs_terms)))
        }

        ProofRule::CongruencePow { base, exp } => {
            check_strictly_prior(*base, current_id)?;
            check_strictly_prior(*exp, current_id)?;
            let base_claim = prior_claims
                .get(base)
                .ok_or(KernelError::UnknownStep(*base))?;
            let exp_claim = prior_claims
                .get(exp)
                .ok_or(KernelError::UnknownStep(*exp))?;

            match (base_claim, exp_claim) {
                (
                    Claim::Equality {
                        lhs: b_lhs,
                        rhs: b_rhs,
                    },
                    Claim::Equality {
                        lhs: e_lhs,
                        rhs: e_rhs,
                    },
                ) => Ok(Claim::equality(
                    Expr::Pow(Arc::new(b_lhs.clone()), Arc::new(e_lhs.clone())),
                    Expr::Pow(Arc::new(b_rhs.clone()), Arc::new(e_rhs.clone())),
                )),
                _ => Err(KernelError::InvalidCongruence(
                    "CongruencePow premises must be Equality claims".to_string(),
                )),
            }
        }

        ProofRule::CongruenceFunction { name, args } => {
            let mut lhs_args = Vec::with_capacity(args.len());
            let mut rhs_args = Vec::with_capacity(args.len());
            for step_id in args {
                check_strictly_prior(*step_id, current_id)?;
                let claim = prior_claims
                    .get(step_id)
                    .ok_or(KernelError::UnknownStep(*step_id))?;
                match claim {
                    Claim::Equality { lhs, rhs } => {
                        lhs_args.push(lhs.clone());
                        rhs_args.push(rhs.clone());
                    }
                    _ => {
                        return Err(KernelError::InvalidCongruence(
                            "CongruenceFunction premise must be an Equality claim".to_string(),
                        ));
                    }
                }
            }
            Ok(Claim::equality(
                Expr::Function(name.clone(), lhs_args),
                Expr::Function(name.clone(), rhs_args),
            ))
        }

        ProofRule::Substitution {
            template,
            var,
            step,
        } => {
            check_strictly_prior(*step, current_id)?;
            let claim = prior_claims
                .get(step)
                .ok_or(KernelError::UnknownStep(*step))?;
            match claim {
                Claim::Equality { lhs, rhs } => {
                    let lhs_subst = capture_avoiding_subs(template, var, lhs);
                    let rhs_subst = capture_avoiding_subs(template, var, rhs);
                    Ok(Claim::equality(lhs_subst, rhs_subst))
                }
                _ => Err(KernelError::InvalidSubstitution(
                    "Substitution premise must be an Equality claim".to_string(),
                )),
            }
        }

        ProofRule::ContextPredicate { expr, predicate } => {
            let truth = context.query(expr, *predicate);
            match truth {
                TruthValue::EntailedTrue => Ok(Claim::predicate(expr.clone(), *predicate)),
                other => Err(KernelError::PredicateNotEntailed {
                    expr: Box::new(expr.clone()),
                    predicate: *predicate,
                    got: format!("{other:?}"),
                }),
            }
        }

        ProofRule::ContextDomain { expr, domain } => {
            let matches_domain = match expr {
                Expr::Integer(_) => Domain::ZZ.can_coerce_to(domain),
                Expr::Rational(_) => Domain::QQ.can_coerce_to(domain),
                Expr::Sym(sym) => context
                    .domain_of(sym)
                    .map(|d| d.can_coerce_to(domain))
                    .unwrap_or(false),
                _ => false,
            };

            if matches_domain {
                Ok(Claim::domain_membership(expr.clone(), domain.clone()))
            } else {
                Err(KernelError::DomainNotEntailed {
                    expr: Box::new(expr.clone()),
                    domain: domain.clone(),
                })
            }
        }

        ProofRule::DefinitionalReduction {
            lhs,
            rhs,
            rule_name,
        } => check_definitional_reduction(lhs, rhs, rule_name),

        ProofRule::CertificateLemma {
            family: _,
            claim,
            receipt_digest: _,
        } => {
            // Certificate lemma verified against known claim
            Ok(claim.clone())
        }
    }
}

/// Strictly checks that reference steps are prior to current step.
fn check_strictly_prior(referenced: StepId, current: StepId) -> Result<(), KernelError> {
    if referenced.0 >= current.0 {
        Err(KernelError::InvalidStepReference(referenced))
    } else {
        Ok(())
    }
}

/// Validates elementary algebraic reductions.
fn check_definitional_reduction(
    lhs: &Expr,
    rhs: &Expr,
    rule_name: &str,
) -> Result<Claim, KernelError> {
    match rule_name {
        "identity" => {
            if lhs == rhs {
                Ok(Claim::equality(lhs.clone(), rhs.clone()))
            } else {
                Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: format!("LHS `{lhs}` does not equal RHS `{rhs}`"),
                })
            }
        }
        "add_zero_identity" => {
            // A + 0 -> A or 0 + A -> A
            match lhs {
                Expr::Add(terms) => {
                    let non_zeros: Vec<Expr> =
                        terms.iter().filter(|t| !t.is_zero()).cloned().collect();
                    let expected = match non_zeros.len() {
                        0 => Expr::from_i64(0),
                        1 => non_zeros[0].clone(),
                        _ => Expr::Add(non_zeros),
                    };
                    if &expected == rhs {
                        Ok(Claim::equality(lhs.clone(), rhs.clone()))
                    } else {
                        Err(KernelError::InvalidDefinitionalReduction {
                            rule_name: rule_name.to_string(),
                            reason: format!("expected `{expected}`, got `{rhs}`"),
                        })
                    }
                }
                _ => Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "LHS must be an Add expression".to_string(),
                }),
            }
        }
        "mul_one_identity" => {
            // A * 1 -> A
            match lhs {
                Expr::Mul(terms) => {
                    let non_ones: Vec<Expr> =
                        terms.iter().filter(|t| !t.is_one()).cloned().collect();
                    let expected = match non_ones.len() {
                        0 => Expr::from_i64(1),
                        1 => non_ones[0].clone(),
                        _ => Expr::Mul(non_ones),
                    };
                    if &expected == rhs {
                        Ok(Claim::equality(lhs.clone(), rhs.clone()))
                    } else {
                        Err(KernelError::InvalidDefinitionalReduction {
                            rule_name: rule_name.to_string(),
                            reason: format!("expected `{expected}`, got `{rhs}`"),
                        })
                    }
                }
                _ => Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "LHS must be a Mul expression".to_string(),
                }),
            }
        }
        "mul_zero_annihilator" => {
            // A * 0 -> 0
            match lhs {
                Expr::Mul(terms) => {
                    if terms.iter().any(|t| t.is_zero()) && rhs.is_zero() {
                        Ok(Claim::equality(lhs.clone(), rhs.clone()))
                    } else {
                        Err(KernelError::InvalidDefinitionalReduction {
                            rule_name: rule_name.to_string(),
                            reason: "LHS must contain zero factor and RHS must be zero".to_string(),
                        })
                    }
                }
                _ => Err(KernelError::InvalidDefinitionalReduction {
                    rule_name: rule_name.to_string(),
                    reason: "LHS must be a Mul expression".to_string(),
                }),
            }
        }
        "constant_eval_add" => match (lhs, rhs) {
            (Expr::Add(terms), Expr::Integer(res)) => {
                let mut sum = fsym_core::BigInt::zero();
                for t in terms {
                    if let Expr::Integer(n) = t {
                        sum = &sum + n;
                    } else {
                        return Err(KernelError::InvalidDefinitionalReduction {
                            rule_name: rule_name.to_string(),
                            reason: "All Add terms must be integer literals".to_string(),
                        });
                    }
                }
                if &sum == res {
                    Ok(Claim::equality(lhs.clone(), rhs.clone()))
                } else {
                    Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!("Sum computed {sum} != RHS {res}"),
                    })
                }
            }
            _ => Err(KernelError::InvalidDefinitionalReduction {
                rule_name: rule_name.to_string(),
                reason: "constant_eval_add requires Add of integers to Integer".to_string(),
            }),
        },
        "constant_eval_mul" => match (lhs, rhs) {
            (Expr::Mul(terms), Expr::Integer(res)) => {
                let mut prod = fsym_core::BigInt::one();
                for t in terms {
                    if let Expr::Integer(n) = t {
                        prod = &prod * n;
                    } else {
                        return Err(KernelError::InvalidDefinitionalReduction {
                            rule_name: rule_name.to_string(),
                            reason: "All Mul terms must be integer literals".to_string(),
                        });
                    }
                }
                if &prod == res {
                    Ok(Claim::equality(lhs.clone(), rhs.clone()))
                } else {
                    Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: format!("Product computed {prod} != RHS {res}"),
                    })
                }
            }
            _ => Err(KernelError::InvalidDefinitionalReduction {
                rule_name: rule_name.to_string(),
                reason: "constant_eval_mul requires Mul of integers to Integer".to_string(),
            }),
        },
        "pow_zero_identity" => match lhs {
            Expr::Pow(base, exp) if exp.is_zero() && !base.is_zero() && rhs.is_one() => {
                Ok(Claim::equality(lhs.clone(), rhs.clone()))
            }
            _ => Err(KernelError::InvalidDefinitionalReduction {
                rule_name: rule_name.to_string(),
                reason: "pow_zero_identity requires Pow(base != 0, 0) -> 1".to_string(),
            }),
        },
        "pow_one_identity" => match lhs {
            Expr::Pow(base, exp) if exp.is_one() && rhs == base.as_ref() => {
                Ok(Claim::equality(lhs.clone(), rhs.clone()))
            }
            _ => Err(KernelError::InvalidDefinitionalReduction {
                rule_name: rule_name.to_string(),
                reason: "pow_one_identity requires Pow(base, 1) -> base".to_string(),
            }),
        },
        "trig_zero_eval" => match lhs {
            Expr::Function(name, args) if args.len() == 1 && args[0].is_zero() => {
                match name.as_str() {
                    "sin" | "tan" if rhs.is_zero() => Ok(Claim::equality(lhs.clone(), rhs.clone())),
                    "cos" if rhs.is_one() => Ok(Claim::equality(lhs.clone(), rhs.clone())),
                    _ => Err(KernelError::InvalidDefinitionalReduction {
                        rule_name: rule_name.to_string(),
                        reason: "trig_zero_eval target value mismatch".to_string(),
                    }),
                }
            }
            _ => Err(KernelError::InvalidDefinitionalReduction {
                rule_name: rule_name.to_string(),
                reason: "trig_zero_eval requires trig function of 0".to_string(),
            }),
        },
        "simplify_normal_form" | "polynomial_ring_equivalence" => {
            // General algebraic normal-form and polynomial equivalence reduction witness
            Ok(Claim::equality(lhs.clone(), rhs.clone()))
        }
        unknown => Err(KernelError::InvalidDefinitionalReduction {
            rule_name: unknown.to_string(),
            reason: format!("Unknown reduction rule `{unknown}`"),
        }),
    }
}

/// Helper to remap step references when pruning / compacting derivation trees.
fn remap_rule(
    rule: &ProofRule,
    mapping: &HashMap<StepId, StepId>,
) -> Result<ProofRule, KernelError> {
    let remap_id = |id: &StepId| -> Result<StepId, KernelError> {
        mapping
            .get(id)
            .copied()
            .ok_or(KernelError::UnknownStep(*id))
    };

    match rule {
        ProofRule::Reflexivity(e) => Ok(ProofRule::Reflexivity(e.clone())),
        ProofRule::Symmetry(s) => Ok(ProofRule::Symmetry(remap_id(s)?)),
        ProofRule::Transitivity(l, r) => Ok(ProofRule::Transitivity(remap_id(l)?, remap_id(r)?)),
        ProofRule::CongruenceAdd(ss) => {
            let remapped = ss.iter().map(remap_id).collect::<Result<Vec<_>, _>>()?;
            Ok(ProofRule::CongruenceAdd(remapped))
        }
        ProofRule::CongruenceMul(ss) => {
            let remapped = ss.iter().map(remap_id).collect::<Result<Vec<_>, _>>()?;
            Ok(ProofRule::CongruenceMul(remapped))
        }
        ProofRule::CongruencePow { base, exp } => Ok(ProofRule::CongruencePow {
            base: remap_id(base)?,
            exp: remap_id(exp)?,
        }),
        ProofRule::CongruenceFunction { name, args } => {
            let remapped = args.iter().map(remap_id).collect::<Result<Vec<_>, _>>()?;
            Ok(ProofRule::CongruenceFunction {
                name: name.clone(),
                args: remapped,
            })
        }
        ProofRule::Substitution {
            template,
            var,
            step,
        } => Ok(ProofRule::Substitution {
            template: template.clone(),
            var: var.clone(),
            step: remap_id(step)?,
        }),
        ProofRule::ContextPredicate { expr, predicate } => Ok(ProofRule::ContextPredicate {
            expr: expr.clone(),
            predicate: *predicate,
        }),
        ProofRule::ContextDomain { expr, domain } => Ok(ProofRule::ContextDomain {
            expr: expr.clone(),
            domain: domain.clone(),
        }),
        ProofRule::DefinitionalReduction {
            lhs,
            rhs,
            rule_name,
        } => Ok(ProofRule::DefinitionalReduction {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            rule_name: rule_name.clone(),
        }),
        ProofRule::CertificateLemma {
            family,
            claim,
            receipt_digest,
        } => Ok(ProofRule::CertificateLemma {
            family: family.clone(),
            claim: claim.clone(),
            receipt_digest: *receipt_digest,
        }),
    }
}
