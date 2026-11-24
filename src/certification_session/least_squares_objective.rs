//! Compile a frozen residual system into its exact squared-error objective.
//!
//! Composition and derivative proposal use ordinary resources. The protected
//! checker reconstructs the held loss from every original residual before
//! admitting its full gradient/Hessian. Neither a compilation receipt nor a
//! numerical fit establishes a root, a stationary point or a minimum.

use super::{
    CertificationResponse, CertificationSessionError, CertificationState, CompiledObjectiveId,
    CompiledResidualSystemId, MAX_OBJECTIVES, ObjectiveId, charge,
};
use crate::certification::{CertificationError, check_problem_sources};
use crate::objective::ObjectiveProblem;
use crate::request_budget::RequestBudget;
use fsym_budget::{BudgetError, BudgetMeter, MeterError, VerifierLease};

impl CertificationState {
    pub(super) fn compile_least_squares_objective(
        &mut self,
        residual_program: CompiledResidualSystemId,
        objective: ObjectiveId,
        program: CompiledObjectiveId,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        budget.checkpoint()?;
        let (_, frozen_context, system) = self
            .compiled
            .source_record(&residual_program)
            .ok_or(CertificationSessionError::UnknownCompiledSystem)?;
        if frozen_context != context_digest {
            return Err(CertificationError::UniverseMismatch {
                found: frozen_context,
                expected: context_digest,
            }
            .into());
        }
        if self.objectives.contains_key(&objective) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.objectives.len() >= MAX_OBJECTIVES {
            return Err(CertificationSessionError::ObjectLimit);
        }
        self.compiled_objectives.admit_id(&program, budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        // Validate that this authority belongs to the same cumulative ledger
        // before spending ordinary work on a candidate that cannot be admitted.
        let _ = budget.verifier_meter(lease).map_err(MeterError::from)?;
        charge(budget, 1, 4096)?;
        let candidate = system.propose_least_squares_objective_with_meter(budget)?;
        let admitted = system.verify_least_squares_objective_with_meter(
            candidate,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;

        // Keep a receiver-owned original statement for existing enclosure,
        // optimization, stationary certification and portable replay requests.
        // This existing source check prepays the complete typed structural
        // copy, including limbs and names, before either clone is allocated.
        check_problem_sources(
            std::slice::from_ref(admitted.objective()),
            admitted.variables(),
            budget,
        )?;
        let problem = ObjectiveProblem::new(
            admitted.objective().clone(),
            admitted.variables().to_vec(),
            budget,
        )?;

        // publish reserves the complete response and reaches the final safe
        // point before inserting its compiled handle. All fallible work for
        // both stores is complete: no charge, checkpoint or refusal may follow
        // that publication. Both objects own their source independently of the
        // residual handle, which can subsequently be released.
        let response = self.compiled_objectives.publish(
            objective.clone(),
            program,
            admitted,
            frozen_context,
            budget,
        )?;
        self.objectives.insert(objective, problem);
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certification_session::{CertificationRequest, ResidualSystemId};
    use fsym_budget::{Budget, BudgetLimits};

    fn compiled_state() -> (CertificationState, Budget, VerifierLease) {
        let mut state = CertificationState::default();
        let mut budget = Budget::new(BudgetLimits::uniform(1 << 40, 1 << 40));
        let lease = budget.verifier_lease().unwrap();
        let source = ResidualSystemId::new("system:loss-context").unwrap();
        for request in [
            CertificationRequest::RegisterResidualSystem {
                schema_version: 1,
                system: source.clone(),
                residuals: vec!["x-1".into(), "x-3".into()],
                variables: vec!["x".into()],
            },
            CertificationRequest::CompileResidualSystem {
                schema_version: 1,
                system: source,
                program: residual_program(),
            },
        ] {
            state
                .handle(request, [1; 32], &mut budget, Some(&lease))
                .unwrap();
        }
        (state, budget, lease)
    }

    fn residual_program() -> CompiledResidualSystemId {
        CompiledResidualSystemId::new("program:loss-context").unwrap()
    }

    fn objective() -> ObjectiveId {
        ObjectiveId::new("objective:loss-context").unwrap()
    }

    fn program() -> CompiledObjectiveId {
        CompiledObjectiveId::new("objective_program:loss-context").unwrap()
    }

    fn derive(
        state: &mut CertificationState,
        budget: &mut Budget,
        context: [u8; 32],
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        let mut checkpoint = || Ok(());
        state.compile_least_squares_objective(
            residual_program(),
            objective(),
            program(),
            context,
            &mut RequestBudget::new(budget, &mut checkpoint),
            lease,
        )
    }

    fn no_derived_objects(state: &CertificationState) {
        assert!(state.objective(&objective()).is_none());
        assert!(state.compiled_objective(&program()).is_none());
        assert!(state.compiled_system(&residual_program()).is_some());
    }

    #[test]
    fn changed_context_cannot_reinterpret_the_frozen_residual_statement() {
        let (mut state, mut budget, lease) = compiled_state();
        let before = budget.snapshot();
        let error = derive(&mut state, &mut budget, [2; 32], Some(&lease)).unwrap_err();
        assert_eq!(error.code(), "universe_mismatch");
        assert_eq!(budget.snapshot(), before);
        no_derived_objects(&state);
        derive(&mut state, &mut budget, [1; 32], Some(&lease)).unwrap();
        assert!(state.objective(&objective()).is_some());
        assert!(state.compiled_objective(&program()).is_some());
    }

    #[test]
    fn absent_or_foreign_verifier_authority_is_rejected_before_generation() {
        let (mut state, mut budget, lease) = compiled_state();
        let before = budget.snapshot();
        let error = derive(&mut state, &mut budget, [1; 32], None).unwrap_err();
        assert_eq!(error.code(), "budget_exhausted");
        assert_eq!(budget.snapshot(), before);
        no_derived_objects(&state);

        let mut unrelated = Budget::new(BudgetLimits::uniform(1 << 40, 1 << 40));
        let foreign = unrelated.verifier_lease().unwrap();
        let error = derive(&mut state, &mut budget, [1; 32], Some(&foreign)).unwrap_err();
        assert_eq!(error.code(), "budget_exhausted");
        assert_eq!(budget.snapshot(), before);
        assert_eq!(unrelated.verifier_remaining(), 1 << 40);
        no_derived_objects(&state);

        derive(&mut state, &mut budget, [1; 32], Some(&lease)).unwrap();
        assert!(budget.verifier_remaining() < before.verifier_pool);
    }
}
