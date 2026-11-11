//! Session ownership and publication for complete parameter-root families.
//!
//! Unknowns and parameters have separate immutable roles. A generated cover
//! remains a candidate until every parameter cell passes independent replay.
//! Sensitivities consume only an accepted family and recompute the original
//! source derivatives with the same session's protected verifier allowance.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationRequest, CertificationResponse,
    CertificationSessionError, ExactInterval, ExactRootWitness, MAX_AXES,
    MAX_SESSION_ROOT_ARTIFACT_BYTES, admit_response, certification_error_code, charge,
    parse_domain, parse_witness, render_domain, render_hessian, render_scalar, valid_label,
};
use crate::certification::CertificationError;
pub use crate::parametric::ParametricFamilyReceipt;
use crate::parametric::{ParametricError, ParametricProblem};
use crate::parametric_transport::{export_parametric_family, replay_parametric_family};
use fsym_budget::{Budget, BudgetError, BudgetMeter, MeterError, VerifierLease};
use fsym_calculus::parametric_cover::ParametricRootCoverOptions;
use fsym_core::{
    ParametricRootBoxWitness, ParametricRootCoverError, ParametricSensitivityReceipt, RootBoxError,
};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

const MAX_PROBLEMS: usize = 32;
const MAX_FAMILIES: usize = 32;
const MAX_SENSITIVITIES: usize = 32;

macro_rules! object_id {
    ($(#[$doc:meta])* $name:ident, $prefix:literal) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
        #[serde(transparent)]
        pub struct $name(Box<str>);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
                let value = value.into();
                if !valid_label(&value, $prefix) {
                    return Err(CertificationSessionError::InvalidLabel);
                }
                Ok(Self(value.into_boxed_str()))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
            }
        }
    };
}

object_id!(
    /// One immutable residual problem with ordered unknown and parameter axes.
    ParametricProblemId, "parametric:"
);
object_id!(
    /// An accepted unique root family on its complete declared parameter box.
    ParametricFamilyId, "family:"
);
object_id!(
    /// Independently recomputed solution-Jacobian bounds for an accepted family.
    ParametricSensitivityId, "sensitivity:"
);

/// Exact scalar inputs for a uniform Krawczyk candidate. Parameter intervals
/// may be points; every unknown coordinate must have strictly positive width.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactParametricRootWitness {
    pub unknown_domain: Vec<ExactInterval>,
    pub parameter_domain: Vec<ExactInterval>,
    pub center: Vec<String>,
    pub preconditioner: Vec<Vec<String>>,
    pub precision_digits: u32,
}

pub(super) fn error_code(error: &ParametricError) -> &'static str {
    match error {
        ParametricError::Certification(error) => certification_error_code(error),
        ParametricError::Cover(ParametricRootCoverError::Verification(error)) => match error {
            RootBoxError::Meter(MeterError::Cancelled) => "cancelled",
            RootBoxError::Meter(MeterError::Budget(_)) => "budget_exhausted",
            RootBoxError::Domain(_) => "domain_not_certified",
            RootBoxError::Unsupported(_) => "unsupported_operation",
            RootBoxError::ResourceLimit(_) => "resource_limit_exceeded",
            _ => "refused",
        },
        ParametricError::Cover(ParametricRootCoverError::Incomplete(_)) => "inconclusive",
        ParametricError::Cover(ParametricRootCoverError::InvalidPartition(_)) => "invalid_artifact",
    }
}

fn parse_uniform_witness(
    witness: ExactParametricRootWitness,
    unknowns: usize,
    parameters: usize,
    meter: &mut impl BudgetMeter,
) -> Result<ParametricRootBoxWitness, CertificationSessionError> {
    let parameter_domain = parse_domain(&witness.parameter_domain, parameters, meter)?;
    let unknown = parse_witness(
        &ExactRootWitness {
            domain: witness.unknown_domain,
            center: witness.center,
            preconditioner: witness.preconditioner,
            precision_digits: witness.precision_digits,
        },
        unknowns,
        meter,
    )?;
    Ok(ParametricRootBoxWitness {
        unknown_domain: unknown.domain,
        parameter_domain,
        center: unknown.center,
        preconditioner: unknown.preconditioner,
        precision_digits: unknown.precision_digits,
    })
}

struct ResidentFamily {
    problem: ParametricProblemId,
    receipt: ParametricFamilyReceipt,
    context_digest: [u8; 32],
}

/// A replay statement belongs to the receiver. An artifact cannot choose its
/// own source, axis roles or complete domain through the transport envelope.
pub(super) struct FamilyReplay {
    pub(super) problem: ParametricProblemId,
    pub(super) family: ParametricFamilyId,
    pub(super) unknown_domain: Vec<ExactInterval>,
    pub(super) parameter_domain: Vec<ExactInterval>,
}

#[derive(Default)]
pub(super) struct ParametricState {
    problems: BTreeMap<ParametricProblemId, ParametricProblem>,
    families: BTreeMap<ParametricFamilyId, ResidentFamily>,
    sensitivities: BTreeMap<ParametricSensitivityId, ParametricSensitivityReceipt>,
}

impl ParametricState {
    pub(super) fn problem(&self, id: &ParametricProblemId) -> Option<&ParametricProblem> {
        self.problems.get(id)
    }

    pub(super) fn family(&self, id: &ParametricFamilyId) -> Option<&ParametricFamilyReceipt> {
        self.families.get(id).map(|entry| &entry.receipt)
    }

    pub(super) fn sensitivity(
        &self,
        id: &ParametricSensitivityId,
    ) -> Option<&ParametricSensitivityReceipt> {
        self.sensitivities.get(id)
    }

    pub(super) fn export_bytes(
        &self,
        family: &ParametricFamilyId,
        meter: &mut impl BudgetMeter,
    ) -> Result<(ParametricProblemId, Vec<u8>), CertificationSessionError> {
        let slot = self
            .families
            .get(family)
            .ok_or(CertificationSessionError::UnknownParametricFamily)?;
        charge(meter, 1, 128)?;
        let bytes = export_parametric_family(&slot.receipt, slot.context_digest, meter)?;
        Ok((slot.problem.clone(), bytes))
    }

    pub(super) fn replay(
        &mut self,
        request: FamilyReplay,
        bytes: &[u8],
        context_digest: [u8; 32],
        budget: &mut Budget,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_family(&request.family, budget)?;
        let source = self
            .problems
            .get(&request.problem)
            .ok_or(CertificationSessionError::UnknownParametricProblem)?;
        let unknown_domain =
            parse_domain(&request.unknown_domain, source.unknowns().len(), budget)?;
        let parameter_domain =
            parse_domain(&request.parameter_domain, source.parameters().len(), budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        let receipt = replay_parametric_family(
            bytes,
            source,
            &unknown_domain,
            &parameter_domain,
            context_digest,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;
        self.publish_family(
            request.problem,
            request.family,
            receipt,
            context_digest,
            budget,
        )
    }

    fn admit_family(
        &self,
        id: &ParametricFamilyId,
        meter: &mut impl BudgetMeter,
    ) -> Result<(), CertificationSessionError> {
        if self.families.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.families.len() >= MAX_FAMILIES {
            return Err(CertificationSessionError::ObjectLimit);
        }
        charge(meter, 1, 4096)
    }

    fn publish_family(
        &mut self,
        problem: ParametricProblemId,
        family: ParametricFamilyId,
        receipt: ParametricFamilyReceipt,
        context_digest: [u8; 32],
        meter: &mut impl BudgetMeter,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        // The maximum local contraction bound is descriptive; each original
        // cell already passed its own independent strict contraction check.
        let mut largest = receipt.leaves()[0].contraction_bound();
        for leaf in receipt.leaves().iter().skip(1) {
            let candidate = leaf.contraction_bound();
            let limbs = (largest.numer().limb_count()
                + largest.denom().limb_count()
                + candidate.numer().limb_count()
                + candidate.denom().limb_count()) as usize;
            charge(meter, limbs.saturating_pow(2).max(1), 128 + 32 * limbs)?;
            if candidate > largest {
                largest = candidate;
            }
        }
        let response = CertificationResponse::ParametricFamilyCertified {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            problem: problem.clone(),
            family: family.clone(),
            evidence: "certified_numeric",
            claim: "for_every_parameter_exactly_one_nonsingular_root_in_unknown_box",
            unknown_domain: render_domain(receipt.unknown_domain(), meter)?,
            parameter_domain: render_domain(receipt.parameter_domain(), meter)?,
            root_enclosure: render_domain(receipt.root_enclosure(), meter)?,
            contraction_bound: render_scalar(largest, meter)?,
            leaf_count: receipt.leaf_count(),
            precision_digits: receipt.precision_digits(),
            context_digest,
        };
        // Accepted data becomes resident only after the entire output and the
        // final cancellation/resource safe point have been admitted.
        admit_response(&response, meter)?;
        self.families.insert(
            family,
            ResidentFamily {
                problem,
                receipt,
                context_digest,
            },
        );
        Ok(response)
    }

    pub(super) fn handle(
        &mut self,
        request: CertificationRequest,
        context_digest: [u8; 32],
        budget: &mut Budget,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        match request {
            CertificationRequest::ExportParametricFamily { family, .. } => {
                let (problem, bytes) = self.export_bytes(&family, budget)?;
                if bytes.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                let artifact = String::from_utf8(bytes)
                    .map_err(|error| CertificationSessionError::Serialization(error.to_string()))?;
                let response = CertificationResponse::ParametricFamilyExported {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    problem,
                    family,
                    artifact,
                };
                admit_response(&response, budget)?;
                Ok(response)
            }
            CertificationRequest::ReplayParametricFamily {
                problem,
                family,
                unknown_domain,
                parameter_domain,
                artifact,
                ..
            } => {
                if artifact.len() > MAX_SESSION_ROOT_ARTIFACT_BYTES {
                    return Err(CertificationSessionError::ArtifactLimit);
                }
                self.replay(
                    FamilyReplay {
                        problem,
                        family,
                        unknown_domain,
                        parameter_domain,
                    },
                    artifact.as_bytes(),
                    context_digest,
                    budget,
                    lease,
                )
            }
            CertificationRequest::RegisterParametricProblem {
                problem,
                residuals,
                unknowns,
                parameters,
                ..
            } => {
                if self.problems.contains_key(&problem) {
                    return Err(CertificationSessionError::DuplicateObject);
                }
                if self.problems.len() >= MAX_PROBLEMS {
                    return Err(CertificationSessionError::ObjectLimit);
                }
                if unknowns.is_empty()
                    || parameters.is_empty()
                    || unknowns.len() > MAX_AXES
                    || parameters.len() > MAX_AXES
                    || unknowns.len() + parameters.len() > MAX_AXES
                    || residuals.len() != unknowns.len()
                {
                    return Err(CertificationSessionError::InvalidDomain);
                }
                charge(budget, 1, 4096)?;
                let sources: Vec<_> = residuals.iter().map(String::as_str).collect();
                let unknown_names: Vec<_> = unknowns.iter().map(String::as_str).collect();
                let parameter_names: Vec<_> = parameters.iter().map(String::as_str).collect();
                let parsed = ParametricProblem::parse_sources(
                    &sources,
                    &unknown_names,
                    &parameter_names,
                    budget,
                )?;
                charge(
                    budget,
                    unknowns.len() + parameters.len(),
                    unknowns
                        .iter()
                        .chain(&parameters)
                        .map(|name| name.len() + 128)
                        .sum(),
                )?;
                let response = CertificationResponse::ParametricProblemRegistered {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    problem: problem.clone(),
                    residuals: parsed.residuals().len(),
                    unknowns: parsed.unknowns().to_vec(),
                    parameters: parsed.parameters().to_vec(),
                    context_digest,
                };
                admit_response(&response, budget)?;
                self.problems.insert(problem, parsed);
                Ok(response)
            }
            CertificationRequest::CertifyParametricRoot {
                problem,
                family,
                witness,
                ..
            } => {
                self.admit_family(&family, budget)?;
                let source = self
                    .problems
                    .get(&problem)
                    .ok_or(CertificationSessionError::UnknownParametricProblem)?;
                let witness = parse_uniform_witness(
                    witness,
                    source.unknowns().len(),
                    source.parameters().len(),
                    budget,
                )?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = source.verify_root_box(
                    &witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_family(
                    problem,
                    family,
                    ParametricFamilyReceipt::RootBox(receipt),
                    context_digest,
                    budget,
                )
            }
            CertificationRequest::CertifyParametricCover {
                problem,
                family,
                unknown_domain,
                parameter_domain,
                precision_digits,
                max_leaves,
                ..
            } => {
                self.admit_family(&family, budget)?;
                let source = self
                    .problems
                    .get(&problem)
                    .ok_or(CertificationSessionError::UnknownParametricProblem)?;
                let unknown_domain =
                    parse_domain(&unknown_domain, source.unknowns().len(), budget)?;
                let parameter_domain =
                    parse_domain(&parameter_domain, source.parameters().len(), budget)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let witness = source.propose_cover(
                    &unknown_domain,
                    &parameter_domain,
                    ParametricRootCoverOptions {
                        precision_digits,
                        max_leaves,
                    },
                    budget,
                )?;
                let receipt = source.verify_cover(
                    &unknown_domain,
                    &parameter_domain,
                    &witness,
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                self.publish_family(
                    problem,
                    family,
                    ParametricFamilyReceipt::Cover(receipt),
                    context_digest,
                    budget,
                )
            }
            CertificationRequest::EncloseParametricSensitivity {
                family,
                sensitivity,
                ..
            } => {
                if self.sensitivities.contains_key(&sensitivity) {
                    return Err(CertificationSessionError::DuplicateObject);
                }
                if self.sensitivities.len() >= MAX_SENSITIVITIES {
                    return Err(CertificationSessionError::ObjectLimit);
                }
                let source = self
                    .families
                    .get(&family)
                    .ok_or(CertificationSessionError::UnknownParametricFamily)?;
                if source.context_digest != context_digest {
                    return Err(CertificationError::UniverseMismatch {
                        found: source.context_digest,
                        expected: context_digest,
                    }
                    .into());
                }
                charge(budget, 1, 4096)?;
                let lease =
                    lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
                let receipt = source.receipt.enclose_sensitivity(
                    &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
                )?;
                let response = CertificationResponse::ParametricSensitivityEnclosed {
                    schema_version: CERTIFICATION_REQUEST_SCHEMA,
                    family,
                    sensitivity: sensitivity.clone(),
                    evidence: "certified_numeric",
                    claim: "implicit_solution_jacobian_enclosure",
                    solution_jacobian: render_hessian(receipt.solution_jacobian(), budget)?,
                    leaf_count: receipt.leaf_count(),
                    precision_digits: receipt.precision_digits(),
                    context_digest,
                };
                admit_response(&response, budget)?;
                self.sensitivities.insert(sensitivity, receipt);
                Ok(response)
            }
            _ => Err(CertificationSessionError::Serialization(
                "request is not a parameter-family operation".into(),
            )),
        }
    }
}
