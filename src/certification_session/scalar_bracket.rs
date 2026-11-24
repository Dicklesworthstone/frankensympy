//! Session ownership for independently checked scalar root-existence brackets.
//!
//! Existence handles cannot substitute for unique-root or complete-isolation
//! evidence. Generation uses ordinary resources, verification uses its protected
//! pool, and final output admission precedes the only accepted-store insertion.

use super::{
    CERTIFICATION_REQUEST_SCHEMA, CertificationResponse, CertificationSessionError,
    CertificationState, ResidualSystemId, admit_response, certification_error_code, charge,
    parse_scalar, render_domain, render_scalar, valid_label,
};
use crate::request_budget::RequestBudget;
use crate::scalar_bracket::{
    ScalarBracketError, export_scalar_root_bracket, replay_scalar_root_bracket,
};
use fsym_budget::{BudgetError, BudgetMeter, MeterError, VerifierLease};
use fsym_calculus::scalar_bracket::ScalarRootBracketOptions;
use fsym_core::{BigRational, RootBoxError, ScalarRootBracketError, ScalarRootBracketReceipt};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

/// Accepted scalar existence brackets have their own resident storage bound.
pub const MAX_SESSION_SCALAR_ROOT_BRACKETS: usize = 32;

/// A session-local existence certificate, distinct from a unique-root certificate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct ScalarRootBracketId(Box<str>);

impl ScalarRootBracketId {
    pub fn new(value: impl Into<String>) -> Result<Self, CertificationSessionError> {
        let value = value.into();
        if !valid_label(&value, "bracket:") {
            return Err(CertificationSessionError::InvalidLabel);
        }
        Ok(Self(value.into_boxed_str()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ScalarRootBracketId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

struct ResidentBracket {
    system: ResidualSystemId,
    context_digest: [u8; 32],
    receipt: ScalarRootBracketReceipt,
}

#[derive(Default)]
pub(super) struct ScalarBracketState {
    brackets: BTreeMap<ScalarRootBracketId, ResidentBracket>,
}

/// The receiver chooses the source and complete original interval. Neither an
/// imported witness nor a current workspace binding can replace them.
pub(super) struct ScalarBracketStatement {
    pub(super) system: ResidualSystemId,
    pub(super) bracket: ScalarRootBracketId,
    pub(super) lower: String,
    pub(super) upper: String,
    pub(super) absolute_tolerance: String,
}

impl ScalarBracketStatement {
    fn parse(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<(BigRational, BigRational, BigRational), CertificationSessionError> {
        Ok((
            parse_scalar(&self.lower, meter)?,
            parse_scalar(&self.upper, meter)?,
            parse_scalar(&self.absolute_tolerance, meter)?,
        ))
    }
}

pub(super) fn error_code(error: &ScalarBracketError) -> &'static str {
    match error {
        ScalarBracketError::Certification(error) => certification_error_code(error),
        ScalarBracketError::ProblemDimension => "invalid_input",
        ScalarBracketError::StatementMismatch => "statement_mismatch",
        ScalarBracketError::Bracket(ScalarRootBracketError::Enclosure(error)) => match error {
            RootBoxError::Meter(MeterError::Cancelled) => "cancelled",
            RootBoxError::Meter(MeterError::Budget(_)) => "budget_exhausted",
            RootBoxError::Domain(_) => "domain_not_certified",
            RootBoxError::Unsupported(_) => "unsupported_operation",
            RootBoxError::InvalidPrecision => "invalid_input",
            RootBoxError::ResourceLimit(_) => "resource_limit_exceeded",
            _ => "refused",
        },
        ScalarBracketError::Bracket(
            ScalarRootBracketError::InvalidTolerance | ScalarRootBracketError::InvalidBracket(_),
        ) => "invalid_input",
        ScalarBracketError::Bracket(ScalarRootBracketError::ResourceLimit(_)) => {
            "resource_limit_exceeded"
        }
        // Equal signs or unresolved rounding cannot prove that no root exists.
        ScalarBracketError::Bracket(
            ScalarRootBracketError::AccuracyNotReached
            | ScalarRootBracketError::NoSignChange
            | ScalarRootBracketError::IndeterminateSign,
        ) => "inconclusive",
    }
}

impl CertificationState {
    pub(crate) fn scalar_root_bracket(
        &self,
        id: &ScalarRootBracketId,
    ) -> Option<&ScalarRootBracketReceipt> {
        self.scalar_brackets
            .brackets
            .get(id)
            .map(|slot| &slot.receipt)
    }

    fn admit_scalar_bracket_id(
        &self,
        id: &ScalarRootBracketId,
    ) -> Result<(), CertificationSessionError> {
        if self.scalar_brackets.brackets.contains_key(id) {
            return Err(CertificationSessionError::DuplicateObject);
        }
        if self.scalar_brackets.brackets.len() >= MAX_SESSION_SCALAR_ROOT_BRACKETS {
            return Err(CertificationSessionError::ObjectLimit);
        }
        Ok(())
    }

    pub(super) fn certify_scalar_bracket(
        &mut self,
        statement: ScalarBracketStatement,
        options: ScalarRootBracketOptions,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_scalar_bracket_id(&statement.bracket)?;
        let problem = self
            .systems
            .get(&statement.system)
            .ok_or(CertificationSessionError::UnknownSystem)?;
        let (lower, upper, tolerance) = statement.parse(budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        // Validate pool ownership before generation, without transferring its
        // resources or spending them on candidate search.
        budget.verifier_meter(lease).map_err(MeterError::from)?;
        let witness =
            problem.propose_scalar_root_bracket(&lower, &upper, &tolerance, options, budget)?;
        let receipt = problem.verify_scalar_root_bracket(
            &lower,
            &upper,
            &tolerance,
            &witness,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?;
        self.publish_scalar_bracket(statement, receipt, context_digest, budget)
    }

    /// Check without publishing, so an import may borrow resident bytes until
    /// independent verification has completed.
    pub(super) fn verify_scalar_bracket_artifact(
        &self,
        statement: &ScalarBracketStatement,
        bytes: &[u8],
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
        lease: Option<&VerifierLease>,
    ) -> Result<ScalarRootBracketReceipt, CertificationSessionError> {
        self.admit_scalar_bracket_id(&statement.bracket)?;
        let problem = self
            .systems
            .get(&statement.system)
            .ok_or(CertificationSessionError::UnknownSystem)?;
        let (lower, upper, tolerance) = statement.parse(budget)?;
        let lease = lease.ok_or(MeterError::Budget(BudgetError::VerifierPoolAccessDenied))?;
        Ok(replay_scalar_root_bracket(
            bytes,
            problem,
            &lower,
            &upper,
            &tolerance,
            context_digest,
            &mut budget.verifier_meter(lease).map_err(MeterError::from)?,
        )?)
    }

    pub(super) fn publish_scalar_bracket(
        &mut self,
        statement: ScalarBracketStatement,
        receipt: ScalarRootBracketReceipt,
        context_digest: [u8; 32],
        budget: &mut RequestBudget<'_>,
    ) -> Result<CertificationResponse, CertificationSessionError> {
        self.admit_scalar_bracket_id(&statement.bracket)?;
        charge(budget, 1, 4096)?;
        let mut enclosure = render_domain(std::slice::from_ref(receipt.root_enclosure()), budget)?;
        let response = CertificationResponse::ScalarRootBracketCertified {
            schema_version: CERTIFICATION_REQUEST_SCHEMA,
            system: statement.system.clone(),
            bracket: statement.bracket.clone(),
            evidence: "certified_numeric",
            claim: "at_least_one_real_root_in_closed_interval",
            lower: render_scalar(receipt.lower(), budget)?,
            upper: render_scalar(receipt.upper(), budget)?,
            absolute_tolerance: render_scalar(receipt.absolute_tolerance(), budget)?,
            enclosure: enclosure.pop().expect("one root enclosure was rendered"),
            endpoint_values: render_domain(receipt.endpoint_values(), budget)?,
            precision_digits: receipt.witness().precision_digits,
            context_digest,
        };
        admit_response(&response, budget)?;
        self.scalar_brackets.brackets.insert(
            statement.bracket,
            ResidentBracket {
                system: statement.system,
                context_digest,
                receipt,
            },
        );
        Ok(response)
    }

    pub(super) fn export_scalar_bracket_bytes(
        &self,
        bracket: &ScalarRootBracketId,
        meter: &mut impl BudgetMeter,
    ) -> Result<(ResidualSystemId, Vec<u8>), CertificationSessionError> {
        let slot = self
            .scalar_brackets
            .brackets
            .get(bracket)
            .ok_or(CertificationSessionError::UnknownScalarRootBracket)?;
        let problem = self
            .systems
            .get(&slot.system)
            .ok_or(CertificationSessionError::UnknownSystem)?;
        let bytes = export_scalar_root_bracket(problem, &slot.receipt, slot.context_digest, meter)?;
        charge(meter, 1, 128)?;
        Ok((slot.system.clone(), bytes))
    }
}
