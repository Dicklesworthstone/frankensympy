//! Source-bound real residual enclosures, complete root isolation and portable
//! root-box witnesses.
//!
//! The public problem retains typed sources and ordered symbol identities.
//! Numerical acceptance always recomputes the original source with the core
//! interval checker. No symbolic derivative, compiled evaluator, or serialized
//! acceptance flag supplies mathematical evidence.

#![forbid(unsafe_code)]

use fsym_budget::BudgetMeter;
use fsym_calculus::root_isolation::{
    RootIsolationOptions, RootIsolationSearch, propose_root_isolation,
};
use fsym_core::root_certificate::artifact::{self, RootCertificateSources};
use fsym_core::{
    Expr, RealBall, ResidualJacobianEnclosure, RootBoxReceipt, RootBoxWitness,
    RootIsolationReceipt, RootIsolationWitness, Symbol, metered_enclose_residual_jacobian,
    parse_raw, verify_root_box, verify_root_isolation,
};
use std::mem::size_of;

// One canonical wire implementation serves the Session and host bridges. These
// reexports preserve the existing native API, bytes, refusal codes and metering.
pub use artifact::{
    CertificationError, MAX_CERTIFICATION_NAME_BYTES, MAX_ROOT_CERTIFICATE_BYTES,
    MAX_ROOT_CERTIFICATE_JSON_DEPTH, ROOT_CERTIFICATE_SCHEMA_VERSION, export_root_certificate,
};
pub(crate) use artifact::{
    JsonNesting, charge, charge_witness_encoding, check_dimensions, check_problem_sources, encode,
    encoding_error,
};

/// Maximum UTF-8 bytes in one textual residual passed to the convenience parser.
pub const MAX_CERTIFICATION_SOURCE_BYTES: usize = 4 * 1024;

/// Immutable typed source for a 1..64 by 1..16 real residual/Jacobian problem.
///
/// Construction checks the source envelope; it does not prove the source is
/// defined or C1 on any domain. Those obligations are checked independently
/// by [`Self::enclose`] and [`Self::verify_root`]. There is no unchecked decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualProblem {
    sources: RootCertificateSources,
}

impl ResidualProblem {
    /// Take ownership of exact sources and their explicit coordinate order.
    /// No workspace substitution, simplification, or symbol-name rebinding occurs.
    pub fn new(
        residuals: Vec<Expr>,
        variables: Vec<Symbol>,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, CertificationError> {
        Ok(Self {
            sources: RootCertificateSources::new(residuals, variables, meter)?,
        })
    }

    /// Parse bounded exact source expressions without simplifying away domains.
    ///
    /// The parser retains ordered Add/Mul/Pow operands, so `0*(1/x)` and
    /// `log(x)-log(x)` still require their original domains. Names construct
    /// plain symbols; identity-bearing symbols use [`Self::new`] instead.
    /// Each at-most-4-KiB parse is a conservatively precharged synchronous work
    /// unit with checkpoints before/after it; interval checking has its own
    /// per-node and arithmetic safe points. No printed string becomes identity.
    pub fn parse_sources(
        residuals: &[String],
        variables: &[String],
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, CertificationError> {
        meter.checkpoint()?;
        check_dimensions(residuals.len(), variables.len())?;
        charge(
            meter,
            1,
            (variables.len() * size_of::<Symbol>() + residuals.len() * size_of::<Expr>()) as u64,
            2,
        )?;
        let mut symbols = Vec::new();
        symbols
            .try_reserve_exact(variables.len())
            .map_err(|_| CertificationError::AllocationFailure)?;
        for name in variables {
            if name.len() > MAX_CERTIFICATION_NAME_BYTES {
                return Err(CertificationError::InvalidVariable);
            }
            charge(meter, name.len() as u64 + 1, name.len() as u64, 1)?;
            let mut chars = name.chars();
            let identifier = chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
                && chars.all(|c| c.is_alphanumeric() || c == '_');
            if !identifier
                || matches!(
                    name.as_str(),
                    "pi" | "Pi" | "E" | "I" | "oo" | "zoo" | "nan" | "NaN"
                )
            {
                return Err(CertificationError::InvalidVariable);
            }
            symbols.push(Symbol::new(name.clone()));
        }
        let mut expressions = Vec::new();
        expressions
            .try_reserve_exact(residuals.len())
            .map_err(|_| CertificationError::AllocationFailure)?;
        for (index, source) in residuals.iter().enumerate() {
            if source.len() > MAX_CERTIFICATION_SOURCE_BYTES {
                return Err(CertificationError::SourceLimit("textual residual bytes"));
            }
            // Raw construction does not evaluate powers. With <=4096 total
            // decimal digits its only numeric work (literal division/decimal
            // normalization) stays far below the core's 1024-limb envelope.
            let length = source.len() as u64;
            let envelope = length
                .saturating_add(1)
                .saturating_mul(length.saturating_add(1));
            charge(
                meter,
                envelope.saturating_mul(4),
                envelope.saturating_mul(2) + 1024,
                length * 8 + 1,
            )?;
            let expression = parse_raw(source).map_err(|_| CertificationError::Parse { index })?;
            meter.checkpoint()?;
            expressions.push(expression);
        }
        Self::new(expressions, symbols, meter)
    }

    pub(crate) fn sources(&self) -> &RootCertificateSources {
        &self.sources
    }

    pub fn residuals(&self) -> &[Expr] {
        self.sources.residuals()
    }

    pub fn variables(&self) -> &[Symbol] {
        self.sources.variables()
    }

    /// Recompute whole-box C1 residual and Jacobian bounds, including every
    /// original domain obligation. An enclosure alone does not establish a root.
    pub fn enclose(
        &self,
        domain: &[RealBall],
        precision: u32,
        meter: &mut impl BudgetMeter,
    ) -> Result<ResidualJacobianEnclosure, CertificationError> {
        metered_enclose_residual_jacobian(
            self.sources.residuals(),
            self.sources.variables(),
            domain,
            precision,
            meter,
        )
        .map_err(Into::into)
    }

    /// Verify a unique nonsingular root in one proposed box of a square system.
    /// The rational preconditioner is an untrusted witness, not an accepted inverse.
    pub fn verify_root(
        &self,
        witness: &RootBoxWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootBoxReceipt, CertificationError> {
        verify_root_box(
            self.sources.residuals(),
            self.sources.variables(),
            witness,
            meter,
        )
        .map_err(Into::into)
    }

    /// Start a bounded, resumable candidate search over this immutable source.
    /// Advances retain their frontier and charge the caller's cumulative meter;
    /// completion still requires independent [`Self::verify_isolation`].
    pub fn begin_isolation_search(
        &self,
        domain: &[RealBall],
        options: RootIsolationOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationSearch, CertificationError> {
        RootIsolationSearch::new(
            self.sources.residuals(),
            self.sources.variables(),
            domain,
            options,
            meter,
        )
        .map_err(Into::into)
    }

    /// Propose a complete cover of a finite box without initial guesses or a
    /// supplied inverse. Exact interval bounds and bounded subdivision guide
    /// this search; the returned witness grants no completeness claim.
    ///
    /// Singular roots, roots on the outer boundary, unresolved source domains
    /// and exhausted search limits refuse without returning a partial root set.
    /// Candidate generation belongs to the caller's ordinary work budget.
    pub fn propose_isolation(
        &self,
        domain: &[RealBall],
        options: RootIsolationOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationWitness, CertificationError> {
        propose_root_isolation(
            self.sources.residuals(),
            self.sources.variables(),
            domain,
            options,
            meter,
        )
        .map_err(Into::into)
    }

    /// Independently check every original-source exclusion and local root
    /// witness, complete box coverage and strict separation of distinct roots.
    /// A zero-root receipt requires a complete exclusion cover.
    pub fn verify_isolation(
        &self,
        domain: &[RealBall],
        witness: &RootIsolationWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationReceipt, CertificationError> {
        verify_root_isolation(
            self.sources.residuals(),
            self.sources.variables(),
            domain,
            witness,
            meter,
        )
        .map_err(Into::into)
    }

    /// Find and independently certify the complete real root set in a finite
    /// box using separate proposal and verification meters. This lane accepts
    /// only nonsingular roots; singular or unresolved regions refuse.
    ///
    /// To share one cumulative [`fsym_budget::Budget`], call
    /// [`Self::propose_isolation`] first, then [`Self::verify_isolation`] through
    /// that budget's protected verifier meter. Neither phase resets a ledger.
    pub fn isolate_roots(
        &self,
        domain: &[RealBall],
        options: RootIsolationOptions,
        proposal_meter: &mut impl BudgetMeter,
        verification_meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationReceipt, CertificationError> {
        let witness = self.propose_isolation(domain, options, proposal_meter)?;
        self.verify_isolation(domain, &witness, verification_meter)
    }
}

/// Recheck the canonical portable witness against the receiver's frozen source.
/// Source, context, wire, resource and independent proof refusals are unchanged.
pub fn replay_root_certificate(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<RootBoxReceipt, CertificationError> {
    artifact::replay_root_certificate(bytes, &expected_problem.sources, expected_context, meter)
}
