//! Source-bound objective enclosures and stationary-point certification.
//!
//! The original expression and ordered typed coordinates remain immutable.
//! Second-order interval automatic differentiation recomputes the value,
//! gradient and Hessian directly from that source; symbolic differentiation
//! and numerical optimization are outside the acceptance boundary.

use crate::certification::{
    CertificationError, MAX_CERTIFICATION_SOURCE_BYTES, ResidualProblem, charge,
};
use fsym_budget::BudgetMeter;
use fsym_core::{
    Expr, ObjectiveEnclosure, RealBall, RootBoxWitness, StationaryPointReceipt, Symbol,
    metered_enclose_objective, verify_stationary_point,
};

/// One scalar objective on 1..=16 ordered real coordinates.
///
/// Registration admits the bounded exact source only. It neither asserts
/// smoothness on a box nor grants a stationary-point or extremum claim.
/// Held zero factors, cancellation and powers retain their original domains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveProblem {
    // Reuse the immutable source envelope and parser rather than creating a
    // second identity, domain, or admission policy for scalar objectives.
    source: ResidualProblem,
}

impl ObjectiveProblem {
    /// Preserve exact symbol identities, including same-name distinct axes.
    pub fn new(
        objective: Expr,
        variables: Vec<Symbol>,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, CertificationError> {
        charge(meter, 1, std::mem::size_of::<Expr>() as u64, 1)?;
        Ok(Self {
            source: ResidualProblem::new(vec![objective], variables, meter)?,
        })
    }

    /// Parse a bounded expression without simplifying away source domains.
    /// Text creates plain symbol identities; use [`Self::new`] for typed axes.
    pub fn parse_source(
        source: &str,
        variables: &[String],
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, CertificationError> {
        meter.checkpoint()?;
        if source.len() > MAX_CERTIFICATION_SOURCE_BYTES {
            return Err(CertificationError::SourceLimit("textual objective bytes"));
        }
        charge(meter, 1, source.len() as u64, 1)?;
        Ok(Self {
            source: ResidualProblem::parse_sources(&[source.to_owned()], variables, meter)?,
        })
    }

    pub fn objective(&self) -> &Expr {
        &self.source.residuals()[0]
    }

    pub fn variables(&self) -> &[Symbol] {
        self.source.variables()
    }

    /// Enclose the original value, gradient and Hessian over the entire box.
    /// Every admitted source must be C2 on a neighborhood of that box.
    /// An enclosure establishes neither stationarity nor convexity.
    pub fn enclose(
        &self,
        domain: &[RealBall],
        precision_digits: u32,
        meter: &mut impl BudgetMeter,
    ) -> Result<ObjectiveEnclosure, CertificationError> {
        metered_enclose_objective(
            self.objective(),
            self.variables(),
            domain,
            precision_digits,
            meter,
        )
        .map_err(Into::into)
    }

    /// Independently check existence and uniqueness of a nonsingular stationary
    /// point in the proposed box. The rational preconditioner is untrusted.
    /// The optional classification is local; no global-optimality claim follows.
    /// All refusals retain consumed work and return no partial receipt.
    pub fn verify_stationary(
        &self,
        witness: &RootBoxWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<StationaryPointReceipt, CertificationError> {
        verify_stationary_point(self.objective(), self.variables(), witness, meter)
            .map_err(Into::into)
    }
}
