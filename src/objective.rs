//! Source-bound objective enclosures and stationary-point certification.
//!
//! The original expression and ordered typed coordinates remain immutable.
//! Second-order interval automatic differentiation recomputes the value,
//! gradient and Hessian directly from that source; symbolic differentiation
//! and numerical optimization are outside the acceptance boundary.

use crate::certification::{
    CertificationError, MAX_CERTIFICATION_NAME_BYTES, MAX_CERTIFICATION_SOURCE_BYTES,
    ResidualProblem, charge, parse_source_expression, parse_source_variable,
};
use fsym_budget::BudgetMeter;
use fsym_calculus::sparse_hessian::MAX_OBJECTIVE_VARIABLES;
use fsym_core::root_certificate::artifact::check_source_expression;
use fsym_core::{
    Expr, MAX_ROOT_BOX_VARIABLES, ObjectiveEnclosure, RealBall, RootBoxError, RootBoxWitness,
    StationaryPointReceipt, Symbol, metered_enclose_objective, verify_stationary_point,
};

/// One scalar objective on 1..=64 ordered real coordinates.
///
/// Registration admits the bounded exact source only. It neither asserts
/// smoothness on a box nor grants a stationary-point or extremum claim.
/// Held zero factors, cancellation and powers retain their original domains.
/// Compiled sparse optimization admits up to 64 coordinates; the independent
/// exact enclosure, stationary and portable-certificate lanes retain their
/// separate 16-coordinate limits. Compilation has additional source bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveProblem {
    source: ObjectiveSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ObjectiveSource {
    // Preserve the complete original <=16-coordinate admission policy,
    // including exact sources too large for numerical derivative compilation.
    Certificate(ResidualProblem),
    // A separate source envelope must not widen any root-certificate checker.
    Wide {
        objective: Expr,
        variables: Vec<Symbol>,
    },
}

impl ObjectiveProblem {
    /// Preserve exact symbol identities, including same-name distinct axes.
    pub fn new(
        objective: Expr,
        variables: Vec<Symbol>,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, CertificationError> {
        charge(meter, 1, std::mem::size_of::<Expr>() as u64, 1)?;
        if variables.len() <= MAX_ROOT_BOX_VARIABLES {
            return Ok(Self {
                source: ObjectiveSource::Certificate(ResidualProblem::new(
                    vec![objective],
                    variables,
                    meter,
                )?),
            });
        }
        if variables.len() > MAX_OBJECTIVE_VARIABLES {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        for (index, variable) in variables.iter().enumerate() {
            if variable.name.len() > MAX_CERTIFICATION_NAME_BYTES {
                return Err(RootBoxError::ResourceLimit("symbol length").into());
            }
            charge(
                meter,
                ((index + 1) * (variable.name.len() + 1)) as u64,
                128 + variable.name.len() as u64,
                1,
            )?;
            if variables[..index].contains(variable) {
                return Err(RootBoxError::DuplicateVariable.into());
            }
        }
        check_source_expression(&objective, meter)?;
        meter.checkpoint()?;
        Ok(Self {
            source: ObjectiveSource::Wide {
                objective,
                variables,
            },
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
        if variables.len() <= MAX_ROOT_BOX_VARIABLES {
            return Ok(Self {
                source: ObjectiveSource::Certificate(ResidualProblem::parse_sources(
                    &[source.to_owned()],
                    variables,
                    meter,
                )?),
            });
        }
        if variables.len() > MAX_OBJECTIVE_VARIABLES {
            return Err(RootBoxError::DimensionMismatch.into());
        }
        charge(
            meter,
            1,
            (variables.len() * std::mem::size_of::<Symbol>()) as u64,
            1,
        )?;
        let mut symbols = Vec::new();
        symbols
            .try_reserve_exact(variables.len())
            .map_err(|_| CertificationError::AllocationFailure)?;
        for name in variables {
            symbols.push(parse_source_variable(name, meter)?);
        }
        let objective = parse_source_expression(source, 0, meter)?;
        Self::new(objective, symbols, meter)
    }

    pub fn objective(&self) -> &Expr {
        match &self.source {
            ObjectiveSource::Certificate(source) => &source.residuals()[0],
            ObjectiveSource::Wide { objective, .. } => objective,
        }
    }

    pub fn variables(&self) -> &[Symbol] {
        match &self.source {
            ObjectiveSource::Certificate(source) => source.variables(),
            ObjectiveSource::Wide { variables, .. } => variables,
        }
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
