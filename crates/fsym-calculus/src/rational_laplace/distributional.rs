//! Checked causal distributional inverses of bounded rational transforms.
//!
//! Under the unilateral convention with lower limit 0-, the transform of
//! delta^(k)(t) is s^k. Polynomial division proposes the impulse coefficients;
//! acceptance independently checks P(s) * B(s) + R(s) = A(s), with R/B
//! reconstructed from the regular inverse's closed transform formulas.
//! The verifier never divides the source numerator to generate its polynomial
//! part or calls the residue generator. Exact denominator quotients are checked
//! when reconstructing the regular component.
//!
//! Impulses and the regular function remain separate: only the regular part
//! receives a Heaviside factor. Multiplying a delta at the origin by a step
//! would change or leave undefined the distribution being returned. Original
//! source exclusions and independent generator/verifier meters are retained.

use super::{
    Capped, Fraction, MAX_BITS, MAX_DEGREE, MAX_TEXT, RationalLaplaceCertificate,
    RationalLaplaceError, Result, Work, admit, bit_size, cauchy_bound, generate_regular_fraction,
    lifted_function, maximum_bound, parse_source, product_expression,
    reconstruct_regular_numerator, scalar_ok, sum_expression,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::Zero;

/// Untrusted complete distributional inverse proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributionalLaplaceCertificate {
    /// Ascending coefficients of P(s), or equivalently of delta^(k)(t).
    /// Canonical zero is the single coefficient zero.
    pub impulse_coefficients: Vec<BigRational>,
    pub regular: RationalLaplaceCertificate,
}

/// A complete causal distributional inverse on a sufficient right half-plane.
/// This receipt is checked in process, not a portable proof-kernel object.
#[derive(Debug, Clone)]
pub struct VerifiedDistributionalLaplace {
    source: Expr,
    transform_variable: Symbol,
    time_variable: Symbol,
    certificate: DistributionalLaplaceCertificate,
    regular_function: Expr,
    impulses: Expr,
    sufficient_abscissa: BigRational,
}

impl VerifiedDistributionalLaplace {
    pub fn source(&self) -> &Expr {
        &self.source
    }
    pub fn transform_variable(&self) -> &Symbol {
        &self.transform_variable
    }
    pub fn time_variable(&self) -> &Symbol {
        &self.time_variable
    }
    pub fn certificate(&self) -> &DistributionalLaplaceCertificate {
        &self.certificate
    }
    /// Ordinary function before causal multiplication by Heaviside(t).
    pub fn regular_function(&self) -> &Expr {
        &self.regular_function
    }
    /// Delta derivatives, which must not acquire a Heaviside multiplier.
    pub fn impulses(&self) -> &Expr {
        &self.impulses
    }
    pub fn sufficient_abscissa(&self) -> &BigRational {
        &self.sufficient_abscissa
    }
    pub fn into_parts(self) -> (Expr, Expr) {
        (self.regular_function, self.impulses)
    }
}

fn validate_impulses(coefficients: &[BigRational]) -> Result<()> {
    if coefficients.is_empty()
        || coefficients.len() > MAX_DEGREE + 1
        || (coefficients.len() > 1 && coefficients.last().is_some_and(Zero::is_zero))
    {
        return Err(RationalLaplaceError::Verification(
            "noncanonical or oversized impulse polynomial",
        ));
    }
    for coefficient in coefficients {
        scalar_ok(coefficient, MAX_BITS)?;
    }
    Ok(())
}

fn lifted_impulses(
    coefficients: &[BigRational],
    time: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Expr> {
    validate_impulses(coefficients)?;
    let bits = coefficients.iter().map(bit_size).max().unwrap_or(1);
    let nodes = 1 + 5 * coefficients.len();
    work.charge(
        nodes,
        nodes * (256 + 4 * bits.div_ceil(8) as usize + 4 * time.name.len()),
    )?;
    let mut terms = Vec::with_capacity(coefficients.len());
    for (order, coefficient) in coefficients.iter().enumerate() {
        work.meter.checkpoint()?;
        if coefficient.is_zero() {
            continue;
        }
        let mut arguments = vec![Expr::Sym(time.clone())];
        if order != 0 {
            arguments.push(Expr::from_i64(order as i64));
        }
        terms.push(product_expression(vec![
            rational_expr(work.number(coefficient)?),
            Expr::Function("DiracDelta".into(), arguments),
        ]));
    }
    work.meter.checkpoint()?;
    Ok(sum_expression(terms))
}

fn generate_distributional(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<DistributionalLaplaceCertificate> {
    admit(source, variable, time, work)?;
    let original = parse_source(source, work)?;
    let (impulse_coefficients, remainder) =
        work.div_rem(&original.numerator, &original.denominator)?;
    let proper = Fraction {
        numerator: remainder,
        denominator: original.denominator,
        excluded_abscissa: original.excluded_abscissa,
    };
    let regular = generate_regular_fraction(&proper, variable, work)?;
    work.meter.checkpoint()?;
    Ok(DistributionalLaplaceCertificate {
        impulse_coefficients,
        regular,
    })
}

fn verify_distributional(
    source: &Expr,
    variable: &Symbol,
    time: &Symbol,
    certificate: &DistributionalLaplaceCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedDistributionalLaplace> {
    let nodes = admit(source, variable, time, work)?;
    let original = parse_source(source, work)?;
    validate_impulses(&certificate.impulse_coefficients)?;
    let (regular_numerator, regular_cells) =
        reconstruct_regular_numerator(&certificate.regular, &original.denominator, work)?;
    // This convolution identity is separate from the generator's long division.
    // It binds the singular and regular components to the complete original.
    let polynomial_numerator =
        work.mul(&certificate.impulse_coefficients, &original.denominator)?;
    let reconstructed = work.add(&polynomial_numerator, &regular_numerator)?;
    if reconstructed != original.numerator {
        return Err(RationalLaplaceError::Verification(
            "complete distributional transform identity",
        ));
    }
    let denominator_bound = cauchy_bound(&original.denominator, work)?;
    let sufficient_abscissa =
        maximum_bound(Some(denominator_bound), original.excluded_abscissa, work)?
            .expect("denominator bound is always present");
    let regular_function = lifted_function(&certificate.regular, time, work)?;
    let impulses = lifted_impulses(&certificate.impulse_coefficients, time, work)?;
    let cells = regular_cells + certificate.impulse_coefficients.len();
    // Cover immutable source, symbol and certificate copies before publication.
    work.charge(
        nodes + cells,
        nodes * 2048 + 4 * MAX_TEXT + (cells + certificate.regular.blocks.len() * 2) * 16_512,
    )?;
    let receipt = VerifiedDistributionalLaplace {
        source: source.clone(),
        transform_variable: variable.clone(),
        time_variable: time.clone(),
        certificate: certificate.clone(),
        regular_function,
        impulses,
        sufficient_abscissa,
    };
    work.meter.checkpoint()?;
    Ok(receipt)
}

pub fn verify_distributional_laplace_certificate(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    certificate: &DistributionalLaplaceCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedDistributionalLaplace> {
    let mut capped = Capped {
        inner: verifier,
        used: [0; 3],
        stopped: None,
    };
    let result = verify_distributional(
        source,
        transform_variable,
        time_variable,
        certificate,
        &mut Work { meter: &mut capped },
    );
    if let Some(error) = capped.stopped {
        return Err(RationalLaplaceError::Meter(error));
    }
    result
}

pub fn inverse_distributional_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
) -> Result<VerifiedDistributionalLaplace> {
    metered_inverse_distributional_laplace(
        source,
        transform_variable,
        time_variable,
        &mut Unbounded,
        &mut Unbounded,
    )
}

/// Internal untrusted candidate entry for compositional delayed generators.
/// Acceptance still requires the complete independent transform verifier.
pub(crate) fn generate_distributional_laplace_certificate(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    generator: &mut impl BudgetMeter,
) -> Result<DistributionalLaplaceCertificate> {
    let mut meter = Capped {
        inner: generator,
        used: [0; 3],
        stopped: None,
    };
    let result = generate_distributional(
        source,
        transform_variable,
        time_variable,
        &mut Work { meter: &mut meter },
    );
    if let Some(error) = meter.stopped {
        return Err(RationalLaplaceError::Meter(error));
    }
    result
}

pub fn metered_inverse_distributional_laplace(
    source: &Expr,
    transform_variable: &Symbol,
    time_variable: &Symbol,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedDistributionalLaplace> {
    let mut generation = Capped {
        inner: generator,
        used: [0; 3],
        stopped: None,
    };
    let mut verification = Capped {
        inner: verifier,
        used: [0; 3],
        stopped: None,
    };
    let candidate = generate_distributional(
        source,
        transform_variable,
        time_variable,
        &mut Work {
            meter: &mut generation,
        },
    );
    if let Some(error) = generation.stopped {
        return Err(RationalLaplaceError::Meter(error));
    }
    let candidate = candidate?;
    generation.checkpoint()?;
    let result = verify_distributional(
        source,
        transform_variable,
        time_variable,
        &candidate,
        &mut Work {
            meter: &mut verification,
        },
    )?;
    generation.checkpoint()?;
    verification.checkpoint()?;
    Ok(result)
}
