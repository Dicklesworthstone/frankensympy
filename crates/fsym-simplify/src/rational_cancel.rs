//! Bounded rational cancellation for native expression consumers.
//!
//! The complete original arithmetic source reaches the checked polynomial
//! operation before any expansion, simplification or zero shortcut. Ordinary
//! expressions are returned here; callers needing the original excluded poles
//! must retain `fsym_polys::VerifiedRationalCancellation` instead. In particular
//! this convenience operation is not a globally valid equality certificate.

use crate::SimplifyError;
use fsym_budget::{BudgetError, BudgetMeter, Dimension, MeterError};
use fsym_core::{Expr, Symbol};
use fsym_polys::{PolyError, checked_rational_cancel};
use std::mem::size_of;
use std::sync::Arc;

const MAX_NODES: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_VARIABLES: usize = 8;
const MAX_NAME_BYTES: usize = 65_536;
const MAX_COMPUTE: u64 = 20_000_000;
const MAX_MEMORY: u64 = 128 * 1024 * 1024;

/// One cumulative phase ledger across recognition, every recursive component,
/// complete cancellation, lifting and the final ordinary-expression folds.
/// Recording the first meter failure preserves its typed cause across the
/// polynomial crate's diagnostic error boundary.
struct Phase<'a, M> {
    meter: &'a mut M,
    compute: u64,
    memory: u64,
    failure: Option<MeterError>,
}

impl<'a, M: BudgetMeter> Phase<'a, M> {
    fn new(meter: &'a mut M) -> Self {
        Self {
            meter,
            compute: 0,
            memory: 0,
            failure: None,
        }
    }

    fn record(&mut self, result: Result<(), MeterError>) -> Result<(), MeterError> {
        if let Err(error) = result {
            self.failure.get_or_insert(error);
        }
        result
    }

    fn tick(&mut self, steps: usize) -> Result<(), SimplifyError> {
        self.checkpoint().map_err(SimplifyError::from_meter)?;
        self.charge(Dimension::ComputeSteps, steps.max(1) as u64)
            .map_err(SimplifyError::from_meter)
    }

    fn allocate(&mut self, bytes: usize) -> Result<(), SimplifyError> {
        self.checkpoint().map_err(SimplifyError::from_meter)?;
        self.charge_batch(&[
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(SimplifyError::from_meter)
    }
}

impl<M: BudgetMeter> BudgetMeter for Phase<'_, M> {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = self.meter.checkpoint();
        self.record(result)
    }

    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let mut compute = self.compute;
        let mut memory = self.memory;
        for &(dimension, amount) in charges {
            let (used, limit) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_COMPUTE),
                Dimension::MemoryBytes => (&mut memory, MAX_MEMORY),
                _ => continue,
            };
            let remaining = limit.saturating_sub(*used);
            if amount > remaining {
                return self.record(Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                })));
            }
            *used += amount;
        }
        let result = self.meter.charge_batch(charges);
        self.record(result)?;
        self.compute = compute;
        self.memory = memory;
        Ok(())
    }
}

fn refused(message: &'static str) -> SimplifyError {
    SimplifyError::General(message.into())
}

/// Borrow the entire source before cloning a scalar, symbol or child. A
/// non-rational node only changes routing: it never skips admission of its
/// siblings or children. The same printed name with different typed identities
/// remains a different polynomial axis.
fn rational_axes(
    source: &Expr,
    work: &mut Phase<'_, impl BudgetMeter>,
) -> Result<Option<Vec<Symbol>>, SimplifyError> {
    work.allocate(128 + (MAX_VARIABLES + 1) * size_of::<&Symbol>())?;
    let mut pending = vec![(source, 0usize)];
    let mut symbols: Vec<&Symbol> = Vec::with_capacity(MAX_VARIABLES + 1);
    let mut rational = true;
    let mut excessive_power = false;
    let mut visited = 0usize;
    while let Some((node, depth)) = pending.pop() {
        work.tick(1)?;
        visited += 1;
        if visited > MAX_NODES || depth > MAX_DEPTH {
            return Err(refused(
                "cancellation source exceeds 4096 nodes or depth 64",
            ));
        }
        match node {
            Expr::Sym(symbol) => {
                if symbol.name.len() > MAX_NAME_BYTES {
                    return Err(refused("cancellation symbol name exceeds 65536 bytes"));
                }
                work.tick((symbols.len() + 1) * (symbol.name.len() + 1))?;
                if symbols.len() <= MAX_VARIABLES && !symbols.contains(&symbol) {
                    symbols.push(symbol);
                }
            }
            Expr::Integer(value) => {
                if value.bits() > 4096 {
                    return Err(refused("cancellation source integer exceeds 4096 bits"));
                }
            }
            Expr::Rational(value) => {
                if value.numer().bits().max(value.denom().bits()) > 4096 {
                    return Err(refused("cancellation source coefficient exceeds 4096 bits"));
                }
            }
            Expr::Const(_) => rational = false,
            Expr::Pow(base, exponent) => {
                if let Expr::Integer(value) = exponent.as_ref() {
                    excessive_power |= !value
                        .to_i64()
                        .is_some_and(|power| (-256..=256).contains(&power));
                } else {
                    rational = false;
                }
                if visited + pending.len() + 2 > MAX_NODES {
                    return Err(refused("cancellation source exceeds 4096 nodes"));
                }
                work.allocate(256)?;
                pending.push((exponent, depth + 1));
                pending.push((base, depth + 1));
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if let Expr::Function(name, _) = node {
                    rational = false;
                    if name.len() > MAX_NAME_BYTES {
                        return Err(refused("cancellation function name exceeds 65536 bytes"));
                    }
                    work.tick(name.len() + 1)?;
                }
                if children.len() > MAX_NODES.saturating_sub(visited + pending.len()) {
                    return Err(refused("cancellation source exceeds 4096 nodes"));
                }
                work.allocate(children.len() * 128)?;
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
        }
    }
    if !rational {
        return Ok(None);
    }
    if excessive_power {
        return Err(refused(
            "rational cancellation requires integer powers between -256 and 256",
        ));
    }
    if symbols.len() > MAX_VARIABLES {
        return Err(refused(
            "rational cancellation supports at most eight typed variables",
        ));
    }
    let name_bytes: usize = symbols.iter().map(|symbol| symbol.name.len()).sum();
    if name_bytes > MAX_NAME_BYTES {
        return Err(refused("cancellation generator names exceed 65536 bytes"));
    }
    work.tick((symbols.len() + 1).pow(2) * (name_bytes + 1))?;
    symbols.sort();
    work.allocate(symbols.len() * size_of::<Symbol>() + name_bytes)?;
    Ok(Some(symbols.into_iter().cloned().collect()))
}

fn polynomial_error(
    error: PolyError,
    generation: &Phase<'_, impl BudgetMeter>,
    verification: &Phase<'_, impl BudgetMeter>,
) -> SimplifyError {
    if let Some(cause) = generation.failure.or(verification.failure) {
        SimplifyError::from_meter(cause)
    } else if let PolyError::IdentityCheckFailed(message) = error {
        SimplifyError::ProofFailed(message)
    } else {
        SimplifyError::General(error.to_string())
    }
}

/// The accepted lift is already a collected polynomial pair: each monomial
/// occurs once and has at most one rational coefficient. The ordinary
/// simplifier only copies/orders those terms and folds coefficients against
/// zero or one. Prepay its bounded shallow-tree buffers, comparison/formatting
/// work and limb arithmetic; its existing node meter alone does not account
/// for these operations. This estimate is deliberately limited to the checked
/// polynomial lift, never used to authorize arbitrary input simplification.
fn prepay_polynomial_ordering(
    expression: &Expr,
    work: &mut Phase<'_, impl BudgetMeter>,
) -> Result<(), SimplifyError> {
    work.allocate(128)?;
    let mut pending = vec![expression];
    let mut nodes = 0usize;
    let mut bytes = 0usize;
    let mut scalar_work = 0usize;
    while let Some(node) = pending.pop() {
        work.tick(1)?;
        nodes += 1;
        if nodes > MAX_NODES {
            return Err(refused("cancelled expression exceeds 4096 output nodes"));
        }
        bytes += size_of::<Expr>() + 64;
        match node {
            Expr::Sym(symbol) => bytes += symbol.name.len(),
            Expr::Integer(value) => {
                let limbs = value.limb_count() as usize + 1;
                bytes += limbs * 8;
                scalar_work += limbs * limbs;
            }
            Expr::Rational(value) => {
                let limbs = (value.numer().limb_count() + value.denom().limb_count()) as usize + 1;
                bytes += limbs * 8;
                scalar_work += limbs * limbs;
            }
            Expr::Add(children) | Expr::Mul(children) => {
                work.allocate(children.len() * size_of::<&Expr>())?;
                pending.extend(children);
            }
            Expr::Pow(base, exponent) => {
                work.allocate(2 * size_of::<&Expr>())?;
                pending.push(base);
                pending.push(exponent);
            }
            _ => {
                return Err(refused(
                    "checked polynomial lift contains a non-rational node",
                ));
            }
        }
    }
    // The lift has fixed shallow depth and at most eight axes per monomial.
    // Sixty-four copies of its logical storage cover transient map/vector,
    // sort-key and scalar buffers; coefficient work has a separate limb bound.
    work.allocate(64 * bytes)?;
    work.tick(64 * bytes + 256 * scalar_work)
}

fn cancel_component(
    source: &Expr,
    generation: &mut Phase<'_, impl BudgetMeter>,
    verification: &mut Phase<'_, impl BudgetMeter>,
) -> Result<Expr, SimplifyError> {
    verification
        .checkpoint()
        .map_err(SimplifyError::from_meter)?;
    if let Some(variables) = rational_axes(source, generation)? {
        let accepted = checked_rational_cancel(source, &variables, generation, verification)
            .map_err(|error| polynomial_error(error, generation, verification))?;
        let expression = accepted
            .metered_expression(generation, verification)
            .map_err(|error| polynomial_error(error, generation, verification))?;
        // This value-only canonical ordering is metered on the same phase.
        // It does not strengthen the checked receipt's original-domain claim.
        prepay_polynomial_ordering(&expression, generation)?;
        let expression = crate::simplify_with(&expression, generation)?;
        generation.checkpoint().map_err(SimplifyError::from_meter)?;
        verification
            .checkpoint()
            .map_err(SimplifyError::from_meter)?;
        return Ok(expression);
    }

    // Preserve the ordinary traversal for non-rational functions/constants,
    // while each rational argument still uses the checked original-source path.
    let mapped = match source {
        Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
            generation.allocate(children.len() * size_of::<Expr>())?;
            let mut output = Vec::with_capacity(children.len());
            for child in children {
                output.push(cancel_component(child, generation, verification)?);
            }
            match source {
                Expr::Add(_) => Expr::Add(output),
                Expr::Mul(_) => Expr::Mul(output),
                Expr::Function(name, _) => {
                    generation.allocate(name.len())?;
                    Expr::Function(name.clone(), output)
                }
                _ => unreachable!(),
            }
        }
        Expr::Pow(base, exponent) => {
            let base = cancel_component(base, generation, verification)?;
            let exponent = cancel_component(exponent, generation, verification)?;
            generation.allocate(2 * (size_of::<Expr>() + 2 * size_of::<usize>()))?;
            Expr::Pow(Arc::new(base), Arc::new(exponent))
        }
        Expr::Const(value) => Expr::Const(*value),
        _ => unreachable!("exact rational atoms were handled above"),
    };
    let output = crate::simplify_with(&mapped, generation)?;
    verification
        .checkpoint()
        .map_err(SimplifyError::from_meter)?;
    Ok(output)
}

/// Cancel exact rational components using independent cumulative phase meters.
///
/// QQ arithmetic supports up to eight typed axes within the existing complete
/// GCD envelope. Source/domain, resource, cancellation and verification failures
/// are terminal. Non-rational functions retain ordinary recursive simplification.
/// The result is a rational-function value, not evidence that removed poles were
/// defined in the original input; use `fsym_polys::checked_rational_cancel` when
/// the original guards must be retained by a proof-producing consumer.
pub fn cancel_with(
    source: &Expr,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<Expr, SimplifyError> {
    let mut generation = Phase::new(generator);
    let mut verification = Phase::new(verifier);
    cancel_component(source, &mut generation, &mut verification)
}
