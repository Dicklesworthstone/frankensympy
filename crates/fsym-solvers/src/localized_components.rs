//! Complete source-domain-preserving projections of localized rational families.
//!
//! Checked localization gives a bijection between original rational solutions
//! and an augmented polynomial variety. Complete component solving happens in
//! the auxiliary-first ring, preferring original free coordinates. A separate
//! reference pass binds every coefficient to the original localization and
//! checks the complete coordinate/component projection before publication.
//!
//! All parameter guards survive projection, including conditions that occurred
//! only in the discarded auxiliary coordinate. A surface that cannot represent
//! such conditions must refuse explicitly rather than returning an open chart
//! as an unconditional tuple family. No auxiliary parameter may escape.

#![forbid(unsafe_code)]

use fsym_budget::{Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{Expr, Symbol};
use fsym_polys::{MultivariatePoly, VerifiedRationalLocalization, bounded_polynomial_from_expr};

use crate::SolverError;
use crate::polynomial_components::{
    PolynomialComponentsOutcome, VerifiedComponentFamily, VerifiedPolynomialComponents,
    metered_rational_components_system,
};

const MAX_VARIABLES: usize = 8;
const MAX_COMPONENTS: usize = 32;
const MAX_PROJECTION_NODES: usize = 32 * 4096;
const MAX_COMPUTE: u64 = 200_000_000;
const MAX_MEMORY: u64 = 512 * 1024 * 1024;

type Result<T> = std::result::Result<T, SolverError>;

#[derive(Debug)]
pub enum LocalizedComponentsOutcome<'a> {
    Empty,
    Families(VerifiedLocalizedComponents<'a>),
    NotApplicable,
}

/// The original source receipt and complete checked augmented union remain
/// bound to this projection. No public constructor creates an acceptance.
#[derive(Debug)]
pub struct VerifiedLocalizedComponents<'a> {
    localization: &'a VerifiedRationalLocalization,
    components: VerifiedPolynomialComponents,
    columns: Vec<usize>,
}

/// A borrowed original-coordinate family with every retained parameter guard.
pub struct LocalizedComponentFamily<'a> {
    family: &'a VerifiedComponentFamily,
}

impl<'a> LocalizedComponentFamily<'a> {
    pub fn values(&self) -> &'a [Expr] {
        &self.family.values()[1..]
    }

    pub fn free_variables(&self) -> &'a [Symbol] {
        self.family.free_variables()
    }

    pub fn nonzero_conditions(&self) -> &'a [Expr] {
        self.family.nonzero_conditions()
    }
}

impl VerifiedLocalizedComponents<'_> {
    pub fn localization(&self) -> &VerifiedRationalLocalization {
        self.localization
    }

    pub fn variables(&self) -> &[Symbol] {
        self.localization.original_variables()
    }

    pub fn families(&self) -> impl ExactSizeIterator<Item = LocalizedComponentFamily<'_>> {
        self.components
            .families()
            .iter()
            .map(|family| LocalizedComponentFamily { family })
    }

    pub fn is_finite(&self) -> bool {
        self.components
            .families()
            .iter()
            .all(|family| family.free_variables().is_empty())
    }

    /// Transfer a complete finite tuple set. Parameterized families must use
    /// into_parts so that no independently retained condition is discarded.
    pub fn into_finite_values(self) -> Result<Vec<Vec<Expr>>> {
        if !self.is_finite() {
            return Err(refuse(
                "parameterized rational families require retained conditions",
            ));
        }
        let columns = self.columns;
        Ok(self
            .components
            .into_values()
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .enumerate()
                    .filter_map(|(column, value)| columns.contains(&column).then_some(value))
                    .collect()
            })
            .collect())
    }

    /// Move every accepted row and every original-domain guard. The result
    /// vectors were reserved before publication; no exact values are cloned.
    pub fn into_parts(self) -> Vec<(Vec<Expr>, Vec<Expr>, Vec<Expr>)> {
        let columns = self.columns;
        self.components
            .into_parts()
            .into_iter()
            .map(|(numerators, denominators, conditions)| {
                let retain = |values: Vec<Expr>| {
                    values
                        .into_iter()
                        .enumerate()
                        .filter_map(|(column, value)| columns.contains(&column).then_some(value))
                        .collect()
                };
                (retain(numerators), retain(denominators), conditions)
            })
            .collect()
    }
}

fn refuse(message: impl std::fmt::Display) -> SolverError {
    SolverError::IncompleteSolutionSet(format!("localized component projection refused: {message}"))
}

fn charge(meter: &mut impl BudgetMeter, steps: usize, bytes: usize) -> Result<()> {
    meter.checkpoint().map_err(refuse)?;
    meter
        .charge_batch(&[
            (Dimension::ComputeSteps, steps.max(1) as u64),
            (Dimension::MemoryBytes, bytes.max(1) as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(refuse)
}

/// The outer projection cap includes source rebinding, every component-search
/// attempt, the reference projection and final owned output. A wrapped failure
/// stays terminal even if a lower-level recognition path changes its error type.
struct Capped<'a, M> {
    inner: &'a mut M,
    compute: u64,
    memory: u64,
    allocations: u64,
    stopped: Option<MeterError>,
}

impl<'a, M: BudgetMeter> Capped<'a, M> {
    fn new(inner: &'a mut M) -> Self {
        Self {
            inner,
            compute: 0,
            memory: 0,
            allocations: 0,
            stopped: None,
        }
    }
}

impl<M: BudgetMeter> BudgetMeter for Capped<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> std::result::Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(
        &mut self,
        charges: &[(Dimension, u64)],
    ) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let (mut compute, mut memory, mut allocations) =
            (self.compute, self.memory, self.allocations);
        for &(dimension, amount) in charges {
            let (used, cap) = match dimension {
                Dimension::ComputeSteps => (&mut compute, MAX_COMPUTE),
                Dimension::MemoryBytes => (&mut memory, MAX_MEMORY),
                Dimension::AllocationCount => (&mut allocations, MAX_COMPUTE),
                _ => continue,
            };
            let remaining = cap.saturating_sub(*used);
            if amount > remaining {
                let error = MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: amount,
                    remaining,
                });
                self.stopped = Some(error);
                return Err(error);
            }
            *used += amount;
        }
        if let Err(error) = self.inner.charge_batch(charges) {
            self.stopped = Some(error);
            return Err(error);
        }
        self.compute = compute;
        self.memory = memory;
        self.allocations = allocations;
        Ok(())
    }

    fn checkpoint(&mut self) -> std::result::Result<(), MeterError> {
        if let Some(error) = self.stopped {
            return Err(error);
        }
        let result = self.inner.checkpoint();
        if let Err(error) = result {
            self.stopped = Some(error);
        }
        result
    }
}

/// Count the full printed-name and typed-identity payload before ring or
/// membership comparisons. Counting variables alone undercharges long names.
fn symbol_comparison_work(variables: &[Symbol]) -> Result<usize> {
    variables
        .iter()
        .try_fold(0usize, |total, variable| {
            total.checked_add(variable.name.len())?.checked_add(64)
        })
        .ok_or_else(|| refuse("typed-symbol comparison charge overflow"))
}

/// Independent direct coefficient-map permutation. No generator expression
/// lifting, substitution or polynomial arithmetic is used by this check.
fn verify_rebinding(
    localization: &VerifiedRationalLocalization,
    variables: &[Symbol],
    equations: &[MultivariatePoly],
    meter: &mut impl BudgetMeter,
) -> Result<()> {
    let original = localization.variables();
    if original.len() < 2
        || original.len() > MAX_VARIABLES
        || variables.len() != original.len()
        || equations.len() != localization.polynomials().len()
    {
        return Err(refuse("localization ring or equation count changed"));
    }
    charge(
        meter,
        symbol_comparison_work(original)? + symbol_comparison_work(variables)?,
        1,
    )?;
    if variables[0] != original[original.len() - 1]
        || variables[1..] != original[..original.len() - 1]
    {
        return Err(refuse("localization typed ring changed"));
    }
    for (source, rebound) in localization.polynomials().iter().zip(equations) {
        charge(
            meter,
            symbol_comparison_work(original)?
                + symbol_comparison_work(variables)?
                + symbol_comparison_work(source.generators())?
                + symbol_comparison_work(rebound.generators())?,
            1,
        )?;
        if source.generators() != original
            || rebound.generators() != variables
            || source.terms().len() != rebound.terms().len()
        {
            return Err(refuse("localization polynomial binding changed"));
        }
        for (powers, coefficient) in source.terms() {
            charge(meter, original.len() + 128, original.len() * 4 + 64)?;
            if powers.len() != original.len() {
                return Err(refuse("localization monomial width changed"));
            }
            let mut projected = Vec::with_capacity(original.len());
            projected.push(powers[original.len() - 1]);
            projected.extend_from_slice(&powers[..original.len() - 1]);
            if rebound.terms().get(&projected) != Some(coefficient) {
                return Err(refuse(
                    "localization projection changed an original coefficient",
                ));
            }
        }
    }
    meter.checkpoint().map_err(refuse)
}

/// Borrowed bounded inspection before recursive value comparison or lifting.
/// An auxiliary symbol is a recognition miss; any other undeclared symbol is
/// an invalid projection. The complete union is withheld in either case.
fn original_symbols_only(
    expression: &Expr,
    originals: &[Symbol],
    auxiliary: &Symbol,
    meter: &mut impl BudgetMeter,
    total_nodes: &mut usize,
) -> Result<bool> {
    charge(meter, 1, 128)?;
    let mut pending = vec![(expression, 0usize)];
    let mut nodes = 0usize;
    while let Some((node, depth)) = pending.pop() {
        nodes += 1;
        *total_nodes += 1;
        charge(meter, 1, 1)?;
        if nodes > 4096 || depth > 64 || *total_nodes > MAX_PROJECTION_NODES {
            return Err(refuse("projected expression exceeds its node/depth bound"));
        }
        match node {
            Expr::Sym(symbol) => {
                let work = (symbol.name.len() + 64)
                    .checked_mul(originals.len() + 1)
                    .ok_or_else(|| refuse("projected symbol comparison charge overflow"))?;
                charge(meter, work, 1)?;
                if symbol == auxiliary {
                    return Ok(false);
                }
                if !originals.contains(symbol) {
                    return Err(refuse(
                        "projected expression contains an undeclared typed symbol",
                    ));
                }
            }
            Expr::Integer(value) if value.bits() > 4096 => {
                return Err(refuse("projected integer exceeds 4096 bits"));
            }
            Expr::Rational(value) if value.numer().bits().max(value.denom().bits()) > 4096 => {
                return Err(refuse("projected rational exceeds 4096 bits"));
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                if children.len() > 4096usize.saturating_sub(nodes + pending.len()) {
                    return Err(refuse("projected expression fanout exceeds its bound"));
                }
                charge(meter, children.len(), children.len() * 32 + 128)?;
                pending.extend(children.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                if nodes + pending.len() + 2 > 4096 {
                    return Err(refuse("projected power exceeds its node bound"));
                }
                charge(meter, 2, 192)?;
                pending.push((base.as_ref(), depth + 1));
                pending.push((exponent.as_ref(), depth + 1));
            }
            _ => {}
        }
    }
    Ok(true)
}

fn verify_projection(
    localization: &VerifiedRationalLocalization,
    components: &VerifiedPolynomialComponents,
    columns: &[usize],
    meter: &mut impl BudgetMeter,
) -> Result<bool> {
    verify_rebinding(
        localization,
        components.variables(),
        components.equations(),
        meter,
    )?;
    let originals = localization.original_variables();
    let variables = components.variables();
    if columns.len() != originals.len()
        || columns
            .iter()
            .enumerate()
            .any(|(index, &column)| column != index + 1)
        || components.families().is_empty()
        || components.families().len() > MAX_COMPONENTS
    {
        return Err(refuse(
            "projection lost an original coordinate or complete component set",
        ));
    }
    let mut total_nodes = 0usize;
    for family in components.families() {
        charge(
            meter,
            symbol_comparison_work(variables)? + symbol_comparison_work(family.variables())?,
            1,
        )?;
        if family.variables() != variables || family.values().len() != variables.len() {
            return Err(refuse("component coordinate binding changed"));
        }
        for free in family.free_variables() {
            let work = (free.name.len() + 64)
                .checked_mul(originals.len() + 1)
                .ok_or_else(|| refuse("free-parameter comparison charge overflow"))?;
            charge(meter, work, 1)?;
            if free == &variables[0] {
                return Ok(false);
            }
            if !originals.contains(free) {
                return Err(refuse("component contains a foreign free parameter"));
            }
        }
        for &column in columns {
            if !original_symbols_only(
                &family.values()[column],
                originals,
                &variables[0],
                meter,
                &mut total_nodes,
            )? {
                return Ok(false);
            }
        }
        for guard in family.nonzero_conditions() {
            if !original_symbols_only(guard, originals, &variables[0], meter, &mut total_nodes)? {
                return Ok(false);
            }
        }
    }
    meter.checkpoint().map_err(refuse)?;
    Ok(true)
}

fn generate<'a>(
    localization: &'a VerifiedRationalLocalization,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<LocalizedComponentsOutcome<'a>> {
    charge(generator, 1, 1)?;
    let input_variables = localization.variables();
    if input_variables.len() < 2 || input_variables.len() > MAX_VARIABLES {
        return Err(refuse("localization requires 1..7 original coordinates"));
    }
    let width = localization.original_width();
    let bytes: usize = input_variables
        .iter()
        .map(|variable| variable.name.len() + 128)
        .sum();
    charge(generator, bytes, bytes + (width + 1) * 16)?;
    let mut variables = Vec::with_capacity(width + 1);
    variables.push(input_variables[width].clone());
    variables.extend_from_slice(&input_variables[..width]);
    charge(
        generator,
        localization.equations().len(),
        localization.equations().len() * 128,
    )?;
    let rebound = localization
        .equations()
        .iter()
        .map(|source| bounded_polynomial_from_expr(source, &variables, generator).map_err(refuse))
        .collect::<Result<Vec<_>>>()?;
    verify_rebinding(localization, &variables, &rebound, verifier)?;
    let outcome = metered_rational_components_system(
        localization.equations(),
        &variables,
        generator,
        verifier,
    )?;
    let components = match outcome {
        PolynomialComponentsOutcome::Empty => {
            verifier.checkpoint().map_err(refuse)?;
            return Ok(LocalizedComponentsOutcome::Empty);
        }
        PolynomialComponentsOutcome::NotApplicable => {
            return Ok(LocalizedComponentsOutcome::NotApplicable);
        }
        PolynomialComponentsOutcome::Families(components) => components,
    };
    charge(generator, width, width * std::mem::size_of::<usize>())?;
    let columns: Vec<_> = (1..=width).collect();
    if !verify_projection(localization, &components, &columns, verifier)? {
        return Ok(LocalizedComponentsOutcome::NotApplicable);
    }
    let count = components.families().len();
    // Reserve both projected coordinate vectors and both outer transfer
    // vectors before acceptance. Extraction moves values and keeps all guards.
    charge(
        generator,
        count * (width + 1) * (width + 1),
        count * (2 * width * std::mem::size_of::<Expr>() + 512),
    )?;
    generator.checkpoint().map_err(refuse)?;
    verifier.checkpoint().map_err(refuse)?;
    Ok(LocalizedComponentsOutcome::Families(
        VerifiedLocalizedComponents {
            localization,
            components,
            columns,
        },
    ))
}

/// Complete component projection of an already independently checked source
/// localization. Recognition misses retain the entire original source, while
/// cancellation or resource failures remain terminal.
pub fn metered_localized_rational_components<'a>(
    localization: &'a VerifiedRationalLocalization,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<LocalizedComponentsOutcome<'a>> {
    let mut generator = Capped::new(generator_meter);
    let mut verifier = Capped::new(verifier_meter);
    let result = generate(localization, &mut generator, &mut verifier);
    if let Some(error) = generator.stopped.or(verifier.stopped) {
        return Err(refuse(error));
    }
    result
}

pub fn localized_rational_components(
    localization: &VerifiedRationalLocalization,
) -> Result<LocalizedComponentsOutcome<'_>> {
    metered_localized_rational_components(
        localization,
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
        &mut Budget::new(BudgetLimits::uniform(MAX_MEMORY, 0)),
    )
}

#[cfg(test)]
mod reference_checks {
    use super::*;
    use fsym_budget::Unbounded;
    use fsym_core::{BigRational, SymbolIdentity, parse};
    use fsym_polys::{RationalSystemAdmission, metered_localize_rational_system};
    use std::sync::Arc;

    fn fixture(source: &str) -> VerifiedRationalLocalization {
        let variables = [Symbol::new("x"), Symbol::new("y")];
        match metered_localize_rational_system(
            &[parse(source).unwrap()],
            &variables,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap()
        {
            RationalSystemAdmission::Localized(value) => *value,
            _ => panic!("fixture requires original reciprocal guards"),
        }
    }

    #[test]
    fn coefficient_map_rebinding_rejects_changed_omitted_and_typed_inputs() {
        let localization = fixture("1/x-y");
        let LocalizedComponentsOutcome::Families(accepted) =
            localized_rational_components(&localization).unwrap()
        else {
            panic!("expected a complete localized graph")
        };
        let variables = accepted.components.variables();
        let equations = accepted.components.equations();
        verify_rebinding(&localization, variables, equations, &mut Unbounded).unwrap();

        let mut changed = equations.to_vec();
        let mut terms = changed[0].terms().clone();
        terms.insert(vec![0; variables.len()], BigRational::from(2_i64));
        changed[0] = MultivariatePoly::new(variables.to_vec(), terms).unwrap();
        assert!(verify_rebinding(&localization, variables, &changed, &mut Unbounded).is_err());

        assert!(
            verify_rebinding(
                &localization,
                variables,
                &equations[..equations.len() - 1],
                &mut Unbounded,
            )
            .is_err()
        );
        let mut reordered = equations.to_vec();
        reordered.swap(0, 1);
        assert!(verify_rebinding(&localization, variables, &reordered, &mut Unbounded).is_err());

        let mut typed = variables.to_vec();
        typed[1] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([103; 32]));
        assert!(verify_rebinding(&localization, &typed, equations, &mut Unbounded).is_err());
    }

    #[test]
    fn projection_rejects_auxiliary_omitted_duplicate_and_reordered_coordinates() {
        let localization = fixture("1/x-y");
        let LocalizedComponentsOutcome::Families(accepted) =
            localized_rational_components(&localization).unwrap()
        else {
            panic!("expected a complete localized graph")
        };
        for columns in [vec![0, 2], vec![1], vec![1, 1], vec![2, 1], vec![1, 3]] {
            assert!(
                verify_projection(
                    &localization,
                    &accepted.components,
                    &columns,
                    &mut Unbounded,
                )
                .is_err()
            );
        }
        assert!(
            verify_projection(&localization, &accepted.components, &[1, 2], &mut Unbounded,)
                .unwrap()
        );
    }

    #[test]
    fn a_checked_subset_or_different_rational_source_cannot_replace_the_complete_union() {
        let original = fixture("(1/x-y)*(1/x+y)");
        let subset = fixture("1/x-y");
        let LocalizedComponentsOutcome::Families(accepted_subset) =
            localized_rational_components(&subset).unwrap()
        else {
            panic!("expected a complete single graph")
        };
        assert!(
            verify_projection(
                &original,
                &accepted_subset.components,
                &[1, 2],
                &mut Unbounded,
            )
            .is_err()
        );
        let changed = fixture("2/x-y");
        assert!(
            verify_projection(
                &changed,
                &accepted_subset.components,
                &[1, 2],
                &mut Unbounded,
            )
            .is_err()
        );
    }

    #[test]
    fn auxiliary_and_foreign_symbols_cannot_escape_in_values_or_guard_expressions() {
        let localization = fixture("1/x-y");
        let originals = localization.original_variables();
        let auxiliary = &localization.certificate().auxiliary;
        let hidden = Expr::Mul(vec![Expr::from_i64(0), Expr::Sym(auxiliary.clone())]);
        assert!(
            !original_symbols_only(&hidden, originals, auxiliary, &mut Unbounded, &mut 0,).unwrap()
        );
        let foreign = Expr::Sym(Symbol::with_identity(
            "x",
            SymbolIdentity::from_assumptions_digest([104; 32]),
        ));
        assert!(
            original_symbols_only(&foreign, originals, auxiliary, &mut Unbounded, &mut 0,).is_err()
        );
        assert!(
            original_symbols_only(
                &parse("x*y+1").unwrap(),
                originals,
                auxiliary,
                &mut Unbounded,
                &mut 0,
            )
            .unwrap()
        );
    }

    #[test]
    fn full_typed_names_are_charged_before_ring_or_output_membership_comparisons() {
        let variables = [Symbol::new("x".repeat(2048)), Symbol::new("y")];
        let input = Expr::Pow(
            Arc::new(Expr::Sym(variables[0].clone())),
            Arc::new(Expr::from_i64(-1)),
        ) - Expr::Sym(variables[1].clone());
        let RationalSystemAdmission::Localized(localization) =
            metered_localize_rational_system(&[input], &variables, &mut Unbounded, &mut Unbounded)
                .unwrap()
        else {
            panic!("expected a localized long-name source")
        };
        let reordered = [
            localization.certificate().auxiliary.clone(),
            variables[0].clone(),
            variables[1].clone(),
        ];
        let rebound: Vec<_> = localization
            .equations()
            .iter()
            .map(|source| bounded_polynomial_from_expr(source, &reordered, &mut Unbounded).unwrap())
            .collect();
        let mut limits = BudgetLimits::uniform(MAX_MEMORY, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = 2047;
        assert!(
            verify_rebinding(
                &localization,
                &reordered,
                &rebound,
                &mut Budget::new(limits),
            )
            .is_err()
        );
        assert!(
            original_symbols_only(
                &Expr::Sym(variables[0].clone()),
                &variables,
                &localization.certificate().auxiliary,
                &mut Budget::new(limits),
                &mut 0,
            )
            .is_err()
        );
        verify_rebinding(&localization, &reordered, &rebound, &mut Unbounded).unwrap();
    }

    #[test]
    fn capped_meter_refusal_is_atomic_and_remains_terminal() {
        let mut limits = BudgetLimits::uniform(32, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = 2;
        let mut inner = Budget::new(limits);
        let mut capped = Capped::new(&mut inner);
        assert!(
            capped
                .charge_batch(&[(Dimension::ComputeSteps, 3), (Dimension::MemoryBytes, 1),])
                .is_err()
        );
        assert_eq!(
            (capped.compute, capped.memory, capped.allocations),
            (0, 0, 0)
        );
        assert!(capped.charge(Dimension::ComputeSteps, 1).is_err());
        assert!(capped.checkpoint().is_err());
    }
}
