//! Complete real solutions of bounded rational systems in shape position.
//!
//! A checked reduced lex basis of the form
//! `x_0 - q_0(t), ..., x_(n-2) - q_(n-2)(t), p(t)` identifies every solution
//! with exactly one root of `p`. A separate reference pass reconstructs these
//! basis elements from the proposed coefficient vectors. The existing proof
//! kernel has already checked both ideal inclusions and all Groebner criteria;
//! the independent real-root checker establishes the complete real root set.
//!
//! The result is an exact rational univariate representation, not a list of
//! approximate tuples or a new native expression encoding for algebraic roots.
//! Nonreal eliminant roots are explicitly visible through
//! `all_complex_roots_real`; a complex-domain caller must not publish only the
//! real tuples unless that predicate is true. General triangular extensions,
//! primitive-element search and positive-dimensional systems remain unsupported.

use std::collections::BTreeMap;

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension};
use fsym_core::real_roots::{MAX_REAL_ROOT_DEGREE, VerifiedRealRoots, isolate_real_roots};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::{
    MultivariatePoly, TermOrder, VerifiedGroebnerBasis, bounded_polynomial_from_expr,
    checked_groebner_basis,
};
use num_traits::{One, Zero};

use crate::SolverError;

const MAX_EQUATIONS: usize = 64;
const MAX_VARIABLES: usize = 8;
const MAX_COEFFICIENT_BITS: u64 = 4096;
const MAX_SHAPE_WORK: usize = 20_000_000;
const MAX_SHAPE_ALLOCATION: usize = 128 * 1024 * 1024;

/// An immutable complete real solution representation bound to its input ideal.
/// The parameter is the last entry of `variables`; coefficients are ascending.
/// Each distinct interval in `roots` supplies one tuple by evaluating every
/// coordinate polynomial at that root. The last coordinate is always `t`.
#[derive(Debug, Clone)]
pub struct VerifiedRealShapeSystem {
    variables: Vec<Symbol>,
    basis: VerifiedGroebnerBasis,
    eliminant: Vec<BigRational>,
    coordinates: Vec<Vec<BigRational>>,
    roots: VerifiedRealRoots,
}

impl VerifiedRealShapeSystem {
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    pub fn basis(&self) -> &VerifiedGroebnerBasis {
        &self.basis
    }

    pub fn eliminant(&self) -> &[BigRational] {
        &self.eliminant
    }

    pub fn coordinates(&self) -> &[Vec<BigRational>] {
        &self.coordinates
    }

    pub fn roots(&self) -> &VerifiedRealRoots {
        &self.roots
    }

    /// The verified real-root count includes original multiplicities. Equality
    /// with the eliminant's degree proves that no nonreal roots are omitted.
    /// Constant one represents the empty ideal variety and also satisfies this.
    pub fn all_complex_roots_real(&self) -> bool {
        self.roots.count_with_multiplicity() == self.eliminant.len() - 1
    }
}

fn refuse(message: impl std::fmt::Display) -> SolverError {
    SolverError::IncompleteSolutionSet(format!("real shape system refused: {message}"))
}

struct ShapeWork<'a, M> {
    meter: &'a mut M,
    compute: usize,
    allocation: usize,
}

impl<'a, M: BudgetMeter> ShapeWork<'a, M> {
    fn new(meter: &'a mut M) -> Result<Self, SolverError> {
        meter.checkpoint().map_err(refuse)?;
        Ok(Self {
            meter,
            compute: 0,
            allocation: 0,
        })
    }

    fn charge(&mut self, steps: usize, bytes: usize) -> Result<(), SolverError> {
        self.meter.checkpoint().map_err(refuse)?;
        self.compute = self
            .compute
            .checked_add(steps.max(1))
            .filter(|&total| total <= MAX_SHAPE_WORK)
            .ok_or_else(|| refuse("cumulative shape work limit"))?;
        self.allocation = self
            .allocation
            .checked_add(bytes.max(1))
            .filter(|&total| total <= MAX_SHAPE_ALLOCATION)
            .ok_or_else(|| refuse("cumulative shape allocation limit"))?;
        self.meter
            .charge_batch(&[
                (Dimension::ComputeSteps, steps.max(1) as u64),
                (Dimension::MemoryBytes, bytes.max(1) as u64),
                (Dimension::AllocationCount, 1),
            ])
            .map_err(refuse)
    }

    fn coefficient(&mut self, value: &BigRational) -> Result<BigRational, SolverError> {
        if value.numer().bits().max(value.denom().bits()) > MAX_COEFFICIENT_BITS {
            return Err(refuse("coefficient exceeds 4096 bits"));
        }
        self.charge(
            1,
            64 + (value.numer().bits() + value.denom().bits()).div_ceil(8) as usize,
        )?;
        Ok(value.clone())
    }

    fn zero_vector(&mut self, count: usize) -> Result<Vec<BigRational>, SolverError> {
        if count == 0 || count > MAX_REAL_ROOT_DEGREE + 1 {
            return Err(refuse("coefficient-vector degree exceeds 64"));
        }
        self.charge(count, count * 128)?;
        Ok(vec![BigRational::zero(); count])
    }
}

fn variables(source: &VerifiedGroebnerBasis) -> &[Symbol] {
    source
        .input_basis()
        .first()
        .or_else(|| source.basis().first())
        .map_or(&[], MultivariatePoly::generators)
}

fn is_unit_ideal(source: &VerifiedGroebnerBasis) -> bool {
    source.basis().len() == 1 && source.basis()[0].is_one()
}

fn check_vector(values: &[BigRational]) -> Result<(), SolverError> {
    if values.is_empty()
        || values.len() > MAX_REAL_ROOT_DEGREE + 1
        || values.len() > 1 && values.last().is_some_and(Zero::is_zero)
    {
        return Err(refuse("noncanonical or oversized coefficient vector"));
    }
    if values
        .iter()
        .any(|value| value.numer().bits().max(value.denom().bits()) > MAX_COEFFICIENT_BITS)
    {
        return Err(refuse("coefficient exceeds 4096 bits"));
    }
    Ok(())
}

/// Independently check an untrusted proposed shape representation against an
/// already verified reduced lex basis. Reconstructing the exact basis, rather
/// than sampling or evaluating roots, establishes both solution directions.
pub fn verify_shape_representation(
    source: &VerifiedGroebnerBasis,
    eliminant: &[BigRational],
    coordinates: &[Vec<BigRational>],
    meter: &mut impl BudgetMeter,
) -> Result<(), SolverError> {
    let mut work = ShapeWork::new(meter)?;
    let vars = variables(source);
    let width = vars.len();
    if source.order() != TermOrder::Lex || width == 0 || width > MAX_VARIABLES {
        return Err(refuse(
            "shape verification requires a bounded lexicographic ring",
        ));
    }
    if coordinates.len() != width || source.basis().len() > MAX_EQUATIONS {
        return Err(refuse("shape dimensions do not match the checked ring"));
    }
    check_vector(eliminant)?;
    for coordinate in coordinates {
        check_vector(coordinate)?;
    }
    if !eliminant.last().is_some_and(One::is_one) {
        return Err(refuse("eliminant must be nonzero and monic"));
    }
    let last = width - 1;
    if coordinates[last].len() != 2
        || !coordinates[last][0].is_zero()
        || !coordinates[last][1].is_one()
    {
        return Err(refuse(
            "the final coordinate must equal the eliminant parameter",
        ));
    }
    if is_unit_ideal(source) {
        if eliminant.len() != 1
            || coordinates[..last]
                .iter()
                .any(|coordinate| coordinate.len() != 1 || !coordinate[0].is_zero())
        {
            return Err(refuse("noncanonical representation of the empty variety"));
        }
        work.charge(width, width * 128)?;
        return work.meter.checkpoint().map_err(refuse);
    }
    if eliminant.len() < 2 || source.basis().len() != width {
        return Err(refuse("the checked basis is not a complete shape basis"));
    }
    let degree = eliminant.len() - 1;
    let symbol_bytes = vars.iter().map(|var| var.name.len() + 64).sum::<usize>();
    for index in 0..width {
        let coeffs = if index == last {
            eliminant
        } else {
            if coordinates[index].len() > degree {
                return Err(refuse("coordinate is not reduced modulo the eliminant"));
            }
            &coordinates[index]
        };
        work.charge(
            coeffs.len() * width + width,
            symbol_bytes + coeffs.len() * 128,
        )?;
        let mut terms = BTreeMap::new();
        if index != last {
            let mut powers = vec![0; width];
            powers[index] = 1;
            terms.insert(powers, BigRational::one());
        }
        for (power, coefficient) in coeffs.iter().enumerate() {
            if !coefficient.is_zero() {
                let mut powers = vec![0; width];
                powers[last] = power as u32;
                let coefficient = work.coefficient(coefficient)?;
                terms.insert(
                    powers,
                    if index == last {
                        coefficient
                    } else {
                        -coefficient
                    },
                );
            }
        }
        let reconstructed = MultivariatePoly {
            generators: vars.to_vec(),
            terms,
        };
        work.charge(source.basis().len() * reconstructed.terms().len().max(1), 1)?;
        if !source.basis().contains(&reconstructed) {
            return Err(refuse(
                "shape polynomial differs from the independently checked basis",
            ));
        }
    }
    work.meter.checkpoint().map_err(refuse)
}

type ShapeCoefficients = (Vec<BigRational>, Vec<Vec<BigRational>>);

fn propose_shape(
    source: &VerifiedGroebnerBasis,
    work: &mut ShapeWork<'_, impl BudgetMeter>,
) -> Result<Option<ShapeCoefficients>, SolverError> {
    let width = variables(source).len();
    if source.order() != TermOrder::Lex || width == 0 || width > MAX_VARIABLES {
        return Ok(None);
    }
    let last = width - 1;
    work.charge(width, width * 64)?;
    if is_unit_ideal(source) {
        let mut coordinates = Vec::with_capacity(width);
        for _ in 0..last {
            coordinates.push(work.zero_vector(1)?);
        }
        work.charge(3, 3 * 128 + 64)?;
        coordinates.push(vec![BigRational::zero(), BigRational::one()]);
        return Ok(Some((vec![BigRational::one()], coordinates)));
    }
    // Recognizing all n required shape polynomials itself establishes a
    // finite quotient; each leading monomial is a pure variable power.
    if source.basis().len() != width {
        return Ok(None);
    }
    let mut eliminant = None;
    let mut coordinates = vec![None; width];
    for polynomial in source.basis() {
        work.charge(polynomial.terms().len() * width, 1)?;
        let Some((head, coefficient)) = polynomial.leading_term(TermOrder::Lex) else {
            return Ok(None);
        };
        if !coefficient.is_one() {
            return Ok(None);
        }
        let Some(variable) = head.iter().position(|&power| power != 0) else {
            return Ok(None);
        };
        if variable != last && (head[variable] != 1 || head.iter().sum::<u32>() != 1) {
            return Ok(None);
        }
        let degree = polynomial.degree_in(last) as usize;
        let mut coeffs = work.zero_vector(degree + 1)?;
        for (powers, coefficient) in polynomial.terms() {
            if variable != last && powers == head {
                continue;
            }
            if powers[..last].iter().any(|&power| power != 0) {
                return Ok(None);
            }
            let coefficient = work.coefficient(coefficient)?;
            coeffs[powers[last] as usize] = if variable == last {
                coefficient
            } else {
                -coefficient
            };
        }
        if variable == last {
            if eliminant.replace(coeffs).is_some() {
                return Ok(None);
            }
        } else if coordinates[variable].replace(coeffs).is_some() {
            return Ok(None);
        }
    }
    let Some(eliminant) = eliminant else {
        return Ok(None);
    };
    work.charge(2, 2 * 128 + width * 32)?;
    coordinates[last] = Some(vec![BigRational::zero(), BigRational::one()]);
    let Some(coordinates) = coordinates.into_iter().collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    Ok(Some((eliminant, coordinates)))
}

/// Reuse a checked basis without recomputing it or resetting caller accounting.
/// `None` means the source is outside this shape lane, not that it has no roots.
pub fn metered_real_shape_from_basis(
    source: VerifiedGroebnerBasis,
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealShapeSystem>, SolverError> {
    let proposal = propose_shape(&source, &mut ShapeWork::new(generator_meter)?)?;
    let Some((eliminant, coordinates)) = proposal else {
        generator_meter.checkpoint().map_err(refuse)?;
        verifier_meter.checkpoint().map_err(refuse)?;
        return Ok(None);
    };
    verify_shape_representation(&source, &eliminant, &coordinates, verifier_meter)?;
    let roots = isolate_real_roots(&eliminant, generator_meter, verifier_meter).map_err(refuse)?;
    let vars = variables(&source);
    let bytes = vars.iter().map(|var| var.name.len() + 64).sum::<usize>();
    generator_meter
        .charge_batch(&[
            (Dimension::ComputeSteps, vars.len() as u64),
            (Dimension::MemoryBytes, bytes as u64),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(refuse)?;
    let variables = vars.to_vec();
    generator_meter.checkpoint().map_err(refuse)?;
    verifier_meter.checkpoint().map_err(refuse)?;
    Ok(Some(VerifiedRealShapeSystem {
        variables,
        basis: source,
        eliminant,
        coordinates,
        roots,
    }))
}

/// Admit every original expression, generate and independently check its basis,
/// then solve a recognized shape system completely over the reals. All budgets
/// are cumulative across admission, basis generation, shape checking and roots.
pub fn metered_solve_real_shape_system(
    equations: &[Expr],
    vars: &[Symbol],
    generator_meter: &mut impl BudgetMeter,
    verifier_meter: &mut impl BudgetMeter,
) -> Result<Option<VerifiedRealShapeSystem>, SolverError> {
    generator_meter.checkpoint().map_err(refuse)?;
    if vars.is_empty()
        || vars.len() > MAX_VARIABLES
        || equations.len() > MAX_EQUATIONS
        || vars
            .iter()
            .enumerate()
            .any(|(index, var)| vars[..index].contains(var))
    {
        return Err(SolverError::InvalidSystem(
            "real shape systems require 1..8 distinct variables and at most 64 equations".into(),
        ));
    }
    generator_meter
        .charge_batch(&[
            (Dimension::ComputeSteps, equations.len().max(1) as u64),
            (Dimension::MemoryBytes, equations.len().max(1) as u64 * 128),
            (Dimension::AllocationCount, 1),
        ])
        .map_err(refuse)?;
    let polynomials = equations
        .iter()
        .map(|expression| bounded_polynomial_from_expr(expression, vars, generator_meter))
        .collect::<Result<Vec<_>, _>>()
        .map_err(refuse)?;
    let source = checked_groebner_basis(
        &polynomials,
        TermOrder::Lex,
        generator_meter,
        verifier_meter,
    )
    .map_err(refuse)?;
    metered_real_shape_from_basis(source, generator_meter, verifier_meter)
}

pub fn solve_real_shape_system(
    equations: &[Expr],
    vars: &[Symbol],
) -> Result<Option<VerifiedRealShapeSystem>, SolverError> {
    metered_solve_real_shape_system(
        equations,
        vars,
        &mut Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0)),
        &mut Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0)),
    )
}
