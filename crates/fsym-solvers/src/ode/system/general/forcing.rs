//! Complete constant-coefficient systems with rational exponential/trigonometric forcing.
//!
//! A finite forcing realization uses the entire functions
//! t^k*exp(a*t)*cos(b*t) and t^k*exp(a*t)*sin(b*t). Its coefficient matrix is
//! only a proposal: a separate coefficient-map checker proves every source
//! coupling, derivative recurrence and initial seed before the checked general
//! ODE verifier admits its fundamental matrix. The physical homogeneous family
//! is checked separately, so auxiliary initial values never become integration
//! constants of the original problem.
//!
//! Polynomial forcing keeps its existing absolute-time convention. A driven
//! particular solution normalized at zero is translated to arbitrary rational
//! initial time by subtracting Phi(t-t0)*p(t0). All functions here are entire
//! on real time; this composition preserves both the original differential
//! equation and its initial values. The physical plus auxiliary dimension is
//! at most eight. Both phases retain one cumulative cap across all subcalls.
//! These are immutable in-process composition receipts, not portable proofs.

use super::{
    Capped, ExpressionMatrix, GeneralOdeSystemCertificate, GeneralOdeSystemError, MAX_DIMENSION,
    MAX_FORCING_DEGREE, MAX_INPUT_BITS, PolynomialVector, RationalMatrix, Result,
    VerifiedGeneralOdeSystem, Work, admit_certificate, admit_source, generate, product, q,
    rational_expr, render, sum, validate_number, verify,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{BigRational, Expr, Symbol};
use num_traits::{Signed, Zero};
use std::collections::{BTreeMap, BTreeSet};

/// One vector-valued t^power*exp(rate*t)*(cosine*cos(frequency*t)+sine*sin(frequency*t)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementaryForcing {
    pub power: usize,
    pub rate: BigRational,
    pub frequency: BigRational,
    pub cosine: Vec<BigRational>,
    pub sine: Vec<BigRational>,
}

/// One canonical auxiliary entire function. Zero frequency has only a cosine slot.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ForcingBasis {
    pub rate: BigRational,
    pub frequency: BigRational,
    pub power: usize,
    pub sine: bool,
}

/// Untrusted augmentation, with physical coordinates preceding auxiliary ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForcingRealization {
    pub basis: Vec<ForcingBasis>,
    pub coefficients: RationalMatrix,
}

/// All untrusted ingredients needed to replay the original-source check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForcedOdeSystemCertificate {
    pub realization: ForcingRealization,
    pub baseline: GeneralOdeSystemCertificate,
    pub driven: GeneralOdeSystemCertificate,
}

/// A complete affine family of the original physical equations.
#[derive(Debug, Clone)]
pub struct VerifiedForcedOdeSystem {
    coefficients: RationalMatrix,
    polynomial: PolynomialVector,
    forcing: Vec<ElementaryForcing>,
    time: Symbol,
    initial_point: BigRational,
    initial_values: Vec<Expr>,
    certificate: ForcedOdeSystemCertificate,
    fundamental: ExpressionMatrix,
    solutions: Vec<Expr>,
}

impl VerifiedForcedOdeSystem {
    pub fn coefficients(&self) -> &[Vec<BigRational>] {
        &self.coefficients
    }
    pub fn polynomial_forcing(&self) -> &[Vec<BigRational>] {
        &self.polynomial
    }
    pub fn elementary_forcing(&self) -> &[ElementaryForcing] {
        &self.forcing
    }
    pub fn time(&self) -> &Symbol {
        &self.time
    }
    pub fn initial_point(&self) -> &BigRational {
        &self.initial_point
    }
    pub fn initial_values(&self) -> &[Expr] {
        &self.initial_values
    }
    pub fn certificate(&self) -> &ForcedOdeSystemCertificate {
        &self.certificate
    }
    pub fn fundamental(&self) -> &[Vec<Expr>] {
        &self.fundamental
    }
    pub fn solutions(&self) -> &[Expr] {
        &self.solutions
    }
    pub fn into_solutions(self) -> Vec<Expr> {
        self.solutions
    }
}

fn active(mode: &ElementaryForcing) -> bool {
    mode.cosine
        .iter()
        .chain(&mode.sine)
        .any(|value| !value.is_zero())
}

fn admit_modes(
    modes: &[ElementaryForcing],
    dimension: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.meter.checkpoint()?;
    if modes.len() > MAX_DIMENSION {
        return Err(GeneralOdeSystemError::Unsupported(
            "at most eight elementary forcing terms",
        ));
    }
    for mode in modes {
        if mode.power > MAX_FORCING_DEGREE
            || mode.cosine.len() != dimension
            || mode.sine.len() != dimension
            || mode.frequency.is_negative()
        {
            return Err(GeneralOdeSystemError::InvalidSource(
                "forcing powers zero through sixteen, nonnegative frequency and matching vectors",
            ));
        }
        validate_number(&mode.rate, MAX_INPUT_BITS)?;
        validate_number(&mode.frequency, MAX_INPUT_BITS)?;
        for value in mode.cosine.iter().chain(&mode.sine) {
            validate_number(value, MAX_INPUT_BITS)?;
        }
        if mode.frequency.is_zero() && mode.sine.iter().any(|value| !value.is_zero()) {
            return Err(GeneralOdeSystemError::InvalidSource(
                "zero-frequency sine coefficients must be zero",
            ));
        }
        // Prepay bounded input retention and all derivative-closure key copies.
        work.charge(
            4 * (dimension + mode.power + 1),
            4096 * (dimension + mode.power + 1),
        )?;
    }
    Ok(())
}

fn basis(mode: &ElementaryForcing, power: usize, sine: bool) -> ForcingBasis {
    ForcingBasis {
        rate: mode.rate.clone(),
        frequency: mode.frequency.clone(),
        power,
        sine,
    }
}

fn propose_realization(
    matrix: &[Vec<BigRational>],
    modes: &[ElementaryForcing],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ForcingRealization> {
    let n = matrix.len();
    let mut keys = BTreeSet::new();
    for mode in modes {
        if !active(mode) {
            continue;
        }
        for power in 0..=mode.power {
            work.charge(2, 1024)?;
            keys.insert(basis(mode, power, false));
            if !mode.frequency.is_zero() {
                keys.insert(basis(mode, power, true));
            }
            if n + keys.len() > MAX_DIMENSION {
                return Err(GeneralOdeSystemError::Unsupported(
                    "physical plus forcing-state dimension exceeds eight",
                ));
            }
        }
    }
    let keys: Vec<_> = keys.into_iter().collect();
    let mut coefficients = work.matrix(n + keys.len())?;
    for (row, source) in matrix.iter().enumerate() {
        for (column, value) in source.iter().enumerate() {
            coefficients[row][column] = work.copy(value)?;
        }
    }
    for mode in modes {
        if !active(mode) {
            continue;
        }
        for (sine, values) in [(false, &mode.cosine), (true, &mode.sine)] {
            if values.iter().all(Zero::is_zero) {
                continue;
            }
            let column = keys
                .binary_search(&basis(mode, mode.power, sine))
                .map_err(|_| {
                    GeneralOdeSystemError::Verification("missing proposed forcing coordinate")
                })?;
            for (row, value) in values.iter().enumerate() {
                coefficients[row][n + column] = work.add(&coefficients[row][n + column], value)?;
            }
        }
    }
    for (index, key) in keys.iter().enumerate() {
        let row = n + index;
        coefficients[row][row] = work.copy(&key.rate)?;
        if key.power > 0 {
            let mut lower = key.clone();
            lower.power -= 1;
            let column = keys.binary_search(&lower).map_err(|_| {
                GeneralOdeSystemError::Verification("missing proposed lower forcing power")
            })?;
            coefficients[row][n + column] = q(key.power);
        }
        if !key.frequency.is_zero() {
            let mut other = key.clone();
            other.sine = !other.sine;
            let column = keys.binary_search(&other).map_err(|_| {
                GeneralOdeSystemError::Verification("missing proposed oscillatory coordinate")
            })?;
            coefficients[row][n + column] = if key.sine {
                work.copy(&key.frequency)?
            } else {
                work.sub(&BigRational::zero(), &key.frequency)?
            };
        }
    }
    work.meter.checkpoint()?;
    Ok(ForcingRealization {
        basis: keys,
        coefficients,
    })
}

fn add_coefficient(
    values: &mut BTreeMap<ForcingBasis, BigRational>,
    key: ForcingBasis,
    coefficient: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<()> {
    work.charge(1, 1024)?;
    let old = values.get(&key).cloned().unwrap_or_else(BigRational::zero);
    let next = work.add(&old, coefficient)?;
    values.insert(key, next);
    Ok(())
}

/// Check source couplings and derivative identities in an independent coefficient map.
fn check_realization(
    matrix: &[Vec<BigRational>],
    modes: &[ElementaryForcing],
    candidate: &ForcingRealization,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<Expr>> {
    let n = matrix.len();
    let count = candidate.basis.len();
    if count > MAX_DIMENSION - n
        || candidate.coefficients.len() != n + count
        || candidate
            .coefficients
            .iter()
            .any(|row| row.len() != n + count)
    {
        return Err(GeneralOdeSystemError::Verification(
            "forcing realization shape or canonical basis",
        ));
    }
    for row in &candidate.coefficients {
        for value in row {
            validate_number(value, MAX_INPUT_BITS)?;
        }
    }
    for key in &candidate.basis {
        validate_number(&key.rate, MAX_INPUT_BITS)?;
        validate_number(&key.frequency, MAX_INPUT_BITS)?;
        if key.power > MAX_FORCING_DEGREE
            || key.frequency.is_negative()
            || (key.frequency.is_zero() && key.sine)
            || !modes.iter().any(|mode| {
                active(mode)
                    && mode.rate == key.rate
                    && mode.frequency == key.frequency
                    && mode.power >= key.power
            })
        {
            return Err(GeneralOdeSystemError::Verification(
                "foreign or invalid forcing basis function",
            ));
        }
    }
    work.charge(4096 * (n + count).pow(2), 4096 * (n + count).pow(2))?;
    if candidate.basis.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(GeneralOdeSystemError::Verification(
            "forcing realization basis is not strictly ordered",
        ));
    }
    for (row, original) in matrix.iter().enumerate() {
        if candidate.coefficients[row][..n] != original[..] {
            return Err(GeneralOdeSystemError::Verification(
                "original physical coefficient matrix",
            ));
        }
        let mut residual = BTreeMap::new();
        for mode in modes {
            for (sine, value) in [(false, &mode.cosine[row]), (true, &mode.sine[row])] {
                if !value.is_zero() {
                    add_coefficient(&mut residual, basis(mode, mode.power, sine), value, work)?;
                }
            }
        }
        for (column, key) in candidate.basis.iter().enumerate() {
            let negative = work.sub(
                &BigRational::zero(),
                &candidate.coefficients[row][n + column],
            )?;
            add_coefficient(&mut residual, key.clone(), &negative, work)?;
        }
        if residual.values().any(|value| !value.is_zero()) {
            return Err(GeneralOdeSystemError::Verification(
                "original elementary forcing identity",
            ));
        }
    }
    for (index, key) in candidate.basis.iter().enumerate() {
        let row = &candidate.coefficients[n + index];
        if row[..n].iter().any(|value| !value.is_zero()) {
            return Err(GeneralOdeSystemError::Verification(
                "forcing realization depends on physical coordinates",
            ));
        }
        let mut residual = BTreeMap::new();
        add_coefficient(&mut residual, key.clone(), &key.rate, work)?;
        if key.power != 0 {
            let mut lower = key.clone();
            lower.power -= 1;
            add_coefficient(&mut residual, lower, &q(key.power), work)?;
        }
        if !key.frequency.is_zero() {
            let mut paired = key.clone();
            paired.sine = !key.sine;
            let coefficient = work.mul(
                &key.frequency,
                &BigRational::from(if key.sine { 1 } else { -1 }),
            )?;
            add_coefficient(&mut residual, paired, &coefficient, work)?;
        }
        for (column, basis) in candidate.basis.iter().enumerate() {
            let negative = work.sub(&BigRational::zero(), &row[n + column])?;
            add_coefficient(&mut residual, basis.clone(), &negative, work)?;
        }
        if residual.values().any(|value| !value.is_zero()) {
            return Err(GeneralOdeSystemError::Verification(
                "forcing derivative recurrence",
            ));
        }
    }
    // These exact values follow from the defining entire basis at zero.
    work.charge(n + count, 128 * (n + count))?;
    let mut initial = vec![Expr::from_i64(0); n];
    initial.extend(
        candidate
            .basis
            .iter()
            .map(|key| Expr::from_i64(i64::from(key.power == 0 && !key.sine))),
    );
    work.meter.checkpoint()?;
    Ok(initial)
}

fn expression_nodes(expression: &Expr, work: &mut Work<'_, impl BudgetMeter>) -> Result<usize> {
    work.charge(1, 64)?;
    let mut pending = vec![(expression, 0usize)];
    let mut nodes = 0usize;
    while let Some((node, depth)) = pending.pop() {
        work.charge(1, 64)?;
        nodes += 1;
        if nodes > 16_384 || depth > 96 {
            return Err(GeneralOdeSystemError::ResourceLimit(
                "forced solution expression shape",
            ));
        }
        match node {
            Expr::Add(values) | Expr::Mul(values) | Expr::Function(_, values) => {
                if nodes + pending.len() + values.len() > 16_384 {
                    return Err(GeneralOdeSystemError::ResourceLimit(
                        "forced solution expression fanout",
                    ));
                }
                pending.extend(values.iter().map(|value| (value, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            _ => {}
        }
    }
    Ok(nodes)
}

fn compose(
    baseline: &VerifiedGeneralOdeSystem,
    driven: &VerifiedGeneralOdeSystem,
    seed: &[Expr],
    point: &BigRational,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<Vec<Expr>> {
    let n = baseline.coefficients.len();
    let mut offsets = Vec::with_capacity(n);
    let mut offset_nodes = Vec::with_capacity(n);
    for row in 0..n {
        let offset = if point.is_zero() {
            Expr::from_i64(0)
        } else {
            let coordinate = rational_expr(work.copy(point)?);
            let mut terms = Vec::new();
            for (column, value) in seed.iter().enumerate() {
                if !value.is_zero() {
                    terms.push(
                        render(
                            &driven.certificate.fundamental[row][column],
                            &coordinate,
                            0,
                            work,
                        )?
                        .0,
                    );
                }
            }
            sum(terms)
        };
        let nodes = expression_nodes(&offset, work)?;
        work.charge(nodes, nodes * 1024)?;
        offsets.push(offset);
        offset_nodes.push(nodes);
    }
    let mut solutions = Vec::with_capacity(n);
    for row in 0..n {
        let mut nodes = 4
            + expression_nodes(&baseline.solutions[row], work)?
            + expression_nodes(&driven.solutions[row], work)?;
        for (column, offset) in offsets.iter().enumerate() {
            if !offset.is_zero() {
                nodes += 4
                    + expression_nodes(&baseline.fundamental[row][column], work)?
                    + offset_nodes[column];
            }
        }
        if nodes > 16_384 {
            return Err(GeneralOdeSystemError::ResourceLimit(
                "assembled forced solution nodes",
            ));
        }
        // Every cloned expression is admitted and paid for before assembly.
        let node_bytes = 16_384
            + 4 * baseline.time.name.len()
            + baseline
                .initial_values
                .iter()
                .filter_map(|value| match value {
                    Expr::Sym(symbol) => Some(symbol.name.len()),
                    _ => None,
                })
                .sum::<usize>();
        work.charge(nodes, nodes * node_bytes)?;
        let mut terms = vec![
            baseline.solutions[row].clone(),
            driven.solutions[row].clone(),
        ];
        for (column, offset) in offsets.iter().enumerate() {
            if !offset.is_zero() {
                terms.push(product(vec![
                    Expr::from_i64(-1),
                    baseline.fundamental[row][column].clone(),
                    offset.clone(),
                ]));
            }
        }
        solutions.push(sum(terms));
    }
    work.meter.checkpoint()?;
    Ok(solutions)
}

#[allow(clippy::too_many_arguments)]
fn verify_source(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    polynomial: &[Vec<BigRational>],
    modes: &[ElementaryForcing],
    initial: &[Expr],
    candidate: &ForcedOdeSystemCertificate,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<VerifiedForcedOdeSystem> {
    let n = admit_source(matrix, time, point, polynomial, initial, work)?;
    admit_modes(modes, n, work)?;
    let seeds = check_realization(matrix, modes, &candidate.realization, work)?;
    let baseline = verify(
        matrix,
        time,
        point,
        polynomial,
        initial,
        &candidate.baseline,
        work,
    )?;
    work.charge(2 * seeds.len(), 256 * seeds.len())?;
    let zero_forcing = vec![vec![BigRational::zero()]; seeds.len()];
    let driven = verify(
        &candidate.realization.coefficients,
        time,
        &BigRational::zero(),
        &zero_forcing,
        &seeds,
        &candidate.driven,
        work,
    )?;
    let solutions = compose(&baseline, &driven, &seeds, point, work)?;
    admit_certificate(&candidate.baseline, n, work)?;
    admit_certificate(&candidate.driven, seeds.len(), work)?;
    // Source, candidate and output copies were admitted under the same ledger.
    work.meter.checkpoint()?;
    let result = VerifiedForcedOdeSystem {
        coefficients: matrix.to_vec(),
        polynomial: polynomial.to_vec(),
        forcing: modes.to_vec(),
        time: time.clone(),
        initial_point: point.clone(),
        initial_values: initial.to_vec(),
        certificate: candidate.clone(),
        fundamental: baseline.fundamental,
        solutions,
    };
    work.meter.checkpoint()?;
    Ok(result)
}

/// Independently replay the exact original forcing, ODE and initial-value identities.
#[allow(clippy::too_many_arguments)]
pub fn verify_forced_ode_system(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    polynomial: &[Vec<BigRational>],
    modes: &[ElementaryForcing],
    initial: &[Expr],
    candidate: &ForcedOdeSystemCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedForcedOdeSystem> {
    let mut capped = Capped {
        inner: verifier,
        used: [0; 3],
    };
    verify_source(
        matrix,
        time,
        point,
        polynomial,
        modes,
        initial,
        candidate,
        &mut Work { meter: &mut capped },
    )
}

/// Generate a complete family with independent coefficient and source verification.
#[allow(clippy::too_many_arguments)]
pub fn metered_forced_ode_system(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    polynomial: &[Vec<BigRational>],
    modes: &[ElementaryForcing],
    initial: &[Expr],
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedForcedOdeSystem> {
    let mut generation = Capped {
        inner: generator,
        used: [0; 3],
    };
    let mut verification = Capped {
        inner: verifier,
        used: [0; 3],
    };
    let mut generate_work = Work {
        meter: &mut generation,
    };
    let mut verify_work = Work {
        meter: &mut verification,
    };
    let n = admit_source(matrix, time, point, polynomial, initial, &mut generate_work)?;
    admit_modes(modes, n, &mut generate_work)?;
    let realization = propose_realization(matrix, modes, &mut generate_work)?;
    let seeds = check_realization(matrix, modes, &realization, &mut verify_work)?;
    let baseline = generate(
        matrix,
        time,
        point,
        polynomial,
        initial,
        &mut generate_work,
        &mut verify_work,
    )?;
    generate_work.charge(2 * seeds.len(), 256 * seeds.len())?;
    let zero_forcing = vec![vec![BigRational::zero()]; seeds.len()];
    admit_source(
        &realization.coefficients,
        time,
        &BigRational::zero(),
        &zero_forcing,
        &seeds,
        &mut generate_work,
    )?;
    let driven = generate(
        &realization.coefficients,
        time,
        &BigRational::zero(),
        &zero_forcing,
        &seeds,
        &mut generate_work,
        &mut verify_work,
    )?;
    let candidate = ForcedOdeSystemCertificate {
        realization,
        baseline,
        driven,
    };
    let checked = verify_source(
        matrix,
        time,
        point,
        polynomial,
        modes,
        initial,
        &candidate,
        &mut verify_work,
    )?;
    generate_work.meter.checkpoint()?;
    verify_work.meter.checkpoint()?;
    Ok(checked)
}

pub fn forced_ode_system(
    matrix: &[Vec<BigRational>],
    time: &Symbol,
    point: &BigRational,
    polynomial: &[Vec<BigRational>],
    modes: &[ElementaryForcing],
    initial: &[Expr],
) -> Result<VerifiedForcedOdeSystem> {
    metered_forced_ode_system(
        matrix,
        time,
        point,
        polynomial,
        modes,
        initial,
        &mut Unbounded,
        &mut Unbounded,
    )
}
