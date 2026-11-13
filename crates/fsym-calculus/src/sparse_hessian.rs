//! Source-bound scalar objectives, checked sparse Hessians and Hessian products.
//!
//! Both differentiation stages replay the independent sparse-Jacobian checker,
//! including all omitted zeros and compiled instruction provenance. These are
//! formal derivatives on their common smooth domain, with approximate binary64
//! evaluation. They do not prove convexity, interval accuracy or global C2
//! regularity. Compilation, independent admission and numerical evaluation can
//! consume caller-owned cumulative meters. Generation and admission accept
//! separate meters so a caller can protect verifier resources. Logical work
//! and storage charges are not allocator RSS or cancellation-latency bounds;
//! bounded synchronous derivative/proof kernels retain their documented
//! safe-point boundaries.

#![forbid(unsafe_code)]

use crate::compile::{
    CompileError, CompiledExpr, CompiledOp, EvalError, EvaluationMemo, MAX_COMPILE_OPS,
    SharedEvaluationPlan, charge_compilation_work, charge_evaluation_allocation,
    charge_evaluation_steps,
};
use crate::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, SparseJacobianEntry, charge_derivative_kernel,
    charge_expression_copy, compiled_expression_matches_with_meter,
    verify_sparse_jacobian_certificate_with_meter,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::{Expr, Symbol};
use fsym_proof_kernel::{Claim, DerivationTree, ProofRule, StepId, derivation_verification_units};
use std::collections::HashMap;
use thiserror::Error;

pub const MAX_OBJECTIVE_VARIABLES: usize = 64;
pub const MAX_OBJECTIVE_SOURCE_NODES: usize = 256;
pub const MAX_OBJECTIVE_SOURCE_DEPTH: usize = 16;
pub const MAX_OBJECTIVE_DERIVATIVE_DEPTH: usize = 32;
pub const MAX_OBJECTIVE_PAYLOAD_BITS: u64 = 4096;
pub const MAX_OBJECTIVE_TOTAL_NODES: usize = 262_144;
pub const MAX_OBJECTIVE_DIFFERENTIATION_UNITS: usize = 20_000_000;
const MAX_NAME_BYTES: usize = 256;

/// Public, untrusted proposal. Importing it always repeats both checkers.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveCandidate {
    pub compiled_objective: CompiledExpr,
    pub gradient: SparseJacobian,
    pub hessian: SparseJacobian,
}

/// Counts describe symbolic acceptance, never numerical error guarantees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveReceipt {
    pub variable_count: usize,
    pub gradient_proofs_verified: usize,
    pub hessian_proofs_verified: usize,
    pub zero_entries_checked: usize,
    pub hessian_nonzeros: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveEvaluation {
    pub value: f64,
    pub gradient: Vec<f64>,
    /// Values in the ordered sparse pattern, not a dense matrix.
    pub hessian: Vec<f64>,
}

/// An accepted artifact has no unchecked constructor or deserialization path.
#[derive(Debug, Clone)]
pub struct VerifiedObjective {
    candidate: ObjectiveCandidate,
    compiled_gradient: Vec<CompiledExpr>,
    receipt: ObjectiveReceipt,
    shared_evaluation: SharedEvaluationPlan,
}

#[derive(Debug, Error)]
pub enum ObjectiveCompilationError {
    #[error("objective compilation failed: {0}")]
    Compilation(#[from] CompileError),
    #[error("objective derivative verification failed: {0}")]
    Verification(#[from] JacobianVerificationError),
    #[error("objective certificate does not bind the supplied {0}")]
    SourceMismatch(&'static str),
    #[error("objective {resource} exceeds the admission limit {limit}")]
    ResourceLimit {
        resource: &'static str,
        limit: usize,
    },
    #[error("objective differentiation proof must be one bounded definitional equality")]
    InvalidProofEnvelope,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ObjectiveEvaluationError {
    #[error("objective evaluation failed: {0}")]
    Evaluation(#[from] EvalError),
    #[error("objective {lane} length must be {expected}, got {actual}")]
    Dimension {
        lane: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("objective evaluation requires finite inputs and finite arithmetic")]
    NonFinite,
}

fn resource(resource: &'static str, limit: usize) -> ObjectiveCompilationError {
    ObjectiveCompilationError::ResourceLimit { resource, limit }
}

fn charge_buffer<T, M: BudgetMeter + ?Sized>(
    count: usize,
    meter: &mut M,
) -> Result<(), ObjectiveCompilationError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CompileError::DimensionOverflow)?;
    charge_compilation_work(meter, count, bytes, usize::from(count != 0))?;
    Ok(())
}

/// Poll on both success and refusal, including a checker or synchronous kernel
/// that observed cancellation only after its bounded work completed.
fn finish_compilation<T, M: BudgetMeter + ?Sized>(
    result: Result<T, ObjectiveCompilationError>,
    meter: &mut M,
) -> Result<T, ObjectiveCompilationError> {
    charge_compilation_work(meter, 0, 0, 0)?;
    result
}

/// Iterative preflight precedes all recursive clone, equality and hash work.
fn inspect_expression<M: BudgetMeter + ?Sized>(
    expression: &Expr,
    depth_limit: usize,
    node_limit: usize,
    total: &mut usize,
    meter: &mut M,
) -> Result<(), ObjectiveCompilationError> {
    charge_buffer::<(&Expr, usize), _>(1, meter)?;
    let mut pending = Vec::new();
    pending
        .try_reserve_exact(1)
        .map_err(|_| CompileError::AllocationFailure)?;
    pending.push((expression, 0usize));
    let mut nodes = 0usize;
    while let Some((expression, depth)) = pending.pop() {
        charge_compilation_work(meter, 1, 0, 0)?;
        if depth > depth_limit {
            return Err(resource("expression depth", depth_limit));
        }
        nodes = nodes.saturating_add(1);
        *total = total.saturating_add(1);
        if nodes > node_limit || *total > MAX_OBJECTIVE_TOTAL_NODES {
            return Err(resource(
                "expression nodes",
                node_limit.min(MAX_OBJECTIVE_TOTAL_NODES),
            ));
        }
        match expression {
            Expr::Integer(value) => {
                if value.bits() > MAX_OBJECTIVE_PAYLOAD_BITS {
                    return Err(resource("integer bits", MAX_OBJECTIVE_PAYLOAD_BITS as usize));
                }
                charge_compilation_work(meter, value.bits().div_ceil(64) as usize + 1, 0, 0)?;
            }
            Expr::Rational(value) => {
                if value.numer().bits().max(value.denom().bits()) > MAX_OBJECTIVE_PAYLOAD_BITS {
                    return Err(resource("rational bits", MAX_OBJECTIVE_PAYLOAD_BITS as usize));
                }
                let limbs = value.numer().bits().div_ceil(64) + value.denom().bits().div_ceil(64);
                charge_compilation_work(meter, limbs as usize + 1, 0, 0)?;
            }
            Expr::Sym(symbol) => {
                if symbol.name.len() > MAX_NAME_BYTES {
                    return Err(resource("symbol bytes", MAX_NAME_BYTES));
                }
                charge_compilation_work(meter, symbol.name.len() + 1, 0, 0)?;
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                if let Expr::Function(name, _) = expression {
                    if name.len() > MAX_NAME_BYTES {
                        return Err(resource("function bytes", MAX_NAME_BYTES));
                    }
                    charge_compilation_work(meter, name.len() + 1, 0, 0)?;
                }
                if args.len() > node_limit.saturating_sub(nodes)
                    || pending.len().saturating_add(args.len()) > node_limit
                {
                    return Err(resource("expression nodes", node_limit));
                }
                charge_buffer::<(&Expr, usize), _>(args.len(), meter)?;
                pending
                    .try_reserve(args.len())
                    .map_err(|_| CompileError::AllocationFailure)?;
                pending.extend(args.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                charge_buffer::<(&Expr, usize), _>(2, meter)?;
                pending
                    .try_reserve(2)
                    .map_err(|_| CompileError::AllocationFailure)?;
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            Expr::Const(_) => {}
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Growth {
    nodes: usize,
    derivative: usize,
    work: usize,
}

/// Conservative expression-growth admission score. This is intentionally
/// separate from mathematical acceptance and does not replace runtime budgets.
fn growth<M: BudgetMeter + ?Sized>(
    expression: &Expr,
    meter: &mut M,
) -> Result<Growth, ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    Ok(match expression {
        Expr::Add(args) | Expr::Mul(args) => {
            let mut nodes = 0usize;
            let mut derivative = 0usize;
            let mut work = 0usize;
            for argument in args {
                let child = growth(argument, meter)?;
                nodes = nodes.saturating_add(child.nodes);
                derivative = derivative.saturating_add(child.derivative);
                work = work.saturating_add(child.work);
            }
            let derivative = if matches!(expression, Expr::Mul(_)) {
                1usize
                    .saturating_add(args.len())
                    .saturating_add(args.len().saturating_sub(1).saturating_mul(nodes))
                    .saturating_add(derivative)
            } else {
                1usize.saturating_add(derivative)
            };
            Growth {
                nodes: nodes.saturating_add(1),
                derivative,
                work: work.saturating_add(derivative.saturating_mul(nodes.saturating_add(1))),
            }
        }
        Expr::Pow(base, exponent) => {
            let base = growth(base, meter)?;
            let exponent_growth = growth(exponent, meter)?;
            let derivative = if matches!(exponent.as_ref(), Expr::Integer(_) | Expr::Rational(_)) {
                4usize
                    .saturating_add(base.nodes)
                    .saturating_add(base.derivative)
            } else {
                20usize
                    .saturating_add(base.nodes.saturating_mul(3))
                    .saturating_add(exponent_growth.nodes.saturating_mul(2))
                    .saturating_add(base.derivative)
                    .saturating_add(exponent_growth.derivative)
            };
            let nodes = 1usize
                .saturating_add(base.nodes)
                .saturating_add(exponent_growth.nodes);
            Growth {
                nodes,
                derivative,
                work: base
                    .work
                    .saturating_add(exponent_growth.work)
                    .saturating_add(derivative.saturating_mul(nodes)),
            }
        }
        Expr::Function(_, args) => {
            let mut nodes = 1usize;
            let mut derivative = 10usize;
            let mut work = 0usize;
            for argument in args {
                let child = growth(argument, meter)?;
                nodes = nodes.saturating_add(child.nodes);
                derivative = derivative
                    .saturating_add(child.nodes.saturating_mul(4))
                    .saturating_add(child.derivative.saturating_mul(2));
                work = work.saturating_add(child.work);
            }
            Growth {
                nodes,
                derivative,
                work: work.saturating_add(derivative.saturating_mul(nodes)),
            }
        }
        _ => Growth {
            nodes: 1,
            derivative: 1,
            work: 1,
        },
    })
}

fn stage_envelope<M: BudgetMeter + ?Sized>(
    sources: &[Expr],
    variables: usize,
    total: &mut usize,
    meter: &mut M,
) -> Result<(), ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    let mut units = 0usize;
    for source in sources {
        inspect_expression(
            source,
            MAX_OBJECTIVE_DERIVATIVE_DEPTH,
            MAX_COMPILE_OPS,
            total,
            meter,
        )?;
        let estimate = growth(source, meter)?;
        if estimate.derivative > MAX_COMPILE_OPS {
            return Err(resource("predicted derivative nodes", MAX_COMPILE_OPS));
        }
        units = units.saturating_add(variables.saturating_mul(estimate.work));
        if units > MAX_OBJECTIVE_DIFFERENTIATION_UNITS {
            return Err(resource(
                "differentiation work",
                MAX_OBJECTIVE_DIFFERENTIATION_UNITS,
            ));
        }
    }
    Ok(())
}

fn source_envelope<M: BudgetMeter + ?Sized>(
    source: &Expr,
    variables: &[Symbol],
    total: &mut usize,
    meter: &mut M,
) -> Result<HashMap<Symbol, usize>, ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if variables.len() > MAX_OBJECTIVE_VARIABLES {
        return Err(resource("variable count", MAX_OBJECTIVE_VARIABLES));
    }
    for variable in variables {
        charge_compilation_work(meter, 1, 0, 0)?;
        if variable.name.len() > MAX_NAME_BYTES {
            return Err(resource("symbol bytes", MAX_NAME_BYTES));
        }
    }
    inspect_expression(
        source,
        MAX_OBJECTIVE_SOURCE_DEPTH,
        MAX_OBJECTIVE_SOURCE_NODES,
        total,
        meter,
    )?;
    charge_buffer::<(Symbol, usize), _>(variables.len(), meter)?;
    let mut mapping = HashMap::new();
    mapping
        .try_reserve(variables.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    for (index, variable) in variables.iter().enumerate() {
        charge_compilation_work(meter, variable.name.len() + 1, variable.name.len(), 1)?;
        if mapping.insert(variable.clone(), index).is_some() {
            charge_compilation_work(meter, variable.name.len() + 1, variable.name.len(), 1)?;
            return Err(CompileError::DuplicateVariable(variable.clone()).into());
        }
    }
    Ok(mapping)
}

fn proof_envelope<M: BudgetMeter + ?Sized>(
    proof: &DerivationTree,
    total: &mut usize,
    meter: &mut M,
) -> Result<(), ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if proof.steps.len() != 1 || proof.root != StepId(1) || proof.steps[0].id != StepId(1) {
        return Err(ObjectiveCompilationError::InvalidProofEnvelope);
    }
    let step = &proof.steps[0];
    let Claim::Equality {
        lhs: claim_lhs,
        rhs: claim_rhs,
    } = &step.claim
    else {
        return Err(ObjectiveCompilationError::InvalidProofEnvelope);
    };
    let ProofRule::DefinitionalReduction {
        lhs,
        rhs,
        rule_name,
    } = &step.rule
    else {
        return Err(ObjectiveCompilationError::InvalidProofEnvelope);
    };
    if rule_name.len() > MAX_NAME_BYTES {
        return Err(resource("rule name bytes", MAX_NAME_BYTES));
    }
    let before = *total;
    for expression in [claim_lhs, claim_rhs, lhs, rhs] {
        inspect_expression(
            expression,
            MAX_OBJECTIVE_DERIVATIVE_DEPTH + 1,
            MAX_COMPILE_OPS + 2,
            total,
            meter,
        )?;
    }
    // The one-step proof's bounded metadata and four already-inspected trees
    // are traversed again by the independent proof-envelope estimator.
    charge_compilation_work(meter, *total - before + rule_name.len() + 1, 0, 0)?;
    derivation_verification_units(proof)
        .map_err(|_| ObjectiveCompilationError::InvalidProofEnvelope)?;
    charge_compilation_work(meter, 1, 0, 0)?;
    Ok(())
}

fn jacobian_envelope<M: BudgetMeter + ?Sized>(
    system: &SparseJacobian,
    rows: usize,
    columns: usize,
    total: &mut usize,
    meter: &mut M,
) -> Result<(), ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if system.residuals.len() != rows
        || system.vars.len() != columns
        || system.entries.len() > rows.saturating_mul(columns)
        || system.pattern.nonzeros.len() > rows.saturating_mul(columns)
        || system.column_colors.len() != columns
    {
        return Err(ObjectiveCompilationError::SourceMismatch(
            "derivative dimensions",
        ));
    }
    for variable in &system.vars {
        charge_compilation_work(meter, 1, 0, 0)?;
        if variable.name.len() > MAX_NAME_BYTES {
            return Err(resource("symbol bytes", MAX_NAME_BYTES));
        }
    }
    for expression in &system.residuals {
        inspect_expression(
            expression,
            MAX_OBJECTIVE_DERIVATIVE_DEPTH,
            MAX_COMPILE_OPS,
            total,
            meter,
        )?;
    }
    for entry in &system.entries {
        inspect_expression(
            &entry.symbolic_deriv,
            MAX_OBJECTIVE_DERIVATIVE_DEPTH,
            MAX_COMPILE_OPS,
            total,
            meter,
        )?;
        proof_envelope(&entry.derivation, total, meter)?;
    }
    system.validate_structure_with_meter(meter)?;
    Ok(())
}

/// Only a proposal: the independent checker recomputes every omitted entry.
/// Retained entries keep their original, unsimplified proof right-hand side.
fn propose_zero_omissions<M: BudgetMeter + ?Sized>(
    system: &mut SparseJacobian,
    meter: &mut M,
) -> Result<(), ObjectiveCompilationError> {
    charge_buffer::<SparseJacobianEntry, _>(system.entries.len(), meter)?;
    let mut retained = Vec::new();
    retained
        .try_reserve_exact(system.entries.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    charge_buffer::<(usize, usize), _>(system.entries.len(), meter)?;
    let mut nonzeros = Vec::new();
    nonzeros
        .try_reserve_exact(system.entries.len())
        .map_err(|_| CompileError::AllocationFailure)?;
    for entry in std::mem::take(&mut system.entries) {
        charge_derivative_kernel(&system.residuals[entry.row], meter)?;
        let derivative = crate::diff(&system.residuals[entry.row], &system.vars[entry.col]);
        charge_compilation_work(meter, 1, 0, 0)?;
        if !derivative.is_zero() {
            nonzeros.push((entry.row, entry.col));
            retained.push(entry);
        }
    }
    charge_compilation_work(meter, 1, 0, 0)?;
    system.entries = retained;
    system.pattern.nonzeros = nonzeros;
    system.pattern.nnz = system.entries.len();
    // Deleting conflict edges preserves the existing valid column coloring.
    Ok(())
}

fn gradient_sources<M: BudgetMeter + ?Sized>(
    gradient: &SparseJacobian,
    count: usize,
    meter: &mut M,
) -> Result<Vec<Expr>, ObjectiveCompilationError> {
    charge_buffer::<Expr, _>(count, meter)?;
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(count)
        .map_err(|_| CompileError::AllocationFailure)?;
    for _ in 0..count {
        charge_compilation_work(meter, 1, 0, 0)?;
        sources.push(Expr::from_i64(0));
    }
    for entry in &gradient.entries {
        charge_expression_copy(&entry.symbolic_deriv, meter)?;
        sources[entry.col] = entry.symbolic_deriv.clone();
        charge_compilation_work(meter, 1, 0, 0)?;
    }
    Ok(sources)
}

/// Exact typed comparisons after both operands pass their structural envelopes.
/// Every visited node and scalar/name payload is charged; no printing,
/// simplification or numeric sampling can establish a source binding.
fn expressions_match<M: BudgetMeter + ?Sized>(
    left: &Expr,
    right: &Expr,
    meter: &mut M,
) -> Result<bool, ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    Ok(match (left, right) {
        (Expr::Sym(left), Expr::Sym(right)) => {
            charge_compilation_work(meter, left.name.len() + right.name.len() + 1, 0, 0)?;
            left == right
        }
        (Expr::Integer(left), Expr::Integer(right)) => {
            let units = left.bits().div_ceil(64) + right.bits().div_ceil(64) + 1;
            charge_compilation_work(meter, units as usize, 0, 0)?;
            left == right
        }
        (Expr::Rational(left), Expr::Rational(right)) => {
            let units = left.numer().bits().div_ceil(64)
                + left.denom().bits().div_ceil(64)
                + right.numer().bits().div_ceil(64)
                + right.denom().bits().div_ceil(64)
                + 1;
            charge_compilation_work(meter, units as usize, 0, 0)?;
            left == right
        }
        (Expr::Const(left), Expr::Const(right)) => left == right,
        (Expr::Pow(left_base, left_exponent), Expr::Pow(right_base, right_exponent)) => {
            expressions_match(left_base, right_base, meter)?
                && expressions_match(left_exponent, right_exponent, meter)?
        }
        (Expr::Add(left), Expr::Add(right)) | (Expr::Mul(left), Expr::Mul(right)) => {
            expression_lists_match(left, right, meter)?
        }
        (Expr::Function(left_name, left), Expr::Function(right_name, right)) => {
            charge_compilation_work(meter, left_name.len() + right_name.len() + 1, 0, 0)?;
            left_name == right_name && expression_lists_match(left, right, meter)?
        }
        _ => false,
    })
}

fn expression_lists_match<M: BudgetMeter + ?Sized>(
    left: &[Expr],
    right: &[Expr],
    meter: &mut M,
) -> Result<bool, ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.iter().zip(right) {
        if !expressions_match(left, right, meter)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn variables_match<M: BudgetMeter + ?Sized>(
    left: &[Symbol],
    right: &[Symbol],
    meter: &mut M,
) -> Result<bool, ObjectiveCompilationError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.iter().zip(right) {
        charge_compilation_work(meter, left.name.len() + right.name.len() + 1, 0, 0)?;
        if left != right {
            return Ok(false);
        }
    }
    Ok(true)
}

impl ObjectiveCandidate {
    pub fn try_build(
        source: &Expr,
        variables: &[Symbol],
    ) -> Result<Self, ObjectiveCompilationError> {
        Self::try_build_with_meter(source, variables, &mut Unbounded)
    }

    /// Generate both derivative stages under one cumulative caller ledger.
    /// The intermediate gradient check is generator work; independent admission
    /// repeats both checks under its own meter before publishing a program.
    pub fn try_build_with_meter<M: BudgetMeter + ?Sized>(
        source: &Expr,
        variables: &[Symbol],
        meter: &mut M,
    ) -> Result<Self, ObjectiveCompilationError> {
        let result = (|| {
            let mut total = 0;
            let mapping = source_envelope(source, variables, &mut total, meter)?;
            // Reject unsupported original nodes before differentiation can erase them.
            let compiled_objective = CompiledExpr::try_compile_with_meter(source, &mapping, meter)?;
            stage_envelope(std::slice::from_ref(source), variables.len(), &mut total, meter)?;
            let mut gradient =
                SparseJacobian::try_build_with_meter(std::slice::from_ref(source), variables, meter)?;
            propose_zero_omissions(&mut gradient, meter)?;
            jacobian_envelope(&gradient, 1, variables.len(), &mut total, meter)?;
            verify_sparse_jacobian_certificate_with_meter(&gradient, &[], meter)?;
            let sources = gradient_sources(&gradient, variables.len(), meter)?;
            stage_envelope(&sources, variables.len(), &mut total, meter)?;
            let mut hessian = SparseJacobian::try_build_with_meter(&sources, variables, meter)?;
            propose_zero_omissions(&mut hessian, meter)?;
            jacobian_envelope(
                &hessian,
                variables.len(),
                variables.len(),
                &mut total,
                meter,
            )?;
            Ok(Self {
                compiled_objective,
                gradient,
                hessian,
            })
        })();
        finish_compilation(result, meter)
    }
}

impl VerifiedObjective {
    pub fn try_compile(
        source: &Expr,
        variables: &[Symbol],
    ) -> Result<Self, ObjectiveCompilationError> {
        Self::try_compile_with_meter(source, variables, &mut Unbounded)
    }

    /// Convenience compilation using one caller meter for generation and
    /// admission. Call the two separate APIs when verification needs a protected
    /// pool; this method never reserves or resets a caller's resources.
    pub fn try_compile_with_meter<M: BudgetMeter + ?Sized>(
        source: &Expr,
        variables: &[Symbol],
        meter: &mut M,
    ) -> Result<Self, ObjectiveCompilationError> {
        let result = (|| {
            let candidate = ObjectiveCandidate::try_build_with_meter(source, variables, meter)?;
            Self::try_from_candidate_with_meter(source, variables, candidate, meter)
        })();
        finish_compilation(result, meter)
    }

    pub fn try_from_candidate(
        source: &Expr,
        variables: &[Symbol],
        candidate: ObjectiveCandidate,
    ) -> Result<Self, ObjectiveCompilationError> {
        Self::try_from_candidate_with_meter(source, variables, candidate, &mut Unbounded)
    }

    /// Independently admit an untrusted proposal with cumulative metering.
    /// Original source, ordered typed axes, every derivative and omitted zero,
    /// both bytecode stages and the reusable execution plan are checked again.
    /// A failed or cancelled admission returns no partially accepted program.
    pub fn try_from_candidate_with_meter<M: BudgetMeter + ?Sized>(
        source: &Expr,
        variables: &[Symbol],
        candidate: ObjectiveCandidate,
        meter: &mut M,
    ) -> Result<Self, ObjectiveCompilationError> {
        let result = (|| {
            let mut total = 0;
            let mapping = source_envelope(source, variables, &mut total, meter)?;
            jacobian_envelope(&candidate.gradient, 1, variables.len(), &mut total, meter)?;
            jacobian_envelope(
                &candidate.hessian,
                variables.len(),
                variables.len(),
                &mut total,
                meter,
            )?;
            if !variables_match(&candidate.gradient.vars, variables, meter)?
                || !variables_match(&candidate.hessian.vars, variables, meter)?
            {
                return Err(ObjectiveCompilationError::SourceMismatch(
                    "typed variable order",
                ));
            }
            if !expressions_match(&candidate.gradient.residuals[0], source, meter)? {
                return Err(ObjectiveCompilationError::SourceMismatch("objective source"));
            }
            if !compiled_expression_matches_with_meter(
                source,
                &mapping,
                &candidate.compiled_objective,
                meter,
            )? {
                return Err(ObjectiveCompilationError::SourceMismatch("compiled objective"));
            }
            stage_envelope(std::slice::from_ref(source), variables.len(), &mut total, meter)?;
            let first =
                verify_sparse_jacobian_certificate_with_meter(&candidate.gradient, &[], meter)?;
            let sources = gradient_sources(&candidate.gradient, variables.len(), meter)?;
            if !expression_lists_match(&candidate.hessian.residuals, &sources, meter)? {
                return Err(ObjectiveCompilationError::SourceMismatch(
                    "checked gradient source",
                ));
            }
            stage_envelope(&sources, variables.len(), &mut total, meter)?;
            let second =
                verify_sparse_jacobian_certificate_with_meter(&candidate.hessian, &[], meter)?;
            // Execute the exact programs admitted by the independent checker,
            // not fresh generator output. Every absent column was proved zero.
            charge_buffer::<CompiledOp, _>(1, meter)?;
            let zero = CompiledExpr::try_from_ops_with_meter(
                vec![CompiledOp::LoadConst(0.0)],
                meter,
            )
            .map_err(CompileError::InvalidBytecode)?;
            charge_buffer::<CompiledExpr, _>(variables.len(), meter)?;
            let mut compiled_gradient = Vec::new();
            compiled_gradient
                .try_reserve_exact(variables.len())
                .map_err(|_| CompileError::AllocationFailure)?;
            for _ in variables {
                charge_buffer::<CompiledOp, _>(zero.ops().len(), meter)?;
                compiled_gradient.push(zero.clone());
            }
            for entry in &candidate.gradient.entries {
                charge_buffer::<CompiledOp, _>(entry.compiled_deriv.ops().len(), meter)?;
                compiled_gradient[entry.col] = entry.compiled_deriv.clone();
            }
            let n = variables.len();
            charge_compilation_work(meter, 1, std::mem::size_of::<ObjectiveReceipt>(), 0)?;
            let receipt = ObjectiveReceipt {
                variable_count: n,
                gradient_proofs_verified: first.proofs_verified,
                hessian_proofs_verified: second.proofs_verified,
                zero_entries_checked: n - first.nnz + n * n - second.nnz,
                hessian_nonzeros: second.nnz,
            };
            // Candidate import rebuilds this derived plan. Proposed memo tables
            // and numerical caches cannot bypass source or proof verification.
            let shared_evaluation = SharedEvaluationPlan::build(
                (0..1 + compiled_gradient.len() + candidate.hessian.entries.len()).map(|index| {
                    if index == 0 {
                        &candidate.compiled_objective
                    } else if index <= compiled_gradient.len() {
                        &compiled_gradient[index - 1]
                    } else {
                        &candidate.hessian.entries[index - 1 - compiled_gradient.len()].compiled_deriv
                    }
                }),
                meter,
            )?;
            Ok(Self {
                candidate,
                compiled_gradient,
                receipt,
                shared_evaluation,
            })
        })();
        finish_compilation(result, meter)
    }

    pub fn objective(&self) -> &Expr {
        &self.candidate.gradient.residuals[0]
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.candidate.gradient.vars
    }
    pub fn gradient(&self) -> &[Expr] {
        &self.candidate.hessian.residuals
    }
    pub fn hessian_pattern(&self) -> &[(usize, usize)] {
        &self.candidate.hessian.pattern.nonzeros
    }
    pub fn receipt(&self) -> &ObjectiveReceipt {
        &self.receipt
    }
    pub fn candidate(&self) -> &ObjectiveCandidate {
        &self.candidate
    }

    /// Reusable pure bytecode spans across the source, gradient and Hessian.
    /// This structural count does not establish a timing or accuracy claim.
    pub fn shared_subexpression_count(&self) -> usize {
        self.shared_evaluation.slot_count()
    }

    fn input_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        values: &[f64],
        lane: &'static str,
        meter: &mut M,
    ) -> Result<(), ObjectiveEvaluationError> {
        charge_evaluation_steps(meter, 1)?;
        if values.len() != self.variables().len() {
            return Err(ObjectiveEvaluationError::Dimension {
                lane,
                expected: self.variables().len(),
                actual: values.len(),
            });
        }
        for value in values {
            charge_evaluation_steps(meter, 1)?;
            if !value.is_finite() {
                return Err(ObjectiveEvaluationError::NonFinite);
            }
        }
        Ok(())
    }

    fn first_order_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        meter: &mut M,
    ) -> Result<(f64, Vec<f64>, EvaluationMemo), ObjectiveEvaluationError> {
        self.input_with_meter(point, "point", meter)?;
        let mut memo = self.shared_evaluation.begin(meter)?;
        let value = self.shared_evaluation.evaluate(
            0,
            &self.candidate.compiled_objective,
            point,
            &mut memo,
            meter,
        )?;
        charge_evaluation_allocation(meter, self.compiled_gradient.len())?;
        let mut gradient = Vec::new();
        gradient
            .try_reserve_exact(self.compiled_gradient.len())
            .map_err(|_| EvalError::AllocationFailure)?;
        for (index, entry) in self.compiled_gradient.iter().enumerate() {
            charge_evaluation_steps(meter, 1)?;
            gradient.push(self.shared_evaluation.evaluate(
                1 + index,
                entry,
                point,
                &mut memo,
                meter,
            )?);
        }
        Ok((value, gradient, memo))
    }

    pub fn evaluate(&self, point: &[f64]) -> Result<ObjectiveEvaluation, ObjectiveEvaluationError> {
        self.evaluate_with_meter(point, &mut Unbounded)
    }

    /// Evaluates the already checked source and both derivative stages using
    /// one cumulative meter. Repeated pure subexpressions share values only
    /// within this call, without hoisting inactive branches or reassociating
    /// arithmetic. Every original domain obligation is retained, and a
    /// cancelled or exhausted evaluation publishes no partial result. Scratch,
    /// memo and result vectors use logical allocation accounting; binary64
    /// values remain approximate.
    pub fn evaluate_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        meter: &mut M,
    ) -> Result<ObjectiveEvaluation, ObjectiveEvaluationError> {
        let (value, gradient, mut memo) = self.first_order_with_meter(point, meter)?;
        charge_evaluation_allocation(meter, self.candidate.hessian.entries.len())?;
        let mut hessian = Vec::new();
        hessian
            .try_reserve_exact(self.candidate.hessian.entries.len())
            .map_err(|_| EvalError::AllocationFailure)?;
        for (index, entry) in self.candidate.hessian.entries.iter().enumerate() {
            charge_evaluation_steps(meter, 1)?;
            hessian.push(self.shared_evaluation.evaluate(
                1 + self.compiled_gradient.len() + index,
                &entry.compiled_deriv,
                point,
                &mut memo,
                meter,
            )?);
        }
        charge_evaluation_steps(meter, 1)?;
        Ok(ObjectiveEvaluation {
            value,
            gradient,
            hessian,
        })
    }

    /// Caller-owned outputs are unchanged on every refusal.
    pub fn try_eval_into(
        &self,
        point: &[f64],
        gradient: &mut [f64],
        hessian: &mut [f64],
    ) -> Result<f64, ObjectiveEvaluationError> {
        self.try_eval_into_with_meter(point, gradient, hessian, &mut Unbounded)
    }

    /// Transactional caller-owned output with the same evaluation accounting.
    /// Both copies are charged and cancellation is checked before either
    /// caller buffer is changed.
    pub fn try_eval_into_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        gradient: &mut [f64],
        hessian: &mut [f64],
        meter: &mut M,
    ) -> Result<f64, ObjectiveEvaluationError> {
        charge_evaluation_steps(meter, 1)?;
        for (lane, expected, actual) in [
            ("gradient output", self.variables().len(), gradient.len()),
            (
                "Hessian output",
                self.hessian_pattern().len(),
                hessian.len(),
            ),
        ] {
            if expected != actual {
                return Err(ObjectiveEvaluationError::Dimension {
                    lane,
                    expected,
                    actual,
                });
            }
        }
        let result = self.evaluate_with_meter(point, meter)?;
        charge_evaluation_steps(
            meter,
            gradient
                .len()
                .checked_add(hessian.len())
                .ok_or(EvalError::DimensionOverflow)?,
        )?;
        charge_evaluation_steps(meter, 0)?;
        gradient.copy_from_slice(&result.gradient);
        hessian.copy_from_slice(&result.hessian);
        Ok(result.value)
    }

    /// Sparse O(n + nnz) multiplication, without materializing a dense Hessian.
    /// Original and first-derivative checks run even when the direction is zero.
    pub fn hessian_vector_product(
        &self,
        point: &[f64],
        direction: &[f64],
    ) -> Result<Vec<f64>, ObjectiveEvaluationError> {
        self.hessian_vector_product_with_meter(point, direction, &mut Unbounded)
    }

    /// Sparse Hessian multiplication using the caller meter for the source,
    /// gradient, each second derivative, each multiply/add, and result storage.
    /// A zero direction cannot skip a source or derivative domain check.
    pub fn hessian_vector_product_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        direction: &[f64],
        meter: &mut M,
    ) -> Result<Vec<f64>, ObjectiveEvaluationError> {
        self.input_with_meter(direction, "direction", meter)?;
        let (_, _, mut memo) = self.first_order_with_meter(point, meter)?;
        charge_evaluation_allocation(meter, self.variables().len())?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(self.variables().len())
            .map_err(|_| EvalError::AllocationFailure)?;
        charge_evaluation_steps(meter, self.variables().len())?;
        result.resize(self.variables().len(), 0.0);
        for (index, entry) in self.candidate.hessian.entries.iter().enumerate() {
            let value = self.shared_evaluation.evaluate(
                1 + self.compiled_gradient.len() + index,
                &entry.compiled_deriv,
                point,
                &mut memo,
                meter,
            )?;
            charge_evaluation_steps(meter, 2)?;
            let contribution = value * direction[entry.col];
            let sum = result[entry.row] + contribution;
            if !contribution.is_finite() || !sum.is_finite() {
                return Err(ObjectiveEvaluationError::NonFinite);
            }
            result[entry.row] = sum;
        }
        charge_evaluation_steps(meter, 1)?;
        Ok(result)
    }

    pub fn try_hessian_vector_product_into(
        &self,
        point: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), ObjectiveEvaluationError> {
        self.try_hessian_vector_product_into_with_meter(point, direction, output, &mut Unbounded)
    }

    /// Publishes the complete sparse product only after all evaluation,
    /// reduction, copy charges and the final cancellation check succeed.
    pub fn try_hessian_vector_product_into_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        direction: &[f64],
        output: &mut [f64],
        meter: &mut M,
    ) -> Result<(), ObjectiveEvaluationError> {
        charge_evaluation_steps(meter, 1)?;
        if output.len() != self.variables().len() {
            return Err(ObjectiveEvaluationError::Dimension {
                lane: "Hessian product output",
                expected: self.variables().len(),
                actual: output.len(),
            });
        }
        let result = self.hessian_vector_product_with_meter(point, direction, meter)?;
        charge_evaluation_steps(meter, output.len())?;
        charge_evaluation_steps(meter, 0)?;
        output.copy_from_slice(&result);
        Ok(())
    }
}
