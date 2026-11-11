//! Source-bound scalar objectives, checked sparse Hessians and Hessian products.
//!
//! Both differentiation stages replay the independent sparse-Jacobian checker,
//! including all omitted zeros and compiled instruction provenance. These are
//! formal derivatives on their common smooth domain, with approximate binary64
//! evaluation. They do not prove convexity, interval accuracy or global C2
//! regularity. Compilation and symbolic acceptance retain admission caps;
//! numerical evaluation can consume a caller's cumulative budget and observe
//! cancellation at operand, derivative, reduction and publication safe points.
//! Evaluation metering does not claim metered compilation or proof replay.

#![forbid(unsafe_code)]

use crate::compile::{
    CompileError, CompiledExpr, CompiledOp, EvalError, EvaluationMemo, MAX_COMPILE_OPS,
    SharedEvaluationPlan, charge_evaluation_allocation, charge_evaluation_steps,
};
use crate::sparse_jacobian::{
    JacobianVerificationError, SparseJacobian, compiled_expression_matches,
    verify_sparse_jacobian_certificate,
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

/// Iterative preflight precedes all recursive clone, equality and hash work.
fn inspect_expression(
    expression: &Expr,
    depth_limit: usize,
    node_limit: usize,
    total: &mut usize,
) -> Result<(), ObjectiveCompilationError> {
    let mut pending = vec![(expression, 0usize)];
    let mut nodes = 0usize;
    while let Some((expression, depth)) = pending.pop() {
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
            Expr::Integer(value) if value.bits() > MAX_OBJECTIVE_PAYLOAD_BITS => {
                return Err(resource(
                    "integer bits",
                    MAX_OBJECTIVE_PAYLOAD_BITS as usize,
                ));
            }
            Expr::Rational(value)
                if value.numer().bits().max(value.denom().bits()) > MAX_OBJECTIVE_PAYLOAD_BITS =>
            {
                return Err(resource(
                    "rational bits",
                    MAX_OBJECTIVE_PAYLOAD_BITS as usize,
                ));
            }
            Expr::Sym(symbol) if symbol.name.len() > MAX_NAME_BYTES => {
                return Err(resource("symbol bytes", MAX_NAME_BYTES));
            }
            Expr::Add(args) | Expr::Mul(args) | Expr::Function(_, args) => {
                if let Expr::Function(name, _) = expression
                    && name.len() > MAX_NAME_BYTES
                {
                    return Err(resource("function bytes", MAX_NAME_BYTES));
                }
                if args.len() > node_limit.saturating_sub(nodes)
                    || pending.len().saturating_add(args.len()) > node_limit
                {
                    return Err(resource("expression nodes", node_limit));
                }
                pending.extend(args.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                pending.push((base, depth + 1));
                pending.push((exponent, depth + 1));
            }
            _ => {}
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
fn growth(expression: &Expr) -> Growth {
    match expression {
        Expr::Add(args) | Expr::Mul(args) => {
            let mut nodes = 0usize;
            let mut derivative = 0usize;
            let mut work = 0usize;
            for argument in args {
                let child = growth(argument);
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
            let base = growth(base);
            let exponent_growth = growth(exponent);
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
                let child = growth(argument);
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
    }
}

fn stage_envelope(
    sources: &[Expr],
    variables: usize,
    total: &mut usize,
) -> Result<(), ObjectiveCompilationError> {
    let mut units = 0usize;
    for source in sources {
        inspect_expression(
            source,
            MAX_OBJECTIVE_DERIVATIVE_DEPTH,
            MAX_COMPILE_OPS,
            total,
        )?;
        let estimate = growth(source);
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

fn source_envelope(
    source: &Expr,
    variables: &[Symbol],
    total: &mut usize,
) -> Result<HashMap<Symbol, usize>, ObjectiveCompilationError> {
    if variables.len() > MAX_OBJECTIVE_VARIABLES {
        return Err(resource("variable count", MAX_OBJECTIVE_VARIABLES));
    }
    for variable in variables {
        if variable.name.len() > MAX_NAME_BYTES {
            return Err(resource("symbol bytes", MAX_NAME_BYTES));
        }
    }
    inspect_expression(
        source,
        MAX_OBJECTIVE_SOURCE_DEPTH,
        MAX_OBJECTIVE_SOURCE_NODES,
        total,
    )?;
    let mut mapping = HashMap::new();
    for (index, variable) in variables.iter().enumerate() {
        if mapping.insert(variable.clone(), index).is_some() {
            return Err(CompileError::DuplicateVariable(variable.clone()).into());
        }
    }
    Ok(mapping)
}

fn proof_envelope(
    proof: &DerivationTree,
    total: &mut usize,
) -> Result<(), ObjectiveCompilationError> {
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
    for expression in [claim_lhs, claim_rhs, lhs, rhs] {
        inspect_expression(
            expression,
            MAX_OBJECTIVE_DERIVATIVE_DEPTH + 1,
            MAX_COMPILE_OPS + 2,
            total,
        )?;
    }
    derivation_verification_units(proof)
        .map_err(|_| ObjectiveCompilationError::InvalidProofEnvelope)?;
    Ok(())
}

fn jacobian_envelope(
    system: &SparseJacobian,
    rows: usize,
    columns: usize,
    total: &mut usize,
) -> Result<(), ObjectiveCompilationError> {
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
        )?;
    }
    for entry in &system.entries {
        inspect_expression(
            &entry.symbolic_deriv,
            MAX_OBJECTIVE_DERIVATIVE_DEPTH,
            MAX_COMPILE_OPS,
            total,
        )?;
        proof_envelope(&entry.derivation, total)?;
    }
    system.validate_structure()?;
    Ok(())
}

/// Only a proposal: the independent checker recomputes every omitted entry.
/// Retained entries keep their original, unsimplified proof right-hand side.
fn propose_zero_omissions(system: &mut SparseJacobian) {
    system.entries.retain(|entry| {
        !crate::diff(&system.residuals[entry.row], &system.vars[entry.col]).is_zero()
    });
    system.pattern.nonzeros = system
        .entries
        .iter()
        .map(|entry| (entry.row, entry.col))
        .collect();
    system.pattern.nnz = system.entries.len();
    // Deleting conflict edges preserves the existing valid column coloring.
}

fn gradient_sources(gradient: &SparseJacobian, count: usize) -> Vec<Expr> {
    let mut sources = vec![Expr::from_i64(0); count];
    for entry in &gradient.entries {
        sources[entry.col] = entry.symbolic_deriv.clone();
    }
    sources
}

impl ObjectiveCandidate {
    pub fn try_build(
        source: &Expr,
        variables: &[Symbol],
    ) -> Result<Self, ObjectiveCompilationError> {
        let mut total = 0;
        let mapping = source_envelope(source, variables, &mut total)?;
        // Reject unsupported original nodes before differentiation can erase them.
        let compiled_objective = CompiledExpr::try_compile(source, &mapping)?;
        stage_envelope(std::slice::from_ref(source), variables.len(), &mut total)?;
        let mut gradient = SparseJacobian::try_build(std::slice::from_ref(source), variables)?;
        propose_zero_omissions(&mut gradient);
        jacobian_envelope(&gradient, 1, variables.len(), &mut total)?;
        verify_sparse_jacobian_certificate(&gradient, &[])?;
        let sources = gradient_sources(&gradient, variables.len());
        stage_envelope(&sources, variables.len(), &mut total)?;
        let mut hessian = SparseJacobian::try_build(&sources, variables)?;
        propose_zero_omissions(&mut hessian);
        jacobian_envelope(&hessian, variables.len(), variables.len(), &mut total)?;
        Ok(Self {
            compiled_objective,
            gradient,
            hessian,
        })
    }
}

impl VerifiedObjective {
    pub fn try_compile(
        source: &Expr,
        variables: &[Symbol],
    ) -> Result<Self, ObjectiveCompilationError> {
        Self::try_from_candidate(
            source,
            variables,
            ObjectiveCandidate::try_build(source, variables)?,
        )
    }

    pub fn try_from_candidate(
        source: &Expr,
        variables: &[Symbol],
        candidate: ObjectiveCandidate,
    ) -> Result<Self, ObjectiveCompilationError> {
        let mut total = 0;
        let mapping = source_envelope(source, variables, &mut total)?;
        jacobian_envelope(&candidate.gradient, 1, variables.len(), &mut total)?;
        jacobian_envelope(
            &candidate.hessian,
            variables.len(),
            variables.len(),
            &mut total,
        )?;
        if candidate.gradient.vars != variables || candidate.hessian.vars != variables {
            return Err(ObjectiveCompilationError::SourceMismatch(
                "typed variable order",
            ));
        }
        if candidate.gradient.residuals[0] != *source {
            return Err(ObjectiveCompilationError::SourceMismatch(
                "objective source",
            ));
        }
        if !compiled_expression_matches(source, &mapping, &candidate.compiled_objective)? {
            return Err(ObjectiveCompilationError::SourceMismatch(
                "compiled objective",
            ));
        }
        stage_envelope(std::slice::from_ref(source), variables.len(), &mut total)?;
        let first = verify_sparse_jacobian_certificate(&candidate.gradient, &[])?;
        let sources = gradient_sources(&candidate.gradient, variables.len());
        if candidate.hessian.residuals != sources {
            return Err(ObjectiveCompilationError::SourceMismatch(
                "checked gradient source",
            ));
        }
        stage_envelope(&sources, variables.len(), &mut total)?;
        let second = verify_sparse_jacobian_certificate(&candidate.hessian, &[])?;
        // Execute the exact programs admitted by the independent checker, not
        // fresh generator output. Every absent column was independently proved
        // zero above, so its complete canonical program is a single zero load.
        let zero = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)])
            .map_err(CompileError::InvalidBytecode)?;
        let mut compiled_gradient = vec![zero; variables.len()];
        for entry in &candidate.gradient.entries {
            compiled_gradient[entry.col] = entry.compiled_deriv.clone();
        }
        let n = variables.len();
        let receipt = ObjectiveReceipt {
            variable_count: n,
            gradient_proofs_verified: first.proofs_verified,
            hessian_proofs_verified: second.proofs_verified,
            zero_entries_checked: n - first.nnz + n * n - second.nnz,
            hessian_nonzeros: second.nnz,
        };
        // Reuse only the exact immutable programs admitted above. The separate
        // bytecode-span checker preserves their instruction order and branch
        // boundaries. Candidate import rebuilds this derived plan; neither a
        // proposed memo table nor a numerical cache can bypass source/proof
        // verification.
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
            &mut Unbounded,
        )?;
        Ok(Self {
            candidate,
            compiled_gradient,
            receipt,
            shared_evaluation,
        })
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
