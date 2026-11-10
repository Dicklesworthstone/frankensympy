//! Fast symbolic-to-numeric compilation for residual systems and symbolically differentiated
//! Jacobians (WS12).

#![forbid(unsafe_code)]

use crate::diff;
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, Constant, Expr, Symbol};
use serde::de::{Error as _, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Maximum recursion depth allowed during expression compilation to avoid stack overflow.
pub const MAX_COMPILE_DEPTH: usize = 128;

/// Maximum number of compiled bytecode operations allowed for a single expression.
pub const MAX_COMPILE_OPS: usize = 8192;

/// Maximum UTF-8 byte length of a symbol or function name in scalar compilation.
///
/// This also bounds an individual standard-library hash/equality operation.
/// Legacy entry points share this admission limit; names are never truncated
/// into a different symbolic identity.
pub const MAX_COMPILE_NAME_BYTES: usize = 4096;

const MAX_COMPILE_DIAGNOSTIC_BYTES: usize = 256;

/// Initial operation-buffer reservation for untrusted compiled-expression input.
///
/// Serde sequence length hints are advisory and attacker-controlled, so even an
/// in-range hint does not justify reserving the entire claimed sequence before
/// observing any elements.
const INITIAL_COMPILED_OP_RESERVE: usize = 16;

/// Every integer whose magnitude uses at most this many bits is represented
/// exactly by an IEEE-754 `f64`. Larger exponent components are refused rather
/// than rounded into a different symbolic power.
const MAX_EXACT_F64_INTEGER_BITS: u64 = 53;

/// Errors emitted during symbolic-to-numeric expression and system compilation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CompileError {
    #[error("Symbolic-to-numeric compilation stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("Variable list contains the duplicate symbol {0}")]
    DuplicateVariable(Symbol),
    #[error("Unmapped variable in expression: {0}")]
    UnmappedVariable(Symbol),
    #[error("Unsupported expression variant for compilation: {0}")]
    UnsupportedExpression(String),
    #[error("Unsupported or non-numeric power exponent: {0}")]
    UnsupportedExponent(String),
    #[error("Unsupported elementary function `{0}` or invalid arity")]
    UnsupportedFunction(String),
    #[error("Numeric conversion error or non-finite literal: {0}")]
    NumericConversion(String),
    #[error("Expression nesting depth {depth} exceeds maximum compile limit {max}")]
    DepthLimitExceeded { depth: usize, max: usize },
    #[error("Compiled bytecode size {ops} exceeds maximum operation limit {max}")]
    OpLimitExceeded { ops: usize, max: usize },
    #[error("Identifier length {bytes} exceeds maximum compile limit {max} bytes")]
    NameLimitExceeded { bytes: usize, max: usize },
    #[error("Residual-system dimensions overflow the platform size type")]
    DimensionOverflow,
    #[error("Allocation failed during symbolic-to-numeric compilation")]
    AllocationFailure,
    #[error("Compiler produced invalid bytecode: {0}")]
    InvalidBytecode(#[source] EvalError),
    #[error("Compiled bytecode does not match the original typed source")]
    SourceProvenanceMismatch,
}

/// Charge an actual compiler/checker work unit before executing it. Buffer
/// charges are cumulative requested storage, not allocator RSS or live bytes.
pub(crate) fn charge_compilation_work<M: BudgetMeter + ?Sized>(
    meter: &mut M,
    steps: usize,
    bytes: usize,
    allocations: usize,
) -> Result<(), CompileError> {
    meter.checkpoint()?;
    let requested = [
        (Dimension::ComputeSteps, steps),
        (Dimension::MemoryBytes, bytes),
        (Dimension::AllocationCount, allocations),
    ];
    let mut charges = [(Dimension::ComputeSteps, 0); 3];
    let mut count = 0;
    for (dimension, amount) in requested {
        if amount != 0 {
            charges[count] = (
                dimension,
                u64::try_from(amount).map_err(|_| CompileError::DimensionOverflow)?,
            );
            count += 1;
        }
    }
    if count != 0 {
        meter.charge_batch(&charges[..count])?;
    }
    meter.checkpoint()?;
    Ok(())
}

pub(crate) fn compilation_finish<T, M: BudgetMeter + ?Sized>(
    result: Result<T, CompileError>,
    meter: &mut M,
) -> Result<T, CompileError> {
    meter.checkpoint()?;
    result
}

pub(crate) fn admit_compilation_name<M: BudgetMeter + ?Sized>(
    name: &str,
    meter: &mut M,
) -> Result<(), CompileError> {
    charge_compilation_work(meter, 1, 0, 0)?;
    if name.len() > MAX_COMPILE_NAME_BYTES {
        return Err(CompileError::NameLimitExceeded {
            bytes: name.len(),
            max: MAX_COMPILE_NAME_BYTES,
        });
    }
    Ok(())
}

/// Error construction must not invoke a recursive printer or decimal bigint
/// conversion. Even unsupported function names have a bounded, charged copy.
pub(crate) fn compilation_diagnostic<M: BudgetMeter + ?Sized>(
    message: &str,
    meter: &mut M,
) -> Result<String, CompileError> {
    let truncated = message.len() > MAX_COMPILE_DIAGNOSTIC_BYTES;
    let mut end = message.len().min(MAX_COMPILE_DIAGNOSTIC_BYTES - 3);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    if !truncated {
        end = message.len();
    }
    let bytes = end + if truncated { 3 } else { 0 };
    charge_compilation_work(meter, bytes, bytes, usize::from(bytes != 0))?;
    let mut result = String::new();
    result
        .try_reserve_exact(bytes)
        .map_err(|_| CompileError::AllocationFailure)?;
    result.push_str(&message[..end]);
    if truncated {
        result.push_str("...");
    }
    compilation_finish(Ok(result), meter)
}

fn compilation_variable<M: BudgetMeter + ?Sized>(
    symbol: &Symbol,
    variables: &HashMap<Symbol, usize>,
    meter: &mut M,
) -> Result<usize, CompileError> {
    admit_compilation_name(&symbol.name, meter)?;
    // Precharge the standard-library hash lookup as one cooperative kernel.
    // Capacity covers sparse/tombstoned tables, and two probe slots per
    // capacity unit cover the table's load-factor envelope. Fixed overhead
    // includes typed-identity/hash tags as well as bounded name bytes.
    // These are logical allowances, not physical instruction or wall-time
    // guarantees. The caller-owned map is never cloned on success.
    let comparison_bytes = symbol
        .name
        .len()
        .checked_add(128)
        .ok_or(CompileError::DimensionOverflow)?;
    let steps = variables
        .capacity()
        .checked_add(1)
        .and_then(|count| count.checked_mul(2))
        .and_then(|count| count.checked_mul(comparison_bytes))
        .ok_or(CompileError::DimensionOverflow)?;
    charge_compilation_work(meter, steps, 0, 0)?;
    if let Some(index) = variables.get(symbol).copied() {
        return compilation_finish(Ok(index), meter);
    }
    let bytes = symbol.name.len();
    charge_compilation_work(meter, bytes, bytes, usize::from(bytes != 0))?;
    let mut name = String::new();
    name.try_reserve_exact(bytes)
        .map_err(|_| CompileError::AllocationFailure)?;
    name.push_str(&symbol.name);
    compilation_finish(
        Err(CompileError::UnmappedVariable(Symbol {
            name,
            identity: symbol.identity,
        })),
        meter,
    )
}

/// Errors emitted during compiled numerical evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EvalError {
    #[error("Compiled numerical evaluation stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("Operand stack underflow during bytecode evaluation")]
    StackUnderflow,
    #[error("Operand stack had {0} remaining elements after evaluation (expected 1)")]
    StackRemaining(usize),
    #[error("Variable index {index} out of bounds for input slice of length {slice_len}")]
    VariableOutOfBounds { index: usize, slice_len: usize },
    #[error("Residual buffer length mismatch: expected {expected}, got {actual}")]
    ResidualBufferMismatch { expected: usize, actual: usize },
    #[error("Jacobian buffer length mismatch: expected {expected}, got {actual}")]
    JacobianBufferMismatch { expected: usize, actual: usize },
    #[error("Compiled {lane} expression count mismatch: expected {expected}, got {actual}")]
    CompiledExpressionCountMismatch {
        lane: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("Compiled bytecode size {ops} exceeds maximum operation limit {max}")]
    OpLimitExceeded { ops: usize, max: usize },
    #[error("Residual-system dimensions overflow the platform size type")]
    DimensionOverflow,
    #[error("Allocation failed during compiled numerical evaluation")]
    AllocationFailure,
    #[error("Compiled `{operation}` operation has invalid zero arity")]
    InvalidReductionArity { operation: &'static str },
    #[error("Invalid sparse Jacobian: {0}")]
    InvalidSparseJacobian(String),
    #[error("Compiled power exponent is not finite")]
    NonFinitePowerExponent,
    #[error("A variable real power requires a strictly positive base")]
    NonPositivePowerBase,
    #[error("Compiled constant is NaN or negative zero, which is not a canonical compiler output")]
    NonCanonicalConstant,
    #[error("Compiled numerical evaluation encountered a non-finite input or intermediate value")]
    NonFiniteValue,
    #[error(
        "Bytecode jump from instruction {instruction} to {target} is not a bounded forward jump"
    )]
    InvalidJump { instruction: usize, target: usize },
    #[error("Bytecode paths join at instruction {instruction} with different stack depths")]
    InconsistentBranchStack { instruction: usize },
    #[error("Bytecode instruction {instruction} is unreachable")]
    UnreachableInstruction { instruction: usize },
    #[error("A compiled condition must evaluate to a canonical boolean (zero or one)")]
    InvalidConditionValue,
    #[error("No Piecewise condition is true at the supplied point")]
    NoMatchingPiecewiseBranch,
}

/// Charge deterministic logical work before executing it, including a safe
/// point for zero-sized operations. The caller owns the ledger across phases.
pub(crate) fn charge_evaluation_steps<M: BudgetMeter + ?Sized>(
    meter: &mut M,
    steps: usize,
) -> Result<(), EvalError> {
    meter.checkpoint()?;
    if steps != 0 {
        let steps = u64::try_from(steps).map_err(|_| EvalError::DimensionOverflow)?;
        meter.charge(Dimension::ComputeSteps, steps)?;
    }
    meter.checkpoint()?;
    Ok(())
}

/// Charge cumulative requested f64 scratch/output storage atomically before
/// allocation. This is logical payload accounting, not allocator RSS tracking;
/// released temporaries do not replenish a computation's cumulative budget.
pub(crate) fn charge_evaluation_allocation<M: BudgetMeter + ?Sized>(
    meter: &mut M,
    elements: usize,
) -> Result<(), EvalError> {
    meter.checkpoint()?;
    if elements != 0 {
        let bytes = elements
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(EvalError::DimensionOverflow)?;
        meter.charge_batch(&[
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, 1),
        ])?;
    }
    meter.checkpoint()?;
    Ok(())
}

/// Fast compiled expression evaluator targeting numerical arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompiledOp {
    LoadConst(f64),
    LoadVar(usize),
    Add(usize),
    Mul(usize),
    Neg,
    Pow(f64),
    /// Binary postfix power: finite positive base, followed by finite exponent.
    /// This real branch remains guarded even when the exponent is integral at
    /// one particular evaluation point.
    PowPositive,
    Sin,
    Cos,
    Tan,
    /// Principal real inverse circular functions. Finite evaluation rejects
    /// asin/acos outside [-1, 1] before any later arithmetic can hide NaN.
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    /// Principal real inverse hyperbolic functions. Finite evaluation rejects
    /// acosh below one and atanh at/outside its poles before later operations.
    Asinh,
    Acosh,
    Atanh,
    Exp,
    Ln,
    Sqrt,
    Abs,
    /// Binary comparisons of the evaluated binary64 operands, producing 0 or 1.
    /// These do not claim exact-real comparison at rounded boundaries.
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    /// Boolean negation. Inputs other than canonical 0 or 1 are refused.
    Not,
    /// Pop a boolean and skip to the absolute target when false. Targets must
    /// be strictly forward and at most the operation count, so loops cannot be
    /// admitted through either construction or deserialization.
    JumpIfFalse(usize),
    Jump(usize),
    /// An explicit undefined branch, never a numeric zero or accepted NaN.
    NoMatchingPiecewiseBranch,
}

/// Bounded forward-only stack program evaluating an expression to an f64 value.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompiledExpr {
    ops: Vec<CompiledOp>,
}

struct BoundedCompiledOps(Vec<CompiledOp>);

struct BoundedCompiledOpsVisitor;

fn try_reserve_compiled_ops<E>(ops: &mut Vec<CompiledOp>, additional: usize) -> Result<(), E>
where
    E: serde::de::Error,
{
    ops.try_reserve_exact(additional)
        .map_err(|_| E::custom("compiled bytecode operation allocation refused"))
}

fn next_compiled_op_capacity(current: usize) -> Option<usize> {
    if current >= MAX_COMPILE_OPS {
        return None;
    }
    Some(
        current
            .checked_mul(2)
            .unwrap_or(MAX_COMPILE_OPS)
            .clamp(1, MAX_COMPILE_OPS),
    )
}

impl<'de> Deserialize<'de> for BoundedCompiledOps {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_seq(BoundedCompiledOpsVisitor)
            .map(Self)
    }
}

impl<'de> Visitor<'de> for BoundedCompiledOpsVisitor {
    type Value = Vec<CompiledOp>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "at most {MAX_COMPILE_OPS} compiled bytecode operations"
        )
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let hint = sequence.size_hint();
        if let Some(hint) = hint
            && hint > MAX_COMPILE_OPS
        {
            return Err(A::Error::invalid_length(hint, &self));
        }

        let mut ops = Vec::new();
        let initial_reserve = hint
            .unwrap_or(0)
            .min(INITIAL_COMPILED_OP_RESERVE)
            .min(MAX_COMPILE_OPS);
        if initial_reserve != 0 {
            try_reserve_compiled_ops::<A::Error>(&mut ops, initial_reserve)?;
        }

        while ops.len() < MAX_COMPILE_OPS {
            let Some(op) = sequence.next_element::<CompiledOp>()? else {
                return Ok(ops);
            };
            if ops.len() == ops.capacity() {
                let target_capacity =
                    next_compiled_op_capacity(ops.capacity()).ok_or_else(|| {
                        A::Error::custom("compiled bytecode operation capacity invariant violated")
                    })?;
                let additional = target_capacity.checked_sub(ops.len()).ok_or_else(|| {
                    A::Error::custom("compiled bytecode operation capacity accounting overflow")
                })?;
                if additional == 0 {
                    return Err(A::Error::custom(
                        "compiled bytecode operation capacity invariant violated",
                    ));
                }
                try_reserve_compiled_ops::<A::Error>(&mut ops, additional)?;
            }
            ops.push(op);
        }

        if sequence.next_element::<IgnoredAny>()?.is_some() {
            return Err(A::Error::invalid_length(
                MAX_COMPILE_OPS.saturating_add(1),
                &self,
            ));
        }
        Ok(ops)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompiledExprWire {
    ops: BoundedCompiledOps,
}

impl<'de> Deserialize<'de> for CompiledExpr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CompiledExprWire::deserialize(deserializer)?;
        Self::try_from_ops(wire.ops.0).map_err(D::Error::custom)
    }
}

impl CompiledExpr {
    /// Builds a compiled expression from externally supplied bytecode.
    ///
    /// This is the trust boundary used by deserialization. It admits only
    /// structurally valid stack programs and numeric parameters admitted by
    /// the compiled format.
    pub fn try_from_ops(ops: Vec<CompiledOp>) -> Result<Self, EvalError> {
        Self::try_from_ops_with_meter(ops, &mut Unbounded)
    }

    /// Admit caller-owned bytecode while charging every validation instruction.
    /// The caller's existing operation vector is not allocated again.
    pub fn try_from_ops_with_meter<M: BudgetMeter + ?Sized>(
        ops: Vec<CompiledOp>,
        meter: &mut M,
    ) -> Result<Self, EvalError> {
        let result = Self::validate_ops_with_meter(&ops, meter).map(|()| Self { ops });
        meter.checkpoint()?;
        result
    }

    /// Returns the validated bytecode operations in evaluation order.
    pub fn ops(&self) -> &[CompiledOp] {
        &self.ops
    }

    fn validate_ops_with_meter<M: BudgetMeter + ?Sized>(
        ops: &[CompiledOp],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        meter.checkpoint()?;
        if ops.len() > MAX_COMPILE_OPS {
            return Err(EvalError::OpLimitExceeded {
                ops: ops.len(),
                max: MAX_COMPILE_OPS,
            });
        }

        let mut stack_depth = 0usize;
        for op in ops {
            charge_evaluation_steps(meter, 1)?;
            if matches!(
                op,
                CompiledOp::JumpIfFalse(_)
                    | CompiledOp::Jump(_)
                    | CompiledOp::NoMatchingPiecewiseBranch
            ) {
                // Straight-line programs retain their allocation-free check.
                // A branching program needs one depth per instruction, not a
                // recursive traversal or exponentially many path executions.
                return Self::validate_control_flow_with_meter(ops, meter);
            }
            stack_depth = Self::stack_depth_after(op, stack_depth)?;
        }

        if stack_depth != 1 {
            return Err(EvalError::StackRemaining(stack_depth));
        }
        meter.checkpoint()?;
        Ok(())
    }

    fn stack_depth_after(op: &CompiledOp, depth: usize) -> Result<usize, EvalError> {
        match op {
            CompiledOp::LoadConst(value) => {
                if value.is_nan() || (*value == 0.0 && value.is_sign_negative()) {
                    return Err(EvalError::NonCanonicalConstant);
                }
                Ok(depth + 1)
            }
            CompiledOp::LoadVar(_) => Ok(depth + 1),
            CompiledOp::Add(count) | CompiledOp::Mul(count) => {
                if *count == 0 {
                    return Err(EvalError::InvalidReductionArity {
                        operation: if matches!(op, CompiledOp::Add(_)) {
                            "Add"
                        } else {
                            "Mul"
                        },
                    });
                }
                if depth < *count {
                    return Err(EvalError::StackUnderflow);
                }
                Ok(depth - *count + 1)
            }
            CompiledOp::Pow(exponent) => {
                if !exponent.is_finite() {
                    return Err(EvalError::NonFinitePowerExponent);
                }
                if depth == 0 {
                    return Err(EvalError::StackUnderflow);
                }
                Ok(depth)
            }
            CompiledOp::PowPositive
            | CompiledOp::Equal
            | CompiledOp::NotEqual
            | CompiledOp::Less
            | CompiledOp::LessEqual
            | CompiledOp::Greater
            | CompiledOp::GreaterEqual => {
                if depth < 2 {
                    return Err(EvalError::StackUnderflow);
                }
                Ok(depth - 1)
            }
            CompiledOp::JumpIfFalse(_) => depth.checked_sub(1).ok_or(EvalError::StackUnderflow),
            CompiledOp::Jump(_) | CompiledOp::NoMatchingPiecewiseBranch => Ok(depth),
            CompiledOp::Neg
            | CompiledOp::Sin
            | CompiledOp::Cos
            | CompiledOp::Tan
            | CompiledOp::Asin
            | CompiledOp::Acos
            | CompiledOp::Atan
            | CompiledOp::Sinh
            | CompiledOp::Cosh
            | CompiledOp::Tanh
            | CompiledOp::Asinh
            | CompiledOp::Acosh
            | CompiledOp::Atanh
            | CompiledOp::Exp
            | CompiledOp::Ln
            | CompiledOp::Sqrt
            | CompiledOp::Abs
            | CompiledOp::Not => {
                if depth == 0 {
                    return Err(EvalError::StackUnderflow);
                }
                Ok(depth)
            }
        }
    }

    fn validate_control_flow_with_meter<M: BudgetMeter + ?Sized>(
        ops: &[CompiledOp],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        let length = ops.len() + 1; // ops is already bounded by MAX_COMPILE_OPS.
        charge_evaluation_steps(meter, length)?;
        meter.charge_batch(&[
            (
                Dimension::MemoryBytes,
                (length * std::mem::size_of::<Option<usize>>()) as u64,
            ),
            (Dimension::AllocationCount, 1),
        ])?;
        let mut depths = Vec::new();
        depths
            .try_reserve_exact(length)
            .map_err(|_| EvalError::AllocationFailure)?;
        depths.resize(length, None);
        depths[0] = Some(0);
        for (instruction, op) in ops.iter().enumerate() {
            charge_evaluation_steps(meter, 4)?;
            let depth =
                depths[instruction].ok_or(EvalError::UnreachableInstruction { instruction })?;
            let next = Self::stack_depth_after(op, depth)?;
            if let CompiledOp::JumpIfFalse(target) | CompiledOp::Jump(target) = op {
                if *target <= instruction || *target > ops.len() {
                    return Err(EvalError::InvalidJump {
                        instruction,
                        target: *target,
                    });
                }
                Self::join_stack_depth(&mut depths, *target, next)?;
            }
            if !matches!(
                op,
                CompiledOp::Jump(_) | CompiledOp::NoMatchingPiecewiseBranch
            ) {
                Self::join_stack_depth(&mut depths, instruction + 1, next)?;
            }
        }
        if depths[ops.len()] != Some(1) {
            return Err(EvalError::StackRemaining(depths[ops.len()].unwrap_or(0)));
        }
        meter.checkpoint()?;
        Ok(())
    }

    fn join_stack_depth(
        depths: &mut [Option<usize>],
        instruction: usize,
        depth: usize,
    ) -> Result<(), EvalError> {
        if let Some(previous) = depths[instruction]
            && previous != depth
        {
            return Err(EvalError::InconsistentBranchStack { instruction });
        }
        depths[instruction] = Some(depth);
        Ok(())
    }

    fn exact_exponent_component<M: BudgetMeter + ?Sized>(
        component: &BigInt,
        meter: &mut M,
    ) -> Result<f64, CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        if component.bits() > MAX_EXACT_F64_INTEGER_BITS {
            return Err(CompileError::UnsupportedExponent(compilation_diagnostic(
                "literal components must be exactly representable within 53 bits",
                meter,
            )?));
        }
        charge_compilation_work(meter, 2, 0, 0)?;
        match component.to_f64().filter(|value| value.is_finite()) {
            Some(value) => compilation_finish(Ok(value), meter),
            None => Err(CompileError::UnsupportedExponent(compilation_diagnostic(
                "literal component cannot be converted to finite binary64",
                meter,
            )?)),
        }
    }

    fn integer_literal<M: BudgetMeter + ?Sized>(
        integer: &BigInt,
        meter: &mut M,
    ) -> Result<f64, CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        let bits = integer.bits();
        // Wider integers cannot round to finite binary64. Refuse before the
        // substrate scans their digits or a diagnostic formats their value.
        if bits <= 1024 {
            charge_compilation_work(meter, bits.div_ceil(32).max(1) as usize, 0, 0)?;
            if let Some(value) = integer.to_f64().filter(|value| value.is_finite()) {
                return compilation_finish(Ok(value), meter);
            }
        }
        Err(CompileError::NumericConversion(compilation_diagnostic(
            "integer literal does not round to finite binary64",
            meter,
        )?))
    }

    /// Compiles a symbolic expression into bytecode given a variable index lookup map.
    /// Fails closed if any unmapped symbol, unsupported function, or complexity limit is reached.
    pub fn try_compile(
        expr: &Expr,
        var_map: &HashMap<Symbol, usize>,
    ) -> Result<Self, CompileError> {
        Self::try_compile_with_meter(expr, var_map, &mut Unbounded)
    }

    /// Compile an immutable source under the caller's cumulative ledger.
    ///
    /// Recursion, instruction emission, actual operation-buffer growth and
    /// validation all consume fuel. Exact rational rounding uses governed
    /// arithmetic. Source/map construction and symbolic differentiation are
    /// outside this operation. Standard-library allocation and a bounded
    /// name lookup are cooperative boundaries, not wall-time guarantees.
    /// The returned program is a candidate; use the checked entry point when
    /// independent original-source admission is required before publication.
    pub fn try_compile_with_meter<M: BudgetMeter + ?Sized>(
        expr: &Expr,
        var_map: &HashMap<Symbol, usize>,
        meter: &mut M,
    ) -> Result<Self, CompileError> {
        meter.checkpoint()?;
        let mut ops = Vec::new();
        let result = Self::compile_recursive(expr, var_map, &mut ops, 0, meter).and_then(|()| {
            Self::try_from_ops_with_meter(ops, meter).map_err(|error| match error {
                EvalError::Meter(error) => CompileError::Meter(error),
                error => CompileError::InvalidBytecode(error),
            })
        });
        compilation_finish(result, meter)
    }

    /// Generate, independently check, and publish using one caller ledger.
    /// No candidate escapes if checking runs out of fuel or is cancelled.
    /// This checks binary64 lowering, not exact-real numerical accuracy or
    /// differentiation proofs, and does not access a protected verifier pool.
    pub fn try_compile_checked_with_meter<M: BudgetMeter + ?Sized>(
        expr: &Expr,
        var_map: &HashMap<Symbol, usize>,
        meter: &mut M,
    ) -> Result<Self, CompileError> {
        let result = Self::try_compile_with_meter(expr, var_map, meter).and_then(|program| {
            if crate::sparse_jacobian::compiled_expression_matches_with_meter(
                expr, var_map, &program, meter,
            )? {
                Ok(program)
            } else {
                Err(CompileError::SourceProvenanceMismatch)
            }
        });
        compilation_finish(result, meter)
    }

    /// Convenience infallible compiler that panics on malformed or unmapped expressions.
    pub fn compile(expr: &Expr, var_map: &HashMap<Symbol, usize>) -> Self {
        Self::try_compile(expr, var_map).expect("expression compilation failed")
    }

    fn compile_recursive<M: BudgetMeter + ?Sized>(
        expr: &Expr,
        var_map: &HashMap<Symbol, usize>,
        ops: &mut Vec<CompiledOp>,
        depth: usize,
        meter: &mut M,
    ) -> Result<(), CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        if depth > MAX_COMPILE_DEPTH {
            return Err(CompileError::DepthLimitExceeded {
                depth,
                max: MAX_COMPILE_DEPTH,
            });
        }
        if ops.len() >= MAX_COMPILE_OPS {
            return Err(CompileError::OpLimitExceeded {
                ops: ops.len().saturating_add(1),
                max: MAX_COMPILE_OPS,
            });
        }

        match expr {
            Expr::Integer(n) => {
                let val = Self::integer_literal(n, meter)?;
                Self::push_op(ops, CompiledOp::LoadConst(val), meter)?;
            }
            Expr::Rational(r) => {
                // Round the exact quotient once. Rounding its components first
                // can change the result or overflow a finite coprime ratio.
                let val = r.metered_to_f64(meter)?.filter(|value| value.is_finite());
                let Some(val) = val else {
                    return Err(CompileError::NumericConversion(compilation_diagnostic(
                        "exact rational quotient does not round to finite binary64",
                        meter,
                    )?));
                };
                // The bytecode format has one canonical zero. This approximate
                // lane permits finite underflow, including negative rationals.
                let val = if val == 0.0 { 0.0 } else { val };
                Self::push_op(ops, CompiledOp::LoadConst(val), meter)?;
            }
            Expr::Const(Constant::Pi) => {
                Self::push_op(ops, CompiledOp::LoadConst(std::f64::consts::PI), meter)?
            }
            Expr::Const(Constant::E) => {
                Self::push_op(ops, CompiledOp::LoadConst(std::f64::consts::E), meter)?
            }
            Expr::Const(Constant::Infinity) => {
                Self::push_op(ops, CompiledOp::LoadConst(f64::INFINITY), meter)?
            }
            Expr::Const(Constant::NegativeInfinity) => {
                Self::push_op(ops, CompiledOp::LoadConst(f64::NEG_INFINITY), meter)?
            }
            Expr::Const(Constant::ComplexInfinity)
            | Expr::Const(Constant::I)
            | Expr::Const(Constant::NaN) => {
                return Err(CompileError::UnsupportedExpression(compilation_diagnostic(
                    "complex infinity, imaginary unit or NaN",
                    meter,
                )?));
            }
            Expr::Sym(s) => {
                let index = compilation_variable(s, var_map, meter)?;
                Self::push_op(ops, CompiledOp::LoadVar(index), meter)?;
            }
            Expr::Add(terms) => {
                if terms.is_empty() {
                    Self::push_op(ops, CompiledOp::LoadConst(0.0), meter)?;
                } else {
                    for t in terms {
                        charge_compilation_work(meter, 1, 0, 0)?;
                        Self::compile_recursive(t, var_map, ops, depth + 1, meter)?;
                    }
                    Self::push_op(ops, CompiledOp::Add(terms.len()), meter)?;
                }
            }
            Expr::Mul(factors) => {
                if factors.is_empty() {
                    Self::push_op(ops, CompiledOp::LoadConst(1.0), meter)?;
                } else {
                    for f in factors {
                        charge_compilation_work(meter, 1, 0, 0)?;
                        Self::compile_recursive(f, var_map, ops, depth + 1, meter)?;
                    }
                    Self::push_op(ops, CompiledOp::Mul(factors.len()), meter)?;
                }
            }
            Expr::Pow(base, exp) => {
                charge_compilation_work(meter, 1, 0, 0)?;
                Self::compile_recursive(base, var_map, ops, depth + 1, meter)?;
                charge_compilation_work(meter, 1, 0, 0)?;
                match exp.as_ref() {
                    Expr::Integer(n) => {
                        let p = Self::exact_exponent_component(n, meter)?;
                        Self::push_op(ops, CompiledOp::Pow(p), meter)?;
                    }
                    Expr::Rational(r) => {
                        let numer = Self::exact_exponent_component(r.numer(), meter)?;
                        let denom = Self::exact_exponent_component(r.denom(), meter)?;
                        charge_compilation_work(meter, 1, 0, 0)?;
                        let exponent = numer / denom;
                        if !exponent.is_finite() {
                            return Err(CompileError::UnsupportedExponent(compilation_diagnostic(
                                "literal exponent quotient is not finite",
                                meter,
                            )?));
                        }
                        Self::push_op(ops, CompiledOp::Pow(exponent), meter)?;
                    }
                    _ => {
                        Self::compile_recursive(exp, var_map, ops, depth + 1, meter)?;
                        Self::push_op(ops, CompiledOp::PowPositive, meter)?;
                    }
                }
            }
            Expr::Function(name, args) => {
                admit_compilation_name(name, meter)?;
                if name == "Piecewise" {
                    return Self::compile_piecewise(args, var_map, ops, depth, meter);
                }
                if args.len() != 1 {
                    return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                        name, meter,
                    )?));
                }
                charge_compilation_work(meter, name.len() + 1, 0, 0)?;
                Self::compile_recursive(&args[0], var_map, ops, depth + 1, meter)?;
                // Share instruction emission across unary functions. A
                // separate fallible call in every arm inflates this recursive
                // frame enough to overflow a normal test-thread stack before
                // MAX_COMPILE_DEPTH can reject a deeply nested source.
                let operation = match name.as_str() {
                    "sin" => CompiledOp::Sin,
                    "cos" => CompiledOp::Cos,
                    "tan" => CompiledOp::Tan,
                    "asin" => CompiledOp::Asin,
                    "acos" => CompiledOp::Acos,
                    "atan" => CompiledOp::Atan,
                    "sinh" => CompiledOp::Sinh,
                    "cosh" => CompiledOp::Cosh,
                    "tanh" => CompiledOp::Tanh,
                    "asinh" => CompiledOp::Asinh,
                    "acosh" => CompiledOp::Acosh,
                    "atanh" => CompiledOp::Atanh,
                    "exp" => CompiledOp::Exp,
                    "ln" | "log" => CompiledOp::Ln,
                    "sqrt" => CompiledOp::Sqrt,
                    "abs" => CompiledOp::Abs,
                    other => {
                        return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                            other, meter,
                        )?));
                    }
                };
                Self::push_op(ops, operation, meter)?;
            }
        }
        Ok(())
    }

    fn patch_slots<M: BudgetMeter + ?Sized>(
        count: usize,
        meter: &mut M,
    ) -> Result<Vec<usize>, CompileError> {
        if count > MAX_COMPILE_OPS {
            return Err(CompileError::OpLimitExceeded {
                ops: count,
                max: MAX_COMPILE_OPS,
            });
        }
        charge_compilation_work(
            meter,
            count,
            count * std::mem::size_of::<usize>(),
            usize::from(count != 0),
        )?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(count)
            .map_err(|_| CompileError::AllocationFailure)?;
        Ok(slots)
    }

    fn compile_piecewise<M: BudgetMeter + ?Sized>(
        branches: &[Expr],
        var_map: &HashMap<Symbol, usize>,
        ops: &mut Vec<CompiledOp>,
        depth: usize,
        meter: &mut M,
    ) -> Result<(), CompileError> {
        if branches.is_empty() {
            return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                "Piecewise requires at least one (value, condition) branch",
                meter,
            )?));
        }
        let mut ends = Self::patch_slots(branches.len(), meter)?;
        for branch in branches {
            charge_compilation_work(meter, 1, 0, 0)?;
            let Expr::Function(tag, pair) = branch else {
                return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                    "Piecewise requires Tuple(value, condition) branches",
                    meter,
                )?));
            };
            if tag != "Tuple" || pair.len() != 2 {
                return Err(CompileError::UnsupportedFunction(compilation_diagnostic(
                    "Piecewise requires Tuple(value, condition) branches",
                    meter,
                )?));
            }
            // Retain source order, including constant conditions. Every source
            // branch must be admitted; evaluation alone skips inactive work.
            Self::compile_condition(&pair[1], var_map, ops, depth + 2, meter)?;
            let condition_jump = ops.len();
            Self::push_op(ops, CompiledOp::JumpIfFalse(0), meter)?;
            Self::compile_recursive(&pair[0], var_map, ops, depth + 2, meter)?;
            ends.push(ops.len());
            Self::push_op(ops, CompiledOp::Jump(0), meter)?;
            charge_compilation_work(meter, 1, 0, 0)?;
            ops[condition_jump] = CompiledOp::JumpIfFalse(ops.len());
        }
        Self::push_op(ops, CompiledOp::NoMatchingPiecewiseBranch, meter)?;
        for jump in ends {
            charge_compilation_work(meter, 1, 0, 0)?;
            ops[jump] = CompiledOp::Jump(ops.len());
        }
        Ok(())
    }

    fn compile_condition<M: BudgetMeter + ?Sized>(
        condition: &Expr,
        var_map: &HashMap<Symbol, usize>,
        ops: &mut Vec<CompiledOp>,
        depth: usize,
        meter: &mut M,
    ) -> Result<(), CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        if depth > MAX_COMPILE_DEPTH {
            return Err(CompileError::DepthLimitExceeded {
                depth,
                max: MAX_COMPILE_DEPTH,
            });
        }
        if let Expr::Sym(symbol) = condition {
            admit_compilation_name(&symbol.name, meter)?;
            // True/False are the legacy Piecewise condition protocol, only
            // within a condition. Identity-bearing names are never booleans.
            if symbol.is_plain() && matches!(symbol.name.as_str(), "True" | "False") {
                return Self::push_op(
                    ops,
                    CompiledOp::LoadConst(if symbol.name == "True" { 1.0 } else { 0.0 }),
                    meter,
                );
            }
        }
        if let Expr::Function(name, args) = condition {
            admit_compilation_name(name, meter)?;
            charge_compilation_work(meter, name.len() + 1, 0, 0)?;
            if name == "Not" && args.len() == 1 {
                Self::compile_condition(&args[0], var_map, ops, depth + 1, meter)?;
                return Self::push_op(ops, CompiledOp::Not, meter);
            }
            if name == "And" || name == "Or" {
                let is_and = name == "And";
                let Some((last, prefix)) = args.split_last() else {
                    return Self::push_op(
                        ops,
                        CompiledOp::LoadConst(if is_and { 1.0 } else { 0.0 }),
                        meter,
                    );
                };
                if prefix.is_empty() {
                    return Self::compile_condition(last, var_map, ops, depth + 1, meter);
                }
                let mut shortcuts = Self::patch_slots(prefix.len(), meter)?;
                for argument in prefix {
                    Self::compile_condition(argument, var_map, ops, depth + 1, meter)?;
                    if !is_and {
                        Self::push_op(ops, CompiledOp::Not, meter)?;
                    }
                    shortcuts.push(ops.len());
                    Self::push_op(ops, CompiledOp::JumpIfFalse(0), meter)?;
                }
                Self::compile_condition(last, var_map, ops, depth + 1, meter)?;
                let end_jump = ops.len();
                Self::push_op(ops, CompiledOp::Jump(0), meter)?;
                let shortcut = ops.len();
                Self::push_op(
                    ops,
                    CompiledOp::LoadConst(if is_and { 0.0 } else { 1.0 }),
                    meter,
                )?;
                for jump in shortcuts {
                    charge_compilation_work(meter, 1, 0, 0)?;
                    ops[jump] = CompiledOp::JumpIfFalse(shortcut);
                }
                charge_compilation_work(meter, 1, 0, 0)?;
                ops[end_jump] = CompiledOp::Jump(ops.len());
                return Ok(());
            }
            if args.len() == 2 {
                let comparison = match name.as_str() {
                    "Equality" => Some(CompiledOp::Equal),
                    "Unequality" => Some(CompiledOp::NotEqual),
                    "StrictLessThan" => Some(CompiledOp::Less),
                    "LessThan" => Some(CompiledOp::LessEqual),
                    "StrictGreaterThan" => Some(CompiledOp::Greater),
                    "GreaterThan" => Some(CompiledOp::GreaterEqual),
                    _ => None,
                };
                if let Some(comparison) = comparison {
                    Self::compile_recursive(&args[0], var_map, ops, depth + 1, meter)?;
                    Self::compile_recursive(&args[1], var_map, ops, depth + 1, meter)?;
                    return Self::push_op(ops, comparison, meter);
                }
            }
        }
        Err(CompileError::UnsupportedFunction(compilation_diagnostic(
            "Piecewise condition requires a boolean, comparison, And, Or or Not",
            meter,
        )?))
    }

    fn push_op<M: BudgetMeter + ?Sized>(
        ops: &mut Vec<CompiledOp>,
        op: CompiledOp,
        meter: &mut M,
    ) -> Result<(), CompileError> {
        charge_compilation_work(meter, 1, 0, 0)?;
        let attempted = ops
            .len()
            .checked_add(1)
            .ok_or(CompileError::OpLimitExceeded {
                ops: usize::MAX,
                max: MAX_COMPILE_OPS,
            })?;
        if attempted > MAX_COMPILE_OPS {
            return Err(CompileError::OpLimitExceeded {
                ops: attempted,
                max: MAX_COMPILE_OPS,
            });
        }
        if ops.len() == ops.capacity() {
            let capacity =
                next_compiled_op_capacity(ops.capacity()).ok_or(CompileError::DimensionOverflow)?;
            let bytes = capacity
                .checked_mul(std::mem::size_of::<CompiledOp>())
                .ok_or(CompileError::DimensionOverflow)?;
            // A growth request may copy all existing instructions. Charge
            // its full requested payload, not merely the capacity difference.
            charge_compilation_work(meter, ops.len(), bytes, 1)?;
            ops.try_reserve_exact(capacity - ops.len())
                .map_err(|_| CompileError::AllocationFailure)?;
        }
        ops.push(op);
        meter.checkpoint()?;
        Ok(())
    }

    /// Evaluates the compiled expression using a verified operand stack with bounds checks.
    pub fn try_eval(&self, vars: &[f64]) -> Result<f64, EvalError> {
        self.try_eval_with_meter(vars, &mut Unbounded)
    }

    /// Evaluate with caller-owned cumulative fuel, allocation accounting and
    /// cancellation. Each instruction and reduction operand is charged before
    /// execution, and cancellation is checked again before returning a value.
    /// Compilation and certificate admission use separate entry points.
    pub fn try_eval_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        vars: &[f64],
        meter: &mut M,
    ) -> Result<f64, EvalError> {
        self.try_eval_with_policy(vars, false, meter)
    }

    /// Evaluate in the finite real binary64 lane, rejecting every non-finite
    /// intermediate before a later operation could conceal a domain failure.
    /// This does not certify rounding error or prevent finite underflow.
    pub fn try_eval_finite(&self, vars: &[f64]) -> Result<f64, EvalError> {
        self.try_eval_finite_with_meter(vars, &mut Unbounded)
    }

    /// Metered finite-real evaluation; finite binary64 results remain
    /// approximate and receive no mathematical evidence from the budget.
    pub fn try_eval_finite_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        vars: &[f64],
        meter: &mut M,
    ) -> Result<f64, EvalError> {
        self.try_eval_with_policy(vars, true, meter)
    }

    fn try_eval_with_policy<M: BudgetMeter + ?Sized>(
        &self,
        vars: &[f64],
        finite_only: bool,
        meter: &mut M,
    ) -> Result<f64, EvalError> {
        meter.checkpoint()?;
        if self.ops.len() > MAX_COMPILE_OPS {
            return Err(EvalError::OpLimitExceeded {
                ops: self.ops.len(),
                max: MAX_COMPILE_OPS,
            });
        }
        if finite_only {
            for value in vars {
                charge_evaluation_steps(meter, 1)?;
                if !value.is_finite() {
                    return Err(EvalError::NonFiniteValue);
                }
            }
        }
        charge_evaluation_allocation(meter, self.ops.len())?;
        let mut stack = Vec::new();
        stack
            .try_reserve_exact(self.ops.len())
            .map_err(|_| EvalError::AllocationFailure)?;
        let mut instruction = 0;
        while instruction < self.ops.len() {
            let op = &self.ops[instruction];
            charge_evaluation_steps(meter, 1)?;
            match op {
                CompiledOp::LoadConst(c) => stack.push(*c),
                CompiledOp::LoadVar(idx) => {
                    if let Some(&v) = vars.get(*idx) {
                        stack.push(v);
                    } else {
                        return Err(EvalError::VariableOutOfBounds {
                            index: *idx,
                            slice_len: vars.len(),
                        });
                    }
                }
                CompiledOp::Add(count) => {
                    let mut sum = 0.0;
                    for _ in 0..*count {
                        charge_evaluation_steps(meter, 1)?;
                        sum += stack.pop().ok_or(EvalError::StackUnderflow)?;
                    }
                    stack.push(sum);
                }
                CompiledOp::Mul(count) => {
                    let mut prod = 1.0;
                    for _ in 0..*count {
                        charge_evaluation_steps(meter, 1)?;
                        prod *= stack.pop().ok_or(EvalError::StackUnderflow)?;
                    }
                    stack.push(prod);
                }
                CompiledOp::Neg => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(-v);
                }
                CompiledOp::Pow(p) => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.powf(*p));
                }
                CompiledOp::PowPositive => {
                    let exponent = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    let base = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    if !base.is_finite() || !exponent.is_finite() {
                        return Err(EvalError::NonFiniteValue);
                    }
                    if base <= 0.0 {
                        return Err(EvalError::NonPositivePowerBase);
                    }
                    stack.push(base.powf(exponent));
                }
                CompiledOp::Sin => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.sin());
                }
                CompiledOp::Cos => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.cos());
                }
                CompiledOp::Tan => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.tan());
                }
                CompiledOp::Asin => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.asin());
                }
                CompiledOp::Acos => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.acos());
                }
                CompiledOp::Atan => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.atan());
                }
                CompiledOp::Sinh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.sinh());
                }
                CompiledOp::Cosh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.cosh());
                }
                CompiledOp::Tanh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.tanh());
                }
                CompiledOp::Asinh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.asinh());
                }
                CompiledOp::Acosh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.acosh());
                }
                CompiledOp::Atanh => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.atanh());
                }
                CompiledOp::Exp => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.exp());
                }
                CompiledOp::Ln => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.ln());
                }
                CompiledOp::Sqrt => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.sqrt());
                }
                CompiledOp::Abs => {
                    let v = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    stack.push(v.abs());
                }
                CompiledOp::Equal
                | CompiledOp::NotEqual
                | CompiledOp::Less
                | CompiledOp::LessEqual
                | CompiledOp::Greater
                | CompiledOp::GreaterEqual => {
                    let right = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    let left = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    // NaN is never allowed to turn an undecidable/domain-error
                    // condition into false (or an inequality into true).
                    if left.is_nan() || right.is_nan() {
                        return Err(EvalError::NonFiniteValue);
                    }
                    let truth = match op {
                        CompiledOp::Equal => left == right,
                        CompiledOp::NotEqual => left != right,
                        CompiledOp::Less => left < right,
                        CompiledOp::LessEqual => left <= right,
                        CompiledOp::Greater => left > right,
                        CompiledOp::GreaterEqual => left >= right,
                        _ => unreachable!("comparison dispatch is exhaustive"),
                    };
                    stack.push(if truth { 1.0 } else { 0.0 });
                }
                CompiledOp::Not => {
                    let value = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    let truth = Self::condition_value(value)?;
                    stack.push(if truth { 0.0 } else { 1.0 });
                }
                CompiledOp::JumpIfFalse(target) => {
                    let value = stack.pop().ok_or(EvalError::StackUnderflow)?;
                    if !Self::condition_value(value)? {
                        instruction = *target;
                        continue;
                    }
                }
                CompiledOp::Jump(target) => {
                    instruction = *target;
                    continue;
                }
                CompiledOp::NoMatchingPiecewiseBranch => {
                    return Err(EvalError::NoMatchingPiecewiseBranch);
                }
            }
            if finite_only && stack.last().is_some_and(|value| !value.is_finite()) {
                return Err(EvalError::NonFiniteValue);
            }
            instruction += 1;
        }
        if stack.len() != 1 {
            return Err(EvalError::StackRemaining(stack.len()));
        }
        meter.checkpoint()?;
        stack.pop().ok_or(EvalError::StackUnderflow)
    }

    fn condition_value(value: f64) -> Result<bool, EvalError> {
        if value == 1.0 {
            Ok(true)
        } else if value == 0.0 && value.is_sign_positive() {
            Ok(false)
        } else {
            Err(EvalError::InvalidConditionValue)
        }
    }

    /// Evaluates the compiled expression, returning `f64::NAN` on failure.
    pub fn eval(&self, vars: &[f64]) -> f64 {
        self.try_eval(vars).unwrap_or(f64::NAN)
    }
}

/// Compiled multi-dimensional residual system and symbolically differentiated Jacobian matrix.
#[derive(Debug, Clone)]
pub struct CompiledResidualSystem {
    pub vars: Vec<Symbol>,
    pub num_residuals: usize,
    pub num_vars: usize,
    pub compiled_residuals: Vec<CompiledExpr>,
    pub compiled_jacobian: Vec<CompiledExpr>,
}

impl CompiledResidualSystem {
    /// Compiles a system of equations $\mathbf{f}(\mathbf{x}) = \mathbf{0}$ and its symbolic
    /// Jacobian.
    /// Fails closed if any expression contains unmapped symbols or unsupported functions.
    pub fn try_compile(exprs: &[Expr], vars: &[Symbol]) -> Result<Self, CompileError> {
        let mut var_map = HashMap::new();
        var_map
            .try_reserve(vars.len())
            .map_err(|_| CompileError::AllocationFailure)?;
        for (i, v) in vars.iter().enumerate() {
            if var_map.insert(v.clone(), i).is_some() {
                return Err(CompileError::DuplicateVariable(v.clone()));
            }
        }

        let num_residuals = exprs.len();
        let num_vars = vars.len();
        let jacobian_len = num_residuals
            .checked_mul(num_vars)
            .ok_or(CompileError::DimensionOverflow)?;

        let mut compiled_residuals = Vec::new();
        compiled_residuals
            .try_reserve_exact(num_residuals)
            .map_err(|_| CompileError::AllocationFailure)?;
        for e in exprs {
            compiled_residuals.push(CompiledExpr::try_compile(e, &var_map)?);
        }

        let mut compiled_jacobian = Vec::new();
        compiled_jacobian
            .try_reserve_exact(jacobian_len)
            .map_err(|_| CompileError::AllocationFailure)?;
        for e in exprs {
            for v in vars {
                let d = diff(e, v);
                compiled_jacobian.push(CompiledExpr::try_compile(&d, &var_map)?);
            }
        }

        Ok(Self {
            vars: vars.to_vec(),
            num_residuals,
            num_vars,
            compiled_residuals,
            compiled_jacobian,
        })
    }

    /// Convenience infallible compiler that panics on compile error.
    pub fn compile(exprs: &[Expr], vars: &[Symbol]) -> Self {
        Self::try_compile(exprs, vars).expect("residual system compilation failed")
    }

    /// Evaluates the residual vector $\mathbf{f}(\mathbf{x})$ in-place with buffer length validation.
    pub fn try_eval_residuals(&self, x: &[f64], out_res: &mut [f64]) -> Result<(), EvalError> {
        self.try_eval_residuals_with_meter(x, out_res, &mut Unbounded)
    }

    /// Evaluate residuals using one caller meter. Every failure leaves the
    /// output untouched and retains charges for work already attempted.
    pub fn try_eval_residuals_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        x: &[f64],
        out_res: &mut [f64],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        meter.checkpoint()?;
        if out_res.len() != self.num_residuals {
            return Err(EvalError::ResidualBufferMismatch {
                expected: self.num_residuals,
                actual: out_res.len(),
            });
        }
        let evaluated = Self::evaluate_batch_with_meter(
            &self.compiled_residuals,
            x,
            self.num_residuals,
            "residual",
            meter,
        )?;
        charge_evaluation_steps(meter, out_res.len())?;
        meter.checkpoint()?;
        out_res.copy_from_slice(&evaluated);
        Ok(())
    }

    /// Infallible residual evaluator for backward compatibility.
    pub fn eval_residuals(&self, x: &[f64], out_res: &mut [f64]) {
        self.try_eval_residuals(x, out_res)
            .expect("residual evaluation failed");
    }

    /// Evaluates the flat Jacobian matrix $J_{i, j}$ in-place (row-major: $i \times n + j$).
    pub fn try_eval_jacobian(&self, x: &[f64], out_jac: &mut [f64]) -> Result<(), EvalError> {
        self.try_eval_jacobian_with_meter(x, out_jac, &mut Unbounded)
    }

    /// Evaluate the Jacobian after its original sources, sharing one meter
    /// across both phases and publishing only after a final safe point.
    pub fn try_eval_jacobian_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        x: &[f64],
        out_jac: &mut [f64],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        meter.checkpoint()?;
        let expected = self
            .num_residuals
            .checked_mul(self.num_vars)
            .ok_or(EvalError::DimensionOverflow)?;
        if out_jac.len() != expected {
            return Err(EvalError::JacobianBufferMismatch {
                expected,
                actual: out_jac.len(),
            });
        }
        // Differentiation can erase a source-domain obligation. Check the
        // original sources before publishing even a derivative-only result.
        Self::evaluate_batch_with_meter(
            &self.compiled_residuals,
            x,
            self.num_residuals,
            "residual",
            meter,
        )?;
        let evaluated = Self::evaluate_batch_with_meter(
            &self.compiled_jacobian,
            x,
            expected,
            "Jacobian",
            meter,
        )?;
        charge_evaluation_steps(meter, out_jac.len())?;
        meter.checkpoint()?;
        out_jac.copy_from_slice(&evaluated);
        Ok(())
    }

    /// Infallible Jacobian evaluator for backward compatibility.
    pub fn eval_jacobian(&self, x: &[f64], out_jac: &mut [f64]) {
        self.try_eval_jacobian(x, out_jac)
            .expect("jacobian evaluation failed");
    }

    /// Evaluates both residuals and Jacobian simultaneously.
    pub fn try_eval_system(
        &self,
        x: &[f64],
        out_res: &mut [f64],
        out_jac: &mut [f64],
    ) -> Result<(), EvalError> {
        self.try_eval_system_with_meter(x, out_res, out_jac, &mut Unbounded)
    }

    /// Evaluate both lanes transactionally with a single cumulative ledger.
    /// Exhaustion or cancellation in the derivative or publication phase
    /// cannot expose an already-computed residual vector.
    pub fn try_eval_system_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        x: &[f64],
        out_res: &mut [f64],
        out_jac: &mut [f64],
        meter: &mut M,
    ) -> Result<(), EvalError> {
        meter.checkpoint()?;
        if out_res.len() != self.num_residuals {
            return Err(EvalError::ResidualBufferMismatch {
                expected: self.num_residuals,
                actual: out_res.len(),
            });
        }
        let jacobian_len = self
            .num_residuals
            .checked_mul(self.num_vars)
            .ok_or(EvalError::DimensionOverflow)?;
        if out_jac.len() != jacobian_len {
            return Err(EvalError::JacobianBufferMismatch {
                expected: jacobian_len,
                actual: out_jac.len(),
            });
        }
        let residuals = Self::evaluate_batch_with_meter(
            &self.compiled_residuals,
            x,
            self.num_residuals,
            "residual",
            meter,
        )?;
        let jacobian = Self::evaluate_batch_with_meter(
            &self.compiled_jacobian,
            x,
            jacobian_len,
            "Jacobian",
            meter,
        )?;
        let published = out_res
            .len()
            .checked_add(out_jac.len())
            .ok_or(EvalError::DimensionOverflow)?;
        charge_evaluation_steps(meter, published)?;
        meter.checkpoint()?;
        out_res.copy_from_slice(&residuals);
        out_jac.copy_from_slice(&jacobian);
        Ok(())
    }

    /// Infallible system evaluator for backward compatibility.
    pub fn eval_system(&self, x: &[f64], out_res: &mut [f64], out_jac: &mut [f64]) {
        self.try_eval_system(x, out_res, out_jac)
            .expect("system evaluation failed");
    }

    /// Runs a fail-closed approximate consistency check against central finite differences.
    ///
    /// This diagnostic is neither an exact verifier nor mathematical evidence for the compiled
    /// Jacobian. It rejects non-finite inputs, outputs, steps, and tolerances.
    pub fn check_with_finite_differences(&self, x: &[f64], eps: f64, tol: f64) -> bool {
        if x.len() != self.num_vars
            || self.vars.len() != self.num_vars
            || !eps.is_finite()
            || eps <= 0.0
            || !tol.is_finite()
            || tol < 0.0
            || x.iter().any(|value| !value.is_finite())
        {
            return false;
        }

        let Some(jacobian_len) = self.num_residuals.checked_mul(self.num_vars) else {
            return false;
        };
        let Ok(jac) = Self::evaluate_batch(&self.compiled_jacobian, x, jacobian_len, "Jacobian")
        else {
            return false;
        };
        if jac.iter().any(|value| !value.is_finite()) {
            return false;
        }

        let Ok(baseline) =
            Self::evaluate_batch(&self.compiled_residuals, x, self.num_residuals, "residual")
        else {
            return false;
        };
        if baseline.iter().any(|value| !value.is_finite()) {
            return false;
        }

        let mut x_plus = Vec::new();
        let mut x_minus = Vec::new();
        if x_plus.try_reserve_exact(x.len()).is_err() || x_minus.try_reserve_exact(x.len()).is_err()
        {
            return false;
        }
        x_plus.extend_from_slice(x);
        x_minus.extend_from_slice(x);

        for j in 0..self.num_vars {
            x_plus[j] = x[j] + eps;
            x_minus[j] = x[j] - eps;
            let denominator = x_plus[j] - x_minus[j];
            if !x_plus[j].is_finite()
                || !x_minus[j].is_finite()
                || !denominator.is_finite()
                || denominator == 0.0
            {
                return false;
            }

            let Ok(res_plus) = Self::evaluate_batch(
                &self.compiled_residuals,
                &x_plus,
                self.num_residuals,
                "residual",
            ) else {
                return false;
            };
            let Ok(res_minus) = Self::evaluate_batch(
                &self.compiled_residuals,
                &x_minus,
                self.num_residuals,
                "residual",
            ) else {
                return false;
            };

            for i in 0..self.num_residuals {
                let numerical_deriv = (res_plus[i] - res_minus[i]) / denominator;
                let compiled_deriv = jac[i * self.num_vars + j];
                let difference = (numerical_deriv - compiled_deriv).abs();
                if !res_plus[i].is_finite()
                    || !res_minus[i].is_finite()
                    || !numerical_deriv.is_finite()
                    || !difference.is_finite()
                    || difference > tol
                {
                    return false;
                }
            }

            x_plus[j] = x[j];
            x_minus[j] = x[j];
        }
        true
    }

    fn evaluate_batch(
        expressions: &[CompiledExpr],
        x: &[f64],
        expected: usize,
        lane: &'static str,
    ) -> Result<Vec<f64>, EvalError> {
        Self::evaluate_batch_with_meter(expressions, x, expected, lane, &mut Unbounded)
    }

    fn evaluate_batch_with_meter<M: BudgetMeter + ?Sized>(
        expressions: &[CompiledExpr],
        x: &[f64],
        expected: usize,
        lane: &'static str,
        meter: &mut M,
    ) -> Result<Vec<f64>, EvalError> {
        meter.checkpoint()?;
        if expressions.len() != expected {
            return Err(EvalError::CompiledExpressionCountMismatch {
                lane,
                expected,
                actual: expressions.len(),
            });
        }
        charge_evaluation_allocation(meter, expected)?;
        let mut evaluated = Vec::new();
        evaluated
            .try_reserve_exact(expected)
            .map_err(|_| EvalError::AllocationFailure)?;
        for expression in expressions {
            charge_evaluation_steps(meter, 1)?;
            evaluated.push(expression.try_eval_with_meter(x, meter)?);
        }
        meter.checkpoint()?;
        Ok(evaluated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::de::DeserializeSeed;
    use serde::de::value::{Error as ValueError, StrDeserializer};
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    struct HintCompiledOpSeq {
        remaining: usize,
        claimed: Option<usize>,
        next_calls: Rc<Cell<usize>>,
    }

    impl HintCompiledOpSeq {
        fn new(remaining: usize, claimed: Option<usize>, next_calls: Rc<Cell<usize>>) -> Self {
            Self {
                remaining,
                claimed,
                next_calls,
            }
        }
    }

    impl<'de> SeqAccess<'de> for HintCompiledOpSeq {
        type Error = ValueError;

        fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
        where
            T: DeserializeSeed<'de>,
        {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            self.next_calls.set(self.next_calls.get() + 1);
            seed.deserialize(StrDeserializer::<ValueError>::new("Sin"))
                .map(Some)
        }

        fn size_hint(&self) -> Option<usize> {
            self.claimed
        }
    }

    #[test]
    fn compiled_expr_serde_round_trips_and_revalidates_the_wire() {
        let expected =
            CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(2.0), CompiledOp::Pow(3.0)])
                .unwrap();
        let wire = serde_json::to_value(&expected).unwrap();

        assert_eq!(wire["ops"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            serde_json::from_value::<CompiledExpr>(wire).unwrap(),
            expected
        );
        let unknown_field = serde_json::from_str::<CompiledExpr>(
            r#"{"ops":[{"LoadConst":1.0}],"unexpected":true}"#,
        )
        .expect_err("unknown compiled-expression fields must be refused");
        assert!(unknown_field.to_string().contains("unknown field"));

        let malformed = serde_json::from_str::<CompiledExpr>(r#"{"ops":[{"Add":0}]}"#)
            .expect_err("deserialization must run the structural bytecode validator");
        assert!(malformed.to_string().contains("invalid zero arity"));
    }

    #[test]
    fn compiled_expr_serde_limit_distrusts_sequence_hints_and_stops_early() {
        for hostile_hint in [Some(MAX_COMPILE_OPS + 1), Some(usize::MAX)] {
            let next_calls = Rc::new(Cell::new(0));
            let error = BoundedCompiledOpsVisitor
                .visit_seq(HintCompiledOpSeq::new(
                    MAX_COMPILE_OPS + 2,
                    hostile_hint,
                    Rc::clone(&next_calls),
                ))
                .expect_err("an over-limit hint must fail before reading elements");
            assert!(error.to_string().contains("at most 8192"));
            assert_eq!(next_calls.get(), 0);
        }

        for underreported_hint in [None, Some(1), Some(MAX_COMPILE_OPS)] {
            let next_calls = Rc::new(Cell::new(0));
            let error = BoundedCompiledOpsVisitor
                .visit_seq(HintCompiledOpSeq::new(
                    MAX_COMPILE_OPS + 2,
                    underreported_hint,
                    Rc::clone(&next_calls),
                ))
                .expect_err("an underreported sequence must not bypass the operation limit");
            assert!(error.to_string().contains("at most 8192"));
            assert_eq!(
                next_calls.get(),
                MAX_COMPILE_OPS + 1,
                "the decoder must stop after observing one excess element"
            );
        }

        let next_calls = Rc::new(Cell::new(0));
        let ops = BoundedCompiledOpsVisitor
            .visit_seq(HintCompiledOpSeq::new(
                MAX_COMPILE_OPS,
                None,
                Rc::clone(&next_calls),
            ))
            .expect("the exact operation limit is admitted by the streaming decoder");
        assert_eq!(ops.len(), MAX_COMPILE_OPS);
        assert_eq!(next_calls.get(), MAX_COMPILE_OPS);
    }

    #[test]
    fn compiled_expr_serde_capacity_growth_is_bounded() {
        assert_eq!(next_compiled_op_capacity(0), Some(1));
        assert_eq!(next_compiled_op_capacity(1), Some(2));
        assert_eq!(next_compiled_op_capacity(2), Some(4));
        assert_eq!(
            next_compiled_op_capacity(MAX_COMPILE_OPS - 1),
            Some(MAX_COMPILE_OPS)
        );
        assert_eq!(next_compiled_op_capacity(MAX_COMPILE_OPS), None);
    }

    #[test]
    fn test_compile_rejection_of_unmapped_symbols() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let mut var_map = HashMap::new();
        var_map.insert(x, 0);

        let expr = Expr::Add(vec![Expr::symbol("x"), Expr::symbol("y")]);
        assert_eq!(
            CompiledExpr::try_compile(&expr, &var_map),
            Err(CompileError::UnmappedVariable(y))
        );
    }

    #[test]
    fn test_compile_rejection_of_unsupported_functions() {
        let x = Symbol::new("x");
        let mut var_map = HashMap::new();
        var_map.insert(x, 0);

        let expr = Expr::Function("custom_bessel".to_string(), vec![Expr::symbol("x")]);
        assert!(matches!(
            CompiledExpr::try_compile(&expr, &var_map),
            Err(CompileError::UnsupportedFunction(_))
        ));
    }

    #[test]
    fn test_compile_positive_real_symbolic_powers() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let mut var_map = HashMap::new();
        var_map.insert(x.clone(), 0);
        var_map.insert(y.clone(), 1);

        let expr = Expr::Pow(Arc::new(Expr::Sym(x)), Arc::new(Expr::Sym(y)));
        let compiled = CompiledExpr::try_compile(&expr, &var_map).unwrap();
        assert_eq!(compiled.ops().last(), Some(&CompiledOp::PowPositive));
        assert_eq!(compiled.try_eval_finite(&[2.0, 3.0]), Ok(8.0));
        assert_eq!(
            compiled.try_eval_finite(&[-2.0, 3.0]),
            Err(EvalError::NonPositivePowerBase)
        );
    }

    #[test]
    fn test_eval_buffer_mismatch_and_bounds() {
        let x = Symbol::new("x");
        let sys = CompiledResidualSystem::try_compile(&[Expr::symbol("x")], &[x]).unwrap();

        let mut wrong_res = [0.0; 2];
        assert!(matches!(
            sys.try_eval_residuals(&[1.0], &mut wrong_res),
            Err(EvalError::ResidualBufferMismatch { .. })
        ));

        let mut wrong_jac = [0.0; 2];
        assert!(matches!(
            sys.try_eval_jacobian(&[1.0], &mut wrong_jac),
            Err(EvalError::JacobianBufferMismatch { .. })
        ));
    }

    #[test]
    fn compile_operation_limit_includes_parent_operations() {
        let boundary = Expr::Add(vec![Expr::from_i64(1); MAX_COMPILE_OPS - 1]);
        assert_eq!(
            CompiledExpr::try_compile(&boundary, &HashMap::new())
                .unwrap()
                .ops()
                .len(),
            MAX_COMPILE_OPS
        );

        let expression = Expr::Add(vec![Expr::from_i64(1); MAX_COMPILE_OPS]);
        assert!(matches!(
            CompiledExpr::try_compile(&expression, &HashMap::new()),
            Err(CompileError::OpLimitExceeded { ops, max })
                if ops == MAX_COMPILE_OPS + 1 && max == MAX_COMPILE_OPS
        ));
    }

    #[test]
    fn non_finite_exponents_are_refused() {
        let huge = fsym_core::BigInt::parse_bytes("9".repeat(400).as_bytes(), 10).unwrap();
        let base = Arc::new(Expr::from_i64(2));

        for exponent in [
            Expr::Integer(huge.clone()),
            Expr::Rational(fsym_core::BigRational::new(
                huge,
                fsym_core::BigInt::from(1),
            )),
        ] {
            assert!(matches!(
                CompiledExpr::try_compile(
                    &Expr::Pow(base.clone(), Arc::new(exponent)),
                    &HashMap::new(),
                ),
                Err(CompileError::UnsupportedExponent(_))
            ));
        }
    }

    #[test]
    fn inexact_f64_exponent_components_are_refused() {
        let largest_safe =
            fsym_core::BigInt::parse_bytes(b"9007199254740991", 10).expect("valid integer");
        let rounded_odd =
            fsym_core::BigInt::parse_bytes(b"9007199254740993", 10).expect("valid integer");
        let rounded_denominator =
            fsym_core::BigInt::parse_bytes(b"9007199254740992", 10).expect("valid integer");
        let base = Arc::new(Expr::from_i64(-1));

        let admitted = CompiledExpr::try_compile(
            &Expr::Pow(base.clone(), Arc::new(Expr::Integer(largest_safe))),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(admitted.try_eval(&[]), Ok(-1.0));

        for exponent in [
            Expr::Integer(rounded_odd.clone()),
            Expr::Rational(fsym_core::BigRational::new(
                rounded_odd,
                rounded_denominator,
            )),
        ] {
            assert!(matches!(
                CompiledExpr::try_compile(
                    &Expr::Pow(base.clone(), Arc::new(exponent)),
                    &HashMap::new(),
                ),
                Err(CompileError::UnsupportedExponent(_))
            ));
        }
    }

    #[test]
    fn rational_literal_range_checks_apply_to_the_exact_quotient() {
        let huge = fsym_core::BigInt::parse_bytes("9".repeat(400).as_bytes(), 10).unwrap();
        for (sign, expected_unit) in [(1_i64, 1.0_f64), (-1, -1.0)] {
            // A component may exceed binary64's range while the exact quotient
            // rounds to a finite value. Finite underflow has canonical +0 bits.
            let near_unit = Expr::Rational(fsym_core::BigRational::new(
                (&huge + 1_i64) * fsym_core::BigInt::from(sign),
                huge.clone(),
            ));
            let tiny = Expr::Rational(fsym_core::BigRational::new(
                fsym_core::BigInt::from(sign),
                huge.clone(),
            ));
            for (source, expected) in [(near_unit, expected_unit), (tiny, 0.0)] {
                let compiled = CompiledExpr::try_compile(&source, &HashMap::new()).unwrap();
                assert_eq!(
                    compiled.try_eval_finite(&[]).unwrap().to_bits(),
                    expected.to_bits()
                );
            }

            // True quotient overflow remains a refusal for either sign.
            let overflowing = Expr::Rational(fsym_core::BigRational::new(
                huge.clone() * fsym_core::BigInt::from(sign),
                fsym_core::BigInt::from(2),
            ));
            assert!(matches!(
                CompiledExpr::try_compile(&overflowing, &HashMap::new()),
                Err(CompileError::NumericConversion(_))
            ));
        }
    }

    #[test]
    fn malformed_bytecode_is_refused_at_the_construction_boundary() {
        for (ops, expected) in [
            (
                vec![CompiledOp::Add(0)],
                EvalError::InvalidReductionArity { operation: "Add" },
            ),
            (
                vec![CompiledOp::Mul(0)],
                EvalError::InvalidReductionArity { operation: "Mul" },
            ),
            (
                vec![CompiledOp::LoadConst(1.0), CompiledOp::Pow(f64::NAN)],
                EvalError::NonFinitePowerExponent,
            ),
            (
                vec![CompiledOp::LoadConst(1.0), CompiledOp::Pow(f64::INFINITY)],
                EvalError::NonFinitePowerExponent,
            ),
            (
                vec![CompiledOp::LoadConst(f64::NAN)],
                EvalError::NonCanonicalConstant,
            ),
            (
                vec![CompiledOp::LoadConst(-0.0)],
                EvalError::NonCanonicalConstant,
            ),
            (
                vec![CompiledOp::LoadConst(1.0), CompiledOp::LoadConst(2.0)],
                EvalError::StackRemaining(2),
            ),
            (vec![CompiledOp::Add(1)], EvalError::StackUnderflow),
        ] {
            assert_eq!(CompiledExpr::try_from_ops(ops), Err(expected));
        }

        let valid = CompiledExpr::try_from_ops(vec![
            CompiledOp::LoadConst(2.0),
            CompiledOp::LoadConst(3.0),
            CompiledOp::Add(2),
        ])
        .unwrap();
        assert_eq!(valid.try_eval(&[]), Ok(5.0));
    }

    #[test]
    fn failed_batch_evaluation_does_not_publish_partial_outputs() {
        let valid = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(1.0)]).unwrap();
        let missing_variable = CompiledExpr::try_from_ops(vec![CompiledOp::LoadVar(0)]).unwrap();
        let system = CompiledResidualSystem {
            vars: Vec::new(),
            num_residuals: 2,
            num_vars: 1,
            compiled_residuals: vec![valid.clone(), missing_variable.clone()],
            compiled_jacobian: vec![valid, missing_variable],
        };

        let mut residuals = [9.0, 9.0];
        assert_eq!(
            system.try_eval_residuals(&[], &mut residuals),
            Err(EvalError::VariableOutOfBounds {
                index: 0,
                slice_len: 0,
            })
        );
        assert_eq!(residuals, [9.0, 9.0]);

        let mut jacobian = [9.0, 9.0];
        assert_eq!(
            system.try_eval_jacobian(&[], &mut jacobian),
            Err(EvalError::VariableOutOfBounds {
                index: 0,
                slice_len: 0,
            })
        );
        assert_eq!(jacobian, [9.0, 9.0]);

        assert_eq!(
            system.try_eval_system(&[], &mut residuals, &mut jacobian),
            Err(EvalError::VariableOutOfBounds {
                index: 0,
                slice_len: 0,
            })
        );
        assert_eq!(residuals, [9.0, 9.0]);
        assert_eq!(jacobian, [9.0, 9.0]);
    }

    #[test]
    fn finite_difference_diagnostic_rejects_invalid_numeric_policy() {
        let x = Symbol::new("x");
        let system = CompiledResidualSystem::try_compile(&[Expr::symbol("x")], &[x]).unwrap();

        assert!(!system.check_with_finite_differences(&[1.0], 0.0, 1e-5));
        assert!(!system.check_with_finite_differences(&[1.0], 1e-6, f64::NAN));
        assert!(!system.check_with_finite_differences(&[f64::NAN], 1e-6, 1e-5));
    }

    #[test]
    fn duplicate_variables_and_oversized_bytecode_are_typed_refusals() {
        let x = Symbol::new("x");
        assert!(matches!(
            CompiledResidualSystem::try_compile(&[Expr::symbol("x")], &[x.clone(), x.clone()],),
            Err(CompileError::DuplicateVariable(ref duplicate)) if duplicate == &x
        ));

        assert_eq!(
            CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(1.0); MAX_COMPILE_OPS + 1]),
            Err(EvalError::OpLimitExceeded {
                ops: MAX_COMPILE_OPS + 1,
                max: MAX_COMPILE_OPS,
            })
        );
    }

    #[test]
    fn inconsistent_public_system_shape_is_refused_without_output_mutation() {
        let system = CompiledResidualSystem {
            vars: Vec::new(),
            num_residuals: 1,
            num_vars: 0,
            compiled_residuals: Vec::new(),
            compiled_jacobian: Vec::new(),
        };
        let mut output = [7.0];
        assert!(matches!(
            system.try_eval_residuals(&[], &mut output),
            Err(EvalError::CompiledExpressionCountMismatch {
                lane: "residual",
                expected: 1,
                actual: 0,
            })
        ));
        assert_eq!(output, [7.0]);
    }
}
