//! Lazy, bytecode-preserving common-subexpression reuse.
//!
//! The optimizer proposes repeated pure postfix spans. A separate admission
//! pass checks complete stack expressions, exact opcode/float-bit equality,
//! and every incoming jump before constructing executable memo tables. The
//! original interpreter and source checker remain the reference lanes. There
//! is no reassociation, branch hoisting, persisted cache, or new numeric claim.

use super::{
    CompileError, CompiledExpr, CompiledOp, EvalError, charge_compilation_work,
    charge_evaluation_allocation, charge_evaluation_steps,
};
use fsym_budget::{BudgetMeter, Dimension, Unbounded};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

const MAX_SHARED_OPERATIONS: usize = 1_048_576;
const MAX_COMPARISON_OPERATIONS: usize = 32 * MAX_SHARED_OPERATIONS;
const FINGERPRINT_BASE: u64 = 0x9e37_79b1_85eb_ca87;

#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    fingerprint: u64,
    factor: u64,
}

#[derive(Debug, Clone, Copy)]
struct Location {
    program: usize,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy)]
struct ReuseRange {
    location: Location,
    slot: usize,
}

struct Seen {
    first: Location,
    slot: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct MemoStart {
    slot: usize,
    end: usize,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ProgramReusePlan {
    starts: Vec<Option<MemoStart>>,
    stores: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SharedEvaluationPlan {
    programs: Vec<ProgramReusePlan>,
    slots: usize,
}

pub(crate) struct EvaluationMemo {
    values: Vec<Option<f64>>,
}

/// An immutable collection of validated scalar programs with shared finite-real
/// evaluation. Inputs have one common variable-index universe. Constructing a
/// batch establishes preservation of those programs, not their relationship
/// to symbolic sources; use the source-checked compiler for that obligation.
///
/// Reuse is lazy and local to one evaluation. An inactive Piecewise expression
/// is never evaluated merely because another branch contains the same code.
/// Admission permits at most 1,048,576 total stored instructions and 33,554,432
/// direct opcode comparisons in each planning/checking pass. Deeply nested raw
/// bytecode can exhaust comparison work before reaching the instruction cap;
/// this is an explicit refusal, never an unchecked optimization fallback.
#[derive(Debug, Clone)]
pub struct CompiledBatch {
    programs: Vec<CompiledExpr>,
    reuse: SharedEvaluationPlan,
}

fn reserve<T, M: BudgetMeter + ?Sized>(
    count: usize,
    meter: &mut M,
) -> Result<Vec<T>, CompileError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CompileError::DimensionOverflow)?;
    charge_compilation_work(meter, count, bytes, usize::from(count != 0))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| CompileError::AllocationFailure)?;
    Ok(values)
}

fn comparison_work<M: BudgetMeter + ?Sized>(
    count: usize,
    used: &mut usize,
    meter: &mut M,
) -> Result<(), CompileError> {
    *used = used
        .checked_add(count)
        .ok_or(CompileError::DimensionOverflow)?;
    if *used > MAX_COMPARISON_OPERATIONS {
        return Err(CompileError::SharedEvaluationLimit("comparison work"));
    }
    charge_compilation_work(meter, count, 0, 0)
}

fn same_operation(left: &CompiledOp, right: &CompiledOp) -> bool {
    match (left, right) {
        (CompiledOp::LoadConst(left), CompiledOp::LoadConst(right))
        | (CompiledOp::Pow(left), CompiledOp::Pow(right)) => left.to_bits() == right.to_bits(),
        _ => left == right,
    }
}

fn operations<'a>(programs: &[&'a CompiledExpr], location: Location) -> &'a [CompiledOp] {
    &programs[location.program].ops()[location.start..=location.end]
}

// Fingerprints are only an index into candidates. Exact opcode comparison
// below resolves all collisions; no digest is used as semantic identity.
fn operation_fingerprint(operation: &CompiledOp) -> u64 {
    let mut hash = DefaultHasher::new();
    std::mem::discriminant(operation).hash(&mut hash);
    match operation {
        CompiledOp::LoadConst(value) | CompiledOp::Pow(value) => value.to_bits().hash(&mut hash),
        CompiledOp::LoadVar(index) | CompiledOp::Add(index) | CompiledOp::Mul(index) => {
            index.hash(&mut hash);
        }
        _ => {}
    }
    hash.finish()
}

fn arity(operation: &CompiledOp) -> Option<usize> {
    match operation {
        CompiledOp::LoadConst(_) | CompiledOp::LoadVar(_) => Some(0),
        CompiledOp::Add(count) | CompiledOp::Mul(count) => Some(*count),
        CompiledOp::PowPositive
        | CompiledOp::Equal
        | CompiledOp::NotEqual
        | CompiledOp::Less
        | CompiledOp::LessEqual
        | CompiledOp::Greater
        | CompiledOp::GreaterEqual => Some(2),
        CompiledOp::JumpIfFalse(_)
        | CompiledOp::Jump(_)
        | CompiledOp::NoMatchingPiecewiseBranch => None,
        _ => Some(1),
    }
}

fn incoming_jumps<M: BudgetMeter + ?Sized>(
    program: &CompiledExpr,
    meter: &mut M,
) -> Result<Vec<bool>, CompileError> {
    let mut targets = reserve(program.ops().len() + 1, meter)?;
    targets.resize(program.ops().len() + 1, false);
    for operation in program.ops() {
        charge_compilation_work(meter, 1, 0, 0)?;
        if let CompiledOp::JumpIfFalse(target) | CompiledOp::Jump(target) = operation {
            let Some(target) = targets.get_mut(*target) else {
                return Err(CompileError::InvalidSharedEvaluationPlan);
            };
            *target = true;
        }
    }
    Ok(targets)
}

fn propose_ranges<M: BudgetMeter + ?Sized>(
    programs: &[&CompiledExpr],
    total: usize,
    meter: &mut M,
) -> Result<(Vec<Location>, Vec<ReuseRange>), CompileError> {
    let mut definitions = reserve(total, meter)?;
    let mut ranges = reserve(total, meter)?;
    // A bucket points into a flat linked list, avoiding one allocation for
    // every unique subtree. Entries are visited in source order only.
    let mut entries: Vec<(Seen, Option<usize>)> = reserve(total, meter)?;
    charge_compilation_work(meter, total, total * 64, usize::from(total != 0))?;
    let mut index: HashMap<(usize, u64), usize> = HashMap::new();
    index
        .try_reserve(total)
        .map_err(|_| CompileError::AllocationFailure)?;
    let mut comparisons = 0;
    for (program_index, program) in programs.iter().enumerate() {
        let targets = incoming_jumps(program, meter)?;
        let mut stack: Vec<Option<Span>> = reserve(program.ops().len(), meter)?;
        for (instruction, operation) in program.ops().iter().enumerate() {
            charge_compilation_work(meter, 8, 0, 0)?;
            if targets[instruction] {
                stack.clear();
            }
            let Some(count) = arity(operation) else {
                stack.clear();
                continue;
            };
            if count > stack.len() {
                // An operand came from another control-flow block. It is
                // deliberately opaque to this local optimization.
                stack.clear();
                stack.push(None);
                continue;
            }
            let child_start = stack.len() - count;
            let mut span = Span {
                start: instruction,
                end: instruction,
                fingerprint: 0,
                factor: 1,
            };
            let mut valid = true;
            for (offset, child) in stack[child_start..].iter().enumerate() {
                charge_compilation_work(meter, 4, 0, 0)?;
                let Some(child) = child else {
                    valid = false;
                    break;
                };
                if offset == 0 {
                    span.start = child.start;
                } else if span.end + 1 != child.start {
                    valid = false;
                    break;
                }
                span.end = child.end;
                span.fingerprint = span
                    .fingerprint
                    .wrapping_mul(child.factor)
                    .wrapping_add(child.fingerprint);
                span.factor = span.factor.wrapping_mul(child.factor);
            }
            if count != 0 && span.end + 1 != instruction {
                valid = false;
            }
            stack.truncate(child_start);
            if !valid {
                stack.push(None);
                continue;
            }
            span.end = instruction;
            span.fingerprint = span
                .fingerprint
                .wrapping_mul(FINGERPRINT_BASE)
                .wrapping_add(operation_fingerprint(operation));
            span.factor = span.factor.wrapping_mul(FINGERPRINT_BASE);
            stack.push(Some(span));
            let length = span.end - span.start + 1;
            if length < 2 {
                continue;
            }
            let location = Location {
                program: program_index,
                start: span.start,
                end: span.end,
            };
            let key = (length, span.fingerprint);
            let head = index.get(&key).copied();
            let mut candidate = head;
            let mut matched = false;
            while let Some(position) = candidate {
                let (seen, next) = &mut entries[position];
                comparison_work(length, &mut comparisons, meter)?;
                if operations(programs, seen.first)
                    .iter()
                    .zip(operations(programs, location))
                    .all(|(left, right)| same_operation(left, right))
                {
                    let slot = if let Some(slot) = seen.slot {
                        slot
                    } else {
                        let slot = definitions.len();
                        definitions.push(seen.first);
                        ranges.push(ReuseRange {
                            location: seen.first,
                            slot,
                        });
                        seen.slot = Some(slot);
                        slot
                    };
                    ranges.push(ReuseRange { location, slot });
                    matched = true;
                    break;
                }
                candidate = *next;
            }
            if !matched {
                index.insert(key, entries.len());
                entries.push((
                    Seen {
                        first: location,
                        slot: None,
                    },
                    head,
                ));
            }
        }
    }
    meter.checkpoint()?;
    Ok((definitions, ranges))
}

/// This pass does not use the optimizer's fingerprints or subtree analysis.
/// It checks each complete proposed span directly against original bytecode,
/// including float bits and branch entry points. Mutated proposals fail closed.
fn verify_ranges<M: BudgetMeter + ?Sized>(
    programs: &[&CompiledExpr],
    definitions: &[Location],
    ranges: &[ReuseRange],
    meter: &mut M,
) -> Result<(), CompileError> {
    let mut targets = reserve(programs.len(), meter)?;
    for program in programs {
        targets.push(incoming_jumps(program, meter)?);
    }
    let mut compared = 0;
    for range in ranges {
        charge_compilation_work(meter, 1, 0, 0)?;
        let location = range.location;
        let definition = definitions
            .get(range.slot)
            .ok_or(CompileError::InvalidSharedEvaluationPlan)?;
        for location in [location, *definition] {
            let Some(program) = programs.get(location.program) else {
                return Err(CompileError::InvalidSharedEvaluationPlan);
            };
            if location.start >= location.end || location.end >= program.ops().len() {
                return Err(CompileError::InvalidSharedEvaluationPlan);
            }
        }
        let original = operations(programs, *definition);
        let proposed = operations(programs, location);
        if original.len() != proposed.len() {
            return Err(CompileError::InvalidSharedEvaluationPlan);
        }
        comparison_work(proposed.len(), &mut compared, meter)?;
        let mut depth = 0;
        for (offset, (left, right)) in original.iter().zip(proposed).enumerate() {
            if !same_operation(left, right)
                || matches!(
                    right,
                    CompiledOp::JumpIfFalse(_)
                        | CompiledOp::Jump(_)
                        | CompiledOp::NoMatchingPiecewiseBranch
                )
                || (offset != 0 && targets[location.program][location.start + offset])
                || (offset != 0 && targets[definition.program][definition.start + offset])
            {
                return Err(CompileError::InvalidSharedEvaluationPlan);
            }
            depth = CompiledExpr::stack_depth_after(right, depth)
                .map_err(|_| CompileError::InvalidSharedEvaluationPlan)?;
        }
        if depth != 1 {
            return Err(CompileError::InvalidSharedEvaluationPlan);
        }
    }
    meter.checkpoint()?;
    Ok(())
}

impl SharedEvaluationPlan {
    pub(crate) fn build<'a, M: BudgetMeter + ?Sized>(
        programs: impl ExactSizeIterator<Item = &'a CompiledExpr>,
        meter: &mut M,
    ) -> Result<Self, CompileError> {
        let count = programs.len();
        if count > MAX_SHARED_OPERATIONS {
            return Err(CompileError::SharedEvaluationLimit("program count"));
        }
        let mut sources = reserve(count, meter)?;
        let mut total = 0usize;
        for program in programs {
            charge_compilation_work(meter, 1, 0, 0)?;
            total = total
                .checked_add(program.ops().len())
                .ok_or(CompileError::DimensionOverflow)?;
            if total > MAX_SHARED_OPERATIONS {
                return Err(CompileError::SharedEvaluationLimit("stored operations"));
            }
            sources.push(program);
        }
        let (definitions, ranges) = propose_ranges(&sources, total, meter)?;
        verify_ranges(&sources, &definitions, &ranges, meter)?;
        let mut programs = reserve(count, meter)?;
        programs.resize_with(count, ProgramReusePlan::default);
        for range in ranges {
            charge_compilation_work(meter, 1, 0, 0)?;
            let plan = &mut programs[range.location.program];
            if plan.starts.is_empty() {
                let length = sources[range.location.program].ops().len();
                plan.starts = reserve(length, meter)?;
                plan.starts.resize(length, None);
                plan.stores = reserve(length, meter)?;
                plan.stores.resize(length, None);
            }
            let start = &mut plan.starts[range.location.start];
            // Nested spans can share a first instruction. The outermost
            // available value skips the most work, without evaluating more.
            if start.is_none_or(|previous| previous.end < range.location.end) {
                *start = Some(MemoStart {
                    slot: range.slot,
                    end: range.location.end,
                });
            }
            let store = &mut plan.stores[range.location.end];
            if store.is_some_and(|previous| previous != range.slot) {
                return Err(CompileError::InvalidSharedEvaluationPlan);
            }
            *store = Some(range.slot);
        }
        meter.checkpoint()?;
        Ok(Self {
            programs,
            slots: definitions.len(),
        })
    }

    pub(crate) fn begin<M: BudgetMeter + ?Sized>(
        &self,
        meter: &mut M,
    ) -> Result<EvaluationMemo, EvalError> {
        charge_evaluation_steps(meter, self.slots)?;
        if self.slots != 0 {
            meter.charge_batch(&[
                (
                    Dimension::MemoryBytes,
                    (self.slots * std::mem::size_of::<Option<f64>>()) as u64,
                ),
                (Dimension::AllocationCount, 1),
            ])?;
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.slots)
            .map_err(|_| EvalError::AllocationFailure)?;
        values.resize(self.slots, None);
        meter.checkpoint()?;
        Ok(EvaluationMemo { values })
    }

    // Crate-private: callers retain the exact immutable programs, in the same
    // order used by build. No external program or persisted plan enters here.
    pub(crate) fn evaluate<M: BudgetMeter + ?Sized>(
        &self,
        index: usize,
        program: &CompiledExpr,
        point: &[f64],
        memo: &mut EvaluationMemo,
        meter: &mut M,
    ) -> Result<f64, EvalError> {
        let plan = &self.programs[index];
        let reuse = (!plan.starts.is_empty()).then_some((plan, memo));
        program.try_eval_with_reuse(point, true, reuse, meter)
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slots
    }
}

impl ProgramReusePlan {
    pub(super) fn load<M: BudgetMeter + ?Sized>(
        &self,
        instruction: usize,
        memo: &EvaluationMemo,
        meter: &mut M,
    ) -> Result<Option<(f64, usize)>, EvalError> {
        charge_evaluation_steps(meter, 1)?;
        Ok(self.starts[instruction]
            .and_then(|start| memo.values[start.slot].map(|value| (value, start.end))))
    }

    pub(super) fn store<M: BudgetMeter + ?Sized>(
        &self,
        instruction: usize,
        value: Option<f64>,
        memo: &mut EvaluationMemo,
        meter: &mut M,
    ) -> Result<(), EvalError> {
        if let Some(slot) = self.stores[instruction] {
            charge_evaluation_steps(meter, 1)?;
            let value = value.ok_or(EvalError::StackUnderflow)?;
            if !value.is_finite() {
                return Err(EvalError::NonFiniteValue);
            }
            memo.values[slot] = Some(value);
        }
        Ok(())
    }
}

impl CompiledBatch {
    pub fn try_from_programs(programs: Vec<CompiledExpr>) -> Result<Self, CompileError> {
        Self::try_from_programs_with_meter(programs, &mut Unbounded)
    }

    /// Build and independently admit reuse tables on one cumulative ledger.
    /// Candidate lookup, comparisons, scratch storage and publication are
    /// charged. Every original program remains available as a reference.
    pub fn try_from_programs_with_meter<M: BudgetMeter + ?Sized>(
        programs: Vec<CompiledExpr>,
        meter: &mut M,
    ) -> Result<Self, CompileError> {
        let reuse = SharedEvaluationPlan::build(programs.iter(), meter)?;
        meter.checkpoint()?;
        Ok(Self { programs, reuse })
    }

    pub fn programs(&self) -> &[CompiledExpr] {
        &self.programs
    }

    /// Number of reusable compound bytecode spans; no speedup is implied.
    pub fn shared_subexpression_count(&self) -> usize {
        self.reuse.slot_count()
    }

    pub fn try_eval_finite(&self, point: &[f64]) -> Result<Vec<f64>, EvalError> {
        self.try_eval_finite_with_meter(point, &mut Unbounded)
    }

    /// All values publish together. Memo slots are discarded on every return,
    /// including cancellation, domain failure and resource exhaustion.
    pub fn try_eval_finite_with_meter<M: BudgetMeter + ?Sized>(
        &self,
        point: &[f64],
        meter: &mut M,
    ) -> Result<Vec<f64>, EvalError> {
        let mut memo = self.reuse.begin(meter)?;
        charge_evaluation_allocation(meter, self.programs.len())?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.programs.len())
            .map_err(|_| EvalError::AllocationFailure)?;
        for (index, program) in self.programs.iter().enumerate() {
            output.push(
                self.reuse
                    .evaluate(index, program, point, &mut memo, meter)?,
            );
        }
        meter.checkpoint()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(ops: Vec<CompiledOp>) -> CompiledExpr {
        CompiledExpr::try_from_ops(ops).unwrap()
    }

    fn rejects(programs: &[&CompiledExpr], definition: Location, location: Location) {
        assert!(matches!(
            verify_ranges(
                programs,
                &[definition],
                &[ReuseRange { location, slot: 0 }],
                &mut Unbounded
            ),
            Err(CompileError::InvalidSharedEvaluationPlan),
        ));
    }

    #[test]
    fn independent_reuse_admission_rejects_operator_axis_and_float_bit_mutants() {
        let base = scalar(vec![CompiledOp::LoadVar(0), CompiledOp::Pow(0.0)]);
        for mutant in [
            scalar(vec![CompiledOp::LoadVar(0), CompiledOp::Pow(-0.0)]),
            scalar(vec![CompiledOp::LoadVar(1), CompiledOp::Pow(0.0)]),
            scalar(vec![CompiledOp::LoadVar(0), CompiledOp::Sin]),
        ] {
            rejects(
                &[&base, &mutant],
                Location {
                    program: 0,
                    start: 0,
                    end: 1,
                },
                Location {
                    program: 1,
                    start: 0,
                    end: 1,
                },
            );
        }
    }

    #[test]
    fn independent_reuse_admission_rejects_branch_entry_into_a_pure_span() {
        let program = scalar(vec![
            CompiledOp::LoadConst(1.0),
            CompiledOp::JumpIfFalse(4),
            CompiledOp::LoadConst(2.0),
            CompiledOp::Jump(5),
            CompiledOp::LoadConst(3.0),
            CompiledOp::Sin,
        ]);
        // This slice is a complete pure expression, but a different branch
        // can enter directly at its sin instruction with a different value.
        let range = Location {
            program: 0,
            start: 4,
            end: 5,
        };
        rejects(&[&program], range, range);
    }

    #[test]
    fn independent_reuse_admission_rejects_incomplete_and_out_of_bounds_spans() {
        let program = scalar(vec![
            CompiledOp::LoadVar(0),
            CompiledOp::LoadVar(1),
            CompiledOp::Sin,
            CompiledOp::Add(2),
        ]);
        for (start, end) in [(0, 2), (2, 3), (0, 7), (3, 1)] {
            let range = Location {
                program: 0,
                start,
                end,
            };
            rejects(&[&program], range, range);
        }
        let range = Location {
            program: 2,
            start: 0,
            end: 1,
        };
        rejects(&[&program], range, range);
    }
}
