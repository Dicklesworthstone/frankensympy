//! Governed normalization for numerical derivative recipes.
//!
//! This boundary retains the ordinary Full normal form while putting exact
//! coefficient work, including held integer powers, on the caller's ledger.
//! Its intrinsic limits also apply to explicitly unbounded convenience callers.
//! Public symbolic simplification keeps its existing policy and representation.

use crate::{SimplifyError, is_explicitly_non_finite_exponent, is_total_expr};
use fsym_budget::{BudgetError, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError};
use fsym_core::{BigRational, Expr};
use fsym_rational::RationalArithmeticError;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::mem::size_of;
use std::sync::Arc;

const MAX_BITS: usize = 4096;
const MAX_NODES: usize = 8192;
const MAX_DEPTH: usize = 128;
const MAX_NAME_BYTES: usize = 4096;
const MAX_VISITS: usize = 131_072;
const MAX_STEPS: usize = 20_000_000;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_ALLOCATIONS: usize = 1_000_000;

fn limit(resource: &'static str, maximum: usize) -> SimplifyError {
    SimplifyError::NumericLimitExceeded {
        resource,
        limit: maximum,
    }
}

fn arithmetic_error(error: RationalArithmeticError) -> SimplifyError {
    match error {
        RationalArithmeticError::Meter(error) => SimplifyError::from_meter(error),
        RationalArithmeticError::SizeOverflow => limit("arithmetic size", MAX_BITS),
        RationalArithmeticError::AllocationFailure => SimplifyError::AllocationFailure,
        RationalArithmeticError::InvariantViolation(reason) => {
            SimplifyError::ArithmeticInvariant(reason)
        }
        RationalArithmeticError::DivisionByZero => {
            SimplifyError::ArithmeticInvariant("unexpected division by zero")
        }
    }
}

/// The intrinsic envelope cannot consume or reset a caller's allowance. Each
/// atomic charge first checks the local envelope, then the caller, and only
/// commits the local counters after the caller has accepted the complete batch.
struct NumericMeter<'a, M: BudgetMeter + ?Sized> {
    caller: &'a mut M,
    used: [u64; DIMENSION_COUNT],
    failure: Option<(&'static str, usize)>,
}

impl<M: BudgetMeter + ?Sized> BudgetMeter for NumericMeter<'_, M> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.caller.checkpoint()?;
        let mut next = self.used;
        for &(dimension, amount) in charges {
            next[dimension.index()] =
                next[dimension.index()]
                    .checked_add(amount)
                    .ok_or(MeterError::Budget(BudgetError::ChargeOverflow {
                        dimension,
                    }))?;
        }
        for (dimension, maximum, resource) in [
            (Dimension::ComputeSteps, MAX_STEPS, "compute steps"),
            (Dimension::MemoryBytes, MAX_BYTES, "requested storage"),
            (Dimension::AllocationCount, MAX_ALLOCATIONS, "allocations"),
        ] {
            if next[dimension.index()] > maximum as u64 {
                self.failure = Some((resource, maximum));
                return Err(MeterError::Budget(BudgetError::Exhausted {
                    dimension,
                    requested: next[dimension.index()] - self.used[dimension.index()],
                    remaining: maximum as u64 - self.used[dimension.index()],
                }));
            }
        }
        self.caller.charge_batch(charges)?;
        self.used = next;
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.caller.checkpoint()
    }
}

#[derive(Clone, Copy, Default)]
struct Shape {
    nodes: usize,
    bytes: usize,
    display_work: usize,
    display_bytes: usize,
    display_allocations: usize,
}

type OrderedTerms<T> = Vec<(Expr, T)>;

struct Normalizer<'a, M: BudgetMeter + ?Sized> {
    meter: NumericMeter<'a, M>,
    visits: usize,
}

/// Preflight the numerical normalization envelope without cloning, rewriting
/// or evaluating any source subtree. Derivative generators and independent
/// reference tables use this before their first scalar subtraction or copy.
pub fn admit_numeric_expression_with<M: BudgetMeter + ?Sized>(
    expression: &Expr,
    meter: &mut M,
) -> Result<(), SimplifyError> {
    let mut normalizer = Normalizer {
        meter: NumericMeter {
            caller: meter,
            used: [0; DIMENSION_COUNT],
            failure: None,
        },
        visits: 0,
    };
    let result = normalizer.inspect(expression).map(|_| ());
    normalizer
        .meter
        .caller
        .checkpoint()
        .map_err(SimplifyError::from_meter)?;
    match (result, normalizer.meter.failure) {
        (Err(_), Some((resource, maximum))) => Err(limit(resource, maximum)),
        (result, _) => result,
    }
}

/// Normalize a numerical derivative under one cumulative caller meter.
///
/// Full-mode coefficient collection, integer-power folding, exponential
/// combination and the registered Pythagorean families retain their ordinary
/// exact forms. Coefficient arithmetic and rational comparison use governed
/// kernels; a restricted historical scalar constructor is preflighted and
/// charged before its bounded synchronous replay. Input/output
/// trees are limited to 8192 nodes, depth 128, 4096-bit scalar components and
/// 4096-byte identifiers; repeated normalization additionally has fixed work,
/// allocation and requested-storage limits. A refusal returns no partial tree
/// and never retries the public unbounded simplifier. Original source-domain
/// checks remain the numerical program's responsibility.
///
/// Nested-power scalar construction additionally admits at most 32 nodes and
/// depth 8, with affine symbolic exponents and small positive numeric radicals.
/// Unsupported constructor shapes return a typed operational refusal before
/// evaluation; accepted trees retain the historical constructor's exact form.
///
/// Structural ordering uses a fallible typed comparator and governed rational
/// comparison. The existing factor-key formatting remains a precharged bounded
/// synchronous operation; these logical charges do not claim allocator-RSS
/// measurement or interruption inside the decimal formatter.
pub fn simplify_numeric_with<M: BudgetMeter + ?Sized>(
    expression: &Expr,
    meter: &mut M,
) -> Result<Expr, SimplifyError> {
    let mut normalizer = Normalizer {
        meter: NumericMeter {
            caller: meter,
            used: [0; DIMENSION_COUNT],
            failure: None,
        },
        visits: 0,
    };
    let result = (|| {
        normalizer.inspect(expression)?;
        let value = normalizer.normalize(expression, 0)?;
        normalizer.inspect(&value)?;
        Ok(value)
    })();
    normalizer
        .meter
        .caller
        .checkpoint()
        .map_err(SimplifyError::from_meter)?;
    match (result, normalizer.meter.failure) {
        (Err(_), Some((resource, maximum))) => Err(limit(resource, maximum)),
        (result, _) => result,
    }
}

impl<M: BudgetMeter + ?Sized> Normalizer<'_, M> {
    fn charge(
        &mut self,
        steps: usize,
        bytes: usize,
        allocations: usize,
    ) -> Result<(), SimplifyError> {
        self.meter.checkpoint().map_err(SimplifyError::from_meter)?;
        let mut charges = [(Dimension::ComputeSteps, 0); 3];
        let mut count = 0;
        for (dimension, amount) in [
            (Dimension::ComputeSteps, steps),
            (Dimension::MemoryBytes, bytes),
            (Dimension::AllocationCount, allocations),
        ] {
            if amount != 0 {
                charges[count] = (dimension, amount as u64);
                count += 1;
            }
        }
        if count != 0 {
            self.meter
                .charge_batch(&charges[..count])
                .map_err(SimplifyError::from_meter)?;
        }
        Ok(())
    }

    fn inspect(&mut self, expression: &Expr) -> Result<Shape, SimplifyError> {
        fn negative_leading(expression: &Expr) -> bool {
            fn numeric_negative(expression: &Expr) -> bool {
                match expression {
                    Expr::Integer(value) => value.is_negative(),
                    Expr::Rational(value) => value.numer().is_negative(),
                    _ => false,
                }
            }
            match expression {
                Expr::Integer(_) | Expr::Rational(_) => numeric_negative(expression),
                Expr::Mul(factors) => factors.first().is_some_and(numeric_negative),
                _ => false,
            }
        }

        fn visit<M: BudgetMeter + ?Sized>(
            normalizer: &mut Normalizer<'_, M>,
            expression: &Expr,
            depth: usize,
            formatter_copies: usize,
            shape: &mut Shape,
        ) -> Result<(), SimplifyError> {
            normalizer.charge(1, 0, 0)?;
            shape.nodes += 1;
            if depth > MAX_DEPTH {
                return Err(limit("expression depth", MAX_DEPTH));
            }
            if shape.nodes > MAX_NODES {
                return Err(limit("expression nodes", MAX_NODES));
            }
            let (payload, display_work, display_bytes, display_allocations) = match expression {
                Expr::Integer(value) => {
                    if value.bits() > MAX_BITS as u64 {
                        return Err(limit("coefficient bits", MAX_BITS));
                    }
                    let limbs = value.limb_count() as usize;
                    let bits = value.bits() as usize;
                    (
                        limbs * 8,
                        64 * limbs * limbs + bits + 24,
                        bits + 24,
                        limbs * limbs + 2,
                    )
                }
                Expr::Rational(value) => {
                    let bits = value.numer().bits().max(value.denom().bits());
                    if bits > MAX_BITS as u64 {
                        return Err(limit("coefficient bits", MAX_BITS));
                    }
                    let numerator = value.numer().limb_count() as usize;
                    let denominator = value.denom().limb_count() as usize;
                    let squares = numerator * numerator + denominator * denominator;
                    let digits = (value.numer().bits() + value.denom().bits()) as usize + 32;
                    (
                        (numerator + denominator) * 8,
                        64 * squares + digits,
                        digits,
                        squares + 4,
                    )
                }
                Expr::Sym(symbol) => {
                    if symbol.name.len() > MAX_NAME_BYTES {
                        return Err(limit("identifier bytes", MAX_NAME_BYTES));
                    }
                    (
                        symbol.name.len(),
                        symbol.name.len() + 32,
                        symbol.name.len() + 32,
                        2,
                    )
                }
                Expr::Function(name, _) => {
                    if name.len() > MAX_NAME_BYTES {
                        return Err(limit("identifier bytes", MAX_NAME_BYTES));
                    }
                    (name.len(), name.len() + 32, name.len() + 32, 2)
                }
                _ => (0, 32, 32, 2),
            };
            normalizer.charge(payload.div_ceil(8).max(1), 0, 0)?;
            let owned = size_of::<Expr>().saturating_add(payload);
            shape.bytes = shape.bytes.saturating_add(owned);
            // Core Display owns a copy of every negative-leading Add term,
            // then copies its non-leading Mul factors again in fmt_magnitude.
            // Two whole-subtree copies bound both operations, including the
            // negated scalar and Vec. Nested negative terms add copy contexts;
            // ordinary unsigned trees retain their previous display envelope.
            // Inspection only computes this metadata; rendering boundaries
            // charge it before invoking a formatter.
            shape.display_work = shape
                .display_work
                .saturating_add(display_work)
                .saturating_add(
                    formatter_copies.saturating_mul(owned.div_ceil(8).saturating_add(1)),
                );
            shape.display_bytes = shape
                .display_bytes
                .saturating_add(display_bytes)
                .saturating_add(formatter_copies.saturating_mul(owned));
            // A node can own both a function-name/number buffer and a child
            // Vec/second rational component. Arc children may share storage;
            // accounting them as copies is a conservative upper bound.
            shape.display_allocations = shape
                .display_allocations
                .saturating_add(display_allocations)
                .saturating_add(formatter_copies.saturating_mul(2));
            match expression {
                Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                    if shape.nodes.saturating_add(children.len()) > MAX_NODES {
                        return Err(limit("expression nodes", MAX_NODES));
                    }
                    let addition = matches!(expression, Expr::Add(_));
                    for child in children {
                        let copies = if addition && negative_leading(child) {
                            formatter_copies.saturating_add(2)
                        } else {
                            formatter_copies
                        };
                        visit(normalizer, child, depth + 1, copies, shape)?;
                    }
                }
                Expr::Pow(base, exponent) => {
                    visit(normalizer, base, depth + 1, formatter_copies, shape)?;
                    visit(normalizer, exponent, depth + 1, formatter_copies, shape)?;
                }
                _ => {}
            }
            Ok(())
        }
        let mut shape = Shape::default();
        visit(self, expression, 0, 0, &mut shape)?;
        Ok(shape)
    }

    fn clone_expr(&mut self, expression: &Expr) -> Result<Expr, SimplifyError> {
        let shape = self.inspect(expression)?;
        self.charge(
            shape.nodes + shape.bytes.div_ceil(8),
            shape.bytes,
            shape.nodes,
        )?;
        Ok(expression.clone())
    }

    fn reserve<T>(&mut self, values: &mut Vec<T>, additional: usize) -> Result<(), SimplifyError> {
        if values.len().saturating_add(additional) > MAX_NODES {
            return Err(limit("work-buffer entries", MAX_NODES));
        }
        self.charge(
            additional,
            additional * size_of::<T>(),
            usize::from(additional != 0),
        )?;
        values
            .try_reserve_exact(additional)
            .map_err(|_| SimplifyError::AllocationFailure)
    }

    fn buffer<T>(&mut self, count: usize) -> Result<Vec<T>, SimplifyError> {
        let mut result = Vec::new();
        self.reserve(&mut result, count)?;
        Ok(result)
    }

    fn number(&mut self, expression: &Expr) -> Result<Option<BigRational>, SimplifyError> {
        Ok(match expression {
            Expr::Integer(_) | Expr::Rational(_) => {
                let copy = self.clone_expr(expression)?;
                match copy {
                    Expr::Integer(value) => Some(BigRational::from_integer(value)),
                    Expr::Rational(value) => Some(value),
                    _ => unreachable!("numeric clone retains its variant"),
                }
            }
            _ => None,
        })
    }

    fn scalar(&mut self, value: BigRational) -> Result<BigRational, SimplifyError> {
        self.charge(1, 0, 0)?;
        if value.numer().bits().max(value.denom().bits()) > MAX_BITS as u64 {
            return Err(limit("coefficient bits", MAX_BITS));
        }
        Ok(value)
    }

    fn add(
        &mut self,
        left: &BigRational,
        right: &BigRational,
    ) -> Result<BigRational, SimplifyError> {
        let result = left
            .metered_add(right, &mut self.meter)
            .map_err(arithmetic_error)?;
        self.scalar(result)
    }

    fn multiply(
        &mut self,
        left: &BigRational,
        right: &BigRational,
    ) -> Result<BigRational, SimplifyError> {
        let result = left
            .metered_mul(right, &mut self.meter)
            .map_err(arithmetic_error)?;
        self.scalar(result)
    }

    fn scalar_expr(&mut self, value: BigRational) -> Result<Expr, SimplifyError> {
        let value = self.scalar(value)?;
        let bytes = (value.numer().limb_count() + value.denom().limb_count()) as usize * 8;
        self.charge(bytes.div_ceil(8).max(1), bytes, 1)?;
        Ok(crate::rational_expr(value))
    }

    // Derived Expr ordering is lexicographic on vectors; the existing
    // canonical factor comparator's structural tie break compares lengths
    // first. Preserve both contracts, using governed rational comparison.
    fn compare(
        &mut self,
        left: &Expr,
        right: &Expr,
        length_first: bool,
    ) -> Result<Ordering, SimplifyError> {
        fn rank(expression: &Expr) -> u8 {
            match expression {
                Expr::Sym(_) => 0,
                Expr::Integer(_) => 1,
                Expr::Rational(_) => 2,
                Expr::Const(_) => 3,
                Expr::Add(_) => 4,
                Expr::Mul(_) => 5,
                Expr::Pow(_, _) => 6,
                Expr::Function(_, _) => 7,
            }
        }
        self.charge(1, 0, 0)?;
        let ordering = rank(left).cmp(&rank(right));
        if ordering != Ordering::Equal {
            return Ok(ordering);
        }
        match (left, right) {
            (Expr::Sym(left), Expr::Sym(right)) => {
                self.charge(left.name.len() + right.name.len() + 64, 0, 0)?;
                Ok(left.cmp(right))
            }
            (Expr::Integer(left), Expr::Integer(right)) => {
                self.charge((left.limb_count() + right.limb_count()) as usize + 1, 0, 0)?;
                Ok(left.cmp(right))
            }
            (Expr::Rational(left), Expr::Rational(right)) => left
                .metered_cmp(right, &mut self.meter)
                .map_err(arithmetic_error),
            (Expr::Const(left), Expr::Const(right)) => Ok(left.cmp(right)),
            (Expr::Add(left), Expr::Add(right)) | (Expr::Mul(left), Expr::Mul(right)) => {
                self.compare_lists(left, right, length_first)
            }
            (Expr::Pow(lb, le), Expr::Pow(rb, re)) => {
                let ordering = self.compare(lb, rb, length_first)?;
                if ordering == Ordering::Equal {
                    self.compare(le, re, length_first)
                } else {
                    Ok(ordering)
                }
            }
            (Expr::Function(ln, la), Expr::Function(rn, ra)) => {
                self.charge(ln.len() + rn.len() + 1, 0, 0)?;
                let ordering = ln.cmp(rn);
                if ordering == Ordering::Equal {
                    self.compare_lists(la, ra, length_first)
                } else {
                    Ok(ordering)
                }
            }
            _ => unreachable!("equal expression ranks retain the same variant"),
        }
    }

    fn compare_lists(
        &mut self,
        left: &[Expr],
        right: &[Expr],
        length_first: bool,
    ) -> Result<Ordering, SimplifyError> {
        self.charge(1, 0, 0)?;
        if length_first && left.len() != right.len() {
            return Ok(left.len().cmp(&right.len()));
        }
        for (left, right) in left.iter().zip(right) {
            let ordering = self.compare(left, right, length_first)?;
            if ordering != Ordering::Equal {
                return Ok(ordering);
            }
        }
        Ok(left.len().cmp(&right.len()))
    }

    fn position<T>(
        &mut self,
        entries: &[(Expr, T)],
        key: &Expr,
    ) -> Result<Result<usize, usize>, SimplifyError> {
        let mut lower = 0;
        let mut upper = entries.len();
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            match self.compare(&entries[middle].0, key, false)? {
                Ordering::Less => lower = middle + 1,
                Ordering::Equal => return Ok(Ok(middle)),
                Ordering::Greater => upper = middle,
            }
        }
        Ok(Err(lower))
    }

    fn insert<T>(
        &mut self,
        entries: &mut OrderedTerms<T>,
        index: usize,
        key: Expr,
        value: T,
    ) -> Result<(), SimplifyError> {
        self.reserve(entries, 1)?;
        self.charge(entries.len() - index + 1, 0, 0)?;
        entries.insert(index, (key, value));
        Ok(())
    }

    fn rendered(&mut self, expression: &Expr, padded: bool) -> Result<String, SimplifyError> {
        let shape = self.inspect(expression)?;
        // Decimal conversion and core Display's negative-term subtree copies
        // are bounded synchronous operations. Both envelopes are accumulated
        // by inspect() and charged before the formatter can allocate.
        self.charge(
            shape.display_work,
            shape
                .display_bytes
                .saturating_mul(4)
                .saturating_add(shape.display_allocations.saturating_mul(32)),
            shape.display_allocations,
        )?;
        Ok(match expression {
            Expr::Integer(value) if padded => format!("{value:0>20}"),
            Expr::Rational(value) if padded => format!("{value:0>20}"),
            _ => format!("{expression}"),
        })
    }

    fn factor_key(&mut self, expression: &Expr) -> Result<(u8, String, String), SimplifyError> {
        self.charge(1, 0, 0)?;
        Ok(match expression {
            Expr::Integer(_) | Expr::Rational(_) => {
                (0, self.rendered(expression, true)?, String::new())
            }
            Expr::Sym(_) => (1, self.rendered(expression, false)?, String::new()),
            Expr::Pow(base, exponent) => {
                let base_key = self.factor_key(base)?;
                let exponent_key =
                    self.rendered(exponent, matches!(exponent.as_ref(), Expr::Integer(_)))?;
                self.charge(base_key.1.len() + 2, base_key.1.len() + 2, 1)?;
                (2, format!("{}|{}", base_key.0, base_key.1), exponent_key)
            }
            Expr::Function(name, arguments) => {
                let mut keys = self.buffer(arguments.len())?;
                let mut bytes = arguments.len().saturating_sub(1);
                for argument in arguments {
                    let key = self.factor_key(argument)?.1;
                    bytes = bytes.saturating_add(key.len());
                    keys.push(key);
                }
                self.charge(bytes + name.len(), bytes + name.len(), 2)?;
                (3, name.clone(), keys.join(","))
            }
            _ => (4, self.rendered(expression, false)?, String::new()),
        })
    }

    fn sort_factors(&mut self, factors: &mut [Expr]) -> Result<(), SimplifyError> {
        if factors.len() < 2 {
            return Ok(());
        }
        let mut keys = self.buffer(factors.len())?;
        for factor in factors.iter() {
            keys.push(self.factor_key(factor)?);
        }
        // Cache the exact legacy keys once. Fallible stable insertion sorting
        // charges only comparisons actually made and keeps typed identity in
        // the structural tie break, including identical printed symbol names.
        for index in 1..factors.len() {
            let mut current = index;
            while current != 0 {
                let left = &keys[current - 1];
                let right = &keys[current];
                self.charge(
                    left.1.len() + left.2.len() + right.1.len() + right.2.len() + 1,
                    0,
                    0,
                )?;
                let mut ordering = left.cmp(right);
                if ordering == Ordering::Equal {
                    ordering = self.compare(&factors[current - 1], &factors[current], true)?;
                }
                if ordering != Ordering::Greater {
                    break;
                }
                self.charge(1, 0, 0)?;
                keys.swap(current - 1, current);
                factors.swap(current - 1, current);
                current -= 1;
            }
        }
        Ok(())
    }

    fn split_coefficient(&mut self, term: Expr) -> Result<(BigRational, Expr), SimplifyError> {
        match term {
            Expr::Integer(value) => Ok((BigRational::from_integer(value), Expr::from_i64(1))),
            Expr::Rational(value) => Ok((value, Expr::from_i64(1))),
            Expr::Mul(factors) => {
                let original = self.clone_expr(&Expr::Mul(factors))?;
                let Expr::Mul(mut pending) = self.clone_expr(&original)? else {
                    unreachable!("Mul clone retains its variant")
                };
                let mut coefficient = BigRational::one();
                let mut rest = self.buffer(pending.len())?;
                while let Some(factor) = pending.pop() {
                    self.charge(1, 0, 0)?;
                    if let Expr::Mul(nested) = factor {
                        self.reserve(&mut pending, nested.len())?;
                        pending.extend(nested);
                    } else if let Some(value) = self.number(&factor)? {
                        coefficient = self.multiply(&coefficient, &value)?;
                    } else {
                        self.reserve(&mut rest, 1)?;
                        rest.push(factor);
                    }
                }
                if coefficient.is_zero() && !rest.iter().all(is_total_expr) {
                    return Ok((BigRational::one(), original));
                }
                self.sort_factors(&mut rest)?;
                let key = match rest.len() {
                    0 => Expr::from_i64(1),
                    1 => rest.pop().expect("length checked"),
                    _ => Expr::Mul(rest),
                };
                Ok((coefficient, key))
            }
            other => Ok((BigRational::one(), other)),
        }
    }

    fn collect_terms(&mut self, mut pending: Vec<Expr>) -> Result<Expr, SimplifyError> {
        let mut collected: OrderedTerms<BigRational> = Vec::new();
        let mut constant = BigRational::zero();
        while let Some(term) = pending.pop() {
            self.charge(1, 0, 0)?;
            if let Expr::Add(children) = term {
                self.reserve(&mut pending, children.len())?;
                pending.extend(children);
                continue;
            }
            let (coefficient, key) = self.split_coefficient(term)?;
            if coefficient.is_zero() {
                continue;
            }
            if key.is_one() {
                constant = self.add(&constant, &coefficient)?;
            } else {
                match self.position(&collected, &key)? {
                    Ok(index) => {
                        collected[index].1 = self.add(&collected[index].1, &coefficient)?
                    }
                    Err(index) => self.insert(&mut collected, index, key, coefficient)?,
                }
            }
        }
        let mut pure = self.buffer(collected.len() + 1)?;
        let mut scaled = self.buffer(collected.len())?;
        for (key, coefficient) in collected {
            self.charge(1, 0, 0)?;
            if coefficient.is_zero() {
                continue;
            }
            if coefficient.is_one() {
                pure.push(key);
            } else {
                let mut factors = self.buffer(2)?;
                factors.push(self.scalar_expr(coefficient)?);
                match key {
                    Expr::Mul(children) => {
                        self.reserve(&mut factors, children.len())?;
                        factors.extend(children);
                        self.sort_factors(&mut factors)?;
                    }
                    other => factors.push(other),
                }
                scaled.push(Expr::Mul(factors));
            }
        }
        let mut output = self.buffer(pure.len() + scaled.len() + 1)?;
        if !constant.is_zero() || (pure.is_empty() && scaled.is_empty()) {
            output.push(self.scalar_expr(constant)?);
        }
        output.extend(pure);
        output.extend(scaled);
        Ok(match output.len() {
            1 => output.pop().expect("length checked"),
            _ => Expr::Add(output),
        })
    }

    fn fold_pairs(&mut self, terms: &[Expr]) -> Result<Option<Vec<Expr>>, SimplifyError> {
        type Pair = Option<[Option<(BigRational, usize)>; 2]>;
        let mut families: [OrderedTerms<Pair>; 4] = std::array::from_fn(|_| Vec::new());
        for (index, term) in terms.iter().enumerate() {
            let copy = self.clone_expr(term)?;
            let (coefficient, key) = self.split_coefficient(copy)?;
            if coefficient.is_zero() {
                continue;
            }
            let Expr::Pow(base, exponent) = key else {
                continue;
            };
            if !matches!(exponent.as_ref(), Expr::Integer(value) if value.to_i64() == Some(2)) {
                continue;
            }
            let Expr::Function(name, arguments) = base.as_ref() else {
                continue;
            };
            if arguments.len() != 1 {
                continue;
            }
            let Some((family, slot)) = crate::rewrite::pythagorean_family_slot(name) else {
                continue;
            };
            let argument = self.clone_expr(&arguments[0])?;
            let slots = &mut families[usize::from(family)];
            match self.position(slots, &argument)? {
                Err(position) => {
                    let mut pair = [None, None];
                    pair[usize::from(slot)] = Some((coefficient, index));
                    self.insert(slots, position, argument, Some(pair))?;
                }
                Ok(position) => {
                    if matches!(&slots[position].1, Some(pair) if pair[usize::from(slot)].is_some())
                    {
                        slots[position].1 = None;
                    } else if let Some(pair) = &mut slots[position].1 {
                        pair[usize::from(slot)] = Some((coefficient, index));
                    }
                }
            }
        }
        let mut remove = self.buffer(terms.len())?;
        remove.resize(terms.len(), false);
        let mut constants = self.buffer(terms.len())?;
        for (family, slots) in families.into_iter().enumerate() {
            for pair in slots.into_iter().filter_map(|(_, pair)| pair) {
                let [Some((left, left_index)), Some((right, right_index))] = pair else {
                    continue;
                };
                self.charge(
                    (left.numer().limb_count()
                        + left.denom().limb_count()
                        + right.numer().limb_count()
                        + right.denom().limb_count()) as usize
                        + 1,
                    0,
                    0,
                )?;
                let folds = if family == 0 {
                    left == right
                } else {
                    self.add(&left, &right)?.is_zero()
                };
                if folds {
                    remove[left_index] = true;
                    remove[right_index] = true;
                    constants.push(self.scalar_expr(left)?);
                }
            }
        }
        if constants.is_empty() {
            return Ok(None);
        }
        let mut output = self.buffer(terms.len() + constants.len())?;
        for (index, term) in terms.iter().enumerate() {
            if !remove[index] {
                output.push(self.clone_expr(term)?);
            }
        }
        output.extend(constants);
        Ok(Some(output))
    }

    fn normalize_children(
        &mut self,
        children: &[Expr],
        depth: usize,
    ) -> Result<Vec<Expr>, SimplifyError> {
        let mut output = self.buffer(children.len())?;
        for child in children {
            output.push(self.normalize(child, depth + 1)?);
        }
        Ok(output)
    }

    fn normalize_product(
        &mut self,
        mut pending: Vec<Expr>,
        depth: usize,
    ) -> Result<Expr, SimplifyError> {
        let mut coefficient = BigRational::one();
        let mut rest = self.buffer(pending.len())?;
        while let Some(factor) = pending.pop() {
            self.charge(1, 0, 0)?;
            if let Expr::Mul(children) = factor {
                self.reserve(&mut pending, children.len())?;
                pending.extend(children);
            } else if let Some(value) = self.number(&factor)? {
                coefficient = self.multiply(&coefficient, &value)?;
            } else {
                self.reserve(&mut rest, 1)?;
                rest.push(factor);
            }
        }
        if coefficient.is_zero() && rest.iter().all(is_total_expr) {
            return Ok(Expr::from_i64(0));
        }
        let mut exponential_arguments = self.buffer(rest.len())?;
        let mut other_factors = self.buffer(rest.len())?;
        for factor in rest {
            if let Expr::Function(name, arguments) = &factor
                && name == "exp"
                && arguments.len() == 1
            {
                exponential_arguments.push(self.clone_expr(&arguments[0])?);
            } else {
                other_factors.push(factor);
            }
        }
        if exponential_arguments.len() > 1 {
            let sum = self.collect_terms(exponential_arguments)?;
            let combined = self.normalize(&Expr::Function("exp".into(), vec![sum]), depth + 1)?;
            if !combined.is_one() {
                self.reserve(&mut other_factors, 1)?;
                other_factors.push(combined);
            }
        } else if !exponential_arguments.is_empty() {
            self.reserve(&mut other_factors, 1)?;
            other_factors.push(Expr::Function("exp".into(), exponential_arguments));
        }
        if other_factors.is_empty() {
            return self.scalar_expr(coefficient);
        }
        let mut powers: OrderedTerms<Vec<Expr>> = Vec::new();
        for factor in other_factors {
            let (base, exponent) = match factor {
                Expr::Pow(base, exponent) => (self.clone_expr(&base)?, self.clone_expr(&exponent)?),
                other => (other, Expr::from_i64(1)),
            };
            match self.position(&powers, &base)? {
                Ok(index) => {
                    self.reserve(&mut powers[index].1, 1)?;
                    powers[index].1.push(exponent);
                }
                Err(index) => {
                    let mut exponents = self.buffer(1)?;
                    exponents.push(exponent);
                    self.insert(&mut powers, index, base, exponents)?;
                }
            }
        }
        let mut parts = self.buffer(powers.len() + 1)?;
        for (base, mut exponents) in powers {
            let exponent = if exponents.len() == 1 {
                exponents.pop().expect("length checked")
            } else {
                let sum = self.collect_terms(exponents)?;
                self.normalize(&sum, depth + 1)?
            };
            if exponent.is_zero() {
                continue;
            }
            if exponent.is_one() {
                parts.push(base);
            } else {
                self.charge(1, 2 * (size_of::<Expr>() + 16), 2)?;
                let power = Expr::Pow(Arc::new(base), Arc::new(exponent));
                let folded = self.normalize(&power, depth + 1)?;
                if !folded.is_one() {
                    parts.push(folded);
                }
            }
        }
        self.sort_factors(&mut parts)?;
        let mut output = self.buffer(parts.len() + 1)?;
        if !coefficient.is_one() {
            output.push(self.scalar_expr(coefficient)?);
        }
        output.extend(parts);
        Ok(match output.len() {
            0 => Expr::from_i64(1),
            1 => output.pop().expect("length checked"),
            _ => Expr::Mul(output),
        })
    }

    /// The Full normal form flattens nested integer powers through Expr's
    /// scalar constructor. Replay that exact constructor only after admitting
    /// a small arithmetic grammar and charging every synchronous substrate
    /// operation; returning a different equivalent exponent would invalidate
    /// historical derivative recipes.
    fn scale_exponent(
        &mut self,
        exponent: &Expr,
        scalar: &Expr,
        _depth: usize,
    ) -> Result<Expr, SimplifyError> {
        let shape = self.inspect(exponent)?;
        self.charge(shape.nodes, 0, 0)?;
        if scalar.is_one() {
            return self.clone_expr(exponent);
        }
        let pole = fsym_core::has_pole(exponent);
        if scalar.is_zero() && !pole {
            return Ok(Expr::from_i64(0));
        }
        if let Some(value) = self.number(exponent)? {
            let factor = self
                .number(scalar)?
                .ok_or(SimplifyError::ArithmeticInvariant(
                    "non-numeric nested-power scalar",
                ))?;
            let product = self.multiply(&value, &factor)?;
            return self.scalar_expr(product);
        }
        if let Expr::Mul(factors) = exponent {
            // Core's zero test uses explicit poles, whereas Full normalization
            // also retains domain-sensitive factors such as log(x).
            if !pole && factors.iter().any(Expr::is_zero) {
                return Ok(Expr::from_i64(0));
            }
            if let [coefficient, sum @ Expr::Add(_)] = factors.as_slice()
                && let Some(coefficient) = self.number(coefficient)?
            {
                let factor = self
                    .number(scalar)?
                    .ok_or(SimplifyError::ArithmeticInvariant(
                        "non-numeric nested-power scalar",
                    ))?;
                let product = self.multiply(&coefficient, &factor)?;
                if !product.is_zero() {
                    if product.is_one() {
                        return self.clone_expr(sum);
                    }
                    let mut factors = self.buffer(2)?;
                    factors.push(self.scalar_expr(product)?);
                    factors.push(self.clone_expr(sum)?);
                    return Ok(Expr::Mul(factors));
                }
            }
        }

        const CONSTRUCTOR_NODES: usize = 32;
        const CONSTRUCTOR_DEPTH: usize = 8;
        let scalar_shape = self.inspect(scalar)?;
        let nodes = shape.nodes.saturating_add(scalar_shape.nodes);
        if nodes > CONSTRUCTOR_NODES {
            return Err(limit("scalar constructor nodes", CONSTRUCTOR_NODES));
        }
        let mut scalar_bits = 0usize;
        self.constructor_payload(exponent, 0, CONSTRUCTOR_DEPTH, &mut scalar_bits)?;
        self.constructor_payload(scalar, 0, CONSTRUCTOR_DEPTH, &mut scalar_bits)?;
        // Numeric radicals are the only admitted constructor path that can
        // materialize a new exact power. The tuple records aggregate base
        // component bits, maximum |p| and maximum q.
        let mut radicals = (0usize, 0u64, 1u64);
        match exponent {
            Expr::Add(_) => self.constructor_affine(exponent)?,
            Expr::Mul(factors) => {
                for (index, factor) in factors.iter().enumerate() {
                    self.constructor_factor(factor, &mut radicals)?;
                    if let Expr::Pow(base, _) = factor
                        && matches!(base.as_ref(), Expr::Integer(_) | Expr::Rational(_))
                    {
                        // Same-base grouping happens before same-exponent
                        // radical grouping in core. It could create a much
                        // larger denominator (1/13 + 1/16 = 29/208), so never
                        // infer a merged-exponent bound from max(p), max(q).
                        for previous in &factors[..index] {
                            if let Expr::Pow(previous_base, _) = previous
                                && self.compare(base, previous_base, false)? == Ordering::Equal
                            {
                                return Err(limit("scalar constructor repeated numeric base", 1));
                            }
                        }
                    }
                }
            }
            other => self.constructor_factor(other, &mut radicals)?,
        }
        let radical_bits = radicals
            .0
            .saturating_mul(radicals.1.max(1) as usize)
            .saturating_mul(radicals.2 as usize);
        if radical_bits > MAX_BITS {
            return Err(limit("radical constructor intermediate bits", MAX_BITS));
        }
        if radicals.0 != 0 {
            self.charge_radical_constructor(radical_bits)?;
        }

        // The admitted affine exponents only multiply a literal by one
        // symbol and add those coefficients. Each original scalar component
        // participates in a product once; collection adds at most log2(nodes)
        // bits. Double the bound for unreduced rational cross-products.
        // Radical integer powers have their independent envelope above.
        let bits = scalar_bits
            .saturating_add(radical_bits)
            .saturating_add(nodes.ilog2() as usize + 3)
            .saturating_mul(2);
        if bits > MAX_BITS {
            return Err(limit("scalar constructor intermediate bits", MAX_BITS));
        }
        let limbs = bits.div_ceil(64).max(1);
        // At most eight complete n^2 passes cover scalar accumulation,
        // affine coefficient extraction/rebuilding, pair grouping, Add
        // collection and repeated canonical comparison. Each pass pays for
        // the full source shape, including sign-aware Display's copies.
        let rounds = nodes.saturating_mul(nodes).saturating_mul(8);
        let arithmetic_allocations = bits.saturating_mul(4).saturating_add(16);
        let arithmetic_work = bits
            .saturating_mul(limbs.saturating_mul(limbs))
            .saturating_mul(64)
            .saturating_add(8);
        let arithmetic_bytes =
            arithmetic_allocations.saturating_mul(limbs.saturating_mul(16).saturating_add(32));
        let display_work = shape.display_work.saturating_add(scalar_shape.display_work);
        let display_bytes = shape
            .display_bytes
            .saturating_add(scalar_shape.display_bytes);
        let display_allocations = shape
            .display_allocations
            .saturating_add(scalar_shape.display_allocations);
        let source_bytes = shape.bytes.saturating_add(scalar_shape.bytes);
        self.charge(
            rounds.saturating_mul(
                arithmetic_work
                    .saturating_add(display_work.saturating_mul(2))
                    .saturating_add(source_bytes.div_ceil(8))
                    .saturating_add(nodes),
            ),
            rounds.saturating_mul(
                arithmetic_bytes
                    .saturating_add(source_bytes.saturating_mul(4))
                    .saturating_add(display_bytes.saturating_mul(8))
                    .saturating_add(display_allocations.saturating_mul(64)),
            ),
            rounds.saturating_mul(
                arithmetic_allocations
                    .saturating_add(nodes.saturating_mul(4))
                    .saturating_add(display_allocations.saturating_mul(2)),
            ),
        )?;
        let value = self.clone_expr(exponent)? * self.clone_expr(scalar)?;
        self.inspect(&value)?;
        self.charge(1, 0, 0)?;
        Ok(value)
    }

    fn constructor_payload(
        &mut self,
        expression: &Expr,
        depth: usize,
        maximum_depth: usize,
        bits: &mut usize,
    ) -> Result<(), SimplifyError> {
        self.charge(1, 0, 0)?;
        if depth > maximum_depth {
            return Err(limit("scalar constructor depth", maximum_depth));
        }
        match expression {
            Expr::Integer(value) => *bits = bits.saturating_add(value.bits() as usize),
            Expr::Rational(value) => {
                *bits = bits
                    .saturating_add(value.numer().bits() as usize)
                    .saturating_add(value.denom().bits() as usize);
            }
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
                for child in children {
                    self.constructor_payload(child, depth + 1, maximum_depth, bits)?;
                }
            }
            Expr::Pow(base, exponent) => {
                self.constructor_payload(base, depth + 1, maximum_depth, bits)?;
                self.constructor_payload(exponent, depth + 1, maximum_depth, bits)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Affine here is deliberately syntactic: typed symbols and exact
    /// coefficients, with no active function, product of symbols, or power.
    /// In particular E/exp grouping can never reach exp(c*log(b)), Gaussian
    /// arithmetic, nested powers, or another elementary function evaluator.
    fn constructor_affine(&mut self, expression: &Expr) -> Result<(), SimplifyError> {
        self.charge(1, 0, 0)?;
        match expression {
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) => Ok(()),
            Expr::Add(terms) => {
                for term in terms {
                    self.constructor_affine(term)?;
                }
                Ok(())
            }
            Expr::Mul(factors)
                if matches!(
                    factors.as_slice(),
                    [Expr::Integer(_) | Expr::Rational(_), Expr::Sym(_)]
                ) =>
            {
                Ok(())
            }
            _ => Err(limit("scalar constructor affine exponent", 1)),
        }
    }

    fn constructor_factor(
        &mut self,
        expression: &Expr,
        radicals: &mut (usize, u64, u64),
    ) -> Result<(), SimplifyError> {
        self.charge(1, 0, 0)?;
        match expression {
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => Ok(()),
            Expr::Function(name, arguments) if name == "exp" => {
                let [argument] = arguments.as_slice() else {
                    return Err(limit("scalar constructor exponential arity", 1));
                };
                self.constructor_affine(argument)
            }
            // These function bodies are opaque to scalar Mul construction.
            Expr::Function(_, _) => Ok(()),
            Expr::Pow(base, exponent) => {
                let base_bits = match base.as_ref() {
                    Expr::Integer(value) if !value.is_negative() && !value.is_zero() => {
                        Some(value.bits() as usize + 1)
                    }
                    Expr::Rational(value) if !value.numer().is_negative() && !value.is_zero() => {
                        Some((value.numer().bits() + value.denom().bits()) as usize)
                    }
                    Expr::Sym(_)
                    | Expr::Const(fsym_core::Constant::Pi | fsym_core::Constant::E) => None,
                    Expr::Function(name, _) if name != "exp" => None,
                    _ => return Err(limit("scalar constructor power-base depth", 1)),
                };
                if let Some(base_bits) = base_bits {
                    let Expr::Rational(exponent) = exponent.as_ref() else {
                        return Err(limit("radical constructor exponent", 16));
                    };
                    let (Some(numerator), Some(denominator)) =
                        (exponent.numer().to_i64(), exponent.denom().to_u64())
                    else {
                        return Err(limit("radical constructor exponent", 16));
                    };
                    if numerator.unsigned_abs() > 16 || denominator == 0 || denominator > 16 {
                        return Err(limit("radical constructor exponent", 16));
                    }
                    radicals.0 = radicals.0.saturating_add(base_bits);
                    if radicals.0 > 32 {
                        return Err(limit("positive radical constructor component bits", 32));
                    }
                    radicals.1 = radicals.1.max(numerator.unsigned_abs());
                    radicals.2 = radicals.2.max(denominator);
                    Ok(())
                } else {
                    self.constructor_affine(exponent)
                }
            }
            _ => Err(limit("scalar constructor factor depth", 1)),
        }
    }

    fn charge_radical_constructor(&mut self, bits: usize) -> Result<(), SimplifyError> {
        // Positive bases have <=32 aggregate component bits and |p|,q <=16.
        // The bit bound also includes p*(q-1) from reciprocal rationalization.
        // Core reaches exact n-th roots, <=32768 trial divisions and bounded
        // integer/rational powers only; no symbolic base or general simplify
        // call enters this part of the adapter.
        let limbs = bits.div_ceil(64).max(1);
        let trials = if bits >= 30 {
            32_768
        } else {
            1usize << bits.div_ceil(2)
        };
        let rounds = trials.saturating_add(bits.saturating_mul(bits).saturating_mul(32));
        self.charge(
            rounds.saturating_mul(limbs.saturating_mul(limbs).saturating_mul(64)),
            rounds.saturating_mul((limbs * 8 + 32).saturating_mul(16)),
            rounds.saturating_mul(16),
        )
    }

    fn normalize(&mut self, expression: &Expr, depth: usize) -> Result<Expr, SimplifyError> {
        self.charge(1, 0, 0)?;
        self.visits += 1;
        if depth > MAX_DEPTH {
            return Err(limit("normalization depth", MAX_DEPTH));
        }
        if self.visits > MAX_VISITS {
            return Err(limit("normalization visits", MAX_VISITS));
        }
        match expression {
            Expr::Add(terms) => {
                let terms = self.normalize_children(terms, depth)?;
                let collected = self.collect_terms(terms)?;
                if let Expr::Add(terms) = &collected
                    && let Some(folded) = self.fold_pairs(terms)?
                {
                    return self.collect_terms(folded);
                }
                Ok(collected)
            }
            Expr::Mul(factors) => {
                let factors = self.normalize_children(factors, depth)?;
                self.normalize_product(factors, depth)
            }
            Expr::Pow(base, exponent) => {
                let base = self.normalize(base, depth + 1)?;
                let exponent = self.normalize(exponent, depth + 1)?;
                if let Expr::Pow(inner_base, inner_exponent) = &base
                    && matches!(exponent, Expr::Integer(_))
                {
                    let scaled = self.scale_exponent(inner_exponent, &exponent, depth + 1)?;
                    self.charge(1, size_of::<Expr>() + 16, 1)?;
                    return self.normalize(&Expr::Pow(inner_base.clone(), Arc::new(scaled)), depth);
                }
                if exponent.is_zero() {
                    return Ok(Expr::from_i64(1));
                }
                if exponent.is_one() {
                    return Ok(base);
                }
                if base.is_zero() {
                    if let Some(value) = crate::zero_base_power_value(&exponent) {
                        return Ok(value);
                    }
                } else if base.is_one() {
                    return Ok(if is_explicitly_non_finite_exponent(&exponent) {
                        Expr::Const(fsym_core::Constant::NaN)
                    } else {
                        Expr::from_i64(1)
                    });
                } else if let (Expr::Integer(integer), Expr::Integer(power)) = (&base, &exponent)
                    && let Some(power) = power.to_i64()
                    && power.unsigned_abs() <= 1000
                {
                    // A lower bound refuses guaranteed growth before any
                    // square. Otherwise the upper bound is at most MAX_BITS
                    // plus 999 bits, so the governed power cannot secretly
                    // materialize a giant intermediate near this boundary.
                    let minimum_bits = integer
                        .bits()
                        .saturating_sub(1)
                        .saturating_mul(power.unsigned_abs())
                        .saturating_add(1);
                    if minimum_bits > MAX_BITS as u64 {
                        return Err(limit("coefficient bits", MAX_BITS));
                    }
                    let numeric = self.number(&base)?.expect("integer numeric view");
                    let result = numeric
                        .metered_pow(power as i32, &mut self.meter)
                        .map_err(arithmetic_error)?;
                    let result = self.scalar(result)?;
                    return if power < 0 {
                        // The historical integer-negative-power rule retains
                        // Rational even when its denominator reduces to one.
                        Ok(Expr::Rational(result))
                    } else {
                        self.scalar_expr(result)
                    };
                }
                self.charge(1, 2 * (size_of::<Expr>() + 16), 2)?;
                Ok(Expr::Pow(Arc::new(base), Arc::new(exponent)))
            }
            Expr::Function(name, arguments) => {
                let arguments = self.normalize_children(arguments, depth)?;
                if arguments.len() == 1 && arguments[0].is_zero() {
                    match name.as_str() {
                        "sin" | "tan" | "sinh" | "tanh" | "asin" | "atan" | "asinh" | "atanh"
                        | "erf" => return Ok(Expr::from_i64(0)),
                        "cos" | "cosh" | "exp" | "sec" | "sech" | "sinc" | "erfc" => {
                            return Ok(Expr::from_i64(1));
                        }
                        _ => {}
                    }
                }
                if arguments.len() == 1
                    && arguments[0].is_one()
                    && matches!(
                        name.as_str(),
                        "acos" | "acosh" | "asec" | "asech" | "ln" | "log"
                    )
                {
                    return Ok(Expr::from_i64(0));
                }
                self.charge(name.len(), name.len(), 1)?;
                Ok(Expr::Function(name.clone(), arguments))
            }
            _ => self.clone_expr(expression),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsym_budget::{Budget, BudgetLimits, Unbounded};
    use fsym_core::{BigInt, Constant};

    #[derive(Default)]
    struct RecordingMeter {
        charges: [u64; DIMENSION_COUNT],
        checkpoints: usize,
        cancel_at: Option<usize>,
    }

    impl BudgetMeter for RecordingMeter {
        fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
            self.charge_batch(&[(dimension, amount)])
        }

        fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
            for &(dimension, amount) in charges {
                self.charges[dimension.index()] += amount;
            }
            Ok(())
        }

        fn checkpoint(&mut self) -> Result<(), MeterError> {
            self.checkpoints += 1;
            if self.cancel_at.is_some_and(|at| self.checkpoints >= at) {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    fn power(base: Expr, exponent: i64) -> Expr {
        Expr::pow(base, Expr::from_i64(exponent))
    }

    fn function(name: &str, argument: Expr) -> Expr {
        Expr::Function(name.into(), vec![argument])
    }

    fn tower() -> Expr {
        power(power(power(Expr::from_i64(2), 1000), 1000), 1000)
    }

    #[test]
    fn negative_term_formatting_precharges_nested_subtree_copies() {
        const LEVELS: usize = 8;
        let leaf = Expr::Add(
            (0..64)
                .map(|index| Expr::Sym(fsym_core::Symbol::new(format!("v{index}"))))
                .collect(),
        );
        let mut positive = leaf.clone();
        let mut negative = leaf.clone();
        for _ in 0..LEVELS {
            positive = Expr::Add(vec![
                Expr::symbol("z"),
                Expr::Mul(vec![Expr::from_i64(1), function("sin", positive)]),
            ]);
            negative = Expr::Add(vec![
                Expr::symbol("z"),
                Expr::Mul(vec![Expr::from_i64(-1), function("sin", negative)]),
            ]);
        }
        let mut measured = RecordingMeter::default();
        let actual = {
            let mut normalizer = Normalizer {
                meter: NumericMeter {
                    caller: &mut measured,
                    used: [0; DIMENSION_COUNT],
                    failure: None,
                },
                visits: 0,
            };
            let leaf_shape = normalizer.inspect(&leaf).unwrap();
            let positive_shape = normalizer.inspect(&positive).unwrap();
            let negative_shape = normalizer.inspect(&negative).unwrap();
            assert_eq!(positive_shape.nodes, negative_shape.nodes);
            assert_eq!(positive_shape.bytes, negative_shape.bytes);
            // Every negative term owns two copies of the complete bottom
            // leaf during core Display; a linear node-only envelope misses
            // these repeated copies even though all shapes are admitted.
            assert!(
                negative_shape.display_bytes
                    >= positive_shape.display_bytes + 2 * LEVELS * leaf_shape.bytes
            );
            assert!(
                negative_shape.display_work
                    >= positive_shape.display_work + 2 * LEVELS * leaf_shape.nodes
            );
            assert!(
                negative_shape.display_allocations
                    >= positive_shape.display_allocations + 2 * LEVELS * leaf_shape.nodes
            );
            normalizer.rendered(&negative, false).unwrap()
        };
        assert_eq!(actual, format!("{negative}"));

        let requested = measured.charges[Dimension::MemoryBytes.index()];
        assert!(requested > 0);
        let mut limits = BudgetLimits::uniform(u64::MAX, 29);
        limits.dimensions[Dimension::MemoryBytes.index()] = requested - 1;
        let mut caller = Budget::new(limits);
        let error = {
            let mut normalizer = Normalizer {
                meter: NumericMeter {
                    caller: &mut caller,
                    used: [0; DIMENSION_COUNT],
                    failure: None,
                },
                visits: 0,
            };
            normalizer.rendered(&negative, false).unwrap_err()
        };
        assert!(matches!(
            error,
            SimplifyError::BudgetExhausted(BudgetError::Exhausted {
                dimension: Dimension::MemoryBytes,
                ..
            })
        ));
        // The complete precharge fails atomically before formatting starts;
        // inspecting the source itself consumes no storage/allocation quota.
        assert_eq!(caller.remaining(Dimension::MemoryBytes), requested - 1);
        assert_eq!(caller.remaining(Dimension::AllocationCount), u64::MAX);
        assert_eq!(caller.verifier_remaining(), 29);
    }

    #[test]
    fn held_powers_refuse_before_large_materialization_in_every_position() {
        let x = Expr::symbol("x");
        for expression in [
            tower(),
            power(power(Expr::from_i64(2), 1000), -1000),
            Expr::Function("_fsym_atan2_dy".into(), vec![x, tower()]),
            Expr::Add(vec![Expr::from_i64(1), tower()]),
            Expr::Mul(vec![Expr::from_i64(2), tower()]),
        ] {
            let original = expression.clone();
            let mut meter = RecordingMeter::default();
            assert!(matches!(
                simplify_numeric_with(&expression, &mut meter),
                Err(SimplifyError::NumericLimitExceeded {
                    resource: "coefficient bits",
                    ..
                })
            ));
            assert_eq!(expression, original);
            assert!(meter.charges[Dimension::MemoryBytes.index()] < 1_000_000);
            assert!(meter.charges[Dimension::ComputeSteps.index()] < 1_000_000);
        }
    }

    #[test]
    fn small_held_powers_and_exact_coefficient_cancellation_keep_their_values() {
        let x = Expr::symbol("x");
        let expression = Expr::Add(vec![
            Expr::Mul(vec![power(Expr::from_i64(3), 8), x.clone()]),
            Expr::Mul(vec![Expr::from_i64(-6560), x.clone()]),
            power(Expr::from_i64(2), -3),
        ]);
        assert_eq!(
            simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
            Expr::Add(vec![Expr::rational(1, 8).unwrap(), x])
        );
        assert_eq!(
            simplify_numeric_with(&power(Expr::from_i64(-1), -2), &mut Unbounded).unwrap(),
            Expr::Rational(BigRational::one())
        );
    }

    #[test]
    fn dynamic_exponents_and_nested_distribution_match_historical_full_forms() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");
        let sum = Expr::Add(vec![y.clone(), z.clone()]);
        let expressions = [
            Expr::pow(x.clone(), y.clone()),
            power(Expr::pow(x.clone(), sum.clone()), 2),
            power(
                Expr::pow(x.clone(), Expr::Mul(vec![Expr::from_i64(2), sum])),
                3,
            ),
            power(
                Expr::pow(x.clone(), Expr::Mul(vec![y.clone(), z.clone()])),
                -2,
            ),
            Expr::Mul(vec![
                Expr::pow(x.clone(), y.clone()),
                Expr::pow(x.clone(), z.clone()),
            ]),
            power(
                Expr::pow(
                    x,
                    Expr::Rational(BigRational::new(BigInt::from(2), BigInt::from(3))),
                ),
                -3,
            ),
        ];
        for expression in expressions {
            assert_eq!(
                simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
                crate::try_simplify(&expression).unwrap(),
                "historical form for {expression}"
            );
        }
    }

    #[test]
    fn full_mode_trigonometric_and_exponential_rules_remain_exact() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        for (left, right, sign) in [
            ("sin", "cos", 1),
            ("sec", "tan", -1),
            ("csc", "cot", -1),
            ("cosh", "sinh", -1),
        ] {
            let expression = Expr::Add(vec![
                Expr::Mul(vec![Expr::from_i64(7), power(function(left, x.clone()), 2)]),
                Expr::Mul(vec![
                    Expr::from_i64(7 * sign),
                    power(function(right, x.clone()), 2),
                ]),
                y.clone(),
            ]);
            assert_eq!(
                simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
                Expr::Add(vec![Expr::from_i64(7), y.clone()])
            );
        }
        let expression = Expr::Mul(vec![
            function("exp", x.clone()),
            function("exp", y.clone()),
            function("exp", Expr::Mul(vec![Expr::from_i64(-1), x])),
        ]);
        assert_eq!(
            simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
            function("exp", y)
        );
    }

    #[test]
    fn historical_small_expression_forms_match_without_weakening_source_comparison() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let atoms = [
            Expr::from_i64(-2),
            Expr::from_i64(0),
            Expr::from_i64(1),
            Expr::from_i64(3),
            Expr::rational(2, 3).unwrap(),
            x.clone(),
            function("sin", x.clone()),
            Expr::Add(vec![x.clone(), y.clone()]),
        ];
        for left in &atoms {
            for right in &atoms {
                for expression in [
                    Expr::Add(vec![left.clone(), right.clone()]),
                    Expr::Mul(vec![left.clone(), right.clone()]),
                    Expr::pow(left.clone(), right.clone()),
                ] {
                    assert_eq!(
                        simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
                        crate::try_simplify(&expression).unwrap(),
                        "historical exact tree for {expression}"
                    );
                }
            }
        }
        for expression in [
            Expr::Mul(vec![Expr::from_i64(0), function("log", x.clone())]),
            Expr::Mul(vec![Expr::from_i64(0), power(x, -1)]),
            Expr::Mul(vec![Expr::from_i64(0), Expr::Const(Constant::Infinity)]),
            Expr::Add(vec![
                power(function("sin", y.clone()), 2),
                power(function("cos", y), 2),
            ]),
        ] {
            assert_eq!(
                simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
                crate::try_simplify(&expression).unwrap()
            );
        }
    }

    #[test]
    fn arithmetic_work_uses_the_caller_budget_beyond_tree_visits() {
        let mut limits = BudgetLimits::uniform(u64::MAX, 71);
        limits.dimensions[Dimension::ComputeSteps.index()] = 80;
        let mut budget = Budget::new(limits);
        let expression = power(Expr::from_i64(3), 256);
        admit_numeric_expression_with(&expression, &mut budget).unwrap();
        assert!(matches!(
            simplify_numeric_with(&expression, &mut budget),
            Err(SimplifyError::BudgetExhausted(BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            }))
        ));
        assert!(budget.remaining(Dimension::ComputeSteps) < 80);
        assert_eq!(budget.verifier_remaining(), 71);
    }

    #[test]
    fn exact_cumulative_budget_edges_and_every_dimension_refuse_atomically() {
        let expression = power(Expr::from_i64(3), 64);
        let mut measured = RecordingMeter::default();
        let expected = simplify_numeric_with(&expression, &mut measured).unwrap();
        let limits = BudgetLimits {
            dimensions: measured.charges,
            verifier_pool: 19,
        };
        let mut exact = Budget::new(limits);
        assert_eq!(
            simplify_numeric_with(&expression, &mut exact).unwrap(),
            expected
        );
        assert_eq!(exact.remaining(Dimension::ComputeSteps), 0);
        assert_eq!(exact.remaining(Dimension::MemoryBytes), 0);
        assert_eq!(exact.verifier_remaining(), 19);
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut reduced = limits;
            reduced.dimensions[dimension.index()] -= 1;
            let mut budget = Budget::new(reduced);
            assert!(
                matches!(simplify_numeric_with(&expression, &mut budget), Err(SimplifyError::BudgetExhausted(BudgetError::Exhausted { dimension: refused, .. })) if refused == dimension)
            );
            assert_eq!(budget.verifier_remaining(), 19);
        }
        let mut caller = RecordingMeter::default();
        let mut intrinsic = NumericMeter {
            caller: &mut caller,
            used: [0; DIMENSION_COUNT],
            failure: None,
        };
        intrinsic.used[Dimension::ComputeSteps.index()] = MAX_STEPS as u64;
        assert!(
            intrinsic
                .charge_batch(&[(Dimension::MemoryBytes, 100), (Dimension::ComputeSteps, 1)])
                .is_err()
        );
        assert_eq!(intrinsic.caller.charges, [0; DIMENSION_COUNT]);
        assert_eq!(intrinsic.used[Dimension::MemoryBytes.index()], 0);
    }

    #[test]
    fn cancellation_during_exact_power_stays_cancellation() {
        let expression = power(Expr::from_i64(3), 256);
        let mut measured = RecordingMeter::default();
        simplify_numeric_with(&expression, &mut measured).unwrap();
        let mut cancelled = RecordingMeter {
            cancel_at: Some(measured.checkpoints / 2),
            ..RecordingMeter::default()
        };
        assert_eq!(
            simplify_numeric_with(&expression, &mut cancelled),
            Err(SimplifyError::Cancelled)
        );
        assert!(cancelled.charges[Dimension::ComputeSteps.index()] > 0);
        assert!(cancelled.charges[Dimension::MemoryBytes.index()] > 0);
        let mut early = RecordingMeter {
            cancel_at: Some(1),
            ..RecordingMeter::default()
        };
        assert_eq!(
            simplify_numeric_with(&expression, &mut early),
            Err(SimplifyError::Cancelled)
        );
        assert_eq!(early.charges, [0; DIMENSION_COUNT]);
    }

    #[test]
    fn source_preflight_refuses_wide_coefficients_without_copying_or_folding() {
        let wide = Expr::Integer(BigInt::from(2).pow(MAX_BITS as u32));
        let expression = Expr::pow(Expr::symbol("x"), wide.clone());
        let mut meter = RecordingMeter::default();
        assert_eq!(
            admit_numeric_expression_with(&expression, &mut meter),
            Err(limit("coefficient bits", MAX_BITS))
        );
        assert_eq!(meter.charges[Dimension::MemoryBytes.index()], 0);
        assert_eq!(meter.charges[Dimension::AllocationCount.index()], 0);
        // Public symbolic normalization retains its independent historical policy.
        assert_eq!(crate::simplify(&wide), wide);
        admit_numeric_expression_with(&tower(), &mut Unbounded).unwrap();
    }

    #[test]
    fn large_fixed_rational_coordinates_do_not_pay_for_nonexistent_comparisons() {
        let coordinate =
            Expr::Rational(BigRational::new(BigInt::from(2).pow(1000), BigInt::from(3)));
        let x = Expr::symbol("x");
        let angle = Expr::Function("atan2".into(), vec![x.clone(), coordinate.clone()]);
        let jet = Expr::Function("_fsym_atan2_dy".into(), vec![x, coordinate]);
        let expression = Expr::Mul(vec![function("cos", angle), jet, Expr::from_i64(1)]);
        let mut meter = RecordingMeter::default();
        assert_eq!(
            simplify_numeric_with(&expression, &mut meter).unwrap(),
            crate::try_simplify(&expression).unwrap()
        );
        assert!(meter.charges[Dimension::ComputeSteps.index()] < 1_000_000);
        assert!(meter.charges[Dimension::AllocationCount.index()] < 100_000);
    }

    #[test]
    fn governed_ordering_and_cached_factor_keys_preserve_both_legacy_orders() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let values = vec![
            x.clone(),
            y.clone(),
            Expr::from_i64(-2),
            Expr::rational(2, 3).unwrap(),
            Expr::Const(Constant::Pi),
            Expr::Add(vec![y.clone()]),
            Expr::Add(vec![x.clone(), y.clone()]),
            Expr::Mul(vec![x.clone(), y.clone()]),
            Expr::pow(x.clone(), y.clone()),
            function("cos", x.clone()),
            function("sin", y.clone()),
            Expr::Function(
                "opaque".into(),
                vec![Expr::pow(x.clone(), Expr::from_i64(2))],
            ),
            Expr::Function("opaque".into(), vec![Expr::pow(x, Expr::from_i64(3))]),
        ];
        let mut unbounded = Unbounded;
        let mut normalizer = Normalizer {
            meter: NumericMeter {
                caller: &mut unbounded,
                used: [0; DIMENSION_COUNT],
                failure: None,
            },
            visits: 0,
        };
        for left in &values {
            for right in &values {
                assert_eq!(
                    normalizer.compare(left, right, false).unwrap(),
                    left.cmp(right)
                );
                assert_eq!(
                    normalizer.compare(left, right, true).unwrap(),
                    fsym_core::cmp_structural(left, right)
                );
            }
        }
        let mut expected = values.clone();
        expected.sort_by(fsym_core::cmp_mul_factors);
        let mut reversed: Vec<_> = values.into_iter().rev().collect();
        normalizer.sort_factors(&mut reversed).unwrap();
        assert_eq!(reversed, expected);
    }

    #[test]
    fn bounded_scalar_constructor_keeps_exponential_alias_and_zero_forms() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");
        let e_power = Expr::pow(Expr::Const(Constant::E), y.clone());
        for exponent in [
            e_power.clone(),
            Expr::Mul(vec![z.clone(), e_power.clone()]),
            Expr::Mul(vec![e_power.clone(), function("exp", y.clone())]),
            Expr::Mul(vec![e_power, function("exp", z)]),
            Expr::Mul(vec![Expr::from_i64(0), function("log", y)]),
        ] {
            let expression = power(Expr::pow(x.clone(), exponent), 2);
            assert_eq!(
                simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
                crate::try_simplify(&expression).unwrap(),
                "constructor-compatible nested power for {expression}",
            );
        }
    }

    #[test]
    fn constructor_refuses_power_expansion_routes_before_substrate_entry() {
        let x = Expr::symbol("x");
        let logarithm = function("log", Expr::from_i64(2));
        let dangerous_argument = Expr::Mul(vec![Expr::from_i64(1_000_000), logarithm]);
        let aliases = Expr::Mul(vec![
            Expr::pow(Expr::Const(Constant::E), dangerous_argument.clone()),
            function("exp", dangerous_argument),
        ]);
        let source = power(Expr::pow(x.clone(), aliases), 2);
        let mut meter = RecordingMeter::default();
        assert!(matches!(
            simplify_numeric_with(&source, &mut meter),
            Err(SimplifyError::NumericLimitExceeded {
                resource: "scalar constructor affine exponent",
                ..
            }),
        ));
        assert!(meter.charges[Dimension::MemoryBytes.index()] < 1_000_000);

        // Core would sum these to -29/208 before reciprocal radical
        // rationalization. Individual exponent bounds cannot admit them.
        let mut unbounded = Unbounded;
        let mut normalizer = Normalizer {
            meter: NumericMeter {
                caller: &mut unbounded,
                used: [0; DIMENSION_COUNT],
                failure: None,
            },
            visits: 0,
        };
        let repeated = Expr::Mul(vec![
            Expr::pow(Expr::from_i64(2), Expr::rational(-1, 13).unwrap()),
            Expr::pow(Expr::from_i64(2), Expr::rational(-1, 16).unwrap()),
        ]);
        assert_eq!(
            normalizer.scale_exponent(&repeated, &Expr::from_i64(2), 0),
            Err(limit("scalar constructor repeated numeric base", 1)),
        );
        let expression = power(Expr::pow(x, repeated), 2);
        assert!(matches!(
            simplify_numeric_with(&expression, &mut Unbounded),
            Err(SimplifyError::NumericLimitExceeded { .. }),
        ));
    }

    #[test]
    fn normalization_depth_boundary_is_safe_on_the_default_test_stack() {
        let mut expression = Expr::symbol("x");
        for _ in 0..MAX_DEPTH {
            expression = function("sin", expression);
        }
        assert_eq!(
            simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
            expression,
        );
        expression = function("sin", expression);
        assert_eq!(
            simplify_numeric_with(&expression, &mut Unbounded),
            Err(limit("expression depth", MAX_DEPTH)),
        );
    }

    #[test]
    fn nested_power_scaling_replays_bounded_numeric_radical_construction() {
        let x = Expr::symbol("x");
        let half = Expr::rational(1, 2).unwrap();
        for (left, right) in [(2, 8), (2, 3), (6, 4)] {
            let radicals = Expr::Mul(vec![
                Expr::pow(Expr::from_i64(left), half.clone()),
                Expr::pow(Expr::from_i64(right), half.clone()),
            ]);
            let expression = power(Expr::pow(x.clone(), radicals), 2);
            assert_eq!(
                simplify_numeric_with(&expression, &mut Unbounded).unwrap(),
                crate::try_simplify(&expression).unwrap()
            );
        }
        let radicals = Expr::Mul(vec![
            Expr::pow(Expr::Integer(BigInt::from(2).pow(40)), half.clone()),
            Expr::pow(Expr::from_i64(3), half),
        ]);
        let oversized = power(Expr::pow(x, radicals), 2);
        assert!(matches!(
            simplify_numeric_with(&oversized, &mut Unbounded),
            Err(SimplifyError::NumericLimitExceeded { .. })
        ));
    }
}
