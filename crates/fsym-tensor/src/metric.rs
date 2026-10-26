//! Exact metric invariants and component index changes over Q.
//!
//! These bounded synchronous operations check symmetry and the supplied inverse
//! before using a metric. A separate reference contraction checks the inverse
//! index change before a generated tensor is returned. This is same-crate exact
//! reference evidence, not a portable proof-kernel or analytic metric claim.

use crate::{IndexVariance, MetricTensor, TensorError, TensorExpr};
use fsym_core::{BigInt, BigRational, Expr};

const MAX_METRIC_DIMENSION: usize = 64;
const MAX_RANK: usize = 32;
const MAX_COEFFICIENT_BITS: u64 = 4_096;
const MAX_STORED_BITS: u64 = 16_777_216;
const MAX_ARITHMETIC_OPERATIONS: usize = 10_000_000;

fn zero() -> BigRational {
    BigRational::from_integer(BigInt::from(0))
}

fn one() -> BigRational {
    BigRational::from_integer(BigInt::from(1))
}

fn is_zero(value: &BigRational) -> bool {
    value.numer() == &BigInt::from(0)
}

fn limit(message: &str) -> TensorError {
    TensorError::MetricResourceLimit(message.to_string())
}

fn invalid(message: &str) -> TensorError {
    TensorError::InvalidMetric(message.to_string())
}

pub(super) fn validate_rational_entries(entries: &[Expr]) -> Result<(), TensorError> {
    // Check all original entries before allocating clones or taking any zero
    // shortcut. Undefined or symbolic children cannot cancel into this lane.
    let mut total_bits = 0u64;
    for entry in entries {
        let (numerator, denominator) = match entry {
            Expr::Integer(value) => (value.bits().max(1), 1),
            Expr::Rational(value) => (value.numer().bits().max(1), value.denom().bits()),
            _ => return Err(TensorError::UnsupportedMetricDomain),
        };
        if numerator > MAX_COEFFICIENT_BITS || denominator > MAX_COEFFICIENT_BITS {
            return Err(limit("input coefficient exceeds 4096 bits"));
        }
        total_bits = total_bits
            .saturating_add(numerator)
            .saturating_add(denominator);
        if total_bits > MAX_STORED_BITS {
            return Err(limit("aggregate input coefficient storage exceeded"));
        }
    }
    Ok(())
}

fn rational_components(entries: &[Expr]) -> Result<Vec<BigRational>, TensorError> {
    validate_rational_entries(entries)?;
    Ok(entries
        .iter()
        .map(|entry| match entry {
            Expr::Integer(value) => BigRational::from_integer(value.clone()),
            Expr::Rational(value) => value.clone(),
            _ => unreachable!("original entries were checked above"),
        })
        .collect())
}

#[derive(Default)]
struct Arithmetic {
    operations: usize,
}

impl Arithmetic {
    fn step(&mut self) -> Result<(), TensorError> {
        if self.operations == MAX_ARITHMETIC_OPERATIONS {
            return Err(limit("arithmetic operation bound exceeded"));
        }
        self.operations += 1;
        Ok(())
    }

    fn product(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, TensorError> {
        self.step()?;
        if is_zero(a) || is_zero(b) {
            return Ok(zero());
        }
        if a.numer().bits().saturating_add(b.numer().bits()) > MAX_COEFFICIENT_BITS
            || a.denom().bits().saturating_add(b.denom().bits()) > MAX_COEFFICIENT_BITS
        {
            return Err(limit("uncancelled product coefficient bound exceeded"));
        }
        Ok(a * b)
    }

    fn sum(&mut self, a: &BigRational, b: &BigRational) -> Result<BigRational, TensorError> {
        self.step()?;
        if a.numer()
            .bits()
            .saturating_add(b.denom().bits())
            .max(b.numer().bits().saturating_add(a.denom().bits()))
            .saturating_add(1)
            > MAX_COEFFICIENT_BITS
            || a.denom().bits().saturating_add(b.denom().bits()) > MAX_COEFFICIENT_BITS
        {
            return Err(limit("uncancelled sum coefficient bound exceeded"));
        }
        Ok(a + b)
    }
}

fn checked_metric(
    metric: &MetricTensor,
) -> Result<(Vec<BigRational>, Vec<BigRational>), TensorError> {
    metric.validate_storage()?;
    let dimension = metric.dimension;
    if dimension > MAX_METRIC_DIMENSION {
        return Err(limit("exact metric dimension exceeds 64"));
    }
    let matrix = rational_components(&metric.matrix)?;
    let inverse = rational_components(&metric.inverse)?;
    for row in 0..dimension {
        for column in 0..dimension {
            if matrix[row * dimension + column] != matrix[column * dimension + row]
                || inverse[row * dimension + column] != inverse[column * dimension + row]
            {
                return Err(invalid("metric and inverse must be symmetric"));
            }
        }
    }
    let mut arithmetic = Arithmetic::default();
    for row in 0..dimension {
        for column in 0..dimension {
            let mut value = zero();
            for inner in 0..dimension {
                let product = arithmetic.product(
                    &matrix[row * dimension + inner],
                    &inverse[inner * dimension + column],
                )?;
                value = arithmetic.sum(&value, &product)?;
            }
            let expected = if row == column { one() } else { zero() };
            if value != expected {
                return Err(invalid(
                    "supplied inverse does not multiply to the identity",
                ));
            }
        }
    }
    // Over Q, a right inverse of a square matrix is also its left inverse.
    Ok((matrix, inverse))
}

fn component_input(tensor: &TensorExpr) -> Result<Vec<BigRational>, TensorError> {
    if tensor.rank() > MAX_RANK {
        return Err(limit("component rank exceeds 32"));
    }
    tensor.validate_components()?;
    let components = tensor
        .components
        .as_deref()
        .ok_or(TensorError::ConcreteComponentsRequired)?;
    rational_components(components)
}

fn transformation_shape(
    metric: &MetricTensor,
    tensor: &TensorExpr,
    position: usize,
) -> Result<(), TensorError> {
    if metric.dimension != tensor.dimension {
        return Err(TensorError::DimensionMismatch(
            metric.dimension,
            tensor.dimension,
        ));
    }
    if position >= tensor.rank() {
        return Err(TensorError::IndexMismatch(
            "metric index position is outside the tensor rank".into(),
        ));
    }
    Ok(())
}

fn change_index(
    metric: &MetricTensor,
    tensor: &TensorExpr,
    position: usize,
    target: IndexVariance,
    name: String,
) -> Result<TensorExpr, TensorError> {
    transformation_shape(metric, tensor, position)?;
    if tensor.indices[position].variance == target {
        return Err(TensorError::IndexMismatch(
            "metric index change requires the opposite input variance".into(),
        ));
    }
    let (matrix, inverse) = checked_metric(metric)?;
    let components = component_input(tensor)?;
    let n = metric.dimension;
    let contraction = if target == IndexVariance::Lower {
        matrix
    } else {
        inverse
    };
    let mut result = Vec::new();
    crate::try_reserve(&mut result, components.len())?;
    let stride = n
        .checked_pow((tensor.rank() - position - 1) as u32)
        .ok_or_else(|| limit("tensor index stride overflow"))?;
    let mut arithmetic = Arithmetic::default();
    let mut stored_bits = 0u64;
    for flat in 0..components.len() {
        let coordinate = flat / stride % n;
        let fixed = flat - coordinate * stride;
        let mut sum = zero();
        for inner in 0..n {
            let product = arithmetic.product(
                &contraction[coordinate * n + inner],
                &components[fixed + inner * stride],
            )?;
            sum = arithmetic.sum(&sum, &product)?;
        }
        stored_bits = stored_bits
            .saturating_add(sum.numer().bits().max(1))
            .saturating_add(sum.denom().bits());
        if stored_bits > MAX_STORED_BITS {
            return Err(limit("aggregate output coefficient storage exceeded"));
        }
        result.push(if sum.is_integer() {
            Expr::Integer(sum.to_integer())
        } else {
            Expr::Rational(sum)
        });
    }
    let mut indices = tensor.indices.clone();
    indices[position].variance = target;
    let result = TensorExpr::with_components(name, n, indices, result)?;
    verify_metric_index_change(metric, tensor, position, &result)?;
    Ok(result)
}

impl MetricTensor {
    /// Check the exact symmetric, nondegenerate rational metric and its supplied
    /// inverse. No signature or positive-definiteness claim is made.
    pub fn verify_exact(&self) -> Result<(), TensorError> {
        checked_metric(self).map(|_| ())
    }

    /// Lower any one upper component index using g, retaining all other slots.
    /// Exact rational components and a checked rational metric are required.
    pub fn lower_index(
        &self,
        tensor: &TensorExpr,
        position: usize,
        new_name: impl Into<String>,
    ) -> Result<TensorExpr, TensorError> {
        change_index(
            self,
            tensor,
            position,
            IndexVariance::Lower,
            new_name.into(),
        )
    }

    /// Raise any one lower component index using the checked inverse metric.
    pub fn raise_index(
        &self,
        tensor: &TensorExpr,
        position: usize,
        new_name: impl Into<String>,
    ) -> Result<TensorExpr, TensorError> {
        change_index(
            self,
            tensor,
            position,
            IndexVariance::Upper,
            new_name.into(),
        )
    }
}

/// Independent reference check of a component index change. After validating
/// the metric and exact slot identities, reconstruct the ORIGINAL tensor by
/// applying the opposite metric to the candidate. Coordinate decoding is
/// independent of the generator's stride-based contraction.
pub fn verify_metric_index_change(
    metric: &MetricTensor,
    original: &TensorExpr,
    position: usize,
    candidate: &TensorExpr,
) -> Result<(), TensorError> {
    transformation_shape(metric, original, position)?;
    let (matrix, inverse) = checked_metric(metric)?;
    let original_components = component_input(original)?;
    let candidate_components = component_input(candidate)?;
    if candidate.dimension != original.dimension || candidate.rank() != original.rank() {
        return Err(invalid(
            "index-change candidate shape differs from the input",
        ));
    }
    for (slot, (input, output)) in original.indices.iter().zip(&candidate.indices).enumerate() {
        let variance = if slot == position {
            input.variance.flip()
        } else {
            input.variance
        };
        if input.symbol != output.symbol || output.variance != variance {
            return Err(invalid(
                "index-change candidate has incorrect index identities or variances",
            ));
        }
    }
    let n = metric.dimension;
    let recovery = if original.indices[position].variance == IndexVariance::Upper {
        inverse
    } else {
        matrix
    };
    let mut arithmetic = Arithmetic::default();
    for (flat, expected) in original_components.iter().enumerate() {
        let mut coordinates = crate::decode_multi_index(flat, original.rank(), n)?;
        let target = coordinates[position];
        let mut recovered = zero();
        for inner in 0..n {
            coordinates[position] = inner;
            let source = crate::encode_multi_index(&coordinates, n);
            let product =
                arithmetic.product(&recovery[target * n + inner], &candidate_components[source])?;
            recovered = arithmetic.sum(&recovered, &product)?;
        }
        if &recovered != expected {
            return Err(invalid(
                "candidate inverse contraction does not recover the original tensor",
            ));
        }
    }
    Ok(())
}
