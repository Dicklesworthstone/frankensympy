//! Complete finite modular solution families from checked integer Smith forms.
//!
//! D=UAV reduces A X=B mod m to scalar congruences, without assuming m
//! prime or A square/full-rank. Nonunit Smith factors contribute torsion
//! directions with explicit finite periods; zero columns contribute period m.
//! The independent proof kernel checks the original sources, modulus, complete
//! family and every period. A zero residual alone is not acceptance.
//!
//! Mixed dense/sparse inputs use borrowed admission before coefficient copies.
//! Source admission, Smith generation, modular products, inversion, lifting and
//! publication share the existing cumulative integer-normal-form envelopes.

use super::integer_system::{IntegerMatrixRef, same_source};
use super::{
    BigInt, IntegerMatrix, MAX_INPUT_BITS, Matrix, MatrixError, RationalArithmeticError,
    RationalSparseMatrix, Result, SparseMatrix, SparseRrefError, Work, coefficient, generate,
    limit, stopped, verification,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::arith::{metered_gcd, metered_mod_inverse};
use fsym_proof_kernel::integer_modular_inverse::MAX_INTEGER_MODULAR_MODULUS_BITS;
use fsym_proof_kernel::integer_modular_system::{
    INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION, IntegerModularSystemError, IntegerModularSystemOutcome,
    IntegerModularSystemReceipt, IntegerModularSystemWitness, verify_integer_modular_system,
};
use fsym_proof_kernel::integer_normal_form::IntegerSmithWitness;

/// Untrusted proposal retaining both exact original matrices and the modulus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerModularSystemCandidate {
    pub source: RationalSparseMatrix,
    pub rhs: RationalSparseMatrix,
    pub modulus: BigInt,
    pub witness: IntegerModularSystemWitness,
}

/// All modular solutions, or independently checked inconsistency.
///
/// A solvable result represents X=P+K*T mod m with independent
/// 0 <= T[i,j] < periods[i] for every RHS column j. The finite family is
/// complete and has no repeated parameterizations. The reported integer Smith
/// rank, accessible through the normal-form receipt, is not a modular rank.
#[derive(Debug)]
pub struct VerifiedIntegerModularSystem {
    source: RationalSparseMatrix,
    rhs: RationalSparseMatrix,
    witness: IntegerModularSystemWitness,
    receipt: IntegerModularSystemReceipt,
}

impl VerifiedIntegerModularSystem {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn rhs(&self) -> &RationalSparseMatrix {
        &self.rhs
    }
    pub fn modulus(&self) -> &BigInt {
        self.receipt.modulus()
    }
    pub fn witness(&self) -> &IntegerModularSystemWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &IntegerModularSystemReceipt {
        &self.receipt
    }
    pub fn is_consistent(&self) -> bool {
        self.receipt.is_consistent()
    }
    pub fn is_unique(&self) -> bool {
        self.receipt.is_unique()
    }
    pub fn contradiction(&self) -> Option<(usize, usize)> {
        self.receipt.contradiction()
    }
    pub fn particular(&self) -> Option<&RationalSparseMatrix> {
        match &self.witness.outcome {
            IntegerModularSystemOutcome::Solvable { particular, .. } => Some(particular),
            IntegerModularSystemOutcome::Inconsistent { .. } => None,
        }
    }
    pub fn generators(&self) -> Option<&RationalSparseMatrix> {
        match &self.witness.outcome {
            IntegerModularSystemOutcome::Solvable { generators, .. } => Some(generators),
            IntegerModularSystemOutcome::Inconsistent { .. } => None,
        }
    }
    pub fn periods(&self) -> Option<&[BigInt]> {
        self.receipt.periods()
    }
    pub fn into_parts(
        self,
    ) -> (
        RationalSparseMatrix,
        RationalSparseMatrix,
        IntegerModularSystemWitness,
        IntegerModularSystemReceipt,
    ) {
        (self.source, self.rhs, self.witness, self.receipt)
    }
}

fn inputs(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<(RationalSparseMatrix, RationalSparseMatrix)> {
    work.tick(1)?;
    if !modulus.is_positive() || modulus.is_one() {
        return Err(MatrixError::InvalidModulus(
            "must be an integer at least two",
        ));
    }
    coefficient(modulus, MAX_INTEGER_MODULAR_MODULUS_BITS)?;
    // No coefficient cloning or mathematical shortcut precedes admission of
    // both originals, including all RHS columns and zero-area shapes.
    source.admit(work)?;
    rhs.admit(work)?;
    let (rows, columns) = source.shape();
    let (rhs_rows, rhs_columns) = rhs.shape();
    if rows != rhs_rows {
        return Err(MatrixError::ShapeMismatch(
            rows,
            columns,
            rhs_rows,
            rhs_columns,
        ));
    }
    Ok((source.snapshot(work)?, rhs.snapshot(work)?))
}

fn residue(
    value: &BigInt,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigInt> {
    let (_, remainder) = work.divide(value, modulus)?;
    if remainder.is_negative() {
        work.subtract(modulus, &(-remainder))
    } else {
        Ok(remainder)
    }
}

fn exact_quotient(
    value: &BigInt,
    divisor: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<BigInt> {
    let (quotient, remainder) = work.divide(value, divisor)?;
    if !remainder.is_zero() {
        return Err(MatrixError::ArithmeticInvariantViolation(
            "nonexact Smith modulus quotient",
        ));
    }
    Ok(quotient)
}

fn reduced(
    source: &RationalSparseMatrix,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<IntegerMatrix> {
    let mut result = IntegerMatrix::zero(source.rows, source.columns, work)?;
    for entry in &source.entries {
        result.values[entry.row * source.columns + entry.column] =
            residue(entry.value.numer(), modulus, work)?;
    }
    Ok(result)
}

/// Generator outer-product accumulation is separate from the kernel's
/// straight reference dot products. Reduce each term and sum.
fn modular_product(
    left: &IntegerMatrix,
    right: &IntegerMatrix,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<IntegerMatrix> {
    if left.columns != right.rows {
        return Err(MatrixError::ArithmeticInvariantViolation(
            "modular product shape",
        ));
    }
    let mut result = IntegerMatrix::zero(left.rows, right.columns, work)?;
    for row in 0..left.rows {
        for inner in 0..left.columns {
            work.tick(1)?;
            let left_value = left.get(row, inner);
            if left_value.is_zero() {
                continue;
            }
            for column in 0..right.columns {
                work.tick(1)?;
                let right_value = right.get(inner, column);
                if right_value.is_zero() {
                    continue;
                }
                let product = work.multiply(left_value, right_value)?;
                let negative = -residue(&product, modulus, work)?;
                let sum = work.subtract(result.get(row, column), &negative)?;
                result.values[row * right.columns + column] = residue(&sum, modulus, work)?;
            }
        }
    }
    Ok(result)
}

fn outcome(
    smith: &IntegerSmithWitness,
    rhs: &RationalSparseMatrix,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<IntegerModularSystemOutcome> {
    let left = reduced(&smith.left, modulus, work)?;
    let rhs = reduced(rhs, modulus, work)?;
    let transformed = modular_product(&left, &rhs, modulus, work)?;
    let rank = smith.diagonal.entries.len();
    let variables = smith.right.rows;
    let mut factors = work.vector(rank)?;
    for entry in &smith.diagonal.entries {
        work.tick(1)?;
        factors.push(metered_gcd(entry.value.numer(), modulus, work).map_err(stopped)?);
    }
    let mut coordinates = IntegerMatrix::zero(variables, rhs.columns, work)?;
    for row in 0..transformed.rows {
        work.tick(1)?;
        if row >= rank {
            for column in 0..transformed.columns {
                work.tick(1)?;
                if !transformed.get(row, column).is_zero() {
                    return Ok(IntegerModularSystemOutcome::Inconsistent {
                        row,
                        rhs_column: column,
                    });
                }
            }
            continue;
        }
        let divisor = &factors[row];
        let reduced_modulus = exact_quotient(modulus, divisor, work)?;
        // If d_i is divisible by m, y_i=0 is canonical and the corresponding
        // full-period generator below supplies all residue values.
        let inverse = if reduced_modulus.is_one() {
            None
        } else {
            let diagonal =
                exact_quotient(smith.diagonal.entries[row].value.numer(), divisor, work)?;
            let diagonal = residue(&diagonal, &reduced_modulus, work)?;
            Some(
                metered_mod_inverse(&diagonal, &reduced_modulus, work)
                    .map_err(stopped)?
                    .ok_or(MatrixError::ArithmeticInvariantViolation(
                        "coprime Smith factor has no modular inverse",
                    ))?,
            )
        };
        for column in 0..transformed.columns {
            let (value, remainder) = work.divide(transformed.get(row, column), divisor)?;
            if !remainder.is_zero() {
                return Ok(IntegerModularSystemOutcome::Inconsistent {
                    row,
                    rhs_column: column,
                });
            }
            if let Some(inverse) = &inverse {
                let product = work.multiply(inverse, &value)?;
                coordinates.values[row * rhs.columns + column] =
                    residue(&product, &reduced_modulus, work)?;
            }
        }
    }

    let right = reduced(&smith.right, modulus, work)?;
    let particular = modular_product(&right, &coordinates, modulus, work)?.to_sparse(work)?;
    let mut directions = work.vector(variables)?;
    let mut periods = work.vector(variables)?;
    for coordinate in 0..variables {
        work.tick(1)?;
        let period = factors.get(coordinate).unwrap_or(modulus);
        if !period.is_one() {
            let step = exact_quotient(modulus, period, work)?;
            directions.push((coordinate, step));
            periods.push(work.copy(period)?);
        }
    }
    let mut generators = IntegerMatrix::zero(variables, directions.len(), work)?;
    for (column, (coordinate, step)) in directions.iter().enumerate() {
        for row in 0..variables {
            let product = work.multiply(right.get(row, *coordinate), step)?;
            generators.values[row * directions.len() + column] = residue(&product, modulus, work)?;
        }
    }
    Ok(IntegerModularSystemOutcome::Solvable {
        particular,
        generators: generators.to_sparse(work)?,
        periods,
    })
}

/// Generate a complete finite family proposal or an actual scalar contradiction.
pub fn generate_integer_modular_system_candidate(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    generator: &mut impl BudgetMeter,
) -> Result<IntegerModularSystemCandidate> {
    let mut work = Work::new(generator);
    let (source, rhs) = inputs(source, rhs, modulus, &mut work)?;
    let smith = generate(source, &mut work)?;
    let proposed = outcome(&smith.witness, &rhs, modulus, &mut work)?;
    let modulus = work.copy(modulus)?;
    work.tick(1)?;
    Ok(IntegerModularSystemCandidate {
        source: smith.source,
        rhs,
        modulus,
        witness: IntegerModularSystemWitness {
            schema_version: INTEGER_MODULAR_SYSTEM_SCHEMA_VERSION,
            normal_form: smith.witness,
            outcome: proposed,
        },
    })
}

fn refused(error: IntegerModularSystemError) -> MatrixError {
    match error {
        IntegerModularSystemError::NormalForm(error) => verification(error),
        IntegerModularSystemError::Reference(SparseRrefError::Meter(error)) => stopped(error),
        IntegerModularSystemError::Reference(SparseRrefError::Arithmetic(error)) => match error {
            RationalArithmeticError::Meter(error) => stopped(error),
            RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
            RationalArithmeticError::InvariantViolation(reason) => {
                MatrixError::ArithmeticInvariantViolation(reason)
            }
            error => limit(&error.to_string()),
        },
        IntegerModularSystemError::Reference(SparseRrefError::Limit(reason)) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

/// Independently bind both exact originals and m, then check finite completeness.
pub fn verify_integer_modular_system_candidate(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    candidate: IntegerModularSystemCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIntegerModularSystem> {
    let mut work = Work::new(verifier);
    let (source, rhs) = inputs(source, rhs, modulus, &mut work)?;
    same_source(&source, &candidate.source, &mut work)?;
    same_source(&rhs, &candidate.rhs, &mut work)?;
    coefficient(&candidate.modulus, MAX_INPUT_BITS)?;
    work.tick(modulus.limb_count().max(candidate.modulus.limb_count()) as usize)?;
    if modulus != &candidate.modulus {
        return Err(MatrixError::InvalidCertificate(
            "modular system candidate differs from the original modulus".into(),
        ));
    }
    let receipt =
        verify_integer_modular_system(&source, &rhs, modulus, &candidate.witness, &mut work)
            .map_err(refused)?;
    work.tick(1)?;
    Ok(VerifiedIntegerModularSystem {
        source,
        rhs,
        witness: candidate.witness,
        receipt,
    })
}

/// Solve all RHS columns with separate cumulative generation/verification meters.
///
/// Dimensions are <=64, original entries/modulus <=4096 bits, and intermediate
/// integers <=8192 bits. Each phase retains the shared 20M-step/256MiB intrinsic
/// envelope. All residues are canonical. Inconsistency, nonuniqueness and
/// resource/cancellation failures remain distinct.
pub fn metered_solve_mod(
    source: IntegerMatrixRef<'_>,
    rhs: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIntegerModularSystem> {
    let candidate = generate_integer_modular_system_candidate(source, rhs, modulus, generator)?;
    generator.checkpoint().map_err(stopped)?;
    let accepted =
        verify_integer_modular_system_candidate(source, rhs, modulus, candidate, verifier)?;
    generator.checkpoint().map_err(stopped)?;
    verifier.checkpoint().map_err(stopped)?;
    Ok(accepted)
}

impl Matrix {
    /// Complete modular solutions, including composite moduli and rectangular A.
    pub fn solve_mod(
        &self,
        rhs: &Matrix,
        modulus: &BigInt,
    ) -> Result<VerifiedIntegerModularSystem> {
        self.metered_solve_mod(rhs, modulus, &mut Unbounded, &mut Unbounded)
    }

    pub fn metered_solve_mod(
        &self,
        rhs: &Matrix,
        modulus: &BigInt,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedIntegerModularSystem> {
        metered_solve_mod(self.into(), rhs.into(), modulus, generator, verifier)
    }
}

impl SparseMatrix {
    /// Complete modular solutions with bounded sparse source admission.
    pub fn solve_mod(
        &self,
        rhs: &SparseMatrix,
        modulus: &BigInt,
    ) -> Result<VerifiedIntegerModularSystem> {
        self.metered_solve_mod(rhs, modulus, &mut Unbounded, &mut Unbounded)
    }

    pub fn metered_solve_mod(
        &self,
        rhs: &SparseMatrix,
        modulus: &BigInt,
        generator: &mut impl BudgetMeter,
        verifier: &mut impl BudgetMeter,
    ) -> Result<VerifiedIntegerModularSystem> {
        metered_solve_mod(self.into(), rhs.into(), modulus, generator, verifier)
    }
}
