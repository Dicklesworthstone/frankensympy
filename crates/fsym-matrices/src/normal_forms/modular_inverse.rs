//! Bounded integer modular inversion by Smith composition.
//!
//! Over Z/mZ, D = U A V is invertible exactly when every Smith factor is a
//! unit. Existing metered scalar inversion supplies D^-1; modular products
//! form V D^-1 U. The proof kernel independently checks the original A,m,
//! complete Smith witness and both modular residuals, or a nonunit divisor.
//! Prime and composite moduli use the same path and cumulative phase meters.

use super::integer_system::{IntegerMatrixRef, same_source};
use super::{
    BigInt, IntegerMatrix, MAX_INPUT_BITS, Matrix, MatrixError, RationalArithmeticError,
    RationalSparseMatrix, Result, SparseMatrix, SparseRrefError, Work, coefficient,
    generate, limit, stopped, verification,
};
use fsym_budget::{BudgetMeter, Unbounded};
use fsym_core::arith::{metered_gcd, metered_mod_inverse};
use fsym_proof_kernel::integer_modular_inverse::{
    INTEGER_MODULAR_INVERSE_SCHEMA_VERSION, IntegerModularInverseError,
    IntegerModularInverseOutcome, IntegerModularInverseReceipt, IntegerModularInverseWitness,
    MAX_INTEGER_MODULAR_MODULUS_BITS, verify_integer_modular_inverse,
};
use fsym_proof_kernel::integer_normal_form::IntegerSmithWitness;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerModularInverseCandidate {
    pub source: RationalSparseMatrix,
    pub modulus: BigInt,
    pub witness: IntegerModularInverseWitness,
}

/// Immutable source-bound inverse, or independently checked noninvertibility.
#[derive(Debug)]
pub struct VerifiedIntegerModularInverse {
    source: RationalSparseMatrix,
    witness: IntegerModularInverseWitness,
    receipt: IntegerModularInverseReceipt,
}

impl VerifiedIntegerModularInverse {
    pub fn source(&self) -> &RationalSparseMatrix {
        &self.source
    }
    pub fn modulus(&self) -> &BigInt {
        self.receipt.modulus()
    }
    pub fn inverse(&self) -> Option<&RationalSparseMatrix> {
        match &self.witness.outcome {
            IntegerModularInverseOutcome::Invertible { inverse } => Some(inverse),
            IntegerModularInverseOutcome::NonUnit { .. } => None,
        }
    }
    pub fn witness(&self) -> &IntegerModularInverseWitness {
        &self.witness
    }
    pub fn receipt(&self) -> &IntegerModularInverseReceipt {
        &self.receipt
    }
    pub fn into_parts(
        self,
    ) -> (
        RationalSparseMatrix,
        IntegerModularInverseWitness,
        IntegerModularInverseReceipt,
    ) {
        (self.source, self.witness, self.receipt)
    }
}

fn inputs(
    source: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<RationalSparseMatrix> {
    work.tick(1)?;
    if !modulus.is_positive() || modulus.is_one() {
        return Err(MatrixError::InvalidModulus("must be an integer at least two"));
    }
    coefficient(modulus, MAX_INTEGER_MODULAR_MODULUS_BITS)?;
    source.admit(work)?;
    let (rows, columns) = source.shape();
    if rows != columns {
        return Err(MatrixError::NotSquare(rows, columns));
    }
    source.snapshot(work)
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

fn reduced_matrix(
    source: &RationalSparseMatrix,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<IntegerMatrix> {
    let mut matrix = IntegerMatrix::zero(source.rows, source.columns, work)?;
    for entry in &source.entries {
        matrix.values[entry.row * source.columns + entry.column] =
            residue(entry.value.numer(), modulus, work)?;
    }
    Ok(matrix)
}

fn outcome(
    smith: &IntegerSmithWitness,
    modulus: &BigInt,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<IntegerModularInverseOutcome> {
    let size = smith.diagonal.rows;
    let rank = smith.diagonal.entries.len();
    if rank < size {
        return Ok(IntegerModularInverseOutcome::NonUnit {
            index: rank,
            divisor: work.copy(modulus)?,
        });
    }
    let mut inverses = work.vector(size)?;
    for (index, entry) in smith.diagonal.entries.iter().enumerate() {
        work.tick(1)?;
        let Some(inverse) =
            metered_mod_inverse(entry.value.numer(), modulus, work).map_err(stopped)?
        else {
            return Ok(IntegerModularInverseOutcome::NonUnit {
                index,
                divisor: metered_gcd(entry.value.numer(), modulus, work).map_err(stopped)?,
            });
        };
        inverses.push(inverse);
    }

    let mut left = reduced_matrix(&smith.left, modulus, work)?;
    let right = reduced_matrix(&smith.right, modulus, work)?;
    // Scale each U row by its diagonal inverse before multiplying by V.
    for (row, inverse) in inverses.iter().enumerate() {
        for column in 0..size {
            let product = work.multiply(inverse, left.get(row, column))?;
            left.values[row * size + column] = residue(&product, modulus, work)?;
        }
    }
    let mut result = IntegerMatrix::zero(size, size, work)?;
    for row in 0..size {
        for inner in 0..size {
            work.tick(1)?;
            let a = right.get(row, inner);
            if a.is_zero() {
                continue;
            }
            for column in 0..size {
                work.tick(1)?;
                let b = left.get(inner, column);
                if b.is_zero() {
                    continue;
                }
                let product = work.multiply(a, b)?;
                let negative = -residue(&product, modulus, work)?;
                let sum = work.subtract(result.get(row, column), &negative)?;
                result.values[row * size + column] = residue(&sum, modulus, work)?;
            }
        }
    }
    Ok(IntegerModularInverseOutcome::Invertible {
        inverse: result.to_sparse(work)?,
    })
}

/// Generate a bounded untrusted inverse or a nonunit Smith-factor witness.
pub fn generate_integer_modular_inverse_candidate(
    source: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    generator: &mut impl BudgetMeter,
) -> Result<IntegerModularInverseCandidate> {
    let mut work = Work::new(generator);
    let original = inputs(source, modulus, &mut work)?;
    let smith = generate(original, &mut work)?;
    let outcome = outcome(&smith.witness, modulus, &mut work)?;
    let modulus = work.copy(modulus)?;
    work.tick(1)?;
    Ok(IntegerModularInverseCandidate {
        source: smith.source,
        modulus,
        witness: IntegerModularInverseWitness {
            schema_version: INTEGER_MODULAR_INVERSE_SCHEMA_VERSION,
            normal_form: smith.witness,
            outcome,
        },
    })
}

fn refused(error: IntegerModularInverseError) -> MatrixError {
    match error {
        IntegerModularInverseError::NormalForm(error) => verification(error),
        IntegerModularInverseError::Reference(SparseRrefError::Meter(error)) => stopped(error),
        IntegerModularInverseError::Reference(SparseRrefError::Arithmetic(error)) => match error {
            RationalArithmeticError::Meter(error) => stopped(error),
            RationalArithmeticError::DivisionByZero => MatrixError::DivisionByZero,
            RationalArithmeticError::InvariantViolation(reason) => {
                MatrixError::ArithmeticInvariantViolation(reason)
            }
            error => limit(&error.to_string()),
        },
        IntegerModularInverseError::Reference(SparseRrefError::Limit(reason)) => limit(reason),
        error => MatrixError::InvalidCertificate(error.to_string()),
    }
}

/// Bind the original matrix and modulus before independent acceptance.
pub fn verify_integer_modular_inverse_candidate(
    source: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    candidate: IntegerModularInverseCandidate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIntegerModularInverse> {
    let mut work = Work::new(verifier);
    let original = inputs(source, modulus, &mut work)?;
    same_source(&original, &candidate.source, &mut work)?;
    coefficient(&candidate.modulus, MAX_INPUT_BITS)?;
    work.tick(modulus.limb_count().max(candidate.modulus.limb_count()) as usize)?;
    if modulus != &candidate.modulus {
        return Err(MatrixError::InvalidCertificate(
            "modular inverse candidate differs from the original modulus".into(),
        ));
    }
    let receipt =
        verify_integer_modular_inverse(&original, modulus, &candidate.witness, &mut work)
            .map_err(refused)?;
    work.tick(1)?;
    Ok(VerifiedIntegerModularInverse {
        source: original,
        witness: candidate.witness,
        receipt,
    })
}

/// Complete exact inversion over an arbitrary integer residue ring.
/// Each dimension is <=64 and source coefficients/modulus have <=4096 bits.
pub fn metered_inverse_mod(
    source: IntegerMatrixRef<'_>,
    modulus: &BigInt,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedIntegerModularInverse> {
    let candidate = generate_integer_modular_inverse_candidate(source, modulus, generator)?;
    generator.checkpoint().map_err(stopped)?;
    let accepted =
        verify_integer_modular_inverse_candidate(source, modulus, candidate, verifier)?;
    generator.checkpoint().map_err(stopped)?;
    verifier.checkpoint().map_err(stopped)?;
    Ok(accepted)
}

impl Matrix {
    pub fn inverse_mod(&self, modulus: &BigInt) -> Result<VerifiedIntegerModularInverse> {
        metered_inverse_mod(self.into(), modulus, &mut Unbounded, &mut Unbounded)
    }
}

impl SparseMatrix {
    pub fn inverse_mod(&self, modulus: &BigInt) -> Result<VerifiedIntegerModularInverse> {
        metered_inverse_mod(self.into(), modulus, &mut Unbounded, &mut Unbounded)
    }
}
