//! Exact primary-axis resultants and universal discriminants over F_p.
//!
//! Coefficients remain polynomials in the other original, ordered typed axes.
//! Generation uses a division-free subset recurrence for the Sylvester
//! determinant. Verification rebuilds its entries directly from the original
//! sources and sums signed permutation products in independent integer columns.
//! A discriminant retains the formal derivative degree even when its leading
//! coefficients vanish in positive characteristic. Its checker proves the
//! leading-coefficient product identity instead of repeating exact division.
//!
//! Inputs and outputs use the existing eight-axis / total-degree-64 / 4096-term
//! envelope, with prime at most 65521 and intermediate degree at most 128.
//! Nontrivial Sylvester matrices have at most eight rows (discriminants have
//! primary degree at most four). Every complete phase shares the cumulative
//! 20M-step / 128MiB logical-allocation envelope, including receipt copies.
//! These exact immutable in-process receipts are not portable proof objects.

#![forbid(unsafe_code)]

use crate::division::{PolynomialWork, refused};
use crate::finite_field_multivariate::{
    Columns, MAX_DEGREE, MAX_TERMS, PrimeFieldMultivariatePolynomial as Polynomial,
    PrimeFieldMultivariateTerms as Terms, accumulate, admit_ring, admit_terms, constant,
    copied_polynomial, copy_terms, divide, empty, identity_zero, monomial_copy, product,
    reference_add, reference_product, same_ring, scaled,
};
use crate::{PolyError, TermOrder};
use fsym_budget::BudgetMeter;

/// Maximum Sylvester dimension admitted before allocating a determinant table.
pub const MAX_PRIME_FIELD_MULTIVARIATE_SYLVESTER_SIZE: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimeFieldMultivariateEliminationOperation {
    /// Eliminate the first declared generator from two original polynomials.
    Resultant,
    /// The universal polynomial discriminant in the first declared generator.
    Discriminant,
}

/// Untrusted coefficient-ring candidate; only verification can accept it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldMultivariateEliminationCertificate {
    pub result: Terms,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedPrimeFieldMultivariateElimination {
    operation: PrimeFieldMultivariateEliminationOperation,
    left: Polynomial,
    right: Option<Polynomial>,
    result: Polynomial,
}

impl VerifiedPrimeFieldMultivariateElimination {
    pub fn operation(&self) -> PrimeFieldMultivariateEliminationOperation {
        self.operation
    }
    pub fn left(&self) -> &Polynomial {
        &self.left
    }
    pub fn right(&self) -> Option<&Polynomial> {
        self.right.as_ref()
    }
    /// The result retains exactly `left.generators()[1..]`, even for constants.
    pub fn result(&self) -> &Polynomial {
        &self.result
    }
    pub fn into_result(self) -> Polynomial {
        self.result
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Constant(u64),
    Sylvester { left: usize, right: usize },
}

fn primary_degree(
    polynomial: &Polynomial,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<usize, PolyError> {
    let mut result = 0;
    for monomial in polynomial.terms().keys() {
        work.tick(1)?;
        result = result.max(monomial[0] as usize);
    }
    Ok(result)
}

fn admission(
    operation: PrimeFieldMultivariateEliminationOperation,
    left: &Polynomial,
    right: Option<&Polynomial>,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Shape, PolyError> {
    if left.generators().is_empty() {
        return Err(refused("elimination requires a declared primary generator"));
    }
    admit_ring(left.generators(), left.prime(), work)?;
    same_ring(left, left.generators(), left.prime(), work)?;
    let m = primary_degree(left, work)?;
    let n = match operation {
        PrimeFieldMultivariateEliminationOperation::Resultant => {
            let right = right.ok_or_else(|| refused("resultant requires two operands"))?;
            same_ring(right, left.generators(), left.prime(), work)?;
            if left.is_zero() || right.is_zero() {
                return Ok(Shape::Constant(0));
            }
            let n = primary_degree(right, work)?;
            if m == 0 && n == 0 {
                return Ok(Shape::Constant(1));
            }
            n
        }
        PrimeFieldMultivariateEliminationOperation::Discriminant => {
            if right.is_some() {
                return Err(refused("discriminant requires one operand"));
            }
            if m <= 1 {
                return Ok(Shape::Constant(u64::from(m == 1)));
            }
            // This is the formal derivative degree, not the degree after
            // reduction modulo p. Trimming it changes the universal invariant.
            m - 1
        }
    };
    if m + n > MAX_PRIME_FIELD_MULTIVARIATE_SYLVESTER_SIZE {
        return Err(refused(
            "primary-axis Sylvester dimension exceeds eight (discriminant degree exceeds four)",
        ));
    }
    work.tick(1)?;
    Ok(Shape::Sylvester { left: m, right: n })
}

// Generation first groups coefficient polynomials, then lays out shifted
// descending rows. The checker below instead scatters original source terms
// directly into its own table and does not call either construction helper.
fn generated_coefficients(
    source: &Polynomial,
    degree: usize,
    derivative: bool,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Terms>, PolyError> {
    let length = if derivative { degree } else { degree + 1 };
    work.allocate(length * std::mem::size_of::<Terms>())?;
    let mut result = Vec::with_capacity(length);
    for _ in 0..length {
        result.push(empty(work)?);
    }
    for (monomial, &coefficient) in source.terms() {
        work.tick(1)?;
        let power = monomial[0] as usize;
        let (power, coefficient) = if derivative {
            if power == 0 {
                continue;
            }
            (power - 1, coefficient * power as u64 % source.prime())
        } else {
            (power, coefficient)
        };
        if coefficient != 0 {
            accumulate(
                &mut result[power],
                monomial_copy(&monomial[1..], work)?,
                coefficient,
                source.prime(),
                work,
            )?;
        }
    }
    Ok(result)
}

fn generated_matrix(
    left: &[Terms],
    right: &[Terms],
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Terms>, PolyError> {
    let (m, n) = (left.len() - 1, right.len() - 1);
    let size = m + n;
    work.allocate(size * size * std::mem::size_of::<Terms>())?;
    let mut matrix = Vec::with_capacity(size * size);
    for row in 0..size {
        let (coefficients, shift, degree) = if row < n {
            (left, row, m)
        } else {
            (right, row - n, n)
        };
        for column in 0..size {
            work.tick(1)?;
            matrix.push(if (shift..=shift + degree).contains(&column) {
                copy_terms(&coefficients[shift + degree - column], work)?
            } else {
                empty(work)?
            });
        }
    }
    Ok(matrix)
}

fn generated_determinant(
    matrix: &[Terms],
    size: usize,
    width: usize,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let count = 1usize << size;
    work.allocate(count * std::mem::size_of::<Terms>())?;
    let mut minors = Vec::with_capacity(count);
    minors.push(constant(width, 1, work)?);
    // A mask denotes columns left for the final popcount(mask) rows.
    // Expanding their first row yields only already-computed smaller masks.
    for mask in 1..count {
        work.tick(size)?;
        let row = size - mask.count_ones() as usize;
        let mut value = empty(work)?;
        let mut position = 0usize;
        for column in 0..size {
            let bit = 1usize << column;
            if mask & bit == 0 {
                continue;
            }
            let cell = &matrix[row * size + column];
            let minor = &minors[mask ^ bit];
            if !cell.is_empty() && !minor.is_empty() {
                let term = product(cell, minor, prime, work)?;
                for (monomial, coefficient) in term {
                    let coefficient = if position.is_multiple_of(2) {
                        coefficient
                    } else {
                        prime - coefficient
                    };
                    accumulate(&mut value, monomial, coefficient, prime, work)?;
                }
            }
            position += 1;
        }
        minors.push(value);
    }
    work.tick(1)?;
    Ok(minors.pop().expect("empty minor is always present"))
}

fn generated_result(
    operation: PrimeFieldMultivariateEliminationOperation,
    left: &Polynomial,
    right: Option<&Polynomial>,
    shape: Shape,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let Shape::Sylvester { left: m, right: n } = shape else {
        let Shape::Constant(value) = shape else {
            unreachable!("shape is constant or Sylvester")
        };
        return constant(left.generators().len() - 1, value, work);
    };
    let coefficients = generated_coefficients(left, m, false, work)?;
    let other = match operation {
        PrimeFieldMultivariateEliminationOperation::Resultant => {
            generated_coefficients(right.expect("admitted resultant operand"), n, false, work)?
        }
        PrimeFieldMultivariateEliminationOperation::Discriminant => {
            generated_coefficients(left, m, true, work)?
        }
    };
    let matrix = generated_matrix(&coefficients, &other, work)?;
    let determinant = generated_determinant(
        &matrix,
        m + n,
        left.generators().len() - 1,
        left.prime(),
        work,
    )?;
    if operation == PrimeFieldMultivariateEliminationOperation::Resultant {
        return Ok(determinant);
    }
    let signed = if (m * (m - 1) / 2).is_multiple_of(2) {
        determinant
    } else {
        scaled(&determinant, left.prime() - 1, left.prime(), work)?
    };
    let mut division = divide(
        &signed,
        &[&coefficients[m]],
        TermOrder::Lex,
        left.prime(),
        work,
    )?;
    if !division.remainder.is_empty() {
        return Err(PolyError::IdentityCheckFailed(
            "universal discriminant requires an exact leading-coefficient quotient".into(),
        ));
    }
    Ok(division.quotients.pop().expect("one exact divisor"))
}

fn columns(work: &mut PolynomialWork<'_, impl BudgetMeter>) -> Result<Columns, PolyError> {
    work.allocate(std::mem::size_of::<Columns>())?;
    Ok(Columns::new())
}

fn reference_insert(
    output: &mut Terms,
    monomial: &[u32],
    coefficient: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    work.tick(monomial.len() + 1)?;
    if coefficient == 0 {
        return Ok(());
    }
    if output.len() == MAX_TERMS {
        return Err(refused("reference coefficient term bound"));
    }
    work.allocate(128)?;
    if output
        .insert(monomial_copy(monomial, work)?, coefficient)
        .is_some()
    {
        return Err(PolyError::IdentityCheckFailed(
            "duplicate original coefficient coordinate".into(),
        ));
    }
    Ok(())
}

fn reference_matrix(
    operation: PrimeFieldMultivariateEliminationOperation,
    left: &Polynomial,
    right: Option<&Polynomial>,
    m: usize,
    n: usize,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Vec<Terms>, PolyError> {
    let size = m + n;
    work.allocate(size * size * std::mem::size_of::<Terms>())?;
    let mut matrix = Vec::with_capacity(size * size);
    for _ in 0..size * size {
        matrix.push(empty(work)?);
    }
    // Scatter each original coefficient through all its shifted rows. This
    // neither consumes the generator's coefficient grouping nor reuses its
    // row/column lookup or derivative construction.
    for (monomial, &coefficient) in left.terms() {
        for row in 0..n {
            let column = row + m - monomial[0] as usize;
            reference_insert(
                &mut matrix[row * size + column],
                &monomial[1..],
                coefficient,
                work,
            )?;
        }
    }
    let derivative = operation == PrimeFieldMultivariateEliminationOperation::Discriminant;
    let source = if derivative {
        left
    } else {
        right.expect("admitted resultant operand")
    };
    for (monomial, &value) in source.terms() {
        work.tick(1)?;
        if derivative && monomial[0] == 0 {
            continue;
        }
        let power = monomial[0] as usize - usize::from(derivative);
        let coefficient = if derivative {
            (u128::from(value) * u128::from(monomial[0]) % u128::from(left.prime())) as u64
        } else {
            value
        };
        for shift in 0..m {
            let column = shift + n - power;
            reference_insert(
                &mut matrix[(n + shift) * size + column],
                &monomial[1..],
                coefficient,
                work,
            )?;
        }
    }
    Ok(matrix)
}

// Independent coefficient-column multiplication. No generation sum, product,
// exact division, subset recurrence or modular monomial update is called.
fn reference_multiply(
    left: &Terms,
    right: &Terms,
    prime: u64,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<Terms, PolyError> {
    let mut integer_columns = columns(work)?;
    reference_product(&mut integer_columns, left, right, 1, work)?;
    let mut result = empty(work)?;
    for (monomial, coefficient) in integer_columns {
        work.tick(1)?;
        let coefficient = coefficient.rem_euclid(i128::from(prime)) as u64;
        if coefficient != 0 {
            if result.len() == MAX_TERMS {
                return Err(refused("reference product term bound"));
            }
            work.allocate(128)?;
            result.insert(monomial, coefficient);
        }
    }
    Ok(result)
}

struct PermutationCheck<'a> {
    matrix: &'a [Terms],
    size: usize,
    prime: u64,
    total: Columns,
}

impl PermutationCheck<'_> {
    fn visit(
        &mut self,
        row: usize,
        occupied: usize,
        value: &Terms,
        negative: bool,
        work: &mut PolynomialWork<'_, impl BudgetMeter>,
    ) -> Result<(), PolyError> {
        work.tick(1)?;
        if row == self.size {
            return reference_add(&mut self.total, value, if negative { -1 } else { 1 }, work);
        }
        for column in 0..self.size {
            work.tick(1)?;
            if occupied & (1usize << column) != 0 {
                continue;
            }
            let cell = &self.matrix[row * self.size + column];
            if cell.is_empty() {
                continue;
            }
            let next = reference_multiply(value, cell, self.prime, work)?;
            let inversions = (occupied >> (column + 1)).count_ones();
            self.visit(
                row + 1,
                occupied | (1usize << column),
                &next,
                negative ^ !inversions.is_multiple_of(2),
                work,
            )?;
        }
        Ok(())
    }
}

fn reference_identity(
    operation: PrimeFieldMultivariateEliminationOperation,
    left: &Polynomial,
    right: Option<&Polynomial>,
    shape: Shape,
    candidate: &Terms,
    work: &mut PolynomialWork<'_, impl BudgetMeter>,
) -> Result<(), PolyError> {
    let Shape::Sylvester { left: m, right: n } = shape else {
        let Shape::Constant(value) = shape else {
            unreachable!("shape is constant or Sylvester")
        };
        let expected = constant(left.generators().len() - 1, value, work)?;
        let mut comparison = columns(work)?;
        reference_add(&mut comparison, &expected, 1, work)?;
        reference_add(&mut comparison, candidate, -1, work)?;
        return identity_zero(&comparison, left.prime(), work);
    };
    let matrix = reference_matrix(operation, left, right, m, n, work)?;
    let one = constant(left.generators().len() - 1, 1, work)?;
    let discriminant = operation == PrimeFieldMultivariateEliminationOperation::Discriminant;
    let mut checker = PermutationCheck {
        matrix: &matrix,
        size: m + n,
        prime: left.prime(),
        total: columns(work)?,
    };
    checker.visit(
        0,
        0,
        &one,
        discriminant && !(m * (m - 1) / 2).is_multiple_of(2),
        work,
    )?;
    if discriminant {
        let mut leading = empty(work)?;
        for (monomial, &coefficient) in left.terms() {
            work.tick(1)?;
            if monomial[0] as usize == m {
                reference_insert(&mut leading, &monomial[1..], coefficient, work)?;
            }
        }
        // In the coefficient polynomial domain the original leading
        // coefficient is nonzero, so this identity uniquely checks the exact
        // quotient even when the full determinant and candidate are zero.
        reference_product(&mut checker.total, &leading, candidate, -1, work)?;
    } else {
        reference_add(&mut checker.total, candidate, -1, work)?;
    }
    identity_zero(&checker.total, left.prime(), work)
}

pub fn verify_prime_field_multivariate_elimination(
    operation: &PrimeFieldMultivariateEliminationOperation,
    left: &Polynomial,
    right: Option<&Polynomial>,
    certificate: &PrimeFieldMultivariateEliminationCertificate,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateElimination, PolyError> {
    let mut work = PolynomialWork::new(verifier);
    let shape = admission(*operation, left, right, &mut work)?;
    admit_terms(
        &certificate.result,
        left.generators().len() - 1,
        left.prime(),
        MAX_DEGREE,
        false,
        &mut work,
    )?;
    reference_identity(
        *operation,
        left,
        right,
        shape,
        &certificate.result,
        &mut work,
    )?;
    work.allocate(std::mem::size_of::<VerifiedPrimeFieldMultivariateElimination>())?;
    let accepted = VerifiedPrimeFieldMultivariateElimination {
        operation: *operation,
        left: copied_polynomial(left.generators(), left.prime(), left.terms(), &mut work)?,
        right: right
            .map(|value| {
                copied_polynomial(value.generators(), value.prime(), value.terms(), &mut work)
            })
            .transpose()?,
        result: copied_polynomial(
            &left.generators()[1..],
            left.prime(),
            &certificate.result,
            &mut work,
        )?,
    };
    work.tick(1)?;
    Ok(accepted)
}

pub fn checked_prime_field_multivariate_elimination(
    operation: &PrimeFieldMultivariateEliminationOperation,
    left: &Polynomial,
    right: Option<&Polynomial>,
    generator: &mut impl BudgetMeter,
    verifier: &mut impl BudgetMeter,
) -> Result<VerifiedPrimeFieldMultivariateElimination, PolyError> {
    let mut generation = PolynomialWork::new(generator);
    let mut verification = PolynomialWork::new(verifier);
    let shape = admission(*operation, left, right, &mut generation)?;
    generation.allocate(std::mem::size_of::<
        PrimeFieldMultivariateEliminationCertificate,
    >())?;
    let certificate = PrimeFieldMultivariateEliminationCertificate {
        result: generated_result(*operation, left, right, shape, &mut generation)?,
    };
    let accepted = verify_prime_field_multivariate_elimination(
        operation,
        left,
        right,
        &certificate,
        &mut verification,
    )?;
    generation.tick(1)?;
    verification.tick(1)?;
    Ok(accepted)
}
