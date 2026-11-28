//! Division-free tridiagonal determinants over Q[x]. The verifier evaluates
//! the original entries and checks pivoted rational LU at enough distinct
//! points to prove the entire polynomial identity, including a zero result.

use crate::{Matrix, MatrixError, SparseMatrix, from_rational, numeric_value};
use fsym_budget::{Budget, BudgetLimits, Dimension};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_polys::UnivariatePoly as Poly;

const MAX_AXIS: usize = 1024;
const MAX_DEGREE: usize = 64;
const MAX_SOURCE_HEIGHT: u64 = 4096;
const MAX_NODES: usize = 8192;
const MAX_SAMPLE_ROWS: usize = 8192;

fn limit() -> MatrixError {
    MatrixError::ResourceLimit("tridiagonal polynomial determinant source or work bound".into())
}

fn poly_error(error: fsym_polys::PolyError) -> MatrixError {
    MatrixError::InvalidPolynomial(error.to_string())
}

// h bounds both the common denominator and the sum of absolute integer
// coefficients after clearing that denominator by 2^h. Addition needs one
// extra bit; multiplication adds heights. Bound syntax *before* expansion.
#[derive(Clone, Copy, Default)]
struct Size {
    degree: usize,
    height: u64,
}

struct Source<'a> {
    symbol: Option<&'a Symbol>,
    nodes: usize,
}

impl<'a> Source<'a> {
    fn inspect(&mut self, expr: &'a Expr, depth: usize) -> Result<Option<Size>, MatrixError> {
        self.nodes += 1;
        if self.nodes > MAX_NODES || depth > 32 {
            return Err(limit());
        }
        let size = match expr {
            Expr::Integer(n) => Size {
                degree: 0,
                height: n.bits(),
            },
            Expr::Rational(r) => Size {
                degree: 0,
                height: r.numer().bits().max(r.denom().bits()),
            },
            Expr::Sym(s) => {
                if s.name.len() > 1024 {
                    return Err(limit());
                }
                if self.symbol.is_some_and(|previous| previous != s) {
                    return Ok(None);
                }
                self.symbol = Some(s);
                Size {
                    degree: 1,
                    height: 0,
                }
            }
            Expr::Add(children) | Expr::Mul(children) => {
                let add = matches!(expr, Expr::Add(_));
                let mut total = Size::default();
                for child in children {
                    let Some(next) = self.inspect(child, depth + 1)? else {
                        return Ok(None);
                    };
                    total.degree = if add {
                        total.degree.max(next.degree)
                    } else {
                        total.degree + next.degree
                    };
                    total.height += next.height + u64::from(add);
                    if total.degree > MAX_DEGREE || total.height > MAX_SOURCE_HEIGHT {
                        return Err(limit());
                    }
                }
                total
            }
            Expr::Pow(base, exponent) => {
                let Expr::Integer(n) = exponent.as_ref() else {
                    return Ok(None);
                };
                // Only positive powers are polynomial on the entire source domain.
                let Ok(power) = usize::try_from(n) else {
                    return Ok(None);
                };
                if power == 0 {
                    return Ok(None);
                }
                if power > MAX_DEGREE {
                    return Err(limit());
                }
                let Some(base) = self.inspect(base, depth + 1)? else {
                    return Ok(None);
                };
                Size {
                    degree: base.degree * power,
                    height: base.height * power as u64,
                }
            }
            _ => return Ok(None),
        };
        if size.degree > MAX_DEGREE || size.height > MAX_SOURCE_HEIGHT {
            return Err(limit());
        }
        Ok(Some(size))
    }
}

// Evaluate the admitted source tree directly. This reference path never
// lowers inputs using the producer's polynomial expansion or its row indexing.
fn evaluate(expr: &Expr, point: &BigRational) -> Result<BigRational, MatrixError> {
    use num_traits::{One, Zero};
    match expr {
        Expr::Integer(n) => Ok(BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Ok(r.clone()),
        Expr::Sym(_) => Ok(point.clone()),
        Expr::Add(children) => children
            .iter()
            .try_fold(BigRational::zero(), |v, e| Ok(v + evaluate(e, point)?)),
        Expr::Mul(children) => children
            .iter()
            .try_fold(BigRational::one(), |v, e| Ok(v * evaluate(e, point)?)),
        Expr::Pow(base, exp) => {
            if let Expr::Integer(n) = exp.as_ref()
                && let Some(k) = n.to_i64()
                && (1..=MAX_DEGREE as i64).contains(&k)
            {
                return evaluate(base, point)?.pow(k as i32).map_err(|_| limit());
            }
            Err(MatrixError::InvalidCertificate(
                "nonpolynomial source power".into(),
            ))
        }
        _ => Err(MatrixError::InvalidCertificate(
            "nonpolynomial source cell".into(),
        )),
    }
}

struct Band<'a> {
    n: usize,
    symbol: &'a Symbol,
    degree_bound: usize,
    entries: Vec<(usize, usize, &'a Expr, Poly)>,
}

impl<'a> Band<'a> {
    fn read(
        n: usize,
        entries: impl Iterator<Item = (usize, usize, &'a Expr)>,
    ) -> Result<Option<Self>, MatrixError> {
        if n > MAX_AXIS {
            return Err(limit());
        }
        let mut source = Source {
            symbol: None,
            nodes: 0,
        };
        let mut row_degrees = vec![0; n];
        let mut height = 0;
        let mut originals = Vec::new();
        for (r, c, entry) in entries {
            if entry.is_zero() {
                continue;
            }
            if r.abs_diff(c) > 1 {
                return Ok(None);
            }
            let Some(size) = source.inspect(entry, 0)? else {
                return Ok(None);
            };
            row_degrees[r] = row_degrees[r].max(size.degree);
            height += size.height;
            if height > MAX_SOURCE_HEIGHT {
                return Err(limit());
            }
            originals.push((r, c, entry));
        }
        let Some(symbol) = source.symbol else {
            return Ok(None);
        };
        // Each determinant monomial chooses exactly one entry from each row.
        let degree_bound: usize = row_degrees.iter().sum();
        if degree_bound > MAX_DEGREE || n * (degree_bound + 1) > MAX_SAMPLE_ROWS {
            return Err(limit());
        }
        let entries = originals
            .into_iter()
            .map(|(r, c, e)| Ok((r, c, e, Poly::from_expr(e, symbol).map_err(poly_error)?)))
            .collect::<Result<_, MatrixError>>()?;
        Ok(Some(Self {
            n,
            symbol,
            degree_bound,
            entries,
        }))
    }

    fn continuant(&self) -> Result<Poly, MatrixError> {
        let zero = Poly::zero(self.symbol.clone());
        let mut rows = vec![[&zero; 3]; self.n];
        for (r, c, _, p) in &self.entries {
            rows[*r][c + 1 - r] = p;
        }
        let mut prior = zero.clone();
        let mut current = Poly::one(self.symbol.clone());
        for (r, &[lower, diagonal, _]) in rows.iter().enumerate() {
            let mut next = diagonal.mul(&current).map_err(poly_error)?;
            if r != 0 && !lower.is_zero() && !rows[r - 1][2].is_zero() {
                let pair = lower.mul(rows[r - 1][2]).map_err(poly_error)?;
                next = next
                    .sub(&pair.mul(&prior).map_err(poly_error)?)
                    .map_err(poly_error)?;
            }
            prior = current;
            current = next;
        }
        Ok(current)
    }

    fn verify(&self, candidate: &Poly) -> Result<(), MatrixError> {
        if candidate.coeffs.len() > self.degree_bound + 1
            || &candidate.gen_sym != self.symbol
            || candidate.validate_shape().is_err()
            || candidate.coeffs.iter().any(|c| {
                c.numer().bits().max(c.denom().bits()) > MAX_SOURCE_HEIGHT + 2 * MAX_AXIS as u64
            })
        {
            return Err(MatrixError::InvalidCertificate(
                "tridiagonal determinant generator or degree".into(),
            ));
        }
        let mut limits = BudgetLimits::uniform(256 * 1024 * 1024, 0);
        limits.dimensions[Dimension::ComputeSteps.index()] = 25_000_000;
        let mut meter = Budget::new(limits);
        for i in 0..=self.degree_bound {
            // Center the distinct integer evaluation points to keep heights small.
            let point =
                BigRational::from_integer((i as i64 - (self.degree_bound / 2) as i64).into());
            let evaluated = self
                .entries
                .iter()
                .map(|(r, c, e, _)| Ok(((*r, *c), from_rational(evaluate(e, &point)?))))
                .collect::<Result<_, MatrixError>>()?;
            let matrix = SparseMatrix::new(self.n, self.n, evaluated)?;
            let reference = matrix.metered_det(&mut meter)?;
            if numeric_value(&reference).as_ref() != Some(&candidate.eval(&point)) {
                return Err(MatrixError::InvalidCertificate(
                    "tridiagonal determinant specialization".into(),
                ));
            }
        }
        Ok(())
    }
}

fn checked<'a>(
    n: usize,
    entries: impl Iterator<Item = (usize, usize, &'a Expr)>,
) -> Result<Option<Expr>, MatrixError> {
    let Some(band) = Band::read(n, entries)? else {
        return Ok(None);
    };
    let candidate = band.continuant()?;
    band.verify(&candidate)?;
    Ok(Some(candidate.to_expr()))
}

impl Matrix {
    pub(crate) fn tridiagonal_polynomial_det(&self) -> Result<Option<Expr>, MatrixError> {
        let n = self.rows;
        checked(
            n,
            self.data.iter().enumerate().map(|(i, e)| (i / n, i % n, e)),
        )
    }
}

impl SparseMatrix {
    /// A checked determinant over Q[x] for a square tridiagonal source.
    /// Unsupported structure or scalar syntax returns None. Resource failures
    /// are terminal. Only stored coordinates are inspected or materialized.
    pub fn tridiagonal_polynomial_det(&self) -> Result<Option<Expr>, MatrixError> {
        if self.rows() != self.cols() {
            return Err(MatrixError::NotSquare(self.rows(), self.cols()));
        }
        checked(
            self.rows(),
            self.entries().iter().map(|(&(r, c), e)| (r, c, e)),
        )
    }

    /// Verify an untrusted determinant polynomial against the original sparse
    /// matrix, independently of the division-free continuant producer.
    pub fn verify_tridiagonal_polynomial_determinant(
        &self,
        candidate: &Poly,
    ) -> Result<(), MatrixError> {
        if self.rows() != self.cols() {
            return Err(MatrixError::NotSquare(self.rows(), self.cols()));
        }
        let band = Band::read(
            self.rows(),
            self.entries().iter().map(|(&(r, c), e)| (r, c, e)),
        )?
        .ok_or_else(|| {
            MatrixError::InvalidCertificate(
                "expected a univariate polynomial tridiagonal source".into(),
            )
        })?;
        band.verify(candidate)
    }
}
