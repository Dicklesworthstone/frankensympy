//! Mathematical sort definitions for WS04 term typing.
//!
//! Provides the semantic sort classification distinguishing scalar quantities,
//! booleans, matrices/tensors, sets, and function types.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

/// Semantic mathematical sort classification for typed symbols and terms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
pub enum Sort {
    /// Scalar generic number.
    Scalar,
    /// Boolean truth value.
    Boolean,
    /// Integer sort $\mathbb{Z}$.
    Integer,
    /// Rational number sort $\mathbb{Q}$.
    Rational,
    /// Real number sort $\mathbb{R}$.
    Real,
    /// Complex number sort $\mathbb{C}$.
    Complex,
    /// Matrix sort with optional row/column dimensions.
    Matrix {
        rows: Option<usize>,
        cols: Option<usize>,
    },
    /// Set / collection sort.
    Set,
    /// Function sort mapping parameter sorts to return sort.
    Function { dom: Vec<Sort>, codom: Box<Sort> },
    /// Unknown / generic untyped sort.
    #[default]
    Unknown,
}

impl Sort {
    /// Returns true if this sort is a numeric scalar (Integer, Rational, Real, Complex, or general Scalar).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Sort::Scalar | Sort::Integer | Sort::Rational | Sort::Real | Sort::Complex
        )
    }

    /// Returns true if this sort is Boolean.
    pub fn is_boolean(&self) -> bool {
        matches!(self, Sort::Boolean)
    }

    /// Checks if `self` is a valid subsort of `other` under the canonical numerical tower:
    /// Integer $\subseteq$ Rational $\subseteq$ Real $\subseteq$ Complex $\subseteq$ Scalar.
    pub fn is_subsort_of(&self, other: &Sort) -> bool {
        if self == other {
            return true;
        }
        if *other == Sort::Unknown || *other == Sort::Scalar {
            return self.is_numeric();
        }
        match self {
            Sort::Integer => matches!(
                other,
                Sort::Rational | Sort::Real | Sort::Complex | Sort::Scalar
            ),
            Sort::Rational => matches!(other, Sort::Real | Sort::Complex | Sort::Scalar),
            Sort::Real => matches!(other, Sort::Complex | Sort::Scalar),
            Sort::Complex => matches!(other, Sort::Scalar),
            Sort::Matrix { rows: r1, cols: c1 } => match other {
                Sort::Matrix { rows: r2, cols: c2 } => {
                    (r2.is_none() || r1 == r2) && (c2.is_none() || c1 == c2)
                }
                _ => false,
            },
            Sort::Function { dom: d1, codom: c1 } => match other {
                Sort::Function { dom: d2, codom: c2 } => {
                    d1.len() == d2.len()
                        && d1.iter().zip(d2.iter()).all(|(a, b)| b.is_subsort_of(a)) // contravariant in domain
                        && c1.is_subsort_of(c2) // covariant in codomain
                }
                _ => false,
            },
            _ => false,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Scalar => write!(f, "Scalar"),
            Sort::Boolean => write!(f, "Boolean"),
            Sort::Integer => write!(f, "Integer"),
            Sort::Rational => write!(f, "Rational"),
            Sort::Real => write!(f, "Real"),
            Sort::Complex => write!(f, "Complex"),
            Sort::Matrix { rows, cols } => match (rows, cols) {
                (Some(r), Some(c)) => write!(f, "Matrix({}, {})", r, c),
                (Some(r), None) => write!(f, "Matrix({}, ?)", r),
                (None, Some(c)) => write!(f, "Matrix(?, {})", c),
                (None, None) => write!(f, "Matrix"),
            },
            Sort::Set => write!(f, "Set"),
            Sort::Function { dom, codom } => {
                write!(f, "(")?;
                for (i, d) in dom.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", d)?;
                }
                write!(f, ") -> {}", codom)
            }
            Sort::Unknown => write!(f, "Unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_hierarchy_and_subsorts() {
        assert!(Sort::Integer.is_subsort_of(&Sort::Rational));
        assert!(Sort::Integer.is_subsort_of(&Sort::Real));
        assert!(Sort::Integer.is_subsort_of(&Sort::Complex));
        assert!(Sort::Integer.is_subsort_of(&Sort::Scalar));

        assert!(Sort::Rational.is_subsort_of(&Sort::Real));
        assert!(Sort::Rational.is_subsort_of(&Sort::Complex));
        assert!(!Sort::Real.is_subsort_of(&Sort::Integer));
        assert!(!Sort::Complex.is_subsort_of(&Sort::Real));

        assert!(Sort::Boolean.is_boolean());
        assert!(!Sort::Boolean.is_numeric());
        assert!(!Sort::Boolean.is_subsort_of(&Sort::Real));

        let m1 = Sort::Matrix {
            rows: Some(3),
            cols: Some(3),
        };
        let m_any = Sort::Matrix {
            rows: None,
            cols: None,
        };
        assert!(m1.is_subsort_of(&m_any));
        assert!(!m_any.is_subsort_of(&m1));
    }

    #[test]
    fn test_sort_display() {
        assert_eq!(Sort::Integer.to_string(), "Integer");
        assert_eq!(Sort::Boolean.to_string(), "Boolean");
        let fn_sort = Sort::Function {
            dom: vec![Sort::Real, Sort::Real],
            codom: Box::new(Sort::Real),
        };
        assert_eq!(fn_sort.to_string(), "(Real, Real) -> Real");
    }
}
