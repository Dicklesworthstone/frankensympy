//! Mathematical domain hierarchy and coercion graph for WS04.
//!
//! Formal domains: $\mathbb{Z}$, $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$,
//! polynomial rings $D[x]$, rational function fields $D(x)$, and finite fields $\mathbb{F}_p$.
//! Coercions are explicit, typed, and emit deterministic receipts.

use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DomainError {
    #[error("Cannot coerce expression {0} from {1} to {2}")]
    CoercionFailed(String, String, String),
    #[error("No common superdomain for {0} and {1}")]
    NoCommonDomain(String, String),
    #[error("Invalid generator list for polynomial/rational domain: {0}")]
    InvalidGenerators(String),
    #[error("Invalid characteristic for finite field (must be prime > 1): {0}")]
    InvalidCharacteristic(u64),
}

/// Mathematical domain specification.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Domain {
    /// Exact integers $\mathbb{Z}$.
    ZZ,
    /// Exact rational numbers $\mathbb{Q}$.
    QQ,
    /// Real numbers $\mathbb{R}$ (exact or numeric).
    RR,
    /// Complex numbers $\mathbb{C}$.
    CC,
    /// Polynomial ring $D[x_1, \ldots, x_k]$.
    PolyRing {
        base: Box<Domain>,
        generators: Vec<Symbol>,
    },
    /// Rational function field $D(x_1, \ldots, x_k)$.
    FractionField {
        base: Box<Domain>,
        generators: Vec<Symbol>,
    },
    /// Galois field $\mathbb{F}_p$.
    FiniteField { characteristic: u64 },
    /// Generic expression domain $EX$.
    ExpressionDomain,
}

/// Classification of domain coercion paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoercionKind {
    /// Exact subring/subfield injection (e.g. $\mathbb{Z} \to \mathbb{Q}$).
    ExactSubdomain,
    /// Natural canonical embedding into polynomial ring (e.g. $D \to D[x]$).
    PolynomialEmbedding,
    /// Natural canonical embedding into fraction field (e.g. $D[x] \to D(x)$).
    FractionEmbedding,
    /// Base domain promotion in ring/field towers (e.g. $\mathbb{Z}[x] \to \mathbb{Q}[x]$).
    BaseCoercion,
    /// Promotion to universal expression domain.
    ExpressionPromotion,
}

/// Explicit audit receipt emitted by a domain coercion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoercionReceipt {
    pub from: Domain,
    pub to: Domain,
    pub kind: CoercionKind,
    pub is_exact: bool,
}

impl Domain {
    /// Canonical binary hashing for deterministic domain identity.
    pub fn hash_canonical(&self, hasher: &mut blake3::Hasher) {
        match self {
            Domain::ZZ => {
                hasher.update(&[0]);
            }
            Domain::QQ => {
                hasher.update(&[1]);
            }
            Domain::RR => {
                hasher.update(&[2]);
            }
            Domain::CC => {
                hasher.update(&[3]);
            }
            Domain::PolyRing { base, generators } => {
                hasher.update(&[4]);
                base.hash_canonical(hasher);
                hasher.update(&(generators.len() as u64).to_le_bytes());
                for g in generators {
                    hasher.update(&(g.name.len() as u64).to_le_bytes());
                    hasher.update(g.name.as_bytes());
                }
            }
            Domain::FractionField { base, generators } => {
                hasher.update(&[5]);
                base.hash_canonical(hasher);
                hasher.update(&(generators.len() as u64).to_le_bytes());
                for g in generators {
                    hasher.update(&(g.name.len() as u64).to_le_bytes());
                    hasher.update(g.name.as_bytes());
                }
            }
            Domain::FiniteField { characteristic } => {
                hasher.update(&[6]);
                hasher.update(&characteristic.to_le_bytes());
            }
            Domain::ExpressionDomain => {
                hasher.update(&[7]);
            }
        }
    }

    /// Create a polynomial ring domain $D[x]$.
    pub fn poly_ring(base: Domain, generators: Vec<Symbol>) -> Self {
        Domain::PolyRing {
            base: Box::new(base),
            generators,
        }
    }

    /// Validates and constructs a polynomial ring domain $D[x_1, \ldots, x_n]$.
    pub fn try_poly_ring(base: Domain, generators: Vec<Symbol>) -> Result<Self, DomainError> {
        if generators.is_empty() {
            return Err(DomainError::InvalidGenerators(
                "Generator list cannot be empty".into(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for g in &generators {
            if !seen.insert(g.clone()) {
                return Err(DomainError::InvalidGenerators(format!(
                    "Duplicate generator: {}",
                    g.name
                )));
            }
        }
        Ok(Domain::PolyRing {
            base: Box::new(base),
            generators,
        })
    }

    /// Create a rational function field domain $D(x)$.
    pub fn fraction_field(base: Domain, generators: Vec<Symbol>) -> Self {
        Domain::FractionField {
            base: Box::new(base),
            generators,
        }
    }

    /// Validates and constructs a rational function field domain $D(x_1, \ldots, x_n)$.
    pub fn try_fraction_field(base: Domain, generators: Vec<Symbol>) -> Result<Self, DomainError> {
        if generators.is_empty() {
            return Err(DomainError::InvalidGenerators(
                "Generator list cannot be empty".into(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for g in &generators {
            if !seen.insert(g.clone()) {
                return Err(DomainError::InvalidGenerators(format!(
                    "Duplicate generator: {}",
                    g.name
                )));
            }
        }
        Ok(Domain::FractionField {
            base: Box::new(base),
            generators,
        })
    }

    /// Validates and constructs a prime finite field $\mathbb{F}_p$.
    pub fn try_finite_field(characteristic: u64) -> Result<Self, DomainError> {
        if characteristic <= 1 || !is_prime_u64(characteristic) {
            return Err(DomainError::InvalidCharacteristic(characteristic));
        }
        Ok(Domain::FiniteField { characteristic })
    }

    /// Whether this domain maintains exact representations.
    pub fn is_exact(&self) -> bool {
        match self {
            Domain::ZZ | Domain::QQ | Domain::FiniteField { .. } => true,
            Domain::RR => false,
            Domain::CC => false,
            Domain::PolyRing { base, .. } | Domain::FractionField { base, .. } => base.is_exact(),
            Domain::ExpressionDomain => true,
        }
    }

    /// Whether this domain forms a field.
    pub fn is_field(&self) -> bool {
        matches!(
            self,
            Domain::QQ
                | Domain::RR
                | Domain::CC
                | Domain::FractionField { .. }
                | Domain::FiniteField { .. }
        )
    }

    /// Whether this domain forms an integral domain or ring.
    pub fn is_ring(&self) -> bool {
        true
    }

    /// Whether this domain has a total ordering compatible with arithmetic operations.
    pub fn is_ordered(&self) -> bool {
        matches!(self, Domain::ZZ | Domain::QQ | Domain::RR)
    }

    /// Characteristic of the underlying ring/field ($0$ for standard characteristic $0$).
    pub fn characteristic(&self) -> u64 {
        match self {
            Domain::FiniteField { characteristic } => *characteristic,
            Domain::PolyRing { base, .. } | Domain::FractionField { base, .. } => {
                base.characteristic()
            }
            _ => 0,
        }
    }

    /// Whether `self` can be coerced into `target`.
    pub fn can_coerce_to(&self, target: &Domain) -> bool {
        self.coercion_to(target).is_some()
    }

    /// Find the direct or multi-step coercion path from `self` to `target`.
    pub fn coercion_to(&self, target: &Domain) -> Option<CoercionReceipt> {
        if self == target {
            return Some(CoercionReceipt {
                from: self.clone(),
                to: target.clone(),
                kind: CoercionKind::ExactSubdomain,
                is_exact: self.is_exact(),
            });
        }

        if matches!(target, Domain::ExpressionDomain) {
            return Some(CoercionReceipt {
                from: self.clone(),
                to: target.clone(),
                kind: CoercionKind::ExpressionPromotion,
                is_exact: true,
            });
        }

        match (self, target) {
            (Domain::ZZ, Domain::QQ)
            | (Domain::ZZ, Domain::RR)
            | (Domain::ZZ, Domain::CC)
            | (Domain::QQ, Domain::RR)
            | (Domain::QQ, Domain::CC)
            | (Domain::RR, Domain::CC) => Some(CoercionReceipt {
                from: self.clone(),
                to: target.clone(),
                kind: CoercionKind::ExactSubdomain,
                is_exact: self.is_exact() && target.is_exact(),
            }),

            (
                Domain::PolyRing {
                    base: b1,
                    generators: g1,
                },
                Domain::PolyRing {
                    base: b2,
                    generators: g2,
                },
            ) => {
                if b1.can_coerce_to(b2) && g1.iter().all(|g| g2.contains(g)) {
                    Some(CoercionReceipt {
                        from: self.clone(),
                        to: target.clone(),
                        kind: if b1 == b2 {
                            CoercionKind::PolynomialEmbedding
                        } else {
                            CoercionKind::BaseCoercion
                        },
                        is_exact: self.is_exact() && target.is_exact(),
                    })
                } else {
                    None
                }
            }

            (
                d,
                Domain::PolyRing {
                    base,
                    generators: _,
                },
            ) => {
                if d == base.as_ref() || d.can_coerce_to(base) {
                    Some(CoercionReceipt {
                        from: self.clone(),
                        to: target.clone(),
                        kind: CoercionKind::PolynomialEmbedding,
                        is_exact: self.is_exact(),
                    })
                } else {
                    None
                }
            }

            (
                Domain::PolyRing { base, generators },
                Domain::FractionField {
                    base: f_base,
                    generators: f_gens,
                },
            ) => {
                if base == f_base && generators == f_gens {
                    Some(CoercionReceipt {
                        from: self.clone(),
                        to: target.clone(),
                        kind: CoercionKind::FractionEmbedding,
                        is_exact: true,
                    })
                } else if (base.can_coerce_to(f_base)
                    || (**base == Domain::QQ && **f_base == Domain::ZZ))
                    && generators.iter().all(|g| f_gens.contains(g))
                {
                    Some(CoercionReceipt {
                        from: self.clone(),
                        to: target.clone(),
                        kind: CoercionKind::BaseCoercion,
                        is_exact: self.is_exact(),
                    })
                } else {
                    None
                }
            }

            (
                Domain::FractionField {
                    base: b1,
                    generators: g1,
                },
                Domain::FractionField {
                    base: b2,
                    generators: g2,
                },
            ) => {
                if (b1.can_coerce_to(b2) || (**b1 == Domain::QQ && **b2 == Domain::ZZ))
                    && g1.iter().all(|g| g2.contains(g))
                {
                    Some(CoercionReceipt {
                        from: self.clone(),
                        to: target.clone(),
                        kind: CoercionKind::BaseCoercion,
                        is_exact: self.is_exact() && target.is_exact(),
                    })
                } else {
                    None
                }
            }

            (d, Domain::FractionField { base, .. }) => {
                if d == base.as_ref()
                    || d.can_coerce_to(base)
                    || (*d == Domain::QQ && **base == Domain::ZZ)
                {
                    Some(CoercionReceipt {
                        from: self.clone(),
                        to: target.clone(),
                        kind: CoercionKind::FractionEmbedding,
                        is_exact: self.is_exact(),
                    })
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Infers the natural minimal domain of an exact expression.
    pub fn of_expr(expr: &Expr) -> Domain {
        match expr {
            Expr::Integer(_) => Domain::ZZ,
            Expr::Rational(_) => Domain::QQ,
            Expr::Const(c) => match c {
                fsym_core::Constant::Pi | fsym_core::Constant::E => Domain::RR,
                fsym_core::Constant::I | fsym_core::Constant::ComplexInfinity => Domain::CC,
                _ => Domain::RR,
            },
            Expr::Sym(s) => Domain::PolyRing {
                base: Box::new(Domain::ZZ),
                generators: vec![s.clone()],
            },
            Expr::Add(terms) | Expr::Mul(terms) => {
                let mut current = Domain::ZZ;
                for t in terms {
                    current = common_domain(&current, &Domain::of_expr(t))
                        .unwrap_or(Domain::ExpressionDomain);
                }
                current
            }
            Expr::Pow(b, e) => {
                let b_dom = Domain::of_expr(b);
                let e_dom = Domain::of_expr(e);
                match (b_dom, e_dom) {
                    (Domain::ZZ, Domain::ZZ) => match e.as_ref() {
                        Expr::Integer(n) if !n.is_negative() => Domain::ZZ,
                        _ => Domain::QQ,
                    },
                    (Domain::QQ, Domain::ZZ) => Domain::QQ,
                    (Domain::RR, Domain::ZZ | Domain::QQ) => Domain::RR,
                    (Domain::CC, _) | (_, Domain::CC) => Domain::CC,
                    (Domain::PolyRing { base, generators }, Domain::ZZ) => match e.as_ref() {
                        Expr::Integer(n) if !n.is_negative() => {
                            Domain::PolyRing { base, generators }
                        }
                        _ => Domain::FractionField { base, generators },
                    },
                    _ => Domain::ExpressionDomain,
                }
            }
            _ => Domain::ExpressionDomain,
        }
    }
}

/// Computes the least common superdomain of two domains.
pub fn common_domain(a: &Domain, b: &Domain) -> Option<Domain> {
    if a == b {
        return Some(a.clone());
    }

    if a.can_coerce_to(b) {
        return Some(b.clone());
    }
    if b.can_coerce_to(a) {
        return Some(a.clone());
    }

    match (a, b) {
        (
            Domain::PolyRing {
                base: b1,
                generators: g1,
            },
            Domain::PolyRing {
                base: b2,
                generators: g2,
            },
        ) => {
            let common_base = common_domain(b1, b2)?;
            let mut merged_gens = g1.clone();
            for g in g2 {
                if !merged_gens.contains(g) {
                    merged_gens.push(g.clone());
                }
            }
            Some(Domain::PolyRing {
                base: Box::new(common_base),
                generators: merged_gens,
            })
        }
        (Domain::PolyRing { base, generators }, other)
        | (other, Domain::PolyRing { base, generators }) => {
            let common_base = common_domain(base, other)?;
            Some(Domain::PolyRing {
                base: Box::new(common_base),
                generators: generators.clone(),
            })
        }
        _ => Some(Domain::ExpressionDomain),
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Domain::ZZ => write!(f, "ZZ"),
            Domain::QQ => write!(f, "QQ"),
            Domain::RR => write!(f, "RR"),
            Domain::CC => write!(f, "CC"),
            Domain::PolyRing { base, generators } => {
                let gens = generators
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{}[{}]", base, gens)
            }
            Domain::FractionField { base, generators } => {
                let gens = generators
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{}({})", base, gens)
            }
            Domain::FiniteField { characteristic } => write!(f, "GF({})", characteristic),
            Domain::ExpressionDomain => write!(f, "EX"),
        }
    }
}

fn is_prime_u64(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut d = n - 1;
    let mut s = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    let bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &a in &bases {
        if n <= a {
            break;
        }
        let mut x = mod_pow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            x = ((x as u128 * x as u128) % n as u128) as u64;
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut res = 1u64;
    base %= modulus;
    while exp > 0 {
        if (exp & 1) != 0 {
            res = ((res as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp /= 2;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn domain_coercion_lattice() {
        assert!(Domain::ZZ.can_coerce_to(&Domain::QQ));
        assert!(Domain::QQ.can_coerce_to(&Domain::RR));
        assert!(Domain::RR.can_coerce_to(&Domain::CC));
        assert!(Domain::ZZ.can_coerce_to(&Domain::CC));

        let x = Symbol::new("x");
        let poly_zz = Domain::poly_ring(Domain::ZZ, vec![x.clone()]);
        let poly_qq = Domain::poly_ring(Domain::QQ, vec![x.clone()]);

        assert!(Domain::ZZ.can_coerce_to(&poly_zz));
        assert!(Domain::QQ.can_coerce_to(&poly_qq));
        assert!(poly_zz.can_coerce_to(&poly_qq));

        let frac_qq = Domain::fraction_field(Domain::QQ, vec![x.clone()]);
        assert!(poly_qq.can_coerce_to(&frac_qq));
    }

    #[test]
    fn common_superdomain_derivation() {
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let poly_x = Domain::poly_ring(Domain::ZZ, vec![x.clone()]);
        let poly_y = Domain::poly_ring(Domain::QQ, vec![y.clone()]);

        let common = common_domain(&poly_x, &poly_y).unwrap();
        assert_eq!(common, Domain::poly_ring(Domain::QQ, vec![x, y]));
    }

    #[test]
    fn finite_field_and_characteristic_properties() {
        let gf7 = Domain::try_finite_field(7).unwrap();
        assert!(gf7.is_exact());
        assert!(gf7.is_field());
        assert_eq!(gf7.characteristic(), 7);

        // Validation failures
        assert!(Domain::try_finite_field(0).is_err());
        assert!(Domain::try_finite_field(1).is_err());
        assert!(Domain::try_finite_field(4).is_err());
        assert!(Domain::try_finite_field(9).is_err());

        assert_eq!(Domain::ZZ.characteristic(), 0);
        assert_eq!(Domain::QQ.characteristic(), 0);
        assert!(Domain::ZZ.is_exact());
        assert!(!Domain::RR.is_exact());
    }

    #[test]
    fn domain_generator_validation() {
        let x = Symbol::new("x");
        // Empty generators error
        assert!(Domain::try_poly_ring(Domain::ZZ, vec![]).is_err());
        assert!(Domain::try_fraction_field(Domain::QQ, vec![]).is_err());

        // Duplicate generators error
        assert!(Domain::try_poly_ring(Domain::ZZ, vec![x.clone(), x.clone()]).is_err());
        assert!(Domain::try_fraction_field(Domain::QQ, vec![x.clone(), x.clone()]).is_err());

        // Valid
        assert!(Domain::try_poly_ring(Domain::ZZ, vec![x.clone()]).is_ok());
        assert!(Domain::try_fraction_field(Domain::QQ, vec![x]).is_ok());
    }

    #[test]
    fn domain_inference_of_exact_expressions() {
        let e1 = Expr::from_i64(42);
        assert_eq!(Domain::of_expr(&e1), Domain::ZZ);

        let e2 = Expr::rational(3, 5).unwrap();
        assert_eq!(Domain::of_expr(&e2), Domain::QQ);

        let e3 = Expr::Const(fsym_core::Constant::Pi);
        assert_eq!(Domain::of_expr(&e3), Domain::RR);

        let x = Symbol::new("x");
        let e4 = Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(1)]);
        assert_eq!(
            Domain::of_expr(&e4),
            Domain::PolyRing {
                base: Box::new(Domain::ZZ),
                generators: vec![x],
            }
        );

        // Power with negative exponent gives QQ, not ZZ
        let pow_neg = Expr::Pow(Arc::new(Expr::from_i64(2)), Arc::new(Expr::from_i64(-1)));
        assert_eq!(Domain::of_expr(&pow_neg), Domain::QQ);
    }
}
