//! # fsym-assumptions
//!
//! Deductive predicate and assumptions engine for symbolic reasoning (WS04):
//! - Multi-valued truth model ([`TruthValue`]: EntailedTrue, EntailedFalse, Unknown, Contradictory);
//! - Mathematical domains ([`Domain`]: $\mathbb{Z}$, $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$, $D[x]$, $D(x)$, $\mathbb{F}_p$) with explicit coercion graph;
//! - Capture-avoiding substitution and alpha-equivalence ([`bindings`]);
//! - Deductive predicate hierarchy ([`Predicate`]) and assumption context ([`AssumptionsContext`]).

#![forbid(unsafe_code)]

pub mod bindings;
pub mod domain;
pub mod predicate;
pub mod truth;

pub use bindings::*;
pub use domain::*;
pub use predicate::*;
pub use truth::*;

use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AssumptionError {
    #[error("Contradictory assumptions inferred")]
    Contradiction,
    #[error("Domain conflict for symbol {0}: {1} vs {2}")]
    DomainConflict(String, String, String),
    #[error("Unknown symbol: {0}")]
    UnknownSymbol(String),
}

pub use fsym_id::ContextId;

/// An immutable, hierarchical, thread-safe assumption context with cryptographic digest and provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmutableAssumptionsSnapshot {
    id: ContextId,
    digest: [u8; 32],
    parent: Option<Arc<ImmutableAssumptionsSnapshot>>,
    facts: HashMap<Symbol, Vec<Predicate>>,
    domains: HashMap<Symbol, Domain>,
    provenance: String,
}

impl ImmutableAssumptionsSnapshot {
    /// Creates the empty root immutable context.
    pub fn empty() -> Arc<Self> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.context.empty.v1");
        let hash = *hasher.finalize().as_bytes();
        let id_bytes: [u8; 8] = hash[..8].try_into().unwrap_or([1; 8]);
        let id_raw = u64::from_le_bytes(id_bytes);

        Arc::new(Self {
            id: ContextId::new(if id_raw == 0 { 1 } else { id_raw })
                .expect("non-zero context id payload"),
            digest: hash,
            parent: None,
            facts: HashMap::new(),
            domains: HashMap::new(),
            provenance: "root".to_string(),
        })
    }

    /// Derives an immutable child context with additional facts and domain assignments.
    pub fn derive_child(
        self: &Arc<Self>,
        additional_facts: HashMap<Symbol, Vec<Predicate>>,
        additional_domains: HashMap<Symbol, Domain>,
        provenance: impl Into<String>,
    ) -> Result<Arc<Self>, AssumptionError> {
        // Validate domain consistency with parent
        for (sym, dom) in &additional_domains {
            if let Some(parent_dom) = self.domain_of(sym)
                && parent_dom != dom
                && !dom.can_coerce_to(parent_dom)
                && !parent_dom.can_coerce_to(dom)
            {
                return Err(AssumptionError::DomainConflict(
                    sym.name.clone(),
                    parent_dom.to_string(),
                    dom.to_string(),
                ));
            }
        }

        let prov = provenance.into();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.context.v2:");
        hasher.update(&self.digest);
        let mut sorted_facts: Vec<(&Symbol, &Vec<Predicate>)> = additional_facts.iter().collect();
        sorted_facts.sort_by_key(|(s, _)| *s);
        for (sym, preds) in sorted_facts {
            hasher.update(b"fact:");
            hasher.update(&(sym.name.len() as u64).to_le_bytes());
            hasher.update(sym.name.as_bytes());
            if let Some(id) = sym.identity {
                hasher.update(&[1u8]);
                hasher.update(&id.assumptions);
            } else {
                hasher.update(&[0u8]);
            }
            let mut sorted_preds = preds.clone();
            sorted_preds.sort();
            sorted_preds.dedup();
            hasher.update(&(sorted_preds.len() as u64).to_le_bytes());
            for p in &sorted_preds {
                hasher.update(&[p.discriminant()]);
            }
        }
        let mut sorted_domains: Vec<(&Symbol, &Domain)> = additional_domains.iter().collect();
        sorted_domains.sort_by_key(|(s, _)| *s);
        for (sym, dom) in sorted_domains {
            hasher.update(b"dom:");
            hasher.update(&(sym.name.len() as u64).to_le_bytes());
            hasher.update(sym.name.as_bytes());
            if let Some(id) = sym.identity {
                hasher.update(&[1u8]);
                hasher.update(&id.assumptions);
            } else {
                hasher.update(&[0u8]);
            }
            dom.hash_canonical(&mut hasher);
        }

        let hash = *hasher.finalize().as_bytes();
        let id_bytes: [u8; 8] = hash[..8].try_into().unwrap_or([1; 8]);
        let id_raw = u64::from_le_bytes(id_bytes);

        Ok(Arc::new(Self {
            id: ContextId::new(if id_raw == 0 { 1 } else { id_raw })
                .expect("non-zero context id payload"),
            digest: hash,
            parent: Some(Arc::clone(self)),
            facts: additional_facts,
            domains: additional_domains,
            provenance: prov,
        }))
    }

    pub fn id(&self) -> ContextId {
        self.id
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    /// Evaluates a predicate query against the immutable snapshot in 4-valued logic.
    pub fn query(&self, expr: &Expr, pred: Predicate) -> TruthValue {
        for sym in expr.free_symbols() {
            let sym_facts = self.deductions(&sym);
            if has_contradiction_in_set(&sym_facts) {
                return TruthValue::Contradictory;
            }
        }

        let known = deduce_expr_facts(expr, &|s| self.deductions(s));

        if has_contradiction_in_set(&known) {
            return TruthValue::Contradictory;
        }

        let has_pred = known.contains(&pred);
        let has_contradiction = known.iter().any(|fact| Predicate::contradicts(*fact, pred));

        if has_pred && has_contradiction {
            TruthValue::Contradictory
        } else if has_pred {
            TruthValue::EntailedTrue
        } else if has_contradiction {
            TruthValue::EntailedFalse
        } else {
            TruthValue::Unknown
        }
    }

    /// Check if predicate holds for an expression under the immutable snapshot.
    pub fn is_true(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        self.query(expr, pred).to_option_bool()
    }

    /// Retrieves the domain assigned to a symbol under this snapshot or any parent.
    pub fn domain_of(&self, sym: &Symbol) -> Option<&Domain> {
        if let Some(dom) = self.domains.get(sym) {
            Some(dom)
        } else if let Some(parent) = &self.parent {
            parent.domain_of(sym)
        } else {
            None
        }
    }

    /// Deduced predicate set for one symbol across this snapshot and parent hierarchy.
    pub fn deductions(&self, sym: &Symbol) -> BTreeSet<Predicate> {
        let mut out = BTreeSet::new();
        if let Some(parent) = &self.parent {
            out.extend(parent.deductions(sym));
        }

        if let Some(preds) = self.facts.get(sym) {
            for p in preds {
                out.extend(p.closure());
            }
        }
        if let Some(dom) = self.domains.get(sym) {
            match dom {
                Domain::ZZ => out.extend(Predicate::Integer.closure()),
                Domain::QQ => out.extend(Predicate::Rational.closure()),
                Domain::RR => out.extend(Predicate::Real.closure()),
                Domain::CC => out.extend(Predicate::Complex.closure()),
                _ => {}
            }
        }
        out
    }
}

fn has_contradiction_in_set(known: &BTreeSet<Predicate>) -> bool {
    known.iter().enumerate().any(|(i, a)| {
        known
            .iter()
            .skip(i + 1)
            .any(|b| Predicate::contradicts(*a, *b))
    })
}

/// Compositionally deduces mathematical predicates for compound expressions
/// from symbol assumptions and literal inherent facts.
pub fn deduce_expr_facts<F>(expr: &Expr, sym_deductions: &F) -> BTreeSet<Predicate>
where
    F: Fn(&Symbol) -> BTreeSet<Predicate>,
{
    if let Some(facts) = inherent_facts(expr) {
        return facts;
    }
    match expr {
        Expr::Sym(s) => sym_deductions(s),
        Expr::Add(terms) if !terms.is_empty() => {
            let child_sets: Vec<BTreeSet<Predicate>> = terms
                .iter()
                .map(|t| deduce_expr_facts(t, sym_deductions))
                .collect();
            let mut out = BTreeSet::new();
            if child_sets.iter().any(has_contradiction_in_set) {
                out.insert(Predicate::Zero);
                out.insert(Predicate::NonZero);
                return out;
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Zero)) {
                out.extend(Predicate::Zero.closure());
                return out;
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Integer)) {
                out.extend(Predicate::Integer.closure());
                if child_sets
                    .iter()
                    .all(|s| s.contains(&Predicate::Even) || s.contains(&Predicate::Odd))
                {
                    let odd_count = child_sets
                        .iter()
                        .filter(|s| s.contains(&Predicate::Odd))
                        .count();
                    if odd_count.is_multiple_of(2) {
                        out.extend(Predicate::Even.closure());
                    } else {
                        out.extend(Predicate::Odd.closure());
                    }
                }
            } else if child_sets.iter().all(|s| s.contains(&Predicate::Rational)) {
                out.extend(Predicate::Rational.closure());
            } else if child_sets.iter().all(|s| s.contains(&Predicate::Algebraic)) {
                out.extend(Predicate::Algebraic.closure());
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Real)) {
                out.extend(Predicate::Real.closure());
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Complex)) {
                out.extend(Predicate::Complex.closure());
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Finite)) {
                out.extend(Predicate::Finite.closure());
            }
            if child_sets
                .iter()
                .all(|s| s.contains(&Predicate::NonNegative))
            {
                if child_sets.iter().any(|s| s.contains(&Predicate::Positive)) {
                    out.extend(Predicate::Positive.closure());
                } else {
                    out.extend(Predicate::NonNegative.closure());
                }
            } else if child_sets
                .iter()
                .all(|s| s.contains(&Predicate::NonPositive))
            {
                if child_sets.iter().any(|s| s.contains(&Predicate::Negative)) {
                    out.extend(Predicate::Negative.closure());
                } else {
                    out.extend(Predicate::NonPositive.closure());
                }
            }
            out
        }
        Expr::Mul(factors) if !factors.is_empty() => {
            let child_sets: Vec<BTreeSet<Predicate>> = factors
                .iter()
                .map(|f| deduce_expr_facts(f, sym_deductions))
                .collect();
            let mut out = BTreeSet::new();
            if child_sets.iter().any(has_contradiction_in_set) {
                out.insert(Predicate::Zero);
                out.insert(Predicate::NonZero);
                return out;
            }
            if child_sets.iter().any(|s| s.contains(&Predicate::Zero))
                && child_sets.iter().all(|s| s.contains(&Predicate::Finite))
            {
                out.extend(Predicate::Zero.closure());
                return out;
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Integer)) {
                out.extend(Predicate::Integer.closure());
                if child_sets.iter().all(|s| s.contains(&Predicate::Odd)) {
                    out.extend(Predicate::Odd.closure());
                } else if child_sets.iter().any(|s| s.contains(&Predicate::Even)) {
                    out.extend(Predicate::Even.closure());
                }
            } else if child_sets.iter().all(|s| s.contains(&Predicate::Rational)) {
                out.extend(Predicate::Rational.closure());
            } else if child_sets.iter().all(|s| s.contains(&Predicate::Algebraic)) {
                out.extend(Predicate::Algebraic.closure());
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Real)) {
                out.extend(Predicate::Real.closure());
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Complex)) {
                out.extend(Predicate::Complex.closure());
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::Finite)) {
                out.extend(Predicate::Finite.closure());
            }
            if child_sets.iter().all(|s| s.contains(&Predicate::NonZero)) {
                out.extend(Predicate::NonZero.closure());
            }
            if child_sets
                .iter()
                .all(|s| s.contains(&Predicate::Positive) || s.contains(&Predicate::Negative))
            {
                let neg_count = child_sets
                    .iter()
                    .filter(|s| s.contains(&Predicate::Negative))
                    .count();
                if neg_count.is_multiple_of(2) {
                    out.extend(Predicate::Positive.closure());
                } else {
                    out.extend(Predicate::Negative.closure());
                }
            } else if child_sets
                .iter()
                .all(|s| s.contains(&Predicate::NonNegative) || s.contains(&Predicate::NonPositive))
            {
                let nonpos_only = child_sets
                    .iter()
                    .filter(|s| {
                        s.contains(&Predicate::NonPositive) && !s.contains(&Predicate::NonNegative)
                    })
                    .count();
                let all_definite_sign = child_sets.iter().all(|s| {
                    s.contains(&Predicate::NonNegative) ^ s.contains(&Predicate::NonPositive)
                        || s.contains(&Predicate::Zero)
                });
                if all_definite_sign {
                    if nonpos_only.is_multiple_of(2) {
                        out.extend(Predicate::NonNegative.closure());
                    } else {
                        out.extend(Predicate::NonPositive.closure());
                    }
                }
            }
            out
        }
        Expr::Pow(base, exp) => {
            let b_facts = deduce_expr_facts(base, sym_deductions);
            let mut out = BTreeSet::new();
            if has_contradiction_in_set(&b_facts) {
                out.insert(Predicate::Zero);
                out.insert(Predicate::NonZero);
                return out;
            }
            if let Expr::Integer(n) = exp.as_ref() {
                if n.is_zero() {
                    if b_facts.contains(&Predicate::NonZero) {
                        out.extend(Predicate::Positive.closure());
                        out.extend(Predicate::Integer.closure());
                        out.extend(Predicate::Odd.closure());
                    }
                    return out;
                }
                let is_even_exp = (n % &fsym_core::BigInt::from(2i64)).is_zero();
                let positive_exp = n.is_positive();
                let can_invert = positive_exp || b_facts.contains(&Predicate::NonZero);
                if !can_invert {
                    return out;
                }
                if positive_exp && b_facts.contains(&Predicate::Integer) {
                    out.extend(Predicate::Integer.closure());
                    if b_facts.contains(&Predicate::Even) {
                        out.extend(Predicate::Even.closure());
                    } else if b_facts.contains(&Predicate::Odd) {
                        out.extend(Predicate::Odd.closure());
                    }
                } else if b_facts.contains(&Predicate::Rational) {
                    out.extend(Predicate::Rational.closure());
                } else if b_facts.contains(&Predicate::Algebraic) {
                    out.extend(Predicate::Algebraic.closure());
                }
                if b_facts.contains(&Predicate::Real) {
                    out.extend(Predicate::Real.closure());
                }
                if b_facts.contains(&Predicate::Complex) {
                    out.extend(Predicate::Complex.closure());
                }
                if b_facts.contains(&Predicate::Finite) {
                    out.extend(Predicate::Finite.closure());
                }
                if b_facts.contains(&Predicate::NonZero) {
                    out.extend(Predicate::NonZero.closure());
                }
                if b_facts.contains(&Predicate::Positive) {
                    out.extend(Predicate::Positive.closure());
                } else if is_even_exp {
                    if b_facts.contains(&Predicate::Real) && b_facts.contains(&Predicate::NonZero) {
                        out.extend(Predicate::Positive.closure());
                    } else if b_facts.contains(&Predicate::Real) {
                        out.extend(Predicate::NonNegative.closure());
                    }
                } else if b_facts.contains(&Predicate::Negative) {
                    out.extend(Predicate::Negative.closure());
                } else if b_facts.contains(&Predicate::NonNegative) {
                    out.extend(Predicate::NonNegative.closure());
                } else if b_facts.contains(&Predicate::NonPositive) {
                    out.extend(Predicate::NonPositive.closure());
                }
            } else if let Expr::Rational(r) = exp.as_ref() {
                if r.numer().is_positive() || b_facts.contains(&Predicate::Positive) {
                    if b_facts.contains(&Predicate::Positive) {
                        out.extend(Predicate::Positive.closure());
                        if b_facts.contains(&Predicate::Finite) {
                            out.extend(Predicate::Finite.closure());
                        }
                        if b_facts.contains(&Predicate::Algebraic) {
                            out.extend(Predicate::Algebraic.closure());
                        }
                    } else if r.numer().is_positive() && b_facts.contains(&Predicate::NonNegative) {
                        out.extend(Predicate::NonNegative.closure());
                        if b_facts.contains(&Predicate::Finite) {
                            out.extend(Predicate::Finite.closure());
                        }
                    }
                }
            } else {
                let e_facts = deduce_expr_facts(exp, sym_deductions);
                if b_facts.contains(&Predicate::Positive) && e_facts.contains(&Predicate::Real) {
                    out.extend(Predicate::Positive.closure());
                    if b_facts.contains(&Predicate::Finite) && e_facts.contains(&Predicate::Finite)
                    {
                        out.extend(Predicate::Finite.closure());
                    }
                }
            }
            out
        }
        Expr::Function(name, args) if args.len() == 1 => {
            let u_facts = deduce_expr_facts(&args[0], sym_deductions);
            let mut out = BTreeSet::new();
            if has_contradiction_in_set(&u_facts) {
                out.insert(Predicate::Zero);
                out.insert(Predicate::NonZero);
                return out;
            }
            match name.as_str() {
                "exp" => {
                    if u_facts.contains(&Predicate::Real) {
                        out.extend(Predicate::Positive.closure());
                    } else if u_facts.contains(&Predicate::Complex) {
                        out.extend(Predicate::Complex.closure());
                        out.extend(Predicate::NonZero.closure());
                    }
                    if u_facts.contains(&Predicate::Finite) {
                        out.extend(Predicate::Finite.closure());
                    }
                }
                "abs" | "Abs" => {
                    if u_facts.contains(&Predicate::Complex) {
                        if u_facts.contains(&Predicate::NonZero) {
                            out.extend(Predicate::Positive.closure());
                        } else {
                            out.extend(Predicate::NonNegative.closure());
                        }
                        if u_facts.contains(&Predicate::Integer) {
                            out.extend(Predicate::Integer.closure());
                        } else if u_facts.contains(&Predicate::Rational) {
                            out.extend(Predicate::Rational.closure());
                        } else if u_facts.contains(&Predicate::Algebraic) {
                            out.extend(Predicate::Algebraic.closure());
                        }
                        if u_facts.contains(&Predicate::Finite) {
                            out.extend(Predicate::Finite.closure());
                        }
                    }
                }
                "cosh" => {
                    if u_facts.contains(&Predicate::Real) {
                        out.extend(Predicate::Positive.closure());
                        if u_facts.contains(&Predicate::Finite) {
                            out.extend(Predicate::Finite.closure());
                        }
                    }
                }
                "sinh" | "tanh" | "atan" | "asinh" | "erf" => {
                    if u_facts.contains(&Predicate::Positive) {
                        out.extend(Predicate::Positive.closure());
                    } else if u_facts.contains(&Predicate::Negative) {
                        out.extend(Predicate::Negative.closure());
                    } else if u_facts.contains(&Predicate::NonNegative) {
                        out.extend(Predicate::NonNegative.closure());
                    } else if u_facts.contains(&Predicate::NonPositive) {
                        out.extend(Predicate::NonPositive.closure());
                    } else if u_facts.contains(&Predicate::Real) {
                        out.extend(Predicate::Real.closure());
                    }
                    if u_facts.contains(&Predicate::Finite) {
                        out.extend(Predicate::Finite.closure());
                    }
                }
                "sin" | "cos" | "sinc" | "erfc" => {
                    if u_facts.contains(&Predicate::Real) {
                        out.extend(Predicate::Real.closure());
                        if u_facts.contains(&Predicate::Finite) {
                            out.extend(Predicate::Finite.closure());
                        }
                    }
                }
                "log" | "ln" => {
                    if u_facts.contains(&Predicate::Positive) {
                        out.extend(Predicate::Real.closure());
                        if u_facts.contains(&Predicate::Finite) {
                            out.extend(Predicate::Finite.closure());
                        }
                    }
                }
                "sqrt" => {
                    if u_facts.contains(&Predicate::Positive) {
                        out.extend(Predicate::Positive.closure());
                    } else if u_facts.contains(&Predicate::NonNegative) {
                        out.extend(Predicate::NonNegative.closure());
                    }
                    if u_facts.contains(&Predicate::Finite) {
                        out.extend(Predicate::Finite.closure());
                    }
                }
                _ => {}
            }
            out
        }
        _ => BTreeSet::new(),
    }
}

/// Assumptions context builder holding mathematical facts and domain assignments for symbols.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssumptionsContext {
    facts: HashMap<Symbol, Vec<Predicate>>,
    domains: HashMap<Symbol, Domain>,
}

impl AssumptionsContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an immutable, thread-safe snapshot of this context.
    pub fn snapshot(&self) -> ImmutableAssumptionsSnapshot {
        let empty = ImmutableAssumptionsSnapshot::empty();
        let child = empty
            .derive_child(self.facts.clone(), self.domains.clone(), "snapshot")
            .expect("valid snapshot derivation");
        (*child).clone()
    }

    /// Records a predicate assumption for a symbol.
    ///
    /// Returns [`AssumptionError::Contradiction`] when the new predicate
    /// contradicts a fact already deduced for `sym`. Unknown remains unknown;
    /// this does not force a two-valued guess.
    pub fn assume(&mut self, sym: Symbol, pred: Predicate) -> Result<(), AssumptionError> {
        if self
            .deductions(&sym)
            .iter()
            .any(|fact| Predicate::contradicts(*fact, pred))
        {
            return Err(AssumptionError::Contradiction);
        }
        self.facts.entry(sym).or_default().push(pred);
        Ok(())
    }

    /// Records an exact domain assignment for a symbol, emitting [`AssumptionError::DomainConflict`]
    /// if an incompatible domain is already registered.
    pub fn assume_domain(&mut self, sym: Symbol, domain: Domain) -> Result<(), AssumptionError> {
        if let Some(existing) = self.domains.get(&sym) {
            if existing != &domain {
                if let Some(common) = common_domain(existing, &domain) {
                    if common == Domain::ExpressionDomain && *existing != Domain::ExpressionDomain {
                        return Err(AssumptionError::DomainConflict(
                            sym.name,
                            existing.to_string(),
                            domain.to_string(),
                        ));
                    }
                    self.domains.insert(sym, common);
                    return Ok(());
                } else {
                    return Err(AssumptionError::DomainConflict(
                        sym.name,
                        existing.to_string(),
                        domain.to_string(),
                    ));
                }
            }
        } else {
            self.domains.insert(sym, domain);
        }
        Ok(())
    }

    /// Retrieves the domain assigned to a symbol, if any.
    pub fn domain_of(&self, sym: &Symbol) -> Option<&Domain> {
        self.domains.get(sym)
    }

    /// Deduced predicate set for one symbol: stated facts plus every consequence
    /// the lattice licenses, as well as facts implied by domain assignments.
    pub fn deductions(&self, sym: &Symbol) -> BTreeSet<Predicate> {
        let mut out = BTreeSet::new();
        if let Some(preds) = self.facts.get(sym) {
            for p in preds {
                out.extend(p.closure());
            }
        }
        if let Some(dom) = self.domains.get(sym) {
            match dom {
                Domain::ZZ => out.extend(Predicate::Integer.closure()),
                Domain::QQ => out.extend(Predicate::Rational.closure()),
                Domain::RR => out.extend(Predicate::Real.closure()),
                Domain::CC => out.extend(Predicate::Complex.closure()),
                _ => {}
            }
        }
        out
    }

    /// Evaluates a predicate query against an expression in 4-valued logic.
    pub fn query(&self, expr: &Expr, pred: Predicate) -> TruthValue {
        for sym in expr.free_symbols() {
            if let Some(preds) = self.facts.get(&sym) {
                for (i, a) in preds.iter().enumerate() {
                    for b in &preds[i + 1..] {
                        if Predicate::contradicts(*a, *b) {
                            return TruthValue::Contradictory;
                        }
                    }
                }
            }
            let sym_facts = self.deductions(&sym);
            if has_contradiction_in_set(&sym_facts) {
                return TruthValue::Contradictory;
            }
        }

        let known = deduce_expr_facts(expr, &|s| self.deductions(s));

        if has_contradiction_in_set(&known) {
            return TruthValue::Contradictory;
        }

        let has_pred = known.contains(&pred);
        let has_contradiction = known.iter().any(|fact| Predicate::contradicts(*fact, pred));

        if has_pred && has_contradiction {
            TruthValue::Contradictory
        } else if has_pred {
            TruthValue::EntailedTrue
        } else if has_contradiction {
            TruthValue::EntailedFalse
        } else {
            TruthValue::Unknown
        }
    }

    /// Check if predicate holds for an expression (backward-compatible 3-valued query).
    pub fn is_true(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        self.query(expr, pred).to_option_bool()
    }

    /// Check if predicate is refutable for an expression.
    pub fn is_false(&self, expr: &Expr, pred: Predicate) -> Option<bool> {
        match self.query(expr, pred) {
            TruthValue::EntailedFalse => Some(true),
            TruthValue::EntailedTrue => Some(false),
            TruthValue::Unknown | TruthValue::Contradictory => None,
        }
    }

    /// Reports [`AssumptionError::Contradiction`] if any symbol carries
    /// mutually exclusive facts or conflicting domain assertions.
    pub fn check_consistency(&self) -> Result<(), AssumptionError> {
        for preds in self.facts.values() {
            for (i, a) in preds.iter().enumerate() {
                for b in &preds[i + 1..] {
                    if Predicate::contradicts(*a, *b) {
                        return Err(AssumptionError::Contradiction);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assumptions_context() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume(x.clone(), Predicate::Positive).unwrap();
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Positive),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::NonZero),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Negative),
            Some(false)
        );
        assert_eq!(ctx.is_true(&Expr::Sym(x.clone()), Predicate::Integer), None);
        assert_eq!(
            ctx.assume(x.clone(), Predicate::Negative),
            Err(AssumptionError::Contradiction)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(x.clone()), Predicate::Positive),
            Some(true)
        );
        assert_eq!(ctx.assume(x.clone(), Predicate::Positive), Ok(()));
    }

    #[test]
    fn test_domain_assignments_imply_predicates() {
        let mut ctx = AssumptionsContext::new();
        let n = Symbol::new("n");
        ctx.assume_domain(n.clone(), Domain::ZZ).unwrap();
        assert_eq!(
            ctx.is_true(&Expr::Sym(n.clone()), Predicate::Integer),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(n.clone()), Predicate::Rational),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(n.clone()), Predicate::Real),
            Some(true)
        );
        assert_eq!(
            ctx.is_true(&Expr::Sym(n.clone()), Predicate::Complex),
            Some(true)
        );
    }

    #[test]
    fn test_domain_conflict_detection() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume_domain(x.clone(), Domain::ZZ).unwrap();
        assert!(ctx.assume_domain(x.clone(), Domain::QQ).is_ok());
        assert_eq!(ctx.domain_of(&x), Some(&Domain::QQ));
    }

    #[test]
    fn immutable_snapshot_preserves_query_behavior() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        ctx.assume(x.clone(), Predicate::Positive).unwrap();
        let snap = ctx.snapshot();

        assert_eq!(
            snap.is_true(&Expr::Sym(x.clone()), Predicate::Positive),
            Some(true)
        );
        assert_eq!(
            snap.is_true(&Expr::Sym(x.clone()), Predicate::NonZero),
            Some(true)
        );
    }

    #[test]
    fn assumptions_refinement_soundness_hierarchical() {
        let root = ImmutableAssumptionsSnapshot::empty();
        let x = Symbol::new("x");
        let mut facts_parent = HashMap::new();
        facts_parent.insert(x.clone(), vec![Predicate::Positive]);

        let parent = root
            .derive_child(facts_parent, HashMap::new(), "parent_fact")
            .unwrap();

        let mut facts_child = HashMap::new();
        facts_child.insert(x.clone(), vec![Predicate::Integer]);
        let child = parent
            .derive_child(facts_child, HashMap::new(), "child_refinement")
            .unwrap();

        // Child entails parent facts AND child facts
        assert_eq!(
            child.query(&Expr::Sym(x.clone()), Predicate::Positive),
            TruthValue::EntailedTrue
        );
        assert_eq!(
            child.query(&Expr::Sym(x.clone()), Predicate::Integer),
            TruthValue::EntailedTrue
        );
        assert_eq!(
            child.query(&Expr::Sym(x.clone()), Predicate::NonZero),
            TruthValue::EntailedTrue
        );
        assert_eq!(
            child.query(&Expr::Sym(x.clone()), Predicate::Real),
            TruthValue::EntailedTrue
        );
    }

    #[test]
    fn monotonicity_under_arbitrary_refinement_property() {
        let x = Symbol::new("x");
        let predicates = [
            Predicate::Positive,
            Predicate::Negative,
            Predicate::NonZero,
            Predicate::Integer,
            Predicate::Rational,
            Predicate::Real,
            Predicate::Complex,
        ];

        let mut base_ctx = AssumptionsContext::new();
        base_ctx.assume(x.clone(), Predicate::Positive).unwrap();
        let base_snap = Arc::new(base_ctx.snapshot());

        // Query all predicates on base
        for &pred in &predicates {
            let base_ans = base_snap.query(&Expr::Sym(x.clone()), pred);
            if base_ans == TruthValue::EntailedTrue {
                // If entailed true in base, must remain entailed true or become contradictory in any valid refinement
                let mut refined_facts = HashMap::new();
                refined_facts.insert(x.clone(), vec![Predicate::Integer]);
                let child = base_snap
                    .derive_child(refined_facts, HashMap::new(), "int_refinement")
                    .unwrap();
                let child_ans = child.query(&Expr::Sym(x.clone()), pred);
                assert!(
                    child_ans == TruthValue::EntailedTrue || child_ans == TruthValue::Contradictory,
                    "Monotonicity violated for predicate {:?}",
                    pred
                );
            }
        }
    }

    #[test]
    fn test_context_digest_invariant_to_provenance_and_fact_order() {
        let root = ImmutableAssumptionsSnapshot::empty();
        let x = Symbol::new("x");
        let y = Symbol::new("y");

        let mut facts1 = HashMap::new();
        facts1.insert(
            x.clone(),
            vec![Predicate::Positive, Predicate::Integer, Predicate::NonZero],
        );
        facts1.insert(y.clone(), vec![Predicate::Real]);

        let mut facts2 = HashMap::new();
        facts2.insert(y.clone(), vec![Predicate::Real]);
        facts2.insert(
            x.clone(),
            vec![Predicate::NonZero, Predicate::Integer, Predicate::Positive],
        );

        let mut domains1 = HashMap::new();
        domains1.insert(x.clone(), Domain::ZZ);
        domains1.insert(y.clone(), Domain::RR);

        let mut domains2 = HashMap::new();
        domains2.insert(y.clone(), Domain::RR);
        domains2.insert(x.clone(), Domain::ZZ);

        let child1 = root
            .derive_child(facts1, domains1, "derived_from_worker_1")
            .unwrap();
        let child2 = root
            .derive_child(
                facts2,
                domains2,
                "derived_from_worker_2_different_provenance",
            )
            .unwrap();

        assert_eq!(child1.digest(), child2.digest());
        assert_eq!(child1.id(), child2.id());
    }

    #[test]
    fn test_compositional_deduction_over_add_mul_pow_and_functions() {
        let mut ctx = AssumptionsContext::new();
        let x = Symbol::new("x");
        let y = Symbol::new("y");
        let n = Symbol::new("n");
        let m = Symbol::new("m");
        ctx.assume(x.clone(), Predicate::Positive).unwrap();
        ctx.assume(y.clone(), Predicate::Negative).unwrap();
        ctx.assume(n.clone(), Predicate::Even).unwrap();
        ctx.assume(m.clone(), Predicate::Odd).unwrap();
        let snap = ctx.snapshot();

        // x + 3 is Positive (Positive + Positive)
        let x_plus_3 = Expr::Add(vec![Expr::Sym(x.clone()), Expr::from_i64(3)]);
        assert_eq!(
            snap.query(&x_plus_3, Predicate::Positive),
            TruthValue::EntailedTrue
        );

        // x * y is Negative (Positive * Negative)
        let x_mul_y = Expr::Mul(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]);
        assert_eq!(
            snap.query(&x_mul_y, Predicate::Negative),
            TruthValue::EntailedTrue
        );
        assert_eq!(
            snap.query(&x_mul_y, Predicate::Positive),
            TruthValue::EntailedFalse
        );

        // y^2 is Positive (Negative^even)
        let y_sq = Expr::pow(Expr::Sym(y.clone()), Expr::from_i64(2));
        assert_eq!(
            snap.query(&y_sq, Predicate::Positive),
            TruthValue::EntailedTrue
        );

        // y^2 + 1 is Positive even if y is only Real
        let mut real_ctx = AssumptionsContext::new();
        let r = Symbol::new("r");
        real_ctx.assume(r.clone(), Predicate::Real).unwrap();
        let r_sq_plus_1 = Expr::Add(vec![
            Expr::pow(Expr::Sym(r.clone()), Expr::from_i64(2)),
            Expr::from_i64(1),
        ]);
        assert_eq!(
            real_ctx.query(&r_sq_plus_1, Predicate::Positive),
            TruthValue::EntailedTrue
        );

        // exp(r) and cosh(r) are Positive for r: Real
        let exp_r = Expr::Function("exp".to_string(), vec![Expr::Sym(r.clone())]);
        let cosh_r = Expr::Function("cosh".to_string(), vec![Expr::Sym(r)]);
        assert_eq!(
            real_ctx.query(&exp_r, Predicate::Positive),
            TruthValue::EntailedTrue
        );
        assert_eq!(
            real_ctx.query(&cosh_r, Predicate::Positive),
            TruthValue::EntailedTrue
        );

        // Parity: n (Even) + m (Odd) is Odd; n * m is Even; m * m is Odd
        let n_plus_m = Expr::Add(vec![Expr::Sym(n.clone()), Expr::Sym(m.clone())]);
        let n_mul_m = Expr::Mul(vec![Expr::Sym(n), Expr::Sym(m.clone())]);
        let m_sq = Expr::pow(Expr::Sym(m), Expr::from_i64(2));
        assert_eq!(
            snap.query(&n_plus_m, Predicate::Odd),
            TruthValue::EntailedTrue
        );
        assert_eq!(
            snap.query(&n_mul_m, Predicate::Even),
            TruthValue::EntailedTrue
        );
        assert_eq!(snap.query(&m_sq, Predicate::Odd), TruthValue::EntailedTrue);
    }

    #[test]
    fn test_context_digest_distinguishes_distinct_symbol_identities_with_same_name() {
        use fsym_core::SymbolIdentity;

        let root = ImmutableAssumptionsSnapshot::empty();
        let x1 = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1u8; 32]));
        let x2 = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2u8; 32]));

        let mut facts1 = HashMap::new();
        facts1.insert(x1, vec![Predicate::Positive]);
        let mut facts2 = HashMap::new();
        facts2.insert(x2, vec![Predicate::Positive]);

        let c1 = root.derive_child(facts1, HashMap::new(), "c1").unwrap();
        let c2 = root.derive_child(facts2, HashMap::new(), "c2").unwrap();
        assert_ne!(
            c1.digest(),
            c2.digest(),
            "symbols with same display name but distinct SymbolIdentity must have distinct context digests"
        );
    }
}
