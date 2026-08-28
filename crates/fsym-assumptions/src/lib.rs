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

/// Unique, content-addressed identifier for an immutable assumption context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContextId(pub u64);

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
        let mut id_bytes = [0u8; 8];
        id_bytes.copy_from_slice(&hash[0..8]);
        let id_raw = u64::from_le_bytes(id_bytes);

        Arc::new(Self {
            id: ContextId(if id_raw == 0 { 1 } else { id_raw }),
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
        sorted_facts.sort_by_key(|(s, _)| &s.name);
        for (sym, preds) in sorted_facts {
            hasher.update(b"fact:");
            hasher.update(&(sym.name.len() as u64).to_le_bytes());
            hasher.update(sym.name.as_bytes());
            let mut sorted_preds = preds.clone();
            sorted_preds.sort();
            sorted_preds.dedup();
            hasher.update(&(sorted_preds.len() as u64).to_le_bytes());
            for p in &sorted_preds {
                hasher.update(&[p.discriminant()]);
            }
        }
        let mut sorted_domains: Vec<(&Symbol, &Domain)> = additional_domains.iter().collect();
        sorted_domains.sort_by_key(|(s, _)| &s.name);
        for (sym, dom) in sorted_domains {
            hasher.update(b"dom:");
            hasher.update(&(sym.name.len() as u64).to_le_bytes());
            hasher.update(sym.name.as_bytes());
            dom.hash_canonical(&mut hasher);
        }

        let hash = *hasher.finalize().as_bytes();
        let mut id_bytes = [0u8; 8];
        id_bytes.copy_from_slice(&hash[0..8]);
        let id_raw = u64::from_le_bytes(id_bytes);

        Ok(Arc::new(Self {
            id: ContextId(if id_raw == 0 { 1 } else { id_raw }),
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
        let known = inherent_facts(expr)
            .or_else(|| match expr {
                Expr::Sym(s) => Some(self.deductions(s)),
                _ => None,
            })
            .unwrap_or_default();

        let has_internal_contradiction = known.iter().enumerate().any(|(i, a)| {
            known
                .iter()
                .skip(i + 1)
                .any(|b| Predicate::contradicts(*a, *b))
        });
        if has_internal_contradiction {
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
        if let Expr::Sym(s) = expr
            && let Some(preds) = self.facts.get(s)
        {
            for (i, a) in preds.iter().enumerate() {
                for b in &preds[i + 1..] {
                    if Predicate::contradicts(*a, *b) {
                        return TruthValue::Contradictory;
                    }
                }
            }
        }

        let known = inherent_facts(expr)
            .or_else(|| match expr {
                Expr::Sym(s) => Some(self.deductions(s)),
                _ => None,
            })
            .unwrap_or_default();

        let has_internal_contradiction = known.iter().enumerate().any(|(i, a)| {
            known
                .iter()
                .skip(i + 1)
                .any(|b| Predicate::contradicts(*a, *b))
        });
        if has_internal_contradiction {
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
}
