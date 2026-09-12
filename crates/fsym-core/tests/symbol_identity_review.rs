//! Reviewer-owned adversarial tests for the typed symbol-identity contract.
//!
//! Gate `fra-rc-lowering-gate-ba5` reviewing implementation `fra-rc-lowering-8w3`
//! at commit `7820929`. This file is written by the gate reviewer, not by the
//! implementation author, and deliberately attacks the contract from outside:
//! the native `Symbol`/`Expr`/DAG layer is exercised through its public API only.
//!
//! EXPECTED FAILURES on commit 7820929 (they encode the gate findings, not
//! reviewer mistakes):
//!
//! * `add_argument_comparator_must_not_declare_unequal_expressions_equal`
//! * `canonical_add_order_must_be_a_function_of_the_term_multiset`
//!
//! Both follow from `cmp_add_args`'s `_ =>` arm, which still compares rendered
//! text. Now that two symbols can render identically while being distinct atoms,
//! two *unequal* compound terms can render identically and therefore compare
//! `Equal`, which makes the "canonical" Add order insertion-order dependent.

#![forbid(unsafe_code)]

use fsym_core::dag::{TermDag, TermNode, compute_term_digest};
use fsym_core::{Expr, Symbol, SymbolIdentity, canonicalize_add_args, cmp_add_args};
use std::cmp::Ordering;

/// Frozen literal from the published cross-architecture golden fixture
/// `artifacts/conformance/fixtures/deterministic_term_identity_v1.json`
/// (`terms.sym_x.digest_hex`), which pins `TermNode::Sym(Symbol::new("x"))`
/// in the `Expression` domain.
const GOLDEN_SYM_X_DIGEST: &str = "fe01bab39fe4f7057029e497c41995eb0e903e75dd0de8a144d2bbd198e07b53";

fn identity(tag: u8) -> SymbolIdentity {
    SymbolIdentity::from_assumptions_digest([tag; 32])
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn symbol_hex(symbol: &Symbol) -> String {
    symbol
        .identity
        .map(|identity| hex(&identity.assumptions))
        .unwrap_or_else(|| "plain".to_string())
}

#[test]
fn same_printed_name_with_distinct_identity_never_merges_and_ord_agrees_with_eq() {
    let plain_x = Symbol::new("x");
    let keyed_x = Symbol::with_identity("x", identity(1));
    let other_x = Symbol::with_identity("x", identity(2));
    let keyed_x_again = Symbol::with_identity("x", identity(1));
    let plain_y = Symbol::new("y");
    let keyed_y = Symbol::with_identity("y", identity(1));

    let all = [
        plain_x.clone(),
        keyed_x.clone(),
        other_x.clone(),
        keyed_x_again.clone(),
        plain_y.clone(),
        keyed_y.clone(),
    ];

    // The printed view is identical for the three x atoms...
    for symbol in [&plain_x, &keyed_x, &other_x] {
        assert_eq!(symbol.to_string(), "x", "printed view must stay plain");
    }
    // ...but no pair of distinct declarations is one atom. A total order that
    // agrees with equality must report `Equal` exactly for equal pairs.
    for left in &all {
        for right in &all {
            assert_eq!(
                left == right,
                left.cmp(right) == Ordering::Equal,
                "Eq and Ord disagree for {} vs {}",
                symbol_hex(left),
                symbol_hex(right)
            );
        }
    }
    assert_ne!(keyed_x, plain_x);
    assert_ne!(keyed_x, other_x);
    assert_ne!(keyed_x, keyed_y);
    assert_eq!(keyed_x, keyed_x_again);

    // A total order must be antisymmetric and must agree with Eq on every pair.
    for left in &all {
        for right in &all {
            assert_eq!(
                left.cmp(right),
                right.cmp(left).reverse(),
                "Ord is not antisymmetric for {} vs {}",
                symbol_hex(left),
                symbol_hex(right)
            );
        }
    }

    // Sorting must not collapse distinct-identity same-name atoms; exactly the
    // five distinct declarations survive.
    let mut sorted = all.clone();
    sorted.sort();
    let mut unique = sorted.clone();
    unique.dedup();
    assert_eq!(
        unique.len(),
        5,
        "sort/dedup merged distinct declarations: {unique:?}"
    );
    for window in sorted.windows(2) {
        assert!(
            window[0].cmp(&window[1]) != Ordering::Greater,
            "sorted output is out of order"
        );
    }

    // The DAG must not intern two distinct declarations to one node, and the
    // identity-bearing atom must never reuse the plain atom's digest.
    let mut dag = TermDag::new();
    let plain_id = dag.insert_node(TermNode::Sym(plain_x.clone())).unwrap();
    let keyed_id = dag.insert_node(TermNode::Sym(keyed_x.clone())).unwrap();
    let other_id = dag.insert_node(TermNode::Sym(other_x.clone())).unwrap();
    assert_ne!(plain_id, keyed_id);
    assert_ne!(keyed_id, other_id);
    assert_ne!(plain_id, other_id);
    let plain_digest = dag.term_digest(plain_id).unwrap();
    assert_ne!(plain_digest, dag.term_digest(keyed_id).unwrap());
    assert_ne!(
        dag.term_digest(keyed_id).unwrap(),
        dag.term_digest(other_id).unwrap()
    );

    // Lifting keeps the typed atom exactly (name and identity payload).
    let lifted = dag.to_expr(keyed_id).unwrap();
    assert_eq!(lifted, Expr::Sym(keyed_x));
    assert_eq!(lifted.free_symbols(), vec![keyed_x]);
}

#[test]
fn plain_symbol_preimage_replays_the_published_cross_architecture_golden() {
    let plain = Symbol::new("x");
    assert!(plain.is_plain());
    assert!(plain.identity.is_none());

    // Independent replay of the frozen golden digest, without the probe binary:
    // a plain symbol's content identity must not have churned.
    let digest = compute_term_digest(&TermNode::Sym(plain.clone())).unwrap();
    assert_eq!(
        hex(&digest),
        GOLDEN_SYM_X_DIGEST,
        "plain symbol preimage churned relative to the published golden"
    );

    let mut dag = TermDag::new();
    let id = dag.insert_node(TermNode::Sym(plain)).unwrap();
    assert_eq!(hex(&dag.term_digest(id).unwrap()), GOLDEN_SYM_X_DIGEST);

    // The identity-bearing atom is a distinct node kind, so it can never
    // collide with the pinned plain preimage.
    let keyed = compute_term_digest(&TermNode::Sym(Symbol::with_identity("x", identity(1)))).unwrap();
    assert_ne!(hex(&keyed), GOLDEN_SYM_X_DIGEST);
}

#[test]
fn identity_is_preserved_by_dag_lifting_and_legacy_serde_payloads() {
    // Persisted artifacts written before this change carry only `name`. They
    // must still deserialize to the plain atom with the pinned golden digest,
    // otherwise every stored artifact silently changes identity.
    let legacy: Symbol = serde_json::from_str(r#"{"name":"x"}"#).unwrap();
    assert!(legacy.identity.is_none());
    assert_eq!(
        hex(&compute_term_digest(&TermNode::Sym(legacy)).unwrap()),
        GOLDEN_SYM_X_DIGEST
    );

    // A plain symbol must not start emitting an identity field...
    let encoded: serde_json::Value = serde_json::from_str(&serde_json::to_string(&Symbol::new("x")).unwrap()).unwrap();
    assert!(
        encoded.get("identity").is_none(),
        "plain symbol grew a persisted identity field: {encoded}"
    );

    // ...while an identity-bearing symbol round-trips its identity exactly.
    let keyed = Symbol::with_identity("x", identity(9));
    let round_tripped: Symbol =
        serde_json::from_str(&serde_json::to_string(&keyed).unwrap()).unwrap();
    assert_eq!(round_tripped, keyed);
    assert_eq!(round_tripped.identity, Some(identity(9)));

    // Lifting through the DAG must return the same typed atom, not a look-alike.
    let mut dag = TermDag::new();
    let id = dag.insert_expr(&Expr::Sym(keyed.clone())).unwrap();
    let lifted = dag.to_expr(id).unwrap();
    assert_eq!(lifted, Expr::Sym(keyed.clone()));
    assert_ne!(lifted, Expr::symbol("x"));
}

#[test]
fn add_argument_comparator_must_not_declare_unequal_expressions_equal() {
    // Two compound terms that differ only inside a symbol's typed identity
    // print identically but are NOT equal expressions.
    let function_plain = Expr::Function(
        "f".to_string(),
        vec![Expr::Sym(Symbol::new("x"))],
    );
    let function_keyed = Expr::Function(
        "f".to_string(),
        vec![Expr::Sym(Symbol::with_identity("x", identity(1)))],
    );
    assert_ne!(function_plain, function_keyed);
    assert_eq!(function_plain.to_string(), function_keyed.to_string());

    let power_plain = Expr::Sym(Symbol::new("x")).pow(Expr::from_i64(2));
    let power_keyed = Expr::Sym(Symbol::with_identity("x", identity(1))).pow(Expr::from_i64(2));
    assert_ne!(power_plain, power_keyed);
    assert_eq!(power_plain.to_string(), power_keyed.to_string());

    // A comparator used to canonicalise Add arguments must not report two
    // unequal expressions as equal, otherwise the "canonical" order is not a
    // function of the term multiset.
    assert_ne!(
        cmp_add_args(&function_plain, &function_keyed),
        Ordering::Equal,
        "cmp_add_args declared unequal expressions equal (both render as \"f(x)\")"
    );
    assert_ne!(
        cmp_add_args(&power_plain, &power_keyed),
        Ordering::Equal,
        "cmp_add_args declared unequal expressions equal (both render as \"x**2\")"
    );
}

#[test]
fn canonical_add_order_must_be_a_function_of_the_term_multiset() {
    let function_plain = Expr::Function("f".to_string(), vec![Expr::Sym(Symbol::new("x"))]);
    let function_keyed =
        Expr::Function("f".to_string(), vec![Expr::Sym(Symbol::with_identity("x", identity(1)))]);

    let mut forward = vec![function_plain.clone(), function_keyed.clone()];
    let mut reverse = vec![function_keyed.clone(), function_plain.clone()];
    canonicalize_add_args(&mut forward);
    canonicalize_add_args(&mut reverse);

    let mut dag_forward = TermDag::new();
    let forward_id = dag_forward.insert_expr(&Expr::Add(forward)).unwrap();
    let mut dag_reverse = TermDag::new();
    let reverse_id = dag_reverse.insert_expr(&Expr::Add(reverse)).unwrap();

    assert_eq!(
        hex(&dag_forward.term_digest(forward_id).unwrap()),
        hex(&dag_reverse.term_digest(reverse_id).unwrap()),
        "canonicalising one term multiset from two insertion orders produced two identities"
    );
}
