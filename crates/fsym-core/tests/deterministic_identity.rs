//! Cross-architecture and deterministic term identity tests (WS04 / SEMANTIC-001).

#![forbid(unsafe_code)]

use fsym_bigint::BigInt;
use fsym_core::dag::{TermDag, TermNode, compute_term_digest};
use fsym_core::{Expr, Symbol, SymbolIdentity};

#[test]
fn test_core_identity_invariants() {
    let mut dag = TermDag::new();
    let x = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
    let zero = dag.insert_node(TermNode::Integer(BigInt::from(0))).unwrap();

    let digest_x = dag.term_digest(x).unwrap();
    let digest_0 = dag.term_digest(zero).unwrap();

    assert_ne!(digest_x, digest_0);
    assert_eq!(
        digest_x,
        compute_term_digest(&TermNode::Sym(Symbol::new("x"))).unwrap()
    );
    assert_eq!(
        digest_0,
        compute_term_digest(&TermNode::Integer(BigInt::from(0))).unwrap()
    );
}

fn identity(tag: u8) -> SymbolIdentity {
    SymbolIdentity::from_assumptions_digest([tag; 32])
}

#[test]
fn printed_name_alone_does_not_establish_atom_identity() {
    let mut dag = TermDag::new();
    let plain = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
    let positive = dag
        .insert_node(TermNode::Sym(Symbol::with_identity("x", identity(1))))
        .unwrap();
    let positive_again = dag
        .insert_node(TermNode::Sym(Symbol::with_identity("x", identity(1))))
        .unwrap();
    let nonzero = dag
        .insert_node(TermNode::Sym(Symbol::with_identity("x", identity(2))))
        .unwrap();

    // Identical identity is one atom; every distinct identity is another.
    assert_eq!(positive, positive_again);
    assert_ne!(positive, nonzero);
    assert_ne!(positive, plain);
    assert_ne!(
        dag.term_digest(positive).unwrap(),
        dag.term_digest(nonzero).unwrap()
    );

    // The printed view is unchanged: identity is not printed, so a symbol's
    // display text must never be used as its semantic preimage.
    assert_eq!(Symbol::with_identity("x", identity(1)).to_string(), "x");
    assert_eq!(Symbol::new("x").to_string(), "x");
}

#[test]
fn plain_symbol_preimages_keep_their_published_digests() {
    // Identity support must not churn the content identity of plain symbols:
    // the cross-architecture golden fixture pins those digests, so a plain
    // symbol with the same name must hash exactly as it did before.
    let plain = compute_term_digest(&TermNode::Sym(Symbol::new("x"))).unwrap();
    let mut dag = TermDag::new();
    let interned = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
    assert_eq!(plain, dag.term_digest(interned).unwrap());

    let keyed =
        compute_term_digest(&TermNode::Sym(Symbol::with_identity("x", identity(1)))).unwrap();
    assert_ne!(plain, keyed);
}

#[test]
fn symbol_identity_survives_dag_lowering_and_lifting() {
    let keyed = Expr::Sym(Symbol::with_identity("x", identity(7)));
    let mut dag = TermDag::new();
    let id = dag.insert_expr(&keyed).unwrap();
    let lifted = dag.to_expr(id).unwrap();
    assert_eq!(lifted, keyed);
    assert_eq!(
        lifted.free_symbols(),
        vec![Symbol::with_identity("x", identity(7))]
    );

    // A same-name plain symbol lowers to a different node and never merges
    // with the keyed atom.
    let plain_id = dag.insert_expr(&Expr::symbol("x")).unwrap();
    assert_ne!(id, plain_id);
}
