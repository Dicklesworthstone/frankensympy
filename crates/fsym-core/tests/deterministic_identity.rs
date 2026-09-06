//! Cross-architecture and deterministic term identity tests (WS04 / SEMANTIC-001).

#![forbid(unsafe_code)]

use fsym_bigint::BigInt;
use fsym_core::Symbol;
use fsym_core::dag::{TermDag, TermNode, compute_term_digest};

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
