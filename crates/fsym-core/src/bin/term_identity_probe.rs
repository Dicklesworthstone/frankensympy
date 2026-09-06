//! Term identity probe binary (WS04 / SEMANTIC-001).
//! Constructs a canonical suite of semantic terms in a TermDag and outputs their
//! TermId and BLAKE3 digests as deterministic JSON.

#![forbid(unsafe_code)]

use fsym_bigint::BigInt;
use fsym_core::dag::{TermDag, TermNode};
use fsym_core::{Constant, Symbol, TermDomain};
use fsym_id::TermId;
use fsym_rational::BigRational;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TermIdentityRecord {
    pub term_id_raw: u64,
    pub digest_hex: String,
    pub domain: String,
}

pub fn collect_canonical_term_identities() -> BTreeMap<String, TermIdentityRecord> {
    let mut dag = TermDag::new();
    let mut records = BTreeMap::new();

    // Atoms
    let sym_x_id = dag.insert_node(TermNode::Sym(Symbol::new("x"))).unwrap();
    let sym_y_id = dag.insert_node(TermNode::Sym(Symbol::new("y"))).unwrap();
    let sym_z_id = dag.insert_node(TermNode::Sym(Symbol::new("z"))).unwrap();

    let int_0_id = dag.insert_node(TermNode::Integer(BigInt::from(0))).unwrap();
    let int_1_id = dag.insert_node(TermNode::Integer(BigInt::from(1))).unwrap();
    let int_neg_1_id = dag
        .insert_node(TermNode::Integer(BigInt::from(-1)))
        .unwrap();
    let int_2_id = dag.insert_node(TermNode::Integer(BigInt::from(2))).unwrap();
    let int_3_id = dag.insert_node(TermNode::Integer(BigInt::from(3))).unwrap();
    let int_42_id = dag
        .insert_node(TermNode::Integer(BigInt::from(42)))
        .unwrap();
    let int_big_id = dag
        .insert_node(TermNode::Integer(
            BigInt::from_str("1234567890123456789012345678901234567890").unwrap(),
        ))
        .unwrap();

    let rat_half_id = dag
        .insert_node(TermNode::Rational(BigRational::new(
            BigInt::from(1),
            BigInt::from(2),
        )))
        .unwrap();
    let rat_neg_22_7_id = dag
        .insert_node(TermNode::Rational(BigRational::new(
            BigInt::from(-22),
            BigInt::from(7),
        )))
        .unwrap();

    let const_pi_id = dag.insert_node(TermNode::Const(Constant::Pi)).unwrap();
    let const_e_id = dag.insert_node(TermNode::Const(Constant::E)).unwrap();
    let const_i_id = dag.insert_node(TermNode::Const(Constant::I)).unwrap();
    let const_inf_id = dag
        .insert_node(TermNode::Const(Constant::Infinity))
        .unwrap();

    // Composites
    let add_x_y_id = dag
        .insert_node(TermNode::Add(vec![sym_x_id, sym_y_id]))
        .unwrap();
    let add_x_1_id = dag
        .insert_node(TermNode::Add(vec![sym_x_id, int_1_id]))
        .unwrap();
    let add_y_1_id = dag
        .insert_node(TermNode::Add(vec![sym_y_id, int_1_id]))
        .unwrap();
    let mul_2_x_id = dag
        .insert_node(TermNode::Mul(vec![int_2_id, sym_x_id]))
        .unwrap();
    let mul_x_y_id = dag
        .insert_node(TermNode::Mul(vec![sym_x_id, sym_y_id]))
        .unwrap();
    let pow_x_2_id = dag.insert_node(TermNode::Pow(sym_x_id, int_2_id)).unwrap();
    let pow_x_plus_1_cubed_id = dag
        .insert_node(TermNode::Pow(add_x_1_id, int_3_id))
        .unwrap();

    let fn_sin_x_id = dag
        .insert_node(TermNode::Function("sin".into(), vec![sym_x_id]))
        .unwrap();
    let fn_cos_x_y_id = dag
        .insert_node(TermNode::Function("cos".into(), vec![add_x_y_id]))
        .unwrap();

    // De Bruijn Lambda binders
    let lambda_x_id = dag
        .insert_lambda(vec![Symbol::new("x")], add_x_1_id)
        .unwrap();
    let lambda_y_id = dag
        .insert_lambda(vec![Symbol::new("y")], add_y_1_id)
        .unwrap();

    let named_nodes: Vec<(&str, TermId)> = vec![
        ("sym_x", sym_x_id),
        ("sym_y", sym_y_id),
        ("sym_z", sym_z_id),
        ("int_0", int_0_id),
        ("int_1", int_1_id),
        ("int_neg_1", int_neg_1_id),
        ("int_2", int_2_id),
        ("int_3", int_3_id),
        ("int_42", int_42_id),
        ("int_big", int_big_id),
        ("rat_half", rat_half_id),
        ("rat_neg_22_7", rat_neg_22_7_id),
        ("const_pi", const_pi_id),
        ("const_e", const_e_id),
        ("const_i", const_i_id),
        ("const_inf", const_inf_id),
        ("add_x_y", add_x_y_id),
        ("add_x_1", add_x_1_id),
        ("add_y_1", add_y_1_id),
        ("mul_2_x", mul_2_x_id),
        ("mul_x_y", mul_x_y_id),
        ("pow_x_2", pow_x_2_id),
        ("pow_x_plus_1_cubed", pow_x_plus_1_cubed_id),
        ("fn_sin_x", fn_sin_x_id),
        ("fn_cos_x_y", fn_cos_x_y_id),
        ("lambda_x_plus_1", lambda_x_id),
        ("lambda_y_plus_1", lambda_y_id),
    ];

    for (name, id) in named_nodes {
        let digest = dag.term_digest(id).expect("interned term must have digest");
        let domain = dag.term_domain(id).unwrap_or(TermDomain::Expression);
        records.insert(
            name.to_string(),
            TermIdentityRecord {
                term_id_raw: id.raw(),
                digest_hex: hex_string(&digest),
                domain: format!("{domain:?}"),
            },
        );
    }

    records
}

fn hex_string(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).unwrap();
    }
    s
}

fn main() {
    let records = collect_canonical_term_identities();
    let json = serde_json::to_string_pretty(&records).expect("serialize term identity records");
    println!("{json}");
}
