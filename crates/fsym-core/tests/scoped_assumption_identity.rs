//! Scoped algorithm facts must constrain the exact atom named by the caller.

use fsym_core::assume::{
    Facts, facts_of_symbol, is_integer, is_nonzero, is_real, sign, with_facts, with_symbol_facts,
};
use fsym_core::elementary::{eval_function, eval_pow};
use fsym_core::{BigRational, Constant, Expr, Symbol, SymbolIdentity};
use std::collections::HashMap;
use std::sync::Arc;

fn tagged(name: &str, tag: u8) -> Symbol {
    Symbol::with_identity(name, SymbolIdentity::from_assumptions_digest([tag; 32]))
}

fn half() -> Expr {
    Expr::Rational(BigRational::new(1.into(), 2.into()))
}

fn square_root_of_square(symbol: &Symbol) -> Expr {
    eval_pow(
        Expr::Pow(
            Arc::new(Expr::Sym(symbol.clone())),
            Arc::new(Expr::from_i64(2)),
        ),
        half(),
    )
}

#[test]
fn plain_facts_do_not_constrain_distinct_same_name_atoms() {
    let plain = Symbol::new("x");
    let other = tagged("x", 1);
    with_symbol_facts(
        HashMap::from([(plain.clone(), Facts::from_names(&["positive", "integer"]))]),
        || {
            assert_eq!(sign(&Expr::Sym(plain)), Some(1));
            assert_eq!(facts_of_symbol(&other), None);
            let other = Expr::Sym(other);
            assert_eq!(sign(&other), None);
            assert!(!is_real(&other));
            assert!(!is_integer(&other));
            assert!(!is_nonzero(&other));
        },
    );
}

#[test]
fn legacy_name_scopes_remain_limited_to_plain_atoms() {
    let plain = Symbol::new("x");
    let other = tagged("x", 2);
    with_facts(
        HashMap::from([("x".into(), Facts::from_names(&["negative"]))]),
        || {
            assert_eq!(sign(&Expr::Sym(plain)), Some(-1));
            assert_eq!(sign(&Expr::Sym(other)), None);
        },
    );
}

#[test]
fn nested_plain_scope_cannot_shadow_an_outer_typed_atom() {
    let atom = tagged("x", 3);
    with_symbol_facts(
        HashMap::from([(atom.clone(), Facts::from_names(&["negative"]))]),
        || {
            with_facts(
                HashMap::from([("x".into(), Facts::from_names(&["positive"]))]),
                || assert_eq!(sign(&Expr::Sym(atom.clone())), Some(-1)),
            );
            assert_eq!(sign(&Expr::Sym(atom.clone())), Some(-1));
        },
    );
    assert_eq!(facts_of_symbol(&atom), None);
}

#[test]
fn exact_typed_atoms_keep_independent_scopes_and_unknown_shadowing() {
    let positive = tagged("x", 4);
    let negative = tagged("x", 5);
    let unknown = tagged("x", 6);
    with_symbol_facts(
        HashMap::from([
            (positive.clone(), Facts::from_names(&["positive"])),
            (negative.clone(), Facts::from_names(&["negative"])),
        ]),
        || {
            assert_eq!(sign(&Expr::Sym(positive.clone())), Some(1));
            assert_eq!(sign(&Expr::Sym(negative)), Some(-1));
            assert_eq!(sign(&Expr::Sym(unknown)), None);
            with_symbol_facts(
                HashMap::from([(positive.clone(), Facts::default())]),
                || {
                    assert_eq!(sign(&Expr::Sym(positive.clone())), None);
                    assert!(!is_real(&Expr::Sym(positive.clone())));
                },
            );
            assert_eq!(sign(&Expr::Sym(positive)), Some(1));
        },
    );
}

#[test]
fn unrelated_positive_atom_does_not_erase_a_principal_square_root_branch() {
    let other = tagged("x", 7);
    let rewritten = with_facts(
        HashMap::from([("x".into(), Facts::from_names(&["positive"]))]),
        || square_root_of_square(&other),
    );
    // At this allowed value, sqrt((-1)^2) is +1, whereas the forbidden
    // sqrt(x^2) -> x rewrite gives -1.
    assert_eq!(
        rewritten.subs(&HashMap::from([(other, Expr::from_i64(-1))])),
        Expr::from_i64(1)
    );
}

#[test]
fn exact_positive_atom_still_discharges_the_power_branch() {
    let atom = tagged("x", 8);
    with_symbol_facts(
        HashMap::from([(atom.clone(), Facts::from_names(&["positive"]))]),
        || assert_eq!(square_root_of_square(&atom), Expr::Sym(atom.clone())),
    );
    assert!(matches!(square_root_of_square(&atom), Expr::Pow(_, _)));
}

#[test]
fn unrelated_positive_atom_does_not_erase_a_zero_base_singularity() {
    let other = tagged("x", 9);
    let rewritten = with_facts(
        HashMap::from([("x".into(), Facts::from_names(&["positive"]))]),
        || eval_pow(Expr::from_i64(0), Expr::Sym(other.clone())),
    );
    assert!(matches!(rewritten, Expr::Pow(_, _)));
    assert_eq!(
        rewritten.subs(&HashMap::from([(other, Expr::from_i64(-1))])),
        Expr::Const(Constant::ComplexInfinity)
    );
}

#[test]
fn unrelated_integer_atom_does_not_authorize_periodic_reduction() {
    let other = tagged("n", 10);
    let argument = Expr::Mul(vec![Expr::Const(Constant::Pi), Expr::Sym(other)]);
    with_facts(
        HashMap::from([("n".into(), Facts::from_names(&["integer"]))]),
        || assert_eq!(eval_function("sin", &[argument]), None),
    );
}

#[test]
fn inconsistent_facts_cannot_authorize_branch_or_integer_rewrites() {
    let atom = tagged("x", 11);
    let contradictions: &[&[&str]] = &[
        &["positive", "negative", "integer"],
        &["positive", "nonpositive", "integer"],
        &["negative", "nonnegative", "integer"],
        &["nonzero", "nonnegative", "nonpositive", "integer"],
    ];
    for names in contradictions {
        let facts = Facts::from_names(names);
        assert!(facts.is_contradictory(), "{names:?}");
        with_symbol_facts(HashMap::from([(atom.clone(), facts)]), || {
            // Retain the source record for diagnostics; it must not be
            // replaced by an apparently empty, merely unknown context.
            assert_eq!(facts_of_symbol(&atom), Some(facts));
            let expression = Expr::Sym(atom.clone());
            assert_eq!(sign(&expression), None, "{names:?}");
            assert!(!is_real(&expression), "{names:?}");
            assert!(!is_integer(&expression), "{names:?}");
            assert!(!is_nonzero(&expression), "{names:?}");
            assert!(matches!(square_root_of_square(&atom), Expr::Pow(_, _)));
            assert!(matches!(
                eval_pow(Expr::from_i64(0), expression),
                Expr::Pow(_, _)
            ));
        });
    }
}

#[test]
fn contradiction_diagnostics_distinguish_zero_unknown_and_direct_records() {
    for names in [
        &[][..],
        &["real", "integer"][..],
        &["positive", "nonzero"][..],
        &["negative", "nonpositive"][..],
        &["nonnegative", "nonpositive"][..],
    ] {
        assert!(!Facts::from_names(names).is_contradictory(), "{names:?}");
    }
    let direct = Facts {
        positive: true,
        nonpositive: true,
        ..Facts::default()
    };
    assert!(direct.is_contradictory());
    let atom = tagged("x", 13);
    with_symbol_facts(HashMap::from([(atom.clone(), direct)]), || {
        assert!(facts_of_symbol(&atom).unwrap().is_contradictory());
        assert_eq!(sign(&Expr::Sym(atom)), None);
    });
}

#[test]
fn contradictory_inner_record_shadows_outer_facts_without_erasing_them() {
    let atom = tagged("x", 14);
    with_symbol_facts(
        HashMap::from([(atom.clone(), Facts::from_names(&["positive"]))]),
        || {
            with_symbol_facts(
                HashMap::from([(atom.clone(), Facts::from_names(&["positive", "negative"]))]),
                || {
                    assert!(facts_of_symbol(&atom).unwrap().is_contradictory());
                    assert_eq!(sign(&Expr::Sym(atom.clone())), None);
                },
            );
            assert_eq!(sign(&Expr::Sym(atom)), Some(1));
        },
    );
}

#[test]
fn unwinding_clears_only_the_failed_scope() {
    let atom = tagged("x", 12);
    with_symbol_facts(
        HashMap::from([(atom.clone(), Facts::from_names(&["positive"]))]),
        || {
            let result = std::panic::catch_unwind(|| {
                with_symbol_facts(
                    HashMap::from([(atom.clone(), Facts::from_names(&["negative"]))]),
                    || panic!("injected callback failure"),
                );
            });
            assert!(result.is_err());
            assert_eq!(sign(&Expr::Sym(atom.clone())), Some(1));
        },
    );
    assert_eq!(facts_of_symbol(&atom), None);
}
