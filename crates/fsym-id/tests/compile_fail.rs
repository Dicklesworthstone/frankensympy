//! Compile-fail proof for WS02: identifiers of different kinds are distinct
//! types and cross-kind substitution must not compile.
//!
//! These UI tests complement the runtime parse-rejection tests in `lib.rs`:
//! even bypassing textual parsing entirely (constructing both ids directly),
//! the type system refuses to unify one kind with another.

#[test]
fn cross_kind_substitution_is_rejected_at_compile_time() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/*.rs");
}
