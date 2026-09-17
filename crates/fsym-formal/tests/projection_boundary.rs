#[path = "../examples/support/mod.rs"]
mod support;
use fsym_core::{BigInt, BigRational};
use fsym_formal::{
    MAX_CAPSULE_BYTES, MAX_JSON_BYTES, ProjectionEnvelope, ProjectionError as E, SafePoint,
    checker::check_projection, project_capsule,
};
use fsym_id::{ContextId, RuleId, VerifierId};
use fsym_proof_kernel::capsule::{Capsule, CapsuleObject, PolyDomain, PolyObject};

fn project(c: &Capsule) -> Result<fsym_formal::CheckedProjection, E> {
    project_capsule(&c.encode().unwrap(), c.claim.digest().unwrap(), &|_| false)
}
fn envelope() -> (ProjectionEnvelope, [u8; 32]) {
    let c = support::capsule();
    (
        project(&c).unwrap().into_envelope(),
        c.claim.digest().unwrap(),
    )
}
fn replace_subject(c: &mut Capsule, object: CapsuleObject) {
    c.objects.remove(&c.claim.subject);
    c.claim.subject = object.id;
    c.objects.insert(object.id, object);
}
fn altered(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => serde_json::Value::String(format!("{s}x")),
        serde_json::Value::Number(_) => serde_json::json!(0),
        serde_json::Value::Array(_) => serde_json::json!([]),
        _ => serde_json::Value::Null,
    }
}
fn refused_json(value: &serde_json::Value, root: [u8; 32]) -> bool {
    ProjectionEnvelope::from_json(&serde_json::to_vec(value).unwrap())
        .and_then(|e| check_projection(&e, root, &|_| false))
        .is_err()
}

#[test]
fn native_first_roundtrip_checks_offline() {
    let (e, root) = envelope();
    let bytes = serde_json::to_vec(&e).unwrap();
    let parsed = ProjectionEnvelope::from_json(&bytes).unwrap();
    assert_eq!(
        check_projection(&parsed, root, &|_| false)
            .unwrap()
            .envelope(),
        &e
    );
}

#[test]
fn every_transport_and_receipt_field_is_required_and_bound() {
    let (e, root) = envelope();
    let value = serde_json::to_value(e).unwrap();
    for (key, current) in value.as_object().unwrap() {
        let mut omitted = value.clone();
        omitted.as_object_mut().unwrap().remove(key);
        assert!(refused_json(&omitted, root), "omitted envelope {key}");
        let mut mutated = value.clone();
        mutated[key] = altered(current);
        assert!(refused_json(&mutated, root), "mutated envelope {key}");
    }
    for (key, current) in value["receipt"].as_object().unwrap() {
        let mut omitted = value.clone();
        omitted["receipt"].as_object_mut().unwrap().remove(key);
        assert!(refused_json(&omitted, root), "omitted receipt {key}");
        let mut mutated = value.clone();
        mutated["receipt"][key] = altered(current);
        assert!(refused_json(&mutated, root), "mutated receipt {key}");
    }
    for key in ["foreign_checker_accepted", "checker_result", "verified"] {
        let mut forged = value.clone();
        forged[key] = serde_json::json!(true);
        assert!(refused_json(&forged, root));
        let mut forged = value.clone();
        forged["receipt"][key] = serde_json::json!(true);
        assert!(refused_json(&forged, root));
    }
}

#[test]
fn source_semantic_changes_refuse_even_with_forged_digests() {
    let (e, root) = envelope();
    for (from, to) in [
        ("[2, -2, 1]", "[3, -2, 1]"),
        ("List Int", "List Nat"),
        ("(s + r0)", "(s - r0)"),
        ("by decide", "by sorry"),
        ("theorem projected_product", "theorem other"),
        ("#print axioms projected_product\n", ""),
    ] {
        let mut changed = e.clone();
        changed.lean_source = changed.lean_source.replace(from, to);
        // Forge the public transport digest consistently: semantic checking,
        // rather than a stale digest alone, must refuse.
        let mut h = blake3::Hasher::new();
        h.update(b"fsym.formal.statement.v1");
        h.update(changed.environment.as_bytes());
        h.update(&[0]);
        h.update(changed.profile_id.as_bytes());
        h.update(&[0]);
        h.update(changed.lean_source.as_bytes());
        changed.statement_root = fsym_formal::encode_hex(h.finalize().as_bytes());
        changed.receipt.formal_statement_root = changed.statement_root.clone();
        changed.receipt.receipt_root.clear();
        let mut h = blake3::Hasher::new();
        h.update(b"fsym.formal.zz-product.receipt.v1\0");
        h.update(&serde_json::to_vec(&changed.receipt).unwrap());
        changed.receipt.receipt_root = fsym_formal::encode_hex(h.finalize().as_bytes());
        assert!(
            check_projection(&changed, root, &|_| false).is_err(),
            "accepted {from}"
        );
    }
}

#[test]
fn false_native_coefficients_refuse_before_projection() {
    let mut c = support::capsule();
    replace_subject(&mut c, support::object(&[(4, 1), (0, 5)]));
    assert!(matches!(project(&c), Err(E::NativeVerificationFailed)));
}

#[test]
fn unsupported_domains_contexts_rules_verifiers_and_shapes_refuse() {
    let mut c = support::capsule();
    c.claim.domain = PolyDomain::Qq;
    assert!(matches!(project(&c), Err(E::UnsupportedFamily)));
    let mut c = support::capsule();
    c.claim.context = ContextId::new(1).unwrap();
    assert!(matches!(project(&c), Err(E::UnrepresentableSemanticField)));
    let mut c = support::capsule();
    c.claim.rule = RuleId::new(1).unwrap();
    assert!(matches!(project(&c), Err(E::UnrepresentableSemanticField)));
    let mut c = support::capsule();
    c.claim.verifier = VerifierId::new(1).unwrap();
    assert!(matches!(project(&c), Err(E::UnrepresentableSemanticField)));
    let mut c = support::capsule();
    c.claim.coefficient = BigRational::from_integer(BigInt::from(2));
    assert!(matches!(project(&c), Err(E::UnsupportedShape)));
    let mut c = support::capsule();
    c.claim.factors[0].1 = 2;
    assert!(matches!(project(&c), Err(E::UnsupportedShape)));
    let mut c = support::capsule();
    c.claim.factors.push((1, 1));
    assert!(matches!(
        project(&c),
        Err(E::ResourceExhausted("factor count"))
    ));
}

#[test]
fn external_claim_root_cannot_be_replaced_by_transport_authority() {
    let (e, root) = envelope();
    let mut wrong = root;
    wrong[0] ^= 1;
    assert!(check_projection(&e, wrong, &|_| false).is_err());
    let c = support::capsule();
    assert!(matches!(
        project_capsule(&c.encode().unwrap(), wrong, &|_| false),
        Err(E::NativeVerificationFailed)
    ));
    // Another valid identity yields valid self-consistent bytes and receipts,
    // but still cannot substitute for the independently requested statement.
    let mut other = support::capsule();
    let a = support::object(&[(1, 1), (0, -1)]);
    let b = support::object(&[(1, 1), (0, 1)]);
    let s = support::object(&[(2, 1), (0, -1)]);
    other.claim.subject = s.id;
    other.claim.factors = vec![(a.id, 1), (b.id, 1)];
    other.objects = [s, a, b].into_iter().map(|o| (o.id, o)).collect();
    let replacement = project(&other).unwrap().into_envelope();
    assert!(check_projection(&replacement, root, &|_| false).is_err());
}

#[test]
fn degree_coefficient_and_byte_exhaustion_are_inconclusive() {
    let mut c = support::capsule();
    replace_subject(&mut c, support::object(&[(65, 1), (0, 4)]));
    assert!(matches!(project(&c), Err(E::ResourceExhausted("degree"))));
    let mut c = support::capsule();
    let huge = BigInt::from(i64::MAX) + BigInt::from(1);
    let object = CapsuleObject::new(
        PolyObject {
            symbol: "x".into(),
            terms: vec![(4, BigRational::from_integer(huge))],
        }
        .encode()
        .unwrap(),
    );
    replace_subject(&mut c, object);
    assert!(matches!(
        project(&c),
        Err(E::ResourceExhausted("coefficient height"))
    ));
    assert!(matches!(
        project_capsule(&vec![0; MAX_CAPSULE_BYTES + 1], [0; 32], &|_| false),
        Err(E::ResourceExhausted("capsule bytes"))
    ));
    assert!(matches!(
        ProjectionEnvelope::from_json(&vec![b' '; MAX_JSON_BYTES + 1]),
        Err(E::ResourceExhausted("JSON bytes"))
    ));
}

#[test]
fn cancellation_at_admission_and_before_output_never_mints_checked_state() {
    let c = support::capsule();
    let bytes = c.encode().unwrap();
    let root = c.claim.digest().unwrap();
    for stop in [
        SafePoint::Admission,
        SafePoint::PreflightObject,
        SafePoint::BeforeNativeVerification,
        SafePoint::AfterNativeVerification,
        SafePoint::BeforeOutput,
    ] {
        assert!(matches!(
            project_capsule(&bytes, root, &|point| point == stop),
            Err(E::Cancelled)
        ));
        let (e, _) = envelope();
        assert!(matches!(
            check_projection(&e, root, &|point| point == stop),
            Err(E::Cancelled)
        ));
    }
}

#[test]
fn malformed_truncated_noncanonical_and_unknown_bytes_refuse() {
    let c = support::capsule();
    let bytes = c.encode().unwrap();
    let root = c.claim.digest().unwrap();
    for length in 0..bytes.len() {
        assert!(project_capsule(&bytes[..length], root, &|_| false).is_err());
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(project_capsule(&trailing, root, &|_| false).is_err());
    let mut schema = bytes.clone();
    schema[7] = 2;
    assert!(project_capsule(&schema, root, &|_| false).is_err());
    let mut corrupt = bytes;
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(project_capsule(&corrupt, root, &|_| false).is_err());
    assert!(ProjectionEnvelope::from_json(b"{}").is_err());
}
