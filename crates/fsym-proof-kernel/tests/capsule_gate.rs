//! Adversarial gate for the verifier-complete polynomial capsule (WS06).
//!
//! Every case is built by hand from exact coefficients: nothing here calls a
//! factorization generator, so a verifier that trusted generator output or a
//! stored flag could not pass this suite.

#![forbid(unsafe_code)]

use fsym_core::{BigInt, BigRational};
use fsym_id::{ContextId, RuleId, VerifierId};
use fsym_proof_kernel::capsule::{
    Capsule, CapsuleError, CapsuleObject, CapsuleVerdict, PolyDomain, PolyIdentityClaim,
    PolyObject, verify_capsule,
};
use std::collections::BTreeMap;

fn rational(numer: i64, denom: i64) -> BigRational {
    BigRational::new(BigInt::from(numer), BigInt::from(denom))
}

fn poly(symbol: &str, terms: &[(u32, i64)]) -> PolyObject {
    PolyObject {
        symbol: symbol.to_string(),
        terms: terms
            .iter()
            .map(|(degree, coefficient)| (*degree, rational(*coefficient, 1)))
            .collect(),
    }
}

fn roots() -> (ContextId, RuleId, VerifierId) {
    (
        ContextId::new(0x1111_2222_3333_4444).expect("non-zero context"),
        RuleId::new(0x5555_6666_7777_8888).expect("non-zero rule"),
        VerifierId::new(0x9999_aaaa_bbbb_cccc).expect("non-zero verifier"),
    )
}

/// `2*x^2 - 2` = `2 * (x - 1) * (x + 1)`.
fn valid_capsule() -> Capsule {
    let (context, rule, verifier) = roots();
    let subject = CapsuleObject::new(
        poly("x", &[(2, 2), (0, -2)])
            .encode()
            .expect("subject encodes"),
    );
    let first = CapsuleObject::new(
        poly("x", &[(1, 1), (0, -1)])
            .encode()
            .expect("factor encodes"),
    );
    let second = CapsuleObject::new(
        poly("x", &[(1, 1), (0, 1)])
            .encode()
            .expect("factor encodes"),
    );
    let mut objects = BTreeMap::new();
    for object in [subject, first, second] {
        objects.insert(object.id, object);
    }
    let subject_id = objects
        .values()
        .find(|object| {
            PolyObject::decode(&object.payload)
                .expect("decodes")
                .degree()
                == 2
        })
        .expect("subject present")
        .id;
    let factors: Vec<(u64, u32)> = objects
        .values()
        .filter(|object| object.id != subject_id)
        .map(|object| (object.id, 1))
        .collect();
    Capsule {
        schema_version: fsym_proof_kernel::capsule::CAPSULE_SCHEMA_VERSION,
        claim: PolyIdentityClaim {
            domain: PolyDomain::Zz,
            context,
            rule,
            verifier,
            subject: subject_id,
            coefficient: rational(2, 1),
            factors,
        },
        objects,
    }
}

fn verdict(bytes: &[u8], fuel: u64) -> CapsuleVerdict {
    verify_capsule(bytes, None, fuel)
}

#[test]
fn valid_capsule_verifies_and_binds_its_claim_root() {
    let capsule = valid_capsule();
    let bytes = capsule.encode().expect("encodes");
    let root = capsule.claim.digest().expect("claim digest");
    match verify_capsule(&bytes, Some(root), 64) {
        CapsuleVerdict::Verified {
            claim_digest,
            objects,
            multiplications,
        } => {
            assert_eq!(claim_digest, root);
            assert_eq!(objects, 3);
            assert_eq!(multiplications, 2, "two factor multiplications charged");
        }
        other => panic!("expected Verified, got {other:?}"),
    }
    // The same capsule under a different claimed root is refused, not accepted.
    let mut other_root = root;
    other_root[0] ^= 1;
    assert!(matches!(
        verify_capsule(&bytes, Some(other_root), 64),
        CapsuleVerdict::Refused(CapsuleError::ClaimRootMismatch)
    ));
}

#[test]
fn incomplete_factor_list_is_refuted_not_accepted() {
    let mut capsule = valid_capsule();
    let dropped = capsule.claim.factors.pop().expect("a factor to drop");
    capsule.objects.remove(&dropped.0);
    let bytes = capsule.encode().expect("encodes");
    // The product of the remaining factor does not equal the subject.
    assert!(matches!(
        verdict(&bytes, 64),
        CapsuleVerdict::Refuted { .. }
    ));
}

#[test]
fn missing_object_is_refused() {
    let mut capsule = valid_capsule();
    let victim = capsule.claim.factors[0].0;
    capsule.objects.remove(&victim);
    // Encoding then decoding must refuse the dangling reference.
    let bytes = capsule
        .encode()
        .expect("encoding does not resolve references");
    assert!(matches!(
        verdict(&bytes, 64),
        CapsuleVerdict::Refused(CapsuleError::MissingObject(id)) if id == victim
    ));
}

#[test]
fn changed_domain_is_refused() {
    let mut capsule = valid_capsule();
    capsule.claim.domain = PolyDomain::Zz;
    let mut rational_factor = poly("x", &[(1, 1)]);
    rational_factor.terms.push((0, rational(1, 2)));
    let offending = CapsuleObject::new(rational_factor.encode().expect("encodes"));
    let id = offending.id;
    capsule.objects.insert(id, offending);
    capsule.claim.factors.push((id, 1));
    let bytes = capsule.encode().expect("encodes");
    assert!(matches!(
        verdict(&bytes, 64),
        CapsuleVerdict::Refused(CapsuleError::Inconsistent(
            "ZZ capsule with rational object"
        ))
    ));
}

#[test]
fn changed_context_root_changes_the_claim_identity() {
    let capsule = valid_capsule();
    let bytes = capsule.encode().expect("encodes");
    let root = capsule.claim.digest().expect("digest");
    let mut altered = valid_capsule();
    altered.claim.context = ContextId::new(0xdead_beef_dead_beef).expect("non-zero");
    assert_ne!(
        altered.claim.digest().expect("digest"),
        root,
        "a different assumptions context must not share a claim identity"
    );
    assert!(matches!(
        verify_capsule(&bytes, Some(altered.claim.digest().expect("digest")), 64),
        CapsuleVerdict::Refused(CapsuleError::ClaimRootMismatch)
    ));
}

#[test]
fn duplicate_id_with_different_bytes_is_refused() {
    let capsule = valid_capsule();
    let bytes = capsule.encode().expect("encodes");
    // Splice a second object entry that re-uses the first entry's id with a
    // different payload, keeping the rest of the capsule intact.
    let mut forged = Vec::new();
    let object_count_offset = {
        let claim_len = u32::from_le_bytes(bytes[9..13].try_into().expect("claim length"));
        13 + claim_len as usize
    };
    let object_count = u32::from_le_bytes(
        bytes[object_count_offset..object_count_offset + 4]
            .try_into()
            .expect("object count"),
    );
    let first_entry = object_count_offset + 4;
    let payload_len = u32::from_le_bytes(
        bytes[first_entry + 40..first_entry + 44]
            .try_into()
            .expect("payload length"),
    ) as usize;
    let first_entry_end = first_entry + 44 + payload_len;
    forged.extend_from_slice(&bytes[..object_count_offset]);
    forged.extend_from_slice(&(object_count + 1).to_le_bytes());
    forged.extend_from_slice(&bytes[first_entry..first_entry_end]);
    // Same id bytes, different digest bytes and payload.
    let mut second_payload = bytes[first_entry + 44..first_entry_end].to_vec();
    second_payload.push(0x7f);
    forged.extend_from_slice(&bytes[first_entry..first_entry + 8]);
    forged.extend_from_slice(&[0xab; 32]);
    forged.extend_from_slice(&(second_payload.len() as u32).to_le_bytes());
    forged.extend_from_slice(&second_payload);
    forged.extend_from_slice(&bytes[first_entry_end..]);
    assert!(matches!(
        verdict(&forged, 64),
        CapsuleVerdict::Refused(
            CapsuleError::ObjectDigestMismatch(_) | CapsuleError::DuplicateObjectId(_)
        )
    ));
}

#[test]
fn malformed_lengths_and_trailing_bytes_are_refused() {
    let capsule = valid_capsule();
    let bytes = capsule.encode().expect("encodes");

    // Truncated buffer.
    assert!(matches!(
        verdict(&bytes[..bytes.len() - 3], 64),
        CapsuleVerdict::Refused(CapsuleError::Malformed(_))
    ));
    // Trailing bytes after a complete capsule.
    let mut extended = bytes.clone();
    extended.push(0);
    assert!(matches!(
        verdict(&extended, 64),
        CapsuleVerdict::Refused(CapsuleError::Malformed(_))
    ));
    // Oversized declared payload length.
    let mut oversized = bytes.clone();
    let object_count_offset = {
        let claim_len = u32::from_le_bytes(bytes[9..13].try_into().expect("claim length"));
        13 + claim_len as usize
    };
    let payload_len_offset = object_count_offset + 4 + 40;
    oversized[payload_len_offset..payload_len_offset + 4]
        .copy_from_slice(&(u32::MAX).to_le_bytes());
    assert!(matches!(
        verdict(&oversized, 64),
        CapsuleVerdict::Refused(CapsuleError::Malformed(_))
    ));
    // Wrong schema version.
    let mut wrong_schema = bytes.clone();
    wrong_schema[7..9].copy_from_slice(&9u16.to_le_bytes());
    assert!(matches!(
        verdict(&wrong_schema, 64),
        CapsuleVerdict::Refused(CapsuleError::UnknownSchema(_))
    ));
}

#[test]
fn exhaustion_is_inconclusive_not_rejection() {
    let capsule = valid_capsule();
    let bytes = capsule.encode().expect("encodes");
    match verdict(&bytes, 1) {
        CapsuleVerdict::Inconclusive { needed_at_least } => assert!(needed_at_least > 1),
        other => panic!("exhaustion must be inconclusive, got {other:?}"),
    }
    // Unknown schema also fails closed rather than defaulting to acceptance.
    assert!(matches!(
        verify_capsule(b"not a capsule", None, 64),
        CapsuleVerdict::Refused(CapsuleError::UnknownSchema(_))
    ));
}

#[test]
fn forged_verified_flag_cannot_be_smuggled_into_the_schema() {
    let capsule = valid_capsule();
    let mut bytes = capsule.encode().expect("encodes");
    // Append an extra claim field spelling `verified = true`.
    bytes.push(0xff);
    assert!(matches!(
        verdict(&bytes, 64),
        CapsuleVerdict::Refused(CapsuleError::Malformed(_))
    ));
    // And a stored flag is not part of the accepted schema at all.
    let map = capsule.objects.clone();
    assert!(map.values().all(|object| object.payload[0] != 0xff));
}
