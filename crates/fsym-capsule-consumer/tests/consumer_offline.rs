//! Offline consumer gate: the same capsule bytes, checked without any
//! generator, planner, runtime, Python, persistence, or network crate present.

#![forbid(unsafe_code)]

use fsym_capsule_consumer::{
    OUTCOME_INCONCLUSIVE, OUTCOME_REFUSED, OUTCOME_VERIFIED, supported_schema_version,
    verify_offline, verify_offline_default,
};
use fsym_core::{BigInt, BigRational};
use fsym_id::{ContextId, RuleId, VerifierId};
use fsym_proof_kernel::capsule::{
    CAPSULE_SCHEMA_VERSION, Capsule, CapsuleObject, PolyDomain, PolyIdentityClaim, PolyObject,
};
use std::collections::BTreeMap;

fn capsule_bytes() -> (Vec<u8>, [u8; 32]) {
    let subject = CapsuleObject::new(
        PolyObject {
            symbol: "x".to_string(),
            terms: vec![
                (3, BigRational::new(BigInt::from(1), BigInt::from(1))),
                (0, BigRational::new(BigInt::from(-1), BigInt::from(1))),
            ],
        }
        .encode()
        .expect("subject encodes"),
    );
    let factor = CapsuleObject::new(
        PolyObject {
            symbol: "x".to_string(),
            terms: vec![
                (1, BigRational::new(BigInt::from(1), BigInt::from(1))),
                (0, BigRational::new(BigInt::from(-1), BigInt::from(1))),
            ],
        }
        .encode()
        .expect("factor encodes"),
    );
    let second = CapsuleObject::new(
        PolyObject {
            symbol: "x".to_string(),
            terms: vec![
                (2, BigRational::new(BigInt::from(1), BigInt::from(1))),
                (1, BigRational::new(BigInt::from(1), BigInt::from(1))),
                (0, BigRational::new(BigInt::from(1), BigInt::from(1))),
            ],
        }
        .encode()
        .expect("second factor encodes"),
    );
    let mut objects = BTreeMap::new();
    for object in [subject, factor, second] {
        objects.insert(object.id, object);
    }
    let mut ids = objects.keys().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    let subject_id = objects
        .values()
        .find(|object| {
            PolyObject::decode(&object.payload)
                .map(|poly| poly.terms.len() == 2 && poly.terms[0].0 == 3)
                .unwrap_or(false)
        })
        .expect("subject present")
        .id;
    let factors = ids
        .into_iter()
        .filter(|id| *id != subject_id)
        .map(|id| (id, 1))
        .collect();
    let capsule = Capsule {
        schema_version: CAPSULE_SCHEMA_VERSION,
        claim: PolyIdentityClaim {
            domain: PolyDomain::Zz,
            context: ContextId::new(0x0102_0304_0506_0708).expect("non-zero"),
            rule: RuleId::new(0x1112_1314_1516_1718).expect("non-zero"),
            verifier: VerifierId::new(0x2122_2324_2526_2728).expect("non-zero"),
            subject: subject_id,
            coefficient: BigRational::new(BigInt::from(1), BigInt::from(1)),
            factors,
        },
        objects,
    };
    let root = capsule.claim.digest().expect("claim digest");
    (capsule.encode().expect("capsule encodes"), root)
}

#[test]
fn consumer_verifies_a_valid_capsule_with_no_generator_present() {
    let (bytes, root) = capsule_bytes();
    assert_eq!(supported_schema_version(), CAPSULE_SCHEMA_VERSION);

    let report = verify_offline(&bytes, Some(root), 64);
    assert_eq!(report.outcome, OUTCOME_VERIFIED);
    assert!(report.is_verified());
    assert_eq!(report.claim_digest, Some(root));
    assert_eq!(report.objects, 3);
    assert_eq!(report.multiplications, 2);
    assert!(report.detail.is_empty());
}

#[test]
fn consumer_refuses_a_capsule_for_another_claim_root() {
    let (bytes, root) = capsule_bytes();
    let mut wrong = root;
    wrong[31] ^= 0x80;
    let report = verify_offline(&bytes, Some(wrong), 64);
    assert_eq!(report.outcome, OUTCOME_REFUSED);
    assert!(!report.is_verified());
}

#[test]
fn consumer_reports_exhaustion_as_inconclusive() {
    let (bytes, _) = capsule_bytes();
    let report = verify_offline(&bytes, None, 1);
    assert_eq!(report.outcome, OUTCOME_INCONCLUSIVE);
    assert!(report.is_inconclusive());
    assert!(!report.is_verified(), "exhaustion must never accept");
}

#[test]
fn consumer_refuses_malformed_input_without_falling_back() {
    let report = verify_offline_default(b"FSYMCAP\x01\x00");
    assert_eq!(report.outcome, OUTCOME_REFUSED);
    assert!(report.detail.contains("malformed") || report.detail.contains("Malformed"));
    let empty = verify_offline_default(&[]);
    assert_eq!(empty.outcome, OUTCOME_REFUSED);
}
