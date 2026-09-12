//! Reviewer-owned adversarial gate for the verifier-complete polynomial capsule
//! (`fra-rc-capsule-gate-5mq`, reviewing implementation `fra-rc-capsule-39v` at
//! commit `992b425`).
//!
//! Independent authorship: this file was written by the gate reviewer, not by
//! the implementation author, and it does not call or reuse the author's tests.
//! Every capsule here is assembled from hand-written coefficients through the
//! *public* API only (`fsym_capsule_consumer::verify_offline` and the public
//! capsule types exported by `fsym-proof-kernel`). No generator, no
//! factorization routine, and no stored `verified` flag participates.
//!
//! Attack obligations, one test each: missing object; changed domain (ZZ vs QQ)
//! and changed assumption/context root (plus changed rule and verifier roots);
//! duplicate object id with different bytes; malformed lengths (truncated,
//! oversized declared length, trailing bytes in capsule claim and object, wrong
//! schema/magic/kind/degree/symbol); exhausted fuel must be Inconclusive and
//! never Verified; incomplete factor list must be Refuted; a forged `verified`
//! flag or any extra trailing field must be Refused; and a cross-check that the
//! accepted product really multiplies back to the subject.
//!
//! Verdicts are asserted as *classes* plus raw detail text, so a verifier that
//! returned a plausible-but-wrong class (for example `Verified` where the claim
//! is false) fails the test rather than passing on a substring.

#![forbid(unsafe_code)]

use fsym_capsule_consumer::{
    CapsuleReport, OUTCOME_INCONCLUSIVE, OUTCOME_REFUTED, OUTCOME_REFUSED, OUTCOME_VERIFIED,
    supported_schema_version, verify_offline, verify_offline_default,
};
use fsym_core::{BigInt, BigRational, Expr};
use fsym_id::{ContextId, RuleId, VerifierId};
use fsym_proof_kernel::capsule::{
    CAPSULE_SCHEMA_VERSION, Capsule, CapsuleObject, PolyDomain, PolyIdentityClaim, PolyObject,
    object_digest,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Hand-written capsule construction (public API only)
// ---------------------------------------------------------------------------

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

fn poly_q(symbol: &str, terms: &[(u32, (i64, i64))]) -> PolyObject {
    PolyObject {
        symbol: symbol.to_string(),
        terms: terms
            .iter()
            .map(|(degree, (n, d))| (*degree, rational(*n, *d)))
            .collect(),
    }
}

const CONTEXT: u64 = 0x1111_2222_3333_4444;
const RULE: u64 = 0x5555_6666_7777_8888;
const VERIFIER: u64 = 0x9999_aaaa_bbbb_cccc;

fn claim(
    domain: PolyDomain,
    subject: u64,
    coefficient: BigRational,
    factors: Vec<(u64, u32)>,
    context: u64,
) -> PolyIdentityClaim {
    PolyIdentityClaim {
        domain,
        context: ContextId::new(context).expect("non-zero context root"),
        rule: RuleId::new(RULE).expect("non-zero rule root"),
        verifier: VerifierId::new(VERIFIER).expect("non-zero verifier root"),
        subject,
        coefficient,
        factors,
    }
}

/// Assemble a capsule from hand-written objects. The subject is always its own
/// object; each factor gets a fresh object entry.
fn assemble(
    domain: PolyDomain,
    subject: PolyObject,
    coefficient: BigRational,
    factors: &[(PolyObject, u32)],
) -> Capsule {
    let subject_object = CapsuleObject::new(subject.encode().expect("subject encodes"));
    let subject_id = subject_object.id;
    let mut objects = BTreeMap::new();
    objects.insert(subject_id, subject_object);
    let mut references = Vec::new();
    for (factor, exponent) in factors {
        let object = CapsuleObject::new(factor.encode().expect("factor encodes"));
        references.push((object.id, *exponent));
        objects.insert(object.id, object);
    }
    Capsule {
        schema_version: CAPSULE_SCHEMA_VERSION,
        claim: claim(domain, subject_id, coefficient, references, CONTEXT),
        objects,
    }
}

fn encoded(capsule: &Capsule) -> Vec<u8> {
    capsule.encode().expect("capsule encodes")
}

fn root_of(capsule: &Capsule) -> [u8; 32] {
    capsule.claim.digest().expect("claim digest")
}

/// Verify under the capsule's own externally computed claim root.
fn check_rooted(capsule: &Capsule, fuel: u64) -> CapsuleReport {
    verify_offline(&encoded(capsule), Some(root_of(capsule)), fuel)
}

/// Raw class + detail, used in assertion messages so a failure shows the real
/// verdict instead of only the expectation.
fn class(report: &CapsuleReport) -> String {
    format!("{}: {}", report.outcome, report.detail)
}

fn expect(report: &CapsuleReport, outcome: &str) -> () {
    assert_eq!(
        report.outcome, outcome,
        "verdict mismatch (observed detail: {})",
        report.detail
    );
}

// Offsets inside canonical capsule bytes:
//   0..7 magic | 7..9 schema version | 9..13 claim length | 13.. claim | ...
const CLAIM_START: usize = 13;

fn claim_len(bytes: &[u8]) -> usize {
    u32::from_le_bytes(bytes[9..13].try_into().expect("claim length")) as usize
}

fn object_count_offset(bytes: &[u8]) -> usize {
    CLAIM_START + claim_len(bytes)
}

fn factor_count_offset(bytes: &[u8]) -> usize {
    // claim preimage: domain(1) context(8) rule(8) verifier(8) subject(8)
    //                 coefficient_len(4) coefficient(...) factor_count(4)
    let coefficient_len = u32::from_le_bytes(
        bytes[CLAIM_START + 33..CLAIM_START + 37]
            .try_into()
            .expect("coefficient length"),
    ) as usize;
    CLAIM_START + 37 + coefficient_len
}

// ---------------------------------------------------------------------------
// Canonical numeric encodings for hand-assembled raw payloads
// ---------------------------------------------------------------------------

/// Canonical coefficient bytes for one small integer, taken from the same
/// canonical numeric codec the capsule format uses. (`PolyObject::encode`
/// refuses a zero coefficient, so the encoder cannot be used to obtain the zero
/// spelling.)
fn coefficient_bytes(value: i64) -> Vec<u8> {
    Expr::Integer(BigInt::from(value))
        .to_canonical_numeric_bytes()
        .expect("integer is canonically encodable")
}

/// Hand-assembled object payload; `kind` is settable so schema attacks are
/// possible without calling the encoder's validating path.
fn raw_object(symbol: &str, kind: u8, terms: &[(u32, i64)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(kind);
    out.extend_from_slice(&(symbol.len() as u32).to_le_bytes());
    out.extend_from_slice(symbol.as_bytes());
    out.extend_from_slice(&(terms.len() as u32).to_le_bytes());
    for (degree, coefficient) in terms {
        out.extend_from_slice(&degree.to_le_bytes());
        let bytes = coefficient_bytes(*coefficient);
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&bytes);
    }
    out
}

/// Rebuild a capsule whose *subject* payload is raw bytes supplied by the test.
fn capsule_with_raw_subject(
    domain: PolyDomain,
    raw_subject: Vec<u8>,
    coefficient: BigRational,
    factors: &[(PolyObject, u32)],
) -> Capsule {
    let subject_object = CapsuleObject::new(raw_subject);
    let subject_id = subject_object.id;
    let mut objects = BTreeMap::new();
    objects.insert(subject_id, subject_object);
    let mut references = Vec::new();
    for (factor, exponent) in factors {
        let object = CapsuleObject::new(factor.encode().expect("factor encodes"));
        references.push((object.id, *exponent));
        objects.insert(object.id, object);
    }
    Capsule {
        schema_version: CAPSULE_SCHEMA_VERSION,
        claim: claim(domain, subject_id, coefficient, references, CONTEXT),
        objects,
    }
}

// ---------------------------------------------------------------------------
// Positive controls
// ---------------------------------------------------------------------------

#[test]
fn review_positive_control_zz_identity_verifies_through_the_consumer() {
    // 2x^2 - 2 = 2 * (x - 1) * (x + 1)
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let report = check_rooted(&capsule, 64);
    expect(&report, OUTCOME_VERIFIED);
    assert_eq!(report.objects, 3, "three closed objects resolved");
    assert_eq!(report.multiplications, 2, "two multiplications charged");
    assert_eq!(report.claim_digest, Some(root_of(&capsule)));
    assert_eq!(supported_schema_version(), CAPSULE_SCHEMA_VERSION);
}

#[test]
fn review_positive_control_qq_identity_verifies() {
    // (1/2)x^2 - 1/2 = (1/2) * (x - 1) * (x + 1)
    let capsule = assemble(
        PolyDomain::Qq,
        poly_q("x", &[(2, (1, 2)), (0, (-1, 2))]),
        rational(1, 2),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    expect(&check_rooted(&capsule, 64), OUTCOME_VERIFIED);
}

#[test]
fn review_no_stored_flag_participates_and_the_claim_is_recomputed() {
    // The user-supplied name is irrelevant and never reaches the wire: a capsule
    // cannot carry authority, only objects. The subject below is wrong and must
    // be caught although the capsule is otherwise well formed.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(1, 1)]),
        rational(1, 1),
        &[],
    );
    let report = check_rooted(&capsule, 64);
    expect(&report, OUTCOME_REFUTED);
}

// ---------------------------------------------------------------------------
// Missing object / no fallback
// ---------------------------------------------------------------------------

#[test]
fn review_missing_factor_object_is_refused() {
    let mut capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let victim = capsule.claim.factors[0].0;
    capsule.objects.remove(&victim);
    let bytes = capsule.encode().expect("encoding does not resolve references");
    let report = verify_offline(&bytes, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(
        report.detail.contains("MissingObject"),
        "expected MissingObject, detail was {}",
        report.detail
    );
}

#[test]
fn review_missing_subject_object_is_refused() {
    let mut capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let subject = capsule.claim.subject;
    capsule.objects.remove(&subject);
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("MissingObject"), "detail {}", report.detail);
}

// ---------------------------------------------------------------------------
// Domain and trust-root attacks
// ---------------------------------------------------------------------------

#[test]
fn review_changed_domain_zz_vs_qq_is_refused() {
    let zz = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let bytes = encoded(&zz);
    // Same objects and coefficients, declared over QQ instead: the claim
    // identity must change, so the ZZ bytes cannot be accepted under the QQ root.
    let mut qq_claim = zz.claim.clone();
    qq_claim.domain = PolyDomain::Qq;
    let qq_root = qq_claim.digest().expect("digest");
    assert_ne!(qq_root, root_of(&zz), "domain is part of claim identity");
    let report = verify_offline(&bytes, Some(qq_root), 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("ClaimRootMismatch"), "detail {}", report.detail);

    // ... and a one-byte domain relabel is caught through the claim root: the
    // same object bytes re-tagged QQ no longer match the ZZ root.
    let mut relabelled = encoded(&zz);
    relabelled[CLAIM_START] = 2; // QQ tag
    let report = verify_offline(&relabelled, Some(root_of(&zz)), 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("ClaimRootMismatch"), "detail {}", report.detail);
    // Under the QQ root of that relabelled claim it does check out — QQ accepts
    // integer objects, so the tag alone is not evidence about the objects. That
    // is exactly why the caller-supplied root, not the capsule, is the authority.
    expect(&verify_offline(&relabelled, Some(qq_root), 64), OUTCOME_VERIFIED);
}

#[test]
fn review_zz_capsule_with_rational_coefficient_is_refused() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly_q("x", &[(2, (1, 2)), (0, (-1, 2))]),
        rational(1, 2),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Inconsistent"), "detail {}", report.detail);
}

#[test]
fn review_zz_capsule_with_rational_object_is_refused() {
    let capsule = capsule_with_raw_subject(
        PolyDomain::Zz,
        raw_object("x", 0x01, &[(1, 1), (0, -1)]),
        rational(1, 1),
        &[],
    );
    // Replace the subject with a rational object while claiming ZZ.
    let mut capsule = capsule;
    let subject = capsule.claim.subject;
    let rational_object = CapsuleObject::new(
        poly_q("x", &[(1, (1, 2))])
            .encode()
            .expect("rational object encodes"),
    );
    let id = rational_object.id;
    capsule.objects.remove(&subject);
    capsule.objects.insert(id, rational_object);
    capsule.claim.subject = id;
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Inconsistent"), "detail {}", report.detail);
}

#[test]
fn review_changed_context_root_breaks_the_claim_identity() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let bytes = encoded(&capsule);
    let original = root_of(&capsule);

    let other_claim = claim(
        PolyDomain::Zz,
        capsule.claim.subject,
        capsule.claim.coefficient.clone(),
        capsule.claim.factors.clone(),
        0xdead_beef_dead_beef,
    );
    let other_root = other_claim.digest().expect("digest");
    assert_ne!(other_root, original, "different context => different claim");
    let report = verify_offline(&bytes, Some(other_root), 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("ClaimRootMismatch"), "detail {}", report.detail);

    // The same bytes under their own root still verify (identity, not a ban).
    expect(&verify_offline(&bytes, Some(original), 64), OUTCOME_VERIFIED);
}

#[test]
fn review_changed_rule_and_verifier_roots_are_bound_into_the_claim() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let bytes = encoded(&capsule);
    let original = root_of(&capsule);

    let mut rule_changed = capsule.claim.clone();
    rule_changed.rule = RuleId::new(0x0123_4567_89ab_cdef).expect("non-zero");
    let mut verifier_changed = capsule.claim.clone();
    verifier_changed.verifier = VerifierId::new(0x0123_4567_89ab_cdef).expect("non-zero");
    for altered in [&rule_changed, &verifier_changed] {
        let root = altered.digest().expect("digest");
        assert_ne!(root, original);
        let report = verify_offline(&bytes, Some(root), 64);
        expect(&report, OUTCOME_REFUSED);
        assert!(report.detail.contains("ClaimRootMismatch"), "detail {}", report.detail);
    }
}

// ---------------------------------------------------------------------------
// Duplicate object id with different bytes
// ---------------------------------------------------------------------------

#[test]
fn review_duplicate_object_id_with_different_bytes_is_refused() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1)],
    );
    let bytes = encoded(&capsule);
    let count_at = object_count_offset(&bytes);
    let count = u32::from_le_bytes(bytes[count_at..count_at + 4].try_into().expect("count"));
    let first = count_at + 4;
    let first_payload_len =
        u32::from_le_bytes(bytes[first + 40..first + 44].try_into().expect("payload length"))
            as usize;
    let first_end = first + 44 + first_payload_len;

    // Second entry: same id bytes, a *different* payload, and the honest digest
    // of that different payload. Both entries are internally well framed.
    let other_payload = poly("x", &[(5, 7)]).encode().expect("encodes");
    let other_digest = object_digest(&other_payload);
    let mut forged = Vec::new();
    forged.extend_from_slice(&bytes[..count_at]);
    forged.extend_from_slice(&(count + 1).to_le_bytes());
    forged.extend_from_slice(&bytes[first..first_end]);
    forged.extend_from_slice(&bytes[first..first + 8]); // reused id
    forged.extend_from_slice(&other_digest); // honest digest of other payload
    forged.extend_from_slice(&(other_payload.len() as u32).to_le_bytes());
    forged.extend_from_slice(&other_payload);
    forged.extend_from_slice(&bytes[first_end..]);

    let report = verify_offline(&forged, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(
        report.detail.contains("ObjectDigestMismatch") || report.detail.contains("DuplicateObjectId"),
        "detail {}",
        report.detail
    );
}

#[test]
fn review_object_id_not_derived_from_its_own_payload_is_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let mut bytes = encoded(&capsule);
    let count_at = object_count_offset(&bytes);
    let first = count_at + 4;
    let payload_len =
        u32::from_le_bytes(bytes[first + 40..first + 44].try_into().expect("payload length"))
            as usize;
    // Recompute the honest digest of the payload, then flip one digest byte so
    // the id no longer follows from the bytes.
    let payload = bytes[first + 44..first + 44 + payload_len].to_vec();
    let mut digest = object_digest(&payload);
    digest[31] ^= 0x80;
    bytes[first + 8..first + 40].copy_from_slice(&digest);
    let report = verify_offline(&bytes, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("ObjectDigestMismatch"), "detail {}", report.detail);
}

#[test]
fn review_payload_tamper_after_framing_is_refused() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let mut bytes = encoded(&capsule);
    let count_at = object_count_offset(&bytes);
    let first = count_at + 4;
    let payload_len =
        u32::from_le_bytes(bytes[first + 40..first + 44].try_into().expect("payload length"))
            as usize;
    // Flip the last byte of an object payload (a coefficient byte).
    bytes[first + 44 + payload_len - 1] ^= 0x01;
    let report = verify_offline(&bytes, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("ObjectDigestMismatch"), "detail {}", report.detail);
}

#[test]
fn review_duplicate_factor_reference_is_refused() {
    let mut capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let first = capsule.claim.factors[0];
    capsule.claim.factors.push(first);
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("DuplicateFactor"), "detail {}", report.detail);
}

#[test]
fn review_zero_exponent_factor_is_refused() {
    let mut capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    capsule.claim.factors[0].1 = 0;
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Inconsistent"), "detail {}", report.detail);
}

#[test]
fn review_zero_coefficient_claim_is_refused() {
    let mut capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    capsule.claim.coefficient = rational(0, 1);
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Inconsistent"), "detail {}", report.detail);
}

// ---------------------------------------------------------------------------
// Malformed lengths, framing, and schema
// ---------------------------------------------------------------------------

#[test]
fn review_truncated_buffer_is_refused() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let bytes = encoded(&capsule);
    for cut in [1usize, 3, 8, 33] {
        let report = verify_offline(&bytes[..bytes.len() - cut], None, 64);
        expect(&report, OUTCOME_REFUSED);
        assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
    }
    expect(&verify_offline(&bytes[..4], None, 64), OUTCOME_REFUSED);
    expect(&verify_offline(&[], None, 64), OUTCOME_REFUSED);
}

#[test]
fn review_trailing_bytes_after_capsule_are_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let mut bytes = encoded(&capsule);
    bytes.push(0x00);
    let report = verify_offline(&bytes, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
}

#[test]
fn review_oversized_declared_claim_length_is_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let mut bytes = encoded(&capsule);
    bytes[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
    let report = verify_offline(&bytes, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
}

#[test]
fn review_oversized_declared_object_length_is_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let bytes = encoded(&capsule);
    let count_at = object_count_offset(&bytes);
    let first = count_at + 4;
    for declared in [u32::MAX, 300 * 1024, 64 * 1024] {
        let mut forged = bytes.clone();
        forged[first + 40..first + 44].copy_from_slice(&declared.to_le_bytes());
        let report = verify_offline(&forged, None, 64);
        expect(&report, OUTCOME_REFUSED);
        assert!(
            report.detail.contains("Malformed"),
            "declared {} => detail {}",
            declared,
            report.detail
        );
    }
}

#[test]
fn review_trailing_bytes_inside_claim_are_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let bytes = encoded(&capsule);
    let len = claim_len(&bytes);
    let mut forged = Vec::new();
    forged.extend_from_slice(&bytes[..9]);
    forged.extend_from_slice(&((len + 1) as u32).to_le_bytes());
    forged.extend_from_slice(&bytes[CLAIM_START..CLAIM_START + len]);
    forged.push(0x00); // smuggled byte after the last claim field
    forged.extend_from_slice(&bytes[CLAIM_START + len..]);
    let report = verify_offline(&forged, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
}

#[test]
fn review_trailing_bytes_inside_an_object_are_refused() {
    let mut payload = raw_object("x", 0x01, &[(1, 1)]);
    payload.push(0x00);
    let capsule = capsule_with_raw_subject(PolyDomain::Zz, payload, rational(1, 1), &[]);
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
}

#[test]
fn review_wrong_schema_version_and_magic_are_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let bytes = encoded(&capsule);
    let mut wrong_version = bytes.clone();
    wrong_version[7..9].copy_from_slice(&9u16.to_le_bytes());
    let report = verify_offline(&wrong_version, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("UnknownSchema"), "detail {}", report.detail);

    let mut wrong_magic = bytes.clone();
    wrong_magic[0] = b'X';
    let report = verify_offline(&wrong_magic, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("UnknownSchema"), "detail {}", report.detail);

    // Wrong schema *before* anything else: never a fallback to a default parse.
    let report = verify_offline(b"not a capsule at all", None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("UnknownSchema"), "detail {}", report.detail);
}

#[test]
fn review_wrong_domain_tag_is_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let mut bytes = encoded(&capsule);
    for tag in [0u8, 3, 0xff] {
        bytes[CLAIM_START] = tag;
        let report = verify_offline(&bytes, None, 64);
        expect(&report, OUTCOME_REFUSED);
        assert!(
            report.detail.contains("UnknownSchema"),
            "tag {} => detail {}",
            tag,
            report.detail
        );
    }
}

#[test]
fn review_zero_context_root_is_refused() {
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let mut bytes = encoded(&capsule);
    bytes[CLAIM_START + 1..CLAIM_START + 9].copy_from_slice(&[0u8; 8]);
    let report = verify_offline(&bytes, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
}

#[test]
fn review_wrong_object_kind_is_refused() {
    let capsule =
        capsule_with_raw_subject(PolyDomain::Zz, raw_object("x", 0x02, &[(1, 1)]), rational(1, 1), &[]);
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("UnknownSchema"), "detail {}", report.detail);
}

#[test]
fn review_non_canonical_object_encodings_are_refused() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("equal degrees", raw_object("x", 0x01, &[(1, 1), (1, 1)])),
        ("ascending degrees", raw_object("x", 0x01, &[(0, 1), (1, 1)])),
        ("zero coefficient", raw_object("x", 0x01, &[(1, 0)])),
        ("empty symbol", raw_object("", 0x01, &[(1, 1)])),
    ];
    for (label, payload) in cases {
        let capsule = capsule_with_raw_subject(PolyDomain::Zz, payload, rational(1, 1), &[]);
        let report = verify_offline(&encoded(&capsule), None, 64);
        assert_eq!(
            report.outcome, OUTCOME_REFUSED,
            "{label}: non-canonical object must be refused, got {}",
            class(&report)
        );
    }
}

/// The zero polynomial is representable (an object with no terms). Comparing it
/// must still be an equality test: 0 != 1.
#[test]
fn review_zero_polynomial_subject_must_not_verify_against_a_constant() {
    let capsule = capsule_with_raw_subject(
        PolyDomain::Zz,
        raw_object("x", 0x01, &[]),
        rational(1, 1),
        &[],
    );
    let report = verify_offline(&encoded(&capsule), None, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUTED,
        "the zero polynomial was accepted as 1; observed {}",
        class(&report)
    );
}

/// A factor carrying an extra term below the subject's top degree must make the
/// identity fail, however plausible the factor list looks.
#[test]
fn review_factor_with_extra_middle_term_must_be_refuted() {
    // subject = x^2 - 1, claimed as 1 * (x^2 + x - 1). Differs by x.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 1), (0, -1)]),
        rational(1, 1),
        &[(poly("x", &[(2, 1), (1, 1), (0, -1)]), 1)],
    );
    let report = check_rooted(&capsule, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUTED,
        "x^2 - 1 was accepted as 1*(x^2 + x - 1); observed {}",
        class(&report)
    );
}

#[test]
fn review_oversized_counts_and_symbol_are_refused() {
    // Declared term count above MAX_TERMS (8192) with no data behind it.
    let mut payload = Vec::new();
    payload.push(0x01);
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.push(b'x');
    payload.extend_from_slice(&(9000u32).to_le_bytes());
    let capsule = capsule_with_raw_subject(PolyDomain::Zz, payload, rational(1, 1), &[]);
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);

    // Symbol longer than MAX_SYMBOL_BYTES.
    let long_symbol = "s".repeat(300);
    let capsule = capsule_with_raw_subject(
        PolyDomain::Zz,
        raw_object(&long_symbol, 0x01, &[(1, 1)]),
        rational(1, 1),
        &[],
    );
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);

    // Factor count above MAX_FACTORS declared in the claim.
    let capsule = assemble(PolyDomain::Zz, poly("x", &[(1, 1)]), rational(1, 1), &[]);
    let mut bytes = encoded(&capsule);
    let at = factor_count_offset(&bytes);
    bytes[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let report = verify_offline(&bytes, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);

    // Oversized whole-capsule buffer (past MAX_CAPSULE_BYTES).
    let oversized = vec![0u8; 1024 * 1024 + 1];
    let report = verify_offline(&oversized, None, 64);
    expect(&report, OUTCOME_REFUSED);
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
}

// ---------------------------------------------------------------------------
// Fuel exhaustion
// ---------------------------------------------------------------------------

#[test]
fn review_exhaustion_is_inconclusive_and_never_verified() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    for fuel in [0u64, 1] {
        let report = check_rooted(&capsule, fuel);
        expect(&report, OUTCOME_INCONCLUSIVE);
        assert!(
            !report.is_verified(),
            "fuel {fuel} must never be an acceptance: {}",
            class(&report)
        );
        assert!(report.is_inconclusive());
    }
    // Exactly the required budget still verifies: exhaustion is a bound, not a
    // rejection heuristic.
    expect(&check_rooted(&capsule, 2), OUTCOME_VERIFIED);
    expect(&check_rooted(&capsule, 3), OUTCOME_VERIFIED);
}

#[test]
fn review_exhaustion_precedes_refutation_for_a_false_claim() {
    // A false claim must not become a refutation (or an acceptance) merely
    // because fuel ran out first.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(3, 1), // wrong coefficient
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let report = check_rooted(&capsule, 1);
    expect(&report, OUTCOME_INCONCLUSIVE);
    expect(&check_rooted(&capsule, 64), OUTCOME_REFUTED);
}

#[test]
fn review_zero_fuel_with_zero_work_is_a_boundary_not_a_false_accept() {
    // Distinct subject/constant: with no factors there is nothing to multiply,
    // so fuel 0 is sufficient and the constant must still be compared.
    let equal = assemble(PolyDomain::Zz, poly("x", &[(0, 5)]), rational(5, 1), &[]);
    expect(&check_rooted(&equal, 0), OUTCOME_VERIFIED);
    let unequal = assemble(PolyDomain::Zz, poly("x", &[(0, 5)]), rational(6, 1), &[]);
    expect(&check_rooted(&unequal, 0), OUTCOME_REFUTED);
}

// ---------------------------------------------------------------------------
// Incomplete factor list
// ---------------------------------------------------------------------------

#[test]
fn review_incomplete_factor_list_is_refuted() {
    // 2x^2 - 2 with only (x - 1): the product is a degree-1 polynomial.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1)],
    );
    let report = check_rooted(&capsule, 64);
    expect(&report, OUTCOME_REFUTED);
    assert!(!report.is_verified());
}

#[test]
fn review_incomplete_factor_list_with_equal_degree_is_refuted() {
    // 2x^2 - 2 with only (x + 1) * 2: degree 1 vs degree 2.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    expect(&check_rooted(&capsule, 64), OUTCOME_REFUTED);
}

#[test]
fn review_dropped_factor_kept_in_object_store_is_still_refuted() {
    // The object store stays closed and complete; only the *claim* is missing a
    // factor. Completeness of the factor list is not something the verifier may
    // assume, so this must be refuted.
    let mut capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    capsule.claim.factors.pop();
    let report = verify_offline(&encoded(&capsule), None, 64);
    expect(&report, OUTCOME_REFUTED);
    assert!(!report.is_verified());
}

// ---------------------------------------------------------------------------
// Forged `verified` flag / extra fields
// ---------------------------------------------------------------------------

#[test]
fn review_trailing_flag_byte_after_capsule_is_refused() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let bytes = encoded(&capsule);
    let mut flag = bytes.clone();
    flag.push(0xff); // a smuggled `verified = true` byte
    let report = verify_offline(&flag, None, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUSED,
        "trailing flag byte accepted; observed {}",
        class(&report)
    );
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
    assert!(!verify_offline(&flag, Some(root_of(&capsule)), 64).is_verified());
}

#[test]
fn review_trailing_named_field_after_capsule_is_refused() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let mut text = encoded(&capsule);
    text.extend_from_slice(b"verified=true");
    let report = verify_offline(&text, None, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUSED,
        "trailing `verified=true` field accepted; observed {}",
        class(&report)
    );
    assert!(report.detail.contains("Malformed"), "detail {}", report.detail);
}

#[test]
fn review_flag_byte_inside_object_store_is_refused() {
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let mut inner = encoded(&capsule);
    let count_at = object_count_offset(&inner);
    let first = count_at + 4;
    let payload_len =
        u32::from_le_bytes(inner[first + 40..first + 44].try_into().expect("payload length"))
            as usize;
    assert!(payload_len > 0, "first object must carry a payload");
    inner[first + 44 + payload_len - 1] ^= 0x80; // last payload byte -> `verified`
    assert!(
        first + 44 + payload_len <= inner.len(),
        "mutation must stay inside the first object"
    );
    let report = verify_offline(&inner, None, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUSED,
        "flag byte inside the object store accepted; observed {}",
        class(&report)
    );
    assert!(report.detail.contains("ObjectDigestMismatch"), "detail {}", report.detail);
}

// ---------------------------------------------------------------------------
// Cross-check: does the accepted product really multiply back?
// ---------------------------------------------------------------------------

#[test]
fn review_wrong_coefficient_is_refuted() {
    // 2x^2 - 2 = 3 * (x - 1)(x + 1) is false.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(3, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let report = check_rooted(&capsule, 64);
    expect(&report, OUTCOME_REFUTED);
    assert!(report.detail.contains("coefficient mismatch"), "detail {}", report.detail);
}

#[test]
fn review_wrong_factor_is_refuted() {
    // 2x^2 - 2 = 2 * (x - 1) * (x + 2)? No: that is x^2 + x - 2.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 2)]), 1)],
    );
    expect(&check_rooted(&capsule, 64), OUTCOME_REFUTED);
}

#[test]
fn review_subtly_wrong_rational_coefficient_is_refuted() {
    // (1/2)x^2 - 1/2 = (1/3)(x - 1)(x + 1) is false by 1/6.
    let capsule = assemble(
        PolyDomain::Qq,
        poly_q("x", &[(2, (1, 2)), (0, (-1, 2))]),
        rational(1, 3),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    expect(&check_rooted(&capsule, 64), OUTCOME_REFUTED);
}

#[test]
fn review_wrong_exponent_is_refuted() {
    // 2x^2 - 2 = 2 * (x - 1)^2 * (x + 1)? A repeated factor must change the product.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 2), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    expect(&check_rooted(&capsule, 64), OUTCOME_REFUTED);
}

/// THE decisive cross-check: the verifier must accept only when the product
/// *equals* the subject. A product with an extra term below the subject's top
/// degree is not equal to the subject and must be Refuted.
#[test]
fn review_extra_product_term_must_be_refuted() {
    // subject = x^2, claimed as 1 * (x^2 + 1).  x^2 != x^2 + 1.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 1)]),
        rational(1, 1),
        &[(poly("x", &[(2, 1), (0, 1)]), 1)],
    );
    let report = check_rooted(&capsule, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUTED,
        "x^2 was accepted as 1*(x^2+1); observed verdict: {}",
        class(&report)
    );
}

#[test]
fn review_missing_constant_term_in_subject_must_be_refuted() {
    // subject object is 2x^2, claimed as 2*(x-1)*(x+1) = 2x^2 - 2.
    // 2x^2 != 2x^2 - 2: the product carries a constant term the subject lacks.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let report = check_rooted(&capsule, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUTED,
        "2x^2 was accepted as 2*(x-1)*(x+1); observed verdict: {}",
        class(&report)
    );
}

#[test]
fn review_extra_high_degree_product_term_must_be_refuted() {
    // subject = x^3, claimed as 1 * (x^3 + x). Different polynomials.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(3, 1)]),
        rational(1, 1),
        &[(poly("x", &[(3, 1), (1, 1)]), 1)],
    );
    let report = check_rooted(&capsule, 64);
    assert_eq!(
        report.outcome, OUTCOME_REFUTED,
        "x^3 was accepted as 1*(x^3+x); observed verdict: {}",
        class(&report)
    );
}

// ---------------------------------------------------------------------------
// Trust boundary of the root-less entry point
// ---------------------------------------------------------------------------

#[test]
fn review_verify_offline_default_accepts_self_asserted_claims() {
    // Documented, but load-bearing: the default wrapper passes `None`, so the
    // capsule supplies the statement it is checked against. Recorded here so the
    // gate report can weigh it explicitly rather than discovering it later.
    let capsule = assemble(
        PolyDomain::Zz,
        poly("x", &[(2, 2), (0, -2)]),
        rational(2, 1),
        &[(poly("x", &[(1, 1), (0, -1)]), 1), (poly("x", &[(1, 1), (0, 1)]), 1)],
    );
    let report = verify_offline_default(&encoded(&capsule));
    expect(&report, OUTCOME_VERIFIED);
}
