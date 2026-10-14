//! Independent mapping validation: parse the frozen Lean grammar and compare
//! coefficients to native sparse objects. Never call the projector or trust a
//! receipt digest as evidence. Shared code is limited to native admission, the
//! frozen operator specification, and cryptographic/transport encodings.
use crate::{
    CHECKER_PIN, CheckedProjection, LEAN_NFACTOR_PREFIX, LEAN_PREFIX, MAX_CAPSULE_BYTES,
    MAX_DEGREE, PROFILE_ID, PROFILE_NFACTOR_ID, ProjectionEnvelope, ProjectionError as E,
    STATEMENT_ENVIRONMENT_ROOT, SafePoint, admission, checkpoint, decode_hex, encode_hex,
    make_nfactor_receipt, make_receipt, statement_root,
};
use fsym_proof_kernel::capsule::PolyObject;

fn mismatch<T>() -> Result<T, E> {
    Err(E::ProjectionReceiptMismatch)
}

fn vector(text: &mut &str) -> Result<Vec<i64>, E> {
    let rest = text.strip_prefix('[').ok_or(E::ProjectionReceiptMismatch)?;
    let (body, tail) = rest.split_once(']').ok_or(E::ProjectionReceiptMismatch)?;
    if body.is_empty() {
        return mismatch();
    }
    let mut coefficients = Vec::new();
    for token in body.split(", ") {
        if coefficients.len() > MAX_DEGREE as usize {
            return Err(E::ResourceExhausted("statement degree"));
        }
        let value = token
            .parse::<i64>()
            .map_err(|_| E::ProjectionReceiptMismatch)?;
        // Exact grammar, including canonical signs/decimal spelling.
        if token != value.to_string() {
            return mismatch();
        }
        coefficients.push(value);
    }
    if coefficients.last() == Some(&0) {
        return mismatch();
    }
    *text = tail;
    Ok(coefficients)
}

fn vector_list(text: &mut &str, max_vectors: usize) -> Result<Vec<Vec<i64>>, E> {
    let mut rest = text.strip_prefix('[').ok_or(E::ProjectionReceiptMismatch)?;
    let mut vectors = Vec::new();
    loop {
        if vectors.len() >= max_vectors {
            return Err(E::ResourceExhausted("factor count"));
        }
        vectors.push(vector(&mut rest)?);
        if let Some(after_close) = rest.strip_prefix(']') {
            *text = after_close;
            return Ok(vectors);
        }
        rest = rest
            .strip_prefix(", ")
            .ok_or(E::ProjectionReceiptMismatch)?;
    }
}

fn compare_sparse(coefficients: &[i64], poly: &PolyObject) -> Result<(), E> {
    if poly.terms.is_empty() || coefficients.len() != poly.degree() as usize + 1 {
        return mismatch();
    }
    let mut native = poly.terms.iter().rev().peekable();
    for (degree, coefficient) in coefficients.iter().enumerate() {
        let expected = match native.peek() {
            Some((d, _)) if *d as usize == degree => {
                let (_, q) = native.next().ok_or(E::ProjectionReceiptMismatch)?;
                if !q.is_integer() {
                    return mismatch();
                }
                q.numer().to_i64().ok_or(E::ProjectionReceiptMismatch)?
            }
            _ => 0,
        };
        if *coefficient != expected {
            return mismatch();
        }
    }
    if native.next().is_some() {
        return mismatch();
    }
    Ok(())
}

/// Check untrusted transport against an independently supplied native claim
/// root. Native verification is mandatory and offline. The result certifies
/// the projection mapping only, never successful Lean elaboration/checking.
pub fn check_projection(
    envelope: &ProjectionEnvelope,
    expected_claim_root: [u8; 32],
    cancel: &impl Fn(SafePoint) -> bool,
) -> Result<CheckedProjection, E> {
    checkpoint(cancel, SafePoint::Admission)?;
    envelope.check_sizes()?;
    if envelope.checker_pin != CHECKER_PIN
        || envelope.environment != STATEMENT_ENVIRONMENT_ROOT
        || envelope.native_verdict != "verified"
        || envelope.native_claim_root != encode_hex(&expected_claim_root)
    {
        return mismatch();
    }
    if envelope.profile_id == PROFILE_NFACTOR_ID {
        return check_nfactor_projection(envelope, expected_claim_root, cancel);
    }
    if envelope.profile_id != PROFILE_ID {
        return mismatch();
    }
    let bytes = decode_hex(&envelope.capsule_hex, MAX_CAPSULE_BYTES)?;
    let capsule = admission::admit(&bytes, expected_claim_root, cancel)?;
    // No call to dense_ascending or render_statement: parse the output as a
    // restricted language and compare directly to sparse native terms.
    let mut text = envelope
        .lean_source
        .strip_prefix(LEAN_PREFIX)
        .and_then(|s| s.strip_prefix("theorem projected_product :\n    polyMul "))
        .ok_or(E::ProjectionReceiptMismatch)?;
    let a = vector(&mut text)?;
    text = text.strip_prefix(' ').ok_or(E::ProjectionReceiptMismatch)?;
    let b = vector(&mut text)?;
    text = text
        .strip_prefix(" = ")
        .ok_or(E::ProjectionReceiptMismatch)?;
    let subject = vector(&mut text)?;
    if text != " := by decide\n#print axioms projected_product\n" {
        return mismatch();
    }
    let native_subject = capsule
        .polynomial(capsule.claim.subject)
        .map_err(|_| E::MalformedInput)?;
    let native_a = capsule
        .polynomial(capsule.claim.factors[0].0)
        .map_err(|_| E::MalformedInput)?;
    let native_b = capsule
        .polynomial(capsule.claim.factors[1].0)
        .map_err(|_| E::MalformedInput)?;
    if native_a.symbol != native_subject.symbol || native_b.symbol != native_subject.symbol {
        return mismatch();
    }
    compare_sparse(&subject, &native_subject)?;
    compare_sparse(&a, &native_a)?;
    compare_sparse(&b, &native_b)?;
    let root = statement_root(
        STATEMENT_ENVIRONMENT_ROOT,
        PROFILE_ID,
        &envelope.lean_source,
    );
    if envelope.statement_root != encode_hex(&root)
        || envelope.receipt != make_receipt(&bytes, &capsule, &expected_claim_root, &root)
    {
        return mismatch();
    }
    let checked = CheckedProjection::new(envelope.clone());
    checkpoint(cancel, SafePoint::BeforeOutput)?;
    Ok(checked)
}

fn check_nfactor_projection(
    envelope: &ProjectionEnvelope,
    expected_claim_root: [u8; 32],
    cancel: &impl Fn(SafePoint) -> bool,
) -> Result<CheckedProjection, E> {
    let bytes = decode_hex(&envelope.capsule_hex, MAX_CAPSULE_BYTES)?;
    let capsule = admission::admit_nfactor(&bytes, expected_claim_root, cancel)?;
    if !capsule.claim.coefficient.is_integer() {
        return mismatch();
    }
    let expected_scalar = capsule
        .claim
        .coefficient
        .numer()
        .to_i64()
        .ok_or(E::ProjectionReceiptMismatch)?;
    if expected_scalar == 0 {
        return mismatch();
    }
    let rest = envelope
        .lean_source
        .strip_prefix(LEAN_NFACTOR_PREFIX)
        .and_then(|s| s.strip_prefix("theorem projected_product :\n    polyScale ("))
        .ok_or(E::ProjectionReceiptMismatch)?;
    let (scalar_token, mut text) = rest
        .split_once(") (polyProd ")
        .ok_or(E::ProjectionReceiptMismatch)?;
    let parsed_scalar = scalar_token
        .parse::<i64>()
        .map_err(|_| E::ProjectionReceiptMismatch)?;
    if scalar_token != parsed_scalar.to_string() || parsed_scalar != expected_scalar {
        return mismatch();
    }
    let max_expanded = crate::MAX_FACTORS * (crate::MAX_EXPONENT as usize);
    let factors = vector_list(&mut text, max_expanded)?;
    text = text
        .strip_prefix(") = ")
        .ok_or(E::ProjectionReceiptMismatch)?;
    let subject = vector(&mut text)?;
    if text != " := by decide\n#print axioms projected_product\n" {
        return mismatch();
    }
    let native_subject = capsule
        .polynomial(capsule.claim.subject)
        .map_err(|_| E::MalformedInput)?;
    compare_sparse(&subject, &native_subject)?;
    let expected_total: usize = capsule
        .claim
        .factors
        .iter()
        .map(|(_, exp)| *exp as usize)
        .sum();
    if factors.len() != expected_total {
        return mismatch();
    }
    let mut cursor = 0usize;
    for &(factor_id, exp) in &capsule.claim.factors {
        let native_factor = capsule.polynomial(factor_id).map_err(|_| E::MalformedInput)?;
        if native_factor.symbol != native_subject.symbol {
            return mismatch();
        }
        for _ in 0..(exp as usize) {
            compare_sparse(&factors[cursor], &native_factor)?;
            cursor += 1;
        }
    }
    let root = statement_root(
        STATEMENT_ENVIRONMENT_ROOT,
        PROFILE_NFACTOR_ID,
        &envelope.lean_source,
    );
    if envelope.statement_root != encode_hex(&root)
        || envelope.receipt
            != make_nfactor_receipt(
                &bytes,
                &capsule,
                expected_scalar,
                &expected_claim_root,
                &root,
            )
    {
        return mismatch();
    }
    let checked = CheckedProjection::new(envelope.clone());
    checkpoint(cancel, SafePoint::BeforeOutput)?;
    Ok(checked)
}
