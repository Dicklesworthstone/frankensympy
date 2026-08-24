//! Bounded canonical serialization for exact numeric leaves (WS03).
//!
//! Wire format (little-endian throughout):
//! - `Integer`:  tag `b'I'`, sign byte (`0` positive/zero, `1` negative),
//!   u64 magnitude length, magnitude as unsigned LE bytes.
//! - `Rational`: tag `b'Q'`, then two Integer payloads (numer, denom).
//!   Decoded denominators must be positive and reduced; encoders emit
//!   reduced rationals by construction.
//!
//! Decoding fails closed *before* allocation on any declared-size
//! violation: a hostile length field can never cause an oversized buffer
//! because the cap is checked against remaining input bytes first.

use crate::CoreError;
use num_bigint::{BigInt, BigUint};
use num_rational::BigRational;
use num_traits::Zero;

/// Maximum accepted payload size per value: 1 MiB of magnitude bytes is
/// far beyond any legitimate CAS leaf while keeping hostile inputs cheap
/// to reject.
pub const MAX_SERIALIZED_BYTES: usize = 1024 * 1024;

fn push_integer(out: &mut Vec<u8>, n: &BigInt) {
    let sign_byte = if n.sign() == num_bigint::Sign::Minus {
        1u8
    } else {
        0u8
    };
    // No per-integer tag: the enclosing value tag (b'I'/b'Q') already
    // fixes the layout, and readers consume [sign][len][magnitude].
    out.push(sign_byte);
    let magnitude_bytes = n.magnitude().to_bytes_le();
    out.extend_from_slice(&(magnitude_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&magnitude_bytes);
}

fn read_integer(buf: &[u8], offset: &mut usize) -> Result<BigInt, CoreError> {
    // Sign byte first, matching the writer.
    let Some(sign_byte) = buf.get(*offset).copied() else {
        return Err(CoreError::InvalidOperation(
            "canonical integer: missing sign byte".into(),
        ));
    };
    if sign_byte > 1 {
        return Err(CoreError::InvalidOperation(
            "canonical integer: malformed sign byte".into(),
        ));
    }
    *offset += 1;

    if buf.len() < *offset + 8 {
        return Err(CoreError::InvalidOperation(
            "canonical integer: truncated length".into(),
        ));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buf[*offset..*offset + 8]);
    let len = u64::from_le_bytes(len_bytes) as usize;
    *offset += 8;

    // Fail closed BEFORE allocation when the declared size is impossible.
    if len > MAX_SERIALIZED_BYTES || len > buf.len() - *offset {
        return Err(CoreError::InvalidOperation(format!(
            "canonical integer: declared magnitude {len} bytes exceeds bounds"
        )));
    }
    let magnitude = BigUint::from_bytes_le(&buf[*offset..*offset + len]);
    *offset += len;

    Ok(match sign_byte {
        0 => BigInt::from(magnitude),
        _ => BigInt::from(magnitude) * -1,
    })
}

impl crate::Expr {
    /// Serializes an exact numeric leaf (`Integer` or `Rational`) into the
    /// canonical bounded wire form. Non-numeric expressions are rejected.
    pub fn to_canonical_numeric_bytes(&self) -> Result<Vec<u8>, CoreError> {
        let mut out = Vec::new();
        match self {
            crate::Expr::Integer(n) => {
                out.push(b'I');
                push_integer(&mut out, n);
            }
            crate::Expr::Rational(q) => {
                out.push(b'Q');
                push_integer(&mut out, q.numer());
                push_integer(&mut out, q.denom());
            }
            _ => {
                return Err(CoreError::InvalidOperation(
                    "only Integer/Rational serialize numerically".into(),
                ));
            }
        }
        if out.len() > MAX_SERIALIZED_BYTES {
            return Err(CoreError::InvalidOperation("serialized too large".into()));
        }
        Ok(out)
    }

    /// Inverse of [`Self::to_canonical_numeric_bytes`]. All size checks
    /// happen before allocation.
    pub fn from_canonical_numeric_bytes(buf: &[u8]) -> Result<Self, CoreError> {
        match buf.first() {
            Some(b'I') => {
                let mut offset = 1usize;
                Ok(crate::Expr::Integer(read_integer(buf, &mut offset)?))
            }
            Some(b'Q') => {
                let mut offset = 1usize;
                let numer = read_integer(buf, &mut offset)?;
                let denom = read_integer(buf, &mut offset)?;
                if denom.is_zero() || denom.sign() == num_bigint::Sign::Minus {
                    return Err(CoreError::InvalidOperation(
                        "canonical rational: denominator must be positive".into(),
                    ));
                }
                let canonical = BigRational::new(numer.clone(), denom.clone());
                if canonical.numer() != &numer || canonical.denom() != &denom {
                    return Err(CoreError::InvalidOperation(
                        "canonical rational: not in reduced positive-denominator form".into(),
                    ));
                }
                Ok(crate::Expr::Rational(canonical))
            }
            other => Err(CoreError::InvalidOperation(format!(
                "unknown numeric tag {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expr;

    #[test]
    fn integer_round_trip_and_sign_preservation() {
        for raw in [0i64, 1, -1, 255, -255, i64::MAX, i64::MIN] {
            let e = Expr::from_i64(raw);
            let bytes = e.to_canonical_numeric_bytes().unwrap();
            assert_eq!(bytes[0], b'I', "value tag for {raw}");
            let expected_sign = if raw < 0 { 1u8 } else { 0u8 };
            assert_eq!(bytes[1], expected_sign, "sign byte for {raw}");
            assert_eq!(
                Expr::from_canonical_numeric_bytes(&bytes).unwrap(),
                e,
                "round trip failed for {raw}"
            );
        }
    }

    #[test]
    fn rational_round_trip_requires_reduced_form() {
        let q = BigRational::new(BigInt::from(-6i32), BigInt::from(4i32));
        // Constructor reduces: -6/4 -> -3/2.
        let e = Expr::Rational(q);
        let bytes = e.to_canonical_numeric_bytes().unwrap();
        assert_eq!(Expr::from_canonical_numeric_bytes(&bytes).unwrap(), e);
    }

    #[test]
    fn hostile_lengths_fail_before_allocation() {
        // Valid single-byte integer, then tamper the declared magnitude
        // length to an absurd value. Decode must reject cheaply.
        let mut bytes = Expr::from_i64(7).to_canonical_numeric_bytes().unwrap();
        // Layout: tag(1) sign(1) len(8) mag(1). Overwrite len with 2^40.
        bytes[2..10].copy_from_slice(&(1u64 << 40).to_le_bytes());
        let err = Expr::from_canonical_numeric_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("exceeds bounds"));

        // Truncated buffers fail at every prefix length.
        let full = Expr::from_i64(-123456)
            .to_canonical_numeric_bytes()
            .unwrap();
        for cut in 1..full.len() {
            assert!(Expr::from_canonical_numeric_bytes(&full[..cut]).is_err());
        }
    }

    #[test]
    fn non_numeric_leaves_are_rejected() {
        let x = Expr::symbol("x");
        assert!(x.to_canonical_numeric_bytes().is_err());
    }
}
