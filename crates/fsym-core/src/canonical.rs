//! Bounded canonical serialization for exact numeric leaves (WS03).
//!
//! Wire format (little-endian throughout):
//! - `Integer`:  tag `b'I'`, sign byte (`0` positive/zero, `1` negative),
//!   u64 magnitude length, magnitude as minimal unsigned LE bytes. Zero
//!   alone uses an empty magnitude; every nonzero magnitude ends nonzero.
//! - `Rational`: tag `b'Q'`, then two Integer payloads (numer, denom).
//!   Decoded denominators must be positive and reduced; encoders emit
//!   reduced rationals by construction.
//!
//! Decoding fails closed *before* allocation on any declared-size
//! violation: a hostile length field can never cause an oversized buffer
//! because the cap is checked against remaining input bytes first.

use crate::CoreError;
use fsym_bigint::{BigInt, BigRational};

/// Maximum accepted payload size per value, including tag and integer
/// headers. The whole input is rejected before any big-integer allocation
/// when it exceeds this limit.
pub const MAX_SERIALIZED_BYTES: usize = 1024 * 1024;

const INTEGER_HEADER_BYTES: usize = 1 + std::mem::size_of::<u64>();

fn magnitude_len(n: &BigInt) -> Result<usize, CoreError> {
    usize::try_from(n.bits().div_ceil(8)).map_err(|_| {
        CoreError::InvalidOperation("canonical integer: magnitude length overflow".into())
    })
}

fn encoded_integer_len(n: &BigInt) -> Result<usize, CoreError> {
    INTEGER_HEADER_BYTES
        .checked_add(magnitude_len(n)?)
        .ok_or_else(|| CoreError::InvalidOperation("canonical integer: length overflow".into()))
}

fn push_integer(out: &mut Vec<u8>, n: &BigInt) {
    let sign_byte = if n.is_negative() { 1u8 } else { 0u8 };
    out.push(sign_byte);
    let magnitude_bytes = if n.is_zero() {
        Vec::new()
    } else {
        n.to_bytes_le()
    };
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

    let length_end = offset
        .checked_add(std::mem::size_of::<u64>())
        .ok_or_else(|| {
            CoreError::InvalidOperation("canonical integer: length offset overflow".into())
        })?;
    if buf.len() < length_end {
        return Err(CoreError::InvalidOperation(
            "canonical integer: truncated length".into(),
        ));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buf[*offset..length_end]);
    let declared_len = u64::from_le_bytes(len_bytes);
    let len = usize::try_from(declared_len).map_err(|_| {
        CoreError::InvalidOperation(format!(
            "canonical integer: declared magnitude {declared_len} bytes exceeds platform bounds"
        ))
    })?;
    *offset = length_end;

    // Fail closed BEFORE allocation when the declared size is impossible.
    if len > MAX_SERIALIZED_BYTES || len > buf.len() - *offset {
        return Err(CoreError::InvalidOperation(format!(
            "canonical integer: declared magnitude {len} bytes exceeds bounds"
        )));
    }
    let magnitude_end = offset.checked_add(len).ok_or_else(|| {
        CoreError::InvalidOperation("canonical integer: magnitude offset overflow".into())
    })?;
    let magnitude_bytes = &buf[*offset..magnitude_end];
    if magnitude_bytes.last() == Some(&0) {
        return Err(CoreError::InvalidOperation(
            "canonical integer: redundant high zero byte".into(),
        ));
    }
    if sign_byte == 1 && magnitude_bytes.is_empty() {
        return Err(CoreError::InvalidOperation(
            "canonical integer: negative zero is not canonical".into(),
        ));
    }
    let magnitude = BigInt::from_bytes_le(magnitude_bytes);
    *offset = magnitude_end;

    Ok(match sign_byte {
        0 => magnitude,
        _ => -magnitude,
    })
}

impl crate::Expr {
    /// Serializes an exact numeric leaf (`Integer` or `Rational`) into the
    /// canonical bounded wire form. Non-numeric expressions are rejected.
    pub fn to_canonical_numeric_bytes(&self) -> Result<Vec<u8>, CoreError> {
        let encoded_len = match self {
            crate::Expr::Integer(n) => 1usize
                .checked_add(encoded_integer_len(n)?)
                .ok_or_else(|| CoreError::InvalidOperation("serialized length overflow".into()))?,
            crate::Expr::Rational(q) => {
                let numer_len = encoded_integer_len(q.numer())?;
                let denom_len = encoded_integer_len(q.denom())?;
                1usize
                    .checked_add(numer_len)
                    .and_then(|len| len.checked_add(denom_len))
                    .ok_or_else(|| {
                        CoreError::InvalidOperation("serialized length overflow".into())
                    })?
            }
            _ => {
                return Err(CoreError::InvalidOperation(
                    "only Integer/Rational serialize numerically".into(),
                ));
            }
        };
        if encoded_len > MAX_SERIALIZED_BYTES {
            return Err(CoreError::InvalidOperation("serialized too large".into()));
        }

        let mut out = Vec::with_capacity(encoded_len);
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
        debug_assert_eq!(out.len(), encoded_len);
        Ok(out)
    }

    /// Inverse of [`Self::to_canonical_numeric_bytes`]. All size checks
    /// happen before allocation.
    pub fn from_canonical_numeric_bytes(buf: &[u8]) -> Result<Self, CoreError> {
        if buf.len() > MAX_SERIALIZED_BYTES {
            return Err(CoreError::InvalidOperation(format!(
                "canonical numeric payload: {} bytes exceeds {MAX_SERIALIZED_BYTES}-byte bound",
                buf.len()
            )));
        }
        match buf.first() {
            Some(b'I') => {
                let mut offset = 1usize;
                let integer = read_integer(buf, &mut offset)?;
                require_end(buf, offset)?;
                Ok(crate::Expr::Integer(integer))
            }
            Some(b'Q') => {
                let mut offset = 1usize;
                let numer = read_integer(buf, &mut offset)?;
                let denom = read_integer(buf, &mut offset)?;
                require_end(buf, offset)?;
                if denom.is_zero() || denom.is_negative() {
                    return Err(CoreError::InvalidOperation(
                        "canonical rational: non-positive denominator".into(),
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

fn require_end(buf: &[u8], offset: usize) -> Result<(), CoreError> {
    if offset == buf.len() {
        Ok(())
    } else {
        Err(CoreError::InvalidOperation(format!(
            "canonical numeric payload: {} trailing bytes",
            buf.len() - offset
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expr;
    use proptest::prelude::*;

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

        assert_eq!(
            Expr::from_i64(0).to_canonical_numeric_bytes().unwrap(),
            [b'I', 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "zero has one canonical empty-magnitude representation"
        );
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

        let oversized = vec![0u8; MAX_SERIALIZED_BYTES + 1];
        let err = Expr::from_canonical_numeric_bytes(&oversized).unwrap_err();
        assert!(err.to_string().contains("exceeds"));
    }

    #[test]
    fn non_canonical_integer_encodings_fail_closed() {
        let negative_zero = [b'I', 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = Expr::from_canonical_numeric_bytes(&negative_zero).unwrap_err();
        assert!(err.to_string().contains("negative zero"));

        let redundant_high_zero = [b'I', 0, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0];
        let err = Expr::from_canonical_numeric_bytes(&redundant_high_zero).unwrap_err();
        assert!(err.to_string().contains("redundant"));
    }

    #[test]
    fn trailing_bytes_fail_closed_for_every_numeric_tag() {
        for value in [Expr::from_i64(7), Expr::rational(3, 5).unwrap()] {
            let mut bytes = value.to_canonical_numeric_bytes().unwrap();
            bytes.push(0xaa);
            let err = Expr::from_canonical_numeric_bytes(&bytes).unwrap_err();
            assert!(err.to_string().contains("trailing"));
        }
    }

    #[test]
    fn malformed_rationals_are_rejected() {
        let value = Expr::rational(3, 5).unwrap();
        let full = value.to_canonical_numeric_bytes().unwrap();
        for cut in 1..full.len() {
            assert!(Expr::from_canonical_numeric_bytes(&full[..cut]).is_err());
        }

        // Q(2, 4) is mathematically meaningful but not a canonical wire value.
        let unreduced = [
            b'Q', 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4,
        ];
        let err = Expr::from_canonical_numeric_bytes(&unreduced).unwrap_err();
        assert!(err.to_string().contains("not in reduced"));
    }

    #[test]
    fn serialized_size_limit_is_an_exact_boundary() {
        let max_magnitude_len = MAX_SERIALIZED_BYTES - 1 - INTEGER_HEADER_BYTES;
        let at_limit = Expr::Integer(BigInt::from_bytes_le(&vec![0xff; max_magnitude_len]));
        let bytes = at_limit.to_canonical_numeric_bytes().unwrap();
        assert_eq!(bytes.len(), MAX_SERIALIZED_BYTES);
        assert_eq!(
            Expr::from_canonical_numeric_bytes(&bytes).unwrap(),
            at_limit
        );

        let over_limit = Expr::Integer(BigInt::from_bytes_le(&vec![0xff; max_magnitude_len + 1]));
        let err = over_limit.to_canonical_numeric_bytes().unwrap_err();
        assert!(err.to_string().contains("too large"));
    }

    proptest! {
        #[test]
        fn arbitrary_big_integers_round_trip(
            magnitude in prop::collection::vec(any::<u8>(), 0..8192),
            negative in any::<bool>(),
        ) {
            let mut n = BigInt::from_bytes_le(&magnitude);
            if negative {
                n = -n;
            }
            let value = Expr::Integer(n);
            let bytes = value.to_canonical_numeric_bytes().unwrap();
            prop_assert_eq!(Expr::from_canonical_numeric_bytes(&bytes).unwrap(), value);
        }

        #[test]
        fn every_accepted_short_payload_has_a_unique_encoding(
            bytes in prop::collection::vec(any::<u8>(), 0..512),
        ) {
            if let Ok(value) = Expr::from_canonical_numeric_bytes(&bytes) {
                prop_assert_eq!(value.to_canonical_numeric_bytes().unwrap(), bytes);
            }
        }
    }

    #[test]
    fn non_numeric_leaves_are_rejected() {
        let x = Expr::symbol("x");
        assert!(x.to_canonical_numeric_bytes().is_err());
    }
}
