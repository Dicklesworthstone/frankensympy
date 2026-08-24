//! RaptorQ repair sidecars and digest validation (WS15 / architecture §17).
//!
//! This module establishes byte recovery and internal envelope integrity only.
//! Callers must still perform authorization, object-schema validation, dependency
//! validation, and mathematical verification before publishing or resuming work.

#![forbid(unsafe_code)]

use asupersync::raptorq::decoder::{DecodeError, InactivationDecoder, ReceivedSymbol};
use asupersync::raptorq::systematic::SystematicEncoder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

const REPAIR_SCHEMA_VERSION: u16 = 1;
const MAX_PAYLOAD_LEN: usize = 64 * 1024 * 1024;
const MAX_SYMBOL_SIZE: usize = 1024 * 1024;
const MAX_SOURCE_SYMBOLS: usize = 56_403;
const MAX_REPAIR_SYMBOLS: usize = 4_096;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RepairError {
    #[error("payload length {actual} exceeds repair limit {maximum}")]
    PayloadTooLarge { actual: usize, maximum: usize },
    #[error("symbol size {0} is outside the supported range")]
    InvalidSymbolSize(usize),
    #[error("source symbol count {actual} exceeds codec limit {maximum}")]
    TooManySourceSymbols { actual: usize, maximum: usize },
    #[error("repair symbol count {actual} exceeds policy limit {maximum}")]
    TooManyRepairSymbols { actual: usize, maximum: usize },
    #[error("unsupported repair envelope schema version {0}")]
    UnsupportedSchemaVersion(u16),
    #[error("repair envelope manifest is inconsistent: {0}")]
    InvalidManifest(&'static str),
    #[error("RaptorQ encoder could not construct a full-rank source block")]
    EncoderConstructionFailed,
    #[error("received {actual} source slots, expected exactly {expected}")]
    SourceSymbolCountMismatch { actual: usize, expected: usize },
    #[error("source symbol {index} has size {actual}, expected {expected}")]
    SourceSymbolSizeMismatch {
        index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("source symbol {0} failed its manifest digest")]
    SourceSymbolDigestMismatch(usize),
    #[error("repair symbol ESI {esi} has size {actual}, expected {expected}")]
    RepairSymbolSizeMismatch {
        esi: u32,
        actual: usize,
        expected: usize,
    },
    #[error("repair symbol ESI {0} is not declared by this envelope")]
    UnknownRepairSymbol(u32),
    #[error("repair symbol ESI {0} was supplied more than once")]
    DuplicateRepairSymbol(u32),
    #[error("repair symbol ESI {0} failed its manifest digest")]
    RepairSymbolDigestMismatch(u32),
    #[error("insufficient symbols received for reconstruction: got {0}, need at least {1}")]
    InsufficientSymbols(usize, usize),
    #[error("received RaptorQ equations are rank-deficient")]
    InsufficientIndependentSymbols,
    #[error("RaptorQ decoder rejected malformed or corrupted input")]
    DecoderRejectedInput,
    #[error("reconstructed payload digest mismatch: expected {0:?}, got {1:?}")]
    DigestMismatch([u8; 32], [u8; 32]),
}

/// A versioned repair envelope containing RFC 6330 repair symbols.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairSidecar {
    pub schema_version: u16,
    pub original_len: usize,
    pub symbol_size: usize,
    pub num_source_symbols: usize,
    pub codec_seed: u64,
    pub source_symbol_digests: Vec<[u8; 32]>,
    pub repair_symbols: Vec<(u32, Vec<u8>)>,
    pub repair_symbol_digests: Vec<[u8; 32]>,
    pub canonical_digest: [u8; 32],
}

impl RepairSidecar {
    /// Encode `payload` into a bounded, deterministic RFC 6330 repair envelope.
    pub fn encode(
        payload: &[u8],
        symbol_size: usize,
        num_repair_symbols: usize,
    ) -> Result<Self, RepairError> {
        validate_encode_limits(payload.len(), symbol_size, num_repair_symbols)?;

        let num_source_symbols = payload.len().div_ceil(symbol_size).max(1);
        if num_source_symbols > MAX_SOURCE_SYMBOLS {
            return Err(RepairError::TooManySourceSymbols {
                actual: num_source_symbols,
                maximum: MAX_SOURCE_SYMBOLS,
            });
        }

        let source_symbols = split_source_symbols(payload, symbol_size, num_source_symbols);
        let canonical_digest = payload_digest(payload);
        let codec_seed = derive_codec_seed(
            canonical_digest,
            payload.len(),
            symbol_size,
            num_source_symbols,
        );
        let encoder = SystematicEncoder::new(&source_symbols, symbol_size, codec_seed)
            .ok_or(RepairError::EncoderConstructionFailed)?;

        let source_symbol_digests = source_symbols
            .iter()
            .enumerate()
            .map(|(index, symbol)| symbol_digest(b"source", index as u32, symbol))
            .collect();
        let first_repair_esi = u32::try_from(num_source_symbols)
            .map_err(|_| RepairError::InvalidManifest("source count does not fit in ESI"))?;
        let mut repair_symbols = Vec::with_capacity(num_repair_symbols);
        let mut repair_symbol_digests = Vec::with_capacity(num_repair_symbols);
        for offset in 0..num_repair_symbols {
            let offset = u32::try_from(offset)
                .map_err(|_| RepairError::InvalidManifest("repair count does not fit in ESI"))?;
            let esi = first_repair_esi
                .checked_add(offset)
                .ok_or(RepairError::InvalidManifest("repair ESI overflow"))?;
            let symbol = encoder.repair_symbol(esi);
            repair_symbol_digests.push(symbol_digest(b"repair", esi, &symbol));
            repair_symbols.push((esi, symbol));
        }

        Ok(Self {
            schema_version: REPAIR_SCHEMA_VERSION,
            original_len: payload.len(),
            symbol_size,
            num_source_symbols,
            codec_seed,
            source_symbol_digests,
            repair_symbols,
            repair_symbol_digests,
            canonical_digest,
        })
    }

    /// Reconstruct payload bytes from received systematic and repair symbols.
    ///
    /// Success proves only that the RaptorQ decoder recovered bytes matching the
    /// envelope's expected digest. Object-schema and mathematical verification
    /// remain separate caller obligations.
    pub fn reconstruct(
        &self,
        received_source: &[Option<Vec<u8>>],
        received_repair: &[(u32, Vec<u8>)],
    ) -> Result<Vec<u8>, RepairError> {
        self.validate_manifest()?;
        if received_source.len() != self.num_source_symbols {
            return Err(RepairError::SourceSymbolCountMismatch {
                actual: received_source.len(),
                expected: self.num_source_symbols,
            });
        }

        let decoder = InactivationDecoder::try_new(
            self.num_source_symbols,
            self.symbol_size,
            self.codec_seed,
        )
        .map_err(|_| RepairError::InvalidManifest("unsupported RaptorQ parameters"))?;
        let mut decoder_symbols = decoder.constraint_symbols();
        let mut received_count = 0usize;

        for (index, maybe_symbol) in received_source.iter().enumerate() {
            let Some(symbol) = maybe_symbol else {
                continue;
            };
            if symbol.len() != self.symbol_size {
                return Err(RepairError::SourceSymbolSizeMismatch {
                    index,
                    actual: symbol.len(),
                    expected: self.symbol_size,
                });
            }
            if symbol_digest(b"source", index as u32, symbol) != self.source_symbol_digests[index] {
                return Err(RepairError::SourceSymbolDigestMismatch(index));
            }
            decoder_symbols.push(ReceivedSymbol::source(index as u32, symbol.clone()));
            received_count = received_count.saturating_add(1);
        }

        let mut seen_repair_esies = BTreeSet::new();
        for (esi, symbol) in received_repair {
            if !seen_repair_esies.insert(*esi) {
                return Err(RepairError::DuplicateRepairSymbol(*esi));
            }
            let manifest_index = self
                .repair_symbols
                .iter()
                .position(|(declared_esi, _)| declared_esi == esi)
                .ok_or(RepairError::UnknownRepairSymbol(*esi))?;
            if symbol.len() != self.symbol_size {
                return Err(RepairError::RepairSymbolSizeMismatch {
                    esi: *esi,
                    actual: symbol.len(),
                    expected: self.symbol_size,
                });
            }
            if symbol_digest(b"repair", *esi, symbol) != self.repair_symbol_digests[manifest_index]
            {
                return Err(RepairError::RepairSymbolDigestMismatch(*esi));
            }
            let (columns, coefficients) = decoder
                .repair_equation(*esi)
                .map_err(|_| RepairError::DecoderRejectedInput)?;
            decoder_symbols.push(ReceivedSymbol::repair(
                *esi,
                columns,
                coefficients,
                symbol.clone(),
            ));
            received_count = received_count.saturating_add(1);
        }

        if received_count < self.num_source_symbols {
            return Err(RepairError::InsufficientSymbols(
                received_count,
                self.num_source_symbols,
            ));
        }

        let decoded = decoder
            .decode(&decoder_symbols) // ubs:ignore — RFC 6330 decoder, not JWT handling
            .map_err(map_decode_error)?;
        let mut payload = Vec::with_capacity(self.original_len);
        for symbol in decoded.source {
            payload.extend_from_slice(&symbol);
        }
        payload.truncate(self.original_len);

        let actual_digest = payload_digest(&payload);
        if actual_digest != self.canonical_digest {
            return Err(RepairError::DigestMismatch(
                self.canonical_digest,
                actual_digest,
            ));
        }
        Ok(payload)
    }

    fn validate_manifest(&self) -> Result<(), RepairError> {
        if self.schema_version != REPAIR_SCHEMA_VERSION {
            return Err(RepairError::UnsupportedSchemaVersion(self.schema_version));
        }
        validate_encode_limits(
            self.original_len,
            self.symbol_size,
            self.repair_symbols.len(),
        )?;
        let expected_source_symbols = self.original_len.div_ceil(self.symbol_size).max(1);
        if self.num_source_symbols != expected_source_symbols {
            return Err(RepairError::InvalidManifest("source-symbol count mismatch"));
        }
        if self.num_source_symbols > MAX_SOURCE_SYMBOLS {
            return Err(RepairError::TooManySourceSymbols {
                actual: self.num_source_symbols,
                maximum: MAX_SOURCE_SYMBOLS,
            });
        }
        if self.source_symbol_digests.len() != self.num_source_symbols {
            return Err(RepairError::InvalidManifest("source digest count mismatch"));
        }
        if self.repair_symbol_digests.len() != self.repair_symbols.len() {
            return Err(RepairError::InvalidManifest("repair digest count mismatch"));
        }
        if self.codec_seed
            != derive_codec_seed(
                self.canonical_digest,
                self.original_len,
                self.symbol_size,
                self.num_source_symbols,
            )
        {
            return Err(RepairError::InvalidManifest("codec seed mismatch"));
        }

        let first_repair_esi = u32::try_from(self.num_source_symbols)
            .map_err(|_| RepairError::InvalidManifest("source count does not fit in ESI"))?;
        let mut seen = BTreeSet::new();
        for (index, (esi, symbol)) in self.repair_symbols.iter().enumerate() {
            if *esi < first_repair_esi || !seen.insert(*esi) {
                return Err(RepairError::InvalidManifest(
                    "invalid or duplicate repair ESI",
                ));
            }
            if symbol.len() != self.symbol_size {
                return Err(RepairError::RepairSymbolSizeMismatch {
                    esi: *esi,
                    actual: symbol.len(),
                    expected: self.symbol_size,
                });
            }
            if symbol_digest(b"repair", *esi, symbol) != self.repair_symbol_digests[index] {
                return Err(RepairError::RepairSymbolDigestMismatch(*esi));
            }
        }
        Ok(())
    }
}

fn validate_encode_limits(
    payload_len: usize,
    symbol_size: usize,
    num_repair_symbols: usize,
) -> Result<(), RepairError> {
    if payload_len > MAX_PAYLOAD_LEN {
        return Err(RepairError::PayloadTooLarge {
            actual: payload_len,
            maximum: MAX_PAYLOAD_LEN,
        });
    }
    if symbol_size == 0 || symbol_size > MAX_SYMBOL_SIZE {
        return Err(RepairError::InvalidSymbolSize(symbol_size));
    }
    if num_repair_symbols > MAX_REPAIR_SYMBOLS {
        return Err(RepairError::TooManyRepairSymbols {
            actual: num_repair_symbols,
            maximum: MAX_REPAIR_SYMBOLS,
        });
    }
    Ok(())
}

fn split_source_symbols(
    payload: &[u8],
    symbol_size: usize,
    num_source_symbols: usize,
) -> Vec<Vec<u8>> {
    let mut symbols = Vec::with_capacity(num_source_symbols);
    for index in 0..num_source_symbols {
        let start = index.saturating_mul(symbol_size);
        let end = start.saturating_add(symbol_size).min(payload.len());
        let mut symbol = vec![0u8; symbol_size];
        if start < end {
            symbol[..end - start].copy_from_slice(&payload[start..end]);
        }
        symbols.push(symbol);
    }
    symbols
}

fn payload_digest(payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.repair.payload.v1:");
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn symbol_digest(kind: &[u8], esi: u32, symbol: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.repair.symbol.v1:");
    hasher.update(kind);
    hasher.update(&esi.to_le_bytes());
    hasher.update(&(symbol.len() as u64).to_le_bytes());
    hasher.update(symbol);
    *hasher.finalize().as_bytes()
}

fn derive_codec_seed(
    canonical_digest: [u8; 32],
    original_len: usize,
    symbol_size: usize,
    num_source_symbols: usize,
) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.raptorq.seed.v1:");
    hasher.update(&canonical_digest);
    hasher.update(&(original_len as u64).to_le_bytes());
    hasher.update(&(symbol_size as u64).to_le_bytes());
    hasher.update(&(num_source_symbols as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut seed = [0u8; 8];
    seed.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(seed)
}

fn map_decode_error(error: DecodeError) -> RepairError {
    match error {
        DecodeError::InsufficientSymbols { received, required } => {
            RepairError::InsufficientSymbols(received, required)
        }
        DecodeError::SingularMatrix { .. } => RepairError::InsufficientIndependentSymbols,
        DecodeError::SymbolSizeMismatch { .. }
        | DecodeError::SymbolEquationArityMismatch { .. }
        | DecodeError::ColumnIndexOutOfRange { .. }
        | DecodeError::SourceEsiOutOfRange { .. }
        | DecodeError::InvalidSourceSymbolEquation { .. }
        | DecodeError::CorruptDecodedOutput { .. }
        | DecodeError::ComputeBudgetExhausted { .. }
        | DecodeError::EsiRateLimitExceeded { .. } => RepairError::DecoderRejectedInput,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_slots(payload: &[u8], sidecar: &RepairSidecar) -> Vec<Option<Vec<u8>>> {
        split_source_symbols(payload, sidecar.symbol_size, sidecar.num_source_symbols)
            .into_iter()
            .map(Some)
            .collect()
    }

    #[test]
    fn raptorq_recovers_multiple_loss_patterns() {
        let payload: Vec<u8> = (0..511).map(|i| ((i * 37 + 11) % 251) as u8).collect();
        let sidecar = RepairSidecar::encode(&payload, 32, 12).unwrap();

        for missing in [
            vec![],
            vec![0],
            vec![1, 7],
            vec![0, 5, 10],
            vec![2, 6, 11, 15],
        ] {
            let mut sources = source_slots(&payload, &sidecar);
            for index in missing {
                sources[index] = None;
            }
            let recovered = sidecar
                .reconstruct(&sources, &sidecar.repair_symbols)
                .unwrap();
            assert_eq!(recovered, payload);
        }
    }

    #[test]
    fn insufficient_repair_symbols_fail_typed() {
        let payload = vec![0xA5; 256];
        let sidecar = RepairSidecar::encode(&payload, 32, 4).unwrap();
        let mut sources = source_slots(&payload, &sidecar);
        sources[0] = None;
        sources[2] = None;
        sources[5] = None;

        let result = sidecar.reconstruct(&sources, &sidecar.repair_symbols[..2]);
        assert_eq!(result, Err(RepairError::InsufficientSymbols(7, 8)));
    }

    #[test]
    fn corrupt_source_and_repair_packets_are_rejected_before_codec_entry() {
        let payload = vec![0x3C; 192];
        let sidecar = RepairSidecar::encode(&payload, 32, 6).unwrap();

        let mut corrupt_source = source_slots(&payload, &sidecar);
        corrupt_source[0].as_mut().unwrap()[0] ^= 1;
        assert_eq!(
            sidecar.reconstruct(&corrupt_source, &sidecar.repair_symbols),
            Err(RepairError::SourceSymbolDigestMismatch(0))
        );

        let mut missing_source = source_slots(&payload, &sidecar);
        missing_source[0] = None;
        let mut corrupt_repair = sidecar.repair_symbols.clone();
        corrupt_repair[0].1[0] ^= 1;
        assert_eq!(
            sidecar.reconstruct(&missing_source, &corrupt_repair),
            Err(RepairError::RepairSymbolDigestMismatch(corrupt_repair[0].0))
        );
    }

    #[test]
    fn unknown_duplicate_and_malformed_packets_fail_closed() {
        let payload = vec![0x17; 96];
        let sidecar = RepairSidecar::encode(&payload, 16, 4).unwrap();
        let mut sources = source_slots(&payload, &sidecar);
        sources[0] = None;

        let mut duplicate = sidecar.repair_symbols[..1].to_vec();
        duplicate.push(duplicate[0].clone());
        assert_eq!(
            sidecar.reconstruct(&sources, &duplicate),
            Err(RepairError::DuplicateRepairSymbol(duplicate[0].0))
        );

        let unknown = vec![(999_999, vec![0u8; sidecar.symbol_size])];
        assert_eq!(
            sidecar.reconstruct(&sources, &unknown),
            Err(RepairError::UnknownRepairSymbol(999_999))
        );

        let mut bad_manifest = sidecar.clone();
        bad_manifest.schema_version = REPAIR_SCHEMA_VERSION + 1;
        assert_eq!(
            bad_manifest.reconstruct(&sources, &[]),
            Err(RepairError::UnsupportedSchemaVersion(
                REPAIR_SCHEMA_VERSION + 1
            ))
        );
    }

    #[test]
    fn empty_payload_round_trips_and_invalid_sizes_do_not_panic() {
        let sidecar = RepairSidecar::encode(&[], 8, 2).unwrap();
        let sources = vec![Some(vec![0u8; 8])];
        assert_eq!(
            sidecar.reconstruct(&sources, &sidecar.repair_symbols),
            Ok(Vec::new())
        );
        assert_eq!(
            RepairSidecar::encode(&[], 0, 0),
            Err(RepairError::InvalidSymbolSize(0))
        );
    }
}
