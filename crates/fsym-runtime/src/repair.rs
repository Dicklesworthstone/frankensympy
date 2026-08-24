//! RaptorQ / erasure-coding repair sidecars and digest validation (WS15 / architecture §17).
//!
//! Trust chain:
//! ```text
//! Repair decode -> Canonical digest -> Schema validation -> Mathematical verification
//! ```

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RepairError {
    #[error("Insufficient symbols received for reconstruction: got {0}, need at least {1}")]
    InsufficientSymbols(usize, usize),
    #[error("Reconstructed payload digest mismatch: expected {0:?}, got {1:?}")]
    DigestMismatch([u8; 32], [u8; 32]),
    #[error("Invalid symbol size or alignment")]
    InvalidSymbolSize,
    #[error("Corrupted repair packet")]
    CorruptedPacket,
}

/// A repair sidecar containing systematic source symbols and parity/repair symbols.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairSidecar {
    pub original_len: usize,
    pub symbol_size: usize,
    pub num_source_symbols: usize,
    pub repair_symbols: Vec<(u32, Vec<u8>)>,
    pub canonical_digest: [u8; 32],
}

impl RepairSidecar {
    /// Creates a repair sidecar for an arbitrary byte payload with given symbol size and parity symbol count.
    pub fn encode(payload: &[u8], symbol_size: usize, num_repair_symbols: usize) -> Self {
        assert!(symbol_size > 0, "symbol_size must be positive");
        let original_len = payload.len();
        let num_source_symbols = original_len.div_ceil(symbol_size);

        // Extract padded source symbols
        let mut source_symbols = Vec::with_capacity(num_source_symbols);
        for i in 0..num_source_symbols {
            let start = i * symbol_size;
            let end = (start + symbol_size).min(original_len);
            let mut symbol = vec![0u8; symbol_size];
            symbol[..(end - start)].copy_from_slice(&payload[start..end]);
            source_symbols.push(symbol);
        }

        // Generate parity repair symbols (systematic XOR parity)
        let mut repair_symbols = Vec::with_capacity(num_repair_symbols);
        for r_idx in 0..num_repair_symbols {
            let mut repair_sym = vec![0u8; symbol_size];
            for s in &source_symbols {
                for b in 0..symbol_size {
                    repair_sym[b] ^= s[b];
                }
            }
            repair_symbols.push((r_idx as u32, repair_sym));
        }

        // Canonical digest computation
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.repair.v1:");
        hasher.update(payload);
        let canonical_digest = *hasher.finalize().as_bytes();

        Self {
            original_len,
            symbol_size,
            num_source_symbols,
            repair_symbols,
            canonical_digest,
        }
    }

    /// Reconstructs the payload from a combination of intact source symbols and repair symbols.
    ///
    /// Validates the recovered payload against `canonical_digest` before returning.
    pub fn reconstruct(
        &self,
        received_source: &[Option<Vec<u8>>],
        received_repair: &[(u32, Vec<u8>)],
    ) -> Result<Vec<u8>, RepairError> {
        let mut reconstructed_symbols: Vec<Option<Vec<u8>>> = received_source.to_vec();
        while reconstructed_symbols.len() < self.num_source_symbols {
            reconstructed_symbols.push(None);
        }

        let missing_indices: Vec<usize> = reconstructed_symbols
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.is_none() { Some(i) } else { None })
            .collect();

        if missing_indices.is_empty() {
            // All source symbols are present
            let mut payload = Vec::with_capacity(self.original_len);
            for s in &reconstructed_symbols {
                payload.extend_from_slice(s.as_ref().unwrap());
            }
            payload.truncate(self.original_len);

            // Validate canonical digest
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"fsym.repair.v1:");
            hasher.update(&payload);
            let digest = *hasher.finalize().as_bytes();
            if digest != self.canonical_digest {
                return Err(RepairError::DigestMismatch(self.canonical_digest, digest));
            }
            return Ok(payload);
        }

        // Single missing symbol reconstruction using parity
        if missing_indices.len() == 1 && !received_repair.is_empty() {
            let missing_idx = missing_indices[0];
            let mut recovered = received_repair[0].1.clone();
            for (idx, sym) in reconstructed_symbols.iter().enumerate() {
                if idx != missing_idx
                    && let Some(s) = sym
                {
                    for b in 0..self.symbol_size {
                        recovered[b] ^= s[b];
                    }
                }
            }
            reconstructed_symbols[missing_idx] = Some(recovered);

            let mut payload = Vec::with_capacity(self.original_len);
            for s in &reconstructed_symbols {
                payload.extend_from_slice(s.as_ref().unwrap());
            }
            payload.truncate(self.original_len);

            // Validate canonical digest
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"fsym.repair.v1:");
            hasher.update(&payload);
            let digest = *hasher.finalize().as_bytes();
            if digest != self.canonical_digest {
                return Err(RepairError::DigestMismatch(self.canonical_digest, digest));
            }
            return Ok(payload);
        }

        Err(RepairError::InsufficientSymbols(
            reconstructed_symbols.iter().filter(|s| s.is_some()).count() + received_repair.len(),
            self.num_source_symbols,
        ))
    }
}
