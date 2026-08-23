//! Deterministic random stream derivation for FrankenSymPy.
//!
//! Layer: L4 (runtime support). Provides replayable, domain-separated
//! randomness for portfolios, fuzz-style exploration, and randomized
//! algorithms. Streams derive from an explicit 32-byte root seed plus a
//! label path; identical (seed, path) pairs reproduce identical sequences
//! across processes, platforms, and runs.
//!
//! Structural guarantees:
//!
//! - **No hidden entropy.** Sequences depend only on `(root seed, label
//!   path, stream counter)`. Wall-clock time, memory addresses, thread
//!   IDs, and process state are never inputs.
//! - **Domain separation.** Every derivation mixes the full label path
//!   through BLAKE3 with length-prefixed framing, so distinct paths cannot
//!   collide or alias a prefix of another stream.
//! - **Cheap forks.** Deriving a child stream copies 32 bytes; parent and
//!   child streams remain independent.
//! - **No crypto claims.** This is a deterministic counter stream keyed by
//!   BLAKE3, suitable for algorithmic randomness and reproducibility — not
//!   a security-sensitive CSPRNG contract.

#![forbid(unsafe_code)]

use blake3::Hasher;

/// Length of a derived stream key in bytes.
pub const KEY_LEN: usize = 32;

/// A named position in the stream derivation tree.
///
/// Labels are appended verbatim with length-prefix framing; use
/// [`StreamPath::push`] rather than concatenating strings so framing stays
/// unambiguous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamPath {
    segments: Vec<String>,
}

impl StreamPath {
    /// Creates an empty root path.
    pub fn root() -> Self {
        StreamPath {
            segments: Vec::new(),
        }
    }

    /// Creates a path with a single top-level segment.
    pub fn of(segment: impl Into<String>) -> Self {
        StreamPath {
            segments: vec![segment.into()],
        }
    }

    /// Appends a segment, returning the extended path.
    pub fn push(mut self, segment: impl Into<String>) -> Self {
        self.segments.push(segment.into());
        self
    }

    /// The ordered label segments of this path.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    fn frame_into(&self, hasher: &mut Hasher) {
        let count = u64::try_from(self.segments.len()).expect("path length fits u64");
        push_chunk(hasher, &count.to_le_bytes());
        for segment in &self.segments {
            let len = u64::try_from(segment.len()).expect("segment length fits u64");
            push_chunk(hasher, &len.to_le_bytes());
            hasher.update(segment.as_bytes());
        }
    }
}

fn push_chunk(hasher: &mut Hasher, chunk: &[u8]) {
    hasher.update(chunk);
}

/// Domain-separation tag mixed into every key derivation.
const DERIVE_TAG: &[u8] = b"frankensympy.rng.stream.v1";

/// The root of a deterministic random stream tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamRoot {
    key: [u8; KEY_LEN],
}

impl StreamRoot {
    /// Derives a root from caller-supplied seed material.
    ///
    /// The seed is hashed, not used raw, so low-entropy seeds (small
    /// integers, short strings) still produce well-distributed keys.
    pub fn from_seed(seed: &[u8]) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(DERIVE_TAG);
        push_chunk(&mut hasher, seed);
        StreamRoot {
            key: *hasher.finalize().as_bytes(),
        }
    }

    /// Derives the child stream identified by `path` relative to this root.
    pub fn derive(&self, path: &StreamPath) -> DerivedStream {
        let mut hasher = Hasher::new();
        hasher.update(DERIVE_TAG);
        hasher.update(&self.key);
        path.frame_into(&mut hasher);
        DerivedStream {
            key: *hasher.finalize().as_bytes(),
            counter: 0,
        }
    }
}

/// A deterministic, seekable random byte stream.
///
/// Drawing advances an internal 64-bit block counter over a BLAKE3 key
/// stream. The sequence is a pure function of `(key, counter)`; cloning,
/// saving, or restoring the cursor reproduces exact positions.
#[derive(Debug, Clone)]
pub struct DerivedStream {
    key: [u8; KEY_LEN],
    counter: u64,
}

impl DerivedStream {
    /// The derivation path-independent identity of this stream, useful for
    /// receipts and traces.
    pub fn stream_id_hex(&self) -> String {
        blake3::hash(&self.key).to_hex().to_string()
    }

    /// Current block counter (number of 64-bit words already produced).
    pub fn cursor(&self) -> u64 {
        self.counter
    }

    /// Restores an earlier cursor, enabling replay from any point.
    pub fn seek(&mut self, cursor: u64) {
        self.counter = cursor;
    }

    /// Produces the next pseudorandom `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut hasher = Hasher::new();
        hasher.update(DERIVE_TAG);
        hasher.update(&self.key);
        push_chunk(&mut hasher, &self.counter.to_le_bytes());
        self.counter += 1;
        let block = hasher.finalize();
        let bytes = block.as_bytes();
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[..8]);
        u64::from_le_bytes(word)
    }

    /// Produces the next uniform `f64` in `[0, 1)` with 53 bits of
    /// mantissa entropy.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Fills `dest` with pseudorandom bytes.
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let word = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }

    /// Produces a uniformly distributed index in `0..bound`.
    ///
    /// Returns `None` when `bound` is zero. Uses rejection sampling on the
    /// half-open range to avoid modulo bias.
    pub fn below(&mut self, bound: u64) -> Option<u64> {
        if bound == 0 {
            return None;
        }
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let draw = self.next_u64();
            if draw < zone {
                return Some(draw % bound);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_and_path_reproduce_identical_sequences() {
        let path = StreamPath::of("portfolio")
            .push("factor")
            .push("strategy-3");
        let mut a = StreamRoot::from_seed(b"run-42").derive(&path);
        let mut b = StreamRoot::from_seed(b"run-42").derive(&path);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let path = StreamPath::of("portfolio");
        let mut a = StreamRoot::from_seed(b"run-42").derive(&path);
        let mut b = StreamRoot::from_seed(b"run-43").derive(&path);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn paths_are_domain_separated_including_prefixes() {
        let root = StreamRoot::from_seed(b"run-42");
        let mut base = root.derive(&StreamPath::of("a"));
        let mut extended = root.derive(&StreamPath::of("a").push("b"));
        // "a" must never alias a prefix of "a.b".
        assert_ne!(base.next_u64(), extended.next_u64());

        let mut framed = root.derive(&StreamPath::of("ab"));
        base.seek(0);
        extended.seek(0);
        assert_ne!(base.next_u64(), framed.next_u64());
    }

    #[test]
    fn seek_replays_exact_positions() {
        let mut stream = StreamRoot::from_seed(b"replay").derive(&StreamPath::of("lab"));
        let drawn: Vec<u64> = (0..16).map(|_| stream.next_u64()).collect();

        stream.seek(0);
        for &expected in &drawn {
            assert_eq!(stream.next_u64(), expected);
        }

        stream.seek(5);
        assert_eq!(stream.cursor(), 5);
        assert_eq!(stream.next_u64(), drawn[5]);
    }

    #[test]
    fn f64_output_stays_in_unit_interval() {
        let mut stream = StreamRoot::from_seed(b"unit").derive(&StreamPath::of("f64"));
        for _ in 0..10_000 {
            let v = stream.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn below_rejects_zero_and_avoids_modulo_bias_bounds() {
        let mut stream = StreamRoot::from_seed(b"bounds").derive(&StreamPath::of("below"));
        assert_eq!(stream.below(0), None);
        for _ in 0..1_000 {
            assert!(stream.below(7).unwrap() < 7);
        }
    }

    #[test]
    fn fill_bytes_handles_partial_words() {
        let mut stream = StreamRoot::from_seed(b"bytes").derive(&StreamPath::of("fill"));
        let mut buf = [0u8; 21];
        stream.fill_bytes(&mut buf);

        let mut reference = StreamRoot::from_seed(b"bytes").derive(&StreamPath::of("fill"));
        let mut expected = Vec::with_capacity(24);
        for _ in 0..3 {
            expected.extend_from_slice(&reference.next_u64().to_le_bytes());
        }
        assert_eq!(&buf[..], &expected[..21]);
    }

    #[test]
    fn stream_ids_distinguish_distinct_derivations() {
        let root = StreamRoot::from_seed(b"id-test");
        let a = root.derive(&StreamPath::of("x")).stream_id_hex();
        let b = root.derive(&StreamPath::of("y")).stream_id_hex();
        assert_ne!(a, b);
        assert_eq!(a, root.derive(&StreamPath::of("x")).stream_id_hex());
    }
}
