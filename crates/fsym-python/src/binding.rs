//! Typed lowering/lifting contract between the Python compatibility shell and
//! the native semantic kernel (WS04).
//!
//! The shell owns Python identity; the kernel owns mathematical identity. A
//! surface symbol therefore crosses the boundary as a [`PySymbolBinding`]: the
//! printed name is a view, while the typed identity is a canonical digest over
//! the declared assumption facts. Two symbols that print identically but were
//! declared differently lower to different native atoms, and no printed string
//! is ever used as a semantic key.

#![forbid(unsafe_code)]

use fsym_core::{Symbol, SymbolIdentity};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Domain separator for the canonical assumptions preimage.
const ASSUMPTIONS_DOMAIN: &[u8] = b"fsym.symbol.assumptions.v1\0";
/// Bound on declared facts per symbol crossing the bridge.
const MAX_ASSUMPTION_FACTS: usize = 64;
/// Bound on a single fact's key/value length in bytes.
const MAX_FACT_BYTES: usize = 256;

/// Canonical digest over declared assumption facts.
///
/// Facts are sorted and de-duplicated, then length-framed so that no two
/// different fact sets share a preimage. The digest depends only on the declared
/// facts: never on insertion order, a session counter, or a surface handle.
fn assumptions_digest(facts: &[(String, String)]) -> [u8; 32] {
    let mut canonical: Vec<&(String, String)> = facts.iter().collect();
    canonical.sort();
    canonical.dedup();
    let mut hasher = blake3::Hasher::new();
    hasher.update(ASSUMPTIONS_DOMAIN);
    for (key, value) in canonical {
        hasher.update(&(key.len() as u64).to_le_bytes());
        hasher.update(key.as_bytes());
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Typed handle for a surface symbol crossing into the native kernel.
#[pyclass(name = "SymbolBinding", module = "fsym_python", frozen)]
#[derive(Clone, Debug)]
pub struct PySymbolBinding {
    #[pyo3(get)]
    pub name: String,
    /// Hex form of the typed identity, for receipts and diagnostics.
    #[pyo3(get)]
    pub identity_hex: String,
    identity: [u8; 32],
    /// Number of distinct declared facts that formed the identity.
    #[pyo3(get)]
    fact_count: usize,
}

impl PySymbolBinding {
    /// The native symbol this binding denotes. A symbol with no declared facts
    /// stays plain, so default-profile behaviour is unchanged.
    pub fn to_symbol(&self) -> Symbol {
        if self.fact_count == 0 {
            Symbol::new(self.name.clone())
        } else {
            Symbol::with_identity(
                self.name.clone(),
                SymbolIdentity::from_assumptions_digest(self.identity),
            )
        }
    }

    /// Declared facts are absent when the binding is plain.
    pub fn is_plain(&self) -> bool {
        self.fact_count == 0
    }

    /// Projects a native symbol's typed identity back onto the surface.
    pub fn from_symbol(symbol: &Symbol) -> Self {
        let digest = symbol
            .identity
            .map(|identity| identity.assumptions)
            .unwrap_or([0u8; 32]);
        Self {
            name: symbol.name.clone(),
            identity_hex: digest.iter().map(|b| format!("{b:02x}")).collect(),
            identity: digest,
            fact_count: usize::from(symbol.identity.is_some()),
        }
    }
}

#[pymethods]
impl PySymbolBinding {
    #[new]
    #[pyo3(signature = (name, assumptions=None))]
    pub(crate) fn new(name: String, assumptions: Option<Vec<(String, String)>>) -> PyResult<Self> {
        if name.is_empty() {
            return Err(PyValueError::new_err("symbol name must not be empty"));
        }
        let facts = assumptions.unwrap_or_default();
        if facts.len() > MAX_ASSUMPTION_FACTS {
            return Err(PyValueError::new_err(format!(
                "at most {MAX_ASSUMPTION_FACTS} assumption facts may bind one symbol"
            )));
        }
        for (key, value) in &facts {
            if key.is_empty() || key.len() > MAX_FACT_BYTES || value.len() > MAX_FACT_BYTES {
                return Err(PyValueError::new_err(format!(
                    "assumption facts must have a non-empty key of at most {MAX_FACT_BYTES} bytes"
                )));
            }
        }
        let digest = assumptions_digest(&facts);
        let mut distinct: Vec<&(String, String)> = facts.iter().collect();
        distinct.sort();
        distinct.dedup();
        Ok(Self {
            name,
            identity_hex: digest.iter().map(|b| format!("{b:02x}")).collect(),
            identity: digest,
            fact_count: distinct.len(),
        })
    }

    /// Whether this binding carries no typed identity.
    #[getter]
    fn plain(&self) -> bool {
        self.fact_count == 0
    }

    /// Rebuild a binding from a serialized typed identity, so a receipt can be
    /// replayed without the original surface session.
    #[staticmethod]
    pub(crate) fn from_identity(name: String, identity_hex: &str) -> PyResult<Self> {
        if identity_hex.len() != 64 {
            return Err(PyValueError::new_err(
                "typed identity must be 64 lowercase hex characters",
            ));
        }
        let mut identity = [0u8; 32];
        for (index, byte) in identity.iter_mut().enumerate() {
            let pair = identity_hex
                .get(index * 2..index * 2 + 2)
                .ok_or_else(|| PyValueError::new_err("typed identity is truncated"))?;
            *byte = u8::from_str_radix(pair, 16)
                .map_err(|_| PyValueError::new_err("typed identity is not lowercase hex"))?;
        }
        Ok(Self {
            name,
            identity_hex: identity_hex.to_string(),
            identity,
            fact_count: 1,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SymbolBinding(name={:?}, facts={}, identity={})",
            self.name,
            self.fact_count,
            &self.identity_hex[..8]
        )
    }
}

/// Extract a typed binding from a Python argument, refusing anything else by
/// name so a caller cannot smuggle a printed string into a semantic slot.
pub fn extract_binding<'py>(value: &Bound<'py, PyAny>) -> PyResult<PyRef<'py, PySymbolBinding>> {
    value.extract::<PyRef<'py, PySymbolBinding>>().map_err(|_| {
        PyValueError::new_err(
            "differentiation requires a typed SymbolBinding for the variable; \
             printed strings are not accepted as semantic identity",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(name: &str, facts: &[(&str, &str)]) -> PySymbolBinding {
        let owned: Vec<(String, String)> = facts
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let digest = assumptions_digest(&owned);
        PySymbolBinding {
            name: name.to_string(),
            identity_hex: digest.iter().map(|b| format!("{b:02x}")).collect(),
            identity: digest,
            fact_count: owned.len(),
        }
    }

    #[test]
    fn identity_depends_only_on_declared_facts() {
        let positive = [("positive", "True")];
        let duplicated = [("positive", "True"), ("positive", "True")];
        assert_eq!(
            assumptions_digest(&positive.map(|(k, v)| (k.to_string(), v.to_string()))),
            assumptions_digest(&duplicated.map(|(k, v)| (k.to_string(), v.to_string())))
        );
        let negative = [("positive", "False")];
        assert_ne!(
            assumptions_digest(&positive.map(|(k, v)| (k.to_string(), v.to_string()))),
            assumptions_digest(&negative.map(|(k, v)| (k.to_string(), v.to_string())))
        );
    }

    #[test]
    fn a_binding_without_facts_stays_a_plain_symbol() {
        let plain = binding("x", &[]);
        assert!(plain.is_plain());
        assert_eq!(plain.to_symbol(), Symbol::new("x"));

        let keyed = binding("x", &[("positive", "True")]);
        assert!(!keyed.is_plain());
        assert_eq!(keyed.to_symbol().name, "x");
        assert!(keyed.to_symbol().identity.is_some());
        assert_ne!(keyed.to_symbol(), Symbol::new("x"));
    }

    #[test]
    fn distinct_declarations_never_share_a_typed_identity() {
        let positive = binding("x", &[("positive", "True")]);
        let nonzero = binding("x", &[("nonzero", "True")]);
        assert_ne!(positive.identity, nonzero.identity);
        assert_ne!(positive.to_symbol(), nonzero.to_symbol());
        // Identity is not derived from the printed name.
        assert_eq!(positive.name, nonzero.name);
    }
}
