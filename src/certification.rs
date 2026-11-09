//! Source-bound real residual enclosures, complete root isolation and portable
//! root-box witnesses.
//!
//! The public problem retains typed sources and ordered symbol identities.
//! Numerical acceptance always recomputes the original source with the core
//! interval checker. No symbolic derivative, compiled evaluator, or serialized
//! acceptance flag supplies mathematical evidence.

#![forbid(unsafe_code)]

use fsym_budget::{BudgetMeter, Dimension, MeterError};
use fsym_calculus::root_isolation::{RootIsolationOptions, propose_root_isolation};
use fsym_core::{
    Expr, MAX_ROOT_BOX_DEPTH, MAX_ROOT_BOX_JACOBIAN_ENTRIES, MAX_ROOT_BOX_LIMBS,
    MAX_ROOT_BOX_NODES, MAX_ROOT_BOX_RESIDUALS, MAX_ROOT_BOX_VARIABLES, RealBall,
    ResidualJacobianEnclosure, RootBoxError, RootBoxReceipt, RootBoxWitness, RootIsolationError,
    RootIsolationReceipt, RootIsolationWitness, Symbol, metered_enclose_residual_jacobian,
    parse_raw, verify_root_box, verify_root_isolation,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::io::{self, Write};
use std::mem::size_of;
use thiserror::Error;

/// Maximum UTF-8 bytes in one textual residual passed to the convenience parser.
pub const MAX_CERTIFICATION_SOURCE_BYTES: usize = 4 * 1024;
/// The root checker's symbol/function payload boundary.
pub const MAX_CERTIFICATION_NAME_BYTES: usize = 4096;
/// Maximum encoded root witness, including all typed source and universe fields.
pub const MAX_ROOT_CERTIFICATE_BYTES: usize = 1024 * 1024;
/// Reject nesting before the JSON decoder allocates or constructs a witness.
pub const MAX_ROOT_CERTIFICATE_JSON_DEPTH: usize = 96;
/// Version of the canonical, proof-replayed root-witness transport.
pub const ROOT_CERTIFICATE_SCHEMA_VERSION: u32 = 1;

/// A refusal never publishes a partial problem, enclosure, receipt, or artifact.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CertificationError {
    #[error("original-source verification refused: {0}")]
    Root(#[source] RootBoxError),
    #[error("complete root isolation refused: {0}")]
    Isolation(#[source] RootIsolationError),
    #[error("certification work stopped: {0}")]
    Meter(#[from] MeterError),
    #[error("certification source limit exceeded: {0}")]
    SourceLimit(&'static str),
    #[error("residual source {index} failed to parse")]
    Parse { index: usize },
    #[error("a parsed variable must be a bounded nonconstant identifier")]
    InvalidVariable,
    #[error("root certificate size {actual} exceeds maximum {max}")]
    ByteLimit { actual: usize, max: usize },
    #[error("root certificate JSON depth {depth} exceeds maximum {max}")]
    DepthLimit { depth: usize, max: usize },
    #[error("unsupported root certificate schema {found}; expected {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("invalid root certificate encoding: {0}")]
    Encoding(String),
    #[error("root certificate is not the canonical typed encoding")]
    NonCanonical,
    #[error(
        "root certificate sources or ordered symbol identities differ from the expected problem"
    )]
    SourceMismatch,
    #[error("root isolation artifact domain differs from the expected domain")]
    DomainMismatch,
    #[error("root certificate assumption universe differs from the expected universe")]
    UniverseMismatch { found: [u8; 32], expected: [u8; 32] },
    #[error("certification allocation refused")]
    AllocationFailure,
}

impl From<RootBoxError> for CertificationError {
    fn from(error: RootBoxError) -> Self {
        match error {
            RootBoxError::Meter(error) => Self::Meter(error),
            error => Self::Root(error),
        }
    }
}

impl From<RootIsolationError> for CertificationError {
    fn from(error: RootIsolationError) -> Self {
        match error {
            RootIsolationError::Verification(error) => error.into(),
            error => Self::Isolation(error),
        }
    }
}

pub(crate) fn charge(
    meter: &mut impl BudgetMeter,
    steps: u64,
    bytes: u64,
    allocations: u64,
) -> Result<(), CertificationError> {
    meter.checkpoint()?;
    let charges = [
        (Dimension::ComputeSteps, steps),
        (Dimension::MemoryBytes, bytes),
        (Dimension::AllocationCount, allocations),
    ];
    let mut nonzero = [(Dimension::ComputeSteps, 1); 3];
    let mut length = 0;
    for entry in charges {
        if entry.1 != 0 {
            nonzero[length] = entry;
            length += 1;
        }
    }
    if length != 0 {
        meter.charge_batch(&nonzero[..length])?;
    }
    Ok(())
}

pub(crate) fn encoding_error(error: &serde_json::Error) -> CertificationError {
    // serde diagnostics may quote an entire untrusted field or payload. Keep
    // protocol/library diagnostics independently bounded and source-private.
    CertificationError::Encoding(format!(
        "typed JSON rejected at line {}, column {}",
        error.line(),
        error.column()
    ))
}

fn check_dimensions(rows: usize, columns: usize) -> Result<(), CertificationError> {
    if !(1..=MAX_ROOT_BOX_RESIDUALS).contains(&rows)
        || !(1..=MAX_ROOT_BOX_VARIABLES).contains(&columns)
        || rows * columns > MAX_ROOT_BOX_JACOBIAN_ENTRIES
    {
        return Err(RootBoxError::DimensionMismatch.into());
    }
    Ok(())
}

fn check_source(
    source: &Expr,
    depth: usize,
    visited: &mut usize,
    meter: &mut impl BudgetMeter,
) -> Result<(), CertificationError> {
    charge(meter, 1, 0, 0)?;
    if depth > MAX_ROOT_BOX_DEPTH || *visited >= MAX_ROOT_BOX_NODES {
        return Err(RootBoxError::ResourceLimit("expression depth or node count").into());
    }
    *visited += 1;
    let payload_bytes = match source {
        Expr::Integer(value) => {
            if value.limb_count() > MAX_ROOT_BOX_LIMBS {
                return Err(RootBoxError::ResourceLimit("integer height").into());
            }
            value.limb_count() * 16
        }
        Expr::Rational(value) => {
            if value.numer().limb_count() > MAX_ROOT_BOX_LIMBS
                || value.denom().limb_count() > MAX_ROOT_BOX_LIMBS
            {
                return Err(RootBoxError::ResourceLimit("rational height").into());
            }
            (value.numer().limb_count() + value.denom().limb_count()) * 16
        }
        Expr::Sym(symbol) => {
            if symbol.name.len() > MAX_CERTIFICATION_NAME_BYTES {
                return Err(RootBoxError::ResourceLimit("symbol length").into());
            }
            symbol.name.len() as u64
        }
        Expr::Function(name, _) => {
            if name.len() > MAX_CERTIFICATION_NAME_BYTES {
                return Err(RootBoxError::ResourceLimit("function name length").into());
            }
            name.len() as u64
        }
        Expr::Const(_) | Expr::Add(_) | Expr::Mul(_) | Expr::Pow(_, _) => 0,
    };
    // Charge retained source/payload ownership as well as the bounded walk.
    // These are logical allocation costs, not a global allocator measurement.
    charge(meter, 1, 128 + payload_bytes, 1)?;
    match source {
        Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => {
            if children.len() > MAX_ROOT_BOX_NODES.saturating_sub(*visited) {
                return Err(RootBoxError::ResourceLimit("variadic expression width").into());
            }
            for child in children {
                check_source(child, depth + 1, visited, meter)?;
            }
        }
        Expr::Pow(base, exponent) => {
            if MAX_ROOT_BOX_NODES.saturating_sub(*visited) < 2 {
                return Err(RootBoxError::ResourceLimit("expression node count").into());
            }
            check_source(base, depth + 1, visited, meter)?;
            check_source(exponent, depth + 1, visited, meter)?;
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn check_problem_sources(
    residuals: &[Expr],
    variables: &[Symbol],
    meter: &mut impl BudgetMeter,
) -> Result<(), CertificationError> {
    meter.checkpoint()?;
    check_dimensions(residuals.len(), variables.len())?;
    for (index, variable) in variables.iter().enumerate() {
        if variable.name.len() > MAX_CERTIFICATION_NAME_BYTES {
            return Err(RootBoxError::ResourceLimit("symbol length").into());
        }
        charge(
            meter,
            (index + 1) as u64,
            128 + variable.name.len() as u64,
            1,
        )?;
        if variables[..index].contains(variable) {
            return Err(RootBoxError::DuplicateVariable.into());
        }
    }
    let mut visited = 0;
    for source in residuals {
        check_source(source, 0, &mut visited, meter)?;
    }
    Ok(())
}

/// Immutable typed source for a 1..64 by 1..16 real residual/Jacobian problem.
///
/// Construction checks the source envelope; it does not prove the source is
/// defined or C1 on any domain. Those obligations are checked independently
/// by [`Self::enclose`] and [`Self::verify_root`]. There is no unchecked decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualProblem {
    residuals: Vec<Expr>,
    variables: Vec<Symbol>,
}

impl ResidualProblem {
    /// Take ownership of exact sources and their explicit coordinate order.
    /// No workspace substitution, simplification, or symbol-name rebinding occurs.
    pub fn new(
        residuals: Vec<Expr>,
        variables: Vec<Symbol>,
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, CertificationError> {
        check_problem_sources(&residuals, &variables, meter)?;
        meter.checkpoint()?;
        Ok(Self {
            residuals,
            variables,
        })
    }

    /// Parse bounded exact source expressions without simplifying away domains.
    ///
    /// The parser retains ordered Add/Mul/Pow operands, so `0*(1/x)` and
    /// `log(x)-log(x)` still require their original domains. Names construct
    /// plain symbols; identity-bearing symbols use [`Self::new`] instead.
    /// Each at-most-4-KiB parse is a conservatively precharged synchronous work
    /// unit with checkpoints before/after it; interval checking has its own
    /// per-node and arithmetic safe points. No printed string becomes identity.
    pub fn parse_sources(
        residuals: &[String],
        variables: &[String],
        meter: &mut impl BudgetMeter,
    ) -> Result<Self, CertificationError> {
        meter.checkpoint()?;
        check_dimensions(residuals.len(), variables.len())?;
        charge(
            meter,
            1,
            (variables.len() * size_of::<Symbol>() + residuals.len() * size_of::<Expr>()) as u64,
            2,
        )?;
        let mut symbols = Vec::new();
        symbols
            .try_reserve_exact(variables.len())
            .map_err(|_| CertificationError::AllocationFailure)?;
        for name in variables {
            if name.len() > MAX_CERTIFICATION_NAME_BYTES {
                return Err(CertificationError::InvalidVariable);
            }
            charge(meter, name.len() as u64 + 1, name.len() as u64, 1)?;
            let mut chars = name.chars();
            let identifier = chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
                && chars.all(|c| c.is_alphanumeric() || c == '_');
            if !identifier
                || matches!(
                    name.as_str(),
                    "pi" | "Pi" | "E" | "I" | "oo" | "zoo" | "nan" | "NaN"
                )
            {
                return Err(CertificationError::InvalidVariable);
            }
            symbols.push(Symbol::new(name.clone()));
        }
        let mut expressions = Vec::new();
        expressions
            .try_reserve_exact(residuals.len())
            .map_err(|_| CertificationError::AllocationFailure)?;
        for (index, source) in residuals.iter().enumerate() {
            if source.len() > MAX_CERTIFICATION_SOURCE_BYTES {
                return Err(CertificationError::SourceLimit("textual residual bytes"));
            }
            // Raw construction does not evaluate powers. With <=4096 total
            // decimal digits its only numeric work (literal division/decimal
            // normalization) stays far below the core's 1024-limb envelope.
            let length = source.len() as u64;
            let envelope = length
                .saturating_add(1)
                .saturating_mul(length.saturating_add(1));
            charge(
                meter,
                envelope.saturating_mul(4),
                envelope.saturating_mul(2) + 1024,
                length * 8 + 1,
            )?;
            let expression = parse_raw(source).map_err(|_| CertificationError::Parse { index })?;
            meter.checkpoint()?;
            expressions.push(expression);
        }
        Self::new(expressions, symbols, meter)
    }

    pub fn residuals(&self) -> &[Expr] {
        &self.residuals
    }

    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }

    /// Recompute whole-box C1 residual and Jacobian bounds, including every
    /// original domain obligation. An enclosure alone does not establish a root.
    pub fn enclose(
        &self,
        domain: &[RealBall],
        precision: u32,
        meter: &mut impl BudgetMeter,
    ) -> Result<ResidualJacobianEnclosure, CertificationError> {
        metered_enclose_residual_jacobian(
            &self.residuals,
            &self.variables,
            domain,
            precision,
            meter,
        )
        .map_err(Into::into)
    }

    /// Verify a unique nonsingular root in one proposed box of a square system.
    /// The rational preconditioner is an untrusted witness, not an accepted inverse.
    pub fn verify_root(
        &self,
        witness: &RootBoxWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootBoxReceipt, CertificationError> {
        verify_root_box(&self.residuals, &self.variables, witness, meter).map_err(Into::into)
    }

    /// Propose a complete cover of a finite box without initial guesses or a
    /// supplied inverse. Exact interval bounds and bounded subdivision guide
    /// this search; the returned witness grants no completeness claim.
    ///
    /// Singular roots, roots on the outer boundary, unresolved source domains
    /// and exhausted search limits refuse without returning a partial root set.
    /// Candidate generation belongs to the caller's ordinary work budget.
    pub fn propose_isolation(
        &self,
        domain: &[RealBall],
        options: RootIsolationOptions,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationWitness, CertificationError> {
        propose_root_isolation(&self.residuals, &self.variables, domain, options, meter)
            .map_err(Into::into)
    }

    /// Independently check every original-source exclusion and local root
    /// witness, complete box coverage and strict separation of distinct roots.
    /// A zero-root receipt requires a complete exclusion cover.
    pub fn verify_isolation(
        &self,
        domain: &[RealBall],
        witness: &RootIsolationWitness,
        meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationReceipt, CertificationError> {
        verify_root_isolation(&self.residuals, &self.variables, domain, witness, meter)
            .map_err(Into::into)
    }

    /// Find and independently certify the complete real root set in a finite
    /// box using separate proposal and verification meters. This lane accepts
    /// only nonsingular roots; singular or unresolved regions refuse.
    ///
    /// To share one cumulative [`fsym_budget::Budget`], call
    /// [`Self::propose_isolation`] first, then [`Self::verify_isolation`] through
    /// that budget's protected verifier meter. Neither phase resets a ledger.
    pub fn isolate_roots(
        &self,
        domain: &[RealBall],
        options: RootIsolationOptions,
        proposal_meter: &mut impl BudgetMeter,
        verification_meter: &mut impl BudgetMeter,
    ) -> Result<RootIsolationReceipt, CertificationError> {
        let witness = self.propose_isolation(domain, options, proposal_meter)?;
        self.verify_isolation(domain, &witness, verification_meter)
    }
}

#[derive(Serialize)]
struct RootArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    witness: &'a RootBoxWitness,
}

// Keep every potentially expensive payload borrowed until its fixed bounds and
// expected typed source are checked. In particular, never decode an attacker
// supplied Expr/BigRational merely to discover it is a different source.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UntrustedArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    #[serde(borrow)]
    residuals: &'a RawValue,
    #[serde(borrow)]
    variables: &'a RawValue,
    #[serde(borrow)]
    witness: &'a RawValue,
}

#[derive(Default)]
pub(crate) struct JsonNesting {
    depth: usize,
    in_string: bool,
    escaped: bool,
}

impl JsonNesting {
    pub(crate) fn scan(&mut self, bytes: &[u8]) -> Result<(), CertificationError> {
        for &byte in bytes {
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.in_string = false;
                }
            } else {
                match byte {
                    b'"' => self.in_string = true,
                    b'[' | b'{' => {
                        self.depth += 1;
                        if self.depth > MAX_ROOT_CERTIFICATE_JSON_DEPTH {
                            return Err(CertificationError::DepthLimit {
                                depth: self.depth,
                                max: MAX_ROOT_CERTIFICATE_JSON_DEPTH,
                            });
                        }
                    }
                    b']' | b'}' => self.depth = self.depth.saturating_sub(1),
                    _ => {}
                }
            }
        }
        // serde_json independently validates syntax and balanced delimiters.
        Ok(())
    }
}

struct MeteredWriter<'a, M> {
    bytes: Vec<u8>,
    nesting: JsonNesting,
    meter: &'a mut M,
    failure: Option<CertificationError>,
}

impl<M: BudgetMeter> MeteredWriter<'_, M> {
    fn fail(&mut self, error: CertificationError) -> io::Error {
        self.failure = Some(error);
        io::Error::other("root certificate encoding refused")
    }
}

impl<M: BudgetMeter> Write for MeteredWriter<'_, M> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::Error::other(
                "root certificate encoding already refused",
            ));
        }
        let required = self.bytes.len().saturating_add(bytes.len());
        if required > MAX_ROOT_CERTIFICATE_BYTES {
            return Err(self.fail(CertificationError::ByteLimit {
                actual: required,
                max: MAX_ROOT_CERTIFICATE_BYTES,
            }));
        }
        let growth = if required > self.bytes.capacity() {
            required
                .max(self.bytes.capacity().saturating_mul(2))
                .min(MAX_ROOT_CERTIFICATE_BYTES)
                - self.bytes.capacity()
        } else {
            0
        };
        if let Err(error) = charge(
            self.meter,
            bytes.len() as u64 + 1,
            growth as u64,
            u64::from(growth != 0),
        ) {
            return Err(self.fail(error));
        }
        if let Err(error) = self.nesting.scan(bytes) {
            return Err(self.fail(error));
        }
        if growth != 0 {
            let additional = self.bytes.capacity() + growth - self.bytes.len();
            if self.bytes.try_reserve_exact(additional).is_err() {
                return Err(self.fail(CertificationError::AllocationFailure));
            }
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn encode<T: Serialize + ?Sized>(
    value: &T,
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, CertificationError> {
    meter.checkpoint()?;
    let mut writer = MeteredWriter {
        bytes: Vec::new(),
        nesting: JsonNesting::default(),
        meter,
        failure: None,
    };
    if let Err(error) = serde_json::to_writer(&mut writer, value) {
        return Err(writer.failure.unwrap_or_else(|| encoding_error(&error)));
    }
    writer.meter.checkpoint()?;
    Ok(writer.bytes)
}

pub(crate) fn charge_witness_encoding(
    witness: &RootBoxWitness,
    meter: &mut impl BudgetMeter,
) -> Result<(), CertificationError> {
    // Reserve a conservative logical allowance for traversing exact scalar
    // payloads in addition to the writer's output capacity. The pinned BigInt
    // serializer streams its digits; this does not assume temporary digit
    // allocation. Recheck shape/height against the current caller's ledger.
    if witness.domain.len() > MAX_ROOT_BOX_VARIABLES
        || witness.center.len() > MAX_ROOT_BOX_VARIABLES
        || witness.preconditioner.len() > MAX_ROOT_BOX_VARIABLES
        || witness
            .preconditioner
            .iter()
            .any(|row| row.len() > MAX_ROOT_BOX_VARIABLES)
    {
        return Err(RootBoxError::DimensionMismatch.into());
    }
    for value in witness
        .domain
        .iter()
        .flat_map(|ball| [ball.midpoint(), ball.radius()])
        .chain(witness.center.iter())
        .chain(witness.preconditioner.iter().flatten())
    {
        let numerator = value.numer().limb_count();
        let denominator = value.denom().limb_count();
        if numerator > MAX_ROOT_BOX_LIMBS || denominator > MAX_ROOT_BOX_LIMBS {
            return Err(RootBoxError::ResourceLimit("rational height").into());
        }
        charge(
            meter,
            numerator + denominator + 1,
            128 + 16 * (numerator + denominator),
            2,
        )?;
    }
    Ok(())
}

/// Export the exact typed statement and untrusted witness of a checked receipt.
///
/// The canonical format contains no accepted flag, root enclosure, or claimed
/// contraction bound. A receiver must provide its own expected typed problem
/// and context and rerun independent verification with its own cumulative meter.
pub fn export_root_certificate(
    receipt: &RootBoxReceipt,
    context_digest: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<Vec<u8>, CertificationError> {
    check_problem_sources(receipt.residuals(), receipt.variables(), meter)?;
    charge_witness_encoding(receipt.witness(), meter)?;
    encode(
        &RootArtifact {
            schema_version: ROOT_CERTIFICATE_SCHEMA_VERSION,
            context_digest,
            residuals: receipt.residuals(),
            variables: receipt.variables(),
            witness: receipt.witness(),
        },
        meter,
    )
}

/// Recheck a portable witness against the receiver's exact typed statement.
///
/// This deliberately accepts only the byte-canonical schema emitted by
/// [`export_root_certificate`]. Whitespace/key-order alternatives, extra nested
/// fields, noncanonical numeric encodings, and substituted sources or variable
/// identities refuse. Byte comparisons are over typed serde encodings, never
/// mathematical printers, expression strings, or unchecked digest assertions.
pub fn replay_root_certificate(
    bytes: &[u8],
    expected_problem: &ResidualProblem,
    expected_context: [u8; 32],
    meter: &mut impl BudgetMeter,
) -> Result<RootBoxReceipt, CertificationError> {
    meter.checkpoint()?;
    if bytes.len() > MAX_ROOT_CERTIFICATE_BYTES {
        return Err(CertificationError::ByteLimit {
            actual: bytes.len(),
            max: MAX_ROOT_CERTIFICATE_BYTES,
        });
    }
    let mut nesting = JsonNesting::default();
    for chunk in bytes.chunks(512) {
        charge(meter, chunk.len() as u64, 0, 0)?;
        nesting.scan(chunk)?;
    }
    // Escaped keys can grow serde's reusable scratch buffer repeatedly.
    // Cover those bounded growth requests and decoder diagnostics.
    charge(
        meter,
        bytes.len() as u64 + 1,
        bytes.len() as u64 * 4 + 1024,
        64,
    )?;
    let wire: UntrustedArtifact<'_> =
        serde_json::from_slice(bytes).map_err(|error| encoding_error(&error))?;
    if wire.schema_version != ROOT_CERTIFICATE_SCHEMA_VERSION {
        return Err(CertificationError::UnsupportedVersion {
            found: wire.schema_version,
            supported: ROOT_CERTIFICATE_SCHEMA_VERSION,
        });
    }
    if wire.context_digest != expected_context {
        return Err(CertificationError::UniverseMismatch {
            found: wire.context_digest,
            expected: expected_context,
        });
    }
    let sources = encode(expected_problem.residuals(), meter)?;
    let variables = encode(expected_problem.variables(), meter)?;
    charge(meter, (sources.len() + variables.len()) as u64 + 1, 0, 0)?;
    if sources.as_slice() != wire.residuals.get().as_bytes()
        || variables.as_slice() != wire.variables.get().as_bytes()
    {
        return Err(CertificationError::SourceMismatch);
    }
    // The bounded core deserializer preflights matrix/sequence lengths and
    // integer limbs before scalar construction or rational normalization.
    // Charge its entire bounded synchronous unit before starting; the original
    // source verifier below supplies fine-grained arithmetic safe points.
    let witness_bytes = wire.witness.get().len() as u64;
    charge(
        meter,
        witness_bytes.saturating_mul(witness_bytes + 1),
        witness_bytes * 32 + 1024,
        witness_bytes + 1,
    )?;
    let witness: RootBoxWitness =
        serde_json::from_str(wire.witness.get()).map_err(|error| encoding_error(&error))?;
    meter.checkpoint()?;
    charge_witness_encoding(&witness, meter)?;
    let canonical = encode(
        &RootArtifact {
            schema_version: ROOT_CERTIFICATE_SCHEMA_VERSION,
            context_digest: expected_context,
            residuals: expected_problem.residuals(),
            variables: expected_problem.variables(),
            witness: &witness,
        },
        meter,
    )?;
    charge(meter, bytes.len() as u64 + 1, 0, 0)?;
    if canonical.as_slice() != bytes {
        return Err(CertificationError::NonCanonical);
    }
    expected_problem.verify_root(&witness, meter)
}
