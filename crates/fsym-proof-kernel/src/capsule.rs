//! Verifier-complete polynomial identity/decomposition capsules (FMAP §7.1).
//!
//! A capsule is the minimal closed object set required to check one typed
//! claim. For the initial ZZ/QQ polynomial slice the claim is exactly:
//!
//! ```text
//! subject  ==  coefficient * product(factor_i ^ exponent_i)      over ZZ or QQ
//! ```
//!
//! Deliberately *not* claimed: irreducibility, completeness of the factor list,
//! uniqueness, or anything about roots. Multiplying the factors back is an
//! identity check, not a factorization proof.
//!
//! Trust rules enforced here:
//! * decode is bounded and fail-closed (lengths, counts, trailing bytes);
//! * every referenced object must be present — there is no generator, planner,
//!   network, or filesystem fallback anywhere in this module;
//! * an object's id is confirmed against the full digest of its canonical
//!   payload, so a re-used id with different bytes is refused;
//! * a stored `verified` flag is not part of the schema and therefore cannot
//!   authorize anything: the only authority is this verifier's verdict;
//! * exhaustion is `Inconclusive`, never a rejection and never an acceptance.

use crate::claim::Claim;
use fsym_core::{BigInt, BigRational, Expr};
use fsym_id::{ContextId, DomainId, RuleId, VerifierId};
use num_traits::Zero;
use std::collections::BTreeMap;

/// Capsule schema this module emits and accepts.
pub const CAPSULE_SCHEMA_VERSION: u16 = 1;
/// Fixed domain separator for capsule digests.
pub const CAPSULE_DIGEST_DOMAIN: &[u8] = b"fsym.capsule.poly-identity.v1\0";
/// Object kind tag for a sparse univariate polynomial.
pub const OBJECT_KIND_POLYNOMIAL: u8 = 0x01;
/// Hard caps applied before any allocation.
pub const MAX_CAPSULE_BYTES: usize = 1024 * 1024;
pub const MAX_OBJECTS: usize = 4096;
pub const MAX_OBJECT_BYTES: usize = 256 * 1024;
pub const MAX_TERMS: usize = 8192;
pub const MAX_FACTORS: usize = 4096;
pub const MAX_SYMBOL_BYTES: usize = 256;
/// Multiplication steps one verification may perform before it is inconclusive.
pub const DEFAULT_FUEL: u64 = 1_000_000;

/// Declared coefficient domain of a capsule claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyDomain {
    /// Integer coefficients (every rational coefficient must have denominator one).
    Zz,
    /// Rational coefficients.
    Qq,
}

impl PolyDomain {
    fn tag(self) -> u8 {
        match self {
            PolyDomain::Zz => 1,
            PolyDomain::Qq => 2,
        }
    }

    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(PolyDomain::Zz),
            2 => Some(PolyDomain::Qq),
            _ => None,
        }
    }

    /// Canonical domain-root payload for this domain.
    pub fn root(self) -> DomainId {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CAPSULE_DIGEST_DOMAIN);
        hasher.update(b"domain");
        hasher.update(&[self.tag()]);
        let digest = hasher.finalize();
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&digest.as_bytes()[..8]);
        DomainId::new(u64::from_le_bytes(raw)).expect("domain digest is non-zero")
    }
}

/// Why a capsule could not even be checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapsuleError {
    /// Declared length or count exceeds a hard cap, or the buffer ends early.
    Malformed(&'static str),
    /// Unknown schema version, kind tag, or domain tag.
    UnknownSchema(&'static str),
    /// Two object entries share an id but not their bytes.
    DuplicateObjectId(u64),
    /// An object's id does not match the digest of its own payload.
    ObjectDigestMismatch(u64),
    /// A referenced object is absent from the capsule.
    MissingObject(u64),
    /// A factor list names the same object twice.
    DuplicateFactor(u64),
    /// The claim root does not match the capsule's own claim.
    ClaimRootMismatch,
    /// The capsule is internally inconsistent (e.g. ZZ claim, rational coefficient).
    Inconsistent(&'static str),
}

/// Outcome of checking a capsule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapsuleVerdict {
    /// The claim holds for the exact objects carried by the capsule.
    Verified {
        /// Digest of the checked claim (canonical preimage).
        claim_digest: [u8; 32],
        /// Objects resolved and inspected.
        objects: usize,
        /// Coefficient multiplications charged.
        multiplications: u64,
    },
    /// The claim is definitely false for these objects.
    Refuted {
        /// Human-readable, deterministic reason.
        reason: String,
    },
    /// The check ran out of fuel: neither accepted nor rejected.
    Inconclusive {
        /// Fuel the capsule asked for.
        needed_at_least: u64,
    },
    /// The capsule itself is unusable.
    Refused(CapsuleError),
}

/// A sparse univariate polynomial over QQ (ZZ coefficients have denominator one).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyObject {
    /// Variable name as carried by the capsule (a view; not an identity).
    pub symbol: String,
    /// `(degree, coefficient)` pairs, strictly descending by degree, no zeros.
    pub terms: Vec<(u32, BigRational)>,
}

impl PolyObject {
    /// Canonical payload bytes for this object.
    pub fn encode(&self) -> Result<Vec<u8>, CapsuleError> {
        if self.symbol.is_empty() || self.symbol.len() > MAX_SYMBOL_BYTES {
            return Err(CapsuleError::Malformed("symbol length out of range"));
        }
        if self.terms.len() > MAX_TERMS {
            return Err(CapsuleError::Malformed("term count exceeds cap"));
        }
        let mut out = Vec::new();
        out.push(OBJECT_KIND_POLYNOMIAL);
        out.extend_from_slice(&(self.symbol.len() as u32).to_le_bytes());
        out.extend_from_slice(self.symbol.as_bytes());
        out.extend_from_slice(&(self.terms.len() as u32).to_le_bytes());
        let mut previous: Option<u32> = None;
        for (degree, coefficient) in &self.terms {
            if coefficient.is_zero() {
                return Err(CapsuleError::Malformed("zero coefficient must be absent"));
            }
            if previous.is_some_and(|last| *degree >= last) {
                return Err(CapsuleError::Malformed("degrees must strictly descend"));
            }
            previous = Some(*degree);
            out.extend_from_slice(&degree.to_le_bytes());
            let coefficient_bytes = encode_rational(coefficient)?;
            out.extend_from_slice(&(coefficient_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&coefficient_bytes);
        }
        if out.len() > MAX_OBJECT_BYTES {
            return Err(CapsuleError::Malformed("object exceeds byte cap"));
        }
        Ok(out)
    }

    /// Decode a canonical payload, rejecting every non-canonical spelling.
    pub fn decode(bytes: &[u8]) -> Result<Self, CapsuleError> {
        if bytes.len() > MAX_OBJECT_BYTES {
            return Err(CapsuleError::Malformed("object exceeds byte cap"));
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.u8()? != OBJECT_KIND_POLYNOMIAL {
            return Err(CapsuleError::UnknownSchema("object kind"));
        }
        let symbol_len = cursor.u32()? as usize;
        if symbol_len == 0 || symbol_len > MAX_SYMBOL_BYTES {
            return Err(CapsuleError::Malformed("symbol length out of range"));
        }
        let symbol = String::from_utf8(cursor.take(symbol_len)?.to_vec())
            .map_err(|_| CapsuleError::Malformed("symbol is not UTF-8"))?;
        let term_count = cursor.u32()? as usize;
        if term_count > MAX_TERMS {
            return Err(CapsuleError::Malformed("term count exceeds cap"));
        }
        let mut terms = Vec::with_capacity(term_count.min(64));
        let mut previous: Option<u32> = None;
        for _ in 0..term_count {
            let degree = cursor.u32()?;
            if previous.is_some_and(|last| degree >= last) {
                return Err(CapsuleError::Malformed("degrees must strictly descend"));
            }
            previous = Some(degree);
            let coefficient_len = cursor.u32()? as usize;
            let coefficient_bytes = cursor.take(coefficient_len)?;
            let coefficient = decode_rational(coefficient_bytes)?;
            if coefficient.is_zero() {
                return Err(CapsuleError::Malformed("zero coefficient must be absent"));
            }
            terms.push((degree, coefficient));
        }
        if !cursor.is_empty() {
            return Err(CapsuleError::Malformed("trailing bytes after object"));
        }
        Ok(Self { symbol, terms })
    }

    /// Degree of the polynomial (zero for an empty term list).
    pub fn degree(&self) -> u32 {
        self.terms.first().map(|(degree, _)| *degree).unwrap_or(0)
    }

    fn is_integral(&self) -> bool {
        self.terms
            .iter()
            .all(|(_, coefficient)| coefficient.denom() == &BigInt::from(1))
    }
}

/// One object entry of a capsule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleObject {
    /// Content-derived id (first eight digest bytes, little endian).
    pub id: u64,
    /// Full digest of `payload`.
    pub digest: [u8; 32],
    /// Canonical object payload.
    pub payload: Vec<u8>,
}

impl CapsuleObject {
    /// Build an entry from canonical payload bytes, deriving id and digest.
    pub fn new(payload: Vec<u8>) -> Self {
        let digest = object_digest(&payload);
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&digest[..8]);
        Self {
            id: u64::from_le_bytes(raw),
            digest,
            payload,
        }
    }
}

/// Full content digest of an object payload.
pub fn object_digest(payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CAPSULE_DIGEST_DOMAIN);
    hasher.update(b"object");
    hasher.update(&(payload.len() as u64).to_le_bytes());
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

/// Canonical wire encoding of one exact coefficient.
fn encode_rational(value: &BigRational) -> Result<Vec<u8>, CapsuleError> {
    Expr::Rational(value.clone())
        .to_canonical_numeric_bytes()
        .map_err(|_| CapsuleError::Malformed("coefficient is not canonically encodable"))
}

/// Inverse of [`encode_rational`]; refuses every non-canonical spelling.
fn decode_rational(bytes: &[u8]) -> Result<BigRational, CapsuleError> {
    match Expr::from_canonical_numeric_bytes(bytes)
        .map_err(|_| CapsuleError::Malformed("coefficient is not canonical"))?
    {
        Expr::Integer(value) => Ok(BigRational::from_integer(value)),
        Expr::Rational(value) => Ok(value),
        _ => Err(CapsuleError::Malformed("coefficient is not numeric")),
    }
}

/// Typed claim carried by the capsule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyIdentityClaim {
    /// Declared coefficient domain.
    pub domain: PolyDomain,
    /// Assumptions context root the claim was stated under.
    pub context: ContextId,
    /// Rewrite/rule root the decomposition was produced under.
    pub rule: RuleId,
    /// Verifier root expected to check this claim.
    pub verifier: VerifierId,
    /// Object id of the subject polynomial.
    pub subject: u64,
    /// Scalar coefficient.
    pub coefficient: BigRational,
    /// `(object id, exponent)` factors; exponents are positive.
    pub factors: Vec<(u64, u32)>,
}

impl PolyIdentityClaim {
    /// Canonical preimage of the claim (roots included, printed text excluded).
    pub fn preimage(&self) -> Result<Vec<u8>, CapsuleError> {
        let mut out = Vec::new();
        out.push(self.domain.tag());
        out.extend_from_slice(&self.context.raw().to_le_bytes());
        out.extend_from_slice(&self.rule.raw().to_le_bytes());
        out.extend_from_slice(&self.verifier.raw().to_le_bytes());
        out.extend_from_slice(&self.subject.to_le_bytes());
        let coefficient_bytes = encode_rational(&self.coefficient)?;
        out.extend_from_slice(&(coefficient_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&coefficient_bytes);
        out.extend_from_slice(&(self.factors.len() as u32).to_le_bytes());
        for (id, exponent) in &self.factors {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&exponent.to_le_bytes());
        }
        Ok(out)
    }

    /// Stable digest of this claim.
    pub fn digest(&self) -> Result<[u8; 32], CapsuleError> {
        let preimage = self.preimage()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CAPSULE_DIGEST_DOMAIN);
        hasher.update(b"claim");
        hasher.update(&preimage);
        Ok(*hasher.finalize().as_bytes())
    }

    /// The kernel claim this capsule asserts, for evidence plumbing.
    pub fn kernel_claim(&self, subject: &Expr, product: &Expr) -> Claim {
        Claim::equality(subject.clone(), product.clone())
    }
}

/// A decoded capsule: typed claim plus its closed object set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capsule {
    /// Declared schema version.
    pub schema_version: u16,
    /// The typed claim.
    pub claim: PolyIdentityClaim,
    /// Objects, indexed by id.
    pub objects: BTreeMap<u64, CapsuleObject>,
}

impl Capsule {
    /// Encode to canonical capsule bytes.
    pub fn encode(&self) -> Result<Vec<u8>, CapsuleError> {
        let claim_preimage = self.claim.preimage()?;
        let mut out = Vec::new();
        out.extend_from_slice(b"FSYMCAP");
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        out.extend_from_slice(&(claim_preimage.len() as u32).to_le_bytes());
        out.extend_from_slice(&claim_preimage);
        out.extend_from_slice(&(self.objects.len() as u32).to_le_bytes());
        for object in self.objects.values() {
            if object.payload.len() > MAX_OBJECT_BYTES {
                return Err(CapsuleError::Malformed("object exceeds byte cap"));
            }
            out.extend_from_slice(&object.id.to_le_bytes());
            out.extend_from_slice(&object.digest);
            out.extend_from_slice(&(object.payload.len() as u32).to_le_bytes());
            out.extend_from_slice(&object.payload);
        }
        if out.len() > MAX_CAPSULE_BYTES {
            return Err(CapsuleError::Malformed("capsule exceeds byte cap"));
        }
        Ok(out)
    }

    /// Decode canonical capsule bytes, failing closed on every anomaly.
    pub fn decode(bytes: &[u8]) -> Result<Self, CapsuleError> {
        if bytes.len() > MAX_CAPSULE_BYTES {
            return Err(CapsuleError::Malformed("capsule exceeds byte cap"));
        }
        let mut cursor = Cursor::new(bytes);
        if cursor.take(7)? != b"FSYMCAP" {
            return Err(CapsuleError::UnknownSchema("magic"));
        }
        let schema_version = cursor.u16()?;
        if schema_version != CAPSULE_SCHEMA_VERSION {
            return Err(CapsuleError::UnknownSchema("schema version"));
        }
        let claim_len = cursor.u32()? as usize;
        let claim_bytes = cursor.take(claim_len)?.to_vec();
        let claim = decode_claim(&claim_bytes)?;
        let object_count = cursor.u32()? as usize;
        if object_count > MAX_OBJECTS {
            return Err(CapsuleError::Malformed("object count exceeds cap"));
        }
        let mut objects: BTreeMap<u64, CapsuleObject> = BTreeMap::new();
        for _ in 0..object_count {
            let id = cursor.u64()?;
            let mut digest = [0u8; 32];
            digest.copy_from_slice(cursor.take(32)?);
            let payload_len = cursor.u32()? as usize;
            if payload_len > MAX_OBJECT_BYTES {
                return Err(CapsuleError::Malformed("object exceeds byte cap"));
            }
            let payload = cursor.take(payload_len)?.to_vec();
            if object_digest(&payload) != digest {
                return Err(CapsuleError::ObjectDigestMismatch(id));
            }
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&digest[..8]);
            if u64::from_le_bytes(raw) != id {
                return Err(CapsuleError::ObjectDigestMismatch(id));
            }
            match objects.get(&id) {
                Some(existing) if existing.payload != payload => {
                    return Err(CapsuleError::DuplicateObjectId(id));
                }
                Some(_) => {}
                None => {
                    objects.insert(
                        id,
                        CapsuleObject {
                            id,
                            digest,
                            payload,
                        },
                    );
                }
            }
        }
        if !cursor.is_empty() {
            return Err(CapsuleError::Malformed("trailing bytes after capsule"));
        }
        let capsule = Self {
            schema_version,
            claim,
            objects,
        };
        capsule.validate_claim()?;
        Ok(capsule)
    }

    fn validate_claim(&self) -> Result<(), CapsuleError> {
        if self.claim.factors.len() > MAX_FACTORS {
            return Err(CapsuleError::Malformed("factor count exceeds cap"));
        }
        let mut seen = BTreeMap::new();
        for (id, exponent) in &self.claim.factors {
            if *exponent == 0 {
                return Err(CapsuleError::Inconsistent(
                    "factor exponent must be positive",
                ));
            }
            if seen.insert(*id, ()).is_some() {
                return Err(CapsuleError::DuplicateFactor(*id));
            }
            if !self.objects.contains_key(id) {
                return Err(CapsuleError::MissingObject(*id));
            }
        }
        if !self.objects.contains_key(&self.claim.subject) {
            return Err(CapsuleError::MissingObject(self.claim.subject));
        }
        if self.claim.domain == PolyDomain::Zz && self.claim.coefficient.denom() != &BigInt::from(1)
        {
            return Err(CapsuleError::Inconsistent(
                "ZZ claim with rational coefficient",
            ));
        }
        for id in self.objects.keys() {
            let object = PolyObject::decode(&self.objects[id].payload)?;
            if self.claim.domain == PolyDomain::Zz && !object.is_integral() {
                return Err(CapsuleError::Inconsistent(
                    "ZZ capsule with rational object",
                ));
            }
        }
        Ok(())
    }

    /// Resolve one object as a polynomial.
    pub fn polynomial(&self, id: u64) -> Result<PolyObject, CapsuleError> {
        let object = self
            .objects
            .get(&id)
            .ok_or(CapsuleError::MissingObject(id))?;
        PolyObject::decode(&object.payload)
    }
}

/// Check a capsule against an independently supplied claim root.
///
/// The root is the authority for *which* claim is being accepted: a capsule
/// whose own claim digest differs is refused, so a capsule cannot redefine the
/// statement it is supposed to support. `fuel` bounds the coefficient
/// multiplications; running out of fuel is `Inconclusive`.
pub fn verify_capsule(
    bytes: &[u8],
    expected_claim_root: Option<[u8; 32]>,
    fuel: u64,
) -> CapsuleVerdict {
    let capsule = match Capsule::decode(bytes) {
        Ok(capsule) => capsule,
        Err(error) => return CapsuleVerdict::Refused(error),
    };
    let claim_digest = match capsule.claim.digest() {
        Ok(digest) => digest,
        Err(error) => return CapsuleVerdict::Refused(error),
    };
    if let Some(expected) = expected_claim_root
        && expected != claim_digest
    {
        return CapsuleVerdict::Refused(CapsuleError::ClaimRootMismatch);
    }
    let subject = match capsule.polynomial(capsule.claim.subject) {
        Ok(subject) => subject,
        Err(error) => return CapsuleVerdict::Refused(error),
    };

    // Exact product of coefficient * factor^exponent, charged against fuel.
    let mut product_terms: BTreeMap<u32, BigRational> = BTreeMap::new();
    product_terms.insert(0, capsule.claim.coefficient.clone());
    let mut multiplications: u64 = 0;
    for (id, exponent) in &capsule.claim.factors {
        let factor = match capsule.polynomial(*id) {
            Ok(factor) => factor,
            Err(error) => return CapsuleVerdict::Refused(error),
        };
        if factor.symbol != subject.symbol {
            return CapsuleVerdict::Refuted {
                reason: format!("factor {id} is in a different variable"),
            };
        }
        for _ in 0..*exponent {
            let mut next: BTreeMap<u32, BigRational> = BTreeMap::new();
            for (left_degree, left) in &product_terms {
                for (right_degree, right) in &factor.terms {
                    let Some(degree) = left_degree.checked_add(*right_degree) else {
                        return CapsuleVerdict::Refuted {
                            reason: "degree overflow".to_string(),
                        };
                    };
                    let term = left.clone() * right.clone();
                    let entry = next.entry(degree).or_insert_with(BigRational::zero);
                    *entry += term;
                    if next.len() > MAX_TERMS {
                        return CapsuleVerdict::Refuted {
                            reason: "product exceeds term cap".to_string(),
                        };
                    }
                }
            }
            next.retain(|_, coefficient| !coefficient.is_zero());
            if next.len() > MAX_TERMS {
                return CapsuleVerdict::Refuted {
                    reason: "product exceeds term cap".to_string(),
                };
            }
            product_terms = next;
            multiplications += 1;
            if multiplications > fuel {
                return CapsuleVerdict::Inconclusive {
                    needed_at_least: multiplications,
                };
            }
        }
    }

    if product_terms.len() > subject.terms.len().max(MAX_TERMS) {
        return CapsuleVerdict::Refuted {
            reason: "product degree exceeds subject".to_string(),
        };
    }
    if subject.degree() != product_terms.keys().next_back().copied().unwrap_or(0) {
        return CapsuleVerdict::Refuted {
            reason: format!(
                "degree mismatch: subject {} vs product {}",
                subject.degree(),
                product_terms.keys().next_back().copied().unwrap_or(0)
            ),
        };
    }
    for (degree, coefficient) in &subject.terms {
        match product_terms.get(degree) {
            Some(actual) if actual == coefficient => {}
            Some(actual) => {
                return CapsuleVerdict::Refuted {
                    reason: format!(
                        "coefficient mismatch at degree {degree}: {actual} != {coefficient}"
                    ),
                };
            }
            None => {
                return CapsuleVerdict::Refuted {
                    reason: format!("missing coefficient at degree {degree}"),
                };
            }
        }
    }
    CapsuleVerdict::Verified {
        claim_digest,
        objects: capsule.objects.len(),
        multiplications,
    }
}

fn decode_claim(bytes: &[u8]) -> Result<PolyIdentityClaim, CapsuleError> {
    let mut cursor = Cursor::new(bytes);
    let domain =
        PolyDomain::from_tag(cursor.u8()?).ok_or(CapsuleError::UnknownSchema("domain tag"))?;
    let context =
        ContextId::new(cursor.u64()?).map_err(|_| CapsuleError::Malformed("zero context root"))?;
    let rule = RuleId::new(cursor.u64()?).map_err(|_| CapsuleError::Malformed("zero rule root"))?;
    let verifier = VerifierId::new(cursor.u64()?)
        .map_err(|_| CapsuleError::Malformed("zero verifier root"))?;
    let subject = cursor.u64()?;
    let coefficient_len = cursor.u32()? as usize;
    let coefficient_bytes = cursor.take(coefficient_len)?;
    let coefficient = decode_rational(coefficient_bytes)?;
    if coefficient.is_zero() {
        return Err(CapsuleError::Inconsistent(
            "claim coefficient must be nonzero",
        ));
    }
    let factor_count = cursor.u32()? as usize;
    if factor_count > MAX_FACTORS {
        return Err(CapsuleError::Malformed("factor count exceeds cap"));
    }
    let mut factors = Vec::with_capacity(factor_count.min(64));
    for _ in 0..factor_count {
        let id = cursor.u64()?;
        let exponent = cursor.u32()?;
        factors.push((id, exponent));
    }
    if !cursor.is_empty() {
        return Err(CapsuleError::Malformed("trailing bytes after claim"));
    }
    Ok(PolyIdentityClaim {
        domain,
        context,
        rule,
        verifier,
        subject,
        coefficient,
        factors,
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], CapsuleError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(CapsuleError::Malformed("length overflow"))?;
        if end > self.bytes.len() {
            return Err(CapsuleError::Malformed(
                "buffer ends before declared length",
            ));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, CapsuleError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, CapsuleError> {
        let mut raw = [0u8; 2];
        raw.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(raw))
    }

    fn u32(&mut self) -> Result<u32, CapsuleError> {
        let mut raw = [0u8; 4];
        raw.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(raw))
    }

    fn u64(&mut self) -> Result<u64, CapsuleError> {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(raw))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
