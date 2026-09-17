use crate::{
    CONTEXT_ROOT, MAX_CAPSULE_BYTES, MAX_DEGREE, ProjectionError as E, RULE_ROOT, SafePoint,
    VERIFIER_ROOT, checkpoint,
};
use fsym_proof_kernel::capsule::{Capsule, CapsuleVerdict, PolyDomain, verify_capsule};

struct Cursor<'a>(&'a [u8]);
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], E> {
        if n > self.0.len() {
            return Err(E::MalformedInput);
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn byte(&mut self) -> Result<u8, E> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, E> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| E::MalformedInput)?,
        ))
    }
    fn u64(&mut self) -> Result<u64, E> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| E::MalformedInput)?,
        ))
    }
    fn blob(&mut self) -> Result<&'a [u8], E> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    fn end(self) -> Result<(), E> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(E::MalformedInput)
        }
    }
}

// Allocation-free magnitude preflight. Native decode still enforces canonical
// integer/rational spellings, reduction, object hashes, and schema integrity.
fn integer(c: &mut Cursor<'_>) -> Result<i64, E> {
    let sign = c.byte()?;
    if sign > 1 {
        return Err(E::MalformedInput);
    }
    let len = c.u64()?;
    if len > 8 {
        return Err(E::ResourceExhausted("coefficient height"));
    }
    let mag = c.take(len as usize)?;
    let mut bytes = [0u8; 8];
    bytes[..mag.len()].copy_from_slice(mag);
    let value = u64::from_le_bytes(bytes);
    if sign == 0 {
        i64::try_from(value).map_err(|_| E::ResourceExhausted("coefficient height"))
    } else if value == 1u64 << 63 {
        Ok(i64::MIN)
    } else {
        i64::try_from(value)
            .map(|v| -v)
            .map_err(|_| E::ResourceExhausted("coefficient height"))
    }
}
fn coefficient(bytes: &[u8]) -> Result<i64, E> {
    let mut c = Cursor(bytes);
    let tag = c.byte()?;
    if tag != b'I' && tag != b'Q' {
        return Err(E::UnrepresentableSemanticField);
    }
    let value = integer(&mut c)?;
    if tag == b'Q' && integer(&mut c)? != 1 {
        return Err(E::UnrepresentableSemanticField);
    }
    c.end()?;
    Ok(value)
}

fn preflight(bytes: &[u8], cancel: &impl Fn(SafePoint) -> bool) -> Result<(), E> {
    if bytes.len() > MAX_CAPSULE_BYTES {
        return Err(E::ResourceExhausted("capsule bytes"));
    }
    let mut c = Cursor(bytes);
    if c.take(7)? != b"FSYMCAP" || c.take(2)? != [1, 0] {
        return Err(E::UnrepresentableSemanticField);
    }
    let mut claim = Cursor(c.blob()?);
    if claim.byte()? != 1 {
        return Err(E::UnsupportedFamily);
    }
    if claim.u64()? != CONTEXT_ROOT || claim.u64()? != RULE_ROOT || claim.u64()? != VERIFIER_ROOT {
        return Err(E::UnrepresentableSemanticField);
    }
    claim.take(8)?;
    if coefficient(claim.blob()?)? != 1 {
        return Err(E::UnsupportedShape);
    }
    let count = claim.u32()?;
    if count > 2 {
        return Err(E::ResourceExhausted("factor count"));
    }
    if count != 2 {
        return Err(E::UnsupportedShape);
    }
    for _ in 0..count {
        claim.take(8)?;
        if claim.u32()? != 1 {
            return Err(E::UnsupportedShape);
        }
    }
    claim.end()?;
    let objects = c.u32()?;
    if objects > 3 {
        return Err(E::ResourceExhausted("object count"));
    }
    if objects != 3 {
        return Err(E::UnsupportedShape);
    }
    for _ in 0..objects {
        checkpoint(cancel, SafePoint::PreflightObject)?;
        c.take(40)?;
        let mut p = Cursor(c.blob()?);
        if p.byte()? != 1 {
            return Err(E::UnrepresentableSemanticField);
        }
        let symbol = p.blob()?;
        if symbol.is_empty() || symbol.len() > 256 {
            return Err(E::ResourceExhausted("symbol bytes"));
        }
        let terms = p.u32()?;
        if terms > 65 {
            return Err(E::ResourceExhausted("term count"));
        }
        if terms == 0 {
            return Err(E::UnsupportedShape);
        }
        for _ in 0..terms {
            if p.u32()? > MAX_DEGREE {
                return Err(E::ResourceExhausted("degree"));
            }
            coefficient(p.blob()?)?;
        }
        p.end()?;
    }
    c.end()
}

pub(crate) fn admit(
    bytes: &[u8],
    root: [u8; 32],
    cancel: &impl Fn(SafePoint) -> bool,
) -> Result<Capsule, E> {
    checkpoint(cancel, SafePoint::Admission)?;
    preflight(bytes, cancel)?;
    checkpoint(cancel, SafePoint::BeforeNativeVerification)?;
    match verify_capsule(bytes, Some(root), 2) {
        CapsuleVerdict::Verified { claim_digest, .. } if claim_digest == root => {}
        CapsuleVerdict::Inconclusive { .. } => {
            return Err(E::ResourceExhausted("native verification fuel"));
        }
        _ => return Err(E::NativeVerificationFailed),
    }
    checkpoint(cancel, SafePoint::AfterNativeVerification)?;
    let capsule = Capsule::decode(bytes).map_err(|_| E::MalformedInput)?;
    // Kernel decoding permits duplicate identical entries and input ordering;
    // this profile admits only the unique canonical complete encoding.
    if capsule.encode().map_err(|_| E::MalformedInput)? != bytes {
        return Err(E::MalformedInput);
    }
    if capsule.claim.domain != PolyDomain::Zz {
        return Err(E::UnsupportedFamily);
    }
    for id in capsule.objects.keys() {
        if *id != capsule.claim.subject
            && !capsule.claim.factors.iter().any(|(factor, _)| factor == id)
        {
            return Err(E::UnrepresentableSemanticField);
        }
    }
    Ok(capsule)
}
