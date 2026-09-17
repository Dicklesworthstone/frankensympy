use fsym_core::{BigInt, BigRational};
use fsym_formal::{CONTEXT_ROOT, RULE_ROOT, VERIFIER_ROOT};
use fsym_id::{ContextId, RuleId, VerifierId};
use fsym_proof_kernel::capsule::{
    CAPSULE_SCHEMA_VERSION, Capsule, CapsuleObject, PolyDomain, PolyIdentityClaim, PolyObject,
};

pub fn object(terms: &[(u32, i64)]) -> CapsuleObject {
    CapsuleObject::new(
        PolyObject {
            symbol: "x".into(),
            terms: terms
                .iter()
                .map(|(d, c)| (*d, BigRational::from_integer(BigInt::from(*c))))
                .collect(),
        }
        .encode()
        .expect("fixture polynomial"),
    )
}

/// The independent caller's authority: x^4+4 = (x^2-2x+2)(x^2+2x+2).
pub fn capsule() -> Capsule {
    let subject = object(&[(4, 1), (0, 4)]);
    let a = object(&[(2, 1), (1, -2), (0, 2)]);
    let b = object(&[(2, 1), (1, 2), (0, 2)]);
    let claim = PolyIdentityClaim {
        domain: PolyDomain::Zz,
        context: ContextId::new(CONTEXT_ROOT).expect("context"),
        rule: RuleId::new(RULE_ROOT).expect("rule"),
        verifier: VerifierId::new(VERIFIER_ROOT).expect("verifier"),
        subject: subject.id,
        coefficient: BigRational::from_integer(BigInt::from(1)),
        factors: vec![(a.id, 1), (b.id, 1)],
    };
    Capsule {
        schema_version: CAPSULE_SCHEMA_VERSION,
        claim,
        objects: [subject, a, b].into_iter().map(|o| (o.id, o)).collect(),
    }
}
