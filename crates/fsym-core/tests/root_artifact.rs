//! The shared transport checks receiver-owned sources and replays real proofs.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::root_certificate::artifact::{
    CertificationError, MAX_ROOT_CERTIFICATE_BYTES, MAX_ROOT_CERTIFICATE_JSON_DEPTH,
    ROOT_CERTIFICATE_SCHEMA_VERSION, RootCertificateSources, export_root_certificate,
    replay_root_certificate,
};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness, Symbol, SymbolIdentity,
    parse_raw,
};
use serde::Serialize;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn source() -> RootCertificateSources {
    RootCertificateSources::new(
        vec![parse_raw("x^2-2").unwrap()],
        vec![Symbol::new("x")],
        &mut Unbounded,
    )
    .unwrap()
}

fn witness() -> RootBoxWitness {
    RootBoxWitness {
        domain: vec![RealBall::new(q(7, 5), q(1, 10)).unwrap()],
        center: vec![q(7, 5)],
        preconditioner: vec![vec![q(5, 14)]],
        precision_digits: 24,
    }
}

#[derive(Serialize)]
struct ProposedArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    witness: &'a RootBoxWitness,
}

fn candidate_bytes(source: &RootCertificateSources, witness: &RootBoxWitness) -> Vec<u8> {
    serde_json::to_vec(&ProposedArtifact {
        schema_version: ROOT_CERTIFICATE_SCHEMA_VERSION,
        context_digest: [7; 32],
        residuals: source.residuals(),
        variables: source.variables(),
        witness,
    })
    .unwrap()
}

#[test]
fn canonical_roundtrip_retains_exact_independent_claim_and_source_ownership() {
    let source = source();
    let receipt = source.verify_root(&witness(), &mut Unbounded).unwrap();
    let bytes = export_root_certificate(&receipt, [7; 32], &mut Unbounded).unwrap();
    assert_eq!(bytes, candidate_bytes(&source, &witness()));

    let mut original = source.residuals().to_vec();
    let receiver =
        RootCertificateSources::copy_from(&original, source.variables(), &mut Unbounded).unwrap();
    original[0] = Expr::from_i64(1);
    let replayed = replay_root_certificate(&bytes, &receiver, [7; 32], &mut Unbounded).unwrap();
    assert_eq!(replayed, receipt);
    assert_eq!(replayed.root_enclosure()[0].lower(), q(197, 140));
    assert_eq!(replayed.root_enclosure()[0].upper(), q(199, 140));
    assert_eq!(replayed.contraction_bound(), &q(1, 14));
    let document: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(document.get("root_enclosure").is_none());
    assert!(document.get("contraction_bound").is_none());
    assert!(document.get("accepted").is_none());
}

#[test]
fn receiver_context_and_typed_source_bind_before_untrusted_witness_decoding() {
    let source = source();
    let bytes = candidate_bytes(&source, &witness());
    assert!(matches!(
        replay_root_certificate(&bytes, &source, [8; 32], &mut Unbounded),
        Err(CertificationError::UniverseMismatch { .. })
    ));
    let other = RootCertificateSources::new(
        source.residuals().to_vec(),
        vec![Symbol::with_identity(
            "x",
            SymbolIdentity::from_assumptions_digest([1; 32]),
        )],
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        replay_root_certificate(&bytes, &other, [7; 32], &mut Unbounded),
        Err(CertificationError::SourceMismatch)
    );

    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["residuals"] = serde_json::json!({"unexpected": ["9".repeat(20_000)]});
    forged["witness"] = serde_json::json!({"also": "not a witness"});
    assert_eq!(
        replay_root_certificate(
            &serde_json::to_vec(&forged).unwrap(),
            &source,
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::SourceMismatch)
    );
}

#[test]
fn canonical_candidate_bytes_are_still_not_an_accepted_proof() {
    let source = source();
    let mut forged = witness();
    forged.preconditioner[0][0] = q(0, 1);
    assert_eq!(
        replay_root_certificate(
            &candidate_bytes(&source, &forged),
            &source,
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::Root(RootBoxError::NotContractive))
    );
    let mut misplaced = witness();
    misplaced.center[0] = q(9, 1);
    assert_eq!(
        replay_root_certificate(
            &candidate_bytes(&source, &misplaced),
            &source,
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::Root(RootBoxError::CenterOutsideBox {
            coordinate: 0,
        }))
    );
}

#[test]
fn alternate_encoding_acceptance_flags_and_oversized_payloads_refuse() {
    let source = source();
    let bytes = candidate_bytes(&source, &witness());
    let mut whitespace = vec![b' '];
    whitespace.extend_from_slice(&bytes);
    assert_eq!(
        replay_root_certificate(&whitespace, &source, [7; 32], &mut Unbounded),
        Err(CertificationError::NonCanonical)
    );
    let mut accepted: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    accepted["accepted"] = serde_json::json!(true);
    assert!(matches!(
        replay_root_certificate(
            &serde_json::to_vec(&accepted).unwrap(),
            &source,
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::Encoding(_))
    ));
    assert!(matches!(
        replay_root_certificate(
            &vec![b' '; MAX_ROOT_CERTIFICATE_BYTES + 1],
            &source,
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::ByteLimit { .. })
    ));
    assert!(matches!(
        replay_root_certificate(
            &[b'['; MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1],
            &source,
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::DepthLimit { .. })
    ));
}

struct Interrupt {
    budget: Budget,
    checkpoints: usize,
    cancel_at: usize,
}

impl Interrupt {
    fn new(cancel_at: usize) -> Self {
        Self {
            budget: Budget::new(BudgetLimits::uniform(1 << 40, 137)),
            checkpoints: 0,
            cancel_at,
        }
    }
}

impl BudgetMeter for Interrupt {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let index = self.checkpoints;
        self.checkpoints += 1;
        if index >= self.cancel_at {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_replay_checkpoint_cancels_without_publishing_or_borrowing_protected_fuel() {
    let source = source();
    let bytes = candidate_bytes(&source, &witness());
    let mut measured = Interrupt::new(usize::MAX);
    replay_root_certificate(&bytes, &source, [7; 32], &mut measured).unwrap();
    assert!(measured.checkpoints > 100);
    for index in 0..measured.checkpoints {
        let mut interrupted = Interrupt::new(index);
        assert_eq!(
            replay_root_certificate(&bytes, &source, [7; 32], &mut interrupted),
            Err(CertificationError::Meter(MeterError::Cancelled)),
            "checkpoint {index}"
        );
        assert_eq!(interrupted.budget.verifier_remaining(), 137);
    }
    assert!(measured.budget.remaining(Dimension::MemoryBytes) < (1 << 40) - 64 * 1024 * 1024);
}
