use frankensympy::artifact_stream::{
    ARTIFACT_CHUNK_BYTES, ARTIFACT_STREAM_SCHEMA_VERSION, ArtifactContentId, ArtifactExport,
    ArtifactImport, ArtifactKind, ArtifactManifest, ArtifactTransferError, ArtifactTransferId,
    MAX_ARTIFACT_BYTES, artifact_chunk_digest, artifact_content_id,
};
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};

fn payload(length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| ((index * 197 + 11) % 256) as u8)
        .collect()
}

fn export(bytes: Vec<u8>) -> ArtifactExport {
    ArtifactExport::new(ArtifactKind::RootIsolation, bytes, &mut Unbounded).unwrap()
}

fn receive(source: &ArtifactExport) -> ArtifactImport {
    let mut receiving = ArtifactImport::new(source.manifest().clone(), &mut Unbounded).unwrap();
    for index in 0..source.manifest().chunk_count() {
        let chunk = source.chunk(index, &mut Unbounded).unwrap();
        receiving
            .append(index, &chunk.digest, &chunk.data_hex, &mut Unbounded)
            .unwrap();
    }
    receiving
}

#[test]
fn maximum_opaque_payload_round_trips_without_borrowing_verifier_fuel() {
    let bytes = payload(MAX_ARTIFACT_BYTES);
    let mut meter = Budget::new(BudgetLimits::uniform(1 << 40, 73));
    let source =
        ArtifactExport::new(ArtifactKind::RootIsolation, bytes.clone(), &mut meter).unwrap();
    assert_eq!(source.manifest().chunk_count(), 64);
    let mut receiving = ArtifactImport::new(source.manifest().clone(), &mut meter).unwrap();
    for index in 0..64 {
        let chunk = source.chunk(index, &mut meter).unwrap();
        assert_eq!(chunk.data_hex.len(), 2 * ARTIFACT_CHUNK_BYTES);
        assert!(serde_json::to_vec(&chunk).unwrap().len() < 64 * 1024);
        let expected = receiving.preview_append(index).unwrap();
        assert_eq!(
            receiving
                .append(index, &chunk.digest, &chunk.data_hex, &mut meter)
                .unwrap(),
            expected,
        );
        assert_eq!(expected.received_bytes, (index + 1) * ARTIFACT_CHUNK_BYTES);
    }
    assert_eq!(
        receiving
            .checked_bytes(ArtifactKind::RootIsolation, &mut meter)
            .unwrap(),
        bytes,
    );
    assert_eq!(meter.verifier_remaining(), 73);
    assert!(meter.remaining(Dimension::ComputeSteps) < 1 << 40);
    assert!(meter.remaining(Dimension::MemoryBytes) < 1 << 40);
    assert!(matches!(
        source.chunk(usize::MAX, &mut meter),
        Err(ArtifactTransferError::InvalidIndex { .. }),
    ));
}

#[test]
fn content_and_chunk_domains_bind_kind_payload_length_and_position() {
    let bytes = vec![0xab; 2 * ARTIFACT_CHUNK_BYTES];
    let source = export(bytes.clone());
    let other_kind =
        ArtifactExport::new(ArtifactKind::RootCertificate, bytes.clone(), &mut Unbounded).unwrap();
    assert_ne!(
        source.manifest().content_id,
        other_kind.manifest().content_id
    );
    let short = artifact_content_id(
        ArtifactKind::RootIsolation,
        &bytes[..bytes.len() - 1],
        &mut Unbounded,
    )
    .unwrap();
    assert_ne!(source.manifest().content_id, short);
    let first = source.chunk(0, &mut Unbounded).unwrap();
    let second = source.chunk(1, &mut Unbounded).unwrap();
    assert_eq!(first.data_hex, second.data_hex);
    assert_ne!(first.digest, second.digest);
    assert_ne!(
        first.digest,
        other_kind.chunk(0, &mut Unbounded).unwrap().digest
    );
    assert_eq!(source.chunk(0, &mut Unbounded).unwrap(), first);

    let mut receiving = ArtifactImport::new(source.manifest().clone(), &mut Unbounded).unwrap();
    receiving
        .append(0, &first.digest, &first.data_hex, &mut Unbounded)
        .unwrap();
    let before = receiving.progress();
    assert_eq!(
        receiving.append(1, &first.digest, &second.data_hex, &mut Unbounded),
        Err(ArtifactTransferError::ChunkDigestMismatch),
    );
    assert_eq!(receiving.progress(), before);
    receiving
        .append(1, &second.digest, &second.data_hex, &mut Unbounded)
        .unwrap();
    assert_eq!(
        receiving.checked_bytes(ArtifactKind::RootCertificate, &mut Unbounded),
        Err(ArtifactTransferError::KindMismatch),
    );
    assert_eq!(
        receiving
            .checked_bytes(ArtifactKind::RootIsolation, &mut Unbounded)
            .unwrap(),
        bytes
    );
}

#[test]
fn typed_ids_manifests_and_retained_capacity_are_bounded() {
    let content = ArtifactContentId::from_digest([0xab; 32]);
    let encoded = format!("\"artifact:{}\"", "ab".repeat(32));
    assert_eq!(serde_json::to_string(&content).unwrap(), encoded);
    assert_eq!(
        serde_json::from_str::<ArtifactContentId>(&encoded).unwrap(),
        content
    );
    assert_eq!(
        serde_json::from_value::<ArtifactContentId>(serde_json::json!(content.to_string()))
            .unwrap(),
        content
    );
    for value in [
        format!("artifact:{}", "AB".repeat(32)),
        format!("proof:{}", "ab".repeat(32)),
        format!("artifact:{}", "0".repeat(63)),
        format!("artifact:{}", "g".repeat(64)),
    ] {
        assert!(serde_json::from_value::<ArtifactContentId>(serde_json::json!(value)).is_err());
    }
    assert!(ArtifactTransferId::new("transfer:bounded_1.test").is_ok());
    let mut spare_label = String::with_capacity(MAX_ARTIFACT_BYTES + 1);
    spare_label.push_str("transfer:small");
    let normalized = ArtifactTransferId::new(spare_label).unwrap();
    assert_eq!(normalized.as_str(), "transfer:small");
    assert_eq!(
        serde_json::to_string(&normalized).unwrap(),
        "\"transfer:small\"",
    );
    for value in ["transfer:", "artifact:wrong_kind", "transfer:has space"] {
        assert_eq!(
            ArtifactTransferId::new(value),
            Err(ArtifactTransferError::InvalidLabel)
        );
    }
    assert!(ArtifactTransferId::new(format!("transfer:{}", "x".repeat(120))).is_err());
    let manifest = ArtifactManifest {
        schema_version: ARTIFACT_STREAM_SCHEMA_VERSION,
        kind: ArtifactKind::RootIsolation,
        total_bytes: 1,
        content_id: content,
    };
    for length in [0, MAX_ARTIFACT_BYTES + 1, usize::MAX] {
        let mut changed = manifest.clone();
        changed.total_bytes = length;
        assert!(ArtifactImport::new(changed, &mut Unbounded).is_err());
    }
    let mut future = manifest.clone();
    future.schema_version += 1;
    assert!(matches!(
        ArtifactImport::new(future, &mut Unbounded),
        Err(ArtifactTransferError::UnsupportedVersion { .. })
    ));
    let mut unknown = serde_json::to_value(&manifest).unwrap();
    unknown["accepted"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ArtifactManifest>(unknown).is_err());

    let mut oversized_capacity = Vec::with_capacity(MAX_ARTIFACT_BYTES + 1);
    oversized_capacity.push(b'x');
    assert!(matches!(
        ArtifactExport::new(
            ArtifactKind::RootIsolation,
            oversized_capacity,
            &mut Unbounded
        ),
        Err(ArtifactTransferError::ByteLimit { .. }),
    ));
    let mut retained = Vec::with_capacity(4096);
    retained.push(b'x');
    let capacity = retained.capacity() as u64;
    let mut meter = Budget::new(BudgetLimits::uniform(1 << 30, 41));
    let memory = meter.remaining(Dimension::MemoryBytes);
    ArtifactExport::new(ArtifactKind::RootIsolation, retained, &mut meter).unwrap();
    assert_eq!(memory - meter.remaining(Dimension::MemoryBytes), capacity);
}

#[test]
fn exact_retries_are_charged_and_conflicting_or_future_chunks_leave_prefix_unchanged() {
    let source = export(vec![0xab; 2 * ARTIFACT_CHUNK_BYTES + 7]);
    let mut receiving = ArtifactImport::new(source.manifest().clone(), &mut Unbounded).unwrap();
    let first = source.chunk(0, &mut Unbounded).unwrap();
    let expected = receiving
        .append(0, &first.digest, &first.data_hex, &mut Unbounded)
        .unwrap();
    let mut meter = Budget::new(BudgetLimits::uniform(1 << 30, 13));
    let before = meter.remaining(Dimension::ComputeSteps);
    assert_eq!(
        receiving
            .append(0, &first.digest, &first.data_hex, &mut meter)
            .unwrap(),
        expected
    );
    assert!(meter.remaining(Dimension::ComputeSteps) < before);
    let last = source.chunk(2, &mut Unbounded).unwrap();
    assert_eq!(
        receiving.append(2, &last.digest, &last.data_hex, &mut Unbounded),
        Err(ArtifactTransferError::OutOfOrder {
            expected: 1,
            found: 2
        }),
    );
    assert!(matches!(
        receiving.append(0, &first.digest, &first.data_hex[2..], &mut Unbounded),
        Err(ArtifactTransferError::ChunkLength { .. }),
    ));
    assert_eq!(
        receiving.append(
            0,
            &first.digest,
            &first.data_hex.to_ascii_uppercase(),
            &mut Unbounded
        ),
        Err(ArtifactTransferError::InvalidHex),
    );
    let mut incorrect_digest = first.digest;
    incorrect_digest[0] ^= 1;
    assert_eq!(
        receiving.append(0, &incorrect_digest, &first.data_hex, &mut Unbounded),
        Err(ArtifactTransferError::ChunkDigestMismatch),
    );
    let changed = vec![0xcd; ARTIFACT_CHUNK_BYTES];
    let changed_digest =
        artifact_chunk_digest(source.manifest().content_id, 0, &changed, &mut Unbounded).unwrap();
    assert_eq!(
        receiving.append(
            0,
            &changed_digest,
            &"cd".repeat(ARTIFACT_CHUNK_BYTES),
            &mut Unbounded
        ),
        Err(ArtifactTransferError::ConflictingChunk),
    );
    assert_eq!(receiving.progress(), expected);
    let second = source.chunk(1, &mut Unbounded).unwrap();
    receiving
        .append(1, &second.digest, &second.data_hex, &mut Unbounded)
        .unwrap();
    receiving
        .append(2, &last.digest, &last.data_hex, &mut Unbounded)
        .unwrap();
    assert_eq!(
        receiving
            .checked_bytes(ArtifactKind::RootIsolation, &mut Unbounded)
            .unwrap(),
        vec![0xab; 2 * ARTIFACT_CHUNK_BYTES + 7]
    );
}

#[test]
fn complete_chunks_cannot_substitute_for_final_content_validation() {
    let bytes = payload(ARTIFACT_CHUNK_BYTES + 7);
    let source = export(bytes.clone());
    let mut forged = source.manifest().clone();
    forged.content_id = ArtifactContentId::from_digest([17; 32]);
    let mut receiving = ArtifactImport::new(forged.clone(), &mut Unbounded).unwrap();
    assert!(matches!(
        receiving.checked_bytes(ArtifactKind::RootIsolation, &mut Unbounded),
        Err(ArtifactTransferError::Incomplete { .. })
    ));
    for (index, raw) in bytes.chunks(ARTIFACT_CHUNK_BYTES).enumerate() {
        let chunk = source.chunk(index, &mut Unbounded).unwrap();
        let digest = artifact_chunk_digest(forged.content_id, index, raw, &mut Unbounded).unwrap();
        receiving
            .append(index, &digest, &chunk.data_hex, &mut Unbounded)
            .unwrap();
    }
    assert_eq!(receiving.progress().received_bytes, bytes.len());
    assert_eq!(
        receiving.checked_bytes(ArtifactKind::RootIsolation, &mut Unbounded),
        Err(ArtifactTransferError::FinalDigestMismatch),
    );

    let valid = receive(&source);
    let mut meter = Budget::new(BudgetLimits::uniform(1 << 30, 29));
    valid
        .checked_bytes(ArtifactKind::RootIsolation, &mut meter)
        .unwrap();
    let after = meter.remaining(Dimension::ComputeSteps);
    valid
        .checked_bytes(ArtifactKind::RootIsolation, &mut meter)
        .unwrap();
    assert!(
        meter.remaining(Dimension::ComputeSteps) < after,
        "each replay rechecks the complete digest"
    );
    assert_eq!(meter.verifier_remaining(), 29);
}

#[test]
fn allocation_and_late_copy_budget_refusals_do_not_accept_a_chunk() {
    let source = export(payload(ARTIFACT_CHUNK_BYTES + 1));
    let first = source.chunk(0, &mut Unbounded).unwrap();
    let mut measured = ArtifactImport::new(source.manifest().clone(), &mut Unbounded).unwrap();
    let mut full = Budget::new(BudgetLimits::uniform(1 << 30, 37));
    measured
        .append(0, &first.digest, &first.data_hex, &mut full)
        .unwrap();
    let spent = (1 << 30) - full.remaining(Dimension::ComputeSteps);
    for (dimension, allowance) in [
        (Dimension::ComputeSteps, spent - 1),
        (Dimension::MemoryBytes, (ARTIFACT_CHUNK_BYTES - 1) as u64),
        (Dimension::AllocationCount, 0),
    ] {
        let mut limited = BudgetLimits::uniform(1 << 30, 37);
        limited.dimensions[dimension.index()] = allowance;
        let mut meter = Budget::new(limited);
        let mut receiving = ArtifactImport::new(source.manifest().clone(), &mut Unbounded).unwrap();
        assert!(matches!(
            receiving.append(0, &first.digest, &first.data_hex, &mut meter),
            Err(ArtifactTransferError::Meter(MeterError::Budget(_))),
        ));
        assert_eq!(receiving.progress().received_bytes, 0);
        assert_eq!(receiving.progress().next_chunk, 0);
        assert_eq!(meter.verifier_remaining(), 37);
        receiving
            .append(0, &first.digest, &first.data_hex, &mut Unbounded)
            .unwrap();
        let last = source.chunk(1, &mut Unbounded).unwrap();
        receiving
            .append(1, &last.digest, &last.data_hex, &mut Unbounded)
            .unwrap();
        assert_eq!(
            receiving
                .checked_bytes(ArtifactKind::RootIsolation, &mut Unbounded)
                .unwrap(),
            payload(ARTIFACT_CHUNK_BYTES + 1)
        );
    }
    let mut limits = BudgetLimits::uniform(1 << 30, 37);
    limits.dimensions[Dimension::MemoryBytes.index()] = source.manifest().total_bytes as u64 - 1;
    let mut insufficient = Budget::new(limits);
    assert!(matches!(
        ArtifactImport::new(source.manifest().clone(), &mut insufficient),
        Err(ArtifactTransferError::Meter(MeterError::Budget(_))),
    ));
    assert_eq!(insufficient.remaining(Dimension::ComputeSteps), 1 << 30);
}

#[derive(Default)]
struct Probe {
    checkpoints: usize,
    interrupt: Option<usize>,
    stopped: bool,
}

impl BudgetMeter for Probe {
    fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped);
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped);
        assert!(charges.iter().all(|(_, amount)| *amount != 0));
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "work continued after cancellation");
        self.checkpoints += 1;
        if self.interrupt == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn every_safe_point(mut operation: impl FnMut(&mut Probe) -> Result<(), ArtifactTransferError>) {
    let mut measured = Probe::default();
    operation(&mut measured).unwrap();
    assert!(measured.checkpoints > 1);
    for interrupt in 1..=measured.checkpoints {
        let mut meter = Probe {
            interrupt: Some(interrupt),
            ..Probe::default()
        };
        assert_eq!(
            operation(&mut meter),
            Err(ArtifactTransferError::Meter(MeterError::Cancelled))
        );
        assert!(meter.stopped);
    }
}

#[test]
fn every_cancellation_safe_point_preserves_the_unpublished_prefix() {
    let bytes = payload(ARTIFACT_CHUNK_BYTES + 3);
    let source = export(bytes.clone());
    let first = source.chunk(0, &mut Unbounded).unwrap();
    every_safe_point(|meter| {
        ArtifactExport::new(ArtifactKind::RootIsolation, bytes.clone(), meter).map(|_| ())
    });
    every_safe_point(|meter| ArtifactImport::new(source.manifest().clone(), meter).map(|_| ()));
    every_safe_point(|meter| source.chunk(0, meter).map(|_| ()));
    every_safe_point(|meter| {
        let mut receiving = ArtifactImport::new(source.manifest().clone(), &mut Unbounded).unwrap();
        let result = receiving.append(0, &first.digest, &first.data_hex, meter);
        if result.is_err() {
            assert_eq!(receiving.progress().received_bytes, 0);
            assert_eq!(receiving.progress().next_chunk, 0);
        }
        result.map(|_| ())
    });
    let complete = receive(&source);
    every_safe_point(|meter| {
        complete
            .checked_bytes(ArtifactKind::RootIsolation, meter)
            .map(|_| ())
    });
}
