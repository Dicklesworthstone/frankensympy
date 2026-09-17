//! Deterministic execution transcript records and bit-for-bit comparison (WS13).
//!
//! This module binds recorded inputs and outcomes; it does not itself re-execute a strategy.

#![forbid(unsafe_code)]

use fsym_budget::Dimension;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_REPLAY_EVENTS: usize = 16_384;
const MAX_EVENT_NAME_BYTES: usize = 256;
const MAX_STRATEGY_NAME_BYTES: usize = 256;
const MAX_EVENT_PAYLOAD_BYTES: usize = 1_048_576;
const MAX_REPLAY_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EVENT_CHARGES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ReplayError {
    #[error("Replay {resource} exceeds limit {limit}")]
    LimitExceeded {
        resource: &'static str,
        limit: usize,
    },
    #[error("Replay event index mismatch: expected {expected}, got {actual}")]
    StepIndexMismatch { expected: u64, actual: u64 },
    #[error("Replay event contains a zero-valued dimension charge")]
    ZeroCharge,
    #[error("Replay {resource} must not be empty")]
    EmptyText { resource: &'static str },
    #[error("Replay event {step_index} digest does not match its recorded fields")]
    EventDigestMismatch { step_index: u64 },
    #[error("Replay total payload byte counter does not match its events")]
    PayloadByteCountMismatch,
    #[error("Replay serialization failed: {0}")]
    Serialization(String),
}

/// An individual trace entry in a deterministic computation replay log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayEvent {
    pub step_index: u64,
    pub event_name: String,
    pub dimension_charges: Vec<(Dimension, u64)>,
    pub payload: Vec<u8>,
    pub outcome_digest: [u8; 32],
}

/// Complete deterministic replay transcript recording seeds, choices, and outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayLog {
    pub initial_seed: u64,
    pub strategy_name: String,
    pub events: Vec<ReplayEvent>,
    pub total_payload_bytes: u64,
    pub final_digest: [u8; 32],
}

impl ReplayLog {
    /// Create a new replay log.
    pub fn new(initial_seed: u64, strategy_name: impl Into<String>) -> Result<Self, ReplayError> {
        let strategy_name = strategy_name.into();
        check_text_len(
            strategy_name.len(),
            MAX_STRATEGY_NAME_BYTES,
            "strategy-name bytes",
        )?;
        Ok(Self {
            initial_seed,
            strategy_name,
            events: Vec::new(),
            total_payload_bytes: 0,
            final_digest: [0u8; 32],
        })
    }

    /// Record a step in the trace.
    pub fn record_event(
        &mut self,
        name: impl Into<String>,
        dimension_charges: Vec<(Dimension, u64)>,
        payload: &[u8],
    ) -> Result<(), ReplayError> {
        let next_event_count =
            self.events
                .len()
                .checked_add(1)
                .ok_or(ReplayError::LimitExceeded {
                    resource: "event count",
                    limit: MAX_REPLAY_EVENTS,
                })?;
        check_len(next_event_count, MAX_REPLAY_EVENTS, "event count")?;
        let event_name = name.into();
        check_text_len(event_name.len(), MAX_EVENT_NAME_BYTES, "event-name bytes")?;
        // Every replay event must charge at least one dimension. An empty
        // dimension_charges vec would record a "phantom" event with no
        // accounting effect and no per-dimension integrity surface; reject
        // it at the trust boundary so a wire-imported ReplayLog cannot
        // smuggle zero-impact events past the integrity check.
        if dimension_charges.is_empty() {
            return Err(ReplayError::EmptyText {
                resource: "dimension-charge count",
            });
        }
        check_len(
            dimension_charges.len(),
            MAX_EVENT_CHARGES,
            "dimension-charge count",
        )?;
        check_len(
            payload.len(),
            MAX_EVENT_PAYLOAD_BYTES,
            "event-payload bytes",
        )?;
        if dimension_charges.iter().any(|(_, amount)| *amount == 0) {
            return Err(ReplayError::ZeroCharge);
        }
        let payload_len = u64::try_from(payload.len()).map_err(|_| ReplayError::LimitExceeded {
            resource: "total event-payload bytes",
            limit: MAX_REPLAY_PAYLOAD_BYTES as usize,
        })?;
        let total_payload_bytes = self.total_payload_bytes.checked_add(payload_len).ok_or(
            ReplayError::LimitExceeded {
                resource: "total event-payload bytes",
                limit: MAX_REPLAY_PAYLOAD_BYTES as usize,
            },
        )?;
        if total_payload_bytes > MAX_REPLAY_PAYLOAD_BYTES {
            return Err(ReplayError::LimitExceeded {
                resource: "total event-payload bytes",
                limit: MAX_REPLAY_PAYLOAD_BYTES as usize,
            });
        }
        let step_index =
            u64::try_from(self.events.len()).map_err(|_| ReplayError::LimitExceeded {
                resource: "event count",
                limit: MAX_REPLAY_EVENTS,
            })?;
        let payload = payload.to_vec();
        let outcome_digest = event_digest(step_index, &event_name, &dimension_charges, &payload)?;

        self.events.push(ReplayEvent {
            step_index,
            event_name,
            dimension_charges,
            payload,
            outcome_digest,
        });
        self.total_payload_bytes = total_payload_bytes;
        Ok(())
    }

    /// Seal and finalize the replay transcript digest.
    pub fn finalize(&mut self) -> Result<[u8; 32], ReplayError> {
        self.validate_events()?;
        let digest = self.computed_final_digest()?;
        self.final_digest = digest;
        Ok(digest)
    }

    /// Verify that every replay field remains bound to the sealed digest.
    pub fn verify_integrity(&self) -> bool {
        self.validate_events().is_ok()
            && self
                .computed_final_digest()
                .is_ok_and(|digest| self.final_digest == digest)
    }

    /// Verify that a candidate replay log matches this reference transcript bit-for-bit.
    pub fn verify_replay_match(&self, candidate: &ReplayLog) -> bool {
        self.verify_integrity() && candidate.verify_integrity() && self == candidate
    }

    fn validate_events(&self) -> Result<(), ReplayError> {
        check_text_len(
            self.strategy_name.len(),
            MAX_STRATEGY_NAME_BYTES,
            "strategy-name bytes",
        )?;
        if self.events.len() > MAX_REPLAY_EVENTS {
            return Err(ReplayError::LimitExceeded {
                resource: "event count",
                limit: MAX_REPLAY_EVENTS,
            });
        }
        let mut total_payload_bytes = 0u64;
        for (index, event) in self.events.iter().enumerate() {
            let expected = u64::try_from(index).map_err(|_| ReplayError::LimitExceeded {
                resource: "event count",
                limit: MAX_REPLAY_EVENTS,
            })?;
            if event.step_index != expected {
                return Err(ReplayError::StepIndexMismatch {
                    expected,
                    actual: event.step_index,
                });
            }
            check_text_len(
                event.event_name.len(),
                MAX_EVENT_NAME_BYTES,
                "event-name bytes",
            )?;
            if event.dimension_charges.is_empty() {
                return Err(ReplayError::EmptyText {
                    resource: "dimension-charge count",
                });
            }
            check_len(
                event.dimension_charges.len(),
                MAX_EVENT_CHARGES,
                "dimension-charge count",
            )?;
            total_payload_bytes = total_payload_bytes
                .checked_add(u64::try_from(event.payload.len()).map_err(|_| {
                    ReplayError::LimitExceeded {
                        resource: "total event-payload bytes",
                        limit: MAX_REPLAY_PAYLOAD_BYTES as usize,
                    }
                })?)
                .ok_or(ReplayError::LimitExceeded {
                    resource: "total event-payload bytes",
                    limit: MAX_REPLAY_PAYLOAD_BYTES as usize,
                })?;
            if total_payload_bytes > MAX_REPLAY_PAYLOAD_BYTES {
                return Err(ReplayError::LimitExceeded {
                    resource: "total event-payload bytes",
                    limit: MAX_REPLAY_PAYLOAD_BYTES as usize,
                });
            }
            if event.outcome_digest
                != event_digest(
                    event.step_index,
                    &event.event_name,
                    &event.dimension_charges,
                    &event.payload,
                )?
            {
                return Err(ReplayError::EventDigestMismatch {
                    step_index: event.step_index,
                });
            }
        }
        if total_payload_bytes != self.total_payload_bytes {
            return Err(ReplayError::PayloadByteCountMismatch);
        }
        Ok(())
    }

    fn computed_final_digest(&self) -> Result<[u8; 32], ReplayError> {
        let canonical_log = serde_json::to_vec(&(
            self.initial_seed,
            &self.strategy_name,
            &self.events,
            self.total_payload_bytes,
        ))
        .map_err(|error| ReplayError::Serialization(error.to_string()))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.replay.log.v3:");
        hasher.update(&canonical_log);
        Ok(*hasher.finalize().as_bytes())
    }
}

fn check_len(actual: usize, limit: usize, resource: &'static str) -> Result<(), ReplayError> {
    if actual > limit {
        Err(ReplayError::LimitExceeded { resource, limit })
    } else {
        Ok(())
    }
}

fn check_text_len(actual: usize, limit: usize, resource: &'static str) -> Result<(), ReplayError> {
    if actual == 0 {
        Err(ReplayError::EmptyText { resource })
    } else {
        check_len(actual, limit, resource)
    }
}

fn event_digest(
    step_index: u64,
    event_name: &str,
    dimension_charges: &[(Dimension, u64)],
    payload: &[u8],
) -> Result<[u8; 32], ReplayError> {
    let canonical_event = serde_json::to_vec(&(step_index, event_name, dimension_charges, payload))
        .map_err(|error| ReplayError::Serialization(error.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.replay.event.v2:");
    hasher.update(&canonical_event);
    Ok(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::{Cx, cx::cap::None as CapNone};
    use fsym_assumptions::ImmutableAssumptionsSnapshot;
    use fsym_budget::{Budget, BudgetLimits, DIMENSION_COUNT};
    use fsym_core::{BigInt, BigRational, Expr, Symbol};
    use fsym_polys::{
        factorization::{metered_complete_factorization, metered_kronecker_factorization},
        univariate::UnivariatePoly,
    };
    use fsym_proof_kernel::{Claim, ProofKernel, verify_derivation_independent};
    use std::sync::{Arc, OnceLock};

    use crate::{
        FsymCpuCx, FsymCx, NamedConcurrentStrategy, PortfolioCandidate, PortfolioError,
        run_portfolio_concurrent_race,
    };

    #[test]
    fn real_factor_race_reexecution_matches_observed_transcript_and_accounting() {
        let polynomial = |coefficients: &[i64]| {
            UnivariatePoly::new(
                Symbol::new("x"),
                coefficients
                    .iter()
                    .map(|coefficient| BigRational::from_integer(BigInt::from(*coefficient)))
                    .collect(),
            )
        };
        let input = polynomial(&[4, 0, 0, 0, 1]);
        // Fix the product independently of either generator's output.
        let expected_product = Expr::Mul(vec![
            polynomial(&[2, -2, 1]).to_expr(),
            polynomial(&[2, 2, 1]).to_expr(),
        ]);
        let requested = Claim::AlgebraicIdentity {
            lhs: input.to_expr(),
            rhs: expected_product.clone(),
        };
        let limits = BudgetLimits::uniform(1_000_000_000, 100_000);
        let run = || {
            // Each invocation owns fresh, equal contexts and budget ledgers.
            let context = Arc::new(ImmutableAssumptionsSnapshot::empty());
            let raw = Cx::detached_cancel_context();
            let mut cx = FsymCx::new(&raw, Budget::new(limits), limits);
            let observations = Arc::new([OnceLock::new(), OnceLock::new()]);
            let strategies: Vec<NamedConcurrentStrategy<CapNone>> = ["zassenhaus", "kronecker"]
                .into_iter()
                .enumerate()
                .map(|(index, name)| {
                    let input = input.clone();
                    let context = Arc::clone(&context);
                    let observations = Arc::clone(&observations);
                    let strategy: crate::ConcurrentStrategyRunner<CapNone> =
                        Box::new(move |cx: &mut FsymCpuCx<'_, CapNone>| {
                            let before = Dimension::ALL.map(|dimension| cx.remaining(dimension));
                            let (scale, mut factors) = if name == "zassenhaus" {
                                let factorization = metered_complete_factorization(&input, cx)
                                    .map_err(|error| {
                                        PortfolioError::AllStrategiesFailed(error.to_string())
                                    })?;
                                (
                                    factorization.scale,
                                    factorization
                                        .factors
                                        .into_iter()
                                        .map(|factor| (factor.poly, factor.multiplicity))
                                        .collect::<Vec<_>>(),
                                )
                            } else {
                                let factorization = metered_kronecker_factorization(&input, cx)
                                    .map_err(|error| {
                                        PortfolioError::AllStrategiesFailed(error.to_string())
                                    })?;
                                (
                                    factorization.scale,
                                    factorization
                                        .factors
                                        .into_iter()
                                        .map(|factor| (factor.poly, factor.multiplicity))
                                        .collect::<Vec<_>>(),
                                )
                            };
                            factors.sort_by(|(left, _), (right, _)| {
                                left.degree()
                                    .cmp(&right.degree())
                                    .then_with(|| left.coeffs.cmp(&right.coeffs))
                            });
                            let mut terms = Vec::new();
                            if scale != BigRational::from_integer(BigInt::from(1)) {
                                terms.push(Expr::Rational(scale));
                            }
                            for (poly, multiplicity) in factors {
                                let term = poly.to_expr();
                                terms.push(if multiplicity == 1 {
                                    term
                                } else {
                                    Expr::Pow(
                                        Arc::new(term),
                                        Arc::new(Expr::Integer(BigInt::from(
                                            u64::try_from(multiplicity)
                                                .expect("multiplicity fits u64"),
                                        ))),
                                    )
                                });
                            }
                            let product = match terms.len() {
                                0 => Expr::from_i64(1),
                                1 => terms.pop().expect("one factor"),
                                _ => Expr::Mul(terms),
                            };
                            let lhs = input.to_expr();
                            let mut kernel = ProofKernel::new((**context).clone());
                            let root = kernel
                                .prove_definitional_reduction(
                                    lhs.clone(),
                                    product.clone(),
                                    "polynomial_ring_equivalence",
                                    cx,
                                )
                                .map_err(|error| {
                                    PortfolioError::AllStrategiesFailed(error.to_string())
                                })?;
                            let candidate = PortfolioCandidate {
                                strategy_name: name.into(),
                                result: product.clone(),
                                claim: Claim::AlgebraicIdentity { lhs, rhs: product },
                                derivation: kernel.export_derivation(root).map_err(|error| {
                                    PortfolioError::AllStrategiesFailed(error.to_string())
                                })?,
                            };
                            let consumed = Dimension::ALL.map(|dimension| {
                                before[dimension.index()] - cx.remaining(dimension)
                            });
                            observations[index]
                                .set((candidate.clone(), consumed))
                                .expect("each worker records one candidate");
                            Ok(candidate)
                        });
                    (name, strategy)
                })
                .collect();
            let outcome = run_portfolio_concurrent_race(&mut cx, &context, &requested, strategies)
                .expect("both real factor generators must complete and a product must verify");
            let observations = Arc::try_unwrap(observations)
                .expect("scoped workers must release observation storage")
                .map(|observation| {
                    observation
                        .into_inner()
                        .expect("each real generator must produce its metered candidate")
                });
            let remaining = Dimension::ALL.map(|dimension| cx.remaining(dimension));
            let verifier_remaining = cx.verifier_remaining();
            let mut consumed = [0; DIMENSION_COUNT];
            for (candidate, charges) in &observations {
                assert_eq!(candidate.result, expected_product);
                assert_eq!(candidate.claim, requested);
                assert_eq!(
                    verify_derivation_independent(&candidate.derivation, &context).unwrap(),
                    requested,
                );
                assert!(charges[Dimension::ComputeSteps.index()] > 0);
                for dimension in Dimension::ALL {
                    consumed[dimension.index()] += charges[dimension.index()];
                }
            }
            for dimension in Dimension::ALL {
                assert_eq!(
                    remaining[dimension.index()],
                    limits.dimensions[dimension.index()] - consumed[dimension.index()],
                );
            }
            assert_eq!(
                outcome.generator_steps_consumed(),
                consumed[Dimension::ComputeSteps.index()],
            );
            assert_eq!(outcome.result(), &expected_product);
            assert_eq!(outcome.evidence().claim, requested);
            assert_eq!(outcome.winning_strategy(), "zassenhaus");
            assert!(outcome.evidence().verify_integrity());
            assert!(verifier_remaining > 0 && verifier_remaining < limits.verifier_pool);

            // Registration order, not worker completion order, defines this transcript.
            // ReplayLog binds observations; it is not an adaptive schedule replay driver.
            let payload = serde_json::to_vec(&(
                &input,
                &requested,
                ["zassenhaus", "kronecker"],
                limits.dimensions,
                limits.verifier_pool,
                outcome.context_digest(),
                &observations,
                outcome.result(),
                &outcome.evidence().claim,
                outcome.winning_strategy(),
                outcome.generator_steps_consumed(),
                remaining,
                verifier_remaining,
            ))
            .unwrap();
            let charges = Dimension::ALL
                .into_iter()
                .filter_map(|dimension| {
                    let amount = consumed[dimension.index()];
                    (amount > 0).then_some((dimension, amount))
                })
                .collect();
            let mut log = ReplayLog::new(0, "zassenhaus-kronecker-quartic").unwrap();
            log.record_event("verified-factor-race", charges, &payload)
                .unwrap();
            log.finalize().unwrap();
            (outcome, remaining, verifier_remaining, observations, log)
        };

        let first = run();
        let second = run();
        assert_eq!(first.0.result(), second.0.result());
        assert_eq!(first.0.evidence().claim, second.0.evidence().claim);
        assert_eq!(first.0.winning_strategy(), second.0.winning_strategy());
        assert_eq!(
            first.0.generator_steps_consumed(),
            second.0.generator_steps_consumed(),
        );
        assert_eq!(first.1, second.1);
        assert_eq!(first.2, second.2);
        assert_eq!(first.3, second.3);
        assert!(first.4.verify_replay_match(&second.4));
    }

    #[test]
    fn dimension_charges_are_integrity_bound() {
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome")
            .unwrap();
        log.finalize().unwrap();
        assert!(log.verify_integrity());

        log.events[0].dimension_charges[0].1 = 300;

        assert!(!log.verify_integrity());
    }

    #[test]
    fn payload_mutation_is_detected_without_a_reference_log() {
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome")
            .unwrap();
        log.finalize().unwrap();

        log.events[0].payload[0] ^= 0xff;

        assert!(!log.verify_integrity());
    }

    #[test]
    fn duplicate_or_out_of_order_indices_cannot_be_resealed() {
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome")
            .unwrap();
        log.events[0].step_index = 7;

        assert_eq!(
            log.finalize(),
            Err(ReplayError::StepIndexMismatch {
                expected: 0,
                actual: 7,
            })
        );
    }

    #[test]
    fn total_payload_counter_is_integrity_bound() {
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        log.record_event("step", vec![(Dimension::ComputeSteps, 3)], b"outcome")
            .unwrap();
        log.finalize().unwrap();

        log.total_payload_bytes += 1;

        assert!(!log.verify_integrity());
        assert_eq!(log.finalize(), Err(ReplayError::PayloadByteCountMismatch));
    }

    #[test]
    fn replay_names_are_nonempty_schema_identifiers() {
        assert_eq!(
            ReplayLog::new(11, ""),
            Err(ReplayError::EmptyText {
                resource: "strategy-name bytes",
            })
        );

        let mut log = ReplayLog::new(11, "strategy").unwrap();
        assert_eq!(
            log.record_event("", vec![(Dimension::ComputeSteps, 1)], b"outcome"),
            Err(ReplayError::EmptyText {
                resource: "event-name bytes",
            })
        );
    }

    #[test]
    fn record_event_refuses_empty_dimension_charges() {
        // An empty dimension_charges vec would be a "phantom" event that
        // records a payload but charges nothing; the trust boundary
        // refuses it so wire-imported ReplayLogs cannot smuggle
        // zero-impact events past the integrity check.
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        assert_eq!(
            log.record_event("step", Vec::new(), b"outcome"),
            Err(ReplayError::EmptyText {
                resource: "dimension-charge count",
            })
        );
    }

    #[test]
    fn validate_events_rejects_zero_impact_event_in_deserialized_log() {
        // A wire-imported ReplayLog with an event whose dimension_charges
        // was deserialized as an empty vec must fail verify_integrity so
        // the rejection lives on the read path as well as the write path.
        let mut log = ReplayLog::new(11, "strategy").unwrap();
        // Hand-build a phantom event bypassing record_event so the test
        // exercises the integrity gate independently of the input gate.
        log.events.push(ReplayEvent {
            step_index: 0,
            event_name: "phantom".into(),
            dimension_charges: Vec::new(),
            payload: b"outcome".to_vec(),
            outcome_digest: [0u8; 32],
        });
        assert!(!log.verify_integrity());
        assert_eq!(
            log.finalize(),
            Err(ReplayError::EmptyText {
                resource: "dimension-charge count",
            })
        );
    }
}
