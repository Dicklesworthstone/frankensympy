# Compatibility profiles and algorithm portfolios

**Status:** normative architecture contract  
**Scope:** observable compatibility, modes, diagnostics, eligibility, expected-loss selection, fallback, decision receipts, independent verification, and performance evidence

## 1. Two orthogonal axes

Compatibility policy and algorithm policy are independent.

- A compatibility profile defines observable Python behavior.
- An algorithm portfolio defines how native work is selected.

The same mathematical operation may run under strict SymPy behavior, native deterministic behavior, or certified behavior. The planner cannot change the profile to make a strategy eligible.

## 2. Observation bundle

A compatibility observation can include:

- value graph and exact types;
- identity/alias relationships;
- ordering and iteration sequence;
- hash and equality outcomes;
- exception and warning records;
- stdout/stderr where contractually relevant;
- repr/string/LaTeX/code-printer forms;
- pickled form and round-trip behavior;
- module/class/MRO metadata;
- mutation and cache effects;
- callback trace;
- timing only as telemetry, never parity.

Normalization must be explicit and profile-scoped. Unregistered normalization is a test bug.

## 3. Surface states

Each inventory row is one of:

- `present_verified`;
- `present_unverified`;
- `partial`;
- `delegated`;
- `known_divergent`;
- `missing`;
- `not_applicable`;
- `unknown`.

Only `present_verified` supports a parity claim for the named observation dimensions and corpus.

## 4. Portfolio state machine

```text
Unplanned
  -> Diagnosed
  -> EligibleSetBuilt
  -> StrategiesRunning
  -> CandidateProduced
  -> Verifying
  -> Verified
  -> Publishing
  -> Published

Any nonterminal state
  -> Cancelled | ResourceExhausted | Refused | InternalFault

CandidateProduced / Verifying
  -> FallbackEligibleSetBuilt
```

No path bypasses verification for a certified claim.

## 5. Hard eligibility

Eligibility is boolean and auditable. Examples:

- domain supported;
- assumptions sufficient;
- input within structural bounds;
- strategy deterministic under requested policy;
- required certificate family available;
- Python effects compatible with execution plan;
- verifier reserve available;
- target architecture supported;
- no quarantined implementation version.

Expected loss ranks only eligible actions.

## 6. Loss dimensions

Loss can combine registered, normalized terms:

- expected CPU and wall time;
- peak memory and allocation risk;
- proof/certificate size;
- verifier cost;
- cancellation waste;
- remote/relay cost;
- probability and cost of generator failure;
- compatibility divergence risk;
- numerical instability for approximate lanes;
- latency to first verifiable subclaim;
- energy/cost telemetry where measured.

Weights are profile/versioned policy, not hidden constants.

## 7. Fallback graph

Fallback edges declare:

- triggering outcomes;
- information carried forward;
- whether a new diagnostic pass is required;
- budget transfer;
- cache/checkpoint reuse;
- evidence class preserved or changed;
- maximum transitions to prevent loops.

Fallback never silently weakens a requested evidence class.

## 8. Decision receipt

Every nontrivial selection records:

- request/profile/universe roots;
- diagnostics and their evidence classes;
- eligible and rejected strategies with reason codes;
- loss-policy version;
- estimated loss vector;
- selected strategy and tie-break;
- launched speculative branches;
- cancellation and fallback events;
- actual resource observations;
- candidate and certificate roots;
- verifier outcome;
- publication result.

The receipt explains execution. It is not proof of the mathematical claim.

## 9. Calibration and monitoring

Adaptive updates are versioned policy events. Historical observations are append-only. Resetting a monitor creates a new generation and preserves prior evidence.

Failures, timeouts, cancellations, refusals, and resource exhaustion remain in the stream. Training data selection is explicit.

## 10. Certified request reserve

Before generator launch, reserve enough resources for:

- bounded certificate decode;
- reference verification;
- final canonical encoding;
- publication or refusal record;
- cleanup/drain.

A generator may not consume the verifier reserve. If the reserve cannot be made, the request is refused before expensive generation.

## 11. Determinism

Given the same request root, universe, profile, policy, seed lineage, and available artifact closure, strict deterministic execution produces the same canonical result and decision receipt modulo explicitly separated host telemetry.

Parallel completion order cannot choose between equally valid canonical results; registered tie-break policy does.

## 12. Performance admission

An optimized strategy enters the default portfolio only after:

- semantic/certificate equivalence;
- hostile and metamorphic tests;
- complete-operation benchmark win in its declared regime;
- no unacceptable tail or memory regression;
- cancellation/drain tests;
- calibrated routing precision against holdout workloads;
- rollback route and quarantine key.

## 13. Research strategies

Research strategies may run in shadow mode. They cannot publish accepted output unless their candidate passes an already admitted certificate verifier. Their performance and success observations cannot strengthen the verifier’s evidence class.
