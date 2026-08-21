# Adaptive selection and anytime monitoring

**Status:** normative architecture contract  
**Scope:** adaptive routing, conformal methods, e-processes, calibration, optional stopping, resets, censoring, multiplicity, decisions, fallbacks, and trust boundaries

## 1. Boundary

Adaptive systems may change what FrankenSymPy tries, when it hedges, what it measures, and when it falls back. They may not change mathematical truth, certificate validity, compatibility observations already recorded, or the authority of the reference verifier.

## 2. Monitor specification

Before activation, every monitor declares:

- stable ID/version and owner;
- operational question and action set;
- population and sampling rule;
- filtration/information available at each update;
- null and alternative or predictive target;
- score, nonconformity, likelihood ratio, or betting construction;
- parameter estimation/calibration source;
- type-I, coverage, false-discovery, or purely heuristic status;
- optional-stopping and optional-continuation scope;
- minimum sample/calibration rules;
- delayed/censored outcomes;
- subgroup and multiplicity policy;
- reset/retraining/change-point policy;
- numerical stability and overflow handling;
- persistence/replay format;
- actions at warning, reject, fault, and inconclusive states;
- deterministic safe fallback.

A monitor missing these fields is telemetry only.

## 3. Conformal methods

A conformal coverage claim requires the exact exchangeability or online-validity assumptions of the selected method. The implementation must use the correct finite-sample rank, including the `+∞` case where no finite order statistic provides the target coverage.

Calibration samples are not scored against the same fitted calibration set unless the method explicitly supports it. Profile/workload drift can invalidate exchangeability and must trigger a registered response.

Without these conditions, a quantile heuristic is called a quantile heuristic, not conformal prediction.

## 4. E-processes

An e-process claim requires a nonnegative supermartingale/e-variable construction under the named null and filtration. The implementation stores log e-values or another stable representation, records the rejection threshold, and preserves all updates.

A multiplicative score that merely grows on bad observations is not automatically an e-process. The expected-value bound must be justified for the actual data-generating assumptions and any adaptive parameter choices.

## 5. Optional stopping and resets

Anytime validity permits observation at arbitrary stopping times only within the construction’s assumptions. It does not permit deleting history or restarting after unfavorable evidence as if no test occurred.

Reset rules:

- reset creates a new monitor generation;
- prior evidence remains immutable and linked;
- promotion decisions name the generation used;
- repeated generations have an explicit multiplicity/evidence policy;
- quarantine cannot be cleared solely by resetting the monitor.

## 6. Censoring and delayed outcomes

Timeouts, cancellations, verifier refusals, crashes, resource exhaustion, and remote-worker loss remain observations. The monitor declares how each enters the stream.

Silently analyzing only successful completions creates selection bias and is prohibited.

## 7. Safe adaptive actions

Permitted actions include:

- select or reorder eligible generators;
- change speculative breadth;
- adjust hedge timing;
- route to a deterministic baseline;
- increase independent-check sampling;
- quarantine an optimized route;
- reduce concurrency or memory pressure;
- request recalibration;
- stop a benchmark campaign under its registered rule.

Prohibited actions include:

- accept a claim without verification;
- weaken assumptions or certificate requirements;
- relabel inconclusive as false/true;
- hide observed failures;
- rewrite historical evidence;
- automatically promote a research strategy to default.

## 8. Decision records

Every action records:

- monitor generation and state before update;
- observation and provenance root;
- update math and numerical status;
- e-value/threshold or prediction set as applicable;
- action and reason;
- policy root;
- resulting fallback/quarantine state;
- replay pointer.

Host timing and machine telemetry are separated from canonical decision fields.

## 9. Initial monitors

### Portfolio outcome regression

Detects sustained degradation in generator success, resource use, or verifier rejection rate. Default action is route quarantine or baseline fallback, never evidence downgrade.

### Compatibility drift

Tracks new divergences across profile/corpus updates. The authoritative facts are differential observations; the monitor prioritizes investigation and may block profile promotion.

### Verifier disagreement

Any disagreement between admitted checkers is an immediate fault/quarantine event. Statistical accumulation is unnecessary for correctness, though monitoring may characterize frequency.

### Obligation/resource leak

Tracks unresolved runtime obligations and cleanup latency. Exact balance invariants remain separately checked; e-process evidence may provide early operational warning.

### Performance regression

Uses same-route, pinned-corpus observations and regime covariates. Warn/reject actions affect route admission, not mathematical acceptance.

## 10. Determinism and replay

For a fixed observation sequence, monitor profile, and generation state, updates and actions are deterministic. If randomized betting or conformal variants are used, seed lineage is explicit and counter-partitioned.

## 11. Release gates

- formula/reference review;
- simulation under null and alternatives;
- adversarial optional stopping;
- reset and repeated-generation tests;
- censored/failure observation tests;
- numerical overflow/underflow tests;
- deterministic replay;
- action/fallback tests;
- wording lint;
- proof that monitor outputs cannot mint claim authority.
