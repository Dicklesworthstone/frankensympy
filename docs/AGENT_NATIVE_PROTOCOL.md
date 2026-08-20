# FrankenSymPy agent-native protocol and collaborative workspace

**Status:** normative architecture contract  
**Scope:** structured machine interfaces, semantic patches, typed requests/results, proof expansion, work packets, collaborative derivation branches, human-readable views

## 1. Design goal

Agents should not have to manipulate mathematical state through fragile strings, scrape pretty-printer output, or infer whether a method returned an exact answer, an unevaluated expression, or a heuristic guess.

FrankenSymPy exposes a versioned symbolic protocol whose primary objects are terms, contexts, claims, evidence, derivations, receipts, checkpoints, and bundles. Human-facing Python and printer APIs remain first-class, but the agent interface is structurally lossless and explicitly budgeted.

## 2. Interface families

### 2.1 In-process Rust API

Typed zero-copy or compact-handle access for native applications and other Franken-suite crates.

### 2.2 Python API

- compatibility shell matching immutable SymPy profiles;
- coexistable `frankensympy` native APIs;
- structured result/evidence objects;
- async and streaming adapters built on asupersync-backed execution.

### 2.3 NDJSON command protocol

A process-safe streaming interface for shells, agent harnesses, batch jobs, and language-neutral tools. Every line is one bounded envelope with a schema version and request/event ID.

### 2.4 Framed RPC

Binary or JSON framing over local transport/network for high-throughput applications. It uses the same semantic schemas as NDJSON, not a separate object model.

### 2.5 Portable bundles

Content-addressed bundles for terms, proofs, replay, checkpoints, counterexamples, conformance fixtures, and distributed work.

## 3. Protocol invariants

- every envelope names a schema version;
- every request has a stable request ID and optional parent/trace ID;
- unknown versions, object kinds, evidence classes, or claim schemas fail closed;
- inputs reference immutable universe IDs;
- inline objects are bounded before allocation;
- stream ordering and terminal events are explicit;
- cancellation targets request/region IDs;
- values and evidence are separate fields;
- printed strings are views, never authoritative IR;
- receipts expose every lowering, planner, verifier, cache, repair, and fallback boundary requested by policy;
- sensitive object content is capability-scoped.

## 4. Core object references

```json
{
  "kind": "term_ref",
  "term_id": "term:v1:...",
  "domain_id": "domain:v1:...",
  "schema": 1
}
```

Other typed references include:

- `surface_ref`;
- `context_ref`;
- `profile_ref`;
- `rule_registry_ref`;
- `algorithm_registry_ref`;
- `verifier_registry_ref`;
- `derivation_ref`;
- `receipt_ref`;
- `checkpoint_ref`;
- `bundle_ref`;
- `workspace_ref` and `branch_ref`.

Stringly typed IDs without kind prefixes are rejected at trust boundaries.

## 5. Request envelope

```json
{
  "schema": 1,
  "type": "request",
  "request_id": "req:...",
  "parent_request_id": null,
  "operation": "factor",
  "inputs": [{"term_id": "term:v1:..."}],
  "universe": {
    "profile_id": "sympy-1.14.0-cpython",
    "context_id": "ctx:v1:...",
    "rule_registry_id": "rules:v1:...",
    "algorithm_registry_id": "alg:v1:...",
    "verifier_registry_id": "verify:v1:..."
  },
  "requirements": {
    "minimum_evidence": "certificate_verified",
    "completeness": "irreducible_factorization",
    "result_form": "profile_canonical"
  },
  "budget": {
    "wall_ms": 10000,
    "memory_bytes": 1073741824,
    "term_nodes": 1000000,
    "proof_bytes": 67108864
  },
  "execution": {
    "determinism": "replay",
    "persistence": "local_durable",
    "allow_remote": false
  }
}
```

Fields omitted by a client receive policy defaults that are included in the accepted-request event and receipt.

## 6. Event stream

A request may emit:

- `request_accepted`;
- `lowering_started/completed`;
- `plan_ready`;
- `strategy_started/progress/candidate/failed`;
- `verification_started/accepted/rejected`;
- `checkpoint_published`;
- `repair_started/completed/failed`;
- `warning`;
- `budget_update`;
- `cancellation_requested/draining`;
- `terminal`.

Progress events are advisory and rate-limited. Only the terminal event defines the request outcome.

A client can request a terse stream that emits only accepted, checkpoint, warning, and terminal events.

## 7. Terminal result

```json
{
  "schema": 1,
  "type": "terminal",
  "request_id": "req:...",
  "execution_outcome": "completed",
  "math_outcome": "accepted",
  "value": {"term_id": "term:v1:..."},
  "claim": {
    "kind": "factorization",
    "source": "term:v1:...",
    "unit": "term:v1:...",
    "factors": [
      {"term_id": "term:v1:...", "multiplicity": 1}
    ],
    "domain_id": "domain:v1:...",
    "irreducible": true
  },
  "evidence": {
    "class": "certificate_verified",
    "verification_id": "verification:v1:...",
    "certificate_id": "certificate:v1:..."
  },
  "receipts": {
    "execution": "receipt:v1:...",
    "decision": "receipt:v1:...",
    "lowering": "receipt:v1:...",
    "lifting": "receipt:v1:..."
  },
  "replay_bundle": "bundle:v1:..."
}
```

A heuristic or conditional result uses a different `math_outcome`; it cannot populate accepted evidence fields.

## 8. Term transport

Clients can provide or retrieve terms in several forms:

- canonical binary term bundle;
- canonical JSON structural form for debugging/interchange;
- profile-bound Python pickle only within the shell compatibility interface;
- parsed text with an explicit parser/profile/domain policy;
- references to existing content-addressed terms.

Text parsing returns a surface graph and lowering receipt. A parse string is never treated as a stable term ID.

### 8.1 Structural JSON

```json
{
  "schema": 1,
  "operator": "core.add",
  "domain": "expr",
  "children": [
    {"operator": "core.symbol", "payload": {"name": "x"}},
    {"operator": "core.integer", "payload": {"decimal": "1"}}
  ]
}
```

Structural JSON is bounded, schema-validated, and canonicalized before receiving a `TermId`.

## 9. Proof and derivation APIs

Agents can:

- request proof summary or full expansion;
- traverse derivation parents/children;
- fetch certificate schemas and verification receipts;
- ask why a side condition is unresolved;
- request a minimal proof or a pedagogical explanation separately;
- query dependencies on assumptions/rules/registries;
- fork a proof branch;
- submit a candidate edge/certificate;
- request local re-verification.

A human explanation is a generated view over verified evidence. It is not itself the proof object.

## 10. Semantic patches

Text diffs are poorly suited to mathematical state. A `SemanticPatch` describes transformations over stable IDs:

```text
SemanticPatch
├── patch_id
├── base workspace/branch/version
├── universe IDs
├── operations
│   ├── add term/object
│   ├── add candidate derivation
│   ├── add verified derivation
│   ├── add/retract context fact
│   ├── add counterexample
│   ├── attach proof/certificate
│   ├── set branch result preference
│   └── request merge
├── preconditions
├── evidence dependencies
└── author/tool provenance
```

Patch application validates preconditions and never upgrades candidate edges. Same printed text with different domain/context/class is a conflict, not a match.

## 11. Collaborative branch workflow

### 11.1 Fork

An agent forks an immutable branch head and receives the exact universe manifest.

### 11.2 Explore

The agent requests bounded transformations, imports or constructs candidates, adds assumptions in a child context, and attaches counterexamples/proofs.

### 11.3 Review

Another agent or verifier checks claims independently. Review events are attached to derivation IDs, not line numbers in a printer output.

### 11.4 Merge

Merge verifies universe compatibility and all imported accepted edges. Conditional/candidate work remains labeled. Conflicts return typed witnesses.

### 11.5 Rebase

Rebase is explicit. The system computes which derivations remain valid under the new universe and which require replay or become invalid.

## 12. Counterexample bundles

A counterexample is first-class:

```text
CounterexampleBundle
├── challenged claim
├── exact or certified assignment/structure
├── assumptions/domain/profile
├── evaluation proof/enclosure
├── minimal reproducer
├── discovery receipt
└── affected rule/algorithm versions
```

Random sampled failures are minimized and then checked exactly or with certified numerics when possible. Counterexamples can automatically quarantine a rule or open a discrepancy without proving a replacement theorem.

## 13. Agent work packets

A bounded packet contains:

- objective and non-goals;
- exact base universe and branch;
- input object IDs;
- allowed operations/strategies;
- required evidence class;
- resource and output budgets;
- acceptance command/schema;
- required differential/metamorphic tests;
- forbidden shortcuts;
- expected discrepancy/claim-registry effects;
- dependency/workstream IDs;
- lease/expiration policy.

An agent cannot declare completion by prose. The coordinator evaluates the named objective artifacts and gates.

## 14. Beads conversion gate

An architectural workstream can become executable Beads work only when it has:

1. one bounded deliverable;
2. explicit dependencies and no hidden cyclic prerequisite;
3. exact files/crates or discovery task;
4. objective acceptance command(s);
5. unit, differential, metamorphic, adversarial, and benchmark obligations as applicable;
6. expected discrepancy and claim-registry updates;
7. forbidden shortcuts;
8. rollback/failure semantics;
9. evidence artifacts required for closure;
10. a named owner for the verifier/gate, not only the generator.

The work graph is single-writer for structural changes. JSONL is authoritative and derived databases are rebuildable.

## 15. Query and introspection operations

Agents can query:

- `describe_term` with bounded depth;
- `free_symbols`, binders, domains, assumptions, and opaque boundaries;
- `explain_lowering` and `explain_lifting`;
- `available_algorithms` with eligibility reasons;
- `plan_only` without execution;
- `verify_only` for supplied claims/certificates;
- `compare_surface` and `compare_semantic` separately;
- `find_derivations` and `find_counterexamples`;
- `estimate_budget` with uncertainty;
- `resume_checkpoint`;
- `export_replay_bundle`;
- `profile_discrepancies`;
- `claim_status`.

Introspection is capability- and budget-controlled. Huge proof graphs are paginated by stable cursors.

## 16. Batch requests

A batch declares dependency structure:

```json
{
  "type": "batch_request",
  "batch_id": "batch:...",
  "nodes": [
    {"id": "a", "request": {"operation": "differentiate", "...": "..."}},
    {"id": "b", "depends_on": ["a"], "request": {"operation": "factor", "input_from": "a.value"}}
  ],
  "failure_policy": "cancel_dependents",
  "budget": {"wall_ms": 60000}
}
```

The runtime owns the batch region. Independent nodes may run concurrently; dependent nodes start only after accepted outputs satisfying their evidence requirements. A heuristic output cannot flow into a node demanding verified exact input unless the edge explicitly accepts it.

## 17. Streaming large objects

Large matrices, polynomials, proof bundles, and generated code use chunked object streams:

- manifest first with total bounds and canonical object ID;
- numbered chunks with per-chunk digest;
- flow control and cancellation;
- optional RaptorQ repair symbols;
- final canonical digest validation;
- object publication only after complete validation.

Partial streams remain staging artifacts and cannot be referenced as complete terms/proofs.

## 18. Error model

Protocol errors are typed:

- schema/version unknown;
- malformed/bounds violation;
- object missing or wrong kind;
- universe mismatch;
- capability denied;
- unsupported operation/domain;
- compatibility profile unavailable;
- invalid semantic patch/precondition;
- verifier rejection;
- budget refusal/exhaustion;
- cancellation/timeout;
- persistence/repair failure;
- internal invariant fault.

Error responses include stable reason codes and bounded diagnostics. They do not rely solely on English messages.

## 19. Human-readable views

Every core object can produce bounded views:

- compact one-line summary;
- profile `str`/`repr`/pretty/LaTeX;
- canonical structural JSON;
- proof outline;
- decision-card explanation;
- resource timeline;
- dependency graph slice;
- uncertainty/unresolved-obligation summary.

View generation is not authoritative and is separately budgeted to prevent a printer from expanding an enormous object.

## 20. Version negotiation

Clients advertise supported:

- protocol schemas;
- term/certificate/bundle encodings;
- evidence and claim registries;
- compression/repair capabilities;
- authentication/capability mechanisms;
- maximum object limits.

The server selects a compatible version or refuses. It never silently drops unknown proof/evidence fields or downgrades an accepted claim.

## 21. Security

- no implicit code execution while decoding terms, proofs, or pickles;
- Python pickle is accepted only through an explicitly unsafe/profile-specific shell capability;
- callback/plugin identities and capabilities are declared;
- parsers preflight sizes and recursion;
- object references are authorization-scoped;
- content-addressed equality leakage is considered in multi-tenant stores;
- remote workers cannot obtain objects outside a packet capability;
- logs/metrics avoid raw formulas by default;
- semantic patches cannot alter registries or profiles without separate governance capabilities.

## 22. Deterministic agent replay

An agent session can export:

- initial workspace/branch manifest;
- all accepted request envelopes;
- semantic patches;
- decision/verification receipts;
- random stream roots;
- terminal outcomes;
- required object bundles;
- optional natural-language transcript as non-authoritative context.

Replaying reconstructs mathematical state independent of the chat transcript. Divergence is reported at the first semantic event/object digest.

## 23. Agent-focused conformance

Tests include:

- schema round trips in multiple languages/processes;
- unknown-version fail-closed behavior;
- stream cancellation and backpressure;
- duplicate/reordered event handling;
- semantic patch conflicts and idempotence;
- candidate-versus-verified flow typing;
- proof pagination and stable cursors;
- huge object/chunk bounds;
- malicious object references/capability denial;
- deterministic session replay;
- branch merge with domain/context/profile conflicts;
- profile-correct Python views from protocol terms;
- human explanation fidelity to proof outline.

## 24. Forbidden shortcuts

- using printed strings as term/proof identity;
- returning only natural-language explanations for proof requests;
- allowing heuristic candidates to populate accepted fields;
- omitting universe IDs from requests or patches;
- silently ignoring unknown evidence/claim fields;
- letting remote clients write verified edges without server verification;
- treating chat transcript order as authoritative workspace history;
- accepting arbitrary pickle/code execution through the normal protocol;
- publishing partial streamed objects;
- merging same-looking formulas without domain/context checks;
- marking agent work complete from a textual assertion rather than gates;
- exposing unbounded graph/proof dumps;
- providing unstable array-index references where stable IDs exist.

## 25. First protocol slice

The first slice must support:

1. create/fetch `Integer`, `Rational`, `Symbol`, `Add`, `Mul`, and `Pow` terms;
2. preserve a held Python surface expression while exposing its semantic term;
3. submit differentiation and polynomial factorization requests;
4. stream planner/candidate/verifier events;
5. return accepted, heuristic, inconclusive, cancelled, and resource-exhausted outcomes distinctly;
6. expand a verified proof/certificate;
7. fork a branch, apply a semantic patch, and merge a verified derivation;
8. export/replay a deterministic bundle;
9. reject an invalid remote candidate and an unknown schema;
10. render profile-correct Python/LaTeX views without using them as identity.
