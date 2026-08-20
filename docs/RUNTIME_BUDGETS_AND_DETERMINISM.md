# FrankenSymPy runtime, budgets, cancellation, and determinism

**Status:** normative architecture contract  
**Scope:** asupersync integration, capability contexts, nested symbolic budgets, cancel-correct execution, deterministic scheduling, replay, continuations, admission control

## 1. Runtime doctrine

Symbolic computations can be tiny, explosive, recursive, branch-heavy, memory-hungry, or effectively nonterminating. Timeouts bolted onto synchronous recursive code do not make them safe. FrankenSymPy treats resource sovereignty and cancellation as part of every algorithm's contract.

All concurrent, asynchronous, remote, persistent-I/O, or potentially blocking work uses asupersync. There is no Tokio runtime, detached thread pool, or hidden executor.

## 2. `Cx` as capability and execution context

Every operation that can block, spawn, allocate from shared resources, invoke a Python hook, access persistence, or consume a significant budget receives `&Cx` or a narrower capability derived from it.

The context carries or references:

```text
MathCx
├── cancellation state
├── deadline/virtual time
├── budget tree
├── execution region
├── determinism/replay policy
├── profile/context/rule universe
├── security capabilities
├── persistence capabilities
├── remote-worker capabilities
├── tracing/receipt sink
└── admission/priority class
```

Read-only contexts cannot express publication or mutation. Local-only contexts cannot dispatch remote work. Compatibility calls receive a profile-bound context. Verifiers receive a narrower context than generators.

## 3. Structured region tree

The task tree mirrors the mathematical request:

```text
request region
├── shell/lowering scope
├── planner scope
├── portfolio scope
│   ├── probes
│   ├── candidate generators
│   ├── verifiers
│   └── publication coordinator
├── lifting scope
├── persistence/checkpoint scope
└── receipt finalizer
```

Rules:

- every child has exactly one owning region;
- every region has a declared close policy;
- return requires region quiescence or a typed non-cooperative boundary outcome;
- no cache writer, verifier, repair task, or remote lease survives its owner accidentally;
- region IDs appear in receipts and deterministic traces.

## 4. Symbolic budget tree

A `MathBudget` is multidimensional and hierarchical.

### 4.1 Core dimensions

- wall-clock deadline;
- virtual-time deadline in lab mode;
- CPU/fuel steps;
- allocation bytes and peak live bytes;
- unique semantic nodes;
- transient surface nodes;
- expression tree/DAG growth;
- recursion and traversal depth;
- rewrite firings;
- e-classes/e-nodes/unions;
- coefficient height/limb operations;
- polynomial monomials and degree growth;
- modular primes/residues;
- matrix dimensions/fill/field operations;
- branch/condition partitions;
- assumptions queries and solver subcalls;
- proof nodes and certificate bytes;
- Python-hook calls and hook wall time;
- persistence bytes and fsyncs;
- remote packets, workers, bytes, and retries;
- RaptorQ source/repair symbols;
- output bytes and printer width/lines.

### 4.2 Reservations

Algorithms reserve expected resources before entering irreversible or high-amplification phases. A reservation can be:

- granted;
- partially granted with a smaller plan;
- denied with a typed reason;
- queued by admission control.

Reservation prevents many concurrent tasks from each assuming they can consume the same remaining memory.

### 4.3 Child budgets

A parent partitions budget among probes, generators, verifiers, and artifact work. Verifier budget is protected: a generator cannot consume the entire request and leave no resources to check its candidate.

Portfolio policies may reclaim unused child reservations and deterministically reallocate them according to the decision card.

### 4.4 Charging discipline

- charge before publication or large allocation when predictable;
- reconcile estimates with actual use;
- include temporary conversion and proof-generation costs;
- never reset counters on fallback;
- record per-strategy consumption;
- make cache hits charge parsing/validation rather than appear free;
- distinguish user budget from system maintenance budget.

## 5. Cancellation protocol

FrankenSymPy follows request → drain → finalize semantics.

### 5.1 Request

Cancellation becomes visible through the context. New speculative work and nonessential artifact generation stop being admitted.

### 5.2 Drain

Algorithms reach declared safe points, release reservations, flush or abandon private buffers, cancel child tasks, revoke remote leases, and leave shared state either unpublished or fully valid.

### 5.3 Finalize

The region emits a terminal outcome and receipt. A resumable algorithm may publish a verified checkpoint. The caller receives only after owned children drain.

## 6. Cancellation safe points

Each algorithm documents maximum work between safe points under supported input classes. Examples:

- big-integer multiplication recursion boundaries;
- modular-prime batches;
- polynomial term-block processing;
- Gröbner pair or matrix-reduction batches;
- e-graph iteration/rule batches;
- matrix pivot/block steps;
- precision-doubling rounds;
- proof-checker node batches;
- serialization chunks;
- remote packet boundaries.

Arbitrary Python callbacks are a special boundary. FrankenSymPy can cancel before/after them and isolate them in supervised execution, but does not claim a universal interruption bound for non-cooperative Python or foreign code.

## 7. Two-phase effects

Shared effects use reserve/commit or prepare/publish protocols.

### 7.1 Verified-result publication

1. reserve candidate slot;
2. verify exact claim;
3. atomically publish value + evidence + receipts;
4. release/abort losers.

### 7.2 Cache publication

1. construct private entry;
2. validate complete key and canonical payload;
3. verify evidence;
4. reserve shard capacity/generation;
5. publish atomically.

### 7.3 Checkpoint publication

1. serialize private checkpoint;
2. compute canonical digest;
3. optionally generate repair symbols;
4. write objects;
5. verify readback/digests according to policy;
6. publish manifest pointer.

Cancellation before commit leaves no authoritative partial effect.

## 8. Outcome model

Runtime and mathematical outcomes are composed rather than conflated.

```text
ExecutionOutcome<T>
├── Completed(MathOutcome<T>)
├── Cancelled(receipt, optional continuation)
├── TimedOut(receipt, optional continuation)
├── ResourceExhausted(dimension, receipt, optional continuation)
├── Refused(policy reason)
├── HookBoundaryUnresponsive(details)
└── InternalFault(details)
```

A mathematical `Inconclusive` can be a successfully completed execution. A timeout is not mathematical evidence. Compatibility lifting maps terminal states to the profile's expected unevaluated return or exception while native APIs preserve distinctions.

## 9. Determinism contract

For fixed canonical inputs, universe IDs, policy, seed, and deterministic mode:

- accepted semantic value is stable;
- stable IDs, proof/certificate, and terminal outcome are stable;
- result ordering and tie breaks are stable;
- decision card is stable in strict mode;
- persistent/replay bundle canonical bytes are stable except explicitly excluded transport metadata.

Wall-clock timing, process IDs, memory addresses, worker IDs, and scheduling races do not enter semantic identity.

## 10. Sources of nondeterminism and controls

| Source | Control |
|---|---|
| hash maps/sets | deterministic map or explicit sorted extraction |
| work stealing | semantic tie breaks independent of completion order |
| random algorithms | counter-based/recorded streams partitioned by strategy ID |
| floating reductions | declared reduction tree and rounding policy |
| remote worker arrival | verify then deterministic candidate selection policy |
| adaptive telemetry | freeze in strict mode; record in replay mode |
| Python hash seed | profile-controlled shell behavior, excluded from stable IDs |
| filesystem enumeration | canonical sorting |
| plugin registration | immutable registry IDs and conflict resolution |
| concurrent interning | content identity plus payload confirmation |
| timeout races | virtual time in lab; terminal boundary recorded in production |

## 11. Randomness

Randomness is a capability. Streams are derived from:

```text
request seed
strategy ID
subtask path
counter range
algorithm schema version
```

This permits reproducible parallelism without schedule-dependent stream consumption. Cryptographic randomness is used only for security boundaries and is never silently mixed into deterministic mathematical algorithms.

A compatibility profile can reproduce upstream randomness behavior where part of a public API, separately from native deterministic streams.

## 12. Lab runtime

The same runtime-facing code executes under deterministic lab contexts with:

- virtual time;
- deterministic task scheduling;
- DPOR/schedule exploration where applicable;
- injected cancellation at safe points;
- allocation/budget failure injection;
- channel close/reorder/delay within protocol constraints;
- persistence torn-write/corruption simulation;
- remote worker loss/duplication/byzantine payloads;
- RaptorQ symbol loss/corruption;
- Python-hook delay/exception simulations;
- seed-replayable failure traces.

The lab verifies runtime protocols, not mathematical truth by itself. Mathematical verifiers run inside the explored schedules.

## 13. Deterministic trace

A trace records logical events rather than raw logs:

```text
TraceEvent
├── logical sequence
├── region/task path
├── event kind
├── object/term/receipt IDs
├── budget delta
├── cancellation state
├── decision/publication transition
├── virtual or monotonic time
└── payload digest
```

Event kinds include spawn, reserve, charge, candidate, verify, reject, commit, cancel-request, drain, checkpoint, remote-lease, repair, and finalize.

Sensitive payloads are referenced by digest and capability-protected artifact, not dumped into logs.

## 14. Replay bundle

A self-contained replay bundle includes all non-rebuildable dependencies:

- request and canonical input objects;
- compatibility/profile manifest;
- assumptions context;
- domain, rule, algorithm, and verifier registries;
- policy/loss matrices;
- random stream roots;
- deterministic trace or adaptive decision card;
- required proof/certificate/checkpoint artifacts;
- environment/build fingerprint;
- expected terminal digest.

A bundle may omit rebuildable caches and indexes. Unknown/missing authoritative inputs cause refusal.

## 15. Resumable continuations

Long-running algorithms expose typed continuation schemas rather than serializing arbitrary stacks.

A continuation includes:

- operation and algorithm version;
- immutable input universe IDs;
- normalized algorithm state;
- completed subclaims and verified artifacts;
- remaining work frontier;
- consumed/remaining budget accounting;
- random counter ranges used/reserved;
- compatibility/profile constraints;
- canonical digest.

Resume validates every dependency. A continuation from a different rule/domain/verifier universe is refused or explicitly migrated with a verified migration receipt.

Checkpointability is designed into algorithms; it is not a memory dump.

## 16. Admission control

Services and agent swarms need global resource governance.

Admission inputs:

- request priority and deadline;
- declared maximum budget;
- current memory/CPU/remote capacity;
- expected amplification and proof cost;
- trust/security class;
- persistence quota;
- fairness/accounting identity.

Policies include weighted fair queuing, per-tenant quotas, deadline-aware ordering, and protected verifier capacity. Admission decisions are auditable and cannot alter mathematical evidence.

A request may be refused before lowering if its declared worst-case resources exceed policy. Hardened/native APIs explain which dimension blocked it.

## 17. Python execution boundary

Python hooks and shell-only algorithms run under a supervised lane:

- GIL/interpreter ownership is explicit;
- native worker tasks send bounded callback requests rather than calling Python unsafely;
- inputs/outputs are shell validated;
- exceptions/warnings are captured according to profile;
- callbacks have wall-time/call/output budgets;
- reentrancy is tracked;
- callback-induced recursive native requests inherit nested budgets;
- unresponsive callbacks produce a typed boundary outcome after all controllable children drain.

No claim of memory safety or cancellation extends through arbitrary third-party Python/C-extension behavior. Certified core paths avoid such callbacks or treat their results as assumptions/candidates.

## 18. Remote work

Remote execution uses capability-scoped, content-addressed work packets. The local coordinator:

- sends only required immutable objects and budgets;
- leases deterministic subtask/seed ranges;
- deduplicates repeated responses;
- treats workers as untrusted generators;
- verifies candidates locally;
- revokes/cancels leases on region closure;
- records transport and worker events outside semantic identity.

Network partitions can delay or waste work but cannot force publication of an unverified result.

## 19. Observability

Metrics are structured by operation, strategy, domain, profile, evidence class, outcome, architecture, and size bucket.

Important metrics:

- outcome mix including refusal/timeout/inconclusive;
- candidate rejection and verifier cost;
- expression/proof growth;
- budget-estimate error;
- cancellation drain latency;
- orphan count (must remain zero for controlled tasks);
- cache evidence-class hits;
- checkpoint/recovery success;
- remote duplicate/defect rates;
- selector shadow regret;
- compatibility drift.

Cardinality is bounded; raw term contents are not metric labels.

## 20. Runtime tests

### 20.1 Protocol tests

- cancel before/after each two-phase boundary;
- candidate and cancellation races;
- verifier rejection while other candidates complete;
- cache/checkpoint publication failure injection;
- nested budget exhaustion;
- child panic/fault containment;
- remote lease duplication/loss;
- hook reentrancy and exception paths.

### 20.2 Schedule exploration

- concurrent interning of identical/different terms;
- assumptions/cache reads across context changes;
- portfolio winner races;
- checkpoint compaction versus resume;
- persistent cache read versus invalidation;
- branch merge and proof publication;
- RaptorQ repair versus garbage collection.

### 20.3 Determinism tests

- repeated production runs under varied core counts;
- lab schedule permutations;
- randomized insertion/worker arrival orders;
- cross-process stable IDs and canonical bundles;
- fixed profile Python hash-seed matrices;
- scalar versus optimized verifier behavior.

## 21. Runtime claims and gates

Public runtime claims require exact gates:

- **“no orphan tasks”**: lab schedule suite plus production region-leak detector;
- **“cancel-correct”**: per-operation safe-point and two-phase publication matrix;
- **“deterministic”**: declared mode, universe, and byte/semantic comparator;
- **“resumable”**: crash/cancel/restart tests across version validation boundaries;
- **“bounded”**: every dimension exposed and adversarial amplification tests;
- **“distributed-safe”**: local re-verification and malicious worker corpus.

No broad adjective is accepted without scope.

## 22. Forbidden shortcuts

- spawning detached background work;
- using a global thread pool outside asupersync ownership;
- treating a wall-clock timeout wrapper as cancel correctness;
- letting a generator consume verifier-reserved budget;
- publishing a candidate before verification;
- caching partial mutable state as a completed result;
- serializing an arbitrary stack/heap image as a continuation;
- making completion order a semantic tie break;
- using process-local handles or timestamps in stable IDs;
- claiming bounded cancellation through arbitrary non-cooperative Python callbacks;
- ignoring temporary allocations and conversion costs;
- resetting budget counters on fallback;
- dropping loser tasks without draining;
- treating deterministic replay as proof of mathematical correctness;
- omitting refused/timed-out cases from operational metrics.

## 23. First runtime vertical slice

The first end-to-end slice must demonstrate:

1. a shell expression lowered under a profile-bound `Cx`;
2. a two-strategy exact polynomial portfolio;
3. protected verifier budget and two-phase winner publication;
4. cancellation injected at every declared boundary;
5. zero orphan tasks and no unverified cache entry;
6. a resumable checkpoint after partial modular work;
7. byte-stable replay under the lab runtime;
8. profile-correct lifted value and native evidence/receipt envelope;
9. remote-worker simulation with duplicate and invalid candidates;
10. metrics and traces that explain the outcome without leaking expression contents.
