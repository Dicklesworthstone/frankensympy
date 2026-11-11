# Native certification and search sessions

This development interface connects the root Rust library and the `frankensympy`
NDJSON command to the existing original-source interval checker. Its scope is
exact real residual/Jacobian enclosures, unique nonsingular local roots and
complete finite-box root sets, objective/gradient/Hessian enclosures, locally
classified stationary points, global box optimum value bounds, proper finite real
definite integrals, complete parameter-root families and implicit sensitivities.
Portable artifacts cover local roots, root isolation, stationary points, global
optimization, integrals and candidate search checkpoints. This interface does not close the complete Certified
Jacobian campaign or certify a SymPy compatibility profile.

## Run an exact root check

Build and start the command with the repository's pinned toolchain and donor
checkouts:

```sh
python3 scripts/fetch_workspace_donors.py
cargo run --locked --bin frankensympy
```

Send these four lines to the same process. Each line receives one JSON response
immediately, while stdin remains open:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:sqrt2","residuals":["x^2-2"],"variables":["x"]}}
{"id":"bounds","type":"EncloseResidualJacobian","payload":{"schema_version":1,"system":"system:sqrt2","domain":[{"lower":"1.4","upper":"1.5"}],"precision_digits":24}}
{"id":"root","type":"CertifyRootBox","payload":{"schema_version":1,"system":"system:sqrt2","certificate":"certificate:sqrt2","witness":{"domain":[{"lower":"1.4","upper":"1.5"}],"center":["1.45"],"preconditioner":[["1/3"]],"precision_digits":24}}}
{"id":"export","type":"ExportRootCertificate","payload":{"schema_version":1,"certificate":"certificate:sqrt2"}}
```

The enclosure response contains exact rational lower and upper endpoints for
the residual and every Jacobian entry over the entire box. This claim does not
assert a root exists. The root response is accepted only after the independent
checker verifies original-source C1 domains, strict Krawczyk inclusion and an
exact contraction bound. Its claim is
`unique_nonsingular_real_root_in_box`; it makes no assertion about other roots
outside that box. For this example the contraction bound is `1/15`.

The rational matrix is an untrusted preconditioner proposal. Its entries, a
small residual, a solver convergence flag or a previously stored acceptance
flag cannot authorize a root certificate.

## Certify a solution for every parameter and bound its sensitivity

`RegisterParametricProblem` freezes original residuals and two separate ordered
coordinate groups: unknowns and parameters. `CertifyParametricRoot` checks an
explicit uniform witness; `CertifyParametricCover` automatically constructs and
independently checks a complete parameter subdivision when a single witness is
insufficient. Both publish a `family:` handle only after the complete claim and
response have been admitted.

For `F(x,p)=x-p`, the following session certifies every `p` in `[-1,1]`, with a
unique nonsingular `x` in `[-2,2]`, then proves the exact sensitivity `dx/dp=1`:

```json
{"id":"register","type":"RegisterParametricProblem","payload":{"schema_version":1,"problem":"parametric:affine","residuals":["x-p"],"unknowns":["x"],"parameters":["p"]}}
{"id":"family","type":"CertifyParametricRoot","payload":{"schema_version":1,"problem":"parametric:affine","family":"family:affine","witness":{"unknown_domain":[{"lower":"-2","upper":"2"}],"parameter_domain":[{"lower":"-1","upper":"1"}],"center":["0"],"preconditioner":[["1"]],"precision_digits":16}}}
{"id":"sensitivity","type":"EncloseParametricSensitivity","payload":{"schema_version":1,"family":"family:affine","sensitivity":"sensitivity:affine"}}
```

The family response has `kind: "parametric_family_certified"`,
`evidence: "certified_numeric"` and claim
`for_every_parameter_exactly_one_nonsingular_root_in_unknown_box`. It retains
the complete unknown and parameter domains, a root enclosure containing all
parameter-dependent roots, the maximum local contraction bound, leaf count,
precision and assumptions-context digest. For this example the root enclosure
is `[-1,1]` and the contraction bound is zero. Strict interior inclusion is
required in every unknown coordinate, including at parameter endpoints.

The sensitivity response has `kind: "parametric_sensitivity_enclosed"` and claim
`implicit_solution_jacobian_enclosure`. Its exact interval matrix has unknowns
as rows and parameters as columns, in registration order. The example returns
`[[{"lower":"1","upper":"1"}]]`. Sensitivity calculation independently
recomputes the original-source Jacobian blocks on every accepted parameter
cell, including the full contraction remainder. It consumes an immutable
accepted family; a user-supplied derivative matrix or sampled root is not an
input to this operation. A sensitivity refusal leaves its input family valid.

For an automatic family that needs adaptive parameter subdivision, use:

```json
{"id":"register-adaptive","type":"RegisterParametricProblem","payload":{"schema_version":1,"problem":"parametric:adaptive","residuals":["(1+p^2)*x-1"],"unknowns":["x"],"parameters":["p"]}}
{"id":"cover","type":"CertifyParametricCover","payload":{"schema_version":1,"problem":"parametric:adaptive","family":"family:adaptive","unknown_domain":[{"lower":"0","upper":"2"}],"parameter_domain":[{"lower":"0","upper":"2"}],"precision_digits":16,"max_leaves":128}}
{"id":"cover-sensitivity","type":"EncloseParametricSensitivity","payload":{"schema_version":1,"family":"family:adaptive","sensitivity":"sensitivity:adaptive"}}
```

Search uses ordinary resources to propose rational centers, preconditioners and
a complete dyadic parameter cover. Independent verification uses the protected
pool on the same cumulative ledger, reconstructs both children of every split,
and checks every original-source cell. All leaves retain the same unknown box,
so uniqueness identifies the same family at shared parameter faces. An
unresolved cell returns `inconclusive` with no partial family. Original-source
poles, including held zero factors, remain domain obligations throughout.

Each group must be nonempty, the residual count must equal the unknown count,
and at most sixteen total coordinates are admitted. Unknown intervals have
positive width; parameter intervals may be points. Precision is 8..128 digits,
and covers have at most 1024 leaves. A session retains at most 32 parametric
problems, 32 accepted families and 32 sensitivity receipts. The normal exact
scalar, source, output, cancellation and cumulative-budget limits also apply.

The claim concerns the declared unknown and parameter boxes. It grants neither
an exact symbolic solution nor roots outside the unknown box, a global
continuation path, or requested accuracy of the derivative enclosure. Family
and sensitivity handles are resident session objects; this slice does not
define a portable family artifact.

Native Rust callers construct `ParametricProblem::from_exprs` for typed symbols
or `ParametricProblem::parse_sources` for bounded raw sources. They can call
`verify_root_box`, or `propose_cover` followed by `verify_cover`, with
caller-owned meters. The core sensitivity functions accept only checked family
receipts. `Session::parametric_problem`, `Session::parametric_family` and
`Session::parametric_sensitivity` expose immutable objects; the accepted family
view also offers `enclose_sensitivity` with an explicit meter.

## Enclose an objective and certify a stationary point

Register a scalar objective with its explicit coordinate order. Enclosure and
certification recompute second-order interval derivatives from the immutable
original expression, including all its domain restrictions:

```json
{"id":"register","type":"RegisterObjective","payload":{"schema_version":1,"objective":"objective:quadratic","source":"(x-1)^2+2*(y+2)^2","variables":["x","y"]}}
{"id":"bounds","type":"EncloseObjective","payload":{"schema_version":1,"objective":"objective:quadratic","domain":[{"lower":"0","upper":"2"},{"lower":"-3","upper":"-1"}],"precision_digits":24}}
{"id":"point","type":"CertifyStationaryPoint","payload":{"schema_version":1,"objective":"objective:quadratic","certificate":"stationary:minimum","witness":{"domain":[{"lower":"0","upper":"2"},{"lower":"-3","upper":"-1"}],"center":["1","-2"],"preconditioner":[["1/2","0"],["0","1/4"]],"precision_digits":24}}}
{"id":"export","type":"ExportStationaryCertificate","payload":{"schema_version":1,"certificate":"stationary:minimum"}}
```

`objective_enclosed` returns exact rational intervals for the value, gradient
and Hessian on the complete box, without asserting a stationary point.
`stationary_point_certified` proves one unique nonsingular stationary point
inside the box through strict Krawczyk inclusion and an exact contraction
bound. Here the point enclosure is exactly `(1, -2)`, the Hessian is
`diag(2, 4)`, the contraction bound is zero, and `classification` is
`strict_minimum`. The preconditioner is an untrusted proposal.

Classification can also be `strict_maximum`, `saddle`, or `unclassified`.
The last variant retains the stationary-point proof without asserting an
extremum. These are local claims, not global optimality or complete enumeration
of stationary points. Objective-value bounds cover the whole supplied box.
Singular stationary points may refuse. All original children are checked, so
`x^2+0*(1/x)` cannot certify a point at zero.

In a fresh session, register the expected objective and replay the unchanged
exported artifact against the expected box:

```json
{"id":"replay","type":"ReplayStationaryCertificate","payload":{"schema_version":1,"objective":"objective:quadratic","certificate":"stationary:received","domain":[{"lower":"0","upper":"2"},{"lower":"-3","upper":"-1"}],"artifact":"<the exact exported result.artifact string, JSON-escaped by your client>"}}
```

The receiver binds the complete typed source, ordered symbol identities, box
and context before bounded witness decoding. It independently recomputes the
stationary enclosure and classification with protected verification resources.
The artifact stores no classification, derivative bounds or accepted flag.
Even adding a constant changes the objective statement; moving the box returns
`domain_mismatch`. Failed verification or output admission publishes no handle
and retains all consumed resources.

Native Rust callers use `ObjectiveProblem::new` for typed sources or
`ObjectiveProblem::parse_source` for raw source syntax, then `enclose` or
`verify_stationary` with a caller-owned meter. The session uses the protected
meter from the same cumulative ledger for acceptance. `Session::objective_problem`
and `Session::stationary_point` expose read-only sources and accepted receipts.
`export_stationary_certificate` and `replay_stationary_certificate` provide
the same transport in Rust; replay requires the receiver's expected objective,
domain and context.

## Find a global minimum or maximum on a finite box

`OptimizeBox` searches an immutable scalar objective over the complete closed
box. It returns an exact feasible point and a certified interval containing the
global minimum or maximum **value**. The reported `gap` bounds the point's
absolute objective-value error and is at most the requested positive exact
`absolute_tolerance`.

For example, the objective below has two irrational global minimizers. No
initial guess or stationary-point preconditioner is required:

```json
{"id":"register","type":"RegisterObjective","payload":{"schema_version":1,"objective":"objective:well","source":"(x^2-2)^2","variables":["x"]}}
{"id":"optimize","type":"OptimizeBox","payload":{"schema_version":1,"objective":"objective:well","optimization":"optimization:minimum","domain":[{"lower":"-2","upper":"2"}],"extremum":"minimum","absolute_tolerance":"1/1000","precision_digits":16,"max_leaves":512}}
{"id":"export","type":"ExportBoxOptimization","payload":{"schema_version":1,"optimization":"optimization:minimum"}}
```

The response has `kind: "box_optimized"`, `evidence: "certified_numeric"` and
claim `global_box_optimum_with_absolute_value_gap_bound`. Its fields distinguish
the optimum-value interval `value` from the original objective's interval
`point_value` at the exact rational `point`. It also retains the requested box,
minimum/maximum direction, tolerance, partition leaf count, precision and
assumptions-context digest. `extremum: "maximum"` requests the corresponding
global maximum. Boundary optima and fixed coordinates with equal endpoints
are supported.

Search produces an untrusted point and complete dyadic partition using ordinary
resources. The independent core checker reconstructs every partition leaf and
reevaluates the original source and point using protected verification resources
from the same cumulative ledger. It visits all regions, including those pruned
by search. Only a complete checked response can publish an
`optimization:<identifier>` handle; failed verification or output admission
keeps that handle free and does not refund consumed work.

This lane admits 1..16 real coordinates, precision 8..128 and 1..1024 leaves.
Supported original source expressions must be C2 on a neighborhood of every
closed cell. Poles, unsupported functions and unresolved Piecewise seams refuse.
An exhausted partition returns `inconclusive` without an accepted partial or
local result. The receipt does not assert exact optimizer coordinates,
coordinate-distance error, uniqueness or a complete set of optimizing points.

To replay in a fresh session, register the receiver's expected original source
and send the exported bytes with the exact expected statement:

```json
{"id":"replay","type":"ReplayBoxOptimization","payload":{"schema_version":1,"objective":"objective:well","optimization":"optimization:received","domain":[{"lower":"-2","upper":"2"}],"extremum":"minimum","absolute_tolerance":"1/1000","artifact":"<the exact exported result.artifact string, JSON-escaped by your client>"}}
```

Export carries the original source, ordered typed axes, box, direction,
tolerance, context and bounded candidate witness. It stores no optimum value,
point value, gap or accepted flag. Replay compares the receiver's statement
before decoding bounded exact coordinates, then independently reconstructs the
complete cover. A changed direction or tolerance returns `statement_mismatch`;
changed source, box or context retain their respective mismatch errors.
Noncanonical encodings, unknown fields and incomplete partitions are rejected.

Native callers use `ObjectiveProblem::propose_box_optimization` and
`ObjectiveProblem::verify_box_optimization` with caller-owned meters and the
existing `BoxOptimizationOptions`. `Session::box_optimization` exposes accepted
receipts. `export_box_optimization` and `replay_box_optimization` provide the
same canonical transport in Rust. Large proofs use `BeginArtifactExport` with
`{"box_optimization":"optimization:minimum"}` and the existing bounded chunk
transfer; `ReplayBoxOptimizationFromArtifact` supplies the same expected
statement and a completed import transfer instead of inline bytes.

## Certify a proper real definite integral

Register exactly one scalar source and one variable, then provide finite exact
endpoints and a positive absolute tolerance. The tolerance bounds the **width
of the final rational enclosure**, rather than a heuristic quadrature estimate.
For example, these requests integrate `x^2` from zero to three:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:integrand","residuals":["x^2"],"variables":["x"]}}
{"id":"integrate","type":"CertifyIntegral","payload":{"schema_version":1,"system":"system:integrand","integral":"integral:area","lower":"0","upper":"3","absolute_tolerance":"1/1000","precision_digits":16,"max_intervals":128}}
{"id":"export","type":"ExportIntegralCertificate","payload":{"schema_version":1,"integral":"integral:area"}}
```

The `integral_certified` response has `enclosure: {"lower":"9","upper":"9"}`,
`evidence: "certified_numeric"`, and claim
`proper_real_definite_integral_with_absolute_enclosure_width_bound`. Its exact
bounds, requested tolerance, precision and interval count accompany the result.
The accepted receipt is available through `Session::integral_certificate`.

The generator proposes a partition using ordinary resources. A separate core
checker verifies complete coverage and original-source C2 bounds on every closed
cell, including endpoints, using the protected verifier pool. This lane admits
one finite real interval and supported twice-continuously-differentiable sources;
improper integrals, principal values and complex contours are outside its claim.
Reversed endpoints negate the enclosure. Equal endpoints still require the
original source to be admitted at that point. Thus `0*(1/x)` at zero cannot
become a certified zero integral by cancellation.

`max_intervals` is between 1 and 1,024. Exhausting subdivision before attaining
the tolerance returns `inconclusive`, with no accepted integral handle. Invalid
domains, unsupported functions and resource exhaustion retain their own errors.
Both generation and verification must succeed before response admission and
publication; a zero or exhausted verifier pool cannot accept a result.

To replay the exported partition, a fresh session registers its own expected
integrand and sends the unchanged `result.artifact` string:

```json
{"id":"replay","type":"ReplayIntegralCertificate","payload":{"schema_version":1,"system":"system:integrand","integral":"integral:received","lower":"0","upper":"3","absolute_tolerance":"1/1000","artifact":"<the exact exported result.artifact string, JSON-escaped by your client>"}}
```

Replay binds the original typed source, axis identity, ordered endpoints,
absolute tolerance and receiving context before independently checking the
partition. A changed tolerance or reversed interval is a different statement
and returns `statement_mismatch`. The artifact contains no cached integral,
enclosure or accepted flag. Export uses ordinary resources; receiving-session
canonical validation and mathematical replay use protected verification.

## Find every root in a finite box

The session can search automatically with no initial guesses or preconditioner.
For example, send these lines to a fresh process using the default CLI budgets:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:polynomial","residuals":["x^2-2"],"variables":["x"]}}
{"id":"isolate","type":"IsolateRealRootsInBox","payload":{"schema_version":1,"system":"system:polynomial","isolation":"isolation:roots","domain":[{"lower":"-2","upper":"2"}],"precision_digits":16,"max_leaves":512}}
{"id":"export-negative","type":"ExportRootCertificate","payload":{"schema_version":1,"certificate":"certificate:roots.root-0"}}
```

The isolation response has `kind: "real_roots_isolated"`, `root_count: 2`, exact
root enclosures, and a complete independently checked cover of the requested
closed box. The roots receive `certificate:roots.root-0` and
`certificate:roots.root-1`; either can be exported and independently replayed
through the existing root-certificate operations. Each child export carries
that local root claim. The full coverage receipt is available in-process through
`Session::root_isolation` and can be exported with `ExportRootIsolation`.

A successful `root_count: 0` proves the whole requested box is root-free.
`inconclusive` instead means the search could not establish completeness, for
example because of a singular root, an outer-boundary root or an exhausted
subdivision limit. It does not return a successful partial root list. Source
domain, budget and cancellation errors retain their separate categories.

Search spends ordinary compute; independent complete-cover verification spends
protected compute. All child labels, collisions, resident capacities and response
resources are checked before either the complete receipt or its child handles
are published. Child handles refer into the immutable accepted receipt, so they
do not duplicate its retained source forest.

## Advance a root search in bounded batches

A live session can retain an unfinished search and advance it over several
requests. After registering `system:polynomial` as above, begin a search:

```json
{"id":"begin","type":"BeginRootIsolationSearch","payload":{"schema_version":1,"system":"system:polynomial","search":"search:roots","domain":[{"lower":"-2","upper":"2"}],"precision_digits":16,"max_leaves":512}}
{"id":"step","type":"AdvanceRootIsolationSearch","payload":{"schema_version":1,"search":"search:roots","max_cells":1}}
{"id":"inspect","type":"InspectRootIsolationSearch","payload":{"schema_version":1,"search":"search:roots"}}
```

Repeat the advance request until `result.progress.ready_for_verification` is
true. `max_cells` bounds committed cell transitions per call, from 1 through
2,047. One cell can require several bounded interval calculations, so this is
a work quantum rather than a wall-clock deadline. Inspection returns current
progress without advancing the search.

Progress reports `processed_cells`, `pending_cells`, `candidate_root_count`,
`excluded_cells`, `leaf_count` and `ready_for_verification`. These replies carry
`evidence: "heuristic_candidate"`; a candidate count is not an accepted root
count, and readiness means only that a complete proposal is available. Finish
the search to request independent verification:

```json
{"id":"finish","type":"FinishRootIsolationSearch","payload":{"schema_version":1,"search":"search:roots","isolation":"isolation:roots"}}
{"id":"release","type":"ReleaseRootIsolationSearch","payload":{"schema_version":1,"search":"search:roots"}}
```

Only a successful finish returns `real_roots_isolated` and publishes the complete
receipt and child certificate handles. It uses the same protected complete-cover
checker and atomic publication path as one-shot isolation. The accepted result
supports all existing local-root and complete-proof export/replay operations,
including chunked transfer. Finishing before readiness returns `search_not_ready`.

The search freezes the original typed equations, ordered coordinates, domain,
precision, options and Session context association. Later workspace bindings
cannot alter that problem. A cell is removed from the pending frontier only
after its exclusion, root proposal or split is fully prepared and admitted.
Cancellation or resource refusal preserves the current unresolved cell and any
earlier completed cells. If a later cell in a batch refuses, inspection exposes
the progress already committed earlier in that batch.

All advances share the Session's cumulative ledger and the search's cumulative
internal work limits. Yielding, retrying or releasing a search resets neither.
An exhausted Session budget remains exhausted. The prepared source and scratch
reservation are retained across batches instead of being recreated for every
cell. Singular roots, unsupported source domains and subdivision limits can
still prevent completion.

Progress-response resources are admitted before advancing any cell. A refused
finish, including output admission or a child-label collision, leaves the search
available and publishes no partial accepted set. Successful finish also retains
the search until explicit release. Sessions admit at most eight resident
searches; release frees a slot and leaves already accepted certificates intact.

Native callers can use a cancellation-aware `BudgetMeter`, retain the search
after cancellation, and resume under the same cumulative ledger once cancellation
is cleared. The synchronous command does not expose an asynchronous cancellation
endpoint. A client can export the retained search for a later process as follows.

## Save and restore an unfinished root search

After registering `system:polynomial`, begin and pause a candidate search:

```json
{"id":"begin","type":"BeginRootIsolationSearch","payload":{"schema_version":1,"system":"system:polynomial","search":"search:paused","domain":[{"lower":"-2","upper":"2"}],"precision_digits":16,"max_leaves":512}}
{"id":"step","type":"AdvanceRootIsolationSearch","payload":{"schema_version":1,"search":"search:paused","max_cells":2}}
{"id":"save","type":"ExportRootIsolationSearch","payload":{"schema_version":1,"search":"search:paused"}}
```

The `root_isolation_search_exported` response carries
`evidence: "heuristic_candidate"` and a canonical `result.artifact` string.
The client may write that string unchanged to a UTF-8 file, terminate the
sender, and read it back later. Save the artifact value itself without
pretty-printing or appending a newline. Register the same original source in the new
session, then supply the expected box and the **same** precision and leaf limit:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:polynomial","residuals":["x^2-2"],"variables":["x"]}}
{"id":"restore","type":"RestoreRootIsolationSearch","payload":{"schema_version":1,"system":"system:polynomial","search":"search:received","domain":[{"lower":"-2","upper":"2"}],"precision_digits":16,"max_leaves":512,"artifact":"<the exact saved result.artifact string, JSON-escaped by your client>"}}
{"id":"continue","type":"AdvanceRootIsolationSearch","payload":{"schema_version":1,"search":"search:received","max_cells":2047}}
{"id":"finish","type":"FinishRootIsolationSearch","payload":{"schema_version":1,"search":"search:received","isolation":"isolation:received"}}
```

Restoration compares the original typed source, ordered axes, exact domain,
frozen options and context before decoding the bounded prefix. The independent
core checker rechecks its root proposals and exclusions and reconstructs the
remaining cells. The returned `root_isolation_search_restored` progress is still
a `heuristic_candidate`, even when the saved traversal was complete. Restoring
uses ordinary resources and requires no verifier lease; only a successful
`FinishRootIsolationSearch` can publish accepted roots through protected
complete-cover verification.

Checkpoint copies retain prior intrinsic step, allocation and evaluation
consumption. Restoration adds preparation and prefix-validation work, then
continues under the unchanged lifetime limits, including **256 MiB of cumulative
intrinsic allocation accounting**. Repeated saves/restores can exhaust that
limit even when the current process has ample memory and a larger Session budget.
The receiver separately pays its own decoding and restoration costs. Counter
validation checks exact source accounting and independently provable minimum
work; an unauthenticated checkpoint cannot establish its sender's full execution
history, and its counters never grant or replenish receiving allowances.

These operations provide explicit portable continuation bytes. They do not
automatically publish files, manage a durable store, recover after a crash, or
apply RaptorQ repair. Those integration gates remain separate.

## Replay the complete root set in another process

Export the accepted coverage proof from the originating session:

```json
{"id":"export-all","type":"ExportRootIsolation","payload":{"schema_version":1,"isolation":"isolation:roots"}}
```

The `root_isolation_exported` response contains a canonical `result.artifact`
JSON string. A fresh receiving process independently registers its expected
equations and supplies its requested exact box when replaying that string:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:polynomial","residuals":["x^2-2"],"variables":["x"]}}
{"id":"replay-all","type":"ReplayRootIsolation","payload":{"schema_version":1,"system":"system:polynomial","isolation":"isolation:received","domain":[{"lower":"-2","upper":"2"}],"artifact":"<the exact exported result.artifact string, JSON-escaped by your client>"}}
```

Success returns `real_roots_isolated` after independently rechecking the entire
cover. It publishes `isolation:received` and, for this two-root example,
`certificate:received.root-0` and `certificate:received.root-1`. Those child
certificates support the same local export/replay operations as discovered
roots. The receiving labels are local handles and are not part of the portable
mathematical statement.

The same operations transport a verified zero-root result, including its
complete exclusion cover. Merely omitting roots or supplying an acceptance flag
cannot establish that a box is empty. Replay binds the source, ordered typed
axes, exact domain and receiving context before independently verifying the
proof. A different requested box receives `domain_mismatch`; the label remains
available for a corrected request. Failed verification, resource admission or
child-label collision publishes neither the complete receipt nor any children.

## Replay in another process

The export response's `result.artifact` is a JSON **string** containing canonical
typed sources, the ordered typed variables, the assumption-context digest and
the root witness. Keep that string unchanged. It contains no accepted flag,
claimed contraction bound or claimed root enclosure.

In a fresh process, register the expected system independently, then send:

```json
{"id":"replay","type":"ReplayRootCertificate","payload":{"schema_version":1,"system":"system:sqrt2","certificate":"certificate:received","artifact":"<the exact result.artifact string, JSON-escaped by your client>"}}
```

The receiver compares the complete typed source and ordered symbol identities
with its own registered problem, checks its own assumption-context digest,
validates the canonical transport and independently recomputes the root claim.
Changing the equations, axis order, universe, schema, witness or nested fields
cannot reuse the sender's acceptance. Unknown fields and duplicate request
keys refuse. Native artifact replay also rejects noncanonical JSON, including
pretty-printing or changed field order; clients should pass the exported
string through their JSON encoder without parsing and rewriting its contents.

## Transfer certificates and checkpoints larger than one frame

All five artifact kinds support a chunked route up to the native 1 MiB artifact
limit. This route preserves the 64 KiB input frame limit. Each chunk carries
at most 16 KiB of bytes encoded as at most 32 KiB of lowercase hexadecimal;
schema 1 has at most 64 chunks. Clients can forward the returned manifest and
chunk objects directly, without decoding, reserializing or concatenating the
canonical artifact themselves.

After obtaining an accepted `isolation:roots`, start an immutable export in
the sending session:

```json
{"id":"begin","type":"BeginArtifactExport","payload":{"schema_version":1,"transfer":"transfer:out","source":{"root_isolation":"isolation:roots"}}}
{"id":"read-0","type":"ReadArtifactChunk","payload":{"schema_version":1,"transfer":"transfer:out","index":0}}
```

Select the source and receiving operation for the object being transferred:

| `BeginArtifactExport` source | Manifest `kind` | Receiving operation |
| --- | --- | --- |
| `{"root_certificate":"certificate:sqrt2"}` | `root_certificate` | `ReplayRootCertificateFromArtifact` |
| `{"root_isolation":"isolation:roots"}` | `root_isolation` | `ReplayRootIsolationFromArtifact` |
| `{"integral_certificate":"integral:area"}` | `integral_certificate` | `ReplayIntegralCertificateFromArtifact` |
| `{"root_isolation_search":"search:paused"}` | `root_search_checkpoint` | `RestoreRootIsolationSearchFromArtifact` |
| `{"stationary_certificate":"stationary:minimum"}` | `stationary_certificate` | `ReplayStationaryCertificateFromArtifact` |
| `{"box_optimization":"optimization:minimum"}` | `box_optimization` | `ReplayBoxOptimizationFromArtifact` |

The begin response's `result.manifest` contains `schema_version`, `kind`,
`total_bytes` and `content_id`. The read response's `result.chunk` contains
`index`, a 32-byte `digest` array and `data_hex`. A search export takes a
candidate snapshot; later advances do not change the exported transfer.

Use the following sequence in the receiving session, which registers its own
expected residual system before replay:

| Request | Payload in addition to `schema_version: 1` |
| --- | --- |
| `BeginArtifactImport` | `transfer: "transfer:in"`, `manifest: <the complete returned manifest object>` |
| `AppendArtifactChunk` | `transfer: "transfer:in"`, `chunk: <the complete returned chunk object>` |
| `ReplayRootIsolationFromArtifact` | `system: "system:polynomial"`, `isolation: "isolation:received"`, `domain: [{"lower":"-2","upper":"2"}]`, `transfer: "transfer:in"` |
| `ReleaseArtifactTransfer` | `transfer: "transfer:in"` |

Read and append chunks in index order, starting at zero. Wait for each append
response before advancing. Both the import-start and append responses report
`progress: {"received_bytes": ..., "next_chunk": ...}`. An exact retry of an
earlier chunk returns the current progress; a conflicting retry or a future
chunk refuses without changing the received prefix. Reads are repeatable and
every retry still consumes cumulative resources. Transfer progress lives in
the current session; automatic persistence and recovery of an unfinished
transfer are not supplied by these operations.

For a local root, the final replay request is instead:

```json
{"id":"replay","type":"ReplayRootCertificateFromArtifact","payload":{"schema_version":1,"system":"system:sqrt2","certificate":"certificate:received","transfer":"transfer:in"}}
```

After importing an integral certificate, use:

```json
{"id":"replay-integral","type":"ReplayIntegralCertificateFromArtifact","payload":{"schema_version":1,"system":"system:integrand","integral":"integral:received","lower":"0","upper":"3","absolute_tolerance":"1/1000","transfer":"transfer:in"}}
```

After importing a search checkpoint, use:

```json
{"id":"restore-search","type":"RestoreRootIsolationSearchFromArtifact","payload":{"schema_version":1,"system":"system:polynomial","search":"search:received","domain":[{"lower":"-2","upper":"2"}],"precision_digits":16,"max_leaves":512,"transfer":"transfer:in"}}
```

The complete content ID is checked before replay or restoration.
Per-chunk digests bind the full content ID, chunk index and chunk bytes; the
content ID binds the artifact kind, length and bytes. Neither successful
transfer nor a matching digest creates a certificate or a search handle.
Root, integral, stationary-point and global-optimization replay require protected independent verification of
the canonical artifact and its expected statement. Checkpoint restoration
rechecks the candidate prefix on ordinary resources; protected finish remains
necessary for acceptance. Using a checkpoint transfer as a proof fails with
`artifact_kind_mismatch`.

Imports remain available after successful or failed replay and restoration,
including response-budget refusal or child-label collision. Release transfers explicitly
when finished on **both** sessions. Imports and exports share eight resident
slots; release frees a slot without refunding any spent budget. Existing
transfer labels cannot be overwritten. A response resource refusal before
publication leaves the corresponding transfer or object state intact.

## Source and object semantics

`RegisterResidualSystem` and `RegisterObjective` convert source text into native `Expr` objects once
and freezes those objects and their coordinate order. Text is input syntax;
typed expressions are the retained mathematical state. Later `Bind`, patch,
fork or checkout requests do not substitute into a registered problem.

The `parse_raw` core entry point retains terms and factors that ordinary
construction can simplify away. Thus `0*(1/x)`, `log(x)-log(x)` and
`0/(x-x)` still undergo their original domain checks. Finite decimal literals
are exact rational numbers. No binary64 conversion enters this interface.

Session labels have distinct kinds: `system:<identifier>`,
`certificate:<identifier>`, `isolation:<identifier>`,
`integral:<identifier>`, `search:<identifier>`, `objective:<identifier>`,
`stationary:<identifier>`, `optimization:<identifier>` and `transfer:<identifier>`.
They are local references, not stable content IDs. Identifiers use ASCII letters,
digits, underscores, hyphens or dots. All labels except isolation labels are
at most 128 bytes; isolation labels are at most 104 bytes,
leaving space for generated
`.root-<index>` child certificate suffixes. Existing labels cannot be overwritten.
Only a successful independent check and successful response admission insert
a certificate into the resident store. A failed attempt leaves its label
available for a corrected request.

## Resources and limits

Every operation uses the session's cumulative ledger. Registration, initial
exact input preparation, artifact transfer and integrity checks, certificate
export and response preparation consume ordinary resources. Candidate search,
integral partition generation, and checkpoint export/restoration also use
ordinary resources. Checking that grants accepted mathematical evidence,
including receiving-session canonical certificate validation and replay,
consumes the protected verifier compute pool through `Budget::verifier_meter`;
memory and allocation limits remain shared.
Failure retains performed work. Reacquiring a meter or retrying a request never
replenishes either pool.

The CLI accepts explicit startup limits:

```sh
cargo run --locked --bin frankensympy -- \
  --compute-steps 2000000 \
  --memory-bytes 1073741824 \
  --allocations 1000000 \
  --verifier-steps 100000000
```

These are also the defaults. Memory is conservative logical accounting, not a
measurement of the process's live heap. Each original-source checker reserves
a bounded scratch allowance, so repeated checks can exhaust memory accounting
even after earlier temporary buffers have been released. Requests have no
budget-reset operation. The Rust `Session::new(SessionBudgets)` interface keeps
its existing small defaults; numerical callers should use
`Session::with_budget_limits` or `with_context_and_budget_limits`.

| Boundary | Limit |
| --- | --- |
| Input NDJSON frame | 64 KiB |
| Request ID | Optional string of at most 256 bytes, or unsigned integer |
| Resident systems / certificates / complete isolations | 32 / 128 / 32; child certificates share the certificate limit |
| Resident integral certificates | 32 |
| Resident objectives / stationary certificates | 32 / 128 |
| Resident global box optimization receipts | 32 |
| Resident artifact transfers | 8 total imports and exports; explicit release frees a slot |
| Resident root searches | 8; explicit release frees a slot without a budget refund |
| Root-search work quantum | 1–2,047 committed cell transitions per advance |
| Root-search intrinsic allocation lifetime | 256 MiB cumulative, including snapshot copies and restoration work |
| Residuals / variables | 1–64 / 1–16 |
| Residual × variable entries | At most 1,024 |
| Automatic isolation leaves / roots | At most 1,024 / 128 |
| Definite integral sources / variables / intervals | Exactly 1 / 1; at most 1,024 intervals |
| Source text per residual | 4 KiB, also subject to cumulative admission cost |
| Typed source depth / total nodes | 32 / 16,384 |
| Exact scalar input | At most 512 bytes: integer, decimal such as `0.5`, or integer fraction such as `1/3` |
| Precision request | 8–128 decimal digits |
| Inline certificate or search-checkpoint artifact | 24 KiB before outer JSON escaping |
| Native or chunked certificate / search-checkpoint artifact | 1 MiB, JSON depth at most 96 |
| Artifact chunk | 16 KiB of bytes, 32 KiB lowercase hex; at most 64 chunks |
| Certification response | 1 MiB including bounded envelope metadata |

The inline export/replay/restore operations retain their single-frame subset. Use
`BeginArtifactExport` and `BeginArtifactImport` for larger artifacts. The artifact
and chunk caps are upper bounds; a caller must still admit the operation under
its cumulative ordinary, memory, allocation and verifier limits.

Typical error codes distinguish `budget_exhausted`, `domain_not_certified`,
`unsupported_operation`, `source_mismatch`, `domain_mismatch`, `universe_mismatch`,
`statement_mismatch`, `invalid_artifact`, `unsupported_schema`, `inconclusive`
and a mathematical `refused` outcome.
An incomplete root search returns `search_not_ready` when asked to finish.
Transfers additionally distinguish `incomplete_transfer`, `out_of_order_chunk`,
`conflicting_chunk`, `artifact_kind_mismatch` and `wrong_transfer_mode`.
An error never contains an accepted root, integral, stationary-point or global-optimization value.

## Proper real definite integration in Rust

For a `ResidualProblem` with exactly one source and one typed variable, call
`propose_integral(lower, upper, absolute_tolerance, options, meter)` on ordinary
resources, then `verify_integral(lower, upper, absolute_tolerance, witness, meter)`
with the protected meter from the same `Budget`. The arguments are exact
`BigRational` values; `QuadratureOptions` contains `precision_digits` and
`max_intervals`. The first call returns only a `QuadratureWitness`. The second
independently checks original-source C2 domains, coverage and enclosure width
before returning an `IntegralEnclosureReceipt`.

`export_integral_certificate(problem, receipt, context_digest, meter)` emits
canonical bytes. Use `replay_integral_certificate(bytes, expected_problem,
expected_lower, expected_upper, expected_tolerance, expected_context, meter)`
to recompute the accepted enclosure without running the adaptive generator.
The receiving caller supplies its own exact statement and protected verifier
meter. Source, axis, bounds, tolerance and context are bound before bounded
partition decoding; cached enclosures and acceptance flags are absent.

## Automatic bounded root isolation in Rust

`ResidualProblem::isolate_roots(domain, options, proposal_meter, verification_meter)`
uses the existing exact subdivision search and independent complete-cover checker.
It needs no initial guesses, hand-written centers or preconditioner matrices.
Accepted results contain every real root in the declared finite closed box, with
an independently checked nonsingular local root enclosure for each. A zero-root
receipt requires a verified exclusion cover of the complete domain.

The two meters keep candidate work separate from acceptance. A session or other
caller sharing one `Budget` can call `propose_isolation` on ordinary resources,
then `verify_isolation` through `Budget::verifier_meter`. Neither phase resets
spent resources. `RootIsolationOptions` specifies precision and a bounded leaf
limit (1 through 1,024). Singular roots, roots on the outer boundary, unresolved
source domains and exhausted subdivision can remain inconclusive; they never
produce a successful partial list or a false root-free result.

## Portable complete-cover proofs in Rust

`export_root_isolation(receipt, context_digest, meter)` exports the entire
complete-cover witness, including a proved empty root set. A receiving process
calls `replay_root_isolation(bytes, expected_problem, expected_domain,
expected_context, meter)` with its own typed equations, coordinate order,
exact requested box and context. Only a fresh independent complete-cover check
can construct the receiving `RootIsolationReceipt`.

The canonical schema binds the original typed sources, ordered symbol
identities, exact outer domain, context and witness. It contains no accepted
flag, asserted root count, claimed root enclosures or claimed contraction bounds.
The receiver compares the source, axes and domain before decoding the bounded
root proposals. It then validates the canonical encoding and rechecks every
local root, exclusion and cover cell. A proof for a smaller or different box
cannot establish completeness for the receiver's requested box.

Both root and complete-cover native artifacts are limited to 1 MiB and JSON
depth 96. A complete-cover witness is limited to 128 local root proposals and
2,047 partition tokens. Sequence lengths and root-witness scalar bounds are
checked during decoding; cumulative charges precede the bounded decoding work.
Receiving sessions should run replay through their protected verifier meter.
Failures retain performed work and produce no accepted receipt.

The root crate also exposes `ArtifactExport`, `ArtifactImport`, typed manifests,
chunks and IDs, and `artifact_content_id` / `artifact_chunk_digest` helpers for
native clients. These transport opaque bytes with caller-owned meters and
cancellation safe points. `ArtifactImport::checked_bytes` validates completeness,
kind and full content integrity on every attempt, returning borrowed bytes for
the caller's independent canonical replay. It does not return a mathematical
receipt. Export ownership avoids a full-payload clone; imports reserve one
bounded buffer at begin and publish each validated chunk with a final copy.

## Native root-search continuations

`ResidualProblem::begin_isolation_search(domain, options, meter)` constructs a
source-owning `RootIsolationSearch`. Its `advance(max_cells, meter)` method
retains completed work and reports `RootIsolationProgress`; `completed_witness()`
returns a borrowed proposal only when the frontier is empty. The existing
`ResidualProblem::verify_isolation` independently verifies that proposal using
the caller's protected verifier meter. Keep one cumulative caller-owned ledger
across advances and retries. The original `propose_isolation` / `isolate_roots`
APIs use the same cell-processing engine for one-shot execution.

`export_root_search_checkpoint(&mut search, context_digest, meter)` returns
canonical bytes suitable for a client-managed file. The mutable search is
required because snapshot copies consume intrinsic resources. Restore with
`restore_root_search_checkpoint(bytes, expected_problem, expected_domain,
expected_options, expected_context, meter)` on ordinary resources. This returns
a `RootIsolationSearch`, not a `RootIsolationReceipt`. Empty, partial and
complete candidate prefixes are admitted only after independent reconstruction;
all still require final protected verification to publish accepted evidence.

For callers retaining typed data, `RootIsolationSearch::checkpoint` returns a
`RootIsolationSearchCheckpoint` containing frozen options, the normalized
witness prefix and `RootIsolationCounters`; `from_checkpoint` independently
rechecks that prefix. The portable transport additionally binds the expected
raw typed source, axis order, box and context. The counters report intrinsic
steps, allocation bytes, source bytes and evaluations; restore adds new work
to the retained history under the existing lifetime caps. They do not replace
the receiver's meter or authenticate remote execution history.

The root crate exposes `ResidualProblem`, exact scalar/ball types, checker
receipts, typed session requests and the portable export/replay functions.
`ResidualProblem::new` accepts typed expressions and symbol identities;
`parse_sources` is the bounded plain-symbol convenience lane. The direct
enclosure, root verification and transport functions accept a caller-owned
`BudgetMeter`, including a region's cancellation-aware adapter. Their tests
exercise cancellation at every observed safe point and cumulative refusal.

The synchronous NDJSON session itself does not start background tasks or expose
an asynchronous cancellation request. A pure `Budget`/`VerifierMeter` has no
cancellation source. This interface does not imply completion of the wider
asupersync lifecycle, compiled-program admission, hybrid callback or performance
gates. The following optional file capability adds explicit checkpoint
publication and fresh-process resumption.

## Saving and resuming root searches on disk

Start the command with `frankensympy --checkpoint-dir ./root-checkpoints`, or
construct a Rust session with
`Session::with_budget_limits(limits, 256).with_checkpoint_directory(path)?`.
The host selects the directory. JSON requests cannot open arbitrary paths, and
ordinary sessions continue to return `persistence_disabled` for storage calls.

After registering a residual system and beginning or advancing a root search,
send `SaveRootIsolationSearch` with `schema_version: 1` and its `search` label.
The `RootIsolationSearchSaved` response includes a `checkpoint` object containing
the exact record digest and payload byte count. Retain that object with the
original equations, typed coordinates, requested box and search options.

A fresh process opens the same directory and registers its own expected
residual system. Send `LoadRootIsolationSearch` with `schema_version: 1`, that
`system`, a new `search` label, the expected `domain`, `precision_digits`,
`max_leaves`, and the returned `checkpoint` object. Loading selects that exact
saved record, including an earlier snapshot if later snapshots also exist.
Modification times and directory iteration order never choose the continuation.

The reader bounds the physical file before allocating, checks canonical
encoding and the complete locked dependency/donor manifest, then independently
reconstructs the original-source traversal prefix. Wrong equations, typed axis
order, context, box, options, forged counters and altered bytes refuse before a
resident search is published. Restored state retains `heuristic_candidate`
evidence. Advance it normally and use `FinishRootIsolationSearch` to spend
protected verification resources and obtain accepted complete-cover evidence.

The Rust `RootSearchCheckpointStore` exposes the same boundary through
`prepare`, `commit` and `restore`. Preparation is in memory; the session admits
its response before writing a staging file. Commit reads back and validates the
exact record, synchronizes it, renames it into the committed namespace and
synchronizes the directories. Cancellation safe points precede publication;
there is no post-commit cancellation check that reports a successful commit as
cancelled. OS synchronization errors after rename can be indeterminate: inspect
or retry the exact content key instead of assuming that no file exists.

Storage, encoding and validation use conservative cumulative ordinary-resource
charges. Stored work counters cannot refill the receiving session or borrow its
protected verifier allowance. The durable envelope is bounded at 4 MiB and the
portable payload at 1 MiB; worst-case encoding admission may refuse a payload
near that maximum. No stored flag or successful file read grants mathematical
evidence. This local file reference lane does not assert the optional
FrankenSQLite adapter, recovery from arbitrary filesystem damage, or completion
of the entire campaign's persistence gate.

## Recovering a saved search after chunk loss

`SaveRepairableRootIsolationSearch` and `LoadRepairableRootIsolationSearch`
provide the same candidate lifecycle with independently stored RaptorQ source
and repair symbols. Enable the same host directory and explicitly fund the
codec work; for example, a small search can use:

```sh
frankensympy --checkpoint-dir ./root-checkpoints --compute-steps 100000000
```

The default two-million-step allowance intentionally refuses repair encoding.
The ordinary exact-record save/load operations remain usable with that default
for small searches. Codec charges scale with the bounded source block and
symbol width, so larger checkpoints can require more than the example budget.
All allowances remain cumulative across requests.

The repairable save request takes `schema_version: 1` and `search`. Its
`RootIsolationSearchSavedWithRepair` response returns a `checkpoint` object
with `manifest_digest`, `manifest_bytes` and `payload_bytes`. Retain this exact
object. The repairable load request takes the same expected `system`, new
`search`, `domain`, `precision_digits` and `max_leaves` fields as ordinary load,
with the repairable checkpoint object instead of an ordinary record key.

Each bundle contains at most 64 source symbols and exactly 16 repair symbols,
with a maximum 1 MiB payload, 64 KiB manifest and 16 KiB symbol width. Every
symbol is a separately bounded, canonical durable record. The exact manifest
is committed only after all symbols are written and read back successfully.
A failed save before manifest publication can leave up to 80 unreferenced
symbol records; this API does not garbage-collect shared records. The same
post-rename synchronization uncertainty described above applies to publication.

Restore first validates the exact manifest, context, dependency pins, lengths
and codec layout. Missing or corrupt symbol records become erasures; genuine
filesystem/backend failures remain errors. The codec recovers the payload when
the available equations suffice and checks its digest. Restore then regenerates
the complete deterministic manifest in memory and requires an exact match,
including the hashes and record identities of erased symbols. This reserves
both a decoder and an encoder pass and does not rewrite files. Sixteen repair symbols
are redundancy, not a guarantee that every pattern of sixteen missing source
symbols is recoverable: decoding can refuse insufficient or dependent equations.
An absent or invalid manifest always refuses; it is never replaced by another
snapshot found on disk.

`RootIsolationSearchRecovered` reports `source_chunks`, `repair_chunks`,
`source_missing`, `source_corrupt`, `repair_missing`, `repair_corrupt` and
`recovered`. The last field means missing/corrupt source bytes were reconstructed;
it remains false when all source records were intact. These are storage facts,
not mathematical evidence. Recovered bytes still pass independent original-source
checkpoint restoration, and the result retains `heuristic_candidate` evidence.
`FinishRootIsolationSearch` must verify the completed cover with protected
resources before any accepted roots are published.

Rust callers use `RepairableRootSearchCheckpointStore::{prepare, commit,
restore}`; restore returns the candidate search and `RootSearchRepairReport`.
Filesystem and codec calls are synchronous. Work is bounded and prepaid, with
cooperative cancellation between operations. This feature does not add a
background scheduler, automatic repair daemon, or a whole-session snapshot.
