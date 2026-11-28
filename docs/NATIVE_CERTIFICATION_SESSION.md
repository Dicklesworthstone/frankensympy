# Native certification and search sessions

This development interface connects the root Rust library and the `frankensympy`
NDJSON command to the existing original-source numerical and exact algebraic
checkers. Its scope is complete finite real polynomial and rational systems,
complete real rational equations, inequalities and Boolean constraint combinations,
exact real residual/Jacobian enclosures, continuous scalar root-existence brackets,
unique nonsingular local roots and
complete finite-box root sets, objective/gradient/Hessian enclosures, locally
classified stationary points, global box optimum value bounds, proper finite real
definite integrals, complete parameter-root families and implicit sensitivities.
Portable artifacts cover scalar existence brackets, local roots, root isolation, stationary points, global
optimization, integrals, candidate search checkpoints and compiled derivative
programs. This interface does not close the complete Certified
Jacobian campaign or certify a SymPy compatibility profile.

## Host-controlled cooperative cancellation

A native host can run a typed request with
`Session::certification_request_with_checkpoint`. Its callback supplies the
owning request's cancellation policy, for example a check of its asupersync
context:

```rust,ignore
let result = session.certification_request_with_checkpoint(request, || {
    if cx.is_cancel_requested() {
        Err(frankensympy::MeterError::Cancelled)
    } else {
        Ok(())
    }
});
```

The same callback runs at ordinary computation, protected independent
verification, and final output-admission safe points. Its first error stays
terminal for that invocation, and `CertificationSessionError::code()` reports
`"cancelled"` for cancellation. Newly registered sources, compiled programs,
accepted certificates, and released handles change only after the final
admission checkpoint. No cancellation check runs after a successful commit to
misreport an already published result as cancelled.

The session retains one cumulative budget across requests. Cancellation does
not refund earlier work or replenish the protected verifier pool. An incremental
search retains completed cells and its valid candidate frontier; the host can
inspect, export, resume, or release it in a subsequent request using the same
session. Acceptance still requires independent `Finish` verification. Dropping
the checkpoint adapter changes neither the budget nor the frozen source and
assumption context.

The callback is synchronous and should be cheap and nonblocking. This API
creates no tasks and supplies no forced interruption of foreign code. The
ordinary `certification_request` remains the standalone entry point, and the
NDJSON command has no asynchronous cancellation endpoint.

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

## Certify a flat or nonsmooth scalar root

`CertifyScalarRootBracket` connects the continuous scalar sign-bracket engine
to the session. It requires one registered original residual and one ordered
typed variable. Its independent checker proves **at least one real root** in a
closed subinterval of the requested interval, with width no greater than an
exact positive tolerance. This claim supports odd repeated roots and supported
continuous absolute-value corners without requiring a nonzero derivative.

For example, a derivative-based nonsingular-root certificate cannot cover the
triple root of `(x-1)^3`. The following session finds and independently checks it:

```json
{"id":"flat-source","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:flat","residuals":["(x-1)^3"],"variables":["x"]}}
{"id":"flat-root","type":"CertifyScalarRootBracket","payload":{"schema_version":1,"system":"system:flat","bracket":"bracket:flat","lower":"0","upper":"2","absolute_tolerance":"1/1000","precision_digits":24,"max_bisections":64}}
{"id":"flat-export","type":"ExportScalarRootBracket","payload":{"schema_version":1,"bracket":"bracket:flat"}}
```

The accepted response uses `kind: "scalar_root_bracket_certified"`,
`evidence: "certified_numeric"`, and
`claim: "at_least_one_real_root_in_closed_interval"`. It includes the complete
original bounds and tolerance, the certified root `enclosure`, independently
computed `endpoint_values`, precision, and assumption-context digest. In this
example the root enclosure is exactly `{"lower":"1","upper":"1"}`.

Other supported cases include `x+abs(x)/2` on `[-1,1]` and an exactly zero
endpoint of `(x-2)^2` on `[2,3]`. A zero-width input interval is admitted only
when evaluation proves that point is exactly zero. The full requested interval
must have certified source continuity: `(x-1)/x` on `[-2,2]` is refused even
though a proposed narrow interval around 1 would avoid its pole.

| Request or outcome | Meaning |
| --- | --- |
| `precision_digits` | Certified function-value precision, from 8 through 128. |
| `max_bisections` | Candidate search bound, from 0 through 1024. Zero can still accept an existing sufficiently narrow sign bracket or exact endpoint zero. |
| `absolute_tolerance` | Exact positive rational upper bound on the returned interval width. |
| `inconclusive` | The supported search cannot establish sufficiently narrow endpoint evidence. Equal endpoint signs do not prove root absence. |
| `domain_not_certified` | The complete original interval contains an unresolved source-domain obligation, such as a pole. |
| `bracket:` handle | Existence evidence only; it cannot be used as a `certificate:` unique-root handle or an `isolation:` complete-root-set handle. |

The sign-change lane does not enumerate all roots, establish multiplicity or
prove uniqueness. For instance, `x^2` on `[-1,1]` has a root but equal positive
endpoint signs, so this search reports `inconclusive`. A wider accepted bracket
may contain several roots. Rounding that merely encloses zero is never treated
as an exactly zero function value.

### Portable bracket replay

`ExportScalarRootBracket` returns canonical artifact bytes containing the
original typed source, variable, complete request interval, tolerance, context,
and candidate witness. It carries no trusted acceptance flag or endpoint value.
A fresh receiver first registers its own source, then supplies the same expected
statement in `ReplayScalarRootBracket`:

```json
{"type":"ReplayScalarRootBracket","payload":{"schema_version":1,"system":"system:flat","bracket":"bracket:received","lower":"0","upper":"2","absolute_tolerance":"1/1000","artifact":"<exact artifact string returned by ExportScalarRootBracket>"}}
```

Replay rejects source, axis, context, interval or tolerance substitutions before
constructing the bounded witness. It rechecks original continuity, containment,
width and endpoint evidence without invoking candidate search. Canonical byte
checks reject extra fields, duplicate keys and alternate encodings.

The chunked transfer API uses a distinct `scalar_root_bracket` artifact kind:
pass `{"scalar_root_bracket":"bracket:flat"}` as the
`BeginArtifactExport` source, then use
`ReplayScalarRootBracketFromArtifact` with the imported `transfer` and the
same receiver-selected source, bounds and tolerance. Byte integrity alone never
publishes a mathematical receipt. Inline artifacts retain the existing 24 KiB
session bound; chunked artifacts retain the existing 1 MiB bound.

Rust hosts can retrieve accepted evidence through
`Session::scalar_root_bracket(&ScalarRootBracketId)`. There are at most
`MAX_SESSION_SCALAR_ROOT_BRACKETS` (32) resident brackets. The independent
checker spends the session's protected verifier pool; generation, transport and
output consume ordinary cumulative resources. Host cancellation reaches both
phases and final publication. Any failed verification or output admission leaves
the new handle absent, and a later retry does not refund spent resources.

## Solve complete finite real polynomial systems

`SolveRealPolynomialSystem` solves all registered polynomial residuals equal to
zero over the reals. The bounded lane admits rational coefficients, one through
eight ordered variables, and at most 64 equations. It returns every distinct
real solution of a recognized finite system, including singular systems. The
independently checked native solver uses a separating parameter when the last
variable alone cannot distinguish all points, and can remove nilpotent structure
through a checked radical while preserving the complete distinct point set.

For example, the independent quadratic coordinates have four real solutions:

```json
{"id":"register-polynomial","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:quadratics","residuals":["x^2-2","y^2-3"],"variables":["x","y"]}}
{"id":"solve-polynomial","type":"SolveRealPolynomialSystem","payload":{"schema_version":1,"system":"system:quadratics"}}
```

The successful response has `kind: "real_polynomial_system_solved"`,
`evidence: "exact_cross_checked"`, `completeness: "complete"`, and
`domain: "real"`. Its claim is
`complete_distinct_real_solution_set_of_original_polynomial_system`.
It retains the registered system, original ordered typed variables and context
digest. Its `solution_set` contains:

| Field | Exact meaning |
| --- | --- |
| `parameter` | The typed algebraic parameter used for this representation; it may be an original axis or a fresh internal symbol. |
| `polynomial` | Nonzero monic defining polynomial, with exact rational coefficient strings in ascending degree. |
| `roots` | Every distinct real parameter root in increasing order, identified by exact rational isolating bounds. |
| `coordinates` | One ascending rational coefficient vector per original variable. Evaluate each vector at the same indexed parameter root to obtain a complete solution tuple. |

For this example, the parameter is `t=x+y`, the defining polynomial is
`["1","0","-10","0","1"]`, and the coordinate vectors are
`["0","-9/2","0","1/2"]` and `["0","11/2","0","-1/2"]`. Thus
each of the four distinct real roots of `t^4-10*t^2+1` represents exactly one
tuple `((t^3-9*t)/2, (11*t-t^3)/2)`. Isolating bounds identify algebraic points;
they are not approximate substituted coordinates or Cartesian solution boxes.

Empty `roots` is a complete empty real solution set, both for inconsistent
polynomial systems and for systems whose solutions are all nonreal. A
positive-dimensional or unrecognized finite system refuses explicitly instead
of returning a partial or empty list. The defining polynomial can have repeated
roots, but output tuples occur once. This interface makes no original-ideal
multiplicity, nonsingularity, complex-domain or portable-proof claim.

All original source expressions are admitted before ideal-dimension or unit-ideal
shortcuts. Unsupported functions and original variable denominators, including
`0*(1/x)` or `x/x-1`, refuse in this polynomial lane. Rational constant
denominators remain supported. The existing rational univariate requests below
retain their broader pole-aware domain semantics. Request payloads cannot
replace the registered source, coordinates, domain or accepted output.

The native conversion, Gröbner, shape, radical and root-isolation bounds remain
in force, including a maximum final parameter degree of 64. Generation and
independent checking spend ordinary and protected verifier resources from one
cumulative Session ledger. A caller-meter failure remains typed even across
older solver error wrappers, and final response admission precedes publication.
The fixed native search envelope can still return an explicit refusal. Larger
systems may require the CLI's documented `--compute-steps`, `--memory-bytes`,
`--allocations` and `--verifier-steps` allowances; no request replenishes them.

Rust callers use `CertificationRequest::SolveRealPolynomialSystem` and the
exported `ExactRealPolynomialSolutionSet` output type.

## Solve complete finite real rational systems

`SolveRealRationalSystem` connects the same finite-system solver to independently
checked rational localization. It solves all registered residuals equal to zero
on their original domains. Denominators are retained before cancellation:
neither a cancelled factor nor a held zero term can admit a pole as a solution.

For example, the polynomial equations below would have four real grid points,
but the original denominator excludes the two diagonal points:

```json
{"id":"register-rational-system","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:off-diagonal","residuals":["(x^2-2)/(x-y)","y^2-2"],"variables":["x","y"]}}
{"id":"solve-rational-system","type":"SolveRealRationalSystem","payload":{"schema_version":1,"system":"system:off-diagonal"}}
```

The response has `kind: "real_rational_system_solved"`,
`evidence: "exact_cross_checked"`, `completeness: "complete"`, and the claim
`complete_distinct_real_solution_set_of_original_rational_system`. It uses
the same `ExactRealPolynomialSolutionSet` representation described above:
a polynomial parameter and coordinate maps describe the exact algebraic points,
even though the original equations are rational.

In this example the internal parameter is `t=1/(x-y)`, its polynomial is
`["-1/8","0","1"]`, and the original coordinate maps are `["0","4"]` and
`["0","-4"]`. The two parameter roots therefore give precisely
`(sqrt(2),-sqrt(2))` and `(-sqrt(2),sqrt(2))`. The auxiliary variable is
never appended to the response's original ordered `variables` or
`coordinates`. Its inverse-product equation gives exactly one auxiliary value
for every admissible original solution, making this projection bijective.

The private native receipt retains the independently checked localization and
complete real polynomial representation. Before publication it checks their
exact augmented equations, coefficient values, monomial support and typed axes.
The original source includes nested reciprocals, zero products such as
`0*(1/x)`, and zero powers such as `(1/x)^0`. These preserve their original
pole restrictions. A source undefined everywhere or outside the rational
grammar refuses rather than being hidden by an inconsistent companion equation.

Polynomial inputs retain the eight-variable/64-equation envelope. A variable
denominator needs one auxiliary variable and equation, so that lane supports
at most seven original variables and 63 residuals. The existing native degree,
term, coefficient, search and output bounds still apply. A recognized empty
real set is complete; a positive-dimensional punctured family or unrecognized
finite system remains an explicit unsupported outcome. This operation does not
claim original-ideal multiplicities, a complete complex locus, or portable proof
transport. `SolveRealPolynomialSystem` retains its polynomial-only admission.

Both stages share the Session's cumulative ordinary and protected verifier
resources. No retry resets either allowance, and a failed final output admission
publishes no partial coordinate map. Rust callers use
`CertificationRequest::SolveRealRationalSystem`; lower-level callers can use
`fsym_solvers::real_rational::metered_solve_real_rational_system` with separate
caller-owned generation and verification meters.

## Solve complete real rational equations and inequalities

`SolveRealEquation` solves the single registered residual equal to zero over
the entire real line. `SolveRealInequality` solves the original residual
`relation` zero, with `relation` equal to `lt`, `le`, `gt`, or `ge`. Both require
exactly one residual and one registered variable. Their bounded exact rational
lane handles high-degree algebraic roots, repeated roots, rational poles,
identities with excluded points, and complete real sign decompositions.

For example, solve a quintic and a rational inequality in the same process:

```json
{"id":"register-quintic","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:quintic","residuals":["x^5-x+1"],"variables":["x"]}}
{"id":"solve-quintic","type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:quintic"}}
{"id":"register-rational","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:rational","residuals":["(x^2-2)/(x^2-3)"],"variables":["x"]}}
{"id":"solve-rational","type":"SolveRealInequality","payload":{"schema_version":1,"system":"system:rational","relation":"ge"}}
```

Successful responses have `kind: "real_equation_solved"` or
`kind: "real_inequality_solved"`, `evidence: "exact_cross_checked"`,
`completeness: "complete"`, and `domain: "real"`. Their claims are
`complete_real_solution_set_of_original_rational_equation` and
`complete_real_solution_set_of_original_rational_inequality`, respectively.
Each response includes its registered `system`, typed `variable`, current
immutable `context_digest`, and an exact `solution_set`:

| Field | Exact meaning |
| --- | --- |
| `polynomial` | Rational coefficient strings in ascending degree, defining a nonzero monic square-free polynomial. |
| `roots` | All distinct real roots of that polynomial, in increasing order, each identified by exact rational `lower` and `upper` isolating bounds. Each item denotes one exact algebraic point. |
| `point_inclusions` | Whether each indexed root belongs to the solution set. |
| `gap_inclusions` | Whether each open gap belongs to the solution set, including the unbounded gaps before the first root and after the last root. There is one more gap flag than root. |

The quintic example has one real root, all gap flags false, and its one point
flag true. No radical formula or floating-point approximation is needed to
identify the answer. For the inequality, the defining polynomial has coefficients
`["6","0","-5","0","1"]`; its ordered roots are
`-sqrt(3), -sqrt(2), sqrt(2), sqrt(3)`. The gap flags are
`[true,false,true,false,true]`, and point flags are
`[false,true,true,false]`. Thus both irrational denominator poles remain excluded,
while both zeros belong to this weak inequality. The `roots` bounds themselves
are isolators, not entire solution intervals or error tolerances.

Registration preserves the original held source. For example:

```json
{"id":"register-holes","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:holes","residuals":["0*(1/x)"],"variables":["x"]}}
{"id":"solve-holes","type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:holes"}}
```

This returns the real line with zero excluded: polynomial `["0","1"]`, gap
flags `[true,true]`, and point flags `[false]`. The same exclusion survives
`x/x-1` and `0*((1/x)^0)`. In contrast, `0/(x-x)` is nowhere defined and has an
empty solution set. With no critical roots, `gap_inclusions: [true]` denotes
the entire real line and `[false]` denotes the empty set. Repeated polynomial
roots occur once in the set; this interface makes no multiplicity or
nonsingularity claim. Later workspace bindings do not change a registered
source or its coordinate.

Polynomial reductions, complete root counts and endpoint/sign decisions use the
existing independent exact checkers. The response is a checked set description;
it is not a portable solver certificate and creates no resident certificate
handle. Rust callers use `CertificationRequest::SolveRealEquation` or
`CertificationRequest::SolveRealInequality` through `Session::certification_request`,
with exported `RealInequalityRelation` and `ExactRealSolutionSet` types.

The solver admits bounded rational arithmetic with literal integer powers.
The final root/critical polynomial has degree at most 64 and coefficient height
at most 4096 bits; source, intermediate growth, work and allocation bounds also
apply. Transcendental, fractional-power or foreign-parameter sources return
`unsupported_operation`, including unsupported children hidden behind a zero.
Unsupported syntax and exhausted resources never become an empty set. Request
payloads cannot override the registered source, variable, real domain, or
accepted membership flags.

Generation spends ordinary resources; the interleaved independent checks spend
the protected verifier pool on the same cumulative session ledger. Reacquiring
a verifier view never replenishes fuel. Complete rendering and response
admission must succeed before any set is returned, and repeated failed requests
retain their consumed work. These operations preserve the separate finite-box
Krawczyk claims above and leave campaign/profile certification gates open.

### Combine exact constraints into one complete feasibility region

`SolveRealConstraints` combines up to 32 registered rational residuals over one
variable. `relations` supplies one comparison with zero per residual, in the
original registration order: `eq`, `ne`, `lt`, `le`, `gt`, or `ge`. A `formula`
references these comparisons by zero-based atom index and combines them with
`all` (intersection) and `any` (union). Nesting is allowed; empty `all` is true
and empty `any` is false.

For example, solve `x^2 <= 2` or `x >= 2`:

```json
{"id":"register-region","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:region","residuals":["x^2-2","x-2"],"variables":["x"]}}
{"id":"solve-region","type":"SolveRealConstraints","payload":{"schema_version":1,"system":"system:region","relations":["le","ge"],"formula":{"any":[{"atom":0},{"atom":1}]}}}
```

The complete result is `[-sqrt(2), sqrt(2)] union [2, infinity)`. The response
has `kind: "real_constraints_solved"`, `evidence: "exact_cross_checked"`,
`completeness: "complete"`, and the claim
`complete_real_solution_set_of_original_rational_constraints`. It echoes the
ordered relations and formula and uses the same exact `solution_set` fields
described above. In this example the three ordered roots are `-sqrt(2)`,
`sqrt(2)`, and `2`; gap flags are `[false,true,false,true]` and point flags
are `[true,true,true]`. The shared critical polynomial may retain boundaries
that do not change final membership. These extra cells do not approximate or
weaken the set.

Every comparison is evaluated on its own original domain. At a pole, that
atom is false for every relation, including `ne`. Union may still include the
point through another defined atom: `(x/x-1 == 0) OR (x == 0)` includes every
real number, whereas `x/x-1 != 0` has no solutions. There is no implicit
Boolean negation that converts undefined values into true. All supplied
sources are admitted even when unused or behind an already decisive branch;
unsupported syntax and resource failures remain terminal.

The native engine independently checks each original-source atom, builds a
square-free least common multiple of their critical polynomials, and isolates
the complete common real root set. Exact factor ownership maps every local
root into that partition. A separate Boolean evaluator checks membership of
every common open gap and algebraic point, including points introduced by a
different atom. Generator and verifier work remain cumulative across the whole
formula. Bounds include 32 atoms, 256 formula nodes, depth 32, fanout 64, common
critical degree 64, and 4096-bit critical coefficients.

Rust callers use `CertificationRequest::SolveRealConstraints`, `RealRelation`
and `RealFormula` through `Session::certification_request`. The lower-level
`fsym_solvers::metered_solve_real_constraints` accepts typed `RealConstraint`
sources and separate meters, returning an immutable `VerifiedRealConstraints`
receipt. These are in-process exact checks, not portable solver certificates,
multivariate quantifier elimination, or a certified compatibility profile.

The Python `reduce_inequalities` facade also uses this common partition for
flat conjunctions of supported univariate rational comparisons, including
`Eq` and `Ne`. It can therefore decide contradictory algebraic bounds without
asking the generic set normalizer to order separately constructed root objects:

```python
from sympy import symbols, reduce_inequalities

x = symbols("x", real=True)
reduce_inequalities([x**2 <= 2, x**2 >= 3], x)  # False
```

The Python path retains held source domains and typed symbol identities and
admits every supplied source before a Boolean short circuit. It returns the
usual relational result. Nested `all`/`any` formulas remain available through
the native and Session APIs; this Python entry point accepts conjunctions.

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

## Export and replay a complete parameter family

An accepted uniform family or adaptive cover can be sent to another process:

```json
{"id":"export","type":"ExportParametricFamily","payload":{"schema_version":1,"family":"family:affine"}}
```

The receiver registers its own expected `ParametricProblem`, then supplies the
unchanged canonical artifact string and both complete domains:

```json
{"id":"replay","type":"ReplayParametricFamily","payload":{"schema_version":1,"problem":"parametric:affine","family":"family:received","unknown_domain":[{"lower":"-2","upper":"2"}],"parameter_domain":[{"lower":"-1","upper":"1"}],"artifact":"<the exact result.artifact string, JSON-escaped by your client>"}}
{"id":"differentiate","type":"EncloseParametricSensitivity","payload":{"schema_version":1,"family":"family:received","sensitivity":"sensitivity:received"}}
```

Replay binds the original typed residuals, the separately ordered unknown and
parameter groups, both complete boxes, schema and receiving context. It checks
the untrusted witness independently, including every leaf of an adaptive cover.
It does not rerun the cover search. Exported artifacts contain candidate centers,
preconditioners and partition data; accepted enclosures, contraction bounds and
sensitivities are recomputed by the receiving checker. A completed transfer or a
matching content digest alone grants no family. Sensitivity requests consume the
newly accepted receiving family and recompute the original source derivatives.

Uniform and cover artifacts share the 24 KiB inline and 1 MiB chunked limits.
For a larger proof, begin an export with
`{"parametric_family":"family:affine"}`, forward its manifest and chunks, then
send `ReplayParametricFamilyFromArtifact` with the same expected statement and a
completed `transfer` instead of an `artifact` string. Unknown fields, duplicate
keys, alternate encodings, reordered coordinate roles, incomplete covers and
oversized exact scalars refuse before publication. Scalar normalization is
bounded and charged per scalar; large partitions are handled linearly.

Rust callers use `export_parametric_family(receipt, context_digest, meter)` and
`replay_parametric_family(bytes, expected_problem, expected_unknown_domain,
expected_parameter_domain, expected_context, meter)`. Both variants return the
same `ParametricFamilyReceipt` accepted type. Export consumes ordinary work;
receiving Session validation and independent replay consume protected verifier
compute on the same cumulative ledger.

## Reuse checked residual programs and matrix-free derivatives

`CompileResidualSystem` retains the original registered residuals, their ordered
coordinates, independently checked symbolic derivatives and admitted binary64
programs. `EvaluateResidualJacobian` reuses that program for residual and sparse
Jacobian values. `EvaluateJacobianVectorProduct` and
`EvaluateJacobianTransposeVectorProduct` return only the requested sparse
linear-map product; they do not construct a dense Jacobian or transpose.

Send the following to one process:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:products","residuals":["x^2+y","sin(y)","7"],"variables":["y","x"]}}
{"id":"compile","type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:products","program":"program:products"}}
{"id":"forward","type":"EvaluateJacobianVectorProduct","payload":{"schema_version":1,"program":"program:products","point":["0","2"],"direction":["3","-2"]}}
{"id":"transpose","type":"EvaluateJacobianTransposeVectorProduct","payload":{"schema_version":1,"program":"program:products","point":["0","2"],"weights":["2","-3","11"]}}
{"id":"release","type":"ReleaseCompiledResidualSystem","payload":{"schema_version":1,"program":"program:products"}}
```

At this point the Jacobian in the declared coordinate order is
`[[1,4],[1,0],[0,0]]`. The expected forward product is `[-5,3,0]`; the
expected transpose product is `[-1,8]`. Directions follow variable order,
whereas transpose weights follow the complete original residual order,
including constant residuals.

Compilation reports the formal symbolic derivative and bytecode provenance
claim. All three numerical evaluation operations report `outcome: "approximate"`
and `numeric_model: "binary64"`, with no numeric-certificate evidence field.
Finite decimal strings are rounded to binary64; fractions such as `"1/2"`,
nonfinite values, dimension mismatches and strings longer than 512 bytes refuse.
Original residual and derivative domains are rechecked on every execution,
including zero directions or zero weights. Overflow refuses the whole result.

Program handles are session-local and immutable even after workspace rebinding.
Their source-bound candidates can be exported and independently replayed in a
fresh session, as described below. At
most 32 residual programs may be resident. Compilation uses ordinary resources
for generation and protected resources for independent acceptance. Repeated
evaluation and response admission consume the same cumulative session budget.
`ReleaseCompiledResidualSystem` frees capacity without refunding spent work;
all later requests against that released handle return `unknown_object`.

Rust callers use the corresponding `CertificationRequest` variants through
`Session::certification_request`, or borrow the checked program through
`Session::compiled_residual_system` for native metered execution.

## Compile and reuse an objective, gradient and sparse Hessian

The session can retain an independently admitted `VerifiedObjective` and reuse
its source, gradient and Hessian programs across numerical requests. Candidate
generation and independent admission now both consume caller-provided budgets;
the session runs admission in its protected verifier pool.

```json
{"id":"register","type":"RegisterObjective","payload":{"schema_version":1,"objective":"objective:quadratic","source":"x^2+x*y+3*y^2","variables":["x","y"]}}
{"id":"compile","type":"CompileObjective","payload":{"schema_version":1,"objective":"objective:quadratic","program":"objective_program:quadratic"}}
{"id":"evaluate","type":"EvaluateObjective","payload":{"schema_version":1,"program":"objective_program:quadratic","point":["2","-1"]}}
{"id":"product","type":"EvaluateHessianVectorProduct","payload":{"schema_version":1,"program":"objective_program:quadratic","point":["2","-1"],"direction":["3","-2"]}}
{"id":"reuse","type":"EvaluateObjective","payload":{"schema_version":1,"program":"objective_program:quadratic","point":["-1","2"]}}
{"id":"release","type":"ReleaseCompiledObjective","payload":{"schema_version":1,"program":"objective_program:quadratic"}}
```

For the first point, the expected value is `5`, the gradient is `[3,-4]`, and
the Hessian has ordered sparse coordinates `[[0,0],[0,1],[1,0],[1,1]]` with
values `[2,1,1,6]`. Multiplying by `[3,-2]` gives `[4,-9]`. The second point
has value `11` and gradient `[0,11]`; numerical memoization is local to each
evaluation and never reuses the first point's values.

`ObjectiveCompiled` binds the original objective, ordered typed coordinates,
context, sparse pattern and numbers of independently checked derivative proofs
and omitted zero entries. Its claim is
`formal_symbolic_gradient_hessian_and_binary64_lowering`. This proves neither
convexity nor smoothness over an arbitrary box. Numerical responses are labeled
`approximate` / `binary64`; use `EncloseObjective`, `CertifyStationaryPoint`
or the existing global box operations for their separate exact-interval claims.

`EvaluateHessianVectorProduct` traverses the sparse entries without constructing
a dense Hessian or returning all Hessian values. The original objective,
gradient and second-derivative domains are checked even for a zero direction.
Nonfinite inputs, arithmetic overflow and derivative singularities refuse the
complete response. Evaluations consume ordinary cumulative resources and do
not spend verifier fuel after the program has been admitted.

At most 32 compiled objectives may be resident, independently of the 32 residual
program slots. Objective registration and compiled sparse optimization admit
1–64 ordered variables. Independent exact enclosures and stationary-point
certificates retain their separate 1–16-variable limits. Compilation also
retains the native source limits of 256 nodes, depth 16,
4096-bit exact scalar payloads and 256-byte names, plus the existing derivative
growth and total-node bounds. Unknown IDs, wrong-kind labels, duplicate handles,
invalid schemas and unsupported original sources fail closed. Releasing a
program frees capacity without refunding budgets; failed response admission
cannot publish a new handle or remove an existing one.

Rust callers use `CompiledObjectiveId` and the matching `CertificationRequest`
variants. `Session::compiled_objective` borrows the immutable admitted program.
For native callers with their own resource policy,
`ObjectiveCandidate::try_build_with_meter` and
`VerifiedObjective::try_from_candidate_with_meter` accept separate generation
and verifier meters. `VerifiedObjective::try_compile_with_meter` is a convenience
entry point using one cumulative meter for both phases. Charges represent
logical work and storage; bounded synchronous derivative and proof kernels
retain their declared cancellation boundaries.

### Bounded numerical differentiation

Numerical derivative generation and independent replay use a bounded exact
normalization lane. Each reconstruction has a private allowance of 20 million
logical compute steps, 64 MiB of cumulative requested storage and one million
allocations, including work spent on intermediate derivatives. These allowances
also apply when the caller has no explicit budget. A supplied meter is charged
throughout; child derivatives, historical recipe attempts and failed
normalizations never replace or refund that caller ledger.

The lane admits at most 8,192 expression nodes, depth 128, 4,096 bytes per
identifier and 4,096 bits per exact scalar component. A compact held power can
therefore be refused before it materializes an enormous integer. Raw numerical
rules run through an explicit work list with cached child derivatives, so a
chain of supported functions does not recursively stack the large rule tables.
The generator and independent replay retain their separate mathematical rules.
Ordinary public symbolic differentiation and simplification keep their existing
entry points and forms.

Native dense callers can use
`CompiledResidualSystem::try_compile_with_meter` to include original residual
compilation, derivative construction and output admission in one caller budget.
Session responses distinguish operational exhaustion and cancellation from a
mathematical rejection. Python exposes normalization resource refusals as
`EvaluationBudgetExceeded`, cancellation as `EvaluationCancelled`, and
allocation failure as `MemoryError`; a numerical search must not treat these as
a domain failure and try another point.

## Derive a least-squares objective from residuals

`CompileLeastSquaresObjective` creates the exact held objective
`(1/2) * sum(residual_i^2)` from a resident compiled residual system. It returns
the existing `objective_compiled` response and publishes both a new objective
source and its checked gradient/Hessian program. The full Hessian includes
both the Jacobian-product and residual-curvature terms.

```json
{"id":"data","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:fit","residuals":["x-1","x-3"],"variables":["x"]}}
{"id":"compile-data","type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:fit","program":"program:fit"}}
{"id":"loss","type":"CompileLeastSquaresObjective","payload":{"schema_version":1,"residual_program":"program:fit","objective":"objective:fit","program":"objective_program:fit"}}
{"id":"check-fit","type":"MinimizeAndCertifyCompiledObjective","payload":{"schema_version":1,"program":"objective_program:fit","certificate":"stationary:fit","initial":["0"],"radii":["1/100"],"precision_digits":24}}
```

These inconsistent residuals have a strict minimum of the squared error at
`x=2`, with objective value `1` and residuals `[1,-1]`. The last operation
independently certifies the stationary point and its local classification.
Compilation itself establishes the formal derivative/lowering claim only.
A least-squares stationary point can also be a maximum or saddle; the existing
checker retains the actual classification.

The operation keeps every original row, including duplicates, zeros and
domain obligations hidden behind algebraic cancellation, and preserves the ordered
typed coordinates. It refuses a different current assumption universe.
Generation spends ordinary resources; independent derivative admission spends
protected verifier resources. Duplicate IDs, either resident cap, budget
exhaustion or cancellation leave both target handles unpublished.

Both target IDs must be new, and the existing separate source/program caps
apply. The derived objects own their source, so releasing the residual program
does not invalidate them. Existing objective evaluation, Hessian-vector
products, certification, global box optimization and compiled-program
export/replay operations work with the resulting handles.

## Export and independently replay compiled derivative programs

A residual/Jacobian or objective/gradient/Hessian program can be transferred to
another process without asking that process to generate new symbolic
derivatives. The receiver registers its own expected original expression(s) and
ordered coordinates, then independently checks the imported derivative proofs,
sparse coverage and bytecode provenance. Only successful admission produces a
new resident program handle and freshly computed verification receipt.

For a residual program, send these lines to the producing session:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:source","residuals":["x^2+y","sin(y)","7"],"variables":["y","x"]}}
{"id":"compile","type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:source","program":"program:source"}}
{"id":"export","type":"ExportCompiledResidualSystem","payload":{"schema_version":1,"program":"program:source"}}
```

Retain the exact `result.artifact` string. In a new process, register the expected
source and replay it under new session-local names:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:expected","residuals":["x^2+y","sin(y)","7"],"variables":["y","x"]}}
{"id":"replay","type":"ReplayCompiledResidualSystem","payload":{"schema_version":1,"system":"system:expected","program":"program:received","artifact":"<the exact result.artifact string, JSON-escaped by your client>"}}
{"id":"evaluate","type":"EvaluateJacobianVectorProduct","payload":{"schema_version":1,"program":"program:received","point":["0","2"],"direction":["3","-2"]}}
```

The product is `[-5,3,0]`. The received handle also supports
`EvaluateResidualJacobian`, `EvaluateJacobianTransposeVectorProduct`,
`SolveCompiledResidualSystem`, `SolveCompiledParametricSystem`,
`SolveCompiledLeastSquares`, reexport, and
`ReleaseCompiledResidualSystem` just like a locally compiled program.

For an objective, use `ExportCompiledObjective` with `program`, followed by
`ReplayCompiledObjective` with `objective`, a new `program`, and `artifact`.
The receiving objective must already be registered. Replay returns the ordinary
`objective_compiled` response, and the new handle supports `EvaluateObjective`,
`EvaluateHessianVectorProduct`, `MinimizeCompiledObjective`, reexport, and
`ReleaseCompiledObjective`.

Inline artifacts are limited to 24 KiB. Both kinds also use the existing 1 MiB
chunked transfer route:

| Export source | Receiving request | Required receiving fields |
| --- | --- | --- |
| `{"compiled_residual_system":"program:source"}` | `ReplayCompiledResidualSystemFromArtifact` | `system`, new `program`, completed `transfer` |
| `{"compiled_objective":"objective_program:source"}` | `ReplayCompiledObjectiveFromArtifact` | `objective`, new `program`, completed `transfer` |

Each request also requires `schema_version: 1`. Forward the manifest and every
chunk without rewriting the artifact. Transfer digests prove byte integrity;
they do not authorize execution. The input frame remains limited to 64 KiB,
each chunk carries at most 16 KiB, and the existing eight-transfer bound applies.
An imported transfer remains available after successful or failed replay until
explicitly released; an accepted program remains valid after transfer release.

The canonical envelope binds the program's frozen compilation context. The
receiver must match that context and the complete original typed source,
including held operand order and multiplicity, exact scalar variants, residual
order, and symbol identities. A source that differs only by a cancelled pole
still differs: `x^2+0*(1/x)` cannot replace `x^2`. Original source and derivative
domain checks run on every numerical execution, including zero-vector products.

Decoding and canonicalization consume ordinary cumulative resources. Source
binding, derivative proof replay, omitted-zero checking, bytecode admission and
reconstruction of derived evaluation plans consume the protected verifier pool
on the same ledger. Response admission and the final cancellation check precede
handle publication. Wrong kind/schema, unknown or duplicate fields, altered
proofs, noncanonical encodings, resource exhaustion and context/source mismatches
leave no new compiled handle. A failed import never refunds spent resources.

The format contains untrusted source, derivative and bytecode candidates. It
does not transport trusted receipts, accepted flags, shared evaluation plans or
numeric caches. Its bounded proof profile admits the current single-step
definitional derivative certificates. Receiving derivative proofs and bytecode
are checked directly; original residual bytecode and local shared evaluation
plans are rebuilt as part of independent admission. All numerical outputs
remain explicitly `approximate` / `binary64`. Replay does not grant an interval,
root, rounding-error, smoothness-on-a-box or optimization certificate.

Rust callers can use `export_residual_program` / `export_objective_program` and
`decode_residual_program` / `decode_objective_program` with a caller-owned meter
and expected context. Decoding returns only an untrusted candidate. Complete
the boundary with `fsym_calculus::program_artifact::admit_residual_candidate` or
`admit_objective_candidate`, supplying the receiver's raw expressions and
ordered typed variables and an independent verifier meter. The typed Session
requests perform those stages and publication together.

For a complete native operation, use `replay_residual_program` with the
receiver's `ResidualProblem`, or `replay_objective_program` with its
`ObjectiveProblem`. Each takes the exact artifact bytes, the receiving problem,
the expected context digest and a caller-owned meter, and returns a verified
program only after both transport decoding and independent source-bound
admission succeed. The same meter pays for both phases. These helpers accept
the existing Session export format, and their returned programs can be
reexported with the functions above. `ProgramReplayError::Transport` and
`ProgramReplayError::Admission` distinguish transport refusals from independent
mathematical admission failures; neither failure returns an executable program.

## Solve with locally compiled or restored programs

Numerical solving consumes the immutable, independently admitted program already
held by a session. The following operations work with either a locally compiled
handle or one returned by replay. They do not generate derivatives again, change
the registered source, or select a replacement backend after a refusal.

| Request | Explicit backend | Successful numerical result |
| --- | --- | --- |
| `SolveCompiledResidualSystem` | `frankennumpy`, `frankenscipy`, `frankenscipy_analytic`, or `sparse` | A point meeting the residual tolerance in every row. Sparse accepts square, overdetermined and underdetermined systems with 1–256 variables and 1–4096 rows. The other backends require square systems |
| `SolveCompiledParametricSystem` | Fixed `frankennumpy` | A root of the unknown prefix at the supplied fixed parameter suffix |
| `SolveCompiledLeastSquares` | `dense` or `sparse` | A point meeting the tolerance for the gradient of `0.5 * sum(residuals^2)` |
| `SolveCompiledBoundedLeastSquares` | Sparse | A feasible point meeting the projected-gradient tolerance on the supplied box |
| `MinimizeCompiledObjective` | `dense` (default, 1–16 variables) or `sparse` (1–64 variables) | A point meeting the objective-gradient tolerance |

For example, compile a square system and refine an initial point:

```json
{"id":"register","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:solve","residuals":["x^2-2","y-x"],"variables":["x","y"]}}
{"id":"compile","type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:solve","program":"program:solve"}}
{"id":"solve","type":"SolveCompiledResidualSystem","payload":{"schema_version":1,"program":"program:solve","initial":["1","1"],"backend":"frankennumpy"}}
```

The result has kind `compiled_residual_system_solved`, a `root` near
`[sqrt(2), sqrt(2)]`, the original residual infinity norm, iteration count,
selected backend, and `termination: "residual_tolerance"`. Point coordinates
always follow the frozen declared variable order, including after workspace
rebinding. The result also identifies the original system, program, and frozen
context digest.

An underdetermined sparse root proposal is one approximate point, not a unique
solution, a minimum-norm solution or a complete family. Coordinates absent from
every residual retain their initial values. The separate root-certificate
operations still require a square system and independently check its original
source on the requested box.

Least squares retains every residual row, including inconsistent and constant
rows. For residuals `["x-1", "x-3"]` and variable `["x"]`, either least-squares
backend with initial `["0"]` returns a point near `[2]`, residuals near `[1,-1]`,
cost near `1`, and gradient norm near zero. This is not a root of the residual
system. Send `SolveCompiledLeastSquares` with `program`, `initial`, `backend`,
and `schema_version: 1`; its result has kind `compiled_least_squares_solved`,
`point`, `residuals`, `cost`, `gradient_norm`, `iterations`, `evaluations`, and
`termination: "gradient_tolerance"`, together with the source/program/context
and backend fields. Dense least squares requires at least as many residuals as
variables. Sparse least squares also supports underdetermined and rank-deficient
systems; it does not claim a unique solution or a minimum-norm solution.

Use `SolveCompiledBoundedLeastSquares` to fit inside coordinate bounds with the
sparse backend. Supply one lower and one upper endpoint per ordered variable.
A finite binary64 decimal string sets an endpoint and JSON `null` leaves that
side unbounded. Equal finite endpoints fix a coordinate. The initial point
must satisfy all bounds. The optional iteration, step, and gradient controls
use the same fields as `SolveCompiledLeastSquares`.

For example, after compiling the two residuals above as `program:fit`, this request limits
`x` to 1 and returns `point: [1]`, `residuals: [0,-2]`, and `cost: 2`.

```json
{"id":"bounded-fit","type":"SolveCompiledBoundedLeastSquares","payload":{"schema_version":1,"program":"program:fit","initial":["0"],"lower":[null],"upper":["1"]}}
```

The response returns `gradient` and `projected_gradient` in the registered
variable order, together with their infinity norms. Here `gradient` is
`[-2]` and `projected_gradient` is `[0]` because increasing `x` would
leave the box. A fixed coordinate also has a zero projected component.
Interior components retain their original source gradients. The solver reports
`projected_gradient_tolerance` when the projected norm meets the requested
tolerance.

For an objective, use the compiled objective handle:

```json
{"id":"register","type":"RegisterObjective","payload":{"schema_version":1,"objective":"objective:descent","source":"(x-1)^2+2*(y+2)^2","variables":["x","y"]}}
{"id":"compile","type":"CompileObjective","payload":{"schema_version":1,"objective":"objective:descent","program":"objective_program:descent"}}
{"id":"minimize","type":"MinimizeCompiledObjective","payload":{"schema_version":1,"program":"objective_program:descent","initial":["0","0"]}}
```

Here the expected point is near `[1,-2]`, with value and gradient near zero.
The `compiled_objective_minimized` result reports `backend`, `point`, `value`, `gradient`,
`gradient_norm`, `iterations`, `evaluations`, and `regularizations`, along with
the objective/program/context fields and `termination: "gradient_tolerance"`.
A stationary initial point can also be a maximum or saddle; successful
termination alone does not classify it.

Select `"backend":"sparse"` in the minimization payload to execute the existing
metered sparse Hessian Newton/CG route. It admits 1–64 declared variables,
including unused coordinates, and applies stored Hessian entries directly
without building a dense Hessian or calling the dense donor solver. Omission
or `"backend":"dense"` retains the 1–16-variable FrankenNumPy route. Either
selection works on locally compiled or independently replayed objective
programs. The original source/gradient/Hessian domain checks, fresh final
gradient test, cumulative resource ledger and terminal cancellation policy
apply to both. A backend refusal never triggers the other backend.

`RegisterObjective` retains up to 64 ordered coordinates, including unused
coordinates and distinct typed symbols with the same name. Registration keeps
the original held source; numerical compilation applies its own stricter
source limits. Independent exact enclosure, stationary-point certification,
and `MinimizeAndCertifyCompiledObjective` retain their separate 1–16-variable
limits. The combined minimization/certification operation continues to use
the dense numerical proposal route.

All three result kinds explicitly report `outcome: "approximate"` and
`numeric_model: "binary64"`. They contain no evidence receipt or certificate.
Use `CertifyRootBox` or `CertifyStationaryPoint` for their exact original-source
interval claims, or the combined operations below to generate and verify a
witness in one request. A numerical proposal does not bypass those independent
checks.

### Sweep fixed parameters without recompilation

For a system with `m` residuals, declare the `m` unknown coordinates first and
one or more parameter coordinates afterwards. Compile once, then submit
different fixed parameter values with the same program:

```json
{"id":"register-parametric-source","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:parameter-sweep","residuals":["x^2-p"],"variables":["x","p"]}}
{"id":"compile-parameter-sweep","type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:parameter-sweep","program":"program:parameter-sweep"}}
{"id":"solve-p2","type":"SolveCompiledParametricSystem","payload":{"schema_version":1,"program":"program:parameter-sweep","initial":["1"],"parameters":["2"]}}
{"id":"solve-p9","type":"SolveCompiledParametricSystem","payload":{"schema_version":1,"program":"program:parameter-sweep","initial":["1"],"parameters":["9"]}}
```

The results have kind `compiled_parametric_system_solved`, approximate roots
near `sqrt(2)` and `3`, and the actual fixed binary64 `parameters` used for
the solve. Ordered typed `unknowns` and `parameter_variables` identify the
coordinate split. Parameter strings follow the finite decimal numerical-input
rules below; reporting `0.1` does not claim exact rational `1/10`.

The Newton step uses only the unknown Jacobian block. Each evaluation still
checks the complete original source and every stored derivative, including
parameter derivatives. Held poles and nonsmooth source endpoints cannot
disappear through substitution, a zero product, a zero power or a zero residual.
A singular unknown Jacobian, oversized step or exhausted iteration allowance
refuses. An initial point already meeting the residual tolerance remains only
an approximate candidate, including when its Jacobian is singular.

This request uses the same cumulative ledger and FrankenNumPy controls as
`SolveCompiledResidualSystem`. It requires `1 <= m < variables <= 16`,
keeps all parameter values fixed, and performs no recompilation or backend
fallback. Its response has `outcome: "approximate"` and
`numeric_model: "binary64"`; it does not publish a family certificate.

The Python interface exposes the same reusable path:

```python
from sympy import symbols, compile_residual_system

x, p = symbols("x p")
program = compile_residual_system([x**2 - p], (x, p))
candidate = program.solve_parametric_candidate([1], [2])
assert candidate.unknown_variables == (x,)
assert candidate.parameters == (p,)
assert candidate.parameter_values == (2.0,)
assert candidate.evidence == "approximate"
```

The candidate retains the exact native program owner and its frozen typed
coordinate order. Legacy and portable program replay retain this method;
replay still independently admits the original program.

### Solve and certify a complete parameter range

`SolveAndCertifyCompiledParametricSystem` combines the numerical proposal
with the existing independent parameter-family checker. Register the exact
same original source and ordered coordinate roles as a `ParametricProblem`,
then supply an exact parameter box and explicit positive radii for the unknowns:

```json
{"id":"family-source","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:family-solve","residuals":["2*x-p"],"variables":["x","p"]}}
{"id":"family-program","type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:family-solve","program":"program:family-solve"}}
{"id":"family-statement","type":"RegisterParametricProblem","payload":{"schema_version":1,"problem":"parametric:family-solve","residuals":["2*x-p"],"unknowns":["x"],"parameters":["p"]}}
{"id":"family-solve","type":"SolveAndCertifyCompiledParametricSystem","payload":{"schema_version":1,"program":"program:family-solve","problem":"parametric:family-solve","family":"family:solved","initial":["0"],"parameter_domain":[{"lower":"3/2","upper":"5/2"}],"radii":["1/2"],"precision_digits":24}}
{"id":"family-sensitivity","type":"EncloseParametricSensitivity","payload":{"schema_version":1,"family":"family:solved","sensitivity":"sensitivity:solved"}}
```

The accepted `parametric_family_certified` response proves that for every
`p` in `[3/2, 5/2]`, exactly one nonsingular unknown root lies in the declared
unknown box `[1/2, 3/2]`. The checked root enclosure is `[3/4, 5/4]`;
the sensitivity request independently returns `dx/dp = [1/2, 1/2]`.
The resulting family supports the existing export, replay and sensitivity APIs.

The driver solves at a binary64 approximation of the exact parameter midpoint.
That numerical point and the donor inverse supply only an untrusted witness.
The full original parameter box, including exact point intervals such as
`[1/10, 1/10]`, is retained unchanged for protected verification. The checker
recomputes source values and derivatives across the entire product domain.
A successful midpoint solve cannot establish a family whose roots leave the
unknown box, whose source has a pole elsewhere in the parameter range, or
whose unknown Jacobian cannot be certified nonsingular.

Source and typed-coordinate binding, context identity, domain parsing and
control validation precede verification. Proposal work uses ordinary resources;
acceptance uses the same ledger's protected verifier pool. Only a complete
check and successful response admission publish the family. Refusal leaves the
requested family label available. This operation proposes one box, does not
widen the caller's radii, and does not invoke adaptive cover search. Use
`CertifyParametricCover` separately when one uniform box cannot certify a
larger parameter range.

Python exposes the corresponding
`program.solve_parametric_certified(initial, parameter_box,
unknown_radii=..., precision=32, budget=...)` method. It captures exact
bounds and checks source eligibility before the numerical solve, retains one
cumulative ledger through solving and checking, and returns the existing
`ParametricRootBoxReceipt` only after independent original-source acceptance.

### Solve and certify an exact root box

`SolveAndCertifyCompiledResidualSystem` composes numerical refinement with
independent original-source root-box verification in one request. It consumes
the same admitted compiled handle, including a replayed handle, and takes exact
positive radii in the frozen variable order:

```json
{"id":"certified-solve","type":"SolveAndCertifyCompiledResidualSystem","payload":{"schema_version":1,"program":"program:solve","certificate":"certificate:solve","initial":["1","1"],"backend":"frankennumpy","radii":["1/1000","1/1000"],"precision_digits":24}}
```

After the registration and compilation example above, this request produces
the existing `root_box_certified` response only if the independently checked
box contains a unique nonsingular real root of the original residual system.
The certificate can immediately be exported with `ExportRootCertificate` and
independently replayed against the same source in a fresh session.

The numerical center and donor inverse are proposals. Their exact binary64
values are encoded as rational witnesses; the supplied rational radii are
retained exactly. Source evaluation and interval differentiation are recomputed
by the root verifier, so a loose numerical tolerance, a small residual, or a
compiled derivative does not establish the root claim. Original logarithmic,
reciprocal, and other supported source-domain restrictions still apply.

All four root backends listed above are available, with their existing explicit
controls. The combined operation is bounded to square systems of 1–16 variables
and precision 8–128 decimal digits. Every radius must be positive. There is no
automatic radius widening or fallback backend. A singular root, an uncertifiable
box, a refused proposal, or resource exhaustion returns an error and creates no
certificate handle. Generation consumes ordinary cumulative resources;
independent acceptance consumes the protected verifier pool. The complete
response is reserved before the certificate is published, and an existing
certificate identifier cannot be overwritten.

### Minimize and certify a stationary point

`MinimizeAndCertifyCompiledObjective` connects the existing numerical descent
to independent original-source stationary-point verification. It accepts a
locally compiled or independently replayed objective handle and exact positive
radii in the frozen coordinate order:

```json
{"id":"certified-minimize","type":"MinimizeAndCertifyCompiledObjective","payload":{"schema_version":1,"program":"objective_program:descent","certificate":"stationary:descent","initial":["0","0"],"radii":["1/100","1/100"],"precision_digits":24}}
```

After the quadratic objective example above, the result is
`stationary_point_certified`, with `classification: "strict_minimum"`, exact
enclosures and a reusable `stationary:descent` handle. The complete original
source is checked over the entire proposed box. Numerical convergence only
supplies the center and inverse-Hessian proposal; exact binary64-to-rational
conversion preserves those proposal values, and the supplied rational radii
are retained without widening. A loose stopping tolerance cannot authorize a
certificate if the box does not contain the stationary point.

The response classifies the point according to the existing independent
Hessian check: strict local minimum, strict local maximum, saddle or
unclassified. A stationary initial maximum or saddle can pass the numerical
stopping criterion and is reported with its actual certified classification.
The operation does not claim global optimality or complete enumeration.
`ExportStationaryCertificate` and `ReplayStationaryCertificate` transport and
independently replay the resulting source-bound certificate unchanged.

The operation supports 1–16 variables, precision 8–128 decimal digits, and the
same optional `CompiledStationarityOptions` controls as
`MinimizeCompiledObjective`. A singular numerical Hessian, invalid source
domain, failed descent, insufficient box, cancellation or exhausted resources
publishes no stationary handle. Ordinary cumulative resources fund descent and
the untrusted witness; the protected verifier pool funds the independent C2,
inclusion, contraction and classification checks. Response admission precedes
publication, failed work is not refunded, and existing identifiers cannot be
overwritten.

### Controls, limits, and failure behavior

The optional `options` object accepts `max_iterations`, `max_step_norm`, and
either `residual_tolerance` for roots or `gradient_tolerance` for the other two
operations. Numerical controls and initial coordinates are finite decimal
strings, with the same 512-byte bound and binary64 rounding as evaluation
points. Iteration counts are nonnegative integers bounded by 1024. A zero
iteration allowance checks the initial point against the stopping criterion.

FrankenNumPy and sparse root solving default to 64 iterations, residual
tolerance `"1e-12"`, and maximum step norm `"1e6"`. Least squares and objective
descent default to 64 iterations, gradient tolerance `"1e-10"`, and maximum
step norm `"1e6"`. Tolerances must be nonnegative; maximum step norm must be
positive. FrankenNumPy Newton refuses an oversized step; the sparse Newton,
least-squares and descent drivers bound the trial step before searching.

Both FrankenSciPy root backends retain their pinned controls: 200 iterations and
residual tolerance `1e-10`. They reject any supplied non-null tuning field in
`options`, even if it equals a default, rather than ignoring it. Omit `options`
or use an empty object for those backends. Unknown controls, duplicate fields,
unknown backend names and numeric controls encoded as JSON numbers refuse as
`malformed_request`.

Residual-system registration retains its existing bounds of 1–16 ordered
variables and 1–64 residuals; selecting sparse root or least-squares solving
does not enlarge those registration limits. Objective registration admits
1–64 ordered variables for the compiled sparse route; dense minimization and
independent exact objective certificates retain their 1–16-variable limits.
Dense and FrankenSciPy root solving require equal residual and variable counts;
sparse roots also admit rectangular systems within the registration bounds.
Fixed-parameter Session root solving requires one unknown per residual followed
by a nonempty parameter suffix. These drivers preserve the original source and
derivative domain checks, including held poles, nonsmooth source endpoints and
Hessian singularities at stationary points. Nonfinite results refuse.

Input conversion, controls, unsuccessful trials, actual consumer work, and
response admission spend the same ordinary cumulative session budget. The
approximate-only solver requests do not spend protected verifier fuel after
the program is admitted. Combined solve-and-certify requests reserve that pool
for their independent original-source acceptance stage.
Cancellation is observed at the existing metered callback and synchronous kernel
boundaries; this interface does not promise preemption inside a donor kernel.

Invalid controls or point dimensions return `invalid_input`; unsupported root
shapes return `unsupported_operation`; domain or nonfinite evaluations return
`numeric_evaluation_refused`; nonconvergence returns `numeric_no_convergence`;
and rejected donor solves return `numeric_consumer_refused`. Existing
`budget_exhausted`, `cancelled`, `unsupported_schema`, and `unknown_object`
codes remain distinct. A failed solve or final response admission retains the
resident handle and all previously spent resources. Releasing a program makes
it unavailable to these consumers as well as to evaluation.

Rust callers use the same `CertificationRequest` variants and the public
`CompiledRootBackend`, `CompiledLeastSquaresBackend`,
`CompiledRootSolveOptions`, and `CompiledStationarityOptions` types. The Session
API returns typed response variants; the NDJSON envelope serializes their
snake-case result kinds.

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

Whole-cell derivative signs also tighten bounds at the corresponding extremal
faces. For example, `exp(x)-x` on `[1,2]` can certify either extremum with
`max_leaves: 1`. Coordinates with uncertain derivative signs retain their full
intervals, and every original-source domain obligation is checked on the whole
cell before any face restriction. These bounds supplement the natural and
centered Taylor enclosures used by both search and independent verification.

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

## Advance a global optimization in bounded steps

A long optimization can retain its complete candidate partition, best feasible
point and priority queue between requests. Register an objective, then begin:

```json
{"id":"begin","type":"BeginBoxOptimization","payload":{"schema_version":1,"objective":"objective:well","search":"optimization-search:well","domain":[{"lower":"-2","upper":"2"}],"extremum":"minimum","absolute_tolerance":"1/1000","precision_digits":16,"max_leaves":1024}}
{"id":"advance","type":"AdvanceBoxOptimization","payload":{"schema_version":1,"search":"optimization-search:well","max_splits":4}}
{"id":"inspect","type":"InspectBoxOptimizationSearch","payload":{"schema_version":1,"search":"optimization-search:well"}}
```

Started and progress responses carry `evidence: "heuristic_candidate"`, a
`status` of `searching`, `complete` or `failed`, leaf/split/evaluation counts, and
cumulative intrinsic work counters. `max_splits` is between 1 and 1024; it bounds
completed bisections, not elapsed time or arithmetic cost. Each bisection checks
both children. A simple objective can complete during construction. Continue
advancing while `status` is `searching`, then request independent verification:

```json
{"id":"finish","type":"FinishBoxOptimization","payload":{"schema_version":1,"search":"optimization-search:well","optimization":"optimization:well"}}
{"id":"release","type":"ReleaseBoxOptimizationSearch","payload":{"schema_version":1,"search":"optimization-search:well"}}
```

Finish independently reconstructs the complete partition and reevaluates the
original objective using protected verifier compute. Only a successful check
and complete response admission publish the accepted `optimization:` handle.
Finishing a live incomplete search returns `search_not_ready`. A refused Finish
leaves the complete candidate intact. Accepted receipts support the same export
and replay operations as `OptimizeBox`.

Source, typed axes, domain, direction, tolerance, precision and leaf limit are
immutable for the search. Ordinary ledger consumption and intrinsic step,
allocation, source and evaluation counters persist across advances. Successful
batch schedules produce the same candidate and intrinsic totals. Invalid work
quanta leave the search untouched. An actual work failure, including native
meter cancellation, makes it terminal: completed work remains charged, later
advances preserve the failure, and no candidate or value bounds escape from a
partially updated transition. Release frees a resident slot without refunding
budget; at most eight searches can be resident.

Rust callers use `ObjectiveProblem::begin_box_optimization_search` or
`BoxOptimizationSearch::new`, then `advance(max_splits, meter)`, `progress()` and
`counters()`. `completed_witness()` returns only a candidate for
`verify_box_optimization`. The blocking proposal API drives the same search
engine. Session getters expose read-only progress and counters.

## Save and restore a global optimization

A candidate search can be exported before the process exits:

```json
{"id":"checkpoint","type":"ExportBoxOptimizationSearch","payload":{"schema_version":1,"search":"optimization-search:well"}}
```

Save the response's `result.artifact` string unchanged. In a fresh session,
register the same original objective and ordered variables, then send:

```json
{"id":"resume","type":"RestoreBoxOptimizationSearch","payload":{"schema_version":1,"objective":"objective:well","search":"optimization-search:continued","domain":[{"lower":"-2","upper":"2"}],"extremum":"minimum","absolute_tolerance":"1/1000","precision_digits":16,"max_leaves":1024,"artifact":"<the unchanged exported artifact string>"}}
{"id":"continue","type":"AdvanceBoxOptimization","payload":{"schema_version":1,"search":"optimization-search:continued","max_splits":4}}
```

The client must insert the artifact through its JSON encoder. Restoration binds
the exact original expression, typed coordinate order, box, minimum/maximum
direction, tolerance, precision, leaf limit and assumption context. It checks
the complete dyadic topology, preserves chronological tie-breaking, recomputes
every leaf enclosure and reevaluates the proposed feasible point. Cached bounds,
priority queues and accepted flags are not serialized. An unresolved source
domain remains unresolved after restoration.

Restored progress carries `evidence: "heuristic_candidate"`, including when
its status is `complete`. Continue with `FinishBoxOptimization` to obtain
independent protected verification. Importing a checkpoint never inserts an
accepted `optimization:` object. Failed imports leave their requested label
available; existing live searches are never overwritten.

For artifacts exceeding the inline limit, export with
`BeginArtifactExport` and
`"source":{"box_optimization_search":"optimization-search:well"}`, then use
the existing bounded chunk transfer. `RestoreBoxOptimizationSearchFromArtifact`
takes the same expected statement and a completed `transfer:` reference
instead of inline bytes. The distinct `box_optimization_checkpoint` artifact
kind cannot be used as an accepted optimization proof.

Rust callers use `export_box_optimization_checkpoint(search, context, meter)`
and `restore_box_optimization_checkpoint(bytes, problem, domain, kind,
tolerance, options, context, meter)`. The lower-level normalized checkpoint
type has no unrestricted deserializer. Copying a checkpoint consumes intrinsic
search work; encoding, decoding and restoration also consume the caller's
ordinary ledger. Prior intrinsic counters are retained and restoration costs
are added, so repeated resumptions can exhaust the lifetime limits. Sender
counters do not authenticate its complete resource history or enlarge the
receiver's allowance. Clients own persistence of these portable bytes; the
host-managed root-search stores and RaptorQ repair APIs remain root-specific.

The optimization checkpoint implementation and source-review evidence are
recorded in `artifacts/audit/2026-11-14-optimization-checkpoint-progress.json`.
Current-source execution remains pending after the development executor
disconnected; this interface does not close campaign or compatibility gates.

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
checker verifies complete coverage and original-source bounds on every closed
cell, including endpoints, using the protected verifier pool. Smooth cells use
C2 bounds. Supported continuous absolute-value compositions also admit cells
crossing corners: independently computed Lipschitz slopes bound the integral
without asserting a derivative or Hessian at the corner. This lane admits one
finite real interval; improper integrals, principal values and complex contours
are outside its claim.
Reversed endpoints negate the enclosure. Equal endpoints still require the
original source to be admitted at that point. Thus `0*(1/x)` at zero cannot
become a certified zero integral by cancellation.

For example, registering `abs(x-1)` and integrating from zero to three with the
same options produces an enclosure containing the exact area `5/2`, with width
at most `1/1000`. Nested absolute values and supported smooth compositions use
the same public request and portable partition format. An integral certificate
does not authorize C1/C2 derivative or root claims for these corners. Unstable
`Piecewise` boundaries and original poles, including `abs(x)+0*(1/x)` across
zero, still refuse. The failed smooth attempt and the Lipschitz evaluation share
one cumulative resource budget; neither resets accounting.

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


### Contraction bounds on scaled boxes

Root, stationary-point and parametric-family responses report the
`contraction_norm` used for their exact `contraction_bound`. The value
`infinity` uses the maximum absolute coordinate. For
`box_radius_scaled_infinity`, divide each coordinate by its box radius
before taking the maximum. For example, the system `(x-4*y^2, y-x^2/128)`
on `[-2,2] × [-1/4,1/4]` has a certified scaled bound of `1/4`.
The residual equations and reported enclosures stay in their original
coordinates. Each root in a complete isolation result carries its own norm.
A parametric cover reports the largest of its verified cell bounds and uses
`per_leaf` when those cells used different norms.

### Export and replay a root proof from Python

The compatibility shell exposes the same bounded native root-witness format
through `RootBoxReceipt.to_bytes`, `RootBoxReceipt.from_bytes`, and a compiled
residual system's `replay_root_certificate` method:

```python
from fractions import Fraction
from sympy import Rational, RootBoxReceipt, Symbol, certify_root_box, compile_residual_system

x = Symbol("portable_x", positive=True)
equation = x*x - 2
proof = certify_root_box(
    equation, x, [(Fraction(13, 10), Fraction(3, 2))],
    center=[Fraction(7, 5)], preconditioner=[[Fraction(5, 14)]],
    precision=24,
)
payload = proof.to_bytes()

# A receiving process supplies its own exact source and typed coordinate.
received = RootBoxReceipt.from_bytes(payload, residuals=equation, args=x)
assert received.contraction_bound == Rational(1, 14)

program = compile_residual_system(equation, x)
received_again = program.replay_root_certificate(payload)
assert received_again.to_bytes() == payload
```

Receipts from the existing `solve_certified` methods use this route as well.
Source-only replay does not compile a symbolic derivative program or run a
numerical solver. It can recheck supported Piecewise sources under the existing
whole-box branch-stability requirements. A compiled receiver uses the exact
source captured when it was constructed, even if the original Python lists
or matrices are later changed.

The payload must be exact built-in `bytes`, at most 1 MiB, in the canonical
schema-1 encoding. It carries the exact typed source and an untrusted witness;
no serialized acceptance flag or claimed enclosure authorizes a result.
Residual order, held expression structure and complete symbol identities must
match. A same-named Symbol with different assumptions or a different Dummy is
not interchangeable. The independent checker recomputes the original-domain
root claim, and the Python adapter also checks captured declared assumptions
throughout the encoded box before returning its immutable receipt.

Python currently uses the empty ambient assumptions context. Native Session
artifacts can interoperate when that context and the entire typed statement
match; an equivalent printed formula alone does not establish a match.
All three Python methods accept `budget=EvaluationBudget(...)`. The supplied
ledger, or one private bounded ledger when omitted, spans native encoding or
decoding, checking and final publication; exhaustion and cancellation return
no receipt. This Python ledger does not provide the Session's separate
protected verifier allowance. The receipt establishes one unique nonsingular
real root in the stated box, with no claim about roots elsewhere.

## Transfer certificates and checkpoints larger than one frame

All certificate, derivative-program, root-checkpoint and optimization-checkpoint artifact kinds support a chunked route up to the native 1 MiB artifact
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
| `{"parametric_family":"family:affine"}` | `parametric_family` | `ReplayParametricFamilyFromArtifact` |
| `{"compiled_residual_system":"program:source"}` | `compiled_residual_system` | `ReplayCompiledResidualSystemFromArtifact` |
| `{"compiled_objective":"objective_program:source"}` | `compiled_objective` | `ReplayCompiledObjectiveFromArtifact` |

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
Root, integral, stationary-point, global-optimization, parameter-family and
compiled-program replay require protected independent verification of
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
`stationary:<identifier>`, `optimization:<identifier>`, `optimization-search:<identifier>`,
`parametric:<identifier>`, `family:<identifier>`, `sensitivity:<identifier>` and
`transfer:<identifier>`.
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
| Resident global optimization searches | 8; explicit release frees a slot without a budget refund |
| Optimization work quantum | 1–1,024 complete bisections per advance |
| Resident parameter problems / families / sensitivities | 32 each |
| Resident artifact transfers | 8 total imports and exports; explicit release frees a slot |
| Resident root searches | 8; explicit release frees a slot without a budget refund |
| Root-search work quantum | 1–2,047 committed cell transitions per advance |
| Root-search intrinsic allocation lifetime | 256 MiB cumulative, including snapshot copies and restoration work |
| Residual-system residuals / variables | 1–64 / 1–16 |
| Objective variables | 1–64 for registration and compiled sparse minimization; 1–16 for dense minimization and independent exact certificates |
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
An incomplete root or optimization search returns `search_not_ready` when asked to finish.
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
independently checks original-source domains, smooth or admitted Lipschitz bounds,
coverage and enclosure width
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

### Exporting certified residual and Jacobian bounds to IEEE formats

`EncloseResidualJacobianIeee` takes the same registered system, exact `domain`,
and `precision_digits` as `EncloseResidualJacobian`, plus `format` set to
`binary32` or `binary64`. Its response preserves the exact input domain.
The arrays `domain_bits`, `residual_bits`, and `jacobian_bits` follow the
original variable, residual, and matrix row/column order. Each interval is a
pair of lowercase hexadecimal strings in lower/upper order, with 8 or 16 digits
per endpoint. This representation preserves signed zero and infinity in JSON.

For a point input of `1/10` and residual `x`, binary32 returns
`["3dcccccc","3dcccccd"]` for the residual. The bounds enclose the original
mathematical function and its derivatives over the exact input box. The
rounded domain endpoints are for interchange. Evaluating on the larger box
formed by those endpoints requires a new enclosure request on that box.

### Prove a supplied point attains a global extremum

`CertifyBoxExtremum` checks an exact rational point on a closed, finite box
against the registered objective. It takes the coordinates in the objective's
variable order. It also works with the objective registered by
`CompileLeastSquaresObjective`, so a bounded fit can be checked by submitting
an exact candidate in a separate request.

```json
{"id":"source","type":"RegisterObjective","payload":{"schema_version":1,"objective":"objective:face","source":"5*x-x^2+(1+x)*(y-1/3)^2","variables":["y","x"]}}
{"id":"check","type":"CertifyBoxExtremum","payload":{"schema_version":1,"objective":"objective:face","domain":[{"lower":"-1","upper":"1"},{"lower":"0","upper":"2"}],"point":["1/3","0"],"extremum":"minimum","precision_digits":24}}
```

Here the minimum value is exactly zero and the minimizer is unique. The full
objective increases in `x` throughout the box. At `x=0`, the remaining
quadratic reaches its unique minimum at `y=1/3`. The response includes the
point, value enclosure, gradient at the point, Hessian over the original box
and `uniqueness_proved`. A false uniqueness flag leaves uniqueness open.
Choose `maximum` for the reversed inequality. The original-source check
consumes protected verifier fuel. Insufficient curvature or first-order
evidence returns `inconclusive`.

For an exact bounded least-squares fit, start with the registered residual
system. `RegisterLeastSquaresObjective` stores one half the sum of squares
of its original rows in their declared order. Each repeated row contributes
separately and each held expression keeps its original domain. The resulting
objective works with `EncloseObjective`, `CertifyBoxExtremum`, and the other
source-based objective operations. Numerical evaluation or optimization can
subsequently compile the objective if needed.

```json
{"id":"rows","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:fit","variables":["x","y","z"],"residuals":["x-3","y+4","z-8","2"]}}
{"id":"loss","type":"RegisterLeastSquaresObjective","payload":{"schema_version":1,"system":"system:fit","objective":"objective:loss"}}
{"id":"minimum","type":"CertifyBoxExtremum","payload":{"schema_version":1,"objective":"objective:loss","domain":[{"lower":"-5","upper":"1"},{"lower":"-1","upper":"5"},{"lower":"2","upper":"2"}],"point":["1","-1","2"],"extremum":"minimum","precision_digits":24}}
```

This gives the unique minimizer `(1,-1,2)` and an exact optimum value of `53/2`.
