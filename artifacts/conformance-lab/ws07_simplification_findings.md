# WS07 Simplification-Family Differential Findings (2026-09-21)

Method: same driver under shell and pinned SymPy 1.14.0 oracle; 156
probes (12 functions x 13 expression shapes); exact srepr payload
comparison. Reproducer: ws07_simplification_differential.py.

Result: 66 divergences in four classes.

## Class 1: construction-time over-evaluation (6)
`sin(x)**2 + cos(x)**2` collapses to 1 AT CONSTRUCTION in the shell;
the oracle keeps the unevaluated Add until simplify/trigsimp. Every
pass-through function (expand/powsimp/ratsimp/together/cancel/collect)
inherits it. Root: native Add canonicalization eagerly folds the
Pythagorean identity. Fix belongs in the native kernel's trig
canonicalization policy, gated behind explicit simplification calls.

## Class 2: Mul factor storage order (~35)
`tan(x)*cos(x)` stores (tan, cos); the oracle stores sorted
(cos, tan) by default_sort_key. The registered corpus (236 fixtures)
does not exercise non-sorted symbol-order Muls, masking this in the
corpus gates. Fix: native Mul factor canonicalization must sort
non-numeric factors by sort key.

## Class 3: simplify depth (part of "other" ~20)
oracle simplify(tan*cos) -> sin(x) (applies trig identities); shell
returns the product. simplify needs the trig rule universe before
parity.

## Class 4: apart domain guards (5)
shell apart SUCCEEDS on inputs where the oracle raises
NotImplementedError('multivariate partial fraction decomposition'):
trig1, trig3, rad1, pow3, rat2. Parity requires the same refusals
(shell is currently more permissive than the pinned profile).

## Status
NOT FIXED - native-kernel work, gated on a bounded WS07 slice. The
registered 236-fixture corpus does not regress (verified: corpus gate
green); these shapes are outside the current corpus and constitute the
next corpus-expansion target once the native fixes land.
