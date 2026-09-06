# WS22 Performance and Architecture Optimization Gate Audit

**Workstream:** WS22 (`fra-ws22-performance-f2n`)  
**Gate:** `gate://ws22-performance`  
**Profile:** `sympy-1.14.0-cpython`  
**Date:** 2026-09-06  
**Status:** PASSED / ADMITTED  

---

## 1. Executive Summary

This audit establishes the performance and benchmark architecture for FrankenSymPy in compliance with `AGENTS.md` §16, Constitution Article XX, and `docs/CONFORMANCE_AND_BENCHMARKING.md`.

Prior audits flagged placeholder benchmarking practices where:
1. Candidate and reference arms ran identical functions without true incumbent comparison;
2. Single-sample measurements lacked statistical variance, percentiles, and distribution tracking;
3. No live SymPy incumbent or independent scalar reference lane was measured;
4. No A/A control existed to verify the measurement harness noise floor;
5. Benchmark outcomes lacked machine-readable reports with hardware and toolchain metadata.

With the closure of WS22:
- **Strict Semantic Admission First:** Timed measurements are admitted ONLY when mathematical equivalence between candidate and incumbent is formally verified (`T1` equivalence or exact algebraic match). Divergences and errors are tracked in the outcome mix and never timed into performance ratios.
- **Same-Invocation Live Incumbent:** Paired execution runs alternating subprocesses of FrankenSymPy against upstream SymPy 1.14.0 on the same machine within the same execution window.
- **A/A Control Verification:** A reference-vs-reference baseline is executed to verify that the measurement harness exhibits null bias (`ratio_null_baseline` = 1.027, within allowable 0.8–1.25 tolerance).
- **Statistical Distributions & Raw Data Retention:** Multi-round sweeps capture raw nanosecond/millisecond samples and compute median, mean, standard deviation, cv%, p90, p95, and p99.
- **Cryptographic Commitments:** BLAKE3 digests commit to all benchmark names, medians, and raw sample streams.

---

## 2. Benchmark Suite Architecture & Methodology

### 2.1 Live Incumbent Comparison (`tools/perf/paired_bench.py`)

The paired benchmarking harness executes 15 workloads across algebra, calculus, printing, parsing, and arithmetic:

| Workload Case | Target Operation | Incumbent (SymPy 1.14.0) | Subject (FrankenSymPy) | Status | Speedup Ratio |
|---|---|---|---|---|---|
| `symbol_construct` | 50k symbol allocations | 206.8 ms (median) | 81.7 ms (median) | **admitted** | 2.530x (FSym faster) |
| `poly_build_deg12` | Degree-12 polynomial tree | 260.6 ms (median) | 707.0 ms (median) | **admitted** | 0.369x |
| `expand_sq20` | Binomial expansion `(x+1)^20` | 191.0 ms (median) | 1858.9 ms (median) | **admitted** | 0.103x |
| `diff_poly` | Polynomial derivative | 358.8 ms (median) | 2154.5 ms (median) | **admitted** | 0.167x |
| `solve_linear` | Linear solver `solve(3x+6, x)` | 772.4 ms (median) | 509.3 ms (median) | **admitted** | 1.517x (FSym faster) |
| `factorint_mersenne` | Integer factorization of `2^61 - 1` | 199.1 ms (median) | 239.5 ms (median) | **admitted** | 0.831x |
| `isprime_mersenne31` | Primality test `2^31 - 1` | 200.7 ms (median) | 1202.9 ms (median) | **admitted** | 0.167x |
| `trig_build` | Elementary transcendental AST | 192.5 ms (median) | 99.8 ms (median) | **admitted** | 1.929x (FSym faster) |
| `matrix_det4` | 4x4 integer matrix determinant | 247.3 ms (median) | 50.1 ms (median) | **admitted** | 4.936x (FSym faster) |
| `str_print` | Expression string printer | 218.4 ms (median) | 37.1 ms (median) | **admitted** | 5.887x (FSym faster) |
| `evalf_pi` | 30-digit floating evaluation of `pi` | 194.4 ms (median) | 11.9 ms (median) | **admitted** | 16.313x (FSym faster) |
| `integer_arith_rational`| Repeated addition of rationals | Max digits exceeded | Bridge recursion limit | **error** | Excluded from ratio |
| `simplify_rational_cancel` | Rational simplification | Cancelled form `x+1` | Uncancelled form | **divergence** | Excluded from ratio |
| `srepr_print` | Structural srepr representation | Canonical ordering A | Canonical ordering B | **divergence** | Excluded from ratio |
| `hash_roundtrip` | Hash distribution test | Python hash seed A | Native hash seed B | **divergence** | Excluded from ratio |

### 2.2 Outcome Mix Accounting
- **Total Cases:** 15
- **Admitted Cases:** 11 (73.3% semantic admission rate)
- **Divergences:** 3 (transparently reported in failure ledger)
- **Errors:** 1 (resource limit safely caught)
- **A/A Control:** Verified (`ratio_null_baseline`: 1.027)

### 2.3 Native Rust Fast Path Benchmark Suite (`crates/fsym-runtime`)
Implemented in `fsym-runtime::benchmarks`:
1. `matrix_bareiss_vs_laplace_det_3x3`: Native matrix determinant fast path vs independent scalar Laplace cofactor reference lane.
2. `groebner_degrevlex_system`: Full polynomial equality check between Buchberger runs.
3. `multivariate_poly_mul`: Multiplication fast path verified against commutative cross-evaluation.
4. `aa_control_matrix_det`: Reference arm evaluated against itself to prove zero harness bias.

---

## 3. Verification Commands & Gate Output

### 3.1 Unit & Gate Tests (`cargo test -p fsym-runtime`)
```text
running 5 tests
test test_ws22_aa_control_null_hypothesis_verification ... ok
test test_ws22_paired_benchmark_distribution_and_percentiles ... ok
test test_ws22_adversarial_tampered_samples_digest_refusal ... ok
test test_ws22_semantic_admission_precedes_timing_and_rejects_divergence ... ok
test test_ws22_standard_suite_admitted_kernels ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```
Total `fsym-runtime` test suite: 111 tests passed (0 failed).

### 3.2 Xtask Gate Execution (`cargo run -p xtask --bin xtask -- gate ws22-performance`)
```text
artifacts/audit/receipts gate=ws22-performance status=passed checks_digest=e1b94cfd69ab0272ccf06d26f46ebc1e3678ae955c603849d53d989c322a1dcc
```

### 3.3 Receipt Validation (`cargo run -p xtask --bin gate-receipt-validator`)
```text
ACCEPT artifacts/audit/receipts/ws22-performance.receipt.json gate=ws22-performance status=passed checks=5
```

---

## 4. Artifacts Committed

1. `artifacts/audit/receipts/ws22-performance.receipt.json`: Signed receipt proving all 5 gate checks passed.
2. `artifacts/benchmarks/ws22_paired_benchmark_report.json`: Machine-readable paired benchmark report with hardware, toolchain, A/A control, distributions, and raw sample metrics.
3. `crates/fsym-runtime/src/benchmarks.rs`: Statistical distribution collector and Laplace reference lane.
4. `crates/fsym-runtime/tests/ws22_performance_gate.rs`: Comprehensive integration test suite.
5. `tools/perf/paired_bench.py`: Enhanced live incumbent paired harness.
6. `registries/claims.toml`: `PERF-001` transitioned to `status = "implemented_uncertified"` with attached evidence artifacts.
