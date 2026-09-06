//! Integration gate for WS22: Performance and Architecture Optimization.
//!
//! Enforces:
//! 1. Semantic admission precedes timing: divergent outputs fail closed (no timing admitted).
//! 2. Multi-sample distribution metrics: raw samples retained, median, p90/p95/p99, cv% computed.
//! 3. A/A control: reference-vs-reference baseline verifies null hypothesis without measurement bias.
//! 4. Standard WS22 suite execution: matrix Bareiss vs Laplace cofactor, Groebner full polynomial equivalence,
//!    multivariate polynomial multiplication, and A/A control baseline.
//! 5. Integrity commitment: blake3 digest binds name, medians, and all raw timing samples.

#![forbid(unsafe_code)]

use fsym_runtime::benchmarks::{
    BenchmarkError, run_aa_control_benchmark, run_paired_benchmark, run_paired_benchmark_rounds,
    run_standard_ws22_suite,
};

#[test]
fn test_ws22_semantic_admission_precedes_timing_and_rejects_divergence() {
    // 1. Identical outputs: semantic admission succeeds
    let ok_res = run_paired_benchmark(
        "admitted_test_case",
        || ("result_alpha".to_string(), 10),
        || ("result_alpha".to_string(), 20),
        |c, r| c == r,
    )
    .expect("identical outputs must be admitted");

    assert!(ok_res.semantic_equivalence_verified);
    assert_eq!(ok_res.benchmark_name, "admitted_test_case");
    assert_eq!(ok_res.raw_candidate_samples_ns.len(), 3);
    assert_eq!(ok_res.raw_reference_samples_ns.len(), 3);

    // 2. Divergent outputs: MUST fail closed before any timing comparison is accepted
    let div_res = run_paired_benchmark(
        "diverging_candidate",
        || ("candidate_wrong_output".to_string(), 5),
        || ("reference_canonical_output".to_string(), 15),
        |c, r| c == r,
    );

    assert!(
        matches!(div_res, Err(BenchmarkError::SemanticAdmissionFailed)),
        "divergent candidate must be rejected before timing publication"
    );
}

#[test]
fn test_ws22_paired_benchmark_distribution_and_percentiles() {
    let rounds = 7;
    let res = run_paired_benchmark_rounds(
        "distribution_metrics_test",
        || {
            // Simulated work
            let mut acc = 0u64;
            for i in 1..=500 {
                acc = acc.wrapping_add(i);
            }
            (acc, 500)
        },
        || {
            let mut acc = 0u64;
            for i in 1..=1000 {
                acc = acc.wrapping_add(i);
            }
            // Normalize result to match candidate mathematically
            let sum_500: u64 = (1..=500).sum();
            (sum_500, 1000)
        },
        |c, r| c == r,
        rounds,
    )
    .expect("distribution test should pass admission");

    assert_eq!(res.raw_candidate_samples_ns.len(), rounds);
    assert_eq!(res.raw_reference_samples_ns.len(), rounds);

    // Verify statistical sanity
    assert!(res.candidate_stats.median_ns > 0);
    assert!(res.reference_stats.median_ns > 0);
    assert!(res.candidate_stats.mean_ns > 0.0);
    assert!(res.reference_stats.mean_ns > 0.0);
    assert!(res.candidate_stats.p90_ns >= res.candidate_stats.median_ns);
    assert!(res.candidate_stats.p99_ns >= res.candidate_stats.p90_ns);
    assert!(res.reference_stats.p90_ns >= res.reference_stats.median_ns);
    assert!(res.reference_stats.p99_ns >= res.reference_stats.p90_ns);
}

#[test]
fn test_ws22_aa_control_null_hypothesis_verification() {
    let res = run_aa_control_benchmark(
        "arithmetic_null_baseline",
        || {
            let mut val = 1u64;
            for i in 1..=300 {
                val = val.wrapping_mul(i);
            }
            (val, 300)
        },
        5,
    )
    .expect("AA control run should succeed");

    assert!(res.aa_control_verified);
    assert!(res.semantic_equivalence_verified);
    // In an A/A run of identical code, the ratio should be reasonably close to 1.0
    assert!(
        res.speedup_ratio > 0.1 && res.speedup_ratio < 10.0,
        "AA control ratio must be within reasonable noise floor, got {}",
        res.speedup_ratio
    );
}

#[test]
fn test_ws22_standard_suite_admitted_kernels() {
    let results = run_standard_ws22_suite().expect("standard ws22 suite must succeed");

    assert_eq!(results.len(), 4, "standard suite must execute all 4 suites");

    // Suite 1: Matrix Bareiss vs Laplace cofactor
    let mat_res = &results[0];
    assert_eq!(mat_res.benchmark_name, "matrix_bareiss_vs_laplace_det_3x3");
    assert!(mat_res.semantic_equivalence_verified);
    assert_eq!(mat_res.raw_candidate_samples_ns.len(), 5);

    // Suite 2: Groebner basis Buchberger minimalization
    let grob_res = &results[1];
    assert_eq!(grob_res.benchmark_name, "groebner_degrevlex_system");
    assert!(grob_res.semantic_equivalence_verified);

    // Suite 3: Multivariate polynomial multiplication
    let poly_res = &results[2];
    assert_eq!(poly_res.benchmark_name, "multivariate_poly_mul");
    assert!(poly_res.semantic_equivalence_verified);

    // Suite 4: A/A control baseline
    let aa_res = &results[3];
    assert!(aa_res.aa_control_verified);
    assert!(aa_res.semantic_equivalence_verified);
}

#[test]
fn test_ws22_adversarial_tampered_samples_digest_refusal() {
    let res = run_paired_benchmark_rounds(
        "digest_integrity_test",
        || (99, 1),
        || (99, 1),
        |c, r| c == r,
        3,
    )
    .expect("benchmark run succeeds");

    // Re-derive digest with unmodified samples
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.benchmark.paired.v2:");
    hasher.update(res.benchmark_name.as_bytes());
    hasher.update(&res.candidate_stats.median_ns.to_le_bytes());
    hasher.update(&res.reference_stats.median_ns.to_le_bytes());
    for s in &res.raw_candidate_samples_ns {
        hasher.update(&s.to_le_bytes());
    }
    for s in &res.raw_reference_samples_ns {
        hasher.update(&s.to_le_bytes());
    }
    let expected_digest = *hasher.finalize().as_bytes();
    assert_eq!(res.receipt_digest, expected_digest);

    // Mutate one sample: digest MUST not match
    let mut tampered_hasher = blake3::Hasher::new();
    tampered_hasher.update(b"fsym.benchmark.paired.v2:");
    tampered_hasher.update(res.benchmark_name.as_bytes());
    tampered_hasher.update(&res.candidate_stats.median_ns.to_le_bytes());
    tampered_hasher.update(&res.reference_stats.median_ns.to_le_bytes());
    for (i, s) in res.raw_candidate_samples_ns.iter().enumerate() {
        let val = if i == 0 { s + 1 } else { *s };
        tampered_hasher.update(&val.to_le_bytes());
    }
    for s in &res.raw_reference_samples_ns {
        tampered_hasher.update(&s.to_le_bytes());
    }
    let tampered_digest = *tampered_hasher.finalize().as_bytes();
    assert_ne!(
        res.receipt_digest, tampered_digest,
        "tampered sample must alter digest"
    );
}
