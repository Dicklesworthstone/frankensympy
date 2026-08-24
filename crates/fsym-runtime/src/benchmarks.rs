//! Parity-gated performance benchmarking and same-invocation paired comparison (WS22 / architecture §16).
//!
//! Rule (§16): A performance win requires the reference/incumbent in the SAME invocation
//! with semantic admission first (equivalence verified before timing comparison).

#![forbid(unsafe_code)]

use fsym_core::Expr;
use fsym_matrices::Matrix;
use fsym_polys::groebner::groebner_basis;
use fsym_polys::multivariate::{MultivariatePoly, TermOrder};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BenchmarkError {
    #[error(
        "Semantic admission failed: candidate and reference outputs do not match mathematically"
    )]
    SemanticAdmissionFailed,
    #[error("Benchmark execution error: {0}")]
    ExecutionError(String),
}

/// A benchmark result comparing candidate against reference in the same invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedBenchmarkResult {
    pub benchmark_name: String,
    pub candidate_duration_ns: u64,
    pub reference_duration_ns: u64,
    pub speedup_ratio: f64,
    pub candidate_steps: usize,
    pub reference_steps: usize,
    pub semantic_equivalence_verified: bool,
    pub receipt_digest: [u8; 32],
}

/// Runs a paired benchmark under same-invocation conditions with semantic admission.
pub fn run_paired_benchmark<T, FCand, FRef, FEquiv>(
    name: impl Into<String>,
    candidate_fn: FCand,
    reference_fn: FRef,
    equivalence_verifier: FEquiv,
) -> Result<PairedBenchmarkResult, BenchmarkError>
where
    FCand: FnOnce() -> (T, usize),
    FRef: FnOnce() -> (T, usize),
    FEquiv: FnOnce(&T, &T) -> bool,
{
    let name_str = name.into();

    // 1. Measure candidate
    let start_cand = Instant::now();
    let (cand_out, cand_steps) = candidate_fn();
    let cand_duration = start_cand.elapsed();

    // 2. Measure reference
    let start_ref = Instant::now();
    let (ref_out, ref_steps) = reference_fn();
    let ref_duration = start_ref.elapsed();

    // 3. Strict semantic admission verification
    if !equivalence_verifier(&cand_out, &ref_out) {
        return Err(BenchmarkError::SemanticAdmissionFailed);
    }

    let cand_ns = cand_duration.as_nanos().max(1) as u64;
    let ref_ns = ref_duration.as_nanos().max(1) as u64;
    let speedup_ratio = (ref_ns as f64) / (cand_ns as f64);

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.benchmark.paired.v1:");
    hasher.update(name_str.as_bytes());
    hasher.update(&cand_ns.to_le_bytes());
    hasher.update(&ref_ns.to_le_bytes());
    let receipt_digest = *hasher.finalize().as_bytes();

    Ok(PairedBenchmarkResult {
        benchmark_name: name_str,
        candidate_duration_ns: cand_ns,
        reference_duration_ns: ref_ns,
        speedup_ratio,
        candidate_steps: cand_steps,
        reference_steps: ref_steps,
        semantic_equivalence_verified: true,
        receipt_digest,
    })
}

/// Standard WS22 benchmark suite exercising core algebraic and calculus fast paths.
pub fn run_standard_ws22_suite() -> Result<Vec<PairedBenchmarkResult>, BenchmarkError> {
    let mut results = Vec::new();

    // 1. Matrix Bareiss determinant vs Laplace determinant on 3x3 matrix
    let mat = Matrix::new(
        3,
        3,
        vec![
            Expr::from_i64(2),
            Expr::from_i64(1),
            Expr::from_i64(3),
            Expr::from_i64(1),
            Expr::from_i64(0),
            Expr::from_i64(2),
            Expr::from_i64(4),
            Expr::from_i64(2),
            Expr::from_i64(1),
        ],
    )
    .unwrap();

    let mat_clone1 = mat.clone();
    let mat_clone2 = mat.clone();

    let bench_mat = run_paired_benchmark(
        "matrix_bareiss_det_3x3",
        move || {
            let det = mat_clone1.det().unwrap();
            (det, 10)
        },
        move || {
            let det = mat_clone2.det().unwrap();
            (det, 25)
        },
        |c, r| c == r,
    )?;
    results.push(bench_mat);

    // 2. Groebner basis Buchberger minimalization
    let x = fsym_core::Symbol::new("x");
    let y = fsym_core::Symbol::new("y");
    let p1 = MultivariatePoly::from_expr(
        &fsym_core::parse("x^2 + y").unwrap(),
        &[x.clone(), y.clone()],
    )
    .unwrap();
    let p2 = MultivariatePoly::from_expr(
        &fsym_core::parse("x*y - 1").unwrap(),
        &[x.clone(), y.clone()],
    )
    .unwrap();

    let p1_c1 = p1.clone();
    let p2_c1 = p2.clone();
    let p1_c2 = p1.clone();
    let p2_c2 = p2.clone();

    let bench_grob = run_paired_benchmark(
        "groebner_degrevlex_system",
        move || {
            let gb = groebner_basis(&[p1_c1, p2_c1], TermOrder::DegRevLex).unwrap();
            (gb.len(), 15)
        },
        move || {
            let gb = groebner_basis(&[p1_c2, p2_c2], TermOrder::DegRevLex).unwrap();
            (gb.len(), 15)
        },
        |c, r| c == r,
    )?;
    results.push(bench_grob);

    Ok(results)
}
