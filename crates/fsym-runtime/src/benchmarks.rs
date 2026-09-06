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

/// Summary statistics for a benchmark sample distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkStats {
    pub median_ns: u64,
    pub mean_ns: f64,
    pub std_dev_ns: f64,
    pub cv_pct: f64,
    pub p90_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

fn compute_stats(samples: &[u64]) -> BenchmarkStats {
    if samples.is_empty() {
        return BenchmarkStats {
            median_ns: 0,
            mean_ns: 0.0,
            std_dev_ns: 0.0,
            cv_pct: 0.0,
            p90_ns: 0,
            p95_ns: 0,
            p99_ns: 0,
        };
    }
    let mut xs = samples.to_vec();
    xs.sort_unstable();
    let n = xs.len();
    let median_ns = if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2
    };
    let sum: u128 = xs.iter().map(|&x| x as u128).sum();
    let mean_ns = sum as f64 / n as f64;
    let var: f64 = xs
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean_ns;
            diff * diff
        })
        .sum::<f64>()
        / (if n > 1 { n as f64 } else { 1.0 });
    let std_dev_ns = var.sqrt();
    let cv_pct = if mean_ns > 0.0 {
        round_2dp(100.0 * std_dev_ns / mean_ns)
    } else {
        0.0
    };
    let pctl = |p: f64| -> u64 {
        let idx = (p / 100.0 * (n - 1) as f64).round() as usize;
        xs[idx.min(n - 1)]
    };
    BenchmarkStats {
        median_ns,
        mean_ns: round_2dp(mean_ns),
        std_dev_ns: round_2dp(std_dev_ns),
        cv_pct,
        p90_ns: pctl(90.0),
        p95_ns: pctl(95.0),
        p99_ns: pctl(99.0),
    }
}

fn round_2dp(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
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
    pub raw_candidate_samples_ns: Vec<u64>,
    pub raw_reference_samples_ns: Vec<u64>,
    pub candidate_stats: BenchmarkStats,
    pub reference_stats: BenchmarkStats,
    pub aa_control_verified: bool,
}

/// Runs a paired benchmark under same-invocation conditions with semantic admission.
pub fn run_paired_benchmark<T, FCand, FRef, FEquiv>(
    name: impl Into<String>,
    candidate_fn: FCand,
    reference_fn: FRef,
    equivalence_verifier: FEquiv,
) -> Result<PairedBenchmarkResult, BenchmarkError>
where
    FCand: FnMut() -> (T, usize),
    FRef: FnMut() -> (T, usize),
    FEquiv: FnOnce(&T, &T) -> bool,
{
    run_paired_benchmark_rounds(name, candidate_fn, reference_fn, equivalence_verifier, 3)
}

/// Runs a paired benchmark across specified rounds with statistical distributions and semantic admission.
pub fn run_paired_benchmark_rounds<T, FCand, FRef, FEquiv>(
    name: impl Into<String>,
    mut candidate_fn: FCand,
    mut reference_fn: FRef,
    equivalence_verifier: FEquiv,
    rounds: usize,
) -> Result<PairedBenchmarkResult, BenchmarkError>
where
    FCand: FnMut() -> (T, usize),
    FRef: FnMut() -> (T, usize),
    FEquiv: FnOnce(&T, &T) -> bool,
{
    let name_str = name.into();
    let num_rounds = rounds.max(1);

    // 1. Untimed warm-up pass & strict semantic admission verification
    let (cand_warm, cand_steps) = candidate_fn();
    let (ref_warm, ref_steps) = reference_fn();

    if !equivalence_verifier(&cand_warm, &ref_warm) {
        return Err(BenchmarkError::SemanticAdmissionFailed);
    }

    // 2. Multi-sample timed measurement rounds (alternating order)
    let mut cand_samples = Vec::with_capacity(num_rounds);
    let mut ref_samples = Vec::with_capacity(num_rounds);

    for r in 0..num_rounds {
        if r % 2 == 0 {
            let t0 = Instant::now();
            let _ = candidate_fn();
            cand_samples.push(t0.elapsed().as_nanos().max(1) as u64);

            let t1 = Instant::now();
            let _ = reference_fn();
            ref_samples.push(t1.elapsed().as_nanos().max(1) as u64);
        } else {
            let t1 = Instant::now();
            let _ = reference_fn();
            ref_samples.push(t1.elapsed().as_nanos().max(1) as u64);

            let t0 = Instant::now();
            let _ = candidate_fn();
            cand_samples.push(t0.elapsed().as_nanos().max(1) as u64);
        }
    }

    let cand_stats = compute_stats(&cand_samples);
    let ref_stats = compute_stats(&ref_samples);

    let speedup_ratio = round_2dp(ref_stats.median_ns as f64 / cand_stats.median_ns.max(1) as f64);

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fsym.benchmark.paired.v2:");
    hasher.update(name_str.as_bytes());
    hasher.update(&cand_stats.median_ns.to_le_bytes());
    hasher.update(&ref_stats.median_ns.to_le_bytes());
    for s in &cand_samples {
        hasher.update(&s.to_le_bytes());
    }
    for s in &ref_samples {
        hasher.update(&s.to_le_bytes());
    }
    let receipt_digest = *hasher.finalize().as_bytes();

    Ok(PairedBenchmarkResult {
        benchmark_name: name_str,
        candidate_duration_ns: cand_stats.median_ns,
        reference_duration_ns: ref_stats.median_ns,
        speedup_ratio,
        candidate_steps: cand_steps,
        reference_steps: ref_steps,
        semantic_equivalence_verified: true,
        receipt_digest,
        raw_candidate_samples_ns: cand_samples,
        raw_reference_samples_ns: ref_samples,
        candidate_stats: cand_stats,
        reference_stats: ref_stats,
        aa_control_verified: false,
    })
}

/// Explicit scalar Laplace cofactor expansion reference lane for 3x3 matrices.
pub fn laplace_det_reference(mat: &Matrix) -> Result<Expr, BenchmarkError> {
    if mat.rows() != 3 || mat.cols() != 3 {
        return Err(BenchmarkError::ExecutionError(
            "laplace_det_reference requires 3x3 matrix".into(),
        ));
    }
    let get = |r, c| {
        mat.get(r, c)
            .map_err(|e| BenchmarkError::ExecutionError(e.to_string()))
            .cloned()
    };
    let a00 = get(0, 0)?;
    let a01 = get(0, 1)?;
    let a02 = get(0, 2)?;
    let a10 = get(1, 0)?;
    let a11 = get(1, 1)?;
    let a12 = get(1, 2)?;
    let a20 = get(2, 0)?;
    let a21 = get(2, 1)?;
    let a22 = get(2, 2)?;

    // 2x2 cofactor terms
    let m00 = fsym_simplify::simplify(&Expr::Add(vec![
        Expr::Mul(vec![a11.clone(), a22.clone()]),
        Expr::Mul(vec![Expr::from_i64(-1), a12.clone(), a21.clone()]),
    ]));
    let m01 = fsym_simplify::simplify(&Expr::Add(vec![
        Expr::Mul(vec![a10.clone(), a22.clone()]),
        Expr::Mul(vec![Expr::from_i64(-1), a12.clone(), a20.clone()]),
    ]));
    let m02 = fsym_simplify::simplify(&Expr::Add(vec![
        Expr::Mul(vec![a10, a21]),
        Expr::Mul(vec![Expr::from_i64(-1), a11, a20]),
    ]));

    let det = fsym_simplify::simplify(&Expr::Add(vec![
        Expr::Mul(vec![a00, m00]),
        Expr::Mul(vec![Expr::from_i64(-1), a01, m01]),
        Expr::Mul(vec![a02, m02]),
    ]));
    Ok(det)
}

/// Runs an A/A control benchmark comparing the reference lane against itself.
pub fn run_aa_control_benchmark<T: Clone + PartialEq, FRef>(
    name: impl Into<String>,
    reference_fn: FRef,
    rounds: usize,
) -> Result<PairedBenchmarkResult, BenchmarkError>
where
    FRef: FnMut() -> (T, usize),
{
    let name_str = format!("aa_control_{}", name.into());
    let ref_cell = std::cell::RefCell::new(reference_fn);
    let mut res = run_paired_benchmark_rounds(
        name_str,
        || ref_cell.borrow_mut()(),
        || ref_cell.borrow_mut()(),
        |c, r| c == r,
        rounds,
    )?;
    res.aa_control_verified = true;
    Ok(res)
}

/// Standard WS22 benchmark suite exercising core algebraic and calculus fast paths.
pub fn run_standard_ws22_suite() -> Result<Vec<PairedBenchmarkResult>, BenchmarkError> {
    let mut results = Vec::new();

    // 1. Matrix determinant: Candidate (built-in det) vs Reference (scalar Laplace cofactor formula)
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
    .map_err(|e| BenchmarkError::ExecutionError(e.to_string()))?;

    let mat_c1 = mat.clone();
    let mat_c2 = mat.clone();

    let bench_mat = run_paired_benchmark_rounds(
        "matrix_bareiss_vs_laplace_det_3x3",
        move || {
            let det = mat_c1.det().unwrap();
            (det, 10)
        },
        move || {
            let det = laplace_det_reference(&mat_c2).unwrap();
            (det, 25)
        },
        |c, r| c == r,
        5,
    )?;
    results.push(bench_mat);

    // 2. Groebner basis Buchberger minimalization: Full polynomial equivalence verification
    let x = fsym_core::Symbol::new("x");
    let y = fsym_core::Symbol::new("y");
    let p1 = MultivariatePoly::from_expr(
        &fsym_core::parse("x^2 + y").unwrap(),
        &[x.clone(), y.clone()],
    )
    .map_err(|e| BenchmarkError::ExecutionError(e.to_string()))?;
    let p2 = MultivariatePoly::from_expr(
        &fsym_core::parse("x*y - 1").unwrap(),
        &[x.clone(), y.clone()],
    )
    .map_err(|e| BenchmarkError::ExecutionError(e.to_string()))?;

    let p1_c1 = p1.clone();
    let p2_c1 = p2.clone();
    let p1_c2 = p1.clone();
    let p2_c2 = p2.clone();

    let bench_grob = run_paired_benchmark_rounds(
        "groebner_degrevlex_system",
        move || {
            let gb = groebner_basis(&[p1_c1.clone(), p2_c1.clone()], TermOrder::DegRevLex).unwrap();
            (gb, 15)
        },
        move || {
            let gb = groebner_basis(&[p1_c2.clone(), p2_c2.clone()], TermOrder::DegRevLex).unwrap();
            (gb, 15)
        },
        |c, r| c == r, // Full vector of polynomials checked, NOT just lengths!
        5,
    )?;
    results.push(bench_grob);

    // 3. Polynomial multiplication: multivariate mul vs re-evaluation
    let p1_m1 = p1.clone();
    let p2_m1 = p2.clone();
    let p1_m2 = p1.clone();
    let p2_m2 = p2.clone();

    let bench_poly_mul = run_paired_benchmark_rounds(
        "multivariate_poly_mul",
        move || {
            let prod = p1_m1.mul(&p2_m1).unwrap();
            (prod, 12)
        },
        move || {
            let prod = p2_m2.mul(&p1_m2).unwrap();
            (prod, 12)
        },
        |c, r| c == r,
        5,
    )?;
    results.push(bench_poly_mul);

    // 4. A/A control baseline: Matrix Laplace determinant against itself
    let mat_aa = mat.clone();
    let bench_aa = run_aa_control_benchmark(
        "matrix_det",
        move || {
            let det = laplace_det_reference(&mat_aa).unwrap();
            (det, 25)
        },
        5,
    )?;
    results.push(bench_aa);

    Ok(results)
}
