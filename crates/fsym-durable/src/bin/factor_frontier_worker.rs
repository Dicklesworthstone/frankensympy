//! Fresh-process worker for the durable factor-frontier crash matrix.
//!
//! `generate <store_root> <boundary> <marker_base>` runs the real
//! two-generator factor race (metered Zassenhaus modular/Hensel and metered
//! Kronecker interpolation over x^4+4), persists the typed continuation
//! through the durable prepare → verify → commit sequence, and stops at the
//! requested boundary: it writes `<marker_base>.<boundary>` and then sleeps
//! until the parent SIGKILLs it, so the on-disk state at kill time is exactly
//! the boundary state and nothing past the boundary executes.
//!
//! `resume <store_root> <marker_base> <boundary>` is a genuinely fresh
//! process: it reads the marker left by `generate`, loads the committed
//! record (promoting the staged one first when the crash happened before
//! commit), validates digest/schema/universe/dependency pins, decodes the
//! typed checkpoint, mathematically re-verifies every drained candidate
//! through the runtime's independent verifier, and writes
//! `<marker_base>.resume.json` describing the accepted result.

#![forbid(unsafe_code)]

use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::{Budget, BudgetLimits, Dimension};
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_durable::{
    DependencyManifest, DurableRecord, DurableStore, FileStore, FsqliteCliStore, PreparedHandle,
    factor_race_universe_id,
};
use fsym_polys::factorization::{metered_complete_factorization, metered_kronecker_factorization};
use fsym_polys::univariate::UnivariatePoly;
use fsym_proof_kernel::{Claim, ProofKernel};
use fsym_runtime::{
    FACTOR_RACE_CONTINUATION_SCHEMA, FactorRaceContinuationState, FsymCpuCx, FsymCx,
    PortfolioCandidate, PortfolioError, RepairSidecar, TypedCheckpoint, factor_race_generate,
    resume_factor_race_from_checkpoint,
};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

const INPUT_COEFFS: [i64; 5] = [4, 0, 0, 0, 1];
const REPAIR_SYMBOL_SIZE: usize = 512;
const REPAIR_SYMBOL_COUNT: usize = 4;

/// Dependency pins bound into every record this worker produces. The crash
/// matrix parent presents the identical map as its validation expectation.
fn dependency_pins() -> DependencyManifest {
    DependencyManifest::new([
        ("factor_race_operation", "factor_race_v1"),
        ("fsym_runtime", env!("CARGO_PKG_VERSION")),
        ("repair_codec", "raptorq-rfc6330-v1"),
    ])
}

fn ipoly(coeffs: &[i64]) -> UnivariatePoly {
    UnivariatePoly::new(
        Symbol::new("x"),
        coeffs
            .iter()
            .map(|c| BigRational::from_integer(BigInt::from(*c)))
            .collect(),
    )
}

/// Normalizes a factorization into the canonical product expression the race
/// fixes as the requested claim (the same fixture contract as the runtime
/// portfolio race tests).
fn normalize_factor_product(
    scale: &BigRational,
    factors: Vec<(UnivariatePoly, usize)>,
) -> Result<Expr, PortfolioError> {
    let one = BigRational::from_integer(BigInt::from(1));
    let mut terms: Vec<Expr> = Vec::new();
    if *scale != one {
        terms.push(Expr::Rational(scale.clone()));
    }
    let mut factors = factors;
    factors.sort_by(|(left, _), (right, _)| {
        left.degree()
            .cmp(&right.degree())
            .then_with(|| left.coeffs.cmp(&right.coeffs))
    });
    for (poly, multiplicity) in factors {
        let term = poly.to_expr();
        terms.push(if multiplicity == 1 {
            term
        } else {
            let multiplicity = u64::try_from(multiplicity)
                .map_err(|_| PortfolioError::InvalidPortfolio("multiplicity exceeds u64".into()))?;
            Expr::Pow(
                Arc::new(term),
                Arc::new(Expr::Integer(BigInt::from(multiplicity))),
            )
        });
    }
    Ok(match terms.len() {
        0 => Expr::from_i64(1),
        1 => terms.pop().expect("one term"),
        _ => Expr::Mul(terms),
    })
}

/// Rebuilds the exact immutable input claim. The resume process presents the
/// same immutable input; the durable record's universe binding is what makes
/// "same input" checkable rather than assumed.
fn fixed_claim() -> (Claim, Arc<ImmutableAssumptionsSnapshot>) {
    let input = ipoly(&INPUT_COEFFS);
    let expected = Expr::Mul(vec![
        ipoly(&[2, -2, 1]).to_expr(),
        ipoly(&[2, 2, 1]).to_expr(),
    ]);
    (
        Claim::AlgebraicIdentity {
            lhs: input.to_expr(),
            rhs: expected,
        },
        ImmutableAssumptionsSnapshot::empty(),
    )
}

fn zassenhaus_generator(
    input: Arc<UnivariatePoly>,
    context: Arc<ImmutableAssumptionsSnapshot>,
) -> fsym_runtime::ConcurrentStrategyRunner<asupersync::cx::cap::None> {
    Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
        let factorization = metered_complete_factorization(&input, cx)
            .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
        let product = normalize_factor_product(
            &factorization.scale,
            factorization
                .factors
                .into_iter()
                .map(|factor| (factor.poly, factor.multiplicity))
                .collect(),
        )?;
        let lhs = input.to_expr();
        let mut kernel = ProofKernel::new((*context).clone());
        let root = kernel
            .prove_definitional_reduction(
                lhs.clone(),
                product.clone(),
                "polynomial_ring_equivalence",
                cx,
            )
            .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
        let derivation = kernel
            .export_derivation(root)
            .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
        Ok(PortfolioCandidate {
            strategy_name: "zassenhaus_modular".into(),
            result: product.clone(),
            claim: Claim::AlgebraicIdentity { lhs, rhs: product },
            derivation,
        })
    })
}

fn kronecker_generator(
    input: Arc<UnivariatePoly>,
    context: Arc<ImmutableAssumptionsSnapshot>,
) -> fsym_runtime::ConcurrentStrategyRunner<asupersync::cx::cap::None> {
    Box::new(move |cx: &mut FsymCpuCx<'_, asupersync::cx::cap::None>| {
        let factorization = metered_kronecker_factorization(&input, cx)
            .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
        let product = normalize_factor_product(
            &factorization.scale,
            factorization
                .factors
                .into_iter()
                .map(|factor| (factor.poly, factor.multiplicity))
                .collect(),
        )?;
        let lhs = input.to_expr();
        let mut kernel = ProofKernel::new((*context).clone());
        let root = kernel
            .prove_definitional_reduction(
                lhs.clone(),
                product.clone(),
                "polynomial_ring_equivalence",
                cx,
            )
            .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
        let derivation = kernel
            .export_derivation(root)
            .map_err(|error| PortfolioError::AllStrategiesFailed(error.to_string()))?;
        Ok(PortfolioCandidate {
            strategy_name: "kronecker_interpolation".into(),
            result: product.clone(),
            claim: Claim::AlgebraicIdentity { lhs, rhs: product },
            derivation,
        })
    })
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Marker {
    universe_hex: String,
    payload_schema: String,
    handle: String,
}

fn write_marker(base: &Path, stage: &str, marker: &Marker) {
    let path = marker_path(base, stage);
    let mut content = serde_json::to_vec(marker).expect("marker serialization");
    content.push(b'\n');
    fs::write(path, content).expect("marker write");
}

fn marker_path(base: &Path, stage: &str) -> PathBuf {
    base.with_extension(stage)
}

fn read_marker(base: &Path, stage: &str) -> Result<Marker, String> {
    let wire =
        fs::read(marker_path(base, stage)).map_err(|error| format!("marker read: {error}"))?;
    serde_json::from_slice(&wire).map_err(|error| format!("marker decode: {error}"))
}

fn decode_universe(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("universe hex length {}", hex.len()));
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)
            .map_err(|error| format!("universe hex: {error}"))?;
    }
    Ok(out)
}

/// Storage lane selector for the crash matrix: the canonical file lane
/// or the pinned-CLI subprocess lane. Both implement the identical
/// `DurableStore` contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    File,
    Cli,
}

fn lane_from_arg(value: &str) -> Option<Lane> {
    match value {
        "file" => Some(Lane::File),
        "cli" => Some(Lane::Cli),
        _ => None,
    }
}

/// The pinned-CLI binary location: the native-enabled build recorded on
/// fra-rc-durable-m5e, overridable for other environments.
fn cli_binary() -> Result<PathBuf, String> {
    if let Ok(env) = std::env::var("FSQLITE_CLI_BIN") {
        return Ok(PathBuf::from(env));
    }
    let recorded = PathBuf::from("/data/tmp/cargo-target/debug/fsqlite");
    if recorded.is_file() {
        return Ok(recorded);
    }
    Err("pinned fsqlite CLI binary not found (set FSQLITE_CLI_BIN)".into())
}

fn open_store(lane: Lane, store_root: &Path) -> Result<Box<dyn DurableStore>, String> {
    match lane {
        Lane::File => FileStore::open(store_root)
            .map(|store| Box::new(store) as Box<dyn DurableStore>)
            .map_err(|error| error.to_string()),
        Lane::Cli => {
            let cli = cli_binary()?;
            let db = store_root.join("records.db");
            FsqliteCliStore::open(cli, db)
                .map(|store| Box::new(store) as Box<dyn DurableStore>)
                .map_err(|error| error.to_string())
        }
    }
}

fn generate_mode(
    store_root: &Path,
    boundary: &str,
    marker_base: &Path,
    lane: Lane,
) -> Result<(), String> {
    let (claim, context) = fixed_claim();
    let input = Arc::new(ipoly(&INPUT_COEFFS));
    let raw = asupersync::Cx::detached_cancel_context();
    let limits = BudgetLimits::uniform(10_000_000, 100_000);
    let mut cx = FsymCx::new(&raw, Budget::new(limits), limits);
    let store = open_store(lane, store_root)?;

    let (_card, continuation) = factor_race_generate(
        &mut cx,
        &context,
        &claim,
        vec![
            (
                "kronecker_interpolation",
                kronecker_generator(Arc::clone(&input), Arc::clone(&context)),
            ),
            (
                "zassenhaus_modular",
                zassenhaus_generator(Arc::clone(&input), Arc::clone(&context)),
            ),
        ],
    )
    .map_err(|error| format!("generation failed: {error}"))?;

    let marker_generation = Marker {
        universe_hex: String::new(),
        payload_schema: FACTOR_RACE_CONTINUATION_SCHEMA.into(),
        handle: String::new(),
    };
    write_marker(marker_base, "generation-done", &marker_generation);
    if boundary == "generation-done" {
        sleep_until_killed();
    }

    let checkpoint_wire = continuation
        .to_wire()
        .map_err(|error| format!("checkpoint serialization: {error}"))?;

    // Decode the typed state from the same wire bytes that will be persisted,
    // so the universe identity is computed over exactly the durable payload.
    let checkpoint: TypedCheckpoint<FactorRaceContinuationState> =
        serde_json::from_slice(&checkpoint_wire)
            .map_err(|error| format!("checkpoint self-decode: {error}"))?;
    let state = &checkpoint.payload;
    let input_serialized = serde_json::to_vec(&state.input_expr)
        .map_err(|error| format!("input serialization: {error}"))?;
    let input_digest = *blake3::hash(&input_serialized).as_bytes();
    let pins = dependency_pins();
    let universe = factor_race_universe_id(
        FACTOR_RACE_CONTINUATION_SCHEMA,
        input_digest,
        state.context_digest,
        &pins,
    );
    let record = DurableRecord::new(
        universe,
        FACTOR_RACE_CONTINUATION_SCHEMA,
        checkpoint_wire,
        pins,
    )
    .map_err(|error| format!("record construction: {error}"))?;

    let handle: PreparedHandle = store.prepare(&record).map_err(|error| error.to_string())?;
    let marker_prepared = Marker {
        universe_hex: fsym_durable::hex_lower(&record.universe_id),
        payload_schema: record.payload_schema.clone(),
        handle: handle.as_str().to_string(),
    };
    write_marker(marker_base, "prepared", &marker_prepared);
    if boundary == "prepared" {
        sleep_until_killed();
    }

    let verified = store
        .verify_prepared(&handle)
        .map_err(|error| error.to_string())?;
    if verified != record {
        return Err("staged record diverged from the prepared record".into());
    }
    write_marker(marker_base, "verified", &marker_prepared);
    if boundary == "verified" {
        sleep_until_killed();
    }

    store.commit(&handle).map_err(|error| error.to_string())?;
    // Retain a RaptorQ repair sidecar over the canonical committed bytes so
    // the parent can exercise loss within the declared symbol envelope.
    let record_wire = record.to_wire().map_err(|error| error.to_string())?;
    let sidecar = RepairSidecar::encode(&record_wire, REPAIR_SYMBOL_SIZE, REPAIR_SYMBOL_COUNT)
        .map_err(|error| format!("repair encode: {error}"))?;
    let sidecar_wire =
        serde_json::to_vec(&sidecar).map_err(|error| format!("sidecar serialization: {error}"))?;
    fs::write(store_root.join("committed.repair"), sidecar_wire)
        .map_err(|error| format!("sidecar write: {error}"))?;
    write_marker(marker_base, "committed", &marker_prepared);
    if boundary == "committed" {
        sleep_until_killed();
    }

    Err(format!("unknown boundary {boundary:?}"))
}

fn sleep_until_killed() -> ! {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// Runs the race exactly like `generate` but persists nothing: this is the
/// ephemeral baseline whose report every durable run must reproduce.
fn ephemeral_mode(marker_base: &Path) -> Result<(), String> {
    let (claim, context) = fixed_claim();
    let input = Arc::new(ipoly(&INPUT_COEFFS));
    let raw = asupersync::Cx::detached_cancel_context();
    let limits = BudgetLimits::uniform(10_000_000, 100_000);
    let mut cx = FsymCx::new(&raw, Budget::new(limits), limits);
    let outcome = fsym_runtime::run_portfolio_concurrent_race(
        &mut cx,
        &context,
        &claim,
        vec![
            (
                "kronecker_interpolation",
                kronecker_generator(Arc::clone(&input), Arc::clone(&context)),
            ),
            (
                "zassenhaus_modular",
                zassenhaus_generator(Arc::clone(&input), Arc::clone(&context)),
            ),
        ],
    )
    .map_err(|error| format!("ephemeral race failed: {error}"))?;
    write_report(marker_base, &outcome, 0)
}

fn write_report(
    marker_base: &Path,
    outcome: &fsym_runtime::VerifiedPortfolioOutcome,
    prior_generation_steps: u64,
) -> Result<(), String> {
    if !outcome.evidence().verify_integrity() {
        return Err("outcome evidence failed integrity".into());
    }
    let report = serde_json::json!({
        "winning_strategy": outcome.winning_strategy(),
        "result": outcome.result(),
        "claim_digest": fsym_durable::hex_lower(&outcome.evidence().claim.digest()),
        "generator_steps_consumed": prior_generation_steps + outcome.generator_steps_consumed(),
    });
    let mut wire =
        serde_json::to_vec_pretty(&report).map_err(|error| format!("report: {error}"))?;
    wire.push(b'\n');
    fs::write(marker_base.with_extension("resume.json"), wire)
        .map_err(|error| format!("report write: {error}"))?;
    Ok(())
}

fn resume_mode(
    lane: Lane,
    store_root: &Path,
    marker_base: &Path,
    boundary: &str,
) -> Result<(), String> {
    let marker = read_marker(marker_base, boundary)?;
    let universe = decode_universe(&marker.universe_hex)?;
    let store = open_store(lane, store_root)?;

    // Promote staged records when the crash happened before commit; after a
    // committed crash this is a no-op because the record is already in the
    // committed namespace.
    let handle = PreparedHandle::from_name(marker.handle.clone());
    if store
        .load_committed(universe, &marker.payload_schema)
        .map_err(|error| error.to_string())?
        .is_none()
    {
        store
            .verify_prepared(&handle)
            .map_err(|error| format!("staged verification before promotion: {error}"))?;
        store
            .commit(&handle)
            .map_err(|error| format!("promotion commit: {error}"))?;
    }
    let record = store
        .load_committed(universe, &marker.payload_schema)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "no committed record after promotion".to_string())?;

    // Full boundary validation against the presenting universe and pins.
    let expected_pins = dependency_pins();
    let payload = record
        .validate(universe, &expected_pins)
        .map_err(|error| error.to_string())?;

    // Decode the typed checkpoint and present the same immutable input.
    let checkpoint: TypedCheckpoint<FactorRaceContinuationState> = serde_json::from_slice(payload)
        .map_err(|error| format!("checkpoint decode refused: {error}"))?;
    let (claim, context) = fixed_claim();

    // Fresh region: its own newly constructed budget. The captured counters
    // are observational; the fresh allowance equals the captured remaining
    // amount so the resumed work is bounded exactly like the interrupted one.
    let mut dimensions = [0u64; fsym_budget::DIMENSION_COUNT];
    for dimension in Dimension::ALL {
        dimensions[dimension.index()] = checkpoint
            .remaining_budget
            .get(&dimension)
            .copied()
            .unwrap_or(0);
    }
    let limits = BudgetLimits {
        dimensions,
        verifier_pool: checkpoint.verifier_remaining,
    };
    let raw = asupersync::Cx::detached_cancel_context();
    let mut cx = FsymCx::new(&raw, Budget::new(limits), limits);

    let outcome = resume_factor_race_from_checkpoint(&mut cx, &context, &claim, &checkpoint)
        .map_err(|error| format!("resumed race refused: {error}"))?;
    // Cross-process accounting: the interrupted process consumed everything
    // between the original full allowance and the captured remaining counters
    // (generation); this process consumed the rest (verification and
    // acceptance). The sum is what an uninterrupted run would have charged.
    let original_compute_allowance = 10_000_000u64;
    let captured_remaining = checkpoint
        .remaining_budget
        .get(&Dimension::ComputeSteps)
        .copied()
        .unwrap_or(0);
    let generation_steps = original_compute_allowance.saturating_sub(captured_remaining);
    write_report(marker_base, &outcome, generation_steps)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match args.as_slice() {
        ["generate", lane, store_root, boundary, marker_base] => match lane_from_arg(lane) {
            Some(lane) => generate_mode(
                Path::new(*store_root),
                boundary,
                Path::new(*marker_base),
                lane,
            ),
            None => Err(format!("unknown lane {lane:?}")),
        },
        ["resume", lane, store_root, marker_base, boundary] => match lane_from_arg(lane) {
            Some(lane) => resume_mode(
                lane,
                Path::new(*store_root),
                Path::new(*marker_base),
                boundary,
            ),
            None => Err(format!("unknown lane {lane:?}")),
        },
        ["ephemeral", marker_base] => ephemeral_mode(Path::new(*marker_base)),
        _ => Err(format!(
            "usage: factor_frontier_worker generate <store_root> <boundary> <marker_base> | resume <store_root> <marker_base> <boundary> | ephemeral <marker_base>; got {args:?}"
        )),
    };
    // The generate mode intentionally never returns; it sleeps until the
    // parent kills it at the boundary. Any actual return of generate_mode is
    // an error (unknown boundary) or a misuse.
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
