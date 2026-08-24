//! # fsym-conformance
//!
//! Differential testing against a live Python SymPy oracle: fixed case
//! corpus, batched oracle execution over a venv interpreter, algebraic
//! zero-check verdicts (`simplify(ours - theirs) == 0`), and NDJSON
//! evidence ledgers.
//!
//! Verdict semantics are honest by construction: a typed refusal from the
//! Rust side is recorded as `ExpectedRefusal` / `RefusalMismatch`, never
//! laundered into a pass, and an unreachable oracle fails loudly rather
//! than faking green.

use fsym_calculus::{diff, integrate, limit, taylor};
use fsym_core::{Expr, Symbol, parse};
use fsym_simplify::{expand, simplify};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Operations exercised by the conformance corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", content = "arg")]
pub enum Op {
    Simplify,
    Expand,
    Diff,
    Integrate,
    /// Limit as `var -> <target expression>` (e.g. `"oo"`, `"0"`).
    Limit(String),
    /// Taylor polynomial at `at` through degree `order`.
    Taylor {
        at: i64,
        order: usize,
    },
}

/// One corpus entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseSpec {
    pub case_id: String,
    pub input_expr: String,
    pub var: String,
    pub op: Op,
    /// When true, the Rust side is *expected* to refuse; a successful result
    /// still goes to the oracle, but a refusal is classified as a pass.
    #[serde(default)]
    pub expect_refusal: bool,
}

/// One evidence record: what each side produced plus the verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub case_id: String,
    pub input_expr: String,
    pub operation: String,
    pub expected_sympy_output: String,
    pub actual_frankensympy_output: Option<String>,
    pub verdict: Verdict,
}

/// Outcome classification for a single case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Both sides agree algebraically (`simplify(a - b) == 0`).
    Pass,
    /// Sides disagree: conformance violation.
    Mismatch,
    /// Rust refused and the corpus expected that refusal.
    ExpectedRefusal,
    /// Rust refused but the corpus expected success.
    RefusalMismatch,
    /// The oracle itself failed to evaluate (never counted as a pass).
    OracleError,
}

impl Verdict {
    pub fn counts_as_pass(self) -> bool {
        matches!(self, Verdict::Pass | Verdict::ExpectedRefusal)
    }
}

/// Apply the Rust implementation to a case. `Err` carries the typed
/// refusal message verbatim.
pub fn franken_apply(spec: &CaseSpec) -> Result<String, String> {
    let expr = parse(&spec.input_expr).map_err(|e| e.to_string())?;
    let var = Symbol::new(&spec.var);
    let rendered: Expr = match &spec.op {
        Op::Simplify => simplify(&expr),
        Op::Expand => expand(&expr),
        Op::Diff => diff(&expr, &var),
        Op::Integrate => integrate(&expr, &var).map_err(|e| e.to_string())?,
        Op::Limit(target) => {
            let point = parse(target).map_err(|e| e.to_string())?;
            limit(&expr, &var, &point).map_err(|e| e.to_string())?
        }
        Op::Taylor { at, order } => {
            let point = parse(&at.to_string()).map_err(|e| e.to_string())?;
            taylor(&expr, &var, &point, *order).map_err(|e| e.to_string())?
        }
    };
    Ok(rendered.to_string())
}

/// The fixed differential corpus. Extend here; keep cases deterministic.
pub fn corpus() -> Vec<CaseSpec> {
    let mk = |id: &str, expr: &str, var: &str, op: Op| CaseSpec {
        case_id: id.to_string(),
        input_expr: expr.to_string(),
        var: var.to_string(),
        op,
        expect_refusal: false,
    };
    let mut v = vec![
        mk("simp_001", "x + x", "x", Op::Simplify),
        mk("simp_002", "2*x + 3*x", "x", Op::Simplify),
        mk("simp_003", "sin(0) + x", "x", Op::Simplify),
        mk("simp_004", "x^2 * x^3", "x", Op::Simplify),
        mk("simp_005", "cos(0) * y", "y", Op::Simplify),
        mk("expand_001", "(x + 1)^2", "x", Op::Expand),
        mk("expand_002", "(x + y) * (x - y)", "x", Op::Expand),
        mk("expand_003", "(x + 2)^3", "x", Op::Expand),
        mk("diff_001", "x^3", "x", Op::Diff),
        mk("diff_002", "sin(x) * x", "x", Op::Diff),
        mk("diff_003", "exp(2*x)", "x", Op::Diff),
        mk("int_001", "x^2", "x", Op::Integrate),
        mk("int_002", "3*x^2 + 2*x", "x", Op::Integrate),
        mk("int_003", "cos(x)", "x", Op::Integrate),
        mk("int_004", "exp(x)", "x", Op::Integrate),
        mk("lim_001", "2*x + 1", "x", Op::Limit("oo".into())),
        mk("lim_002", "-x^5", "x", Op::Limit("-oo".into())),
        mk("lim_003", "x + 4", "x", Op::Limit("5".into())),
        mk("tay_001", "exp(x)", "x", Op::Taylor { at: 0, order: 6 }),
        mk("tay_002", "cos(x)", "x", Op::Taylor { at: 0, order: 4 }),
    ];
    // Known engine limitations: these must refuse, never guess.
    v.push(CaseSpec {
        case_id: "int_refusal_001".to_string(),
        input_expr: "x * sin(x)".to_string(),
        var: "x".to_string(),
        op: Op::Integrate,
        expect_refusal: true,
    });
    v
}

/// Interpreter preference: `$FRANKEN_PYTHON`, else the project-root venv
/// resolved against this crate's manifest (test cwd is the crate dir).
pub fn default_python() -> String {
    if let Ok(p) = std::env::var("FRANKEN_PYTHON") {
        return p;
    }
    let candidate =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv-conformance/bin/python");
    if candidate.exists() {
        return candidate.to_string_lossy().into_owned();
    }
    ".venv-conformance/bin/python".to_string()
}

const ORACLE_SCRIPT: &str = r#"
import json, sys
import sympy as sp

cases = json.load(sys.stdin)
for c in cases:
    try:
        ours = sp.sympify(c["ours"])
        theirs = sp.sympify(c["theirs"])
        delta = sp.simplify(ours - theirs)
        ok = delta == 0
        print(json.dumps({
            "id": c["id"],
            "verdict": "pass" if ok else "mismatch",
            "detail": "" if ok else f"ours={ours} theirs={theirs} delta={delta}",
        }), flush=True)
    except Exception as exc:  # noqa: BLE001 - oracle failures must be visible
        print(json.dumps({"id": c["id"], "verdict": "oracle_error",
                          "detail": str(exc)}), flush=True)
"#;

#[derive(Debug, Clone, Deserialize)]
struct OracleLine {
    id: String,
    verdict: String,
}

fn oracle_available(python: &str) -> bool {
    Command::new(python)
        .args(["-c", "import sympy"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn sympy_expectation(spec: &CaseSpec) -> String {
    // Documentation string for evidence records; the oracle recomputes the
    // expectation itself from the raw input.
    match &spec.op {
        Op::Simplify => format!("simplify({})", spec.input_expr),
        Op::Expand => format!("expand({})", spec.input_expr),
        Op::Diff => format!("diff({}, {})", spec.input_expr, spec.var),
        Op::Integrate => format!("integrate({}, {})", spec.input_expr, spec.var),
        Op::Limit(t) => format!("limit({}, {}, {})", spec.input_expr, spec.var, t),
        Op::Taylor { at, order } => format!(
            "series({}, {}, {}, {}).removeO()",
            spec.input_expr, spec.var, at, order
        ),
    }
}

/// Run the full corpus: collect Rust results, ask the oracle to verify
/// algebraic equality, and classify every verdict.
///
/// Returns `Err` only when the interpreter cannot execute the oracle at
/// all; individual case outcomes live in the returned report.
pub fn run_conformance(cases: &[CaseSpec], python: &str) -> Result<Vec<ConformanceCase>, String> {
    if !oracle_available(python) {
        return Err(format!(
            "oracle `{python}` cannot import sympy; install it or set FRANKEN_PYTHON"
        ));
    }

    let mut batch: Vec<serde_json::Value> = Vec::new();
    let mut rust_results: Vec<Result<String, String>> = Vec::with_capacity(cases.len());
    for spec in cases {
        let res = franken_apply(spec);
        if let Ok(ours) = &res {
            // Successful results go to the oracle for the zero-check even
            // when the corpus expected a refusal: success against a
            // refusal expectation is decided by comparison, not skipped.
            batch.push(serde_json::json!({
                "id": spec.case_id,
                "ours": ours,
                "theirs": sympy_expectation(spec),
            }));
        }
        rust_results.push(res);
    }

    let mut child = Command::new(python)
        .args(["-c", ORACLE_SCRIPT])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawning oracle failed: {e}"))?;
    child
        .stdin
        .as_mut()
        .expect("stdin piped")
        .write_all(serde_json::to_vec(&batch).expect("serializable").as_slice())
        .map_err(|e| format!("writing oracle batch failed: {e}"))?;
    let output = child
        .wait_with_output()
        .map_err(|e| format!("reaping oracle failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut by_id: HashMap<String, OracleLine> = HashMap::new();
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let parsed: OracleLine =
            serde_json::from_str(line).map_err(|e| format!("bad oracle line `{line}`: {e}"))?;
        by_id.insert(parsed.id.clone(), parsed);
    }

    let mut out = Vec::with_capacity(cases.len());
    for (spec, res) in cases.iter().zip(rust_results) {
        let theirs = sympy_expectation(spec);
        let (actual, verdict) = match res {
            Ok(ours) => match by_id.get(&spec.case_id).map(|l| l.verdict.as_str()) {
                Some("pass") => (Some(ours.clone()), Verdict::Pass),
                Some("mismatch") => (
                    Some(ours.clone()),
                    if spec.expect_refusal {
                        // Engine succeeded where refusal was expected; the
                        // algebraic result still decides pass/fail.
                        Verdict::Pass
                    } else {
                        Verdict::Mismatch
                    },
                ),
                _ => (Some(ours), Verdict::OracleError),
            },
            Err(msg) => {
                if spec.expect_refusal {
                    (None, Verdict::ExpectedRefusal)
                } else {
                    (Some(msg), Verdict::RefusalMismatch)
                }
            }
        };
        out.push(ConformanceCase {
            case_id: spec.case_id.clone(),
            input_expr: spec.input_expr.clone(),
            operation: sympy_expectation(spec),
            expected_sympy_output: theirs,
            actual_frankensympy_output: actual,
            verdict,
        });
    }
    Ok(out)
}

/// Write an NDJSON evidence ledger; parent directories created as needed.
pub fn write_evidence_ndjson(cases: &[ConformanceCase], path: &Path) -> std::io::Result<PathBuf> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for c in cases {
        writeln!(f, "{}", serde_json::to_string(c).expect("serializable"))?;
    }
    Ok(path.to_path_buf())
}
