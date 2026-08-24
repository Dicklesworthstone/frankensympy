#![forbid(unsafe_code)]

//! # fsym-conformance
//!
//! A narrow native-mathematics differential harness against a live,
//! version-pinned Python SymPy oracle: fixed scalar corpus, batched oracle
//! execution over an isolated interpreter, exact-or-algebraic verdicts
//! (`ours == theirs` or `simplify(ours - theirs) == 0`), and NDJSON
//! evidence ledgers.
//!
//! Verdict semantics are honest by construction: a typed refusal from the
//! Rust side is recorded as `ExpectedRefusal` / `RefusalMismatch`, never
//! laundered into a pass, and an unreachable oracle fails loudly rather
//! than faking green.
//!
//! This crate does **not** exercise Python class identity, held forms,
//! warnings, pickles, or arbitrary subclasses. It is not the WS01 object
//! compatibility laboratory and cannot support a drop-in compatibility
//! claim.

use fsym_calculus::{diff, integrate, limit, taylor};
use fsym_core::{Expr, Symbol, parse};
use fsym_simplify::{expand, simplify};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const PROFILE_ID: &str = "sympy-1.14.0-cpython";
pub const PINNED_SYMPY_VERSION: &str = "1.14.0";

const COMPARATOR_ID: &str = "exact_or_algebraic_difference_zero";
const MAX_CASES: usize = 1_024;
const MAX_CASE_FIELD_BYTES: usize = 16 * 1_024;

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
    /// When true, the current Rust capability boundary is expected to refuse.
    /// Such a refusal is an expected harness outcome, but is not conformance.
    #[serde(default)]
    pub expect_refusal: bool,
}

/// One evidence record: what each side produced plus the verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub profile_id: String,
    pub oracle_sympy_version: String,
    pub case_id: String,
    pub input_expr: String,
    pub operation: String,
    pub comparator: String,
    pub expected_sympy_output: Option<String>,
    pub actual_frankensympy_output: Option<String>,
    pub frankensympy_refusal: Option<String>,
    pub oracle_detail: Option<String>,
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
    /// Rust succeeded where the corpus required a typed refusal.
    UnexpectedSuccess,
    /// Rust refused but the corpus expected success.
    RefusalMismatch,
    /// The oracle itself failed to evaluate (never counted as a pass).
    OracleError,
}

impl Verdict {
    /// True only for an actual differential conformance result.
    pub fn is_conformant(self) -> bool {
        matches!(self, Verdict::Pass)
    }

    /// True when the outcome matches the current fixed corpus expectation.
    /// Expected refusals remain non-conformant capability gaps.
    pub fn is_expected_outcome(self) -> bool {
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
import json
import signal
import sys

if not hasattr(signal, "alarm"):
    raise RuntimeError("this oracle runner requires signal.alarm for a hard wall-time bound")
signal.alarm(30)

import sympy as sp

payload = json.load(sys.stdin)
actual_version = sp.__version__
print(json.dumps({"kind": "meta", "sympy_version": actual_version}), flush=True)
if actual_version != payload["required_sympy_version"]:
    print(
        f"SymPy version mismatch: required={payload['required_sympy_version']} actual={actual_version}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(3)

for c in payload["cases"]:
    try:
        expr = sp.sympify(c["input_expr"])
        var = sp.Symbol(c["var"])
        op = c["op"]
        name = op["name"]
        if name == "simplify":
            theirs = sp.simplify(expr)
        elif name == "expand":
            theirs = sp.expand(expr)
        elif name == "diff":
            theirs = sp.diff(expr, var)
        elif name == "integrate":
            theirs = sp.integrate(expr, var)
        elif name == "limit":
            theirs = sp.limit(expr, var, sp.sympify(op["target"]))
        elif name == "taylor":
            theirs = sp.series(
                expr, var, op["at"], op["series_terms"]
            ).removeO()
        else:
            raise ValueError(f"unknown operation: {name}")

        expected = str(theirs)
        if c["ours"] is None:
            verdict = "expectation_only"
            detail = None
        else:
            ours = sp.sympify(c["ours"])
            exact = ours == theirs
            delta = sp.S.Zero if exact else sp.simplify(ours - theirs)
            agrees = exact or delta == 0
            verdict = "pass" if agrees else "mismatch"
            detail = None if agrees else f"ours={ours} theirs={theirs} delta={delta}"
        print(json.dumps({
            "kind": "case",
            "id": c["id"],
            "verdict": verdict,
            "expected": expected,
            "detail": detail,
        }), flush=True)
    except Exception as exc:  # oracle failures are evidence, never a pass
        print(json.dumps({
            "kind": "case",
            "id": c["id"],
            "verdict": "oracle_error",
            "expected": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }), flush=True)

signal.alarm(0)
"#;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OracleRecord {
    Meta {
        sympy_version: String,
    },
    Case {
        id: String,
        verdict: String,
        expected: Option<String>,
        detail: Option<String>,
    },
}

#[derive(Debug, Clone)]
struct OracleLine {
    verdict: String,
    expected: Option<String>,
    detail: Option<String>,
}

fn series_terms(order: usize) -> Result<usize, String> {
    order
        .checked_add(1)
        .ok_or_else(|| "Taylor order overflows the oracle series bound".to_string())
}

fn sympy_operation(spec: &CaseSpec) -> Result<String, String> {
    Ok(match &spec.op {
        Op::Simplify => format!("simplify({})", spec.input_expr),
        Op::Expand => format!("expand({})", spec.input_expr),
        Op::Diff => format!("diff({}, {})", spec.input_expr, spec.var),
        Op::Integrate => format!("integrate({}, {})", spec.input_expr, spec.var),
        Op::Limit(t) => format!("limit({}, {}, {})", spec.input_expr, spec.var, t),
        Op::Taylor { at, order } => format!(
            "series({}, {}, {}, {}).removeO()",
            spec.input_expr,
            spec.var,
            at,
            series_terms(*order)?
        ),
    })
}

fn validate_cases(cases: &[CaseSpec]) -> Result<(), String> {
    if cases.len() > MAX_CASES {
        return Err(format!(
            "corpus has {} cases; maximum is {MAX_CASES}",
            cases.len()
        ));
    }
    let mut ids = HashSet::with_capacity(cases.len());
    for spec in cases {
        if spec.case_id.trim().is_empty() {
            return Err("case_id must not be empty".to_string());
        }
        if !ids.insert(spec.case_id.as_str()) {
            return Err(format!("duplicate case_id `{}`", spec.case_id));
        }
        for (field, value) in [
            ("case_id", spec.case_id.as_str()),
            ("input_expr", spec.input_expr.as_str()),
            ("var", spec.var.as_str()),
        ] {
            if value.len() > MAX_CASE_FIELD_BYTES {
                return Err(format!(
                    "case `{}` field `{field}` exceeds {MAX_CASE_FIELD_BYTES} bytes",
                    spec.case_id
                ));
            }
        }
        if let Op::Limit(target) = &spec.op
            && target.len() > MAX_CASE_FIELD_BYTES
        {
            return Err(format!(
                "case `{}` limit target exceeds {MAX_CASE_FIELD_BYTES} bytes",
                spec.case_id
            ));
        }
        let _ = sympy_operation(spec)?;
    }
    Ok(())
}

fn oracle_request(spec: &CaseSpec, ours: Option<&String>) -> Result<serde_json::Value, String> {
    let op = match &spec.op {
        Op::Simplify => serde_json::json!({"name": "simplify"}),
        Op::Expand => serde_json::json!({"name": "expand"}),
        Op::Diff => serde_json::json!({"name": "diff"}),
        Op::Integrate => serde_json::json!({"name": "integrate"}),
        Op::Limit(target) => serde_json::json!({"name": "limit", "target": target}),
        Op::Taylor { at, order } => serde_json::json!({
            "name": "taylor",
            "at": at,
            "series_terms": series_terms(*order)?,
        }),
    };
    Ok(serde_json::json!({
        "id": spec.case_id,
        "input_expr": spec.input_expr,
        "var": spec.var,
        "op": op,
        "ours": ours,
    }))
}

fn parse_oracle_stdout(stdout: &str) -> Result<(String, HashMap<String, OracleLine>), String> {
    let mut version = None;
    let mut by_id = HashMap::new();
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let parsed: OracleRecord =
            serde_json::from_str(line).map_err(|e| format!("bad oracle line `{line}`: {e}"))?;
        match parsed {
            OracleRecord::Meta { sympy_version } => {
                if version.replace(sympy_version).is_some() {
                    return Err("oracle emitted duplicate metadata records".to_string());
                }
            }
            OracleRecord::Case {
                id,
                verdict,
                expected,
                detail,
            } => {
                let oracle_line = OracleLine {
                    verdict,
                    expected,
                    detail,
                };
                if by_id.insert(id.clone(), oracle_line).is_some() {
                    return Err(format!("oracle emitted duplicate case_id `{id}`"));
                }
            }
        }
    }
    let version = version.ok_or_else(|| "oracle emitted no metadata record".to_string())?;
    if version != PINNED_SYMPY_VERSION {
        return Err(format!(
            "oracle SymPy version mismatch: required={PINNED_SYMPY_VERSION} actual={version}"
        ));
    }
    Ok((version, by_id))
}

fn validate_oracle_ids(
    cases: &[CaseSpec],
    by_id: &HashMap<String, OracleLine>,
) -> Result<(), String> {
    let expected: HashSet<&str> = cases.iter().map(|spec| spec.case_id.as_str()).collect();
    for id in by_id.keys() {
        if !expected.contains(id.as_str()) {
            return Err(format!("oracle emitted unknown case_id `{id}`"));
        }
    }
    for id in expected {
        if !by_id.contains_key(id) {
            return Err(format!("oracle omitted case_id `{id}`"));
        }
    }
    Ok(())
}

fn classify(
    spec: &CaseSpec,
    rust_result: &Result<String, String>,
    oracle: &OracleLine,
) -> Result<Verdict, String> {
    match (rust_result, oracle.verdict.as_str()) {
        (_, "oracle_error") => Ok(Verdict::OracleError),
        (Err(_), "expectation_only") if spec.expect_refusal => Ok(Verdict::ExpectedRefusal),
        (Err(_), "expectation_only") => Ok(Verdict::RefusalMismatch),
        (Ok(_), "pass" | "mismatch") if spec.expect_refusal => Ok(Verdict::UnexpectedSuccess),
        (Ok(_), "pass") => Ok(Verdict::Pass),
        (Ok(_), "mismatch") => Ok(Verdict::Mismatch),
        (Ok(_), "expectation_only") => Err(format!(
            "oracle did not compare successful case `{}`",
            spec.case_id
        )),
        (Err(_), "pass" | "mismatch") => {
            Err(format!("oracle compared refused case `{}`", spec.case_id))
        }
        (_, other) => Err(format!(
            "oracle emitted unknown verdict `{other}` for case `{}`",
            spec.case_id
        )),
    }
}

fn assemble_report(
    cases: &[CaseSpec],
    rust_results: Vec<Result<String, String>>,
    oracle_version: &str,
    mut by_id: HashMap<String, OracleLine>,
) -> Result<Vec<ConformanceCase>, String> {
    validate_oracle_ids(cases, &by_id)?;
    let mut report = Vec::with_capacity(cases.len());
    for (spec, rust_result) in cases.iter().zip(rust_results) {
        let oracle = by_id
            .remove(&spec.case_id)
            .ok_or_else(|| format!("oracle omitted case_id `{}`", spec.case_id))?;
        let verdict = classify(spec, &rust_result, &oracle)?;
        report.push(ConformanceCase {
            profile_id: PROFILE_ID.to_string(),
            oracle_sympy_version: oracle_version.to_string(),
            case_id: spec.case_id.clone(),
            input_expr: spec.input_expr.clone(),
            operation: sympy_operation(spec)?,
            comparator: COMPARATOR_ID.to_string(),
            expected_sympy_output: oracle.expected,
            actual_frankensympy_output: rust_result.as_ref().ok().cloned(),
            frankensympy_refusal: rust_result.as_ref().err().cloned(),
            oracle_detail: oracle.detail,
            verdict,
        });
    }
    Ok(report)
}

/// Run the full fixed corpus against a separate live SymPy process.
///
/// `Err` means the run itself is invalid: malformed/duplicate cases, a
/// missing or wrong-version oracle, child-process failure, or a broken
/// result protocol. Individual mathematical disagreements and typed Rust
/// refusals are returned as explicit case verdicts.
pub fn run_conformance(cases: &[CaseSpec], python: &str) -> Result<Vec<ConformanceCase>, String> {
    validate_cases(cases)?;
    let rust_results: Vec<Result<String, String>> = cases.iter().map(franken_apply).collect();
    let oracle_cases: Vec<serde_json::Value> = cases
        .iter()
        .zip(&rust_results)
        .map(|(spec, result)| oracle_request(spec, result.as_ref().ok()))
        .collect::<Result<_, _>>()?;
    let payload = serde_json::json!({
        "required_sympy_version": PINNED_SYMPY_VERSION,
        "cases": oracle_cases,
    });
    let payload = serde_json::to_vec(&payload)
        .map_err(|e| format!("serializing oracle request failed: {e}"))?;

    let mut child = Command::new(python)
        .args(["-I", "-c", ORACLE_SCRIPT])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawning oracle `{python}` failed: {e}"))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "oracle stdin was not piped".to_string())?;
    stdin
        .write_all(&payload)
        .map_err(|e| format!("writing oracle batch failed: {e}"))?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .map_err(|e| format!("reaping oracle failed: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "oracle exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }
    let stdout =
        String::from_utf8(output.stdout).map_err(|e| format!("oracle stdout is not UTF-8: {e}"))?;
    let (oracle_version, by_id) = parse_oracle_stdout(&stdout)?;
    assemble_report(cases, rust_results, &oracle_version, by_id)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn case(expect_refusal: bool) -> CaseSpec {
        CaseSpec {
            case_id: "planted_001".to_string(),
            input_expr: "x + x".to_string(),
            var: "x".to_string(),
            op: Op::Simplify,
            expect_refusal,
        }
    }

    fn oracle(verdict: &str) -> OracleLine {
        OracleLine {
            verdict: verdict.to_string(),
            expected: Some("2*x".to_string()),
            detail: None,
        }
    }

    #[test]
    fn planted_mismatch_is_never_conformant() {
        let verdict = classify(&case(false), &Ok("x".to_string()), &oracle("mismatch"))
            .expect("known verdict");
        assert_eq!(verdict, Verdict::Mismatch);
        assert!(!verdict.is_conformant());
        assert!(!verdict.is_expected_outcome());
    }

    #[test]
    fn expected_refusal_is_not_a_conformance_pass() {
        let verdict = classify(
            &case(true),
            &Err("typed integration refusal".to_string()),
            &oracle("expectation_only"),
        )
        .expect("known verdict");
        assert_eq!(verdict, Verdict::ExpectedRefusal);
        assert!(!verdict.is_conformant());
        assert!(verdict.is_expected_outcome());
    }

    #[test]
    fn success_where_refusal_was_required_is_unexpected() {
        for oracle_verdict in ["pass", "mismatch"] {
            let verdict = classify(&case(true), &Ok("2*x".to_string()), &oracle(oracle_verdict))
                .expect("known verdict");
            assert_eq!(verdict, Verdict::UnexpectedSuccess);
            assert!(!verdict.is_expected_outcome());
        }
    }

    #[test]
    fn unexpected_refusal_remains_a_mismatch() {
        let verdict = classify(
            &case(false),
            &Err("unsupported".to_string()),
            &oracle("expectation_only"),
        )
        .expect("known verdict");
        assert_eq!(verdict, Verdict::RefusalMismatch);
        assert!(!verdict.is_expected_outcome());
    }

    #[test]
    fn taylor_operation_translates_degree_to_series_term_count() {
        let spec = CaseSpec {
            case_id: "tay_degree".to_string(),
            input_expr: "exp(x)".to_string(),
            var: "x".to_string(),
            op: Op::Taylor { at: 0, order: 6 },
            expect_refusal: false,
        };
        assert_eq!(
            sympy_operation(&spec).expect("bounded order"),
            "series(exp(x), x, 0, 7).removeO()"
        );
        let request = oracle_request(&spec, Some(&"1 + x".to_string())).expect("request");
        assert_eq!(request["op"]["series_terms"], 7);
    }

    #[test]
    fn duplicate_case_ids_fail_before_oracle_execution() {
        let duplicate = vec![case(false), case(true)];
        let err = validate_cases(&duplicate).expect_err("duplicate must fail");
        assert!(err.contains("duplicate case_id"));
    }

    #[test]
    fn oracle_protocol_rejects_duplicate_and_missing_results() {
        let duplicate = concat!(
            "{\"kind\":\"meta\",\"sympy_version\":\"1.14.0\"}\n",
            "{\"kind\":\"case\",\"id\":\"planted_001\",\"verdict\":\"pass\",\"expected\":\"2*x\",\"detail\":null}\n",
            "{\"kind\":\"case\",\"id\":\"planted_001\",\"verdict\":\"pass\",\"expected\":\"2*x\",\"detail\":null}\n"
        );
        assert!(
            parse_oracle_stdout(duplicate)
                .expect_err("duplicate output must fail")
                .contains("duplicate case_id")
        );

        let meta_only = "{\"kind\":\"meta\",\"sympy_version\":\"1.14.0\"}\n";
        let (_, by_id) = parse_oracle_stdout(meta_only).expect("valid metadata");
        assert!(
            validate_oracle_ids(&[case(false)], &by_id)
                .expect_err("missing result must fail")
                .contains("omitted case_id")
        );
    }

    #[test]
    fn malformed_oracle_output_fails_closed() {
        let err = parse_oracle_stdout("not-json\n").expect_err("malformed output must fail");
        assert!(err.contains("bad oracle line"));
    }

    #[test]
    fn unreachable_oracle_fails_closed() {
        let err = run_conformance(&[case(false)], "/definitely/missing/frankensympy-python")
            .expect_err("missing interpreter must fail");
        assert!(err.contains("spawning oracle"));
    }

    #[test]
    fn wrong_sympy_version_fails_closed() {
        let stdout = "{\"kind\":\"meta\",\"sympy_version\":\"1.13.3\"}\n";
        let err = parse_oracle_stdout(stdout).expect_err("wrong version must fail");
        assert!(err.contains("required=1.14.0 actual=1.13.3"));
    }
}
