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

use fsym_calculus::{CalculusError, diff, integrate, limit, taylor};
use fsym_core::{Constant, Expr, Symbol, parse};
use fsym_simplify::{SimplifyError, try_expand, try_simplify};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const PROFILE_ID: &str = "native-math-corpus-v1";
pub const PINNED_SYMPY_VERSION: &str = "1.14.0";

const COMPARATOR_ID: &str = "exact_or_algebraic_difference_zero";
const PROTOCOL_SCHEMA_VERSION: u32 = 1;
const MAX_CASES: usize = 1_024;
const MAX_CASE_FIELD_BYTES: usize = 16 * 1_024;
const MAX_TAYLOR_ORDER: usize = 12;
const ORACLE_TIMEOUT: Duration = Duration::from_secs(35);
const MAX_ORACLE_REQUEST_BYTES: usize = 16 * 1024;
const MAX_ORACLE_OUTPUT_BYTES: usize = 32 * 1024;

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
    /// Exact typed capability refusal required from the Rust lane. A matching
    /// refusal is an expected harness outcome, but is not conformance.
    #[serde(default)]
    pub expected_refusal: Option<RefusalKind>,
}

/// Capability-boundary refusal kinds understood by this fixed corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalKind {
    IntegrationUnsupported,
    LimitUndetermined,
    TaylorUnsupported,
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
    pub frankensympy_refusal_kind: Option<RefusalKind>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrankenFailure {
    kind: Option<RefusalKind>,
    detail: String,
}

impl FrankenFailure {
    fn parse(error: impl ToString) -> Self {
        Self {
            kind: None,
            detail: error.to_string(),
        }
    }

    fn calculus(error: CalculusError, expected_variant: RefusalKind) -> Self {
        let kind = match (&error, expected_variant) {
            (CalculusError::IntegrationFailed(_), RefusalKind::IntegrationUnsupported)
            | (CalculusError::Undetermined(_), RefusalKind::LimitUndetermined)
            | (CalculusError::NonDifferentiable(_), RefusalKind::TaylorUnsupported) => {
                Some(expected_variant)
            }
            _ => None,
        };
        Self {
            kind,
            detail: error.to_string(),
        }
    }

    fn simplify(error: SimplifyError) -> Self {
        Self {
            kind: None,
            detail: error.to_string(),
        }
    }
}

fn franken_apply_expr(spec: &CaseSpec) -> Result<Expr, FrankenFailure> {
    let expr = parse(&spec.input_expr).map_err(FrankenFailure::parse)?;
    let var = Symbol::new(&spec.var);
    match &spec.op {
        Op::Simplify => try_simplify(&expr).map_err(FrankenFailure::simplify),
        Op::Expand => try_expand(&expr).map_err(FrankenFailure::simplify),
        Op::Diff => Ok(diff(&expr, &var)),
        Op::Integrate => integrate(&expr, &var)
            .map_err(|error| FrankenFailure::calculus(error, RefusalKind::IntegrationUnsupported)),
        Op::Limit(target) => {
            let point = parse(target).map_err(FrankenFailure::parse)?;
            limit(&expr, &var, &point)
                .map_err(|error| FrankenFailure::calculus(error, RefusalKind::LimitUndetermined))
        }
        Op::Taylor { at, order } => {
            let point = Expr::from_i64(*at);
            taylor(&expr, &var, &point, *order)
                .map_err(|error| FrankenFailure::calculus(error, RefusalKind::TaylorUnsupported))
        }
    }
}

/// The fixed differential corpus. Extend here; keep cases deterministic.
pub fn corpus() -> Vec<CaseSpec> {
    let mk = |id: &str, expr: &str, var: &str, op: Op| CaseSpec {
        case_id: id.to_string(),
        input_expr: expr.to_string(),
        var: var.to_string(),
        op,
        expected_refusal: None,
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
        // Integration by parts (capability outgrew the old int_refusal_001
        // expectation; residual d/dx[-x*cos(x) + sin(x)] == x*sin(x) verified
        // against the pinned oracle lane).
        mk("int_005", "x * sin(x)", "x", Op::Integrate),
        mk("lim_001", "2*x + 1", "x", Op::Limit("oo".into())),
        mk("lim_002", "-x^5", "x", Op::Limit("-oo".into())),
        mk("lim_003", "x + 4", "x", Op::Limit("5".into())),
        mk("tay_001", "exp(x)", "x", Op::Taylor { at: 0, order: 6 }),
        mk("tay_002", "cos(x)", "x", Op::Taylor { at: 0, order: 4 }),
    ];
    // Known engine limitations: these must refuse, never guess. `x*exp(x^2)`
    // has no elementary antiderivative and the engine's integrator refuses it
    // (live-verified 2026-09-04); the refusal lane must keep at least one such
    // genuinely unsupported form.
    v.push(CaseSpec {
        case_id: "int_refusal_001".to_string(),
        input_expr: "x * exp(x^2)".to_string(),
        var: "x".to_string(),
        op: Op::Integrate,
        expected_refusal: Some(RefusalKind::IntegrationUnsupported),
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

payload = json.loads(sys.argv[1])
actual_version = sp.__version__
print(json.dumps({
    "kind": "meta",
    "schema_version": payload["schema_version"],
    "profile_id": payload["profile_id"],
    "sympy_version": actual_version,
}), flush=True)
if actual_version != payload["required_sympy_version"]:
    print(
        f"SymPy version mismatch: required={payload['required_sympy_version']} actual={actual_version}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(3)

CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
    "i": sp.I,
    "infinity": sp.oo,
    "negative_infinity": -sp.oo,
    "complex_infinity": sp.zoo,
    "nan": sp.nan,
}
KNOWN_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "exp": sp.exp,
    "log": sp.log,
    "gamma": sp.gamma,
    "zeta": sp.zeta,
}

def from_expr(node):
    kind = node["kind"]
    if kind == "symbol":
        return sp.Symbol(node["name"])
    if kind == "integer":
        return sp.Integer(node["value"])
    if kind == "rational":
        return sp.Rational(node["numerator"], node["denominator"])
    if kind == "constant":
        return CONSTANTS[node["name"]]
    if kind == "add":
        return sp.Add(*(from_expr(child) for child in node["children"]))
    if kind == "mul":
        return sp.Mul(*(from_expr(child) for child in node["children"]))
    if kind == "pow":
        return sp.Pow(from_expr(node["base"]), from_expr(node["exponent"]))
    if kind == "function":
        function = KNOWN_FUNCTIONS.get(node["name"])
        if function is None:
            function = sp.Function(node["name"])
        return function(*(from_expr(arg) for arg in node["args"]))
    raise ValueError(f"unknown expression node kind: {kind}")

for c in payload["cases"]:
    try:
        expr = from_expr(c["input"])
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
            theirs = sp.limit(expr, var, from_expr(op["target"]))
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
            ours = from_expr(c["ours"])
            exact = ours == theirs
            delta = sp.S.Zero if exact else sp.simplify(ours - theirs)
            agrees = exact or delta == 0
            verdict = "pass" if agrees else "mismatch"
            detail = None if agrees else f"ours={ours} theirs={theirs} delta={delta}"
        print(json.dumps({
            "kind": "case",
            "id": c["id"],
            "lane": "rust_native",
            "verdict": verdict,
            "comparator": "native_math_exact",
            "expected": expected,
            "detail": detail,
        }), flush=True)
    except Exception as exc:  # oracle failures are evidence, never a pass
        print(json.dumps({
            "kind": "case",
            "id": c["id"],
            "lane": "rust_native",
            "verdict": "oracle_error",
            "comparator": "native_math_exact",
            "expected": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }), flush=True)

signal.alarm(0)
"#;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum OracleRecord {
    Meta {
        schema_version: u32,
        profile_id: String,
        sympy_version: String,
    },
    Case {
        id: String,
        #[allow(dead_code)]
        lane: String,
        #[allow(dead_code)]
        comparator: Option<String>,
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
    if order > MAX_TAYLOR_ORDER {
        return Err(format!(
            "Taylor order {order} exceeds harness maximum {MAX_TAYLOR_ORDER}"
        ));
    }
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
    if cases.is_empty() {
        return Err("corpus must not be empty".to_string());
    }
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
        parse(&spec.input_expr)
            .map_err(|error| format!("case `{}` input failed parsing: {error}", spec.case_id))?;
        if let Op::Limit(target) = &spec.op {
            parse(target).map_err(|error| {
                format!(
                    "case `{}` limit target failed parsing: {error}",
                    spec.case_id
                )
            })?;
        }
        let _ = sympy_operation(spec)?;
    }
    Ok(())
}

fn expr_payload(expr: &Expr) -> serde_json::Value {
    match expr {
        Expr::Sym(symbol) => serde_json::json!({"kind": "symbol", "name": symbol.name}),
        Expr::Integer(value) => {
            serde_json::json!({"kind": "integer", "value": value.to_string()})
        }
        Expr::Rational(value) => serde_json::json!({
            "kind": "rational",
            "numerator": value.numer().to_string(),
            "denominator": value.denom().to_string(),
        }),
        Expr::Const(value) => {
            let name = match value {
                Constant::Pi => "pi",
                Constant::E => "e",
                Constant::I => "i",
                Constant::Infinity => "infinity",
                Constant::NegativeInfinity => "negative_infinity",
                Constant::ComplexInfinity => "complex_infinity",
                Constant::NaN => "nan",
            };
            serde_json::json!({"kind": "constant", "name": name})
        }
        Expr::Add(children) => serde_json::json!({
            "kind": "add",
            "children": children.iter().map(expr_payload).collect::<Vec<_>>(),
        }),
        Expr::Mul(children) => serde_json::json!({
            "kind": "mul",
            "children": children.iter().map(expr_payload).collect::<Vec<_>>(),
        }),
        Expr::Pow(base, exponent) => serde_json::json!({
            "kind": "pow",
            "base": expr_payload(base),
            "exponent": expr_payload(exponent),
        }),
        Expr::Function(name, args) => serde_json::json!({
            "kind": "function",
            "name": name,
            "args": args.iter().map(expr_payload).collect::<Vec<_>>(),
        }),
    }
}

fn oracle_request(spec: &CaseSpec, ours: Option<&Expr>) -> Result<serde_json::Value, String> {
    let input = parse(&spec.input_expr)
        .map_err(|error| format!("case `{}` input failed parsing: {error}", spec.case_id))?;
    let op = match &spec.op {
        Op::Simplify => serde_json::json!({"name": "simplify"}),
        Op::Expand => serde_json::json!({"name": "expand"}),
        Op::Diff => serde_json::json!({"name": "diff"}),
        Op::Integrate => serde_json::json!({"name": "integrate"}),
        Op::Limit(target) => {
            let target = parse(target).map_err(|error| {
                format!(
                    "case `{}` limit target failed parsing: {error}",
                    spec.case_id
                )
            })?;
            serde_json::json!({"name": "limit", "target": expr_payload(&target)})
        }
        Op::Taylor { at, order } => serde_json::json!({
            "name": "taylor",
            "at": at,
            "series_terms": series_terms(*order)?,
        }),
    };
    Ok(serde_json::json!({
        "id": spec.case_id,
        "input": expr_payload(&input),
        "var": spec.var,
        "op": op,
        "ours": ours.map(expr_payload),
    }))
}

fn parse_oracle_stdout(stdout: &str) -> Result<(String, HashMap<String, OracleLine>), String> {
    let mut metadata = None;
    let mut by_id = HashMap::new();
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let parsed: OracleRecord =
            serde_json::from_str(line).map_err(|e| format!("bad oracle line `{line}`: {e}"))?;
        match parsed {
            OracleRecord::Meta {
                schema_version,
                profile_id,
                sympy_version,
            } => {
                if metadata
                    .replace((schema_version, profile_id, sympy_version))
                    .is_some()
                {
                    return Err("oracle emitted duplicate metadata records".to_string());
                }
            }
            OracleRecord::Case {
                id,
                lane: _,
                comparator: _,
                verdict,
                expected,
                detail,
            } => {
                let valid_shape = match verdict.as_str() {
                    "pass" | "expectation_only" => expected.is_some() && detail.is_none(),
                    "mismatch" => expected.is_some() && detail.is_some(),
                    "oracle_error" => expected.is_none() && detail.is_some(),
                    _ => false,
                };
                if !valid_shape {
                    return Err(format!(
                        "oracle emitted invalid `{verdict}` record shape for case_id `{id}`"
                    ));
                }
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
    let (schema_version, profile_id, version) =
        metadata.ok_or_else(|| "oracle emitted no metadata record".to_string())?;
    if schema_version != PROTOCOL_SCHEMA_VERSION {
        return Err(format!(
            "oracle schema mismatch: required={PROTOCOL_SCHEMA_VERSION} actual={schema_version}"
        ));
    }
    if profile_id != PROFILE_ID {
        return Err(format!(
            "oracle corpus profile mismatch: required={PROFILE_ID} actual={profile_id}"
        ));
    }
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
    rust_result: &Result<Expr, FrankenFailure>,
    oracle: &OracleLine,
) -> Result<Verdict, String> {
    match (rust_result, oracle.verdict.as_str()) {
        (_, "oracle_error") => Ok(Verdict::OracleError),
        (Err(failure), "expectation_only")
            if spec.expected_refusal.is_some() && spec.expected_refusal == failure.kind =>
        {
            Ok(Verdict::ExpectedRefusal)
        }
        (Err(_), "expectation_only") => Ok(Verdict::RefusalMismatch),
        (Ok(_), "pass" | "mismatch") if spec.expected_refusal.is_some() => {
            Ok(Verdict::UnexpectedSuccess)
        }
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
    rust_results: Vec<Result<Expr, FrankenFailure>>,
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
            actual_frankensympy_output: rust_result.as_ref().ok().map(ToString::to_string),
            frankensympy_refusal_kind: rust_result.as_ref().err().and_then(|failure| failure.kind),
            frankensympy_refusal: rust_result
                .as_ref()
                .err()
                .map(|failure| failure.detail.clone()),
            oracle_detail: oracle.detail,
            verdict,
        });
    }
    Ok(report)
}

/// Run the compiled, fixed native-mathematics corpus against a separate live
/// SymPy process. External callers cannot inject cases into this lane.
///
/// `Err` means the run itself is invalid: malformed/duplicate cases, a
/// missing or wrong-version oracle, child-process failure, or a broken
/// result protocol. Individual mathematical disagreements and typed Rust
/// refusals are returned as explicit case verdicts.
pub fn run_conformance(python: &str) -> Result<Vec<ConformanceCase>, String> {
    run_cases_with_timeout(&corpus(), python, ORACLE_TIMEOUT)
}

fn run_cases_with_timeout(
    cases: &[CaseSpec],
    python: &str,
    timeout: Duration,
) -> Result<Vec<ConformanceCase>, String> {
    validate_cases(cases)?;
    let rust_results: Vec<Result<Expr, FrankenFailure>> =
        cases.iter().map(franken_apply_expr).collect();
    let oracle_cases: Vec<serde_json::Value> = cases
        .iter()
        .zip(&rust_results)
        .map(|(spec, result)| oracle_request(spec, result.as_ref().ok()))
        .collect::<Result<_, _>>()?;
    let payload = serde_json::json!({
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "required_sympy_version": PINNED_SYMPY_VERSION,
        "cases": oracle_cases,
    });
    let payload = serde_json::to_string(&payload)
        .map_err(|e| format!("serializing oracle request failed: {e}"))?;
    if payload.len() > MAX_ORACLE_REQUEST_BYTES {
        return Err(format!(
            "oracle request size {} exceeds limit of {} bytes",
            payload.len(),
            MAX_ORACLE_REQUEST_BYTES
        ));
    }

    let mut command = Command::new(python);
    command
        .args(["-I", "-W", "error", "-c", ORACLE_SCRIPT])
        .arg(&payload);
    let output = run_bounded_oracle(command, timeout)?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "oracle exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }
    let stderr =
        String::from_utf8(output.stderr).map_err(|e| format!("oracle stderr is not UTF-8: {e}"))?;
    if !stderr.trim().is_empty() {
        return Err(format!(
            "oracle emitted unexpected stderr: {}",
            stderr.trim()
        ));
    }
    let stdout =
        String::from_utf8(output.stdout).map_err(|e| format!("oracle stdout is not UTF-8: {e}"))?;
    let (oracle_version, by_id) = parse_oracle_stdout(&stdout)?;
    assemble_report(cases, rust_results, &oracle_version, by_id)
}

#[cfg(unix)]
fn run_bounded_oracle(
    mut command: Command,
    timeout: Duration,
) -> Result<std::process::Output, String> {
    use std::os::unix::process::CommandExt;

    command.process_group(0);
    let deadline = Instant::now() + timeout;
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawning oracle process failed: {e}"))?;
    loop {
        let status = match child.try_wait() {
            Ok(status) => status,
            Err(error) => {
                let cleanup = abort_oracle_group(&mut child);
                return Err(match cleanup {
                    Ok(()) => format!("polling oracle failed: {error}"),
                    Err(cleanup) => {
                        format!("polling oracle failed: {error}; cleanup failed: {cleanup}")
                    }
                });
            }
        };
        match status {
            Some(_) => {
                kill_oracle_group(child.id())?;
                break;
            }
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            None => {
                let cleanup = abort_oracle_group(&mut child);
                return Err(match cleanup {
                    Ok(()) => format!(
                        "oracle exceeded parent wall-time bound of {} seconds",
                        timeout.as_secs_f64()
                    ),
                    Err(cleanup) => format!(
                        "oracle exceeded parent wall-time bound of {} seconds; cleanup failed: {cleanup}",
                        timeout.as_secs_f64()
                    ),
                });
            }
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("reaping oracle failed: {e}"))?;
    let output_bytes = output
        .stdout
        .len()
        .checked_add(output.stderr.len())
        .ok_or_else(|| "oracle output length overflowed".to_string())?;
    if output_bytes > MAX_ORACLE_OUTPUT_BYTES {
        return Err(format!(
            "oracle output has {output_bytes} bytes; maximum is {MAX_ORACLE_OUTPUT_BYTES}"
        ));
    }
    Ok(output)
}

#[cfg(not(unix))]
fn run_bounded_oracle(
    _command: Command,
    _timeout: Duration,
) -> Result<std::process::Output, String> {
    Err("live oracle execution requires Unix process-group containment".to_string())
}

#[cfg(unix)]
fn oracle_group_exists(process_group: u32) -> Result<bool, String> {
    let target = format!("-{process_group}");
    let status = Command::new("/bin/kill")
        .args(["-0", "--", &target])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("probing oracle process group failed: {error}"))?;
    Ok(status.success())
}

#[cfg(all(unix, test))]
fn oracle_process_exists(process: u32) -> Result<bool, String> {
    let status = Command::new("/bin/kill")
        .args(["-0", "--", &process.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("probing oracle descendant failed: {error}"))?;
    Ok(status.success())
}

#[cfg(unix)]
fn kill_oracle_group(process_group: u32) -> Result<(), String> {
    if !oracle_group_exists(process_group)? {
        return Ok(());
    }
    let target = format!("-{process_group}");
    let output = Command::new("/bin/kill")
        .args(["-KILL", "--", &target])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("killing oracle process group failed: {error}"))?;
    if output.status.success() || !oracle_group_exists(process_group)? {
        return Ok(());
    }
    Err(format!(
        "killing oracle process group failed with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

#[cfg(unix)]
fn abort_oracle_group(child: &mut std::process::Child) -> Result<(), String> {
    if let Err(group_error) = kill_oracle_group(child.id()) {
        let direct_error = child.kill().err();
        let wait_error = child.wait().err();
        return Err(format!(
            "{group_error}; direct kill error={direct_error:?}; reap error={wait_error:?}"
        ));
    }
    child
        .wait()
        .map(|_| ())
        .map_err(|error| format!("reaping killed oracle failed: {error}"))
}

/// Write an NDJSON evidence ledger; parent directories created as needed.
pub fn write_evidence_ndjson(cases: &[ConformanceCase], path: &Path) -> std::io::Result<PathBuf> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    for c in cases {
        serde_json::to_writer(&mut f, c).map_err(std::io::Error::other)?;
        f.write_all(b"\n")?;
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
            expected_refusal: if expect_refusal {
                Some(RefusalKind::IntegrationUnsupported)
            } else {
                None
            },
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
    fn native_expand_limit_is_recorded_as_failure_instead_of_panicking() {
        let left = (0..65)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>()
            .join("+");
        let right = (0..65)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>()
            .join("+");
        let spec = CaseSpec {
            case_id: "oversized_expand".to_string(),
            input_expr: format!("({left})*({right})"),
            var: "x".to_string(),
            op: Op::Expand,
            expected_refusal: None,
        };

        let failure = franken_apply_expr(&spec).expect_err("expansion must refuse");
        assert_eq!(failure.kind, None);
        assert!(failure.detail.contains("term limit"));
    }

    #[test]
    fn planted_mismatch_is_never_conformant() {
        let x_expr = parse("x").unwrap();
        let verdict =
            classify(&case(false), &Ok(x_expr), &oracle("mismatch")).expect("known verdict");
        assert_eq!(verdict, Verdict::Mismatch);
        assert!(!verdict.is_conformant());
        assert!(!verdict.is_expected_outcome());
    }

    #[test]
    fn expected_refusal_is_not_a_conformance_pass() {
        let err = FrankenFailure {
            kind: Some(RefusalKind::IntegrationUnsupported),
            detail: "typed integration refusal".to_string(),
        };
        let verdict =
            classify(&case(true), &Err(err), &oracle("expectation_only")).expect("known verdict");
        assert_eq!(verdict, Verdict::ExpectedRefusal);
        assert!(!verdict.is_conformant());
        assert!(verdict.is_expected_outcome());
    }

    #[test]
    fn wrong_failure_kind_cannot_satisfy_expected_refusal() {
        for err in [
            FrankenFailure {
                kind: None,
                detail: "parser regression".to_string(),
            },
            FrankenFailure {
                kind: Some(RefusalKind::LimitUndetermined),
                detail: "wrong capability refusal".to_string(),
            },
        ] {
            let verdict = classify(&case(true), &Err(err), &oracle("expectation_only"))
                .expect("known verdict");
            assert_eq!(verdict, Verdict::RefusalMismatch);
            assert!(!verdict.is_expected_outcome());
        }
    }

    #[test]
    fn success_where_refusal_was_required_is_unexpected() {
        let two_x = parse("2*x").unwrap();
        for oracle_verdict in ["pass", "mismatch"] {
            let verdict = classify(&case(true), &Ok(two_x.clone()), &oracle(oracle_verdict))
                .expect("known verdict");
            assert_eq!(verdict, Verdict::UnexpectedSuccess);
            assert!(!verdict.is_expected_outcome());
        }
    }

    #[test]
    fn unexpected_refusal_remains_a_mismatch() {
        let err = FrankenFailure {
            kind: None,
            detail: "unsupported".to_string(),
        };
        let verdict =
            classify(&case(false), &Err(err), &oracle("expectation_only")).expect("known verdict");
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
            expected_refusal: None,
        };
        assert_eq!(
            sympy_operation(&spec).expect("bounded order"),
            "series(exp(x), x, 0, 7).removeO()"
        );
        let one_plus_x = parse("1 + x").unwrap();
        let request = oracle_request(&spec, Some(&one_plus_x)).expect("request");
        assert_eq!(request["op"]["series_terms"], 7);
        assert!(request.get("input_expr").is_none());
        assert_eq!(request["input"]["kind"], "function");
    }

    #[test]
    fn taylor_operation_refuses_order_above_harness_maximum() {
        // series_terms refuses orders > MAX_TAYLOR_ORDER so that the
        // oracle call below is bounded. Pin the boundary at the
        // boundary value (MAX_TAYLOR_ORDER is still allowed;
        // MAX_TAYLOR_ORDER + 1 is refused) so a future change to
        // MAX_TAYLOR_ORDER cannot silently loosen the gate.
        assert_eq!(
            sympy_operation(&CaseSpec {
                case_id: "tay_at_max".to_string(),
                input_expr: "exp(x)".to_string(),
                var: "x".to_string(),
                op: Op::Taylor {
                    at: 0,
                    order: MAX_TAYLOR_ORDER
                },
                expected_refusal: None,
            })
            .expect("max order is still admitted"),
            format!("series(exp(x), x, 0, {}).removeO()", MAX_TAYLOR_ORDER + 1)
        );

        let err = sympy_operation(&CaseSpec {
            case_id: "tay_over_max".to_string(),
            input_expr: "exp(x)".to_string(),
            var: "x".to_string(),
            op: Op::Taylor {
                at: 0,
                order: MAX_TAYLOR_ORDER + 1,
            },
            expected_refusal: None,
        })
        .unwrap_err();
        assert!(err.contains("exceeds harness maximum"));
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
            "{\"kind\":\"meta\",\"schema_version\":1,\"profile_id\":\"native-math-corpus-v1\",\"sympy_version\":\"1.14.0\"}\n",
            "{\"kind\":\"case\",\"id\":\"planted_001\",\"lane\":\"rust_native\",\"comparator\":null,\"verdict\":\"pass\",\"expected\":\"2*x\",\"detail\":null}\n",
            "{\"kind\":\"case\",\"id\":\"planted_001\",\"lane\":\"rust_native\",\"comparator\":null,\"verdict\":\"pass\",\"expected\":\"2*x\",\"detail\":null}\n"
        );
        assert!(
            parse_oracle_stdout(duplicate)
                .expect_err("duplicate output must fail")
                .contains("duplicate case_id")
        );

        let meta_only = "{\"kind\":\"meta\",\"schema_version\":1,\"profile_id\":\"native-math-corpus-v1\",\"sympy_version\":\"1.14.0\"}\n";
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
    fn oracle_protocol_rejects_unknown_fields_and_invalid_verdict_shapes() {
        let unknown = "{\"kind\":\"meta\",\"schema_version\":1,\"profile_id\":\"native-math-corpus-v1\",\"sympy_version\":\"1.14.0\",\"extra\":true}\n";
        assert!(
            parse_oracle_stdout(unknown)
                .expect_err("unknown field must fail")
                .contains("unknown field")
        );

        let invalid = concat!(
            "{\"kind\":\"meta\",\"schema_version\":1,\"profile_id\":\"native-math-corpus-v1\",\"sympy_version\":\"1.14.0\"}\n",
            "{\"kind\":\"case\",\"id\":\"planted_001\",\"lane\":\"rust_native\",\"comparator\":null,\"verdict\":\"pass\",\"expected\":null,\"detail\":null}\n"
        );
        assert!(
            parse_oracle_stdout(invalid)
                .expect_err("pass without expected output must fail")
                .contains("invalid `pass` record shape")
        );
    }

    #[test]
    fn taylor_order_is_bounded_before_oracle_execution() {
        let mut spec = case(false);
        spec.op = Op::Taylor {
            at: 0,
            order: MAX_TAYLOR_ORDER + 1,
        };
        let err = validate_cases(&[spec]).expect_err("oversized order must fail");
        assert!(err.contains("exceeds harness maximum"));
    }

    #[test]
    fn unreachable_oracle_fails_closed() {
        let err = run_cases_with_timeout(
            &[case(false)],
            "/definitely/missing/frankensympy-python",
            Duration::from_millis(100),
        )
        .expect_err("missing interpreter must fail");
        assert!(err.contains("spawning oracle"));
    }

    #[cfg(unix)]
    #[test]
    fn output_flood_is_killed_and_reaped_within_parent_deadline() {
        let started = Instant::now();
        let err = run_bounded_oracle(Command::new("/usr/bin/yes"), Duration::from_millis(100))
            .expect_err("flooding child must fail");
        assert!(err.contains("parent wall-time bound"));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[cfg(unix)]
    #[test]
    fn exited_parent_cannot_leave_a_descendant_holding_oracle_pipes() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "sleep 5 & echo $!"]);
        let started = Instant::now();
        let output = run_bounded_oracle(command, Duration::from_secs(1))
            .expect("supervisor must kill the inherited-pipe holder");
        assert!(output.status.success());
        assert!(started.elapsed() < Duration::from_secs(10));
        let descendant = String::from_utf8(output.stdout)
            .expect("pid output is UTF-8")
            .trim()
            .parse::<u32>()
            .expect("pid output is numeric");
        let mut alive = oracle_process_exists(descendant).expect("process probe works");
        for _ in 0..100 {
            if !alive {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
            alive = oracle_process_exists(descendant).expect("process probe works");
        }
        assert!(!alive, "descendant {descendant} survived group cleanup");
    }

    #[test]
    fn wrong_sympy_version_fails_closed() {
        let stdout = "{\"kind\":\"meta\",\"schema_version\":1,\"profile_id\":\"native-math-corpus-v1\",\"sympy_version\":\"1.13.3\"}\n";
        let err = parse_oracle_stdout(stdout).expect_err("wrong version must fail");
        assert!(err.contains("required=1.14.0 actual=1.13.3"));
    }

    #[test]
    fn empty_and_executable_source_corpora_fail_before_oracle() {
        assert!(
            validate_cases(&[])
                .expect_err("empty corpus")
                .contains("must not be empty")
        );
        let mut executable = case(true);
        executable.input_expr = "__import__('builtins').len([1,2,3])".to_string();
        let err = validate_cases(&[executable]).expect_err("Rust grammar rejects Python source");
        assert!(err.contains("input failed parsing"));
    }

    #[test]
    fn validate_cases_rejects_empty_oversized_and_unparseable_target() {
        // The validate_cases() guards upstream of the oracle call:
        // empty case_id, oversized fields, unparseable limit target,
        // and unparseable input must all refuse before any side effect.
        // Pin each guard so a future refactor that drops one of them
        // becomes a test failure rather than a silent
        // over-the-wire dependency.

        // Empty case_id (whitespace-only counts as empty).
        let mut blank = case(false);
        blank.case_id = "   ".to_string();
        let err = validate_cases(&[blank]).expect_err("blank case_id");
        assert!(err.contains("case_id must not be empty"));

        // Oversized case_id: 17 KiB exceeds the 16 KiB field limit.
        let mut giant = case(false);
        giant.case_id = "x".repeat(17 * 1_024);
        let err = validate_cases(&[giant]).expect_err("oversized case_id");
        assert!(err.contains("exceeds 16384 bytes"));

        // Oversized input_expr: same limit applies to all three
        // string fields (case_id, input_expr, var).
        let mut giant_input = case(false);
        giant_input.input_expr = "x".repeat(17 * 1_024);
        let err = validate_cases(&[giant_input]).expect_err("oversized input");
        assert!(err.contains("exceeds 16384 bytes"));

        // Unparseable limit target: the case has valid structure
        // but the Op::Limit target string fails to parse as Expr.
        let bad_limit = CaseSpec {
            case_id: "bad_limit".to_string(),
            input_expr: "x".to_string(),
            var: "x".to_string(),
            op: Op::Limit("not-an-expression-%%%".into()),
            expected_refusal: None,
        };
        let err = validate_cases(&[bad_limit]).expect_err("bad limit target");
        assert!(err.contains("limit target failed parsing"));
    }
    #[test]
    fn op_and_case_spec_serde_roundtrip_preserves_wire_format() {
        // The conformance corpus and remote oracle exchange
        // CaseSpec through serde_json. Pin the wire format of every
        // Op variant so a future refactor cannot silently change the
        // tag content (which would invalidate stored corpus files
        // and remote cache entries).
        let cases = [
            (Op::Simplify, r#"{"op":"Simplify"}"#),
            (Op::Expand, r#"{"op":"Expand"}"#),
            (Op::Diff, r#"{"op":"Diff"}"#),
            (Op::Integrate, r#"{"op":"Integrate"}"#),
            (Op::Limit("oo".into()), r#"{"op":"Limit","arg":"oo"}"#),
            (
                Op::Taylor { at: 0, order: 6 },
                r#"{"op":"Taylor","arg":{"at":0,"order":6}}"#,
            ),
        ];
        for (op, expected_json) in cases {
            assert_eq!(serde_json::to_string(&op).unwrap(), expected_json);
            let round: Op = serde_json::from_str(expected_json).unwrap();
            assert_eq!(round, op);
        }

        // CaseSpec: every field survives a round-trip including
        // the optional expected_refusal.
        let spec = CaseSpec {
            case_id: "wire_001".to_string(),
            input_expr: "x + x".to_string(),
            var: "x".to_string(),
            op: Op::Taylor { at: 2, order: 3 },
            expected_refusal: Some(RefusalKind::IntegrationUnsupported),
        };
        let json = serde_json::to_string(&spec).unwrap();
        let round: CaseSpec = serde_json::from_str(&json).unwrap();
        // CaseSpec does not derive PartialEq, so verify each field
        // individually. Together these cover the wire round-trip.
        assert_eq!(round.case_id, spec.case_id);
        assert_eq!(round.input_expr, spec.input_expr);
        assert_eq!(round.var, spec.var);
        assert_eq!(round.op, spec.op);
        assert_eq!(round.expected_refusal, spec.expected_refusal);
    }
}

#[cfg(test)]
mod verdict_schema_alignment {
    use crate::ORACLE_SCRIPT;
    /// The emitted oracle-script verdict records must carry every required
    /// field of the SHARED verdict schema (tools/conformance-lab/schema/
    /// verdict.schema.json) so drift counts are comparable across lanes
    /// (fra-conformance-corpus-200-b75; Art. XXII documentation law).
    const SCHEMA: &str = include_str!("../../../tools/conformance-lab/schema/verdict.schema.json");

    #[test]
    fn oracle_script_emits_shared_schema_required_fields() {
        let schema: serde_json::Value =
            serde_json::from_str(SCHEMA).expect("shared verdict schema must be valid JSON");
        let required = schema["required"]
            .as_array()
            .expect("schema.required must be an array")
            .iter()
            .map(|v| v.as_str().expect("required entries are strings"))
            .collect::<Vec<_>>();
        assert!(required.contains(&"lane"));
        assert!(required.contains(&"comparator"));
        assert!(required.contains(&"verdict"));
        for field in &required {
            let needle = format!("\"{field}\":");
            assert!(
                ORACLE_SCRIPT.contains(&needle),
                "ORACLE_SCRIPT verdict records miss shared-schema field {field}"
            );
        }
        let lane = schema["properties"]["lane"]["enum"]
            .as_array()
            .expect("lane enum")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(
            lane.contains(&"rust_native"),
            "schema must know the rust_native lane"
        );
    }

    #[test]
    fn emitted_verdict_values_are_in_shared_vocabulary() {
        let schema: serde_json::Value = serde_json::from_str(SCHEMA).unwrap();
        let allowed: Vec<&str> = schema["properties"]["verdict"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for emitted in ["pass", "mismatch", "oracle_error", "expectation_only"] {
            assert!(
                allowed.contains(&emitted),
                "emitted verdict {emitted} missing from shared verdict vocabulary"
            );
        }
    }
}
