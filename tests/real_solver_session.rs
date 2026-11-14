//! Complete exact real solving through frozen native sources and fresh NDJSON.

use frankensympy::{
    BigRational, BudgetLimits, CertificationRequest, CertificationResponse, Dimension,
    ExactRealSolutionSet, RealInequalityRelation, ResidualSystemId, Session,
};
use fsym_budget::Unbounded;
use fsym_core::real_roots::{
    RealRootInterval, RealRootIsolationCertificate, verify_real_root_isolation,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits
}

fn session() -> Session {
    Session::with_budget_limits(limits(), 256)
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:real").unwrap()
}

fn registration(sources: &[&str], variables: &[&str]) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: sources.iter().map(|value| (*value).into()).collect(),
        variables: variables.iter().map(|value| (*value).into()).collect(),
    }
}

fn equation() -> CertificationRequest {
    CertificationRequest::SolveRealEquation {
        schema_version: 1,
        system: system(),
    }
}

fn inequality(relation: RealInequalityRelation) -> CertificationRequest {
    CertificationRequest::SolveRealInequality {
        schema_version: 1,
        system: system(),
        relation,
    }
}

fn registered(source: &str) -> Session {
    let mut session = session();
    session
        .certification_request(registration(&[source], &["x"]))
        .unwrap();
    session
}

fn set(response: CertificationResponse) -> ExactRealSolutionSet {
    match response {
        CertificationResponse::RealEquationSolved { solution_set, .. }
        | CertificationResponse::RealInequalitySolved { solution_set, .. } => solution_set,
        _ => panic!("expected complete real solution set"),
    }
}

fn rational(value: &str) -> BigRational {
    if let Some((numerator, denominator)) = value.split_once('/') {
        BigRational::new(numerator.parse().unwrap(), denominator.parse().unwrap())
    } else {
        BigRational::from_integer(value.parse().unwrap())
    }
}

/// Independently replay the output root basis, so the wire representation
/// cannot accidentally lose a coefficient, reorder a root, or broaden a point.
fn check_basis(set: &ExactRealSolutionSet) {
    assert_eq!(set.point_inclusions.len(), set.roots.len());
    assert_eq!(set.gap_inclusions.len(), set.roots.len() + 1);
    let coefficients: Vec<_> = set.polynomial.iter().map(|value| rational(value)).collect();
    let candidate = RealRootIsolationCertificate {
        intervals: set
            .roots
            .iter()
            .map(|interval| RealRootInterval {
                lower: rational(&interval.lower),
                upper: rational(&interval.upper),
                multiplicity: 1,
            })
            .collect(),
    };
    verify_real_root_isolation(&coefficients, &candidate, &mut Unbounded).unwrap();
}

#[test]
fn quintic_solutions_are_complete_and_bound_to_the_frozen_source_and_context() {
    let mut session = registered("x^5-x+1");
    let source = session.residual_problem(&system()).unwrap().clone();
    let context = session.context_digest();
    session.construct("x", "100").unwrap();
    let before = session.budget_snapshot();
    let response = session.certification_request(equation()).unwrap();
    let CertificationResponse::RealEquationSolved {
        system: solved,
        variable,
        evidence,
        claim,
        completeness,
        domain,
        solution_set,
        context_digest,
        ..
    } = response
    else {
        panic!("equation response")
    };
    assert_eq!(solved, system());
    assert_eq!(variable, source.variables()[0]);
    assert_eq!(context_digest, context);
    assert_eq!(evidence, "exact_cross_checked");
    assert_eq!(
        claim,
        "complete_real_solution_set_of_original_rational_equation"
    );
    assert_eq!((completeness, domain), ("complete", "real"));
    assert_eq!(solution_set.polynomial, ["1", "-1", "0", "0", "0", "1"]);
    assert_eq!(solution_set.roots.len(), 1);
    assert_eq!(solution_set.gap_inclusions, [false, false]);
    assert_eq!(solution_set.point_inclusions, [true]);
    check_basis(&solution_set);
    assert_eq!(session.residual_problem(&system()).unwrap(), &source);
    let after = session.budget_snapshot();
    assert!(after.verifier_pool < before.verifier_pool);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(after.dimensions[dimension.index()] < before.dimensions[dimension.index()]);
    }
}

#[test]
fn singular_polynomial_roots_are_returned_once_without_a_nonsingularity_claim() {
    let mut session = registered("x^4*(x-1)^2");
    let solution = set(session.certification_request(equation()).unwrap());
    assert_eq!(solution.polynomial, ["0", "-1", "1"]);
    assert_eq!(solution.roots.len(), 2);
    assert_eq!(solution.point_inclusions, [true, true]);
    assert_eq!(solution.gap_inclusions, [false; 3]);
    check_basis(&solution);
}

#[test]
fn identities_keep_original_poles_including_nested_and_zero_power_sources() {
    for source in ["x/x-1", "0*(1/x)", "0*((1/x)^0)"] {
        let mut session = registered(source);
        let solution = set(session.certification_request(equation()).unwrap());
        assert_eq!(solution.polynomial, ["0", "1"], "{source}");
        assert_eq!(solution.point_inclusions, [false], "{source}");
        assert_eq!(solution.gap_inclusions, [true, true], "{source}");
        check_basis(&solution);
    }
    let mut session = registered("0/((x^2-2)*(x^5-x+1))");
    let solution = set(session.certification_request(equation()).unwrap());
    assert_eq!(solution.roots.len(), 3);
    assert_eq!(solution.point_inclusions, [false; 3]);
    assert_eq!(solution.gap_inclusions, [true; 4]);
    check_basis(&solution);
}

#[test]
fn finite_sets_empty_sets_and_everywhere_defined_identities_remain_distinct() {
    for (source, expected_all_real) in [
        ("0", true),
        ("7", false),
        ("x^2+1", false),
        ("0/(x-x)", false),
        ("(x^2-2)^2/(x^2-2)", false),
        ("1/(1/x)", false),
    ] {
        let mut session = registered(source);
        let solution = set(session.certification_request(equation()).unwrap());
        assert!(solution.roots.is_empty(), "{source}");
        assert!(solution.point_inclusions.is_empty(), "{source}");
        assert_eq!(solution.gap_inclusions, [expected_all_real], "{source}");
        check_basis(&solution);
    }
    let mut session = registered("(x^2-1)/(x-1)");
    let solution = set(session.certification_request(equation()).unwrap());
    assert_eq!(solution.polynomial, ["1", "1"]);
    assert_eq!(solution.point_inclusions, [true]);
    assert_eq!(solution.gap_inclusions, [false, false]);
    check_basis(&solution);
}

#[test]
fn rational_inequalities_distinguish_irrational_zeros_from_excluded_poles() {
    let mut session = registered("(x^2-2)/(x^2-3)");
    let response = session
        .certification_request(inequality(RealInequalityRelation::Ge))
        .unwrap();
    let CertificationResponse::RealInequalitySolved {
        relation,
        evidence,
        completeness,
        domain,
        solution_set,
        context_digest,
        ..
    } = response
    else {
        panic!("inequality response")
    };
    assert_eq!(relation, RealInequalityRelation::Ge);
    assert_eq!(evidence, "exact_cross_checked");
    assert_eq!((completeness, domain), ("complete", "real"));
    assert_eq!(context_digest, session.context_digest());
    assert_eq!(solution_set.polynomial, ["6", "0", "-5", "0", "1"]);
    assert_eq!(
        solution_set.gap_inclusions,
        [true, false, true, false, true]
    );
    assert_eq!(solution_set.point_inclusions, [false, true, true, false]);
    check_basis(&solution_set);
    let strict = set(session
        .certification_request(inequality(RealInequalityRelation::Gt))
        .unwrap());
    assert_eq!(strict.polynomial, solution_set.polynomial);
    assert_eq!(strict.gap_inclusions, solution_set.gap_inclusions);
    assert_eq!(strict.point_inclusions, [false; 4]);
    check_basis(&strict);
}

#[test]
fn all_four_relations_retain_even_multiplicity_endpoint_ownership() {
    for (relation, gap, point) in [
        (RealInequalityRelation::Lt, false, false),
        (RealInequalityRelation::Le, false, true),
        (RealInequalityRelation::Gt, true, false),
        (RealInequalityRelation::Ge, true, true),
    ] {
        let mut session = registered("x^2");
        let solution = set(session.certification_request(inequality(relation)).unwrap());
        assert_eq!(solution.polynomial, ["0", "1"]);
        assert_eq!(solution.gap_inclusions, [gap; 2]);
        assert_eq!(solution.point_inclusions, [point]);
        check_basis(&solution);
    }
    for relation in [
        RealInequalityRelation::Lt,
        RealInequalityRelation::Le,
        RealInequalityRelation::Gt,
        RealInequalityRelation::Ge,
    ] {
        let mut session = registered("0/(x-x)");
        let solution = set(session.certification_request(inequality(relation)).unwrap());
        assert_eq!(solution.gap_inclusions, [false]);
        assert!(solution.roots.is_empty());
    }
}

#[test]
fn unknown_sources_wrong_dimensions_and_unsupported_grammar_never_mean_empty_sets() {
    let mut unknown = session();
    assert_eq!(
        unknown
            .certification_request(equation())
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    for (sources, variables) in [(vec!["x", "x^2"], vec!["x"]), (vec!["x+y"], vec!["x", "y"])] {
        let mut session = session();
        session
            .certification_request(registration(&sources, &variables))
            .unwrap();
        for request in [equation(), inequality(RealInequalityRelation::Le)] {
            assert_eq!(
                session.certification_request(request).unwrap_err().code(),
                "invalid_input"
            );
        }
    }
    for source in ["sin(x)", "0*sin(x)", "x^(1/2)"] {
        let mut session = registered(source);
        for request in [equation(), inequality(RealInequalityRelation::Le)] {
            assert_eq!(
                session.certification_request(request).unwrap_err().code(),
                "unsupported_operation",
                "{source}"
            );
        }
    }
    let mut oversized = registered("x^65");
    assert_eq!(
        oversized
            .certification_request(equation())
            .unwrap_err()
            .code(),
        "resource_limit_exceeded"
    );
}

#[test]
fn missing_and_exhausted_verifier_allowances_never_accept_or_reset_consumed_work() {
    for request in [equation(), inequality(RealInequalityRelation::Le)] {
        for verifier_pool in [0, 1] {
            let mut specification = limits();
            specification.verifier_pool = verifier_pool;
            let mut session = Session::with_budget_limits(specification, 256);
            session
                .certification_request(registration(&["x^2-2"], &["x"]))
                .unwrap();
            for _ in 0..2 {
                let before = session.budget_snapshot();
                assert_eq!(
                    session
                        .certification_request(request.clone())
                        .unwrap_err()
                        .code(),
                    "budget_exhausted"
                );
                let after = session.budget_snapshot();
                assert!(after.verifier_pool <= before.verifier_pool);
                if verifier_pool == 0 {
                    // A zero initial pool issues no lease. Authorization fails
                    // before generator or response work can begin.
                    assert_eq!(after, before);
                } else {
                    assert!(
                        after.dimensions[Dimension::ComputeSteps.index()]
                            < before.dimensions[Dimension::ComputeSteps.index()]
                    );
                }
            }
        }
    }
}

#[test]
fn ordinary_exhaustion_cannot_spend_reserved_compute_or_reset_shared_resources() {
    let mut probe = session();
    probe
        .certification_request(registration(&["x"], &["x"]))
        .unwrap();
    let after_registration = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut specification = limits();
        let slot = dimension.index();
        specification.dimensions[slot] -= after_registration.dimensions[slot];
        let mut session = Session::with_budget_limits(specification, 256);
        session
            .certification_request(registration(&["x"], &["x"]))
            .unwrap();
        let before = session.budget_snapshot();
        assert_eq!(before.dimensions[slot], 0);
        assert_eq!(
            session
                .certification_request(equation())
                .unwrap_err()
                .code(),
            "budget_exhausted"
        );
        assert_eq!(session.budget_snapshot(), before);
    }
}

#[test]
fn response_admission_failure_does_not_return_a_partially_checked_set() {
    // Exact output must retain coefficients far beyond binary64 and the
    // small numeric-point input envelope. Its entire rendering is charged.
    let magnitude = format!("1{}", "0".repeat(1100));
    let source = format!("x-{magnitude}");
    let mut probe = registered(&source);
    let solution = set(probe.certification_request(equation()).unwrap());
    assert_eq!(solution.polynomial, [format!("-{magnitude}"), "1".into()]);
    assert_eq!(solution.point_inclusions, [true]);
    check_basis(&solution);
    let completed = probe.budget_snapshot();
    let mut specification = limits();
    let slot = Dimension::ComputeSteps.index();
    specification.dimensions[slot] -= completed.dimensions[slot] + 1;
    let mut bounded = Session::with_budget_limits(specification, 256);
    bounded
        .certification_request(registration(&[&source], &["x"]))
        .unwrap();
    let before = bounded.budget_snapshot();
    assert_eq!(
        bounded
            .certification_request(equation())
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    let failed = bounded.budget_snapshot();
    assert!(failed.verifier_pool < before.verifier_pool);
    assert!(failed.dimensions[slot] < before.dimensions[slot]);
    assert_eq!(
        bounded
            .certification_request(equation())
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(bounded.budget_snapshot().verifier_pool <= failed.verifier_pool);
}

#[test]
fn strict_wire_rejects_forged_acceptance_masks_unknown_relations_and_duplicate_keys() {
    let mut session = registered("x");
    for raw in [
        r#"{"type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:real","accepted":true}}"#,
        r#"{"type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:real","solution_set":{"gap_inclusions":[true]}}}"#,
        r#"{"type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:real","residuals":["0"]}}"#,
        r#"{"type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:real","variables":["y"]}}"#,
        r#"{"type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:real","domain":["0","1"]}}"#,
        r#"{"type":"SolveRealEquation","payload":{"schema_version":1,"system":"system:real","system":"system:other"}}"#,
        r#"{"type":"SolveRealInequality","payload":{"schema_version":1,"system":"system:real","relation":"eq"}}"#,
        r#"{"type":"SolveRealInequality","payload":{"schema_version":1,"system":"system:real","relation":"lt","relation":"ge"}}"#,
    ] {
        let before = session.budget_snapshot().verifier_pool;
        let response: Value = serde_json::from_str(&session.handle_envelope(raw)).unwrap();
        assert_eq!(response["status"], "error", "{raw}");
        assert_eq!(response["code"], "malformed_request", "{raw}");
        assert_eq!(session.budget_snapshot().verifier_pool, before);
    }
    let mut request = serde_json::to_value(equation()).unwrap();
    request["payload"]["schema_version"] = json!(2);
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
    assert_eq!(response["code"], "unsupported_schema");
    let accepted = set(session.certification_request(equation()).unwrap());
    assert_eq!(accepted.gap_inclusions, [false, false]);
    assert_eq!(accepted.point_inclusions, [true]);
}

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args([
                "--compute-steps",
                "100000000",
                "--verifier-steps",
                "100000000",
                "--memory-bytes",
                "1073741824",
                "--allocations",
                "4000000",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take();
        let output = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            input,
            output,
        }
    }

    fn send(&mut self, request: CertificationRequest, id: &str) -> Value {
        let mut frame = serde_json::to_value(request).unwrap();
        frame["id"] = json!(id);
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{frame}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], id);
        response
    }

    fn finish(&mut self) {
        self.input.take();
        assert!(self.child.wait().unwrap().success());
    }
}

impl Drop for Cli {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn fresh_cli_returns_exact_complete_sets_and_remains_usable_after_refusal() {
    let mut cli = Cli::new();
    assert_eq!(cli.send(equation(), "missing")["code"], "unknown_object");
    assert_eq!(
        cli.send(registration(&["(x^2-2)/(x^2-3)"], &["x"]), "register")["status"],
        "success"
    );
    let equation = cli.send(equation(), "roots");
    assert_eq!(equation["status"], "success", "{equation}");
    assert_eq!(equation["result"]["kind"], "real_equation_solved");
    assert_eq!(equation["result"]["completeness"], "complete");
    assert_eq!(equation["result"]["domain"], "real");
    assert_eq!(
        equation["result"]["solution_set"]["polynomial"],
        json!(["-2", "0", "1"])
    );
    assert_eq!(
        equation["result"]["solution_set"]["point_inclusions"],
        json!([true, true])
    );
    let inequality = cli.send(inequality(RealInequalityRelation::Ge), "inequality");
    assert_eq!(inequality["status"], "success", "{inequality}");
    assert_eq!(inequality["result"]["kind"], "real_inequality_solved");
    assert_eq!(inequality["result"]["relation"], "ge");
    assert_eq!(inequality["result"]["evidence"], "exact_cross_checked");
    assert_eq!(
        inequality["result"]["solution_set"]["gap_inclusions"],
        json!([true, false, true, false, true])
    );
    assert_eq!(
        inequality["result"]["solution_set"]["point_inclusions"],
        json!([false, true, true, false])
    );
    cli.finish();
}
