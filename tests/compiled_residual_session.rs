//! Typed native and fresh-process numerical consumers of checked sparse programs.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, CompiledResidualSystemId, Dimension,
    MAX_SESSION_COMPILED_SYSTEMS, ResidualSystemId, Session,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(50_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits
}

fn session() -> Session {
    Session::with_budget_limits(limits(), 256)
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:residuals").unwrap()
}

fn program(name: &str) -> CompiledResidualSystemId {
    CompiledResidualSystemId::new(format!("program:{name}")).unwrap()
}

fn registration(residuals: &[&str], variables: &[&str]) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: residuals.iter().map(|source| (*source).into()).collect(),
        variables: variables
            .iter()
            .map(|variable| (*variable).into())
            .collect(),
    }
}

fn compile(name: &str) -> CertificationRequest {
    CertificationRequest::CompileResidualSystem {
        schema_version: 1,
        system: system(),
        program: program(name),
    }
}

fn evaluate(name: &str, point: &[&str]) -> CertificationRequest {
    CertificationRequest::EvaluateResidualJacobian {
        schema_version: 1,
        program: program(name),
        point: point.iter().map(|value| (*value).into()).collect(),
    }
}

fn release(name: &str) -> CertificationRequest {
    CertificationRequest::ReleaseCompiledResidualSystem {
        schema_version: 1,
        program: program(name),
    }
}

#[test]
fn compiled_program_preserves_frozen_source_order_and_honest_numeric_outcomes() {
    let mut session = session();
    session
        .certification_request(registration(&["x^2+y", "sin(y)", "7"], &["y", "x"]))
        .unwrap();
    session.construct("x", "100").unwrap();
    let before = session.budget_snapshot();
    let response = session.certification_request(compile("checked")).unwrap();
    let CertificationResponse::ResidualSystemCompiled {
        system: source,
        program: handle,
        evidence,
        claim,
        variables,
        residual_count,
        jacobian_pattern,
        proofs_verified,
        stored_operations,
        context_digest,
        ..
    } = response
    else {
        panic!("compiled program response")
    };
    assert_eq!(source, system());
    assert_eq!(handle, program("checked"));
    assert_eq!(evidence, "certificate_verified");
    assert_eq!(claim, "formal_symbolic_jacobian_and_binary64_lowering");
    assert_eq!(
        variables
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>(),
        ["y", "x"]
    );
    assert_eq!(residual_count, 3);
    assert_eq!(jacobian_pattern, [(0, 0), (0, 1), (1, 0)]);
    assert_eq!(proofs_verified, 3);
    assert!(stored_operations > 0);
    let compiled = session.compiled_residual_system(&handle).unwrap();
    assert_eq!(
        compiled.residuals(),
        session.residual_problem(&system()).unwrap().residuals()
    );
    let compiled_operations = compiled.stored_operation_count();
    let after = session.budget_snapshot();
    assert!(after.verifier_pool < before.verifier_pool);
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );

    session.construct("x", "-999").unwrap();
    for _ in 0..2 {
        let before = session.budget_snapshot();
        let response = session
            .certification_request(evaluate("checked", &["0e0", "2.0"]))
            .unwrap();
        let CertificationResponse::ResidualJacobianEvaluated {
            outcome,
            numeric_model,
            point,
            residuals,
            jacobian_pattern,
            jacobian,
            context_digest: evaluated_context,
            ..
        } = response
        else {
            panic!("sparse numerical response")
        };
        assert_eq!((outcome, numeric_model), ("approximate", "binary64"));
        assert_eq!(point, [0.0, 2.0]);
        assert_eq!(residuals, [4.0, 0.0, 7.0]);
        assert_eq!(jacobian_pattern, [(0, 0), (0, 1), (1, 0)]);
        assert_eq!(jacobian, [1.0, 4.0, 1.0]);
        assert_eq!(evaluated_context, context_digest);
        let after = session.budget_snapshot();
        assert_eq!(after.verifier_pool, before.verifier_pool);
        assert!(
            after.dimensions[Dimension::ComputeSteps.index()]
                < before.dimensions[Dimension::ComputeSteps.index()]
        );
        assert_eq!(
            session
                .compiled_residual_system(&handle)
                .unwrap()
                .stored_operation_count(),
            compiled_operations
        );
    }
}

#[test]
fn inputs_domains_and_nonfinite_results_fail_without_partial_numeric_output() {
    for (source, invalid_point, valid_point) in [
        ("x/x", "0", "2"),
        ("0*log(x)", "-1", "1"),
        ("x^2", "1e308", "2"),
    ] {
        let mut session = session();
        session
            .certification_request(registration(&[source], &["x"]))
            .unwrap();
        session.certification_request(compile("guarded")).unwrap();
        assert_eq!(
            session
                .certification_request(evaluate("guarded", &[invalid_point]))
                .unwrap_err()
                .code(),
            "numeric_evaluation_refused",
            "{source}"
        );
        let response = session
            .certification_request(evaluate("guarded", &[valid_point]))
            .unwrap();
        assert!(matches!(
            response,
            CertificationResponse::ResidualJacobianEvaluated { .. }
        ));
        assert!(
            session
                .compiled_residual_system(&program("guarded"))
                .is_some()
        );
    }

    let mut session = session();
    session
        .certification_request(registration(&["x"], &["x"]))
        .unwrap();
    session.certification_request(compile("finite")).unwrap();
    for point in [
        vec![],
        vec!["1", "2"],
        vec!["NaN"],
        vec!["inf"],
        vec!["-inf"],
        vec!["1e999"],
        vec!["1/2"],
        vec![" 1"],
        vec![""],
    ] {
        assert_eq!(
            session
                .certification_request(evaluate("finite", &point))
                .unwrap_err()
                .code(),
            "invalid_input",
            "{point:?}"
        );
    }
    let oversized = "1".repeat(513);
    assert_eq!(
        session
            .certification_request(evaluate("finite", &[&oversized]))
            .unwrap_err()
            .code(),
        "invalid_input"
    );
    let response = session
        .certification_request(evaluate("finite", &["-0"]))
        .unwrap();
    let CertificationResponse::ResidualJacobianEvaluated { point, .. } = response else {
        panic!("finite response")
    };
    assert!(point[0].is_sign_negative());
}

#[test]
fn each_exhausted_resource_retains_charges_and_prevents_handle_publication() {
    let mut probe = session();
    probe
        .certification_request(registration(&["x^2"], &["x"]))
        .unwrap();
    probe.certification_request(compile("bounded")).unwrap();
    let remaining = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut specification = limits();
        let spent =
            specification.dimensions[dimension.index()] - remaining.dimensions[dimension.index()];
        specification.dimensions[dimension.index()] = spent - 1;
        let mut session = Session::with_budget_limits(specification, 256);
        session
            .certification_request(registration(&["x^2"], &["x"]))
            .unwrap();
        let before = session.budget_snapshot();
        let error = session
            .certification_request(compile("bounded"))
            .unwrap_err();
        assert_eq!(error.code(), "budget_exhausted", "{dimension:?}: {error}");
        assert!(
            session
                .compiled_residual_system(&program("bounded"))
                .is_none()
        );
        assert!(
            session.budget_snapshot().dimensions[dimension.index()]
                < before.dimensions[dimension.index()]
        );
    }
    for protected_steps in [0, 1] {
        let mut specification = limits();
        specification.verifier_pool = protected_steps;
        let mut session = Session::with_budget_limits(specification, 256);
        session
            .certification_request(registration(&["x^2"], &["x"]))
            .unwrap();
        let before = session.budget_snapshot();
        assert_eq!(
            session
                .certification_request(compile("bounded"))
                .unwrap_err()
                .code(),
            "budget_exhausted"
        );
        assert!(
            session
                .compiled_residual_system(&program("bounded"))
                .is_none()
        );
        let after = session.budget_snapshot();
        if protected_steps == 0 {
            // Missing authority is rejected before any candidate construction.
            assert_eq!(after, before);
        } else {
            assert!(
                after.dimensions[Dimension::ComputeSteps.index()]
                    < before.dimensions[Dimension::ComputeSteps.index()]
            );
            assert!(after.verifier_pool < before.verifier_pool);
        }
    }
}

#[test]
fn bounded_handles_reject_duplicates_release_without_refunds_and_reuse_capacity() {
    let mut session = session();
    assert_eq!(
        session
            .certification_request(compile("missing"))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    session
        .certification_request(registration(&["x"], &["x"]))
        .unwrap();
    for index in 0..MAX_SESSION_COMPILED_SYSTEMS {
        session
            .certification_request(compile(&format!("p{index}")))
            .unwrap();
    }
    assert_eq!(
        session
            .certification_request(compile("p0"))
            .unwrap_err()
            .code(),
        "duplicate_object"
    );
    assert_eq!(
        session
            .certification_request(compile("extra"))
            .unwrap_err()
            .code(),
        "resource_limit_exceeded"
    );
    let before = session.budget_snapshot();
    session.certification_request(release("p0")).unwrap();
    let after = session.budget_snapshot();
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    assert_eq!(after.verifier_pool, before.verifier_pool);
    assert!(session.compiled_residual_system(&program("p0")).is_none());
    assert_eq!(
        session
            .certification_request(evaluate("p0", &["2"]))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    assert_eq!(
        session
            .certification_request(release("p0"))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    session.certification_request(compile("extra")).unwrap();
    assert!(
        session
            .compiled_residual_system(&program("extra"))
            .is_some()
    );
}

#[test]
fn failed_release_admission_retains_the_program_and_unsupported_sources_publish_none() {
    let mut probe = session();
    probe
        .certification_request(registration(&["x"], &["x"]))
        .unwrap();
    probe.certification_request(compile("retain")).unwrap();
    probe.certification_request(release("retain")).unwrap();
    let mut specification = limits();
    let dimension = Dimension::ComputeSteps.index();
    specification.dimensions[dimension] -= probe.budget_snapshot().dimensions[dimension] + 1;
    let mut bounded = Session::with_budget_limits(specification, 256);
    bounded
        .certification_request(registration(&["x"], &["x"]))
        .unwrap();
    bounded.certification_request(compile("retain")).unwrap();
    assert_eq!(
        bounded
            .certification_request(release("retain"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(
        bounded
            .compiled_residual_system(&program("retain"))
            .is_some()
    );

    let mut unsupported = session();
    unsupported
        .certification_request(registration(&["unknown(x)"], &["x"]))
        .unwrap();
    assert_eq!(
        unsupported
            .certification_request(compile("unsupported"))
            .unwrap_err()
            .code(),
        "unsupported_operation"
    );
    assert!(
        unsupported
            .compiled_residual_system(&program("unsupported"))
            .is_none()
    );
}

#[test]
fn strict_wire_validation_does_not_publish_unknown_schemas_fields_or_duplicate_keys() {
    let mut session = session();
    session
        .certification_request(registration(&["x"], &["x"]))
        .unwrap();
    let mut request = serde_json::to_value(compile("wire")).unwrap();
    request["payload"]["schema_version"] = json!(2);
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
    assert_eq!(response["code"], "unsupported_schema");
    request["payload"]["schema_version"] = json!(1);
    request["payload"]["accepted"] = json!(true);
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
    assert_eq!(response["code"], "malformed_request");
    for raw in [
        r#"{"type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:residuals","program":"program:wire","program":"program:wire"}}"#,
        r#"{"type":"CompileResidualSystem","payload":{"schema_version":1,"system":"system:residuals","program":"system:wire"}}"#,
    ] {
        let response: Value = serde_json::from_str(&session.handle_envelope(raw)).unwrap();
        assert_eq!(response["code"], "malformed_request");
    }
    assert!(session.compiled_residual_system(&program("wire")).is_none());
    for invalid in ["program:", "system:x", "program:space here"] {
        assert!(CompiledResidualSystemId::new(invalid).is_err());
    }
    assert!(CompiledResidualSystemId::new(format!("program:{}", "a".repeat(128))).is_err());
}

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new(arguments: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args(arguments)
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
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "one complete JSON response per request"
        );
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
fn fresh_cli_compiles_evaluates_releases_and_keeps_approximation_explicit() {
    let mut cli = Cli::new(&[]);
    assert_eq!(
        cli.send(registration(&["x^2+y", "sin(y)"], &["y", "x"]), "register")["status"],
        "success"
    );
    let compiled = cli.send(compile("cli"), "compile");
    assert_eq!(compiled["status"], "success", "{compiled}");
    assert_eq!(compiled["result"]["kind"], "residual_system_compiled");
    let evaluated = cli.send(evaluate("cli", &["0", "2"]), "evaluate");
    assert_eq!(evaluated["status"], "success", "{evaluated}");
    assert_eq!(evaluated["result"]["kind"], "residual_jacobian_evaluated");
    assert_eq!(evaluated["result"]["outcome"], "approximate");
    assert_eq!(evaluated["result"]["numeric_model"], "binary64");
    assert_eq!(evaluated["result"]["residuals"], json!([4.0, 0.0]));
    assert_eq!(
        evaluated["result"]["jacobian_pattern"],
        json!([[0, 0], [0, 1], [1, 0]])
    );
    assert_eq!(evaluated["result"]["jacobian"], json!([1.0, 4.0, 1.0]));
    assert!(evaluated["result"].get("evidence").is_none());
    assert_eq!(cli.send(release("cli"), "release")["status"], "success");
    assert_eq!(
        cli.send(evaluate("cli", &["0", "2"]), "unknown")["code"],
        "unknown_object"
    );
    cli.finish();

    let mut fresh = Cli::new(&[]);
    assert_eq!(
        fresh.send(evaluate("cli", &["0", "2"]), "session-local")["code"],
        "unknown_object"
    );
    fresh.finish();
}

#[test]
fn fresh_cli_budget_and_source_domain_failures_publish_no_successful_numeric_value() {
    let mut cli = Cli::new(&["--verifier-steps", "0"]);
    assert_eq!(
        cli.send(registration(&["x"], &["x"]), "register")["status"],
        "success"
    );
    assert_eq!(
        cli.send(compile("refused"), "compile")["code"],
        "budget_exhausted"
    );
    assert_eq!(
        cli.send(evaluate("refused", &["1"]), "unpublished")["code"],
        "unknown_object"
    );
    cli.finish();

    let mut cli = Cli::new(&[]);
    assert_eq!(
        cli.send(registration(&["0*log(x)"], &["x"]), "register")["status"],
        "success"
    );
    let compiled = cli.send(compile("domain"), "compile");
    assert_eq!(compiled["status"], "success", "{compiled}");
    let refused = cli.send(evaluate("domain", &["-1"]), "domain");
    assert_eq!(refused["code"], "numeric_evaluation_refused");
    assert!(refused.get("result").is_none());
    assert_eq!(
        cli.send(evaluate("domain", &["1"]), "valid")["status"],
        "success"
    );
    cli.finish();
}
