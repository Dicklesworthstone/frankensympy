//! Matrix-free products over frozen, independently checked session programs.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, CompiledResidualSystemId, Dimension,
    ResidualSystemId, Session,
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
    ResidualSystemId::new("system:products").unwrap()
}

fn program() -> CompiledResidualSystemId {
    CompiledResidualSystemId::new("program:products").unwrap()
}

fn registration(residuals: &[&str], variables: &[&str]) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: residuals.iter().map(|value| (*value).into()).collect(),
        variables: variables.iter().map(|value| (*value).into()).collect(),
    }
}

fn compile() -> CertificationRequest {
    CertificationRequest::CompileResidualSystem {
        schema_version: 1,
        system: system(),
        program: program(),
    }
}

fn product(transpose: bool, point: &[&str], vector: &[&str]) -> CertificationRequest {
    let point = point.iter().map(|value| (*value).into()).collect();
    let vector = vector.iter().map(|value| (*value).into()).collect();
    if transpose {
        CertificationRequest::EvaluateJacobianTransposeVectorProduct {
            schema_version: 1,
            program: program(),
            point,
            weights: vector,
        }
    } else {
        CertificationRequest::EvaluateJacobianVectorProduct {
            schema_version: 1,
            program: program(),
            point,
            direction: vector,
        }
    }
}

fn prepare(session: &mut Session, residuals: &[&str], variables: &[&str]) {
    session
        .certification_request(registration(residuals, variables))
        .unwrap();
    session.certification_request(compile()).unwrap();
}

fn wire(session: &mut Session, request: CertificationRequest) -> Value {
    serde_json::from_str(&session.handle_envelope(&serde_json::to_string(&request).unwrap()))
        .unwrap()
}

#[test]
fn rectangular_products_match_independent_derivatives_and_the_adjoint_identity() {
    let mut session = session();
    prepare(&mut session, &["x^2+y", "sin(y)", "7"], &["y", "x"]);
    // The variable order is intentionally different from lexical order:
    // J(0, 2) = [[1, 4], [1, 0], [0, 0]].
    let direction = [3.0, -2.0];
    let weights = [2.0, -3.0, 11.0];
    let before = session.budget_snapshot();
    let response = session
        .certification_request(product(false, &["0", "2"], &["3", "-2"]))
        .unwrap();
    let CertificationResponse::JacobianVectorProductEvaluated {
        system: source,
        program: handle,
        outcome,
        numeric_model,
        point,
        direction: admitted_direction,
        product: forward,
        context_digest,
        ..
    } = response
    else {
        panic!("forward product response");
    };
    assert_eq!(source, system());
    assert_eq!(handle, program());
    assert_eq!((outcome, numeric_model), ("approximate", "binary64"));
    assert_eq!(point, [0.0, 2.0]);
    assert_eq!(admitted_direction, direction);
    assert_eq!(forward, [-5.0, 3.0, 0.0]);

    // Workspace rebinding must not mutate a resident problem or its axes.
    session.construct("x", "1000").unwrap();
    let response = session
        .certification_request(product(true, &["0", "2"], &["2", "-3", "11"]))
        .unwrap();
    let CertificationResponse::JacobianTransposeVectorProductEvaluated {
        weights: admitted_weights,
        product: transpose,
        context_digest: second_context,
        outcome,
        numeric_model,
        ..
    } = response
    else {
        panic!("transpose product response");
    };
    assert_eq!(admitted_weights, weights);
    assert_eq!(transpose, [-1.0, 8.0]);
    assert_eq!(second_context, context_digest);
    assert_eq!((outcome, numeric_model), ("approximate", "binary64"));
    let left: f64 = forward.iter().zip(weights).map(|(a, b)| a * b).sum();
    let right: f64 = transpose.iter().zip(direction).map(|(a, b)| a * b).sum();
    assert_eq!(left, -19.0);
    assert_eq!(left, right);
    let after = session.budget_snapshot();
    assert_eq!(after.verifier_pool, before.verifier_pool);
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );

    // Reuse at a different point cannot retain a previous numerical memo.
    let response = wire(&mut session, product(false, &["0", "3"], &["3", "-2"]));
    assert_eq!(response["result"]["product"], json!([-9.0, 3.0, 0.0]));
    let response = wire(&mut session, product(true, &["0", "3"], &["2", "-3", "11"]));
    assert_eq!(response["result"]["product"], json!([-1.0, 12.0]));
    assert!(response["result"].get("jacobian").is_none());
    assert!(response["result"].get("jacobian_pattern").is_none());
    assert!(response["result"].get("evidence").is_none());
}

#[test]
fn zero_vectors_cannot_erase_original_or_derivative_domain_failures() {
    for source in ["x/x", "0*log(x)", "x^(1/2)"] {
        let mut session = session();
        prepare(&mut session, &[source], &["x"]);
        for transpose in [false, true] {
            let response = wire(&mut session, product(transpose, &["0"], &["0"]));
            assert_eq!(
                response["code"], "numeric_evaluation_refused",
                "{source}: {response}"
            );
            assert!(response.get("result").is_none());
            let response = wire(&mut session, product(transpose, &["1"], &["0"]));
            assert_eq!(response["status"], "success", "{source}: {response}");
            assert_eq!(response["result"]["product"], json!([0.0]));
        }
        assert!(session.compiled_residual_system(&program()).is_some());
    }
}

#[test]
fn finite_inputs_cannot_publish_overflowed_products_or_reductions() {
    for transpose in [false, true] {
        let mut session = session();
        prepare(&mut session, &["x^2"], &["x"]);
        // Source and derivative are finite; the product is not.
        let response = wire(&mut session, product(transpose, &["1e150"], &["1e200"]));
        assert_eq!(response["code"], "numeric_evaluation_refused");
        assert!(response.get("result").is_none());
    }
    let mut session = session();
    prepare(&mut session, &["x", "x"], &["x"]);
    // Both products are finite; their transpose accumulation overflows.
    let response = wire(&mut session, product(true, &["1"], &["1e308", "1e308"]));
    assert_eq!(response["code"], "numeric_evaluation_refused");
    assert!(response.get("result").is_none());
    assert!(session.compiled_residual_system(&program()).is_some());
}

#[test]
fn both_input_spaces_have_independent_dimension_and_finite_scalar_admission() {
    let mut session = session();
    prepare(&mut session, &["x+y", "x-y", "7"], &["x", "y"]);
    for request in [
        product(false, &["1"], &["1", "2"]),
        product(false, &["1", "2"], &["1"]),
        product(false, &["1", "2"], &["1", "2", "3"]),
        product(true, &["1", "2", "3"], &["1", "2", "3"]),
        product(true, &["1", "2"], &["1", "2"]),
        product(true, &["1", "2"], &["1", "2", "3", "4"]),
    ] {
        assert_eq!(
            session.certification_request(request).unwrap_err().code(),
            "invalid_input"
        );
    }
    let oversized = "1".repeat(513);
    for invalid in ["NaN", "inf", "-inf", "1e999", "1/2", " 1", "", &oversized] {
        for request in [
            product(false, &[invalid, "2"], &["1", "2"]),
            product(false, &["1", "2"], &[invalid, "2"]),
            product(true, &["1", "2"], &["1", invalid, "3"]),
        ] {
            let response = wire(&mut session, request);
            assert_eq!(response["code"], "invalid_input", "{invalid}: {response}");
            assert!(response.get("result").is_none());
        }
    }
}

#[test]
fn output_admission_exhaustion_retains_programs_and_cumulative_work() {
    for transpose in [false, true] {
        let request = product(transpose, &["2"], &["3"]);
        let mut probe = session();
        prepare(&mut probe, &["x^2"], &["x"]);
        probe.certification_request(request.clone()).unwrap();
        let remaining = probe.budget_snapshot();

        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut specification = limits();
            let spent = specification.dimensions[dimension.index()]
                - remaining.dimensions[dimension.index()];
            specification.dimensions[dimension.index()] = spent - 1;
            let mut bounded = Session::with_budget_limits(specification, 256);
            prepare(&mut bounded, &["x^2"], &["x"]);
            let before = bounded.budget_snapshot();
            assert_eq!(
                bounded
                    .certification_request(request.clone())
                    .unwrap_err()
                    .code(),
                "budget_exhausted",
                "{dimension:?}"
            );
            assert!(bounded.compiled_residual_system(&program()).is_some());
            let after = bounded.budget_snapshot();
            assert_eq!(after.verifier_pool, before.verifier_pool);
            assert!(after.dimensions[dimension.index()] < before.dimensions[dimension.index()]);
            // The remaining ledger is not replaced by a new per-call budget.
            assert_eq!(
                bounded
                    .certification_request(request.clone())
                    .unwrap_err()
                    .code(),
                "budget_exhausted"
            );
        }
    }
}

#[test]
fn wire_schema_and_handle_lifetime_fail_closed_for_both_product_directions() {
    let mut session = session();
    for transpose in [false, true] {
        let response = wire(&mut session, product(transpose, &["1"], &["1"]));
        assert_eq!(response["code"], "unknown_object");
    }
    prepare(&mut session, &["x"], &["x"]);
    for transpose in [false, true] {
        let mut request = serde_json::to_value(product(transpose, &["1"], &["1"])).unwrap();
        request["payload"]["schema_version"] = json!(2);
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
        assert_eq!(response["code"], "unsupported_schema");
        request["payload"]["schema_version"] = json!(1);
        request["payload"]["accepted"] = json!(true);
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&request.to_string())).unwrap();
        assert_eq!(response["code"], "malformed_request");
    }
    let raw = r#"{"type":"EvaluateJacobianVectorProduct","payload":{"schema_version":1,"program":"program:products","point":["1"],"direction":["1"],"direction":["2"]}}"#;
    let response: Value = serde_json::from_str(&session.handle_envelope(raw)).unwrap();
    assert_eq!(response["code"], "malformed_request");
    session
        .certification_request(CertificationRequest::ReleaseCompiledResidualSystem {
            schema_version: 1,
            program: program(),
        })
        .unwrap();
    for transpose in [false, true] {
        let response = wire(&mut session, product(transpose, &["1"], &["1"]));
        assert_eq!(response["code"], "unknown_object");
    }
}

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
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
fn fresh_cli_supports_both_products_without_dense_or_certified_numeric_payloads() {
    let mut cli = Cli::new();
    assert_eq!(
        cli.send(
            registration(&["x^2+y", "sin(y)", "7"], &["y", "x"]),
            "register"
        )["status"],
        "success"
    );
    assert_eq!(cli.send(compile(), "compile")["status"], "success");
    for (transpose, vector, expected, kind) in [
        (
            false,
            vec!["3", "-2"],
            json!([-5.0, 3.0, 0.0]),
            "jacobian_vector_product_evaluated",
        ),
        (
            true,
            vec!["2", "-3", "11"],
            json!([-1.0, 8.0]),
            "jacobian_transpose_vector_product_evaluated",
        ),
    ] {
        let response = cli.send(product(transpose, &["0", "2"], &vector), kind);
        assert_eq!(response["status"], "success", "{response}");
        assert_eq!(response["result"]["kind"], kind);
        assert_eq!(response["result"]["product"], expected);
        assert_eq!(response["result"]["outcome"], "approximate");
        assert_eq!(response["result"]["numeric_model"], "binary64");
        assert!(response["result"].get("jacobian").is_none());
        assert!(response["result"].get("evidence").is_none());
    }
    cli.finish();

    let mut fresh = Cli::new();
    for transpose in [false, true] {
        assert_eq!(
            fresh.send(product(transpose, &["0", "2"], &["1", "2"]), "local")["code"],
            "unknown_object"
        );
    }
    fresh.finish();
}
