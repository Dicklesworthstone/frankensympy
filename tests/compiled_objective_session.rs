//! Native and fresh-process consumers of resident checked objective programs.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, CompiledObjectiveId, Dimension,
    MAX_SESSION_COMPILED_OBJECTIVES, ObjectiveId, Session,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const AMPLE: u64 = 1 << 40;

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn session() -> Session {
    Session::with_budget_limits(limits(), 256)
}

fn objective() -> ObjectiveId {
    ObjectiveId::new("objective:quadratic").unwrap()
}

fn program(name: &str) -> CompiledObjectiveId {
    CompiledObjectiveId::new(format!("objective_program:{name}")).unwrap()
}

fn registration(source: &str, variables: &[&str]) -> CertificationRequest {
    CertificationRequest::RegisterObjective {
        schema_version: 1,
        objective: objective(),
        source: source.into(),
        variables: variables
            .iter()
            .map(|variable| (*variable).into())
            .collect(),
    }
}

fn compile(name: &str) -> CertificationRequest {
    CertificationRequest::CompileObjective {
        schema_version: 1,
        objective: objective(),
        program: program(name),
    }
}

fn evaluate(name: &str, point: &[&str]) -> CertificationRequest {
    CertificationRequest::EvaluateObjective {
        schema_version: 1,
        program: program(name),
        point: point.iter().map(|value| (*value).into()).collect(),
    }
}

fn product(name: &str, point: &[&str], direction: &[&str]) -> CertificationRequest {
    CertificationRequest::EvaluateHessianVectorProduct {
        schema_version: 1,
        program: program(name),
        point: point.iter().map(|value| (*value).into()).collect(),
        direction: direction.iter().map(|value| (*value).into()).collect(),
    }
}

fn release(name: &str) -> CertificationRequest {
    CertificationRequest::ReleaseCompiledObjective {
        schema_version: 1,
        program: program(name),
    }
}

fn wire(session: &mut Session, request: CertificationRequest) -> Value {
    let request = serde_json::to_string(&request).unwrap();
    serde_json::from_str(&session.handle_envelope(&request)).unwrap()
}

fn assert_approximate(value: &Value) {
    assert_eq!(value["outcome"], "approximate");
    assert_eq!(value["numeric_model"], "binary64");
    assert!(value.get("evidence").is_none());
    assert!(value.get("claim").is_none());
}

#[test]
fn frozen_objective_compiles_once_and_returns_hand_derived_values_and_sparse_products() {
    let mut session = session();
    session
        .certification_request(registration("x^2+x*y+3*y^2", &["x", "y"]))
        .unwrap();
    session.construct("x", "100").unwrap();
    let before = session.budget_snapshot();
    let response = session.certification_request(compile("checked")).unwrap();
    let CertificationResponse::ObjectiveCompiled {
        objective: source,
        program: handle,
        evidence,
        claim,
        variables,
        hessian_pattern,
        gradient_proofs_verified,
        hessian_proofs_verified,
        zero_entries_checked,
        context_digest,
        ..
    } = response
    else {
        panic!("compiled objective response")
    };
    assert_eq!(source, objective());
    assert_eq!(handle, program("checked"));
    assert_eq!(evidence, "certificate_verified");
    assert_eq!(
        claim,
        "formal_symbolic_gradient_hessian_and_binary64_lowering"
    );
    assert_eq!(
        variables
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect::<Vec<_>>(),
        ["x", "y"]
    );
    assert_eq!(hessian_pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(gradient_proofs_verified, 2);
    assert_eq!(hessian_proofs_verified, 4);
    assert_eq!(zero_entries_checked, 0);
    assert_eq!(
        session.compiled_objective(&handle).unwrap().objective(),
        session.objective_problem(&objective()).unwrap().objective()
    );
    let after = session.budget_snapshot();
    assert!(after.verifier_pool < before.verifier_pool);
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );

    session.construct("x", "-999").unwrap();
    for (point, value, gradient) in [
        (["2e0", "-1.0"], 5.0, [3.0, -4.0]),
        (["-1", "2"], 11.0, [0.0, 11.0]),
        (["2e0", "-1.0"], 5.0, [3.0, -4.0]),
    ] {
        let before = session.budget_snapshot();
        let response = session
            .certification_request(evaluate("checked", &point))
            .unwrap();
        let json = serde_json::to_value(&response).unwrap();
        assert_approximate(&json);
        let CertificationResponse::ObjectiveEvaluated {
            objective: evaluated_source,
            program: evaluated_program,
            value: actual,
            gradient: actual_gradient,
            hessian_pattern,
            hessian,
            context_digest: evaluated_context,
            ..
        } = response
        else {
            panic!("objective evaluation response")
        };
        assert_eq!(evaluated_source, objective());
        assert_eq!(evaluated_program, handle);
        assert_eq!(actual, value);
        assert_eq!(actual_gradient, gradient);
        assert_eq!(hessian_pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(hessian, [2.0, 1.0, 1.0, 6.0]);
        assert_eq!(evaluated_context, context_digest);
        let after = session.budget_snapshot();
        assert_eq!(after.verifier_pool, before.verifier_pool);
        assert!(
            after.dimensions[Dimension::ComputeSteps.index()]
                < before.dimensions[Dimension::ComputeSteps.index()]
        );
    }

    let before = session.budget_snapshot();
    let response = session
        .certification_request(product("checked", &["2", "-1"], &["3", "-2"]))
        .unwrap();
    assert_approximate(&serde_json::to_value(&response).unwrap());
    let CertificationResponse::ObjectiveHessianVectorProduct {
        objective: evaluated_source,
        program: evaluated_program,
        point,
        direction,
        product,
        context_digest: evaluated_context,
        ..
    } = response
    else {
        panic!("Hessian-vector response")
    };
    assert_eq!(evaluated_source, objective());
    assert_eq!(evaluated_program, handle);
    assert_eq!(point, [2.0, -1.0]);
    assert_eq!(direction, [3.0, -2.0]);
    assert_eq!(product, [4.0, -9.0]);
    assert_eq!(evaluated_context, context_digest);
    assert_eq!(
        session.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
}

#[test]
fn original_and_second_derivative_domains_refuse_even_zero_directions_without_poisoning_handles() {
    for (source, invalid_point, valid_point) in [
        ("x/x", "0", "2"),
        ("0*log(x)", "-1", "1"),
        ("x^(3/2)", "0", "1"),
        ("x^2", "1e308", "2"),
    ] {
        let mut session = session();
        session
            .certification_request(registration(source, &["x"]))
            .unwrap();
        session.certification_request(compile("domain")).unwrap();
        for request in [
            evaluate("domain", &[invalid_point]),
            product("domain", &[invalid_point], &["0"]),
        ] {
            let response = wire(&mut session, request);
            assert_eq!(
                response["code"], "numeric_evaluation_refused",
                "{source}: {response}"
            );
            assert!(response.get("result").is_none());
        }
        assert!(
            session
                .certification_request(evaluate("domain", &[valid_point]))
                .is_ok(),
            "{source}"
        );
        assert!(
            session
                .certification_request(product("domain", &[valid_point], &["0"]))
                .is_ok(),
            "{source}"
        );
        assert!(session.compiled_objective(&program("domain")).is_some());
    }
}

#[test]
fn point_and_direction_validation_retains_finite_decimal_and_dimension_contracts() {
    let mut session = session();
    session
        .certification_request(registration("x^2", &["x"]))
        .unwrap();
    session.certification_request(compile("finite")).unwrap();
    for values in [
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
        for request in [
            evaluate("finite", &values),
            product("finite", &values, &["0"]),
            product("finite", &["1"], &values),
        ] {
            assert_eq!(
                session.certification_request(request).unwrap_err().code(),
                "invalid_input",
                "{values:?}"
            );
        }
    }
    let oversized = "1".repeat(513);
    for request in [
        evaluate("finite", &[&oversized]),
        product("finite", &["1"], &[&oversized]),
    ] {
        assert_eq!(
            session.certification_request(request).unwrap_err().code(),
            "invalid_input"
        );
    }
    assert_eq!(
        session
            .certification_request(product("finite", &["1"], &["1e308"]))
            .unwrap_err()
            .code(),
        "numeric_evaluation_refused"
    );
    let response = session
        .certification_request(evaluate("finite", &["-0"]))
        .unwrap();
    let CertificationResponse::ObjectiveEvaluated { point, value, .. } = response else {
        panic!("finite evaluation")
    };
    assert!(point[0].is_sign_negative());
    assert_eq!(value, 0.0);
    assert!(session.compiled_objective(&program("finite")).is_some());
}

#[test]
fn compilation_cannot_borrow_missing_or_exhausted_verifier_authority() {
    for protected_steps in [0, 1] {
        let mut specification = limits();
        specification.verifier_pool = protected_steps;
        let mut session = Session::with_budget_limits(specification, 256);
        session
            .certification_request(registration("x^2", &["x"]))
            .unwrap();
        let before = session.budget_snapshot();
        assert_eq!(
            session
                .certification_request(compile("refused"))
                .unwrap_err()
                .code(),
            "budget_exhausted"
        );
        assert!(session.compiled_objective(&program("refused")).is_none());
        let after = session.budget_snapshot();
        if protected_steps == 0 {
            assert_eq!(after, before, "missing authority precedes candidate work");
        } else {
            assert_eq!(after.verifier_pool, 0);
            assert!(
                after.dimensions[Dimension::ComputeSteps.index()]
                    < before.dimensions[Dimension::ComputeSteps.index()]
            );
        }
    }
}

#[test]
fn every_final_publication_resource_can_refuse_a_fully_checked_objective_handle() {
    let mut probe = session();
    probe
        .certification_request(registration("x^2", &["x"]))
        .unwrap();
    probe.certification_request(compile("publication")).unwrap();
    let accepted = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut specification = limits();
        let index = dimension.index();
        let spent = specification.dimensions[index] - accepted.dimensions[index];
        specification.dimensions[index] = spent - 1;
        let mut session = Session::with_budget_limits(specification, 256);
        session
            .certification_request(registration("x^2", &["x"]))
            .unwrap();
        let before = session.budget_snapshot();
        let error = session
            .certification_request(compile("publication"))
            .unwrap_err();
        assert_eq!(error.code(), "budget_exhausted", "{dimension:?}: {error}");
        assert!(
            session
                .compiled_objective(&program("publication"))
                .is_none()
        );
        let after = session.budget_snapshot();
        assert_eq!(
            after.verifier_pool, accepted.verifier_pool,
            "independent verification completed before output admission"
        );
        assert!(after.dimensions[index] < before.dimensions[index]);
    }
}

#[test]
fn bounded_handles_are_session_local_and_release_reuses_capacity_without_refunds() {
    let mut session = session();
    for request in [
        compile("missing"),
        evaluate("missing", &["1"]),
        product("missing", &["1"], &["1"]),
        release("missing"),
    ] {
        assert_eq!(
            session.certification_request(request).unwrap_err().code(),
            "unknown_object"
        );
    }
    session
        .certification_request(registration("x", &["x"]))
        .unwrap();
    for index in 0..MAX_SESSION_COMPILED_OBJECTIVES {
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
    let released = session.certification_request(release("p0")).unwrap();
    assert!(matches!(
        released,
        CertificationResponse::CompiledObjectiveReleased { program: id, .. }
            if id == program("p0")
    ));
    let after = session.budget_snapshot();
    for dimension in Dimension::ALL {
        assert!(after.dimensions[dimension.index()] <= before.dimensions[dimension.index()]);
    }
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    assert_eq!(after.verifier_pool, before.verifier_pool);
    assert!(session.compiled_objective(&program("p0")).is_none());
    assert_eq!(
        session
            .certification_request(release("p0"))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    assert_eq!(
        session
            .certification_request(product("p0", &["2"], &["1"]))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    session.certification_request(compile("p0")).unwrap();
    let evaluated = session
        .certification_request(evaluate("p0", &["-2"]))
        .unwrap();
    let CertificationResponse::ObjectiveEvaluated {
        value,
        gradient,
        hessian,
        ..
    } = evaluated
    else {
        panic!("reused capacity")
    };
    assert_eq!(value, -2.0);
    assert_eq!(gradient, [1.0]);
    assert!(hessian.is_empty());

    let mut fresh = Session::with_budget_limits(limits(), 256);
    fresh
        .certification_request(registration("x", &["x"]))
        .unwrap();
    assert_eq!(
        fresh
            .certification_request(evaluate("p0", &["1"]))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
}

#[test]
fn failed_release_preserves_the_handle_and_unsupported_sources_publish_none() {
    let mut probe = session();
    probe
        .certification_request(registration("x", &["x"]))
        .unwrap();
    probe.certification_request(compile("retain")).unwrap();
    probe.certification_request(release("retain")).unwrap();
    let mut specification = limits();
    let index = Dimension::ComputeSteps.index();
    specification.dimensions[index] -= probe.budget_snapshot().dimensions[index] + 1;
    let mut bounded = Session::with_budget_limits(specification, 256);
    bounded
        .certification_request(registration("x", &["x"]))
        .unwrap();
    bounded.certification_request(compile("retain")).unwrap();
    let before = bounded.budget_snapshot();
    assert_eq!(
        bounded
            .certification_request(release("retain"))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(bounded.compiled_objective(&program("retain")).is_some());
    assert_eq!(
        bounded.budget_snapshot().verifier_pool,
        before.verifier_pool
    );

    let mut unsupported = session();
    unsupported
        .certification_request(registration("unregistered(x)", &["x"]))
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
            .compiled_objective(&program("unsupported"))
            .is_none()
    );
}

#[test]
fn numerical_output_exhaustion_keeps_the_program_and_never_borrows_verifier_fuel() {
    for request in [
        evaluate("bounded", &["2"]),
        product("bounded", &["2"], &["3"]),
    ] {
        let mut probe = session();
        probe
            .certification_request(registration("x^2", &["x"]))
            .unwrap();
        probe.certification_request(compile("bounded")).unwrap();
        let before = probe.budget_snapshot();
        let accepted = wire(&mut probe, request.clone());
        assert_eq!(accepted["status"], "success", "{accepted}");
        assert_eq!(probe.budget_snapshot().verifier_pool, before.verifier_pool);

        let mut specification = limits();
        let index = Dimension::ComputeSteps.index();
        specification.dimensions[index] -= probe.budget_snapshot().dimensions[index] + 1;
        let mut bounded = Session::with_budget_limits(specification, 256);
        bounded
            .certification_request(registration("x^2", &["x"]))
            .unwrap();
        bounded.certification_request(compile("bounded")).unwrap();
        let before = bounded.budget_snapshot();
        let refused = wire(&mut bounded, request);
        assert_eq!(refused["code"], "budget_exhausted", "{refused}");
        assert!(refused.get("result").is_none());
        assert!(bounded.compiled_objective(&program("bounded")).is_some());
        let after = bounded.budget_snapshot();
        assert_eq!(after.verifier_pool, before.verifier_pool);
        assert!(after.dimensions[index] < before.dimensions[index]);
    }
}

#[test]
fn strict_wire_schemas_fields_duplicate_keys_and_typed_ids_are_enforced() {
    let mut session = session();
    session
        .certification_request(registration("x^2", &["x"]))
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
        r#"{"type":"CompileObjective","payload":{"schema_version":1,"objective":"objective:quadratic","program":"objective_program:wire","program":"objective_program:wire"}}"#,
        r#"{"type":"CompileObjective","payload":{"schema_version":1,"objective":"objective:quadratic","program":"objective:wire"}}"#,
    ] {
        let response: Value = serde_json::from_str(&session.handle_envelope(raw)).unwrap();
        assert_eq!(response["code"], "malformed_request");
    }
    assert!(session.compiled_objective(&program("wire")).is_none());
    for invalid in [
        "objective_program:",
        "program:x",
        "objective_program:space here",
    ] {
        assert!(CompiledObjectiveId::new(invalid).is_err());
    }
    assert!(CompiledObjectiveId::new(format!("objective_program:{}", "a".repeat(128))).is_err());
    session.certification_request(compile("wire")).unwrap();
    for request in [
        evaluate("wire", &["2"]),
        product("wire", &["2"], &["1"]),
        release("wire"),
    ] {
        let mut frame = serde_json::to_value(request).unwrap();
        frame["payload"]["schema_version"] = json!(2);
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
        assert_eq!(response["code"], "unsupported_schema");
        frame["payload"]["schema_version"] = json!(1);
        frame["payload"]["certified"] = json!(true);
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
        assert_eq!(response["code"], "malformed_request");
    }
    assert!(session.compiled_objective(&program("wire")).is_some());
}

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new(arguments: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args([
                "--compute-steps",
                "1099511627776",
                "--memory-bytes",
                "1099511627776",
                "--allocations",
                "1099511627776",
                "--verifier-steps",
                "1099511627776",
            ])
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
fn fresh_cli_executes_objective_and_hessian_product_lifecycle_with_explicit_approximation() {
    let mut cli = Cli::new(&[]);
    let registered = cli.send(registration("x^2+x*y+3*y^2", &["x", "y"]), "register");
    assert_eq!(registered["status"], "success", "{registered}");
    let compiled = cli.send(compile("cli"), "compile");
    assert_eq!(compiled["status"], "success", "{compiled}");
    assert_eq!(compiled["result"]["kind"], "objective_compiled");
    assert_eq!(compiled["result"]["evidence"], "certificate_verified");
    let evaluated = cli.send(evaluate("cli", &["2", "-1"]), "evaluate");
    assert_eq!(evaluated["status"], "success", "{evaluated}");
    assert_eq!(evaluated["result"]["kind"], "objective_evaluated");
    assert_approximate(&evaluated["result"]);
    assert_eq!(evaluated["result"]["value"], json!(5.0));
    assert_eq!(evaluated["result"]["gradient"], json!([3.0, -4.0]));
    assert_eq!(evaluated["result"]["hessian"], json!([2.0, 1.0, 1.0, 6.0]));
    assert_eq!(
        evaluated["result"]["hessian_pattern"],
        json!([[0, 0], [0, 1], [1, 0], [1, 1]])
    );
    let multiplied = cli.send(product("cli", &["2", "-1"], &["3", "-2"]), "product");
    assert_eq!(multiplied["status"], "success", "{multiplied}");
    assert_eq!(
        multiplied["result"]["kind"],
        "objective_hessian_vector_product"
    );
    assert_eq!(multiplied["result"]["product"], json!([4.0, -9.0]));
    assert_approximate(&multiplied["result"]);
    assert_eq!(
        multiplied["result"]["context_digest"],
        compiled["result"]["context_digest"]
    );
    assert_eq!(cli.send(release("cli"), "release")["status"], "success");
    assert_eq!(
        cli.send(evaluate("cli", &["2", "-1"]), "released")["code"],
        "unknown_object"
    );
    cli.finish();

    let mut fresh = Cli::new(&[]);
    assert_eq!(
        fresh.send(product("cli", &["2", "-1"], &["3", "-2"]), "isolated")["code"],
        "unknown_object"
    );
    fresh.finish();
}

#[test]
fn fresh_cli_budget_and_second_derivative_failures_publish_no_numeric_result() {
    let mut cli = Cli::new(&["--verifier-steps", "0"]);
    assert_eq!(
        cli.send(registration("x", &["x"]), "register")["status"],
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
        cli.send(registration("x^(3/2)", &["x"]), "register")["status"],
        "success"
    );
    let compiled = cli.send(compile("domain"), "compile");
    assert_eq!(compiled["status"], "success", "{compiled}");
    for (request, id) in [
        (evaluate("domain", &["0"]), "second-derivative"),
        (product("domain", &["0"], &["0"]), "zero-direction"),
    ] {
        let refused = cli.send(request, id);
        assert_eq!(refused["code"], "numeric_evaluation_refused");
        assert!(refused.get("result").is_none());
    }
    assert_eq!(
        cli.send(evaluate("domain", &["1"]), "valid")["status"],
        "success"
    );
    cli.finish();
}
