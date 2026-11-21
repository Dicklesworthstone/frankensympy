//! Fixed-parameter root proposals reuse the complete frozen source and its
//! independently admitted Jacobian, including after fresh-process replay.

use frankensympy::{BudgetLimits, CompiledResidualSystemId, Dimension, Session};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

const AMPLE: u64 = 1 << 40;
const SYSTEM: &str = "system:parametric-solve";
const PROGRAM: &str = "program:parametric-solve";

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn frame(operation: &str, mut payload: Value) -> String {
    payload["schema_version"] = json!(1);
    json!({"id":"parametric-solve", "type":operation, "payload":payload}).to_string()
}

fn call(session: &mut Session, operation: &str, payload: Value) -> Value {
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&frame(operation, payload))).unwrap();
    assert_eq!(response["id"], "parametric-solve");
    response
}

fn success(response: Value) -> Value {
    assert_eq!(response["status"], "success", "{response}");
    response["result"].clone()
}

fn refused(response: Value, code: &str) {
    assert_eq!(response["status"], "error", "{response}");
    assert_eq!(response["code"], code, "{response}");
    assert!(response.get("result").is_none(), "{response}");
}

fn register(session: &mut Session, residuals: &[&str], variables: &[&str]) {
    success(call(
        session,
        "RegisterResidualSystem",
        json!({"system":SYSTEM, "residuals":residuals, "variables":variables}),
    ));
}

fn compiled(limits: BudgetLimits, residuals: &[&str], variables: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits, 256);
    register(&mut session, residuals, variables);
    success(call(
        &mut session,
        "CompileResidualSystem",
        json!({"system":SYSTEM, "program":PROGRAM}),
    ));
    session
}

fn solve(session: &mut Session, initial: &[&str], parameters: &[&str]) -> Value {
    call(
        session,
        "SolveCompiledParametricSystem",
        json!({"program":PROGRAM, "initial":initial, "parameters":parameters}),
    )
}

fn retained(session: &Session) {
    assert!(
        session
            .compiled_residual_system(&CompiledResidualSystemId::new(PROGRAM).unwrap())
            .is_some()
    );
}

fn approximate(result: &Value) {
    assert_eq!(result["kind"], "compiled_parametric_system_solved");
    assert_eq!(result["outcome"], "approximate");
    assert_eq!(result["numeric_model"], "binary64");
    assert_eq!(result["backend"], "frankennumpy");
    assert_eq!(result["termination"], "residual_tolerance");
    for field in ["evidence", "claim", "certificate", "family"] {
        assert!(result.get(field).is_none(), "{field}: {result}");
    }
}

#[test]
fn local_and_replayed_programs_keep_unknown_and_parameter_order_across_repeated_solves() {
    let sources = ["x^2-a", "y^2-b"];
    let axes = ["y", "x", "b", "a"];
    let mut sender = compiled(limits(), &sources, &axes);
    let artifact = success(call(
        &mut sender,
        "ExportCompiledResidualSystem",
        json!({"program":PROGRAM}),
    ))["artifact"]
        .as_str()
        .unwrap()
        .to_owned();
    for replay in [false, true] {
        let mut session = if replay {
            let mut session = Session::with_budget_limits(limits(), 256);
            register(&mut session, &sources, &axes);
            success(call(
                &mut session,
                "ReplayCompiledResidualSystem",
                json!({"system":SYSTEM, "program":PROGRAM, "artifact":artifact}),
            ));
            session
        } else {
            compiled(limits(), &sources, &axes)
        };
        for name in axes {
            session.construct(name, "999").unwrap();
        }
        for (parameters, expected) in [(["9", "4"], [3.0, 2.0]), (["16", "25"], [4.0, 5.0])] {
            let before = session.budget_snapshot();
            let result = success(solve(&mut session, &["1", "1"], &parameters));
            approximate(&result);
            assert_eq!(result["system"], SYSTEM);
            assert_eq!(result["program"], PROGRAM);
            assert_eq!(result["unknowns"][0]["name"], "y");
            assert_eq!(result["unknowns"][1]["name"], "x");
            assert_eq!(result["parameter_variables"][0]["name"], "b");
            assert_eq!(result["parameter_variables"][1]["name"], "a");
            assert_eq!(result["context_digest"], json!(session.context_digest()));
            let y = result["root"][0].as_f64().unwrap();
            let x = result["root"][1].as_f64().unwrap();
            let b = result["parameters"][0].as_f64().unwrap();
            let a = result["parameters"][1].as_f64().unwrap();
            assert_eq!(
                [b, a],
                parameters.map(|value| value.parse::<f64>().unwrap())
            );
            assert!((y - expected[0]).abs() < 1e-10);
            assert!((x - expected[1]).abs() < 1e-10);
            let original_norm = (x * x - a).abs().max((y * y - b).abs());
            assert_eq!(result["residual_norm"].as_f64().unwrap(), original_norm);
            assert!(original_norm <= 1e-12);
            let after = session.budget_snapshot();
            assert_eq!(after.verifier_pool, before.verifier_pool);
            assert!(
                after.dimensions[Dimension::ComputeSteps.index()]
                    < before.dimensions[Dimension::ComputeSteps.index()]
            );
            retained(&session);
        }
    }
}

#[test]
fn fixed_parameter_poles_and_derivative_domains_cannot_disappear_at_a_zero_residual() {
    for source in ["x-1+0*(1/p)", "x-1+0*sqrt(p)"] {
        let mut session = compiled(limits(), &[source], &["x", "p"]);
        refused(
            solve(&mut session, &["1"], &["0"]),
            "numeric_evaluation_refused",
        );
        retained(&session);
        let result = success(solve(&mut session, &["1"], &["1"]));
        approximate(&result);
        assert_eq!(result["root"], json!([1.0]));
    }
}

#[test]
fn singular_roots_loose_tolerances_and_failed_refinement_never_grant_certificates() {
    let mut session = compiled(limits(), &["x^2-p"], &["x", "p"]);
    let initial_root = success(solve(&mut session, &["0"], &["0"]));
    approximate(&initial_root);
    assert_eq!(initial_root["iterations"], 0);
    refused(
        solve(&mut session, &["0"], &["1"]),
        "numeric_consumer_refused",
    );
    retained(&session);
    for options in [json!({"max_iterations":0}), json!({"max_step_norm":"0.01"})] {
        refused(
            call(
                &mut session,
                "SolveCompiledParametricSystem",
                json!({"program":PROGRAM, "initial":["1"], "parameters":["4"], "options":options}),
            ),
            "numeric_no_convergence",
        );
    }
    let result = success(call(
        &mut session,
        "SolveCompiledParametricSystem",
        json!({"program":PROGRAM, "initial":["1"], "parameters":["4"],
            "options":{"residual_tolerance":"4", "max_iterations":0}}),
    ));
    approximate(&result);
    assert_eq!(result["root"], json!([1.0]));
    assert_eq!(result["residual_norm"], 3.0);
}

#[test]
fn malformed_values_dimensions_options_and_released_handles_refuse() {
    let mut session = compiled(limits(), &["x-p"], &["x", "p"]);
    for parameters in [
        vec![],
        vec!["1", "2"],
        vec!["NaN"],
        vec!["inf"],
        vec!["1/2"],
        vec![" 1"],
    ] {
        refused(solve(&mut session, &["0"], &parameters), "invalid_input");
    }
    for initial in [vec![], vec!["1", "2"], vec!["1e999"]] {
        refused(solve(&mut session, &initial, &["1"]), "invalid_input");
    }
    for options in [
        json!({"max_iterations":1025}),
        json!({"residual_tolerance":"-1"}),
        json!({"max_step_norm":"0"}),
    ] {
        refused(
            call(
                &mut session,
                "SolveCompiledParametricSystem",
                json!({"program":PROGRAM, "initial":["0"], "parameters":["1"], "options":options}),
            ),
            "invalid_input",
        );
    }
    for extra in [
        json!({"backend":"automatic"}),
        json!({"options":{"fallback":true}}),
    ] {
        let mut payload = json!({"program":PROGRAM, "initial":["0"], "parameters":["1"]});
        payload
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        refused(
            call(&mut session, "SolveCompiledParametricSystem", payload),
            "malformed_request",
        );
    }
    let duplicate = r#"{"id":"parametric-solve","type":"SolveCompiledParametricSystem","payload":{"schema_version":1,"program":"program:parametric-solve","initial":["0"],"parameters":["1"],"parameters":["2"]}}"#;
    refused(
        serde_json::from_str(&session.handle_envelope(duplicate)).unwrap(),
        "malformed_request",
    );
    retained(&session);
    success(call(
        &mut session,
        "ReleaseCompiledResidualSystem",
        json!({"program":PROGRAM}),
    ));
    refused(solve(&mut session, &["0"], &["1"]), "unknown_object");
    let mut square = compiled(limits(), &["x-1"], &["x"]);
    refused(solve(&mut square, &["0"], &[]), "invalid_input");
}

#[test]
fn compilation_and_repeated_parameter_solves_share_a_cumulative_ordinary_ledger() {
    let mut probe = compiled(limits(), &["x^2-p"], &["x", "p"]);
    let verifier_cost = AMPLE - probe.budget_snapshot().verifier_pool;
    success(solve(&mut probe, &["1"], &["2"]));
    let remaining = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let total = AMPLE - remaining.dimensions[dimension.index()];
        let mut bounded = limits();
        bounded.dimensions[dimension.index()] = total - 1;
        bounded.verifier_pool = verifier_cost;
        let mut session = compiled(bounded, &["x^2-p"], &["x", "p"]);
        let before = session.budget_snapshot();
        refused(solve(&mut session, &["1"], &["2"]), "budget_exhausted");
        assert_eq!(session.budget_snapshot().verifier_pool, 0);
        assert!(
            session.budget_snapshot().dimensions[dimension.index()]
                < before.dimensions[dimension.index()]
        );
        retained(&session);
    }
    let mut bounded = limits();
    bounded.dimensions[Dimension::ComputeSteps.index()] =
        AMPLE - remaining.dimensions[Dimension::ComputeSteps.index()];
    bounded.verifier_pool = verifier_cost;
    let mut session = compiled(bounded, &["x^2-p"], &["x", "p"]);
    approximate(&success(solve(&mut session, &["1"], &["2"])));
    refused(solve(&mut session, &["1"], &["3"]), "budget_exhausted");
    retained(&session);
}

#[test]
fn ndjson_fresh_process_replays_and_solves_the_full_parameterized_program() {
    let mut sender = compiled(limits(), &["x^2-p"], &["x", "p"]);
    let artifact = success(call(
        &mut sender,
        "ExportCompiledResidualSystem",
        json!({"program":PROGRAM}),
    ))["artifact"]
        .clone();
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
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    for (operation, payload) in [
        (
            "RegisterResidualSystem",
            json!({"system":SYSTEM, "residuals":["x^2-p"], "variables":["x","p"]}),
        ),
        (
            "ReplayCompiledResidualSystem",
            json!({"system":SYSTEM, "program":PROGRAM, "artifact":artifact}),
        ),
        (
            "SolveCompiledParametricSystem",
            json!({"program":PROGRAM, "initial":["1"], "parameters":["9"]}),
        ),
    ] {
        writeln!(stdin, "{}", frame(operation, payload)).unwrap();
        stdin.flush().unwrap();
        let mut response = String::new();
        assert!(stdout.read_line(&mut response).unwrap() > 0);
        let result = success(serde_json::from_str(&response).unwrap());
        if operation == "SolveCompiledParametricSystem" {
            approximate(&result);
            assert!((result["root"][0].as_f64().unwrap() - 3.0).abs() < 1e-10);
            assert_eq!(result["parameters"], json!([9.0]));
        }
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
