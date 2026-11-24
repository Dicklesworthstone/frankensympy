//! Real numerical consumers of immutable local and replayed Session programs.

use frankensympy::{
    BudgetLimits, CompiledObjectiveId, CompiledResidualSystemId, Dimension, Session,
};
use serde_json::{Value, json};

const AMPLE: u64 = 1 << 40;
const SYSTEM: &str = "system:solver";
const PROGRAM: &str = "program:solver";
const OBJECTIVE: &str = "objective:solver";
const OBJECTIVE_PROGRAM: &str = "objective_program:solver";

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn request(operation: &str, mut payload: Value) -> String {
    payload["schema_version"] = json!(1);
    json!({"id":"solver-request", "type":operation, "payload":payload}).to_string()
}

fn call(session: &mut Session, operation: &str, payload: Value) -> Value {
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&request(operation, payload))).unwrap();
    assert_eq!(response["id"], "solver-request");
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

fn approximate(result: &Value) {
    assert_eq!(result["outcome"], "approximate");
    assert_eq!(result["numeric_model"], "binary64");
    for field in ["evidence", "claim", "certificate", "classification"] {
        assert!(result.get(field).is_none(), "{field}: {result}");
    }
}

fn number(value: &Value) -> f64 {
    value.as_f64().unwrap()
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected} by more than {tolerance}"
    );
}

fn near_vector(actual: &Value, expected: &[f64], tolerance: f64) {
    let actual = actual.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        near(number(actual), *expected, tolerance);
    }
}

fn register_residuals(session: &mut Session, sources: &[&str], variables: &[&str]) {
    success(call(
        session,
        "RegisterResidualSystem",
        json!({"system":SYSTEM, "residuals":sources, "variables":variables}),
    ));
}

fn residual_session(
    budget: BudgetLimits,
    sources: &[&str],
    variables: &[&str],
) -> (Session, Value) {
    let mut session = Session::with_budget_limits(budget, 256);
    register_residuals(&mut session, sources, variables);
    let compiled = success(call(
        &mut session,
        "CompileResidualSystem",
        json!({"system":SYSTEM, "program":PROGRAM}),
    ));
    (session, compiled)
}

fn objective_session(budget: BudgetLimits, source: &str, variables: &[&str]) -> (Session, Value) {
    let mut session = Session::with_budget_limits(budget, 256);
    success(call(
        &mut session,
        "RegisterObjective",
        json!({"objective":OBJECTIVE, "source":source, "variables":variables}),
    ));
    let compiled = success(call(
        &mut session,
        "CompileObjective",
        json!({"objective":OBJECTIVE, "program":OBJECTIVE_PROGRAM}),
    ));
    (session, compiled)
}

fn solve(session: &mut Session, backend: &str, initial: &[&str]) -> Value {
    call(
        session,
        "SolveCompiledResidualSystem",
        json!({"program":PROGRAM, "initial":initial, "backend":backend}),
    )
}

fn fit(session: &mut Session, backend: &str, initial: &[&str]) -> Value {
    call(
        session,
        "SolveCompiledLeastSquares",
        json!({"program":PROGRAM, "initial":initial, "backend":backend}),
    )
}

fn minimize(session: &mut Session, initial: &[&str]) -> Value {
    call(
        session,
        "MinimizeCompiledObjective",
        json!({"program":OBJECTIVE_PROGRAM, "initial":initial}),
    )
}

#[test]
fn all_root_backends_execute_local_and_replayed_programs_in_the_frozen_axis_order() {
    let sources = ["x^2-2", "y^2-3"];
    let variables = ["y", "x"];
    let (mut sender, _) = residual_session(limits(), &sources, &variables);
    let artifact = success(call(
        &mut sender,
        "ExportCompiledResidualSystem",
        json!({"program":PROGRAM}),
    ))["artifact"]
        .as_str()
        .unwrap()
        .to_owned();

    for imported in [false, true] {
        let (mut session, admitted) = if imported {
            let mut receiver = Session::with_budget_limits(limits(), 256);
            register_residuals(&mut receiver, &sources, &variables);
            let replayed = success(call(
                &mut receiver,
                "ReplayCompiledResidualSystem",
                json!({"system":SYSTEM, "program":PROGRAM, "artifact":artifact}),
            ));
            (receiver, replayed)
        } else {
            residual_session(limits(), &sources, &variables)
        };
        session.construct("x", "999").unwrap();
        session.construct("y", "-999").unwrap();
        for backend in [
            "frankennumpy",
            "frankenscipy",
            "frankenscipy_analytic",
            "sparse",
        ] {
            let before = session.budget_snapshot();
            let result = success(solve(&mut session, backend, &["1.5", "1"]));
            approximate(&result);
            assert_eq!(result["kind"], "compiled_residual_system_solved");
            assert_eq!(result["system"], SYSTEM);
            assert_eq!(result["program"], PROGRAM);
            assert_eq!(result["backend"], backend);
            assert_eq!(result["termination"], "residual_tolerance");
            assert_eq!(result["context_digest"], admitted["context_digest"]);
            near_vector(&result["root"], &[3.0_f64.sqrt(), 2.0_f64.sqrt()], 1e-9);
            let y = number(&result["root"][0]);
            let x = number(&result["root"][1]);
            let original_norm = (x * x - 2.0).abs().max((y * y - 3.0).abs());
            near(number(&result["residual_norm"]), original_norm, 1e-12);
            assert!(number(&result["residual_norm"]) <= 1e-10);
            assert!(result["iterations"].as_u64().unwrap() > 0);
            let after = session.budget_snapshot();
            assert_eq!(after.verifier_pool, before.verifier_pool);
            assert!(
                after.dimensions[Dimension::ComputeSteps.index()]
                    < before.dimensions[Dimension::ComputeSteps.index()]
            );
        }
    }
}

#[test]
fn sparse_root_backend_accepts_overdetermined_systems_and_rechecks_every_row() {
    let (mut session, _) = residual_session(limits(), &["x-2", "2*x-4", "x+y-5"], &["y", "x"]);
    let result = success(solve(&mut session, "sparse", &["0", "0"]));
    approximate(&result);
    near_vector(&result["root"], &[3.0, 2.0], 1e-10);
    assert!(number(&result["residual_norm"]) <= 1e-12);

    let (mut inconsistent, _) = residual_session(limits(), &["x-1", "x-3"], &["x"]);
    refused(
        solve(&mut inconsistent, "sparse", &["2"]),
        "numeric_no_convergence",
    );
}

#[test]
fn both_least_squares_backends_retain_every_inconsistent_residual_row() {
    let (mut session, compiled) = residual_session(limits(), &["x-1", "x-3", "x-5"], &["x"]);
    for backend in ["dense", "sparse"] {
        let before = session.budget_snapshot();
        let result = success(fit(&mut session, backend, &["0"]));
        approximate(&result);
        assert_eq!(result["kind"], "compiled_least_squares_solved");
        assert_eq!(result["backend"], backend);
        assert_eq!(result["system"], SYSTEM);
        assert_eq!(result["program"], PROGRAM);
        assert_eq!(result["termination"], "gradient_tolerance");
        assert_eq!(result["context_digest"], compiled["context_digest"]);
        near_vector(&result["point"], &[3.0], 1e-10);
        near_vector(&result["residuals"], &[2.0, 0.0, -2.0], 1e-10);
        near(number(&result["cost"]), 4.0, 1e-12);
        assert!(number(&result["gradient_norm"]) <= 1e-10);
        let iterations = result["iterations"].as_u64().unwrap();
        assert!(iterations > 0);
        assert!(result["evaluations"].as_u64().unwrap() >= iterations + 2);
        assert_eq!(
            session.budget_snapshot().verifier_pool,
            before.verifier_pool
        );
    }
    refused(
        solve(&mut session, "frankennumpy", &["0"]),
        "unsupported_operation",
    );
}

#[test]
fn explicit_sparse_selection_supports_rank_deficiency_and_underdetermined_systems() {
    let (mut session, _) =
        residual_session(limits(), &["x+y-1", "2*x+2*y-2", "3*x+3*y-4"], &["x", "y"]);
    refused(
        fit(&mut session, "dense", &["0", "0"]),
        "numeric_consumer_refused",
    );
    let result = success(fit(&mut session, "sparse", &["0", "0"]));
    approximate(&result);
    // All minimizers have x+y=17/14. No unique coordinate split is asserted.
    near(
        number(&result["point"][0]) + number(&result["point"][1]),
        17.0 / 14.0,
        1e-10,
    );
    near_vector(
        &result["residuals"],
        &[3.0 / 14.0, 3.0 / 7.0, -5.0 / 14.0],
        1e-10,
    );
    near(number(&result["cost"]), 5.0 / 28.0, 1e-12);
    assert!(number(&result["gradient_norm"]) <= 1e-10);

    let (mut session, _) = residual_session(limits(), &["x+y-3"], &["x", "y", "z"]);
    refused(
        fit(&mut session, "dense", &["0", "0", "7"]),
        "invalid_input",
    );
    let result = success(fit(&mut session, "sparse", &["0", "0", "7"]));
    approximate(&result);
    near(
        number(&result["point"][0]) + number(&result["point"][1]),
        3.0,
        1e-10,
    );
    assert_eq!(result["point"][2], 7.0);
    near(number(&result["cost"]), 0.0, 1e-18);
}

#[test]
fn objective_descent_returns_the_original_value_gradient_and_frozen_context() {
    let (mut session, compiled) =
        objective_session(limits(), "3*(x-2)^2+2*(x-2)*(y+1)+4*(y+1)^2+7", &["y", "x"]);
    session.construct("x", "1234").unwrap();
    let before = session.budget_snapshot();
    let result = success(minimize(&mut session, &["6", "-4"]));
    approximate(&result);
    assert_eq!(result["kind"], "compiled_objective_minimized");
    assert_eq!(result["objective"], OBJECTIVE);
    assert_eq!(result["program"], OBJECTIVE_PROGRAM);
    assert_eq!(result["termination"], "gradient_tolerance");
    assert_eq!(result["context_digest"], compiled["context_digest"]);
    near_vector(&result["point"], &[-1.0, 2.0], 1e-9);
    let b = number(&result["point"][0]) + 1.0;
    let a = number(&result["point"][1]) - 2.0;
    near(
        number(&result["value"]),
        3.0 * a * a + 2.0 * a * b + 4.0 * b * b + 7.0,
        1e-12,
    );
    near(number(&result["value"]), 7.0, 1e-12);
    near_vector(
        &result["gradient"],
        &[2.0 * a + 8.0 * b, 6.0 * a + 2.0 * b],
        1e-11,
    );
    assert!(number(&result["gradient_norm"]) <= 1e-10);
    let iterations = result["iterations"].as_u64().unwrap();
    assert!(iterations > 0);
    assert!(result["evaluations"].as_u64().unwrap() >= iterations + 2);
    assert!(result["regularizations"].is_u64());
    assert_eq!(
        session.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
}

#[test]
fn stationary_nonroots_maxima_and_saddles_remain_approximate_proposals() {
    let (mut session, _) = residual_session(limits(), &["x^2-1"], &["x"]);
    for backend in ["dense", "sparse"] {
        let result = success(call(
            &mut session,
            "SolveCompiledLeastSquares",
            json!({"program":PROGRAM, "backend":backend, "initial":["0"],
                "options":{"max_iterations":0}}),
        ));
        approximate(&result);
        assert_eq!(result["point"], json!([0.0]));
        assert_eq!(result["residuals"], json!([-1.0]));
        assert_eq!(result["cost"], 0.5);
        assert_eq!(result["gradient_norm"], 0.0);
        assert_eq!(result["iterations"], 0);
    }
    refused(
        solve(&mut session, "sparse", &["0"]),
        "numeric_no_convergence",
    );

    for (source, variables, initial, value) in [
        ("3-x^2", vec!["x"], vec!["0"], 3.0),
        ("x^2-y^2", vec!["x", "y"], vec!["0", "0"], 0.0),
    ] {
        let (mut session, _) = objective_session(limits(), source, &variables);
        let result = success(minimize(&mut session, &initial));
        approximate(&result);
        assert_eq!(result["value"], value);
        assert_eq!(result["gradient_norm"], 0.0);
        assert_eq!(result["iterations"], 0);
        assert_eq!(result["regularizations"], 0);
        assert!(result["evaluations"].as_u64().unwrap() >= 2);
        near_vector(&result["gradient"], &vec![0.0; variables.len()], 0.0);
    }
}

#[test]
fn original_poles_and_singular_second_derivatives_cannot_be_hidden_by_stationarity() {
    let (mut session, _) = residual_session(limits(), &["x-1+0*(1/x)"], &["x"]);
    for backend in [
        "frankennumpy",
        "frankenscipy",
        "frankenscipy_analytic",
        "sparse",
    ] {
        refused(
            solve(&mut session, backend, &["0"]),
            "numeric_evaluation_refused",
        );
        let result = success(solve(&mut session, backend, &["1"]));
        near_vector(&result["root"], &[1.0], 0.0);
    }
    for backend in ["dense", "sparse"] {
        refused(
            fit(&mut session, backend, &["0"]),
            "numeric_evaluation_refused",
        );
        let result = success(fit(&mut session, backend, &["1"]));
        assert_eq!(result["cost"], 0.0);
    }
    let (mut session, _) = objective_session(limits(), "(x-1)^2+0*(1/x)", &["x"]);
    refused(minimize(&mut session, &["0"]), "numeric_evaluation_refused");
    let result = success(minimize(&mut session, &["1"]));
    assert_eq!(result["value"], 0.0);
    assert!(
        session
            .compiled_objective(&CompiledObjectiveId::new(OBJECTIVE_PROGRAM).unwrap())
            .is_some()
    );

    // The value and gradient are finite at zero, but the retained Hessian is not.
    let (mut session, _) = objective_session(limits(), "x^(3/2)", &["x"]);
    refused(minimize(&mut session, &["0"]), "numeric_evaluation_refused");
}

#[test]
fn iteration_and_step_controls_are_honored_without_backend_fallback() {
    let (mut session, _) = residual_session(limits(), &["x-2"], &["x"]);
    let controls =
        json!({"max_iterations":16, "residual_tolerance":"1e-12", "max_step_norm":"0.25"});
    refused(
        call(
            &mut session,
            "SolveCompiledResidualSystem",
            json!({"program":PROGRAM, "backend":"frankennumpy", "initial":["0"], "options":controls}),
        ),
        "numeric_no_convergence",
    );
    let result = success(call(
        &mut session,
        "SolveCompiledResidualSystem",
        json!({"program":PROGRAM, "backend":"sparse", "initial":["0"], "options":controls}),
    ));
    near_vector(&result["root"], &[2.0], 1e-12);
    assert!(result["iterations"].as_u64().unwrap() >= 8);
    for backend in ["frankennumpy", "sparse"] {
        refused(
            call(
                &mut session,
                "SolveCompiledResidualSystem",
                json!({"program":PROGRAM, "backend":backend, "initial":["0"], "options":{"max_iterations":0}}),
            ),
            "numeric_no_convergence",
        );
        let result = success(call(
            &mut session,
            "SolveCompiledResidualSystem",
            json!({"program":PROGRAM, "backend":backend, "initial":["2"], "options":{"max_iterations":0}}),
        ));
        assert_eq!(result["iterations"], 0);
    }
    let result = success(call(
        &mut session,
        "SolveCompiledLeastSquares",
        json!({"program":PROGRAM, "backend":"dense", "initial":["0"],
            "options":{"max_iterations":16, "gradient_tolerance":"1e-10", "max_step_norm":"0.25"}}),
    ));
    near_vector(&result["point"], &[2.0], 1e-12);
    assert!(result["iterations"].as_u64().unwrap() >= 8);
    let (mut session, _) = objective_session(limits(), "(x-2)^2", &["x"]);
    let result = success(call(
        &mut session,
        "MinimizeCompiledObjective",
        json!({"program":OBJECTIVE_PROGRAM, "initial":["0"],
            "options":{"max_iterations":16, "gradient_tolerance":"1e-10", "max_step_norm":"0.25"}}),
    ));
    near_vector(&result["point"], &[2.0], 1e-12);
    assert!(result["iterations"].as_u64().unwrap() >= 8);
}

#[test]
fn numeric_inputs_controls_and_fixed_scipy_options_refuse_explicitly() {
    let (mut residual, _) = residual_session(limits(), &["x-2"], &["x"]);
    let (mut objective, _) = objective_session(limits(), "x^2+7", &["x"]);
    for initial in [
        vec![],
        vec!["1", "2"],
        vec!["NaN"],
        vec!["inf"],
        vec!["1e999"],
        vec!["1/2"],
        vec![" 1"],
    ] {
        refused(
            solve(&mut residual, "frankennumpy", &initial),
            "invalid_input",
        );
        refused(fit(&mut residual, "dense", &initial), "invalid_input");
        refused(minimize(&mut objective, &initial), "invalid_input");
    }
    for backend in ["frankenscipy", "frankenscipy_analytic"] {
        for options in [
            json!({"max_iterations":200}),
            json!({"residual_tolerance":"1e-10"}),
            json!({"max_step_norm":"1000000"}),
        ] {
            refused(
                call(
                    &mut residual,
                    "SolveCompiledResidualSystem",
                    json!({"program":PROGRAM, "backend":backend, "initial":["0"], "options":options}),
                ),
                "invalid_input",
            );
        }
    }
    for options in [
        json!({"max_iterations":1025}),
        json!({"residual_tolerance":"-1"}),
        json!({"residual_tolerance":"NaN"}),
        json!({"max_step_norm":"0"}),
    ] {
        refused(
            call(
                &mut residual,
                "SolveCompiledResidualSystem",
                json!({"program":PROGRAM, "backend":"frankennumpy", "initial":["0"], "options":options}),
            ),
            "invalid_input",
        );
    }
    for options in [
        json!({"max_iterations":1025}),
        json!({"gradient_tolerance":"-1"}),
        json!({"gradient_tolerance":"NaN"}),
        json!({"max_step_norm":"0"}),
    ] {
        refused(
            call(
                &mut residual,
                "SolveCompiledLeastSquares",
                json!({"program":PROGRAM, "backend":"dense", "initial":["0"], "options":options}),
            ),
            "invalid_input",
        );
        refused(
            call(
                &mut objective,
                "MinimizeCompiledObjective",
                json!({"program":OBJECTIVE_PROGRAM, "initial":["0"], "options":options}),
            ),
            "invalid_input",
        );
    }
}

#[test]
fn solver_wire_schemas_reject_unknown_and_duplicate_controls() {
    let (mut session, _) = residual_session(limits(), &["x-2"], &["x"]);
    for payload in [
        json!({"program":PROGRAM, "backend":"automatic", "initial":["0"]}),
        json!({"program":PROGRAM, "backend":"frankennumpy", "initial":["0"], "options":{"secret_fallback":true}}),
        json!({"program":PROGRAM, "backend":"frankennumpy", "initial":["0"], "options":{"residual_tolerance":0.1}}),
    ] {
        refused(
            call(&mut session, "SolveCompiledResidualSystem", payload),
            "malformed_request",
        );
    }
    let duplicate = r#"{"id":"duplicate","type":"SolveCompiledResidualSystem","payload":{"schema_version":1,"program":"program:solver","backend":"frankennumpy","initial":["0"],"options":{"max_iterations":4,"max_iterations":8}}}"#;
    refused(
        serde_json::from_str(&session.handle_envelope(duplicate)).unwrap(),
        "malformed_request",
    );
    let future = json!({"type":"SolveCompiledResidualSystem", "payload":{
        "schema_version":2, "program":PROGRAM, "backend":"frankennumpy", "initial":["0"]
    }});
    refused(
        serde_json::from_str(&session.handle_envelope(&future.to_string())).unwrap(),
        "unsupported_schema",
    );
    let invalid_id = json!({"id":"x".repeat(257), "type":"SolveCompiledResidualSystem", "payload":{
        "schema_version":1, "program":PROGRAM, "backend":"frankennumpy", "initial":["0"]
    }});
    refused(
        serde_json::from_str(&session.handle_envelope(&invalid_id.to_string())).unwrap(),
        "invalid_request_id",
    );
}

#[test]
fn released_programs_are_unavailable_to_every_numerical_consumer() {
    let (mut session, _) = residual_session(limits(), &["x-2"], &["x"]);
    success(call(
        &mut session,
        "ReleaseCompiledResidualSystem",
        json!({"program":PROGRAM}),
    ));
    refused(
        solve(&mut session, "frankennumpy", &["0"]),
        "unknown_object",
    );
    refused(fit(&mut session, "sparse", &["0"]), "unknown_object");
    let (mut session, _) = objective_session(limits(), "x^2", &["x"]);
    success(call(
        &mut session,
        "ReleaseCompiledObjective",
        json!({"program":OBJECTIVE_PROGRAM}),
    ));
    refused(minimize(&mut session, &["1"]), "unknown_object");
}

#[derive(Clone, Copy, Debug)]
enum SolverCase {
    Root,
    LeastSquares,
    Objective,
}

impl SolverCase {
    fn session(self, limits: BudgetLimits) -> Session {
        match self {
            Self::Root => residual_session(limits, &["x^2-2"], &["x"]).0,
            Self::LeastSquares => residual_session(limits, &["x-1", "x-3", "x-5"], &["x"]).0,
            Self::Objective => objective_session(limits, "3*(x-2)^2+7", &["x"]).0,
        }
    }

    fn run(self, session: &mut Session) -> Value {
        match self {
            Self::Root => solve(session, "frankennumpy", &["1"]),
            Self::LeastSquares => fit(session, "dense", &["0"]),
            Self::Objective => minimize(session, &["0"]),
        }
    }

    fn assert_program_retained(self, session: &Session) {
        match self {
            Self::Root | Self::LeastSquares => assert!(
                session
                    .compiled_residual_system(&CompiledResidualSystemId::new(PROGRAM).unwrap())
                    .is_some()
            ),
            Self::Objective => assert!(
                session
                    .compiled_objective(&CompiledObjectiveId::new(OBJECTIVE_PROGRAM).unwrap())
                    .is_some()
            ),
        }
    }
}

#[test]
fn final_response_resource_refusal_retains_programs_and_all_previous_work() {
    for case in [
        SolverCase::Root,
        SolverCase::LeastSquares,
        SolverCase::Objective,
    ] {
        let mut probe = case.session(limits());
        success(case.run(&mut probe));
        let remaining = probe.budget_snapshot();
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            // Keep identical work through the solver and make the final response
            // admission one unit too expensive. Failed charges cannot refund it.
            let mut bounded = limits();
            bounded.dimensions[dimension.index()] =
                AMPLE - remaining.dimensions[dimension.index()] - 1;
            let mut session = case.session(bounded);
            let before = session.budget_snapshot();
            refused(case.run(&mut session), "budget_exhausted");
            let after = session.budget_snapshot();
            assert!(
                after.dimensions[dimension.index()] < before.dimensions[dimension.index()],
                "{case:?} {dimension:?}"
            );
            assert_eq!(after.verifier_pool, before.verifier_pool);
            case.assert_program_retained(&session);
        }
    }
}

#[test]
fn solver_allowances_are_cumulative_and_need_no_remaining_verifier_fuel() {
    for case in [
        SolverCase::Root,
        SolverCase::LeastSquares,
        SolverCase::Objective,
    ] {
        let mut probe = case.session(limits());
        let compile_verifier_cost = AMPLE - probe.budget_snapshot().verifier_pool;
        success(case.run(&mut probe));
        let total_compute =
            AMPLE - probe.budget_snapshot().dimensions[Dimension::ComputeSteps.index()];
        let mut bounded = limits();
        bounded.dimensions[Dimension::ComputeSteps.index()] = total_compute;
        bounded.verifier_pool = compile_verifier_cost;
        let mut session = case.session(bounded);
        assert_eq!(session.budget_snapshot().verifier_pool, 0);
        approximate(&success(case.run(&mut session)));
        assert_eq!(
            session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()],
            0
        );
        assert_eq!(session.budget_snapshot().verifier_pool, 0);
        refused(case.run(&mut session), "budget_exhausted");
        assert_eq!(
            session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()],
            0
        );
        assert_eq!(session.budget_snapshot().verifier_pool, 0);
        case.assert_program_retained(&session);
    }
}
