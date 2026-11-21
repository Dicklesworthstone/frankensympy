//! Midpoint proposals become whole-parameter-box claims only after
//! source binding and protected independent original-source checking.

use frankensympy::{
    BigRational, BudgetLimits, CompiledResidualSystemId, Dimension, ParametricFamilyId,
    ParametricFamilyReceipt, Session,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

const AMPLE: u64 = 1 << 40;
const SYSTEM: &str = "system:family-solve";
const PROGRAM: &str = "program:family-solve";
const PROBLEM: &str = "parametric:family-solve";
const FAMILY: &str = "family:solved";

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: &str, upper: &str) -> Value {
    json!({"lower": lower, "upper": upper})
}

fn frame(operation: &str, mut payload: Value) -> String {
    payload["schema_version"] = json!(1);
    json!({"id": "family-solve", "type": operation, "payload": payload}).to_string()
}

fn call(session: &mut Session, operation: &str, payload: Value) -> Value {
    let result: Value =
        serde_json::from_str(&session.handle_envelope(&frame(operation, payload))).unwrap();
    assert_eq!(result["id"], "family-solve");
    result
}

fn success(response: Value) -> Value {
    assert_eq!(response["status"], "success", "{response}");
    response["result"].clone()
}

fn refused(response: Value, code: &str) {
    assert_eq!(response["status"], "error", "{response}");
    assert_eq!(response["code"], code, "{response}");
    assert!(response.get("result").is_none());
}

fn register_problem(session: &mut Session, source: &str, unknown: &str, parameter: &str) {
    success(call(
        session,
        "RegisterParametricProblem",
        json!({
            "problem": PROBLEM,
            "residuals": [source],
            "unknowns": [unknown],
            "parameters": [parameter],
        }),
    ));
}

fn setup(limits: BudgetLimits, source: &str) -> Session {
    let mut session = Session::with_budget_limits(limits, 256);
    success(call(
        &mut session,
        "RegisterResidualSystem",
        json!({"system": SYSTEM, "residuals": [source], "variables": ["x", "p"]}),
    ));
    success(call(
        &mut session,
        "CompileResidualSystem",
        json!({"system": SYSTEM, "program": PROGRAM}),
    ));
    register_problem(&mut session, source, "x", "p");
    session
}

fn payload(initial: &str, lower: &str, upper: &str, radius: &str) -> Value {
    json!({
        "program": PROGRAM,
        "problem": PROBLEM,
        "family": FAMILY,
        "initial": [initial],
        "parameter_domain": [interval(lower, upper)],
        "radii": [radius],
        "precision_digits": 24,
    })
}

fn certify(session: &mut Session, payload: Value) -> Value {
    call(session, "SolveAndCertifyCompiledParametricSystem", payload)
}

fn no_family(session: &Session) {
    assert!(
        session
            .parametric_family(&ParametricFamilyId::new(FAMILY).unwrap())
            .is_none()
    );
    assert!(
        session
            .compiled_residual_system(&CompiledResidualSystemId::new(PROGRAM).unwrap())
            .is_some()
    );
}

#[test]
fn a_midpoint_solve_certifies_every_parameter_and_supplies_checked_sensitivity() {
    let mut session = setup(limits(), "2*x-p");
    session.construct("x", "999").unwrap();
    session.construct("p", "-999").unwrap();
    let before = session.budget_snapshot();
    let result = success(certify(&mut session, payload("0", "3/2", "5/2", "1/2")));
    assert_eq!(result["kind"], "parametric_family_certified");
    assert_eq!(result["evidence"], "certified_numeric");
    assert_eq!(
        result["claim"],
        "for_every_parameter_exactly_one_nonsingular_root_in_unknown_box"
    );
    assert_eq!(result["unknown_domain"], json!([interval("1/2", "3/2")]));
    assert_eq!(result["parameter_domain"], json!([interval("3/2", "5/2")]));
    assert_eq!(result["root_enclosure"], json!([interval("3/4", "5/4")]));
    assert_eq!(result["contraction_bound"], "0");
    assert_eq!(result["context_digest"], json!(session.context_digest()));
    let family = session
        .parametric_family(&ParametricFamilyId::new(FAMILY).unwrap())
        .unwrap();
    assert_eq!(family.unknown_domain()[0].radius(), &q(1, 2));
    let ParametricFamilyReceipt::RootBox(receipt) = family else {
        panic!("one uniform box");
    };
    let replay = fsym_core::verify_parametric_root_box(
        receipt.residuals(),
        receipt.unknowns(),
        receipt.parameters(),
        receipt.witness(),
        &mut fsym_budget::Unbounded,
    )
    .unwrap();
    assert_eq!(&replay, receipt);
    let after = session.budget_snapshot();
    assert!(after.verifier_pool < before.verifier_pool);
    assert!(
        after.dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
    let sensitivity = success(call(
        &mut session,
        "EncloseParametricSensitivity",
        json!({"family": FAMILY, "sensitivity": "sensitivity:solved"}),
    ));
    assert_eq!(
        sensitivity["solution_jacobian"],
        json!([[interval("1/2", "1/2")]])
    );
}

#[test]
fn fixed_fractional_parameters_keep_the_exact_domain_instead_of_the_rounded_solver_value() {
    let mut session = setup(limits(), "2*x-p");
    let result = success(certify(
        &mut session,
        payload("0", "1/10", "1/10", "1/1000"),
    ));
    assert_eq!(
        result["parameter_domain"],
        json!([interval("1/10", "1/10")])
    );
    assert_eq!(result["root_enclosure"], json!([interval("1/20", "1/20")]));
    let receipt = session
        .parametric_family(&ParametricFamilyId::new(FAMILY).unwrap())
        .unwrap();
    assert_eq!(receipt.parameter_domain()[0].midpoint(), &q(1, 10));
    assert_eq!(receipt.parameter_domain()[0].radius(), &q(0, 1));
    assert_eq!(receipt.unknown_domain()[0].radius(), &q(1, 1000));
}

#[test]
fn midpoint_success_loose_tolerances_and_singular_roots_do_not_authorize_a_family() {
    for (source, request, code) in [
        ("x-p", payload("0", "-1", "1", "1/10"), "refused"),
        // An interval division containing zero follows the existing
        // arithmetic-refusal category, without granting a family.
        ("x-1/p", payload("1", "-1", "2", "10"), "refused"),
        (
            "x^2-p",
            payload("0", "0", "0", "1"),
            "numeric_consumer_refused",
        ),
    ] {
        let mut session = setup(limits(), source);
        refused(certify(&mut session, request), code);
        no_family(&session);
    }
    let mut session = setup(limits(), "x-p");
    let mut request = payload("0", "1", "1", "1/100");
    request["options"] = json!({"max_iterations": 0, "residual_tolerance": "2"});
    refused(certify(&mut session, request), "refused");
    no_family(&session);
    success(certify(&mut session, payload("0", "1", "1", "1/10")));
}

#[test]
fn registered_problem_and_compiled_source_must_match_before_any_proposal_can_be_certified() {
    for (source, unknown, parameter) in
        [("x-p+1", "x", "p"), ("x-p", "p", "x"), ("x-p+0", "x", "p")]
    {
        let mut session = Session::with_budget_limits(limits(), 256);
        success(call(
            &mut session,
            "RegisterResidualSystem",
            json!({"system": SYSTEM, "residuals": ["x-p"], "variables": ["x", "p"]}),
        ));
        success(call(
            &mut session,
            "CompileResidualSystem",
            json!({"system": SYSTEM, "program": PROGRAM}),
        ));
        register_problem(&mut session, source, unknown, parameter);
        let before = session.budget_snapshot().verifier_pool;
        refused(
            certify(&mut session, payload("0", "-1", "1", "2")),
            "source_mismatch",
        );
        no_family(&session);
        assert_eq!(session.budget_snapshot().verifier_pool, before);
    }
}

#[test]
fn invalid_exact_controls_and_duplicate_publication_keep_existing_objects_unchanged() {
    let mut session = setup(limits(), "x-p");
    for radius in ["0", "-1", "NaN", "1/0"] {
        refused(
            certify(&mut session, payload("0", "-1", "1", radius)),
            "invalid_input",
        );
        no_family(&session);
    }
    for change in [
        json!({"precision_digits": 7}),
        json!({"parameter_domain": []}),
        json!({"parameter_domain": [interval("2", "1")]}),
        json!({"radii": []}),
        json!({"initial": []}),
        json!({"options": {"max_iterations": 1025}}),
    ] {
        let mut request = payload("0", "-1", "1", "2");
        request
            .as_object_mut()
            .unwrap()
            .extend(change.as_object().unwrap().clone());
        refused(certify(&mut session, request), "invalid_input");
        no_family(&session);
    }
    let accepted = success(certify(&mut session, payload("0", "-1", "1", "2")));
    refused(
        certify(&mut session, payload("0", "1", "2", "2")),
        "duplicate_object",
    );
    let receipt = session
        .parametric_family(&ParametricFamilyId::new(FAMILY).unwrap())
        .unwrap();
    assert_eq!(receipt.parameter_domain()[0].midpoint(), &q(0, 1));
    assert_eq!(accepted["parameter_domain"], json!([interval("-1", "1")]));
}

#[test]
fn final_publication_is_atomic_and_protected_work_cannot_borrow_ordinary_fuel() {
    let request = payload("0", "-1", "1", "2");
    let mut probe = setup(limits(), "x-p");
    let setup_cost = AMPLE - probe.budget_snapshot().verifier_pool;
    success(certify(&mut probe, request.clone()));
    let after = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut bounded = limits();
        bounded.dimensions[dimension.index()] = AMPLE - after.dimensions[dimension.index()] - 1;
        let mut session = setup(bounded, "x-p");
        refused(certify(&mut session, request.clone()), "budget_exhausted");
        no_family(&session);
        assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
    }
    for verifier_limit in [setup_cost + 1, AMPLE - after.verifier_pool - 1] {
        let mut bounded = limits();
        bounded.verifier_pool = verifier_limit;
        let mut session = setup(bounded, "x-p");
        refused(certify(&mut session, request.clone()), "budget_exhausted");
        no_family(&session);
        assert!(session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    }
}

#[test]
fn the_accepted_family_exports_replays_and_checks_sensitivity_in_a_fresh_native_process() {
    let mut session = setup(limits(), "2*x-p");
    let certified = success(certify(&mut session, payload("0", "3/2", "5/2", "1/2")));
    let artifact = success(call(
        &mut session,
        "ExportParametricFamily",
        json!({"family": FAMILY}),
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
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    for (operation, payload) in [
        (
            "RegisterParametricProblem",
            json!({
                "problem": PROBLEM,
                "residuals": ["2*x-p"],
                "unknowns": ["x"],
                "parameters": ["p"],
            }),
        ),
        (
            "ReplayParametricFamily",
            json!({
                "problem": PROBLEM,
                "family": FAMILY,
                "artifact": artifact,
                "unknown_domain": certified["unknown_domain"],
                "parameter_domain": certified["parameter_domain"],
            }),
        ),
        (
            "EncloseParametricSensitivity",
            json!({"family": FAMILY, "sensitivity": "sensitivity:replayed"}),
        ),
    ] {
        writeln!(input, "{}", frame(operation, payload)).unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        let result = success(serde_json::from_str(&line).unwrap());
        if operation == "ReplayParametricFamily" {
            assert_eq!(result["root_enclosure"], certified["root_enclosure"]);
        }
        if operation == "EncloseParametricSensitivity" {
            assert_eq!(
                result["solution_jacobian"],
                json!([[interval("1/2", "1/2")]])
            );
        }
    }
    drop(input);
    let status = child.wait_with_output().unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
}
