//! A minimizer proposes; protected independent original-source checking proves
//! the stationary point and its local classification before atomic publication.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, CompiledObjectiveId,
    CompiledStationarityOptions, Dimension, ObjectiveId, Session, StationaryPointClassification,
    StationaryPointId,
};
use fsym_core::{BigInt, BigRational};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const AMPLE: u64 = 1 << 40;
const OBJECTIVE: &str = "objective:certified-minimize";
const PROGRAM: &str = "objective_program:certified-minimize";
const CERTIFICATE: &str = "stationary:certified-minimize";

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn frame(operation: &str, mut payload: Value) -> String {
    payload["schema_version"] = json!(1);
    json!({"id":"certified-minimize", "type":operation, "payload":payload}).to_string()
}

fn call(session: &mut Session, operation: &str, payload: Value) -> Value {
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&frame(operation, payload))).unwrap();
    assert_eq!(response["id"], "certified-minimize");
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

fn register(session: &mut Session, source: &str, variables: &[&str]) {
    success(call(
        session,
        "RegisterObjective",
        json!({"objective":OBJECTIVE, "source":source, "variables":variables}),
    ));
}

fn compiled_session(limits: BudgetLimits, source: &str, variables: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits, 256);
    register(&mut session, source, variables);
    success(call(
        &mut session,
        "CompileObjective",
        json!({"objective":OBJECTIVE, "program":PROGRAM}),
    ));
    session
}

fn payload(initial: &[&str], radii: &[&str]) -> Value {
    json!({
        "program":PROGRAM, "certificate":CERTIFICATE, "initial":initial,
        "radii":radii, "precision_digits":24,
    })
}

fn certify(session: &mut Session, payload: Value) -> Value {
    call(session, "MinimizeAndCertifyCompiledObjective", payload)
}

fn no_certificate(session: &Session) {
    assert!(
        session
            .stationary_point(&StationaryPointId::new(CERTIFICATE).unwrap())
            .is_none()
    );
    assert!(
        session
            .compiled_objective(&CompiledObjectiveId::new(PROGRAM).unwrap())
            .is_some()
    );
}

#[test]
fn typed_minimization_certifies_frozen_source_in_original_coordinate_order() {
    let mut session =
        compiled_session(limits(), "3*(x-2)^2+2*(x-2)*(y+1)+4*(y+1)^2+7", &["y", "x"]);
    session.construct("x", "999").unwrap();
    session.construct("y", "-999").unwrap();
    let before = session.budget_snapshot();
    let result = session
        .certification_request(CertificationRequest::MinimizeAndCertifyCompiledObjective {
            schema_version: 1,
            program: CompiledObjectiveId::new(PROGRAM).unwrap(),
            certificate: StationaryPointId::new(CERTIFICATE).unwrap(),
            initial: vec!["6".into(), "-4".into()],
            options: CompiledStationarityOptions::default(),
            radii: vec!["1/100".into(), "0.02".into()],
            precision_digits: 24,
        })
        .unwrap();
    let CertificationResponse::StationaryPointCertified {
        objective,
        classification,
        claim,
        evidence,
        context_digest,
        ..
    } = result
    else {
        panic!("stationary certificate expected")
    };
    assert_eq!(objective, ObjectiveId::new(OBJECTIVE).unwrap());
    assert_eq!(classification, StationaryPointClassification::StrictMinimum);
    assert_eq!(claim, "unique_nonsingular_stationary_point_in_box");
    assert_eq!(evidence, "certificate_verified");
    assert_eq!(context_digest, session.context_digest());
    let receipt = session
        .stationary_point(&StationaryPointId::new(CERTIFICATE).unwrap())
        .unwrap();
    assert_eq!(receipt.variables()[0].name, "y");
    assert_eq!(receipt.variables()[1].name, "x");
    assert_eq!(receipt.witness().domain[0].radius(), &q(1, 100));
    assert_eq!(receipt.witness().domain[1].radius(), &q(1, 50));
    assert!(receipt.stationary_enclosure()[0].contains(&q(-1, 1)));
    assert!(receipt.stationary_enclosure()[1].contains(&q(2, 1)));
    assert!(receipt.objective_enclosure().contains(&q(7, 1)));
    assert_eq!(
        fsym_core::certify_stationary_point(
            receipt.objective(),
            receipt.variables(),
            receipt.witness(),
        )
        .unwrap(),
        *receipt
    );
    assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    assert!(
        session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );
}

#[test]
fn nonlinear_minimum_and_initial_maximum_or_saddle_have_exact_local_classifications() {
    for (source, variables, initial, expected) in [
        ("(x^2-1)^2", vec!["x"], vec!["0.1"], "strict_minimum"),
        ("3-x^2", vec!["x"], vec!["0"], "strict_maximum"),
        ("x^2-y^2", vec!["x", "y"], vec!["0", "0"], "saddle"),
    ] {
        let mut session = compiled_session(limits(), source, &variables);
        let result = success(certify(
            &mut session,
            payload(&initial, &vec!["1/100"; variables.len()]),
        ));
        assert_eq!(result["kind"], "stationary_point_certified");
        assert_eq!(result["classification"], expected);
        assert_eq!(
            result["claim"],
            "unique_nonsingular_stationary_point_in_box"
        );
        assert!(result.get("global_minimum").is_none());
    }
}

#[test]
fn loose_numerical_stopping_tolerance_does_not_prove_stationarity_inside_the_box() {
    let mut session = compiled_session(limits(), "(x-1)^2", &["x"]);
    let options = json!({"gradient_tolerance":"3"});
    let approximate = success(call(
        &mut session,
        "MinimizeCompiledObjective",
        json!({
            "program":PROGRAM, "initial":["0"], "options":options,
        }),
    ));
    assert_eq!(approximate["iterations"], 0);
    assert_eq!(approximate["outcome"], "approximate");
    let before = session.budget_snapshot();
    let mut request = payload(&["0"], &["1/100"]);
    request["options"] = options;
    refused(certify(&mut session, request), "refused");
    no_certificate(&session);
    assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    success(certify(&mut session, payload(&["0"], &["1/100"])));
}

#[test]
fn whole_box_source_domains_and_singular_hessians_refuse_without_publication() {
    for (source, refusal) in [
        ("log(x)^2", "domain_not_certified"),
        ("(x-1)^2+0*(1/x)", "refused"),
    ] {
        let mut session = compiled_session(limits(), source, &["x"]);
        refused(certify(&mut session, payload(&["1"], &["2"])), refusal);
        no_certificate(&session);
        success(certify(&mut session, payload(&["1"], &["1/100"])));
        refused(
            certify(&mut session, payload(&["1"], &["1/100"])),
            "duplicate_object",
        );
    }
    let mut singular = compiled_session(limits(), "x^4", &["x"]);
    let before = singular.budget_snapshot();
    refused(
        certify(&mut singular, payload(&["0"], &["1/100"])),
        "numeric_consumer_refused",
    );
    no_certificate(&singular);
    assert_eq!(
        singular.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
}

#[test]
fn invalid_controls_and_failed_descent_never_spend_protected_fuel_or_publish() {
    let mut session = compiled_session(limits(), "(x-1)^2", &["x"]);
    for radii in [
        vec![],
        vec!["0"],
        vec!["-1"],
        vec!["NaN"],
        vec!["1/0"],
        vec!["1e-3"],
        vec!["1/10", "1/10"],
    ] {
        let before = session.budget_snapshot();
        refused(
            certify(&mut session, payload(&["0"], &radii)),
            "invalid_input",
        );
        no_certificate(&session);
        assert_eq!(
            session.budget_snapshot().verifier_pool,
            before.verifier_pool
        );
    }
    for precision in [0, 7, 129, u32::MAX] {
        let mut request = payload(&["0"], &["1/10"]);
        request["precision_digits"] = json!(precision);
        refused(certify(&mut session, request), "invalid_input");
    }
    for options in [
        json!({"gradient_tolerance":"-1"}),
        json!({"max_step_norm":"0"}),
        json!({"max_iterations":1025}),
    ] {
        let mut request = payload(&["0"], &["1/10"]);
        request["options"] = options;
        refused(certify(&mut session, request), "invalid_input");
    }
    let before = session.budget_snapshot();
    let mut request = payload(&["0"], &["1/10"]);
    request["options"] = json!({"max_iterations":0});
    refused(certify(&mut session, request), "numeric_no_convergence");
    no_certificate(&session);
    assert_eq!(
        session.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
}

#[test]
fn response_admission_is_atomic_in_every_ordinary_resource_dimension() {
    let request = payload(&["1"], &["1/100"]);
    let mut probe = compiled_session(limits(), "x^2", &["x"]);
    success(certify(&mut probe, request.clone()));
    let after = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut bounded = limits();
        bounded.dimensions[dimension.index()] = AMPLE - after.dimensions[dimension.index()] - 1;
        let mut session = compiled_session(bounded, "x^2", &["x"]);
        refused(certify(&mut session, request.clone()), "budget_exhausted");
        no_certificate(&session);
        assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
    }
}

#[test]
fn protected_allowance_is_cumulative_and_cannot_borrow_ordinary_fuel() {
    let request = payload(&["1"], &["1/100"]);
    let mut probe = compiled_session(limits(), "x^2", &["x"]);
    let setup = AMPLE - probe.budget_snapshot().verifier_pool;
    success(certify(&mut probe, request.clone()));
    let spent = AMPLE - probe.budget_snapshot().verifier_pool;
    for protected in [setup + 1, spent - 1] {
        let mut bounded = limits();
        bounded.verifier_pool = protected;
        let mut session = compiled_session(bounded, "x^2", &["x"]);
        refused(certify(&mut session, request.clone()), "budget_exhausted");
        no_certificate(&session);
        assert!(session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    }
    let mut bounded = limits();
    bounded.verifier_pool = spent;
    let mut session = compiled_session(bounded, "x^2", &["x"]);
    success(certify(&mut session, request.clone()));
    assert_eq!(session.budget_snapshot().verifier_pool, 0);
    let mut next = request;
    next["certificate"] = json!("stationary:another-minimize");
    refused(certify(&mut session, next), "budget_exhausted");
    assert!(
        session
            .stationary_point(&StationaryPointId::new("stationary:another-minimize").unwrap())
            .is_none()
    );
    assert!(
        session
            .stationary_point(&StationaryPointId::new(CERTIFICATE).unwrap())
            .is_some()
    );
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
        Self {
            input: child.stdin.take(),
            output: BufReader::new(child.stdout.take().unwrap()),
            child,
        }
    }

    fn send(&mut self, operation: &str, payload: Value) -> Value {
        let request = frame(operation, payload);
        assert!(request.len() <= frankensympy::MAX_ENVELOPE_BYTES);
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut response = String::new();
        assert!(self.output.read_line(&mut response).unwrap() > 0);
        serde_json::from_str(&response).unwrap()
    }

    fn register(&mut self) {
        success(self.send(
            "RegisterObjective",
            json!({
                "objective":OBJECTIVE, "source":"(x-1)^2+7", "variables":["x"],
            }),
        ));
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
fn fresh_processes_restore_program_minimize_certify_and_replay_the_stationary_proof() {
    let mut compiler = Cli::new();
    compiler.register();
    success(compiler.send(
        "CompileObjective",
        json!({"objective":OBJECTIVE, "program":PROGRAM}),
    ));
    let artifact =
        success(compiler.send("ExportCompiledObjective", json!({"program":PROGRAM})))["artifact"]
            .as_str()
            .unwrap()
            .to_owned();
    compiler.finish();

    let mut solver = Cli::new();
    solver.register();
    success(solver.send(
        "ReplayCompiledObjective",
        json!({
            "objective":OBJECTIVE, "program":PROGRAM, "artifact":artifact,
        }),
    ));
    let accepted = success(solver.send(
        "MinimizeAndCertifyCompiledObjective",
        payload(&["0"], &["1/100"]),
    ));
    assert_eq!(accepted["classification"], "strict_minimum");
    let proof = success(solver.send(
        "ExportStationaryCertificate",
        json!({"certificate":CERTIFICATE}),
    ))["artifact"]
        .as_str()
        .unwrap()
        .to_owned();
    solver.finish();

    let mut verifier = Cli::new();
    verifier.register();
    let replayed = success(verifier.send("ReplayStationaryCertificate", json!({
        "objective":OBJECTIVE, "certificate":CERTIFICATE, "artifact":proof, "domain":accepted["domain"],
    })));
    assert_eq!(replayed, accepted);
    verifier.finish();
}
