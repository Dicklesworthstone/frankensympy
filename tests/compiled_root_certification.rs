//! Numerical proposals become portable root proofs only through original-source
//! verification on the protected pool of the same cumulative session ledger.

use frankensympy::{BudgetLimits, CompiledResidualSystemId, Dimension, RootCertificateId, Session};
use fsym_core::{BigInt, BigRational};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const AMPLE: u64 = 1 << 40;
const SYSTEM: &str = "system:certified-solve";
const PROGRAM: &str = "program:certified-solve";
const CERTIFICATE: &str = "certificate:certified-solve";

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn frame(operation: &str, mut payload: Value) -> String {
    payload["schema_version"] = json!(1);
    json!({"id":"certified-solve", "type":operation, "payload":payload}).to_string()
}

fn call(session: &mut Session, operation: &str, payload: Value) -> Value {
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&frame(operation, payload))).unwrap();
    assert_eq!(response["id"], "certified-solve");
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

fn register(session: &mut Session, sources: &[&str], variables: &[&str]) {
    success(call(
        session,
        "RegisterResidualSystem",
        json!({"system":SYSTEM, "residuals":sources, "variables":variables}),
    ));
}

fn compiled_session(limits: BudgetLimits, sources: &[&str], variables: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits, 256);
    register(&mut session, sources, variables);
    success(call(
        &mut session,
        "CompileResidualSystem",
        json!({"system":SYSTEM, "program":PROGRAM}),
    ));
    session
}

fn payload(backend: &str, initial: &[&str], radii: &[&str]) -> Value {
    json!({
        "program":PROGRAM, "certificate":CERTIFICATE,
        "initial":initial, "backend":backend,
        "radii":radii, "precision_digits":24,
    })
}

fn certify(session: &mut Session, payload: Value) -> Value {
    call(session, "SolveAndCertifyCompiledResidualSystem", payload)
}

fn no_certificate(session: &Session) {
    assert!(
        session
            .root_certificate(&RootCertificateId::new(CERTIFICATE).unwrap())
            .is_none()
    );
    assert!(
        session
            .compiled_residual_system(&CompiledResidualSystemId::new(PROGRAM).unwrap())
            .is_some()
    );
}

#[test]
fn all_backends_certify_local_and_restored_sources_in_the_original_axis_order() {
    let sources = ["x^2-2", "y^2-3"];
    let variables = ["y", "x"];
    let mut sender = compiled_session(limits(), &sources, &variables);
    let artifact = success(call(
        &mut sender,
        "ExportCompiledResidualSystem",
        json!({"program":PROGRAM}),
    ))["artifact"]
        .as_str()
        .unwrap()
        .to_owned();
    for imported in [false, true] {
        for backend in [
            "frankennumpy",
            "frankenscipy",
            "frankenscipy_analytic",
            "sparse",
        ] {
            let mut session = if imported {
                let mut session = Session::with_budget_limits(limits(), 256);
                register(&mut session, &sources, &variables);
                success(call(
                    &mut session,
                    "ReplayCompiledResidualSystem",
                    json!({"system":SYSTEM, "program":PROGRAM, "artifact":artifact}),
                ));
                session
            } else {
                compiled_session(limits(), &sources, &variables)
            };
            session.construct("x", "999").unwrap();
            session.construct("y", "-999").unwrap();
            let before = session.budget_snapshot();
            let result = success(certify(
                &mut session,
                payload(backend, &["1.5", "1"], &["1/1000", "0.002"]),
            ));
            assert_eq!(result["kind"], "root_box_certified");
            assert_eq!(result["evidence"], "certificate_verified");
            assert_eq!(result["claim"], "unique_nonsingular_real_root_in_box");
            assert_eq!(result["system"], SYSTEM);
            assert_eq!(result["certificate"], CERTIFICATE);
            assert_eq!(result["context_digest"], json!(session.context_digest()));
            let receipt = session
                .root_certificate(&RootCertificateId::new(CERTIFICATE).unwrap())
                .unwrap();
            assert_eq!(receipt.variables()[0].name, "y");
            assert_eq!(receipt.variables()[1].name, "x");
            assert_eq!(receipt.witness().domain[0].radius(), &q(1, 1000));
            assert_eq!(receipt.witness().domain[1].radius(), &q(1, 500));
            assert_eq!(receipt.witness().precision_digits, 24);
            assert_eq!(
                fsym_core::certify_root_box(
                    receipt.residuals(),
                    receipt.variables(),
                    receipt.witness(),
                )
                .unwrap(),
                *receipt
            );
            assert!(receipt.contraction_bound() < &q(1, 1));
            assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
            assert!(
                session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
                    < before.dimensions[Dimension::ComputeSteps.index()]
            );
        }
    }
}

#[test]
fn numerical_convergence_never_substitutes_for_root_existence_or_box_inclusion() {
    for (source, initial, options) in [
        ("exp(x)", "-40", json!({})),
        ("x-1", "0", json!({"residual_tolerance":"2"})),
    ] {
        let mut session = compiled_session(limits(), &[source], &["x"]);
        let approximate = success(call(
            &mut session,
            "SolveCompiledResidualSystem",
            json!({"program":PROGRAM, "backend":"frankennumpy",
                "initial":[initial], "options":options}),
        ));
        assert_eq!(approximate["iterations"], 0);
        assert_eq!(approximate["outcome"], "approximate");
        let before = session.budget_snapshot();
        let mut request = payload("frankennumpy", &[initial], &["1/100"]);
        request["options"] = options;
        refused(certify(&mut session, request), "refused");
        no_certificate(&session);
        assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    }
}

#[test]
fn whole_box_domain_and_singular_root_refusals_keep_the_handle_available() {
    for (source, refusal) in [
        ("log(x)", "domain_not_certified"),
        ("x-1+0*(1/x)", "refused"),
    ] {
        let mut session = compiled_session(limits(), &[source], &["x"]);
        refused(
            certify(&mut session, payload("sparse", &["1"], &["2"])),
            refusal,
        );
        no_certificate(&session);
        success(certify(&mut session, payload("sparse", &["1"], &["1/100"])));
        assert!(
            session
                .root_certificate(&RootCertificateId::new(CERTIFICATE).unwrap())
                .unwrap()
                .root_enclosure()[0]
                .contains(&q(1, 1))
        );
        refused(
            certify(&mut session, payload("sparse", &["1"], &["1/100"])),
            "duplicate_object",
        );
    }

    let mut singular = compiled_session(limits(), &["x^2"], &["x"]);
    let before = singular.budget_snapshot();
    refused(
        certify(&mut singular, payload("frankennumpy", &["0"], &["1/100"])),
        "numeric_consumer_refused",
    );
    no_certificate(&singular);
    assert_eq!(
        singular.budget_snapshot().verifier_pool,
        before.verifier_pool
    );
}

#[test]
fn exact_radii_precision_shape_and_backend_controls_refuse_before_verification() {
    let mut session = compiled_session(limits(), &["x-1"], &["x"]);
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
            certify(&mut session, payload("frankennumpy", &["0"], &radii)),
            "invalid_input",
        );
        no_certificate(&session);
        assert_eq!(
            session.budget_snapshot().verifier_pool,
            before.verifier_pool
        );
    }
    for precision in [0, 7, 129, u32::MAX] {
        let mut request = payload("frankennumpy", &["0"], &["1/10"]);
        request["precision_digits"] = json!(precision);
        refused(certify(&mut session, request), "invalid_input");
        no_certificate(&session);
    }
    for backend in ["frankenscipy", "frankenscipy_analytic"] {
        let mut request = payload(backend, &["0"], &["1/10"]);
        request["options"] = json!({"max_iterations":3});
        refused(certify(&mut session, request), "invalid_input");
    }
    let mut request = payload("frankennumpy", &["0"], &["1/10"]);
    request["options"] = json!({"max_iterations":0});
    refused(certify(&mut session, request), "numeric_no_convergence");
    no_certificate(&session);
    let mut rectangular = compiled_session(limits(), &["x-1", "2*x-2"], &["x"]);
    refused(
        certify(&mut rectangular, payload("sparse", &["0"], &["1/10"])),
        "unsupported_operation",
    );
    no_certificate(&rectangular);
}

#[test]
fn publication_exhaustion_is_atomic_in_every_ordinary_resource_dimension() {
    let request = payload("frankennumpy", &["1"], &["1/100"]);
    let mut probe = compiled_session(limits(), &["x^2-2"], &["x"]);
    success(certify(&mut probe, request.clone()));
    let after = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut bounded = limits();
        bounded.dimensions[dimension.index()] = AMPLE - after.dimensions[dimension.index()] - 1;
        let mut session = compiled_session(bounded, &["x^2-2"], &["x"]);
        let before = session.budget_snapshot();
        refused(certify(&mut session, request.clone()), "budget_exhausted");
        no_certificate(&session);
        assert_eq!(session.budget_snapshot().verifier_pool, after.verifier_pool);
        assert!(
            session.budget_snapshot().dimensions[dimension.index()]
                < before.dimensions[dimension.index()]
        );
    }
}

#[test]
fn ordinary_and_protected_allowances_are_cumulative_and_never_substitute_for_each_other() {
    let request = payload("frankennumpy", &["1"], &["1/100"]);
    let mut probe = compiled_session(limits(), &["x^2-2"], &["x"]);
    let setup = AMPLE - probe.budget_snapshot().verifier_pool;
    success(certify(&mut probe, request.clone()));
    let spent = AMPLE - probe.budget_snapshot().verifier_pool;
    for protected_fuel in [setup + 1, spent - 1] {
        let mut bounded = limits();
        bounded.verifier_pool = protected_fuel;
        let mut session = compiled_session(bounded, &["x^2-2"], &["x"]);
        assert_eq!(
            session.budget_snapshot().verifier_pool,
            protected_fuel - setup
        );
        refused(certify(&mut session, request.clone()), "budget_exhausted");
        no_certificate(&session);
        assert!(session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()] > 0);
    }

    let mut bounded = limits();
    bounded.verifier_pool = spent;
    let mut session = compiled_session(bounded, &["x^2-2"], &["x"]);
    success(certify(&mut session, request.clone()));
    assert_eq!(session.budget_snapshot().verifier_pool, 0);
    let mut second = request;
    second["certificate"] = json!("certificate:another-solve");
    refused(certify(&mut session, second), "budget_exhausted");
    assert!(
        session
            .root_certificate(&RootCertificateId::new("certificate:another-solve").unwrap())
            .is_none()
    );
    assert!(
        session
            .root_certificate(&RootCertificateId::new(CERTIFICATE).unwrap())
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
            "RegisterResidualSystem",
            json!({"system":SYSTEM, "residuals":["x^2-2"], "variables":["x"]}),
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
fn fresh_processes_replay_a_program_solve_certify_and_replay_only_the_root_proof() {
    let mut compiler = Cli::new();
    compiler.register();
    success(compiler.send(
        "CompileResidualSystem",
        json!({"system":SYSTEM, "program":PROGRAM}),
    ));
    let artifact = success(
        compiler.send("ExportCompiledResidualSystem", json!({"program":PROGRAM})),
    )["artifact"]
        .as_str()
        .unwrap()
        .to_owned();
    compiler.finish();

    let mut solver = Cli::new();
    solver.register();
    success(solver.send(
        "ReplayCompiledResidualSystem",
        json!({"system":SYSTEM, "program":PROGRAM, "artifact":artifact}),
    ));
    let accepted = success(solver.send(
        "SolveAndCertifyCompiledResidualSystem",
        payload("frankenscipy_analytic", &["1"], &["1/100"]),
    ));
    let proof = success(solver.send(
        "ExportRootCertificate",
        json!({"certificate":CERTIFICATE}),
    ))["artifact"]
        .as_str()
        .unwrap()
        .to_owned();
    solver.finish();

    let mut verifier = Cli::new();
    verifier.register();
    let replayed = success(verifier.send(
        "ReplayRootCertificate",
        json!({"system":SYSTEM, "certificate":CERTIFICATE, "artifact":proof}),
    ));
    assert_eq!(replayed, accepted);
    verifier.finish();
}
