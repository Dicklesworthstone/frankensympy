//! Source-bound residual fitting through the native Session and NDJSON paths.

use frankensympy::{
    BudgetLimits, CertificationRequest, CertificationResponse, CompiledObjectiveId,
    CompiledResidualSystemId, CompiledStationarityOptions, Dimension,
    MAX_SESSION_COMPILED_OBJECTIVES, MeterError, ObjectiveId, ResidualSystemId, Session,
    StationaryPointClassification, StationaryPointId,
};
use fsym_core::{BigInt, BigRational, Expr};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const AMPLE: u64 = 1 << 40;
const SOURCE: &str = "system:loss";
const RESIDUAL_PROGRAM: &str = "program:loss";
const OBJECTIVE: &str = "objective:loss";
const PROGRAM: &str = "objective_program:loss";

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn objective(name: &str) -> ObjectiveId {
    ObjectiveId::new(name).unwrap()
}

fn program(name: &str) -> CompiledObjectiveId {
    CompiledObjectiveId::new(name).unwrap()
}

fn residual_program() -> CompiledResidualSystemId {
    CompiledResidualSystemId::new(RESIDUAL_PROGRAM).unwrap()
}

fn compiled_session(
    specification: BudgetLimits,
    residuals: &[&str],
    variables: &[&str],
) -> Session {
    let mut session = Session::with_budget_limits(specification, 256);
    session
        .certification_request(CertificationRequest::RegisterResidualSystem {
            schema_version: 1,
            system: ResidualSystemId::new(SOURCE).unwrap(),
            residuals: residuals.iter().map(|source| (*source).into()).collect(),
            variables: variables
                .iter()
                .map(|variable| (*variable).into())
                .collect(),
        })
        .unwrap();
    session
        .certification_request(CertificationRequest::CompileResidualSystem {
            schema_version: 1,
            system: ResidualSystemId::new(SOURCE).unwrap(),
            program: residual_program(),
        })
        .unwrap();
    session
}

fn request_named(source: &str, compiled: &str) -> CertificationRequest {
    CertificationRequest::CompileLeastSquaresObjective {
        schema_version: 1,
        residual_program: residual_program(),
        objective: objective(source),
        program: program(compiled),
    }
}

fn request() -> CertificationRequest {
    request_named(OBJECTIVE, PROGRAM)
}

fn derive(session: &mut Session) -> CertificationResponse {
    session.certification_request(request()).unwrap()
}

fn no_derived_objects(session: &Session, source: &str, compiled: &str) {
    assert!(session.objective_problem(&objective(source)).is_none());
    assert!(session.compiled_objective(&program(compiled)).is_none());
    assert!(
        session
            .compiled_residual_system(&residual_program())
            .is_some()
    );
}

fn evaluate(
    session: &mut Session,
    compiled: &str,
    point: &[&str],
) -> Result<CertificationResponse, frankensympy::CertificationSessionError> {
    session.certification_request(CertificationRequest::EvaluateObjective {
        schema_version: 1,
        program: program(compiled),
        point: point.iter().map(|value| (*value).into()).collect(),
    })
}

fn minimize_and_certify(
    session: &mut Session,
    compiled: &str,
    initial: &str,
) -> CertificationResponse {
    session
        .certification_request(CertificationRequest::MinimizeAndCertifyCompiledObjective {
            schema_version: 1,
            program: program(compiled),
            certificate: StationaryPointId::new("stationary:loss").unwrap(),
            initial: vec![initial.into()],
            options: CompiledStationarityOptions::default(),
            radii: vec!["1/100".into()],
            precision_digits: 24,
        })
        .unwrap()
}

#[test]
fn inconsistent_residuals_produce_a_certified_nonzero_minimum_without_a_root_claim() {
    let mut session = compiled_session(limits(), &["x-1", "x-3"], &["x"]);
    let before = session.budget_snapshot();
    let CertificationResponse::ObjectiveCompiled {
        objective: source,
        program: compiled,
        evidence,
        claim,
        variables,
        context_digest,
        ..
    } = derive(&mut session)
    else {
        panic!("compiled objective response");
    };
    assert_eq!(source, objective(OBJECTIVE));
    assert_eq!(compiled, program(PROGRAM));
    assert_eq!(evidence, "certificate_verified");
    assert_eq!(
        claim,
        "formal_symbolic_gradient_hessian_and_binary64_lowering"
    );
    assert_eq!(variables[0].name, "x");
    assert_eq!(context_digest, session.context_digest());
    assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    assert!(
        session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
            < before.dimensions[Dimension::ComputeSteps.index()]
    );

    let CertificationResponse::StationaryPointCertified {
        objective: source,
        evidence,
        claim,
        classification,
        ..
    } = minimize_and_certify(&mut session, PROGRAM, "0")
    else {
        panic!("independently certified stationary point");
    };
    assert_eq!(source, objective(OBJECTIVE));
    assert_eq!(evidence, "certificate_verified");
    assert_eq!(claim, "unique_nonsingular_stationary_point_in_box");
    assert_eq!(classification, StationaryPointClassification::StrictMinimum);
    let receipt = session
        .stationary_point(&StationaryPointId::new("stationary:loss").unwrap())
        .unwrap();
    assert!(receipt.stationary_enclosure()[0].contains(&q(2, 1)));
    assert!(receipt.objective_enclosure().contains(&q(1, 1)));
    assert_eq!(
        fsym_core::certify_stationary_point(
            receipt.objective(),
            receipt.variables(),
            receipt.witness(),
        )
        .unwrap(),
        *receipt
    );
    assert_eq!(
        session
            .compiled_residual_system(&residual_program())
            .unwrap()
            .evaluate_sparse(&[2.0])
            .unwrap()
            .residuals,
        [1.0, -1.0],
        "the inconsistent equations still have nonzero residuals at the fit"
    );
}

#[test]
fn full_nonlinear_curvature_can_certify_a_local_maximum_of_the_loss() {
    let mut session = compiled_session(limits(), &["x^2-1"], &["x"]);
    derive(&mut session);
    let CertificationResponse::ObjectiveEvaluated {
        value,
        gradient,
        hessian,
        ..
    } = evaluate(&mut session, PROGRAM, &["0"]).unwrap()
    else {
        panic!("objective evaluation");
    };
    // F=(x^2-1)^2/2: F''(0)=-2, while J-transpose-J would be zero.
    assert_eq!(value, 0.5);
    assert_eq!(gradient, [0.0]);
    assert_eq!(hessian, [-2.0]);
    let CertificationResponse::StationaryPointCertified { classification, .. } =
        minimize_and_certify(&mut session, PROGRAM, "0")
    else {
        panic!("stationary certificate");
    };
    assert_eq!(classification, StationaryPointClassification::StrictMaximum);
}

#[test]
fn original_coordinate_order_duplicate_rows_and_zero_rows_survive_composition() {
    let mut session = compiled_session(limits(), &["x*y-1", "x+y-3", "0", "x+y-3"], &["y", "x"]);
    session.construct("x", "999").unwrap();
    session.construct("y", "-999").unwrap();
    derive(&mut session);
    let compiled = session.compiled_objective(&program(PROGRAM)).unwrap();
    let source = session.objective_problem(&objective(OBJECTIVE)).unwrap();
    assert_eq!(compiled.objective(), source.objective());
    assert_eq!(compiled.variables(), source.variables());
    let original = session
        .compiled_residual_system(&residual_program())
        .unwrap();
    assert_eq!(compiled.variables(), original.vars());
    let Expr::Mul(factors) = source.objective() else {
        panic!("held squared-error source");
    };
    let Expr::Add(squares) = &factors[1] else {
        panic!("ordered complete residual sum");
    };
    assert_eq!(squares.len(), 4);
    for (square, residual) in squares.iter().zip(original.residuals()) {
        let Expr::Pow(base, _) = square else {
            panic!("one square per original row");
        };
        assert_eq!(base.as_ref(), residual);
    }
    let CertificationResponse::ObjectiveEvaluated {
        value,
        gradient,
        hessian,
        ..
    } = evaluate(&mut session, PROGRAM, &["1", "2"]).unwrap()
    else {
        panic!("objective evaluation");
    };
    assert_eq!(value, 0.5);
    assert_eq!(gradient, [2.0, 1.0]);
    assert_eq!(hessian, [6.0, 5.0, 5.0, 3.0]);
    let CertificationResponse::ObjectiveHessianVectorProduct { product, .. } = session
        .certification_request(CertificationRequest::EvaluateHessianVectorProduct {
            schema_version: 1,
            program: program(PROGRAM),
            point: vec!["1".into(), "2".into()],
            direction: vec!["2".into(), "-1".into()],
        })
        .unwrap()
    else {
        panic!("Hessian-vector product");
    };
    assert_eq!(product, [7.0, 7.0]);
}

#[test]
fn retained_source_holes_survive_parent_release_export_and_independent_replay() {
    let sources = ["x-1", "0*(1/x)"];
    let mut sender = compiled_session(limits(), &sources, &["x"]);
    derive(&mut sender);
    let domain_code = evaluate(&mut sender, PROGRAM, &["0"]).unwrap_err().code();
    assert!(evaluate(&mut sender, PROGRAM, &["2"]).is_ok());
    let CertificationResponse::CompiledObjectiveExported { artifact, .. } = sender
        .certification_request(CertificationRequest::ExportCompiledObjective {
            schema_version: 1,
            program: program(PROGRAM),
        })
        .unwrap()
    else {
        panic!("compiled objective artifact");
    };
    sender
        .certification_request(CertificationRequest::ReleaseCompiledResidualSystem {
            schema_version: 1,
            program: residual_program(),
        })
        .unwrap();
    assert!(
        sender
            .compiled_residual_system(&residual_program())
            .is_none()
    );
    assert_eq!(
        evaluate(&mut sender, PROGRAM, &["0"]).unwrap_err().code(),
        domain_code
    );
    assert!(evaluate(&mut sender, PROGRAM, &["2"]).is_ok());

    // Recreate the exact expected statement from the receiver's own residuals.
    // Releasing its temporary program leaves that statement resident.
    let mut receiver = compiled_session(limits(), &sources, &["x"]);
    derive(&mut receiver);
    receiver
        .certification_request(CertificationRequest::ReleaseCompiledObjective {
            schema_version: 1,
            program: program(PROGRAM),
        })
        .unwrap();
    let before = receiver.budget_snapshot();
    receiver
        .certification_request(CertificationRequest::ReplayCompiledObjective {
            schema_version: 1,
            objective: objective(OBJECTIVE),
            program: program("objective_program:replayed"),
            artifact: artifact.clone(),
        })
        .unwrap();
    assert!(receiver.budget_snapshot().verifier_pool < before.verifier_pool);
    assert_eq!(
        evaluate(&mut receiver, "objective_program:replayed", &["0"])
            .unwrap_err()
            .code(),
        domain_code
    );
    assert!(evaluate(&mut receiver, "objective_program:replayed", &["2"]).is_ok());

    // A mathematically cancelling zero row still cannot be replaced by a
    // domain-free zero in the expected source statement.
    let mut changed = compiled_session(limits(), &["x-1", "0"], &["x"]);
    derive(&mut changed);
    let error = changed
        .certification_request(CertificationRequest::ReplayCompiledObjective {
            schema_version: 1,
            objective: objective(OBJECTIVE),
            program: program("objective_program:forged"),
            artifact,
        })
        .unwrap_err();
    assert_eq!(error.code(), "source_mismatch");
    assert!(
        changed
            .compiled_objective(&program("objective_program:forged"))
            .is_none()
    );
}

#[test]
fn unknown_sources_and_duplicate_targets_cannot_publish_half_a_pair() {
    let mut session = compiled_session(limits(), &["x-1", "x-3"], &["x"]);
    let before = session.budget_snapshot();
    let error = session
        .certification_request(CertificationRequest::CompileLeastSquaresObjective {
            schema_version: 1,
            residual_program: CompiledResidualSystemId::new("program:missing").unwrap(),
            objective: objective(OBJECTIVE),
            program: program(PROGRAM),
        })
        .unwrap_err();
    assert_eq!(error.code(), "unknown_object");
    assert_eq!(session.budget_snapshot(), before);
    no_derived_objects(&session, OBJECTIVE, PROGRAM);

    derive(&mut session);
    let original = session
        .objective_problem(&objective(OBJECTIVE))
        .unwrap()
        .clone();
    for (source, compiled) in [
        (OBJECTIVE, "objective_program:fresh"),
        ("objective:fresh", PROGRAM),
    ] {
        let before = session.budget_snapshot();
        let error = session
            .certification_request(request_named(source, compiled))
            .unwrap_err();
        assert_eq!(error.code(), "duplicate_object");
        assert_eq!(session.budget_snapshot(), before);
        assert_eq!(
            session.objective_problem(&objective(OBJECTIVE)),
            Some(&original)
        );
    }
    assert!(
        session
            .compiled_objective(&program("objective_program:fresh"))
            .is_none()
    );
    assert!(
        session
            .objective_problem(&objective("objective:fresh"))
            .is_none()
    );
    assert!(evaluate(&mut session, PROGRAM, &["2"]).is_ok());
}

#[test]
fn both_resident_caps_are_checked_before_compilation() {
    let mut source_full = compiled_session(limits(), &["x-1"], &["x"]);
    // Native Session has the existing independent 32-objective source cap.
    for index in 0..32 {
        source_full
            .certification_request(CertificationRequest::RegisterObjective {
                schema_version: 1,
                objective: objective(&format!("objective:occupied-{index}")),
                source: "x^2".into(),
                variables: vec!["x".into()],
            })
            .unwrap();
    }
    let before = source_full.budget_snapshot();
    let error = source_full.certification_request(request()).unwrap_err();
    assert_eq!(error.code(), "resource_limit_exceeded");
    assert_eq!(source_full.budget_snapshot(), before);
    no_derived_objects(&source_full, OBJECTIVE, PROGRAM);

    let mut program_full = compiled_session(limits(), &["x-1"], &["x"]);
    let existing = objective("objective:occupied");
    program_full
        .certification_request(CertificationRequest::RegisterObjective {
            schema_version: 1,
            objective: existing.clone(),
            source: "x^2".into(),
            variables: vec!["x".into()],
        })
        .unwrap();
    for index in 0..MAX_SESSION_COMPILED_OBJECTIVES {
        program_full
            .certification_request(CertificationRequest::CompileObjective {
                schema_version: 1,
                objective: existing.clone(),
                program: program(&format!("objective_program:occupied-{index}")),
            })
            .unwrap();
    }
    let before = program_full.budget_snapshot();
    let error = program_full.certification_request(request()).unwrap_err();
    assert_eq!(error.code(), "resource_limit_exceeded");
    assert_eq!(program_full.budget_snapshot(), before);
    no_derived_objects(&program_full, OBJECTIVE, PROGRAM);
}

#[test]
fn ordinary_protected_and_late_allocation_exhaustion_never_publish_partial_state() {
    let sources = ["x-1", "x-3"];
    let mut probe = compiled_session(limits(), &sources, &["x"]);
    let before = probe.budget_snapshot();
    derive(&mut probe);
    let after = probe.budget_snapshot();
    let compute = Dimension::ComputeSteps.index();
    let memory = Dimension::MemoryBytes.index();
    assert!(after.dimensions[compute] < before.dimensions[compute]);
    assert!(after.dimensions[memory] < before.dimensions[memory]);
    assert!(after.verifier_pool < before.verifier_pool);

    let mut ordinary_limits = limits();
    ordinary_limits.dimensions[compute] = AMPLE - before.dimensions[compute];
    let mut protected_limits = limits();
    protected_limits.verifier_pool = AMPLE - before.verifier_pool + 1;
    let mut memory_limits = limits();
    memory_limits.dimensions[memory] = AMPLE - after.dimensions[memory] - 1;
    for specification in [ordinary_limits, protected_limits, memory_limits] {
        let mut session = compiled_session(specification, &sources, &["x"]);
        let before_attempt = session.budget_snapshot();
        let error = session.certification_request(request()).unwrap_err();
        assert_eq!(error.code(), "budget_exhausted", "{error}");
        no_derived_objects(&session, OBJECTIVE, PROGRAM);
        let after_attempt = session.budget_snapshot();
        assert!(after_attempt.verifier_pool <= before_attempt.verifier_pool);
        for dimension in Dimension::ALL {
            assert!(
                after_attempt.dimensions[dimension.index()]
                    <= before_attempt.dimensions[dimension.index()]
            );
        }
    }
}

fn cancel_at(session: &mut Session, target: usize) -> u64 {
    let before = session.budget_snapshot();
    let mut seen = 0;
    let error = session
        .certification_request_with_checkpoint(request(), || {
            seen += 1;
            if seen == target {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(error.code(), "cancelled", "{error}");
    assert_eq!(seen, target, "a terminal host cancellation is not retried");
    no_derived_objects(session, OBJECTIVE, PROGRAM);
    before.verifier_pool - session.budget_snapshot().verifier_pool
}

#[test]
fn cancellation_reaches_generation_protected_verification_and_final_pair_publication() {
    let sources = ["x-1", "x-3"];
    let mut probe = compiled_session(limits(), &sources, &["x"]);
    let before = probe.budget_snapshot();
    let mut checkpoints = 0;
    probe
        .certification_request_with_checkpoint(request(), || {
            checkpoints += 1;
            Ok(())
        })
        .unwrap();
    let verifier_work = before.verifier_pool - probe.budget_snapshot().verifier_pool;
    assert!(checkpoints > 4);
    assert!(verifier_work > 1);

    for target in [1, checkpoints / 2, checkpoints - 1, checkpoints] {
        let mut session = compiled_session(limits(), &sources, &["x"]);
        cancel_at(&mut session, target);
        let cancelled = session.budget_snapshot();
        derive(&mut session);
        assert!(session.budget_snapshot().verifier_pool < cancelled.verifier_pool);
        assert!(
            session.budget_snapshot().dimensions[Dimension::ComputeSteps.index()]
                < cancelled.dimensions[Dimension::ComputeSteps.index()]
        );
    }

    // Locate an actual protected-work safe point without relying on how many
    // generator checkpoints a particular compiler implementation happens to use.
    let (mut lower, mut upper) = (1, checkpoints);
    let mut interrupted_verification = false;
    while lower <= upper {
        let target = lower + (upper - lower) / 2;
        let mut session = compiled_session(limits(), &sources, &["x"]);
        let used = cancel_at(&mut session, target);
        if used > 0 && used < verifier_work {
            interrupted_verification = true;
            break;
        }
        if used == 0 {
            lower = target + 1;
        } else {
            upper = target - 1;
        }
    }
    assert!(interrupted_verification);
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

    fn send(&mut self, operation: &str, mut payload: Value) -> Value {
        payload["schema_version"] = json!(1);
        let frame = json!({"id":"loss", "type":operation, "payload":payload});
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{frame}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "response before EOF"
        );
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], "loss");
        assert_eq!(response["status"], "success", "{response}");
        response["result"].clone()
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
fn fresh_ndjson_process_compiles_and_independently_certifies_the_residual_loss() {
    let mut cli = Cli::new();
    cli.send(
        "RegisterResidualSystem",
        json!({"system":SOURCE, "residuals":["x-1","x-3"], "variables":["x"]}),
    );
    cli.send(
        "CompileResidualSystem",
        json!({"system":SOURCE, "program":RESIDUAL_PROGRAM}),
    );
    let compiled = cli.send(
        "CompileLeastSquaresObjective",
        json!({"residual_program":RESIDUAL_PROGRAM, "objective":OBJECTIVE, "program":PROGRAM}),
    );
    assert_eq!(compiled["kind"], "objective_compiled");
    assert_eq!(compiled["objective"], OBJECTIVE);
    assert_eq!(compiled["program"], PROGRAM);
    assert_eq!(
        compiled["claim"],
        "formal_symbolic_gradient_hessian_and_binary64_lowering"
    );
    let certified = cli.send(
        "MinimizeAndCertifyCompiledObjective",
        json!({
            "program":PROGRAM, "certificate":"stationary:loss", "initial":["0"],
            "radii":["1/100"], "precision_digits":24,
        }),
    );
    assert_eq!(certified["kind"], "stationary_point_certified");
    assert_eq!(certified["classification"], "strict_minimum");
    assert_eq!(
        certified["claim"],
        "unique_nonsingular_stationary_point_in_box"
    );
    assert!(certified.get("global_minimum").is_none());
    assert!(certified.get("root_count").is_none());
    cli.finish();
}
