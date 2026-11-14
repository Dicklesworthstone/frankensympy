//! Portable candidates bind the receiving statement before independent replay.

use frankensympy::{
    CertificationError, MAX_COMPILED_PROGRAM_ARTIFACT_BYTES, ObjectiveProblem, ProgramReplayError,
    ProgramTransportError, ResidualProblem, Session, export_objective_program,
    export_residual_program, replay_objective_program, replay_residual_program,
};
use fsym_budget::{Budget, BudgetLimits, Dimension, Unbounded};
use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompiledExpr, CompiledOp};
use fsym_calculus::program_artifact::{ProgramAdmissionError, ProgramArtifactError};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_core::{Expr, Symbol, SymbolIdentity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json, value::RawValue};

fn residual_problem(sources: &[&str], variables: &[&str]) -> ResidualProblem {
    ResidualProblem::parse_sources(
        &sources.iter().map(|s| (*s).into()).collect::<Vec<_>>(),
        &variables.iter().map(|s| (*s).into()).collect::<Vec<_>>(),
        &mut Unbounded,
    )
    .unwrap()
}

fn residual_program(problem: &ResidualProblem) -> VerifiedResidualSystem {
    VerifiedResidualSystem::try_compile(problem.residuals(), problem.variables()).unwrap()
}

fn objective_problem(source: &str) -> ObjectiveProblem {
    ObjectiveProblem::parse_source(source, &["x".into(), "y".into()], &mut Unbounded).unwrap()
}

fn objective_program(problem: &ObjectiveProblem) -> VerifiedObjective {
    VerifiedObjective::try_compile(problem.objective(), problem.variables()).unwrap()
}

#[derive(Serialize)]
struct CandidateWire<'a, T> {
    schema_version: u32,
    kind: &'static str,
    candidate: &'a T,
}

#[derive(Serialize)]
struct ArtifactWire<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    candidate: &'a RawValue,
}

// Construct adversarial but canonical wire, so failure must come from source
// binding or mathematical admission rather than incidental JSON formatting.
fn forged<T: Serialize>(candidate_kind: &'static str, candidate: &T) -> Vec<u8> {
    let candidate = serde_json::to_string(&CandidateWire {
        schema_version: 1,
        kind: candidate_kind,
        candidate,
    })
    .unwrap();
    let candidate: &RawValue = serde_json::from_str(&candidate).unwrap();
    serde_json::to_vec(&ArtifactWire {
        schema_version: 1,
        context_digest: [3; 32],
        candidate,
    })
    .unwrap()
}

#[test]
fn replay_restores_both_program_kinds_and_hand_derived_numeric_products() {
    let problem = residual_problem(&["x^2+y", "x*y+sin(x)"], &["x", "y"]);
    let original = residual_program(&problem);
    let bytes = export_residual_program(&original, [3; 32], &mut Unbounded).unwrap();
    let replayed =
        replay_residual_program(&bytes, &problem, [3; 32], &mut Unbounded).unwrap();
    assert_eq!(original.receipt(), replayed.receipt());
    assert_eq!(
        export_residual_program(&replayed, [3; 32], &mut Unbounded).unwrap(),
        bytes
    );
    let values = replayed
        .evaluate_sparse_with_meter(&[2.0, 3.0], &mut Unbounded)
        .unwrap();
    assert_eq!(values.residuals, [7.0, 6.0 + 2.0f64.sin()]);
    assert_eq!(values.jacobian, [4.0, 1.0, 3.0 + 2.0f64.cos(), 2.0]);
    let forward = replayed
        .jacobian_vector_product(&[2.0, 3.0], &[1.0, 3.0])
        .unwrap();
    assert_eq!(forward, [7.0, 9.0 + 2.0f64.cos()]);
    let transpose = replayed
        .jacobian_transpose_vector_product(&[2.0, 3.0], &[2.0, -1.0])
        .unwrap();
    assert!((transpose[0] - (5.0 - 2.0f64.cos())).abs() < 1e-12);
    assert_eq!(transpose[1], 0.0);

    let problem = objective_problem("x^2+x*y+y^2");
    let original = objective_program(&problem);
    let bytes = export_objective_program(&original, [3; 32], &mut Unbounded).unwrap();
    let replayed = replay_objective_program(&bytes, &problem, [3; 32], &mut Unbounded).unwrap();
    assert_eq!(original.receipt(), replayed.receipt());
    assert_eq!(
        export_objective_program(&replayed, [3; 32], &mut Unbounded).unwrap(),
        bytes
    );
    let value = replayed.evaluate(&[2.0, 3.0]).unwrap();
    assert_eq!(value.value, 19.0);
    assert_eq!(value.gradient, [7.0, 8.0]);
    assert_eq!(value.hessian, [2.0, 1.0, 1.0, 2.0]);
    assert_eq!(
        replayed
            .hessian_vector_product(&[2.0, 3.0], &[1.0, 3.0])
            .unwrap(),
        [5.0, 7.0]
    );
}

#[test]
fn receiver_binds_exact_sources_axis_order_typed_identity_and_context() {
    let problem = residual_problem(&["x^2+y", "x*y"], &["x", "y"]);
    let bytes =
        export_residual_program(&residual_program(&problem), [3; 32], &mut Unbounded)
            .unwrap();
    for other in [
        residual_problem(&["x^2+y+1", "x*y"], &["x", "y"]),
        residual_problem(&["x*y", "x^2+y"], &["x", "y"]),
        residual_problem(&["x^2+y", "x*y"], &["y", "x"]),
    ] {
        assert!(matches!(
            replay_residual_program(&bytes, &other, [3; 32], &mut Unbounded),
            Err(ProgramReplayError::Admission(ProgramAdmissionError::SourceMismatch(_)))
        ));
    }
    assert!(matches!(
        replay_residual_program(&bytes, &problem, [4; 32], &mut Unbounded),
        Err(ProgramReplayError::Transport(ProgramTransportError::Envelope(
            CertificationError::UniverseMismatch { .. }
        )))
    ));

    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([7; 32]));
    let typed_problem =
        ResidualProblem::new(vec![Expr::Sym(typed.clone())], vec![typed], &mut Unbounded).unwrap();
    let bytes =
        export_residual_program(&residual_program(&typed_problem), [3; 32], &mut Unbounded)
            .unwrap();
    assert!(
        replay_residual_program(&bytes, &typed_problem, [3; 32], &mut Unbounded).is_ok()
    );
    assert!(matches!(
        replay_residual_program(
            &bytes,
            &residual_problem(&["x"], &["x"]),
            [3; 32],
            &mut Unbounded
        ),
        Err(ProgramReplayError::Admission(ProgramAdmissionError::SourceMismatch(_)))
    ));

    let objective = objective_problem("x^2+x*y+y^2");
    let bytes =
        export_objective_program(&objective_program(&objective), [3; 32], &mut Unbounded).unwrap();
    assert!(matches!(
        replay_objective_program(
            &bytes,
            &objective_problem("x^2+x*y+y^2+1"),
            [3; 32],
            &mut Unbounded
        ),
        Err(ProgramReplayError::Admission(ProgramAdmissionError::SourceMismatch(_)))
    ));
    assert!(matches!(
        replay_residual_program(&bytes, &problem, [3; 32], &mut Unbounded),
        Err(ProgramReplayError::Transport(ProgramTransportError::Candidate(
            ProgramArtifactError::WrongKind { .. }
        )))
    ));
}

#[test]
fn canonical_inner_source_derivative_and_bytecode_forgery_cannot_gain_admission() {
    let problem = residual_problem(&["x^2"], &["x"]);
    let original = residual_program(&problem);
    for mutation in 0..3 {
        let mut candidate = original.sparse_jacobian().clone();
        match mutation {
            0 => candidate.residuals[0] = fsym_core::parse("x^3").unwrap(),
            1 => candidate.entries[0].symbolic_deriv = Expr::from_i64(999),
            _ => {
                candidate.entries[0].compiled_deriv =
                    CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(999.0)]).unwrap()
            }
        }
        let bytes = forged("residual", &candidate);
        let error =
            replay_residual_program(&bytes, &problem, [3; 32], &mut Unbounded).unwrap_err();
        if mutation == 0 {
            assert!(matches!(
                error,
                ProgramReplayError::Admission(ProgramAdmissionError::SourceMismatch(_))
            ));
        } else {
            assert!(
                matches!(error, ProgramReplayError::Admission(ProgramAdmissionError::Residual(_))),
                "{error}"
            );
        }
    }
    let problem = objective_problem("x^2+x*y+y^2");
    let original = objective_program(&problem);
    let mut candidate = original.candidate().clone();
    candidate.compiled_objective =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(99.0)]).unwrap();
    let bytes = forged("objective", &candidate);
    assert!(matches!(
        replay_objective_program(&bytes, &problem, [3; 32], &mut Unbounded),
        Err(ProgramReplayError::Admission(ProgramAdmissionError::Objective(_)))
    ));
}

#[test]
fn transport_keeps_original_domains_and_singular_derivatives_even_for_zero_products() {
    for source in ["0*(1/x)", "sqrt(x)"] {
        let problem = residual_problem(&[source], &["x"]);
        let bytes =
            export_residual_program(&residual_program(&problem), [3; 32], &mut Unbounded)
                .unwrap();
        let replayed =
            replay_residual_program(&bytes, &problem, [3; 32], &mut Unbounded).unwrap();
        assert!(
            replayed
                .evaluate_sparse_with_meter(&[4.0], &mut Unbounded)
                .is_ok()
        );
        assert!(
            replayed
                .evaluate_sparse_with_meter(&[0.0], &mut Unbounded)
                .is_err()
        );
        assert!(replayed.jacobian_vector_product(&[0.0], &[0.0]).is_err());
        assert!(
            replayed
                .jacobian_transpose_vector_product(&[0.0], &[0.0])
                .is_err()
        );
        let objective = ObjectiveProblem::new(
            problem.residuals()[0].clone(),
            problem.variables().to_vec(),
            &mut Unbounded,
        )
        .unwrap();
        let bytes =
            export_objective_program(&objective_program(&objective), [3; 32], &mut Unbounded)
                .unwrap();
        let replayed =
            replay_objective_program(&bytes, &objective, [3; 32], &mut Unbounded).unwrap();
        assert!(replayed.evaluate(&[4.0]).is_ok());
        assert!(replayed.hessian_vector_product(&[0.0], &[0.0]).is_err());
    }
}

#[test]
fn outer_wire_rejects_unknown_duplicate_noncanonical_oversized_and_unfunded_inputs() {
    let problem = residual_problem(&["x"], &["x"]);
    let bytes =
        export_residual_program(&residual_program(&problem), [3; 32], &mut Unbounded)
            .unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    for malformed in [
        format!(" {text}"),
        text.replacen("{", "{\"accepted\":true,", 1),
        text.replacen("{", "{\"schema_version\":1,", 1),
        text.replacen("\"schema_version\":1", "\"schema_version\":2", 1),
    ] {
        assert!(
            replay_residual_program(
                malformed.as_bytes(),
                &problem,
                [3; 32],
                &mut Unbounded
            )
            .is_err()
        );
    }
    assert!(matches!(
        replay_residual_program(
            &vec![b' '; MAX_COMPILED_PROGRAM_ARTIFACT_BYTES + 1],
            &problem,
            [3; 32],
            &mut Unbounded
        ),
        Err(ProgramReplayError::Transport(ProgramTransportError::Envelope(
            CertificationError::ByteLimit { .. }
        )))
    ));
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(1 << 40, 1 << 40);
        limits.dimensions[dimension.index()] = 0;
        let mut budget = Budget::new(limits);
        assert!(replay_residual_program(&bytes, &problem, [3; 32], &mut budget).is_err());
    }
}

#[derive(Deserialize)]
struct ArtifactContext {
    context_digest: [u8; 32],
}

fn session_success(session: &mut Session, operation: &str, mut payload: Value) -> Value {
    payload["schema_version"] = json!(1);
    let frame = json!({"type": operation, "payload": payload});
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
    assert_eq!(response["status"], "success", "{response}");
    response["result"].clone()
}

fn interop_session() -> Session {
    Session::with_budget_limits(BudgetLimits::uniform(1 << 40, 1 << 40), 256)
}

#[test]
fn session_and_native_replay_interoperate_in_both_directions_for_both_kinds() {
    let sources = ["x^2+y", "x*y"];
    let variables = ["x", "y"];
    let expected = residual_problem(&sources, &variables);
    let mut sender = interop_session();
    session_success(
        &mut sender,
        "RegisterResidualSystem",
        json!({"system":"system:interop", "residuals":sources, "variables":variables}),
    );
    session_success(
        &mut sender,
        "CompileResidualSystem",
        json!({"system":"system:interop", "program":"program:sender"}),
    );
    let exported = session_success(
        &mut sender,
        "ExportCompiledResidualSystem",
        json!({"program":"program:sender"}),
    );
    let artifact = exported["artifact"].as_str().unwrap();
    let context = serde_json::from_str::<ArtifactContext>(artifact)
        .unwrap()
        .context_digest;
    // A Session export is consumed directly by the complete native facade.
    let native =
        replay_residual_program(artifact.as_bytes(), &expected, context, &mut Unbounded).unwrap();
    let values = native
        .evaluate_sparse_with_meter(&[2.0, 3.0], &mut Unbounded)
        .unwrap();
    assert_eq!(values.residuals, [7.0, 6.0]);
    assert_eq!(values.jacobian, [4.0, 1.0, 3.0, 2.0]);

    // Independently compiled native evidence travels in the same wire to a
    // fresh Session, under the receiving problem's actual context digest.
    let native_bytes =
        export_residual_program(&residual_program(&expected), context, &mut Unbounded).unwrap();
    let mut receiver = interop_session();
    session_success(
        &mut receiver,
        "RegisterResidualSystem",
        json!({"system":"system:interop", "residuals":sources, "variables":variables}),
    );
    let replayed = session_success(
        &mut receiver,
        "ReplayCompiledResidualSystem",
        json!({
            "system":"system:interop", "program":"program:from-native",
            "artifact":String::from_utf8(native_bytes).unwrap()
        }),
    );
    assert_eq!(replayed["evidence"], "certificate_verified");
    assert_eq!(replayed["kind"], "residual_system_compiled");
    let evaluated = session_success(
        &mut receiver,
        "EvaluateResidualJacobian",
        json!({"program":"program:from-native", "point":["2","3"]}),
    );
    assert_eq!(evaluated["outcome"], "approximate");
    assert_eq!(evaluated["numeric_model"], "binary64");
    assert_eq!(evaluated["residuals"], json!([7.0, 6.0]));
    assert_eq!(evaluated["jacobian"], json!([4.0, 1.0, 3.0, 2.0]));

    let source = "x^2+x*y+y^2";
    let expected = objective_problem(source);
    session_success(
        &mut sender,
        "RegisterObjective",
        json!({"objective":"objective:interop", "source":source, "variables":variables}),
    );
    session_success(
        &mut sender,
        "CompileObjective",
        json!({"objective":"objective:interop", "program":"objective_program:sender"}),
    );
    let exported = session_success(
        &mut sender,
        "ExportCompiledObjective",
        json!({"program":"objective_program:sender"}),
    );
    let artifact = exported["artifact"].as_str().unwrap();
    let context = serde_json::from_str::<ArtifactContext>(artifact)
        .unwrap()
        .context_digest;
    let native =
        replay_objective_program(artifact.as_bytes(), &expected, context, &mut Unbounded).unwrap();
    let value = native.evaluate(&[2.0, 3.0]).unwrap();
    assert_eq!(value.value, 19.0);
    assert_eq!(value.gradient, [7.0, 8.0]);
    assert_eq!(value.hessian, [2.0, 1.0, 1.0, 2.0]);

    let native_bytes =
        export_objective_program(&objective_program(&expected), context, &mut Unbounded).unwrap();
    session_success(
        &mut receiver,
        "RegisterObjective",
        json!({"objective":"objective:interop", "source":source, "variables":variables}),
    );
    let replayed = session_success(
        &mut receiver,
        "ReplayCompiledObjective",
        json!({
            "objective":"objective:interop", "program":"objective_program:from-native",
            "artifact":String::from_utf8(native_bytes).unwrap()
        }),
    );
    assert_eq!(replayed["evidence"], "certificate_verified");
    assert_eq!(replayed["kind"], "objective_compiled");
    let product = session_success(
        &mut receiver,
        "EvaluateHessianVectorProduct",
        json!({
            "program":"objective_program:from-native",
            "point":["2","3"], "direction":["1","3"]
        }),
    );
    assert_eq!(product["outcome"], "approximate");
    assert_eq!(product["numeric_model"], "binary64");
    assert_eq!(product["product"], json!([5.0, 7.0]));
}
