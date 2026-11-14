//! Portable programs acquire fresh source-bound authority in the receiving session.

use frankensympy::certification_session::MAX_SESSION_ROOT_ARTIFACT_BYTES;
use frankensympy::{
    BudgetLimits, CompiledObjectiveId, CompiledResidualSystemId, Dimension, Session,
};
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_budget::Unbounded;
use fsym_calculus::compile::{CompiledExpr, CompiledOp};
use fsym_calculus::program_artifact::{decode_objective_candidate, decode_residual_candidate};
use fsym_calculus::sparse_hessian::ObjectiveCandidate;
use fsym_calculus::sparse_jacobian::SparseJacobian;
use fsym_core::{Expr, Symbol, SymbolIdentity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json, value::RawValue};
use std::collections::HashMap;

const AMPLE: u64 = 1 << 40;

fn limits() -> BudgetLimits {
    BudgetLimits::uniform(AMPLE, AMPLE)
}

fn session() -> Session {
    Session::with_budget_limits(limits(), 256)
}

fn call(session: &mut Session, operation: &str, mut payload: Value) -> Value {
    payload["schema_version"] = json!(1);
    let frame = json!({"type": operation, "payload": payload});
    serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap()
}

fn success(response: Value) -> Value {
    assert_eq!(response["status"], "success", "{response}");
    response["result"].clone()
}

fn register_residuals(session: &mut Session, sources: &[&str], variables: &[&str]) {
    success(call(
        session,
        "RegisterResidualSystem",
        json!({"system":"system:receiver", "residuals":sources, "variables":variables}),
    ));
}

fn register_objective(session: &mut Session, source: &str, variables: &[&str]) {
    success(call(
        session,
        "RegisterObjective",
        json!({"objective":"objective:receiver", "source":source, "variables":variables}),
    ));
}

fn compile_residual(session: &mut Session) -> Value {
    success(call(
        session,
        "CompileResidualSystem",
        json!({"system":"system:receiver", "program":"program:checked"}),
    ))
}

fn compile_objective(session: &mut Session) -> Value {
    success(call(
        session,
        "CompileObjective",
        json!({"objective":"objective:receiver", "program":"objective_program:checked"}),
    ))
}

fn export_residual(session: &mut Session) -> String {
    let exported = success(call(
        session,
        "ExportCompiledResidualSystem",
        json!({"program":"program:checked"}),
    ));
    assert_eq!(exported["kind"], "compiled_residual_system_exported");
    exported["artifact"].as_str().unwrap().to_owned()
}

fn export_objective(session: &mut Session) -> String {
    let exported = success(call(
        session,
        "ExportCompiledObjective",
        json!({"program":"objective_program:checked"}),
    ));
    assert_eq!(exported["kind"], "compiled_objective_exported");
    exported["artifact"].as_str().unwrap().to_owned()
}

fn replay_residual(session: &mut Session, artifact: &str) -> Value {
    call(
        session,
        "ReplayCompiledResidualSystem",
        json!({"system":"system:receiver", "program":"program:received", "artifact":artifact}),
    )
}

fn replay_objective(session: &mut Session, artifact: &str) -> Value {
    call(
        session,
        "ReplayCompiledObjective",
        json!({"objective":"objective:receiver", "program":"objective_program:received", "artifact":artifact}),
    )
}

fn residual_absent(session: &Session) {
    assert!(
        session
            .compiled_residual_system(&CompiledResidualSystemId::new("program:received").unwrap())
            .is_none()
    );
}

fn objective_absent(session: &Session) {
    assert!(
        session
            .compiled_objective(&CompiledObjectiveId::new("objective_program:received").unwrap())
            .is_none()
    );
}

fn assert_approximate(result: &Value) {
    assert_eq!(result["outcome"], "approximate");
    assert_eq!(result["numeric_model"], "binary64");
    assert!(result.get("evidence").is_none());
    assert!(result.get("claim").is_none());
}

#[derive(Deserialize)]
struct RawArtifact<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    #[serde(borrow)]
    candidate: &'a RawValue,
}

#[derive(Serialize)]
struct NativeEnvelope<'a, T> {
    schema_version: u32,
    kind: &'static str,
    candidate: &'a T,
}

#[derive(Serialize)]
struct Artifact<'a, T> {
    schema_version: u32,
    context_digest: [u8; 32],
    candidate: NativeEnvelope<'a, T>,
}

fn residual_mutant(original: &str, mutate: impl FnOnce(&mut SparseJacobian)) -> String {
    let original: RawArtifact<'_> = serde_json::from_str(original).unwrap();
    let mut candidate =
        decode_residual_candidate(original.candidate.get().as_bytes(), &mut Unbounded).unwrap();
    mutate(&mut candidate);
    serde_json::to_string(&Artifact {
        schema_version: original.schema_version,
        context_digest: original.context_digest,
        candidate: NativeEnvelope {
            schema_version: 1,
            kind: "residual",
            candidate: &candidate,
        },
    })
    .unwrap()
}

fn objective_mutant(original: &str, mutate: impl FnOnce(&mut ObjectiveCandidate)) -> String {
    let original: RawArtifact<'_> = serde_json::from_str(original).unwrap();
    let mut candidate =
        decode_objective_candidate(original.candidate.get().as_bytes(), &mut Unbounded).unwrap();
    mutate(&mut candidate);
    serde_json::to_string(&Artifact {
        schema_version: original.schema_version,
        context_digest: original.context_digest,
        candidate: NativeEnvelope {
            schema_version: 1,
            kind: "objective",
            candidate: &candidate,
        },
    })
    .unwrap()
}

#[test]
fn residual_program_replays_in_fresh_session_and_reuses_all_sparse_products() {
    let sources = ["x^2+y", "sin(y)", "7"];
    let variables = ["y", "x"];
    let mut sender = session();
    register_residuals(&mut sender, &sources, &variables);
    let compiled = compile_residual(&mut sender);
    let before_export = sender.budget_snapshot();
    let artifact = export_residual(&mut sender);
    assert_eq!(
        sender.budget_snapshot().verifier_pool,
        before_export.verifier_pool
    );
    let raw: Value = serde_json::from_str(&artifact).unwrap();
    assert!(raw.get("receipt").is_none());
    assert!(
        raw["candidate"]["candidate"]
            .get("shared_evaluation")
            .is_none()
    );
    assert!(artifact.len() <= MAX_SESSION_ROOT_ARTIFACT_BYTES);

    let mut receiver = session();
    register_residuals(&mut receiver, &sources, &variables);
    receiver.construct("x", "999").unwrap();
    let before = receiver.budget_snapshot();
    let replayed = success(replay_residual(&mut receiver, &artifact));
    assert_eq!(replayed["kind"], "residual_system_compiled");
    assert_eq!(replayed["program"], "program:received");
    for field in [
        "variables",
        "jacobian_pattern",
        "proofs_verified",
        "claim",
        "context_digest",
    ] {
        assert_eq!(replayed[field], compiled[field], "{field}");
    }
    assert_eq!(replayed["evidence"], "certificate_verified");
    assert!(receiver.budget_snapshot().verifier_pool < before.verifier_pool);
    let protected = receiver.budget_snapshot().verifier_pool;
    let evaluated = success(call(
        &mut receiver,
        "EvaluateResidualJacobian",
        json!({"program":"program:received", "point":["0","2"]}),
    ));
    assert_approximate(&evaluated);
    assert_eq!(evaluated["residuals"], json!([4.0, 0.0, 7.0]));
    assert_eq!(
        evaluated["jacobian_pattern"],
        json!([[0, 0], [0, 1], [1, 0]])
    );
    assert_eq!(evaluated["jacobian"], json!([1.0, 4.0, 1.0]));
    let forward = success(call(
        &mut receiver,
        "EvaluateJacobianVectorProduct",
        json!({"program":"program:received", "point":["0","2"], "direction":["3","-2"]}),
    ));
    assert_approximate(&forward);
    assert_eq!(forward["product"], json!([-5.0, 3.0, 0.0]));
    let transpose = success(call(
        &mut receiver,
        "EvaluateJacobianTransposeVectorProduct",
        json!({"program":"program:received", "point":["0","2"], "weights":["2","-3","11"]}),
    ));
    assert_approximate(&transpose);
    assert_eq!(transpose["product"], json!([-1.0, 8.0]));
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    let reexported = success(call(
        &mut receiver,
        "ExportCompiledResidualSystem",
        json!({"program":"program:received"}),
    ));
    assert_eq!(reexported["artifact"], artifact);
}

#[test]
fn objective_program_replays_with_new_receipt_and_no_cross_call_numeric_cache() {
    let source = "x^2+x*y+3*y^2";
    let mut sender = session();
    register_objective(&mut sender, source, &["x", "y"]);
    let compiled = compile_objective(&mut sender);
    let artifact = export_objective(&mut sender);
    let mut receiver = session();
    register_objective(&mut receiver, source, &["x", "y"]);
    let before = receiver.budget_snapshot();
    let replayed = success(replay_objective(&mut receiver, &artifact));
    assert_eq!(replayed["kind"], "objective_compiled");
    for field in [
        "variables",
        "hessian_pattern",
        "gradient_proofs_verified",
        "hessian_proofs_verified",
        "zero_entries_checked",
        "claim",
        "context_digest",
    ] {
        assert_eq!(replayed[field], compiled[field], "{field}");
    }
    assert!(receiver.budget_snapshot().verifier_pool < before.verifier_pool);
    let protected = receiver.budget_snapshot().verifier_pool;
    for (point, value, gradient) in [
        (["2", "-1"], 5.0, [3.0, -4.0]),
        (["-1", "2"], 11.0, [0.0, 11.0]),
        (["2", "-1"], 5.0, [3.0, -4.0]),
    ] {
        let evaluated = success(call(
            &mut receiver,
            "EvaluateObjective",
            json!({"program":"objective_program:received", "point":point}),
        ));
        assert_approximate(&evaluated);
        assert_eq!(evaluated["value"], value);
        assert_eq!(evaluated["gradient"], json!(gradient));
        assert_eq!(evaluated["hessian"], json!([2.0, 1.0, 1.0, 6.0]));
    }
    let product = success(call(
        &mut receiver,
        "EvaluateHessianVectorProduct",
        json!({"program":"objective_program:received", "point":["2","-1"], "direction":["3","-2"]}),
    ));
    assert_approximate(&product);
    assert_eq!(product["product"], json!([4.0, -9.0]));
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    let exported = success(call(
        &mut receiver,
        "ExportCompiledObjective",
        json!({"program":"objective_program:received"}),
    ));
    assert_eq!(exported["artifact"], artifact);
}

#[test]
fn original_domains_survive_replay_even_for_zero_direction_or_weights() {
    let source = "x^2+0*(1/x)";
    let mut sender = session();
    register_residuals(&mut sender, &[source], &["x"]);
    compile_residual(&mut sender);
    let residual = export_residual(&mut sender);
    register_objective(&mut sender, source, &["x"]);
    compile_objective(&mut sender);
    let objective = export_objective(&mut sender);
    let mut receiver = session();
    register_residuals(&mut receiver, &[source], &["x"]);
    register_objective(&mut receiver, source, &["x"]);
    success(replay_residual(&mut receiver, &residual));
    success(replay_objective(&mut receiver, &objective));
    for (operation, payload) in [
        (
            "EvaluateResidualJacobian",
            json!({"program":"program:received", "point":["0"]}),
        ),
        (
            "EvaluateJacobianVectorProduct",
            json!({"program":"program:received", "point":["0"], "direction":["0"]}),
        ),
        (
            "EvaluateJacobianTransposeVectorProduct",
            json!({"program":"program:received", "point":["0"], "weights":["0"]}),
        ),
        (
            "EvaluateObjective",
            json!({"program":"objective_program:received", "point":["0"]}),
        ),
        (
            "EvaluateHessianVectorProduct",
            json!({"program":"objective_program:received", "point":["0"], "direction":["0"]}),
        ),
    ] {
        let response = call(&mut receiver, operation, payload);
        assert_eq!(response["code"], "numeric_evaluation_refused", "{response}");
        assert!(response.get("result").is_none());
    }
    let evaluated = success(call(
        &mut receiver,
        "EvaluateObjective",
        json!({"program":"objective_program:received", "point":["2"]}),
    ));
    assert_eq!(evaluated["value"], 4.0);
    assert_eq!(evaluated["gradient"], json!([4.0]));
}

#[test]
fn replay_binds_raw_sources_ordered_axes_and_context_before_any_publication() {
    let mut sender = session();
    register_residuals(&mut sender, &["x^2+y", "sin(y)"], &["y", "x"]);
    compile_residual(&mut sender);
    let artifact = export_residual(&mut sender);
    for (sources, variables) in [
        (["x^3+y", "sin(y)"], ["y", "x"]),
        (["sin(y)", "x^2+y"], ["y", "x"]),
        (["x^2+y", "sin(y)"], ["x", "y"]),
        (["x^2+y+0*(1/x)", "sin(y)"], ["y", "x"]),
    ] {
        let mut receiver = session();
        register_residuals(&mut receiver, &sources, &variables);
        let rejected = replay_residual(&mut receiver, &artifact);
        assert_eq!(rejected["code"], "source_mismatch", "{rejected}");
        residual_absent(&receiver);
    }
    let identity_mutant = residual_mutant(&artifact, |candidate| {
        candidate.vars[0] =
            Symbol::with_identity("y", SymbolIdentity::from_assumptions_digest([31; 32]));
    });
    let mut receiver = session();
    register_residuals(&mut receiver, &["x^2+y", "sin(y)"], &["y", "x"]);
    assert_eq!(
        replay_residual(&mut receiver, &identity_mutant)["code"],
        "source_mismatch"
    );
    residual_absent(&receiver);
    let context = ImmutableAssumptionsSnapshot::empty()
        .derive_child(
            HashMap::new(),
            HashMap::new(),
            "different receiving context",
        )
        .unwrap();
    let mut receiver = Session::with_context_and_budget_limits(context, limits(), 256);
    register_residuals(&mut receiver, &["x^2+y", "sin(y)"], &["y", "x"]);
    let rejected = replay_residual(&mut receiver, &artifact);
    assert_eq!(rejected["code"], "universe_mismatch", "{rejected}");
    residual_absent(&receiver);

    register_objective(&mut sender, "x^2+3*y^2", &["x", "y"]);
    compile_objective(&mut sender);
    let artifact = export_objective(&mut sender);
    for (source, variables) in [
        ("x^2+4*y^2", ["x", "y"]),
        ("x^2+3*y^2", ["y", "x"]),
        ("x^2+3*y^2+0*(1/x)", ["x", "y"]),
    ] {
        let mut receiver = session();
        register_objective(&mut receiver, source, &variables);
        let rejected = replay_objective(&mut receiver, &artifact);
        assert_eq!(rejected["code"], "source_mismatch", "{rejected}");
        objective_absent(&receiver);
    }
}

#[test]
fn canonical_derivative_bytecode_and_sparsity_forgery_never_becomes_a_program() {
    let mut sender = session();
    register_residuals(&mut sender, &["x^2"], &["x"]);
    compile_residual(&mut sender);
    let original = export_residual(&mut sender);
    let mutants = [
        residual_mutant(&original, |candidate| {
            candidate.entries[0].symbolic_deriv = Expr::from_i64(0);
        }),
        residual_mutant(&original, |candidate| {
            candidate.entries[0].compiled_deriv =
                CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
        }),
        residual_mutant(&original, |candidate| {
            candidate.entries.clear();
            candidate.pattern.nonzeros.clear();
            candidate.pattern.nnz = 0;
        }),
    ];
    let mut receiver = session();
    register_residuals(&mut receiver, &["x^2"], &["x"]);
    for mutant in mutants {
        let before = receiver.budget_snapshot();
        let rejected = replay_residual(&mut receiver, &mutant);
        assert_eq!(rejected["code"], "verification_rejected", "{rejected}");
        assert!(receiver.budget_snapshot().verifier_pool < before.verifier_pool);
        residual_absent(&receiver);
    }
    success(replay_residual(&mut receiver, &original));

    register_objective(&mut sender, "x^2+x*y+3*y^2", &["x", "y"]);
    compile_objective(&mut sender);
    let original = export_objective(&mut sender);
    let mutants = [
        objective_mutant(&original, |candidate| {
            candidate.compiled_objective =
                CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(99.0)]).unwrap();
        }),
        objective_mutant(&original, |candidate| {
            candidate.hessian.entries[0].symbolic_deriv = Expr::from_i64(99);
        }),
        objective_mutant(&original, |candidate| {
            candidate.hessian.entries[0].compiled_deriv =
                CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(99.0)]).unwrap();
        }),
    ];
    let mut receiver = session();
    register_objective(&mut receiver, "x^2+x*y+3*y^2", &["x", "y"]);
    for mutant in mutants {
        let rejected = replay_objective(&mut receiver, &mutant);
        assert_eq!(rejected["code"], "verification_rejected", "{rejected}");
        objective_absent(&receiver);
    }
    success(replay_objective(&mut receiver, &original));
}

#[test]
fn malformed_schemas_unknown_metadata_and_noncanonical_payloads_refuse() {
    let mut sender = session();
    register_residuals(&mut sender, &["x"], &["x"]);
    compile_residual(&mut sender);
    let original = export_residual(&mut sender);
    let mut receiver = session();
    register_residuals(&mut receiver, &["x"], &["x"]);
    let mut extra: Value = serde_json::from_str(&original).unwrap();
    extra["accepted"] = json!(true);
    let unknown_schema = original.replacen("\"schema_version\":1", "\"schema_version\":2", 1);
    let duplicate_schema = original.replacen(
        "\"schema_version\":1",
        "\"schema_version\":1,\"schema_version\":1",
        1,
    );
    let duplicate_kind = original.replacen(
        "\"kind\":\"residual\"",
        "\"kind\":\"residual\",\"kind\":\"residual\"",
        1,
    );
    for (artifact, expected) in [
        (format!(" {original}"), "invalid_artifact"),
        (extra.to_string(), "invalid_artifact"),
        (duplicate_schema, "invalid_artifact"),
        (duplicate_kind, "invalid_artifact"),
        (unknown_schema, "unsupported_schema"),
    ] {
        let rejected = replay_residual(&mut receiver, &artifact);
        assert_eq!(rejected["code"], expected, "{rejected}");
        residual_absent(&receiver);
    }
    let request = json!({
        "type":"ReplayCompiledResidualSystem",
        "payload":{"schema_version":2,"system":"system:receiver","program":"program:received","artifact":original}
    });
    let response: Value =
        serde_json::from_str(&receiver.handle_envelope(&request.to_string())).unwrap();
    assert_eq!(response["code"], "unsupported_schema");
    residual_absent(&receiver);

    register_objective(&mut sender, "x^2", &["x"]);
    compile_objective(&mut sender);
    let objective = export_objective(&mut sender);
    let rejected = replay_residual(&mut receiver, &objective);
    assert_eq!(rejected["code"], "artifact_kind_mismatch", "{rejected}");
    residual_absent(&receiver);
    register_objective(&mut receiver, "x^2", &["x"]);
    let rejected = replay_objective(&mut receiver, &original);
    assert_eq!(rejected["code"], "artifact_kind_mismatch", "{rejected}");
    objective_absent(&receiver);
}

#[test]
fn replay_resource_exhaustion_including_output_admission_publishes_no_handle() {
    let mut sender = session();
    register_residuals(&mut sender, &["x^2"], &["x"]);
    compile_residual(&mut sender);
    let artifact = export_residual(&mut sender);
    let mut probe = session();
    register_residuals(&mut probe, &["x^2"], &["x"]);
    success(replay_residual(&mut probe, &artifact));
    let remaining = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut bound = limits();
        bound.dimensions[dimension.index()] = AMPLE - remaining.dimensions[dimension.index()] - 1;
        let mut receiver = Session::with_budget_limits(bound, 256);
        register_residuals(&mut receiver, &["x^2"], &["x"]);
        let before = receiver.budget_snapshot();
        let rejected = replay_residual(&mut receiver, &artifact);
        assert_eq!(
            rejected["code"], "budget_exhausted",
            "{dimension:?}: {rejected}"
        );
        assert!(
            receiver.budget_snapshot().dimensions[dimension.index()]
                < before.dimensions[dimension.index()]
        );
        residual_absent(&receiver);
    }
    for verifier_pool in [0, 1] {
        let mut bound = limits();
        bound.verifier_pool = verifier_pool;
        let mut receiver = Session::with_budget_limits(bound, 256);
        register_residuals(&mut receiver, &["x^2"], &["x"]);
        let rejected = replay_residual(&mut receiver, &artifact);
        assert_eq!(rejected["code"], "budget_exhausted", "{rejected}");
        residual_absent(&receiver);
    }

    register_objective(&mut sender, "x^2", &["x"]);
    compile_objective(&mut sender);
    let artifact = export_objective(&mut sender);
    let mut probe = session();
    register_objective(&mut probe, "x^2", &["x"]);
    success(replay_objective(&mut probe, &artifact));
    let remaining = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut bound = limits();
        bound.dimensions[dimension.index()] = AMPLE - remaining.dimensions[dimension.index()] - 1;
        let mut receiver = Session::with_budget_limits(bound, 256);
        register_objective(&mut receiver, "x^2", &["x"]);
        let rejected = replay_objective(&mut receiver, &artifact);
        assert_eq!(
            rejected["code"], "budget_exhausted",
            "{dimension:?}: {rejected}"
        );
        objective_absent(&receiver);
    }
}

#[test]
fn large_program_uses_chunked_transport_and_digest_checks_never_grant_acceptance() {
    let sources = ["sin(x)+x*y+y^2"; 32];
    let mut sender = session();
    register_residuals(&mut sender, &sources, &["y", "x"]);
    compile_residual(&mut sender);
    let exported = success(call(
        &mut sender,
        "BeginArtifactExport",
        json!({"transfer":"transfer:out", "source":{"compiled_residual_system":"program:checked"}}),
    ));
    let manifest = exported["manifest"].clone();
    assert_eq!(manifest["kind"], "compiled_residual_system");
    let bytes = manifest["total_bytes"].as_u64().unwrap() as usize;
    assert!(
        bytes > MAX_SESSION_ROOT_ARTIFACT_BYTES,
        "fixture contains only {bytes} bytes"
    );
    let inline = call(
        &mut sender,
        "ExportCompiledResidualSystem",
        json!({"program":"program:checked"}),
    );
    assert_eq!(inline["code"], "resource_limit_exceeded", "{inline}");
    success(call(
        &mut sender,
        "ReleaseCompiledResidualSystem",
        json!({"program":"program:checked"}),
    ));

    let mut receiver = session();
    register_residuals(&mut receiver, &sources, &["y", "x"]);
    register_objective(&mut receiver, "x^2", &["x"]);
    let protected = receiver.budget_snapshot().verifier_pool;
    success(call(
        &mut receiver,
        "BeginArtifactImport",
        json!({"transfer":"transfer:in", "manifest":manifest}),
    ));
    let replay = || json!({"system":"system:receiver", "program":"program:received", "transfer":"transfer:in"});
    let incomplete = call(
        &mut receiver,
        "ReplayCompiledResidualSystemFromArtifact",
        replay(),
    );
    assert_eq!(incomplete["code"], "incomplete_transfer", "{incomplete}");
    residual_absent(&receiver);
    let count = bytes.div_ceil(frankensympy::artifact_stream::ARTIFACT_CHUNK_BYTES);
    for index in 0..count {
        let chunk = success(call(
            &mut sender,
            "ReadArtifactChunk",
            json!({"transfer":"transfer:out", "index":index}),
        ));
        success(call(
            &mut receiver,
            "AppendArtifactChunk",
            json!({"transfer":"transfer:in", "chunk":chunk["chunk"]}),
        ));
    }
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    residual_absent(&receiver);
    let wrong_kind = call(
        &mut receiver,
        "ReplayCompiledObjectiveFromArtifact",
        json!({"objective":"objective:receiver", "program":"objective_program:received", "transfer":"transfer:in"}),
    );
    assert_eq!(wrong_kind["code"], "artifact_kind_mismatch", "{wrong_kind}");
    assert_eq!(receiver.budget_snapshot().verifier_pool, protected);
    objective_absent(&receiver);
    success(call(
        &mut receiver,
        "ReplayCompiledResidualSystemFromArtifact",
        replay(),
    ));
    assert!(receiver.budget_snapshot().verifier_pool < protected);
    let evaluated = success(call(
        &mut receiver,
        "EvaluateJacobianVectorProduct",
        json!({"program":"program:received", "point":["1","0"], "direction":["3","-2"]}),
    ));
    assert_eq!(evaluated["product"], json!(vec![2.0; 32]));
    assert_approximate(&evaluated);
    let duplicate = call(
        &mut receiver,
        "ReplayCompiledResidualSystemFromArtifact",
        replay(),
    );
    assert_eq!(duplicate["code"], "duplicate_object");
    success(call(
        &mut receiver,
        "ReleaseArtifactTransfer",
        json!({"transfer":"transfer:in"}),
    ));
    let evaluated = success(call(
        &mut receiver,
        "EvaluateResidualJacobian",
        json!({"program":"program:received", "point":["1","0"]}),
    ));
    assert_eq!(evaluated["residuals"], json!(vec![1.0; 32]));
}

#[test]
fn objective_hessian_candidate_transfers_across_multiple_chunks() {
    let source = "(a+b+c+d+e+f+g+h)^2";
    let variables = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let mut sender = session();
    register_objective(&mut sender, source, &variables);
    compile_objective(&mut sender);
    let exported = success(call(
        &mut sender,
        "BeginArtifactExport",
        json!({"transfer":"transfer:out", "source":{"compiled_objective":"objective_program:checked"}}),
    ));
    let manifest = exported["manifest"].clone();
    assert_eq!(manifest["kind"], "compiled_objective");
    let bytes = manifest["total_bytes"].as_u64().unwrap() as usize;
    assert!(
        bytes > MAX_SESSION_ROOT_ARTIFACT_BYTES,
        "fixture contains only {bytes} bytes"
    );
    let wrong_mode = call(
        &mut sender,
        "ReplayCompiledObjectiveFromArtifact",
        json!({"objective":"objective:receiver", "program":"objective_program:received", "transfer":"transfer:out"}),
    );
    assert_eq!(wrong_mode["code"], "wrong_transfer_mode", "{wrong_mode}");
    objective_absent(&sender);

    let mut receiver = session();
    register_objective(&mut receiver, source, &variables);
    success(call(
        &mut receiver,
        "BeginArtifactImport",
        json!({"transfer":"transfer:in", "manifest":manifest}),
    ));
    for index in 0..bytes.div_ceil(frankensympy::artifact_stream::ARTIFACT_CHUNK_BYTES) {
        let chunk = success(call(
            &mut sender,
            "ReadArtifactChunk",
            json!({"transfer":"transfer:out", "index":index}),
        ));
        success(call(
            &mut receiver,
            "AppendArtifactChunk",
            json!({"transfer":"transfer:in", "chunk":chunk["chunk"]}),
        ));
    }
    objective_absent(&receiver);
    let replayed = success(call(
        &mut receiver,
        "ReplayCompiledObjectiveFromArtifact",
        json!({"objective":"objective:receiver", "program":"objective_program:received", "transfer":"transfer:in"}),
    ));
    assert_eq!(replayed["hessian_proofs_verified"], 64);
    let point = vec!["1"; variables.len()];
    let evaluated = success(call(
        &mut receiver,
        "EvaluateObjective",
        json!({"program":"objective_program:received", "point":point}),
    ));
    assert_eq!(evaluated["value"], 64.0);
    assert_eq!(evaluated["gradient"], json!(vec![16.0; 8]));
    assert_eq!(evaluated["hessian"], json!(vec![2.0; 64]));
    assert_approximate(&evaluated);
    let product = success(call(
        &mut receiver,
        "EvaluateHessianVectorProduct",
        json!({"program":"objective_program:received", "point":point, "direction":point}),
    ));
    assert_eq!(product["product"], json!(vec![16.0; 8]));
    assert_approximate(&product);
}
