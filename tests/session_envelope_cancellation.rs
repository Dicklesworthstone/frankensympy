//! The serialized host boundary shares real certification safe points and budgets.

use frankensympy::{
    Budget, BudgetLimits, CertificationRequest, CompiledResidualSystemId, Dimension, ExactInterval,
    ExactRootWitness, MAX_ENVELOPE_BYTES, MeterError, ResidualSystemId, RootCertificateId, Session,
};
use serde_json::{Value, json};

fn session() -> Session {
    Session::with_budget_limits(BudgetLimits::uniform(1 << 40, 1 << 40), 256)
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:envelope-cancel").unwrap()
}

fn certificate() -> RootCertificateId {
    RootCertificateId::new("certificate:envelope-cancel").unwrap()
}

fn program() -> CompiledResidualSystemId {
    CompiledResidualSystemId::new("program:envelope-cancel").unwrap()
}

fn registration() -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: vec!["x^2-2".into()],
        variables: vec!["x".into()],
    }
}

fn certify() -> CertificationRequest {
    CertificationRequest::CertifyRootBox {
        schema_version: 1,
        system: system(),
        certificate: certificate(),
        witness: ExactRootWitness {
            domain: vec![ExactInterval {
                lower: "7/5".into(),
                upper: "3/2".into(),
            }],
            center: vec!["29/20".into()],
            preconditioner: vec![vec!["1/3".into()]],
            precision_digits: 16,
        },
    }
}

fn compile() -> CertificationRequest {
    CertificationRequest::CompileResidualSystem {
        schema_version: 1,
        system: system(),
        program: program(),
    }
}

fn envelope(request: CertificationRequest, id: Value) -> String {
    let mut value = serde_json::to_value(request).unwrap();
    value["id"] = id;
    value.to_string()
}

fn response(line: String) -> Value {
    serde_json::from_str(&line).unwrap()
}

fn send(session: &mut Session, request: CertificationRequest) -> Value {
    response(
        session.handle_envelope_with_checkpoint(&envelope(request, json!("same-id")), || Ok(())),
    )
}

fn registered() -> Session {
    let mut session = session();
    assert_eq!(send(&mut session, registration())["status"], "success");
    session
}

fn measured(session: &mut Session, request: CertificationRequest) -> usize {
    let mut calls = 0;
    let result = response(session.handle_envelope_with_checkpoint(
        &envelope(request, json!("same-id")),
        || {
            calls += 1;
            Ok(())
        },
    ));
    assert_eq!(result["status"], "success", "{result}");
    assert!(calls > 4, "the policy must reach execution after decoding");
    calls
}

fn cancel_at(session: &mut Session, request: CertificationRequest, target: usize) -> Value {
    let mut calls = 0;
    let result = response(session.handle_envelope_with_checkpoint(
        &envelope(request, json!("same-id")),
        || {
            calls += 1;
            if calls == target {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        },
    ));
    assert_eq!(result["code"], "cancelled", "{result}");
    assert_eq!(calls, target, "terminal host failure must never be retried");
    if target == 1 {
        assert!(result.get("id").is_none());
    } else {
        assert_eq!(result["id"], "same-id");
    }
    result
}

#[test]
fn no_op_policy_preserves_results_context_and_exact_cumulative_accounting() {
    let mut ordinary = session();
    let mut supervised = session();
    for (request, id) in [
        (registration(), json!("same-id")),
        (certify(), json!(u64::MAX)),
        (compile(), json!("same-id")),
    ] {
        let line = envelope(request, id);
        let expected = response(ordinary.handle_envelope(&line));
        let actual = response(supervised.handle_envelope_with_checkpoint(&line, || Ok(())));
        assert_eq!(expected["status"], "success", "{expected}");
        assert_eq!(actual, expected);
        assert_eq!(supervised.budget_snapshot(), ordinary.budget_snapshot());
        assert_eq!(supervised.context_digest(), ordinary.context_digest());
    }
}

#[test]
fn pre_decode_failure_is_terminal_unpublished_and_does_not_echo_unparsed_metadata() {
    for line in [envelope(registration(), json!("unused")), "{broken".into()] {
        let mut session = session();
        let before = session.budget_snapshot();
        let mut calls = 0;
        let result = response(session.handle_envelope_with_checkpoint(&line, || {
            calls += 1;
            if calls == 1 {
                Err(MeterError::Cancelled)
            } else {
                Ok(())
            }
        }));
        assert_eq!(result["code"], "cancelled");
        assert!(result.get("id").is_none());
        assert_eq!(calls, 1);
        assert_eq!(session.budget_snapshot(), before);
        assert!(session.residual_problem(&system()).is_none());
    }
}

#[test]
fn cancellation_at_every_registration_boundary_keeps_atomic_state_and_spent_work() {
    let calls = measured(&mut session(), registration());
    for target in 1..=calls {
        let mut session = session();
        let before = session.budget_snapshot();
        cancel_at(&mut session, registration(), target);
        assert!(session.residual_problem(&system()).is_none());
        let cancelled = session.budget_snapshot();
        if target <= 2 {
            assert_eq!(cancelled, before);
        } else {
            assert!(
                cancelled.dimensions[Dimension::ComputeSteps.index()]
                    < before.dimensions[Dimension::ComputeSteps.index()]
            );
        }
        assert_eq!(send(&mut session, registration())["status"], "success");
        assert!(session.residual_problem(&system()).is_some());
        for dimension in Dimension::ALL {
            assert!(
                session.budget_snapshot().dimensions[dimension.index()]
                    <= cancelled.dimensions[dimension.index()]
            );
        }
    }
}

#[test]
fn serialized_cancellation_interrupts_protected_verification_before_publication() {
    let mut completed = registered();
    let before = completed.budget_snapshot();
    let calls = measured(&mut completed, certify());
    let full_verifier_work = before.verifier_pool - completed.budget_snapshot().verifier_pool;
    assert!(full_verifier_work > 0);
    let mut interrupted_verifier = false;
    for target in [
        calls / 8,
        calls / 4,
        calls / 2,
        calls * 3 / 4,
        calls * 7 / 8,
        calls - 1,
        calls,
    ] {
        let mut session = registered();
        let before = session.budget_snapshot();
        cancel_at(&mut session, certify(), target.max(1));
        let cancelled = session.budget_snapshot();
        let work = before.verifier_pool - cancelled.verifier_pool;
        interrupted_verifier |= work > 0 && work < full_verifier_work;
        assert!(session.root_certificate(&certificate()).is_none());
        assert_eq!(send(&mut session, certify())["status"], "success");
        assert!(session.root_certificate(&certificate()).is_some());
        assert!(session.budget_snapshot().verifier_pool < cancelled.verifier_pool);
    }
    assert!(
        interrupted_verifier,
        "a pre/post wrapper cannot interrupt protected mathematical checking"
    );
}

#[test]
fn cancelled_compilation_does_not_publish_or_replace_existing_evidence() {
    let calls = measured(&mut registered(), compile());
    for target in [calls / 2, calls - 1, calls] {
        let mut session = registered();
        assert_eq!(send(&mut session, certify())["status"], "success");
        let accepted = session.root_certificate(&certificate()).unwrap().clone();
        cancel_at(&mut session, compile(), target.max(1));
        assert!(session.compiled_residual_system(&program()).is_none());
        assert_eq!(session.root_certificate(&certificate()), Some(&accepted));
        assert_eq!(send(&mut session, compile())["status"], "success");
        assert!(session.compiled_residual_system(&program()).is_some());
    }
}

#[test]
fn unsupervised_operations_refuse_without_mutation_or_accounting_claims() {
    let mut session = session();
    let initial_root = session.handle_envelope(r#"{"type":"StateRoot"}"#);
    for request in [
        json!({"type":"Bind", "payload":{"symbol":"a", "expr":"7"}}),
        json!({"type":"Fork", "payload":{"branch_name":"unpublished"}}),
        json!({"type":"Eval", "payload":{"expr":"2+2"}}),
        json!({"type":"Diff", "payload":{"expr":"x^2", "var":"x"}}),
        json!({"type":"ConstructClaim", "payload":{"source":"x=x"}}),
        json!({"type":"UnknownOperation", "payload":{}}),
    ] {
        let mut request = request;
        request["id"] = json!(17);
        let before = session.budget_snapshot();
        let result =
            response(session.handle_envelope_with_checkpoint(&request.to_string(), || Ok(())));
        assert_eq!(result["code"], "unsupported_operation", "{result}");
        assert_eq!(result["id"], 17);
        assert_eq!(session.budget_snapshot(), before);
        assert_eq!(session.active_branch(), "session");
        assert_eq!(
            session.handle_envelope(r#"{"type":"StateRoot"}"#),
            initial_root
        );
    }
    let ordinary = response(
        session.handle_envelope(r#"{"type":"Bind","payload":{"symbol":"a","expr":"7"},"id":17}"#),
    );
    assert_eq!(ordinary["status"], "success");
    assert_eq!(ordinary["result"], "7");
    assert_ne!(
        session.handle_envelope(r#"{"type":"StateRoot"}"#),
        initial_root
    );
}

#[test]
fn malformed_schema_and_duplicate_key_refusals_match_the_ordinary_decoder() {
    let mut future = serde_json::to_value(registration()).unwrap();
    future["payload"]["schema_version"] = json!(2);
    let mut invalid_id = serde_json::to_value(registration()).unwrap();
    invalid_id["id"] = json!({"not":"an id"});
    let mut unknown_field = serde_json::to_value(registration()).unwrap();
    unknown_field["ignored"] = json!(true);
    let duplicate = r#"{"id":"duplicated","type":"RegisterResidualSystem","payload":{"schema_version":1,"system":"system:other","system":"system:envelope-cancel","residuals":["x^2-2"],"variables":["x"]}}"#;
    for line in [
        future.to_string(),
        invalid_id.to_string(),
        unknown_field.to_string(),
        duplicate.to_string(),
        "{broken".to_string(),
        " ".repeat(MAX_ENVELOPE_BYTES + 1),
    ] {
        let mut ordinary = session();
        let mut supervised = session();
        let expected = response(ordinary.handle_envelope(&line));
        let actual = response(supervised.handle_envelope_with_checkpoint(&line, || Ok(())));
        assert_eq!(actual, expected);
        assert_eq!(actual["status"], "error");
        assert_eq!(supervised.budget_snapshot(), ordinary.budget_snapshot());
        assert!(supervised.residual_problem(&system()).is_none());
    }
}

#[test]
fn host_budget_and_admission_exhaustion_remain_distinct_from_cancellation() {
    let line = envelope(registration(), json!("budget"));
    let mut limits = BudgetLimits::uniform(1 << 40, 1 << 40);
    limits.dimensions[Dimension::ComputeSteps.index()] = line.len() as u64 - 1;
    let mut limited = Session::with_budget_limits(limits, 256);
    let before = limited.budget_snapshot();
    let result = response(limited.handle_envelope_with_checkpoint(&line, || Ok(())));
    assert_eq!(result["code"], "budget_exhausted");
    assert_eq!(result["id"], "budget");
    assert_eq!(limited.budget_snapshot(), before);
    assert!(limited.residual_problem(&system()).is_none());

    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    let error = empty
        .try_charge_batch(&[(Dimension::ComputeSteps, 1)])
        .unwrap_err();
    let mut session = session();
    let before = session.budget_snapshot();
    let mut calls = 0;
    let result = response(session.handle_envelope_with_checkpoint(&line, || {
        calls += 1;
        if calls == 2 {
            Err(MeterError::Budget(error))
        } else {
            Ok(())
        }
    }));
    assert_eq!(result["code"], "budget_exhausted");
    assert_eq!(result["id"], "budget");
    assert_eq!(calls, 2);
    assert_eq!(session.budget_snapshot(), before);
    assert!(session.residual_problem(&system()).is_none());
}
