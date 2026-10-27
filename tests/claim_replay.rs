//! Portable proof transport: supplied evidence, frozen statements and fresh processes.
#![forbid(unsafe_code)]

use frankensympy::{ClaimExportPage, ClaimStatus, Session, SessionBudgets, SessionError};
use fsym_proof_kernel::{Claim, DerivationTree, ProofRule};
use serde_json::{Value, json};

fn integer(value: i64) -> frankensympy::Expr {
    frankensympy::Expr::from_i64(value)
}

fn exported(source: &str, page_size: usize) -> Vec<ClaimExportPage> {
    let mut session = Session::new(SessionBudgets::default());
    let candidate = session.construct_claim(source).unwrap();
    let accepted = session.verify_claim(&candidate).unwrap();
    let first: ClaimExportPage =
        serde_json::from_value(session.export_claim(&accepted, 0, page_size).unwrap()).unwrap();
    (0..first.total_pages)
        .map(|page| {
            serde_json::from_value(session.export_claim(&accepted, page, page_size).unwrap())
                .unwrap()
        })
        .collect()
}

fn replay(pages: &[ClaimExportPage]) -> Result<[u8; 32], SessionError> {
    Session::new(SessionBudgets::default()).replay_claim_pages(pages)
}

fn refresh_digest(pages: &mut [ClaimExportPage]) {
    let derivation = DerivationTree {
        steps: pages.iter().flat_map(|p| p.steps.iter().cloned()).collect(),
        root: pages[0].root,
    };
    let digest = derivation.digest();
    for page in pages {
        page.claim.derivation_digest = Some(digest);
    }
}

#[test]
fn complete_pages_replay_without_any_resident_proof() {
    for source in ["42", "(x+1)^2 = x^2+2*x+1", "sin(x) = sin(x)"] {
        for page_size in [1, 2, 256] {
            let pages = exported(source, page_size);
            assert_eq!(
                replay(&pages).unwrap(),
                pages[0].claim.derivation_digest.unwrap()
            );
            let mut receiver = Session::new(SessionBudgets::default());
            let value = serde_json::to_value(&pages).unwrap();
            let context = receiver.context_digest();
            assert_eq!(
                receiver.replay_claim(&value, context).unwrap(),
                replay(&pages).unwrap()
            );
        }
    }
}

#[test]
fn changed_steps_are_rejected_even_with_a_recomputed_digest() {
    let pages = exported("(x+1)^2 = x^2+2*x+1", 1);
    let mut changed_claim = pages.clone();
    changed_claim[0].steps[0].claim = Claim::equality(integer(1), integer(2));
    refresh_digest(&mut changed_claim);
    assert!(replay(&changed_claim).is_err());

    let mut changed_rule = pages.clone();
    changed_rule[0].steps[0].rule = ProofRule::Reflexivity(integer(99));
    refresh_digest(&mut changed_rule);
    assert!(replay(&changed_rule).is_err());

    let mut changed_reference = pages.clone();
    changed_reference[1].steps[0].rule = ProofRule::Symmetry(fsym_proof_kernel::StepId(99));
    refresh_digest(&mut changed_reference);
    assert!(replay(&changed_reference).is_err());

    let mut changed_digest = pages;
    for page in &mut changed_digest {
        page.claim.derivation_digest = Some([7; 32]);
    }
    assert!(matches!(
        replay(&changed_digest),
        Err(SessionError::ReplayDigestMismatch)
    ));
}

#[test]
fn valid_proof_cannot_be_relabelled_as_a_different_claim() {
    let mut pages = exported("42", 1);
    for page in &mut pages {
        page.claim.source = "1 = 2".into();
        page.claim.statement = Claim::equality(integer(1), integer(2));
    }
    assert!(matches!(replay(&pages), Err(SessionError::ClaimMismatch)));
    let mut pages = exported("42", 1);
    for page in &mut pages {
        page.claim.source = "99".into();
    }
    assert!(matches!(replay(&pages), Err(SessionError::ClaimMismatch)));
}

#[test]
fn page_completeness_order_schema_and_uniform_metadata_are_required() {
    let pages = exported("(x+1)^2 = x^2+2*x+1", 1);
    assert!(replay(&[]).is_err());
    assert!(replay(&pages[..2]).is_err());
    let mut reordered = pages.clone();
    reordered.swap(0, 1);
    assert!(replay(&reordered).is_err());
    let mut duplicate = pages.clone();
    duplicate[1] = duplicate[0].clone();
    assert!(replay(&duplicate).is_err());
    let mut empty = pages.clone();
    empty[1].steps.clear();
    assert!(replay(&empty).is_err());
    let mut schema = pages.clone();
    schema[0].schema_version += 1;
    assert!(replay(&schema).is_err());
    let mut candidate = pages;
    for page in &mut candidate {
        page.claim.status = ClaimStatus::Candidate;
    }
    assert!(matches!(
        replay(&candidate),
        Err(SessionError::CandidateNotAccepted)
    ));
}

#[test]
fn a_caller_cannot_override_the_receiving_assumption_universe() {
    let mut context = fsym_assumptions::AssumptionsContext::new();
    context
        .assume(
            frankensympy::Symbol::new("x"),
            fsym_assumptions::Predicate::Real,
        )
        .unwrap();
    let mut sender = Session::with_context(
        std::sync::Arc::new(context.snapshot()),
        SessionBudgets::default(),
    );
    let candidate = sender.construct_claim("x").unwrap();
    let accepted = sender.verify_claim(&candidate).unwrap();
    let export = sender.export_claim(&accepted, 0, 256).unwrap();
    let mut receiver = Session::new(SessionBudgets::default());
    // Supplying the sender's digest used to bypass the session-context comparison.
    assert!(matches!(
        receiver.replay_claim(&export, accepted.context_digest),
        Err(SessionError::UniverseMismatch { .. })
    ));
    assert!(matches!(
        receiver.verify_claim(&candidate),
        Err(SessionError::UniverseMismatch { .. })
    ));
}

#[test]
fn candidate_bindings_are_frozen_and_verified_without_receiver_state() {
    let mut sender = Session::new(SessionBudgets::default());
    sender.construct("x", "2").unwrap();
    let candidate = sender.construct_claim("x+1 = 3").unwrap();
    sender.construct("x", "100").unwrap();
    let accepted = sender.verify_claim(&candidate).unwrap();
    assert_eq!(accepted.statement, candidate.statement);
    assert_eq!(accepted.statement.rhs(), Some(&integer(3)));
    let export = sender.export_claim(&accepted, 0, 256).unwrap();
    let mut receiver = Session::new(SessionBudgets::default());
    receiver.construct("x", "-7").unwrap();
    let context = receiver.context_digest();
    receiver.replay_claim(&export, context).unwrap();
    let evaluated: Value = serde_json::from_str(
        &receiver.handle_envelope(r#"{"type":"Eval","payload":{"expr":"x"}}"#),
    )
    .unwrap();
    assert_eq!(evaluated["result"], "-7");
    let mut tampered = candidate;
    tampered.bindings[0].1 = integer(4);
    assert!(matches!(
        sender.verify_claim(&tampered),
        Err(SessionError::ClaimMismatch)
    ));
}

#[test]
fn replay_spends_persistent_verifier_budget_and_enforces_step_caps() {
    let pages = exported("42", 1);
    let mut too_small = Session::new(SessionBudgets {
        uniform_dimension_budget: 100,
        verify_step_budget: 1,
    });
    assert!(matches!(
        too_small.replay_claim_pages(&pages),
        Err(SessionError::InvalidPages) | Err(SessionError::BudgetExhausted { .. })
    ));
    let mut once = Session::new(SessionBudgets {
        uniform_dimension_budget: 1,
        verify_step_budget: 256,
    });
    once.replay_claim_pages(&pages).unwrap();
    assert!(matches!(
        once.replay_claim_pages(&pages),
        Err(SessionError::BudgetExhausted { .. })
    ));
    let mut zero = Session::new(SessionBudgets {
        uniform_dimension_budget: 0,
        verify_step_budget: 256,
    });
    assert!(matches!(
        zero.replay_claim_pages(&pages),
        Err(SessionError::BudgetExhausted { .. })
    ));
}

#[test]
fn substituted_expansion_and_later_page_depth_are_bounded_before_comparison() {
    let mut sender = Session::new(SessionBudgets::default());
    sender.construct("x", "1").unwrap();
    let source = format!("f({})", vec!["x"; 256].join(","));
    let mut candidate = sender.construct_claim(&source).unwrap();
    candidate.bindings[0].1 = frankensympy::Expr::Function("g".into(), vec![integer(1); 2048]);
    assert!(matches!(
        sender.verify_claim(&candidate),
        Err(SessionError::ExportLimit)
    ));

    let mut pages = exported("42", 1);
    let mut deep = integer(1);
    for _ in 0..300 {
        deep = frankensympy::Expr::Function("f".into(), vec![deep]);
    }
    pages[1].claim.statement = Claim::equality(deep, integer(1));
    assert!(matches!(replay(&pages), Err(SessionError::ExportLimit)));
}

#[test]
fn claim_instantiation_never_eagerly_evaluates_power_towers() {
    let mut sender = Session::new(SessionBudgets::default());
    sender.construct("x", "2").unwrap();
    let candidate = sender.construct_claim("(x^100000)^100000").unwrap();
    let accepted = sender.verify_claim(&candidate).unwrap();
    assert!(matches!(
        accepted.statement.lhs(),
        Some(frankensympy::Expr::Pow(_, _))
    ));
    let export = sender.export_claim(&accepted, 0, 256).unwrap();
    let mut receiver = Session::new(SessionBudgets::default());
    let context = receiver.context_digest();
    assert_eq!(
        receiver.replay_claim(&export, context).unwrap(),
        accepted.derivation_digest.unwrap()
    );
    assert!(matches!(
        sender.construct_claim("Lambda(y, x+y)"),
        Err(SessionError::UnsupportedSubstitution)
    ));
    assert!(matches!(
        sender.construct_claim("diff(x, x)"),
        Err(SessionError::UnsupportedSubstitution)
    ));
    // An unrelated workspace binding must not change an untouched bound form.
    sender.construct_claim("Lambda(y, y+1)").unwrap();
}

fn cli(requests: &[Value]) -> Vec<Value> {
    use std::io::Write;
    let mut process = std::process::Command::new(env!("CARGO_BIN_EXE_frankensympy"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = process.stdin.take().unwrap();
    for request in requests {
        writeln!(input, "{request}").unwrap();
    }
    drop(input);
    let output = process.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn cli_constructs_verifies_exports_and_replays_in_a_second_process() {
    let source = "(x+1)^2 = x^2+2*x+1";
    let mut template = Session::new(SessionBudgets::default());
    let candidate = template.construct_claim(source).unwrap();
    let accepted = template.verify_claim(&candidate).unwrap();
    let responses = cli(&[
        json!({"id":"construct", "type":"ConstructClaim", "payload":{"source":source}}),
        json!({"id":"verify", "type":"VerifyClaim", "payload":{"claim":candidate}}),
        json!({"type":"ExportClaim", "payload":{"claim":accepted, "page":0, "page_size":256}}),
    ]);
    assert_eq!(responses[0]["id"], "construct");
    assert_eq!(responses[0]["result"]["status"], "candidate");
    assert_eq!(responses[1]["id"], "verify");
    assert_eq!(responses[1]["result"]["status"], "accepted");
    assert_eq!(responses[2]["status"], "success");
    let page = responses[2]["result"].clone();
    let replayed = cli(&[json!({"id":"fresh", "type":"ReplayClaim", "payload":{"pages":[page]}})]);
    assert_eq!(replayed[0]["id"], "fresh");
    assert_eq!(replayed[0]["status"], "success");
    assert_eq!(
        replayed[0]["result"]["derivation_digest"],
        responses[1]["result"]["derivation_digest"]
    );
}

#[test]
fn unknown_nested_proof_and_statement_fields_are_never_silently_discarded() {
    let original = serde_json::to_value(exported("(x+1)^2 = x^2+2*x+1", 256)).unwrap();
    let mut changed_step = original.clone();
    changed_step[0]["steps"][0]["unverified_evidence"] = json!(true);
    let mut changed_rule = original.clone();
    changed_rule[0]["steps"][0]["rule"]["CertificateLemma"]["unverified_condition"] =
        json!("x > 0");
    let mut changed_statement = original;
    changed_statement[0]["claim"]["statement"]["Equality"]["unverified_condition"] = json!("x > 0");
    for value in [changed_step, changed_rule, changed_statement] {
        let mut receiver = Session::new(SessionBudgets::default());
        let context = receiver.context_digest();
        assert!(matches!(
            receiver.replay_claim(&value, context),
            Err(SessionError::Serialization(_))
        ));
        let envelope = json!({"type":"ReplayClaim", "payload":{"pages":value}});
        let response: Value =
            serde_json::from_str(&receiver.handle_envelope(&envelope.to_string())).unwrap();
        assert_eq!(response["code"], "malformed_request");
    }
}
