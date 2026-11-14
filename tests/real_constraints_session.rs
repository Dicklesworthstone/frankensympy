//! Exact Boolean feasibility regions through typed sessions and fresh NDJSON.

use frankensympy::{
    BigRational, BudgetLimits, CertificationRequest, CertificationResponse, Dimension,
    ExactRealSolutionSet, RealFormula, RealRelation, ResidualSystemId, Session,
};
use fsym_budget::Unbounded;
use fsym_core::real_roots::{
    RealRootInterval, RealRootIsolationCertificate, verify_real_root_isolation,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(100_000_000, 100_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:constraints").unwrap()
}

fn register(sources: &[&str], variables: &[&str]) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: sources.iter().map(|source| (*source).into()).collect(),
        variables: variables.iter().map(|name| (*name).into()).collect(),
    }
}

fn registered(sources: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    session
        .certification_request(register(sources, &["x"]))
        .unwrap();
    session
}

fn request(relations: &[RealRelation], formula: RealFormula) -> CertificationRequest {
    CertificationRequest::SolveRealConstraints {
        schema_version: 1,
        system: system(),
        relations: relations.to_vec(),
        formula,
    }
}

fn all(count: usize) -> RealFormula {
    RealFormula::All((0..count).map(RealFormula::Atom).collect())
}

fn rational(value: &str) -> BigRational {
    match value.split_once('/') {
        Some((n, d)) => BigRational::new(n.parse().unwrap(), d.parse().unwrap()),
        None => BigRational::from_integer(value.parse().unwrap()),
    }
}

fn check_set(solution: &ExactRealSolutionSet) {
    assert_eq!(solution.roots.len(), solution.point_inclusions.len());
    assert_eq!(solution.roots.len() + 1, solution.gap_inclusions.len());
    let coefficients: Vec<_> = solution.polynomial.iter().map(|q| rational(q)).collect();
    let certificate = RealRootIsolationCertificate {
        intervals: solution
            .roots
            .iter()
            .map(|root| RealRootInterval {
                lower: rational(&root.lower),
                upper: rational(&root.upper),
                multiplicity: 1,
            })
            .collect(),
    };
    verify_real_root_isolation(&coefficients, &certificate, &mut Unbounded).unwrap();
}

fn solution(response: CertificationResponse) -> ExactRealSolutionSet {
    let CertificationResponse::RealConstraintsSolved { solution_set, .. } = response else {
        panic!("expected a complete real constraint result")
    };
    check_set(&solution_set);
    solution_set
}

#[test]
fn shared_partition_preserves_irrational_endpoints_original_poles_and_frozen_sources() {
    let mut session = registered(&["(x^2-2)/(x^2-3)", "x"]);
    let original = session.residual_problem(&system()).unwrap().clone();
    let context = session.context_digest();
    session.construct("x", "100").unwrap();
    let before = session.budget_snapshot();
    let response = session
        .certification_request(request(&[RealRelation::Ge, RealRelation::Ge], all(2)))
        .unwrap();
    let CertificationResponse::RealConstraintsSolved {
        system: solved,
        variable,
        relations,
        formula,
        evidence,
        claim,
        completeness,
        domain,
        solution_set,
        context_digest,
        ..
    } = response
    else {
        panic!("constraint response")
    };
    assert_eq!(solved, system());
    assert_eq!(variable, original.variables()[0]);
    assert_eq!(relations, [RealRelation::Ge, RealRelation::Ge]);
    assert_eq!(formula, all(2));
    assert_eq!(evidence, "exact_cross_checked");
    assert_eq!(
        claim,
        "complete_real_solution_set_of_original_rational_constraints"
    );
    assert_eq!((completeness, domain), ("complete", "real"));
    assert_eq!(context_digest, context);
    assert_eq!(solution_set.polynomial, ["0", "6", "0", "-5", "0", "1"]);
    assert_eq!(
        solution_set.gap_inclusions,
        [false, false, false, true, false, true]
    );
    assert_eq!(
        solution_set.point_inclusions,
        [false, false, true, true, false]
    );
    check_set(&solution_set);
    assert_eq!(session.residual_problem(&system()).unwrap(), &original);
    let after = session.budget_snapshot();
    assert!(after.verifier_pool < before.verifier_pool);
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(after.dimensions[dimension.index()] < before.dimensions[dimension.index()]);
    }
}

#[test]
fn nested_union_retains_isolated_equality_points_and_foreign_boundary_values() {
    let mut session = registered(&["x^2-2", "x", "x+2"]);
    let formula = RealFormula::Any(vec![all(2), RealFormula::Atom(2)]);
    let set = solution(
        session
            .certification_request(request(
                &[RealRelation::Eq, RealRelation::Gt, RealRelation::Le],
                formula,
            ))
            .unwrap(),
    );
    // (-infinity,-2] union {sqrt(2)}; other critical roots remain exact cells.
    assert_eq!(set.roots.len(), 4);
    assert_eq!(set.gap_inclusions, [true, false, false, false, false]);
    assert_eq!(set.point_inclusions, [true, false, false, true]);
}

#[test]
fn a_defined_union_branch_can_include_another_atoms_hole_but_disequality_cannot() {
    let mut session = registered(&["x/x-1", "x"]);
    for (relation, gaps) in [
        (RealRelation::Eq, [true, true]),
        (RealRelation::Ne, [false, false]),
    ] {
        let set = solution(
            session
                .certification_request(request(
                    &[relation, RealRelation::Eq],
                    RealFormula::Any(vec![RealFormula::Atom(0), RealFormula::Atom(1)]),
                ))
                .unwrap(),
        );
        assert_eq!(set.polynomial, ["0", "1"]);
        assert_eq!(set.gap_inclusions, gaps);
        assert_eq!(set.point_inclusions, [true]);
    }
    let mut session = registered(&["0*(1/x)", "x"]);
    let set = solution(
        session
            .certification_request(request(&[RealRelation::Eq, RealRelation::Ne], all(2)))
            .unwrap(),
    );
    assert_eq!(set.gap_inclusions, [true, true]);
    assert_eq!(set.point_inclusions, [false]);
}

#[test]
fn constant_formulas_and_unused_atoms_obey_explicit_set_semantics() {
    let mut session = registered(&["x"]);
    for (formula, expected) in [
        (RealFormula::All(vec![]), true),
        (RealFormula::Any(vec![]), false),
    ] {
        let set = solution(
            session
                .certification_request(request(&[RealRelation::Eq], formula))
                .unwrap(),
        );
        assert!(
            set.gap_inclusions
                .iter()
                .all(|&included| included == expected)
        );
        assert!(
            set.point_inclusions
                .iter()
                .all(|&included| included == expected)
        );
    }
    for source in ["sin(x)", "0*sin(x)", "x^(1/2)"] {
        let mut session = registered(&["0", source]);
        for formula in [
            RealFormula::Any(vec![RealFormula::Atom(0)]),
            RealFormula::All(vec![]),
        ] {
            let error = session
                .certification_request(request(&[RealRelation::Eq, RealRelation::Eq], formula))
                .unwrap_err();
            assert_eq!(error.code(), "unsupported_operation", "{source}");
        }
    }
}

#[test]
fn invalid_dimensions_relations_and_formula_indices_do_not_return_empty_regions() {
    let mut missing = Session::with_budget_limits(limits(), 256);
    assert_eq!(
        missing
            .certification_request(request(&[RealRelation::Eq], all(1)))
            .unwrap_err()
            .code(),
        "unknown_object"
    );
    let mut session = registered(&["x", "x^2-1"]);
    for (relations, formula) in [
        (vec![RealRelation::Eq], all(1)),
        (
            vec![RealRelation::Eq, RealRelation::Ge],
            RealFormula::Atom(2),
        ),
    ] {
        assert_eq!(
            session
                .certification_request(request(&relations, formula))
                .unwrap_err()
                .code(),
            "invalid_input"
        );
    }
    let mut multivariate = Session::with_budget_limits(limits(), 256);
    multivariate
        .certification_request(register(&["x+y"], &["x", "y"]))
        .unwrap();
    assert_eq!(
        multivariate
            .certification_request(request(&[RealRelation::Eq], all(1)))
            .unwrap_err()
            .code(),
        "invalid_input"
    );
    let mut too_many = registered(&["0"; 33]);
    assert_eq!(
        too_many
            .certification_request(request(&[RealRelation::Eq; 33], all(1)))
            .unwrap_err()
            .code(),
        "invalid_input"
    );
}

#[test]
fn structural_bounds_and_short_circuited_degree_limits_fail_explicitly() {
    let mut session = registered(&["x"]);
    let mut formula = RealFormula::Atom(0);
    for _ in 0..40 {
        formula = RealFormula::All(vec![formula]);
    }
    assert_eq!(
        session
            .certification_request(request(&[RealRelation::Eq], formula))
            .unwrap_err()
            .code(),
        "resource_limit_exceeded"
    );
    let mut session = registered(&["0", "x^65"]);
    assert_eq!(
        session
            .certification_request(request(
                &[RealRelation::Eq, RealRelation::Eq],
                RealFormula::Any(vec![RealFormula::Atom(0)]),
            ))
            .unwrap_err()
            .code(),
        "resource_limit_exceeded"
    );
}

#[test]
fn verifier_authority_and_cumulative_limits_cannot_be_reset_by_retries() {
    for verifier_pool in [0, 1] {
        let mut bounded = limits();
        bounded.verifier_pool = verifier_pool;
        let mut session = Session::with_budget_limits(bounded, 256);
        session
            .certification_request(register(&["x^2-2", "x"], &["x"]))
            .unwrap();
        for _ in 0..2 {
            let before = session.budget_snapshot();
            let error = session
                .certification_request(request(&[RealRelation::Ge, RealRelation::Ge], all(2)))
                .unwrap_err();
            assert_eq!(error.code(), "budget_exhausted");
            let after = session.budget_snapshot();
            assert!(after.verifier_pool <= before.verifier_pool);
            if verifier_pool == 0 {
                assert_eq!(before, after);
            }
        }
    }
}

#[test]
fn output_admission_failure_does_not_publish_a_partial_region_or_modify_sources() {
    let mut probe = registered(&["x^2-2", "x"]);
    probe
        .certification_request(request(&[RealRelation::Ge, RealRelation::Gt], all(2)))
        .unwrap();
    let completed = probe.budget_snapshot();
    let mut bounded = limits();
    let slot = Dimension::ComputeSteps.index();
    bounded.dimensions[slot] -= completed.dimensions[slot] + 1;
    let mut session = Session::with_budget_limits(bounded, 256);
    session
        .certification_request(register(&["x^2-2", "x"], &["x"]))
        .unwrap();
    let original = session.residual_problem(&system()).unwrap().clone();
    let before = session.budget_snapshot();
    assert_eq!(
        session
            .certification_request(request(&[RealRelation::Ge, RealRelation::Gt], all(2)))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
    assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    assert_eq!(session.residual_problem(&system()).unwrap(), &original);
    assert_eq!(
        session
            .certification_request(request(&[RealRelation::Ge, RealRelation::Gt], all(2)))
            .unwrap_err()
            .code(),
        "budget_exhausted"
    );
}

#[test]
fn strict_formula_wire_rejects_unknown_keys_duplicate_members_and_forged_results() {
    let mut session = registered(&["x"]);
    for formula in [
        r#"{"not":{"atom":0}}"#,
        r#"{"atom":0,"atom":0}"#,
        r#"{"all":[],"any":[]}"#,
        r#"{"atom":-1}"#,
        r#"{"atom":{"index":0}}"#,
    ] {
        let frame = format!(
            r#"{{"type":"SolveRealConstraints","payload":{{"schema_version":1,"system":"system:constraints","relations":["eq"],"formula":{formula}}}}}"#
        );
        let before = session.budget_snapshot().verifier_pool;
        let result: Value = serde_json::from_str(&session.handle_envelope(&frame)).unwrap();
        assert_eq!(result["code"], "malformed_request", "{frame}: {result}");
        assert_eq!(session.budget_snapshot().verifier_pool, before);
    }
    let mut frame = serde_json::to_value(request(&[RealRelation::Eq], all(1))).unwrap();
    frame["payload"]["solution_set"] = json!({"gap_inclusions":[true]});
    let result: Value = serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
    assert_eq!(result["code"], "malformed_request");
    let mut frame = serde_json::to_value(request(&[RealRelation::Eq], all(1))).unwrap();
    frame["payload"]["schema_version"] = json!(2);
    let result: Value = serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
    assert_eq!(result["code"], "unsupported_schema");
}

#[test]
fn fresh_cli_composes_exact_regions_with_default_limits_and_survives_refusal() {
    // Default limits are part of usability: this ordinary constraint example
    // must not need a private test-only budget increase.
    let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut send = |request: CertificationRequest, id: &str| {
        let mut frame = serde_json::to_value(request).unwrap();
        frame["id"] = json!(id);
        writeln!(input, "{frame}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        let result: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(result["id"], id);
        result
    };
    assert_eq!(
        send(register(&["x^2-2", "x-2"], &["x"]), "register")["status"],
        "success"
    );
    let relations = [RealRelation::Le, RealRelation::Ge];
    assert_eq!(
        send(request(&relations, RealFormula::Atom(2)), "bad-index")["code"],
        "invalid_input"
    );
    let result = send(
        request(
            &relations,
            RealFormula::Any(vec![RealFormula::Atom(0), RealFormula::Atom(1)]),
        ),
        "region",
    );
    assert_eq!(result["status"], "success", "{result}");
    assert_eq!(result["result"]["kind"], "real_constraints_solved");
    assert_eq!(result["result"]["completeness"], "complete");
    assert_eq!(result["result"]["evidence"], "exact_cross_checked");
    assert_eq!(
        result["result"]["solution_set"]["gap_inclusions"],
        json!([false, true, false, true])
    );
    assert_eq!(
        result["result"]["solution_set"]["point_inclusions"],
        json!([true, true, true])
    );
    drop(input);
    assert!(child.wait().unwrap().success());
}
