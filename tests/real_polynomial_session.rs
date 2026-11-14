//! Complete finite real polynomial systems through Session and fresh NDJSON.

use frankensympy::{
    BigRational, BudgetLimits, CertificationRequest, CertificationResponse, Dimension,
    ExactRealPolynomialSolutionSet, ResidualSystemId, Session,
};
use fsym_budget::Unbounded;
use fsym_core::real_roots::{
    RealRootInterval, RealRootIsolationCertificate, verify_real_root_isolation,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn limits() -> BudgetLimits {
    let mut limits = BudgetLimits::uniform(200_000_000, 200_000_000);
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits
}

fn system() -> ResidualSystemId {
    ResidualSystemId::new("system:polynomial").unwrap()
}

fn registration(sources: &[&str], variables: &[&str]) -> CertificationRequest {
    CertificationRequest::RegisterResidualSystem {
        schema_version: 1,
        system: system(),
        residuals: sources.iter().map(|source| (*source).into()).collect(),
        variables: variables.iter().map(|name| (*name).into()).collect(),
    }
}

fn request() -> CertificationRequest {
    CertificationRequest::SolveRealPolynomialSystem {
        schema_version: 1,
        system: system(),
    }
}

fn registered(sources: &[&str], variables: &[&str]) -> Session {
    let mut session = Session::with_budget_limits(limits(), 256);
    session
        .certification_request(registration(sources, variables))
        .unwrap();
    session
}

fn solution(response: CertificationResponse) -> ExactRealPolynomialSolutionSet {
    let CertificationResponse::RealPolynomialSystemSolved { solution_set, .. } = response else {
        panic!("expected an exact polynomial solution set")
    };
    solution_set
}

fn rational(value: &str) -> BigRational {
    match value.split_once('/') {
        Some((n, d)) => BigRational::new(n.parse().unwrap(), d.parse().unwrap()),
        None => BigRational::from_integer(value.parse().unwrap()),
    }
}

fn check_roots(solution: &ExactRealPolynomialSolutionSet, multiplicity: usize) {
    let coefficients = solution
        .polynomial
        .iter()
        .map(|q| rational(q))
        .collect::<Vec<_>>();
    let certificate = RealRootIsolationCertificate {
        intervals: solution
            .roots
            .iter()
            .map(|root| RealRootInterval {
                lower: rational(&root.lower),
                upper: rational(&root.upper),
                multiplicity,
            })
            .collect(),
    };
    verify_real_root_isolation(&coefficients, &certificate, &mut Unbounded).unwrap();
}

fn evaluate(coefficients: &[String], point: i64) -> BigRational {
    let point = BigRational::from_integer(point.into());
    coefficients
        .iter()
        .rev()
        .fold(rational("0"), |value, coefficient| {
            value * &point + rational(coefficient)
        })
}

#[test]
fn complete_nonshape_quadratic_system_has_four_exact_irrational_tuples() {
    let mut session = registered(&["x^2-2", "y^2-3"], &["x", "y"]);
    let original = session.residual_problem(&system()).unwrap().clone();
    let context = session.context_digest();
    session.construct("x", "100").unwrap();
    let before = session.budget_snapshot();
    let response = session.certification_request(request()).unwrap();
    let CertificationResponse::RealPolynomialSystemSolved {
        system: solved,
        variables,
        evidence,
        claim,
        completeness,
        domain,
        solution_set,
        context_digest,
        ..
    } = response
    else {
        panic!("expected polynomial system response")
    };
    assert_eq!(solved, system());
    assert_eq!(variables, original.variables());
    assert_eq!(context_digest, context);
    assert_eq!(evidence, "exact_cross_checked");
    assert_eq!(
        claim,
        "complete_distinct_real_solution_set_of_original_polynomial_system"
    );
    assert_eq!((completeness, domain), ("complete", "real"));
    // With t=x+y, p(t)=(t^2-5)^2-24; these two maps are the
    // exact inverse of that separating form on all four original points.
    assert_eq!(solution_set.polynomial, ["1", "0", "-10", "0", "1"]);
    assert_eq!(solution_set.coordinates[0], ["0", "-9/2", "0", "1/2"]);
    assert_eq!(solution_set.coordinates[1], ["0", "11/2", "0", "-1/2"]);
    assert_eq!(solution_set.roots.len(), 4);
    assert!(!variables.contains(&solution_set.parameter));
    check_roots(&solution_set, 1);
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
fn separating_form_collisions_cannot_drop_grid_points() {
    let mut session = registered(&["x^2-x", "y^2-y"], &["x", "y"]);
    let set = solution(session.certification_request(request()).unwrap());
    // t=x+y and t=y-x both collide; the checked t=2*x+y map must
    // distinguish all four tuples, including the two mixed points.
    assert_eq!(set.polynomial, ["0", "-6", "11", "-6", "1"]);
    for (t, (x, y)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
        assert_eq!(
            evaluate(&set.coordinates[0], t as i64),
            rational(&x.to_string())
        );
        assert_eq!(
            evaluate(&set.coordinates[1], t as i64),
            rational(&y.to_string())
        );
    }
    assert_eq!(set.roots.len(), 4);
    check_roots(&set, 1);
}

#[test]
fn caller_coordinate_order_and_redundant_exact_equations_are_preserved() {
    let mut session = registered(&["3*y^2-9", "x^2/7-2/7", "0", "x^2-2"], &["y", "x"]);
    let response = session.certification_request(request()).unwrap();
    let CertificationResponse::RealPolynomialSystemSolved {
        variables,
        solution_set,
        ..
    } = response
    else {
        panic!("polynomial response")
    };
    assert_eq!(
        variables
            .iter()
            .map(|axis| axis.name.as_str())
            .collect::<Vec<_>>(),
        ["y", "x"]
    );
    assert_eq!(solution_set.coordinates[0], ["0", "11/2", "0", "-1/2"]);
    assert_eq!(solution_set.coordinates[1], ["0", "-9/2", "0", "1/2"]);
    check_roots(&solution_set, 1);
}

#[test]
fn shape_quintic_needs_no_radical_formula_or_numerical_approximation() {
    let mut session = registered(&["x-y^2-1", "y^5-y+1"], &["x", "y"]);
    let set = solution(session.certification_request(request()).unwrap());
    assert_eq!(set.parameter.name, "y");
    assert_eq!(set.polynomial, ["1", "-1", "0", "0", "0", "1"]);
    assert_eq!(set.coordinates[0], ["1", "0", "1"]);
    assert_eq!(set.coordinates[1], ["0", "1"]);
    assert_eq!(set.roots.len(), 1);
    check_roots(&set, 1);
}

#[test]
fn singular_and_nonmonogenic_ideals_preserve_complete_distinct_points() {
    let mut repeated = registered(&["x^2", "y^2-3"], &["x", "y"]);
    let set = solution(repeated.certification_request(request()).unwrap());
    assert_eq!(set.polynomial, ["9", "0", "-6", "0", "1"]);
    assert_eq!(set.roots.len(), 2);
    check_roots(&set, 2);
    let mut nonmonogenic = registered(&["x^2", "x*y", "y^2"], &["x", "y"]);
    let set = solution(nonmonogenic.certification_request(request()).unwrap());
    assert_eq!(set.polynomial, ["0", "1"]);
    assert_eq!(set.roots.len(), 1);
    assert_eq!(evaluate(&set.coordinates[0], 0), rational("0"));
    assert_eq!(evaluate(&set.coordinates[1], 0), rational("0"));
    check_roots(&set, 1);
}

#[test]
fn unit_and_complex_only_ideals_are_complete_empty_real_sets() {
    for sources in [vec!["x^2-2", "y^2-3", "1"], vec!["x^2-2", "y^2+1"]] {
        let mut session = registered(&sources, &["x", "y"]);
        let set = solution(session.certification_request(request()).unwrap());
        assert!(set.roots.is_empty());
        assert_eq!(set.coordinates.len(), 2);
        check_roots(&set, 1);
    }
}

#[test]
fn positive_dimensional_systems_never_become_empty_or_incomplete_lists() {
    for sources in [vec!["x*y-1"], vec!["x-1"], vec!["0"]] {
        let mut session = registered(&sources, &["x", "y"]);
        assert_eq!(
            session.certification_request(request()).unwrap_err().code(),
            "unsupported_operation"
        );
    }
}

#[test]
fn all_original_sources_are_admitted_before_zero_or_unit_ideal_shortcuts() {
    for source in [
        "sin(x)",
        "0*sin(x)",
        "x^(1/2)",
        "x/x-1",
        "0*(1/x)",
        "0*((1/x)^0)",
        "0/(x-x)",
        "0^0",
    ] {
        let mut session = registered(&["1", source, "y"], &["x", "y"]);
        let before = session.residual_problem(&system()).unwrap().clone();
        assert_eq!(
            session.certification_request(request()).unwrap_err().code(),
            "refused",
            "{source}"
        );
        assert_eq!(session.residual_problem(&system()).unwrap(), &before);
    }
}

#[test]
fn unknown_systems_invalid_dimensions_and_wire_source_forgery_refuse() {
    let mut missing = Session::with_budget_limits(limits(), 256);
    assert_eq!(
        missing.certification_request(request()).unwrap_err().code(),
        "unknown_object"
    );
    let mut many_axes = registered(&["a"], &["a", "b", "c", "d", "e", "f", "g", "h", "i"]);
    assert_eq!(
        many_axes
            .certification_request(request())
            .unwrap_err()
            .code(),
        "invalid_input"
    );
    let mut session = registered(&["x-y", "y^2-2"], &["x", "y"]);
    for (key, value) in [
        ("residuals", json!(["1"])),
        ("variables", json!(["y", "x"])),
        ("domain", json!("complex")),
        ("solution_set", json!({"roots":[]})),
        ("accepted", json!(true)),
    ] {
        let mut frame = serde_json::to_value(request()).unwrap();
        frame["payload"][key] = value;
        let before = session.budget_snapshot();
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
        assert_eq!(response["code"], "malformed_request", "{frame}");
        // Envelope admission still spends ordinary resources; malformed
        // solver payloads cannot start independent mathematical verification.
        assert_eq!(
            session.budget_snapshot().verifier_pool,
            before.verifier_pool
        );
    }
    let mut frame = serde_json::to_value(request()).unwrap();
    frame["payload"]["schema_version"] = json!(2);
    let response: Value =
        serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
    assert_eq!(response["code"], "unsupported_schema");
}

#[test]
fn protected_verifier_and_ordinary_exhaustion_keep_typed_failures_across_retries() {
    for verifier_pool in [0, 1] {
        let mut bounded = limits();
        bounded.verifier_pool = verifier_pool;
        let mut session = Session::with_budget_limits(bounded, 256);
        session
            .certification_request(registration(&["x-y", "y^2-2"], &["x", "y"]))
            .unwrap();
        for _ in 0..2 {
            let before = session.budget_snapshot();
            assert_eq!(
                session.certification_request(request()).unwrap_err().code(),
                "budget_exhausted"
            );
            let after = session.budget_snapshot();
            assert!(after.verifier_pool <= before.verifier_pool);
            if verifier_pool == 0 {
                assert_eq!(after, before);
            }
        }
    }
    let mut probe = registered(&["x-y", "y^2-2"], &["x", "y"]);
    let after_registration = probe.budget_snapshot();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut bounded = limits();
        let index = dimension.index();
        bounded.dimensions[index] -= after_registration.dimensions[index];
        let mut session = Session::with_budget_limits(bounded, 256);
        session
            .certification_request(registration(&["x-y", "y^2-2"], &["x", "y"]))
            .unwrap();
        let before = session.budget_snapshot();
        assert_eq!(
            session.certification_request(request()).unwrap_err().code(),
            "budget_exhausted"
        );
        let after = session.budget_snapshot();
        assert_eq!(after.dimensions[index], 0);
        assert_eq!(after.verifier_pool, before.verifier_pool);
        for (remaining, previous) in after.dimensions.iter().zip(before.dimensions) {
            assert!(*remaining <= previous);
        }
    }
    // Keep the successful solve in the same test to establish that refusal is
    // the supplied budget, not an unsupported mathematical fixture.
    assert_eq!(
        solution(probe.certification_request(request()).unwrap())
            .roots
            .len(),
        2
    );
}

#[test]
fn final_response_admission_cannot_publish_a_partial_coordinate_map() {
    let mut probe = registered(&["x-y", "y^2-2"], &["x", "y"]);
    probe.certification_request(request()).unwrap();
    let completed = probe.budget_snapshot();
    let slot = Dimension::ComputeSteps.index();
    let mut bounded = limits();
    bounded.dimensions[slot] -= completed.dimensions[slot] + 1;
    let mut session = Session::with_budget_limits(bounded, 256);
    session
        .certification_request(registration(&["x-y", "y^2-2"], &["x", "y"]))
        .unwrap();
    let original = session.residual_problem(&system()).unwrap().clone();
    let before = session.budget_snapshot();
    assert_eq!(
        session.certification_request(request()).unwrap_err().code(),
        "budget_exhausted"
    );
    assert!(session.budget_snapshot().verifier_pool < before.verifier_pool);
    assert_eq!(session.residual_problem(&system()).unwrap(), &original);
    assert_eq!(
        session.certification_request(request()).unwrap_err().code(),
        "budget_exhausted"
    );
}

#[test]
fn fresh_cli_returns_complete_algebraic_tuples_with_default_limits() {
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
    assert_eq!(send(request(), "missing")["code"], "unknown_object");
    assert_eq!(
        send(registration(&["x-y^2-1", "y^3-y"], &["x", "y"]), "register")["status"],
        "success"
    );
    let result = send(request(), "solve");
    assert_eq!(result["status"], "success", "{result}");
    assert_eq!(result["result"]["kind"], "real_polynomial_system_solved");
    assert_eq!(result["result"]["completeness"], "complete");
    assert_eq!(
        result["result"]["solution_set"]["polynomial"],
        json!(["0", "-1", "0", "1"])
    );
    assert_eq!(
        result["result"]["solution_set"]["roots"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        result["result"]["solution_set"]["coordinates"],
        json!([["1", "0", "1"], ["0", "1"]])
    );
    drop(input);
    let exit = child.wait_with_output().unwrap();
    assert!(exit.status.success());
    assert!(
        exit.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&exit.stderr)
    );
}
