//! Complete finite real rational systems through the production Session and CLI.

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
    ResidualSystemId::new("system:rational").unwrap()
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
    CertificationRequest::SolveRealRationalSystem {
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
    let CertificationResponse::RealRationalSystemSolved { solution_set, .. } = response else {
        panic!("expected an exact rational-system solution")
    };
    solution_set
}

fn rational(value: &str) -> BigRational {
    match value.split_once('/') {
        Some((n, d)) => BigRational::new(n.parse().unwrap(), d.parse().unwrap()),
        None => BigRational::from_integer(value.parse().unwrap()),
    }
}

fn single_tuple(solution: &ExactRealPolynomialSolutionSet) -> Vec<BigRational> {
    assert_eq!(solution.roots.len(), 1);
    assert_eq!(solution.polynomial.len(), 2);
    let root = -rational(&solution.polynomial[0]) / rational(&solution.polynomial[1]);
    solution
        .coordinates
        .iter()
        .map(|coordinate| {
            coordinate
                .iter()
                .rev()
                .fold(rational("0"), |value, coefficient| {
                    value * &root + rational(coefficient)
                })
        })
        .collect()
}

#[test]
fn cancelled_denominator_factors_keep_only_valid_original_tuples() {
    let mut session = registered(&["(x^2-1)/(x-1)", "y-x"], &["x", "y"]);
    let original = session.residual_problem(&system()).unwrap().clone();
    let context = session.context_digest();
    session.construct("x", "100").unwrap();
    let before = session.budget_snapshot();
    let response = session.certification_request(request()).unwrap();
    let CertificationResponse::RealRationalSystemSolved {
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
        panic!("expected rational-system response")
    };
    assert_eq!(solved, system());
    assert_eq!(variables, original.variables());
    assert_eq!(context_digest, context);
    assert_eq!(evidence, "exact_cross_checked");
    assert_eq!(
        claim,
        "complete_distinct_real_solution_set_of_original_rational_system"
    );
    assert_eq!((completeness, domain), ("complete", "real"));
    assert_eq!(
        single_tuple(&solution_set),
        [rational("-1"), rational("-1")]
    );
    assert_eq!(solution_set.coordinates.len(), 2);
    assert!(!variables.contains(&solution_set.parameter));
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
fn irrational_grid_projection_keeps_both_cross_points_and_excludes_the_diagonal() {
    let mut session = registered(&["(x^2-2)/(x-y)", "y^2-2"], &["x", "y"]);
    let set = solution(session.certification_request(request()).unwrap());
    // t=1/(x-y), t^2=1/8 gives exactly (4t,-4t). These are the
    // two off-diagonal points; the other two polynomial roots are poles.
    assert_eq!(set.polynomial, ["-1/8", "0", "1"]);
    assert_eq!(set.coordinates, [vec!["0", "4"], vec!["0", "-4"]]);
    assert_eq!(set.roots.len(), 2);
    let certificate = RealRootIsolationCertificate {
        intervals: set
            .roots
            .iter()
            .map(|root| RealRootInterval {
                lower: rational(&root.lower),
                upper: rational(&root.upper),
                multiplicity: 1,
            })
            .collect(),
    };
    verify_real_root_isolation(
        &set.polynomial
            .iter()
            .map(|q| rational(q))
            .collect::<Vec<_>>(),
        &certificate,
        &mut Unbounded,
    )
    .unwrap();
}

#[test]
fn original_hidden_poles_and_order_survive_registration_and_projection() {
    for sources in [
        vec!["x+0*(1/y)", "y*(y-1)"],
        vec!["x", "(1/y)^0-1", "y*(y-1)"],
    ] {
        let mut session = registered(&sources, &["y", "x"]);
        let response = session.certification_request(request()).unwrap();
        let CertificationResponse::RealRationalSystemSolved {
            variables,
            solution_set,
            ..
        } = response
        else {
            panic!("expected projected solution")
        };
        assert_eq!(
            variables
                .iter()
                .map(|axis| axis.name.as_str())
                .collect::<Vec<_>>(),
            ["y", "x"]
        );
        assert_eq!(single_tuple(&solution_set), [rational("1"), rational("0")]);
    }
}

#[test]
fn complete_empty_sets_are_distinct_from_unrecognized_positive_dimensional_sets() {
    for sources in [
        vec!["1/x"],
        vec!["x", "0*(1/x)"],
        vec!["1/(1/x)"],
        vec!["(x^2+1)/x"],
    ] {
        let mut session = registered(&sources, &["x"]);
        let set = solution(session.certification_request(request()).unwrap());
        assert!(set.roots.is_empty(), "{sources:?}");
        assert_eq!(set.coordinates.len(), 1);
    }
    for sources in [vec!["x/y"], vec!["0*(1/x)"], vec!["x/x-1"], vec!["x*y-1"]] {
        let mut session = registered(&sources, &["x", "y"]);
        assert_eq!(
            session.certification_request(request()).unwrap_err().code(),
            "unsupported_operation"
        );
    }
}

#[test]
fn polynomial_inputs_reuse_the_existing_complete_representation() {
    let mut session = registered(&["x^2-2", "y^2-3"], &["x", "y"]);
    let polynomial = session
        .certification_request(CertificationRequest::SolveRealPolynomialSystem {
            schema_version: 1,
            system: system(),
        })
        .unwrap();
    let CertificationResponse::RealPolynomialSystemSolved {
        solution_set: expected,
        ..
    } = polynomial
    else {
        panic!("expected polynomial response")
    };
    assert_eq!(
        solution(session.certification_request(request()).unwrap()),
        expected
    );
    let mut rational_source = registered(&["(x^2-1)/(x-1)"], &["x"]);
    assert_eq!(
        rational_source
            .certification_request(CertificationRequest::SolveRealPolynomialSystem {
                schema_version: 1,
                system: system(),
            })
            .unwrap_err()
            .code(),
        "refused"
    );
    assert_eq!(
        single_tuple(&solution(
            rational_source.certification_request(request()).unwrap()
        )),
        [rational("-1")]
    );
}

#[test]
fn original_invalid_sources_cannot_hide_behind_inconsistent_equations() {
    for source in ["sin(x)", "0*sin(x)", "x^(1/2)", "0/(x-x)", "0^0"] {
        let mut session = registered(&["1", source], &["x"]);
        let original = session.residual_problem(&system()).unwrap().clone();
        assert_eq!(
            session.certification_request(request()).unwrap_err().code(),
            "refused",
            "{source}"
        );
        assert_eq!(session.residual_problem(&system()).unwrap(), &original);
    }
}

#[test]
fn wire_requests_cannot_replace_sources_or_publish_a_claimed_solution() {
    let mut session = registered(&["(x^2-1)/(x-1)"], &["x"]);
    for (key, value) in [
        ("residuals", json!(["1"])),
        ("variables", json!(["y"])),
        ("domain", json!("complex")),
        ("solution_set", json!({"roots":[]})),
        ("accepted", json!(true)),
    ] {
        let mut frame = serde_json::to_value(request()).unwrap();
        frame["payload"][key] = value;
        let before = session.budget_snapshot();
        let response: Value =
            serde_json::from_str(&session.handle_envelope(&frame.to_string())).unwrap();
        assert_eq!(response["code"], "malformed_request");
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
    let mut missing = Session::with_budget_limits(limits(), 256);
    assert_eq!(
        missing.certification_request(request()).unwrap_err().code(),
        "unknown_object"
    );
    let mut oversized = registered(&["a"], &["a", "b", "c", "d", "e", "f", "g", "h", "i"]);
    assert_eq!(
        oversized
            .certification_request(request())
            .unwrap_err()
            .code(),
        "invalid_input"
    );
}

#[test]
fn localization_and_solving_share_protected_and_ordinary_session_resources() {
    let source = ["(x^2-1)/(x-1)"];
    for verifier_pool in [0, 1] {
        let mut bounded = limits();
        bounded.verifier_pool = verifier_pool;
        let mut session = Session::with_budget_limits(bounded, 256);
        session
            .certification_request(registration(&source, &["x"]))
            .unwrap();
        for _ in 0..2 {
            let before = session.budget_snapshot();
            assert_eq!(
                session.certification_request(request()).unwrap_err().code(),
                "budget_exhausted"
            );
            assert!(session.budget_snapshot().verifier_pool <= before.verifier_pool);
            if verifier_pool == 0 {
                assert_eq!(session.budget_snapshot(), before);
            }
        }
    }
    let mut probe = registered(&source, &["x"]);
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
            .certification_request(registration(&source, &["x"]))
            .unwrap();
        let before = session.budget_snapshot();
        assert_eq!(
            session.certification_request(request()).unwrap_err().code(),
            "budget_exhausted"
        );
        let after = session.budget_snapshot();
        assert_eq!(after.dimensions[index], 0);
        assert_eq!(after.verifier_pool, before.verifier_pool);
    }
    assert_eq!(
        single_tuple(&solution(probe.certification_request(request()).unwrap())),
        [rational("-1")]
    );
}

#[test]
fn final_output_admission_cannot_publish_a_partial_projected_solution() {
    let source = ["(x^2-1)/(x-1)"];
    let mut probe = registered(&source, &["x"]);
    probe.certification_request(request()).unwrap();
    let slot = Dimension::ComputeSteps.index();
    let mut bounded = limits();
    bounded.dimensions[slot] -= probe.budget_snapshot().dimensions[slot] + 1;
    let mut session = Session::with_budget_limits(bounded, 256);
    session
        .certification_request(registration(&source, &["x"]))
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
fn fresh_cli_solves_a_rational_system_with_default_allowances() {
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
        send(
            registration(&["(x^2-1)/(x-1)", "y-x"], &["x", "y"]),
            "register"
        )["status"],
        "success"
    );
    let result = send(request(), "solve");
    assert_eq!(result["status"], "success", "{result}");
    assert_eq!(result["result"]["kind"], "real_rational_system_solved");
    assert_eq!(result["result"]["completeness"], "complete");
    assert_eq!(
        result["result"]["solution_set"]["coordinates"],
        json!([["-1"], ["-1"]])
    );
    assert_eq!(
        result["result"]["solution_set"]["roots"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(result["result"]["variables"].as_array().unwrap().len(), 2);
    drop(input);
    let exit = child.wait_with_output().unwrap();
    assert!(exit.status.success());
    assert!(
        exit.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&exit.stderr)
    );
}
