//! Real NDJSON family certification and implicit derivative responses.

use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Cli {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Cli {
    fn new(arguments: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_frankensympy"))
            .args(arguments)
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

    fn send(&mut self, request: Value) -> Value {
        self.send_raw(&request.to_string())
    }

    fn send_raw(&mut self, request: &str) -> Value {
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{request}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
        serde_json::from_str(&line).unwrap()
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

fn registration(source: &str) -> Value {
    json!({"id":"register","type":"RegisterParametricProblem","payload":{
        "schema_version":1,"problem":"parametric:source",
        "residuals":[source],"unknowns":["x"],"parameters":["p"]
    }})
}

fn certificate() -> Value {
    json!({"id":"uniform","type":"CertifyParametricRoot","payload":{
        "schema_version":1,"problem":"parametric:source","family":"family:accepted",
        "witness":{
            "unknown_domain":[{"lower":"-2","upper":"2"}],
            "parameter_domain":[{"lower":"-1","upper":"1"}],
            "center":["0"],"preconditioner":[["1"]],"precision_digits":16
        }
    }})
}

fn derivative() -> Value {
    json!({"id":"derivative","type":"EncloseParametricSensitivity","payload":{
        "schema_version":1,"family":"family:accepted","sensitivity":"sensitivity:accepted"
    }})
}

fn cover(max_leaves: usize) -> Value {
    json!({"id":"cover","type":"CertifyParametricCover","payload":{
        "schema_version":1,"problem":"parametric:source","family":"family:accepted",
        "unknown_domain":[{"lower":"0","upper":"2"}],
        "parameter_domain":[{"lower":"0","upper":"2"}],
        "precision_digits":16,"max_leaves":max_leaves
    }})
}

#[test]
fn interactive_cli_certifies_every_parameter_and_emits_exact_implicit_derivatives() {
    let mut cli = Cli::new(&[]);
    let registered = cli.send(registration("x-p"));
    assert_eq!(registered["status"], "success", "{registered}");
    assert_eq!(
        registered["result"]["kind"],
        "parametric_problem_registered"
    );
    assert_eq!(registered["result"]["unknowns"][0]["name"], "x");
    assert_eq!(registered["result"]["parameters"][0]["name"], "p");
    let checked = cli.send(certificate());
    assert_eq!(checked["status"], "success", "{checked}");
    assert_eq!(checked["id"], "uniform");
    assert_eq!(checked["result"]["kind"], "parametric_family_certified");
    assert_eq!(checked["result"]["evidence"], "certified_numeric");
    assert_eq!(
        checked["result"]["claim"],
        "for_every_parameter_exactly_one_nonsingular_root_in_unknown_box"
    );
    assert_eq!(
        checked["result"]["root_enclosure"],
        json!([{"lower":"-1","upper":"1"}])
    );
    assert_eq!(
        checked["result"]["parameter_domain"],
        json!([{"lower":"-1","upper":"1"}])
    );
    assert_eq!(checked["result"]["contraction_bound"], "0");
    assert_eq!(checked["result"]["leaf_count"], 1);
    assert_eq!(
        checked["result"]["context_digest"],
        registered["result"]["context_digest"]
    );
    let differentiated = cli.send(derivative());
    assert_eq!(differentiated["status"], "success", "{differentiated}");
    assert_eq!(differentiated["id"], "derivative");
    assert_eq!(
        differentiated["result"]["kind"],
        "parametric_sensitivity_enclosed"
    );
    assert_eq!(differentiated["result"]["evidence"], "certified_numeric");
    assert_eq!(
        differentiated["result"]["claim"],
        "implicit_solution_jacobian_enclosure"
    );
    assert_eq!(
        differentiated["result"]["solution_jacobian"],
        json!([[{"lower":"1","upper":"1"}]])
    );
    assert_eq!(
        differentiated["result"]["context_digest"],
        checked["result"]["context_digest"]
    );
    assert_eq!(cli.send(derivative())["code"], "duplicate_object");
    cli.finish();
}

#[test]
fn adaptive_cli_refuses_incomplete_families_then_certifies_and_differentiates_the_cover() {
    let mut cli = Cli::new(&[
        "--compute-steps",
        "20000000",
        "--verifier-steps",
        "100000000",
        "--memory-bytes",
        "1073741824",
    ]);
    assert_eq!(cli.send(registration("(1+p^2)*x-1"))["status"], "success");
    let incomplete = cli.send(cover(1));
    assert_eq!(incomplete["code"], "inconclusive", "{incomplete}");
    assert_eq!(cli.send(derivative())["code"], "unknown_object");
    let accepted = cli.send(cover(128));
    assert_eq!(accepted["status"], "success", "{accepted}");
    assert_eq!(accepted["id"], "cover");
    assert!(accepted["result"]["leaf_count"].as_u64().unwrap() > 1);
    assert_eq!(
        accepted["result"]["parameter_domain"],
        json!([{"lower":"0","upper":"2"}])
    );
    let differentiated = cli.send(derivative());
    assert_eq!(differentiated["status"], "success", "{differentiated}");
    assert_eq!(
        differentiated["result"]["leaf_count"],
        accepted["result"]["leaf_count"]
    );
    for interval in accepted["result"]["root_enclosure"]
        .as_array()
        .unwrap()
        .iter()
        .chain(
            differentiated["result"]["solution_jacobian"][0]
                .as_array()
                .unwrap(),
        )
    {
        assert!(interval["lower"].is_string());
        assert!(interval["upper"].is_string());
    }
    assert_eq!(cli.send(cover(128))["code"], "duplicate_object");
    cli.finish();
}

#[test]
fn zero_protected_allowance_cannot_publish_a_family_or_derive_sensitivity() {
    let mut cli = Cli::new(&["--verifier-steps", "0"]);
    assert_eq!(cli.send(registration("x-p"))["status"], "success");
    let refused = cli.send(certificate());
    assert_eq!(refused["code"], "budget_exhausted", "{refused}");
    assert_eq!(cli.send(derivative())["code"], "unknown_object");
    cli.finish();
}

#[test]
fn family_wire_rejects_forged_evidence_ambiguous_fields_and_inexact_coordinates() {
    let mut cli = Cli::new(&[]);
    let mut future = registration("x-p");
    future["payload"]["schema_version"] = json!(2);
    assert_eq!(cli.send(future)["code"], "unsupported_schema");
    let duplicate = r#"{"type":"RegisterParametricProblem","payload":{"schema_version":1,"problem":"parametric:source","residuals":["x-p"],"unknowns":["x"],"parameters":["p"],"parameters":["x"]}}"#;
    assert_eq!(cli.send_raw(duplicate)["code"], "malformed_request");
    let mut wrong_type = registration("x-p");
    wrong_type["payload"]["problem"] = json!("system:source");
    assert_eq!(cli.send(wrong_type)["code"], "malformed_request");
    assert_eq!(cli.send(registration("x-p"))["status"], "success");
    let mut forged = certificate();
    forged["payload"]["witness"]["verified"] = json!(true);
    assert_eq!(cli.send(forged)["code"], "malformed_request");
    let mut inexact = certificate();
    inexact["payload"]["witness"]["center"][0] = json!(0.0);
    assert_eq!(cli.send(inexact)["code"], "malformed_request");
    let mut expression = certificate();
    expression["payload"]["witness"]["center"][0] = json!("0+0");
    assert_eq!(cli.send(expression)["code"], "invalid_input");
    let mut midpoint_only = certificate();
    midpoint_only["payload"]["witness"]["unknown_domain"] = json!([{"lower":"-1","upper":"1"}]);
    midpoint_only["payload"]["witness"]["parameter_domain"] = json!([{"lower":"-2","upper":"2"}]);
    assert_eq!(cli.send(midpoint_only)["code"], "refused");
    assert_eq!(cli.send(derivative())["code"], "unknown_object");
    let valid = cli.send(certificate());
    assert_eq!(valid["status"], "success", "{valid}");
    assert_eq!(cli.send(derivative())["status"], "success");
    cli.finish();
}
