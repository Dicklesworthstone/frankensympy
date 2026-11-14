//! Fresh-process replay followed by actual numerical solving through NDJSON.

use frankensympy::MAX_ENVELOPE_BYTES;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

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

    fn send(&mut self, operation: &str, mut payload: Value, id: &str) -> Value {
        payload["schema_version"] = json!(1);
        let frame = json!({"id":id, "type":operation, "payload":payload}).to_string();
        assert!(frame.len() <= MAX_ENVELOPE_BYTES);
        let input = self.input.as_mut().unwrap();
        writeln!(input, "{frame}").unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "response before EOF"
        );
        assert!(line.len() <= MAX_ENVELOPE_BYTES);
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], id, "{response}");
        response
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

fn success(response: Value) -> Value {
    assert_eq!(response["status"], "success", "{response}");
    response["result"].clone()
}

fn refused(response: Value, code: &str) {
    assert_eq!(response["status"], "error", "{response}");
    assert_eq!(response["code"], code, "{response}");
    assert!(response.get("result").is_none());
}

fn approximate(result: &Value) {
    assert_eq!(result["outcome"], "approximate");
    assert_eq!(result["numeric_model"], "binary64");
    assert!(result.get("evidence").is_none());
    assert!(result.get("claim").is_none());
    assert!(result.get("certificate").is_none());
}

fn near(actual: &Value, expected: f64, tolerance: f64) {
    let actual = actual.as_f64().unwrap();
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected}"
    );
}

fn register_sources(cli: &mut Cli) {
    success(cli.send(
        "RegisterResidualSystem",
        json!({"system":"system:portable", "residuals":["x^2-2"], "variables":["x"]}),
        "register-residual",
    ));
    success(cli.send(
        "RegisterObjective",
        json!({"objective":"objective:portable", "source":"x^2+3", "variables":["x"]}),
        "register-objective",
    ));
}

#[test]
fn default_cli_replays_then_solves_and_minimizes_after_the_sender_process_exits() {
    let mut sender = Cli::new();
    register_sources(&mut sender);
    success(sender.send(
        "CompileResidualSystem",
        json!({"system":"system:portable", "program":"program:sender"}),
        "compile-residual",
    ));
    success(sender.send(
        "CompileObjective",
        json!({"objective":"objective:portable", "program":"objective_program:sender"}),
        "compile-objective",
    ));
    let residual = success(sender.send(
        "ExportCompiledResidualSystem",
        json!({"program":"program:sender"}),
        "export-residual",
    ))["artifact"]
        .as_str()
        .unwrap()
        .to_owned();
    let objective = success(sender.send(
        "ExportCompiledObjective",
        json!({"program":"objective_program:sender"}),
        "export-objective",
    ))["artifact"]
        .as_str()
        .unwrap()
        .to_owned();
    sender.finish();

    let mut receiver = Cli::new();
    refused(
        receiver.send(
            "SolveCompiledResidualSystem",
            json!({"program":"program:sender", "backend":"frankennumpy", "initial":["1"]}),
            "no-shared-program",
        ),
        "unknown_object",
    );
    register_sources(&mut receiver);
    let admitted_residual = success(receiver.send(
        "ReplayCompiledResidualSystem",
        json!({"system":"system:portable", "program":"program:received", "artifact":residual}),
        "replay-residual",
    ));
    let admitted_objective = success(receiver.send(
        "ReplayCompiledObjective",
        json!({"objective":"objective:portable", "program":"objective_program:received", "artifact":objective}),
        "replay-objective",
    ));
    let solved = success(receiver.send(
        "SolveCompiledResidualSystem",
        json!({"program":"program:received", "backend":"frankenscipy_analytic", "initial":["1"]}),
        "solve-restored",
    ));
    approximate(&solved);
    assert_eq!(solved["kind"], "compiled_residual_system_solved");
    assert_eq!(solved["backend"], "frankenscipy_analytic");
    assert_eq!(solved["termination"], "residual_tolerance");
    assert_eq!(
        solved["context_digest"],
        admitted_residual["context_digest"]
    );
    near(&solved["root"][0], 2.0_f64.sqrt(), 1e-9);
    assert!(solved["residual_norm"].as_f64().unwrap() <= 1e-10);

    let fitted = success(receiver.send(
        "SolveCompiledLeastSquares",
        json!({"program":"program:received", "backend":"dense", "initial":["1"]}),
        "fit-restored",
    ));
    approximate(&fitted);
    assert_eq!(fitted["kind"], "compiled_least_squares_solved");
    assert_eq!(fitted["termination"], "gradient_tolerance");
    near(&fitted["point"][0], 2.0_f64.sqrt(), 1e-9);
    near(&fitted["cost"], 0.0, 1e-18);

    let minimized = success(receiver.send(
        "MinimizeCompiledObjective",
        json!({"program":"objective_program:received", "initial":["3"]}),
        "minimize-restored",
    ));
    approximate(&minimized);
    assert_eq!(minimized["kind"], "compiled_objective_minimized");
    assert_eq!(minimized["termination"], "gradient_tolerance");
    assert_eq!(
        minimized["context_digest"],
        admitted_objective["context_digest"]
    );
    near(&minimized["point"][0], 0.0, 1e-10);
    near(&minimized["value"], 3.0, 1e-12);
    near(&minimized["gradient"][0], 0.0, 1e-10);

    success(receiver.send(
        "ReleaseCompiledResidualSystem",
        json!({"program":"program:received"}),
        "release-residual",
    ));
    refused(
        receiver.send(
            "SolveCompiledResidualSystem",
            json!({"program":"program:received", "backend":"sparse", "initial":["1"]}),
            "released-root",
        ),
        "unknown_object",
    );
    refused(
        receiver.send(
            "SolveCompiledLeastSquares",
            json!({"program":"program:received", "backend":"sparse", "initial":["1"]}),
            "released-fit",
        ),
        "unknown_object",
    );
    success(receiver.send(
        "ReleaseCompiledObjective",
        json!({"program":"objective_program:received"}),
        "release-objective",
    ));
    refused(
        receiver.send(
            "MinimizeCompiledObjective",
            json!({"program":"objective_program:received", "initial":["1"]}),
            "released-objective",
        ),
        "unknown_object",
    );
    receiver.finish();
}

#[test]
fn interactive_refusals_do_not_turn_nonroot_stationarity_into_convergence_or_poison_handles() {
    let mut cli = Cli::new();
    success(cli.send(
        "RegisterResidualSystem",
        json!({"system":"system:nonroot", "residuals":["x^2-1"], "variables":["x"]}),
        "register",
    ));
    success(cli.send(
        "CompileResidualSystem",
        json!({"system":"system:nonroot", "program":"program:nonroot"}),
        "compile",
    ));
    refused(
        cli.send(
            "SolveCompiledResidualSystem",
            json!({"program":"program:nonroot", "backend":"sparse", "initial":["0"]}),
            "stationary-nonroot",
        ),
        "numeric_no_convergence",
    );
    let fit = success(cli.send(
        "SolveCompiledLeastSquares",
        json!({"program":"program:nonroot", "backend":"dense", "initial":["0"]}),
        "stationary-fit",
    ));
    approximate(&fit);
    assert_eq!(fit["point"], json!([0.0]));
    assert_eq!(fit["residuals"], json!([-1.0]));
    assert_eq!(fit["cost"], 0.5);
    assert_eq!(fit["gradient_norm"], 0.0);
    refused(
        cli.send(
            "SolveCompiledResidualSystem",
            json!({"program":"program:nonroot", "backend":"frankenscipy", "initial":["2"],
            "options":{"max_iterations":200}}),
            "fixed-controls",
        ),
        "invalid_input",
    );
    refused(
        cli.send(
            "SolveCompiledResidualSystem",
            json!({"program":"program:nonroot", "backend":"automatic", "initial":["2"]}),
            "unknown-backend",
        ),
        "malformed_request",
    );
    let solved = success(cli.send(
        "SolveCompiledResidualSystem",
        json!({"program":"program:nonroot", "backend":"sparse", "initial":["2"],
            "options":{"max_iterations":64, "residual_tolerance":"1e-12", "max_step_norm":"1"}}),
        "valid-after-refusals",
    ));
    approximate(&solved);
    near(&solved["root"][0], 1.0, 1e-10);
    assert!(solved["residual_norm"].as_f64().unwrap() <= 1e-12);
    cli.finish();
}
