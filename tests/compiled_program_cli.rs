//! Fresh processes exchange derivative candidates through the real NDJSON binary.

use frankensympy::MAX_ENVELOPE_BYTES;
use frankensympy::artifact_stream::ARTIFACT_CHUNK_BYTES;
use frankensympy::certification_session::MAX_SESSION_ROOT_ARTIFACT_BYTES;
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
        let response: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["id"], id);
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

fn register_residuals(cli: &mut Cli, sources: &[&str]) {
    success(cli.send(
        "RegisterResidualSystem",
        json!({"system":"system:portable", "residuals":sources, "variables":["y","x"]}),
        "register-residuals",
    ));
}

fn register_objective(cli: &mut Cli) {
    success(cli.send(
        "RegisterObjective",
        json!({"objective":"objective:portable", "source":"x^2+x*y+3*y^2", "variables":["x","y"]}),
        "register-objective",
    ));
}

fn assert_approximate(result: &Value) {
    assert_eq!(result["outcome"], "approximate");
    assert_eq!(result["numeric_model"], "binary64");
    assert!(result.get("evidence").is_none());
}

#[test]
fn default_cli_exports_and_replays_both_program_kinds_after_sender_exits() {
    let sources = ["x^2+y", "sin(y)", "7"];
    let mut sender = Cli::new(&[]);
    register_residuals(&mut sender, &sources);
    register_objective(&mut sender);
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

    let mut receiver = Cli::new(&[]);
    let absent = receiver.send(
        "EvaluateResidualJacobian",
        json!({"program":"program:sender", "point":["0","2"]}),
        "no-shared-state",
    );
    assert_eq!(absent["code"], "unknown_object");
    register_residuals(&mut receiver, &sources);
    register_objective(&mut receiver);
    let residual_result = success(receiver.send(
        "ReplayCompiledResidualSystem",
        json!({"system":"system:portable", "program":"program:received", "artifact":residual}),
        "replay-residual",
    ));
    assert_eq!(residual_result["kind"], "residual_system_compiled");
    assert_eq!(residual_result["proofs_verified"], 3);
    let objective_result = success(receiver.send(
        "ReplayCompiledObjective",
        json!({"objective":"objective:portable", "program":"objective_program:received", "artifact":objective}),
        "replay-objective",
    ));
    assert_eq!(objective_result["kind"], "objective_compiled");
    let forward = success(receiver.send(
        "EvaluateJacobianVectorProduct",
        json!({"program":"program:received", "point":["0","2"], "direction":["3","-2"]}),
        "forward",
    ));
    assert_approximate(&forward);
    assert_eq!(forward["product"], json!([-5.0, 3.0, 0.0]));
    let transpose = success(receiver.send(
        "EvaluateJacobianTransposeVectorProduct",
        json!({"program":"program:received", "point":["0","2"], "weights":["2","-3","11"]}),
        "transpose",
    ));
    assert_approximate(&transpose);
    assert_eq!(transpose["product"], json!([-1.0, 8.0]));
    let hessian = success(receiver.send(
        "EvaluateHessianVectorProduct",
        json!({"program":"objective_program:received", "point":["2","-1"], "direction":["3","-2"]}),
        "hessian",
    ));
    assert_approximate(&hessian);
    assert_eq!(hessian["product"], json!([4.0, -9.0]));
    success(receiver.send(
        "ReleaseCompiledResidualSystem",
        json!({"program":"program:received"}),
        "release",
    ));
    let absent = receiver.send(
        "EvaluateResidualJacobian",
        json!({"program":"program:received", "point":["0","2"]}),
        "released",
    );
    assert_eq!(absent["code"], "unknown_object");
    receiver.finish();
}

#[test]
fn cli_streams_a_large_program_without_raising_frame_limits_or_trusting_transfer() {
    let arguments = [
        "--compute-steps",
        "1000000000",
        "--memory-bytes",
        "4294967296",
        "--allocations",
        "10000000",
        "--verifier-steps",
        "1000000000",
    ];
    let sources = ["sin(x)+x*y+y^2"; 32];
    let mut sender = Cli::new(&arguments);
    register_residuals(&mut sender, &sources);
    success(sender.send(
        "CompileResidualSystem",
        json!({"system":"system:portable", "program":"program:sender"}),
        "compile",
    ));
    let exported = success(sender.send(
        "BeginArtifactExport",
        json!({"transfer":"transfer:out", "source":{"compiled_residual_system":"program:sender"}}),
        "begin-export",
    ));
    let manifest = exported["manifest"].clone();
    let bytes = manifest["total_bytes"].as_u64().unwrap() as usize;
    assert!(bytes > MAX_SESSION_ROOT_ARTIFACT_BYTES);
    let mut chunks = Vec::new();
    for index in 0..bytes.div_ceil(ARTIFACT_CHUNK_BYTES) {
        let response = success(sender.send(
            "ReadArtifactChunk",
            json!({"transfer":"transfer:out", "index":index}),
            "read",
        ));
        chunks.push(response["chunk"].clone());
    }
    sender.finish();

    let mut receiver = Cli::new(&arguments);
    register_residuals(&mut receiver, &sources);
    success(receiver.send(
        "BeginArtifactImport",
        json!({"transfer":"transfer:in", "manifest":manifest}),
        "begin-import",
    ));
    for chunk in chunks {
        success(receiver.send(
            "AppendArtifactChunk",
            json!({"transfer":"transfer:in", "chunk":chunk}),
            "append",
        ));
    }
    let absent = receiver.send(
        "EvaluateResidualJacobian",
        json!({"program":"program:received", "point":["1","0"]}),
        "bytes-not-proof",
    );
    assert_eq!(absent["code"], "unknown_object");
    let replayed = success(receiver.send(
        "ReplayCompiledResidualSystemFromArtifact",
        json!({"system":"system:portable", "program":"program:received", "transfer":"transfer:in"}),
        "replay",
    ));
    assert_eq!(replayed["residual_count"], 32);
    let evaluated = success(receiver.send(
        "EvaluateResidualJacobian",
        json!({"program":"program:received", "point":["1","0"]}),
        "evaluate",
    ));
    assert_eq!(evaluated["residuals"], json!(vec![1.0; 32]));
    assert_approximate(&evaluated);
    receiver.finish();
}
