mod support;
use fsym_formal::{MAX_JSON_BYTES, ProjectionEnvelope, checker::check_projection, project_capsule};
use std::io::Read;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = support::capsule();
    // Authority is derived from fixed hand-built constants, never from JSON.
    let root = fixture
        .claim
        .digest()
        .map_err(|e| format!("fixture claim: {e:?}"))?;
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let envelope = if args.is_empty() {
        let bytes = fixture
            .encode()
            .map_err(|e| format!("fixture bytes: {e:?}"))?;
        project_capsule(&bytes, root, &|_| false)?.into_envelope()
    } else if args.len() == 2 && args[0] == "--check" {
        let file = std::fs::File::open(&args[1])?;
        let mut bytes = Vec::new();
        file.take(MAX_JSON_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        ProjectionEnvelope::from_json(&bytes)?
    } else {
        return Err("usage: project [--check JSON_PATH]".into());
    };
    let checked = check_projection(&envelope, root, &|_| false)?;
    println!("{}", serde_json::to_string(checked.envelope())?);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
