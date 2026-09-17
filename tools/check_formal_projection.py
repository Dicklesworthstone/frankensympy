#!/usr/bin/env python3
"""Offline Lean-core projection gate. Never changes native evidence classes.

Supply the built fsym-formal `project` example, an explicit Lean executable,
and an independently frozen environment digest. Artifacts are retained, never
removed. No downloads, shell execution, plugins, or dynamic code loaders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import resource

PIN = "Lean (version 4.32.2, x86_64-unknown-linux-gnu, commit f3b06c705e6c85f5314019d5d3baab0fec5b580c, Release)"
ENVIRONMENT_SHA256 = "88d1bfed5e2ba13e7dd28043c70a303f1ea1b301220023fe6b9a1a5d14368f9f"
MAX_OUTPUT = 131072


def environment(lean: Path) -> dict:
    """Bind executable plus complete installed lib tree, not a version banner alone."""
    prefix = lean.parent.parent
    files = [lean] + sorted(p for p in (prefix / "lib").rglob("*") if p.is_file())
    rows = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        rows.append({"path": str(path.relative_to(prefix)), "sha256": digest.hexdigest()})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "files": rows}


def child_limits() -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT, MAX_OUTPUT))


def run(command: list[str], work: Path, label: str, seconds: float = 60) -> dict:
    """One owned synchronous subprocess; timeout kills and waits for direct child.

    The admitted commands create no child processes (Lean is run without --run).
    Native admission and fixed source grammar precede Lean invocation.
    """
    out_path, err_path = work / f"{label}.stdout", work / f"{label}.stderr"
    clean_env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    clean_env["LEAN_PATH"] = ""
    with out_path.open("wb") as out, err_path.open("wb") as err:
        try:
            result = subprocess.run(command, cwd=work, env=clean_env, stdout=out, stderr=err,
                                    timeout=seconds, check=False, preexec_fn=child_limits)
            status, code = "completed", result.returncode
        except subprocess.TimeoutExpired:
            status, code = "ResourceExhausted", None
        except FileNotFoundError:
            status, code = "MissingFormalDependency", None
    if out_path.stat().st_size > MAX_OUTPUT or err_path.stat().st_size > MAX_OUTPUT:
        return {"status": "ResourceExhausted", "exit": code, "command": command}
    return {"status": status, "exit": code, "command": command,
            "stdout": out_path.read_text(errors="replace"), "stderr": err_path.read_text(errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", type=Path, required=True)
    parser.add_argument("--projector", type=Path)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-environment")
    parser.add_argument("--inspect-environment", action="store_true")
    args = parser.parse_args()
    work = args.artifacts.resolve()
    work.mkdir(parents=True, exist_ok=False)
    lean = args.lean.resolve()
    manifest = environment(lean)
    (work / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.inspect_environment:
        print(manifest["sha256"])
        return 0
    if not args.projector or not args.expected_environment:
        parser.error("--projector and --expected-environment are required for checking")
    if not manifest["sha256"] == args.expected_environment == ENVIRONMENT_SHA256:
        print("REFUSE: wrong formal environment")
        return 1
    projector = str(args.projector.resolve())
    results = {}

    def record(label: str, command: list[str], seconds: float = 60) -> dict:
        result = run(command, work, label, seconds)
        results[label] = result
        (work / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        return result

    version = record("version", [str(lean), "--version"])
    if version["exit"] != 0 or version.get("stdout", "").strip() != PIN:
        print("REFUSE: wrong checker pin")
        return 1
    emitted = record("native-offline", [projector])
    if emitted["exit"] != 0:
        return 1
    envelope = json.loads(emitted["stdout"])
    envelope_path = work / "projection.json"
    envelope_path.write_text(json.dumps(envelope, indent=2) + "\n")
    checked = record("independent-mapping", [projector, "--check", str(envelope_path)])
    if checked["exit"] != 0:
        return 1
    replay = record("replay", [projector])
    if replay["exit"] != 0 or json.loads(replay["stdout"]) != envelope:
        print("FAIL: deterministic projection replay")
        return 1
    source = envelope["lean_source"]
    source_path = work / "Projected.lean"
    source_path.write_text(source)
    # -t0 rechecks imported declarations; -j1 excludes scheduler variation.
    flags = [str(lean), "-t0", "-j1", "-M512", "-T200000"]
    foreign = record("foreign-positive", flags + [str(source_path)])
    expected_axioms = "'projected_product' depends on axioms: [propext]"
    if foreign["exit"] != 0 or foreign.get("stdout", "").strip() != expected_axioms or foreign.get("stderr"):
        print("FAIL: foreign acceptance or unexpected axioms/diagnostics")
        return 1
    # Every top-level omission and substitution must fail strict native mapping.
    for key in envelope:
        for operation in ("omit", "alter"):
            mutant = dict(envelope)
            if operation == "omit":
                del mutant[key]
            else:
                mutant[key] = "forged"
            path = work / f"{operation}-{key}.json"
            path.write_text(json.dumps(mutant))
            result = record(f"{operation}-{key}", [projector, "--check", str(path)])
            if result["exit"] in (0, None):
                print(f"FAIL: mutation not rejected: {operation} {key}")
                return 1
    forged = dict(envelope, foreign_checker_accepted=True)
    path = work / "forged-checker.json"
    path.write_text(json.dumps(forged))
    if record("forged-checker", [projector, "--check", str(path)])["exit"] in (0, None):
        return 1
    # Deliberately false theorem uses the same operator environment; checker,
    # not a pre-recorded boolean, must reject it.
    false_source = source.split("theorem projected_product", 1)[0] + (
        "theorem projected_product : polyMul [1] [1] = [2] := by decide\n"
        "#print axioms projected_product\n"
    )
    false_path = work / "Rejected.lean"
    false_path.write_text(false_source)
    if record("foreign-negative", flags + [str(false_path)])["exit"] in (0, None):
        return 1
    missing = record("missing-checker", [str(work / "missing-lean"), str(source_path)])
    if missing["status"] != "MissingFormalDependency":
        return 1
    exhausted = record("foreign-resource", [str(lean), "-j1", "-M1", str(source_path)])
    if exhausted["exit"] in (0, None):
        return 1
    print("PASS: native offline, independent mapping mutations, fixed replay, pinned foreign positive/negative, missing checker and resource refusal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
