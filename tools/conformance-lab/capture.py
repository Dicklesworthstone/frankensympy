#!/usr/bin/env python3
"""Capture driver for the FrankenSymPy conformance laboratory.

Spawns tools/conformance-lab/oracle_runner.py inside the isolated oracle
interpreter (one process per fixture file; no shared imports or objects),
validates every observation envelope against the required-field contract,
and writes:

  artifacts/conformance/<profile>/goldens/<fixture>.ndjson      goldens
  artifacts/conformance/<profile>/runs/<utc-stamp>.manifest.json

Usage:
  capture.py capture <profile-manifest.toml> [--oracle-python PATH]
  capture.py self-test <profile-manifest.toml> [--oracle-python PATH]

`capture` regenerates goldens. Goldens include the full environment
fingerprint per envelope (docs/CONFORMANCE_AND_BENCHMARKING.md section 3).

`self-test` is the harness-level mutation gate required by campaign stage C1:
it captures fresh observations and asserts the exact comparator REJECTS
deliberately weakened variants (printer flip, hash swap, dropped field). If
tampering is NOT detected, the gate fails closed.

Exit codes: 0 = success, 1 = gate failure, 2 = misuse.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "conformance"

REQUIRED_ENVELOPE_KEYS = {
    "schema_version",
    "profile_id",
    "fixture_id",
    "side",
    "outcome_class",
    "observations",
    "environment",
}
REQUIRED_ENV_KEYS = {"sympy_version", "python", "implementation", "platform", "env"}
REQUIRED_OBSERVATION_KEYS_RETURNED = {
    "type",
    "module",
    "mro",
    "hash_sha256_of_py_hash",
    "printers",
    "args_repr",
    "func",
    "pickle_v4",
    "pickle_v5",
}


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def load_profile(manifest_path: Path) -> dict:
    with open(manifest_path, "rb") as fh:
        profile = tomllib.load(fh)
    for section in ("profile_id", "upstream", "environment", "inventory"):
        if section not in profile:
            raise SystemExit(f"profile manifest missing section/key: {section}")
    return profile


def oracle_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("FSYM_ORACLE_PYTHON")
    if env:
        return env
    default = Path.home() / ".venvs" / "fsym-oracle-sympy-1.14.0" / "bin" / "python"
    if default.exists():
        return str(default)
    raise SystemExit(
        "no oracle interpreter found; pass --oracle-python or set FSYM_ORACLE_PYTHON"
    )


def runner_path() -> Path:
    path = Path(__file__).resolve().parent / "oracle_runner.py"
    if not path.exists():
        raise SystemExit(f"missing runner: {path}")
    return path


def validate_envelope(envelope: dict, expected_profile: str) -> None:
    missing = REQUIRED_ENVELOPE_KEYS - envelope.keys()
    if missing:
        raise ValueError(f"envelope missing keys: {sorted(missing)}")
    if envelope["schema_version"] != 1:
        raise ValueError("unknown schema_version (fail closed)")
    if envelope["profile_id"] != expected_profile:
        raise ValueError(f"profile mismatch: {envelope['profile_id']!r}")
    env_keys = set(envelope["environment"].keys())
    if not REQUIRED_ENV_KEYS <= env_keys:
        raise ValueError(f"environment fingerprint incomplete: {sorted(env_keys)}")
    if envelope["outcome_class"] == "returned":
        obs_keys = set(envelope["observations"].keys())
        missing_obs = REQUIRED_OBSERVATION_KEYS_RETURNED - obs_keys
        if missing_obs:
            raise ValueError(f"returned observation missing keys: {sorted(missing_obs)}")
    elif envelope["outcome_class"] != "raised":
        raise ValueError(f"unknown outcome_class: {envelope['outcome_class']!r}")


def capture_file(profile: dict, fixture_path: Path, py: str) -> list[dict]:
    """Runs one fixture file in a fresh isolated oracle subprocess."""
    env = dict(os.environ)
    env.update(profile["environment"]["env_overrides"])
    proc = subprocess.run(
        [py, str(runner_path()), str(fixture_path), profile["profile_id"]],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner exited {proc.returncode} for {fixture_path.name}: {proc.stderr[-400:]}"
        )
    envelopes = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        envelope = json.loads(line)
        validate_envelope(envelope, profile["profile_id"])
        envelopes.append(envelope)
    return envelopes


def golden_name_for(fixture_rel: str) -> str:
    return (
        fixture_rel.removeprefix("fixtures/")
        .removesuffix(".json")
        + ".ndjson"
    )


def write_goldens(profile: dict, captured: dict[str, list[dict]]) -> Path:
    golden_dir = ARTIFACT_ROOT / profile["profile_id"] / "goldens"
    golden_dir.mkdir(parents=True, exist_ok=True)
    for name, envelopes in captured.items():
        target = golden_dir / name
        with open(target, "w", encoding="utf-8") as fh:
            for envelope in envelopes:
                fh.write(json.dumps(envelope, sort_keys=True) + "\n")
    return golden_dir


def compare(left: list[dict], right: list[dict]) -> list[str]:
    """Exact-surface comparator over normalized envelopes."""
    if len(left) != len(right):
        return ["envelope count differs"]
    return [
        lo.get("fixture_id", "?")
        for lo, ro in zip(left, right)
        if lo != ro
    ]


def cmd_capture(profile: dict, py: str) -> int:
    base = Path(__file__).resolve().parent
    captured: dict[str, list[dict]] = {}
    for rel in profile["inventory"]["fixtures"]:
        fixture_path = base / rel
        captured[golden_name_for(rel)] = capture_file(profile, fixture_path, py)

    golden_dir = write_goldens(profile, captured)

    total = sum(len(v) for v in captured.values())
    raised = sum(
        1 for v in captured.values() for e in v if e["outcome_class"] == "raised"
    )
    run_manifest = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "profile_id": profile["profile_id"],
        "upstream_commit": profile["upstream"]["commit"],
        "fixture_files": len(captured),
        "envelopes": total,
        "raised_outcomes": raised,
        "golden_dir": str(golden_dir.relative_to(REPO_ROOT)),
        "golden_digests": {
            name: hashlib.sha256((golden_dir / name).read_bytes()).hexdigest()
            for name in captured
        },
    }
    runs_dir = ARTIFACT_ROOT / profile["profile_id"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = runs_dir / f"{stamp}.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(run_manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


def load_goldens(profile: dict) -> dict[str, list[dict]]:
    golden_dir = ARTIFACT_ROOT / profile["profile_id"] / "goldens"
    goldens: dict[str, list[dict]] = {}
    for rel in profile["inventory"]["fixtures"]:
        name = golden_name_for(rel)
        path = golden_dir / name
        if not path.exists():
            raise SystemExit(f"missing golden {name}; run `capture.py capture` first")
        with open(path, encoding="utf-8") as fh:
            goldens[name] = [json.loads(line) for line in fh if line.strip()]
    return goldens


def weakened_variants(envelopes: list[dict]) -> dict[str, list[dict]]:
    """Deliberately weakened envelope variants the comparator must reject."""
    variants: dict[str, list[dict]] = {}

    printer_flip = copy.deepcopy(envelopes)
    for envelope in printer_flip:
        printers = envelope.get("observations", {}).get("printers")
        if printers and "latex" in printers:
            printers["latex"] += "\\;"
            break
    variants["printer-weakened"] = printer_flip

    variants["hash-swapped"] = [
        {
            **e,
            "observations": {
                **e["observations"],
                "hash_sha256_of_py_hash": "f" * 64,
            },
        }
        for e in envelopes
        if "hash_sha256_of_py_hash" in e.get("observations", {})
    ]

    variants["dropped-field"] = [
        {k: v for k, v in e.items() if k != "environment"} for e in envelopes
    ]

    variants["count-shrunk"] = envelopes[:-1] if len(envelopes) > 1 else []

    return variants


def cmd_self_test(profile: dict, py: str) -> int:
    base = Path(__file__).resolve().parent
    goldens = load_goldens(profile)

    # Sanity: unmodified fresh observations must match goldens exactly.
    first_rel = profile["inventory"]["fixtures"][0]
    fresh = capture_file(profile, base / first_rel, py)
    golden_first = goldens[golden_name_for(first_rel)]
    diffs = compare(golden_first, fresh)
    if diffs:
        return fail(f"fresh capture does not match goldens before mutation: {diffs}")

    # Gate: every weakened variant must be REJECTED by the exact comparator.
    rejected = 0
    checked_files = 0
    for name, golden in goldens.items():
        for label, mutated in weakened_variants(golden).items():
            checked_files += 1
            if compare(golden, mutated):
                rejected += 1
            else:
                return fail(f"comparator FAILED to reject mutant: {label} ({name})")

    print(
        json.dumps(
            {
                "self_test": "passed",
                "fresh_matches_golden": True,
                "mutants_checked": checked_files,
                "mutants_rejected": rejected,
                "note": "comparator rejects printer/hash weakening, dropped fields, count shrink",
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    mode = args[0]
    py = None
    if "--oracle-python" in args:
        idx = args.index("--oracle-python")
        py = args[idx + 1] if idx + 1 < len(args) else None
    profile = load_profile(Path(args[1]))
    interpreter = oracle_python(py)
    if mode == "capture":
        return cmd_capture(profile, interpreter)
    if mode == "self-test":
        return cmd_self_test(profile, interpreter)
    print(f"unknown mode: {mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
