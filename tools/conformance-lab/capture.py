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
  capture.py inventory <profile-manifest.toml> [--oracle-python PATH]

`capture` creates goldens for a new profile and refuses to overwrite an
existing profile. Goldens include the environment fingerprint per envelope
(docs/CONFORMANCE_AND_BENCHMARKING.md section 3).

`self-test` is the harness-level mutation gate required by campaign stage C1:
it captures fresh observations and asserts the exact comparator REJECTS
deliberately weakened variants (printer flip, hash swap, dropped field). If
tampering is NOT detected, the gate fails closed.

Exit codes: 0 = success, 1 = gate failure, 2 = misuse.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparators import REGISTRY, diff_envelopes

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
REQUIRED_ENV_KEYS = {
    "sympy_version",
    "python",
    "implementation",
    "platform",
    "sympy_path",
    "env",
}
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
    "is_number",
    "free_symbols",
}
REQUIRED_OBSERVATION_KEYS_RAISED = {
    "exception_module",
    "exception_type",
    "message_head",
}
MAX_FIXTURE_BYTES = 256 * 1024
MAX_FIXTURES_PER_FILE = 4_096
MAX_RUNNER_OUTPUT_BYTES = 8 * 1024 * 1024


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def load_profile(manifest_path: Path) -> dict:
    with open(manifest_path, "rb") as fh:
        profile = tomllib.load(fh)
    validate_profile(profile)
    return profile


def validate_profile(profile: dict) -> None:
    required = {
        "schema_version",
        "profile_id",
        "profile_status",
        "upstream",
        "environment",
        "runtime",
        "inventory",
        "comparators",
    }
    missing = required - profile.keys()
    unknown = profile.keys() - required
    if missing or unknown:
        raise ValueError(
            f"profile top-level keys invalid: missing={sorted(missing)} "
            f"unknown={sorted(unknown)}"
        )
    if profile["schema_version"] != 1:
        raise ValueError("unknown profile schema_version (fail closed)")
    if not isinstance(profile["profile_id"], str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9.-]*", profile["profile_id"]
    ):
        raise ValueError("profile_id contains unsafe or unsupported characters")
    if profile["profile_status"] != "active":
        raise ValueError("only active profiles can be captured")

    upstream = profile["upstream"]
    if set(upstream) != {"distribution", "version", "commit"}:
        raise ValueError("upstream profile keys do not match schema version 1")
    if upstream["distribution"] != "sympy" or not isinstance(upstream["version"], str):
        raise ValueError("profile must name an exact SymPy distribution version")
    commit = upstream["commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(ch not in "0123456789abcdef" for ch in commit)
    ):
        raise ValueError(
            "upstream commit must be a lowercase 40-digit hexadecimal digest"
        )

    environment = profile["environment"]
    required_environment = {
        "implementation",
        "python_version",
        "locale",
        "timezone",
        "hash_seed",
        "env_overrides",
    }
    if set(environment) != required_environment:
        raise ValueError("environment profile keys do not match schema version 1")
    overrides = environment["env_overrides"]
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in overrides.items()
    ):
        raise ValueError("environment.env_overrides must map strings to strings")
    if overrides.get("PYTHONHASHSEED") != str(environment["hash_seed"]):
        raise ValueError("PYTHONHASHSEED does not match the profile hash_seed")
    if overrides.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("profile must disable Python bytecode writes")

    runtime = profile["runtime"]
    if set(runtime) != {"oracle_isolation", "pickle_protocols", "printers"}:
        raise ValueError("runtime profile keys do not match schema version 1")
    if runtime["oracle_isolation"] != "subprocess":
        raise ValueError("profile requires an unsupported oracle isolation mode")
    if runtime["pickle_protocols"] != [4, 5]:
        raise ValueError("profile pickle protocols do not match the runner")
    if runtime["printers"] != ["str", "repr", "srepr", "latex", "pretty_ascii"]:
        raise ValueError("profile printer inventory does not match the runner")

    inventory = profile["inventory"]
    if set(inventory) != {"fixtures"}:
        raise ValueError("inventory profile keys do not match schema version 1")
    fixtures = inventory["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("inventory.fixtures must be a non-empty list")
    if len(fixtures) > MAX_FIXTURES_PER_FILE:
        raise ValueError("inventory fixture-file count exceeds the harness limit")
    if len(set(fixtures)) != len(fixtures):
        raise ValueError("inventory.fixtures contains duplicate paths")
    for rel in fixtures:
        if not isinstance(rel, str):
            raise TypeError("fixture paths must be strings")
        path = PurePosixPath(rel)
        if (
            path.is_absolute()
            or len(path.parts) != 2
            or path.parts[0] != "fixtures"
            or path.suffix != ".json"
            or path.name in {"", ".", ".."}
        ):
            raise ValueError(f"unsafe or unsupported fixture path: {rel!r}")
    if set(profile["comparators"]) != set(REGISTRY) or not all(
        isinstance(description, str) and description
        for description in profile["comparators"].values()
    ):
        raise ValueError("profile comparator registry does not match harness code")


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


def oracle_environment(
    profile: dict, *, pinned_hash_seed: bool = True
) -> dict[str, str]:
    """Build a deterministic child environment without Python path injection."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("PYTHON")
    }
    env.update(profile["environment"]["env_overrides"])
    env["LC_ALL"] = profile["environment"]["locale"]
    env["TZ"] = profile["environment"]["timezone"]
    if not pinned_hash_seed:
        env.pop("PYTHONHASHSEED", None)
    return env


def validate_environment(fingerprint: object, profile: dict) -> None:
    if not isinstance(fingerprint, dict):
        raise TypeError("environment fingerprint must be an object")
    missing = REQUIRED_ENV_KEYS - fingerprint.keys()
    if missing:
        raise ValueError(f"environment fingerprint missing keys: {sorted(missing)}")
    if fingerprint["sympy_version"] != profile["upstream"]["version"]:
        raise ValueError(
            "SymPy version mismatch: "
            f"required={profile['upstream']['version']!r} "
            f"actual={fingerprint['sympy_version']!r}"
        )
    expected_python = str(profile["environment"]["python_version"])
    actual_python = fingerprint["python"]
    if not isinstance(actual_python, str) or not (
        actual_python == expected_python
        or actual_python.startswith(expected_python + ".")
    ):
        raise ValueError(
            f"Python version mismatch: required={expected_python!r} actual={actual_python!r}"
        )
    expected_impl = str(profile["environment"]["implementation"])
    actual_impl = fingerprint["implementation"]
    if (
        not isinstance(actual_impl, str)
        or actual_impl.casefold() != expected_impl.casefold()
    ):
        raise ValueError(
            f"Python implementation mismatch: required={expected_impl!r} actual={actual_impl!r}"
        )
    if fingerprint["env"] != profile["environment"]["env_overrides"]:
        raise ValueError("oracle environment variables do not match the profile")
    if not isinstance(fingerprint["platform"], str) or not fingerprint["platform"]:
        raise ValueError("oracle platform fingerprint must be a non-empty string")
    if not isinstance(fingerprint["sympy_path"], str) or not fingerprint["sympy_path"]:
        raise ValueError("oracle SymPy path fingerprint must be a non-empty string")


def validate_envelope(envelope: dict, profile: dict) -> None:
    if not isinstance(envelope, dict):
        raise TypeError("observation envelope must be an object")
    missing = REQUIRED_ENVELOPE_KEYS - envelope.keys()
    unknown = envelope.keys() - REQUIRED_ENVELOPE_KEYS
    if missing or unknown:
        raise ValueError(
            f"envelope keys invalid: missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    if envelope["schema_version"] != 1:
        raise ValueError("unknown schema_version (fail closed)")
    if envelope["profile_id"] != profile["profile_id"]:
        raise ValueError(f"profile mismatch: {envelope['profile_id']!r}")
    if envelope["side"] != "upstream_oracle":
        raise ValueError(f"wrong observation side: {envelope['side']!r}")
    if not isinstance(envelope["fixture_id"], str) or not envelope["fixture_id"]:
        raise ValueError("fixture_id must be a non-empty string")
    validate_environment(envelope["environment"], profile)
    observations = envelope["observations"]
    if not isinstance(observations, dict):
        raise TypeError("observations must be an object")
    if envelope["outcome_class"] == "returned":
        if set(observations) != REQUIRED_OBSERVATION_KEYS_RETURNED:
            raise ValueError("returned observation keys do not match schema version 1")
    elif envelope["outcome_class"] == "raised":
        if set(observations) != REQUIRED_OBSERVATION_KEYS_RAISED:
            raise ValueError("raised observation keys do not match schema version 1")
    else:
        raise ValueError(f"unknown outcome_class: {envelope['outcome_class']!r}")


def load_fixture_ids(fixture_path: Path) -> list[str]:
    fixture_root = (Path(__file__).resolve().parent / "fixtures").resolve()
    resolved = fixture_path.resolve()
    if not resolved.is_relative_to(fixture_root) or not resolved.is_file():
        raise ValueError(f"fixture path escapes the fixture root: {fixture_path}")
    size = fixture_path.stat().st_size
    if size > MAX_FIXTURE_BYTES:
        raise ValueError(
            f"fixture {fixture_path.name} is {size} bytes; maximum is {MAX_FIXTURE_BYTES}"
        )
    try:
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"fixture {fixture_path.name} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(fixtures, list):
        raise TypeError(f"fixture {fixture_path.name} must contain a JSON array")
    if not fixtures:
        raise ValueError(
            f"fixture {fixture_path.name} must contain a non-empty JSON array"
        )
    if len(fixtures) > MAX_FIXTURES_PER_FILE:
        raise ValueError(f"fixture {fixture_path.name} exceeds the fixture-count limit")
    ids = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise TypeError(f"fixture entries in {fixture_path.name} must be objects")
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ValueError(f"fixture entry in {fixture_path.name} has an invalid id")
        ids.append(fixture_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"fixture {fixture_path.name} contains duplicate ids")
    return ids


def parse_capture_output(
    stdout: str, stderr: str, expected_ids: list[str], profile: dict
) -> list[dict]:
    output_bytes = len(stdout.encode()) + len(stderr.encode())
    if output_bytes > MAX_RUNNER_OUTPUT_BYTES:
        raise ValueError(
            f"oracle output is {output_bytes} bytes; maximum is {MAX_RUNNER_OUTPUT_BYTES}"
        )
    if stderr.strip():
        raise ValueError(f"oracle emitted unexpected stderr: {stderr[-400:]}")
    envelopes = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"oracle output line {line_number} is not valid JSON: {exc}"
            ) from exc
        validate_envelope(envelope, profile)
        envelopes.append(envelope)
    actual_ids = [envelope["fixture_id"] for envelope in envelopes]
    if actual_ids != expected_ids:
        raise ValueError(
            f"oracle fixture sequence mismatch: expected={expected_ids!r} actual={actual_ids!r}"
        )
    return envelopes


def capture_file(profile: dict, fixture_path: Path, py: str) -> list[dict]:
    """Runs one fixture file in a fresh isolated oracle subprocess."""
    expected_ids = load_fixture_ids(fixture_path)
    proc = subprocess.run(
        [py, "-P", "-s", str(runner_path()), str(fixture_path), profile["profile_id"]],
        capture_output=True,
        text=True,
        env=oracle_environment(profile),
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner exited {proc.returncode} for {fixture_path.name}: {proc.stderr[-400:]}"
        )
    return parse_capture_output(proc.stdout, proc.stderr, expected_ids, profile)


def golden_name_for(fixture_rel: str) -> str:
    return fixture_rel.removeprefix("fixtures/").removesuffix(".json") + ".ndjson"


def write_goldens(profile: dict, captured: dict[str, list[dict]]) -> Path:
    golden_dir = ARTIFACT_ROOT / profile["profile_id"] / "goldens"
    golden_dir.mkdir(parents=True, exist_ok=True)
    targets = [golden_dir / name for name in captured]
    existing = [target.name for target in targets if target.exists()]
    if existing:
        raise FileExistsError(
            "immutable goldens already exist; use self-test or create a new profile: "
            + ", ".join(sorted(existing))
        )
    for name, envelopes in captured.items():
        target = golden_dir / name
        with open(target, "x", encoding="utf-8") as fh:
            fh.writelines(
                json.dumps(envelope, sort_keys=True) + "\n" for envelope in envelopes
            )
    return golden_dir


def compare(left: list[dict], right: list[dict]) -> list[str]:
    """Exact-surface comparison over normalized envelope lists."""
    if len(left) != len(right):
        return ["envelope count differs"]
    return [
        lo.get("fixture_id", "?")
        for lo, ro in zip(left, right)
        if diff_envelopes(lo, ro, "exact_surface")
    ]


def compare_construction_only(left: list[dict], right: list[dict]) -> list[str]:
    """Construction-contract comparison: identity/structure fields only."""
    if len(left) != len(right):
        return ["envelope count differs"]
    return [
        lo.get("fixture_id", "?")
        for lo, ro in zip(left, right)
        if diff_envelopes(lo, ro, "construction_only")
    ]


def cmd_capture(profile: dict, py: str) -> int:
    base = Path(__file__).resolve().parent
    golden_dir = ARTIFACT_ROOT / profile["profile_id"] / "goldens"
    existing = [
        golden_name_for(rel)
        for rel in profile["inventory"]["fixtures"]
        if (golden_dir / golden_name_for(rel)).exists()
    ]
    if existing:
        raise FileExistsError(
            "immutable goldens already exist; use self-test or create a new profile: "
            + ", ".join(sorted(existing))
        )
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
        "declared_upstream_commit": profile["upstream"]["commit"],
        "oracle_environment": next(iter(captured.values()))[0]["environment"],
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
    with open(manifest_path, "x", encoding="utf-8") as fh:
        json.dump(run_manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


def load_goldens(profile: dict) -> dict[str, list[dict]]:
    base = Path(__file__).resolve().parent
    golden_dir = ARTIFACT_ROOT / profile["profile_id"] / "goldens"
    goldens: dict[str, list[dict]] = {}
    for rel in profile["inventory"]["fixtures"]:
        name = golden_name_for(rel)
        path = golden_dir / name
        if not path.exists():
            raise SystemExit(f"missing golden {name}; run `capture.py capture` first")
        stdout = path.read_text(encoding="utf-8")
        expected_ids = load_fixture_ids(base / rel)
        goldens[name] = parse_capture_output(stdout, "", expected_ids, profile)
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

    hash_swapped = copy.deepcopy(envelopes)
    for envelope in hash_swapped:
        observations = envelope.get("observations", {})
        if "hash_sha256_of_py_hash" in observations:
            old = observations["hash_sha256_of_py_hash"]
            observations["hash_sha256_of_py_hash"] = "f" * 64
            if observations["hash_sha256_of_py_hash"] == old:
                observations["hash_sha256_of_py_hash"] = "e" * 64
            break
    variants["hash-swapped"] = hash_swapped

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

    # Isolation/determinism: a second independent oracle subprocess must
    # produce byte-identical envelopes (no shared interpreter state).
    fresh_again = capture_file(profile, base / first_rel, py)
    if json.dumps(fresh, sort_keys=True) != json.dumps(fresh_again, sort_keys=True):
        return fail("two isolated oracle subprocesses disagreed: nondeterminism")

    # Environment pinning is enforced, not decorative: an oracle subprocess
    # launched with an unpinned hash seed must refuse to observe (exit 2).
    unpinned = subprocess.run(
        [
            py,
            "-P",
            "-s",
            str(runner_path()),
            str(base / first_rel),
            profile["profile_id"],
        ],
        capture_output=True,
        text=True,
        env=oracle_environment(profile, pinned_hash_seed=False),
        timeout=60,
        check=False,
    )
    if (
        unpinned.returncode != 2
        or '"error_class": "harness_misuse"' not in unpinned.stdout
    ):
        return fail(
            "oracle runner accepted an unpinned PYTHONHASHSEED; "
            "hash observations would be irreproducible"
        )

    # The profile checks themselves are a trust boundary. Each planted
    # environment mismatch must be rejected before it can become a golden.
    profile_mutants = {
        "wrong SymPy": ("sympy_version", "0.0.0"),
        "wrong Python": ("python", "0.0.0"),
        "wrong implementation": ("implementation", "PyPy"),
    }
    for label, (field, replacement) in profile_mutants.items():
        mutant = copy.deepcopy(fresh[0])
        mutant["environment"][field] = replacement
        try:
            validate_envelope(mutant, profile)
        except ValueError:
            pass
        else:
            return fail(f"profile validator accepted {label} mutant")
    wrong_side = copy.deepcopy(fresh[0])
    wrong_side["side"] = "candidate"
    try:
        validate_envelope(wrong_side, profile)
    except ValueError:
        pass
    else:
        return fail("profile validator accepted a candidate as the upstream oracle")
    try:
        parse_capture_output("", "", [fresh[0]["fixture_id"]], profile)
    except ValueError:
        pass
    else:
        return fail("capture accepted empty oracle output")
    try:
        write_goldens(profile, {golden_name_for(first_rel): fresh})
    except FileExistsError:
        pass
    else:
        return fail("capture allowed an immutable golden to be overwritten")
    # Comparator registry behavior: construction_only must ACCEPT pure
    # surface drift that exact_surface REJECTS (printers changed), and both
    # must reject identity drift.
    drifted = copy.deepcopy(fresh)
    for envelope in drifted:
        printers = envelope.get("observations", {}).get("printers")
        if printers:
            printers["latex"] += "\\;"
            break
    if not compare(golden_first, drifted):
        return fail("exact_surface accepted printer drift")
    if compare_construction_only(golden_first, drifted):
        return fail("construction_only rejected pure printer drift")
    identity_drift = copy.deepcopy(drifted)
    identity_drift[0]["observations"]["type"] = "WrongType"
    if not compare_construction_only(golden_first, identity_drift):
        return fail("construction_only accepted type identity drift")

    # Registry ids in code must stay aligned with the profile manifest.
    with open(
        Path(__file__).parent / "profiles" / f"{profile['profile_id']}.toml", "rb"
    ) as fh:
        import tomllib as _tomllib

        declared = set(_tomllib.load(fh)["comparators"].keys())
    if declared != set(REGISTRY):
        return fail(
            f"comparator registry drift: manifest={sorted(declared)} "
            f"code={sorted(REGISTRY)}"
        )

    # Gate: every weakened variant must be REJECTED by the exact comparator.
    rejected = 0
    checked_files = 0
    for name, golden in goldens.items():
        variants = weakened_variants(golden)
        if len(variants["hash-swapped"]) != len(golden):
            return fail(
                f"hash mutant changed envelope count instead of hash content ({name})"
            )
        for label, mutated in variants.items():
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
                "determinism_two_subprocesses": True,
                "profile_environment_mutants_rejected": len(profile_mutants) + 1,
                "empty_oracle_output_rejected": True,
                "immutable_golden_overwrite_rejected": True,
                "hash_mutant_preserves_envelope_count": True,
                "registry_matches_profile_manifest": True,
                "construction_only_semantics": "accepts printer drift, rejects identity drift",
                "mutants_checked": checked_files,
                "mutants_rejected": rejected,
            },
            indent=2,
        )
    )
    return 0


INVENTORY_REQUIRED_KEYS = {
    "schema_version",
    "kind",
    "profile_id",
    "environment",
    "modules",
    "identity",
    "upstream_test_tree",
    "content_sha256",
}


def validate_inventory(inventory: object, profile: dict) -> None:
    if not isinstance(inventory, dict):
        raise TypeError("inventory must be an object")
    if set(inventory) != INVENTORY_REQUIRED_KEYS:
        raise ValueError("inventory keys do not match schema version 1")
    if inventory["schema_version"] != 1 or inventory["kind"] != "reflection_inventory":
        raise ValueError("inventory schema or kind mismatch")
    if inventory["profile_id"] != profile["profile_id"]:
        raise ValueError("inventory profile mismatch")
    validate_environment(inventory["environment"], profile)
    claimed = inventory["content_sha256"]
    unsigned = dict(inventory)
    unsigned.pop("content_sha256")
    canonical = json.dumps(unsigned, sort_keys=True).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if not isinstance(claimed, str) or not hmac.compare_digest(claimed, actual):
        raise ValueError(
            f"inventory content digest mismatch: claimed={claimed} actual={actual}"
        )


def capture_inventory_once(profile: dict, py: str, runner: Path) -> tuple[dict, str]:
    proc = subprocess.run(
        [py, "-P", "-s", str(runner), profile["profile_id"]],
        capture_output=True,
        text=True,
        env=oracle_environment(profile),
        timeout=180,
        check=False,
    )
    output_bytes = len(proc.stdout.encode()) + len(proc.stderr.encode())
    if output_bytes > MAX_RUNNER_OUTPUT_BYTES:
        raise ValueError("inventory runner output exceeds the harness limit")
    if proc.returncode != 0:
        raise RuntimeError(
            f"inventory runner exited {proc.returncode}: {proc.stderr[-300:]}"
        )
    if proc.stderr.strip():
        raise RuntimeError(
            f"inventory runner emitted unexpected stderr: {proc.stderr[-300:]}"
        )
    try:
        inventory = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"inventory runner output is not valid JSON: {exc}") from exc
    validate_inventory(inventory, profile)
    return inventory, proc.stdout


def cmd_inventory(profile: dict, py: str) -> int:
    """Generates the C1 reflection/source inventory as a digest-pinned artifact."""
    runner = Path(__file__).resolve().parent / "inventory_runner.py"
    inventory, stdout = capture_inventory_once(profile, py, runner)
    second, _ = capture_inventory_once(profile, py, runner)
    if second != inventory:
        return fail("reflection inventory is nondeterministic across subprocesses")

    # Publish only after both isolated runs validate and agree. Existing
    # profile artifacts are immutable: an identical artifact is retained;
    # different bytes fail closed instead of being overwritten.
    out_dir = ARTIFACT_ROOT / profile["profile_id"] / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "inventory.json"
    payload = stdout if stdout.endswith("\n") else stdout + "\n"
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != payload:
            return fail("immutable reflection inventory differs; create a new profile")
    else:
        with open(target, "x", encoding="utf-8") as fh:
            fh.write(payload)

    print(
        json.dumps(
            {
                "inventory": str(target.relative_to(REPO_ROOT)),
                "content_sha256": inventory["content_sha256"],
                "modules_inventoried": len(inventory["modules"]),
                "deterministic": True,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if len(args) not in {2, 4}:
        print(__doc__)
        return 2
    mode = args[0]
    if mode not in {"capture", "self-test", "inventory"}:
        print(f"unknown mode: {mode}", file=sys.stderr)
        return 2
    py = None
    if len(args) == 4:
        if args[2] != "--oracle-python" or not args[3]:
            print(__doc__)
            return 2
        py = args[3]
    try:
        profile = load_profile(Path(args[1]))
        interpreter = oracle_python(py)
        if mode == "capture":
            return cmd_capture(profile, interpreter)
        if mode == "self-test":
            return cmd_self_test(profile, interpreter)
        return cmd_inventory(profile, interpreter)
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as exc:
        return fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
