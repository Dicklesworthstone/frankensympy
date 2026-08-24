#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${1:-all}"

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi

say() { printf '%s\n' "${BOLD}$*${RESET}"; }
ok() { printf '%s\n' "${GREEN}ok:${RESET} $*"; }
refuse() { printf '%s\n' "${YELLOW}refused:${RESET} $*" >&2; exit 2; }
fail() { printf '%s\n' "${RED}error:${RESET} $*" >&2; exit 1; }
run() { say "+ $*"; "$@"; }

require_python() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN is required"
  "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required for tomllib")
PY
}

tooling_self_check() {
  require_python
  run "$PYTHON_BIN" -m py_compile "$ROOT"/tools/*.py
  run "$PYTHON_BIN" "$ROOT/tools/validate_planning.py" --self-test
  run "$PYTHON_BIN" "$ROOT/tools/validate_planning.py"
  run bash -n "$ROOT/scripts/check.sh"
  ok "validator and shell syntax"
}

source_clean() {
  command -v git >/dev/null 2>&1 || fail "git is required"
  local status
  status="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
  [[ -z "$status" ]] || { printf '%s\n' "$status" >&2; fail "source tree is not clean"; }
  ok "source tree clean"
}

registries() {
  tooling_self_check
  run "$PYTHON_BIN" "$ROOT/tools/verify_cross_cutting.py" --root "$ROOT" --no-write
  run "$PYTHON_BIN" "$ROOT/tools/verify_registry_bundle.py" --root "$ROOT" --no-write
  run "$PYTHON_BIN" "$ROOT/tools/audit_donor_sources.py" --root "$ROOT" --no-write
  run "$PYTHON_BIN" "$ROOT/tools/verify_dependency_and_safety.py" --root "$ROOT" --no-write
  run "$PYTHON_BIN" "$ROOT/tools/verify_performance_kernels.py" --root "$ROOT" --no-write
  ok "all executable planning registries passed"
}

architecture() {
  require_python
  run "$PYTHON_BIN" "$ROOT/tools/verify_cross_cutting.py" --root "$ROOT" --no-write
  run "$PYTHON_BIN" "$ROOT/tools/verify_registry_bundle.py" --root "$ROOT" --no-write
  ok "architecture registries coherent"
}

metadata() {
  require_python
  run "$PYTHON_BIN" "$ROOT/tools/audit_donor_sources.py" --root "$ROOT" --no-write
  run "$PYTHON_BIN" "$ROOT/tools/verify_dependency_and_safety.py" --root "$ROOT" --no-write
  run "$PYTHON_BIN" "$ROOT/tools/verify_performance_kernels.py" --root "$ROOT" --no-write
  ok "metadata, source pins, safety policy, and kernel registry passed"
}

format() {
  run cargo fmt --check
  run cargo clippy --workspace --all-targets -- -D warnings
  ok "formatting and clippy passed across workspace"
}

release_readiness() {
  require_python
  "$PYTHON_BIN" - "$ROOT" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
import tomllib

root = Path(sys.argv[1])
blockers = []


def load_toml(relative_path):
    with (root / relative_path).open("rb") as handle:
        return tomllib.load(handle)


quality = load_toml("quality_gates.toml")
if quality.get("registry_status") != "implemented" or not quality.get("enforced", False):
    blockers.append("quality_gates.toml is not enforced")
for section in ("coverage", "flake", "runtime"):
    if quality.get(section, {}).get("measurement_status") != "implemented":
        blockers.append(f"quality gate measurement '{section}' is not implemented")
activation = quality.get("activation_requirements")
if not isinstance(activation, dict):
    blockers.append("quality gate activation requirements are missing or malformed")
else:
    for field in (
        "monitor_implementation_exists",
        "measurement_schema_frozen",
        "representative_test_inventory_exists",
        "claims_registry_update_required",
    ):
        if activation.get(field) is not True:
            blockers.append(f"quality gate activation '{field}' is not satisfied")

compatibility = load_toml("registries/compatibility_profiles.toml")
targets = [
    profile
    for profile in compatibility.get("profiles", [])
    if profile.get("kind") == "certification_target"
]
certified_targets = [profile for profile in targets if profile.get("status") == "certified"]
known_compatibility_statuses = {"planned", "certified"}
for profile in targets:
    if profile.get("status") not in known_compatibility_statuses:
        blockers.append(
            f"profile '{profile.get('profile_id', '<unknown>')}' has unknown status "
            f"'{profile.get('status')}'"
        )
if not certified_targets:
    blockers.append("no immutable compatibility target is certified")
else:
    required_profile_fields = (
        "source_tree_digest",
        "test_tree_digest",
        "public_surface_manifest",
        "class_manifest",
        "signature_manifest",
        "behavior_manifest",
        "assumptions_manifest",
        "printer_manifest",
        "serialization_manifest",
        "test_inventory_manifest",
        "comparator_registry",
        "discrepancy_ledger_digest",
        "rule_registry_digest",
        "algorithm_registry_digest",
        "verifier_registry_digest",
        "gate_bundle",
        "certified_at_commit",
    )
    for profile in certified_targets:
        profile_id = profile.get("profile_id", "<unknown>")
        for field in required_profile_fields:
            if profile.get(field) in {None, "", "UNSET"}:
                blockers.append(f"profile '{profile_id}' leaves '{field}' unset")
        for field in ("python_versions", "abi_tags", "platform_tags"):
            if not profile.get(field):
                blockers.append(f"profile '{profile_id}' has no '{field}' matrix")
        for field in ("lowering_schema_version", "result_schema_version"):
            if not isinstance(profile.get(field), int) or profile[field] <= 0:
                blockers.append(f"profile '{profile_id}' has no active '{field}'")
        # A tracked registry cannot contain the SHA of its own enclosing commit.
        # Same-commit source/tree and artifact digests must instead be established
        # by an external canonical gate receipt. That validator has not landed.
        blockers.append(
            f"profile '{profile_id}' cannot certify until external same-commit gate receipt validation is implemented"
        )

claims = load_toml("registries/claims.toml")
release_claims = [claim for claim in claims.get("claims", []) if claim.get("kind") == "release"]
if not release_claims or any(claim.get("status") != "certified" for claim in release_claims):
    blockers.append("release claim is not certified")
for claim in release_claims:
    artifacts = claim.get("evidence_artifacts", [])
    if not artifacts or any(artifact in {"", "UNSET"} for artifact in artifacts):
        blockers.append(f"release claim '{claim.get('id', '<unknown>')}' has no evidence bundle")

packaging = load_toml("registries/packaging_profiles.toml")
resolver_profiles = [
    profile
    for profile in packaging.get("profile", [])
    if profile.get("satisfies_requires_dist_sympy") is True
]
for profile in packaging.get("profile", []):
    if profile.get("status") not in {"planned", "certified"}:
        blockers.append(
            f"packaging profile '{profile.get('id', '<unknown>')}' has unknown status "
            f"'{profile.get('status')}'"
        )
if not any(profile.get("status") == "certified" for profile in resolver_profiles):
    blockers.append("no resolver-transparent replacement packaging profile is certified")

release_gates = load_toml("registries/release_gates.toml")
if release_gates.get("registry_status") != "implemented":
    blockers.append("release gate registry is not implemented")

cross_cutting = subprocess.run(
    [
        sys.executable,
        str(root / "tools/verify_cross_cutting.py"),
        "--root",
        str(root),
        "--no-write",
    ],
    check=False,
    capture_output=True,
    text=True,
)
if cross_cutting.returncode != 0:
    blockers.append("cross-cutting release audit failed to execute cleanly")
else:
    try:
        cross_cutting_report = json.loads(cross_cutting.stdout)
    except json.JSONDecodeError:
        blockers.append("cross-cutting release audit did not emit valid JSON")
    else:
        incomplete = cross_cutting_report.get("incomplete_release_blockers")
        if (
            cross_cutting_report.get("schema_version") != 1
            or not isinstance(incomplete, list)
            or not all(isinstance(obligation, str) for obligation in incomplete)
        ):
            blockers.append("cross-cutting release audit emitted an unsupported schema")
        else:
            for obligation in incomplete:
                blockers.append(
                    f"cross-cutting obligation '{obligation}' remains a release blocker"
                )

if blockers:
    print("release readiness blockers:", file=sys.stderr)
    for blocker in sorted(set(blockers)):
        print(f"  - {blocker}", file=sys.stderr)
    raise SystemExit(1)
PY
}

write_audits() {
  require_python
  mkdir -p "$ROOT/artifacts/audit"
  run "$PYTHON_BIN" "$ROOT/tools/verify_cross_cutting.py" --root "$ROOT" --json-output artifacts/audit/cross_cutting_obligations.json
  run "$PYTHON_BIN" "$ROOT/tools/verify_registry_bundle.py" --root "$ROOT" --json-output artifacts/audit/registry_bundle.json
  run "$PYTHON_BIN" "$ROOT/tools/audit_donor_sources.py" --root "$ROOT" --json-output artifacts/audit/donor_sources.json
  run "$PYTHON_BIN" "$ROOT/tools/verify_dependency_and_safety.py" --root "$ROOT" --json-output artifacts/audit/dependency_safety.json
  run "$PYTHON_BIN" "$ROOT/tools/verify_performance_kernels.py" --root "$ROOT" --json-output artifacts/audit/performance_kernels.json
  ok "audit artifacts written"
}

unit() {
  run cargo test --workspace
  ok "unit tests passed across workspace"
}

conformance() {
  run cargo test -p fsym-conformance
  ok "conformance tests passed"
}

portable_verifiers() {
  run cargo test -p fsym-proof-kernel
  ok "portable proof-kernel verifiers passed"
}

bench_smoke() {
  run cargo test -p fsym-runtime --lib -- test_standard_ws22_suite_execution
  ok "paired benchmark suite passed"
}

unimplemented_release_profile() {
  refuse "profile '$1' is defined by the release contract but has no landed implementation evidence yet"
}

lab() { unimplemented_release_profile lab; }
fuzz_smoke() { unimplemented_release_profile fuzz-smoke; }
matrix() { unimplemented_release_profile matrix; }
reproducibility() { unimplemented_release_profile reproducibility; }
package() { unimplemented_release_profile package; }
sign() { unimplemented_release_profile sign; }

release_candidate() {
  source_clean
  release_readiness || refuse "release readiness preconditions are not satisfied"
  registries
  format
  unit
  conformance
  portable_verifiers
  lab
  fuzz_smoke
  bench_smoke
  matrix
  reproducibility
  package
  write_audits
  # Audit generation is admissible only when it reproduces the tracked bytes exactly. A dirty
  # diff here means the candidate no longer matches the evidence bundle checked above.
  source_clean
  ok "release candidate locally validated against every required implemented gate"
}

case "$PROFILE" in
  source-clean) source_clean ;;
  format) format ;;
  architecture) architecture ;;
  registries) registries ;;
  metadata) metadata ;;
  audit) write_audits ;;
  unit) unit ;;
  conformance) conformance ;;
  portable-verifiers) portable_verifiers ;;
  bench-smoke) bench_smoke ;;
  lab) lab ;;
  fuzz-smoke) fuzz_smoke ;;
  matrix) matrix ;;
  reproducibility) reproducibility ;;
  package) package ;;
  sign) sign ;;
  release-candidate) release_candidate ;;
  all) release_candidate ;;
  *)
    printf 'usage: %s {source-clean|format|architecture|registries|metadata|audit|unit|conformance|portable-verifiers|bench-smoke|lab|fuzz-smoke|matrix|reproducibility|package|sign|release-candidate|all}\n' "$0" >&2
    exit 2
    ;;
esac
