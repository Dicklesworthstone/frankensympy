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

release_candidate() {
  registries
  unit
  conformance
  portable_verifiers
  bench_smoke
  write_audits
  ok "release candidate certified (G1-G8 release bundle passed)"
}

case "$PROFILE" in
  source-clean) source_clean ;;
  architecture) architecture ;;
  registries) registries ;;
  metadata) metadata ;;
  audit) write_audits ;;
  unit) unit ;;
  conformance) conformance ;;
  portable-verifiers) portable_verifiers ;;
  bench-smoke) bench_smoke ;;
  release-candidate) release_candidate ;;
  all) release_candidate ;;
  lab|fuzz-smoke|matrix|reproducibility|package|sign)
    refuse "profile '$PROFILE' is defined by the release contract but has no landed implementation evidence yet"
    ;;
  *)
    printf 'usage: %s {source-clean|architecture|registries|metadata|audit|unit|conformance|portable-verifiers|bench-smoke|release-candidate|all}\n' "$0" >&2
    exit 2
    ;;
esac
