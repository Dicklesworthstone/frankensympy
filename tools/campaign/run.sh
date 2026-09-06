#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="${1:-all}"

case "$MODE" in
  all|run|hero-v1|jacobian-v1)
    echo "=== Executing Certified Jacobian Pipeline Campaign ($MODE) ==="
    "$PYTHON_BIN" "$SCRIPT_DIR/harness.py"
    ;;
  *)
    echo "usage: $0 [all|run|hero-v1|jacobian-v1]" >&2
    exit 2
    ;;
esac
