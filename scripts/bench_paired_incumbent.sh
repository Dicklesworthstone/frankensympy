#!/usr/bin/env bash
# WS22 paired live-incumbent baseline (bead fra-fra-ws22-paired-incumbent-adoption-lfx).
# Builds the release-perf cdylib, installs it, runs the paired harness, and
# files the result into .bench-history. Raw paired data is retained.
set -euo pipefail
cd "$(dirname "$0")/.."

ROUNDS="${1:-9}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/perf/paired_incumbent_${STAMP}.json"

echo "[bench] building release-perf cdylib (local: remote workers OOM on thin-LTO cdylib link; observed 3x SIGKILL on rch workers)..."
RCH_SHIM_LOCAL_IDE=1 cargo build --profile release-perf -p fsym-python
# CARGO_TARGET_DIR (set by rch tooling) moves the artifact out of target/.
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
install -m 0755 "$TARGET_DIR/release-perf/libfsym_python.so" "python/fsym_python.so"

echo "[bench] running paired sweep (rounds=${ROUNDS})..."
python3 tools/perf/paired_bench.py run --out "$OUT" --rounds "$ROUNDS"

mkdir -p .bench-history
cp "$OUT" ".bench-history/paired_incumbent.latest.json"

echo "[bench] baseline: $OUT"
echo "[bench] summary:"
python3 - "$OUT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print(json.dumps(report["summary"], indent=2))
PY
