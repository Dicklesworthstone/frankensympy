#!/usr/bin/env bash
# Build the fsym_python CPython extension and install it under the importable name.
#
# Declared local path until the maturin packaging profile is implemented (WS23):
#   1. cargo build -p fsym-python (cdylib libfsym_python.so) into a local target dir
#   2. copy it to python/fsym_python.so, the name CPython's finder and the
#      python/sympy/core/__init__.py fallback both resolve
#   3. smoke-load the module with the conformance interpreter and fail closed
#      if PyInit_fsym_python does not load
#
# Environment:
#   FSYM_PYTHON        interpreter used for the smoke load and PYO3 configuration
#                      (default: .venv-conformance/bin/python when present)
#   FSYM_EXT_TARGET_DIR cargo target dir (default: <repo>/target_fsym; kept out of
#                      rch transfer by .rchignore)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

py="${FSYM_PYTHON:-}"
if [[ -z "$py" ]]; then
  if [[ -x ".venv-conformance/bin/python" ]]; then
    py="$repo_root/.venv-conformance/bin/python"
  else
    py="python3"
  fi
fi

target_dir="${FSYM_EXT_TARGET_DIR:-$repo_root/target_fsym}"
# The rch cargo shim offloads build|test|check to remote workers, but remote
# build artifacts never land back in local target dirs (.rchignore excludes
# target*/), and the extension must exist on THIS host for CPython to load it.
# RCH_SHIM_LOCAL_IDE=1 is the shim's documented per-invocation local escape.
PYO3_PYTHON="$py" CARGO_TARGET_DIR="$target_dir" RCH_SHIM_LOCAL_IDE=1 \
  cargo build -p fsym-python --lib

echo "build_python_extension: interpreter=$py target=$target_dir"

so=""
for candidate in "$target_dir/debug/libfsym_python.so" "$target_dir/debug/fsym_python.so"; do
  if [[ -f "$candidate" ]]; then
    so="$candidate"
    break
  fi
done
if [[ -z "$so" ]]; then
  echo "refused: cdylib libfsym_python.so not found under $target_dir/debug" >&2
  exit 1
fi

# pyproject [tool.maturin] module-name is fsym_python; the installed artifact
# must carry exactly that module name (abi3 tagging is a packaging-profile decision).
install_path="$repo_root/python/fsym_python.so"
# python/fsym_python.so may exist as a stale dangling symlink from an old build
# layout; replacing it destroys no content (a symlink has no target payload).
rm -f "$install_path"
cp "$so" "$install_path"

"$py" - "$install_path" <<'PY'
import importlib.machinery
import importlib.util
import sys

path = sys.argv[1]
loader = importlib.machinery.ExtensionFileLoader("fsym_python", path)
spec = importlib.util.spec_from_loader("fsym_python", loader)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert hasattr(module, "Expr"), "fsym_python loaded but exposes no Expr"
print(f"build_python_extension: loaded {path}")
PY

echo "build_python_extension: installed $install_path"
