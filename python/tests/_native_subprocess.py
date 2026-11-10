"""Isolate tests which irreversibly alter CPython heap-type allocation slots."""

import os
from pathlib import Path
import subprocess
import sys

import sympy
from sympy.core import _native


def isolated_heap_new_probe(request, hook):
    """Run a __new__ adversary in a child using the same extension and shell.

    Restoring an inherited object.__new__ after replacing it does not restore
    CPython's original tp_new slot. Keep that interpreter-level side effect
    out of later test cases while retaining every adversarial assertion.
    """
    marker = "FRANKENSYMPY_HEAP_NEW_PROBE"
    if hook != "__new__" or os.environ.get(marker) == request.node.name:
        return False
    bootstrap = """
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2])
native = importlib.util.module_from_spec(spec)
sys.modules[sys.argv[1]] = native
spec.loader.exec_module(native)
assert Path(native.__file__).resolve() == Path(sys.argv[2]).resolve()
sys.path.insert(0, sys.argv[3])
import sympy
from sympy.core import _native
assert Path(sympy.__file__).resolve() == Path(sys.argv[3]) / "sympy" / "__init__.py"
assert _native is native
import pytest
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", sys.argv[4]]))
"""
    node = str(request.node.path) + "::" + request.node.name
    process = subprocess.run(
        [sys.executable, "-c", bootstrap, _native.__name__, _native.__file__,
         str(Path(sympy.__file__).resolve().parent.parent), node],
        env={**os.environ, marker: request.node.name},
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return True
