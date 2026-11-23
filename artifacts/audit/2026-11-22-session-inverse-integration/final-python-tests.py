"""Run retained real PyO3 numerical paths with exact import identities."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

root = Path("/workspace/scratch/78afbd38037e/frankensympy")
component = Path(__file__).resolve().parent
binary = (root / "target/debug/libfsym_python.so").resolve()
expected = json.loads((component / "native-binary.json").read_text())
with binary.open("rb") as stream:
    binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
assert binary_hash == expected["sha256"]
assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
spec = importlib.util.spec_from_file_location("fsym_python", binary)
assert spec is not None and spec.loader is not None
native = importlib.util.module_from_spec(spec)
sys.modules["fsym_python"] = native
spec.loader.exec_module(native)
assert Path(native.__file__).resolve() == binary
sys.path.insert(0, str(root / "python"))
import sympy
from sympy.core import _native
assert Path(sympy.__file__).resolve() == root / "python/sympy/__init__.py"
assert _native is native
import pytest

targets = [
    "python/tests/test_inverse_derivative_numeric.py",
    "python/tests/test_nsolve_inverse_jets.py",
    "python/tests/test_scalar_numeric_derivatives.py",
    "python/tests/test_inverse_circular_numeric.py",
    "python/tests/test_inverse_hyperbolic_numeric.py",
    "python/tests/test_nsolve_certified.py",
    "python/tests/test_nsolve_bisect.py",
]
print(json.dumps({
    "scope": "Retained mathematical PyO3 component; six unrelated simplify/expand helpers and full async runtime omitted",
    "native": str(binary), "native_sha256": binary_hash,
    "sympy": sympy.__file__, "python": sys.version, "pytest": pytest.__version__,
    "targets": targets,
}), flush=True)
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", *targets]))
