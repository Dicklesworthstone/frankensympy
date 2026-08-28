"""Real FrankenSymPy candidate envelopes when the native extension is present."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import capture_candidate_file, load_profile  # noqa: E402

LAB = Path(__file__).resolve().parent
PROFILE_PATH = LAB / "profiles" / "sympy-1.14.0-cpython.toml"
CORE = LAB / "fixtures" / "seed_core_atoms.json"


def _extension_available() -> bool:
    import os

    names = ("fsym_python.so", "libfsym_python.so")
    dirs = []
    explicit = os.environ.get("FSYM_PYTHON_EXT_DIR")
    if explicit:
        dirs.append(Path(explicit))
    cargo = os.environ.get("CARGO_TARGET_DIR")
    if cargo:
        dirs.append(Path(cargo) / "debug")
        dirs.append(Path(cargo) / "release")
    for directory in dirs:
        for name in names:
            if (directory / name).is_file():
                return True
    return False


class RealCandidateTests(unittest.TestCase):
    def test_integer_fixture_returns_native_integer_when_extension_present(self) -> None:
        if not _extension_available():
            self.skipTest("fsym_python cdylib not on CARGO_TARGET_DIR/FSYM_PYTHON_EXT_DIR")
        profile = load_profile(PROFILE_PATH)
        envelopes = capture_candidate_file(profile, CORE, sys.executable, broken=False)
        by_id = {envelope["fixture_id"]: envelope for envelope in envelopes}
        integer = by_id["core/integer/42"]
        self.assertEqual(integer["side"], "frankensympy_candidate")
        self.assertEqual(integer["outcome_class"], "returned")
        self.assertEqual(integer["observations"]["type"], "Integer")
        refused = [
            envelope["fixture_id"]
            for envelope in envelopes
            if envelope["outcome_class"] == "refused"
        ]
        self.assertEqual(refused, [])
        symbol = by_id["core/symbol/x_positive"]
        self.assertEqual(symbol["outcome_class"], "raised")
        self.assertEqual(symbol["observations"]["exception_type"], "NotImplementedError")


if __name__ == "__main__":
    unittest.main()
