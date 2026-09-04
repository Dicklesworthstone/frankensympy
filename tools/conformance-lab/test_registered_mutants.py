"""Registered harness-mutant kill tests (fra-conformance-corpus-200-b75).

Each registered mutant in mutants.json is a weakening a cheating agent could
apply to a candidate observation envelope. For every mutant we take a REAL
golden envelope pair from the active profile, apply the weakening to the
candidate copy, and assert the exact_surface comparator FLAGS it (the mutant
is killed): Art. VIII.5 applied to the harness lane.
"""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parent
ARTIFACT = LAB.parent.parent / "artifacts" / "conformance" / "sympy-1.14.0-cpython-r2-corpus"
PROFILE_FIXTURES = ["adversarial_corpus_r2", "generated_corpus_r2",
                    "seed_core_atoms", "seed_held_forms", "seed_function_subclass"]


def _load_goldens() -> dict[str, list[dict]]:
    golden_dir = ARTIFACT / "goldens"
    envs: dict[str, list[dict]] = {}
    for stem in PROFILE_FIXTURES:
        path = golden_dir / f"{stem}.ndjson"
        envs[stem] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return envs


def _first_env(test, kinds: tuple[str, ...] = (), top_level_key: str | None = None) -> dict:
    for stem in PROFILE_FIXTURES:
        for env in test.envs[stem]:
            if top_level_key and top_level_key in env:
                return env
            if kinds and any(k in env.get("observations", {}) for k in kinds):
                return env
    raise unittest.SkipTest(f"no golden envelope with {kinds or top_level_key!r}")


class RegisteredMutantKills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mutants = json.loads((LAB / "mutants.json").read_text())["mutants"]
        cls.envs = _load_goldens()
        from capture import compare  # lazy: capture.py is a CLI module
        cls.compare = staticmethod(compare)

    def _assert_killed(self, mutant_id: str, golden: dict, weaken) -> None:
        candidate = copy.deepcopy(golden)
        weaken(candidate)
        flagged = self.compare([golden], [candidate])
        self.assertTrue(
            flagged,
            f"mutant {mutant_id!r} NOT killed: comparator accepted a weakened candidate",
        )

    def test_registry_is_nonempty_and_ids_unique(self):
        ids = [m["id"] for m in self.mutants]
        self.assertGreaterEqual(len(ids), 6)
        self.assertEqual(len(ids), len(set(ids)))

    def test_kill_treat_expected_refusal_as_pass(self):
        env = _first_env(self, top_level_key="outcome_class")
        original = env["outcome_class"]

        def weaken(c):
            c["outcome_class"] = "returned" if original != "returned" else "raised"

        self._assert_killed("treat-expected-refusal-as-pass", env, weaken)

    def test_kill_drop_type_drift_class(self):
        env = _first_env(self, kinds=("type",))

        def weaken(c):
            c["observations"]["type"] = "DefinitelyNotTheOracleType"

        self._assert_killed("drop-type-drift-class", env, weaken)

    def test_kill_loosen_pickle_byte_compare(self):
        env = _first_env(self, kinds=("pickle_v4",))

        def weaken(c):
            obs = c["observations"]["pickle_v4"]
            obs["sha256"] = "0" * 64

        self._assert_killed("loosen-pickle-byte-compare", env, weaken)

    def test_kill_accept_oracle_version_mismatch(self):
        env = _first_env(self, top_level_key="environment")

        def weaken(c):
            c["environment"]["sympy_version"] = "1.13.9-wrong"

        self._assert_killed("accept-oracle-version-mismatch", env, weaken)

    def test_kill_drop_warning_class(self):
        # Warning observations ride the warnings-only lane; the envelope shape
        # here mirrors observe_warnings_real() output exactly.
        oracle = {
            "schema_version": 1,
            "profile_id": "sympy-1.14.0-cpython-r2-corpus",
            "fixture_id": "synthetic/warning_pair",
            "side": "upstream_oracle",
            "outcome_class": "returned",
            "observations": {
                "kind": "warning_observation",
                "warnings": [
                    {"module": "sympy.core.numbers", "name": "SymPyDeprecationWarning"},
                    {"module": "builtins", "name": "UserWarning"},
                ],
            },
        }

        def weaken(c):
            c["observations"]["warnings"].pop()

        self._assert_killed("drop-warning-class", oracle, weaken)

    def test_kill_swap_hash_value(self):
        env = _first_env(self, kinds=("hash_sha256_of_py_hash",))

        def weaken(c):
            c["observations"]["hash_sha256_of_py_hash"] = "f" * 64

        self._assert_killed("swap-hash-value", env, weaken)

    def test_kill_shrink_args_tuple(self):
        env = None
        for stem in PROFILE_FIXTURES:
            for cand in self.envs[stem]:
                args = cand.get("observations", {}).get("args_repr")
                if isinstance(args, list) and len(args) > 1:
                    env = cand
                    break
            if env:
                break
        if env is None:
            self.skipTest("no multi-arg observation in corpus goldens")

        def weaken(c):
            args = c["observations"]["args_repr"]
            if isinstance(args, list) and len(args) > 1:
                args.pop()
            else:
                raise AssertionError("no multi-arg observation found to shrink")

        self._assert_killed("shrink-args-tuple", env, weaken)


if __name__ == "__main__":
    unittest.main()
