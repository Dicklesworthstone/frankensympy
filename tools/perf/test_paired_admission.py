"""Exercise the ordering and failure semantics of a live paired run."""

import json
import io
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import paired_bench


class PairedAdmissionTests(unittest.TestCase):
    def run_pair(self, outputs):
        calls = []

        def observe(engine, case, *, timed):
            calls.append((engine, timed))
            value = next(outputs)
            return {"case": case, "result": value if isinstance(value, str) else None,
                    "error": str(value) if isinstance(value, Exception) else None,
                    "ms": (4 if engine == "subject" else 2) if timed else None}

        with patch.object(paired_bench, "run_case_isolated", side_effect=observe):
            entry = paired_bench.measure_pair("workload", 2)
        return entry, calls

    def test_both_engines_agree_before_either_is_timed_and_order_alternates(self):
        entry, calls = self.run_pair(iter(["A"] * 6))
        self.assertEqual(calls, [("subject", False), ("oracle", False),
                                 ("subject", True), ("oracle", True),
                                 ("oracle", True), ("subject", True)])
        self.assertEqual(entry["status"], "admitted")
        self.assertEqual(entry["ratio_oracle_over_subject"], 0.5)
        self.assertEqual(entry["subject_samples"], [4, 4])

    def test_failed_and_divergent_admissions_never_launch_timed_work(self):
        for outputs, status in [(["A", "B"], "divergence"),
                                ([ValueError("refused"), "B"], "error")]:
            with self.subTest(status=status):
                entry, calls = self.run_pair(iter(outputs))
                self.assertEqual(calls, [("subject", False), ("oracle", False)])
                self.assertEqual(entry["status"], status)
                self.assertNotIn("ratio_oracle_over_subject", entry)

    def test_late_mismatch_or_failure_invalidates_even_previously_paired_samples(self):
        for outputs, status in [(["A", "B"], "divergence"),
                                (["B", "B"], "error"),
                                (["A", ValueError("refused")], "error")]:
            with self.subTest(outputs=outputs):
                entry, _ = self.run_pair(iter(["A"] * 4 + outputs))
                self.assertEqual(entry["status"], status)
                self.assertNotIn("subject_samples", entry)

    def test_untimed_worker_emits_result_without_a_time(self):
        output = io.StringIO()
        with (patch.object(paired_bench, "CASES", [("integer", "def run(): return 7")]),
              patch.object(paired_bench.time, "perf_counter_ns", side_effect=AssertionError),
              redirect_stdout(output)):
            paired_bench.case_worker("integer", timed=False)
        self.assertEqual(json.loads(output.getvalue()),
                         {"case": "integer", "result": "int:7", "error": None, "ms": None})


if __name__ == "__main__":
    unittest.main()
