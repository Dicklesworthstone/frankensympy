"""Campaign property 12 must execute both engines for this revision."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import harness

HEAD = "0123456789abcdef0123456789abcdef01234567"


class PairedCampaignTests(unittest.TestCase):
    def test_an_existing_report_cannot_substitute_for_no_producer_output(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            report = bundle / "paired_incumbent.json"
            report.write_text('{"old":true}')
            with (patch.object(harness, "BUNDLE_DIR", bundle),
                  patch.object(harness, "run_cmd") as run):
                with self.assertRaises(FileNotFoundError):
                    harness.run_property_12_paired_benchmark(HEAD)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(json.loads(report.read_text()), {"old": True})

    def test_fresh_report_is_bound_checked_and_failures_remain_visible(self):
        for status in ("admitted", "divergence", "error"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                calls = []

                def run(args, **kwargs):
                    calls.append(args)
                    if "--out" in args:
                        path = Path(args[args.index("--out") + 1])
                        self.assertFalse(path.exists())
                        path.write_text(json.dumps({
                            "subject_head": HEAD, "cases": {"workload": {"status": status}},
                        }))
                    else:
                        self.assertEqual(args[args.index("--expected-subject-head") + 1], HEAD)
                        self.assertEqual(args[args.index("--expected-profile") + 1], "sympy-1.14.0-cpython")
                        self.assertIn("--require-settings", args)

                with (patch.object(harness, "ROOT", root), patch.object(harness, "BUNDLE_DIR", root),
                      patch.object(harness, "run_cmd", side_effect=run)):
                    result = harness.run_property_12_paired_benchmark(HEAD)
                self.assertEqual(len(calls), 2)
                self.assertEqual(result["status"], "passed" if status == "admitted" else "blocked")
                self.assertEqual(result["report_git_commit"], HEAD)
                self.assertEqual(result["paired_records"], [{"case_id": "workload", "status": status}])
                self.assertTrue((root / result["report_path"]).is_file())
