"""Mutation checks for the numerical campaign evidence schema."""

import base64
from fractions import Fraction
import hashlib
import math
import unittest

if __package__:
    from .numeric_case import CASE_ID, validate_numeric_case
else:
    from numeric_case import CASE_ID, validate_numeric_case


def _example_record():
    # A synthetic manifest for testing only the offline validator. These roots
    # satisfy the analytic diagnostic, and every mutation below changes a field
    # independently of the generator and native proof checker.
    blob = b"synthetic manifest test"
    x, y = 0.2444294514144251, 1.8700032849244765
    return {
        "case_id": CASE_ID, "status": "completed", "num_residuals": 2, "num_vars": 2,
        "program_base64": base64.b64encode(blob).decode(),
        "program_sha256": hashlib.sha256(blob).hexdigest(),
        "evaluation_point": [1, 1],
        "compiled_residuals": [math.sqrt(math.e) - 2, math.sin(1) - 0.2],
        "compiled_jacobian": [[math.sqrt(math.e) / 2, 1.0], [1.0, math.cos(1)]],
        "box_enclosure": {
            "precision_digits": 30, "domain": [["1/8", "2"], ["0", "2"]],
            "residual_bounds": [["-2", "2"], ["-2", "2"]],
            "jacobian_bounds": [[["1/2", "3/2"], ["1", "1"]],
                                [["1", "1"], ["-1", "1"]]],
        },
        "consumers": {
            backend: {
                "backend": backend, "evidence": "approximate", "root": [x, y],
                "point_domain": [[str(Fraction(value))] * 2 for value in (x, y)],
                "residual_bounds": [["-1/10000000000000", "1/10000000000000"]] * 2,
            }
            for backend in ("frankennumpy", "frankenscipy")
        },
    }


class NumericCampaignValidation(unittest.TestCase):
    def setUp(self):
        self.case = _example_record()
        validate_numeric_case(self.case)

    def test_wrong_program_bytes(self):
        self.case["program_base64"] = base64.b64encode(b"different program").decode()
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_numeric_case(self.case)

    def test_swapped_jacobian_rows(self):
        self.case["compiled_jacobian"].reverse()
        with self.assertRaisesRegex(ValueError, "analytic reference"):
            validate_numeric_case(self.case)

    def test_nonfinite_point_evaluation(self):
        self.case["compiled_residuals"][0] = math.nan
        with self.assertRaisesRegex(ValueError, "analytic reference"):
            validate_numeric_case(self.case)

    def test_narrowed_box_jacobian(self):
        self.case["box_enclosure"]["jacobian_bounds"][1][1] = ["0", "1"]
        with self.assertRaisesRegex(ValueError, "excludes"):
            validate_numeric_case(self.case)

    def test_missing_or_approximate_bounds(self):
        for pair in (["1"], [0.1, 1.0], ["2", "1"]):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                self.case["box_enclosure"]["residual_bounds"][0] = pair
                validate_numeric_case(self.case)

    def test_consumer_residual_bound_over_threshold(self):
        self.case["consumers"]["frankennumpy"]["residual_bounds"][1] = ["0", "1/100000000000"]
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_numeric_case(self.case)

    def test_residual_checked_at_nearby_but_different_point(self):
        self.case["consumers"]["frankenscipy"]["point_domain"][0] = ["1/4", "1/4"]
        with self.assertRaisesRegex(ValueError, "returned iterate"):
            validate_numeric_case(self.case)

    def test_roots_agree_but_solve_the_wrong_system(self):
        for consumer in self.case["consumers"].values():
            consumer.update(root=[1.0, 1.0], point_domain=[["1", "1"], ["1", "1"]])
        with self.assertRaisesRegex(ValueError, "analytic residual diagnostic"):
            validate_numeric_case(self.case)


if __name__ == "__main__":
    unittest.main()
