# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path
import tempfile
import unittest

from signature import bias_aggregates, parse_args


class AggregateBiasTest(unittest.TestCase):
    SUM = "(bigint) -> bigint -> bigint"
    AVG = "(bigint) -> row(double,bigint) -> double"
    COLLECT_SET = "(T) -> array(T) -> array(T)"
    COLLECT_SET_NULLS = "(T,constant boolean) -> array(T) -> array(T)"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.previous_output = os.environ.get("GITHUB_OUTPUT")
        os.environ["GITHUB_OUTPUT"] = str(self.directory / "outputs")
        self.addCleanup(self.restore_output)

    def restore_output(self):
        if self.previous_output is None:
            os.environ.pop("GITHUB_OUTPUT", None)
        else:
            os.environ["GITHUB_OUTPUT"] = self.previous_output

    def run_bias(self, before, after, eligible=None, group="spark"):
        base = self.directory / "base.json"
        contender = self.directory / "contender.json"
        output = self.directory / "bias"
        base.write_text(json.dumps(before))
        contender.write_text(json.dumps(after))
        args = [
            "bias_aggregates",
            "--group",
            group,
            str(base),
            str(contender),
            str(output),
            str(self.directory / "errors"),
        ]
        if eligible is not None:
            eligible_file = self.directory / "eligible"
            eligible_file.write_text("\n".join(eligible))
            args.extend(["--fuzzable_functions", str(eligible_file)])
        status = bias_aggregates(parse_args(args))
        outputs = dict(
            line.split("=", 1)
            for line in (self.directory / "outputs").read_text().splitlines()
        )
        return status, outputs, output

    def test_excluded_companion_keeps_regular_fuzzer_selected(self):
        base = {"collect_set": [self.COLLECT_SET], "sum": [self.SUM]}
        contender = {
            **base,
            "collect_set_merge_extract": ["(array(T)) -> array(T) -> array(T)"],
        }
        (self.directory / "bias").write_text("collect_set")
        status, outputs, output = self.run_bias(base, contender, ["sum"])
        self.assertEqual(status, 0)
        self.assertEqual(outputs["spark_aggregate_functions"], "false")
        self.assertEqual(outputs["spark_aggregate_error"], "false")
        self.assertEqual(output.read_text(), "")

    def test_mixed_changes_select_supported_names_once_in_sorted_order(self):
        base = {
            "collect_set": [self.COLLECT_SET],
            "sum": [self.SUM],
            "avg": [self.AVG],
        }
        contender = {
            **base,
            "collect_set_merge_extract": ["(array(T)) -> array(T) -> array(T)"],
            "sum_partial": [self.SUM],
            "sum_merge": [self.SUM],
            "avg_partial": ["(bigint) -> row(double,bigint) -> row(double,bigint)"],
        }
        status, outputs, output = self.run_bias(base, contender, ["sum", "avg"])
        self.assertEqual(status, 0)
        self.assertEqual(outputs["spark_aggregate_functions"], "true")
        self.assertEqual(output.read_text(), "avg,sum")

    def test_excluded_signature_removal_still_reports_incompatibility(self):
        base = {
            "collect_set": [self.COLLECT_SET, self.COLLECT_SET_NULLS],
            "sum": [self.SUM],
        }
        contender = {"collect_set": [self.COLLECT_SET], "sum": [self.SUM]}
        _, outputs, output = self.run_bias(base, contender, ["sum"])
        self.assertEqual(outputs["spark_aggregate_error"], "true")
        self.assertEqual(outputs["spark_aggregate_functions"], "false")
        self.assertEqual(output.read_text(), "")
        self.assertIn(self.COLLECT_SET_NULLS, (self.directory / "errors").read_text())

    def test_invalid_exported_names_are_rejected(self):
        for eligible in ([], ["not_registered"]):
            with self.subTest(eligible=eligible), self.assertRaises(ValueError):
                self.run_bias({"sum": [self.SUM]}, {"sum": [self.SUM]}, eligible)

    def test_unfiltered_presto_selection_is_preserved(self):
        _, outputs, output = self.run_bias(
            {"sum": [self.SUM]},
            {"sum": [self.SUM], "sum_partial": [self.SUM]},
            group="presto",
        )
        self.assertEqual(outputs["presto_aggregate_functions"], "true")
        self.assertEqual(output.read_text(), "sum")


if __name__ == "__main__":
    unittest.main()
