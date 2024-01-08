# Copyright (c) Facebook, Inc. and its affiliates.
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

import pyvelox.pyvelox as pv
import unittest
import shutil, tempfile
import os

# Run the tests from the base path of the project
BASE_PATH = os.path.join(os.getcwd(), "velox/substrait/tests/data/")


class TestVeloxSubstrait(unittest.TestCase):
    def setUp(self):
        # create a temporary directory
        pv.initialize_substrait()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove the temporary directory
        pv.finalize_substrait()
        shutil.rmtree(self.test_dir)

    def test_simple(self):
        other_path = BASE_PATH + "substrait_virtualTable.json"
        res = pv.run_substrait_query(other_path)
        # TODO: Vectors cannot be created with all None values in Python.
        #   which forces a string comparison for the moment.
        expected_vec_str = (
            "0: {2499109626526694126, 581869302, 0.9057919341454927, true, null}\n1: {2342493223442167775, "
            "-708632711, 0.9688677711242314, false, null}\n2: {4077358421272316858, -133711905, 0.6323592500344464, "
            "false, null}"
        )
        assert str(res) == expected_vec_str

    def test_file_type_failure(self):
        other_path = self.test_dir + "dummy_plan.txt"
        with open(other_path, "w") as fp:
            fp.write("no json")

        with self.assertRaises(ValueError) as cm:
            pv.run_substrait_query(other_path)
        self.assertEqual('plan should be path to a plan in JSON format.', str(cm.exception))
