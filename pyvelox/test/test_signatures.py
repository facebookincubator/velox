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

import unittest

import pyvelox.pyvelox as pv


class TestFunctionSignatures(unittest.TestCase):
    def test_clear_signatures(self):
        pv.clear_signatures()
        signatures = pv.get_function_signatures()
        self.assertEqual(len(signatures), 0)

    def test_get_signatures(self):
        pv.register_presto_signatures()
        presto_signatures = pv.get_function_signatures()
        self.assertTrue(len(presto_signatures) > 0)

        pv.clear_signatures()
        pv.register_spark_signatures()
        spark_signatures = pv.get_function_signatures()
        self.assertTrue(len(spark_signatures) > 0)

    def test_function_signature(self):
        pv.clear_signatures()
        pv.register_presto_signatures()
        presto_signatures = pv.get_function_signatures()

        concat_signatures = presto_signatures["concat"]
        self.assertTrue(len(concat_signatures) > 0)
        self.assertEqual(str(concat_signatures[0].return_type()), "varchar")
        self.assertEqual(str(concat_signatures[0]), "(varchar,varchar...) -> varchar")

    def test_function_prefix(self):
        pv.clear_signatures()
        pv.register_presto_signatures("foo")
        presto_signatures = pv.get_function_signatures()

        concat_signatures = presto_signatures["fooconcat"]
        self.assertTrue(len(concat_signatures) > 0)

        pv.clear_signatures()
        pv.register_spark_signatures("bar")
        spark_signatures = pv.get_function_signatures()

        concat_signatures = spark_signatures["barconcat"]
        self.assertTrue(len(concat_signatures) > 0)
