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


class TestVeloxExpression(unittest.TestCase):
    def test_from_string(self):
        pv.Expression.from_string("a + b")

    def test_eval(self):
        expr = pv.Expression.from_string("a + b")
        a = pv.from_list([1, 2, 3, None])
        b = pv.from_list([4, 5, None, 6])
        c = expr.evaluate(["a", "b"], [a, b])
        self.assertEqual(c[0], 5)
        self.assertEqual(c[1], 7)
        self.assertEqual(c[2], None)
        self.assertEqual(c[3], None)

        with self.assertRaises(ValueError):
            d = pv.from_list([10, 11])
            expr.evaluate(["a", "b"], [a, d])

        with self.assertRaises(RuntimeError):
            d = pv.from_list(["hi", "bye", "hello", "goodbye"])
            expr.evaluate(["a", "b"], [a, d])

