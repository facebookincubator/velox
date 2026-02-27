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

from pyvelox.expression import col, lit


class TestPyVeloxExpressionAuthoring(unittest.TestCase):
    def test_expression_authoring(self):
        # Column references.
        c0 = col("c0")
        c1 = col("c1")

        self.assertEqual(str(c0), "\"c0\"")
        self.assertEqual(str(c1), "\"c1\"")

        # Literals.
        li = lit(42)
        ld = lit(0.42)
        ls = lit("my str")

        l_false = lit(False)
        l_true = lit(True)
        l_none = lit(None)

        self.assertEqual(str(li), "42")
        self.assertEqual(str(ld), "0.42")
        self.assertEqual(str(ls), "my str")
        self.assertEqual(str(l_false), "false")
        self.assertEqual(str(l_true), "true")
        self.assertEqual(str(l_none), "null")

        # Arithmetics.
        self.assertEqual(str(c0 + c1), "plus(\"c0\",\"c1\")")
        self.assertEqual(str(c0 + li), "plus(\"c0\",42)")
        self.assertEqual(str(c0 + 42), "plus(\"c0\",42)")
        self.assertEqual(str(42 + c0), "plus(42,\"c0\")")

        self.assertEqual(str(c0 - c1), "minus(\"c0\",\"c1\")")
        self.assertEqual(str(c0 - li), "minus(\"c0\",42)")
        self.assertEqual(str(c0 - 42), "minus(\"c0\",42)")
        self.assertEqual(str(42 - c0), "minus(42,\"c0\")")

        self.assertEqual(str(c0 * c1), "multiply(\"c0\",\"c1\")")
        self.assertEqual(str(c0 * li), "multiply(\"c0\",42)")
        self.assertEqual(str(c0 * 42), "multiply(\"c0\",42)")
        self.assertEqual(str(42 * c0), "multiply(42,\"c0\")")

        self.assertEqual(str(c0 / c1), "divide(\"c0\",\"c1\")")
        self.assertEqual(str(c0 / li), "divide(\"c0\",42)")
        self.assertEqual(str(c0 / 42), "divide(\"c0\",42)")
        self.assertEqual(str(42 / c0), "divide(42,\"c0\")")

        # Comparisons.
        self.assertEqual(str(c0 > c1), "gt(\"c0\",\"c1\")")
        self.assertEqual(str(c0 > 19.2), "gt(\"c0\",19.2)")
        self.assertEqual(str("asd" > c0), "lt(\"c0\",asd)")

        self.assertEqual(str(c0 < c1), "lt(\"c0\",\"c1\")")
        self.assertEqual(str(c0 < 1), "lt(\"c0\",1)")
        self.assertEqual(str(1 < c1), "gt(\"c1\",1)")

        self.assertEqual(str(c0 <= c1), "lte(\"c0\",\"c1\")")
        self.assertEqual(str(c0 <= 1), "lte(\"c0\",1)")
        self.assertEqual(str(1 <= c1), "gte(\"c1\",1)")

        self.assertEqual(str(c0 >= c1), "gte(\"c0\",\"c1\")")
        self.assertEqual(str(c0 >= 1), "gte(\"c0\",1)")
        self.assertEqual(str(1 >= c1), "lte(\"c1\",1)")

        self.assertEqual(str(c0 == c1), "eq(\"c0\",\"c1\")")
        self.assertEqual(str(c0 == 1), "eq(\"c0\",1)")
        self.assertEqual(str(1 == c1), "eq(\"c1\",1)")

        self.assertEqual(str(c0 != c1), "neq(\"c0\",\"c1\")")
        self.assertEqual(str(c0 != 1), "neq(\"c0\",1)")
        self.assertEqual(str(1 != c1), "neq(\"c1\",1)")

        # Conjuncts
        self.assertEqual(str(c0 & c1), "and(\"c0\",\"c1\")")
        self.assertEqual(str(c0 & True), "and(\"c0\",true)")
        self.assertEqual(str(True & c0), "and(true,\"c0\")")

        self.assertEqual(str(c0 | c1), "or(\"c0\",\"c1\")")
        self.assertEqual(str(c0 | True), "or(\"c0\",true)")
        self.assertEqual(str(True | c0), "or(true,\"c0\")")

        self.assertEqual(str(~c0), "not(\"c0\")")
        self.assertEqual(str(c0 & (c1 | ~c0)), "and(\"c0\",or(\"c1\",not(\"c0\")))")
