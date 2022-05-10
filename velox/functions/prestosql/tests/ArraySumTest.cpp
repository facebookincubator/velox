/*
* Copyright (c) Facebook, Inc. and its affiliates.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <optional>
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {

// Class to test the array_duplicates operator.
class ArraySumTest : public FunctionBaseTest {
protected:
 // Evaluate an expression.
 void testExpr(
     const VectorPtr& expected,
     const std::string& expression,
     const std::vector<VectorPtr>& input) {
   auto result = evaluate<ArrayVector>(expression, makeRowVector(input));
   assertEqualVectors(expected, result);
 }
};

} // namespace

// Test integer arrays.
TEST_F(ArraySumTest, integerArrays) {
  auto input = makeNullableArrayVector<int64_t>({{0, 1, 2},
    {std::nullopt, 1, 2},
    {std::nullopt}});
  auto expected = makeNullableFlatVector<int64_t>({3, 3, 0});
  testExpr(expected, "array_sum(C0)", {input});
}

