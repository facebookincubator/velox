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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {

class ArrayUnionTest : public FunctionBaseTest {
 protected:
  // Evaluate an expression.
  void testExpr(
      const VectorPtr& expected,
      const VectorPtr& array1,
      const VectorPtr& array2) {
    auto result = evaluate<ArrayVector>(
        "array_union(c0, c1)", makeRowVector({array1, array2}));
    assertEqualVectors(expected, result);
  }

  template <typename T>
  void testNumerics() {
    auto array1 = makeNullableArrayVector<T>(
        {{1, std::nullopt, 2, 2, 4}, {5, std::nullopt, 6, 6}});
    auto array2 =
        makeNullableArrayVector<T>({{3, 1, std::nullopt, 2}, {9, 5, 6}});
    auto expected = makeNullableArrayVector<T>(
        {{1, std::nullopt, 2, 4, 3}, {5, std::nullopt, 6, 9}});
    testExpr(expected, array1, array2);
  }

  template <typename T>
  void testNullFreeNumerics() {
    auto array1 = makeArrayVector<T>({{1, 2, 2, 4}, {5, 6, 6}});
    auto array2 = makeArrayVector<T>({{3, 1, 2}, {9, 5, 6}});
    auto expected = makeArrayVector<T>({{1, 2, 4, 3}, {5, 6, 9}});
    testExpr(expected, array1, array2);
  }
};

} // namespace

TEST_F(ArrayUnionTest, numerics) {
  testNumerics<int8_t>();
  testNumerics<int16_t>();
  testNumerics<int32_t>();
  testNumerics<int64_t>();
  testNumerics<float>();
  testNumerics<double>();
}

TEST_F(ArrayUnionTest, strings) {
  auto array1 = makeNullableArrayVector<StringView>(
      {{"A", "B", "C", std::nullopt},
       {"Sadly I stand on the toes of their predecessors, not on their shoulders.",
        "E"}});
  auto array2 = makeNullableArrayVector<StringView>(
      {{std::nullopt, "C", "A", "E"},
       {"E", "D", "Velox is a novel C++ database acceleration library"}});
  auto expected = makeNullableArrayVector<StringView>(
      {{"A", "B", "C", std::nullopt, "E"},
       {"Sadly I stand on the toes of their predecessors, not on their shoulders.",
        "E",
        "D",
        "Velox is a novel C++ database acceleration library"}});
  testExpr(expected, array1, array2);
}

TEST_F(ArrayUnionTest, booleans) {
  auto array1 = makeNullableArrayVector<bool>({{true, std::nullopt}, {false}});
  auto array2 =
      makeNullableArrayVector<bool>({{std::nullopt, false}, {std::nullopt}});
  auto expected = makeNullableArrayVector<bool>(
      {{true, std::nullopt, false}, {false, std::nullopt}});
  testExpr(expected, array1, array2);
}

TEST_F(ArrayUnionTest, dates) {
  using D = Date;
  auto array1 = makeNullableArrayVector<Date>(
      {{D{1}, std::nullopt, D{1024}}, {D{2}, D{2048}}});
  auto array2 = makeNullableArrayVector<Date>(
      {{
           D{3},
           std::nullopt,
       },
       {D{128}}});
  auto expected = makeNullableArrayVector<Date>(
      {{D{1}, std::nullopt, D{1024}, D{3}}, {D{2}, D{2048}, D{128}}});
  testExpr(expected, array1, array2);
}

TEST_F(ArrayUnionTest, timestamps) {
  using T = Timestamp;
  auto array1 = makeNullableArrayVector<Timestamp>(
      {{T{0, 1}, T{0, 1024}}, {T{1, 2}, std::nullopt, T{1, 2048}}});
  auto array2 = makeNullableArrayVector<Timestamp>({{T{0, 3}}, {T{1, 128}}});
  auto expected = makeNullableArrayVector<Timestamp>(
      {{T{0, 1}, T{0, 1024}, T{0, 3}},
       {T{1, 2}, std::nullopt, T{1, 2048}, T{1, 128}}});
  testExpr(expected, array1, array2);
}

TEST_F(ArrayUnionTest, nullFreeNumerics) {
  testNullFreeNumerics<int8_t>();
  testNullFreeNumerics<int16_t>();
  testNullFreeNumerics<int32_t>();
  testNullFreeNumerics<int64_t>();
  testNullFreeNumerics<float>();
  testNullFreeNumerics<double>();
}

TEST_F(ArrayUnionTest, nullFreeBooleans) {
  auto array1 = makeArrayVector<bool>({{true, true}, {false, true}});
  auto array2 = makeArrayVector<bool>({{true}, {true, true}});
  auto expected = makeArrayVector<bool>({{true}, {false, true}});
  testExpr(expected, array1, array2);
}

TEST_F(ArrayUnionTest, nullFreeDates) {
  using D = Date;
  auto array1 = makeArrayVector<Date>({{D{1}, D{1024}}, {D{2}, D{2048}}});
  auto array2 = makeArrayVector<Date>({{D{3}}, {D{128}}});
  auto expected =
      makeArrayVector<Date>({{D{1}, D{1024}, D{3}}, {D{2}, D{2048}, D{128}}});
  testExpr(expected, array1, array2);
}

TEST_F(ArrayUnionTest, nullFreeTimestamps) {
  using T = Timestamp;
  auto array1 = makeArrayVector<Timestamp>(
      {{T{0, 1}, T{0, 1024}}, {T{1, 2}, T{1, 2048}}});
  auto array2 = makeArrayVector<Timestamp>({{T{0, 3}}, {T{1, 128}}});
  auto expected = makeArrayVector<Timestamp>(
      {{T{0, 1}, T{0, 1024}, T{0, 3}}, {T{1, 2}, T{1, 2048}, T{1, 128}}});
  testExpr(expected, array1, array2);
}

TEST_F(ArrayUnionTest, nullFreeStrings) {
  auto array1 = makeArrayVector<StringView>(
      {{"A", "B", "C"},
       {"Sadly I stand on the toes of their predecessors, not on their shoulders.",
        "E"}});
  auto array2 = makeArrayVector<StringView>(
      {{"C", "A", "E"},
       {"E", "D", "Velox is a novel C++ database acceleration library"}});
  auto expected = makeArrayVector<StringView>(
      {{"A", "B", "C", "E"},
       {"Sadly I stand on the toes of their predecessors, not on their shoulders.",
        "E",
        "D",
        "Velox is a novel C++ database acceleration library"}});
  testExpr(expected, array1, array2);
}
