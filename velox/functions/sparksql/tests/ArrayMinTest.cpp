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

#include <limits>
#include <optional>
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class ArrayMinTest : public SparkFunctionBaseTest {
 protected:
  template <typename T, typename TExpected = T>
  void testExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const std::vector<VectorPtr>& input) {
    auto result =
        evaluate<SimpleVector<TExpected>>(expression, makeRowVector(input));
    assertEqualVectors(expected, result);
  }

  template <typename T>
  void testIntNullable() {
    auto arrayVector = makeNullableArrayVector<T>(
        {{-1, 0, 1, 2, 3, 4},
         {4, 3, 2, 1, std::nullopt, 0, -1, -2},
         {-5, -4, -3, -2, -1},
         {101, 102, 103, 104, std::nullopt},
         {std::nullopt, -1, -2, std::nullopt, -3, -4},
         {},
         {std::nullopt, std::nullopt}});
    auto expected = makeNullableFlatVector<T>(
        {-1, -2, -5, 101, -4, std::nullopt, std::nullopt});
    testExpr<T>(expected, "array_min(C0)", {arrayVector});
  }

  template <typename T>
  void testInt() {
    auto arrayVector = makeArrayVector<T>(
        {{-1, 0, 1, 2, 3, 4},
         {4, 3, 2, 1, 0, -1, -2},
         {-5, -4, -3, -2, -1},
         {101, 102, 103, 104, 105},
         {}});
    auto expected = makeNullableFlatVector<T>({-1, -2, -5, 101, std::nullopt});
    testExpr<T>(expected, "array_min(C0)", {arrayVector});
  }

  void testInLineVarcharNullable() {
    using S = StringView;

    auto arrayVector = makeNullableArrayVector<S>({
        {S("red"), S("blue")},
        {std::nullopt, S("blue"), S("yellow"), S("orange")},
        {},
        {std::nullopt, std::nullopt},
        {S("red"), S("purple"), S("green")},
    });
    auto expected = makeNullableFlatVector<S>(
        {S("blue"), S("blue"), std::nullopt, std::nullopt, S("green")});
    testExpr<S>(expected, "array_min(C0)", {arrayVector});
  }

  void testVarcharNullable() {
    using S = StringView;
    // use > 12 length string to avoid inlining
    auto arrayVector = makeNullableArrayVector<S>({
        {S("red shiny car ahead"), S("blue clear sky above")},
        {std::nullopt,
         S("blue clear sky above"),
         S("yellow rose flowers"),
         S("orange beautiful sunset")},
        {},
        {std::nullopt, std::nullopt},
        {S("red shiny car ahead"),
         S("purple is an elegant color"),
         S("green plants make us happy")},
    });
    auto expected = makeNullableFlatVector<S>(
        {S("blue clear sky above"),
         S("blue clear sky above"),
         std::nullopt,
         std::nullopt,
         S("green plants make us happy")});
    testExpr<S>(expected, "array_min(C0)", {arrayVector});
  }

  void testBoolNullable() {
    auto arrayVector = makeNullableArrayVector<bool>(
        {{true, false},
         {true},
         {false},
         {},
         {std::nullopt, std::nullopt},
         {true, false, true, std::nullopt},
         {std::nullopt, true, false, true},
         {false, false, false},
         {true, true, true}});

    auto expected = makeNullableFlatVector<bool>(
        {false,
         true,
         false,
         std::nullopt,
         std::nullopt,
         false,
         false,
         false,
         true});
    testExpr<bool>(expected, "array_min(C0)", {arrayVector});
  }

  void testBool() {
    auto arrayVector = makeArrayVector<bool>(
        {{true, false},
         {true},
         {false},
         {},
         {false, false, false},
         {true, true, true}});

    auto expected = makeNullableFlatVector<bool>(
        {false, true, false, std::nullopt, false, true});
    testExpr<bool>(expected, "array_min(C0)", {arrayVector});
  }

  template <typename T>
  void testNumNullable() {
    constexpr T kNaN = std::numeric_limits<T>::quiet_NaN();
    auto arrayVector = makeNullableArrayVector<T>(
        {{-1.0, 0.0, 1.0, kNaN, 2.0, 3.0, 4.0},
         {4.0, 3.0, 2.0, 1.0, std::nullopt, 0.0, -1.0, -2.0, kNaN},
         {-5.0, -4.0, -3.0, -2.0, -1.0},
         {101.0, 102.0, 103.0, 104.0, std::nullopt},
         {std::nullopt, -1.0, -2.0, std::nullopt, -3.0, -4.0},
         {},
         {std::nullopt, std::nullopt},
         {kNaN, std::nullopt, -1.0, kNaN, -2.0},
         {std::nullopt, kNaN}});
    auto expected = makeNullableFlatVector<T>(
        {-1.0,
         -2.0,
         -5.0,
         101.0,
         -4.0,
         std::nullopt,
         std::nullopt,
         -2.0,
         kNaN});
    testExpr<T>(expected, "array_min(C0)", {arrayVector});
  }
};

TEST_F(ArrayMinTest, intArrays) {
  testIntNullable<int8_t>();
  testIntNullable<int16_t>();
  testIntNullable<int32_t>();
  testIntNullable<int64_t>();
  testInt<int8_t>();
  testInt<int16_t>();
  testInt<int32_t>();
  testInt<int64_t>();
}

TEST_F(ArrayMinTest, varcharArrays) {
  testInLineVarcharNullable();
  testVarcharNullable();
}

TEST_F(ArrayMinTest, boolArrays) {
  testBoolNullable();
  testBool();
}

TEST_F(ArrayMinTest, numArrays) {
  testNumNullable<float>();
  testNumNullable<double>();
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
