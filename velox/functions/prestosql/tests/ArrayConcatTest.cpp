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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {

class ArrayConcatTest : public FunctionBaseTest {
 protected:
  // Evaluate an expression.
  void testArrayConcat(
      const VectorPtr& expected,
      const VectorPtr& input,
      const std::string exp) {
    std::string expr = "concat(c0, " + exp + ")";
    auto result = evaluate(expr, makeRowVector({input}));
    assertEqualVectors(expected, result);

    // Test constant input.
    const vector_size_t firstRow = 0;
    result = evaluate(
        expr, makeRowVector({BaseVector::wrapInConstant(10, firstRow, input)}));
    assertEqualVectors(
        BaseVector::wrapInConstant(10, firstRow, expected), result);

    const vector_size_t lastRow = input->size() - 1;
    result = evaluate(
        expr, makeRowVector({BaseVector::wrapInConstant(10, lastRow, input)}));
    assertEqualVectors(
        BaseVector::wrapInConstant(10, lastRow, expected), result);
  }
};

} // namespace

// Test integer arrays.
TEST_F(ArrayConcatTest, int64Input) {
  auto input = makeNullableArrayVector<int64_t>(
      {{0, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt}});
  auto expected = makeNullableArrayVector<int64_t>(
      {{0, 1, 2, 1}, {std::nullopt, 1, 2, 1}, {std::nullopt, 1}});
  testArrayConcat(expected, input, "Array[1]");
}

TEST_F(ArrayConcatTest, int32Input) {
  auto input = makeNullableArrayVector<int32_t>(
      {{0, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt}});
  auto expected = makeNullableArrayVector<int32_t>(
      {{0, 1, 2, 1}, {std::nullopt, 1, 2, 1}, {std::nullopt, 1}});
  testArrayConcat(expected, input, "Array[1]");
}

TEST_F(ArrayConcatTest, int16Input) {
  auto input = makeNullableArrayVector<int16_t>(
      {{0, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt}});
  auto expected = makeNullableArrayVector<int16_t>(
      {{0, 1, 2, 1}, {std::nullopt, 1, 2, 1}, {std::nullopt, 1}});
  testArrayConcat(expected, input, "Array[cast (1 as smallint)]");
}

TEST_F(ArrayConcatTest, int8Input) {
  auto input = makeNullableArrayVector<int8_t>(
      {{0, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt}});
  auto expected = makeNullableArrayVector<int8_t>(
      {{0, 1, 2, 1}, {std::nullopt, 1, 2, 1}, {std::nullopt, 1}});
  testArrayConcat(expected, input, "Array[cast (1 as tinyInt)]");
}

// Test floating point arrays
TEST_F(ArrayConcatTest, realInput) {
  auto input = makeNullableArrayVector<float_t>(
      {{0, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt}});
  auto expected = makeNullableArrayVector<float_t>(
      {{0, 1, 2, 1.0}, {std::nullopt, 1, 2, 1.0}, {std::nullopt, 1.0}});
  testArrayConcat(expected, input, "Array[cast(1.0 as float)]");
}

TEST_F(ArrayConcatTest, doubleInput) {
  auto input = makeNullableArrayVector<double_t>(
      {{0, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt}});
  auto expected = makeNullableArrayVector<double_t>(
      {{0, 1, 2, 1.0}, {std::nullopt, 1, 2, 1.0}, {std::nullopt, 1.0}});
  testArrayConcat(expected, input, "Array[1.0]");
}

TEST_F(ArrayConcatTest, StringInput) {
  auto input = makeNullableArrayVector<StringView>(
      {{"test", "tess1", "test2"},
       {std::nullopt, "test1", "test2"},
       {std::nullopt}});
  auto expected = makeNullableArrayVector<StringView>(
      {{"test", "tess1", "test2", "concat"},
       {std::nullopt, "test1", "test2", "concat"},
       {std::nullopt, "concat"}});
  testArrayConcat(expected, input, "Array['concat']");
}

TEST_F(ArrayConcatTest, TimestampInput) {
  auto input = makeNullableArrayVector<Timestamp>(
      {{Timestamp(0, 0), Timestamp(0201, 0400000), Timestamp(-0201, 0400000)},
       {std::nullopt, Timestamp(0, 0), Timestamp(0201, 0400000)},
       {std::nullopt}});

  auto expected = makeNullableArrayVector<Timestamp>(
      {{Timestamp(0, 0),
        Timestamp(0201, 0400000),
        Timestamp(-0201, 0400000),
        Timestamp(946684800, 0)},
       {std::nullopt,
        Timestamp(0, 0),
        Timestamp(0201, 0400000),
        Timestamp(946684800, 0)},
       {std::nullopt, Timestamp(946684800, 0)}});
  testArrayConcat(expected, input, "Array[cast ('2000-01-01' as timestamp)]");
}

TEST_F(ArrayConcatTest, DateInput) {
  auto input = makeNullableArrayVector<Date>(
      {{Date(0), Date(-121), Date(500)},
       {std::nullopt, Date(0), Date(-121)},
       {std::nullopt}});
  //	10957, fromDateString("2000-01-01")
  auto expected = makeNullableArrayVector<Date>(
      {{Date(0), Date(-121), Date(500), Date(10957)},
       {std::nullopt, Date(0), Date(-121), Date(10957)},
       {std::nullopt, Date(10957)}});
  testArrayConcat(expected, input, "Array[cast ('2000-01-01' as date)]");
}

TEST_F(ArrayConcatTest, castInput) {
  auto input = makeNullableArrayVector<float_t>(
      {{0, 1.0f, 2.0f}, {std::nullopt, 1, 2}, {std::nullopt}});
  auto expected = makeNullableArrayVector<double_t>(
      {{0, 1, 2, 1}, {std::nullopt, 1, 2, 1}, {std::nullopt, 1}});
  testArrayConcat(expected, input, "Array[1]");
}
