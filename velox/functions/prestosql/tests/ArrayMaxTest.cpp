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
using namespace facebook::velox::functions::test;

namespace {

template <typename Type>
class ArrayMaxTest : public FunctionBaseTest {
 public:
  using T = typename Type::NativeType::NativeType;

  void testMaxExpr(VectorPtr input, VectorPtr expected) {
    auto result =
        evaluate<SimpleVector<T>>("array_max(C0)", makeRowVector({input}));
    assertEqualVectors(expected, result);
  }
  void testMaxNullableExpr() {
    ASSERT_FALSE(true) << "No test added for this type";
  }
  void testMaxExpr() {
    ASSERT_FALSE(true) << "No test added for this type";
  }
};

template <typename T>
void getTypedNullableIntegerInputExpectedArrays(
    FunctionBaseTest* test,
    VectorPtr& input,
    VectorPtr& expected) {
  input = test->makeNullableArrayVector<T>(
      {{std::numeric_limits<T>::min(),
        0,
        1,
        2,
        3,
        std::numeric_limits<T>::max()},
       {std::numeric_limits<T>::max(),
        3,
        2,
        1,
        0,
        -1,
        std::numeric_limits<T>::min()},
       {101, 102, 103, std::numeric_limits<T>::max(), std::nullopt},
       {std::nullopt, -1, -2, -3, std::numeric_limits<T>::min()},
       {},
       {std::nullopt}});
  expected = test->makeNullableFlatVector<T>(
      {std::numeric_limits<T>::max(),
       std::numeric_limits<T>::max(),
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
}

template <typename T>
void getTypedIntegerInputExpectedArrays(
    FunctionBaseTest* test,
    VectorPtr& input,
    VectorPtr& expected) {
  input = test->makeArrayVector<T>(
      {{std::numeric_limits<T>::min(), 0, 1, 2, 3, 4},
       {std::numeric_limits<T>::max(), 3, 2, 1, 0, -1, -2},
       {std::numeric_limits<T>::max(),
        101,
        102,
        103,
        104,
        105,
        std::numeric_limits<T>::max()},
       {}});
  expected = test->makeNullableFlatVector<T>(
      {4,
       std::numeric_limits<T>::max(),
       std::numeric_limits<T>::max(),
       std::nullopt});
}
template <>
void ArrayMaxTest<SmallintType>::testMaxNullableExpr() {
  VectorPtr input, expected;
  getTypedNullableIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<SmallintType>::testMaxExpr() {
  VectorPtr input, expected;
  getTypedIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<TinyintType>::testMaxNullableExpr() {
  VectorPtr input, expected;
  getTypedNullableIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<TinyintType>::testMaxExpr() {
  VectorPtr input, expected;
  getTypedIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<IntegerType>::testMaxNullableExpr() {
  VectorPtr input, expected;
  getTypedNullableIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<IntegerType>::testMaxExpr() {
  VectorPtr input, expected;
  getTypedIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<BigintType>::testMaxNullableExpr() {
  VectorPtr input, expected;
  getTypedNullableIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<BigintType>::testMaxExpr() {
  VectorPtr input, expected;
  getTypedIntegerInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<BooleanType>::testMaxNullableExpr() {
  auto input = makeNullableArrayVector<T>(
      {{true, false},
       {true},
       {false},
       {},
       {true, false, true, std::nullopt},
       {std::nullopt, true, false, true},
       {false, false, false},
       {true, true, true}});
  auto expected = makeNullableFlatVector<T>(
      {true,
       true,
       false,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       false,
       true});
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<BooleanType>::testMaxExpr() {
  auto input = makeNullableArrayVector<T>(
      {{true, false},
       {true},
       {false},
       {},
       {false, false, false},
       {true, true, true}});
  auto expected =
      makeNullableFlatVector<T>({true, true, false, std::nullopt, false, true});
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<VarcharType>::testMaxNullableExpr() {
  using S = StringView;
  auto input = makeNullableArrayVector<S>({
      {S("red"), S("blue")},
      {std::nullopt, S("blue"), S("yellow"), S("orange")},
      {},
      {S("red"), S("purple"), S("green")},
  });
  auto expected = makeNullableFlatVector<StringView>(
      {S("red"), std::nullopt, std::nullopt, S("red")});
}

template <>
void ArrayMaxTest<VarcharType>::testMaxExpr() {
  using S = StringView;
  auto input = makeArrayVector<S>({
      {S("red"), S("blue")},
      {S("blue"), S("yellow"), S("orange")},
      {},
      {S("red"), S("purple"), S("green")},
  });
  auto expected = makeNullableFlatVector<StringView>(
      {S("red"), S("yellow"), std::nullopt, S("red")});
  testMaxExpr(input, expected);
}

template <typename T>
void getTypedNullableFloatingInputExpectedArrays(
    FunctionBaseTest* test,
    VectorPtr& input,
    VectorPtr& expected) {
  input = test->makeNullableArrayVector<T>(
      {{0.0000, 0.00001},
       {std::nullopt, 1.1, 1.11, -2.2, -1.0, std::numeric_limits<T>::min()},
       {std::numeric_limits<T>::min(),
        -0.0001,
        -0.0002,
        -0.0003,
        std::numeric_limits<T>::max()},
       {},
       {std::numeric_limits<T>::min(), 1.1, 1.22222, 1.33, std::nullopt},
       {-0.00001, -0.0002, 0.0001}});
  expected = test->makeNullableFlatVector<T>(
      {0.00001,
       std::nullopt,
       std::numeric_limits<T>::max(),
       std::nullopt,
       std::nullopt,
       0.0001});
}

template <typename T>
void getTypedFloatingInputExpectedArrays(
    FunctionBaseTest* test,
    VectorPtr& input,
    VectorPtr& expected) {
  input = test->makeArrayVector<T>(
      {{0.0000, 0.00001},
       {std::numeric_limits<T>::max(),
        1.1,
        1.11,
        -2.2,
        -1.0,
        std::numeric_limits<T>::min()},
       {std::numeric_limits<T>::min(),
        -0.0001,
        -0.0002,
        -0.0003,
        std::numeric_limits<T>::max()},
       {},
       {1.1, 1.22222, 1.33},
       {-0.00001, -0.0002, 0.0001}});
  expected = test->makeNullableFlatVector<T>(
      {0.00001,
       std::numeric_limits<T>::max(),
       std::numeric_limits<T>::max(),
       std::nullopt,
       1.33,
       0.0001});
}

template <>
void ArrayMaxTest<RealType>::testMaxNullableExpr() {
  VectorPtr input, expected;
  getTypedNullableFloatingInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<RealType>::testMaxExpr() {
  VectorPtr input, expected;
  getTypedFloatingInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<DoubleType>::testMaxNullableExpr() {
  VectorPtr input, expected;
  getTypedNullableFloatingInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

template <>
void ArrayMaxTest<DoubleType>::testMaxExpr() {
  VectorPtr input, expected;
  getTypedFloatingInputExpectedArrays<T>(this, input, expected);
  testMaxExpr(input, expected);
}

} // namespace

TYPED_TEST_SUITE(
    ArrayMaxTest,
    FunctionBaseTest::ScalarTypes,
    FunctionBaseTest::TypeNames);

TYPED_TEST(ArrayMaxTest, arrayMaxNullable) {
  this->testMaxNullableExpr();
}

TYPED_TEST(ArrayMaxTest, arrayMax) {
  this->testMaxExpr();
}

// Test non-inlined (> 12 length) nullable strings.
using LongVarcharTest = ArrayMaxTest<VarcharType>;
TEST_F(LongVarcharTest, arrayMaxNullableLongVarchar) {
  using S = StringView;
  auto input = makeNullableArrayVector<S>({
      {S("red shiny car ahead"), S("blue clear sky above")},
      {std::nullopt,
       S("blue clear sky above"),
       S("yellow rose flowers"),
       S("orange beautiful sunset")},
      {},
      {S("red shiny car ahead"),
       S("purple is an elegant color"),
       S("green plants make us happy")},
  });
  auto expected = makeNullableFlatVector<S>(
      {S("red shiny car ahead"),
       std::nullopt,
       std::nullopt,
       S("red shiny car ahead")});

  this->testMaxExpr(input, expected);
}

// Test non-inlined (> 12 length) strings.
TEST_F(LongVarcharTest, arrayMaxLongVarchar) {
  using S = StringView;
  auto input = makeNullableArrayVector<S>({
      {S("red shiny car ahead"), S("blue clear sky above")},
      {S("blue clear sky above"),
       S("yellow rose flowers"),
       S("orange beautiful sunset")},
      {},
      {S("red shiny car ahead"),
       S("purple is an elegant color"),
       S("green plants make us happy")},
  });
  ;
  auto expected = makeNullableFlatVector<S>(
      {S("red shiny car ahead"),
       S("yellow rose flowers"),
       std::nullopt,
       S("red shiny car ahead")});
  this->testMaxExpr(input, expected);
}

// Test documented example.
using DocExample = ArrayMaxTest<IntegerType>;
TEST_F(DocExample, arrayMaxDocExample) {
  auto input = makeNullableArrayVector<T>(
      {{1, 2, 3}, {-1, -2, -2}, {-1, -2, std::nullopt}, {}});
  auto expected =
      makeNullableFlatVector<T>({3, -1, std::nullopt, std::nullopt});
  this->testMaxExpr(input, expected);
}