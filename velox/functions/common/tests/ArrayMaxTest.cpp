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
#include "velox/functions/common/tests/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;

namespace {

template <typename Type>
class ArrayMaxTest : public FunctionBaseTest {
 public:
  using T = typename Type::NativeType::NativeType;

  void setInputExpectedArrays(VectorPtr input, VectorPtr expected) {
    input_ = input;
    expected_ = expected;
  }

  void initDocData() {
    input_ = makeNullableArrayVector<T>(
        {{1, 2, 3}, {-1, -2, -2}, {-1, -2, std::nullopt}, {}});
    expected_ = makeNullableFlatVector<T>({3, -1, std::nullopt, std::nullopt});
  }

  void initDefaultNullableInputArray() {
    input_ = makeSample1NullableArrayVector<T>();
  }

  void initDefaultInputArray() {
    input_ = makeSample1ArrayVector<T>();
  }

  void initDefaultNullableExpectedArray() {}

  void initDefaultExpectedArray() {}

  void testMaxExpr() {
    auto result =
        evaluate<SimpleVector<T>>("array_max(C0)", makeRowVector({input_}));
    assertEqualVectors(expected_, result);
  }

 private:
  VectorPtr input_ = nullptr;
  VectorPtr expected_ = nullptr;
};

template <>
void ArrayMaxTest<SmallintType>::initDefaultNullableExpectedArray() {
  expected_ = makeNullableFlatVector<T>(
      {4, 4, -1, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

template <>
void ArrayMaxTest<SmallintType>::initDefaultExpectedArray() {
  expected_ = makeNullableFlatVector<T>({4, 4, -1, 105, std::nullopt});
}

template <>
void ArrayMaxTest<TinyintType>::initDefaultNullableExpectedArray() {
  expected_ = makeNullableFlatVector<T>(
      {4, 4, -1, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

template <>
void ArrayMaxTest<TinyintType>::initDefaultExpectedArray() {
  expected_ = makeNullableFlatVector<T>({4, 4, -1, 105, std::nullopt});
}

template <>
void ArrayMaxTest<IntegerType>::initDefaultNullableExpectedArray() {
  expected_ = makeNullableFlatVector<T>(
      {4, 4, -1, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

template <>
void ArrayMaxTest<IntegerType>::initDefaultExpectedArray() {
  expected_ = makeNullableFlatVector<T>({4, 4, -1, 105, std::nullopt});
}

template <>
void ArrayMaxTest<BigintType>::initDefaultNullableExpectedArray() {
  expected_ = makeNullableFlatVector<T>(
      {4, 4, -1, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

template <>
void ArrayMaxTest<BigintType>::initDefaultExpectedArray() {
  expected_ = makeNullableFlatVector<T>({4, 4, -1, 105, std::nullopt});
}

template <>
void ArrayMaxTest<BooleanType>::initDefaultNullableExpectedArray() {
  expected_ = makeNullableFlatVector<T>(
      {true,
       true,
       false,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       false,
       true});
}

template <>
void ArrayMaxTest<BooleanType>::initDefaultExpectedArray() {
  expected_ =
      makeNullableFlatVector<T>({true, true, false, std::nullopt, false, true});
}

template <>
void ArrayMaxTest<VarcharType>::initDefaultNullableExpectedArray() {
  using S = StringView;
  expected_ = makeNullableFlatVector<StringView>(
      {S("red"), std::nullopt, std::nullopt, S("red")});
}

template <>
void ArrayMaxTest<VarcharType>::initDefaultExpectedArray() {
  using S = StringView;
  expected_ = makeNullableFlatVector<StringView>(
      {S("red"), S("yellow"), std::nullopt, S("red")});
}

template <>
void ArrayMaxTest<RealType>::initDefaultNullableExpectedArray() {
  expected_ = makeNullableFlatVector<float>(
      {0.00001, std::nullopt, -0.0001, std::nullopt, std::nullopt, 0.0001});
}

template <>
void ArrayMaxTest<RealType>::initDefaultExpectedArray() {
  expected_ = makeNullableFlatVector<float>(
      {0.00001, 1.11, -0.0001, std::nullopt, 1.33, 0.0001});
}

template <>
void ArrayMaxTest<DoubleType>::initDefaultNullableExpectedArray() {
  expected_ = makeNullableFlatVector<double>(
      {0.00001, std::nullopt, -0.0001, std::nullopt, std::nullopt, 0.0001});
}

template <>
void ArrayMaxTest<DoubleType>::initDefaultExpectedArray() {
  expected_ = makeNullableFlatVector<double>(
      {0.00001, 1.11, -0.0001, std::nullopt, 1.33, 0.0001});
}

} // namespace

using Types = ::testing::Types<
    TinyintType,
    SmallintType,
    IntegerType,
    BigintType,
    VarcharType,
    BooleanType,
    DoubleType,
    RealType>;
TYPED_TEST_SUITE(ArrayMaxTest, Types, FunctionBaseTest::TypeNames);

// test nullable.
TYPED_TEST(ArrayMaxTest, ArrayMaxNullable) {
  this->initDefaultNullableInputArray();
  this->initDefaultNullableExpectedArray();
  this->testMaxExpr();
}

// test non-nullable.
TYPED_TEST(ArrayMaxTest, ArrayMax) {
  this->initDefaultInputArray();
  this->initDefaultExpectedArray();
  this->testMaxExpr();
}

// Test non-inlined (> 12 length) strings.
using longVarcharTest = ArrayMaxTest<VarcharType>;
TEST_F(longVarcharTest, ArrayMaxLongVarchar) {
  using S = StringView;
  // test nullable.
  auto expectedNullable = makeNullableFlatVector<StringView>(
      {S("red shiny car ahead"),
       std::nullopt,
       std::nullopt,
       S("red shiny car ahead")});
  this->setInputExpectedArrays(
      makeSample1NullableLongVarcharArrayVector(), expectedNullable);
  this->testMaxExpr();
  // test non-nullable.
  auto expected = makeNullableFlatVector<StringView>(
      {S("red shiny car ahead"),
       S("yellow rose flowers"),
       std::nullopt,
       S("red shiny car ahead")});
  this->setInputExpectedArrays(makeSample1LongVarcharArrayVector(), expected);
  this->testMaxExpr();
}

// Test documented example.
using docExample = ArrayMaxTest<IntegerType>;
TEST_F(docExample, ArrayMaxDocExample) {
  this->initDocData();
  this->testMaxExpr();
}