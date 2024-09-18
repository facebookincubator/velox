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
#include "velox/functions/sparksql/specialforms/AtLeastNNonNulls.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {
using namespace facebook::velox::test;

static constexpr auto kNaNDouble = std::numeric_limits<double>::quiet_NaN();
static constexpr auto kNaNFloat = std::numeric_limits<float>::quiet_NaN();
static constexpr auto kMaxDouble = std::numeric_limits<double>::max();
static constexpr auto kMaxFloat = std::numeric_limits<float>::max();

class AtLeastNNonNullsTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  core::CallTypedExprPtr createAtLeastNNonNulls(
      const std::vector<VectorPtr>& data) {
    std::vector<core::TypedExprPtr> inputs;

    for (int i = 0; i < data.size(); ++i) {
      if (data[i]->isConstantEncoding()) {
        auto constVector = data[i]->asUnchecked<ConstantVector<T>>();
        if (constVector->isNullAt(0)) {
          inputs.emplace_back(std::make_shared<core::ConstantTypedExpr>(
              data[i]->type(), variant(data[i]->type()->kind())));
        } else {
          inputs.emplace_back(std::make_shared<core::ConstantTypedExpr>(
              data[i]->type(), variant(constVector->valueAt(0))));
        }
      } else {
        inputs.emplace_back(std::make_shared<core::FieldAccessTypedExpr>(
            data[i]->type(), fmt::format("c{}", i)));
      }
    }
    return std::make_shared<const core::CallTypedExpr>(
        BOOLEAN(),
        std::move(inputs),
        AtLeastNNonNullsCallToSpecialForm::kAtLeastNNonNulls);
  }

  template <typename T>
  VectorPtr atLeastNNonNulls(const std::vector<VectorPtr>& input) {
    return evaluate(createAtLeastNNonNulls<T>(input), makeRowVector(input));
  }

  void testAtLeastNNonNulls(
      int32_t n,
      const std::vector<VectorPtr>& input,
      const VectorPtr& expected) {
    std::vector<VectorPtr> data;
    data.emplace_back(makeConstant<int32_t>(n, input[0]->size()));
    for (auto i = 0; i < input.size(); ++i) {
      data.emplace_back(input[i]);
    }
    const auto result = atLeastNNonNulls<int32_t>(data);
    assertEqualVectors(expected, result);
  }
};

TEST_F(AtLeastNNonNullsTest, basic) {
  auto stringInput = makeNullableFlatVector<StringView>(
      {std::nullopt, "1", "", std::nullopt, ""});
  auto boolInput = makeNullableFlatVector<bool>(
      {std::nullopt, true, false, std::nullopt, std::nullopt});
  auto intInput =
      makeNullableFlatVector<int32_t>({-1, 0, 1, std::nullopt, std::nullopt});
  auto floatInput = makeNullableFlatVector<float>(
      {kMaxFloat, kNaNFloat, 0.1f, 0.0f, std::nullopt});
  auto doubleInput = makeNullableFlatVector<double>(
      {std::log(-2.0), kMaxDouble, kNaNDouble, std::nullopt, 0.1});
  auto arrayInput = makeArrayVectorFromJson<int32_t>(
      {"[1, null, 3]", "[1, 2, 3]", "null", "[null]", "[]"});
  auto mapInput = makeMapVectorFromJson<int32_t, int32_t>(
      {"{1: 10, 2: null, 3: null}", "{1: 10, 2: 20}", "{1: 2}", "{}", "null"});
  auto constInput = makeConstant<int32_t>(2, 5);
  auto indices = makeIndices(5, [](auto row) { return (row + 1) % 5; });
  auto dictInput = wrapInDictionary(indices, 5, doubleInput);

  auto expected = makeFlatVector<bool>({false, true, true, false, false});
  testAtLeastNNonNulls(2, {stringInput, boolInput}, expected);

  expected = makeFlatVector<bool>({false, false, false, false, false});
  testAtLeastNNonNulls(3, {stringInput, boolInput}, expected);

  expected = makeFlatVector<bool>({true, true, true, true, true});
  testAtLeastNNonNulls(0, {stringInput, boolInput}, expected);
  testAtLeastNNonNulls(-1, {stringInput, boolInput}, expected);

  expected = makeFlatVector<bool>({true, false, true, true, false});
  testAtLeastNNonNulls(1, {floatInput}, expected);

  expected = makeFlatVector<bool>({false, true, false, false, true});
  testAtLeastNNonNulls(1, {doubleInput}, expected);

  expected = makeFlatVector<bool>({false, true, true, false, false});
  testAtLeastNNonNulls(2, {stringInput, boolInput, floatInput}, expected);

  expected = makeFlatVector<bool>({false, true, true, false, false});
  testAtLeastNNonNulls(
      3, {boolInput, intInput, floatInput, doubleInput}, expected);

  expected = makeFlatVector<bool>({false, false, false, false, false});
  testAtLeastNNonNulls(2, {floatInput, doubleInput}, expected);

  expected = makeFlatVector<bool>({true, false, false, true, false});
  testAtLeastNNonNulls(
      4, {mapInput, arrayInput, constInput, dictInput}, expected);
}

TEST_F(AtLeastNNonNullsTest, error) {
  auto input =
      makeNullableFlatVector<int32_t>({-1, 0, 1, std::nullopt, std::nullopt});

  VELOX_ASSERT_USER_THROW(
      atLeastNNonNulls<float>({makeConstant<float>(1.0f, 5), input}),
      "The first input type should be INTEGER but got REAL");
  VELOX_ASSERT_USER_THROW(
      atLeastNNonNulls<int32_t>({makeConstant<int32_t>(1, 5)}),
      "AtLeastNNonNulls expects to receive at least 2 arguments");
  VELOX_ASSERT_USER_THROW(
      atLeastNNonNulls<int32_t>({input, input}),
      "The first parameter should be constant expression");
  VELOX_ASSERT_USER_THROW(
      atLeastNNonNulls<int32_t>(
          {makeConstant<int32_t>(std::nullopt, 5), input}),
      "The first parameter should not be null");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
