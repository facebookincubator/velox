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
#include <gtest/gtest.h>

#include "functions/sparksql/specialforms/GetStructField.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class GetStructFieldTest : public SparkFunctionBaseTest {
 protected:
  void GetStructFieldSimple(
      const VectorPtr& parameter,
      const VectorPtr& index,
      const TypePtr& inputType,
      const TypePtr& resultType,
      const VectorPtr& expected = nullptr) {
    core::TypedExprPtr data =
        std::make_shared<const core::FieldAccessTypedExpr>(inputType, "c0");
    core::TypedExprPtr ordinal =
        std::make_shared<const core::FieldAccessTypedExpr>(INTEGER(), "c1");
    auto getStructFieldExpr = std::make_shared<const core::CallTypedExpr>(
        resultType,
        std::vector<core::TypedExprPtr>{data, ordinal},
        "get_struct_field");
    auto result = evaluate(
        getStructFieldExpr, makeRowVector({"c0", "c1"}, {parameter, index}));
    if (expected) {
      ::facebook::velox::test::assertEqualVectors(expected, result);
    }
  }
};

TEST_F(GetStructFieldTest, simpleInteger) {
  auto dataType = ROW({"k1", "k2", "a"}, {BIGINT(), BIGINT(), BIGINT()});
  auto data = makeRowVector(
      {"k1", "k2", "a"},
      {makeNullableFlatVector<int32_t>({12}),
       makeNullableFlatVector<int32_t>({2}),
       makeNullableFlatVector<int32_t>({1})});
  auto result = makeNullableFlatVector<int32_t>({2});
  auto index = makeConstant<int32_t>(1, 1);
  GetStructFieldSimple(data, index, dataType, INTEGER(), result);
}

TEST_F(GetStructFieldTest, simpleVarchar) {
  auto dataType = ROW({"k1", "k2", "a"}, {BIGINT(), VARCHAR(), BIGINT()});
  auto data = makeRowVector(
      {"k1", "k2", "a"},
      {makeNullableFlatVector<int32_t>({12}),
       makeNullableFlatVector<std::string>({"Milly"}),
       makeNullableFlatVector<int32_t>({1})});
  auto result = makeNullableFlatVector<std::string>({"Milly"});
  auto index = makeConstant<int32_t>(1, 1);
  GetStructFieldSimple(data, index, dataType, VARCHAR(), result);
}

TEST_F(GetStructFieldTest, simpleNull) {
  auto dataType = ROW({"k1", "k2", "a"}, {BIGINT(), BIGINT(), BIGINT()});
  auto data = makeRowVector(
      {"k1", "k2", "a"},
      {makeNullableFlatVector<int32_t>({12}),
       makeNullableFlatVector<int32_t>({std::nullopt}),
       makeNullableFlatVector<int32_t>({1})});
  auto result = makeNullableFlatVector<int32_t>({std::nullopt});
  auto index = makeConstant<int32_t>(1, 1);
  GetStructFieldSimple(data, index, dataType, INTEGER(), result);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
