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

#include "velox/exec/fuzzer/PrestoQueryRunnerIntermediateTypeTransforms.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/JsonType.h"

namespace facebook::velox::exec::test {
namespace {
class PrestoQueryRunnerJsonTransformTest
    : public functions::test::FunctionBaseTest {
 public:
  void test(const VectorPtr& vector) {
    const auto colName = "col";
    const auto input =
        makeRowVector({colName}, {transformIntermediateOnlyType(vector)});

    auto expr = getIntermediateOnlyTypeProjectionExpr(
        vector->type(),
        std::make_shared<core::FieldAccessExpr>(
            colName,
            std::nullopt,
            std::vector<core::ExprPtr>{std::make_shared<core::InputExpr>()}),
        colName);

    core::PlanNodePtr plan =
        PlanBuilder().values({input}).projectExpressions({expr}).planNode();

    AssertQueryBuilder(plan).assertResults(makeRowVector({colName}, {vector}));
  }
};

TEST_F(PrestoQueryRunnerJsonTransformTest, verifyTransformExpression) {
  core::PlanNodePtr plan =
      PlanBuilder()
          .values({makeRowVector(
              {"c0"},
              {transformIntermediateOnlyType(makeNullableFlatVector(
                  std::vector<std::optional<StringView>>{}, JSON()))})})
          .projectExpressions({getIntermediateOnlyTypeProjectionExpr(
              JSON(),
              std::make_shared<core::FieldAccessExpr>(
                  "c0",
                  std::nullopt,
                  std::vector<core::ExprPtr>{
                      std::make_shared<core::InputExpr>()}),
              "c0")})
          .planNode();

  VELOX_CHECK_EQ(
      plan->toString(true, false),
      "-- Project[1][expressions: (p0:JSON, try(json_parse(ROW[\"c0\"])))] -> p0:JSON\n");
  AssertQueryBuilder(plan).assertTypeAndNumRows(JSON(), 0);
}

TEST_F(PrestoQueryRunnerJsonTransformTest, happyPath) {
  std::vector<std::optional<StringView>> no_nulls{"1", "2", "3"};
  test(makeNullableFlatVector(no_nulls, JSON()));

  std::vector<std::optional<StringView>> some_nulls{"1", std::nullopt, "3"};
  test(makeNullableFlatVector(some_nulls, JSON()));

  std::vector<std::optional<StringView>> all_nulls{
      std::nullopt, std::nullopt, std::nullopt};
  test(makeNullableFlatVector(all_nulls, JSON()));
}

TEST_F(PrestoQueryRunnerJsonTransformTest, isIntermediateOnlyType) {
  ASSERT_TRUE(isIntermediateOnlyType(JSON()));
  ASSERT_TRUE(isIntermediateOnlyType(ARRAY(JSON())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(JSON(), SMALLINT())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(VARBINARY(), JSON())));
  ASSERT_TRUE(isIntermediateOnlyType(ROW({JSON(), SMALLINT()})));
  ASSERT_TRUE(isIntermediateOnlyType(ROW({BOOLEAN(), ARRAY(JSON())})));
  ASSERT_TRUE(isIntermediateOnlyType(
      ROW({SMALLINT(), TIMESTAMP(), ARRAY(ROW({MAP(VARCHAR(), JSON())}))})));
}

} // namespace
} // namespace facebook::velox::exec::test
