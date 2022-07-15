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
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/WindowFunction.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/VectorTestBase.h"

namespace facebook::velox::exec::test {

class PlanBuilderTest : public testing::Test,
                        public velox::test::VectorTestBase {
 public:
  PlanBuilderTest() {
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
  }
};

TEST_F(PlanBuilderTest, duplicateSubfield) {
  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(
              ROW({"a", "b"}, {BIGINT(), BIGINT()}),
              {"a < 5", "b = 7", "a > 0"},
              "a + b < 100")
          .planNode(),
      "Duplicate subfield: a");
}

TEST_F(PlanBuilderTest, invalidScalarFunctionCall) {
  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(ROW({"a", "b"}, {BIGINT(), BIGINT()}))
          .project({"to_unixtime(a)"})
          .planNode(),
      "Scalar function signature is not supported: to_unixtime(BIGINT).");

  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(ROW({"a", "b"}, {BIGINT(), BIGINT()}))
          .project({"to_unitime(a)"})
          .planNode(),
      "Scalar function doesn't exist: to_unitime.");
}

TEST_F(PlanBuilderTest, invalidAggregateFunctionCall) {
  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(ROW({"a", "b"}, {VARCHAR(), BIGINT()}))
          .partialAggregation({}, {"sum(a)"})
          .planNode(),
      "Aggregate function signature is not supported: sum(VARCHAR).");

  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(ROW({"a", "b"}, {VARCHAR(), BIGINT()}))
          .partialAggregation({}, {"maxx(a)"})
          .planNode(),
      "Aggregate function doesn't exist: maxx.");
}

TEST_F(PlanBuilderTest, windowFunctionCall) {
  PlanBuilder::WindowFrame frame = {
      core::WindowNode::WindowType::kRange,
      core::WindowNode::BoundType::kPreceding,
      "10",
      core::WindowNode::BoundType::kFollowing,
      "b"};
  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(ROW({"a", "b", "c"}, {VARCHAR(), BIGINT(), BIGINT()}))
          .window({"a"}, {"b"}, {"window1(c) AS d"}, {frame})
          .planNode(),
      "Registry of window functions is empty.");

  std::vector<exec::FunctionSignaturePtr> signatures{
      exec::FunctionSignatureBuilder()
          .argumentType("BIGINT")
          .returnType("BIGINT")
          .build(),
  };
  exec::registerWindowFunction("window1", std::move(signatures), nullptr);

  VELOX_CHECK_NOT_NULL(
      PlanBuilder()
          .tableScan(ROW({"a", "b", "c"}, {VARCHAR(), BIGINT(), BIGINT()}))
          .window({"a"}, {"b"}, {"window1(c) AS d"}, {frame})
          .planNode());

  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(ROW({"a", "b"}, {VARCHAR(), BIGINT()}))
          .window({}, {}, {"window1(a) AS c"}, {})
          .planNode(),
      "Window function signature is not supported: window1(VARCHAR).");

  VELOX_ASSERT_THROW(
      PlanBuilder()
          .tableScan(ROW({"a", "b"}, {VARCHAR(), BIGINT()}))
          .window({}, {}, {"window2(a) AS c"}, {})
          .planNode(),
      "Window function doesn't exist: window2.");
}
} // namespace facebook::velox::exec::test
