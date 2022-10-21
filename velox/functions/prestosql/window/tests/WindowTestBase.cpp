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
#include "velox/functions/prestosql/window/tests/WindowTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

namespace {
struct QueryInfo {
  const core::PlanNodePtr planNode;
  const std::string functionSql;
  const std::string querySql;
};

QueryInfo buildWindowQuery(
    const std::vector<RowVectorPtr>& input,
    const std::string& function,
    const std::string& overClause,
    const std::optional<std::string> frameClause = "") {
  auto functionSql = frameClause
      ? fmt::format(
            "{} over ({} {})", function, overClause, frameClause.value())
      : fmt::format("{} over ({})", function, overClause);

  auto op = PlanBuilder().values(input).window({functionSql}).planNode();

  auto rowType = asRowType(input[0]->type());
  std::string columnsString = folly::join(", ", rowType->names());
  std::string querySql =
      fmt::format("SELECT {}, {} FROM tmp", columnsString, functionSql);

  return {op, functionSql, querySql};
}
}; // namespace

std::vector<RowVectorPtr> WindowTestBase::makeVectors(
    const RowTypePtr& rowType,
    vector_size_t size,
    int numVectors,
    float nullRatio) {
  std::vector<RowVectorPtr> vectors;
  VectorFuzzer::Options options;
  options.vectorSize = size;
  options.nullRatio = nullRatio;
  options.useMicrosecondPrecisionTimestamp = true;
  VectorFuzzer fuzzer(options, pool_.get(), 0);
  for (int32_t i = 0; i < numVectors; ++i) {
    auto vector = std::dynamic_pointer_cast<RowVector>(fuzzer.fuzzRow(rowType));
    vectors.push_back(vector);
  }
  return vectors;
}

void WindowTestBase::testWindowFunction(
    const std::vector<RowVectorPtr>& input,
    const std::string& function,
    const std::string& overClause,
    const std::optional<std::string>& frameClause) {
  auto queryInfo = buildWindowQuery(input, function, overClause, frameClause);
  SCOPED_TRACE(queryInfo.functionSql);
  assertQuery(queryInfo.planNode, queryInfo.querySql);
}

void WindowTestBase::assertWindowFunctionError(
    const std::vector<RowVectorPtr>& input,
    const std::string& function,
    const std::string& overClause,
    const std::string& errorMessage,
    const std::optional<std::string>& frameClause) {
  auto queryInfo = buildWindowQuery(input, function, overClause, frameClause);
  SCOPED_TRACE(queryInfo.functionSql);

  VELOX_ASSERT_THROW(
      assertQuery(queryInfo.planNode, queryInfo.querySql), errorMessage);
}

void WindowTestBase::testTwoColumnOverClauses(
    const std::vector<RowVectorPtr>& input,
    const std::string& windowFunction,
    const std::string& overClause,
    const std::optional<std::string>& frameClause) {
  VELOX_CHECK_GE(input[0]->childrenSize(), 2);

  createDuckDbTable(input);
  if (frameClause) {
    testWindowFunction(input, windowFunction, overClause, frameClause);
  } else {
    testWindowFunction(input, windowFunction, overClause);
  }

  // Invoking with same vector set twice so that the underlying WindowFunction
  // receives the same data set multiple times and does a full processing
  // (partition, sort) + apply of it.
  std::vector<RowVectorPtr> doubleInput;
  doubleInput.insert(doubleInput.end(), input.begin(), input.end());
  doubleInput.insert(doubleInput.end(), input.begin(), input.end());
  createDuckDbTable(doubleInput);
  if (frameClause) {
    testWindowFunction(doubleInput, windowFunction, overClause, frameClause);
  } else {
    testWindowFunction(doubleInput, windowFunction, overClause);
  }
}

}; // namespace facebook::velox::window::test
