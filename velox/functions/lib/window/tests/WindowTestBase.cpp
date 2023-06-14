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
#include "velox/functions/lib/window/tests/WindowTestBase.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::window::test {

WindowTestBase::QueryInfo WindowTestBase::buildWindowQuery(
    const std::vector<RowVectorPtr>& input,
    const std::string& function,
    const std::string& overClause,
    const std::string& frameClause) {
  std::string functionSql =
      fmt::format("{} over ({} {})", function, overClause, frameClause);
  auto op = PlanBuilder()
                .setParseOptions(options_)
                .values(input)
                .window({functionSql})
                .planNode();

  auto rowType = asRowType(input[0]->type());
  std::string columnsString = folly::join(", ", rowType->names());
  std::string querySql =
      fmt::format("SELECT {}, {} FROM tmp", columnsString, functionSql);

  return {op, functionSql, querySql};
}

RowVectorPtr WindowTestBase::makeSimpleVector(vector_size_t size) {
  return makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
      makeFlatVector<int32_t>(
          size, [](auto row) { return row % 7; }, nullEvery(11)),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 6 + 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 4 + 1; }),
  });
}

RowVectorPtr WindowTestBase::makeSinglePartitionVector(vector_size_t size) {
  return makeRowVector({
      makeFlatVector<int32_t>(size, [](auto /* row */) { return 1; }),
      makeFlatVector<int32_t>(
          size, [](auto row) { return row; }, nullEvery(7)),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 6 + 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 4 + 1; }),
  });
}

RowVectorPtr WindowTestBase::makeSingleRowPartitionsVector(vector_size_t size) {
  return makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      makeFlatVector<int64_t>(size, [](auto row) { return row % 6 + 1; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 4 + 1; }),
  });
}

VectorPtr WindowTestBase::makeRandomInputVector(
    const TypePtr& type,
    vector_size_t size,
    float nullRatio) {
  VectorFuzzer::Options options;
  options.vectorSize = size;
  options.nullRatio = nullRatio;
  options.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMicroSeconds;
  VectorFuzzer fuzzer(options, pool_.get(), 0);
  return fuzzer.fuzzFlat(type);
}

RowVectorPtr WindowTestBase::makeRandomInputVector(vector_size_t size) {
  boost::random::mt19937 gen;
  // Frame index values require integer values > 0.
  auto genRandomFrameValue = [&](vector_size_t /*row*/) {
    return boost::random::uniform_int_distribution<int>(1)(gen);
  };
  return makeRowVector(
      {makeRandomInputVector(BIGINT(), size, 0.2),
       makeRandomInputVector(VARCHAR(), size, 0.3),
       makeFlatVector<int64_t>(size, genRandomFrameValue),
       makeFlatVector<int64_t>(size, genRandomFrameValue)});
}

void WindowTestBase::testWindowFunction(
    const std::vector<RowVectorPtr>& input,
    const std::string& function,
    const std::vector<std::string>& overClauses,
    const std::vector<std::string>& frameClauses,
    bool createTable) {
  if (createTable) {
    createDuckDbTable(input);
  }
  for (const auto& overClause : overClauses) {
    for (auto& frameClause : frameClauses) {
      auto queryInfo =
          buildWindowQuery(input, function, overClause, frameClause);
      SCOPED_TRACE(queryInfo.functionSql);
      assertQuery(queryInfo.planNode, queryInfo.querySql);
    }
  }
}

void WindowTestBase::testKRangeFrames(const std::string& function) {
  vector_size_t size = 20;

  auto rangeFrameTest = [&](const VectorPtr& startColumn,
                            const VectorPtr& endColumn,
                            const std::string& overClause,
                            const std::string& veloxFrame,
                            const std::string& duckFrame) {
    auto vectors = makeRowVector({
        makeFlatVector<int32_t>(size, [](auto row) { return row % 5; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row; }),
        startColumn,
        endColumn,
    });
    createDuckDbTable({vectors});

    std::string veloxFunction =
        fmt::format("{} over ({} {})", function, overClause, veloxFrame);
    std::string duckFunction =
        fmt::format("{} over ({} {})", function, overClause, duckFrame);
    auto op = PlanBuilder()
                  .setParseOptions(options_)
                  .values({vectors})
                  .window({veloxFunction})
                  .planNode();

    auto rowType = asRowType(vectors->type());
    std::string columnsString = folly::join(", ", rowType->names());
    std::string querySql =
        fmt::format("SELECT {}, {} FROM tmp", columnsString, duckFunction);
    SCOPED_TRACE(veloxFunction);
    assertQuery(op, querySql);
  };

  // For frames with k RANGE PRECEDING/FOLLOWING, columns with the range values
  // computed according to the frame type are added to the plan by Presto and
  // Spark. The Velox Window operator also requires these columns to be computed
  // and setup accordingly.
  std::string overClause = "partition by c0 order by c1";
  auto startColumn =
      makeFlatVector<int64_t>(size, [](auto row) { return row - 4; });
  auto endColumn =
      makeFlatVector<int64_t>(size, [](auto row) { return row + 2; });
  std::string veloxFrame = "range between c2 preceding and c3 following";
  std::string duckFrame = "range between 4 preceding and 2 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  duckFrame = "range between 4 preceding and current row";
  veloxFrame = "range between c2 preceding and current row";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  // There are no rows between 2 preceding and 4 preceding frames. So this tests
  // empty frames.
  endColumn = makeFlatVector<int64_t>(size, [](auto row) { return row - 2; });
  duckFrame = "range between 4 preceding and 2 preceding";
  veloxFrame = "range between c2 preceding and c3 preceding";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  // There is exactly one row in the frames between 2 and 6 preceding values.
  startColumn = makeFlatVector<int64_t>(size, [](auto row) { return row - 6; });
  duckFrame = "range between 6 preceding and 2 preceding";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  // There are no rows between 2 and 4 following frames. So this tests empty
  // frames.
  startColumn = makeFlatVector<int64_t>(size, [](auto row) { return row + 2; });
  endColumn = makeFlatVector<int64_t>(size, [](auto row) { return row + 4; });
  duckFrame = "range between 2 following and 4 following";
  veloxFrame = "range between c2 following and c3 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  duckFrame = "range between current row and 4 following";
  veloxFrame = "range between current row and c3 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  // There is exactly one row between 2 and 6 following frame values.
  endColumn = makeFlatVector<int64_t>(size, [](auto row) { return row + 6; });
  duckFrame = "range between 2 following and 6 following";
  veloxFrame = "range between c2 following and c3 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  // The below tests are for a descending order by column. For such cases,
  // preceding rows have greater values than the current row (and following has
  // smaller values).
  overClause = "partition by c0 order by c1 desc";
  startColumn = makeFlatVector<int64_t>(size, [](auto row) { return row + 4; });
  endColumn = makeFlatVector<int64_t>(size, [](auto row) { return row - 2; });
  veloxFrame = "range between c2 preceding and c3 following";
  duckFrame = "range between 4 preceding and 2 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  duckFrame = "range between 4 preceding and current row";
  veloxFrame = "range between c2 preceding and current row";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  endColumn = makeFlatVector<int64_t>(size, [](auto row) { return row + 2; });
  veloxFrame = "range between c2 preceding and c3 preceding";
  duckFrame = "range between 4 preceding and 2 preceding";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  startColumn = makeFlatVector<int64_t>(size, [](auto row) { return row + 6; });
  duckFrame = "range between 6 preceding and 2 preceding";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  startColumn = makeFlatVector<int64_t>(size, [](auto row) { return row - 2; });
  endColumn = makeFlatVector<int64_t>(size, [](auto row) { return row - 4; });
  veloxFrame = "range between c2 following and c3 following";
  duckFrame = "range between 2 following and 4 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  veloxFrame = "range between current row and c3 following";
  duckFrame = "range between current row and 4 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);

  endColumn = makeFlatVector<int64_t>(size, [](auto row) { return row - 6; });
  veloxFrame = "range between c2 following and c3 following";
  duckFrame = "range between 2 following and 6 following";
  rangeFrameTest(startColumn, endColumn, overClause, veloxFrame, duckFrame);
}

void WindowTestBase::assertWindowFunctionError(
    const std::vector<RowVectorPtr>& input,
    const std::string& function,
    const std::string& overClause,
    const std::string& errorMessage) {
  assertWindowFunctionError(input, function, overClause, "", errorMessage);
}

void WindowTestBase::assertWindowFunctionError(
    const std::vector<RowVectorPtr>& input,
    const std::string& function,
    const std::string& overClause,
    const std::string& frameClause,
    const std::string& errorMessage) {
  auto queryInfo = buildWindowQuery(input, function, overClause, frameClause);
  SCOPED_TRACE(queryInfo.functionSql);

  VELOX_ASSERT_THROW(
      assertQuery(queryInfo.planNode, queryInfo.querySql), errorMessage);
}

}; // namespace facebook::velox::window::test
