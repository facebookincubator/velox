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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

using core::QueryConfig;
using facebook::velox::test::BatchMaker;
using namespace common::testutil;

class ToCudfSelectionTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer fuzzer({.vectorSize = size}, pool());
    for (int32_t i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }
    return vectors;
  }

  bool wasCudfAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType.starts_with("CudfAggregation")) {
          return true;
        }
      }
    }
    return false;
  }

  bool wasDefaultHashAggregationUsed(const std::shared_ptr<exec::Task>& task) {
    auto stats = task->taskStats();
    for (const auto& pipelineStats : stats.pipelineStats) {
      for (const auto& operatorStats : pipelineStats.operatorStats) {
        if (operatorStats.operatorType == OperatorType::kAggregation) {
          return true;
        }
      }
    }
    return false;
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           DOUBLE(),
           DOUBLE(),
           VARCHAR()})};
};

// Test supported aggregation should use CUDF
TEST_F(ToCudfSelectionTest, supportedAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "count(c2)", "min(c3)", "max(c4)", "avg(c5)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, sum(c1), count(c2), min(c3), max(c4), avg(c5) FROM tmp GROUP BY c0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation should fall back to CPU
TEST_F(ToCudfSelectionTest, unsupportedAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"stddev(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0, stddev(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test mixed supported/unsupported should fall back
TEST_F(ToCudfSelectionTest, mixedSupportFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "stddev(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), stddev(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported global aggregation should use CUDF
TEST_F(ToCudfSelectionTest, supportedGlobalAggregationUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"sum(c1)", "count(c2)", "max(c3)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT sum(c1), count(c2), max(c3) FROM tmp");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported global aggregation should fall back
TEST_F(ToCudfSelectionTest, unsupportedGlobalAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"stddev(c1)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT stddev(c1) FROM tmp");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported grouping key expressions should use CUDF
TEST_F(ToCudfSelectionTest, supportedGroupingKeyExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .project(
                      {"c0",
                       "c1",
                       "c2",
                       "c3",
                       "c4",
                       "c5",
                       "c6",
                       "c0 + c1 as key1",
                       "length(c6) as key2"})
                  .aggregation(
                      {"key1", "key2"},
                      {"sum(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults(
              "SELECT c0 + c1, length(c6), sum(c2) FROM tmp GROUP BY c0 + c1, length(c6)");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation functions should fall back
TEST_F(ToCudfSelectionTest, unsupportedAggregationFunctionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "stddev(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), stddev(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test complex grouping key expressions should fall back
TEST_F(ToCudfSelectionTest, complexGroupingKeyExpressionsFallsBack) {
  // Use a deterministic input vector.
  // Input: c0=[1,2,1,2], c2=[100,200,300,400]
  // After grouping by to_big_endian_64(c0):
  // - Group 1 (c0=1): sum(c2) = 100 + 300 = 400
  // - Group 2 (c0=2): sum(c2) = 200 + 400 = 600
  auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 1, 2}), // c0
      makeFlatVector<int16_t>({10, 20, 30, 40}), // c1
      makeFlatVector<int32_t>({100, 200, 300, 400}), // c2
  });

  auto plan =
      PlanBuilder()
          .values({input})
          .project({"c0", "c1", "c2", "to_big_endian_64(c0) AS complex_key"})
          .aggregation(
              {"complex_key"},
              {"sum(c2)"},
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode();

  std::shared_ptr<Task> task;
  AssertQueryBuilder(plan).config("cudf.enabled", true).countResults(task);

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test supported aggregation input expressions should use CUDF
TEST_F(ToCudfSelectionTest, supportedAggregationInputExpressionsUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "max(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", true)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), max(c2) FROM tmp GROUP BY c0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test unsupported aggregation input expressions should fall back
TEST_F(ToCudfSelectionTest, unsupportedAggregationInputExpressionsFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "variance(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults(
                      "SELECT c0, sum(c1), variance(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Non-count constant aggregates are not supported by cuDF aggregation.
TEST_F(ToCudfSelectionTest, nonCountConstantAggregationFallsBack) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan =
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {}, {"sum(1)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT sum(1) FROM tmp");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

// Test zero-column count(*) should stay on GPU.
TEST_F(ToCudfSelectionTest, zeroColumnCountStarUsesCudf) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .filter("c0 > 0")
          .project({})
          .aggregation(
              {}, {"count(*)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test zero-column count(constant) should use cudf GPU.
TEST_F(ToCudfSelectionTest, zeroColumnCountConstantUsesGpu) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan =
      PlanBuilder()
          .values({data})
          .filter("c0 > 0")
          .project({})
          .aggregation(
              {}, {"count(1)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT count(1) FROM tmp WHERE c0 > 0");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Count-only aggregation plans (single and partial/final) should use
// CudfAggregation.
TEST_F(ToCudfSelectionTest, countAggregatesOnlyUsesCudf) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto assertCountUsesCudf = [this](
                                 const core::PlanNodePtr& plan,
                                 const std::string& duckSql,
                                 const char* caseLabel) {
    SCOPED_TRACE(caseLabel);
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config("cudf.enabled", true)
                    .plan(plan)
                    .assertResults(duckSql);
    ASSERT_TRUE(wasCudfAggregationUsed(task));
    ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
  };

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {}, {"count(*)"}, {}, core::AggregationNode::Step::kSingle, false)
          .planNode(),
      "SELECT count(*) FROM tmp",
      "global count(*) kSingle");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {},
              {"count(c0)"},
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode(),
      "SELECT count(c0) FROM tmp",
      "global count(c0) kSingle");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .aggregation(
              {"c0"},
              {"count(c2)"},
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode(),
      "SELECT c0, count(c2) FROM tmp GROUP BY c0",
      "group by c0, count(c2) kSingle");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .partialAggregation({}, {"count(*)"})
          .finalAggregation()
          .planNode(),
      "SELECT count(*) FROM tmp",
      "global count(*) partial+final");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .partialAggregation({"c0"}, {"count(c2)"})
          .finalAggregation()
          .planNode(),
      "SELECT c0, count(c2) FROM tmp GROUP BY c0",
      "group by c0, count(c2) partial+final");

  assertCountUsesCudf(
      PlanBuilder()
          .values(vectors)
          .partialAggregation({"c0"}, {"count(*)"})
          .finalAggregation()
          .planNode(),
      "SELECT c0, count(*) FROM tmp GROUP BY c0",
      "group by c0, count(*) partial+final");
}

// Test count(NULL) runs on cudf GPU (returns 0).
TEST_F(ToCudfSelectionTest, countNullAggregationUsesGpu) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {},
                      {"count(null)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config("cudf.enabled", true)
                  .plan(plan)
                  .assertResults("SELECT count(NULL) FROM tmp");

  ASSERT_TRUE(wasCudfAggregationUsed(task));
  ASSERT_FALSE(wasDefaultHashAggregationUsed(task));
}

// Test CUDF disabled should always use regular aggregation
TEST_F(ToCudfSelectionTest, cudfDisabledUsesRegularAggregation) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  auto plan = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0"},
                      {"sum(c1)", "count(c2)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config("cudf.enabled", false)
          .plan(plan)
          .assertResults("SELECT c0, sum(c1), count(c2) FROM tmp GROUP BY c0");

  ASSERT_FALSE(wasCudfAggregationUsed(task));
  ASSERT_TRUE(wasDefaultHashAggregationUsed(task));
}

} // namespace facebook::velox::exec::test
