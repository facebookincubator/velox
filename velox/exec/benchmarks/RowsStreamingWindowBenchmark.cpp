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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

constexpr int32_t kRowsPerVector = 10'000;
constexpr int32_t kOutputBatchRows = 1'000;

class RowsStreamingWindowBenchmark : public VectorTestBase {
 public:
  void addBenchmarks() {
    addBenchmarksForSize("10K", 1, 10'000);
    addBenchmarksForSize("100K", 10, 25'000);
    addBenchmarksForSize("1M", 100, 25'000);
  }

 private:
  struct TestCase {
    std::string name;
    std::vector<RowVectorPtr> data;
    core::PlanNodePtr plan;
    int64_t numRows;
  };

  std::vector<RowVectorPtr> makeData(
      int32_t numVectors,
      int32_t rowsPerPartition) {
    std::vector<RowVectorPtr> data;
    data.reserve(numVectors);
    const auto rowType = ROW({"p", "s", "v"}, {INTEGER(), INTEGER(), BIGINT()});
    for (auto vectorIndex = 0; vectorIndex < numVectors; ++vectorIndex) {
      data.push_back(makeRowVector(
          rowType->names(),
          {
              makeFlatVector<int32_t>(
                  kRowsPerVector,
                  [vectorIndex, rowsPerPartition](auto row) {
                    const auto globalRow = vectorIndex * kRowsPerVector + row;
                    return globalRow / rowsPerPartition;
                  }),
              makeFlatVector<int32_t>(
                  kRowsPerVector,
                  [vectorIndex, rowsPerPartition](auto row) {
                    const auto globalRow = vectorIndex * kRowsPerVector + row;
                    return globalRow % rowsPerPartition;
                  }),
              makeFlatVector<int64_t>(
                  kRowsPerVector, [](auto row) { return row % 97; }),
          }));
    }
    return data;
  }

  void addBenchmarksForSize(
      const std::string& sizeName,
      int32_t numVectors,
      int32_t rowsPerPartition) {
    auto data = makeData(numVectors, rowsPerPartition);
    addBenchmark(
        "rowsStreamingWindowRank_" + sizeName,
        data,
        {"rank() over (partition by p order by s)"});
    addBenchmark(
        "rowsStreamingWindowSum_" + sizeName,
        data,
        {"sum(v) over (partition by p order by s)"});
    addBenchmark(
        "rowsStreamingWindowRankAndSum_" + sizeName,
        data,
        {"rank() over (partition by p order by s)",
         "sum(v) over (partition by p order by s)"});
    addBenchmark(
        "rowsStreamingWindowSevenFunctions_" + sizeName,
        data,
        {"rank() over (partition by p order by s)",
         "dense_rank() over (partition by p order by s)",
         "row_number() over (partition by p order by s)",
         "sum(v) over (partition by p order by s)",
         "count(v) over (partition by p order by s)",
         "min(v) over (partition by p order by s)",
         "max(v) over (partition by p order by s)"});
  }

  void addBenchmark(
      const std::string& name,
      const std::vector<RowVectorPtr>& data,
      const std::vector<std::string>& windowFunctions) {
    auto testCase = std::make_unique<TestCase>();
    testCase->name = name;
    testCase->data = data;
    testCase->numRows = data.size() * kRowsPerVector;
    testCase->plan = exec::test::PlanBuilder()
                         .values(testCase->data)
                         .streamingWindow(windowFunctions)
                         .planNode();

    const auto* test = testCase.get();
    folly::addBenchmark(
        __FILE__,
        name,
        [this, test](folly::UserCounters& counters, unsigned iterations) {
          CpuWallTiming windowTiming;
          uint64_t numResultRows = 0;
          for (auto i = 0; i < iterations; ++i) {
            numResultRows += runOnce(*test, windowTiming);
          }

          BENCHMARK_SUSPEND {
            counters["rows"] = folly::UserMetric(
                static_cast<int64_t>(test->numRows),
                folly::UserMetric::Type::METRIC);
            counters["windowCpu"] = folly::UserMetric(
                static_cast<double>(windowTiming.cpuNanos) / iterations / 1e9,
                folly::UserMetric::Type::TIME);
            counters["windowWall"] = folly::UserMetric(
                static_cast<double>(windowTiming.wallNanos) / iterations / 1e9,
                folly::UserMetric::Type::TIME);
            folly::doNotOptimizeAway(numResultRows);
          }
          return iterations;
        });

    cases_.push_back(std::move(testCase));
  }

  uint64_t runOnce(const TestCase& test, CpuWallTiming& windowTiming) const {
    std::unique_ptr<TaskCursor> cursor;
    BENCHMARK_SUSPEND {
      CursorParameters params;
      params.planNode = test.plan;
      params.serialExecution = true;
      params.queryConfigs = {
          {core::QueryConfig::kPreferredOutputBatchRows,
           std::to_string(kOutputBatchRows)}};
      cursor = TaskCursor::create(params);
    }

    uint64_t numResultRows = 0;
    while (cursor->moveNext()) {
      numResultRows += cursor->current()->size();
    }

    BENCHMARK_SUSPEND {
      VELOX_CHECK_EQ(numResultRows, test.numRows);
      const auto stats = cursor->task()->taskStats();
      for (const auto& pipeline : stats.pipelineStats) {
        for (const auto& op : pipeline.operatorStats) {
          if (op.operatorType == OperatorType::kWindow) {
            windowTiming.add(op.addInputTiming);
            windowTiming.add(op.getOutputTiming);
          }
        }
      }
    }
    return numResultRows;
  }

  std::vector<std::unique_ptr<TestCase>> cases_;
};

std::unique_ptr<RowsStreamingWindowBenchmark> benchmark;

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  aggregate::prestosql::registerAllAggregateFunctions();
  window::prestosql::registerAllWindowFunctions();
  benchmark = std::make_unique<RowsStreamingWindowBenchmark>();
  benchmark->addBenchmarks();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
