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
#include <folly/Random.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

DEFINE_int64(
    krange_rows,
    1'000'000,
    "Total number of input rows of each benchmark case.");
DEFINE_double(
    krange_null_ratio,
    0.05,
    "Fraction of rows whose ORDER BY key is null.");

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;
using facebook::velox::exec::test::PlanBuilder;

// Measures Window operators with k RANGE PRECEDING / FOLLOWING frames, from
// plan to results, in the shape of Presto plans: a project node computes the
// bound values s - k and s + k, and the frame refers to them as columns.
//
// Each case varies the partition size (1P is one partition, 100P is 100) and
// the ORDER BY keys of a partition:
// - unique: 0, 1, 2, ... in input order.
// - 100peers: each key on 100 consecutive rows. Keys with many peer rows
//   exercise the end bound search the most.
// - random: drawn uniformly from [0, partition size) in random order, so
//   frames vary in size.
// In every case a fraction --krange_null_ratio of keys, at random rows, is
// null.
namespace {

constexpr int32_t kRowsPerVector = 10'000;
constexpr int64_t kOffset = 10;

class KRangeWindowBenchmark : public VectorTestBase {
 public:
  void addBenchmarks() {
    const auto numRows = FLAGS_krange_rows;
    for (const auto& [partitionName, rowsPerPartition] :
         std::vector<std::pair<std::string, int64_t>>{
             {"1P", numRows}, {"100P", numRows / 100}}) {
      for (const auto keys : {Keys::kUnique, Keys::k100Peers, Keys::kRandom}) {
        addCases(
            fmt::format("{}_{}", partitionName, keysName(keys)),
            makeData(numRows, rowsPerPartition, keys));
      }
    }
  }

 private:
  struct TestCase {
    std::vector<RowVectorPtr> data;
    core::PlanNodePtr plan;
    int64_t numRows;
  };

  enum class Keys { kUnique, k100Peers, kRandom };

  static std::string keysName(Keys keys) {
    switch (keys) {
      case Keys::kUnique:
        return "unique";
      case Keys::k100Peers:
        return "100peers";
      case Keys::kRandom:
        return "random";
    }
    VELOX_UNREACHABLE();
  }

  // Makes 'numRows' rows in partitions of 'rowsPerPartition' consecutive rows,
  // with ORDER BY keys as described by 'keys'.
  std::vector<RowVectorPtr>
  makeData(int64_t numRows, int64_t rowsPerPartition, Keys keys) {
    std::mt19937 rng(static_cast<uint32_t>(keys));
    std::vector<RowVectorPtr> data;
    for (int64_t offset = 0; offset < numRows; offset += kRowsPerVector) {
      const auto size = static_cast<vector_size_t>(
          std::min<int64_t>(kRowsPerVector, numRows - offset));
      std::vector<std::optional<int64_t>> keyValues(size);
      for (auto row = 0; row < size; ++row) {
        const int64_t position = (offset + row) % rowsPerPartition;
        switch (keys) {
          case Keys::kUnique:
            keyValues[row] = position;
            break;
          case Keys::k100Peers:
            keyValues[row] = position / 100;
            break;
          case Keys::kRandom:
            keyValues[row] = folly::Random::rand64(rowsPerPartition, rng);
            break;
        }
        if (folly::Random::randDouble01(rng) < FLAGS_krange_null_ratio) {
          keyValues[row] = std::nullopt;
        }
      }
      data.push_back(makeRowVector(
          {"p", "s", "v"},
          {
              makeFlatVector<int32_t>(
                  size,
                  [=](auto row) { return (offset + row) / rowsPerPartition; }),
              makeNullableFlatVector<int64_t>(keyValues),
              makeFlatVector<int64_t>(size, [](auto row) { return row % 97; }),
          }));
    }
    return data;
  }

  void addCases(const std::string& name, std::vector<RowVectorPtr> data) {
    struct Frame {
      std::string name;
      std::string start;
      std::string end;
    };
    const std::vector<Frame> frames{
        {"precToCur", "off0 preceding", "current row"},
        {"curToFoll", "current row", "off1 following"},
        {"precToFoll", "off0 preceding", "off1 following"},
    };
    const std::vector<std::string> projections{
        "p",
        "s",
        "v",
        fmt::format("s - cast({} as bigint) as off0", kOffset),
        fmt::format("s + cast({} as bigint) as off1", kOffset)};
    // first_value() takes constant time per row, so frame bound computation
    // dominates. sum() aggregates over every row of a sliding frame.
    const std::vector<std::pair<std::string, std::string>> functions{
        {"first", "first_value"}, {"sum", "sum"}};
    for (const auto& [functionName, function] : functions) {
      for (const auto& frame : frames) {
        auto plan =
            PlanBuilder()
                .values(data)
                .project(projections)
                .window({fmt::format(
                    "{}(v) over (partition by p order by s range between {} and {})",
                    function,
                    frame.start,
                    frame.end)})
                .planNode();
        addBenchmark(
            fmt::format("{}_{}_{}", functionName, name, frame.name),
            data,
            std::move(plan));
      }
    }
  }

  void addBenchmark(
      const std::string& name,
      const std::vector<RowVectorPtr>& data,
      core::PlanNodePtr plan) {
    auto testCase = std::make_unique<TestCase>();
    testCase->data = data;
    testCase->numRows = 0;
    for (const auto& vector : data) {
      testCase->numRows += vector->size();
    }
    testCase->plan = std::move(plan);

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
            counters["windowCpu"] = folly::UserMetric(
                static_cast<double>(windowTiming.cpuNanos) / iterations / 1e9,
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
        for (const auto& operatorStats : pipeline.operatorStats) {
          if (operatorStats.operatorType == OperatorType::kWindow) {
            windowTiming.add(operatorStats.addInputTiming);
            windowTiming.add(operatorStats.finishTiming);
            windowTiming.add(operatorStats.getOutputTiming);
          }
        }
      }
    }
    return numResultRows;
  }

  std::vector<std::unique_ptr<TestCase>> cases_;
};

std::unique_ptr<KRangeWindowBenchmark> benchmark;

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  parse::registerTypeResolver();
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  window::prestosql::registerAllWindowFunctions();
  benchmark = std::make_unique<KRangeWindowBenchmark>();
  benchmark->addBenchmarks();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
