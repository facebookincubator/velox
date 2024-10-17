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

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <string>

#include "velox/common/memory/SharedArbitrator.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int64(fuzzer_seed, 99887766, "Seed for random input dataset generator");

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

static constexpr int32_t kNumVectors = 50;
static constexpr int32_t kRowsPerVector = 1'0000;

namespace {

class WindowKRangeFrameBenchmark : public HiveConnectorTestBase {
 public:
  explicit WindowKRangeFrameBenchmark() {
    memory::SharedArbitrator::registerFactory();
    HiveConnectorTestBase::SetUp();
    aggregate::prestosql::registerAllAggregateFunctions();
    window::prestosql::registerAllWindowFunctions();

    inputType_ = ROW({
        {"k_array", INTEGER()},
        {"k_norm", INTEGER()},
        {"k_hash", INTEGER()},
        {"k_sort", INTEGER()},
        {"c0", INTEGER()},
        {"c1", INTEGER()},
        {"i32", INTEGER()},
    });

    VectorFuzzer::Options opts;
    opts.vectorSize = kRowsPerVector;
    opts.nullRatio = 0;
    VectorFuzzer fuzzer(opts, pool_.get(), FLAGS_fuzzer_seed);
    std::vector<RowVectorPtr> inputVectors;
    for (auto i = 0; i < kNumVectors; ++i) {
      std::vector<VectorPtr> children;

      // Generate key with a small number of unique values from a small range
      // (0-16).
      children.emplace_back(makeFlatVector<int32_t>(
          kRowsPerVector, [](auto row) { return row % 17; }));

      // Generate key with a small number of unique values from a large range
      // (300 total values).
      children.emplace_back(
          makeFlatVector<int32_t>(kRowsPerVector, [](auto row) {
            if (row % 3 == 0) {
              return std::numeric_limits<int32_t>::max() - row % 100;
            } else if (row % 3 == 1) {
              return row % 100;
            } else {
              return std::numeric_limits<int32_t>::min() + row % 100;
            }
          }));

      // Generate key with many unique values from a large range (500K total
      // values).
      children.emplace_back(fuzzer.fuzzFlat(INTEGER()));

      // Generate a column with increasing values to get a deterministic sort
      // order.
      children.emplace_back(makeFlatVector<int32_t>(
          kRowsPerVector, [](auto row) { return row; }));

      const int kPrecedingOffset = 3;
      const int kFollowingOffset = 5;
      // Only consider ascending order.
      // Generate preceding column.
      children.emplace_back(makeFlatVector<int32_t>(
          kRowsPerVector,
          [kPrecedingOffset](auto row) { return row - kPrecedingOffset; }));
      // Generate following column.
      children.emplace_back(makeFlatVector<int32_t>(
          kRowsPerVector,
          [kFollowingOffset](auto row) { return row + kFollowingOffset; }));

      children.emplace_back(fuzzer.fuzzFlat(INTEGER()));

      inputVectors.emplace_back(makeRowVector(inputType_->names(), children));
    }

    sourceFilePath_ = TempFilePath::create();
    writeToFile(sourceFilePath_->getPath(), inputVectors);
  }

  ~WindowKRangeFrameBenchmark() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  void run(const std::string& key, const std::string& aggregate) {
    folly::BenchmarkSuspender suspender1;

    std::string functionSql = fmt::format(
        "{} over (partition by {} order by k_sort range between c0 preceding and c1 following)",
        aggregate,
        key);

    core::PlanNodeId tableScanPlanId;
    core::PlanFragment plan = PlanBuilder()
                                  .tableScan(inputType_)
                                  .capturePlanNodeId(tableScanPlanId)
                                  .window({functionSql})
                                  .planFragment();

    vector_size_t numResultRows = 0;
    auto task = makeTask(plan);
    task->addSplit(
        tableScanPlanId,
        exec::Split(makeHiveConnectorSplit(sourceFilePath_->getPath())));
    task->noMoreSplits(tableScanPlanId);
    suspender1.dismiss();

    while (auto result = task->next()) {
      numResultRows += result->size();
    }

    folly::doNotOptimizeAway(numResultRows);
  }

  std::shared_ptr<exec::Task> makeTask(core::PlanFragment plan) {
    return exec::Task::create(
        "t",
        std::move(plan),
        0,
        core::QueryCtx::create(executor_.get()),
        Task::ExecutionMode::kSerial);
  }

 private:
  RowTypePtr inputType_;
  std::shared_ptr<TempFilePath> sourceFilePath_;
};

std::unique_ptr<WindowKRangeFrameBenchmark> benchmark;

void doRun(uint32_t, const std::string& key, const std::string& aggregate) {
  benchmark->run(key, aggregate);
}

#define AGG_BENCHMARKS(_name_, _key_)     \
  BENCHMARK_NAMED_PARAM(                  \
      doRun,                              \
      _name_##_INTEGER_##_key_,           \
      #_key_,                             \
      fmt::format("{}(i32)", (#_name_))); \
  BENCHMARK_DRAW_LINE();

// Count aggregate.
AGG_BENCHMARKS(count, k_array)
AGG_BENCHMARKS(count, k_norm)
AGG_BENCHMARKS(count, k_hash)
} // namespace

int main(int argc, char** argv) {
  folly::Init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});

  benchmark = std::make_unique<WindowKRangeFrameBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
