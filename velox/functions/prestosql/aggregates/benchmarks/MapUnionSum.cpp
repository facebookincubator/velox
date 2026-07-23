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

#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

// Emulates the "deltoid" workload: many small integer-keyed maps with ~5-20
// keys each, aggregated with map_union_sum both globally and grouped.
static constexpr int32_t kNumVectors = 100;
static constexpr int32_t kRowsPerVector = 10'000;

// Number of distinct map keys drawn per row. Keys come from a small universe so
// that the accumulator map stays compact and repeatedly hits existing keys.
static constexpr int32_t kKeysPerMap = 10;
static constexpr int32_t kKeyUniverse = 64;

// Number of distinct group-by keys for the grouped benchmark.
static constexpr int32_t kNumGroups = 1'000;

class MapUnionSumBenchmark : public HiveConnectorTestBase {
 public:
  MapUnionSumBenchmark() {
    HiveConnectorTestBase::SetUp();

    inputType_ = ROW({
        {"g", INTEGER()},
        {"m_i64", MAP(INTEGER(), BIGINT())},
        {"m_i64_nulls", MAP(INTEGER(), BIGINT())},
        {"m_f64", MAP(INTEGER(), DOUBLE())},
    });

    for (auto vectorIndex = 0; vectorIndex < kNumVectors; ++vectorIndex) {
      const auto base = vectorIndex * kRowsPerVector;

      auto groupKeys = makeFlatVector<int32_t>(
          kRowsPerVector, [](auto row) { return row % kNumGroups; });

      auto sizeAt = [](vector_size_t /*row*/) { return kKeysPerMap; };
      // Spread the key window across rows so different rows touch overlapping
      // but not identical key ranges, exercising both insert and update paths.
      auto keyAt = [base](vector_size_t idx) {
        const auto row = idx / kKeysPerMap;
        const auto offset = idx % kKeysPerMap;
        return static_cast<int32_t>(((base + row) + offset) % kKeyUniverse);
      };
      auto i64ValueAt = [](vector_size_t idx) {
        return static_cast<int64_t>(idx % 7);
      };
      auto f64ValueAt = [](vector_size_t idx) {
        return static_cast<double>(idx % 7);
      };
      // Every 4th value is null to exercise the null-value (+0) path.
      auto valueIsNullAt = [](vector_size_t idx) { return idx % 4 == 0; };

      auto mapI64 = makeMapVector<int32_t, int64_t>(
          kRowsPerVector, sizeAt, keyAt, i64ValueAt);
      auto mapI64Nulls = makeMapVector<int32_t, int64_t>(
          kRowsPerVector, sizeAt, keyAt, i64ValueAt, nullptr, valueIsNullAt);
      auto mapF64 = makeMapVector<int32_t, double>(
          kRowsPerVector, sizeAt, keyAt, f64ValueAt);

      vectors_.emplace_back(makeRowVector(
          inputType_->names(), {groupKeys, mapI64, mapI64Nulls, mapF64}));
    }

    filePath_ = TempFilePath::create();
    writeToFile(filePath_->getPath(), vectors_);
  }

  ~MapUnionSumBenchmark() override {
    // Release the input vectors before tearing down the connector/memory pools;
    // as a member they would otherwise outlive TearDown() and leave their
    // reservation held when the arbitrator removes the root pool.
    vectors_.clear();
    HiveConnectorTestBase::TearDown();
  }

  MapUnionSumBenchmark(const MapUnionSumBenchmark&) = delete;
  MapUnionSumBenchmark& operator=(const MapUnionSumBenchmark&) = delete;
  MapUnionSumBenchmark(MapUnionSumBenchmark&&) = delete;
  MapUnionSumBenchmark& operator=(MapUnionSumBenchmark&&) = delete;

  void TestBody() override {}

  void run(const std::vector<std::string>& keys, const std::string& aggregate) {
    folly::BenchmarkSuspender suspender;

    auto plan = PlanBuilder()
                    .tableScan(inputType_)
                    .partialAggregation(keys, {aggregate})
                    .finalAggregation()
                    .planFragment();

    vector_size_t numResultRows{0};
    auto task = makeTask(plan);

    task->addSplit(
        "0", exec::Split(makeHiveConnectorSplit(filePath_->getPath())));
    task->noMoreSplits("0");

    suspender.dismiss();

    while (auto result = task->next()) {
      numResultRows += result->size();
    }

    folly::doNotOptimizeAway(numResultRows);
  }

  // Micro variant: feeds the pre-built vectors straight through a Values node
  // (no table scan, no file I/O), so the measured time isolates the
  // aggregation. This captures the map_union_sum improvement more accurately
  // than run(), where fixed table-scan overhead dilutes the ratio.
  void runInMemory(
      const std::vector<std::string>& keys,
      const std::string& aggregate) {
    folly::BenchmarkSuspender suspender;

    auto plan = PlanBuilder()
                    .values(vectors_)
                    .partialAggregation(keys, {aggregate})
                    .finalAggregation()
                    .planFragment();

    auto task = makeTask(plan);

    suspender.dismiss();

    vector_size_t numResultRows{0};
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
        exec::Task::ExecutionMode::kSerial,
        exec::Consumer{});
  }

 private:
  RowTypePtr inputType_;
  std::vector<RowVectorPtr> vectors_;
  std::shared_ptr<TempFilePath> filePath_;
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::unique_ptr<MapUnionSumBenchmark> benchmark;

void doRunGlobal(uint32_t, const std::string& aggregate) {
  benchmark->run({}, aggregate);
}

void doRunGrouped(uint32_t, const std::string& aggregate) {
  benchmark->run({"g"}, aggregate);
}

void doRunGlobalMicro(uint32_t, const std::string& aggregate) {
  benchmark->runInMemory({}, aggregate);
}

void doRunGroupedMicro(uint32_t, const std::string& aggregate) {
  benchmark->runInMemory({"g"}, aggregate);
}

// End-to-end (table scan -> aggregation). Realistic pipeline; the scan overhead
// is fixed across baseline/optimized so it understates the aggregate speedup.
// Global (no group-by) map_union_sum.
BENCHMARK_NAMED_PARAM(doRunGlobal, global_i64, "map_union_sum(m_i64)");
BENCHMARK_NAMED_PARAM(
    doRunGlobal,
    global_i64_nulls,
    "map_union_sum(m_i64_nulls)");
BENCHMARK_NAMED_PARAM(doRunGlobal, global_f64, "map_union_sum(m_f64)");
BENCHMARK_DRAW_LINE();

// Grouped map_union_sum.
BENCHMARK_NAMED_PARAM(doRunGrouped, grouped_i64, "map_union_sum(m_i64)");
BENCHMARK_NAMED_PARAM(
    doRunGrouped,
    grouped_i64_nulls,
    "map_union_sum(m_i64_nulls)");
BENCHMARK_NAMED_PARAM(doRunGrouped, grouped_f64, "map_union_sum(m_f64)");
BENCHMARK_DRAW_LINE();

// Micro (Values -> aggregation, no table scan). Isolates the aggregate so the
// map_union_sum improvement is captured without fixed scan overhead.
// Global (no group-by) map_union_sum.
BENCHMARK_NAMED_PARAM(
    doRunGlobalMicro,
    micro_global_i64,
    "map_union_sum(m_i64)");
BENCHMARK_NAMED_PARAM(
    doRunGlobalMicro,
    micro_global_i64_nulls,
    "map_union_sum(m_i64_nulls)");
BENCHMARK_NAMED_PARAM(
    doRunGlobalMicro,
    micro_global_f64,
    "map_union_sum(m_f64)");
BENCHMARK_DRAW_LINE();

// Grouped map_union_sum.
BENCHMARK_NAMED_PARAM(
    doRunGroupedMicro,
    micro_grouped_i64,
    "map_union_sum(m_i64)");
BENCHMARK_NAMED_PARAM(
    doRunGroupedMicro,
    micro_grouped_i64_nulls,
    "map_union_sum(m_i64_nulls)");
BENCHMARK_NAMED_PARAM(
    doRunGroupedMicro,
    micro_grouped_f64,
    "map_union_sum(m_f64)");
BENCHMARK_DRAW_LINE();

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  OperatorTestBase::SetUpTestCase();
  benchmark = std::make_unique<MapUnionSumBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  OperatorTestBase::TearDownTestCase();
  return 0;
}
