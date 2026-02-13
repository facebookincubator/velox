/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

static constexpr int32_t kNumVectors = 200;
static constexpr int32_t kRowsPerVector = 10'000;

namespace {
class BenchmarkRecorder {
 public:
  BenchmarkRecorder() = default;

  void record(std::string name, uint64_t numBytes) {
    // Only record the first apperance.
    if (numBytesRecords_.count(name) == 0) {
      numBytesRecords_[name] = {1, numBytes};
      names_.push_back(name);
    } else {
      auto& record = numBytesRecords_[name];
      record.numAppearance++;
      record.totalCount += numBytes;
    }
  }

  std::string report() {
    std::string result = "name, memory(MB)\n";
    for (auto& name : names_) {
      auto& record = numBytesRecords_[name];
      result += fmt::format(
          "{}, {}MB\n",
          name,
          record.totalCount / 1024 / 1024 / record.numAppearance);
    }
    return result;
  }

 private:
  struct Counter {
    int32_t numAppearance{0};
    uint64_t totalCount{0};
  };
  std::vector<std::string> names_;
  std::unordered_map<std::string, Counter> numBytesRecords_;
};

class WindowSubPartitionedSortBenchmark : public HiveConnectorTestBase {
 public:
  WindowSubPartitionedSortBenchmark(
      int32_t numVectors,
      int32_t rowsPerVector,
      std::shared_ptr<BenchmarkRecorder> recorder)
      : numVectors_(numVectors),
        rowsPerVector_(rowsPerVector),
        recorder_(recorder) {
    memory::SharedArbitrator::registerFactory();
    HiveConnectorTestBase::SetUp();
    aggregate::prestosql::registerAllAggregateFunctions();
    window::prestosql::registerAllWindowFunctions();

    inputType_ = ROW({
        {"k_array", INTEGER()},
        {"k_norm", INTEGER()},
        {"k_hash", INTEGER()},
        {"k_sort", INTEGER()},
        {"i32", INTEGER()},
        {"i64", BIGINT()},
        {"f32", REAL()},
        {"f64", DOUBLE()},
        {"i32_halfnull", INTEGER()},
        {"i64_halfnull", BIGINT()},
        {"f32_halfnull", REAL()},
        {"f64_halfnull", DOUBLE()},
    });

    VectorFuzzer::Options opts;
    opts.vectorSize = rowsPerVector_;
    opts.nullRatio = 0;
    VectorFuzzer fuzzer(opts, pool_.get(), FLAGS_fuzzer_seed);
    std::vector<RowVectorPtr> inputVectors;
    for (auto i = 0; i < numVectors_; ++i) {
      std::vector<VectorPtr> children;

      // Generate key with a small number of unique values from a small range
      // (0-16).
      children.emplace_back(
          makeFlatVector<int32_t>(
              rowsPerVector_, [](auto row) { return row % 17; }));

      // Generate key with a small number of unique values from a large range
      // (300 total values).
      children.emplace_back(
          makeFlatVector<int32_t>(rowsPerVector_, [](auto row) {
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
      children.emplace_back(
          makeFlatVector<int32_t>(
              rowsPerVector_, [](auto row) { return row; }));

      // Generate random values without nulls.
      children.emplace_back(fuzzer.fuzzFlat(INTEGER()));
      children.emplace_back(fuzzer.fuzzFlat(BIGINT()));
      children.emplace_back(fuzzer.fuzzFlat(REAL()));
      children.emplace_back(fuzzer.fuzzFlat(DOUBLE()));

      // Generate random values with nulls.
      opts.nullRatio = 0.05; // 5%
      fuzzer.setOptions(opts);

      children.emplace_back(fuzzer.fuzzFlat(INTEGER()));
      children.emplace_back(fuzzer.fuzzFlat(BIGINT()));
      children.emplace_back(fuzzer.fuzzFlat(REAL()));
      children.emplace_back(fuzzer.fuzzFlat(DOUBLE()));

      inputVectors.emplace_back(makeRowVector(inputType_->names(), children));
    }

    sourceFilePath_ = TempFilePath::create();
    writeToFile(sourceFilePath_->getPath(), inputVectors);
  }

  ~WindowSubPartitionedSortBenchmark() override {
    HiveConnectorTestBase::TearDown();
  }

  CpuWallTiming windowNanos() {
    return windowNanos_;
  }

  void TestBody() override {}

  void run(
      const std::string& recordName,
      const std::string& key,
      const std::string& aggregate,
      int32_t numSubPartitions) {
    folly::BenchmarkSuspender suspender1;

    windowNanos_.clear();
    windowMems_.clear();

    std::string functionSql = fmt::format(
        "{} over (partition by {} order by k_sort)", aggregate, key);

    core::PlanNodeId tableScanPlanId;
    core::PlanFragment plan = PlanBuilder()
                                  .tableScan(inputType_)
                                  .capturePlanNodeId(tableScanPlanId)
                                  .window({functionSql})
                                  .planFragment();

    vector_size_t numResultRows = 0;
    auto task = makeTask(plan, numSubPartitions);
    task->addSplit(
        tableScanPlanId,
        exec::Split(makeHiveConnectorSplit(sourceFilePath_->getPath())));
    task->noMoreSplits(tableScanPlanId);
    suspender1.dismiss();

    while (auto result = task->next()) {
      numResultRows += result->size();
    }

    folly::BenchmarkSuspender suspender2;
    auto stats = task->taskStats();
    for (auto& pipeline : stats.pipelineStats) {
      for (auto& op : pipeline.operatorStats) {
        if (op.operatorType == "Window") {
          windowNanos_.add(op.addInputTiming);
          windowNanos_.add(op.getOutputTiming);
          windowMems_.add(op.memoryStats);
        }
        if (op.operatorType == "Values") {
          // This is the timing for Window::noMoreInput() where the window
          // sorting happens. So including in the cpu timing.
          windowNanos_.add(op.finishTiming);
        }
      }
    }
    recorder_->record(recordName, windowMems_.peakTotalMemoryReservation);
    suspender2.dismiss();
    folly::doNotOptimizeAway(numResultRows);
  }

  std::shared_ptr<exec::Task> makeTask(
      core::PlanFragment plan,
      int32_t numSubPartitions) {
    bool subPartitionedSort = numSubPartitions > 1;
    if (subPartitionedSort) {
      const std::unordered_map<std::string, std::string> queryConfigMap(
          {{core::QueryConfig::kWindowNumSubPartitions,
            std::to_string(numSubPartitions)}});
      return exec::Task::create(
          "t",
          std::move(plan),
          0,
          core::QueryCtx::create(
              executor_.get(), core::QueryConfig(queryConfigMap)),
          Task::ExecutionMode::kSerial);

    } else {
      return exec::Task::create(
          "t",
          std::move(plan),
          0,
          core::QueryCtx::create(executor_.get()),
          Task::ExecutionMode::kSerial);
    }
  }

  uint64_t getLatestMemoryUsage() {
    return windowMems_.peakTotalMemoryReservation;
  }

 private:
  const int32_t numVectors_;
  const int32_t rowsPerVector_;
  const std::shared_ptr<BenchmarkRecorder> recorder_;
  RowTypePtr inputType_;
  std::shared_ptr<TempFilePath> sourceFilePath_;

  CpuWallTiming windowNanos_;
  MemoryStats windowMems_;
};

std::unique_ptr<WindowSubPartitionedSortBenchmark> benchmark;
auto recorder = std::make_shared<BenchmarkRecorder>();

void doSortRun(
    uint32_t,
    const std::string& recordName,
    int32_t numSubPartitions,
    const std::string& key,
    const std::string& aggregate) {
  benchmark->run(recordName, key, aggregate, numSubPartitions);
}

#define BENCHMARK_AND_RECORD_HEAD(_num_, _name_, _key_, _agg_) \
  BENCHMARK_NAMED_PARAM(                                       \
      doSortRun,                                               \
      num##_num_##_##_name_,                                   \
      fmt::format("num{}_{}", #_num_, #_name_),                \
      _num_,                                                   \
      _key_,                                                   \
      _agg_);

#define BENCHMARK_AND_RECORD_TAIL(_num_, _name_, _key_, _agg_) \
  BENCHMARK_RELATIVE_NAMED_PARAM(                              \
      doSortRun,                                               \
      num##_num_##_##_name_,                                   \
      fmt::format("num{}_{}", #_num_, #_name_),                \
      _num_,                                                   \
      _key_,                                                   \
      _agg_);

#define BATCHED_BENCHMARKS(_name_, _key_, _agg_)         \
  BENCHMARK_AND_RECORD_HEAD(1, _name_, _key_, _agg_);    \
  BENCHMARK_AND_RECORD_TAIL(2, _name_, _key_, _agg_);    \
  BENCHMARK_AND_RECORD_TAIL(4, _name_, _key_, _agg_);    \
  BENCHMARK_AND_RECORD_TAIL(8, _name_, _key_, _agg_);    \
  BENCHMARK_AND_RECORD_TAIL(16, _name_, _key_, _agg_);   \
  BENCHMARK_AND_RECORD_TAIL(32, _name_, _key_, _agg_);   \
  BENCHMARK_AND_RECORD_TAIL(64, _name_, _key_, _agg_);   \
  BENCHMARK_AND_RECORD_TAIL(128, _name_, _key_, _agg_);  \
  BENCHMARK_AND_RECORD_TAIL(256, _name_, _key_, _agg_);  \
  BENCHMARK_AND_RECORD_TAIL(512, _name_, _key_, _agg_);  \
  BENCHMARK_AND_RECORD_TAIL(1024, _name_, _key_, _agg_); \
  BENCHMARK_AND_RECORD_TAIL(2048, _name_, _key_, _agg_);

#define AGG_BENCHMARKS(_name_, _key_)                                       \
  BATCHED_BENCHMARKS(                                                       \
      _name_##_INTEGER_##_key_, #_key_, fmt::format("{}(i32)", (#_name_))); \
  BATCHED_BENCHMARKS(                                                       \
      _name_##_REAL_##_key_, #_key_, fmt::format("{}(f32)", (#_name_)));    \
  BATCHED_BENCHMARKS(                                                       \
      _name_##_INTEGER_NULLS_##_key_,                                       \
      #_key_,                                                               \
      fmt::format("{}(i32_halfnull)", (#_name_)));                          \
  BATCHED_BENCHMARKS(                                                       \
      _name_##_REAL_NULLS_##_key_,                                          \
      #_key_,                                                               \
      fmt::format("{}(f32_halfnull)", (#_name_)));

#define MULTI_KEY_AGG_BENCHMARKS(_name_, _key1_, _key2_) \
  BATCHED_BENCHMARKS(                                    \
      _name_##_BIGINT_##_key1_##_key2_,                  \
      fmt::format("{},{}", (#_key1_), (#_key2_)),        \
      fmt::format("{}(i64)", (#_name_)));                \
  BATCHED_BENCHMARKS(                                    \
      _name_##_BIGINT_NULLS_##_key1_##_key2_,            \
      fmt::format("{},{}", (#_key1_), (#_key2_)),        \
      fmt::format("{}(i64_halfnull)", (#_name_)));       \
  BATCHED_BENCHMARKS(                                    \
      _name_##_DOUBLE_##_key1_##_key2_,                  \
      fmt::format("{},{}", (#_key1_), (#_key2_)),        \
      fmt::format("{}(f64)", (#_name_)));                \
  BATCHED_BENCHMARKS(                                    \
      _name_##_DOUBLE_NULLS_##_key1_##_key2_,            \
      fmt::format("{},{}", (#_key1_), (#_key2_)),        \
      fmt::format("{}(f64_halfnull)", (#_name_)));

// Count(1) aggregate.
BATCHED_BENCHMARKS(count_k_array, "k_array", "count(1)");
BATCHED_BENCHMARKS(count_k_norm, "k_norm", "count(1)");
BATCHED_BENCHMARKS(count_k_hash, "k_hash", "count(1)");
BATCHED_BENCHMARKS(count_k_array_k_hash, "k_array,i32", "count(1)");
BENCHMARK_DRAW_LINE();

// Count aggregate.
AGG_BENCHMARKS(count, k_array)
AGG_BENCHMARKS(count, k_norm)
AGG_BENCHMARKS(count, k_hash)
MULTI_KEY_AGG_BENCHMARKS(count, k_array, i32)
MULTI_KEY_AGG_BENCHMARKS(count, k_array, i64)
MULTI_KEY_AGG_BENCHMARKS(count, k_hash, f32)
MULTI_KEY_AGG_BENCHMARKS(count, k_hash, f64)
BENCHMARK_DRAW_LINE();

// Avg aggregate.
AGG_BENCHMARKS(avg, k_array)
AGG_BENCHMARKS(avg, k_norm)
AGG_BENCHMARKS(avg, k_hash)
MULTI_KEY_AGG_BENCHMARKS(avg, k_array, i32)
MULTI_KEY_AGG_BENCHMARKS(avg, k_array, i64)
MULTI_KEY_AGG_BENCHMARKS(avg, k_hash, f32)
MULTI_KEY_AGG_BENCHMARKS(avg, k_hash, f64)
BENCHMARK_DRAW_LINE();

// Min aggregate.
AGG_BENCHMARKS(min, k_array)
AGG_BENCHMARKS(min, k_norm)
AGG_BENCHMARKS(min, k_hash)
MULTI_KEY_AGG_BENCHMARKS(min, k_array, i32)
MULTI_KEY_AGG_BENCHMARKS(min, k_array, i64)
MULTI_KEY_AGG_BENCHMARKS(min, k_hash, f32)
MULTI_KEY_AGG_BENCHMARKS(min, k_hash, f64)
BENCHMARK_DRAW_LINE();

// Max aggregate.
AGG_BENCHMARKS(max, k_array)
AGG_BENCHMARKS(max, k_norm)
AGG_BENCHMARKS(max, k_hash)
MULTI_KEY_AGG_BENCHMARKS(max, k_array, i32)
MULTI_KEY_AGG_BENCHMARKS(max, k_array, i64)
MULTI_KEY_AGG_BENCHMARKS(max, k_hash, f32)
MULTI_KEY_AGG_BENCHMARKS(max, k_hash, f64)
BENCHMARK_DRAW_LINE();

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});

  benchmark = std::make_unique<WindowSubPartitionedSortBenchmark>(
      kNumVectors, kRowsPerVector, recorder);
  folly::runBenchmarks();
  benchmark.reset();

  std::cout << std::endl << recorder->report();
  return 0;
}
