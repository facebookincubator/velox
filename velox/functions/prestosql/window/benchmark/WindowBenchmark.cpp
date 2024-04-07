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

#include "velox/exec/tests/utils/Cursor.h"
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

static constexpr int32_t kNumVectors = 10;
static constexpr int32_t kRowsPerVector = 30'000;

namespace {

class WindowBenchmark : public HiveConnectorTestBase {
public:
 explicit WindowBenchmark() {
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
   opts.vectorSize = kRowsPerVector;
   opts.nullRatio = 0;
   VectorFuzzer fuzzer(opts, pool_.get(), FLAGS_fuzzer_seed);

   std::vector<RowVectorPtr> vectors;
   for (auto i = 0; i < kNumVectors; ++i) {
     std::vector<VectorPtr> children;

     // Generate key with a small number of unique values from a small range
     // (0-16).
     children.emplace_back(makeFlatVector<int32_t>(
         kRowsPerVector, [](auto row) { return row % 4; }));

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

     // Generate random values without nulls.
     children.emplace_back(fuzzer.fuzzFlat(INTEGER()));
     children.emplace_back(fuzzer.fuzzFlat(BIGINT()));
     children.emplace_back(fuzzer.fuzzFlat(REAL()));
     children.emplace_back(fuzzer.fuzzFlat(DOUBLE()));

     // Generate random values with nulls.

     opts.nullRatio = 0.5; // 50%
     fuzzer.setOptions(opts);

     children.emplace_back(fuzzer.fuzzFlat(INTEGER()));
     children.emplace_back(fuzzer.fuzzFlat(BIGINT()));
     children.emplace_back(fuzzer.fuzzFlat(REAL()));
     children.emplace_back(fuzzer.fuzzFlat(DOUBLE()));

     vectors.emplace_back(makeRowVector(inputType_->names(), children));
   }
   filePath_ = TempFilePath::create();
   writeToFile(filePath_->path, vectors);
 }

 ~WindowBenchmark() override {
   HiveConnectorTestBase::TearDown();
 }

 void TestBody() override {}

 // Enhance this function to include a frame clause.
 void run(
     const std::string& key,
     const std::string& aggregate,
     const std::string& frameType,
     const uint32_t frameSize,
     bool enableSegmentTree) {
   folly::BenchmarkSuspender suspender;

   std::string functionSql = fmt::format(
       "{} over (partition by {} order by k_sort {} between {} PRECEDING and {} FOLLOWING)",
       aggregate,
       key,
       frameType,
       frameSize / 2,
       frameSize / 2);
   core::PlanFragment plan;

   plan =
       PlanBuilder().tableScan(inputType_).window({functionSql}).planFragment();

   vector_size_t numResultRows = 0;
   auto task = makeTask(plan, numResultRows, enableSegmentTree);

   task->addSplit("0", exec::Split(makeHiveConnectorSplit(filePath_->path)));
   task->noMoreSplits("0");

   suspender.dismiss();

   task->start(1);
   auto& executor = folly::QueuedImmediateExecutor::instance();
   auto future = task->stateChangeFuture(60'000'000).via(&executor);
   future.wait();

   folly::doNotOptimizeAway(numResultRows);
 }

 std::shared_ptr<exec::Task> makeTask(
     core::PlanFragment plan,
     vector_size_t& numResultRows,
     bool enableSegmentTree) {
   int32_t minFrameSizeUseSegmentTree = 0;
   if (!enableSegmentTree) {
     minFrameSizeUseSegmentTree = std::numeric_limits<int32_t>::max();
   }

   std::unordered_map<std::string, std::string> queryConfig{
       {core::QueryConfig::kMinFrameSizeUseSegmentTree,
        fmt::format("{}", minFrameSizeUseSegmentTree)}};

   return exec::Task::create(
       "t",
       std::move(plan),
       0,
       std::make_shared<core::QueryCtx>(
           executor_.get(),
           core::QueryConfig(std::move(queryConfig))),
       [&](auto vector, auto* /*future*/) {
         if (vector) {
           numResultRows += vector->size();
         }
         return exec::BlockingReason::kNotBlocked;
       });
 }

private:
 RowTypePtr inputType_;

 std::shared_ptr<TempFilePath> filePath_;
};

std::unique_ptr<WindowBenchmark> benchmark;

void rawRowFrame(uint32_t, const std::string& key, const std::string& aggregate, uint32_t frameSize) {
  benchmark->run(key, aggregate, "rows", frameSize, false);
}

void segmentOptRowFrame(uint32_t, const std::string& key, const std::string& aggregate, uint32_t frameSize) {
 benchmark->run(key, aggregate, "rows", frameSize, true);
}

#define WINDOW_AGG_BENCHMARKS(_name_, _key_)       \
 BENCHMARK_NAMED_PARAM(                            \
     rawRowFrame,                                  \
     _name_##_frame_16_##_key_,                    \
     #_key_,                                       \
     fmt::format("{}(i32)", (#_name_)),            \
      16);                                         \
 BENCHMARK_NAMED_PARAM(                            \
     segmentOptRowFrame,                           \
     _name_##_frame_16_##_key_,                    \
     #_key_,                                       \
     fmt::format("{}(i32)", (#_name_)),            \
      16);                                         \
  BENCHMARK_NAMED_PARAM(                           \
     rawRowFrame,                                  \
     _name_##_frame_64_##_key_,                    \
     #_key_,                                       \
     fmt::format("{}(i32)", (#_name_)),            \
      64);                                         \
 BENCHMARK_NAMED_PARAM(                            \
     segmentOptRowFrame,                           \
     _name_##_frame_64_##_key_,                    \
     #_key_,                                       \
     fmt::format("{}(i32)", (#_name_)),            \
      64);                                         \
  BENCHMARK_NAMED_PARAM(                           \
      rawRowFrame,                                 \
      _name_##_frame_256_##_key_,                  \
      #_key_,                                      \
      fmt::format("{}(i32)", (#_name_)),           \
       256);                                       \
  BENCHMARK_NAMED_PARAM(                           \
      segmentOptRowFrame,                          \
      _name_##_frame_256_##_key_,                  \
      #_key_,                                      \
      fmt::format("{}(i32)", (#_name_)),           \
       256);                                       \
   BENCHMARK_NAMED_PARAM(                          \
      rawRowFrame,                                 \
      _name_##_frame_512_##_key_,                  \
      #_key_,                                      \
      fmt::format("{}(i32)", (#_name_)),           \
       512);                                       \
  BENCHMARK_NAMED_PARAM(                           \
      segmentOptRowFrame,                          \
      _name_##_frame_512_##_key_,                  \
      #_key_,                                      \
      fmt::format("{}(i32)", (#_name_)),           \
       512);                                       \
  BENCHMARK_NAMED_PARAM(                           \
     rawRowFrame,                                  \
     _name_##_frame_2048_##_key_,                  \
     #_key_,                                       \
     fmt::format("{}(i32)", (#_name_)),            \
      2048);                                       \
 BENCHMARK_NAMED_PARAM(                            \
     segmentOptRowFrame,                           \
     _name_##_frame_2048_##_key_,                  \
     #_key_,                                       \
     fmt::format("{}(i32)", (#_name_)),            \
      2048);                                       \
 BENCHMARK_DRAW_LINE();

// Sum aggregate.
WINDOW_AGG_BENCHMARKS(sum, k_array)
BENCHMARK_DRAW_LINE();

// Avg aggregate.
WINDOW_AGG_BENCHMARKS(avg, k_array)
BENCHMARK_DRAW_LINE();

// Min aggregate.
WINDOW_AGG_BENCHMARKS(min, k_array)
BENCHMARK_DRAW_LINE();

// Max aggregate.
WINDOW_AGG_BENCHMARKS(max, k_array)
BENCHMARK_DRAW_LINE();

// Stddev aggregate.
WINDOW_AGG_BENCHMARKS(stddev, k_array)
BENCHMARK_DRAW_LINE();

} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});

  benchmark = std::make_unique<WindowBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}