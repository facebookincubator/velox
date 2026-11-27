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
#include <string>
#include <vector>

#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int64(fuzzer_seed, 99887766, "Seed for random input dataset generator");

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

static constexpr int32_t kNumVectors = 100;
static constexpr int32_t kRowsPerVector = 10'000;

namespace {

class StringMinMaxBenchmark : public OperatorTestBase {
 public:
  StringMinMaxBenchmark() {
    OperatorTestBase::SetUp();

    VectorFuzzer::Options opts;
    opts.vectorSize = kRowsPerVector;
    opts.nullRatio = 0;
    opts.stringVariableLength = false;
    VectorFuzzer fuzzer(opts, pool(), FLAGS_fuzzer_seed);

    vectors_.reserve(kNumVectors);
    for (int32_t i = 0; i < kNumVectors; ++i) {
      std::vector<VectorPtr> children;

      // Low-cardinality key (100 groups).
      children.emplace_back(
          makeFlatVector<int32_t>(
              kRowsPerVector, [](auto row) { return row % 100; }));

      // High-cardinality key (10,000 groups).
      children.emplace_back(
          makeFlatVector<int32_t>(
              kRowsPerVector, [](auto row) { return row % 10'000; }));

      // Short (inline, 8 bytes) strings without nulls.
      opts.nullRatio = 0;
      opts.stringLength = 8;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));

      // Long (non-inline, 48 bytes) strings without nulls.
      opts.stringLength = 48;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));

      // Short (inline, 8 bytes) strings with 50% nulls.
      opts.nullRatio = 0.5;
      opts.stringLength = 8;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));

      // Long (non-inline, 48 bytes) strings with 50% nulls.
      opts.stringLength = 48;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));

      // Extra long string columns for multi-aggregate benchmark.
      opts.nullRatio = 0;
      opts.stringLength = 48;
      fuzzer.setOptions(opts);
      for (int32_t c = 0; c < 8; ++c) {
        children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));
      }

      std::vector<std::string> names = {
          "k_low",
          "k_high",
          "short_str",
          "long_str",
          "short_str_null",
          "long_str_null",
          "s0",
          "s1",
          "s2",
          "s3",
          "s4",
          "s5",
          "s6",
          "s7",
      };
      vectors_.emplace_back(makeRowVector(names, children));
    }
  }

  ~StringMinMaxBenchmark() override {
    vectors_.clear();
    OperatorTestBase::TearDown();
  }

  void TestBody() override {}

  void run(
      const std::vector<std::string>& keys,
      const std::vector<std::string>& aggregates) {
    folly::BenchmarkSuspender suspender;

    auto plan = PlanBuilder()
                    .values(vectors_)
                    .partialAggregation(keys, aggregates)
                    .finalAggregation()
                    .planFragment();

    auto task = exec::Task::create(
        "t",
        std::move(plan),
        0,
        core::QueryCtx::create(executor_.get()),
        exec::Task::ExecutionMode::kSerial,
        exec::Consumer{});

    vector_size_t numResultRows = 0;
    suspender.dismiss();

    while (auto result = task->next()) {
      numResultRows += result->size();
    }

    folly::doNotOptimizeAway(numResultRows);
  }

 private:
  std::vector<RowVectorPtr> vectors_;
};

std::unique_ptr<StringMinMaxBenchmark> benchmark;

void runSingle(
    uint32_t,
    const std::vector<std::string>& keys,
    const std::string& agg) {
  benchmark->run(keys, {agg});
}

void runMulti(uint32_t, const std::vector<std::string>& keys) {
  benchmark->run(
      keys,
      {
          "min(s0)",
          "max(s0)",
          "min(s1)",
          "max(s1)",
          "min(s2)",
          "max(s2)",
          "min(s3)",
          "max(s3)",
          "min(s4)",
          "max(s4)",
          "min(s5)",
          "max(s5)",
          "min(s6)",
          "max(s6)",
          "min(s7)",
          "max(s7)",
      });
}

// Global (single group) aggregation.
BENCHMARK_NAMED_PARAM(runSingle, global_min_short, {}, "min(short_str)");
BENCHMARK_NAMED_PARAM(runSingle, global_max_long, {}, "max(long_str)");
BENCHMARK_DRAW_LINE();

// Low-cardinality grouping (100 groups).
BENCHMARK_NAMED_PARAM(runSingle, k_low_min_short, {"k_low"}, "min(short_str)");
BENCHMARK_NAMED_PARAM(runSingle, k_low_max_short, {"k_low"}, "max(short_str)");
BENCHMARK_NAMED_PARAM(runSingle, k_low_min_long, {"k_low"}, "min(long_str)");
BENCHMARK_NAMED_PARAM(runSingle, k_low_max_long, {"k_low"}, "max(long_str)");
BENCHMARK_NAMED_PARAM(
    runSingle,
    k_low_min_short_null,
    {"k_low"},
    "min(short_str_null)");
BENCHMARK_NAMED_PARAM(
    runSingle,
    k_low_max_long_null,
    {"k_low"},
    "max(long_str_null)");
BENCHMARK_DRAW_LINE();

// High-cardinality grouping (10,000 groups).
BENCHMARK_NAMED_PARAM(
    runSingle,
    k_high_min_short,
    {"k_high"},
    "min(short_str)");
BENCHMARK_NAMED_PARAM(runSingle, k_high_max_long, {"k_high"}, "max(long_str)");
BENCHMARK_DRAW_LINE();

// 16 min/max string aggregates per group.
BENCHMARK_NAMED_PARAM(runMulti, k_low_16_minmax_long, {"k_low"});
BENCHMARK_NAMED_PARAM(runMulti, k_high_16_minmax_long, {"k_high"});

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  OperatorTestBase::SetUpTestCase();
  benchmark = std::make_unique<StringMinMaxBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  OperatorTestBase::TearDownTestCase();
  return 0;
}
