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

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

constexpr int32_t kNumVectors = 100;
constexpr int32_t kRowsPerVector = 10'000;

// Measures approx_percentile on unique grouping keys with partial aggregation
// abandoned after the first input, so the remaining inputs are converted to
// intermediate results row by row.
class ApproxPercentileBenchmark : public OperatorTestBase {
 public:
  ApproxPercentileBenchmark() {
    OperatorTestBase::SetUp();
    for (auto i = 0; i < kNumVectors; ++i) {
      const int64_t offset = i * kRowsPerVector;
      data_.push_back(makeRowVector(
          {"k", "v", "w"},
          {
              makeFlatVector<int64_t>(
                  kRowsPerVector, [&](auto row) { return offset + row; }),
              makeFlatVector<double>(
                  kRowsPerVector,
                  [&](auto row) { return (offset + row) % 1'009 * 0.5; }),
              makeFlatVector<int64_t>(
                  kRowsPerVector,
                  [&](auto row) { return (offset + row) % 1'000 + 1; }),
          }));
    }
  }

  ~ApproxPercentileBenchmark() override {
    // Release the input vectors before TearDown() drops the memory pools.
    data_.clear();
    OperatorTestBase::TearDown();
  }

  void TestBody() override {}

  static inline const std::string kUnweighted = "approx_percentile(v, 0.5)";
  static inline const std::string kWeighted = "approx_percentile(v, w, 0.5)";

  // Checks that partial aggregation is abandoned, so the benchmarks measure
  // the abandoned path.
  void verify() {
    for (const auto& aggregate : {kUnweighted, kWeighted}) {
      core::PlanNodeId partialNodeId;
      auto task = makeTask(makePlan(aggregate, false, partialNodeId));
      while (task->next()) {
      }
      const auto stats = toPlanStats(task->taskStats());
      VELOX_CHECK_GT(
          stats.at(partialNodeId)
              .customStats.at("abandonedPartialAggregationRows")
              .sum,
          0);
    }
  }

  void run(const std::string& aggregate, bool withFinal) {
    folly::BenchmarkSuspender suspender;
    core::PlanNodeId partialNodeId;
    auto task = makeTask(makePlan(aggregate, withFinal, partialNodeId));
    suspender.dismiss();

    vector_size_t numResultRows = 0;
    while (auto result = task->next()) {
      numResultRows += result->size();
    }
    folly::doNotOptimizeAway(numResultRows);
  }

 private:
  core::PlanNodePtr makePlan(
      const std::string& aggregate,
      bool withFinal,
      core::PlanNodeId& partialNodeId) {
    PlanBuilder builder;
    builder.values(data_)
        .partialAggregation({"k"}, {aggregate})
        .capturePlanNodeId(partialNodeId);
    if (withFinal) {
      builder.finalAggregation();
    }
    return builder.planNode();
  }

  std::shared_ptr<Task> makeTask(const core::PlanNodePtr& plan) {
    return Task::create(
        "t",
        core::PlanFragment{plan},
        0,
        core::QueryCtx::create(
            executor_.get(),
            core::QueryConfig({
                {core::QueryConfig::kAbandonPartialAggregationMinRows, "1"},
                {core::QueryConfig::kAbandonPartialAggregationMinPct, "0"},
            })),
        Task::ExecutionMode::kSerial,
        Consumer{});
  }

  std::vector<RowVectorPtr> data_;
};

std::unique_ptr<ApproxPercentileBenchmark> benchmark;

BENCHMARK(partialUnweighted) {
  benchmark->run(ApproxPercentileBenchmark::kUnweighted, false);
}

BENCHMARK(partialWeighted) {
  benchmark->run(ApproxPercentileBenchmark::kWeighted, false);
}

BENCHMARK(partialFinalUnweighted) {
  benchmark->run(ApproxPercentileBenchmark::kUnweighted, true);
}

BENCHMARK(partialFinalWeighted) {
  benchmark->run(ApproxPercentileBenchmark::kWeighted, true);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  OperatorTestBase::SetUpTestCase();
  benchmark = std::make_unique<ApproxPercentileBenchmark>();
  benchmark->verify();
  folly::runBenchmarks();
  benchmark.reset();
  OperatorTestBase::TearDownTestCase();
  return 0;
}
