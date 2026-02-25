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

#include "velox/exec/tests/SimpleAggregateFunctionsRegistration.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

static constexpr int32_t kNumVectors = 1'000;
static constexpr int32_t kRowsPerVector = 1'000;

class SimpleVariadicSumAggregateBenchmark : public OperatorTestBase {
 public:
  SimpleVariadicSumAggregateBenchmark() {
    OperatorTestBase::SetUp();

    // Register the baseline Aggregate-based implementation (default null
    // behavior).
    facebook::velox::aggregate::registerVariadicSumAggregateDefaultNull(
        "variadic_sum_agg_default_null");

    // Register the SimpleAggregateAdapter-based implementation (default null
    // behavior).
    facebook::velox::aggregate::registerSimpleVariadicSumAggregateDefaultNull(
        "simple_variadic_sum_default_null");

    // Register the non-default-null Aggregate-based implementation.
    facebook::velox::aggregate::registerVariadicSumAggregateNonDefaultNull(
        "variadic_sum_agg_non_default_null");

    // Register the non-default-null SimpleAggregateAdapter-based
    // implementation.
    facebook::velox::aggregate::
        registerSimpleVariadicSumAggregateNonDefaultNull(
            "simple_variadic_sum_non_default_null");
  }

  ~SimpleVariadicSumAggregateBenchmark() override {
    OperatorTestBase::TearDown();
  }

  void TestBody() override {}

  std::vector<RowVectorPtr> makeData() {
    auto vector = makeRowVector(
        {// Grouping key with ~7 unique values.
         makeFlatVector<int32_t>(
             kRowsPerVector, [](auto row) { return row % 7; }),
         // Dummy argument (required by the function signature).
         makeFlatVector<int64_t>(
             kRowsPerVector, [](auto /*row*/) { return 0; }),
         // Variadic argument 1.
         makeFlatVector<int64_t>(
             kRowsPerVector,
             [](auto row) { return row % 100; },
             [](auto row) { return row % 23 == 0; }),
         // Variadic argument 2.
         makeFlatVector<int64_t>(
             kRowsPerVector,
             [](auto row) { return row % 50; },
             [](auto row) { return row % 17 == 0; }),
         // Variadic argument 3.
         makeFlatVector<int64_t>(
             kRowsPerVector,
             [](auto row) { return row % 30; },
             [](auto row) { return row % 13 == 0; })});
    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < kNumVectors; ++i) {
      vectors.push_back(vector);
    }
    return vectors;
  }

  void run(const std::string& aggregate) {
    folly::BenchmarkSuspender suspender;

    auto vectors = makeData();
    auto plan = PlanBuilder()
                    .values(vectors)
                    .partialAggregation({"c0"}, {aggregate})
                    .finalAggregation()
                    .planFragment();

    vector_size_t numResultRows = 0;
    auto task = Task::create(
        "t",
        std::move(plan),
        0,
        core::QueryCtx::create(executor_.get()),
        Task::ExecutionMode::kSerial);

    suspender.dismiss();

    while (auto result = task->next()) {
      numResultRows += result->size();
    }

    folly::doNotOptimizeAway(numResultRows);
  }

  void runVariadicSumAggregateDefaultNull() {
    run("variadic_sum_agg_default_null(c1, c2, c3, c4)");
  }

  void runSimpleVariadicSumAggregateDefaultNull() {
    run("simple_variadic_sum_default_null(c1, c2, c3, c4)");
  }

  void runVariadicSumAggregateNonDefaultNull() {
    run("variadic_sum_agg_non_default_null(c1, c2, c3, c4)");
  }

  void runSimpleVariadicSumAggregateNonDefaultNull() {
    run("simple_variadic_sum_non_default_null(c1, c2, c3, c4)");
  }
};

std::unique_ptr<SimpleVariadicSumAggregateBenchmark> benchmark;

BENCHMARK(variadicSumAggregateDefaultNull) {
  benchmark->runVariadicSumAggregateDefaultNull();
}

BENCHMARK_RELATIVE(simpleVariadicSumDefaultNull) {
  benchmark->runSimpleVariadicSumAggregateDefaultNull();
}

BENCHMARK(variadicSumAggregateNonDefaultNull) {
  benchmark->runVariadicSumAggregateNonDefaultNull();
}

BENCHMARK_RELATIVE(simpleVariadicSumNonDefaultNull) {
  benchmark->runSimpleVariadicSumAggregateNonDefaultNull();
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  OperatorTestBase::SetUpTestCase();
  benchmark = std::make_unique<SimpleVariadicSumAggregateBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  OperatorTestBase::TearDownTestCase();
  return 0;
}
