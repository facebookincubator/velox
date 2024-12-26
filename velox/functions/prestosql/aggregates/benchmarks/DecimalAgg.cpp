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

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int64(fuzzer_seed, 99887766, "Seed for random input dataset generator");

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

static constexpr int32_t kNumVectors = 10;
static constexpr int32_t kRowsPerVector = 10'000;

namespace {

class DecimalAggBenchmark : public AggregationTestBase {
 public:
  void TestBody() override {}

  static inline const std::string kSum = "sum(c0)";
  static inline const std::string kAvg = "avg(c0)";

  void runSumAndAvg(const TypePtr& decimalType) {
    run(decimalType);
  }

 private:
  void run(const TypePtr& decimalType) {
    folly::BenchmarkSuspender suspender;

    auto plan = makePlan(decimalType);
    AssertQueryBuilder assertQueryBuilder(plan);
    suspender.dismiss();

    vector_size_t numResultRows = 0;
    auto result = assertQueryBuilder.copyResults(pool());
    numResultRows += result->size();

    folly::doNotOptimizeAway(numResultRows);
  }

  core::PlanNodePtr makePlan(const TypePtr& decimalType) {
    auto rowType = ROW({"c0", "c1"}, {decimalType, BOOLEAN()});

    VectorFuzzer::Options opts;
    opts.vectorSize = kRowsPerVector;
    opts.nullRatio = 0;
    VectorFuzzer fuzzer(opts, pool(), FLAGS_fuzzer_seed);

    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < kNumVectors; ++i) {
      vectors.emplace_back(fuzzer.fuzzInputRow(rowType));
    }
    return PlanBuilder()
        .values(vectors)
        .partialAggregation({"c1"}, {kSum, kSum, kAvg, kAvg})
        .finalAggregation()
        .planNode();
  }
};

std::unique_ptr<DecimalAggBenchmark> benchmark;

BENCHMARK(decimal_5) {
  benchmark->runSumAndAvg(DECIMAL(5, 2));
}

BENCHMARK(decimal_10) {
  benchmark->runSumAndAvg(DECIMAL(10, 2));
}

BENCHMARK(decimal_20) {
  benchmark->runSumAndAvg(DECIMAL(20, 2));
}

BENCHMARK(decimal_30) {
  benchmark->runSumAndAvg(DECIMAL(30, 2));
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  OperatorTestBase::SetUpTestCase();
  benchmark = std::make_unique<DecimalAggBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  OperatorTestBase::TearDownTestCase();
  return 0;
}
