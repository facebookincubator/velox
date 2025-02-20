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

#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int64(fuzzer_seed, 99887766, "Seed for random input dataset generator");

DEFINE_int32(num_vectors, 1000, "Number of vectors to generate");

DEFINE_int32(rows_per_vector, 1024, "Number of rows per vector");

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::exec::test;

namespace {

class VarianceAggBenchmark : public HiveConnectorTestBase {
 public:
  VarianceAggBenchmark() {
    HiveConnectorTestBase::SetUp();

    inputType_ = ROW({
        {"i32", INTEGER()},
        {"double", DOUBLE()},
    });

    VectorFuzzer::Options opts;
    opts.vectorSize = FLAGS_rows_per_vector;
    opts.nullRatio = 0.0;
    VectorFuzzer fuzzer(opts, pool(), FLAGS_fuzzer_seed);

    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < FLAGS_num_vectors; ++i) {
      vectors.emplace_back(fuzzer.fuzzInputRow(inputType_));
    }

    filePath_ = TempFilePath::create();
    writeToFile((filePath_->getPath()), vectors);
  }

  ~VarianceAggBenchmark() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  static inline const std::string kStddevSamp = "stddev_samp";
  static inline const std::string kStddevPop = "stddev_pop";
  static inline const std::string kVarSamp = "var_samp";
  static inline const std::string kVarPop = "var_pop";

  void runGlobalAgg(const std::string& aggregate, const std::string& input) {
    runGlobal(fmt::format("{}({})", aggregate, input));
  }

 private:
  void runGlobal(const std::string& aggregate) {
    folly::BenchmarkSuspender suspender;

    auto plan = makeGlobalPlan(aggregate);
    auto task = makeTask(plan);
    task->addSplit(
        "0", exec::Split(makeHiveConnectorSplit(filePath_->getPath())));
    task->noMoreSplits("0");

    suspender.dismiss();

    vector_size_t numResultRows = 0;
    while (auto result = task->next()) {
      numResultRows += result->size();
    }

    folly::doNotOptimizeAway(numResultRows);
  }

  core::PlanFragment makeGlobalPlan(const std::string& aggregate) {
    return PlanBuilder()
        .tableScan(inputType_)
        .partialAggregation({}, {aggregate})
        .finalAggregation()
        .planFragment();
  }

  std::shared_ptr<exec::Task> makeTask(core::PlanFragment plan) {
    auto task = exec::Task::create(
        "t",
        std::move(plan),
        0,
        core::QueryCtx::create(executor_.get()),
        exec::Task::ExecutionMode::kSerial);
    return task;
  }

  RowTypePtr inputType_;
  std::shared_ptr<TempFilePath> filePath_;
};

std::unique_ptr<VarianceAggBenchmark> benchmark;

BENCHMARK(stddev_samp_i32_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kStddevSamp, "i32");
}

BENCHMARK(stddev_samp_double_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kStddevSamp, "double");
}

BENCHMARK(stddev_pop_i32_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kStddevPop, "i32");
}

BENCHMARK(stddev_pop_double_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kStddevPop, "double");
}
BENCHMARK(var_samp_i32_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kVarSamp, "i32");
}

BENCHMARK(var_samp_double_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kVarSamp, "double");
}

BENCHMARK(var_pop_i32_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kVarPop, "i32");
}

BENCHMARK(var_pop_double_global) {
  benchmark->runGlobalAgg(VarianceAggBenchmark::kVarPop, "double");
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  OperatorTestBase::SetUpTestCase();
  benchmark = std::make_unique<VarianceAggBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  OperatorTestBase::TearDownTestCase();
  return 0;
}
