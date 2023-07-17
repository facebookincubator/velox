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

#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

class AverageAggregateBenchmark
    : public functions::test::FunctionBenchmarkBase {
 public:
  static constexpr vector_size_t size = 1'000;
  AverageAggregateBenchmark() : FunctionBenchmarkBase() {
    aggregate::prestosql::registerAllAggregateFunctions();
  }

  RowVectorPtr makeFlatData() {
    return vectorMaker_.rowVector(
        {vectorMaker_.flatVector<int32_t>(
             size, [](auto row) { return row % 7; }),
         vectorMaker_.flatVector<int64_t>(
             size,
             [](auto row) { return row % 5; },
             [](auto row) { return row % 23 == 0; }),
         vectorMaker_.flatVector<double>(
             size,
             [](auto row) { return row % 5 + 0.1; },
             [](auto row) { return row % 37 == 0; })});
  }

  RowVectorPtr makeConstantData() {
    auto c1 = vectorMaker_.flatVector<int64_t>(
        size,
        [](auto row) { return row % 5; },
        [](auto row) { return row % 23 == 0; });
    auto c2 = vectorMaker_.flatVector<double>(
        size,
        [](auto row) { return row % 5 + 0.1; },
        [](auto row) { return row % 37 == 0; });
    return vectorMaker_.rowVector(
        {vectorMaker_.flatVector<int32_t>(
             size, [](auto row) { return row % 7; }),
         BaseVector::wrapInConstant(size, size / 2, c1),
         BaseVector::wrapInConstant(size, size / 2, c2)});
  }

  RowVectorPtr makeDictionaryData() {
    auto indices = facebook::velox::test::makeIndices(
        size, [](auto row) { return row; }, pool());
    auto nulls = facebook::velox::allocateNulls(size, pool());
    auto rawNulls = nulls->asMutable<uint64_t>();
    for (auto i = 0; i < size; i += 4) {
      bits::setNull(rawNulls, i, true);
    }

    auto c1 = vectorMaker_.flatVector<int64_t>(
        size,
        [](auto row) { return row % 5; },
        [](auto row) { return row % 23 == 0; });
    auto c2 = vectorMaker_.flatVector<double>(
        size,
        [](auto row) { return row % 5 + 0.1; },
        [](auto row) { return row % 37 == 0; });
    return vectorMaker_.rowVector(
        {vectorMaker_.flatVector<int32_t>(
             size, [](auto row) { return row % 7; }),
         BaseVector::wrapInDictionary(nulls, indices, size, c1),
         BaseVector::wrapInDictionary(nulls, indices, size, c2)});
  }

  RowVectorPtr makeData(VectorEncoding::Simple encoding) {
    switch (encoding) {
      case VectorEncoding::Simple::FLAT:
        return makeFlatData();
      case VectorEncoding::Simple::CONSTANT:
        return makeConstantData();
      case VectorEncoding::Simple::DICTIONARY:
        return makeDictionaryData();
      default:
        VELOX_UNREACHABLE();
    }
  }

  void doRun(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      VectorEncoding::Simple encoding) {
    folly::BenchmarkSuspender suspender;

    const int32_t kLoopSize = 10'000;
    std::vector<RowVectorPtr> values;
    for (auto i = 0; i < kLoopSize; ++i) {
      values.push_back(makeData(encoding));
    }
    auto plan = exec::test::PlanBuilder(pool())
                    .values(values)
                    .partialAggregation(groupingKeys, aggregates)
                    .intermediateAggregation()
                    .finalAggregation()
                    .planNode();

    std::shared_ptr<folly::Executor> executor{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};
    auto task = exec::Task::create(
        "avg aggregate benchmark task",
        core::PlanFragment{plan, core::ExecutionStrategy::kUngrouped, 1, {}},
        0,
        std::make_shared<core::QueryCtx>(executor.get()));

    suspender.dismiss();

    int cnt = 0;
    auto result = task->next();
    while (result != nullptr) {
      cnt += result->size();
      result = task->next();
    }
    folly::doNotOptimizeAway(cnt);
  }

  void runWithFlatInput() {
    doRun({"c0"}, {"avg(c1)", "avg(c2)"}, VectorEncoding::Simple::FLAT);
  }

  void runWithConstantInput() {
    doRun({"c0"}, {"avg(c1)", "avg(c2)"}, VectorEncoding::Simple::CONSTANT);
  }

  void runWithDictionaryInput() {
    doRun({"c0"}, {"avg(c1)", "avg(c2)"}, VectorEncoding::Simple::DICTIONARY);
  }
};

BENCHMARK(flat) {
  AverageAggregateBenchmark benchmark;
  benchmark.runWithFlatInput();
}

BENCHMARK_RELATIVE(constant) {
  AverageAggregateBenchmark benchmark;
  benchmark.runWithConstantInput();
}

BENCHMARK_RELATIVE(dictionary) {
  AverageAggregateBenchmark benchmark;
  benchmark.runWithDictionaryInput();
}

} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  folly::runBenchmarks();
  return 0;
}
