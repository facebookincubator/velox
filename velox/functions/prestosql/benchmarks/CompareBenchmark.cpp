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
#include "folly/Random.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/SimpleFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

class CompareBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  CompareBenchmark() : FunctionBenchmarkBase() {
    functions::registerFunctions();
  }

  VectorPtr makeData() {
    constexpr vector_size_t size = 1000;

    return vectorMaker_.flatVector<int64_t>(
        size,
        [](auto row) { return row % 2 ? row : folly::Random::rand32() % size; },
        VectorMaker::nullEvery(5));
  }

  void run(const std::string& functionName) {
    folly::BenchmarkSuspender suspender;
    auto inputs = vectorMaker_.rowVector({makeData(), makeData()});

    auto exprSet = compileExpression(
        fmt::format("c0 {} c1", functionName), inputs->type());
    suspender.dismiss();

    doRun(exprSet, inputs);
  }

  void doRun(ExprSet& exprSet, const RowVectorPtr& rowVector) {
    int cnt = 0;
    for (auto i = 0; i < 100; i++) {
      cnt += evaluate(exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

BENCHMARK(Eq_Velox) {
  CompareBenchmark benchmark;
  benchmark.run("==");
}

BENCHMARK(Neq_Velox) {
  CompareBenchmark benchmark;
  benchmark.run("=");
}

BENCHMARK(Gt_Velox) {
  CompareBenchmark benchmark;
  benchmark.run(">");
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  return 0;
}
