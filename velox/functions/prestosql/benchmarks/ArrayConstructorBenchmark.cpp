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

#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions;

namespace {

class ArrayConstructorBenchmark
    : public functions::test::FunctionBenchmarkBase {
 public:
  ArrayConstructorBenchmark() : FunctionBenchmarkBase() {
    functions::prestosql::registerArrayFunctions();
  }

  void run(size_t numArgs, bool everyOther) {
    folly::BenchmarkSuspender suspender;
    vector_size_t size = 10'000;
    std::vector<VectorPtr> args;
    std::vector<std::string> argNames;
    for (int i = 0; i < numArgs; i++) {
      args.push_back(vectorMaker_.flatVector<int32_t>(
          size, [](vector_size_t row) { return row; }));
      argNames.push_back(fmt::format("c{}", i));
    }

    auto rowVector = vectorMaker_.rowVector(args);
    auto exprSet = compileExpression(
        fmt::format("array_constructor({})", folly::join(",", argNames)),
        rowVector->type());

    SelectivityVector rows(size);
    if (everyOther) {
      for (int i = 0; i < size; i += 2) {
        rows.setValid(i, false);
      }
      rows.updateBounds();
    }

    suspender.dismiss();

    doRun(exprSet, rowVector, rows);
  }

  void doRun(
      ExprSet& exprSet,
      const RowVectorPtr& rowVector,
      const SelectivityVector& rows) {
    int cnt = 0;
    for (auto i = 0; i < 100; i++) {
      cnt += evaluate(exprSet, rowVector, rows)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

BENCHMARK(NoArgs) {
  ArrayConstructorBenchmark benchmark;
  benchmark.run(0, false);
}

BENCHMARK(NoArgsEveryOther) {
  ArrayConstructorBenchmark benchmark;
  benchmark.run(0, true);
}

BENCHMARK(OneArg) {
  ArrayConstructorBenchmark benchmark;
  benchmark.run(1, false);
}

BENCHMARK(OneArgEveryOther) {
  ArrayConstructorBenchmark benchmark;
  benchmark.run(1, true);
}

BENCHMARK(SeveralArgs) {
  ArrayConstructorBenchmark benchmark;
  benchmark.run(3, false);
}

BENCHMARK(SeveralArgsEveryOther) {
  ArrayConstructorBenchmark benchmark;
  benchmark.run(3, true);
}
} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv);

  folly::runBenchmarks();
  return 0;
}
