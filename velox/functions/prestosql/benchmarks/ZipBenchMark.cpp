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
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/SimpleFunctions.h"
#include "velox/functions/prestosql/VectorFunctions.h"
#include "velox/expression/tests/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions;

namespace {

class ZipBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  ZipBenchmark() : FunctionBenchmarkBase() {
    functions::registerFunctions();
    functions::registerVectorFunctions();
  }

  void runZip() {
    folly::BenchmarkSuspender suspender;

    VectorFuzzer::Options opts;
    opts.vectorSize = 1'000;
    VectorFuzzer fuzzer(opts, execCtx_.pool());
    auto vectorFirst = fuzzer.fuzzFlat(BIGINT());
    auto vectorSecond = fuzzer.fuzzFlat(DOUBLE());
    auto vectorThird = fuzzer.fuzzFlat(VARCHAR());

    auto rowVector = vectorMaker_.rowVector({vectorFirst, vectorSecond, vectorThird});
    auto exprSet = compileExpression("zip(c0, c1, c2)", rowVector->type());

    suspender.dismiss();
    doRun(exprSet, rowVector);
  }

  void doRun(ExprSet& exprSet, const RowVectorPtr& rowVector) {
    uint32_t cnt = 0;
    for (auto i = 0; i < 100; i++) {
      cnt += evaluate(exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

BENCHMARK(zip) {
  ZipBenchmark benchmark;
  benchmark.runZip();
}
}

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  return 0;
}