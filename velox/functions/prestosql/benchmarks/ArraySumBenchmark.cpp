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
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

template <typename T>
struct udf_array_sum {
  VELOX_DEFINE_FUNCTION_TYPES(T)
  template <typename IT, typename OT>
  FOLLY_ALWAYS_INLINE bool call(OT& out, const arg_type<Array<IT>>& array) {
    if (array.mayHaveNulls()) {
      bool allNulls = true;
      OT sum = 0;
      for (const auto& item : array) {
        if (item.has_value()) {
          allNulls = false;
          sum += *item;
        }
      }
      if (allNulls) {
        return false;
      }
      out = sum;
      return true;
    }

    // Not nulls path
    OT sum = 0;
    for (const auto& item : array) {
      sum += *item;
    }
    out = sum;
    return true;
  }
};

class ArraySumBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  ArraySumBenchmark() : FunctionBenchmarkBase() {
    functions::prestosql::registerArrayFunctions();
    functions::prestosql::registerGeneralFunctions();

    registerFunction<
        udf_array_sum,
        int64_t,
        facebook::velox::Array<int32_t>>({"array_sum_alt"});
  }

  void runInteger(const std::string& functionName) {
    folly::BenchmarkSuspender suspender;
    vector_size_t size = 1'000;
    auto arrayVector = vectorMaker_.arrayVector<int32_t>(
        size,
        [](auto row) { return row % 5; },
        [](auto row) { return row % 23; });

    auto rowVector = vectorMaker_.rowVector({arrayVector});
    auto exprSet = compileExpression(
        fmt::format("{}(c0)", functionName), rowVector->type());
    suspender.dismiss();

    doRun(exprSet, rowVector);
  }

  void doRun(ExprSet& exprSet, const RowVectorPtr& rowVector) {
    int cnt = 0;
    for (auto i = 0; i < 100; i++) {
      cnt += evaluate(exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

BENCHMARK(vectorSimpleFunction) {
  ArraySum benchmark;
  benchmark.runInteger("array_sum_alt");
}

BENCHMARK_RELATIVE(vectorFunctionInteger) {
  ArraySum benchmark;
  benchmark.runInteger("array_sum");
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  return 0;
}