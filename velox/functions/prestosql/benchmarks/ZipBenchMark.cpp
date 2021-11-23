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
#include "velox/expression/tests/VectorFuzzer.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/VectorFunctions.h"

namespace {
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions;

class ZipBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  ZipBenchmark() : FunctionBenchmarkBase() {
    functions::registerVectorFunctions();
  }

  void runFunction() {
    folly::BenchmarkSuspender suspender;

    VectorFuzzer::Options opts;
    opts.vectorSize = 1'000;
    VectorFuzzer fuzzer(opts, execCtx_.pool());
    auto vectorFirst = fuzzer.fuzzComplex(ARRAY(BIGINT()));
    auto vectorSecond = fuzzer.fuzzComplex(ARRAY(DOUBLE()));
    auto vectorThird = fuzzer.fuzzComplex(ARRAY(VARCHAR()));

    auto rowVector =
        vectorMaker_.rowVector({vectorFirst, vectorSecond, vectorThird});
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

  template <uint32_t numArrays, bool useVectorized, uint32_t numTimes>
  void runLoop() {
    folly::BenchmarkSuspender suspender;

    VectorFuzzer::Options opts;
    opts.vectorSize = 1'000;
    VectorFuzzer fuzzer(opts, execCtx_.pool());

    std::vector<VectorPtr> inputArrays(numArrays);

    for (int i = 0; i < numArrays; i++) {
      inputArrays[i] = fuzzer.fuzzComplex(ARRAY(BIGINT()));
    }

    auto rowVector = vectorMaker_.rowVector(inputArrays);
    SelectivityVector rows(opts.vectorSize);
    exec::EvalCtx evalCtx(&execCtx_, nullptr, rowVector.get());

    exec::DecodedArgs decodedArgs(rows, inputArrays, &evalCtx);
    std::vector<const vector_size_t*> rawSizes(numArrays);
    std::vector<const vector_size_t*> indices(numArrays);

    for (int i = 0; i < numArrays; i++) {
      auto baseVector = decodedArgs.at(i)->base()->as<ArrayVector>();
      rawSizes[i] = baseVector->rawSizes();
      indices[i] = decodedArgs.at(i)->indices();
    }

    vector_size_t resultElementsSize = 0;
    BufferPtr resultArraySizesBuffer =
        AlignedBuffer::allocate<vector_size_t>(rows.end(), pool_.get(), 0);
    auto rawResultArraySizes =
        resultArraySizesBuffer->asMutable<vector_size_t>();

    suspender.dismiss();

    for (int i = 0; i < numTimes; i++) {
      if constexpr (useVectorized) {
        for (int i = 0; i < numArrays; i++) {
          rows.applyToSelected([&](auto row) {
            auto x = rawSizes[i][indices[i][row]];
            auto y = rawResultArraySizes[row];

            if (x > y) {
              rawResultArraySizes[row] = x;
            }
          });
        }

        rows.applyToSelected(
            [&](auto row) { resultElementsSize += rawResultArraySizes[row]; });
      } else {
        rows.applyToSelected([&](auto row) {
          vector_size_t maxSize = 0;
          for (int i = 0; i < numArrays; i++) {
            auto x = rawSizes[i][indices[i][row]];
            auto y = maxSize;
            if (x > y) {
              maxSize = x;
            }
          }
          resultElementsSize += maxSize;
          rawResultArraySizes[row] = maxSize;
        });
      }
    }

    folly::doNotOptimizeAway(resultElementsSize);
    folly::doNotOptimizeAway(rawResultArraySizes);
  }

  template <uint32_t numArrays, bool useLambda, uint32_t numTimes>
  void runLambdaCheck() {
    folly::BenchmarkSuspender suspender;

    VectorFuzzer::Options opts;
    opts.vectorSize = 1'000;
    opts.nullChance = 0;
    VectorFuzzer fuzzer(opts, execCtx_.pool());

    std::vector<FlatVector<int64_t>*> inputVectors(numArrays);
    std::vector<int64_t*> rawBuffers(numArrays);
    SelectivityVector rows(opts.vectorSize);

    for (int i = 0; i < numArrays; i++) {
      auto vec = fuzzer.fuzzFlat(BIGINT());
      inputVectors[i] = vec->as<FlatVector<int64_t>>();
      rawBuffers[i] = inputVectors[i]->values()->template asMutable<int64_t>();
    }

    suspender.dismiss();
    int64_t runningSum = 0;

    for (int i = 0; i < numTimes; i++) {
      if constexpr (useLambda) {
        for (int k = 0; k < numArrays; k++) {
          rows.template applyToSelected(
              [&](auto row) { runningSum += rawBuffers[k][row]; });
        }
      } else {
        auto range = rows.asRange();
        for (int k = 0; k < numArrays; k++) {
          for (auto j = range.begin(); j < range.end(); j++) {
            runningSum += rawBuffers[k][j];
          }
        }
      }
    }
    folly::doNotOptimizeAway(runningSum);
  }
};

BENCHMARK(zip) {
  ZipBenchmark benchmark;
  benchmark.runFunction();
}

BENCHMARK(vectorized_loop) {
  ZipBenchmark benchmark;
  benchmark.runLoop<5, true, 1000000>();
}

BENCHMARK_RELATIVE(default_loop) {
  ZipBenchmark benchmark;
  benchmark.runLoop<5, false, 1000000>();
}

BENCHMARK(lambda_loop) {
  ZipBenchmark benchmark;
  benchmark.runLambdaCheck<5, true, 1000000>();
}

BENCHMARK_RELATIVE(range_loop) {
  ZipBenchmark benchmark;
  benchmark.runLambdaCheck<5, false, 1000000>();
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  return 0;
}