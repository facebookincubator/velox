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
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

class MapZipWithBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  explicit MapZipWithBenchmark(uint32_t seed)
      : FunctionBenchmarkBase(), seed_{seed} {
    functions::prestosql::registerAllScalarFunctions();

    VectorFuzzer::Options options;
    options.vectorSize = 10'024;

    VectorFuzzer fuzzer(options, pool(), seed_);
    dictionaryKeysMap_ = fuzzer.fuzzMap(
        fuzzer.fuzzDictionary(
            fuzzer.fuzzFlat(INTEGER(), 100),
            options.vectorSize * options.containerLength),
        fuzzer.fuzzFlat(BIGINT(), options.vectorSize * options.containerLength),
        options.vectorSize);

    flatKeysMap_ = BaseVector::create(
        dictionaryKeysMap_->type(), options.vectorSize, pool());
    flatKeysMap_->copy(dictionaryKeysMap_.get(), 0, 0, options.vectorSize);

    auto sortedKeysMap = BaseVector::create<MapVector>(
        flatKeysMap_->type(), options.vectorSize, pool());
    sortedKeysMap->copy(flatKeysMap_.get(), 0, 0, options.vectorSize);
    MapVector::canonicalize(sortedKeysMap);
    sortedKeysMap_ = std::move(sortedKeysMap);
  }

  void test() {
    auto flatResult =
        evaluate(kBasicExpression, vectorMaker_.rowVector({flatKeysMap_}));
    auto dictionaryResult = evaluate(
        kBasicExpression, vectorMaker_.rowVector({dictionaryKeysMap_}));
    test::assertEqualVectors(flatResult, dictionaryResult);
  }

  size_t runFlatKeys(size_t times) {
    return run(flatKeysMap_, times);
  }

  size_t runSortedKeys(size_t times) {
    return run(sortedKeysMap_, times);
  }

  size_t runDictionaryKeys(size_t times) {
    return run(dictionaryKeysMap_, times);
  }

 private:
  static const std::string kBasicExpression;

  size_t run(const VectorPtr& vector, size_t times) {
    folly::BenchmarkSuspender suspender;
    auto data = vectorMaker_.rowVector({vector});
    auto exprSet = compileExpression(kBasicExpression, asRowType(data->type()));
    suspender.dismiss();

    int cnt = 0;
    for (auto i = 0; i < times * 1'000; i++) {
      cnt += evaluate(exprSet, data)->size();
    }
    return cnt;
  }

  const uint32_t seed_;
  VectorPtr flatKeysMap_;
  VectorPtr sortedKeysMap_;
  VectorPtr dictionaryKeysMap_;
};

const std::string MapZipWithBenchmark::kBasicExpression =
    "map_zip_with(c0, c0, (k, v1, v2) -> v1 + v2)";

const uint32_t seed = folly::Random::rand32();

// Use flat map with flat keys and values. No nulls.
BENCHMARK_MULTI(flatKeys, n) {
  MapZipWithBenchmark benchmark(seed);
  return benchmark.runFlatKeys(n);
}

// Use flat map with sorted keys. Keys and values will be dictionary encoded as
// a result of sorting.
BENCHMARK_MULTI(sortedKeys, n) {
  MapZipWithBenchmark benchmark(seed);
  return benchmark.runSortedKeys(n);
}

// Use flat map with dictionary encoded keys and flat values.
BENCHMARK_MULTI(dictionaryKeys, n) {
  MapZipWithBenchmark benchmark(seed);
  return benchmark.runDictionaryKeys(n);
}

} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv);

  LOG(ERROR) << "Seed: " << seed;
  {
    MapZipWithBenchmark benchmark(seed);
    benchmark.test();
  }
  folly::runBenchmarks();
  return 0;
}
