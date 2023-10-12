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
#include <cstdint>

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  ExpressionBenchmarkBuilder benchmarkBuilder;
  facebook::velox::functions::prestosql::registerAllScalarFunctions();

  auto* pool = benchmarkBuilder.pool();
  auto& vm = benchmarkBuilder.vectorMaker();

  auto createSet = [&](const TypePtr& mapType) {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.containerLength = 20;
    options.containerVariableLength = 20;

    VectorFuzzer fuzzer(options, pool);
    std::vector<VectorPtr> columns;

    // Ratio = elements vector/ elements in base.
    auto makeMapVector = [&](auto ratio) {
      auto baseSize = options.vectorSize / ratio;
      auto flatBase = fuzzer.fuzzFlat(mapType, baseSize);
      auto dictionary = fuzzer.fuzzDictionary(flatBase, options.vectorSize);
      return dictionary;
    };

    // Fuzz input vectors.
    columns.push_back(makeMapVector(1));
    columns.push_back(makeMapVector(2));
    columns.push_back(makeMapVector(3));
    columns.push_back(makeMapVector(4));

    // Fuzz valid keys for map at columns[index].
    auto makeKeys = [&](int index) {
      DecodedVector decoded(*columns[index - 1]);
      auto* map = decoded.base()->as<MapVector>();
      auto indices = allocateIndices(1000, pool);
      auto* mutableIndices = indices->asMutable<vector_size_t>();
      for (int i = 0; i < 1000; i++) {
        int keyIndex = folly::Random::rand32() % 20;
        // We use the keyIndex as the key.
        mutableIndices[i] = keyIndex;
      }
      return BaseVector::wrapInDictionary(
          nullptr, indices, 1000, map->mapKeys());
    };

    columns.push_back(makeKeys(1));
    columns.push_back(makeKeys(2));
    columns.push_back(makeKeys(3));
    columns.push_back(makeKeys(4));

    auto indicesFlat = vm.flatVector<int64_t>(
        options.vectorSize,
        [&](auto row) { return row % options.containerLength; });
    columns.push_back(indicesFlat);
    benchmarkBuilder
        .addBenchmarkSet(
            fmt::format("map_subscript_{}", mapType->toString()),
            vm.rowVector(columns))
        .addExpression("1", "subscript(c0, c4)")
        .addExpression("2", "subscript(c1, c5)")
        .addExpression("3", "subscript(c2, c6)")
        .addExpression("4", "subscript(c3, c7)");
  };

  createSet(MAP(INTEGER(), INTEGER()));
  createSet(MAP(VARCHAR(), INTEGER()));
  createSet(MAP(ARRAY(VARCHAR()), INTEGER()));

  benchmarkBuilder.registerBenchmarks();

  folly::runBenchmarks();
  return 0;
}
