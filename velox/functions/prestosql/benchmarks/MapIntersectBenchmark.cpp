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

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize({});

  ExpressionBenchmarkBuilder benchmarkBuilder;
  functions::prestosql::registerAllScalarFunctions();

  auto* pool = benchmarkBuilder.pool();
  auto& vm = benchmarkBuilder.vectorMaker();

  // ---------------------------------------------------------------------------
  // Scenario 1: Large map (100 entries), small key array (3 keys).
  // Tests the early-exit (toFind) optimization — stops after finding 3 matches
  // instead of scanning all 100 entries.
  // ---------------------------------------------------------------------------
  {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.containerLength = 100;
    options.containerVariableLength = false;
    options.complexElementsMaxSize = 1'000'000;
    VectorFuzzer fuzzer(options, pool);

    auto mapVector = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));

    // Small constant key array: 3 keys.
    auto keysVector = vm.arrayVector<int32_t>(
        std::vector<std::vector<int32_t>>(1'000, {1, 50, 99}));

    benchmarkBuilder
        .addBenchmarkSet(
            "large_map_small_keys_int", vm.rowVector({mapVector, keysVector}))
        .addExpression("map_intersect", "map_intersect(c0, c1)")
        .addExpression("map_subset", "map_subset(c0, c1)");
  }

  // ---------------------------------------------------------------------------
  // Scenario 2: Small map (5 entries), large key array (50 keys).
  // Tests the hash set reserve() optimization — pre-allocates for 50 keys.
  // ---------------------------------------------------------------------------
  {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.containerLength = 5;
    options.containerVariableLength = false;
    options.complexElementsMaxSize = 1'000'000;
    VectorFuzzer fuzzer(options, pool);

    auto mapVector = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));

    std::vector<int32_t> manyKeys(50);
    std::iota(manyKeys.begin(), manyKeys.end(), 0);
    auto keysVector = vm.arrayVector<int32_t>(
        std::vector<std::vector<int32_t>>(1'000, manyKeys));

    benchmarkBuilder
        .addBenchmarkSet(
            "small_map_large_keys_int", vm.rowVector({mapVector, keysVector}))
        .addExpression("map_intersect", "map_intersect(c0, c1)")
        .addExpression("map_subset", "map_subset(c0, c1)");
  }

  // ---------------------------------------------------------------------------
  // Scenario 3: Medium map (20 entries), medium key array (10 keys).
  // Balanced workload — tests combined effect of all optimizations.
  // ---------------------------------------------------------------------------
  {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.containerLength = 20;
    options.containerVariableLength = false;
    options.complexElementsMaxSize = 1'000'000;
    VectorFuzzer fuzzer(options, pool);

    auto mapVector = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));

    std::vector<int32_t> mediumKeys(10);
    std::iota(mediumKeys.begin(), mediumKeys.end(), 0);
    auto keysVector = vm.arrayVector<int32_t>(
        std::vector<std::vector<int32_t>>(1'000, mediumKeys));

    benchmarkBuilder
        .addBenchmarkSet(
            "medium_map_medium_keys_int", vm.rowVector({mapVector, keysVector}))
        .addExpression("map_intersect", "map_intersect(c0, c1)")
        .addExpression("map_subset", "map_subset(c0, c1)");
  }

  // ---------------------------------------------------------------------------
  // Scenario 4: VARCHAR keys — large map, small key array.
  // Tests the varchar-specific optimizations (inline string, F14FastSet).
  // ---------------------------------------------------------------------------
  {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.containerLength = 50;
    options.containerVariableLength = false;
    options.stringLength = 10;
    options.complexElementsMaxSize = 1'000'000;
    VectorFuzzer fuzzer(options, pool);

    auto mapVector = fuzzer.fuzz(MAP(VARCHAR(), INTEGER()));

    auto keysVector =
        vm.arrayVector<StringView>(std::vector<std::vector<StringView>>(
            1'000, {StringView("abc"), StringView("xyz"), StringView("foo")}));

    benchmarkBuilder
        .addBenchmarkSet(
            "large_map_small_keys_varchar",
            vm.rowVector({mapVector, keysVector}))
        .addExpression("map_intersect", "map_intersect(c0, c1)")
        .addExpression("map_subset", "map_subset(c0, c1)");
  }

  // ---------------------------------------------------------------------------
  // Scenario 5: Empty key array — tests the early-exit guard optimization.
  // ---------------------------------------------------------------------------
  {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.containerLength = 50;
    options.containerVariableLength = false;
    options.complexElementsMaxSize = 1'000'000;
    VectorFuzzer fuzzer(options, pool);

    auto mapVector = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));

    auto keysVector = vm.arrayVector<int32_t>(
        std::vector<std::vector<int32_t>>(1'000, std::vector<int32_t>{}));

    benchmarkBuilder
        .addBenchmarkSet(
            "empty_keys_int", vm.rowVector({mapVector, keysVector}))
        .addExpression("map_intersect", "map_intersect(c0, c1)")
        .addExpression("map_subset", "map_subset(c0, c1)");
  }

  // ---------------------------------------------------------------------------
  // Scenario 6: Constant keys (compile-time known).
  // Tests the constant-key caching optimization in initialize().
  // ---------------------------------------------------------------------------
  {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.containerLength = 50;
    options.containerVariableLength = false;
    options.complexElementsMaxSize = 1'000'000;
    VectorFuzzer fuzzer(options, pool);

    auto mapVector = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));

    benchmarkBuilder
        .addBenchmarkSet("constant_keys_int", vm.rowVector({mapVector}))
        .addExpression(
            "map_intersect",
            "map_intersect(c0, array_constructor(cast(1 as integer), cast(5 as integer), cast(10 as integer), cast(25 as integer), cast(42 as integer)))")
        .addExpression(
            "map_subset",
            "map_subset(c0, array_constructor(cast(1 as integer), cast(5 as integer), cast(10 as integer), cast(25 as integer), cast(42 as integer)))");
  }

  benchmarkBuilder.registerBenchmarks();
  benchmarkBuilder.testBenchmarks();
  folly::runBenchmarks();
  return 0;
}
