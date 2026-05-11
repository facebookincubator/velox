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
#include "velox/common/memory/Memory.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

/// Benchmarks for map_except(map, array) covering:
///   - Constant vs non-constant exclusion keys
///   - Small vs large maps (early termination benefit)
///   - Empty vs small vs large exclusion sets
///   - Bigint vs varchar key types
class MapExceptBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  MapExceptBenchmark() : FunctionBenchmarkBase() {
    functions::prestosql::registerAllScalarFunctions();
  }

  /// Creates a RowVector with columns:
  ///   c0: MAP<BIGINT, BIGINT> with mapSize entries per row (keys 0..mapSize-1)
  ///   c1: ARRAY<BIGINT> with excludeSize elements per row (keys
  ///   0..excludeSize-1)
  RowVectorPtr makeBigintData(int mapSize, int excludeSize) {
    const vector_size_t numRows = 1'000;
    auto mapVector = vectorMaker_.mapVector<int64_t, int64_t>(
        numRows,
        [mapSize](auto) { return mapSize; },
        [mapSize](auto idx) { return idx % mapSize; },
        [](auto idx) { return idx * 10; });

    auto excludeArray = vectorMaker_.arrayVector<int64_t>(
        numRows,
        [excludeSize](auto) { return excludeSize; },
        [excludeSize](auto idx) {
          return excludeSize > 0 ? idx % excludeSize : 0;
        });

    return vectorMaker_.rowVector({mapVector, excludeArray});
  }

  /// Creates a RowVector with columns:
  ///   c0: MAP<VARCHAR, BIGINT> with mapSize entries per row
  ///   c1: ARRAY<VARCHAR> with excludeSize elements per row
  RowVectorPtr makeVarcharData(int mapSize, int excludeSize) {
    const vector_size_t numRows = 1'000;
    auto mapVector = vectorMaker_.mapVector<std::string, int64_t>(
        numRows,
        [mapSize](auto) { return mapSize; },
        [mapSize](auto idx) { return std::to_string(idx % mapSize); },
        [](auto idx) { return idx * 10; });

    auto excludeArray = vectorMaker_.arrayVector<std::string>(
        numRows,
        [excludeSize](auto) { return excludeSize; },
        [excludeSize](auto idx) {
          return std::to_string(excludeSize > 0 ? idx % excludeSize : 0);
        });

    return vectorMaker_.rowVector({mapVector, excludeArray});
  }

  unsigned int run(
      const RowVectorPtr& data,
      const std::string& expression,
      unsigned int times) {
    folly::BenchmarkSuspender suspender;
    auto exprSet = compileExpression(expression, asRowType(data->type()));
    suspender.dismiss();

    unsigned int cnt = 0;
    for (unsigned int i = 0; i < times * 100; i++) {
      cnt += evaluate(exprSet, data)->size();
    }
    return cnt;
  }
};

// ============================================================================
// Bigint keys, map size 100
// Exercises: constant key optimization, out.reserve(), toExclude counter.
// ============================================================================

BENCHMARK_MULTI(bigint_map100_constExclude3, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeBigintData(100, 3);
  return benchmark.run(data, "map_except(c0, array_constructor(0, 1, 2))", n);
}

BENCHMARK_MULTI(bigint_map100_nonConstExclude3, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeBigintData(100, 3);
  return benchmark.run(data, "map_except(c0, c1)", n);
}

BENCHMARK_MULTI(bigint_map100_excludeEmpty, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeBigintData(100, 0);
  return benchmark.run(
      data, "map_except(c0, array_constructor()::bigint[])", n);
}

BENCHMARK_MULTI(bigint_map100_nonConstExclude50, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeBigintData(100, 50);
  return benchmark.run(data, "map_except(c0, c1)", n);
}

// ============================================================================
// Bigint keys, map size 1000
// Exercises: early termination — after 3 exclusion keys are found, the
// remaining ~997 entries skip the hash lookup entirely.
// ============================================================================

BENCHMARK_MULTI(bigint_map1000_constExclude3, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeBigintData(1'000, 3);
  return benchmark.run(data, "map_except(c0, array_constructor(0, 1, 2))", n);
}

BENCHMARK_MULTI(bigint_map1000_nonConstExclude3, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeBigintData(1'000, 3);
  return benchmark.run(data, "map_except(c0, c1)", n);
}

BENCHMARK_MULTI(bigint_map1000_constExclude10, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeBigintData(1'000, 10);
  return benchmark.run(
      data,
      "map_except(c0, array_constructor(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))",
      n);
}

// ============================================================================
// Varchar keys, map size 100
// Exercises: varchar fast path with inline string optimization, constant key
// caching with searchKeyStrings_ lifetime management.
// ============================================================================

BENCHMARK_MULTI(varchar_map100_constExclude3, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeVarcharData(100, 3);
  return benchmark.run(
      data, "map_except(c0, array_constructor('0', '1', '2'))", n);
}

BENCHMARK_MULTI(varchar_map100_nonConstExclude3, n) {
  MapExceptBenchmark benchmark;
  auto data = benchmark.makeVarcharData(100, 3);
  return benchmark.run(data, "map_except(c0, c1)", n);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
