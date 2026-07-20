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
#include <gflags/gflags.h>

#include "velox/common/memory/Memory.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <algorithm>
#include <numeric>
#include <random>

DEFINE_int32(batch_size, 10'000, "Number of rows per batch");
DEFINE_int32(num_batches, 10, "Number of batches per iteration");
DEFINE_int32(seed, 42, "Random seed for key shuffling");

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

class MapConcatBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  MapConcatBenchmark() {
    functions::prestosql::registerAllScalarFunctions();
  }

  // Generate two map vectors with configurable key overlap and run the
  // benchmark.
  //
  // overlapFraction: 0.0 = no overlap, 0.5 = half overlap, 1.0 = full overlap.
  // mapSize: number of entries per map.
  // sorted: if true, keys are in ascending order; if false, keys are shuffled.
  unsigned
  run(double overlapFraction, int mapSize, unsigned times, bool sorted) {
    folly::BenchmarkSuspender suspender;

    const int numRows{FLAGS_batch_size};

    // Build key arrays.
    std::vector<int64_t> keys1(mapSize);
    std::iota(keys1.begin(), keys1.end(), 0);

    int overlap{static_cast<int>(mapSize * overlapFraction)};
    int shift{mapSize - overlap};
    std::vector<int64_t> keys2(mapSize);
    std::iota(keys2.begin(), keys2.end(), shift);

    if (!sorted) {
      std::default_random_engine gen(FLAGS_seed);
      std::shuffle(keys1.begin(), keys1.end(), gen);
      std::shuffle(keys2.begin(), keys2.end(), gen);
    }

    // Map1: shuffled keys from [0, mapSize).
    auto map1 = maker().mapVector<int64_t, int64_t>(
        numRows,
        [&](auto /*row*/) { return mapSize; },
        [&](auto /*row*/, auto entry) { return keys1[entry]; },
        [](auto row, auto entry) { return row * 100 + entry; });

    // Map2: shuffled keys from [shift, shift + mapSize).
    auto map2 = maker().mapVector<int64_t, int64_t>(
        numRows,
        [&](auto /*row*/) { return mapSize; },
        [&](auto /*row*/, auto entry) { return keys2[entry]; },
        [](auto row, auto entry) { return row * 100 + entry + 1'000; });

    auto data = maker().rowVector({"c0", "c1"}, {map1, map2});
    auto exprSet =
        compileExpression("map_concat(c0, c1)", asRowType(data->type()));

    suspender.dismiss();

    unsigned count{0};
    for (unsigned i = 0; i < times * FLAGS_num_batches; ++i) {
      count += evaluate(exprSet, data)->size();
    }
    return count;
  }

  // Dictionary-encoded variant.  Wraps flat maps in a dictionary layer.
  unsigned
  runDict(double overlapFraction, int mapSize, unsigned times, bool sorted) {
    folly::BenchmarkSuspender suspender;

    const int numRows{FLAGS_batch_size};

    std::vector<int64_t> keys1(mapSize);
    std::iota(keys1.begin(), keys1.end(), 0);

    int overlap{static_cast<int>(mapSize * overlapFraction)};
    int shift{mapSize - overlap};
    std::vector<int64_t> keys2(mapSize);
    std::iota(keys2.begin(), keys2.end(), shift);

    if (!sorted) {
      std::default_random_engine gen(FLAGS_seed);
      std::shuffle(keys1.begin(), keys1.end(), gen);
      std::shuffle(keys2.begin(), keys2.end(), gen);
    }

    auto flatMap1 = maker().mapVector<int64_t, int64_t>(
        numRows,
        [&](auto /*row*/) { return mapSize; },
        [&](auto /*row*/, auto entry) { return keys1[entry]; },
        [](auto row, auto entry) { return row * 100 + entry; });

    auto flatMap2 = maker().mapVector<int64_t, int64_t>(
        numRows,
        [&](auto /*row*/) { return mapSize; },
        [&](auto /*row*/, auto entry) { return keys2[entry]; },
        [](auto row, auto entry) { return row * 100 + entry + 1'000; });

    // Wrap in identity dictionary encoding.
    auto indices =
        maker().flatVector<int32_t>(numRows, [](auto row) { return row; });
    auto map1 = BaseVector::wrapInDictionary(
        BufferPtr(nullptr), indices->values(), numRows, flatMap1);
    auto map2 = BaseVector::wrapInDictionary(
        BufferPtr(nullptr), indices->values(), numRows, flatMap2);

    auto data = maker().rowVector({"c0", "c1"}, {map1, map2});
    auto exprSet =
        compileExpression("map_concat(c0, c1)", asRowType(data->type()));

    suspender.dismiss();

    unsigned count{0};
    for (unsigned i = 0; i < times * FLAGS_num_batches; ++i) {
      count += evaluate(exprSet, data)->size();
    }
    return count;
  }

  // VARCHAR key variant.
  unsigned
  runVarchar(double overlapFraction, int mapSize, unsigned times, bool sorted) {
    folly::BenchmarkSuspender suspender;

    const int numRows{FLAGS_batch_size};

    // Build string key arrays.
    std::vector<std::string> keys1(mapSize);
    for (int i = 0; i < mapSize; ++i) {
      keys1[i] = fmt::format("key_{:06d}", i);
    }

    int overlap{static_cast<int>(mapSize * overlapFraction)};
    int shift{mapSize - overlap};
    std::vector<std::string> keys2(mapSize);
    for (int i = 0; i < mapSize; ++i) {
      keys2[i] = fmt::format("key_{:06d}", i + shift);
    }

    if (!sorted) {
      std::default_random_engine gen(FLAGS_seed);
      std::shuffle(keys1.begin(), keys1.end(), gen);
      std::shuffle(keys2.begin(), keys2.end(), gen);
    }

    auto map1 = maker().mapVector<StringView, int64_t>(
        numRows,
        [&](auto /*row*/) { return mapSize; },
        [&](auto /*row*/, auto entry) { return StringView(keys1[entry]); },
        [](auto row, auto entry) { return row * 100 + entry; });

    auto map2 = maker().mapVector<StringView, int64_t>(
        numRows,
        [&](auto /*row*/) { return mapSize; },
        [&](auto /*row*/, auto entry) { return StringView(keys2[entry]); },
        [](auto row, auto entry) { return row * 100 + entry + 1'000; });

    auto data = maker().rowVector({"c0", "c1"}, {map1, map2});
    auto exprSet =
        compileExpression("map_concat(c0, c1)", asRowType(data->type()));

    suspender.dismiss();

    unsigned count{0};
    for (unsigned i = 0; i < times * FLAGS_num_batches; ++i) {
      count += evaluate(exprSet, data)->size();
    }
    return count;
  }
};

// Map size 100.
BENCHMARK_MULTI(noOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.0, 100, n, false);
}

BENCHMARK_MULTI(halfOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.5, 100, n, false);
}

BENCHMARK_MULTI(fullOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(1.0, 100, n, false);
}

// Map size 1000.
BENCHMARK_MULTI(noOverlap_1000, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.0, 1'000, n, false);
}

BENCHMARK_MULTI(halfOverlap_1000, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.5, 1'000, n, false);
}

BENCHMARK_MULTI(fullOverlap_1000, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(1.0, 1'000, n, false);
}

BENCHMARK_DRAW_LINE();

// Sorted keys, map size 100.
BENCHMARK_MULTI(sorted_noOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.0, 100, n, true);
}

BENCHMARK_MULTI(sorted_halfOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.5, 100, n, true);
}

BENCHMARK_MULTI(sorted_fullOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(1.0, 100, n, true);
}

// Sorted keys, map size 1000.
BENCHMARK_MULTI(sorted_noOverlap_1000, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.0, 1'000, n, true);
}

BENCHMARK_MULTI(sorted_halfOverlap_1000, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(0.5, 1'000, n, true);
}

BENCHMARK_MULTI(sorted_fullOverlap_1000, n) {
  MapConcatBenchmark benchmark;
  return benchmark.run(1.0, 1'000, n, true);
}

BENCHMARK_DRAW_LINE();

// Dictionary-encoded maps, map size 100.
BENCHMARK_MULTI(dict_noOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.runDict(0.0, 100, n, false);
}

BENCHMARK_MULTI(dict_halfOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.runDict(0.5, 100, n, false);
}

BENCHMARK_MULTI(dict_fullOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.runDict(1.0, 100, n, false);
}

BENCHMARK_DRAW_LINE();

// VARCHAR keys, map size 100.
BENCHMARK_MULTI(varchar_noOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.runVarchar(0.0, 100, n, false);
}

BENCHMARK_MULTI(varchar_halfOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.runVarchar(0.5, 100, n, false);
}

BENCHMARK_MULTI(varchar_fullOverlap_100, n) {
  MapConcatBenchmark benchmark;
  return benchmark.runVarchar(1.0, 100, n, false);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
