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
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <algorithm>
#include <numeric>
#include <random>

DEFINE_int32(batch_size, 10'000, "Number of rows per batch");
DEFINE_int32(seed, 42, "Random seed for key shuffling and dictionary indices");

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

enum class KeyKind { kBigint, kVarchar };

// Describes the two inputs of map_concat(c0, c1). Every row of c0 has keys
// [0, mapSize). Every row of c1 has 'mapSize' keys, of which
// 'overlapFraction' also appear in c0.
struct CaseSpec {
  double overlapFraction;
  int mapSize;
  // Keeps keys in ascending order within each row instead of shuffling them.
  bool sortedKeys{false};
  KeyKind keyKind{KeyKind::kBigint};
  // Wraps each input in its own dictionary with shuffled indices.
  bool dictionaryEncoded{false};

  bool operator==(const CaseSpec&) const = default;
};

std::pair<std::vector<int64_t>, std::vector<int64_t>> makeKeys(
    const CaseSpec& spec) {
  std::vector<int64_t> firstKeys(spec.mapSize);
  std::iota(firstKeys.begin(), firstKeys.end(), 0);

  const int overlap{static_cast<int>(spec.mapSize * spec.overlapFraction)};
  std::vector<int64_t> secondKeys(spec.mapSize);
  std::iota(secondKeys.begin(), secondKeys.end(), spec.mapSize - overlap);

  if (!spec.sortedKeys) {
    std::default_random_engine gen(FLAGS_seed);
    std::shuffle(firstKeys.begin(), firstKeys.end(), gen);
    std::shuffle(secondKeys.begin(), secondKeys.end(), gen);
  }
  return {std::move(firstKeys), std::move(secondKeys)};
}

template <typename TKey, typename TKeyAt>
VectorPtr makeMap(
    test::VectorMaker& maker,
    vector_size_t mapSize,
    int64_t valueOffset,
    TKeyAt keyAt) {
  return maker.mapVector<TKey, int64_t>(
      FLAGS_batch_size,
      [&](auto /*row*/) { return mapSize; },
      [&](auto /*row*/, auto entry) { return keyAt(entry); },
      [&](auto row, auto entry) { return row * 100 + entry + valueOffset; });
}

VectorPtr makeMapInput(
    test::VectorMaker& maker,
    const std::vector<int64_t>& keys,
    KeyKind keyKind,
    int64_t valueOffset) {
  const auto mapSize = static_cast<vector_size_t>(keys.size());
  if (keyKind == KeyKind::kBigint) {
    return makeMap<int64_t>(
        maker, mapSize, valueOffset, [&](auto entry) { return keys[entry]; });
  }

  std::vector<std::string> names;
  names.reserve(keys.size());
  for (auto key : keys) {
    names.push_back(fmt::format("key_{:06d}", key));
  }
  return makeMap<StringView>(maker, mapSize, valueOffset, [&](auto entry) {
    return StringView(names[entry]);
  });
}

class MapConcatBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  MapConcatBenchmark() {
    functions::prestosql::registerAllScalarFunctions();
  }

  // Evaluates map_concat(c0, c1) over one batch 'times' times. Returns the
  // number of rows evaluated, so folly reports time per row.
  unsigned run(const CaseSpec& spec, unsigned times) {
    folly::BenchmarkSuspender suspender;
    prepare(spec);
    auto exprSet =
        compileExpression("map_concat(c0, c1)", asRowType(data_->type()));
    suspender.dismiss();

    unsigned numRows{0};
    for (unsigned i = 0; i < times; ++i) {
      numRows += evaluate(exprSet, data_)->size();
    }

    suspender.rehire();
    return numRows;
  }

 private:
  void prepare(const CaseSpec& spec) {
    if (spec_ == spec) {
      return;
    }
    data_.reset();

    const auto [firstKeys, secondKeys] = makeKeys(spec);
    auto first = makeMapInput(maker(), firstKeys, spec.keyKind, 0);
    auto second = makeMapInput(maker(), secondKeys, spec.keyKind, 1'000);
    if (spec.dictionaryEncoded) {
      // Each input needs its own indices buffer: the expression evaluator
      // peels a dictionary shared by all arguments, and map_concat would then
      // see the unwrapped maps.
      VectorFuzzer fuzzer({}, pool(), FLAGS_seed);
      const auto numRows = first->size();
      auto wrap = [&](const VectorPtr& input) {
        return BaseVector::wrapInDictionary(
            nullptr, fuzzer.fuzzIndices(numRows, numRows), numRows, input);
      };
      first = wrap(first);
      second = wrap(second);
    }
    data_ = maker().rowVector({"c0", "c1"}, {first, second});
    spec_ = spec;
  }

  // Case whose inputs are in 'data_'. Folly runs every sample of one benchmark
  // before starting the next, so caching only the latest case keeps input
  // generation out of the timed region and holds one case in memory at a time.
  std::optional<CaseSpec> spec_;
  RowVectorPtr data_;
};

std::unique_ptr<MapConcatBenchmark> benchmark;

// Map size 100.
BENCHMARK_MULTI(noOverlap_100, n) {
  return benchmark->run({.overlapFraction = 0.0, .mapSize = 100}, n);
}

BENCHMARK_MULTI(halfOverlap_100, n) {
  return benchmark->run({.overlapFraction = 0.5, .mapSize = 100}, n);
}

BENCHMARK_MULTI(fullOverlap_100, n) {
  return benchmark->run({.overlapFraction = 1.0, .mapSize = 100}, n);
}

// Map size 1000.
BENCHMARK_MULTI(noOverlap_1000, n) {
  return benchmark->run({.overlapFraction = 0.0, .mapSize = 1'000}, n);
}

BENCHMARK_MULTI(halfOverlap_1000, n) {
  return benchmark->run({.overlapFraction = 0.5, .mapSize = 1'000}, n);
}

BENCHMARK_MULTI(fullOverlap_1000, n) {
  return benchmark->run({.overlapFraction = 1.0, .mapSize = 1'000}, n);
}

BENCHMARK_DRAW_LINE();

// Sorted keys, map size 100.
BENCHMARK_MULTI(sorted_noOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 0.0, .mapSize = 100, .sortedKeys = true}, n);
}

BENCHMARK_MULTI(sorted_halfOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 0.5, .mapSize = 100, .sortedKeys = true}, n);
}

BENCHMARK_MULTI(sorted_fullOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 1.0, .mapSize = 100, .sortedKeys = true}, n);
}

// Sorted keys, map size 1000.
BENCHMARK_MULTI(sorted_noOverlap_1000, n) {
  return benchmark->run(
      {.overlapFraction = 0.0, .mapSize = 1'000, .sortedKeys = true}, n);
}

BENCHMARK_MULTI(sorted_halfOverlap_1000, n) {
  return benchmark->run(
      {.overlapFraction = 0.5, .mapSize = 1'000, .sortedKeys = true}, n);
}

BENCHMARK_MULTI(sorted_fullOverlap_1000, n) {
  return benchmark->run(
      {.overlapFraction = 1.0, .mapSize = 1'000, .sortedKeys = true}, n);
}

BENCHMARK_DRAW_LINE();

// Dictionary-encoded maps, map size 100.
BENCHMARK_MULTI(dict_noOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 0.0, .mapSize = 100, .dictionaryEncoded = true}, n);
}

BENCHMARK_MULTI(dict_halfOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 0.5, .mapSize = 100, .dictionaryEncoded = true}, n);
}

BENCHMARK_MULTI(dict_fullOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 1.0, .mapSize = 100, .dictionaryEncoded = true}, n);
}

BENCHMARK_DRAW_LINE();

// VARCHAR keys, map size 100.
BENCHMARK_MULTI(varchar_noOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 0.0, .mapSize = 100, .keyKind = KeyKind::kVarchar},
      n);
}

BENCHMARK_MULTI(varchar_halfOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 0.5, .mapSize = 100, .keyKind = KeyKind::kVarchar},
      n);
}

BENCHMARK_MULTI(varchar_fullOverlap_100, n) {
  return benchmark->run(
      {.overlapFraction = 1.0, .mapSize = 100, .keyKind = KeyKind::kVarchar},
      n);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  benchmark = std::make_unique<MapConcatBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
