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
#include <folly/String.h>
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

// Measures map_except(c0, c1), which removes the keys listed in c1 from the map
// c0. Every row of c0 has the keys [0, mapSize) in a shuffled order. Every row
// of c1 lists `numExcluded` keys, of which a fraction `hitFraction` are in c0
// and the rest are not.
//
// Case names encode the map size, the number of excluded keys, and whether none
// or all of them are in the map. Prefixes mark variations:
//  - `const_`: the excluded keys are a constant array in the expression.
//  - `array_`: ARRAY(BIGINT) values instead of REAL.
//  - `varchar_`: VARCHAR keys instead of BIGINT.
//  - `dict_`: maps wrapped in a dictionary, as a filter produces.
//
// Every case reports time per row.

DEFINE_int32(batch_size, 1'000, "Number of rows per batch");
DEFINE_int32(seed, 42, "Random seed for key shuffling and dictionary indices");

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

enum class KeyKind { kBigint, kVarchar };

struct CaseSpec {
  int mapSize;
  int numExcluded;
  // Fraction of the excluded keys that are in the map.
  double hitFraction;
  bool constantKeys{false};
  KeyKind keyKind{KeyKind::kBigint};
  // ARRAY(BIGINT) values instead of REAL.
  bool arrayValues{false};
  bool dictionaryEncoded{false};

  bool operator==(const CaseSpec&) const = default;
};

std::string keyName(int64_t key) {
  return fmt::format("key_{:06d}", key);
}

// Returns the excluded keys. The ones in the map are spread evenly over
// [0, mapSize); the others start at mapSize.
std::vector<int64_t> makeExcludedKeys(const CaseSpec& spec) {
  const int numHits{static_cast<int>(spec.numExcluded * spec.hitFraction)};
  std::vector<int64_t> keys;
  keys.reserve(spec.numExcluded);
  for (int i = 0; i < numHits; ++i) {
    keys.push_back(static_cast<int64_t>(i) * spec.mapSize / numHits);
  }
  for (int i = numHits; i < spec.numExcluded; ++i) {
    keys.push_back(spec.mapSize + i);
  }
  return keys;
}

// Returns a vector of `size` keys, where `keyAt` gives the key at each index.
template <typename TKeyAt>
VectorPtr makeKeyVector(
    test::VectorMaker& maker,
    KeyKind keyKind,
    vector_size_t size,
    TKeyAt keyAt) {
  if (keyKind == KeyKind::kBigint) {
    return maker.flatVector<int64_t>(size, keyAt);
  }
  // flatVector() copies each string before asking for the next one.
  std::string name;
  return maker.flatVector<StringView>(size, [&](auto index) {
    name = keyName(keyAt(index));
    return StringView(name);
  });
}

// Returns an array of the excluded keys as an expression literal.
std::string makeKeysLiteral(const CaseSpec& spec) {
  std::vector<std::string> literals;
  for (auto key : makeExcludedKeys(spec)) {
    literals.push_back(
        spec.keyKind == KeyKind::kBigint ? std::to_string(key)
                                         : fmt::format("'{}'", keyName(key)));
  }
  return fmt::format("array_constructor({})", folly::join(", ", literals));
}

class MapExceptBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  MapExceptBenchmark() {
    functions::prestosql::registerAllScalarFunctions();
  }

  // Evaluates map_except over one batch 'times' times. Returns the number of
  // rows evaluated, so folly reports time per row.
  unsigned run(const CaseSpec& spec, unsigned times) {
    folly::BenchmarkSuspender suspender;
    prepare(spec);
    const auto expression = fmt::format(
        "map_except(c0, {})", spec.constantKeys ? makeKeysLiteral(spec) : "c1");
    auto exprSet = compileExpression(expression, asRowType(data_->type()));
    suspender.dismiss();

    unsigned numRows{0};
    for (unsigned i = 0; i < times; ++i) {
      numRows += evaluate(exprSet, data_)->size();
    }

    suspender.rehire();
    return numRows;
  }

 private:
  VectorPtr makeMaps(const CaseSpec& spec) {
    const vector_size_t numRows = FLAGS_batch_size;
    const vector_size_t numEntries = numRows * spec.mapSize;

    std::vector<int64_t> mapKeys(spec.mapSize);
    std::iota(mapKeys.begin(), mapKeys.end(), 0);
    std::default_random_engine gen(FLAGS_seed);
    std::shuffle(mapKeys.begin(), mapKeys.end(), gen);

    auto offsets = allocateOffsets(numRows, pool());
    auto sizes = allocateSizes(numRows, pool());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < numRows; ++row) {
      rawOffsets[row] = row * spec.mapSize;
      rawSizes[row] = spec.mapSize;
    }

    auto keys =
        makeKeyVector(maker(), spec.keyKind, numEntries, [&](auto index) {
          return mapKeys[index % spec.mapSize];
        });
    VectorPtr values;
    if (spec.arrayValues) {
      values = maker().arrayVector<int64_t>(
          numEntries,
          [](auto /*row*/) { return 3; },
          [](auto index) { return index; });
    } else {
      values = maker().flatVector<float>(
          numEntries, [](auto index) { return static_cast<float>(index); });
    }

    auto mapType = MAP(keys->type(), values->type());
    return std::make_shared<MapVector>(
        pool(),
        std::move(mapType),
        nullptr,
        numRows,
        std::move(offsets),
        std::move(sizes),
        std::move(keys),
        std::move(values));
  }

  VectorPtr makeExcludedKeyArrays(const CaseSpec& spec) {
    const vector_size_t numRows = FLAGS_batch_size;
    const auto excludedKeys = makeExcludedKeys(spec);

    auto offsets = allocateOffsets(numRows, pool());
    auto sizes = allocateSizes(numRows, pool());
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < numRows; ++row) {
      rawOffsets[row] = row * spec.numExcluded;
      rawSizes[row] = spec.numExcluded;
    }

    auto elements = makeKeyVector(
        maker(), spec.keyKind, numRows * spec.numExcluded, [&](auto index) {
          return excludedKeys[index % spec.numExcluded];
        });
    auto arrayType = ARRAY(elements->type());
    return std::make_shared<ArrayVector>(
        pool(),
        std::move(arrayType),
        nullptr,
        numRows,
        std::move(offsets),
        std::move(sizes),
        std::move(elements));
  }

  void prepare(const CaseSpec& spec) {
    if (spec_ == spec) {
      return;
    }
    data_.reset();

    auto maps = makeMaps(spec);
    if (spec.dictionaryEncoded) {
      // The excluded keys stay flat: the expression evaluator peels a
      // dictionary when every other argument is constant, and map_except would
      // then see the unwrapped maps.
      VELOX_CHECK(!spec.constantKeys);
      VectorFuzzer fuzzer({}, pool(), FLAGS_seed);
      const auto numRows = maps->size();
      maps = BaseVector::wrapInDictionary(
          nullptr, fuzzer.fuzzIndices(numRows, numRows), numRows, maps);
    }
    data_ =
        maker().rowVector({"c0", "c1"}, {maps, makeExcludedKeyArrays(spec)});
    spec_ = spec;
  }

  // Case whose inputs are in 'data_'. Folly runs every sample of one benchmark
  // before starting the next, so caching only the latest case keeps input
  // generation out of the timed region and holds one case in memory at a time.
  std::optional<CaseSpec> spec_;
  RowVectorPtr data_;
};

std::unique_ptr<MapExceptBenchmark> benchmark;

// Map size 100.
BENCHMARK_MULTI(map100_exclude3_noHit, n) {
  return benchmark->run(
      {.mapSize = 100, .numExcluded = 3, .hitFraction = 0.0}, n);
}

BENCHMARK_MULTI(map100_exclude3_allHit, n) {
  return benchmark->run(
      {.mapSize = 100, .numExcluded = 3, .hitFraction = 1.0}, n);
}

BENCHMARK_MULTI(map100_exclude30_noHit, n) {
  return benchmark->run(
      {.mapSize = 100, .numExcluded = 30, .hitFraction = 0.0}, n);
}

BENCHMARK_MULTI(map100_exclude30_allHit, n) {
  return benchmark->run(
      {.mapSize = 100, .numExcluded = 30, .hitFraction = 1.0}, n);
}

// Map size 1000.
BENCHMARK_MULTI(map1000_exclude3_noHit, n) {
  return benchmark->run(
      {.mapSize = 1'000, .numExcluded = 3, .hitFraction = 0.0}, n);
}

BENCHMARK_MULTI(map1000_exclude3_allHit, n) {
  return benchmark->run(
      {.mapSize = 1'000, .numExcluded = 3, .hitFraction = 1.0}, n);
}

BENCHMARK_MULTI(map1000_exclude30_noHit, n) {
  return benchmark->run(
      {.mapSize = 1'000, .numExcluded = 30, .hitFraction = 0.0}, n);
}

BENCHMARK_MULTI(map1000_exclude30_allHit, n) {
  return benchmark->run(
      {.mapSize = 1'000, .numExcluded = 30, .hitFraction = 1.0}, n);
}

BENCHMARK_DRAW_LINE();

// Constant excluded keys.
BENCHMARK_MULTI(const_map100_exclude3_allHit, n) {
  return benchmark->run(
      {.mapSize = 100,
       .numExcluded = 3,
       .hitFraction = 1.0,
       .constantKeys = true},
      n);
}

BENCHMARK_MULTI(const_map100_exclude30_allHit, n) {
  return benchmark->run(
      {.mapSize = 100,
       .numExcluded = 30,
       .hitFraction = 1.0,
       .constantKeys = true},
      n);
}

BENCHMARK_MULTI(const_map1000_exclude3_allHit, n) {
  return benchmark->run(
      {.mapSize = 1'000,
       .numExcluded = 3,
       .hitFraction = 1.0,
       .constantKeys = true},
      n);
}

BENCHMARK_MULTI(const_map1000_exclude30_allHit, n) {
  return benchmark->run(
      {.mapSize = 1'000,
       .numExcluded = 30,
       .hitFraction = 1.0,
       .constantKeys = true},
      n);
}

BENCHMARK_DRAW_LINE();

// ARRAY(BIGINT) values.
BENCHMARK_MULTI(array_map100_exclude3_allHit, n) {
  return benchmark->run(
      {.mapSize = 100,
       .numExcluded = 3,
       .hitFraction = 1.0,
       .arrayValues = true},
      n);
}

BENCHMARK_MULTI(array_map1000_exclude30_allHit, n) {
  return benchmark->run(
      {.mapSize = 1'000,
       .numExcluded = 30,
       .hitFraction = 1.0,
       .arrayValues = true},
      n);
}

BENCHMARK_DRAW_LINE();

// VARCHAR keys.
BENCHMARK_MULTI(varchar_map100_exclude3_allHit, n) {
  return benchmark->run(
      {.mapSize = 100,
       .numExcluded = 3,
       .hitFraction = 1.0,
       .keyKind = KeyKind::kVarchar},
      n);
}

BENCHMARK_MULTI(varchar_map100_exclude30_allHit, n) {
  return benchmark->run(
      {.mapSize = 100,
       .numExcluded = 30,
       .hitFraction = 1.0,
       .keyKind = KeyKind::kVarchar},
      n);
}

BENCHMARK_DRAW_LINE();

// Dictionary-encoded maps.
BENCHMARK_MULTI(dict_map100_exclude3_allHit, n) {
  return benchmark->run(
      {.mapSize = 100,
       .numExcluded = 3,
       .hitFraction = 1.0,
       .dictionaryEncoded = true},
      n);
}

BENCHMARK_MULTI(dict_map1000_exclude30_allHit, n) {
  return benchmark->run(
      {.mapSize = 1'000,
       .numExcluded = 30,
       .hitFraction = 1.0,
       .dictionaryEncoded = true},
      n);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  benchmark = std::make_unique<MapExceptBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
