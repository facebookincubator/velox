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

#include "velox/functions/lib/MakeRowFromMap.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <folly/Benchmark.h>
#include <folly/container/F14Map.h>
#include <folly/init/Init.h>

// Measures MakeRowFromMap, which projects a few keys of a map into the fields
// of a row, over feature-like maps generated with VectorFuzzer: 1,000 rows of
// BIGINT keys, and REAL or ARRAY(BIGINT) values, 5% of them null.
//
// Case names encode the number of distinct keys in the maps, the number of
// keys projected, and the percentage of rows that contain each key. Suffixes
// mark variations:
//  - `_array`: ARRAY(BIGINT) values instead of REAL.
//  - `_wrapped`: maps wrapped in a dictionary, as a filter produces.
//  - `_defaults`: nulls replaced with default values (replaceNulls).
//
// Every case reports time per row, over the same maps passed as:
//  - map: a MapVector with flat keys and values, as readers produce.
//  - toMap: a FlatMapVector, converted to a MapVector with toMapVector()
//    before projecting. Reported relative to `map`.
//  - flatMap: a FlatMapVector, projected directly. Reported relative to `map`.

DEFINE_int32(seed, 1, "Seed for the fuzzer that generates the maps");

namespace facebook::velox::functions::test {
namespace {

constexpr vector_size_t kNumRows = 1'000;

struct CaseSpec {
  // Distinct keys in the maps.
  vector_size_t numKeys;
  // Keys projected into the row.
  vector_size_t numProjected;
  // Percentage of rows that contain each key.
  int32_t density;
  bool replaceNulls{false};
  // ARRAY(BIGINT) values instead of REAL.
  bool arrayValues{false};
  // Maps wrapped in a dictionary.
  bool wrapped{false};

  std::string key() const {
    return fmt::format(
        "{}_{}_{}_{}_{}_{}",
        numKeys,
        numProjected,
        density,
        replaceNulls,
        arrayValues,
        wrapped);
  }
};

// Builds flat maps with about `numKeys` distinct keys; fewer if the fuzzer
// draws duplicates. Each row contains each key with a chance of `density`
// percent.
FlatMapVectorPtr makeFlatMap(
    memory::MemoryPool* pool,
    vector_size_t numKeys,
    int32_t density,
    const TypePtr& valueType) {
  VectorFuzzer::Options keyOptions;
  // fuzzUniqueFlatNotNull() draws `containerLength` values per requested key.
  keyOptions.containerLength = 1;
  auto distinctKeys = VectorFuzzer(keyOptions, pool, FLAGS_seed)
                          .fuzzUniqueFlatNotNull(BIGINT(), numKeys);

  VectorFuzzer::Options valueOptions;
  valueOptions.nullRatio = 0.05;
  valueOptions.inMapRatio = density / 100.0;
  // A key in every row has no in-map buffer.
  valueOptions.inMapNullRatio = density == 100 ? 1.0 : 0.0;
  VectorFuzzer fuzzer(valueOptions, pool, FLAGS_seed);
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;
  mapValues.reserve(distinctKeys->size());
  inMaps.reserve(distinctKeys->size());
  for (vector_size_t i = 0; i < distinctKeys->size(); ++i) {
    mapValues.push_back(fuzzer.fuzzFlat(valueType, kNumRows));
    inMaps.push_back(fuzzer.fuzzFlatMapInMap(kNumRows));
  }

  return std::make_shared<FlatMapVector>(
      pool,
      MAP(BIGINT(), valueType),
      nullptr,
      kNumRows,
      std::move(distinctKeys),
      std::move(mapValues),
      std::move(inMaps));
}

// Returns a MakeRowFromMap that projects `numProjected` keys of `flatMap`,
// spread evenly over its distinct keys.
std::unique_ptr<MakeRowFromMap<TypeKind::BIGINT>> makeProjection(
    memory::MemoryPool* pool,
    const FlatMapVector& flatMap,
    vector_size_t numProjected,
    bool replaceNulls) {
  const auto* distinctKeys = flatMap.distinctKeys()->asFlatVector<int64_t>();
  auto keysToProject =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), numProjected, pool);
  std::vector<std::string> fieldNames;
  fieldNames.reserve(numProjected);
  for (vector_size_t i = 0; i < numProjected; ++i) {
    const auto channel = i * flatMap.numDistinctKeys() / numProjected;
    keysToProject->set(i, distinctKeys->valueAt(channel));
    fieldNames.push_back(fmt::format("f{}", i));
  }

  return std::make_unique<MakeRowFromMap<TypeKind::BIGINT>>(
      MakeRowFromMapOptions{
          .keysToProject = std::move(keysToProject),
          .outputFieldNames = std::move(fieldNames),
          .replaceNulls = replaceNulls});
}

struct TestCase {
  FlatMapVectorPtr flatMap;
  // The same maps as `flatMap`.
  MapVectorPtr map;
  // Indices of the dictionary that wraps the maps, if the case wraps them.
  BufferPtr wrapIndices;
  std::unique_ptr<MakeRowFromMap<TypeKind::BIGINT>> projection;
};

// Returns `maps` wrapped in the dictionary of `testCase`, if it has one.
VectorPtr wrap(const TestCase& testCase, VectorPtr maps) {
  if (testCase.wrapIndices == nullptr) {
    return maps;
  }
  return BaseVector::wrapInDictionary(
      nullptr, testCase.wrapIndices, kNumRows, std::move(maps));
}

// Returns the test cases that the benchmarks below run.
std::vector<CaseSpec> caseSpecs() {
  std::vector<CaseSpec> specs;

  // REAL and ARRAY(BIGINT) values, for every combination of the parameters.
  for (bool arrayValues : {false, true}) {
    for (vector_size_t numKeys : {100, 1'000}) {
      for (vector_size_t numProjected : {10, 100}) {
        for (int32_t density : {100, 50}) {
          for (bool replaceNulls : {false, true}) {
            specs.push_back(
                {.numKeys = numKeys,
                 .numProjected = numProjected,
                 .density = density,
                 .replaceNulls = replaceNulls,
                 .arrayValues = arrayValues});
          }
        }
      }
    }
  }

  // Wrapped maps, and sparse keys.
  for (vector_size_t numProjected : {10, 100}) {
    specs.push_back(
        {.numKeys = 1'000,
         .numProjected = numProjected,
         .density = 100,
         .wrapped = true});
    for (bool replaceNulls : {false, true}) {
      specs.push_back(
          {.numKeys = 1'000,
           .numProjected = numProjected,
           .density = 10,
           .replaceNulls = replaceNulls});
    }
  }
  return specs;
}

// Builds every test case up front and owns the memory pool they use.
class BenchmarkData {
 public:
  BenchmarkData() : pool_(memory::memoryManager()->addLeafPool()) {
    const auto wrapIndices = VectorFuzzer({}, pool_.get(), FLAGS_seed)
                                 .fuzzIndices(kNumRows, kNumRows);

    // Cases that differ only in their projection share their maps.
    folly::F14FastMap<std::string, std::pair<FlatMapVectorPtr, MapVectorPtr>>
        maps;
    for (const auto& spec : caseSpecs()) {
      const auto mapsKey =
          fmt::format("{}_{}_{}", spec.numKeys, spec.density, spec.arrayValues);
      auto it = maps.find(mapsKey);
      if (it == maps.end()) {
        auto flatMap = makeFlatMap(
            pool_.get(),
            spec.numKeys,
            spec.density,
            spec.arrayValues ? TypePtr{ARRAY(BIGINT())} : TypePtr{REAL()});

        // toMapVector() wraps the distinct keys in a dictionary; readers
        // produce flat keys.
        auto map = flatMap->toMapVector();
        BaseVector::flattenVector(map->mapKeys());
        it = maps.emplace(mapsKey, std::pair{flatMap, map}).first;
      }

      const auto& [flatMap, map] = it->second;
      TestCase testCase{
          flatMap,
          map,
          spec.wrapped ? wrapIndices : nullptr,
          makeProjection(
              pool_.get(), *flatMap, spec.numProjected, spec.replaceNulls)};
      checkFlatMapMatchesMap(testCase);
      testCases_.emplace(spec.key(), std::move(testCase));
    }
  }

  const TestCase& testCase(const CaseSpec& spec) const {
    return testCases_.at(spec.key());
  }

  const SelectivityVector& rows() const {
    return rows_;
  }

 private:
  // Checks that projecting the flat map returns the same rows as projecting the
  // map.
  void checkFlatMapMatchesMap(const TestCase& testCase) const {
    auto expected = testCase.projection->apply(
        *wrap(testCase, testCase.map), rows_, nullptr);
    auto actual = testCase.projection->apply(
        *wrap(testCase, testCase.flatMap), rows_, nullptr);
    for (vector_size_t row = 0; row < kNumRows; ++row) {
      VELOX_CHECK(
          expected->equalValueAt(actual.get(), row, row),
          "Projecting the flat map differs from the map at row {}",
          row);
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  const SelectivityVector rows_{kNumRows};
  folly::F14FastMap<std::string, TestCase> testCases_;
};

std::unique_ptr<BenchmarkData> data;

// The functions below project one batch and return its number of rows, so
// folly reports time per row.

vector_size_t runMap(const TestCase& testCase) {
  folly::doNotOptimizeAway(testCase.projection->apply(
      *wrap(testCase, testCase.map), data->rows(), nullptr));
  return kNumRows;
}

vector_size_t runToMap(const TestCase& testCase) {
  folly::doNotOptimizeAway(testCase.projection->apply(
      *wrap(testCase, testCase.flatMap->toMapVector()), data->rows(), nullptr));
  return kNumRows;
}

vector_size_t runFlatMap(const TestCase& testCase) {
  folly::doNotOptimizeAway(testCase.projection->apply(
      *wrap(testCase, testCase.flatMap), data->rows(), nullptr));
  return kNumRows;
}

// Defines the benchmarks of the case `name`, whose CaseSpec fields follow.
#define MAKE_ROW_FROM_MAP_BENCHMARKS(name, ...)               \
  BENCHMARK_MULTI(map_##name) {                               \
    return runMap(data->testCase(CaseSpec{__VA_ARGS__}));     \
  }                                                           \
  BENCHMARK_RELATIVE_MULTI(toMap_##name) {                    \
    return runToMap(data->testCase(CaseSpec{__VA_ARGS__}));   \
  }                                                           \
  BENCHMARK_RELATIVE_MULTI(flatMap_##name) {                  \
    return runFlatMap(data->testCase(CaseSpec{__VA_ARGS__})); \
  }                                                           \
  BENCHMARK_DRAW_LINE();

MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_100pct,
    .numKeys = 100,
    .numProjected = 10,
    .density = 100)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_100pct_defaults,
    .numKeys = 100,
    .numProjected = 10,
    .density = 100,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_50pct,
    .numKeys = 100,
    .numProjected = 10,
    .density = 50)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_50pct_defaults,
    .numKeys = 100,
    .numProjected = 10,
    .density = 50,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_100pct,
    .numKeys = 100,
    .numProjected = 100,
    .density = 100)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_100pct_defaults,
    .numKeys = 100,
    .numProjected = 100,
    .density = 100,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_50pct,
    .numKeys = 100,
    .numProjected = 100,
    .density = 50)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_50pct_defaults,
    .numKeys = 100,
    .numProjected = 100,
    .density = 50,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_100pct,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 100)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_100pct_defaults,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 100,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_50pct,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 50)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_50pct_defaults,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 50,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_100pct,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 100)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_100pct_defaults,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 100,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_50pct,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 50)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_50pct_defaults,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 50,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_100pct_array,
    .numKeys = 100,
    .numProjected = 10,
    .density = 100,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_100pct_array_defaults,
    .numKeys = 100,
    .numProjected = 10,
    .density = 100,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_50pct_array,
    .numKeys = 100,
    .numProjected = 10,
    .density = 50,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_10proj_50pct_array_defaults,
    .numKeys = 100,
    .numProjected = 10,
    .density = 50,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_100pct_array,
    .numKeys = 100,
    .numProjected = 100,
    .density = 100,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_100pct_array_defaults,
    .numKeys = 100,
    .numProjected = 100,
    .density = 100,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_50pct_array,
    .numKeys = 100,
    .numProjected = 100,
    .density = 50,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    100keys_100proj_50pct_array_defaults,
    .numKeys = 100,
    .numProjected = 100,
    .density = 50,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_100pct_array,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 100,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_100pct_array_defaults,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 100,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_50pct_array,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 50,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_50pct_array_defaults,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 50,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_100pct_array,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 100,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_100pct_array_defaults,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 100,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_50pct_array,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 50,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_50pct_array_defaults,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 50,
    .replaceNulls = true,
    .arrayValues = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_100pct_wrapped,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 100,
    .wrapped = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_10pct,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 10)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_10proj_10pct_defaults,
    .numKeys = 1'000,
    .numProjected = 10,
    .density = 10,
    .replaceNulls = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_100pct_wrapped,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 100,
    .wrapped = true)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_10pct,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 10)
MAKE_ROW_FROM_MAP_BENCHMARKS(
    1000keys_100proj_10pct_defaults,
    .numKeys = 1'000,
    .numProjected = 100,
    .density = 10,
    .replaceNulls = true)

} // namespace
} // namespace facebook::velox::functions::test

int main(int argc, char* argv[]) {
  using namespace facebook::velox;
  using namespace facebook::velox::functions::test;
  folly::Init follyInit(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  data = std::make_unique<BenchmarkData>();
  folly::runBenchmarks();
  data.reset();

  return 0;
}
