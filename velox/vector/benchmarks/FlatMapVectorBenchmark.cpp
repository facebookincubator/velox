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
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <folly/Benchmark.h>
#include <folly/container/F14Map.h>
#include <folly/init/Init.h>

// Measures the cost of the FlatMapVector key index, which maps each distinct
// key to its channel (the position of that key's values and in-map buffer).
//
// For each key type (BIGINT, VARCHAR) and each number of distinct keys (100,
// 1,000, 10,000), it builds one flat map of 1,000 rows over keys generated
// with VectorFuzzer. All keys share a single values vector and there are no
// in-map buffers, so the cases below exercise the key index rather than the
// values. Every case reports time per key.
//
//  - construct: constructs a FlatMapVector over the same keys and values.
//  - constructAndLookup: constructs one and looks up a single key. Reported
//    relative to `construct`, so the gap is the cost of building the key
//    index when construction defers it.
//  - slice: slices half of the rows.
//  - lookupGeneric{Hit,Miss}: getKeyChannel() with a keys vector and an index,
//    for keys that are present / absent.
//  - lookupTyped{Hit,Miss}: getKeyChannel() with a C++ value. Reported
//    relative to the generic lookup above it, since the typed overloads are
//    meant to be the faster path.

DEFINE_int32(seed, 1, "Seed for the fuzzer that generates the keys");

namespace facebook::velox {
namespace {

constexpr vector_size_t kNumRows = 1'000;

// Builds a flat map with about `numKeys` distinct keys of `keyType`; fewer if
// the fuzzer draws duplicates. Every key shares one values vector, since only
// the key index is being measured.
FlatMapVectorPtr makeFlatMap(
    memory::MemoryPool* pool,
    VectorFuzzer& fuzzer,
    const TypePtr& keyType,
    vector_size_t numKeys) {
  auto distinctKeys = fuzzer.fuzzUniqueFlatNotNull(keyType, numKeys);
  auto values = fuzzer.fuzzFlat(INTEGER(), kNumRows);
  return std::make_shared<FlatMapVector>(
      pool,
      MAP(keyType, INTEGER()),
      nullptr,
      kNumRows,
      distinctKeys,
      std::vector<VectorPtr>(distinctKeys->size(), values),
      std::vector<BufferPtr>{});
}

// Returns keys absent from `flatMap`, for measuring lookup misses.
VectorPtr fuzzMissingKeys(
    memory::MemoryPool* pool,
    VectorFuzzer& fuzzer,
    const FlatMapVectorPtr& flatMap) {
  auto candidates = fuzzer.fuzzUniqueFlatNotNull(
      flatMap->keyType(), flatMap->numDistinctKeys());
  auto missingKeys = BaseVector::create(flatMap->keyType(), 0, pool);
  for (vector_size_t i = 0; i < candidates->size(); ++i) {
    if (!flatMap->getKeyChannel(candidates, i).has_value()) {
      const auto size = missingKeys->size();
      missingKeys->resize(size + 1);
      missingKeys->copy(candidates.get(), size, i, 1);
    }
  }
  return missingKeys;
}

struct TestCase {
  FlatMapVectorPtr flatMap;
  VectorPtr missingKeys;
};

// Builds every test case up front and owns the memory pool they use.
class BenchmarkData {
 public:
  BenchmarkData() : pool_(memory::memoryManager()->addLeafPool()) {
    VectorFuzzer::Options options;
    // fuzzUniqueFlatNotNull() draws `containerLength` values per requested key.
    options.containerLength = 1;
    VectorFuzzer fuzzer(options, pool_.get(), FLAGS_seed);
    for (const auto& keyType : std::vector<TypePtr>{BIGINT(), VARCHAR()}) {
      for (vector_size_t numKeys : {100, 1'000, 10'000}) {
        auto flatMap = makeFlatMap(pool_.get(), fuzzer, keyType, numKeys);
        auto missingKeys = fuzzMissingKeys(pool_.get(), fuzzer, flatMap);
        testCases_.emplace(
            key(keyType->toString(), numKeys), TestCase{flatMap, missingKeys});
      }
    }
  }

  const TestCase& testCase(std::string_view keyType, vector_size_t numKeys)
      const {
    return testCases_.at(key(keyType, numKeys));
  }

 private:
  static std::string key(std::string_view keyType, vector_size_t numKeys) {
    return fmt::format("{}_{}", keyType, numKeys);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  folly::F14FastMap<std::string, TestCase> testCases_;
};

std::unique_ptr<BenchmarkData> data;

FlatMapVectorPtr construct(const FlatMapVectorPtr& flatMap) {
  return std::make_shared<FlatMapVector>(
      flatMap->pool(),
      flatMap->type(),
      nullptr,
      flatMap->size(),
      flatMap->distinctKeys(),
      flatMap->mapValues(),
      flatMap->inMaps());
}

// The fuzzer may drop duplicate keys, so the functions below return the actual
// number of keys, and folly reports time per key.
//
// Lookups pass `value_or(0)` to doNotOptimizeAway() because GCC rejects a
// std::optional as its asm register operand.

vector_size_t runConstruct(const TestCase& testCase) {
  folly::doNotOptimizeAway(construct(testCase.flatMap));
  return testCase.flatMap->numDistinctKeys();
}

// The first lookup builds the key index if construction did not. `T` is the
// C++ type of the keys.
template <typename T>
vector_size_t runConstructAndLookup(const TestCase& testCase) {
  const auto& flatMap = testCase.flatMap;
  const auto key = flatMap->distinctKeys()->asFlatVector<T>()->valueAt(0);
  folly::doNotOptimizeAway(construct(flatMap)->getKeyChannel(key).value_or(0));
  return flatMap->numDistinctKeys();
}

vector_size_t runSlice(const TestCase& testCase) {
  folly::doNotOptimizeAway(testCase.flatMap->slice(0, kNumRows / 2));
  return testCase.flatMap->numDistinctKeys();
}

// Looks up every key in `keys`, passing each as a C++ value of type `T`.
template <typename T>
vector_size_t runLookupTyped(
    const FlatMapVector& flatMap,
    const VectorPtr& keys) {
  const auto* rawKeys = keys->asFlatVector<T>()->rawValues();
  for (vector_size_t i = 0; i < keys->size(); ++i) {
    folly::doNotOptimizeAway(flatMap.getKeyChannel(rawKeys[i]).value_or(0));
  }
  return keys->size();
}

// Looks up every key in `keys`, passing the vector and an index.
vector_size_t runLookupGeneric(
    const FlatMapVector& flatMap,
    const VectorPtr& keys) {
  for (vector_size_t i = 0; i < keys->size(); ++i) {
    folly::doNotOptimizeAway(flatMap.getKeyChannel(keys, i).value_or(0));
  }
  return keys->size();
}

#define FLAT_MAP_BENCHMARKS(type, T, numKeys)                          \
  BENCHMARK_MULTI(construct_##type##_##numKeys) {                      \
    return runConstruct(data->testCase(#type, numKeys));               \
  }                                                                    \
  BENCHMARK_RELATIVE_MULTI(constructAndLookup_##type##_##numKeys) {    \
    return runConstructAndLookup<T>(data->testCase(#type, numKeys));   \
  }                                                                    \
  BENCHMARK_MULTI(slice_##type##_##numKeys) {                          \
    return runSlice(data->testCase(#type, numKeys));                   \
  }                                                                    \
  BENCHMARK_MULTI(lookupGenericHit_##type##_##numKeys) {               \
    const auto& flatMap = *data->testCase(#type, numKeys).flatMap;     \
    return runLookupGeneric(flatMap, flatMap.distinctKeys());          \
  }                                                                    \
  BENCHMARK_RELATIVE_MULTI(lookupTypedHit_##type##_##numKeys) {        \
    const auto& flatMap = *data->testCase(#type, numKeys).flatMap;     \
    return runLookupTyped<T>(flatMap, flatMap.distinctKeys());         \
  }                                                                    \
  BENCHMARK_MULTI(lookupGenericMiss_##type##_##numKeys) {              \
    const auto& testCase = data->testCase(#type, numKeys);             \
    return runLookupGeneric(*testCase.flatMap, testCase.missingKeys);  \
  }                                                                    \
  BENCHMARK_RELATIVE_MULTI(lookupTypedMiss_##type##_##numKeys) {       \
    const auto& testCase = data->testCase(#type, numKeys);             \
    return runLookupTyped<T>(*testCase.flatMap, testCase.missingKeys); \
  }                                                                    \
  BENCHMARK_DRAW_LINE();

FLAT_MAP_BENCHMARKS(BIGINT, int64_t, 100)
FLAT_MAP_BENCHMARKS(BIGINT, int64_t, 1000)
FLAT_MAP_BENCHMARKS(BIGINT, int64_t, 10000)
FLAT_MAP_BENCHMARKS(VARCHAR, StringView, 100)
FLAT_MAP_BENCHMARKS(VARCHAR, StringView, 1000)
FLAT_MAP_BENCHMARKS(VARCHAR, StringView, 10000)

} // namespace
} // namespace facebook::velox

int main(int argc, char* argv[]) {
  using namespace facebook::velox;
  folly::Init follyInit(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  data = std::make_unique<BenchmarkData>();
  folly::runBenchmarks();
  data.reset();

  return 0;
}
