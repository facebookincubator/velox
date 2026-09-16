/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <tsl/robin_map.h>

#include <folly/container/F14Map-fwd.h>
#include "absl/container/flat_hash_map.h"
#include "common/init/light.h"
#include "folly/Benchmark.h"
#include "folly/Demangle.h"
#include "folly/container/F14Map.h"

constexpr size_t kDataSize = 20 * 1024 * 1024; // 20M
constexpr size_t kRepeatSize = 5 * 1024; // 5K

std::vector<int64_t> repeatingData;
std::vector<int64_t> uniqueData;

template <typename Map>
struct TestParameters {
  size_t elementCount;
  bool preAllocate;
  const std::vector<typename Map::key_type>& data;
};

template <class Map>
struct is_robin_map : std::integral_constant<
                          bool,
                          std::is_same_v<
                              Map,
                              tsl::robin_map<
                                  typename Map::key_type,
                                  typename Map::mapped_type>> ||
                              std::is_same_v<
                                  Map,
                                  tsl::robin_pg_map<
                                      typename Map::key_type,
                                      typename Map::mapped_type>>> {};

template <class Map, std::enable_if_t<!is_robin_map<Map>::value, bool> = true>
void incrementValue(Map& map, typename Map::key_type key) {
  ++(map.try_emplace(key, 0).first->second);
}

template <class Map, std::enable_if_t<is_robin_map<Map>::value, bool> = true>
void incrementValue(Map& map, typename Map::key_type key) {
  ++(map.try_emplace(key, 0).first.value());
}

template <class Map>
void incrementCounts(size_t iters, const TestParameters<Map>& parameters) {
  for (int i = 0; i < iters; ++i) {
    Map map;
    if (parameters.preAllocate) {
      map.reserve(10000);
    }
    for (auto j = 0; j < parameters.elementCount; ++j) {
      incrementValue<Map>(map, parameters.data[i]);
    }
  }
}

BENCHMARK_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Repeating20K,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Repeating20KNoReserve,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Repeating20K,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Repeating20KNoReserve,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Repeating20K,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Repeating20KNoReserve,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Repeating20K,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Repeating20KNoReserve,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Repeating20K,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Repeating20KNoReserve,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Repeating20K,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Repeating20KNoReserve,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Repeating20M,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Repeating20MNoReserve,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Repeating20M,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Repeating20MNoReserve,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Repeating20M,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Repeating20MNoReserve,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Repeating20M,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Repeating20MNoReserve,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Repeating20M,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Repeating20MNoReserve,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Repeating20M,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = repeatingData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Repeating20MNoReserve,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = repeatingData})

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Unique20K,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Unique20KNoReserve,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Unique20K,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Unique20KNoReserve,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Unique20K,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Unique20KNoReserve,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Unique20K,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Unique20KNoReserve,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Unique20K,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Unique20KNoReserve,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Unique20K,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Unique20KNoReserve,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 20000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Unique20M,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    UnorderedMapInt64Unique20MNoReserve,
    TestParameters<std::unordered_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Unique20M,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinMapInt64Unique20MNoReserve,
    TestParameters<tsl::robin_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Unique20M,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    RobinPGMapInt64Unique20MNoReserve,
    TestParameters<tsl::robin_pg_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Unique20M,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14ValueMapInt64Unique20MNoReserve,
    TestParameters<folly::F14ValueMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Unique20M,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    F14NodeMapInt64Unique20MNoReserve,
    TestParameters<folly::F14NodeMap<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Unique20M,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = true,
        .data = uniqueData})

BENCHMARK_RELATIVE_NAMED_PARAM(
    incrementCounts,
    AbslMapInt64Unique20MNoReserve,
    TestParameters<absl::flat_hash_map<int64_t, uint64_t>>{
        .elementCount = 2000000,
        .preAllocate = false,
        .data = uniqueData})

int main(int argc, char** argv) {
  facebook::init::initFacebookLight(&argc, &argv);
  repeatingData.resize(kDataSize);
  uniqueData.resize(kDataSize);
  for (auto i = 0; i < kDataSize; ++i) {
    repeatingData[i] = i % kRepeatSize - (kRepeatSize / 2);
    uniqueData[i] = i - kRepeatSize;
  }

  folly::runBenchmarks();
  return 0;
}
