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

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <random>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/benchmarks/HashJoinBenchmarkBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

// Benchmarks semi and anti hash joins on keys that repeat on both sides.
//
// The anti and left semi cases also have 1,000 keys that appear only on the
// build side and 1,000 that appear only on the probe side. Each runs with no
// filter, a filter that passes every probe row and one that passes 1/4 of
// them. These filters read only probe columns, so the build side keeps one row
// per key.
//
// Uses 4,096-row input and output batches.
namespace {

constexpr vector_size_t kBatchRows = 4'096;

class HashJoinSemiAntiBenchmark : public HashJoinBenchmarkBase {
 public:
  HashJoinSemiAntiBenchmark() : HashJoinBenchmarkBase(kBatchRows) {}

  // Keys [0, numMatchedKeys) repeat 'buildDuplicates' times on the build side
  // and 'probeDuplicates' times on the probe side. Each side also has
  // 'numOnlyKeys' keys of its own that appear once.
  struct Keys {
    int32_t numMatchedKeys;
    int32_t buildDuplicates;
    int32_t probeDuplicates;
    int32_t numOnlyKeys{0};
  };

  // Registers a benchmark of 'joinType' with 'filter' on the rows 'keys'
  // describes.
  void
  add(core::JoinType joinType, const std::string& filter, const Keys& keys) {
    const auto probeKeys = makeKeys(
        keys.numMatchedKeys,
        keys.probeDuplicates,
        keys.numMatchedKeys + keys.numOnlyKeys,
        keys.numOnlyKeys,
        /*seed=*/97);
    const auto buildKeys = makeKeys(
        keys.numMatchedKeys,
        keys.buildDuplicates,
        keys.numMatchedKeys,
        keys.numOnlyKeys,
        /*seed=*/31);
    addBenchmark(
        fmt::format("dup{}x{}", keys.buildDuplicates, keys.probeDuplicates),
        {.joinType = joinType,
         .filter = filter,
         .configs =
             {{core::QueryConfig::kPreferredOutputBatchRows,
               std::to_string(kBatchRows)},
              {core::QueryConfig::kMaxOutputBatchRows,
               std::to_string(kBatchRows)}}},
        makeBatches(
            {"t0", "t1"},
            probeKeys.size(),
            [&](int64_t row) { return probeKeys[row]; }),
        makeBatches({"u0", "u1"}, buildKeys.size(), [&](int64_t row) {
          return buildKeys[row];
        }));
  }

 private:
  // Returns keys [0, numMatchedKeys) repeated 'duplicates' times and keys
  // [onlyKeysStart, onlyKeysStart + numOnlyKeys) once, in a deterministic
  // random order.
  static std::vector<int64_t> makeKeys(
      int32_t numMatchedKeys,
      int32_t duplicates,
      int64_t onlyKeysStart,
      int32_t numOnlyKeys,
      uint32_t seed) {
    VELOX_CHECK_GT(numMatchedKeys, 0);
    VELOX_CHECK_GT(duplicates, 0);
    VELOX_CHECK_GE(numOnlyKeys, 0);

    const int64_t numRepeatedKeys = numMatchedKeys * duplicates;
    std::vector<int64_t> keys(numRepeatedKeys + numOnlyKeys);
    for (int64_t i = 0; i < numRepeatedKeys; ++i) {
      keys[i] = i % numMatchedKeys;
    }
    for (int32_t i = 0; i < numOnlyKeys; ++i) {
      keys[numRepeatedKeys + i] = onlyKeysStart + i;
    }

    std::mt19937 random(seed);
    std::shuffle(keys.begin(), keys.end(), random);
    return keys;
  }
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});

  HashJoinSemiAntiBenchmark benchmark;
  // 't1' holds the row number, so 't1 >= 0' passes every row.
  benchmark.add(
      core::JoinType::kRightSemiFilter,
      "",
      {.numMatchedKeys = 20'000, .buildDuplicates = 8, .probeDuplicates = 8});
  for (const auto* filter : {"", "t1 >= 0"}) {
    benchmark.add(
        core::JoinType::kRightSemiFilter,
        filter,
        {.numMatchedKeys = 10'000,
         .buildDuplicates = 32,
         .probeDuplicates = 32});
  }

  for (const auto joinType :
       {core::JoinType::kAnti,
        core::JoinType::kLeftSemiFilter,
        core::JoinType::kLeftSemiProject}) {
    for (const auto buildDuplicates : {1, 4, 32}) {
      for (const auto* filter : {"", "t1 >= 0", "t1 % 4 = 0"}) {
        benchmark.add(
            joinType,
            filter,
            {.numMatchedKeys = 1'000,
             .buildDuplicates = buildDuplicates,
             .probeDuplicates = 128,
             .numOnlyKeys = 1'000});
      }
    }
  }

  folly::runBenchmarks();
  return 0;
}
