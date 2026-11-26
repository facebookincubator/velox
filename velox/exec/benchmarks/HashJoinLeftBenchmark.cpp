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
#include <folly/hash/Hash.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/benchmarks/HashJoinBenchmarkBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

constexpr vector_size_t kBatchSize = 100'000;
constexpr int64_t kProbePatternRows = 10'000'000;
constexpr int32_t kProbeRepeats = 500;
constexpr int64_t kNumProbeRows = kProbePatternRows * kProbeRepeats;
constexpr uint64_t kBloomFilterMaxBytes = 256UL << 20;

int64_t buildKey(uint64_t row) {
  return static_cast<int64_t>(folly::hash::twang_mix64(row) & ~uint64_t{1});
}

class HashJoinLeftBenchmark : public HashJoinBenchmarkBase {
 public:
  HashJoinLeftBenchmark() : HashJoinBenchmarkBase(kBatchSize) {}

  std::vector<RowVectorPtr> makeBuild(int64_t numBuildRows) {
    return makeBatches({"u0", "u1"}, numBuildRows, buildKey);
  }

  std::vector<RowVectorPtr> makeProbe(int64_t numBuildRows, int32_t hitPct) {
    const auto pattern =
        makeBatches({"t0"}, kProbePatternRows, [&](int64_t row) {
          const auto random = folly::hash::twang_mix64(row);
          if (random % 100 < hitPct) {
            return buildKey(random % numBuildRows);
          }
          return static_cast<int64_t>(random | uint64_t{1});
        });
    std::vector<RowVectorPtr> probe;
    for (int32_t repeat = 0; repeat < kProbeRepeats; ++repeat) {
      probe.insert(probe.end(), pattern.begin(), pattern.end());
    }
    return probe;
  }
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});

  HashJoinLeftBenchmark benchmark;
  for (const auto numBuildRows : {1'000'000, 10'000'000}) {
    const auto build = benchmark.makeBuild(numBuildRows);
    for (const auto hitPct : {1, 10, 100}) {
      const auto probe = benchmark.makeProbe(numBuildRows, hitPct);
      for (const auto enableBloomFilter : {false, true}) {
        benchmark.addBenchmark(
            fmt::format(
                "build_{}M_probe_5B_hit_{}pct_bloom_{}",
                numBuildRows / 1'000'000,
                hitPct,
                enableBloomFilter ? "on" : "off"),
            {.joinType = core::JoinType::kLeft,
             .configs =
                 {{core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize,
                   std::to_string(kBloomFilterMaxBytes)},
                  {core::QueryConfig::kBypassHashProbeBloomFilterMinRows,
                   enableBloomFilter ? "100000" : "0"},
                  {core::QueryConfig::kBypassHashProbeBloomFilterMinPct, "85"}},
             .expectedNumOutputRows = kNumProbeRows},
            probe,
            build);
      }
    }
  }

  folly::runBenchmarks();
  return 0;
}
