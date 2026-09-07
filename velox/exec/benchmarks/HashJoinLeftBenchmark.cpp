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
#include <folly/hash/Hash.h>
#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

namespace {

constexpr vector_size_t kBatchSize = 100'000;
constexpr int64_t kProbePatternRows = 10'000'000;
constexpr int32_t kProbeRepeats = 500;
constexpr int64_t kNumProbeRows = kProbePatternRows * kProbeRepeats;
constexpr uint64_t kBloomFilterMaxBytes = 256UL << 20;

struct BenchmarkParams {
  int64_t numBuildRows;
  int32_t hitPct;
  bool enableBloomFilter;
};

struct BenchmarkCase {
  BenchmarkParams params;
  std::shared_ptr<std::vector<RowVectorPtr>> buildVectors;
  std::shared_ptr<std::vector<RowVectorPtr>> probeVectors;
};

int64_t buildKey(uint64_t row) {
  return static_cast<int64_t>(folly::hash::twang_mix64(row) & ~uint64_t{1});
}

template <typename MakeBatch>
std::vector<RowVectorPtr> makeBatches(int64_t numRows, MakeBatch makeBatch) {
  std::vector<RowVectorPtr> batches;
  for (int64_t row = 0; row < numRows; row += kBatchSize) {
    const auto size = static_cast<vector_size_t>(
        std::min<int64_t>(kBatchSize, numRows - row));
    batches.push_back(makeBatch(row, size));
  }
  return batches;
}

class HashJoinLeftBenchmark : public VectorTestBase {
 public:
  std::vector<RowVectorPtr> prepareBuildData(int64_t numBuildRows) {
    return makeBatches(numBuildRows, [&](int64_t row, vector_size_t size) {
      return makeRowVector(
          {"u0", "u1"},
          {
              makeFlatVector<int64_t>(
                  size,
                  [&](vector_size_t index) { return buildKey(row + index); }),
              makeFlatVector<int64_t>(
                  size, [&](vector_size_t index) { return row + index; }),
          });
    });
  }

  std::vector<RowVectorPtr> prepareProbeData(
      int64_t numBuildRows,
      int32_t hitPct) {
    auto pattern =
        makeBatches(kProbePatternRows, [&](int64_t row, vector_size_t size) {
          return makeRowVector(
              {"t0"}, {makeFlatVector<int64_t>(size, [&](vector_size_t index) {
                const auto probeRow = row + index;
                const auto random = folly::hash::twang_mix64(probeRow);
                if (random % 100 < hitPct) {
                  return buildKey(random % numBuildRows);
                }
                return static_cast<int64_t>(random | uint64_t{1});
              })});
        });
    std::vector<RowVectorPtr> probeVectors;
    for (int32_t repeat = 0; repeat < kProbeRepeats; ++repeat) {
      probeVectors.insert(probeVectors.end(), pattern.begin(), pattern.end());
    }
    return probeVectors;
  }

  void run(
      const BenchmarkParams& params,
      const std::vector<RowVectorPtr>& buildVectors,
      const std::vector<RowVectorPtr>& probeVectors) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(probeVectors)
                    .hashJoin(
                        {"t0"},
                        {"u0"},
                        PlanBuilder(planNodeIdGenerator, pool_.get())
                            .values(buildVectors)
                            .planNode(),
                        "",
                        {"t0", "u1"},
                        core::JoinType::kLeft)
                    .singleAggregation({}, {"count(1)"})
                    .planNode();

    AssertQueryBuilder query(plan);
    query.maxDrivers(1)
        .config(
            core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize,
            std::to_string(kBloomFilterMaxBytes))
        .config(
            core::QueryConfig::kBypassHashProbeBloomFilterMinRows,
            params.enableBloomFilter ? std::to_string(100'000)
                                     : std::to_string(0))
        .config(
            core::QueryConfig::kBypassHashProbeBloomFilterMinPct,
            std::to_string(85));
    auto result = query.copyResults(pool());
    VELOX_CHECK_EQ(result->size(), 1);
    VELOX_CHECK_EQ(
        result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(0),
        kNumProbeRows);
  }
};

std::string benchmarkName(const BenchmarkParams& params) {
  return fmt::format(
      "build_{}M_probe_5B_hit_{}pct_bloom_{}",
      params.numBuildRows / 1'000'000,
      params.hitPct,
      params.enableBloomFilter ? "enabled" : "disabled");
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();

  auto benchmark = std::make_unique<HashJoinLeftBenchmark>();
  std::vector<BenchmarkCase> benchmarkCases;
  for (const auto numBuildRows : {1'000'000, 10'000'000}) {
    auto buildVectors = std::make_shared<std::vector<RowVectorPtr>>(
        benchmark->prepareBuildData(numBuildRows));
    for (const auto hitPct : {1, 10, 100}) {
      auto probeVectors = std::make_shared<std::vector<RowVectorPtr>>(
          benchmark->prepareProbeData(numBuildRows, hitPct));
      for (const auto enableBloomFilter : {false, true}) {
        benchmarkCases.push_back(
            {{numBuildRows, hitPct, enableBloomFilter},
             buildVectors,
             probeVectors});
      }
    }
  }

  for (const auto& benchmarkCase : benchmarkCases) {
    folly::addBenchmark(
        __FILE__,
        benchmarkName(benchmarkCase.params),
        [&benchmark, &benchmarkCase]() {
          benchmark->run(
              benchmarkCase.params,
              *benchmarkCase.buildVectors,
              *benchmarkCase.probeVectors);
          return 1;
        });
  }

  folly::runBenchmarks();
  return 0;
}
