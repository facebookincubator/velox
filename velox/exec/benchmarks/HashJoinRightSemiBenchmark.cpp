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

#include <random>
#include <tuple>

#include "velox/common/memory/Memory.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

// Benchmarks hash joins that track probed build rows: right semi filter,
// right anti, right and full joins, with and without a filter that only
// references probe columns.
namespace {

struct BenchmarkParams {
  core::JoinType joinType;
  // Number of distinct keys on both sides.
  int32_t numMatchedKeys;
  // Number of distinct keys that only appear once on the build side.
  int32_t numBuildOnlyKeys{0};
  // Number of distinct keys that only appear once on the probe side.
  int32_t numProbeOnlyKeys{0};
  int32_t buildDuplicates;
  int32_t probeDuplicates;
  int32_t numProbeBatches;
  // Join filter. Empty means no filter.
  std::string filter{};
  std::vector<std::string> outputLayout;
  std::string title;
};

struct BenchmarkData {
  std::vector<RowVectorPtr> buildVectors;
  std::vector<RowVectorPtr> probeVectors;
};

struct BenchmarkCase {
  BenchmarkParams params;
  BenchmarkData data;
};

class HashJoinRightSemiBenchmark : public VectorTestBase {
 public:
  BenchmarkData prepareData(const BenchmarkParams& params) {
    BenchmarkData data;

    auto buildKeys = makeKeys(
        params.numMatchedKeys,
        params.buildDuplicates,
        params.numMatchedKeys,
        params.numBuildOnlyKeys,
        31 /*seed for deterministic shuffling*/);
    auto probeKeys = makeKeys(
        params.numMatchedKeys,
        params.probeDuplicates,
        params.numMatchedKeys + params.numBuildOnlyKeys,
        params.numProbeOnlyKeys,
        97 /*seed for deterministic shuffling*/);

    data.buildVectors = makeBatchesFromKeys(buildKeys, 1, 0);
    data.probeVectors =
        makeBatchesFromKeys(probeKeys, params.numProbeBatches, 10'000'000);
    return data;
  }

  int64_t run(const BenchmarkParams& params, const BenchmarkData& data) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(data.probeVectors)
                    .project({"c0 AS t0", "c1 AS t1"})
                    .hashJoin(
                        {"t0"},
                        {"u0"},
                        PlanBuilder(planNodeIdGenerator, pool_.get())
                            .values(data.buildVectors)
                            .project({"c0 AS u0", "c1 AS u1"})
                            .planNode(),
                        params.filter,
                        params.outputLayout,
                        params.joinType)
                    .planNode();

    return AssertQueryBuilder(plan).maxDrivers(1).countResults();
  }

 private:
  // Returns keys [0, numMatchedKeys) repeated 'duplicates' times and keys
  // [onlyKeysStart, onlyKeysStart + numOnlyKeys) once, in a deterministic
  // random order.
  std::vector<int64_t> makeKeys(
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

  std::vector<RowVectorPtr> makeBatchesFromKeys(
      const std::vector<int64_t>& keys,
      int32_t numBatches,
      int64_t payloadBase) {
    VELOX_CHECK_GT(numBatches, 0);

    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);

    const int32_t totalSize = keys.size();
    const int32_t batchSize = std::max(1, totalSize / numBatches);
    int32_t start = 0;

    while (start < totalSize) {
      const int32_t end = std::min(totalSize, start + batchSize);
      const int32_t size = end - start;

      std::vector<VectorPtr> children;
      children.push_back(makeFlatVector<int64_t>(size, [&](vector_size_t row) {
        return keys[start + row];
      }));
      children.push_back(makeFlatVector<int64_t>(size, [&](vector_size_t row) {
        return payloadBase + static_cast<int64_t>(start + row);
      }));

      batches.push_back(makeRowVector(children));
      start = end;
    }

    return batches;
  }
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();

  auto benchmark = std::make_unique<HashJoinRightSemiBenchmark>();
  std::vector<BenchmarkCase> benchmarkCases;

  std::vector<BenchmarkParams> params = {
      {// A moderate duplicate workload.
       .joinType = core::JoinType::kRightSemiFilter,
       .numMatchedKeys = 20'000,
       .buildDuplicates = 8,
       .probeDuplicates = 8,
       .numProbeBatches = 4,
       .outputLayout = {"u1"},
       .title = "right_semi_no_filter_dup8x8"},
      {// A heavy duplicate/skew-sensitive workload.
       .joinType = core::JoinType::kRightSemiFilter,
       .numMatchedKeys = 10'000,
       .buildDuplicates = 32,
       .probeDuplicates = 32,
       .numProbeBatches = 4,
       .outputLayout = {"u1"},
       .title = "right_semi_no_filter_dup32x32"},
      {// Control: keep filter=true path to compare against no-filter fast path.
       .joinType = core::JoinType::kRightSemiFilter,
       .numMatchedKeys = 10'000,
       .buildDuplicates = 32,
       .probeDuplicates = 32,
       .numProbeBatches = 4,
       .filter = "t1 >= 0",
       .outputLayout = {"u1"},
       .title = "right_semi_with_filter_dup32x32"},
  };

  // Each probe key matches a few build rows and appears in many probe rows,
  // so a probe-only filter is evaluated once per probe row instead of once
  // per matching build row. Compares filters that pass all and 1/4 of the
  // probe rows with no filter. Some keys only appear on one side.
  const std::vector<
      std::tuple<core::JoinType, std::string, std::vector<std::string>>>
      joinTypes = {
          {core::JoinType::kRight, "right_outer", {"t1", "u1"}},
          {core::JoinType::kFull, "full_outer", {"t1", "u1"}},
          {core::JoinType::kRightSemiFilter, "right_semi", {"u1"}},
          {core::JoinType::kRightAnti, "right_anti", {"u1"}},
      };
  const std::vector<std::pair<std::string, std::string>> filters = {
      {"", "no_filter"},
      {"t1 >= 0", "with_filter"},
      {"t1 % 4 = 0", "selective_filter"},
  };
  for (const auto& [joinType, joinName, outputLayout] : joinTypes) {
    for (const auto& [filter, filterName] : filters) {
      params.push_back(
          {.joinType = joinType,
           .numMatchedKeys = 1'000,
           .numBuildOnlyKeys = 1'000,
           .numProbeOnlyKeys = 1'000,
           .buildDuplicates = 4,
           .probeDuplicates = 128,
           .numProbeBatches = 8,
           .filter = filter,
           .outputLayout = outputLayout,
           .title = fmt::format("{}_{}_dup4x128", joinName, filterName)});
    }
  }

  for (const auto& param : params) {
    benchmarkCases.push_back({param, benchmark->prepareData(param)});
  }

  for (const auto& benchmarkCase : benchmarkCases) {
    folly::addBenchmark(
        __FILE__, benchmarkCase.params.title, [&benchmark, &benchmarkCase]() {
          auto outputRows =
              benchmark->run(benchmarkCase.params, benchmarkCase.data);
          folly::doNotOptimizeAway(outputRows);
          return 1;
        });
  }

  folly::runBenchmarks();
  return 0;
}
