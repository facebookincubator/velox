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

#include <algorithm>
#include <random>

#include "velox/common/memory/Memory.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

namespace {

struct BenchmarkParams {
  core::JoinType joinType;
  int32_t numMatchedKeys;
  int32_t numBuildOnlyKeys;
  int32_t numProbeOnlyKeys;
  int32_t buildDuplicates;
  int32_t probeDuplicates;
  int32_t numProbeBatches;
  std::string filter;
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

class HashJoinProbedFlagBenchmark : public VectorTestBase {
 public:
  BenchmarkData prepareData(const BenchmarkParams& params) {
    BenchmarkData data;

    auto buildKeys = makeRepeatedKeys(
        0,
        params.numMatchedKeys,
        params.buildDuplicates,
        31 /*seed for deterministic shuffling*/);
    appendRepeatedKeys(
        buildKeys,
        params.numMatchedKeys,
        params.numBuildOnlyKeys,
        1 /*duplicates*/);
    shuffle(buildKeys, 43);

    auto probeKeys = makeRepeatedKeys(
        0,
        params.numMatchedKeys,
        params.probeDuplicates,
        97 /*seed for deterministic shuffling*/);
    appendRepeatedKeys(
        probeKeys,
        params.numMatchedKeys + params.numBuildOnlyKeys,
        params.numProbeOnlyKeys,
        1 /*duplicates*/);
    shuffle(probeKeys, 109);

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
  std::vector<int64_t> makeRepeatedKeys(
      int64_t start,
      int32_t numKeys,
      int32_t duplicates,
      uint32_t seed) {
    VELOX_CHECK_GE(numKeys, 0);
    VELOX_CHECK_GT(duplicates, 0);

    std::vector<int64_t> keys;
    keys.reserve(numKeys * duplicates);
    appendRepeatedKeys(keys, start, numKeys, duplicates);
    shuffle(keys, seed);
    return keys;
  }

  void appendRepeatedKeys(
      std::vector<int64_t>& keys,
      int64_t start,
      int32_t numKeys,
      int32_t duplicates) {
    for (auto i = 0; i < numKeys; ++i) {
      for (auto j = 0; j < duplicates; ++j) {
        keys.push_back(start + i);
      }
    }
  }

  void shuffle(std::vector<int64_t>& keys, uint32_t seed) {
    std::mt19937 random(seed);
    std::shuffle(keys.begin(), keys.end(), random);
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

  auto benchmark = std::make_unique<HashJoinProbedFlagBenchmark>();
  std::vector<BenchmarkCase> benchmarkCases;

  constexpr int32_t kMatchedKeys = 1'000;
  constexpr int32_t kBuildDuplicates = 4;
  constexpr int32_t kProbeDuplicates = 128;
  constexpr int32_t kOuterOnlyKeys = 1'000;
  constexpr int32_t kProbeBatches = 8;

  const std::vector<BenchmarkParams> params = {
      {.joinType = core::JoinType::kRight,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = 0,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "",
       .outputLayout = {"t1", "u1"},
       .title = "right_outer_dup4x128"},
      {.joinType = core::JoinType::kRight,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = 0,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "t1 >= 0",
       .outputLayout = {"t1", "u1"},
       .title = "right_outer_with_filter_dup4x128"},
      {.joinType = core::JoinType::kRight,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = 0,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "t1 % 4 = 0",
       .outputLayout = {"t1", "u1"},
       .title = "right_outer_selective_filter_dup4x128"},
      {.joinType = core::JoinType::kFull,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = kOuterOnlyKeys,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "",
       .outputLayout = {"t1", "u1"},
       .title = "full_outer_dup4x128"},
      {.joinType = core::JoinType::kFull,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = kOuterOnlyKeys,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "t1 >= 0",
       .outputLayout = {"t1", "u1"},
       .title = "full_outer_with_filter_dup4x128"},
      {.joinType = core::JoinType::kFull,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = kOuterOnlyKeys,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "t1 % 4 = 0",
       .outputLayout = {"t1", "u1"},
       .title = "full_outer_selective_filter_dup4x128"},
      {.joinType = core::JoinType::kRightSemiFilter,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = 0,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "t1 >= 0",
       .outputLayout = {"u1"},
       .title = "right_semi_with_filter_dup4x128"},
      {.joinType = core::JoinType::kRightSemiFilter,
       .numMatchedKeys = kMatchedKeys,
       .numBuildOnlyKeys = kOuterOnlyKeys,
       .numProbeOnlyKeys = 0,
       .buildDuplicates = kBuildDuplicates,
       .probeDuplicates = kProbeDuplicates,
       .numProbeBatches = kProbeBatches,
       .filter = "t1 % 4 = 0",
       .outputLayout = {"u1"},
       .title = "right_semi_selective_filter_dup4x128"},
  };

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
