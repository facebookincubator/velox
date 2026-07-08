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
  int32_t numDistinctKeys;
  int32_t buildDuplicates;
  int32_t probeDuplicates;
  int32_t numProbeBatches;
  bool withFilter;
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
        params.numDistinctKeys,
        params.buildDuplicates,
        31 /*seed for deterministic shuffling*/);
    auto probeKeys = makeKeys(
        params.numDistinctKeys,
        params.probeDuplicates,
        97 /*seed for deterministic shuffling*/);

    data.buildVectors = makeBatchesFromKeys(buildKeys, 1, 0);
    data.probeVectors =
        makeBatchesFromKeys(probeKeys, params.numProbeBatches, 10'000'000);
    return data;
  }

  int64_t run(const BenchmarkParams& params, const BenchmarkData& data) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    const std::string filter = params.withFilter ? "t1 >= 0" : "";

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
                        filter,
                        {"u1"},
                        core::JoinType::kRightSemiFilter)
                    .planNode();

    auto results = AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());
    return results->size();
  }

 private:
  std::vector<int64_t>
  makeKeys(int32_t numDistinctKeys, int32_t duplicates, uint32_t seed) {
    VELOX_CHECK_GT(numDistinctKeys, 0);
    VELOX_CHECK_GT(duplicates, 0);

    std::vector<int64_t> keys(numDistinctKeys * duplicates);
    for (int64_t i = 0; i < keys.size(); ++i) {
      keys[i] = i % numDistinctKeys;
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

  const std::vector<BenchmarkParams> params = {
      {// A moderate duplicate workload.
       .numDistinctKeys = 20'000,
       .buildDuplicates = 8,
       .probeDuplicates = 8,
       .numProbeBatches = 4,
       .withFilter = false,
       .title = "right_semi_no_filter_dup8x8"},
      {// A heavy duplicate/skew-sensitive workload.
       .numDistinctKeys = 10'000,
       .buildDuplicates = 32,
       .probeDuplicates = 32,
       .numProbeBatches = 4,
       .withFilter = false,
       .title = "right_semi_no_filter_dup32x32"},
      {// Control: keep filter=true path to compare against no-filter fast path.
       .numDistinctKeys = 10'000,
       .buildDuplicates = 32,
       .probeDuplicates = 32,
       .numProbeBatches = 4,
       .withFilter = true,
       .title = "right_semi_with_filter_dup32x32"},
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
