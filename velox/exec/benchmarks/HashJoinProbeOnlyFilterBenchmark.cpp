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

// Benchmarks hash joins with a filter that only references probe columns. Such
// a filter can't be pushed below joins that keep probe rows without a match:
// left, full, left anti and left semi project joins. Compares filters that
// pass all and 1/4 of the probe rows with no filter, for 1, 4 and 32 build rows
// per key. Some keys only appear on one side. Reports the number of output
// rows as 'outputRows'. Also runs filters on a lazy probe column that is not
// projected to the output. One of them never reads the column, and the other
// reads it on the probe rows with a key match. The lazy column copies the rows
// it loads, and reports their number as 'loadedRows'.
namespace {

struct BenchmarkParams {
  core::JoinType joinType;
  // Number of distinct keys on both sides.
  int32_t numMatchedKeys;
  // Number of distinct keys that only appear once on the build side.
  int32_t numBuildOnlyKeys;
  // Number of distinct keys that only appear once on the probe side.
  int32_t numProbeOnlyKeys;
  // Number of build rows per key on both sides.
  int32_t buildDuplicates;
  // Number of probe rows per key on both sides.
  int32_t probeDuplicates;
  int32_t numProbeBatches;
  // Join filter. Empty means no filter.
  std::string filter;
  // If true, the probe side has a lazy column 't2' with the values of 't1'.
  bool lazyFilterColumn;
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

class HashJoinProbeOnlyFilterBenchmark : public VectorTestBase {
 public:
  BenchmarkData prepareData(const BenchmarkParams& params) {
    const auto buildKeys = makeKeys(
        params.numMatchedKeys,
        params.buildDuplicates,
        params.numMatchedKeys,
        params.numBuildOnlyKeys,
        /*seed=*/31);
    const auto probeKeys = makeKeys(
        params.numMatchedKeys,
        params.probeDuplicates,
        params.numMatchedKeys + params.numBuildOnlyKeys,
        params.numProbeOnlyKeys,
        /*seed=*/97);

    return {
        .buildVectors = makeBatches(buildKeys, 1, 0),
        .probeVectors =
            makeBatches(probeKeys, params.numProbeBatches, 10'000'000)};
  }

  // Returns the number of output rows. Adds the number of rows loaded from the
  // lazy column 't2' to 'numLoadedRows'.
  uint64_t run(
      const BenchmarkParams& params,
      const BenchmarkData& data,
      int64_t& numLoadedRows) {
    std::vector<RowVectorPtr> probeVectors;
    std::vector<std::string> probeProjections{"c0 AS t0", "c1 AS t1"};
    BENCHMARK_SUSPEND {
      probeVectors = data.probeVectors;
      if (params.lazyFilterColumn) {
        // A lazy vector is loaded only once, so each run needs new ones.
        for (auto& probe : probeVectors) {
          probe = makeRowVector({
              probe->childAt(0),
              probe->childAt(1),
              makeLazyCopy(probe->childAt(1), numLoadedRows),
          });
        }
        probeProjections.emplace_back("c2 AS t2");
      }
    }

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values(probeVectors)
                    .project(probeProjections)
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
  // Returns a lazy vector that copies the rows it loads from the flat BIGINT
  // vector 'source', and adds their number to 'numLoadedRows'.
  VectorPtr makeLazyCopy(const VectorPtr& source, int64_t& numLoadedRows) {
    return std::make_shared<LazyVector>(
        pool(),
        BIGINT(),
        source->size(),
        std::make_unique<SimpleVectorLoader>(
            [this, source, &numLoadedRows](RowSet rows) {
              numLoadedRows += rows.size();
              auto* sourceValues = source->asFlatVector<int64_t>();
              auto values = BaseVector::create<FlatVector<int64_t>>(
                  BIGINT(), source->size(), pool());
              for (const auto row : rows) {
                values->set(row, sourceValues->valueAt(row));
              }
              return values;
            }));
  }

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

  // Splits 'keys' into 'numBatches' batches with the key in column 'c0' and a
  // distinct non-negative payload starting at 'payloadBase' in column 'c1'.
  std::vector<RowVectorPtr> makeBatches(
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
      const int32_t size = std::min(totalSize, start + batchSize) - start;
      batches.push_back(makeRowVector({
          makeFlatVector<int64_t>(
              size, [&](vector_size_t row) { return keys[start + row]; }),
          makeFlatVector<int64_t>(
              size,
              [&](vector_size_t row) { return payloadBase + start + row; }),
      }));
      start += size;
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

  auto benchmark = std::make_unique<HashJoinProbeOnlyFilterBenchmark>();

  const std::vector<
      std::tuple<core::JoinType, std::string, std::vector<std::string>>>
      joinTypes = {
          {core::JoinType::kLeft, "left", {"t1", "u1"}},
          {core::JoinType::kFull, "full", {"t1", "u1"}},
          {core::JoinType::kAnti, "anti", {"t1"}},
          {core::JoinType::kLeftSemiProject,
           "left_semi_project",
           {"t1", "match"}},
      };
  // Filters with a flag that tells whether the probe side has a lazy column
  // 't2' with the values of 't1'. Probe keys are non-negative, so 't0 >= 0'
  // passes every row.
  const std::vector<std::tuple<std::string, std::string, bool>> filters = {
      {"", "no_filter", false},
      {"t1 >= 0", "with_filter", false},
      {"t1 % 4 = 0", "selective_filter", false},
      {"t0 >= 0 OR t2 % 4 = 0", "lazy_short_circuit_filter", true},
      {"t2 % 4 = 0", "lazy_selective_filter", true},
  };
  const std::vector<int32_t> buildDuplicates = {1, 4, 32};
  constexpr int32_t kProbeDuplicates = 128;

  std::vector<BenchmarkCase> benchmarkCases;
  for (const auto& [joinType, joinName, outputLayout] : joinTypes) {
    for (const auto duplicates : buildDuplicates) {
      for (const auto& [filter, filterName, lazyFilterColumn] : filters) {
        BenchmarkParams params{
            .joinType = joinType,
            .numMatchedKeys = 1'000,
            .numBuildOnlyKeys = 1'000,
            .numProbeOnlyKeys = 1'000,
            .buildDuplicates = duplicates,
            .probeDuplicates = kProbeDuplicates,
            .numProbeBatches = 8,
            .filter = filter,
            .lazyFilterColumn = lazyFilterColumn,
            .outputLayout = outputLayout,
            .title = fmt::format(
                "{}_{}_dup{}x{}",
                joinName,
                filterName,
                duplicates,
                kProbeDuplicates)};
        auto data = benchmark->prepareData(params);
        benchmarkCases.push_back({std::move(params), std::move(data)});
      }
    }
  }

  for (const auto& benchmarkCase : benchmarkCases) {
    folly::addBenchmark(
        __FILE__,
        benchmarkCase.params.title,
        [&benchmark, &benchmarkCase](folly::UserCounters& counters) {
          int64_t numLoadedRows{0};
          const auto outputRows = benchmark->run(
              benchmarkCase.params, benchmarkCase.data, numLoadedRows);
          counters["outputRows"] = static_cast<int64_t>(outputRows);
          if (benchmarkCase.params.lazyFilterColumn) {
            counters["loadedRows"] = numLoadedRows;
          }
          return 1;
        });
  }

  folly::runBenchmarks();
  return 0;
}
