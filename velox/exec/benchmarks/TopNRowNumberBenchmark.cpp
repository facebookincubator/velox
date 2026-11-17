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

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using facebook::velox::RowVector;
using facebook::velox::RowVectorPtr;
using facebook::velox::vector_size_t;
using facebook::velox::core::PlanNodePtr;
using facebook::velox::core::QueryConfig;
using facebook::velox::exec::test::AssertQueryBuilder;
using facebook::velox::exec::test::PlanBuilder;
using facebook::velox::memory::MemoryManager;
using facebook::velox::test::VectorTestBase;

namespace {

// Measures candidate maintenance with large and small peer groups.
class TopNRowNumberBenchmark : public VectorTestBase {
 public:
  // Registers a query with input construction outside the timed execution.
  void addBenchmark(
      const std::string& function,
      int32_t limit,
      int32_t numPartitions,
      bool manyPeers) {
    constexpr vector_size_t kNumRows = 100'000;
    constexpr vector_size_t kBatchSize = 4'096;
    auto data = makeRowVector({
        makeFlatVector<int64_t>(
            kNumRows,
            [numPartitions](auto row) { return row % numPartitions; }),
        makeFlatVector<int64_t>(
            kNumRows,
            [numPartitions, manyPeers](auto row) {
              const auto partitionRow = row / numPartitions;
              return manyPeers ? (partitionRow % 128 == 127 ? 99 : 100)
                               : (partitionRow * 7'919) % 10'000;
            }),
    });
    std::vector<RowVectorPtr> batches;
    for (vector_size_t offset = 0; offset < kNumRows; offset += kBatchSize) {
      batches.push_back(
          std::dynamic_pointer_cast<RowVector>(data->slice(
              offset, std::min<vector_size_t>(kBatchSize, kNumRows - offset))));
    }
    const auto planIndex = plans_.size();
    plans_.push_back(
        PlanBuilder()
            .values(batches)
            .topNRank(function, {"c0"}, {"c1 DESC"}, limit, false)
            .planNode());
    folly::addBenchmark(
        __FILE__,
        fmt::format(
            "{}_limit{}_partitions{}_{}",
            function,
            limit,
            numPartitions,
            manyPeers ? "manyPeers" : "fewPeers"),
        [this, planIndex]() {
          const auto count =
              AssertQueryBuilder(plans_[planIndex])
                  .maxDrivers(1)
                  .config(
                      QueryConfig::kAbandonPartialTopNRowNumberMinRows,
                      2'147'483'647)
                  .countResults();
          folly::doNotOptimizeAway(count);
          return 1;
        });
  }

 private:
  // Releases input vectors before the base class destroys their memory pool.
  std::vector<PlanNodePtr> plans_;
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  MemoryManager::initialize(MemoryManager::Options{});
  TopNRowNumberBenchmark benchmark;
  for (const auto& function : {"row_number", "rank", "dense_rank"}) {
    for (int32_t limit : {10, 100}) {
      for (int32_t numPartitions : {1, 64}) {
        for (bool manyPeers : {false, true}) {
          benchmark.addBenchmark(function, limit, numPartitions, manyPeers);
        }
      }
    }
  }
  folly::runBenchmarks();
  return 0;
}
