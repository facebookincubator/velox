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
#include <iostream>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;

namespace {

/// Benchmark for the MarkSorted operator.
/// Measures overhead of sortedness validation with different configurations:
/// - Small vs large batches (zero-copy vs copy mode)
/// - Single vs multiple keys
/// - BIGINT keys (fast path) vs VARCHAR keys (generic path)
/// - Sorted vs unsorted data

class MarkSortedBenchmark {
 public:
  void addBenchmark(
      const std::string& benchmarkName,
      int32_t numBatches,
      int32_t batchSize,
      int32_t numKeys,
      TypeKind keyType,
      double unsortedRatio,
      int32_t zeroCopyThreshold) {
    folly::addBenchmark(__FILE__, benchmarkName, [=, this]() {
      runBenchmark(
          numBatches,
          batchSize,
          numKeys,
          keyType,
          unsortedRatio,
          zeroCopyThreshold);
      return 1;
    });
  }

  void addBaselineBenchmark(
      const std::string& benchmarkName,
      int32_t numBatches,
      int32_t batchSize,
      int32_t numKeys,
      TypeKind keyType) {
    // Baseline: just Values operator without MarkSorted
    folly::addBenchmark(__FILE__, benchmarkName, [=, this]() {
      runBaseline(numBatches, batchSize, numKeys, keyType);
      return 1;
    });
  }

 private:
  std::vector<RowVectorPtr> generateSortedData(
      int32_t numBatches,
      int32_t batchSize,
      int32_t numKeys,
      TypeKind keyType,
      double unsortedRatio) {
    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);

    int64_t currentValue = 0;
    const int32_t unsortedInterval =
        unsortedRatio > 0 ? static_cast<int32_t>(1.0 / unsortedRatio) : 0;

    for (int32_t b = 0; b < numBatches; ++b) {
      std::vector<VectorPtr> children;
      children.reserve(numKeys);

      for (int32_t k = 0; k < numKeys; ++k) {
        if (keyType == TypeKind::BIGINT) {
          std::vector<int64_t> values(batchSize);
          for (int32_t i = 0; i < batchSize; ++i) {
            // Introduce unsorted values periodically.
            if (unsortedInterval > 0 && i > 0 && (i % unsortedInterval) == 0) {
              values[i] = currentValue - 10; // Out of order
            } else {
              values[i] = currentValue++;
            }
          }
          children.push_back(
              BaseVector::create<FlatVector<int64_t>>(
                  BIGINT(), batchSize, pool_.get()));
          auto* flatVector = children.back()->as<FlatVector<int64_t>>();
          for (int32_t i = 0; i < batchSize; ++i) {
            flatVector->set(i, values[i]);
          }
        } else if (keyType == TypeKind::VARCHAR) {
          children.push_back(
              BaseVector::create<FlatVector<StringView>>(
                  VARCHAR(), batchSize, pool_.get()));
          auto* flatVector = children.back()->as<FlatVector<StringView>>();
          for (int32_t i = 0; i < batchSize; ++i) {
            auto value = currentValue++;
            // Introduce unsorted values periodically.
            if (unsortedInterval > 0 && i > 0 && (i % unsortedInterval) == 0) {
              value = currentValue - 20;
            }
            auto str = fmt::format("value_{:012d}", value);
            flatVector->set(i, StringView(str));
          }
        }
      }

      std::vector<std::string> names;
      std::vector<TypePtr> types;
      for (int32_t k = 0; k < numKeys; ++k) {
        names.push_back(fmt::format("c{}", k));
        types.push_back(children[k]->type());
      }
      auto rowType = ROW(std::move(names), std::move(types));
      batches.push_back(
          std::make_shared<RowVector>(
              pool_.get(), rowType, nullptr, batchSize, std::move(children)));
    }

    return batches;
  }

  void runBenchmark(
      int32_t numBatches,
      int32_t batchSize,
      int32_t numKeys,
      TypeKind keyType,
      double unsortedRatio,
      int32_t zeroCopyThreshold) {
    folly::BenchmarkSuspender suspender;

    auto batches = generateSortedData(
        numBatches, batchSize, numKeys, keyType, unsortedRatio);

    std::vector<std::string> sortingKeys;
    std::vector<core::SortOrder> sortingOrders;
    for (int32_t k = 0; k < numKeys; ++k) {
      sortingKeys.push_back(fmt::format("c{}", k));
      sortingOrders.push_back(core::kAscNullsLast);
    }

    core::PlanNodeId markSortedNodeId;
    auto plan = PlanBuilder()
                    .values(batches)
                    .markSorted("is_sorted", sortingKeys, sortingOrders)
                    .capturePlanNodeId(markSortedNodeId)
                    .planNode();

    suspender.dismiss();

    std::shared_ptr<Task> task;
    AssertQueryBuilder(plan)
        .config(
            core::QueryConfig::kMarkSortedZeroCopyThreshold,
            std::to_string(zeroCopyThreshold))
        .countResults(task);

    auto taskStats = exec::toPlanStats(task->taskStats());
    auto& stats = taskStats.at(markSortedNodeId);
    std::cout << "MarkSorted: Input "
              << succinctNanos(stats.addInputTiming.wallNanos) << " Output "
              << succinctNanos(stats.getOutputTiming.wallNanos) << std::endl;
  }

  void runBaseline(
      int32_t numBatches,
      int32_t batchSize,
      int32_t numKeys,
      TypeKind keyType) {
    folly::BenchmarkSuspender suspender;

    auto batches =
        generateSortedData(numBatches, batchSize, numKeys, keyType, 0.0);

    auto plan = PlanBuilder().values(batches).planNode();

    suspender.dismiss();

    std::shared_ptr<Task> task;
    AssertQueryBuilder(plan).countResults(task);
  }

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      rootPool_->addLeafChild("MarkSortedBenchmark")};
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});

  MarkSortedBenchmark bm;

  // Baseline benchmarks (no MarkSorted).
  bm.addBaselineBenchmark(
      "Baseline_SmallBatch_SingleKey_Bigint",
      100, // numBatches
      1000, // batchSize
      1, // numKeys
      TypeKind::BIGINT);

  bm.addBaselineBenchmark(
      "Baseline_LargeBatch_SingleKey_Bigint",
      10, // numBatches
      10000, // batchSize
      1, // numKeys
      TypeKind::BIGINT);

  // Small batch benchmarks (zero-copy path).
  bm.addBenchmark(
      "MarkSorted_SmallBatch_SingleKey_Bigint",
      100, // numBatches
      1000, // batchSize
      1, // numKeys
      TypeKind::BIGINT,
      0.0, // unsortedRatio
      2000); // zeroCopyThreshold

  bm.addBenchmark(
      "MarkSorted_SmallBatch_MultiKey_Bigint",
      100, // numBatches
      1000, // batchSize
      3, // numKeys
      TypeKind::BIGINT,
      0.0, // unsortedRatio
      2000); // zeroCopyThreshold

  bm.addBenchmark(
      "MarkSorted_SmallBatch_SingleKey_Varchar",
      100, // numBatches
      1000, // batchSize
      1, // numKeys
      TypeKind::VARCHAR,
      0.0, // unsortedRatio
      2000); // zeroCopyThreshold

  // Large batch benchmarks (copy path).
  bm.addBenchmark(
      "MarkSorted_LargeBatch_SingleKey_Bigint",
      10, // numBatches
      10000, // batchSize
      1, // numKeys
      TypeKind::BIGINT,
      0.0, // unsortedRatio
      5000); // zeroCopyThreshold (triggers copy mode)

  bm.addBenchmark(
      "MarkSorted_LargeBatch_MultiKey_Bigint",
      10, // numBatches
      10000, // batchSize
      3, // numKeys
      TypeKind::BIGINT,
      0.0, // unsortedRatio
      5000); // zeroCopyThreshold

  bm.addBenchmark(
      "MarkSorted_LargeBatch_SingleKey_Varchar",
      10, // numBatches
      10000, // batchSize
      1, // numKeys
      TypeKind::VARCHAR,
      0.0, // unsortedRatio
      5000); // zeroCopyThreshold

  // Unsorted data benchmarks.
  bm.addBenchmark(
      "MarkSorted_SmallBatch_SingleKey_Bigint_1pctUnsorted",
      100, // numBatches
      1000, // batchSize
      1, // numKeys
      TypeKind::BIGINT,
      0.01, // 1% unsorted
      2000); // zeroCopyThreshold

  folly::runBenchmarks();
  return 0;
}
