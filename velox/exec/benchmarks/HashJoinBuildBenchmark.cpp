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
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <iostream>
#include <utility>

#include "velox/exec/HashBuild.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/VectorTestUtil.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

namespace {
struct BenchmarkParams {
  BenchmarkParams() = default;

  // Benchmark params, we need to provide:
  //  -the expect hash mode,
  //  -the build row schema,
  //  -the duplicate factor,
  //  -number of building rows,
  //  -number of probing rows,
  //  -the abandon percentage,
  //  -the number of build vector batches.
  BenchmarkParams(
      BaseHashTable::HashMode mode,
      const TypePtr& buildType,
      double dupFactor,
      int64_t buildSize,
      int64_t probeSize,
      int32_t abandonPct,
      int32_t numBuildBatches)
      : mode{mode},
        buildType{buildType},
        hashTableSize{static_cast<long>(std::floor(buildSize / dupFactor))},
        buildSize{buildSize},
        probeSize{probeSize},
        numBuildBatches{numBuildBatches},
        dupFactor{dupFactor},
        abandonPct{abandonPct} {
    VELOX_CHECK_LE(hashTableSize, buildSize);
    VELOX_CHECK_GE(numBuildBatches, 1);

    if (hashTableSize > BaseHashTable::kArrayHashMaxSize &&
        mode == BaseHashTable::HashMode::kArray) {
      VELOX_FAIL("Bad hash mode.");
    }

    numFields = buildType->size();
    if (mode == BaseHashTable::HashMode::kNormalizedKey) {
      extraValue = BaseHashTable::kArrayHashMaxSize + 100;
    } else if (mode == BaseHashTable::HashMode::kHash) {
      extraValue = std::numeric_limits<int64_t>::max() - 1;
    } else {
      extraValue = 0;
    }

    title = fmt::format(
        "dupFactor:{:<2},abandonPct:{},{}",
        dupFactor,
        abandonPct,
        BaseHashTable::modeString(mode));
  }

  // Expected mode.
  BaseHashTable::HashMode mode;

  // Type of build & probe row.
  TypePtr buildType;

  // Distinct rows in the table.
  int64_t hashTableSize;

  // Number of build rows.
  int64_t buildSize;

  // Number of probe rows.
  int64_t probeSize;

  // Number of build RowContainers.
  int32_t numBuildBatches;

  // Title for reporting.
  std::string title;

  // The duplicate factor, 2 means every row will repeat 2 times.
  double dupFactor;

  // This parameter controls the hashing mode. It is incorporated into the keys
  // on the build side. If the expected mode is an array, its value is 0. If
  // the expected mode is a normalized key, its value is 'kArrayHashMaxSize' +
  // 100 to make the key range > 'kArrayHashMaxSize'. If the expected mode is a
  // hash, its value is the maximum value of int64_t minus 1 to make the key
  // range  == 'kRangeTooLarge'.
  int64_t extraValue;

  // Number of fields.
  int32_t numFields;

  int32_t abandonPct;

  std::string toString() const {
    return fmt::format(
        "DupFactor:{:<2}, AbandonPct:{}, HashMode:{:<14}",
        dupFactor,
        abandonPct,
        BaseHashTable::modeString(mode));
  }
};

struct BenchmarkResult {
  BenchmarkParams params;

  uint64_t totalClock{0};

  uint64_t hashBuildPeakMemoryBytes{0};

  bool isBuildNoDupHashTableAbandon{false};

  // The mode of the table.
  BaseHashTable::HashMode hashMode;

  std::string toString() const {
    return fmt::format(
        "{}, isAbandon:{:<5}, totalClock:{}ms, peakMemoryBytes:{}",
        params.toString(),
        isBuildNoDupHashTableAbandon,
        totalClock / 1000'000,
        succinctBytes(hashBuildPeakMemoryBytes));
  }
};

class HashJoinBuildBenchmark : public VectorTestBase {
 public:
  HashJoinBuildBenchmark() : randomEngine_((std::random_device{}())) {}

  BenchmarkResult run(BenchmarkParams params) {
    params_ = std::move(params);
    BenchmarkResult result;
    result.params = params_;
    result.hashMode = params_.mode;

    std::vector<RowVectorPtr> buildVectors;
    makeBuildBatches(buildVectors);

    int64_t sequence = 0;
    int64_t batchSize = params_.probeSize / 4;
    std::vector<RowVectorPtr> probeVectors;
    for (auto i = 0; i < 4; ++i) {
      auto batch = makeProbeVector(batchSize, params_.hashTableSize, sequence);
      probeVectors.emplace_back(batch);
    }

    uint64_t totalClocks{0};
    {
      ClockTimer timer(totalClocks);
      auto plan = makeHashJoinPlan(buildVectors, probeVectors);
      CursorParameters cursorParams;
      cursorParams.planNode = std::move(plan);
      cursorParams.queryCtx = core::QueryCtx::create(
          executor_.get(),
          core::QueryConfig{{}},
          {},
          cache::AsyncDataCache::getInstance(),
          rootPool_);
      cursorParams.queryCtx->testingOverrideConfigUnsafe({
          {core::QueryConfig::kAbandonDedupHashMapMinPct,
           std::to_string(params_.abandonPct)},
          {core::QueryConfig::kAbandonDedupHashMapMinRows, "1000000"},
      });

      cursorParams.maxDrivers = 1;
      auto cursor = TaskCursor::create(cursorParams);
      auto* task = cursor->task().get();
      while (cursor->moveNext()) {
      }
      waitForTaskCompletion(task);
      result.isBuildNoDupHashTableAbandon = isBuildNoDupHashTableAbandon(task);
    }
    result.totalClock = totalClocks;

    result.hashBuildPeakMemoryBytes = getHashBuildPeakMemory(rootPool_.get());
    return result;
  }

 private:
  std::shared_ptr<const core::PlanNode> makeHashJoinPlan(
      const std::vector<RowVectorPtr>& buildVectors,
      const std::vector<RowVectorPtr>& probeVectors) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
        .values(probeVectors)
        .project({"c0 AS t0", "c1 as t1", "c2 as t2"})
        .hashJoin(
            {"t0"},
            {"u0"},
            exec::test::PlanBuilder(planNodeIdGenerator)
                .values(buildVectors)
                .project({"c0 AS u0"})
                .planNode(),
            "",
            {"t0", "t1", "match"},
            core::JoinType::kLeftSemiProject)
        .planNode();
  }

  // Create the row vector for the build side, where the first column is used
  // as the join key, and the remaining columns are dependent fields.
  // If expect mode is array, the key is within the range [0, hashTableSize];
  // If expect mode is normalized key, the key is within the range
  // [0, hashTableSize] + extraValue(kArrayHashMaxSize + 100);
  // If expect mode is hash, the key is within the range [0, hashTableSize] +
  // extraValue(max_int64 -1);
  RowVectorPtr makeBuildRows(
      std::vector<int64_t>& data,
      int64_t start,
      int64_t end,
      bool addExtraValue) {
    auto subData =
        std::vector<int64_t>(data.begin() + start, data.begin() + end);
    if (addExtraValue) {
      subData[0] = params_.extraValue;
    }

    std::vector<VectorPtr> children;
    children.push_back(makeFlatVector<int64_t>(subData));
    return makeRowVector(children);
  }

  // Generate the build side data batches.
  void makeBuildBatches(std::vector<RowVectorPtr>& batches) {
    int64_t buildKey = 0;
    std::vector<int64_t> data;
    for (auto i = 0; i < params_.buildSize; ++i) {
      data.emplace_back((buildKey++) % params_.hashTableSize);
    }
    std::shuffle(data.begin(), data.end(), randomEngine_);

    auto size = params_.buildSize / params_.numBuildBatches;
    for (auto i = 0; i < params_.numBuildBatches; ++i) {
      batches.push_back(makeBuildRows(
          data,
          i * size,
          (i + 1) * size + 1,
          i == params_.numBuildBatches - 1));
    }
  }

  // Create the row vector for the probe side, where the first column is used
  // as the join key, and the remaining columns are dependent fields.
  // Probe key is within the range [0, hashTableSize].
  RowVectorPtr
  makeProbeVector(int64_t size, int64_t hashTableSize, int64_t& sequence) {
    std::vector<VectorPtr> children;
    for (int32_t i = 0; i < params_.numFields; ++i) {
      children.push_back(
          makeFlatVector<int64_t>(
              size,
              [&](vector_size_t row) {
                return (sequence + row) % hashTableSize;
              },
              nullptr));
    }
    sequence += size;

    for (int32_t i = 0; i < 2; ++i) {
      children.push_back(
          makeFlatVector<int64_t>(
              size, [&](vector_size_t row) { return row + size; }, nullptr));
    }
    return makeRowVector(children);
  }

  static int64_t getHashBuildPeakMemory(memory::MemoryPool* rootPool) {
    int64_t hashBuildPeakBytes = 0;
    std::vector<memory::MemoryPool*> pools;
    pools.push_back(rootPool);
    while (!pools.empty()) {
      std::vector<memory::MemoryPool*> childPools;
      for (auto pool : pools) {
        pool->visitChildren([&](memory::MemoryPool* childPool) -> bool {
          if (childPool->name().find("HashBuild") != std::string::npos) {
            hashBuildPeakBytes += childPool->peakBytes();
          }
          childPools.push_back(childPool);
          return true;
        });
      }
      pools.swap(childPools);
    }
    if (hashBuildPeakBytes == 0) {
      VELOX_FAIL("Failed to get HashBuild peak memory");
    }
    return hashBuildPeakBytes;
  }

  static bool isBuildNoDupHashTableAbandon(exec::Task* task) {
    for (auto& pipelineStat : task->taskStats().pipelineStats) {
      for (auto& operatorStat : pipelineStat.operatorStats) {
        if (operatorStat.operatorType == OperatorType::kHashBuild) {
          return operatorStat
                     .runtimeStats[std::string(
                         HashBuild::kAbandonBuildNoDupHash)]
                     .count != 0;
        }
      }
    }
    return false;
  }

  std::default_random_engine randomEngine_;
  BenchmarkParams params_;
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::Options options;
  options.useMmapAllocator = true;
  options.allocatorCapacity = 10UL << 30;
  options.useMmapArena = true;
  options.mmapArenaCapacityRatio = 1;
  memory::MemoryManager::initialize(options);

  auto bm = std::make_unique<HashJoinBuildBenchmark>();
  std::vector<BenchmarkResult> results;

  auto buildRowSize = (2L << 20) - 3;
  auto probeRowSize = 100000000L;

  TypePtr twoKeyType{ROW({"k1"}, {BIGINT()})};

  const std::vector<BaseHashTable::HashMode> hashModes = {
      BaseHashTable::HashMode::kArray,
      BaseHashTable::HashMode::kNormalizedKey,
      BaseHashTable::HashMode::kHash,
  };
  const std::vector<double> dupFactorVector = {
      2,
      8,
      32,
  };
  const std::vector<int32_t> abandonPcts = {
      90,
      80,
      70,
      50,
      0,
  };

  std::vector<BenchmarkParams> params;
  for (auto mode : hashModes) {
    for (auto dupFactor : dupFactorVector) {
      for (auto pct : abandonPcts) {
        params.push_back(BenchmarkParams(
            mode, twoKeyType, dupFactor, buildRowSize, probeRowSize, pct, 512));
      }
    }
  }

  for (auto& param : params) {
    BenchmarkResult result;
    folly::addBenchmark(__FILE__, param.title, [param, &results, &bm]() {
      results.emplace_back(bm->run(param));
      return 1;
    });
  }

  folly::runBenchmarks();

  for (auto& result : results) {
    std::cout << result.toString() << std::endl;
  }
  return 0;
}
