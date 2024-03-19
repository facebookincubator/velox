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

#include "velox/common/base/SelectivityInfo.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/VectorHasher.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <folly/Benchmark.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <memory>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {
struct HashTableBenchmarkParams {
  HashTableBenchmarkParams() = default;

  HashTableBenchmarkParams(
      BaseHashTable::HashMode mode,
      TypePtr buildType,
      int64_t hashTableSize,
      int64_t probeSize,
      const std::vector<std::pair<int32_t, int32_t>>&
          keyRepeatTimesDistribution)
      : mode(mode),
        buildType(buildType),
        hashTableSize(hashTableSize),
        probeSize(probeSize),
        keyRepeatTimesDistribution(std::move(keyRepeatTimesDistribution)) {
    int32_t distSum = 0;
    buildSize = 0;
    buildKeyRepeat.reserve(keyRepeatTimesDistribution.size());
    for (auto dist : keyRepeatTimesDistribution) {
      if (dist.first > 100 || dist.first < 0 || dist.second < 0) {
        VELOX_FAIL(
            "Bad distribution: [distribution:{}, key duplicate count:{}]",
            dist.first,
            dist.second);
      }
      buildSize += (hashTableSize * dist.first / 100) * (dist.second + 1);
      distSum += dist.first;
      buildKeyRepeat.emplace_back(
          std::make_pair(hashTableSize * distSum / 100, dist.second));
    }
    VELOX_CHECK_EQ(distSum, 100, "Sum of distributions should be 100");

    if (hashTableSize > BaseHashTable::kArrayHashMaxSize &&
        mode == BaseHashTable::HashMode::kArray) {
      VELOX_FAIL("Bad hash mode.");
    }

    numDependentFields = buildType->size() - 1;
    if (mode == BaseHashTable::HashMode::kNormalizedKey) {
      extraValue = (2L << 20) + 100;
    } else if (mode == BaseHashTable::HashMode::kHash) {
      extraValue = std::numeric_limits<int64_t>::max() - 1;
    } else {
      extraValue = 0;
    }
    std::string modeString = mode == BaseHashTable::HashMode::kArray ? "array"
        : mode == BaseHashTable::HashMode::kHash                     ? "hash"
                                                 : "normalized key";
    std::stringstream distStr;
    for (auto dist : keyRepeatTimesDistribution) {
      distStr << fmt::format("{}%:{};", dist.first, dist.second);
    }
    title = fmt::format(
        "{}_{}_{}", modeString, numDependentFields + 1, distStr.str());
  }

  // Title for reporting
  std::string title;

  // Expected mode.
  BaseHashTable::HashMode mode;

  // Type of build & probe row.
  TypePtr buildType{ROW({"k1"}, {BIGINT()})};

  // Distinct rows in the table.
  int64_t hashTableSize;

  // Number of probe rows.
  int64_t probeSize;

  // Number of build rows.
  int64_t buildSize;

  // This is used to control the hash mode.
  int64_t extraValue;

  std::vector<std::pair<int32_t, int32_t>> keyRepeatTimesDistribution;

  std::vector<std::pair<int32_t, int32_t>> buildKeyRepeat;

  int32_t numDependentFields;

  int32_t numTables{7};

  std::string toString() const {
    std::string modeString = mode == BaseHashTable::HashMode::kArray ? "array"
        : mode == BaseHashTable::HashMode::kHash                     ? "hash"
                                                 : "normalized key";
    std::stringstream distStr;
    for (auto dist : keyRepeatTimesDistribution) {
      distStr << fmt::format("{}%:{}; ", dist.first, dist.second);
    }
    return fmt::format(
        "HashTableSize:{}, BuildInputSize:{}, ProbeInputSize:{}, ExpectHashMode:{}, KeyRepeatTimesDistribution: {}",
        hashTableSize,
        buildSize,
        probeSize,
        modeString,
        distStr.str());
  }
};

struct HashTableBenchmarkRun {
  HashTableBenchmarkParams params;

  // __rdtsc clocks per hashed probe row. Same for Velox and F14 cases.
  double prepareJoinTableClocks{0};

  // Result in __rdtsc clocks for total probe over number of probed rows.
  double listJoinResultClocks{0};

  int64_t numOutput;

  int32_t numIter{1};

  // The mode of the table.
  BaseHashTable::HashMode hashMode;

  void merge(HashTableBenchmarkRun other) {
    prepareJoinTableClocks += other.prepareJoinTableClocks;
    listJoinResultClocks += other.listJoinResultClocks;
    numIter++;
  }

  std::string toString() const {
    std::stringstream out;
    out << params.toString();

    std::string modeString = hashMode == BaseHashTable::HashMode::kArray
        ? "array"
        : hashMode == BaseHashTable::HashMode::kHash ? "hash"
                                                     : "normalized key";
    out << std::endl << " mode=" << modeString << " numOutput=" << numOutput;
    return out.str();
  }
};

class HashTableListJoinResultBenchmark : public VectorTestBase {
 public:
  int64_t getBuildKey(int64_t& buildKey, int32_t& iterTimes, int32_t& repeat) {
    if (iterTimes >= repeat) {
      iterTimes = 0;
      buildKey++;
      for (auto iter : params_.buildKeyRepeat) {
        if (buildKey < iter.first) {
          repeat = iter.second;
          break;
        }
      }
    } else {
      iterTimes++;
    }
    return buildKey;
  }

  RowVectorPtr makeBuildVector(
      int32_t size,
      int64_t& buildKey,
      int32_t& iterTimes,
      int32_t& repeat) {
    std::vector<int64_t> data;
    for (int32_t i = 0; i < size; ++i) {
      auto key = getBuildKey(buildKey, iterTimes, repeat);
      data.emplace_back(key);
    }
    data[0] = params_.extraValue;
    std::random_shuffle(data.begin(), data.end());
    std::vector<VectorPtr> children;
    children.push_back(vectorMaker_->flatVector<int64_t>(data));
    for (int32_t i = 0; i < params_.numDependentFields; ++i) {
      children.push_back(vectorMaker_->flatVector<int64_t>(
          size, [&](vector_size_t row) { return row + size; }, nullptr));
    }
    return vectorMaker_->rowVector(children);
  }

  void makeBuildRows(std::vector<RowVectorPtr>& batches) {
    int64_t buildKey = -1;
    int32_t iterTimes = 0;
    int32_t repeat = 0;
    int32_t size = params_.buildSize / params_.numTables;
    for (auto i = 0; i < params_.numTables; ++i) {
      batches.push_back(makeBuildVector(size, buildKey, iterTimes, repeat));
    }
  }

  RowVectorPtr
  makeProbeVector(int32_t size, int64_t hashTableSize, int64_t& sequence) {
    std::vector<VectorPtr> children;
    children.push_back(vectorMaker_->flatVector<int64_t>(
        size,
        [&](vector_size_t row) { return (sequence + row) % hashTableSize; },
        nullptr));
    sequence += size;
    for (int32_t i = 0; i < params_.numDependentFields; ++i) {
      children.push_back(vectorMaker_->flatVector<int64_t>(
          size, [&](vector_size_t row) { return row + size; }, nullptr));
    }
    return vectorMaker_->rowVector(children);
  }

  void copyVectorsToTable(RowVectorPtr batch, BaseHashTable* table) {
    int32_t batchSize = batch->size();
    raw_vector<uint64_t> dummy(batchSize);
    auto rowContainer = table->rows();
    auto& hashers = table->hashers();
    auto numKeys = hashers.size();
    auto numDependentFields = batch->childrenSize() - numKeys;

    std::vector<DecodedVector> decoders;
    decoders.reserve(numDependentFields);
    SelectivityVector rows(batchSize);

    for (auto i = 0; i < batch->childrenSize(); ++i) {
      if (i < numKeys) {
        auto hasher = table->hashers()[i].get();
        hasher->decode(*batch->childAt(i), rows);
        if (table->hashMode() != BaseHashTable::HashMode::kHash &&
            hasher->mayUseValueIds()) {
          hasher->computeValueIds(rows, dummy);
        }
      } else {
        decoders[i - numKeys].decode(*batch->childAt(i), rows);
      }
    }
    rows.applyToSelected([&](auto rowIndex) {
      char* newRow = rowContainer->newRow();
      *reinterpret_cast<char**>(newRow + rowContainer->nextOffset()) = nullptr;
      for (auto i = 0; i < numKeys; ++i) {
        rowContainer->store(hashers[i]->decodedVector(), rowIndex, newRow, i);
      }
      for (auto i = 0; i < numDependentFields; ++i) {
        rowContainer->store(decoders[i], rowIndex, newRow, i + numKeys);
      }
    });
  }

  void setParams(HashTableBenchmarkParams params) {
    params_ = params;
    topTable_.reset();
    vectorMaker_.reset();
    pool_.reset();
    pool_ = memory::memoryManager()->addLeafPool();
    vectorMaker_ = std::make_unique<VectorMaker>(pool_.get());
    // std::cout << params_.toString();
  }

  void buildTable() {
    std::vector<TypePtr> dependentTypes;
    std::vector<std::unique_ptr<BaseHashTable>> otherTables;
    std::vector<RowVectorPtr> batches;
    makeBuildRows(batches);
    for (auto i = 0; i < params_.numTables; ++i) {
      std::vector<std::unique_ptr<VectorHasher>> keyHashers;
      keyHashers.emplace_back(
          std::make_unique<VectorHasher>(params_.buildType->childAt(0), 0));
      auto table = HashTable<true>::createForJoin(
          std::move(keyHashers),
          dependentTypes,
          true,
          false,
          1'000,
          pool_.get());

      copyVectorsToTable(batches[i], table.get());
      if (i == 0) {
        topTable_ = std::move(table);
      } else {
        otherTables.push_back(std::move(table));
      }
    }
    SelectivityInfo buildTime;
    {
      SelectivityTimer timer(buildTime, 0);
      topTable_->prepareJoinTable(std::move(otherTables), executor_.get());
    }
    buildTime_ += (buildTime.timeToDropValue() / params_.buildSize);
    // std::cout << "Made table " << topTable_->toString();
  }

  int64_t probeTableAndListResult() {
    auto lookup = std::make_unique<HashLookup>(topTable_->hashers());
    auto numBatch = params_.probeSize / params_.hashTableSize;
    auto batchSize = params_.hashTableSize;
    SelectivityVector rows(batchSize);
    auto mode = topTable_->hashMode();
    SelectivityInfo listJoinResultTime;

    auto& hashers = topTable_->hashers();
    VectorHasher::ScratchMemory scratchMemory;
    BaseHashTable::JoinResultIterator results;
    BufferPtr outputRowMapping;
    auto outputBatchSize = batchSize;
    std::vector<char*> outputTableRows;
    int64_t sequence = 0;
    int64_t numJoinListResult = 0;
    for (auto i = 0; i < numBatch; ++i) {
      auto batch = makeProbeVector(batchSize, params_.hashTableSize, sequence);
      lookup->reset(batch->size());
      rows.setAll();
      for (auto i = 0; i < hashers.size(); ++i) {
        auto key = batch->childAt(i);
        if (mode != BaseHashTable::HashMode::kHash) {
          hashers[i]->lookupValueIds(*key, rows, scratchMemory, lookup->hashes);
        } else {
          hashers[i]->decode(*key, rows);
          hashers[i]->hash(rows, i > 0, lookup->hashes);
        }
      }
      lookup->rows.resize(rows.size());
      std::iota(lookup->rows.begin(), lookup->rows.end(), 0);
      topTable_->joinProbe(*lookup);
      int32_t numHit = 0;
      for (auto i = 0; i < lookup->rows.size(); ++i) {
        auto key = lookup->rows[i];
        numHit += lookup->hits[key] != nullptr;
      }
      // VELOX_CHECK_EQ(numHit, lookup->rows.size());
      results.reset(*lookup);
      auto mapping = initializeRowNumberMapping(
          outputRowMapping, outputBatchSize, pool_.get());
      outputTableRows.resize(outputBatchSize);
      {
        SelectivityTimer timer(listJoinResultTime, 0);
        while (!results.atEnd()) {
          numJoinListResult += topTable_->listJoinResults(
              results,
              false,
              mapping,
              folly::Range(outputTableRows.data(), outputTableRows.size()));
        }
        listJoinResultTime_ += listJoinResultTime.timeToDropValue();
      }
    }
    listJoinResultTime_ = listJoinResultTime_ / params_.probeSize;
    return numJoinListResult;
  }

  HashTableBenchmarkRun run() {
    HashTableBenchmarkRun result;
    result.params = params_;
    result.numOutput = probeTableAndListResult();
    result.prepareJoinTableClocks += buildTime_;
    result.listJoinResultClocks += listJoinResultTime_;
    result.hashMode = topTable_->hashMode();
    return result;
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};

  std::unique_ptr<VectorMaker> vectorMaker_{
      std::make_unique<VectorMaker>(pool_.get())};

  std::unique_ptr<HashTable<true>> topTable_;
  HashTableBenchmarkParams params_;

  float buildTime_{0};
  float listJoinResultTime_{0};
};

void combineResults(
    std::vector<HashTableBenchmarkRun>& results,
    HashTableBenchmarkRun run) {
  if (!results.empty() && results.back().params.title == run.params.title) {
    results.back().merge(run);
    return;
  }
  results.push_back(run);
}
} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManagerOptions options;
  options.useMmapAllocator = true;
  options.allocatorCapacity = 10UL << 30;
  options.useMmapArena = true;
  options.mmapArenaCapacityRatio = 1;
  memory::MemoryManager::initialize(options);

  auto bm = std::make_unique<HashTableListJoinResultBenchmark>();
  std::vector<HashTableBenchmarkRun> results;

  auto hashTableSize = (2L << 20) - 1;
  auto probeRowSize = 100000000L;

  TypePtr onlyKeyType{ROW({"k1"}, {BIGINT()})};

  TypePtr keyAndDependentType{
      ROW({"k1", "d1", "d2"}, {BIGINT(), BIGINT(), BIGINT()})};

  std::vector<TypePtr> buildTypes = {onlyKeyType};

  std::vector<BaseHashTable::HashMode> hashModes = {
      BaseHashTable::HashMode::kArray,
      BaseHashTable::HashMode::kNormalizedKey,
      BaseHashTable::HashMode::kHash};

  std::vector<std::vector<std::pair<int32_t, int32_t>>> keyRepeatDists = {
      // 20% of the rows are repeated only once, and 80% of the rows are not
      // repeated.
      {{20, 1}, {80, 0}},
      {{20, 5}, {80, 0}},
      {{20, 10}, {80, 0}},
      {{20, 20}, {80, 0}},
      {{20, 50}, {80, 0}},
      {{10, 5}, {10, 1}, {80, 0}},
      {{10, 10}, {10, 5}, {10, 1}, {70, 0}},
      {{10, 20}, {10, 10}, {10, 5}, {10, 1}, {60, 0}},
      {{10, 50}, {10, 20}, {10, 10}, {10, 5}, {10, 1}, {50, 0}}};
  std::vector<HashTableBenchmarkParams> params;
  for (auto mode : hashModes) {
    for (auto type : buildTypes) {
      for (auto& dist : keyRepeatDists) {
        params.emplace_back(HashTableBenchmarkParams(
            mode, type, hashTableSize, probeRowSize, dist));
      }
    }
  }

  for (auto& param : params) {
    folly::addBenchmark(__FILE__, param.title, [param, &bm, &results]() {
      {
        folly::BenchmarkSuspender suspender;
        bm->setParams(param);
        bm->buildTable();
      }
      combineResults(results, bm->run());
      return 1;
    });
  }
  folly::runBenchmarks();
  std::cout << "*** Results:" << std::endl;
  for (auto& result : results) {
    std::cout << result.toString() << std::endl;
  }
  return 0;
}
