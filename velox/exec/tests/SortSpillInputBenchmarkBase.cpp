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

#include "velox/exec/tests/SortSpillInputBenchmarkBase.h"
#include <gflags/gflags.h>

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::memory;
using namespace facebook::velox::exec;
namespace facebook::velox::exec::test {

namespace {
std::unique_ptr<RowContainer> makeRowContainer(
    const std::vector<TypePtr>& keyTypes,
    const std::vector<TypePtr>& dependentTypes,
    std::shared_ptr<velox::memory::MemoryPool>& pool) {
  return std::make_unique<RowContainer>(keyTypes, dependentTypes, pool.get());
}

std::unique_ptr<RowContainer> setupSpillContainer(
    const RowTypePtr& rowType,
    uint32_t numKeys,
    std::shared_ptr<velox::memory::MemoryPool>& pool) {
  const auto& childTypes = rowType->children();
  std::vector<TypePtr> keys(childTypes.begin(), childTypes.begin() + numKeys);
  std::vector<TypePtr> dependents;
  if (numKeys < childTypes.size()) {
    dependents.insert(
        dependents.end(), childTypes.begin() + numKeys, childTypes.end());
  }
  return makeRowContainer(keys, dependents, pool);
}
} // namespace

void SortSpillInputBenchmarkBase::setUp(
    RowTypePtr rowType,
    bool serializeRowContainer,
    int32_t stringMaxLength) {
  SpillerBenchmarkBase::setUp(rowType, stringMaxLength);
  serializeRowContainer_ = serializeRowContainer;
  rowContainer_ = setupSpillContainer(
      rowType_, FLAGS_spiller_benchmark_num_key_columns, pool_);
  writeSpillData();
  spiller_ = makeSpiller();
}

void SortSpillInputBenchmarkBase::writeSpillData() {
  vector_size_t numRows = 0;
  for (const auto& rowVector : rowVectors_) {
    numRows += rowVector->size();
  }

  std::vector<char*> rows;
  rows.resize(numRows);
  for (int i = 0; i < numRows; ++i) {
    rows[i] = rowContainer_->newRow();
  }

  vector_size_t nextRow = 0;
  for (const auto& rowVector : rowVectors_) {
    const SelectivityVector allRows(rowVector->size());
    for (int index = 0; index < rowVector->size(); ++index, ++nextRow) {
      for (int i = 0; i < rowType_->size(); ++i) {
        DecodedVector decodedVector(*rowVector->childAt(i), allRows);
        rowContainer_->store(decodedVector, index, rows[nextRow], i);
      }
    }
  }
}

void SortSpillInputBenchmarkBase::run() {
  MicrosecondTimer timer(&executionTimeUs_);
  auto sortInputSpiller = dynamic_cast<SortInputSpiller*>(spiller_.get());
  VELOX_CHECK_NOT_NULL(sortInputSpiller);
  sortInputSpiller->spill();
  rowContainer_->clear();
}

std::unique_ptr<SpillerBase> SortSpillInputBenchmarkBase::makeSpiller() {
  common::SpillConfig spillConfig;
  spillConfig.getSpillDirPathCb = [&]() -> std::string_view {
    return spillDir_;
  };
  spillConfig.updateAndCheckSpillLimitCb = [&](uint64_t) {};
  spillConfig.fileNamePrefix = FLAGS_spiller_benchmark_name;
  spillConfig.writeBufferSize = FLAGS_spiller_benchmark_write_buffer_size;
  spillConfig.executor = executor_.get();

  spillConfig.compressionKind = common::CompressionKind::CompressionKind_LZ4;
  spillConfig.prefixSortConfig = PrefixSortConfig();

  spillConfig.maxSpillRunRows = 0;
  spillConfig.fileCreateConfig = {};
  spillConfig.serializeRowContainer = serializeRowContainer_;

  const auto sortingKeys = SpillState::makeSortingKeys(
      std::vector<CompareFlags>(rowContainer_->keyTypes().size()));
  return std::make_unique<SortInputSpiller>(
      rowContainer_.get(), rowType_, sortingKeys, &spillConfig, &spillStats_);
}

void SortSpillInputBenchmarkBase::printStats() const {
  LOG(INFO) << "total execution time: " << succinctMicros(executionTimeUs_);
  const auto memStats = pool_->stats();
  LOG(INFO) << numInputVectors_ << " vectors each with " << inputVectorSize_
            << " rows have been processed";
  LOG(INFO) << "peak memory usage[" << succinctBytes(memStats.peakBytes)
            << "] cumulative memory usage["
            << succinctBytes(memStats.cumulativeBytes) << "]";
  LOG(INFO) << spiller_->stats().toString();
  SpillPartitionSet partitionSet;
  spiller_->finishSpill(partitionSet);
  VELOX_CHECK_EQ(partitionSet.size(), 1);
}
} // namespace facebook::velox::exec::test
