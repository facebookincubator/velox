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

#include "velox/connectors/hive/PartitionWriter.h"

#include "velox/common/time/Timer.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::connector::hive {
namespace {

/// Extract data columns from input, producing a RowVector with only the
/// non-partition columns.
RowVectorPtr makeDataInput(
    const std::vector<column_index_t>& dataCols,
    const RowTypePtr& dataType,
    const RowVectorPtr& input) {
  std::vector<VectorPtr> childVectors;
  childVectors.reserve(dataCols.size());
  for (int dataCol : dataCols) {
    childVectors.push_back(input->childAt(dataCol));
  }
  return std::make_shared<RowVector>(
      input->pool(),
      dataType,
      input->nulls(),
      input->size(),
      std::move(childVectors),
      input->getNullCount());
}

} // namespace

PartitionWriter::PartitionWriter(
    uint32_t maxOpenWriters,
    const std::vector<column_index_t>& dataChannels,
    RowTypePtr dataType,
    WriterFactory writerFactory,
    memory::MemoryPool* pool)
    : maxOpenWriters_(maxOpenWriters),
      dataChannels_(dataChannels),
      dataType_(std::move(dataType)),
      writerFactory_(std::move(writerFactory)),
      pool_(pool) {
  VELOX_CHECK_NOT_NULL(dataType_);
  VELOX_CHECK(writerFactory_);
  VELOX_CHECK_NOT_NULL(pool_);
}

void PartitionWriter::write(const HiveWriterId& id, RowVectorPtr input) {
  const auto index = ensureWriter(id);
  writeToWriter(index, input);
}

void PartitionWriter::write(
    RowVectorPtr input,
    const raw_vector<uint64_t>& partitionIds,
    const std::vector<uint32_t>& bucketIds,
    bool isPartitioned,
    bool isBucketed) {
  splitInputRowsAndEnsureWriters(
      partitionIds, bucketIds, isPartitioned, isBucketed);

  for (auto index = 0; index < writers_.size(); ++index) {
    const vector_size_t partitionSize = partitionSizes_[index];
    if (partitionSize == 0) {
      continue;
    }

    RowVectorPtr writerInput = partitionSize == input->size()
        ? input
        : exec::wrap(partitionSize, partitionRows_[index], input);
    writeToWriter(index, writerInput);
  }
}

bool PartitionWriter::finish(uint64_t timeSliceLimitMs) {
  const uint64_t startTimeMs = getCurrentTimeMs();
  for (auto& writer : writers_) {
    if (!writer->finish()) {
      return false;
    }
    if (getCurrentTimeMs() - startTimeMs > timeSliceLimitMs) {
      return false;
    }
  }
  return true;
}

void PartitionWriter::close() {
  for (auto& writer : writers_) {
    writer->close();
  }
}

void PartitionWriter::abort() {
  for (auto& writer : writers_) {
    writer->abort();
  }
}

uint32_t PartitionWriter::ensureWriter(const HiveWriterId& id) {
  auto it = writerIndexMap_.find(id);
  if (it != writerIndexMap_.end()) {
    return it->second;
  }

  VELOX_USER_CHECK_LT(
      writers_.size(), maxOpenWriters_, "Exceeded open writer limit");
  VELOX_CHECK_EQ(writerIndexMap_.size(), writers_.size());

  const auto writerIndex = writers_.size();
  auto writer = writerFactory_(id, writerIndex);
  VELOX_CHECK_NOT_NULL(writer);
  writers_.emplace_back(std::move(writer));

  partitionSizes_.emplace_back(0);
  partitionRows_.emplace_back(nullptr);
  rawPartitionRows_.emplace_back(nullptr);

  writerIndexMap_.emplace(id, writerIndex);

  if (onWriterCreated_) {
    onWriterCreated_(writerIndex, id);
  }

  return writerIndex;
}

void PartitionWriter::writeToWriter(size_t index, RowVectorPtr input) {
  auto dataInput = makeDataInput(dataChannels_, dataType_, input);
  writers_[index]->write(dataInput);
}

HiveWriterId PartitionWriter::getWriterId(
    size_t row,
    const raw_vector<uint64_t>& partitionIds,
    const std::vector<uint32_t>& bucketIds,
    bool isPartitioned,
    bool isBucketed) const {
  std::optional<uint32_t> partitionId;
  if (isPartitioned) {
    VELOX_CHECK_LT(partitionIds[row], std::numeric_limits<uint32_t>::max());
    partitionId = static_cast<uint32_t>(partitionIds[row]);
  }

  std::optional<uint32_t> bucketId;
  if (isBucketed) {
    bucketId = bucketIds[row];
  }
  return HiveWriterId{partitionId, bucketId};
}

void PartitionWriter::splitInputRowsAndEnsureWriters(
    const raw_vector<uint64_t>& partitionIds,
    const std::vector<uint32_t>& bucketIds,
    bool isPartitioned,
    bool isBucketed) {
  VELOX_CHECK(isPartitioned || isBucketed);
  if (isBucketed && isPartitioned) {
    VELOX_CHECK_EQ(bucketIds.size(), partitionIds.size());
  }

  std::fill(partitionSizes_.begin(), partitionSizes_.end(), 0);

  const auto numRows = isPartitioned ? partitionIds.size() : bucketIds.size();
  for (auto row = 0; row < numRows; ++row) {
    const auto id =
        getWriterId(row, partitionIds, bucketIds, isPartitioned, isBucketed);
    const uint32_t index = ensureWriter(id);
    updatePartitionRows(index, numRows, row);
  }

  for (uint32_t i = 0; i < partitionSizes_.size(); ++i) {
    if (partitionSizes_[i] != 0) {
      VELOX_CHECK_NOT_NULL(partitionRows_[i]);
      partitionRows_[i]->setSize(partitionSizes_[i] * sizeof(vector_size_t));
    }
  }
}

void PartitionWriter::updatePartitionRows(
    uint32_t index,
    vector_size_t numRows,
    vector_size_t row) {
  VELOX_DCHECK_LT(index, partitionSizes_.size());
  VELOX_DCHECK_EQ(partitionSizes_.size(), partitionRows_.size());
  VELOX_DCHECK_EQ(partitionRows_.size(), rawPartitionRows_.size());
  if (FOLLY_UNLIKELY(partitionRows_[index] == nullptr) ||
      (partitionRows_[index]->capacity() < numRows * sizeof(vector_size_t))) {
    partitionRows_[index] = allocateIndices(numRows, pool_);
    rawPartitionRows_[index] =
        partitionRows_[index]->asMutable<vector_size_t>();
  }
  rawPartitionRows_[index][partitionSizes_[index]] = row;
  ++partitionSizes_[index];
}

} // namespace facebook::velox::connector::hive
