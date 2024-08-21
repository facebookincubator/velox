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

#include "velox/connectors/hive/iceberg/IcebergDataSink.h"

#include "velox/common/base/Fs.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {
#define WRITER_NON_RECLAIMABLE_SECTION_GUARD(index)       \
  memory::NonReclaimableSectionGuard nonReclaimableGuard( \
      writerInfo_[(index)]->nonReclaimableSectionHolder.get())
} // namespace

IcebergDataSink::IcebergDataSink(
    facebook::velox::RowTypePtr inputType,
    std::shared_ptr<const IcebergInsertTableHandle> insertTableHandle,
    const facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
    facebook::velox::connector::CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig)
    : HiveDataSink(
          std::move(inputType),
          std::move(insertTableHandle),
          connectorQueryCtx,
          commitStrategy,
          hiveConfig) {}

void IcebergDataSink::appendData(facebook::velox::RowVectorPtr input) {
  checkRunning();

  // Write to unpartitioned table.
  if (!isPartitioned()) {
    const auto index = ensureWriter(HiveWriterId::unpartitionedId());
    write(index, input);
    return;
  }

  dataChannels_ = getDataChannels(partitionChannels_, inputType_->size());

  // Compute partition and bucket numbers.
  computePartitionAndBucketIds(input);

  // Lazy load all the input columns.
  for (column_index_t i = 0; i < input->childrenSize(); ++i) {
    input->childAt(i)->loadedVector();
  }

  // All inputs belong to a single non-bucketed partition. The partition id
  // must be zero.
  if (!isBucketed() && partitionIdGenerator_->numPartitions() == 1) {
    const auto index = ensureWriter(HiveWriterId{0});
    write(index, input);
    return;
  }

  splitInputRowsAndEnsureWriters(input);

  for (auto index = 0; index < writers_.size(); ++index) {
    const vector_size_t partitionSize = partitionSizes_[index];
    if (partitionSize == 0) {
      continue;
    }

    RowVectorPtr writerInput = partitionSize == input->size()
        ? input
        : exec::wrap(partitionSize, partitionRows_[index], input);

    write(index, writerInput);
  }
}

void IcebergDataSink::write(size_t index, RowVectorPtr input) {
  WRITER_NON_RECLAIMABLE_SECTION_GUARD(index);

  writers_[index]->write(input);
  writerInfo_[index]->numWrittenRows += input->size();
}

std::vector<std::string> IcebergDataSink::close() {
  checkRunning();
  state_ = State::kClosed;
  closeInternal();

  auto icebergInsertTableHandle =
      std::dynamic_pointer_cast<const IcebergInsertTableHandle>(
          insertTableHandle_);

  std::vector<std::string> commitTasks;
  commitTasks.reserve(writerInfo_.size());
  auto fileFormat = toString(insertTableHandle_->tableStorageFormat());
  std::string finalFileFormat(fileFormat);
  std::transform(
      finalFileFormat.begin(),
      finalFileFormat.end(),
      finalFileFormat.begin(),
      ::toupper);

  for (int i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);
    // TODO(imjalpreet): Collect Iceberg file format metrics required by
    // com.facebook.presto.iceberg.MetricsWrapper->org.apache.iceberg.Metrics#Metrics
    // clang-format off
      auto commitDataJson = folly::toJson(
       folly::dynamic::object
          ("path", info->writerParameters.writeDirectory() + "/" + info->writerParameters.writeFileName())
          ("fileSizeInBytes", ioStats_.at(i)->rawBytesWritten())
          ("metrics", folly::dynamic::object
            ("recordCount", info->numWrittenRows))
          ("partitionSpecId", icebergInsertTableHandle->partitionSpec()->specId)
          ("partitionDataJson", partitionData_[i]->toJson())
          ("fileFormat", finalFileFormat)
          ("referencedDataFile", nullptr));
    // clang-format on
    commitTasks.push_back(commitDataJson);
  }
  return commitTasks;
}

void IcebergDataSink::splitInputRowsAndEnsureWriters(RowVectorPtr input) {
  VELOX_CHECK(isPartitioned());

  std::fill(partitionSizes_.begin(), partitionSizes_.end(), 0);

  const auto numRows = partitionIds_.size();
  for (auto row = 0; row < numRows; ++row) {
    auto id = getWriterId(row);
    uint32_t index = ensureWriter(id);

    VELOX_DCHECK_LT(index, partitionSizes_.size());
    VELOX_DCHECK_EQ(partitionSizes_.size(), partitionRows_.size());
    VELOX_DCHECK_EQ(partitionRows_.size(), rawPartitionRows_.size());
    if (FOLLY_UNLIKELY(partitionRows_[index] == nullptr) ||
        (partitionRows_[index]->capacity() < numRows * sizeof(vector_size_t))) {
      partitionRows_[index] =
          allocateIndices(numRows, connectorQueryCtx_->memoryPool());
      rawPartitionRows_[index] =
          partitionRows_[index]->asMutable<vector_size_t>();
    }
    rawPartitionRows_[index][partitionSizes_[index]] = row;
    ++partitionSizes_[index];

    if (partitionData_[index] != nullptr) {
      continue;
    }

    std::vector<std::string> partitionValues;
    partitionValues.reserve(partitionChannels_.size());

    auto icebergInsertTableHandle =
        std::dynamic_pointer_cast<const IcebergInsertTableHandle>(
            insertTableHandle_);

    for (auto i = 0; i < partitionChannels_.size(); ++i) {
      auto type =
          icebergInsertTableHandle->inputColumns()[partitionChannels_[i]]
              ->dataType();
      auto block = input->childAt(partitionChannels_[i]);
      partitionValues.insert(partitionValues.begin() + i, block->toString(row));
    }

    partitionData_.insert(
        partitionData_.begin() + index,
        std::make_shared<PartitionData>(partitionValues));
  }

  for (uint32_t i = 0; i < partitionSizes_.size(); ++i) {
    if (partitionSizes_[i] != 0) {
      VELOX_CHECK_NOT_NULL(partitionRows_[i]);
      partitionRows_[i]->setSize(partitionSizes_[i] * sizeof(vector_size_t));
    }
  }
}

void IcebergDataSink::extendBuffersForPartitionedTables() {
  // Extends the buffer used for partition rows calculations.
  partitionSizes_.emplace_back(0);
  partitionRows_.emplace_back(nullptr);
  rawPartitionRows_.emplace_back(nullptr);

  // Extend the buffer used for partitionData
  partitionData_.emplace_back(nullptr);
}

std::string IcebergDataSink::makePartitionDirectory(
    const std::string& tableDirectory,
    const std::optional<std::string>& partitionSubdirectory) const {
  std::string tableLocation = fmt::format("{}/data", tableDirectory);
  if (partitionSubdirectory.has_value()) {
    return fs::path(tableLocation) / partitionSubdirectory.value();
  }
  return tableLocation;
}

// Returns the column indices of data columns.
std::vector<column_index_t> IcebergDataSink::getDataChannels(
    const std::vector<column_index_t>& partitionChannels,
    const column_index_t childrenSize) const {
  // Create a vector of all possible channels
  std::vector<column_index_t> dataChannels(childrenSize);
  std::iota(dataChannels.begin(), dataChannels.end(), 0);

  return dataChannels;
}
} // namespace facebook::velox::connector::hive::iceberg
