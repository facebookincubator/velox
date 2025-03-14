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

namespace facebook::velox::connector::hive::iceberg {

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

std::vector<std::string> IcebergDataSink::close() {
  setState(State::kClosed);
  closeInternal();

  auto icebergInsertTableHandle =
      std::dynamic_pointer_cast<const IcebergInsertTableHandle>(
          insertTableHandle_);

  std::vector<std::string> commitTasks;
  commitTasks.reserve(writerInfo_.size());
  auto fileFormat = toString(insertTableHandle_->storageFormat());
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
          ("partitionDataJson", partitionData_[i] == nullptr ? "" : partitionData_[i]->toJson())
          ("fileFormat", finalFileFormat)
          ("referencedDataFile", nullptr)
          ("content", "DATA"));
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

    updatePartitionRows(index, numRows, row);

    ++partitionSizes_[index];

    if (partitionData_[index] != nullptr) {
      continue;
    }

    std::vector<std::string> partitionValues(partitionChannels_.size());

    for (auto i = 0; i < partitionChannels_.size(); ++i) {
      auto block = input->childAt(partitionChannels_[i]);
      partitionValues[i] = block->toString(row);
    }

    partitionData_[index] = std::make_shared<PartitionData>(partitionValues);
  }

  for (uint32_t i = 0; i < partitionSizes_.size(); ++i) {
    if (partitionSizes_[i] != 0) {
      VELOX_CHECK_NOT_NULL(partitionRows_[i]);
      partitionRows_[i]->setSize(partitionSizes_[i] * sizeof(vector_size_t));
    }
  }
}

void IcebergDataSink::extendBuffersForPartitionedTables() {
  HiveDataSink::extendBuffersForPartitionedTables();

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
std::vector<column_index_t> IcebergDataSink::getDataChannels() const {
  // Create a vector of all possible channels
  std::vector<column_index_t> dataChannels(inputType_->size());
  std::iota(dataChannels.begin(), dataChannels.end(), 0);

  return dataChannels;
}
} // namespace facebook::velox::connector::hive::iceberg
