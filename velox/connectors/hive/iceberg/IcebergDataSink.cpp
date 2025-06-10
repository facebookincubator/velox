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
static const std::string kPartitionValuesField = "partitionValues";

IcebergDataSink::IcebergDataSink(
    facebook::velox::RowTypePtr inputType,
    std::shared_ptr<const IcebergInsertTableHandle> insertTableHandle,
    const facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
    facebook::velox::connector::CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig)
    : IcebergDataSink(
          std::move(inputType),
          insertTableHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig,
          [&insertTableHandle]() {
            std::vector<column_index_t> channels;
            const auto& inputColumns = insertTableHandle->inputColumns();

            for (column_index_t i = 0; i < inputColumns.size(); i++) {
              if (inputColumns[i]->isPartitionKey()) {
                channels.push_back(i);
              }
            }
            return channels;
          }(),
          [&insertTableHandle]() {
            std::vector<column_index_t> channels(
                insertTableHandle->inputColumns().size());
            std::iota(channels.begin(), channels.end(), 0);
            return channels;
          }()) {}

IcebergDataSink::IcebergDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const std::vector<column_index_t>& partitionChannels,
    const std::vector<column_index_t>& dataChannels)
    : HiveDataSink(
          inputType,
          insertTableHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig,
          0,
          nullptr,
          partitionChannels,
          dataChannels) {}

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
    auto commitDataJson = folly::toJson(
        folly::dynamic::object(
            "path",
            info->writerParameters.writeDirectory() + "/" +
                info->writerParameters.writeFileName())(
            "fileSizeInBytes", ioStats_.at(i)->rawBytesWritten())(
            "metrics",
            folly::dynamic::object("recordCount", info->numWrittenRows))(
            "partitionSpecId",
            icebergInsertTableHandle->partitionSpec()->specId)(
            "partitionDataJson",
            partitionData_[i] == nullptr ? "" : partitionData_[i]->toJson())(
            "fileFormat", finalFileFormat)("content", "DATA"));
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

std::string PartitionData::toJson() const {
  folly::dynamic jsonObject = folly::dynamic::object();
  folly::dynamic valuesArray = folly::dynamic::array();
  for (const auto& value : partitionValues_) {
    valuesArray.push_back(value); // Directly use the string values
  }
  jsonObject[kPartitionValuesField] = valuesArray;
  return folly::toJson(jsonObject); // Convert dynamic object to JSON string
}

} // namespace facebook::velox::connector::hive::iceberg
