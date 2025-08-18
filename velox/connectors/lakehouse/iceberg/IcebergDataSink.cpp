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

#include "velox/connectors/lakehouse/iceberg/IcebergDataSink.h"
#include "velox/common/base/Fs.h"
#include "velox/connectors/lakehouse/common/HiveConnectorUtil.h"

namespace facebook::velox::connector::lakehouse::iceberg {

namespace {

std::string toJson(const std::vector<folly::dynamic>& partitionValues) {
  folly::dynamic jsonObject = folly::dynamic::object();
  folly::dynamic valuesArray = folly::dynamic::array();
  for (const auto& value : partitionValues) {
    valuesArray.push_back(value);
  }
  jsonObject["partitionValues"] = valuesArray;
  return folly::toJson(jsonObject);
}

template <TypeKind Kind>
folly::dynamic extractPartitionValue(
    const DecodedVector* block,
    vector_size_t row) {
  using T = typename TypeTraits<Kind>::NativeType;
  return block->valueAt<T>(row);
}

template <>
folly::dynamic extractPartitionValue<TypeKind::VARCHAR>(
    const DecodedVector* block,
    vector_size_t row) {
  return block->toString(row);
}

template <>
folly::dynamic extractPartitionValue<TypeKind::VARBINARY>(
    const DecodedVector* block,
    vector_size_t row) {
  VELOX_NYI("Partition on varbinary column is not supported yet.");
}

class IcebergFileNameGenerator : public common::FileNameGenerator {
 public:
  IcebergFileNameGenerator() {}

  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const common::HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      bool commitRequired) const override;

  folly::dynamic serialize() const override;

  std::string toString() const override;
};

std::pair<std::string, std::string> IcebergFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const std::shared_ptr<const common::HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    bool commitRequired) const {
  auto targetFileName = insertTableHandle->locationHandle()->targetFileName();
  if (targetFileName.empty()) {
    targetFileName = fmt::format("{}", common::makeUuid());
  }

  return {
      fmt::format("{}{}", targetFileName, ".parquet"),
      fmt::format("{}{}", targetFileName, ".parquet")};
}

folly::dynamic IcebergFileNameGenerator::serialize() const {
  VELOX_UNREACHABLE("Unexpected code path, implement serialize() first.");
}

std::string IcebergFileNameGenerator::toString() const {
  return "IcebergFileNameGenerator";
}

} // namespace

IcebergInsertTableHandle::IcebergInsertTableHandle(
    std::vector<std::shared_ptr<const common::HiveColumnHandle>> inputColumns,
    std::shared_ptr<const common::LocationHandle> locationHandle,
    std::shared_ptr<const IcebergPartitionSpec> partitionSpec,
    dwio::common::FileFormat tableStorageFormat,
    std::shared_ptr<common::HiveBucketProperty> bucketProperty,
    std::optional<velox::common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters)
    : common::HiveInsertTableHandle(
          std::move(inputColumns),
          std::move(locationHandle),
          tableStorageFormat,
          std::move(bucketProperty),
          compressionKind,
          serdeParameters,
          nullptr,
          false,
          std::make_shared<const IcebergFileNameGenerator>()),
      partitionSpec_(std::move(partitionSpec)) {}

IcebergDataSink::IcebergDataSink(
    facebook::velox::RowTypePtr inputType,
    std::shared_ptr<const IcebergInsertTableHandle> insertTableHandle,
    const facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx,
    facebook::velox::connector::CommitStrategy commitStrategy,
    const std::shared_ptr<const common::HiveConfig>& hiveConfig)
    : IcebergDataSink(
          std::move(inputType),
          insertTableHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig,
          [&insertTableHandle]() {
            std::vector<column_index_t> channels(
                insertTableHandle->inputColumns().size());
            std::iota(channels.begin(), channels.end(), 0);
            return channels;
          }()) {}

IcebergDataSink::IcebergDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const common::HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const common::HiveConfig>& hiveConfig,
    const std::vector<column_index_t>& dataChannels)
    : common::HiveDataSink(
          inputType,
          insertTableHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig,
          0,
          nullptr,
          dataChannels) {
  if (isPartitioned()) {
    partitionData_.resize(maxOpenWriters_);
  }
}

std::vector<std::string> IcebergDataSink::commitMessage() const {
  auto icebergInsertTableHandle =
      std::dynamic_pointer_cast<const IcebergInsertTableHandle>(
          insertTableHandle_);

  std::vector<std::string> commitTasks;
  commitTasks.reserve(writerInfo_.size());
  std::string fileFormat(toString(insertTableHandle_->storageFormat()));
  std::transform(
      fileFormat.begin(), fileFormat.end(), fileFormat.begin(), ::toupper);

  for (auto i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);
    // Following metadata (json format) is consumed by Presto CommitTaskData.
    // It contains the minimal subset of metadata.
    // Complete metrics is missing now and this could lead to suboptimal query
    // plan, will collect full iceberg metrics in following PR.
    // clang-format off
    folly::dynamic commitData = folly::dynamic::object(
        "path", info->writerParameters.writeDirectory() + "/" +
        info->writerParameters.writeFileName())
      ("fileSizeInBytes", ioStats_.at(i)->rawBytesWritten())
      ("metrics",
        folly::dynamic::object("recordCount", info->numWrittenRows))
      ("partitionSpecJson", icebergInsertTableHandle->partitionSpec()->specId)
      ("fileFormat", fileFormat)
      ("content", "DATA");
    // clang-format on
    if (!(partitionData_.empty() || partitionData_[i].empty())) {
      commitData["partitionDataJson"] = toJson(partitionData_[i]);
    }
    auto commitDataJson = folly::toJson(commitData);
    commitTasks.push_back(commitDataJson);
  }
  return commitTasks;
}

void IcebergDataSink::splitInputRowsAndEnsureWriters(RowVectorPtr input) {
  VELOX_CHECK(isPartitioned());

  std::fill(partitionSizes_.begin(), partitionSizes_.end(), 0);

  const auto numRows = partitionIds_.size();
  for (auto row = 0; row < numRows; ++row) {
    auto id = getIcebergWriterId(row);
    uint32_t index = ensureWriter(id);

    updatePartitionRows(index, numRows, row);

    if (!partitionData_[index].empty()) {
      continue;
    }

    std::vector<folly::dynamic> partitionValues(partitionChannels_.size());

    for (auto i = 0; i < partitionChannels_.size(); ++i) {
      auto block = input->childAt(partitionChannels_[i]);
      if (block->type()->isDecimal()) {
        VELOX_NYI("Partition on decimal column is not supported yet.");
      }
      DecodedVector decoded(*block);
      partitionValues[i] = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          extractPartitionValue, block->typeKind(), &decoded, row);
    }

    partitionData_[index] = partitionValues;
  }

  for (auto i = 0; i < partitionSizes_.size(); ++i) {
    if (partitionSizes_[i] != 0) {
      VELOX_CHECK_NOT_NULL(partitionRows_[i]);
      partitionRows_[i]->setSize(partitionSizes_[i] * sizeof(vector_size_t));
    }
  }
}

common::HiveWriterId IcebergDataSink::getIcebergWriterId(size_t row) const {
  std::optional<uint32_t> partitionId;
  if (isPartitioned()) {
    VELOX_CHECK_LT(partitionIds_[row], std::numeric_limits<uint32_t>::max());
    partitionId = static_cast<uint32_t>(partitionIds_[row]);
  }

  return common::HiveWriterId{partitionId, std::nullopt};
}

std::optional<std::string> IcebergDataSink::getPartitionName(
    const common::HiveWriterId& id) const {
  std::optional<std::string> partitionName;
  if (isPartitioned()) {
    partitionName =
        partitionIdGenerator_->partitionName(id.partitionId.value(), "null");
  }
  return partitionName;
}

} // namespace facebook::velox::connector::lakehouse::iceberg
