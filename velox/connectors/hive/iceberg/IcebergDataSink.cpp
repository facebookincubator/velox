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
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/IcebergPartitionIdGenerator.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/writer/Writer.h"
#endif
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::connector::hive::iceberg {

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
  auto value = block->valueAt<StringView>(row);
  return value.str();
}

template <>
folly::dynamic extractPartitionValue<TypeKind::VARBINARY>(
    const DecodedVector* block,
    vector_size_t row) {
  auto value = block->valueAt<StringView>(row);
  return value.str();
}

template <>
folly::dynamic extractPartitionValue<TypeKind::TIMESTAMP>(
    const DecodedVector* block,
    vector_size_t row) {
  auto timestamp = block->valueAt<Timestamp>(row);
  return timestamp.toMicros();
}

class IcebergFileNameGenerator : public FileNameGenerator {
 public:
  IcebergFileNameGenerator() {}

  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      bool commitRequired) const override;

  folly::dynamic serialize() const override;

  std::string toString() const override;
};

std::pair<std::string, std::string> IcebergFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    bool commitRequired) const {
  auto targetFileName = insertTableHandle->locationHandle()->targetFileName();
  if (targetFileName.empty()) {
    targetFileName = fmt::format("{}", makeUuid());
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
    std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
    std::shared_ptr<const LocationHandle> locationHandle,
    std::shared_ptr<const IcebergPartitionSpec> partitionSpec,
    memory::MemoryPool* pool,
    dwio::common::FileFormat tableStorageFormat,
    std::shared_ptr<HiveBucketProperty> bucketProperty,
    std::optional<common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters)
    : HiveInsertTableHandle(
          std::move(inputColumns),
          std::move(locationHandle),
          tableStorageFormat,
          std::move(bucketProperty),
          compressionKind,
          serdeParameters,
          nullptr,
          false,
          std::make_shared<const IcebergFileNameGenerator>()),
      partitionSpec_(std::move(partitionSpec)),
      columnTransforms_(
          parsePartitionTransformSpecs(partitionSpec_->fields, pool)) {}

IcebergDataSink::IcebergDataSink(
    facebook::velox::RowTypePtr inputType,
    const std::shared_ptr<const IcebergInsertTableHandle>& insertTableHandle,
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
            const auto& inputColumns = insertTableHandle->inputColumns();
            const auto& partitionSpec = insertTableHandle->partitionSpec();
            std::unordered_map<std::string, column_index_t> partitionKeyMap;
            for (auto i = 0; i < inputColumns.size(); ++i) {
              if (inputColumns[i]->isPartitionKey()) {
                partitionKeyMap[inputColumns[i]->name()] = i;
              }
            }
            std::vector<column_index_t> channels;
            channels.reserve(partitionSpec->fields.size());
            for (const auto& field : partitionSpec->fields) {
              if (auto it = partitionKeyMap.find(field.name);
                  it != partitionKeyMap.end()) {
                channels.push_back(it->second);
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
    const std::shared_ptr<const IcebergInsertTableHandle>& insertTableHandle,
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
          dataChannels,
          !partitionChannels.empty()
              ? std::make_unique<IcebergPartitionIdGenerator>(
                    partitionChannels,
                    hiveConfig->maxPartitionsPerWriters(
                        connectorQueryCtx->sessionProperties()),
                    connectorQueryCtx->memoryPool(),
                    insertTableHandle->columnTransforms(),
                    hiveConfig->isPartitionPathAsLowerCase(
                        connectorQueryCtx->sessionProperties()))
              : nullptr) {
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
    folly::dynamic commitData =
      folly::dynamic::object
        ("path",
          (fs::path(info->writerParameters.writeDirectory()) /
                        info->writerParameters.writeFileName()).string())
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
    auto icebergPartitionIdGenerator =
        dynamic_cast<const IcebergPartitionIdGenerator*>(
            partitionIdGenerator_.get());
    VELOX_CHECK_NOT_NULL(icebergPartitionIdGenerator);
    const RowVectorPtr transformedValues =
        icebergPartitionIdGenerator->partitionValues();
    for (auto i = 0; i < partitionChannels_.size(); ++i) {
      auto block = transformedValues->childAt(i);
      if (block->isNullAt(index)) {
        partitionValues[i] = nullptr;
      } else {
        DecodedVector decoded(*block);
        partitionValues[i] = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            extractPartitionValue, block->typeKind(), &decoded, index);
      }
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

void IcebergDataSink::appendData(RowVectorPtr input) {
  checkRunning();
  if (!isPartitioned() || partitionIdGenerator_->numPartitions() == 1) {
    const auto index = ensureWriter(HiveWriterId::unpartitionedId());
    write(index, input);
    return;
  }

  // Compute partition and bucket numbers.
  computePartitionAndBucketIds(input);

  splitInputRowsAndEnsureWriters(input);

  for (auto index = 0; index < writers_.size(); ++index) {
    const vector_size_t partitionSize = partitionSizes_[index];
    if (partitionSize == 0) {
      continue;
    }

    const RowVectorPtr writerInput = partitionSize == input->size()
        ? input
        : exec::wrap(partitionSize, partitionRows_[index], input);
    write(index, writerInput);
  }
}

HiveWriterId IcebergDataSink::getIcebergWriterId(size_t row) const {
  std::optional<uint32_t> partitionId;
  if (isPartitioned()) {
    VELOX_CHECK_LT(partitionIds_[row], std::numeric_limits<uint32_t>::max());
    partitionId = static_cast<uint32_t>(partitionIds_[row]);
  }

  return HiveWriterId{partitionId, std::nullopt};
}

std::shared_ptr<dwio::common::WriterOptions>
IcebergDataSink::createWriterOptions() const {
  auto options = HiveDataSink::createWriterOptions();
#ifdef VELOX_ENABLE_PARQUET
  auto parquetOptions =
      std::dynamic_pointer_cast<parquet::WriterOptions>(options);
  VELOX_CHECK_NOT_NULL(parquetOptions);
  parquetOptions->parquetWriteTimestampTimeZone = std::nullopt;
  parquetOptions->parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
#endif
  return options;
}

std::optional<std::string> IcebergDataSink::getPartitionName(
    const HiveWriterId& id) const {
  std::optional<std::string> partitionName;
  if (isPartitioned()) {
    partitionName =
        partitionIdGenerator_->partitionName(id.partitionId.value());
  }
  return partitionName;
}

} // namespace facebook::velox::connector::hive::iceberg
