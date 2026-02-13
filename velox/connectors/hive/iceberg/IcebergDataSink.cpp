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

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "velox/common/base/Fs.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/connectors/hive/iceberg/IcebergColumnHandle.h"
#include "velox/connectors/hive/iceberg/TransformExprBuilder.h"
#include "velox/exec/OperatorUtils.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

template <TypeKind Kind>
folly::dynamic extractPartitionValue(
    const VectorPtr& child,
    vector_size_t row) {
  using T = typename TypeTraits<Kind>::NativeType;
  return child->asChecked<SimpleVector<T>>()->valueAt(row);
}

template <>
folly::dynamic extractPartitionValue<TypeKind::VARCHAR>(
    const VectorPtr& child,
    vector_size_t row) {
  return child->asChecked<SimpleVector<StringView>>()->valueAt(row).str();
}

template <>
folly::dynamic extractPartitionValue<TypeKind::VARBINARY>(
    const VectorPtr& child,
    vector_size_t row) {
  return encoding::Base64::encode(
      child->asChecked<SimpleVector<StringView>>()->valueAt(row));
}

template <>
folly::dynamic extractPartitionValue<TypeKind::TIMESTAMP>(
    const VectorPtr& child,
    vector_size_t row) {
  return child->asChecked<SimpleVector<Timestamp>>()->valueAt(row).toMicros();
}

class IcebergFileNameGenerator : public FileNameGenerator {
 public:
  std::pair<std::string, std::string> gen(
      std::optional<uint32_t> bucketId,
      const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
      const ConnectorQueryCtx& connectorQueryCtx,
      bool commitRequired) const override;

  folly::dynamic serialize() const override;

  std::string toString() const override;
};

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

std::pair<std::string, std::string> IcebergFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    bool commitRequired) const {
  auto targetFileName = insertTableHandle->locationHandle()->targetFileName();
  if (targetFileName.empty()) {
    targetFileName = fmt::format("{}", makeUuid());
  }
  auto fileFormat = dwio::common::toString(insertTableHandle->storageFormat());
  auto fileName = fmt::format("{}.{}", targetFileName, fileFormat);
  return {fileName, fileName};
}

folly::dynamic IcebergFileNameGenerator::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "IcebergFileNameGenerator";
  return obj;
}

std::string IcebergFileNameGenerator::toString() const {
  return "IcebergFileNameGenerator";
}

} // namespace

IcebergInsertTableHandle::IcebergInsertTableHandle(
    std::vector<IcebergColumnHandlePtr> inputColumns,
    LocationHandlePtr locationHandle,
    dwio::common::FileFormat tableStorageFormat,
    IcebergPartitionSpecPtr partitionSpec,
    std::optional<common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters)
    : HiveInsertTableHandle(
          std::vector<HiveColumnHandlePtr>(
              inputColumns.begin(),
              inputColumns.end()),
          std::move(locationHandle),
          tableStorageFormat,
          nullptr,
          compressionKind,
          serdeParameters,
          nullptr,
          false,
          std::make_shared<const HiveInsertFileNameGenerator>()),
      partitionSpec_(partitionSpec) {
  VELOX_USER_CHECK(
      !inputColumns_.empty(),
      "Input columns cannot be empty for Iceberg tables.");
  VELOX_USER_CHECK_NOT_NULL(
      locationHandle_, "Location handle is required for Iceberg tables.");
  VELOX_USER_CHECK_EQ(
      tableStorageFormat,
      dwio::common::FileFormat::PARQUET,
      "Only Parquet file format is supported when writing Iceberg tables.");
}

namespace {

// Creates partition channels by mapping partition spec fields to input column
// indices. For each field in the partition spec, finds the corresponding
// partition key column in the input columns and records its index.
//
// @param inputColumns The input columns from the insert table handle.
// @param partitionSpec The Iceberg partition specification, or nullptr if
// unpartitioned.
// @return A vector of column indices representing the partition channels. Each
// index corresponds to a partition field in the spec and points to the
// matching partition key column in the input. Returns an empty vector if
// partitionSpec is nullptr.
std::vector<column_index_t> createPartitionChannels(
    const std::vector<HiveColumnHandlePtr>& inputColumns,
    const IcebergPartitionSpecPtr& partitionSpec) {
  std::vector<column_index_t> channels;
  if (!partitionSpec) {
    return channels;
  }

  // Build a map from partition key column names to their indices in the input.
  std::unordered_map<std::string, column_index_t> partitionKeyMap;
  for (auto i = 0; i < inputColumns.size(); ++i) {
    if (inputColumns[i]->isPartitionKey()) {
      partitionKeyMap[inputColumns[i]->name()] = i;
    }
  }

  // For each field in the partition spec, find its corresponding input column
  // index.
  channels.reserve(partitionSpec->fields.size());
  for (const auto& field : partitionSpec->fields) {
    if (auto it = partitionKeyMap.find(field.name);
        it != partitionKeyMap.end()) {
      channels.push_back(it->second);
    }
  }

  return channels;
}

std::vector<column_index_t> createDataChannels(
    const IcebergInsertTableHandlePtr& insertTableHandle) {
  std::vector<column_index_t> dataChannels(
      insertTableHandle->inputColumns().size());
  std::iota(dataChannels.begin(), dataChannels.end(), 0);
  return dataChannels;
}

// Creates a RowType schema for transformed partition values based on the
// partition specification. This RowType is used to wrap the transformed
// partition columns before passing them to the partition ID generator.
//
// For each partition field in the spec:
// - The column type is the result type of the partition transform (e.g.,
//   INTEGER for year transform, DATE for day transform).
// - The column name is the source column name for identity transforms, or
//   "columnName_transformName" for non-identity transforms (e.g., "birth_year"
//   for a year transform on a birth column).
//
// @param partitionSpec The Iceberg partition specification, or nullptr if
// unpartitioned.
// @return A RowType containing one column per partition field with appropriate
// names and types. Returns nullptr if partitionSpec is nullptr.
RowTypePtr createPartitionRowType(
    const IcebergPartitionSpecPtr& partitionSpec) {
  if (!partitionSpec) {
    return nullptr;
  }

  std::vector<TypePtr> partitionKeyTypes;
  std::vector<std::string> partitionKeyNames;

  // Build column names and types for each partition field.
  // Identity transforms use the source column name directly.
  // Non-identity transforms use "columnName_transformName" format.
  for (const auto& field : partitionSpec->fields) {
    partitionKeyTypes.emplace_back(field.resultType());
    std::string key = field.transformType == TransformType::kIdentity
        ? field.name
        : fmt::format(
              "{}_{}",
              field.name,
              TransformTypeName::toName(field.transformType));
    partitionKeyNames.emplace_back(std::move(key));
  }

  return ROW(std::move(partitionKeyNames), std::move(partitionKeyTypes));
}

} // namespace

IcebergDataSink::IcebergDataSink(
    RowTypePtr inputType,
    IcebergInsertTableHandlePtr insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const IcebergConfigPtr& icebergConfig)
    : IcebergDataSink(
          std::move(inputType),
          insertTableHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig,
          createPartitionChannels(
              insertTableHandle->inputColumns(),
              insertTableHandle->partitionSpec()),
          createDataChannels(insertTableHandle),
          createPartitionRowType(insertTableHandle->partitionSpec()),
          icebergConfig) {}

IcebergDataSink::IcebergDataSink(
    RowTypePtr inputType,
    IcebergInsertTableHandlePtr insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const std::vector<column_index_t>& partitionChannels,
    const std::vector<column_index_t>& dataChannels,
    RowTypePtr partitionRowType,
    const IcebergConfigPtr& icebergConfig)
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
              ? std::make_unique<PartitionIdGenerator>(
                    partitionRowType,
                    [&partitionChannels]() {
                      std::vector<column_index_t> transformedChannels(
                          partitionChannels.size());
                      std::iota(
                          transformedChannels.begin(),
                          transformedChannels.end(),
                          0);
                      return transformedChannels;
                    }(),
                    hiveConfig->maxPartitionsPerWriters(
                        connectorQueryCtx->sessionProperties()),
                    connectorQueryCtx->memoryPool())
              : nullptr),
      partitionSpec_(insertTableHandle->partitionSpec()),
      transformEvaluator_(
          !partitionChannels.empty() ? std::make_unique<TransformEvaluator>(
                                           TransformExprBuilder::toExpressions(
                                               partitionSpec_,
                                               partitionChannels_,
                                               inputType_,
                                               icebergConfig->functionPrefix()),
                                           connectorQueryCtx_)
                                     : nullptr),
      icebergPartitionName_(
          partitionSpec_ != nullptr
              ? std::make_unique<IcebergPartitionName>(partitionSpec_)
              : nullptr),
      partitionRowType_(std::move(partitionRowType)) {
  commitPartitionValue_.resize(maxOpenWriters_);
}

std::vector<std::string> IcebergDataSink::commitMessage() const {
  std::vector<std::string> commitTasks;
  commitTasks.reserve(writerInfo_.size());

  auto icebergInsertTableHandle =
      std::dynamic_pointer_cast<const IcebergInsertTableHandle>(
          insertTableHandle_);

  for (auto i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);
    // Following metadata (json format) is consumed by Presto CommitTaskData.
    // It contains the minimal subset of metadata.
    // TODO: Complete metrics is missing now and this could lead to suboptimal
    // query plan, will collect full iceberg metrics in following PR.
    // clang-format off
    folly::dynamic commitData = folly::dynamic::object(
    "path", (fs::path(info->writerParameters.writeDirectory()) /
                    info->writerParameters.writeFileName()).string())
      ("fileSizeInBytes", ioStats_.at(i)->rawBytesWritten())
      ("metrics",
        folly::dynamic::object("recordCount", info->numWrittenRows))
      ("partitionSpecJson",
        icebergInsertTableHandle->partitionSpec() ? icebergInsertTableHandle->partitionSpec()->specId : 0)
      ("fileFormat", "PARQUET")
      ("content", "DATA");
    // clang-format on
    if (!commitPartitionValue_.empty() && !commitPartitionValue_[i].isNull()) {
      commitData["partitionDataJson"] = folly::toJson(
          folly::dynamic::object("partitionValues", commitPartitionValue_[i]));
    }
    auto commitDataJson = folly::toJson(commitData);
    commitTasks.push_back(commitDataJson);
  }
  return commitTasks;
}

void IcebergDataSink::computePartitionAndBucketIds(const RowVectorPtr& input) {
  VELOX_CHECK(isPartitioned());
  VELOX_CHECK_NOT_NULL(transformEvaluator_);
  VELOX_CHECK_NOT_NULL(partitionIdGenerator_);
  // Step 1: Apply transforms to input partition columns.
  auto transformedColumns = transformEvaluator_->evaluate(input);

  // Step 2: Create RowVector based on transformed columns.
  const auto& transformedRowVector = std::make_shared<RowVector>(
      connectorQueryCtx_->memoryPool(),
      partitionRowType_,
      nullptr,
      input->size(),
      std::move(transformedColumns));
  partitionIdGenerator_->run(transformedRowVector, partitionIds_);
}

std::string IcebergDataSink::getPartitionName(uint32_t partitionId) const {
  VELOX_CHECK_NOT_NULL(icebergPartitionName_);

  return icebergPartitionName_->partitionName(
      partitionId,
      partitionIdGenerator_->partitionValues(),
      partitionKeyAsLowerCase_);
}

uint32_t IcebergDataSink::ensureWriter(const HiveWriterId& id) {
  auto writerId = HiveDataSink::ensureWriter(id);
  if (commitPartitionValue_[writerId].isNull()) {
    commitPartitionValue_[writerId] = makeCommitPartitionValue(writerId);
  }
  return writerId;
}

std::shared_ptr<dwio::common::WriterOptions>
IcebergDataSink::createWriterOptions() const {
  auto options = HiveDataSink::createWriterOptions();
  // Per Iceberg specification (https://iceberg.apache.org/spec/#parquet):
  // - Timestamps must be stored with microsecond precision.
  // - Timestamps must NOT be adjusted to UTC timezone; they should be written
  //   as-is without timezone conversion (empty string disables conversion).
  //
  // These settings are passed via serdeParameters to avoid including
  // parquet-specific headers. The keys must match kParquetSerdeTimestampUnit
  // and kParquetSerdeTimestampTimezone defined in
  // velox/dwio/parquet/writer/Writer.h. The value "6" represents microseconds
  // (TimestampPrecision::kMicroseconds).
  options->serdeParameters["parquet.writer.timestamp.unit"] = "6";
  options->serdeParameters["parquet.writer.timestamp.timezone"] = "";
  // Re-process configs to apply the serde parameters we just set.
  options->processConfigs(
      *hiveConfig_->config(), *connectorQueryCtx_->sessionProperties());
  return options;
}

folly::dynamic IcebergDataSink::makeCommitPartitionValue(
    uint32_t writerIndex) const {
  folly::dynamic partitionValues = folly::dynamic::array();
  const auto& transformedValues = partitionIdGenerator_->partitionValues();
  for (auto i = 0; i < partitionChannels_.size(); ++i) {
    const auto& child = transformedValues->childAt(i);
    if (child->isNullAt(writerIndex)) {
      partitionValues.push_back(nullptr);
    } else {
      partitionValues.push_back(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          extractPartitionValue, child->typeKind(), child, writerIndex));
    }
  }
  return partitionValues;
}

} // namespace facebook::velox::connector::hive::iceberg
