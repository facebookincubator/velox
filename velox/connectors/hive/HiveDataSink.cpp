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

#include "velox/connectors/hive/HiveDataSink.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/Fs.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/SortingWriter.h"
#include "velox/exec/SortBuffer.h"

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <re2/re2.h>

namespace facebook::velox::connector::hive {
namespace {
#define WRITER_NON_RECLAIMABLE_SECTION_GUARD(index)       \
  memory::NonReclaimableSectionGuard nonReclaimableGuard( \
      writerInfo_[(index)]->nonReclaimableSectionHolder.get())

// Appends a sequence number to a filename for file rotation.
// Returns the original filename if sequenceNumber is 0 (no rotation yet).
// Example: "file.orc" with seq 0 remains "file.orc"
// Example: "file.orc" with seq 2 becomes "file_2.orc"
std::string makeSequencedFileName(
    const std::string& filename,
    uint32_t sequenceNumber) {
  if (sequenceNumber == 0) {
    return filename;
  }
  const auto dotPos = filename.rfind('.');
  if (dotPos == std::string::npos) {
    // No extension, just append the sequence number
    return fmt::format("{}_{}", filename, sequenceNumber);
  }
  // Insert sequence number before the extension
  return fmt::format(
      "{}_{}{}",
      filename.substr(0, dotPos),
      sequenceNumber,
      filename.substr(dotPos));
}

std::unique_ptr<dwio::common::FileSink> createHiveFileSink(
    const std::string& path,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    memory::MemoryPool* sinkPool,
    io::IoStatistics* ioStats,
    IoStats* fileSystemStats,
    const std::unordered_map<std::string, std::string>& storageParameters) {
  return dwio::common::FileSink::create(
      path,
      {
          .bufferWrite = false,
          .connectorProperties = hiveConfig->config(),
          .fileCreateConfig = hiveConfig->writeFileCreateConfig(),
          .pool = sinkPool,
          .metricLogger = dwio::common::MetricsLog::voidLog(),
          .stats = ioStats,
          .fileSystemStats = fileSystemStats,
          .storageParameters = storageParameters,
      });
}

// Creates a PartitionIdGenerator if the table is partitioned, otherwise returns
// nullptr.
std::unique_ptr<PartitionIdGenerator> createPartitionIdGenerator(
    const RowTypePtr& inputType,
    const std::shared_ptr<const HiveInsertTableHandle>& insertTableHandle,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const ConnectorQueryCtx* connectorQueryCtx) {
  auto partitionChannels = insertTableHandle->partitionChannels();
  if (partitionChannels.empty()) {
    return nullptr;
  }
  return std::make_unique<PartitionIdGenerator>(
      inputType,
      partitionChannels,
      hiveConfig->maxPartitionsPerWriters(
          connectorQueryCtx->sessionProperties()),
      connectorQueryCtx->memoryPool());
}

std::string makePartitionDirectory(
    const std::string& tableDirectory,
    const std::optional<std::string>& partitionSubdirectory) {
  if (partitionSubdirectory.has_value()) {
    return fs::path(tableDirectory) / partitionSubdirectory.value();
  }
  return tableDirectory;
}

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

std::unordered_map<LocationHandle::TableType, std::string> tableTypeNames() {
  return {
      {LocationHandle::TableType::kNew, "kNew"},
      {LocationHandle::TableType::kExisting, "kExisting"},
  };
}

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& mapping) {
  std::unordered_map<V, K> inverted;
  for (const auto& [key, value] : mapping) {
    inverted.emplace(value, key);
  }
  return inverted;
}

std::unique_ptr<core::PartitionFunction> createBucketFunction(
    const HiveBucketProperty& bucketProperty,
    const RowTypePtr& inputType) {
  const auto& bucketedBy = bucketProperty.bucketedBy();
  const auto& bucketedTypes = bucketProperty.bucketedTypes();
  std::vector<column_index_t> bucketedByChannels;
  bucketedByChannels.reserve(bucketedBy.size());
  for (int32_t i = 0; i < bucketedBy.size(); ++i) {
    const auto& bucketColumn = bucketedBy[i];
    const auto& bucketType = bucketedTypes[i];
    const auto inputChannel = inputType->getChildIdx(bucketColumn);
    if (FOLLY_UNLIKELY(
            !inputType->childAt(inputChannel)->equivalent(*bucketType))) {
      VELOX_USER_FAIL(
          "Input column {} type {} doesn't match bucket type {}",
          inputType->nameOf(inputChannel),
          inputType->childAt(inputChannel)->toString(),
          bucketType->toString());
    }
    bucketedByChannels.push_back(inputChannel);
  }
  return std::make_unique<HivePartitionFunction>(
      bucketProperty.bucketCount(), bucketedByChannels);
}

std::string computeBucketedFileName(
    const std::string& queryId,
    uint32_t maxBucketCount,
    uint32_t bucket) {
  const uint32_t kMaxBucketCountPadding =
      std::to_string(maxBucketCount - 1).size();
  const std::string bucketValueStr = std::to_string(bucket);
  return fmt::format(
      "0{:0>{}}_0_{}", bucketValueStr, kMaxBucketCountPadding, queryId);
}

uint64_t getFinishTimeSliceLimitMsFromHiveConfig(
    const std::shared_ptr<const HiveConfig>& config,
    const config::ConfigBase* sessions) {
  const uint64_t flushTimeSliceLimitMsFromConfig =
      config->sortWriterFinishTimeSliceLimitMs(sessions);
  // NOTE: if the flush time slice limit is set to 0, then we treat it as no
  // limit.
  return flushTimeSliceLimitMsFromConfig == 0
      ? std::numeric_limits<uint64_t>::max()
      : flushTimeSliceLimitMsFromConfig;
}

FOLLY_ALWAYS_INLINE int32_t
getBucketCount(const HiveBucketProperty* bucketProperty) {
  return bucketProperty == nullptr ? 0 : bucketProperty->bucketCount();
}

std::vector<column_index_t> computePartitionChannels(
    const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns) {
  std::vector<column_index_t> channels;
  for (auto i = 0; i < inputColumns.size(); i++) {
    if (inputColumns[i]->isPartitionKey()) {
      channels.push_back(i);
    }
  }
  return channels;
}

std::vector<column_index_t> computeNonPartitionChannels(
    const std::vector<std::shared_ptr<const HiveColumnHandle>>& inputColumns) {
  std::vector<column_index_t> channels;
  for (auto i = 0; i < inputColumns.size(); i++) {
    if (!inputColumns[i]->isPartitionKey()) {
      channels.push_back(i);
    }
  }
  return channels;
}

} // namespace

const std::string LocationHandle::tableTypeName(
    LocationHandle::TableType type) {
  static const auto tableTypes = tableTypeNames();
  return tableTypes.at(type);
}

LocationHandle::TableType LocationHandle::tableTypeFromName(
    const std::string& name) {
  static const auto nameTableTypes = invertMap(tableTypeNames());
  return nameTableTypes.at(name);
}

HiveSortingColumn::HiveSortingColumn(
    const std::string& sortColumn,
    const core::SortOrder& sortOrder)
    : sortColumn_(sortColumn), sortOrder_(sortOrder) {
  VELOX_USER_CHECK(!sortColumn_.empty(), "hive sort column must be set");

  if (FOLLY_UNLIKELY(
          (sortOrder_.isAscending() && !sortOrder_.isNullsFirst()) ||
          (!sortOrder_.isAscending() && sortOrder_.isNullsFirst()))) {
    VELOX_USER_FAIL("Bad hive sort order: {}", toString());
  }
}

folly::dynamic HiveSortingColumn::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveSortingColumn";
  obj["columnName"] = sortColumn_;
  obj["sortOrder"] = sortOrder_.serialize();
  return obj;
}

std::shared_ptr<HiveSortingColumn> HiveSortingColumn::deserialize(
    const folly::dynamic& obj,
    void* context) {
  const std::string columnName = obj["columnName"].asString();
  const auto sortOrder = core::SortOrder::deserialize(obj["sortOrder"]);
  return std::make_shared<HiveSortingColumn>(columnName, sortOrder);
}

std::string HiveSortingColumn::toString() const {
  return fmt::format(
      "[COLUMN[{}] ORDER[{}]]", sortColumn_, sortOrder_.toString());
}

void HiveSortingColumn::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("HiveSortingColumn", HiveSortingColumn::deserialize);
}

HiveBucketProperty::HiveBucketProperty(
    Kind kind,
    int32_t bucketCount,
    const std::vector<std::string>& bucketedBy,
    const std::vector<TypePtr>& bucketTypes,
    const std::vector<std::shared_ptr<const HiveSortingColumn>>& sortedBy)
    : kind_(kind),
      bucketCount_(bucketCount),
      bucketedBy_(bucketedBy),
      bucketTypes_(bucketTypes),
      sortedBy_(sortedBy) {
  validate();
}

void HiveBucketProperty::validate() const {
  VELOX_USER_CHECK_GT(bucketCount_, 0, "Hive bucket count can't be zero");
  VELOX_USER_CHECK(!bucketedBy_.empty(), "Hive bucket columns must be set");
  VELOX_USER_CHECK_EQ(
      bucketedBy_.size(),
      bucketTypes_.size(),
      "The number of hive bucket columns and types do not match {}",
      toString());
}

std::string HiveBucketProperty::kindString(Kind kind) {
  switch (kind) {
    case Kind::kHiveCompatible:
      return "HIVE_COMPATIBLE";
    case Kind::kPrestoNative:
      return "PRESTO_NATIVE";
    default:
      return fmt::format("UNKNOWN {}", static_cast<int>(kind));
  }
}

folly::dynamic HiveBucketProperty::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveBucketProperty";
  obj["kind"] = static_cast<int64_t>(kind_);
  obj["bucketCount"] = bucketCount_;
  obj["bucketedBy"] = ISerializable::serialize(bucketedBy_);
  obj["bucketedTypes"] = ISerializable::serialize(bucketTypes_);
  obj["sortedBy"] = ISerializable::serialize(sortedBy_);
  return obj;
}

std::shared_ptr<HiveBucketProperty> HiveBucketProperty::deserialize(
    const folly::dynamic& obj,
    void* context) {
  const Kind kind = static_cast<Kind>(obj["kind"].asInt());
  const int32_t bucketCount = obj["bucketCount"].asInt();
  const auto buckectedBy =
      ISerializable::deserialize<std::vector<std::string>>(obj["bucketedBy"]);
  const auto bucketedTypes = ISerializable::deserialize<std::vector<Type>>(
      obj["bucketedTypes"], context);
  const auto sortedBy =
      ISerializable::deserialize<std::vector<HiveSortingColumn>>(
          obj["sortedBy"], context);
  return std::make_shared<HiveBucketProperty>(
      kind, bucketCount, buckectedBy, bucketedTypes, sortedBy);
}

void HiveBucketProperty::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("HiveBucketProperty", HiveBucketProperty::deserialize);
}

std::string HiveBucketProperty::toString() const {
  std::stringstream out;
  out << "\nHiveBucketProperty[<" << kind_ << " " << bucketCount_ << ">\n";
  out << "\tBucket Columns:\n";
  for (const auto& column : bucketedBy_) {
    out << "\t\t" << column << "\n";
  }
  out << "\tBucket Types:\n";
  for (const auto& type : bucketTypes_) {
    out << "\t\t" << type->toString() << "\n";
  }
  if (!sortedBy_.empty()) {
    out << "\tSortedBy Columns:\n";
    for (const auto& sortColum : sortedBy_) {
      out << "\t\t" << sortColum->toString() << "\n";
    }
  }
  out << "]\n";
  return out.str();
}

HiveInsertTableHandle::HiveInsertTableHandle(
    std::vector<std::shared_ptr<const HiveColumnHandle>> inputColumns,
    std::shared_ptr<const LocationHandle> locationHandle,
    dwio::common::FileFormat storageFormat,
    std::shared_ptr<const HiveBucketProperty> bucketProperty,
    std::optional<common::CompressionKind> compressionKind,
    const std::unordered_map<std::string, std::string>& serdeParameters,
    const std::shared_ptr<dwio::common::WriterOptions>& writerOptions,
    // When this option is set the HiveDataSink will always write a file even
    // if there's no data. This is useful when the table is bucketed, but the
    // engine handles ensuring a 1 to 1 mapping from task to bucket.
    const bool ensureFiles,
    std::shared_ptr<const FileNameGenerator> fileNameGenerator,
    const std::unordered_map<std::string, std::string>& storageParameters)
    : inputColumns_(std::move(inputColumns)),
      locationHandle_(std::move(locationHandle)),
      storageFormat_(storageFormat),
      bucketProperty_(std::move(bucketProperty)),
      compressionKind_(compressionKind),
      serdeParameters_(serdeParameters),
      writerOptions_(writerOptions),
      ensureFiles_(ensureFiles),
      fileNameGenerator_(std::move(fileNameGenerator)),
      storageParameters_(storageParameters),
      partitionChannels_(computePartitionChannels(inputColumns_)),
      nonPartitionChannels_(computeNonPartitionChannels(inputColumns_)) {
  if (compressionKind.has_value()) {
    VELOX_CHECK(
        compressionKind.value() != common::CompressionKind_MAX,
        "Unsupported compression type: CompressionKind_MAX");
  }

  if (ensureFiles_) {
    // If ensureFiles is set and either the bucketProperty is set or some
    // partition keys are in the data, there is not a 1:1 mapping from Task to
    // files so we can't proactively create writers.
    VELOX_CHECK(
        bucketProperty_ == nullptr || bucketProperty_->bucketCount() == 0,
        "ensureFiles is not supported with bucketing");

    for (const auto& inputColumn : inputColumns_) {
      VELOX_CHECK(
          !inputColumn->isPartitionKey(),
          "ensureFiles is not supported with partition keys in the data");
    }
  }
}

HiveDataSink::HiveDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig)
    : HiveDataSink(
          inputType,
          insertTableHandle,
          connectorQueryCtx,
          commitStrategy,
          hiveConfig,
          getBucketCount(insertTableHandle->bucketProperty()),
          getBucketCount(insertTableHandle->bucketProperty()) > 0
              ? createBucketFunction(
                    *insertTableHandle->bucketProperty(),
                    inputType)
              : nullptr,
          insertTableHandle->partitionChannels(),
          insertTableHandle->nonPartitionChannels(),
          createPartitionIdGenerator(
              inputType,
              insertTableHandle,
              hiveConfig,
              connectorQueryCtx)) {}

HiveDataSink::HiveDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    uint32_t bucketCount,
    std::unique_ptr<core::PartitionFunction> bucketFunction,
    const std::vector<column_index_t>& partitionChannels,
    const std::vector<column_index_t>& dataChannels,
    std::unique_ptr<PartitionIdGenerator> partitionIdGenerator)
    : FileDataSink(
          std::move(inputType),
          connectorQueryCtx,
          commitStrategy,
          insertTableHandle->storageFormat(),
          hiveConfig->maxPartitionsPerWriters(
              connectorQueryCtx->sessionProperties()),
          partitionChannels,
          dataChannels,
          static_cast<int32_t>(bucketCount),
          std::move(bucketFunction),
          std::move(partitionIdGenerator),
          dwio::common::getWriterFactory(insertTableHandle->storageFormat()),
          hiveConfig->maxTargetFileSizeBytes(
              connectorQueryCtx->sessionProperties()),
          hiveConfig->isPartitionPathAsLowerCase(
              connectorQueryCtx->sessionProperties()),
          connectorQueryCtx->spillConfig(),
          getFinishTimeSliceLimitMsFromHiveConfig(
              hiveConfig,
              connectorQueryCtx->sessionProperties())),
      insertTableHandle_(std::move(insertTableHandle)),
      hiveConfig_(hiveConfig),
      updateMode_(getUpdateMode()),
      fileNameGenerator_(insertTableHandle_->fileNameGenerator()) {
  if (isBucketed()) {
    VELOX_USER_CHECK_LT(
        bucketCount_,
        hiveConfig_->maxBucketCount(connectorQueryCtx->sessionProperties()),
        "bucketCount exceeds the limit");
  }
  VELOX_USER_CHECK(
      (commitStrategy_ == CommitStrategy::kNoCommit) ||
          (commitStrategy_ == CommitStrategy::kTaskCommit),
      "Unsupported commit strategy: {}",
      CommitStrategyName::toName(commitStrategy_));

  if (insertTableHandle_->ensureFiles()) {
    VELOX_CHECK(
        !isPartitioned() && !isBucketed(),
        "ensureFiles is not supported with bucketing or partition keys in the data");
    ensureWriter(WriterId::unpartitionedId());
  }

  if (!isBucketed()) {
    return;
  }
  const auto& sortedProperty = insertTableHandle_->bucketProperty()->sortedBy();
  if (!sortedProperty.empty()) {
    sortColumnIndices_.reserve(sortedProperty.size());
    sortCompareFlags_.reserve(sortedProperty.size());
    for (int i = 0; i < sortedProperty.size(); ++i) {
      auto columnIndex =
          getNonPartitionTypes(dataChannels_, inputType_)
              ->getChildIdxIfExists(sortedProperty.at(i)->sortColumn());
      if (columnIndex.has_value()) {
        sortColumnIndices_.push_back(columnIndex.value());
        sortCompareFlags_.push_back(
            {sortedProperty.at(i)->sortOrder().isNullsFirst(),
             sortedProperty.at(i)->sortOrder().isAscending(),
             false,
             CompareFlags::NullHandlingMode::kNullAsValue});
      }
    }
    sortWrite_ = !sortColumnIndices_.empty();
  }
}

bool HiveDataSink::canReclaim() const {
  // Currently, we only support memory reclaim on dwrf file writer.
  return (spillConfig_ != nullptr) &&
      (insertTableHandle_->storageFormat() == dwio::common::FileFormat::DWRF ||
       insertTableHandle_->storageFormat() == dwio::common::FileFormat::NIMBLE);
}

void HiveDataSink::computePartitionAndBucketIds(const RowVectorPtr& input) {
  VELOX_CHECK(isPartitioned() || isBucketed());
  if (isPartitioned()) {
    if (!hiveConfig_->allowNullPartitionKeys(
            connectorQueryCtx_->sessionProperties())) {
      // Check that there are no nulls in the partition keys.
      for (auto& partitionIdx : partitionChannels_) {
        auto col = input->childAt(partitionIdx);
        if (col->mayHaveNulls()) {
          for (auto i = 0; i < col->size(); ++i) {
            VELOX_USER_CHECK(
                !col->isNullAt(i),
                "Partition key must not be null: {}",
                input->type()->asRow().nameOf(partitionIdx));
          }
        }
      }
    }
    partitionIdGenerator_->run(input, partitionIds_);
  }

  if (isBucketed()) {
    bucketFunction_->partition(*input, bucketIds_);
  }
}

void HiveDataSink::setMemoryReclaimers(
    WriterInfo* writerInfo,
    io::IoStatistics* ioStats) {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  if (connectorPool->reclaimer() == nullptr) {
    return;
  }
  writerInfo->writerPool->setReclaimer(
      WriterReclaimer::create(this, writerInfo, ioStats));
  writerInfo->sinkPool->setReclaimer(exec::MemoryReclaimer::create());
  // NOTE: we set the memory reclaimer for sort pool when we construct the sort
  // writer.
}

std::vector<std::string> HiveDataSink::commitMessage() const {
  std::vector<std::string> partitionUpdates;
  partitionUpdates.reserve(writerInfo_.size());
  for (int i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);

    folly::dynamic fileWriteInfosArray = folly::dynamic::array;
    for (const auto& fileInfo : info->writtenFiles) {
      fileWriteInfosArray.push_back(
          folly::dynamic::object(
              HiveCommitMessage::kWriteFileName, fileInfo.writeFileName)(
              HiveCommitMessage::kTargetFileName, fileInfo.targetFileName)(
              HiveCommitMessage::kFileSize, fileInfo.fileSize));
    }

    // clang-format off
      auto partitionUpdateJson = folly::toJson(
       folly::dynamic::object
          (HiveCommitMessage::kName, info->writerParameters.partitionName().value_or(""))
          (HiveCommitMessage::kUpdateMode,
            WriterParameters::updateModeToString(
              info->writerParameters.updateMode()))
          (HiveCommitMessage::kWritePath, info->writerParameters.writeDirectory())
          (HiveCommitMessage::kTargetPath, info->writerParameters.targetDirectory())
          (HiveCommitMessage::kFileWriteInfos, std::move(fileWriteInfosArray))
          (HiveCommitMessage::kRowCount, info->numWrittenRows)
          (HiveCommitMessage::kInMemoryDataSizeInBytes, info->inputSizeInBytes)
          (HiveCommitMessage::kOnDiskDataSizeInBytes, ioStats_.at(i)->rawBytesWritten())
          (HiveCommitMessage::kContainsNumberedFileNames, true));
    // clang-format on
    partitionUpdates.push_back(partitionUpdateJson);
  }
  return partitionUpdates;
}

std::shared_ptr<dwio::common::WriterOptions> HiveDataSink::createWriterOptions()
    const {
  // Default: use the last writer's info (for appendWriter which just added it)
  return createWriterOptions(writerInfo_.size() - 1);
}

std::shared_ptr<dwio::common::WriterOptions> HiveDataSink::createWriterOptions(
    size_t writerIndex) const {
  VELOX_CHECK_LT(writerIndex, writerInfo_.size());

  // Take the writer options provided by the user as a starting point, or
  // allocate a new one.
  auto options = insertTableHandle_->writerOptions();
  if (!options) {
    options = writerFactory_->createWriterOptions();
  }

  const auto* connectorSessionProperties =
      connectorQueryCtx_->sessionProperties();

  // Only overwrite options in case they were not already provided.
  if (options->schema == nullptr) {
    options->schema = getNonPartitionTypes(dataChannels_, inputType_);
  }

  if (options->memoryPool == nullptr) {
    options->memoryPool = writerInfo_[writerIndex]->writerPool.get();
  }

  if (!options->compressionKind) {
    options->compressionKind = insertTableHandle_->compressionKind();
  }

  if (options->spillConfig == nullptr && canReclaim()) {
    options->spillConfig = spillConfig_;
  }

  // Always set nonReclaimableSection to the current writer's holder.
  // Since insertTableHandle_->writerOptions() returns a shared_ptr, we need
  // to ensure each writer has its own nonReclaimableSection pointer.
  options->nonReclaimableSection =
      writerInfo_[writerIndex]->nonReclaimableSectionHolder.get();

  if (options->memoryReclaimerFactory == nullptr ||
      options->memoryReclaimerFactory() == nullptr) {
    options->memoryReclaimerFactory = []() {
      return exec::MemoryReclaimer::create();
    };
  }

  if (options->serdeParameters.empty()) {
    options->serdeParameters = std::map<std::string, std::string>(
        insertTableHandle_->serdeParameters().begin(),
        insertTableHandle_->serdeParameters().end());
  }

  options->sessionTimezoneName = connectorQueryCtx_->sessionTimezone();
  options->adjustTimestampToTimezone =
      connectorQueryCtx_->adjustTimestampToTimezone();
  options->processConfigs(*hiveConfig_->config(), *connectorSessionProperties);
  return options;
}

std::unique_ptr<dwio::common::Writer> HiveDataSink::createWriterForIndex(
    size_t writerIndex) {
  VELOX_CHECK_LT(writerIndex, writerInfo_.size());
  VELOX_CHECK_LT(writerIndex, ioStats_.size());

  auto& info = writerInfo_[writerIndex];
  const auto& params = info->writerParameters;

  // Compute and store the new file names.
  info->currentWriteFileName =
      makeSequencedFileName(params.writeFileName(), info->fileSequenceNumber);
  info->currentTargetFileName =
      makeSequencedFileName(params.targetFileName(), info->fileSequenceNumber);

  const auto writePath =
      (fs::path(params.writeDirectory()) / info->currentWriteFileName).string();

  auto options = createWriterOptions(writerIndex);

  // Prevents the memory allocation during the writer creation.
  WRITER_NON_RECLAIMABLE_SECTION_GUARD(writerIndex);
  auto writer = writerFactory_->createWriter(
      createHiveFileSink(
          writePath,
          hiveConfig_,
          info->sinkPool.get(),
          ioStats_[writerIndex].get(),
          fileSystemStats_.get(),
          insertTableHandle_->storageParameters()),
      options);
  return maybeCreateBucketSortWriter(writerIndex, std::move(writer));
}

std::string HiveDataSink::getPartitionName(uint32_t partitionId) const {
  VELOX_CHECK_NOT_NULL(partitionIdGenerator_);

  return HivePartitionName::partitionName(
      partitionId,
      partitionIdGenerator_->partitionValues(),
      partitionKeyAsLowerCase_);
}

std::unique_ptr<facebook::velox::dwio::common::Writer>
HiveDataSink::maybeCreateBucketSortWriter(
    size_t writerIndex,
    std::unique_ptr<facebook::velox::dwio::common::Writer> writer) {
  if (!sortWrite()) {
    return writer;
  }
  auto* sortPool = writerInfo_[writerIndex]->sortPool.get();
  VELOX_CHECK_NOT_NULL(sortPool);
  auto sortBuffer = std::make_unique<exec::SortBuffer>(
      getNonPartitionTypes(dataChannels_, inputType_),
      sortColumnIndices_,
      sortCompareFlags_,
      sortPool,
      writerInfo_[writerIndex]->nonReclaimableSectionHolder.get(),
      connectorQueryCtx_->prefixSortConfig(),
      spillConfig_,
      writerInfo_[writerIndex]->spillStats.get());

  return std::make_unique<dwio::common::SortingWriter>(
      std::move(writer),
      std::move(sortBuffer),
      hiveConfig_->sortWriterMaxOutputRows(
          connectorQueryCtx_->sessionProperties()),
      hiveConfig_->sortWriterMaxOutputBytes(
          connectorQueryCtx_->sessionProperties()),
      sortWriterFinishTimeSliceLimitMs_);
}

WriterParameters HiveDataSink::getWriterParameters(
    const std::optional<std::string>& partition,
    std::optional<uint32_t> bucketId) const {
  auto [targetFileName, writeFileName] = getWriterFileNames(bucketId);

  return WriterParameters{
      updateMode_,
      partition,
      targetFileName,
      makePartitionDirectory(
          insertTableHandle_->locationHandle()->targetPath(), partition),
      writeFileName,
      makePartitionDirectory(
          insertTableHandle_->locationHandle()->writePath(), partition)};
}

std::pair<std::string, std::string> HiveDataSink::getWriterFileNames(
    std::optional<uint32_t> bucketId) const {
  if (auto hiveInsertFileNameGenerator =
          std::dynamic_pointer_cast<const HiveInsertFileNameGenerator>(
              fileNameGenerator_)) {
    return hiveInsertFileNameGenerator->gen(
        bucketId,
        insertTableHandle_,
        *connectorQueryCtx_,
        hiveConfig_,
        isCommitRequired());
  }

  return fileNameGenerator_->gen(
      bucketId, insertTableHandle_, *connectorQueryCtx_, isCommitRequired());
}

std::pair<std::string, std::string> HiveInsertFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    bool commitRequired) const {
  auto defaultHiveConfig =
      std::make_shared<const HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));

  return this->gen(
      bucketId,
      insertTableHandle,
      connectorQueryCtx,
      defaultHiveConfig,
      commitRequired);
}

std::pair<std::string, std::string> HiveInsertFileNameGenerator::gen(
    std::optional<uint32_t> bucketId,
    const std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx& connectorQueryCtx,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    bool commitRequired) const {
  auto targetFileName = insertTableHandle->locationHandle()->targetFileName();
  const bool generateFileName = targetFileName.empty();
  if (bucketId.has_value()) {
    VELOX_CHECK(generateFileName);
    // TODO: add hive.file_renaming_enabled support.
    targetFileName = computeBucketedFileName(
        connectorQueryCtx.queryId(),
        hiveConfig->maxBucketCount(connectorQueryCtx.sessionProperties()),
        bucketId.value());
    // queryId may contain unsafe characters.
    sanitizeFileName(targetFileName);
  } else if (generateFileName) {
    // targetFileName includes planNodeId and Uuid. As a result, different
    // table writers run by the same task driver or the same table writer
    // run in different task tries would have different targetFileNames.
    targetFileName = fmt::format(
        "{}_{}_{}_{}",
        connectorQueryCtx.taskId(),
        connectorQueryCtx.driverId(),
        connectorQueryCtx.planNodeId(),
        makeUuid());
    // taskId, planNodeId may contain unsafe characters.
    sanitizeFileName(targetFileName);
  }
  // do not try to sanitize user provided targetFileName
  VELOX_CHECK(!targetFileName.empty());
  const std::string writeFileName = commitRequired
      ? fmt::format(".tmp.velox.{}_{}", targetFileName, makeUuid())
      : targetFileName;
  if (generateFileName &&
      insertTableHandle->storageFormat() == dwio::common::FileFormat::PARQUET) {
    return {
        fmt::format("{}{}", targetFileName, ".parquet"),
        fmt::format("{}{}", writeFileName, ".parquet")};
  }
  return {targetFileName, writeFileName};
}

void HiveInsertFileNameGenerator::sanitizeFileName(std::string& name) {
  static const re2::RE2 re("[^a-zA-Z0-9._-]");
  re2::RE2::GlobalReplace(&name, re, "_");
}

folly::dynamic HiveInsertFileNameGenerator::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveInsertFileNameGenerator";
  return obj;
}

std::shared_ptr<HiveInsertFileNameGenerator>
HiveInsertFileNameGenerator::deserialize(
    const folly::dynamic& /* obj */,
    void* /* context */) {
  return std::make_shared<HiveInsertFileNameGenerator>();
}

void HiveInsertFileNameGenerator::registerSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register(
      "HiveInsertFileNameGenerator", HiveInsertFileNameGenerator::deserialize);
}

std::string HiveInsertFileNameGenerator::toString() const {
  return "HiveInsertFileNameGenerator";
}

WriterParameters::UpdateMode HiveDataSink::getUpdateMode() const {
  if (insertTableHandle_->isExistingTable()) {
    if (insertTableHandle_->isPartitioned()) {
      const auto insertBehavior = hiveConfig_->insertExistingPartitionsBehavior(
          connectorQueryCtx_->sessionProperties());
      switch (insertBehavior) {
        case HiveConfig::InsertExistingPartitionsBehavior::kOverwrite:
          return WriterParameters::UpdateMode::kOverwrite;
        case HiveConfig::InsertExistingPartitionsBehavior::kError:
          return WriterParameters::UpdateMode::kNew;
        default:
          VELOX_UNSUPPORTED(
              "Unsupported insert existing partitions behavior: {}",
              HiveConfig::insertExistingPartitionsBehaviorString(
                  insertBehavior));
      }
    } else {
      if (hiveConfig_->immutablePartitions()) {
        VELOX_USER_FAIL("Unpartitioned Hive tables are immutable.");
      }
      return WriterParameters::UpdateMode::kAppend;
    }
  } else {
    return WriterParameters::UpdateMode::kNew;
  }
}

bool HiveInsertTableHandle::isPartitioned() const {
  return std::any_of(
      inputColumns_.begin(), inputColumns_.end(), [](auto column) {
        return column->isPartitionKey();
      });
}

const HiveBucketProperty* HiveInsertTableHandle::bucketProperty() const {
  return bucketProperty_.get();
}

bool HiveInsertTableHandle::isBucketed() const {
  return bucketProperty() != nullptr;
}

bool HiveInsertTableHandle::isExistingTable() const {
  return locationHandle_->tableType() == LocationHandle::TableType::kExisting;
}

folly::dynamic HiveInsertTableHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "HiveInsertTableHandle";
  folly::dynamic arr = folly::dynamic::array;
  for (const auto& ic : inputColumns_) {
    arr.push_back(ic->serialize());
  }

  obj["inputColumns"] = arr;
  obj["locationHandle"] = locationHandle_->serialize();
  obj["tableStorageFormat"] = dwio::common::toString(storageFormat_);

  if (bucketProperty_) {
    obj["bucketProperty"] = bucketProperty_->serialize();
  }

  if (compressionKind_.has_value()) {
    obj["compressionKind"] = common::compressionKindToString(*compressionKind_);
  }

  folly::dynamic params = folly::dynamic::object;
  for (const auto& [key, value] : serdeParameters_) {
    params[key] = value;
  }
  obj["serdeParameters"] = params;

  folly::dynamic storageParams = folly::dynamic::object;
  for (const auto& [key, value] : storageParameters_) {
    storageParams[key] = value;
  }
  obj["storageParameters"] = storageParams;

  obj["ensureFiles"] = ensureFiles_;
  obj["fileNameGenerator"] = fileNameGenerator_->serialize();
  return obj;
}

HiveInsertTableHandlePtr HiveInsertTableHandle::create(
    const folly::dynamic& obj) {
  auto inputColumns = ISerializable::deserialize<std::vector<HiveColumnHandle>>(
      obj["inputColumns"]);
  auto locationHandle =
      ISerializable::deserialize<LocationHandle>(obj["locationHandle"]);
  auto storageFormat =
      dwio::common::toFileFormat(obj["tableStorageFormat"].asString());

  std::optional<common::CompressionKind> compressionKind = std::nullopt;
  if (obj.count("compressionKind") > 0) {
    compressionKind =
        common::stringToCompressionKind(obj["compressionKind"].asString());
  }

  std::shared_ptr<const HiveBucketProperty> bucketProperty;
  if (obj.count("bucketProperty") > 0) {
    bucketProperty =
        ISerializable::deserialize<HiveBucketProperty>(obj["bucketProperty"]);
  }

  std::unordered_map<std::string, std::string> serdeParameters;
  for (const auto& pair : obj["serdeParameters"].items()) {
    serdeParameters.emplace(pair.first.asString(), pair.second.asString());
  }

  std::unordered_map<std::string, std::string> storageParameters;
  if (obj.count("storageParameters") > 0) {
    for (const auto& pair : obj["storageParameters"].items()) {
      storageParameters.emplace(pair.first.asString(), pair.second.asString());
    }
  }

  bool ensureFiles = obj["ensureFiles"].asBool();

  auto fileNameGenerator =
      ISerializable::deserialize<FileNameGenerator>(obj["fileNameGenerator"]);
  return std::make_shared<HiveInsertTableHandle>(
      inputColumns,
      locationHandle,
      storageFormat,
      bucketProperty,
      compressionKind,
      serdeParameters,
      nullptr, // writerOptions is not serializable
      ensureFiles,
      fileNameGenerator,
      storageParameters);
}

void HiveInsertTableHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveInsertTableHandle", HiveInsertTableHandle::create);
}

std::string HiveInsertTableHandle::toString() const {
  std::ostringstream out;
  out << "HiveInsertTableHandle [" << dwio::common::toString(storageFormat_);
  if (compressionKind_.has_value()) {
    out << " " << common::compressionKindToString(compressionKind_.value());
  } else {
    out << " none";
  }
  out << "], [inputColumns: [";
  for (const auto& i : inputColumns_) {
    out << " " << i->toString();
  }
  out << " ], locationHandle: " << locationHandle_->toString();
  if (bucketProperty_) {
    out << ", bucketProperty: " << bucketProperty_->toString();
  }

  if (serdeParameters_.size() > 0) {
    std::map<std::string, std::string> sortedSerdeParams(
        serdeParameters_.begin(), serdeParameters_.end());
    out << ", serdeParameters: ";
    for (const auto& [key, value] : sortedSerdeParams) {
      out << "[" << key << ", " << value << "] ";
    }
  }
  out << ", fileNameGenerator: " << fileNameGenerator_->toString();
  out << "]";
  return out.str();
}

std::string LocationHandle::toString() const {
  return fmt::format(
      "LocationHandle [targetPath: {}, writePath: {}, tableType: {}, tableFileName: {}]",
      targetPath_,
      writePath_,
      tableTypeName(tableType_),
      targetFileName_);
}

void LocationHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("LocationHandle", LocationHandle::create);
}

folly::dynamic LocationHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "LocationHandle";
  obj["targetPath"] = targetPath_;
  obj["writePath"] = writePath_;
  obj["tableType"] = tableTypeName(tableType_);
  obj["targetFileName"] = targetFileName_;
  return obj;
}

LocationHandlePtr LocationHandle::create(const folly::dynamic& obj) {
  auto targetPath = obj["targetPath"].asString();
  auto writePath = obj["writePath"].asString();
  auto tableType = tableTypeFromName(obj["tableType"].asString());
  auto targetFileName = obj["targetFileName"].asString();
  return std::make_shared<LocationHandle>(
      targetPath, writePath, tableType, targetFileName);
}

std::unique_ptr<memory::MemoryReclaimer> HiveDataSink::WriterReclaimer::create(
    HiveDataSink* dataSink,
    WriterInfo* writerInfo,
    io::IoStatistics* ioStats) {
  return std::unique_ptr<memory::MemoryReclaimer>(
      new HiveDataSink::WriterReclaimer(dataSink, writerInfo, ioStats));
}

bool HiveDataSink::WriterReclaimer::reclaimableBytes(
    const memory::MemoryPool& pool,
    uint64_t& reclaimableBytes) const {
  VELOX_CHECK_EQ(pool.name(), writerInfo_->writerPool->name());
  reclaimableBytes = 0;
  if (!dataSink_->canReclaim()) {
    return false;
  }
  return exec::MemoryReclaimer::reclaimableBytes(pool, reclaimableBytes);
}

uint64_t HiveDataSink::WriterReclaimer::reclaim(
    memory::MemoryPool* pool,
    uint64_t targetBytes,
    uint64_t maxWaitMs,
    memory::MemoryReclaimer::Stats& stats) {
  VELOX_CHECK_EQ(pool->name(), writerInfo_->writerPool->name());
  if (!dataSink_->canReclaim()) {
    return 0;
  }

  if (*writerInfo_->nonReclaimableSectionHolder.get()) {
    RECORD_METRIC_VALUE(kMetricMemoryNonReclaimableCount);
    LOG(WARNING) << "Can't reclaim from hive writer pool " << pool->name()
                 << " which is under non-reclaimable section, root pool: "
                 << pool->root()->name()
                 << ", state: " << stateString(dataSink_->state_)
                 << ", used: " << succinctBytes(pool->usedBytes())
                 << ", reservation: " << succinctBytes(pool->reservedBytes())
                 << ", root pool reservation: "
                 << succinctBytes(pool->root()->reservedBytes());
    ++stats.numNonReclaimableAttempts;
    return 0;
  }

  const uint64_t memoryUsageBeforeReclaim = pool->reservedBytes();
  const std::string memoryUsageTreeBeforeReclaim = pool->treeMemoryUsage();
  const auto writtenBytesBeforeReclaim = ioStats_->rawBytesWritten();
  const auto reclaimedBytes =
      exec::MemoryReclaimer::reclaim(pool, targetBytes, maxWaitMs, stats);
  const auto earlyFlushedRawBytes =
      ioStats_->rawBytesWritten() - writtenBytesBeforeReclaim;
  addThreadLocalRuntimeStat(
      kEarlyFlushedRawBytes,
      RuntimeCounter(
          saturateCast(earlyFlushedRawBytes), RuntimeCounter::Unit::kBytes));
  if (earlyFlushedRawBytes > 0) {
    RECORD_METRIC_VALUE(
        kMetricFileWriterEarlyFlushedRawBytes, earlyFlushedRawBytes);
  }
  const uint64_t memoryUsageAfterReclaim = pool->reservedBytes();
  if (memoryUsageAfterReclaim > memoryUsageBeforeReclaim) {
    VELOX_FAIL(
        "Unexpected memory growth after memory reclaim from {}, the memory usage before reclaim: {}, after reclaim: {}\nThe memory tree usage before reclaim:\n{}\nThe memory tree usage after reclaim:\n{}",
        pool->name(),
        succinctBytes(memoryUsageBeforeReclaim),
        succinctBytes(memoryUsageAfterReclaim),
        memoryUsageTreeBeforeReclaim,
        pool->treeMemoryUsage());
  }
  return reclaimedBytes;
}
} // namespace facebook::velox::connector::hive
