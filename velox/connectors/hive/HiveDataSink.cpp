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
#include "velox/common/config/Config.h"
#include "velox/connectors/common/FileConnectorUtil.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/connectors/hive/SortingWriter.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/SortBuffer.h"

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

std::unique_ptr<core::PartitionFunction> createBucketFunction(
    const HiveBucketProperty& bucketProperty,
    const RowTypePtr& inputType) {
  VELOX_USER_CHECK_EQ(
      bucketProperty.kind(),
      HiveBucketProperty::Kind::kHiveCompatible,
      "Unsupported bucket function");
  const auto& bucketedBy = bucketProperty.bucketedBy();
  const auto& bucketedTypes = bucketProperty.bucketedTypes();
  std::vector<column_index_t> bucketedByChannels;
  bucketedByChannels.reserve(bucketedBy.size());
  for (int32_t i = 0; i < bucketedBy.size(); ++i) {
    const auto& bucketColumn = bucketedBy[i];
    const auto& bucketType = bucketedTypes[i];
    const auto inputChannel = inputType->getChildIdx(bucketColumn);
    if (!inputType->childAt(inputChannel)->equivalent(*bucketType))
        [[unlikely]] {
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

} // namespace

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
              insertTableHandle->storageFormat(),
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

  if (insertTableHandle_->writeEmptyFile()) {
    VELOX_CHECK(
        !isPartitioned() && !isBucketed(),
        "writeEmptyFile is not supported with bucketing or partition keys in the data");
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

  std::shared_ptr<dwio::common::WriterOptions> options =
      writerFactory_->createWriterOptions();

  const auto* connectorSessionProperties =
      connectorQueryCtx_->sessionProperties();

  if (options->schema == nullptr) {
    options->schema = getNonPartitionTypes(dataChannels_, inputType_);
  }

  if (!options->compressionKind) {
    options->compressionKind = insertTableHandle_->compressionKind();
  }

  if (options->spillConfig == nullptr && canReclaim()) {
    options->spillConfig = spillConfig_;
  }

  options->memoryPool = writerInfo_[writerIndex]->writerPool.get();
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
  options->maxTargetFileSizeBytes =
      (isBucketed() || sortWrite()) ? 0 : maxTargetFileBytes_;
  auto formatScopedConfigs = connector::makeFormatScopedConfigs(
      *hiveConfig_, *connectorSessionProperties, writerFactory_->fileFormat());
  auto sessionFormatOptions = writerFactory_->createFormatOptions(
      formatScopedConfigs.connectorConfig,
      formatScopedConfigs.sessionProperties);
  if (options->formatSpecificOptions == nullptr) {
    options->formatSpecificOptions = std::move(sessionFormatOptions);
  } else if (sessionFormatOptions != nullptr) {
    options->formatSpecificOptions->merge(*sessionFormatOptions);
  }
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

  return std::make_unique<SortingWriter>(
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
  return fileNameGenerator_->gen(
      bucketId,
      insertTableHandle_,
      *connectorQueryCtx_,
      hiveConfig_->maxBucketCount(connectorQueryCtx_->sessionProperties()),
      isCommitRequired());
}

WriterParameters::UpdateMode HiveDataSink::getUpdateMode() const {
  if (insertTableHandle_->isExistingTable()) {
    if (insertTableHandle_->isPartitioned()) {
      auto insertBehavior =
          insertTableHandle_->insertExistingPartitionsBehavior();
      if (!insertBehavior.has_value()) {
        insertBehavior = hiveConfig_->insertExistingPartitionsBehavior(
            connectorQueryCtx_->sessionProperties());
      }
      switch (insertBehavior.value()) {
        case HiveConfig::InsertExistingPartitionsBehavior::kOverwrite:
          return WriterParameters::UpdateMode::kOverwrite;
        case HiveConfig::InsertExistingPartitionsBehavior::kError:
          return WriterParameters::UpdateMode::kNew;
      }
      VELOX_UNREACHABLE();
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
