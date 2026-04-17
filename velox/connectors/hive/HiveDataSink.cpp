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

#include "velox/connectors/hive/BucketSortingWriter.h"
#include "velox/connectors/hive/PartitionWriter.h"
#include "velox/connectors/hive/RotationWriter.h"
#include "velox/connectors/hive/HiveInsertTableHandle.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/Fs.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/connectors/hive/HivePartitionName.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/OperatorUtils.h"

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::connector::hive {
namespace {

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

// Returns the type of non-partition data columns.
RowTypePtr getNonPartitionTypes(
    const std::vector<column_index_t>& dataCols,
    const RowTypePtr& inputType) {
  std::vector<std::string> childNames;
  std::vector<TypePtr> childTypes;
  const auto& dataSize = dataCols.size();
  childNames.reserve(dataSize);
  childTypes.reserve(dataSize);
  for (int dataCol : dataCols) {
    childNames.push_back(inputType->nameOf(dataCol));
    childTypes.push_back(inputType->childAt(dataCol));
  }

  return ROW(std::move(childNames), std::move(childTypes));
}

std::shared_ptr<dwio::common::WriterOptions> cloneWriterOptions(
    const std::shared_ptr<dwio::common::WriterOptions>& writerOptions) {
  VELOX_CHECK_NOT_NULL(writerOptions);
  return writerOptions->clone();
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

std::shared_ptr<memory::MemoryPool> createSinkPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sink", writerPool->name()));
}

std::shared_ptr<memory::MemoryPool> createSortPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sort", writerPool->name()));
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
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      commitStrategy_(commitStrategy),
      hiveConfig_(hiveConfig),
      updateMode_(getUpdateMode()),
      maxOpenWriters_(hiveConfig_->maxPartitionsPerWriters(
          connectorQueryCtx->sessionProperties())),
      partitionChannels_(partitionChannels),
      partitionIdGenerator_(std::move(partitionIdGenerator)),
      dataChannels_(dataChannels),
      bucketCount_(static_cast<int32_t>(bucketCount)),
      bucketFunction_(std::move(bucketFunction)),
      writerFactory_(
          dwio::common::getWriterFactory(insertTableHandle_->storageFormat())),
      spillConfig_(connectorQueryCtx->spillConfig()),
      maxTargetFileBytes_(hiveConfig_->maxTargetFileSizeBytes(
          connectorQueryCtx->sessionProperties())),
      partitionKeyAsLowerCase_(hiveConfig_->isPartitionPathAsLowerCase(
          connectorQueryCtx_->sessionProperties())),
      fileNameGenerator_(insertTableHandle_->fileNameGenerator()) {
  fileSystemStats_ = std::make_unique<IoStats>();

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

  // Compute the data column type once, used by BucketSortingWriter and
  // PartitionWriter.
  auto dataType = getNonPartitionTypes(dataChannels_, inputType_);

  // Build bucket sort configuration.
  std::vector<column_index_t> sortColumnIndices;
  std::vector<CompareFlags> sortCompareFlags;
  if (isBucketed()) {
    const auto& sortedProperty =
        insertTableHandle_->bucketProperty()->sortedBy();
    if (!sortedProperty.empty()) {
      sortColumnIndices.reserve(sortedProperty.size());
      sortCompareFlags.reserve(sortedProperty.size());
      for (int i = 0; i < sortedProperty.size(); ++i) {
        auto columnIndex =
            dataType->getChildIdxIfExists(sortedProperty.at(i)->sortColumn());
        if (columnIndex.has_value()) {
          sortColumnIndices.push_back(columnIndex.value());
          sortCompareFlags.push_back(
              {sortedProperty.at(i)->sortOrder().isNullsFirst(),
               sortedProperty.at(i)->sortOrder().isAscending(),
               false,
               CompareFlags::NullHandlingMode::kNullAsValue});
        }
      }
    }
  }
  bucketSortingWriter_ = std::make_unique<BucketSortingWriter>(
      dataType,
      std::move(sortColumnIndices),
      std::move(sortCompareFlags),
      getFinishTimeSliceLimitMsFromHiveConfig(
          hiveConfig_, connectorQueryCtx->sessionProperties()),
      hiveConfig_->sortWriterMaxOutputRows(
          connectorQueryCtx->sessionProperties()),
      hiveConfig_->sortWriterMaxOutputBytes(
          connectorQueryCtx->sessionProperties()),
      connectorQueryCtx_->prefixSortConfig(),
      spillConfig_);

  partitionWriter_ = std::make_unique<PartitionWriter>(
      maxOpenWriters_,
      dataChannels_,
      std::move(dataType),
      [this](const HiveWriterId& id, uint32_t writerIndex) {
        return createRotationWriter(id, writerIndex);
      },
      connectorQueryCtx_->memoryPool());

  if (insertTableHandle_->ensureFiles()) {
    VELOX_CHECK(
        !isPartitioned() && !isBucketed(),
        "ensureFiles is not supported with bucketing or partition keys in the data");
    partitionWriter_->ensureWriter(HiveWriterId::unpartitionedId());
  }
}

HiveDataSink::~HiveDataSink() = default;

bool HiveDataSink::canReclaim() const {
  // Currently, we only support memory reclaim on dwrf file writer.
  return (spillConfig_ != nullptr) &&
      (insertTableHandle_->storageFormat() == dwio::common::FileFormat::DWRF ||
       insertTableHandle_->storageFormat() == dwio::common::FileFormat::NIMBLE);
}

void HiveDataSink::appendData(RowVectorPtr input) {
  checkRunning();

  // Lazy load all the input columns.
  input->loadedVector();

  // Write to unpartitioned (and unbucketed) table.
  if (!isPartitioned() && !isBucketed()) {
    partitionWriter_->write(HiveWriterId::unpartitionedId(), input);
    return;
  }

  // Compute partition and bucket numbers.
  computePartitionAndBucketIds(input);

  // All inputs belong to a single non-bucketed partition. The partition id
  // must be zero.
  if (!isBucketed() && partitionIdGenerator_->numPartitions() == 1) {
    partitionWriter_->write(HiveWriterId{0}, input);
    return;
  }

  partitionWriter_->write(
      input, partitionIds_, bucketIds_, isPartitioned(), isBucketed());
}

std::string HiveDataSink::stateString(State state) {
  switch (state) {
    case State::kRunning:
      return "RUNNING";
    case State::kFinishing:
      return "FLUSHING";
    case State::kClosed:
      return "CLOSED";
    case State::kAborted:
      return "ABORTED";
    default:
      VELOX_UNREACHABLE("BAD STATE: {}", static_cast<int>(state));
  }
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

DataSink::Stats HiveDataSink::stats() const {
  Stats stats;
  if (state_ == State::kAborted) {
    return stats;
  }

  const auto& writers = partitionWriter_->writers();
  for (const auto& writer : writers) {
    stats.numWrittenBytes += writer->ioStats()->rawBytesWritten();
    stats.writeIOTimeUs += writer->ioStats()->writeIOTimeUs();
  }

  if (state_ != State::kClosed) {
    return stats;
  }

  // Count total files written, including rotated files.
  stats.numWrittenFiles = 0;
  for (const auto& writer : writers) {
    const auto& info = writer->writerInfo();
    VELOX_CHECK_NOT_NULL(info);
    stats.numWrittenFiles += info->writtenFiles.size();
    if (!info->spillStats->empty()) {
      stats.spillStats += *info->spillStats;
    }
  }
  return stats;
}

std::unordered_map<std::string, RuntimeCounter> HiveDataSink::runtimeStats()
    const {
  std::unordered_map<std::string, RuntimeCounter> runtimeStats;

  const auto fsStatsMap = fileSystemStats_->stats();
  for (const auto& [statName, statValue] : fsStatsMap) {
    runtimeStats.emplace(
        statName, RuntimeCounter(statValue.sum, statValue.unit));
  }

  return runtimeStats;
}

std::shared_ptr<memory::MemoryPool> HiveDataSink::createWriterPool(
    const HiveWriterId& writerId) {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  return connectorPool->addAggregateChild(
      fmt::format("{}.{}", connectorPool->name(), writerId.toString()));
}

void HiveDataSink::setMemoryReclaimers(
    HiveWriterInfo* writerInfo,
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

void HiveDataSink::setState(State newState) {
  checkStateTransition(state_, newState);
  state_ = newState;
}

/// Validates the state transition from 'oldState' to 'newState'.
void HiveDataSink::checkStateTransition(State oldState, State newState) {
  switch (oldState) {
    case State::kRunning:
      if (newState == State::kAborted || newState == State::kFinishing) {
        return;
      }
      break;
    case State::kFinishing:
      if (newState == State::kAborted || newState == State::kClosed ||
          // The finishing state is reentry state if we yield in the middle of
          // finish processing if a single run takes too long.
          newState == State::kFinishing) {
        return;
      }
      [[fallthrough]];
    case State::kAborted:
    case State::kClosed:
    default:
      break;
  }
  VELOX_FAIL("Unexpected state transition from {} to {}", oldState, newState);
}

bool HiveDataSink::finish() {
  // Flush is reentry state.
  setState(State::kFinishing);

  // As for now, only sorted writer needs flush buffered data. For non-sorted
  // writer, data is directly written to the underlying file writer.
  if (!bucketSortingWriter_->enabled()) {
    return true;
  }

  return partitionWriter_->finish(
      bucketSortingWriter_->finishTimeSliceLimitMs());
}

std::vector<std::string> HiveDataSink::close() {
  setState(State::kClosed);
  closeInternal();
  return commitMessage();
}

std::vector<std::string> HiveDataSink::commitMessage() const {
  const auto& writers = partitionWriter_->writers();
  std::vector<std::string> partitionUpdates;
  partitionUpdates.reserve(writers.size());
  for (size_t i = 0; i < writers.size(); ++i) {
    const auto& info = writers.at(i)->writerInfo();
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
            HiveWriterParameters::updateModeToString(
              info->writerParameters.updateMode()))
          (HiveCommitMessage::kWritePath, info->writerParameters.writeDirectory())
          (HiveCommitMessage::kTargetPath, info->writerParameters.targetDirectory())
          (HiveCommitMessage::kFileWriteInfos, std::move(fileWriteInfosArray))
          (HiveCommitMessage::kRowCount, info->numWrittenRows)
          (HiveCommitMessage::kInMemoryDataSizeInBytes, info->inputSizeInBytes)
          (HiveCommitMessage::kOnDiskDataSizeInBytes, writers.at(i)->ioStats()->rawBytesWritten())
          (HiveCommitMessage::kContainsNumberedFileNames, true));
    // clang-format on
    partitionUpdates.push_back(partitionUpdateJson);
  }
  return partitionUpdates;
}

void HiveDataSink::abort() {
  setState(State::kAborted);
  closeInternal();
}

void HiveDataSink::closeInternal() {
  VELOX_CHECK_NE(state_, State::kRunning);
  VELOX_CHECK_NE(state_, State::kFinishing);

  TestValue::adjust(
      "facebook::velox::connector::hive::HiveDataSink::closeInternal", this);

  if (state_ == State::kClosed) {
    partitionWriter_->close();
  } else {
    partitionWriter_->abort();
  }
}

std::shared_ptr<dwio::common::WriterOptions> HiveDataSink::createWriterOptions(
    const HiveWriterInfo* writerInfo) const {
  VELOX_CHECK_NOT_NULL(writerInfo);

  // Clone writer options for each writer instance to avoid sharing mutable
  // writer-scoped state (e.g. memoryPool, nonReclaimableSection) across
  // different partition writers.
  std::shared_ptr<dwio::common::WriterOptions> options;
  if (insertTableHandle_->writerOptions() != nullptr) {
    options = cloneWriterOptions(insertTableHandle_->writerOptions());
  } else {
    options = writerFactory_->createWriterOptions();
  }

  const auto* connectorSessionProperties =
      connectorQueryCtx_->sessionProperties();

  // Only overwrite options in case they were not already provided.
  if (options->schema == nullptr) {
    options->schema = getNonPartitionTypes(dataChannels_, inputType_);
  }

  if (options->memoryPool == nullptr) {
    options->memoryPool = writerInfo->writerPool.get();
  }

  if (!options->compressionKind) {
    options->compressionKind = insertTableHandle_->compressionKind();
  }

  if (options->spillConfig == nullptr && canReclaim()) {
    options->spillConfig = spillConfig_;
  }

  // Always set nonReclaimableSection to the current writer's holder.
  options->nonReclaimableSection =
      writerInfo->nonReclaimableSectionHolder.get();

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

std::unique_ptr<PartitionWriterInterface> HiveDataSink::createRotationWriter(
    const HiveWriterId& id,
    uint32_t writerIndex) {
  (void)writerIndex;

  std::optional<std::string> partitionName;
  if (isPartitioned()) {
    partitionName = getPartitionName(id.partitionId.value());
  }

  // Without explicitly setting flush policy, the default memory based flush
  // policy is used.
  auto writerParameters = getWriterParameters(partitionName, id.bucketId);
  auto writerPool = createWriterPool(id);
  auto sinkPool = createSinkPool(writerPool);
  std::shared_ptr<memory::MemoryPool> sortPool{nullptr};
  if (bucketSortingWriter_->enabled()) {
    sortPool = createSortPool(writerPool);
  }
  auto writerInfo = std::make_shared<HiveWriterInfo>(
      std::move(writerParameters),
      std::move(writerPool),
      std::move(sinkPool),
      std::move(sortPool));
  auto ioStats = std::make_unique<io::IoStatistics>();

  setMemoryReclaimers(writerInfo.get(), ioStats.get());

  auto formatWriter = createFormatWriter(writerInfo.get(), ioStats.get());
  auto* ioStatsPtr = ioStats.get();

  const bool canRotate = !isBucketed() && !bucketSortingWriter_->enabled();
  auto rotationWriter = std::make_unique<RotationWriter>(
      std::move(formatWriter),
      writerInfo,
      std::move(ioStats),
      maxTargetFileBytes_,
      canRotate,
      [this, writerInfo, ioStatsPtr]() {
        return createFormatWriter(writerInfo.get(), ioStatsPtr);
      });

  addThreadLocalRuntimeStat(
      fmt::format(
          "{}WriterCount",
          dwio::common::toString(insertTableHandle_->storageFormat())),
      RuntimeCounter(1));

  return rotationWriter;
}

std::unique_ptr<dwio::common::Writer> HiveDataSink::createFormatWriter(
    HiveWriterInfo* writerInfo,
    io::IoStatistics* ioStats) {
  VELOX_CHECK_NOT_NULL(writerInfo);
  VELOX_CHECK_NOT_NULL(ioStats);

  const auto& params = writerInfo->writerParameters;

  // Compute and store the new file names.
  writerInfo->currentWriteFileName = makeSequencedFileName(
      params.writeFileName(), writerInfo->fileSequenceNumber);
  writerInfo->currentTargetFileName = makeSequencedFileName(
      params.targetFileName(), writerInfo->fileSequenceNumber);

  const auto writePath =
      (fs::path(params.writeDirectory()) / writerInfo->currentWriteFileName)
          .string();

  auto options = createWriterOptions(writerInfo);

  // Prevents the memory allocation during the writer creation.
  memory::NonReclaimableSectionGuard nonReclaimableGuard(
      writerInfo->nonReclaimableSectionHolder.get());
  auto writer = writerFactory_->createWriter(
      createHiveFileSink(
          writePath,
          hiveConfig_,
          writerInfo->sinkPool.get(),
          ioStats,
          fileSystemStats_.get(),
          insertTableHandle_->storageParameters()),
      options);
  return bucketSortingWriter_->wrap(writerInfo, std::move(writer));
}

std::string HiveDataSink::getPartitionName(uint32_t partitionId) const {
  VELOX_CHECK_NOT_NULL(partitionIdGenerator_);

  return HivePartitionName::partitionName(
      partitionId,
      partitionIdGenerator_->partitionValues(),
      partitionKeyAsLowerCase_);
}

HiveWriterParameters HiveDataSink::getWriterParameters(
    const std::optional<std::string>& partition,
    std::optional<uint32_t> bucketId) const {
  auto [targetFileName, writeFileName] = getWriterFileNames(bucketId);

  return HiveWriterParameters{
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

HiveWriterParameters::UpdateMode HiveDataSink::getUpdateMode() const {
  if (insertTableHandle_->isExistingTable()) {
    if (insertTableHandle_->isPartitioned()) {
      const auto insertBehavior = hiveConfig_->insertExistingPartitionsBehavior(
          connectorQueryCtx_->sessionProperties());
      switch (insertBehavior) {
        case HiveConfig::InsertExistingPartitionsBehavior::kOverwrite:
          return HiveWriterParameters::UpdateMode::kOverwrite;
        case HiveConfig::InsertExistingPartitionsBehavior::kError:
          return HiveWriterParameters::UpdateMode::kNew;
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
      return HiveWriterParameters::UpdateMode::kAppend;
    }
  } else {
    return HiveWriterParameters::UpdateMode::kNew;
  }
}

std::unique_ptr<memory::MemoryReclaimer> HiveDataSink::WriterReclaimer::create(
    HiveDataSink* dataSink,
    HiveWriterInfo* writerInfo,
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
