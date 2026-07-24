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

#include "velox/connectors/hive/paimon/PaimonDataSink.h"

#include "velox/common/base/Fs.h"
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/connectors/hive/HivePartitionName.h"

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace facebook::velox::connector::hive::paimon {
namespace {

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

// Appends a sequence number to a filename for file rotation.
// "data-xxx.orc" with seq 0 -> "data-xxx.orc"
// "data-xxx.orc" with seq 2 -> "data-xxx_2.orc"
std::string makeSequencedFileName(
    const std::string& filename,
    uint32_t sequenceNumber) {
  // The first file uses the base name without a suffix.
  if (sequenceNumber == 0) {
    return filename;
  }
  // Insert the sequence number before the file extension.
  const auto dotPos = filename.rfind('.');
  if (dotPos == std::string::npos) {
    return fmt::format("{}_{}", filename, sequenceNumber);
  }
  return fmt::format(
      "{}_{}{}",
      filename.substr(0, dotPos),
      sequenceNumber,
      filename.substr(dotPos));
}

int32_t getBucketCount(const HiveBucketProperty* bucketProperty) {
  return bucketProperty == nullptr ? 0 : bucketProperty->bucketCount();
}

std::unique_ptr<core::PartitionFunction> createBucketFunction(
    const HiveBucketProperty& bucketProperty,
    const RowTypePtr& inputType) {
  const auto& bucketedBy = bucketProperty.bucketedBy();
  const auto& bucketedTypes = bucketProperty.bucketedTypes();
  VELOX_CHECK_EQ(
      bucketedBy.size(),
      bucketedTypes.size(),
      "bucketedBy and bucketedTypes must have the same size");
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

std::unique_ptr<PartitionIdGenerator> createPartitionIdGenerator(
    const RowTypePtr& inputType,
    const std::shared_ptr<const HiveInsertTableHandle>& insertTableHandle,
    uint32_t maxPartitions,
    memory::MemoryPool* pool) {
  const auto& partitionChannels = insertTableHandle->partitionChannels();
  if (partitionChannels.empty()) {
    return nullptr;
  }
  return std::make_unique<PartitionIdGenerator>(
      inputType, partitionChannels, maxPartitions, pool);
}

std::string makeBucketDirectory(std::optional<uint32_t> bucketId) {
  return fmt::format("bucket-{}", bucketId.value_or(0));
}

// Builds a directory path: {base}/{partition}/{bucket-N}.
std::string makePaimonDirectory(
    const std::string& baseDir,
    const std::optional<std::string>& partition,
    const std::string& bucketDir) {
  fs::path path(baseDir);
  if (partition.has_value()) {
    path /= partition.value();
  }
  path /= bucketDir;
  return path.string();
}

// Returns storage format file extension.
std::string formatExtension(dwio::common::FileFormat format) {
  switch (format) {
    case dwio::common::FileFormat::PARQUET:
      return ".parquet";
    case dwio::common::FileFormat::DWRF:
      [[fallthrough]];
    case dwio::common::FileFormat::ORC:
      return ".orc";
    default:
      VELOX_UNSUPPORTED(
          "Unsupported file format: {}", dwio::common::toString(format));
  }
}

// Default max open writers for Paimon (matches Hive's default of 128).
constexpr uint32_t kDefaultMaxOpenWriters = 128;

} // namespace

PaimonDataSink::PaimonDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const HiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const PaimonConfig>& paimonConfig)
    : FileDataSink(
          inputType,
          connectorQueryCtx,
          commitStrategy,
          insertTableHandle->storageFormat(),
          kDefaultMaxOpenWriters,
          insertTableHandle->partitionChannels(),
          insertTableHandle->nonPartitionChannels(),
          getBucketCount(insertTableHandle->bucketProperty()),
          getBucketCount(insertTableHandle->bucketProperty()) > 0
              ? createBucketFunction(
                    *insertTableHandle->bucketProperty(),
                    inputType)
              : nullptr,
          createPartitionIdGenerator(
              inputType,
              insertTableHandle,
              kDefaultMaxOpenWriters,
              connectorQueryCtx->memoryPool()),
          dwio::common::getWriterFactory(insertTableHandle->storageFormat()),
          /*maxTargetFileBytes=*/0,
          /*partitionKeyAsLowerCase=*/true,
          connectorQueryCtx->spillConfig(),
          /*sortWriterFinishTimeSliceLimitMs=*/0),
      insertTableHandle_(std::move(insertTableHandle)),
      paimonConfig_(paimonConfig) {
  VELOX_USER_CHECK(
      (commitStrategy_ == CommitStrategy::kNoCommit) ||
          (commitStrategy_ == CommitStrategy::kTaskCommit),
      "Unsupported commit strategy: {}",
      CommitStrategyName::toName(commitStrategy_));
}

std::vector<std::string> PaimonDataSink::commitMessage() const {
  std::vector<std::string> messages;
  messages.reserve(writerInfo_.size());

  for (int i = 0; i < writerInfo_.size(); ++i) {
    const auto& info = writerInfo_.at(i);
    VELOX_CHECK_NOT_NULL(info);

    // Build per-file write info array.
    folly::dynamic fileWriteInfos = folly::dynamic::array;
    for (const auto& fileInfo : info->writtenFiles) {
      fileWriteInfos.push_back(
          folly::dynamic::object(
              CommitMessage::kWriteFileName, fileInfo.writeFileName)(
              CommitMessage::kTargetFileName, fileInfo.targetFileName)(
              CommitMessage::kFileSize, fileInfo.fileSize)(
              CommitMessage::kFileRowCount, fileInfo.numRows));
    }

    // Extract partition values from the partition name (key=value format).
    // The partition name is stored in writerParameters.
    folly::dynamic partitionValues = folly::dynamic::object;
    // Note: partition values are not directly available from the writer
    // parameters as a map. The coordinator tracks partition values separately.
    // For now, we pass the partition name string.

    // Extract bucket number from writer ID. We reconstruct it by looking at
    // the writer index map.
    int32_t bucketNumber = -1;
    for (const auto& [writerId, index] : writerIndexMap_) {
      if (index == i) {
        bucketNumber = writerId.bucketId.value_or(-1);
        break;
      }
    }

    auto commitJson = folly::toJson(
        folly::dynamic::object(
            CommitMessage::kPartitionValues, std::move(partitionValues))(
            CommitMessage::kBucketNumber, bucketNumber)(
            CommitMessage::kWritePath, info->writerParameters.writeDirectory())(
            CommitMessage::kTargetPath,
            info->writerParameters.targetDirectory())(
            CommitMessage::kFileWriteInfos, std::move(fileWriteInfos))(
            CommitMessage::kTotalRowCount, info->numWrittenRows)(
            CommitMessage::kInMemoryDataSizeInBytes, info->inputSizeInBytes)(
            CommitMessage::kOnDiskDataSizeInBytes,
            ioStats_.at(i)->rawBytesWritten()));

    messages.push_back(std::move(commitJson));
  }
  return messages;
}

void PaimonDataSink::computePartitionAndBucketIds(const RowVectorPtr& input) {
  VELOX_CHECK(isPartitioned() || isBucketed());

  if (isPartitioned()) {
    partitionIdGenerator_->run(input, partitionIds_);
  }

  if (isBucketed()) {
    bucketFunction_->partition(*input, bucketIds_);
  }
}

std::unique_ptr<dwio::common::Writer> PaimonDataSink::createWriterForIndex(
    size_t writerIndex) {
  VELOX_CHECK_LT(writerIndex, writerInfo_.size());
  VELOX_CHECK_LT(writerIndex, ioStats_.size());

  auto& info = writerInfo_[writerIndex];
  const auto& params = info->writerParameters;

  // Compute and store the new file names (handles file rotation).
  info->currentWriteFileName =
      makeSequencedFileName(params.writeFileName(), info->fileSequenceNumber);
  info->currentTargetFileName =
      makeSequencedFileName(params.targetFileName(), info->fileSequenceNumber);

  const auto writePath =
      (fs::path(params.writeDirectory()) / info->currentWriteFileName).string();

  auto options = createWriterOptions(writerIndex);

  auto writer = writerFactory_->createWriter(
      dwio::common::FileSink::create(
          writePath,
          {
              .bufferWrite = false,
              .connectorProperties = paimonConfig_->config(),
              .pool = info->sinkPool.get(),
              .metricLogger = dwio::common::MetricsLog::voidLog(),
              .stats = ioStats_[writerIndex].get(),
              .fileSystemStats = fileSystemStats_.get(),
          }),
      options);
  return writer;
}

std::shared_ptr<dwio::common::WriterOptions>
PaimonDataSink::createWriterOptions() const {
  return createWriterOptions(writerInfo_.size() - 1);
}

std::shared_ptr<dwio::common::WriterOptions>
PaimonDataSink::createWriterOptions(size_t writerIndex) const {
  VELOX_CHECK_LT(writerIndex, writerInfo_.size());

  auto options = insertTableHandle_->writerOptions();
  if (!options) {
    options = writerFactory_->createWriterOptions();
  }

  if (options->schema == nullptr) {
    options->schema = getNonPartitionTypes(dataChannels_, inputType_);
  }

  if (options->memoryPool == nullptr) {
    options->memoryPool = writerInfo_[writerIndex]->writerPool.get();
  }

  if (!options->compressionKind) {
    options->compressionKind = insertTableHandle_->compressionKind();
  }

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
  return options;
}

std::string PaimonDataSink::getPartitionName(uint32_t partitionId) const {
  VELOX_CHECK_NOT_NULL(partitionIdGenerator_);
  return HivePartitionName::partitionName(
      partitionId,
      partitionIdGenerator_->partitionValues(),
      partitionKeyAsLowerCase_);
}

WriterParameters PaimonDataSink::getWriterParameters(
    const std::optional<std::string>& partition,
    std::optional<uint32_t> bucketId) const {
  // Paimon file naming: data-{uuid}.{format}
  auto uuid = makeUuid();
  auto targetFileName =
      fmt::format("data-{}{}", uuid, formatExtension(storageFormat_));
  auto writeFileName = isCommitRequired()
      ? fmt::format(".tmp.paimon.{}", targetFileName)
      : targetFileName;

  // Paimon directory layout: {root}/{partition=value}/bucket-{N}/
  auto bucketDir = makeBucketDirectory(bucketId);

  return WriterParameters{
      WriterParameters::UpdateMode::kNew,
      partition,
      targetFileName,
      makePaimonDirectory(
          insertTableHandle_->locationHandle()->targetPath(),
          partition,
          bucketDir),
      writeFileName,
      makePaimonDirectory(
          insertTableHandle_->locationHandle()->writePath(),
          partition,
          bucketDir)};
}

} // namespace facebook::velox::connector::hive::paimon
