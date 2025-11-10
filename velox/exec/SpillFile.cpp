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

#include "velox/exec/SpillFile.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/serializers/SerializedPageFile.h"

namespace facebook::velox::exec {
namespace {
// Spilling currently uses the default PrestoSerializer which by default
// serializes timestamp with millisecond precision to maintain compatibility
// with presto. Since velox's native timestamp implementation supports
// nanosecond precision, we use this serde option to ensure the serializer
// preserves precision.
static const bool kDefaultUseLosslessTimestamp = true;
} // namespace

SpillWriter::SpillWriter(
    const RowTypePtr& type,
    const std::vector<SpillSortKey>& sortingKeys,
    common::CompressionKind compressionKind,
    const std::string& pathPrefix,
    uint64_t targetFileSize,
    uint64_t writeBufferSize,
    const std::string& fileCreateConfig,
    const common::UpdateAndCheckSpillLimitCB& updateAndCheckSpillLimitCb,
    memory::MemoryPool* pool,
    folly::Synchronized<common::SpillStats>* stats)
    : serializer::SerializedPageFileWriter(
          pathPrefix,
          targetFileSize,
          writeBufferSize,
          fileCreateConfig,
          std::make_unique<
              serializer::presto::PrestoVectorSerde::PrestoOptions>(
              kDefaultUseLosslessTimestamp,
              compressionKind,
              0.8,
              /*_nullsFirst=*/true),
          getNamedVectorSerde(VectorSerde::Kind::kPresto),
          pool),
      type_(type),
      sortingKeys_(sortingKeys),
      stats_(stats),
      updateAndCheckLimitCb_(updateAndCheckSpillLimitCb) {}

void SpillWriter::updateAppendStats(
    uint64_t numRows,
    uint64_t serializationTimeNs) {
  auto statsLocked = stats_->wlock();
  statsLocked->spilledRows += numRows;
  statsLocked->spillSerializationTimeNanos += serializationTimeNs;
  common::updateGlobalSpillAppendStats(numRows, serializationTimeNs);
}

void SpillWriter::updateWriteStats(
    uint64_t spilledBytes,
    uint64_t flushTimeNs,
    uint64_t fileWriteTimeNs) {
  auto statsLocked = stats_->wlock();
  statsLocked->spilledBytes += spilledBytes;
  statsLocked->spillFlushTimeNanos += flushTimeNs;
  statsLocked->spillWriteTimeNanos += fileWriteTimeNs;
  ++statsLocked->spillWrites;
  common::updateGlobalSpillWriteStats(
      spilledBytes, flushTimeNs, fileWriteTimeNs);
  updateAndCheckLimitCb_(spilledBytes);
}

void SpillWriter::updateFileStats(
    const serializer::SerializedPageFile::FileInfo& file) {
  ++stats_->wlock()->spilledFiles;
  addThreadLocalRuntimeStat(
      "spillFileSize", RuntimeCounter(file.size, RuntimeCounter::Unit::kBytes));
  common::incrementGlobalSpilledFiles();
}

SpillFiles SpillWriter::finish() {
  const auto serializedPageFiles =
      serializer::SerializedPageFileWriter::finish();
  SpillFiles spillFiles;
  spillFiles.reserve(serializedPageFiles.size());
  for (const auto& fileInfo : serializedPageFiles) {
    spillFiles.push_back(
        SpillFileInfo{
            .id = fileInfo.id,
            .type = type_,
            .path = fileInfo.path,
            .size = fileInfo.size,
            .sortingKeys = sortingKeys_,
            .compressionKind = serdeOptions_->compressionKind});
  }
  return spillFiles;
}

std::vector<std::string> SpillWriter::testingSpilledFilePaths() const {
  checkNotFinished();

  std::vector<std::string> spilledFilePaths;
  spilledFilePaths.reserve(
      finishedFiles_.size() + (currentFile_ != nullptr ? 1 : 0));
  for (auto& file : finishedFiles_) {
    spilledFilePaths.push_back(file.path);
  }
  if (currentFile_ != nullptr) {
    spilledFilePaths.push_back(currentFile_->path());
  }
  return spilledFilePaths;
}

std::vector<uint32_t> SpillWriter::testingSpilledFileIds() const {
  checkNotFinished();

  std::vector<uint32_t> fileIds;
  for (auto& file : finishedFiles_) {
    fileIds.push_back(file.id);
  }
  if (currentFile_ != nullptr) {
    fileIds.push_back(currentFile_->id());
  }
  return fileIds;
}

std::unique_ptr<SpillReadFile> SpillReadFile::create(
    const SpillFileInfo& fileInfo,
    uint64_t bufferSize,
    memory::MemoryPool* pool,
    folly::Synchronized<common::SpillStats>* stats) {
  return std::unique_ptr<SpillReadFile>(new SpillReadFile(
      fileInfo.id,
      fileInfo.path,
      fileInfo.size,
      bufferSize,
      fileInfo.type,
      fileInfo.sortingKeys,
      fileInfo.compressionKind,
      pool,
      stats));
}

SpillReadFile::SpillReadFile(
    uint32_t id,
    const std::string& path,
    uint64_t size,
    uint64_t bufferSize,
    const RowTypePtr& type,
    const std::vector<SpillSortKey>& sortingKeys,
    common::CompressionKind compressionKind,
    memory::MemoryPool* pool,
    folly::Synchronized<common::SpillStats>* stats)
    : serializer::SerializedPageFileReader(
          path,
          bufferSize,
          type,
          getNamedVectorSerde(VectorSerde::Kind::kPresto),
          std::make_unique<
              serializer::presto::PrestoVectorSerde::PrestoOptions>(
              kDefaultUseLosslessTimestamp,
              compressionKind,
              0.8,
              /*_nullsFirst=*/true),
          pool),
      id_(id),
      path_(path),
      size_(size),
      sortingKeys_(sortingKeys),
      stats_(stats) {}

void SpillReadFile::updateFinalStats() {
  VELOX_CHECK(input_->atEnd());
  const auto readStats = input_->stats();
  common::updateGlobalSpillReadStats(
      readStats.numReads, readStats.readBytes, readStats.readTimeNs);
  auto lockedSpillStats = stats_->wlock();
  lockedSpillStats->spillReads += readStats.numReads;
  lockedSpillStats->spillReadTimeNanos += readStats.readTimeNs;
  lockedSpillStats->spillReadBytes += readStats.readBytes;
};

void SpillReadFile::updateSerializationTimeStats(uint64_t timeNs) {
  stats_->wlock()->spillDeserializationTimeNanos += timeNs;
  common::updateGlobalSpillDeserializationTimeNs(timeNs);
};

} // namespace facebook::velox::exec
