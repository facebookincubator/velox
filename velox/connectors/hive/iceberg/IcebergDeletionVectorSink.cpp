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

#include "velox/connectors/hive/iceberg/IcebergDeletionVectorSink.h"

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <folly/json.h>

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/dwio/common/FileSink.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Position-delete input rows always carry exactly two columns:
// file_path: VARCHAR, pos: BIGINT.
constexpr int32_t kFilePathChannel = 0;
constexpr int32_t kPositionChannel = 1;
constexpr int32_t kExpectedChannelCount = 2;

// Builds the JSON commit fragment the coordinator consumes for one written
// deletion-vector Puffin file.
std::string buildDeletionVectorCommitMessage(
    const std::string& puffinPath,
    uint64_t fileSize,
    uint64_t recordCount,
    int64_t partitionSpecId,
    const std::string& referencedDataFile,
    uint64_t contentOffset,
    uint64_t contentLength) {
  folly::dynamic msg = folly::dynamic::object;
  msg["path"] = puffinPath;
  msg["fileSizeInBytes"] = static_cast<int64_t>(fileSize);
  msg["metrics"] =
      folly::dynamic::object("recordCount", static_cast<int64_t>(recordCount));
  msg["partitionSpecJson"] = partitionSpecId;
  msg["fileFormat"] = "PUFFIN";
  msg["referencedDataFile"] = referencedDataFile;
  msg["content"] = "POSITION_DELETES";
  msg["contentOffset"] = static_cast<int64_t>(contentOffset);
  msg["contentSizeInBytes"] = static_cast<int64_t>(contentLength);
  return folly::toJson(msg);
}

} // namespace

IcebergDeletionVectorSink::IcebergDeletionVectorSink(
    RowTypePtr inputType,
    IcebergInsertTableHandlePtr insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    std::shared_ptr<const HiveConfig> hiveConfig)
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      commitStrategy_(commitStrategy),
      hiveConfig_(std::move(hiveConfig)) {
  VELOX_USER_CHECK_NOT_NULL(
      insertTableHandle_,
      "IcebergDeletionVectorSink requires a non-null insert table handle.");
  VELOX_USER_CHECK_NOT_NULL(
      inputType_, "IcebergDeletionVectorSink requires a non-null input type.");
  VELOX_USER_CHECK_NOT_NULL(
      hiveConfig_,
      "IcebergDeletionVectorSink requires a non-null hive config.");
  VELOX_USER_CHECK_EQ(
      static_cast<int32_t>(inputType_->size()),
      kExpectedChannelCount,
      "IcebergDeletionVectorSink expects a 2-column input (file_path, pos)");
}

void IcebergDeletionVectorSink::appendData(RowVectorPtr input) {
  VELOX_USER_CHECK(!finished_, "appendData() called after finish()");
  VELOX_USER_CHECK(!aborted_, "appendData() called after abort()");
  if (input == nullptr || input->size() == 0) {
    return;
  }
  VELOX_USER_CHECK_EQ(
      static_cast<int32_t>(input->childrenSize()),
      kExpectedChannelCount,
      "IcebergDeletionVectorSink expects 2-column input pages");

  const auto* filePathVector = input->childAt(kFilePathChannel)->loadedVector();
  const auto* positionVector = input->childAt(kPositionChannel)->loadedVector();

  DecodedVector decodedFilePath(
      *filePathVector, SelectivityVector(input->size()));
  DecodedVector decodedPosition(
      *positionVector, SelectivityVector(input->size()));

  for (vector_size_t i = 0; i < input->size(); ++i) {
    VELOX_USER_CHECK(
        !decodedFilePath.isNullAt(i), "Null file_path in DELETE input row");
    VELOX_USER_CHECK(
        !decodedPosition.isNullAt(i), "Null pos in DELETE input row");
    const auto pathSlice = decodedFilePath.valueAt<StringView>(i);
    const std::string path(pathSlice.data(), pathSlice.size());
    PerFileState& state = findOrCreatePerFile(path);
    state.writer.addDeletedPosition(decodedPosition.valueAt<int64_t>(i));
  }
}

IcebergDeletionVectorSink::PerFileState&
IcebergDeletionVectorSink::findOrCreatePerFile(const std::string& path) {
  if (auto it = perFileIndex_.find(path); it != perFileIndex_.end()) {
    return perFile_[it->second].second;
  }
  const size_t index = perFile_.size();
  perFile_.emplace_back(path, PerFileState{});
  perFileIndex_.emplace(path, index);
  return perFile_.back().second;
}

bool IcebergDeletionVectorSink::finish() {
  // Single-shot finish: serialize each per-file roaring bitmap, write a Puffin
  // file, and emit a commit message. No yielding mid-finish today; can be
  // extended once we expose intermediate progress.
  if (finished_) {
    return true;
  }
  finished_ = true;
  // Memory pool used to stage puffin bytes for the FileSink::write call.
  // Use the operator memory pool from the connector context. In test paths
  // that pass nullptr for connectorQueryCtx_, fall back to a leaf pool
  // derived from the connector handle's pool. Tests without a connector
  // context must supply a non-null connectorQueryCtx_ for this path; the
  // existing test fixture is updated accordingly.
  VELOX_CHECK_NOT_NULL(
      connectorQueryCtx_,
      "IcebergDeletionVectorSink::finish requires a connector query ctx for "
      "FileSink buffer allocation.");
  auto* pool = connectorQueryCtx_->memoryPool();
  for (auto& entry : perFile_) {
    if (entry.second.writer.numPositions() == 0) {
      continue;
    }
    const std::string puffinPath = puffinPathFor(entry.first);
    const std::string blob = entry.second.writer.serialize();

    // Build a per-puffin FileSink. The Options mirror the data-file sink
    // initialised by createHiveFileSink in HiveDataSink.cpp so that the
    // puffin file lands through the same registered filesystem dispatch
    // (local, warm storage, S3, ...) as the data file. Stats pointers are
    // null because the deletion vector sink does not surface per-puffin
    // IO statistics; cumulative numWrittenBytes_/numWrittenFiles_ are
    // tracked locally below.
    dwio::common::FileSink::Options sinkOptions{
        .bufferWrite = false,
        .connectorProperties = hiveConfig_->config(),
        .fileCreateConfig = hiveConfig_->writeFileCreateConfig(),
        .pool = pool,
        .metricLogger = dwio::common::MetricsLog::voidLog(),
        .stats = nullptr,
        .fileSystemStats = nullptr,
        .storageParameters = insertTableHandle_->storageParameters(),
    };
    auto sink = dwio::common::FileSink::create(puffinPath, sinkOptions);
    VELOX_CHECK_NOT_NULL(
        sink, "Failed to create file sink for Puffin file: {}", puffinPath);

    const auto [offset, length] =
        writePuffinFile(*sink, *pool, blob, entry.first);
    const uint64_t fileSize = sink->size();
    sink->close();

    numWrittenBytes_ += fileSize;
    numWrittenFiles_ += 1;

    // Use the real partition spec id when the target table is partitioned;
    // unpartitioned tables (or sinks constructed without a spec) emit 0
    // (the default unpartitioned spec id).
    const int64_t partitionSpecId = insertTableHandle_->partitionSpec()
        ? insertTableHandle_->partitionSpec()->specId
        : 0;
    commitMessages_.push_back(buildDeletionVectorCommitMessage(
        puffinPath,
        fileSize,
        entry.second.writer.numPositions(),
        partitionSpecId,
        entry.first,
        offset,
        length));
  }
  return true;
}

std::vector<std::string> IcebergDeletionVectorSink::close() {
  if (!finished_) {
    (void)finish();
  }
  return commitMessages_;
}

void IcebergDeletionVectorSink::abort() {
  if (finished_ || aborted_) {
    return;
  }
  aborted_ = true;
  perFile_.clear();
  perFileIndex_.clear();
}

DataSink::Stats IcebergDeletionVectorSink::stats() const {
  Stats stats;
  stats.numWrittenBytes = numWrittenBytes_;
  stats.numWrittenFiles = numWrittenFiles_;
  return stats;
}

std::string IcebergDeletionVectorSink::puffinPathFor(
    const std::string& dataFile) const {
  // Co-locate the puffin file with the referenced data file so that a
  // partitioned V3 table's puffin blobs land in the same partition
  // sub-directory as their data files. Falling back to the location handle's
  // target path keeps unpartitioned tables and tests (which pass a synthetic
  // dataFile without a parent directory) functional.
  //
  // Derive the parent directory via string ops rather than
  // std::filesystem::path, which is not URI-safe: for scheme-based paths like
  // "ws://bucket/dir/file" it collapses "//" to "/" and corrupts the
  // scheme/authority prefix.
  const auto lastSlash = dataFile.rfind('/');
  std::string parentDir;
  if (lastSlash == std::string::npos) {
    parentDir = insertTableHandle_->locationHandle()->targetPath();
  } else {
    parentDir = dataFile.substr(0, lastSlash);
  }
  const std::string uuid =
      boost::uuids::to_string(boost::uuids::random_generator()());
  return parentDir + "/dv-" + uuid + ".puffin";
}

} // namespace facebook::velox::connector::hive::iceberg
