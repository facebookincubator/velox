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

#include <filesystem>

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <folly/json.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Position-delete input rows always carry exactly two columns:
// file_path: VARCHAR, pos: BIGINT.
constexpr int32_t kFilePathChannel = 0;
constexpr int32_t kPositionChannel = 1;
constexpr int32_t kExpectedChannelCount = 2;

// Lookup or lazily insert a per-file state entry, preserving the original
// insertion order so commit messages come back deterministically.
IcebergDeletionVectorSink::PerFileState* findOrCreate(
    std::vector<
        std::pair<std::string, IcebergDeletionVectorSink::PerFileState>>&
        perFile,
    const std::string& path) {
  for (auto& entry : perFile) {
    if (entry.first == path) {
      return &entry.second;
    }
  }
  perFile.emplace_back(path, IcebergDeletionVectorSink::PerFileState{});
  return &perFile.back().second;
}

} // namespace

IcebergDeletionVectorSink::IcebergDeletionVectorSink(
    RowTypePtr inputType,
    IcebergInsertTableHandlePtr insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy)
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      commitStrategy_(commitStrategy) {
  VELOX_USER_CHECK_NOT_NULL(
      insertTableHandle_,
      "IcebergDeletionVectorSink requires a non-null insert table handle.");
  VELOX_USER_CHECK_NOT_NULL(
      inputType_, "IcebergDeletionVectorSink requires a non-null input type.");
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
    PerFileState* state = findOrCreate(perFile_, path);
    state->writer.addDeletedPosition(decodedPosition.valueAt<int64_t>(i));
  }
}

bool IcebergDeletionVectorSink::finish() {
  // Single-shot finish: serialize each per-file roaring bitmap, write a Puffin
  // file, and emit a commit message. No yielding mid-finish today; can be
  // extended once we expose intermediate progress.
  if (finished_) {
    return true;
  }
  finished_ = true;
  for (auto& entry : perFile_) {
    if (entry.second.writer.numPositions() == 0) {
      continue;
    }
    const std::string puffinPath = puffinPathFor(entry.first);
    const std::string blob = entry.second.writer.serialize();
    const auto [offset, length] =
        writePuffinFile(puffinPath, blob, entry.first);

    const auto fileSize = std::filesystem::file_size(puffinPath);
    numWrittenBytes_ += fileSize;
    numWrittenFiles_ += 1;

    folly::dynamic msg = folly::dynamic::object;
    msg["path"] = puffinPath;
    msg["fileSizeInBytes"] = static_cast<int64_t>(fileSize);
    msg["metrics"] = folly::dynamic::object(
        "recordCount",
        static_cast<int64_t>(entry.second.writer.numPositions()));
    // Use the real partition spec id when the target table is partitioned;
    // unpartitioned tables (or sinks constructed without a spec) emit 0
    // (the default unpartitioned spec id).
    msg["partitionSpecJson"] = static_cast<int64_t>(
        insertTableHandle_->partitionSpec()
            ? insertTableHandle_->partitionSpec()->specId
            : 0);
    msg["fileFormat"] = "PUFFIN";
    msg["referencedDataFile"] = entry.first;
    msg["content"] = "POSITION_DELETES";
    msg["contentOffset"] = static_cast<int64_t>(offset);
    msg["contentSizeInBytes"] = static_cast<int64_t>(length);
    commitMessages_.push_back(folly::toJson(msg));
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
  std::string parentDir;
  if (dataFile.find('/') != std::string::npos) {
    parentDir = std::filesystem::path(dataFile).parent_path().string();
  } else {
    parentDir = insertTableHandle_->locationHandle()->targetPath();
  }
  const std::string uuid =
      boost::uuids::to_string(boost::uuids::random_generator()());
  return parentDir + "/dv-" + uuid + ".puffin";
}

} // namespace facebook::velox::connector::hive::iceberg
