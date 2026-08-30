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
#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
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
  // The V3 DELETE plan delivers the row-id as a synthesized
  // ROW<file_path VARCHAR, pos BIGINT, ...> column (getDeleteRowIdColumn),
  // and may prepend passthrough columns (e.g. a data/partition column), so the
  // input can be ROW<id, $row_id:ROW<...>> — not a bare 2-column page. Detect
  // the row-id by locating the ROW-typed column whose first two fields are
  // (VARCHAR file_path, BIGINT pos). Fall back to the legacy two flat columns
  // (file_path, pos) produced by IcebergMergeSink::makeDeleteBatch.
  const auto isRowIdStruct = [](const TypePtr& type) {
    return type->isRow() && type->asRow().size() >= kExpectedChannelCount &&
        type->asRow().childAt(kFilePathChannel)->isVarchar() &&
        type->asRow().childAt(kPositionChannel)->isBigint();
  };
  const auto numColumns = static_cast<int32_t>(inputType_->size());
  bool found = false;
  for (int32_t channel = 0; channel < numColumns; ++channel) {
    if (isRowIdStruct(inputType_->childAt(channel))) {
      rowIdAsStruct_ = true;
      rowIdChannel_ = channel;
      found = true;
      break;
    }
  }
  if (!found) {
    if (numColumns == kExpectedChannelCount &&
        inputType_->childAt(kFilePathChannel)->isVarchar() &&
        inputType_->childAt(kPositionChannel)->isBigint()) {
      rowIdAsStruct_ = false;
    } else {
      VELOX_USER_FAIL(
          "IcebergDeletionVectorSink expects a two-column (file_path, pos) "
          "input or a ROW<file_path, pos, ...> row-id column, got {}",
          inputType_->toString());
    }
  }
}

void IcebergDeletionVectorSink::appendData(RowVectorPtr input) {
  VELOX_USER_CHECK(!finished_, "appendData() called after finish()");
  VELOX_USER_CHECK(!aborted_, "appendData() called after abort()");
  if (input == nullptr || input->size() == 0) {
    return;
  }

  const auto numRows = input->size();
  const SelectivityVector allRows(numRows);

  if (rowIdAsStruct_) {
    // The row-id is a synthesized ROW<file_path, pos, ...> column (at
    // 'rowIdChannel_') that may be dictionary-encoded (the delete operator
    // selects deleted rows via a dictionary over the row-id). Decode the ROW
    // column to peel the outer encoding, then read file_path/pos from the base
    // ROW's fields through the decoded indices. Reading the base children
    // directly (ignoring the dictionary) mis-indexes and can read out of
    // bounds.
    DecodedVector decodedRowId(*input->childAt(rowIdChannel_), allRows);
    const auto* rowId = decodedRowId.base()->as<RowVector>();
    VELOX_USER_CHECK_NOT_NULL(rowId, "row-id column must be a ROW vector");
    VELOX_USER_CHECK_GE(
        rowId->childrenSize(),
        kExpectedChannelCount,
        "row-id ROW must have at least 2 fields (file_path, pos), got {}",
        rowId->childrenSize());
    const SelectivityVector baseRows(rowId->size());
    DecodedVector decodedFilePath(*rowId->childAt(kFilePathChannel), baseRows);
    DecodedVector decodedPosition(*rowId->childAt(kPositionChannel), baseRows);
    for (vector_size_t i = 0; i < numRows; ++i) {
      if (decodedRowId.isNullAt(i)) {
        continue;
      }
      const auto row = decodedRowId.index(i);
      VELOX_USER_CHECK(
          !decodedFilePath.isNullAt(row), "Null file_path in DELETE input row");
      VELOX_USER_CHECK(
          !decodedPosition.isNullAt(row), "Null pos in DELETE input row");
      const auto pathSlice = decodedFilePath.valueAt<StringView>(row);
      PerFileState& state =
          findOrCreatePerFile(std::string(pathSlice.data(), pathSlice.size()));
      state.writer.addDeletedPosition(decodedPosition.valueAt<int64_t>(row));
    }
    return;
  }

  VELOX_USER_CHECK_EQ(
      static_cast<int32_t>(input->childrenSize()),
      kExpectedChannelCount,
      "IcebergDeletionVectorSink expects 2-column input pages");
  DecodedVector decodedFilePath(*input->childAt(kFilePathChannel), allRows);
  DecodedVector decodedPosition(*input->childAt(kPositionChannel), allRows);
  for (vector_size_t i = 0; i < numRows; ++i) {
    VELOX_USER_CHECK(
        !decodedFilePath.isNullAt(i), "Null file_path in DELETE input row");
    VELOX_USER_CHECK(
        !decodedPosition.isNullAt(i), "Null pos in DELETE input row");
    const auto pathSlice = decodedFilePath.valueAt<StringView>(i);
    PerFileState& state =
        findOrCreatePerFile(std::string(pathSlice.data(), pathSlice.size()));
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
  PerFileState& state = perFile_.back().second;
  seedFromExistingDeletionVector(state, path);
  return state;
}

void IcebergDeletionVectorSink::seedFromExistingDeletionVector(
    PerFileState& state,
    const std::string& dataFile) {
  const auto& existing = insertTableHandle_->existingDeletionVectors();
  const auto it = existing.find(dataFile);
  if (it == existing.end()) {
    return;
  }
  const auto& descriptor = it->second;

  // Reconstruct the existing DV as an IcebergDeleteFile and read its positions
  // through the same DeletionVectorReader used on the read path. The reader
  // ignores the memory pool argument, so nullptr is safe; the connector config
  // resolves the (possibly warm-storage / Manifold) filesystem for the Puffin.
  const IcebergDeleteFile dvFile(
      FileContent::kDeletionVector,
      descriptor.puffinPath,
      dwio::common::FileFormat::PUFFIN,
      static_cast<uint64_t>(descriptor.recordCount),
      static_cast<uint64_t>(descriptor.fileSizeInBytes),
      /*equalityFieldIds=*/{},
      /*lowerBounds=*/{},
      /*upperBounds=*/{},
      /*dataSequenceNumber=*/0,
      descriptor.contentOffset,
      descriptor.contentLength,
      dataFile);
  DeletionVectorReader reader(
      dvFile, /*splitOffset=*/0, /*pool=*/nullptr, hiveConfig_->config());
  state.writer.addDeletedPositions(reader.deletedPositions());
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

    // Iceberg V3 treats the DV blob's cardinality as authoritative, so it must
    // match the de-duplicated bitmap that serialize() emits — not the raw
    // insertion count (seeding an existing DV plus overlapping new deletes can
    // introduce duplicates).
    const size_t cardinality = entry.second.writer.numDistinctPositions();

    const auto [offset, length] = writePuffinFile(
        *sink, *pool, blob, entry.first, static_cast<int64_t>(cardinality));
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
        cardinality,
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
