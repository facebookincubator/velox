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
#include "velox/connectors/hive/paimon/PaimonSplitReader.h"

#include <limits>

namespace facebook::velox::connector::hive::paimon {

PaimonSplitReader::PaimonSplitReader(
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : FileSplitReader(
          fileSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          readerOutputType,
          ioStatistics,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      paimonSplit_(std::move(paimonSplit)) {
  // Validate all data files upfront.
  for (const auto& dataFile : paimonSplit_->dataFiles()) {
    VELOX_CHECK(
        !dataFile.deletionFile.has_value(),
        "Paimon deletion vector reading is not yet implemented. "
        "File '{}' has a deletion vector with {} deleted rows.",
        dataFile.path,
        dataFile.deletionFile.has_value() ? dataFile.deletionFile->cardinality
                                          : 0);
    VELOX_CHECK_EQ(
        dataFile.type,
        PaimonDataFile::Type::kData,
        "Paimon changelog file reading is not yet supported. "
        "File '{}' has type {}.",
        dataFile.path,
        dataFile.type);
  }

  // Build FileConnectorSplits for all data files so we can switch between
  // them during multi-file iteration.
  const auto& dataFiles = paimonSplit_->dataFiles();
  fileSplits_.reserve(dataFiles.size());
  for (const auto& dataFile : dataFiles) {
    fileSplits_.emplace_back(makeFileConnectorSplit(dataFile));
  }
}

void PaimonSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& /*fileReadOps*/) {
  // Save for re-use when advancing to subsequent files.
  metadataFilter_ = std::move(metadataFilter);
  runtimeStats_ = &runtimeStats;
}

uint64_t PaimonSplitReader::next(uint64_t size, VectorPtr& output) {
  while (ensureFileSplitReader()) {
    // When deletion vectors are implemented, apply the deletion bitmap
    // here via Mutation.deletedRows, following the same pattern as
    // IcebergSplitReader::next().
    const auto rowsRead = FileSplitReader::next(size, output);
    if (rowsRead > 0) {
      return rowsRead;
    }
    // Current file exhausted, try the next one.
    finishFileSplitReader();
  }
  return 0;
}

bool PaimonSplitReader::ensureFileSplitReader() {
  if (baseReader_ != nullptr) {
    VELOX_CHECK_NOT_NULL(baseRowReader_);
    return true;
  }
  VELOX_CHECK_NULL(baseRowReader_);

  while (currentFileIndex_ < fileSplits_.size()) {
    // Point to the current file.
    fileSplit_ = fileSplits_[currentFileIndex_];

    // Initialize the base reader for this file.
    FileSplitReader::prepareSplit(metadataFilter_, *runtimeStats_);
    if (!emptySplit_) {
      return true;
    }
    ++currentFileIndex_;
  }
  return false; // All files exhausted.
}

void PaimonSplitReader::finishFileSplitReader() {
  VELOX_CHECK_LT(currentFileIndex_, fileSplits_.size());
  VELOX_CHECK_NOT_NULL(baseReader_);
  VELOX_CHECK_NOT_NULL(baseRowReader_);
  ++currentFileIndex_;
  baseReader_ = nullptr;
  baseRowReader_ = nullptr;
  emptySplit_ = false;
}

std::shared_ptr<FileConnectorSplit> PaimonSplitReader::makeFileConnectorSplit(
    const PaimonDataFile& dataFile) const {
  return std::make_shared<FileConnectorSplit>(
      paimonSplit_->connectorId,
      dataFile.path,
      paimonSplit_->fileFormat(),
      /*_start=*/0,
      /*_length=*/std::numeric_limits<uint64_t>::max(),
      /*splitWeight=*/0,
      /*cacheable=*/true,
      /*_properties=*/std::nullopt,
      paimonSplit_->partitionKeys());
}

} // namespace facebook::velox::connector::hive::paimon
