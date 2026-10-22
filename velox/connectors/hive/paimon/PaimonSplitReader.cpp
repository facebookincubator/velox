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

#include "velox/common/Casts.h"
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

namespace facebook::velox::connector::hive::paimon {

PaimonSplitReader::PaimonSplitReader(
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
    const std::shared_ptr<const PaimonTableHandle>& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : FileSplitReader(
          fileSplit,
          tableHandle,
          &ownedPartitionKeys_,
          connectorQueryCtx,
          fileConfig,
          readerOutputType,
          dataIoStats,
          metadataIoStats,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      paimonTableHandle_(checkedNotNull(tableHandle)),
      ownedPartitionKeys_(*checkedNotNull(partitionKeys)),
      paimonSplit_(std::move(paimonSplit)) {
  VELOX_CHECK_NOT_NULL(paimonSplit_);

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
    dwio::common::RuntimeStats& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& /*fileReadOps*/) {
  // Save for re-use when advancing to subsequent files.
  metadataFilter_ = std::move(metadataFilter);
  runtimeStats_ = &runtimeStats;
}

uint64_t PaimonSplitReader::next(uint64_t size, VectorPtr& output) {
  while (ensureFileSplitReader()) {
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

    createReader();
    VELOX_USER_CHECK_NOT_NULL(baseReader_, "Missing Paimon file reader");
    paimonTableHandle_->validateFileSchema(baseReader_->rowType());
    auto rowType = getAdaptedRowType();
    if (!checkIfSplitIsEmpty(*runtimeStats_)) {
      createRowReader(metadataFilter_, std::move(rowType), std::nullopt);
    }
    if (!emptySplit_) {
      return true;
    }
    baseReader_.reset();
    emptySplit_ = false;
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

void PaimonSplitReader::configureBaseReaderOptions() {
  FileSplitReader::configureBaseReaderOptions();
  // Preserve footer names so a contradictory file cannot be renamed into
  // the target schema by positional mapping before validation.
  baseReaderOpts_.setColumnMappingMode(dwio::common::ColumnMappingMode::kName);
  // Some readers use the requested schema to convert footer types. Inspect
  // the physical schema first; the ScanSpec still carries the target types.
  baseReaderOpts_.setFileSchema(nullptr);
  baseReaderOpts_.setAllowEmptyFile(false);
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
