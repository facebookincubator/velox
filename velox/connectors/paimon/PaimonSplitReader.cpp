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
#include "velox/connectors/paimon/PaimonSplitReader.h"

#include <cstring>
#include <limits>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/Mutation.h"

namespace facebook::velox::connector::paimon {

PaimonSplitReader::PaimonSplitReader(
    const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
    std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
    const std::shared_ptr<const PaimonTableHandle>& tableHandle,
    const std::unordered_map<std::string, connector::FileColumnHandlePtr>*
        partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const connector::FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : connector::FileSplitReader(
          fileSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          readerOutputType,
          dataIoStats,
          metadataIoStats,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      paimonSplit_(std::move(paimonSplit)) {
  VELOX_CHECK_NOT_NULL(paimonSplit_);

  // Build FileConnectorSplits for all files so we can switch between them
  // during multi-file iteration.
  const auto& dataFiles = paimonSplit_->dataFiles();
  fileSplits_.reserve(dataFiles.size());
  for (const auto& dataFile : dataFiles) {
    fileSplits_.emplace_back(makeFileConnectorSplit(*paimonSplit_, dataFile));
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
    // TODO: Decode internal columns and row kinds for primary-key/changelog
    // files.
    const auto actualSize = baseRowReader_->nextReadSize(size);
    if (actualSize == dwio::common::RowReader::kAtEnd) {
      finishFileSplitReader();
      continue;
    }

    dwio::common::Mutation mutation;
    mutation.randomSkip = baseReaderOpts_.randomSkip().get();
    if (deletionVectorReader_ != nullptr &&
        !deletionVectorReader_->noMoreData()) {
      // The bitmap is reused across batches and files, so clear every word
      // this batch reads before setting its deletes.
      const auto numBytes = bits::nwords(actualSize) * sizeof(uint64_t);
      dwio::common::ensureCapacity<int8_t>(
          deleteBitmap_, numBytes, connectorQueryCtx_->memoryPool());
      std::memset(deleteBitmap_->asMutable<int8_t>(), 0, numBytes);
      deletionVectorReader_->readDeletePositions(
          baseRowReader_->nextRowNumber(), actualSize, deleteBitmap_);
      mutation.deletedRows = deleteBitmap_->as<uint64_t>();
    }
    const auto rowsRead = baseRowReader_->next(actualSize, output, &mutation);
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

    // TODO: Add historical-schema adaptation using per-file schema metadata.
    connector::FileSplitReader::prepareSplit(metadataFilter_, *runtimeStats_);
    if (!emptySplit_) {
      const auto& dataFile = paimonSplit_->dataFiles()[currentFileIndex_];
      if (dataFile.deletionFile.has_value()) {
        deletionVectorReader_ = std::make_unique<PaimonDeletionVectorReader>(
            *dataFile.deletionFile);
      } else {
        deletionVectorReader_ = nullptr;
      }
      return true;
    }
    // Empty preparation may leave a file reader but no row reader. Reset
    // the per-file state before advancing so the next file is prepared.
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
  deletionVectorReader_ = nullptr;
  emptySplit_ = false;
}

// static
std::shared_ptr<connector::FileConnectorSplit>
PaimonSplitReader::makeFileConnectorSplit(
    const PaimonConnectorSplit& paimonSplit,
    const PaimonDataFile& dataFile) {
  // Primary-key files put the key, sequence number and row kind columns
  // before the table columns, so bind table columns to file columns by name.
  std::optional<dwio::common::ColumnMappingMode> columnMappingMode;
  if (paimonSplit.tableType() == PaimonTableType::kPrimaryKey) {
    columnMappingMode = dwio::common::ColumnMappingMode::kName;
  }
  return std::make_shared<connector::FileConnectorSplit>(
      paimonSplit.connectorId,
      dataFile.path,
      paimonSplit.fileFormat(),
      /*_start=*/0,
      /*_length=*/std::numeric_limits<uint64_t>::max(),
      /*splitWeight=*/0,
      /*cacheable=*/true,
      /*_properties=*/std::nullopt,
      paimonSplit.partitionKeys(),
      columnMappingMode);
}

} // namespace facebook::velox::connector::paimon
