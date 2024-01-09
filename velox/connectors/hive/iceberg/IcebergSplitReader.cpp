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

#include "velox/connectors/hive/iceberg/IcebergSplitReader.h"

#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/Mutation.h"
#include "velox/dwio/common/Reader.h"

using namespace facebook::velox::dwio::common;

namespace facebook::velox::connector::hive::iceberg {

IcebergSplitReader::IcebergSplitReader(
    const std::shared_ptr<velox::connector::hive::HiveConnectorSplit>&
        hiveSplit,
    const std::shared_ptr<HiveTableHandle>& hiveTableHandle,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    const RowTypePtr readerOutputType,
    std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
        partitionKeys,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    core::ExpressionEvaluator* expressionEvaluator,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig> hiveConfig,
    std::shared_ptr<io::IoStatistics> ioStats)
    : SplitReader(
          hiveSplit,
          hiveTableHandle,
          scanSpec,
          readerOutputType,
          partitionKeys,
          fileHandleFactory,
          executor,
          expressionEvaluator,
          connectorQueryCtx,
          hiveConfig,
          ioStats),
      baseReadOffset_(0),
      splitOffset_(0),
      originalScanSpec_(nullptr),
      originalRemainingFilters_(nullptr) {}

IcebergSplitReader::~IcebergSplitReader() {
  if (originalScanSpec_) {
    *scanSpec_ = *originalScanSpec_;
  }
}

void IcebergSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    std::shared_ptr<exec::ExprSet>& remainingFilterExprSet,
    dwio::common::RuntimeStatistics& runtimeStats) {
  createReader();

  if (testEmptySplit(runtimeStats)) {
    return;
  }

  createRowReader(metadataFilter);

  std::shared_ptr<HiveIcebergSplit> icebergSplit =
      std::dynamic_pointer_cast<HiveIcebergSplit>(hiveSplit_);
  baseReadOffset_ = 0;
  splitOffset_ = baseRowReader_->nextRowNumber();
  positionalDeleteFileReaders_.clear();

  const auto& deleteFiles = icebergSplit->deleteFiles;

  // Process the equality delete files to update the scan spec and remaining
  // filters. It needs to be done after creating the Reader and before creating
  // the RowReader.

  SubfieldFilters subfieldFilters;
  std::unique_ptr<exec::ExprSet> exprSet;

  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kEqualityDeletes &&
        deleteFile.recordCount > 0) {
      // TODO: build cache of <DeleteFile, ExprSet> to avoid repeating file
      // parsing across queries
      auto equalityDeleteReader = std::make_unique<EqualityDeleteFileReader>(
          deleteFile,
          baseFileSchema(),
          fileHandleFactory_,
          executor_,
          expressionEvaluator_,
          connectorQueryCtx_,
          hiveConfig_,
          ioStats_,
          runtimeStats,
          hiveSplit_->connectorId);
      equalityDeleteReader->readDeleteValues(subfieldFilters, exprSet);
    }
  }

  if (!subfieldFilters.empty()) {
    originalScanSpec_ = scanSpec_->clone();

    for (auto iter = subfieldFilters.begin(); iter != subfieldFilters.end();
         iter++) {
      auto childSpec = scanSpec_->getOrCreateChild(iter->first);
      childSpec->addFilter(*iter->second);
      childSpec->setSubscript(scanSpec_->children().size() - 1);
    }
  }

  if (exprSet && exprSet->size() > 0) {
    originalRemainingFilters_ = remainingFilterExprSet;
    remainingFilterExprSet->addExprs(exprSet->exprs());
  }

  createRowReader(metadataFilter);

  baseReadOffset_ = 0;
  splitOffset_ = baseRowReader_->nextRowNumber();

  // Create the positional deletes file readers. They need to be created after
  // the RowReader is created.
  positionalDeleteFileReaders_.clear();
  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kPositionalDeletes &&
        deleteFile.recordCount > 0) {
      positionalDeleteFileReaders_.push_back(
          std::make_unique<PositionalDeleteFileReader>(
              deleteFile,
              hiveSplit_->filePath,
              fileHandleFactory_,
              connectorQueryCtx_,
              executor_,
              hiveConfig_,
              ioStats_,
              runtimeStats,
              splitOffset_,
              hiveSplit_->connectorId));
    }
  }
}

uint64_t IcebergSplitReader::next(uint64_t size, VectorPtr& output) {
  Mutation mutation;
  mutation.randomSkip = baseReaderOpts_.randomSkip().get();
  mutation.deletedRows = nullptr;

  if (!positionalDeleteFileReaders_.empty()) {
    auto numBytes = bits::nbytes(size);
    dwio::common::ensureCapacity<int8_t>(
        deleteBitmap_, numBytes, connectorQueryCtx_->memoryPool());
    std::memset((void*)deleteBitmap_->as<int8_t>(), 0L, numBytes);

    for (auto iter = positionalDeleteFileReaders_.begin();
         iter != positionalDeleteFileReaders_.end();
         iter++) {
      (*iter)->readDeletePositions(
          baseReadOffset_, size, deleteBitmap_->asMutable<int8_t>());
      if ((*iter)->endOfFile()) {
        iter = positionalDeleteFileReaders_.erase(iter);
      }
    }

    deleteBitmap_->setSize(numBytes);
    mutation.deletedRows = deleteBitmap_->as<uint64_t>();
  }

  auto rowsScanned = baseRowReader_->next(size, output, &mutation);
  baseReadOffset_ += rowsScanned;

  return rowsScanned;
}

} // namespace facebook::velox::connector::hive::iceberg
