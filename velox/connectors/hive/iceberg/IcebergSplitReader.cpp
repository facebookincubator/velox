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

using namespace facebook::velox::dwio::common;

namespace facebook::velox::connector::hive::iceberg {

IcebergSplitReader::IcebergSplitReader(
    const std::shared_ptr<const hive::HiveConnectorSplit>& hiveSplit,
    const HiveTableHandlePtr& hiveTableHandle,
    const std::unordered_map<std::string, HiveColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStats,
    const std::shared_ptr<filesystems::File::IoStats>& fsStats,
    FileHandleFactory* const fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    core::ExpressionEvaluator* expressionEvaluator,
    std::atomic<uint64_t>& totalRemainingFilterTime)
    : SplitReader(
          hiveSplit,
          hiveTableHandle,
          partitionKeys,
          connectorQueryCtx,
          hiveConfig,
          readerOutputType,
          ioStats,
          fsStats,
          fileHandleFactory,
          executor,
          scanSpec),
      baseReadOffset_(0),
      splitOffset_(0),
      deleteBitmap_(nullptr),
      deleteExprSet_(nullptr),
      expressionEvaluator_(expressionEvaluator),
      totalRemainingFilterMs_(totalRemainingFilterTime) {}

IcebergSplitReader::~IcebergSplitReader() {}

void IcebergSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats) {
  createReader();
  if (emptySplit_) {
    return;
  }
  auto rowType = getAdaptedRowType();

  std::shared_ptr<const HiveIcebergSplit> icebergSplit =
      std::dynamic_pointer_cast<const HiveIcebergSplit>(hiveSplit_);
  const auto& deleteFiles = icebergSplit->deleteFiles;
  std::unordered_set<int32_t> equalityFieldIds;
  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kEqualityDeletes &&
        deleteFile.recordCount > 0) {
      equalityFieldIds.insert(
          deleteFile.equalityFieldIds.begin(),
          deleteFile.equalityFieldIds.end());
    }
  }

  // checkIfSplitIsEmpty needs to use the base reader's schemaWithId_. For that
  // we need to update the base RowReader to include these extra fields from the
  // equality delete file first, so that the schemaWithId_ of the base file is
  // updated when we call baseFileSchema() later.
  baseReader_->setRequiredExtraFieldIds(equalityFieldIds);

  if (checkIfSplitIsEmpty(runtimeStats)) {
    VELOX_CHECK(emptySplit_);
    return;
  }

  // Process the equality delete files to update the scan spec and remaining
  // filters. It needs to be done after creating the Reader and before creating
  // the RowReader.

  SubfieldFilters subfieldFilters;
  std::vector<core::TypedExprPtr> conjunctInputs;

  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kEqualityDeletes &&
        deleteFile.recordCount > 0) {
      // TODO: build cache of <Partition, ExprSet> to avoid repeating file
      // parsing across partitions. Within a single partition, the splits should
      // be with the same equality delete files and only need to be parsed once.
      auto equalityDeleteReader = std::make_unique<EqualityDeleteFileReader>(
          deleteFile,
          baseFileSchema(),
          fileHandleFactory_,
          executor_,
          connectorQueryCtx_,
          hiveConfig_,
          ioStats_,
          fsStats_,
          hiveSplit_->connectorId);
      equalityDeleteReader->readDeleteValues(subfieldFilters, conjunctInputs);
    }
  }

  if (!subfieldFilters.empty()) {
    for (const auto& [key, filter] : subfieldFilters) {
      auto childSpec = scanSpec_->getOrCreateChild(key, true);
      childSpec->addFilter(*filter);
      childSpec->setHasTempFilter(true);
      childSpec->setSubscript(scanSpec_->children().size() - 1);
    }
  }

  if (!conjunctInputs.empty()) {
    core::TypedExprPtr expression =
        std::make_shared<core::CallTypedExpr>(BOOLEAN(), conjunctInputs, "and");
    deleteExprSet_ = expressionEvaluator_->compile(expression);
    VELOX_CHECK_EQ(deleteExprSet_->size(), 1);
  }

  createRowReader(std::move(metadataFilter), std::move(rowType));

  baseReadOffset_ = 0;
  splitOffset_ = baseRowReader_->nextRowNumber();

  // Create the positional deletes file readers. They need to be created after
  // the RowReader is created.
  positionalDeleteFileReaders_.clear();
  for (const auto& deleteFile : deleteFiles) {
    if (deleteFile.content == FileContent::kPositionalDeletes) {
      if (deleteFile.recordCount > 0) {
        positionalDeleteFileReaders_.push_back(
            std::make_unique<PositionalDeleteFileReader>(
                deleteFile,
                hiveSplit_->filePath,
                fileHandleFactory_,
                connectorQueryCtx_,
                executor_,
                hiveConfig_,
                ioStats_,
                fsStats_,
                runtimeStats,
                splitOffset_,
                hiveSplit_->connectorId));
      }
    }
  }
}

std::shared_ptr<const dwio::common::TypeWithId>
IcebergSplitReader::baseFileSchema() {
  VELOX_CHECK_NOT_NULL(baseReader_.get());
  return baseReader_->typeWithId();
}

uint64_t IcebergSplitReader::next(uint64_t size, VectorPtr& output) {
  Mutation mutation;
  mutation.randomSkip = baseReaderOpts_.randomSkip().get();
  mutation.deletedRows = nullptr;

  if (deleteBitmap_) {
    std::memset(
        (void*)(deleteBitmap_->asMutable<int8_t>()), 0L, deleteBitmap_->size());
  }

  const auto actualSize = baseRowReader_->nextReadSize(size);

  if (actualSize == dwio::common::RowReader::kAtEnd) {
    return 0;
  }

  if (!positionalDeleteFileReaders_.empty()) {
    auto numBytes = bits::nbytes(actualSize);
    dwio::common::ensureCapacity<int8_t>(
        deleteBitmap_, numBytes, connectorQueryCtx_->memoryPool(), false, true);

    for (auto iter = positionalDeleteFileReaders_.begin();
         iter != positionalDeleteFileReaders_.end();) {
      (*iter)->readDeletePositions(baseReadOffset_, actualSize, deleteBitmap_);

      if ((*iter)->noMoreData()) {
        iter = positionalDeleteFileReaders_.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  mutation.deletedRows = deleteBitmap_ && deleteBitmap_->size() > 0
      ? deleteBitmap_->as<uint64_t>()
      : nullptr;

  auto rowsScanned = baseRowReader_->next(actualSize, output, &mutation);

  // Evaluate the remaining filter deleteExprSet_ for every batch and update the
  // output vector if it reduces any rows.
  if (deleteExprSet_) {
    auto filterStartMs = getCurrentTimeMs();

    filterRows_.resize(output->size());
    auto rowVector = std::dynamic_pointer_cast<RowVector>(output);
    expressionEvaluator_->evaluate(
        deleteExprSet_.get(), filterRows_, *rowVector, filterResult_);
    auto numRemainingRows = exec::processFilterResults(
        filterResult_, filterRows_, filterEvalCtx_, pool_);

    if (numRemainingRows < output->size()) {
      output = exec::wrap(
          numRemainingRows, filterEvalCtx_.selectedIndices, rowVector);
    }

    totalRemainingFilterMs_.fetch_add(
        (getCurrentTimeMs() - filterStartMs), std::memory_order_relaxed);
  }

  baseReadOffset_ += rowsScanned;

  if (rowsScanned == 0) {
    scanSpec_->deleteTempNodes();
  }

  return rowsScanned;
}

} // namespace facebook::velox::connector::hive::iceberg
