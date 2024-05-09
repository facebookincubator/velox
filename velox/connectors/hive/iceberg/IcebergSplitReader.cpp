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
using namespace facebook::velox::exec;

namespace facebook::velox::connector::hive::iceberg {

IcebergSplitReader::IcebergSplitReader(
    const std::shared_ptr<const hive::HiveConnectorSplit>& hiveSplit,
    const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
    const std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
        partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& ioStats,
    FileHandleFactory* const fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<exec::ExprSet>& remainingFilterExprSet,
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
          fileHandleFactory,
          executor,
          scanSpec),
      originalScanSpec_(nullptr),
      baseReadOffset_(0),
      splitOffset_(0),
      deleteExprSet_(nullptr),
      expressionEvaluator_(expressionEvaluator),
      totalRemainingFilterTime_(totalRemainingFilterTime) {}

IcebergSplitReader::~IcebergSplitReader() {
  if (originalScanSpec_) {
    *scanSpec_ = *originalScanSpec_;
  }
}

void IcebergSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats) {
  createReader();

  if (checkIfSplitIsEmpty(runtimeStats)) {
    VELOX_CHECK(emptySplit_);
    return;
  }

  std::shared_ptr<const HiveIcebergSplit> icebergSplit =
      std::dynamic_pointer_cast<const HiveIcebergSplit>(hiveSplit_);
  const auto& deleteFiles = icebergSplit->deleteFiles;

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
          runtimeStats,
          hiveSplit_->connectorId);
      equalityDeleteReader->readDeleteValues(subfieldFilters, conjunctInputs);
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

  if (!conjunctInputs.empty()) {
    core::TypedExprPtr expression =
        std::make_shared<core::CallTypedExpr>(BOOLEAN(), conjunctInputs, "and");
    deleteExprSet_ = expressionEvaluator_->compile(expression);
    VELOX_CHECK_EQ(deleteExprSet_->size(), 1);
  }

  createRowReader(metadataFilter);

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
                runtimeStats,
                splitOffset_,
                hiveSplit_->connectorId));
      }
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
         iter != positionalDeleteFileReaders_.end();) {
      (*iter)->readDeletePositions(
          baseReadOffset_, size, deleteBitmap_->asMutable<int8_t>());
      if ((*iter)->endOfFile()) {
        iter = positionalDeleteFileReaders_.erase(iter);
      } else {
        ++iter;
      }
    }

    deleteBitmap_->setSize(numBytes);
    mutation.deletedRows = deleteBitmap_->as<uint64_t>();
  }

  auto rowsScanned = baseRowReader_->next(size, output, &mutation);

  // Evaluate the remaining filter deleteExprSet_ for every batch and update the
  // output vector if it reduces any rows.
  if (deleteExprSet_) {
    auto filterStartMicros = getCurrentTimeMicro();

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

    totalRemainingFilterTime_.fetch_add(
        (getCurrentTimeMicro() - filterStartMicros) * 1000,
        std::memory_order_relaxed);
  }

  baseReadOffset_ += rowsScanned;

  return rowsScanned;
}

} // namespace facebook::velox::connector::hive::iceberg
