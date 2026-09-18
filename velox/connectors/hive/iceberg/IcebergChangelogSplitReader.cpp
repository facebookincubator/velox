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

#include "velox/connectors/hive/iceberg/IcebergChangelogSplitReader.h"

#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

/// Returns the canonical string representation of a ChangelogOperation.
/// All four enumerators are handled; the default branch is unreachable.
std::string_view operationName(ChangelogOperation op) {
  switch (op) {
    case ChangelogOperation::INSERT:
      return kChangelogOpInsert;
    case ChangelogOperation::DELETE:
      return kChangelogOpDelete;
    case ChangelogOperation::UPDATE_BEFORE:
      return kChangelogOpUpdateBefore;
    case ChangelogOperation::UPDATE_AFTER:
      return kChangelogOpUpdateAfter;
  }
  VELOX_UNREACHABLE("Unknown ChangelogOperation: {}", static_cast<int>(op));
}

} // namespace

IcebergChangelogSplitReader::IcebergChangelogSplitReader(
    const std::shared_ptr<const HiveIcebergSplit>& icebergSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ChangelogScanContext& scanContext,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const RowTypePtr& changelogOutputType,
    ColumnHandleMap changelogColumnHandles,
    const common::SubfieldFilters* changelogFilters)
    : IcebergSplitReader(
          icebergSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          scanContext.dataReaderOutputType,
          dataIoStats,
          metadataIoStats,
          ioStats,
          fileHandleFactory,
          executor,
          scanContext.dataScanSpec,
          scanContext.dataColumnHandles),
      changelogOutputType_(changelogOutputType),
      changelogColumnHandles_(std::move(changelogColumnHandles)),
      changelogFilters_(changelogFilters) {}

void IcebergChangelogSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStats& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  auto icebergSplit =
      std::dynamic_pointer_cast<const HiveIcebergSplit>(fileSplit_);
  VELOX_CHECK_NOT_NULL(icebergSplit, "Expected HiveIcebergSplit");
  VELOX_CHECK(
      icebergSplit->changelogSplitInfo.has_value(),
      "HiveIcebergSplit missing changelogSplitInfo for changelog query");
  changelogSplitInfo_ = &*icebergSplit->changelogSplitInfo;

  if (!applyChangelogFilters()) {
    emptySplit_ = true;
    return;
  }

  // Split passed constant-column filters — proceed with full preparation.
  //
  // NOTE: metadataFilter was constructed by FileDataSource against the
  // changelog-space scanSpec_ (column names: operation/ordinal/snapshotid/
  // rowdata).  The row reader, however, uses dataScanSpec (column names from
  // the base table: id, name, …).  MetadataFilter::LeafNode::addToScanSpec()
  // registered each leaf on changelog-space spec nodes that are absent from
  // dataScanSpec.  As a result, LeafNode::eval() always returns nullptr, so
  // the filter evaluates to "pass all row groups" — changelog scans silently
  // lose stats-based row-group skipping entirely, rather than producing wrong
  // results.
  // Fixing this requires either building metadataFilter against dataScanSpec
  // (stripping the "rowdata." prefix from leaf subfields), or registering the
  // leaves on dataScanSpec after construction.
  IcebergSplitReader::prepareSplit(metadataFilter, runtimeStats, fileReadOps);
}

bool IcebergChangelogSplitReader::applyChangelogFilters() const {
  if (changelogFilters_ == nullptr) {
    return true;
  }
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_);

  const auto& filters = *changelogFilters_;
  const auto opName = operationName(changelogSplitInfo_->operation);

  auto evaluate = [&](std::string_view colName, auto evaluateFn) -> bool {
    auto it = filters.find(common::Subfield(std::string(colName)));
    if (it == filters.end()) {
      return true; // No filter on this column — pass.
    }
    return evaluateFn(it->second.get());
  };

  return evaluate(
             kChangelogColOperation,
             [&](const common::Filter* filter) {
               return filter->testBytes(opName.data(), opName.size());
             }) &&
      evaluate(
             kChangelogColOrdinal,
             [&](const common::Filter* filter) {
               return filter->testInt64(changelogSplitInfo_->ordinal);
             }) &&
      evaluate(kChangelogColSnapshotId, [&](const common::Filter* filter) {
           return filter->testInt64(changelogSplitInfo_->snapshotId);
         });
}

VectorPtr IcebergChangelogSplitReader::buildChangelogColumn(
    const RowVectorPtr& dataOutput,
    const std::string& fieldName,
    column_index_t colIdx,
    vector_size_t positionCount) const {
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_); // set in prepareSplit()

  if (fieldName == kChangelogColOperation) {
    const auto opName = operationName(changelogSplitInfo_->operation);
    return BaseVector::createConstant(
        VARCHAR(), variant(std::string(opName)), positionCount, pool_);
  }

  if (fieldName == kChangelogColOrdinal) {
    return BaseVector::createConstant(
        BIGINT(), variant(changelogSplitInfo_->ordinal), positionCount, pool_);
  }

  if (fieldName == kChangelogColSnapshotId) {
    return BaseVector::createConstant(
        BIGINT(),
        variant(changelogSplitInfo_->snapshotId),
        positionCount,
        pool_);
  }

  if (fieldName == kChangelogColRowdata) {
    auto rowdataType = changelogOutputType_->childAt(colIdx);
    auto rowdataRowType = std::dynamic_pointer_cast<const RowType>(rowdataType);
    VELOX_CHECK_NOT_NULL(
        rowdataRowType, "Changelog 'rowdata' column type must be a RowType");
    VELOX_CHECK(
        rowdataRowType->equivalent(*dataOutput->type()),
        "Changelog rowdata type ({}) does not match data output type ({})",
        rowdataRowType->toString(),
        dataOutput->type()->toString());
    return std::make_shared<RowVector>(
        pool_,
        rowdataType,
        BufferPtr(nullptr),
        positionCount,
        dataOutput->children());
  }

  VELOX_FAIL("Unknown changelog column field name: '{}'", fieldName);
}

uint64_t IcebergChangelogSplitReader::next(uint64_t size, VectorPtr& output) {
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_); // set in prepareSplit()

  // Pre-allocate the data output buffer if needed. The base row reader
  // (SelectiveStructColumnReaderBase::next) requires a non-null output vector
  // before it calls getValues(). readerOutputType_ holds the base-table schema.
  if (!dataOutput_) {
    dataOutput_ = BaseVector::create(readerOutputType_, 0, pool_);
  }

  // Read base-table rows into our private buffer.
  const uint64_t rowsScanned = IcebergSplitReader::next(size, dataOutput_);
  if (rowsScanned == 0) {
    return 0;
  }

  auto* dataRowVector = dataOutput_->as<RowVector>();
  VELOX_CHECK_NOT_NULL(dataRowVector, "Expected RowVector from base reader.");
  const auto positionCount = static_cast<vector_size_t>(dataOutput_->size());

  // Build the changelog output columns.
  std::vector<VectorPtr> changelogColumns;
  changelogColumns.reserve(changelogOutputType_->size());

  for (column_index_t i = 0; i < changelogOutputType_->size(); ++i) {
    const auto& outputColName = changelogOutputType_->nameOf(i);

    // Resolve the physical field name via changelog column handles.
    auto it = changelogColumnHandles_.find(outputColName);
    VELOX_CHECK(
        it != changelogColumnHandles_.end(),
        "No column handle found for changelog output column '{}'.",
        outputColName);
    const auto& fieldName =
        static_cast<const FileColumnHandle*>(it->second.get())->name();

    changelogColumns.push_back(buildChangelogColumn(
        std::dynamic_pointer_cast<RowVector>(dataOutput_),
        fieldName,
        i,
        positionCount));
  }

  // Replace the caller-supplied output with the changelog RowVector.
  output = std::make_shared<RowVector>(
      pool_,
      changelogOutputType_,
      BufferPtr(nullptr),
      positionCount,
      std::move(changelogColumns));

  return rowsScanned;
}

} // namespace facebook::velox::connector::hive::iceberg
