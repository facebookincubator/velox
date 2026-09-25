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

// Returns the canonical string representation of a ChangelogOperation.
std::string_view operationName(ChangelogOperation operation) {
  switch (operation) {
    case ChangelogOperation::kInsert:
      return kChangelogOpInsert;
    case ChangelogOperation::kDelete:
      return kChangelogOpDelete;
    case ChangelogOperation::kUpdateBefore:
      return kChangelogOpUpdateBefore;
    case ChangelogOperation::kUpdateAfter:
      return kChangelogOpUpdateAfter;
  }
  VELOX_UNREACHABLE(
      "Unknown ChangelogOperation: {}", static_cast<int>(operation));
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
  VELOX_CHECK(
      icebergSplit_->changelogSplitInfo.has_value(),
      "HiveIcebergSplit missing changelogSplitInfo for changelog query");
  VELOX_CHECK(
      icebergSplit_->deleteFiles.empty(),
      "Changelog splits do not support delete files");
  changelogSplitInfo_ = &icebergSplit_->changelogSplitInfo.value();

  if (!applyChangelogFilters()) {
    emptySplit_ = true;
    return;
  }

  // Changelog scans get no stats-based row-group skipping: metadataFilter is
  // built against the changelog-space scan spec (operation/ordinal/snapshotid/
  // rowdata) while the row reader uses dataScanSpec (base-table column names).
  // TODO: Fix by building metadataFilter against dataScanSpec, or by
  // registering MetadataFilter leaves on dataScanSpec after construction.
  IcebergSplitReader::prepareSplit(metadataFilter, runtimeStats, fileReadOps);
}

bool IcebergChangelogSplitReader::applyChangelogFilters() const {
  if (changelogFilters_ == nullptr) {
    return true;
  }
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_);

  const auto& filters = *changelogFilters_;
  const auto operationStr = operationName(changelogSplitInfo_->operation);

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
               return filter->testBytes(
                   operationStr.data(), operationStr.size());
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
    const RowVector& dataOutput,
    const std::string& fieldName,
    column_index_t columnIndex,
    vector_size_t positionCount) const {
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_); // set in prepareSplit()

  if (fieldName == kChangelogColOperation) {
    const auto operationStr = operationName(changelogSplitInfo_->operation);
    return BaseVector::createConstant(
        VARCHAR(), variant(std::string(operationStr)), positionCount, pool_);
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
    auto rowdataType = changelogOutputType_->childAt(columnIndex);
    auto rowdataRowType = std::dynamic_pointer_cast<const RowType>(rowdataType);
    VELOX_CHECK_NOT_NULL(
        rowdataRowType, "Changelog 'rowdata' column type must be a RowType");
    VELOX_CHECK(
        rowdataRowType->equivalent(*dataOutput.type()),
        "Changelog rowdata type ({}) does not match data output type ({})",
        rowdataRowType->toString(),
        dataOutput.type()->toString());
    return std::make_shared<RowVector>(
        pool_,
        rowdataType,
        BufferPtr(nullptr),
        positionCount,
        dataOutput.children());
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

    changelogColumns.push_back(
        buildChangelogColumn(*dataRowVector, fieldName, i, positionCount));
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
