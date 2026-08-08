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

IcebergChangelogSplitReader::IcebergChangelogSplitReader(
    const std::shared_ptr<const HiveIcebergSplit>& icebergSplit,
    const FileTableHandlePtr& tableHandle,
    const std::unordered_map<std::string, FileColumnHandlePtr>* partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const RowTypePtr& dataReaderOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<ColumnHandleMap> columnHandles,
    const RowTypePtr& changelogOutputType,
    ColumnHandleMap changelogColumnHandles,
    const common::SubfieldFilters* changelogFilters)
    : IcebergSplitReader(
          icebergSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          dataReaderOutputType,
          dataIoStats,
          metadataIoStats,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec,
          std::move(columnHandles)),
      changelogOutputType_(changelogOutputType),
      changelogColumnHandles_(std::move(changelogColumnHandles)),
      changelogFilters_(changelogFilters) {}

// ── prepareSplit
// ────────────────────────────────────────────────────────────── Extracts
// ChangelogSplitInfo from the split, evaluates the constant-column subfield
// filters (operation/ordinal/snapshotid) against it, and — if any filter
// rejects the constant value — marks the split as empty so no file I/O is
// performed. For accepted splits, delegates to IcebergSplitReader to open
// delete files and set up the row reader.

void IcebergChangelogSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStatistics& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  auto icebergSplit =
      std::dynamic_pointer_cast<const HiveIcebergSplit>(fileSplit_);
  VELOX_CHECK_NOT_NULL(icebergSplit, "Expected HiveIcebergSplit");
  VELOX_CHECK_NOT_NULL(
      icebergSplit->changelogSplitInfo,
      "HiveIcebergSplit missing changelogSplitInfo for changelog query");
  changelogSplitInfo_ = icebergSplit->changelogSplitInfo;

  if (!applyChangelogFilters()) {
    emptySplit_ = true;
    return;
  }

  // Split passed constant-column filters — proceed with full preparation.
  IcebergSplitReader::prepareSplit(metadataFilter, runtimeStats, fileReadOps);
}

bool IcebergChangelogSplitReader::applyChangelogFilters() const {
  if (changelogFilters_ == nullptr) {
    return true;
  }
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_);

  const auto& filters = *changelogFilters_;

  std::string operationStr;
  switch (changelogSplitInfo_->operation) {
    case ChangelogOperation::INSERT:
      operationStr = "INSERT";
      break;
    case ChangelogOperation::DELETE:
      operationStr = "DELETE";
      break;
    case ChangelogOperation::UPDATE_BEFORE:
      operationStr = "UPDATE_BEFORE";
      break;
    case ChangelogOperation::UPDATE_AFTER:
      operationStr = "UPDATE_AFTER";
      break;
    default:
      VELOX_FAIL("Unknown changelog operation");
  }

  auto test = [&](const std::string& colName, auto testFn) -> bool {
    auto it = filters.find(common::Subfield(colName));
    if (it == filters.end()) {
      return true; // No filter on this column — pass.
    }
    return testFn(it->second.get());
  };

  return test(
             "operation",
             [&](const common::Filter* f) {
               return f->testBytes(operationStr.data(), operationStr.size());
             }) &&
      test("ordinal",
           [&](const common::Filter* f) {
             return f->testInt64(changelogSplitInfo_->ordinal);
           }) &&
      test("snapshotid", [&](const common::Filter* f) {
           return f->testInt64(changelogSplitInfo_->snapshotId);
         });
}

// ── buildChangelogColumn
// ──────────────────────────────────────────────────────

VectorPtr IcebergChangelogSplitReader::buildChangelogColumn(
    const RowVectorPtr& dataOutput,
    const std::string& fieldName,
    column_index_t colIdx,
    vector_size_t positionCount) const {
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_);

  if (fieldName == "operation") {
    std::string operationStr;
    switch (changelogSplitInfo_->operation) {
      case ChangelogOperation::INSERT:
        operationStr = "INSERT";
        break;
      case ChangelogOperation::DELETE:
        operationStr = "DELETE";
        break;
      case ChangelogOperation::UPDATE_BEFORE:
        operationStr = "UPDATE_BEFORE";
        break;
      case ChangelogOperation::UPDATE_AFTER:
        operationStr = "UPDATE_AFTER";
        break;
      default:
        VELOX_FAIL("Unknown changelog operation");
    }
    return BaseVector::createConstant(
        VARCHAR(), variant(operationStr), positionCount, pool_);
  }

  if (fieldName == "ordinal") {
    return BaseVector::createConstant(
        BIGINT(), variant(changelogSplitInfo_->ordinal), positionCount, pool_);
  }

  if (fieldName == "snapshotid") {
    return BaseVector::createConstant(
        BIGINT(),
        variant(changelogSplitInfo_->snapshotId),
        positionCount,
        pool_);
  }

  if (fieldName == "rowdata") {
    auto rowdataType = changelogOutputType_->childAt(colIdx);
    auto rowdataRowType = std::dynamic_pointer_cast<const RowType>(rowdataType);
    VELOX_CHECK_NOT_NULL(
        rowdataRowType, "Changelog 'rowdata' column type must be a RowType");
    VELOX_CHECK_EQ(
        rowdataRowType->size(),
        dataOutput->childrenSize(),
        "Changelog rowdata field count ({}) != data output column count ({})",
        rowdataRowType->size(),
        dataOutput->childrenSize());
    return std::make_shared<RowVector>(
        pool_,
        rowdataType,
        BufferPtr(nullptr),
        positionCount,
        dataOutput->children());
  }

  VELOX_FAIL("Unknown changelog column field name: '{}'", fieldName);
}

// ── next
// ────────────────────────────────────────────────────────────────────── Reads
// a base-table batch from IcebergSplitReader (which handles delete files,
// schema evolution, etc.), then reshapes it into the changelog output schema:
// (operation VARCHAR, ordinal BIGINT, snapshotid BIGINT, rowdata ROW). The
// caller (FileDataSource::next) allocated output with changelogOutputType_, so
// we replace it with the freshly constructed changelog RowVector.

uint64_t IcebergChangelogSplitReader::next(uint64_t size, VectorPtr& output) {
  VELOX_CHECK_NOT_NULL(changelogSplitInfo_);

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
