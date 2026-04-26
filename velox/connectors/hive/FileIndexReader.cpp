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

#include "velox/connectors/hive/FileIndexReader.h"

#include "velox/common/Casts.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::velox::connector::hive {
namespace {

// Gets the index of a column in the request type by name.
// Throws if the column is not found.
column_index_t getRequestColumnIndex(
    const std::string& name,
    const RowTypePtr& requestType) {
  const auto idxOpt = requestType->getChildIdxIfExists(name);
  VELOX_CHECK(
      idxOpt.has_value(), "Request column {} not found in request type", name);
  return idxOpt.value();
}

// Processes a between condition bound (lower or upper).
// Returns true if the bound is a field access (non-constant).
void processBetweenBound(
    const core::TypedExprPtr& bound,
    size_t boundIndex,
    const char* boundName,
    std::vector<std::optional<variant>>& constantBoundValues,
    std::vector<column_index_t>& requestColumnIndices,
    const RowTypePtr& requestType) {
  if (auto constantExpr =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(bound)) {
    VELOX_CHECK(
        !constantExpr->hasValueVector(),
        "Complex constant values not supported for between condition {} bound",
        boundName);
    VELOX_CHECK(
        !constantExpr->value().isNull(),
        "Null constant value not allowed for between condition {} bound",
        boundName);
    constantBoundValues[boundIndex] = constantExpr->value();
    requestColumnIndices[boundIndex] = kConstantChannel;
  } else {
    auto fieldAccess =
        checkedPointerCast<const core::FieldAccessTypedExpr>(bound);
    requestColumnIndices[boundIndex] =
        getRequestColumnIndex(fieldAccess->name(), requestType);
  }
}

} // namespace

FileIndexReader::FileIndexReader(
    std::shared_ptr<const HiveConnectorSplit> hiveSplit,
    const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions,
    const RowTypePtr& requestType,
    const RowTypePtr& outputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    uint32_t maxRowsPerRequest)
    : tableHandle_{hiveTableHandle},
      connectorQueryCtx_{connectorQueryCtx},
      fileConfig_{fileConfig},
      fileHandleFactory_{fileHandleFactory},
      requestType_{requestType},
      outputType_{outputType},
      ioStatistics_{ioStatistics},
      ioStats_{ioStats},
      ioExecutor_{ioExecutor},
      pool_{connectorQueryCtx->memoryPool()},
      scanSpec_{scanSpec},
      indexLookupConditions_{indexLookupConditions},
      maxRowsPerRequest_{maxRowsPerRequest},
      hiveSplit_{std::move(hiveSplit)},
      fileReader_{createFileReader()},
      indexReader_{createIndexReader()} {
  parseIndexLookupConditions();
}

void FileIndexReader::parseIndexLookupConditions() {
  VELOX_CHECK(
      !indexLookupConditions_.empty(),
      "Index lookup conditions cannot be empty");
  const auto& indexColumns = tableHandle_->indexColumns();
  VELOX_CHECK(
      !indexColumns.empty(), "Index columns not set in hive table handle");
  VELOX_CHECK_LE(indexLookupConditions_.size(), indexColumns.size());
  VELOX_CHECK_LE(
      indexLookupConditions_.size(),
      indexColumns.size(),
      "Too many index lookup conditions");
  const auto numIndexLookupConditions = indexLookupConditions_.size();
  requestColumnIndices_.resize(numIndexLookupConditions);
  constantBoundValues_.resize(numIndexLookupConditions);

  // For building indexBoundType_ during condition processing.
  std::vector<std::string> indexColumnNames;
  indexColumnNames.reserve(numIndexLookupConditions);
  std::vector<TypePtr> indexColumnTypes;
  indexColumnTypes.reserve(numIndexLookupConditions);

  // Validate and process index lookup conditions:
  // - Index lookup conditions combined with index filters must cover all table
  // index
  //   columns in order.
  // - For each index column, it must have either an index lookup condition OR a
  // filter
  //   in the scan spec, but not both.
  // - At least one index lookup condition must have a non-constant value (field
  // access
  //   from probe side).
  // - For between conditions, at least one bound must be non-constant.
  // - Processing stops when we encounter a range filter or between condition.
  size_t numValidIndexLookupConditions{0};
  bool hasNonConstantCondition{false};
  for (size_t i = 0; i < indexLookupConditions_.size(); ++i) {
    const auto& indexColumn = indexColumns[i];
    // Process the index lookup condition for this index column.
    const auto& condition = indexLookupConditions_[i];
    // Validate that the condition's key column matches the expected index
    // column.
    const auto& conditionKeyName =
        condition->key->asUnchecked<core::FieldAccessTypedExpr>()->name();
    VELOX_CHECK_EQ(
        conditionKeyName,
        indexColumn,
        "Index lookup condition key column does not match expected index column at position {}. Index lookup conditions must follow index column order.",
        i);
    const auto* spec = scanSpec_->childByName(indexColumn);
    VELOX_CHECK(
        spec == nullptr || !spec->hasFilter(),
        "Index column '{}' cannot have both an index lookup condition and a filter at position {}",
        indexColumn,
        i);

    // Determine the request column index for the condition value.
    if (auto equalCondition =
            std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
                condition)) {
      // Check if the value is a constant or a field access.
      if (auto constantValue =
              std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
                  equalCondition->value)) {
        // Constant value - store in constantBoundValues_ and use
        // kConstantChannel.
        VELOX_CHECK(
            !constantValue->hasValueVector(),
            "Complex constant values not supported for equal condition value");
        VELOX_CHECK(
            !constantValue->value().isNull(),
            "Null constant value not allowed for equal condition value");
        constantBoundValues_[i].resize(1);
        constantBoundValues_[i][0] = constantValue->value();
        requestColumnIndices_[i] = {kConstantChannel};
      } else {
        // Field access - get request column index.
        const auto requestFieldAccess =
            checkedPointerCast<const core::FieldAccessTypedExpr>(
                equalCondition->value);
        requestColumnIndices_[i] = {
            getRequestColumnIndex(requestFieldAccess->name(), requestType_)};
        hasNonConstantCondition = true;
      }
      // Collect column name and type for indexBoundType_.
      indexColumnNames.push_back(indexColumn);
      indexColumnTypes.push_back(condition->key->type());
      ++numValidIndexLookupConditions;
      continue;
    }

    if (auto betweenCondition =
            std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
                condition)) {
      constantBoundValues_[i].resize(2);
      requestColumnIndices_[i].resize(2);

      processBetweenBound(
          betweenCondition->lower,
          0,
          "lower",
          constantBoundValues_[i],
          requestColumnIndices_[i],
          requestType_);
      processBetweenBound(
          betweenCondition->upper,
          1,
          "upper",
          constantBoundValues_[i],
          requestColumnIndices_[i],
          requestType_);

      // Track if this between condition has at least one non-constant bound.
      if (requestColumnIndices_[i][0] != kConstantChannel ||
          requestColumnIndices_[i][1] != kConstantChannel) {
        hasNonConstantCondition = true;
      }

      // Collect column name and type for indexBoundType_.
      indexColumnNames.push_back(indexColumn);
      indexColumnTypes.push_back(condition->key->type());
      ++numValidIndexLookupConditions;
      // Between condition is a range condition, stop processing further.
      break;
    }

    VELOX_FAIL(
        "Unsupported index lookup condition type: {}", condition->toString());
  }
  VELOX_CHECK_EQ(numValidIndexLookupConditions, indexLookupConditions_.size());
  VELOX_CHECK(
      hasNonConstantCondition,
      "At least one index lookup condition must have a non-constant value");

  // Build and cache the index bound row type.
  indexBoundType_ =
      ROW(std::move(indexColumnNames), std::move(indexColumnTypes));
}

std::unique_ptr<dwio::common::Reader> FileIndexReader::createFileReader() {
  VELOX_CHECK_NOT_NULL(hiveSplit_);

  dwio::common::ReaderOptions readerOpts(connectorQueryCtx_->memoryPool());
  hive::configureReaderOptions(
      fileConfig_,
      connectorQueryCtx_,
      tableHandle_,
      hiveSplit_,
      hiveSplit_->serdeParameters,
      readerOpts);
  readerOpts.setScanSpec(scanSpec_);
  readerOpts.setFileFormat(hiveSplit_->fileFormat);
  VELOX_CHECK_NULL(readerOpts.randomSkip());

  FileHandleKey fileHandleKey{
      .filename = hiveSplit_->filePath,
      .tokenProvider = connectorQueryCtx_->fsTokenProvider()};

  auto fileProperties = hiveSplit_->properties.value_or(FileProperties{});
  auto fileHandleCachePtr = fileHandleFactory_->generate(
      fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
  VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());

  auto baseFileInput = BufferedInputBuilder::getInstance()->create(
      *fileHandleCachePtr,
      readerOpts,
      connectorQueryCtx_,
      ioStatistics_,
      ioStats_,
      ioExecutor_,
      /*fileReadOps=*/{});

  auto reader = dwio::common::getReaderFactory(readerOpts.fileFormat())
                    ->createReader(std::move(baseFileInput), readerOpts);
  VELOX_CHECK_NOT_NULL(reader);
  return reader;
}

std::unique_ptr<dwio::common::IndexReader>
FileIndexReader::createIndexReader() {
  VELOX_CHECK_NOT_NULL(fileReader_);
  VELOX_CHECK(
      hiveSplit_->fileFormat == dwio::common::FileFormat::NIMBLE ||
          hiveSplit_->fileFormat == dwio::common::FileFormat::FLUX ||
          hiveSplit_->fileFormat == dwio::common::FileFormat::SST,
      "FileIndexReader only supports Nimble, Flux and SST file formats");

  dwio::common::RowReaderOptions rowReaderOpts;
  configureRowReaderOptions(
      tableHandle_->tableParameters(),
      scanSpec_,
      /*metadataFilter=*/nullptr,
      outputType_,
      hiveSplit_,
      hiveSplit_->serdeParameters,
      fileConfig_,
      connectorQueryCtx_->sessionProperties(),
      ioExecutor_,
      rowReaderOpts);
  if (hiveSplit_->fileFormat != dwio::common::FileFormat::SST) {
    rowReaderOpts.setIndexEnabled(true);
    // Disable eager first stripe load since FileIndexReader loads stripes
    // on-demand based on index lookup results.
    rowReaderOpts.setEagerFirstStripeLoad(false);
  }
  return fileReader_->createIndexReader(rowReaderOpts);
}

void FileIndexReader::startLookup(
    const Request& request,
    const Options& options) {
  // Empty files have no cluster index, so indexReader_ is null.
  if (indexReader_ == nullptr) {
    return;
  }
  VELOX_CHECK(
      !indexReader_->hasNext(),
      "Previous request not finished. Call next() first.");
  VELOX_CHECK_NOT_NULL(request.input);
  VELOX_CHECK(requestType_->equivalent(*request.input->type()));

  // Use caller-provided maxRowsPerRequest if set, otherwise fall back to
  // the construction-time default.
  const auto maxRows = options.maxRowsPerRequest != 0
      ? options.maxRowsPerRequest
      : static_cast<vector_size_t>(maxRowsPerRequest_);
  auto indexBounds = buildRequestIndexBounds(request.input);
  indexReader_->startLookup(indexBounds, {.maxRowsPerRequest = maxRows});
}

serializer::IndexBounds FileIndexReader::buildRequestIndexBounds(
    const RowVectorPtr& request) {
  VELOX_CHECK_NOT_NULL(request);
  VELOX_CHECK_NOT_NULL(indexBoundType_);

  const auto numRows = request->size();

  // Resize and clear reusable column vectors.
  lowerBoundColumns_.resize(indexLookupConditions_.size());
  upperBoundColumns_.resize(indexLookupConditions_.size());

  for (size_t i = 0; i < indexLookupConditions_.size(); ++i) {
    const auto& condition = indexLookupConditions_[i];
    const auto& type = condition->key->type();

    if (auto equalCondition =
            std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
                condition)) {
      // For equal condition, lower and upper bounds have the same value.
      const auto colIdx = requestColumnIndices_[i][0];
      if (colIdx == kConstantChannel) {
        auto constVector = BaseVector::createConstant(
            type, constantBoundValues_[i][0].value(), numRows, pool_);
        lowerBoundColumns_[i] = constVector;
        upperBoundColumns_[i] = constVector;
      } else {
        auto valueVector = request->childAt(colIdx);
        lowerBoundColumns_[i] = valueVector;
        upperBoundColumns_[i] = valueVector;
      }
    } else if (
        auto betweenCondition =
            std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
                condition)) {
      // Handle lower bound.
      if (constantBoundValues_[i][0].has_value()) {
        auto constVector = BaseVector::createConstant(
            type, constantBoundValues_[i][0].value(), numRows, pool_);
        lowerBoundColumns_[i] = constVector;
      } else {
        const auto colIdx = requestColumnIndices_[i][0];
        lowerBoundColumns_[i] = request->childAt(colIdx);
      }

      // Handle upper bound.
      if (constantBoundValues_[i][1].has_value()) {
        auto constVector = BaseVector::createConstant(
            type, constantBoundValues_[i][1].value(), numRows, pool_);
        upperBoundColumns_[i] = constVector;
      } else {
        const auto colIdx = requestColumnIndices_[i][1];
        upperBoundColumns_[i] = request->childAt(colIdx);
      }
    } else {
      VELOX_FAIL(
          "Unsupported index lookup condition type: {}", condition->toString());
    }
  }

  // Build RowVectors for lower and upper bounds.
  auto lowerBoundVector = std::make_shared<RowVector>(
      pool_, indexBoundType_, nullptr, numRows, lowerBoundColumns_);
  auto upperBoundVector = std::make_shared<RowVector>(
      pool_, indexBoundType_, nullptr, numRows, upperBoundColumns_);

  // Collect column names for indexBounds.
  std::vector<std::string> indexColumnNames;
  indexColumnNames.reserve(indexLookupConditions_.size());
  for (const auto& col : indexBoundType_->names()) {
    indexColumnNames.push_back(col);
  }

  serializer::IndexBounds bounds;
  bounds.indexColumns = std::move(indexColumnNames);
  bounds.set(
      serializer::IndexBound{std::move(lowerBoundVector), /*inclusive=*/true},
      serializer::IndexBound{std::move(upperBoundVector), /*inclusive=*/true});
  return bounds;
}

bool FileIndexReader::hasNext() {
  return indexReader_ != nullptr && indexReader_->hasNext();
}

std::unique_ptr<FileIndexReader::Result> FileIndexReader::next(
    vector_size_t maxOutputRows) {
  VELOX_CHECK_NOT_NULL(indexReader_);
  return indexReader_->next(maxOutputRows);
}

std::unordered_map<std::string, RuntimeMetric> FileIndexReader::runtimeStats() {
  // TODO: Populate with format-specific stats in a follow-up.
  // File-based IO stats are tracked externally via IoStatistics.
  return {};
}

std::string FileIndexReader::toString() const {
  return fmt::format(
      "FileIndexReader: split={} scanSpec={} requestType={} outputType={}",
      hiveSplit_->toString(),
      scanSpec_->toString(),
      requestType_->toString(),
      outputType_->toString());
}

} // namespace facebook::velox::connector::hive
