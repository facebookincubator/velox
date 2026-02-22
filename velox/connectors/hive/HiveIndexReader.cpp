/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/connectors/hive/HiveIndexReader.h"

#include "velox/common/Casts.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/serializers/KeyEncoder.h"

namespace facebook::velox::connector::hive {
namespace {
// Returns the single split from the vector. Checks that the vector is not empty
// and has exactly one element.
// TODO: Support multiple splits.
std::shared_ptr<const HiveConnectorSplit> getSingleSplit(
    const std::vector<std::shared_ptr<const HiveConnectorSplit>>& hiveSplits) {
  VELOX_CHECK_EQ(
      hiveSplits.size(),
      1,
      "HiveIndexReader currently only supports a single split");
  return hiveSplits[0];
}

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

HiveIndexReader::HiveIndexReader(
    const std::vector<std::shared_ptr<const HiveConnectorSplit>>& hiveSplits,
    const std::shared_ptr<const HiveTableHandle>& hiveTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    const std::vector<core::IndexLookupConditionPtr>& joinConditions,
    const RowTypePtr& requestType,
    const RowTypePtr& outputType,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor)
    : hiveSplit_{getSingleSplit(hiveSplits)},
      tableHandle_{hiveTableHandle},
      connectorQueryCtx_{connectorQueryCtx},
      hiveConfig_{hiveConfig},
      fileHandleFactory_{fileHandleFactory},
      requestType_{requestType},
      outputType_{outputType},
      ioStatistics_{ioStatistics},
      ioStats_{ioStats},
      ioExecutor_{ioExecutor},
      pool_{connectorQueryCtx->memoryPool()},
      scanSpec_{scanSpec},
      fileReader_{createFileReader()},
      indexReader_{createIndexReader()},
      joinConditions_{joinConditions} {
  parseJoinConditions();
}

void HiveIndexReader::parseJoinConditions() {
  VELOX_CHECK(!joinConditions_.empty(), "Join conditions cannot be empty");
  const auto& indexColumns = tableHandle_->indexColumns();
  VELOX_CHECK(
      !indexColumns.empty(), "Index columns not set in hive table handle");
  VELOX_CHECK_LE(joinConditions_.size(), indexColumns.size());
  VELOX_CHECK_LE(
      joinConditions_.size(), indexColumns.size(), "Too many join conditions");
  const auto numJoinConditions = joinConditions_.size();
  requestColumnIndices_.resize(numJoinConditions);
  constantBoundValues_.resize(numJoinConditions);

  // For building indexBoundType_ during condition processing.
  std::vector<std::string> indexColumnNames;
  indexColumnNames.reserve(numJoinConditions);
  std::vector<TypePtr> indexColumnTypes;
  indexColumnTypes.reserve(numJoinConditions);

  // Validate and process join conditions:
  // - Join conditions combined with index filters must cover all table index
  //   columns in order.
  // - For each index column, it must have either a join condition OR a filter
  //   in the scan spec, but not both.
  // - At least one join condition must have a non-constant value (field access
  //   from probe side).
  // - For between conditions, at least one bound must be non-constant.
  // - Processing stops when we encounter a range filter or between condition.
  size_t numValidJoinConditions{0};
  bool hasNonConstantCondition{false};
  for (size_t i = 0; i < joinConditions_.size(); ++i) {
    const auto& indexColumn = indexColumns[i];
    // Process the join condition for this index column.
    const auto& condition = joinConditions_[i];
    // Validate that the condition's key column matches the expected index
    // column.
    const auto& conditionKeyName =
        condition->key->asUnchecked<core::FieldAccessTypedExpr>()->name();
    VELOX_CHECK_EQ(
        conditionKeyName,
        indexColumn,
        "Join condition key column does not match expected index column at position {}. Join conditions must follow index column order.",
        i);
    const auto* spec = scanSpec_->childByName(indexColumn);
    VELOX_CHECK(
        spec == nullptr || !spec->hasFilter(),
        "Index column '{}' cannot have both a join condition and a filter at position {}",
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
      ++numValidJoinConditions;
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
      ++numValidJoinConditions;
      // Between condition is a range condition, stop processing further.
      break;
    }

    VELOX_FAIL("Unsupported join condition type: {}", condition->toString());
  }
  VELOX_CHECK_EQ(numValidJoinConditions, joinConditions_.size());
  VELOX_CHECK(
      hasNonConstantCondition,
      "At least one join condition must have a non-constant value");

  // Build and cache the index bound row type.
  indexBoundType_ =
      ROW(std::move(indexColumnNames), std::move(indexColumnTypes));
}

std::unique_ptr<dwio::common::Reader> HiveIndexReader::createFileReader() {
  VELOX_CHECK_NOT_NULL(hiveSplit_);

  dwio::common::ReaderOptions readerOpts(connectorQueryCtx_->memoryPool());
  hive::configureReaderOptions(
      hiveConfig_, connectorQueryCtx_, tableHandle_, hiveSplit_, readerOpts);
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
HiveIndexReader::createIndexReader() {
  VELOX_CHECK_NOT_NULL(fileReader_);
  VELOX_CHECK_EQ(
      hiveSplit_->fileFormat,
      dwio::common::FileFormat::NIMBLE,
      "HiveIndexReader only supports Nimble file format");

  dwio::common::RowReaderOptions rowReaderOpts;
  configureRowReaderOptions(
      tableHandle_->tableParameters(),
      scanSpec_,
      /*metadataFilter=*/nullptr,
      outputType_,
      hiveSplit_,
      hiveConfig_,
      connectorQueryCtx_->sessionProperties(),
      ioExecutor_,
      rowReaderOpts);
  rowReaderOpts.setIndexEnabled(true);
  // Disable eager first stripe load since HiveIndexReader loads stripes
  // on-demand based on index lookup results.
  rowReaderOpts.setEagerFirstStripeLoad(false);
  return fileReader_->createIndexReader(rowReaderOpts);
}

void HiveIndexReader::startLookup(
    const Request& request,
    const Options& options) {
  VELOX_CHECK(
      !indexReader_->hasNext(),
      "Previous request not finished. Call next() first.");
  VELOX_CHECK_NOT_NULL(request.input);
  VELOX_CHECK(requestType_->equivalent(*request.input->type()));

  // Build index bounds from request and pass to the index reader.
  auto indexBounds = buildRequestIndexBounds(request.input);
  indexReader_->startLookup(indexBounds, options);
}

serializer::IndexBounds HiveIndexReader::buildRequestIndexBounds(
    const RowVectorPtr& request) {
  VELOX_CHECK_NOT_NULL(request);
  VELOX_CHECK_NOT_NULL(indexBoundType_);

  const auto numRows = request->size();

  // Resize and clear reusable column vectors.
  lowerBoundColumns_.resize(joinConditions_.size());
  upperBoundColumns_.resize(joinConditions_.size());

  for (size_t i = 0; i < joinConditions_.size(); ++i) {
    const auto& condition = joinConditions_[i];
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
      VELOX_FAIL("Unsupported join condition type: {}", condition->toString());
    }
  }

  // Build RowVectors for lower and upper bounds.
  auto lowerBoundVector = std::make_shared<RowVector>(
      pool_, indexBoundType_, nullptr, numRows, lowerBoundColumns_);
  auto upperBoundVector = std::make_shared<RowVector>(
      pool_, indexBoundType_, nullptr, numRows, upperBoundColumns_);

  // Collect column names for indexBounds.
  std::vector<std::string> indexColumnNames;
  indexColumnNames.reserve(joinConditions_.size());
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

bool HiveIndexReader::hasNext() const {
  return indexReader_->hasNext();
}

std::unique_ptr<HiveIndexReader::Result> HiveIndexReader::next(
    vector_size_t maxOutputRows) {
  return indexReader_->next(maxOutputRows);
}

std::string HiveIndexReader::toString() const {
  return fmt::format(
      "HiveIndexReader: hiveSplit_{} scanSpec_{} requestType_{} outputType_{}",
      hiveSplit_->toString(),
      scanSpec_->toString(),
      requestType_->toString(),
      outputType_->toString());
}

} // namespace facebook::velox::connector::hive
