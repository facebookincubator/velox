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

#include "velox/connectors/hive/HiveIndexReader.h"

#include <folly/container/F14Set.h>

#include "velox/common/Casts.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/vector/SelectivityVector.h"

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

template <TypeKind kind>
variant extractValue(const DecodedVector& decoded, vector_size_t row) {
  using T = typename TypeTraits<kind>::NativeType;
  return variant(decoded.valueAt<T>(row));
}

// Checks if a filter is a point lookup (single value equality filter).
// Returns true if the filter represents a point lookup, false if it's a range
// filter or unsupported filter type.
// This is used to determine if we can continue processing join conditions after
// an index column has a filter. Only point lookup filters allow subsequent
// index columns to be used for join conditions.
bool isPointLookupFilter(const common::Filter* filter) {
  VELOX_CHECK_NOT_NULL(filter);

  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      const auto* range = filter->as<common::BigintRange>();
      return range->isSingleValue();
    }
    case common::FilterKind::kDoubleRange: {
      const auto* range = filter->as<common::DoubleRange>();
      return !range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive();
    }
    case common::FilterKind::kFloatRange: {
      const auto* range = filter->as<common::FloatRange>();
      return !range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive();
    }
    case common::FilterKind::kBytesRange: {
      const auto* range = filter->as<common::BytesRange>();
      return range->isSingleValue();
    }
    case common::FilterKind::kBytesValues: {
      const auto* values = filter->as<common::BytesValues>();
      return values->values().size() == 1;
    }
    case common::FilterKind::kBoolValue:
      // These are always point lookups or not convertible.
      return true;
    default:
      // Unsupported filter types are not point lookups.
      return false;
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
    const std::shared_ptr<io::IoStatistics>& ioStats,
    const std::shared_ptr<filesystems::File::IoStats>& fsStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor)
    : hiveSplit_{getSingleSplit(hiveSplits)},
      tableHandle_{hiveTableHandle},
      connectorQueryCtx_{connectorQueryCtx},
      hiveConfig_{hiveConfig},
      fileHandleFactory_{fileHandleFactory},
      requestType_{requestType},
      outputType_{outputType},
      ioStats_{ioStats},
      fsStats_{fsStats},
      ioExecutor_{ioExecutor},
      pool_{connectorQueryCtx->memoryPool()},
      scanSpec_{scanSpec},
      fileReader_{createFileReader()},
      rowReader_{createRowReader()},
      joinConditions_{joinConditions} {
  initJoinConditions();
}

void HiveIndexReader::initJoinConditions() {
  VELOX_CHECK(!joinConditions_.empty(), "Join conditions cannot be empty");
  VELOX_CHECK(
      !tableHandle_->indexColumns().empty(),
      "Index columns not set in hive table handle");
  VELOX_CHECK(joinIndexColumnSpecs_.empty());
  VELOX_CHECK(requestColumnIndices_.empty());

  const auto& tableIndexColumns = tableHandle_->indexColumns();
  VELOX_CHECK_LE(
      joinConditions_.size(),
      tableIndexColumns.size(),
      "Too many join conditions");
  joinIndexColumnSpecs_.reserve(joinConditions_.size());
  requestColumnIndices_.reserve(joinConditions_.size());
  decodedRequestVectors_.resize(joinConditions_.size());

  auto getRequestColumnIndex = [&](const std::string& name) {
    const auto idxOpt = requestType_->getChildIdxIfExists(name);
    VELOX_CHECK(
        idxOpt.has_value(),
        "Request column {} not found in request type",
        name);
    return idxOpt.value();
  };

  // Validate and process join conditions:
  // - Join conditions combined with index filters must cover all table index
  //   columns in order.
  // - For each index column, it must have either a join condition OR a filter
  //   in the scan spec, but not both.
  // - Processing stops when we encounter a range filter or between condition.
  size_t conditionIdx = 0;
  for (size_t i = 0; i < tableIndexColumns.size(); ++i) {
    const auto& indexColumn = tableIndexColumns[i];
    auto* spec = scanSpec_->childByName(indexColumn);
    if (spec == nullptr) {
      break;
    }

    // Check if this index column has a filter in scanSpec.
    const bool hasFilter = spec->hasFilter();

    // Check if there's a join condition for this index column.
    const bool hasJoinCondition =
        (conditionIdx < joinConditions_.size() &&
         joinConditions_[conditionIdx]->key->name() == indexColumn);

    if (!hasFilter && !hasJoinCondition) {
      break;
    }
    VELOX_CHECK(
        !(hasFilter && hasJoinCondition),
        "Index column '{}' cannot have both a join condition and a pushdown filter",
        indexColumn);

    if (hasFilter) {
      // If the filter is not a point lookup (range filter), we cannot continue
      // processing join conditions after this point.
      if (!isPointLookupFilter(spec->filter())) {
        break;
      }
      continue;
    }

    joinIndexColumnSpecs_.push_back(spec);
    // Process the join condition for this index column.
    const auto& condition = joinConditions_[conditionIdx];
    VELOX_CHECK(!condition->isFilter());

    // Determine the request column index for the condition value.
    if (auto equalCondition =
            std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
                condition)) {
      const auto requestFieldAccess =
          checkedPointerCast<const core::FieldAccessTypedExpr>(
              equalCondition->value);
      requestColumnIndices_.push_back(
          {getRequestColumnIndex(requestFieldAccess->name())});
      decodedRequestVectors_[conditionIdx].resize(1);
      ++conditionIdx;
      continue;
    }

    if (auto betweenCondition =
            std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
                condition)) {
      auto lowerFieldAccess =
          checkedPointerCast<const core::FieldAccessTypedExpr>(
              betweenCondition->lower);
      auto upperFieldAccess =
          checkedPointerCast<const core::FieldAccessTypedExpr>(
              betweenCondition->upper);
      requestColumnIndices_.push_back(
          {getRequestColumnIndex(lowerFieldAccess->name()),
           getRequestColumnIndex(upperFieldAccess->name())});
      decodedRequestVectors_[conditionIdx].resize(2);
      ++conditionIdx;
      // Between condition is a range condition, stop processing further.
      break;
    }

    VELOX_FAIL("Unsupported join condition type: {}", condition->toString());
  }

  VELOX_CHECK_EQ(
      conditionIdx,
      joinConditions_.size(),
      "Not all join conditions were processed.");
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
      fileHandleKey, &fileProperties, fsStats_ ? fsStats_.get() : nullptr);
  VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());

  auto baseFileInput = BufferedInputBuilder::getInstance()->create(
      *fileHandleCachePtr,
      readerOpts,
      connectorQueryCtx_,
      ioStats_,
      fsStats_,
      ioExecutor_,
      /*fileReadOps=*/{});

  auto reader = dwio::common::getReaderFactory(readerOpts.fileFormat())
                    ->createReader(std::move(baseFileInput), readerOpts);
  VELOX_CHECK_NOT_NULL(reader);
  return reader;
}

std::unique_ptr<dwio::common::RowReader> HiveIndexReader::createRowReader() {
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
      fileReader_->rowType(),
      hiveSplit_,
      hiveConfig_,
      connectorQueryCtx_->sessionProperties(),
      ioExecutor_,
      rowReaderOpts);
  rowReaderOpts.setIndexEnabled(true);
  // Disable eager first stripe load since HiveIndexReader loads stripes
  // on-demand based on index lookup results.
  rowReaderOpts.setEagerFirstStripeLoad(false);
  return fileReader_->createRowReader(rowReaderOpts);
}

std::unique_ptr<HiveIndexReader::Result> HiveIndexReader::next(
    vector_size_t maxOutputRows) {
  VELOX_CHECK_NOT_NULL(request_, "No request set. Call setRequest first.");
  VELOX_CHECK(hasNext(), "No more rows to process.");

  SCOPE_EXIT {
    if (requestRow_ == request_->size()) {
      reset();
    }
  };

  std::vector<RowVectorPtr> outputChunks;
  std::vector<vector_size_t> inputIndices;
  vector_size_t numOutputRows{0};
  const vector_size_t numRequestRows = request_->size();
  // Check the limit before starting a new input row, but once we start
  // processing an input row, produce all its output rows.
  for (; requestRow_ < numRequestRows && numOutputRows < maxOutputRows;
       ++requestRow_) {
    applyFiltersFromRequest(requestRow_);

    // Read all matching rows for this input row regardless of the limit.
    // We need a loop here because rowReader_->next() may return partial results
    // when crossing stripe boundaries.
    while (true) {
      VectorPtr output;
      const uint64_t numRows = readNext(output);
      if (numRows == 0) {
        break;
      }
      if (output->size() == 0) {
        continue;
      }

      numOutputRows += output->size();
      // Record the input row index for each output row.
      for (uint64_t i = 0; i < output->size(); ++i) {
        inputIndices.push_back(requestRow_);
      }
      // Load all lazy vectors because lazy loading doesn't work across batches.
      output->loadedVector();
      outputChunks.push_back(
          std::dynamic_pointer_cast<RowVector>(std::move(output)));
    }

    clearKeyFilters();
  }

  if (numOutputRows == 0) {
    return nullptr;
  }

  // Populate inputHits buffer.
  auto inputHits = AlignedBuffer::allocate<vector_size_t>(numOutputRows, pool_);
  auto* rawInputHits = inputHits->asMutable<vector_size_t>();
  std::copy(inputIndices.begin(), inputIndices.end(), rawInputHits);

  RowVectorPtr output;
  if (outputChunks.size() == 1) {
    output = std::move(outputChunks[0]);
  } else {
    output = BaseVector::create<RowVector>(outputType_, numOutputRows, pool_);
    vector_size_t offset = 0;
    for (const auto& chunk : outputChunks) {
      output->copy(chunk.get(), offset, 0, chunk->size());
      offset += chunk->size();
    }
  }
  return std::make_unique<Result>(std::move(inputHits), std::move(output));
}

void HiveIndexReader::setRequest(const Request& request) {
  VELOX_CHECK_NULL(request_, "Request already set. Call reset first.");
  VELOX_CHECK_EQ(requestRow_, 0, "Request already set. Call reset first.");
  request_ = request.input;
  VELOX_CHECK_NOT_NULL(request_);
  VELOX_CHECK(requestType_->equivalent(*request_->type()));
  requestRow_ = 0;

  // Decode all input vectors used in join conditions.
  SelectivityVector allRows(request_->size());
  for (size_t i = 0; i < joinConditions_.size(); ++i) {
    const auto& indices = requestColumnIndices_[i];
    for (size_t j = 0; j < indices.size(); ++j) {
      decodedRequestVectors_[i][j].decode(
          *request_->childAt(indices[j]), allRows);
    }
  }
}

bool HiveIndexReader::hasNext() const {
  return request_ != nullptr && requestRow_ < request_->size();
}

void HiveIndexReader::reset() {
  request_.reset();
  requestRow_ = 0;
}

void HiveIndexReader::applyFiltersFromRequest(vector_size_t row) {
  VELOX_CHECK_NOT_NULL(request_);
  VELOX_CHECK_EQ(decodedRequestVectors_.size(), joinConditions_.size());
  VELOX_CHECK_LT(row, request_->size());

  for (size_t i = 0; i < joinConditions_.size(); ++i) {
    auto* spec = joinIndexColumnSpecs_[i];
    const auto& condition = joinConditions_[i];

    if (auto equalCondition =
            std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
                condition)) {
      VELOX_CHECK_EQ(decodedRequestVectors_[i].size(), 1);
      const auto& decoded = decodedRequestVectors_[i][0];
      VELOX_CHECK(
          !decoded.isNullAt(row), "Null value not allowed for equal condition");
      const auto& type = requestType_->childAt(requestColumnIndices_[i][0]);
      const auto value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          extractValue, type->kind(), decoded, row);
      spec->setFilter(createPointFilter(type, value));
    } else if (
        auto betweenCondition =
            std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
                condition)) {
      const auto& lowerDecoded = decodedRequestVectors_[i][0];
      VELOX_CHECK(
          !lowerDecoded.isNullAt(row),
          "Null value not allowed for lower bound");
      const auto& upperDecoded = decodedRequestVectors_[i][1];
      VELOX_CHECK(
          !upperDecoded.isNullAt(row),
          "Null value not allowed for upper bound");
      const auto& type = requestType_->childAt(requestColumnIndices_[i][0]);
      const auto lower = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          extractValue, type->kind(), lowerDecoded, row);
      const auto upper = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          extractValue, type->kind(), upperDecoded, row);
      spec->setFilter(createRangeFilter(type, lower, upper));
    } else {
      VELOX_FAIL("Unsupported join condition type: {}", condition->toString());
    }
  }
  scanSpec_->resetCachedValues(/*doReorder=*/false);
  // Resets the row reader for the new index lookup. This resets internal state
  // and re-applies index bounds based on the updated filters set above.
  rowReader_->reset();
}

void HiveIndexReader::clearKeyFilters() {
  for (auto* spec : joinIndexColumnSpecs_) {
    spec->setFilter(nullptr);
  }
}

uint64_t HiveIndexReader::readNext(VectorPtr& output) {
  VELOX_CHECK_NOT_NULL(rowReader_);
  // Pre-allocate output vector with correct schema to ensure the reader returns
  // results with expected schema even when no rows match.
  if (output == nullptr) {
    output = BaseVector::create(outputType_, 0, pool_);
  }
  return rowReader_->next(1024, output);
}

void HiveIndexReader::resetFilterCaches() {
  VELOX_CHECK_NOT_NULL(rowReader_);
  rowReader_->resetFilterCaches();
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
