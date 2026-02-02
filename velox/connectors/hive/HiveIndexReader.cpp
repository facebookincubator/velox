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
  VELOX_CHECK(indexColumnSpecs_.empty());
  VELOX_CHECK(requestColumnIndices_.empty());

  indexColumnSpecs_.reserve(joinConditions_.size());
  requestColumnIndices_.reserve(joinConditions_.size());
  decodedVectors_.resize(joinConditions_.size());

  auto getRequestColumnIndex = [&](const std::string& name) {
    const auto idxOpt = requestType_->getChildIdxIfExists(name);
    VELOX_CHECK(
        idxOpt.has_value(),
        "Request column {} not found in request type",
        name);
    return idxOpt.value();
  };

  // Validate join conditions:
  // 1. Index columns must follow the order of table index columns (prefix).
  // 2. Equal conditions must come before between conditions.
  const auto& tableIndexColumns = tableHandle_->indexColumns();
  bool seenBetweenCondition = false;
  for (size_t i = 0; i < joinConditions_.size(); ++i) {
    const auto& condition = joinConditions_[i];
    VELOX_CHECK(!condition->isFilter());

    const auto& keyName = condition->key->name();
    VELOX_CHECK_LT(
        i,
        tableIndexColumns.size(),
        "Join condition index column '{}' exceeds the number of table index columns {}",
        keyName,
        tableIndexColumns.size());
    VELOX_CHECK_EQ(
        keyName,
        tableIndexColumns[i],
        "Join condition index column '{}' at position {} does not match table index column '{}'. "
        "Join conditions must use index columns in order as a prefix.",
        keyName,
        i,
        tableIndexColumns[i]);

    const bool isBetweenCondition =
        std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
            condition) != nullptr;
    if (isBetweenCondition) {
      seenBetweenCondition = true;
    } else {
      VELOX_CHECK(
          !seenBetweenCondition,
          "Equal join condition on index column '{}' cannot come after a between condition. "
          "Equal conditions must precede between conditions.",
          keyName);
    }
    auto* spec = scanSpec_->childByName(keyName);
    VELOX_CHECK_NOT_NULL(
        spec, "Index column {} not found in scan spec", keyName);
    VELOX_CHECK(
        !spec->hasFilter(),
        "Index column '{}' used in join condition cannot have a pushdown filter",
        keyName);
    indexColumnSpecs_.push_back(spec);

    // Determine the request column index for the condition value.
    if (auto equalCondition =
            std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
                condition)) {
      const auto requestFieldAccess =
          checkedPointerCast<const core::FieldAccessTypedExpr>(
              equalCondition->value);
      requestColumnIndices_.push_back(
          {getRequestColumnIndex(requestFieldAccess->name())});
      decodedVectors_[i].resize(1);
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
      decodedVectors_[i].resize(2);
      continue;
    }

    VELOX_FAIL("Unsupported join condition type: {}", condition->toString());
  }
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
      decodedVectors_[i][j].decode(*request_->childAt(indices[j]), allRows);
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
  VELOX_CHECK_EQ(decodedVectors_.size(), joinConditions_.size());
  VELOX_CHECK_LT(row, request_->size());

  for (size_t i = 0; i < joinConditions_.size(); ++i) {
    auto* spec = indexColumnSpecs_[i];
    const auto& condition = joinConditions_[i];

    if (auto equalCondition =
            std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
                condition)) {
      VELOX_CHECK_EQ(decodedVectors_[i].size(), 1);
      const auto& decoded = decodedVectors_[i][0];
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
      const auto& lowerDecoded = decodedVectors_[i][0];
      VELOX_CHECK(
          !lowerDecoded.isNullAt(row),
          "Null value not allowed for lower bound");
      const auto& upperDecoded = decodedVectors_[i][1];
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
  for (auto* spec : indexColumnSpecs_) {
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
