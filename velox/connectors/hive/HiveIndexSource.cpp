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
#include "velox/connectors/hive/HiveIndexSource.h"

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/FileIndexReader.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/LazyVector.h"

namespace facebook::velox::connector::hive {
namespace {

// Extracts a constant value from a point lookup filter (single value equality).
// Returns nullopt if the filter is not a point lookup or cannot be converted.
std::optional<variant> extractPointLookupValue(const common::Filter* filter) {
  VELOX_CHECK_NOT_NULL(filter);

  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      const auto* range = filter->as<common::BigintRange>();
      if (range->isSingleValue()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kDoubleRange: {
      const auto* range = filter->as<common::DoubleRange>();
      if (!range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kFloatRange: {
      const auto* range = filter->as<common::FloatRange>();
      if (!range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBytesRange: {
      const auto* range = filter->as<common::BytesRange>();
      if (range->isSingleValue()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBytesValues: {
      const auto* values = filter->as<common::BytesValues>();
      if (values->values().size() == 1) {
        return variant(*values->values().begin());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBoolValue: {
      const auto* boolFilter = filter->as<common::BoolValue>();
      // BoolValue doesn't expose value() getter - use testBool to determine the
      // value
      return variant(boolFilter->testBool(true));
    }
    default:
      return std::nullopt;
  }
}

// Extracts range bounds from a range filter.
// Returns a pair of (lower, upper) variants. If a bound is unbounded, the
// corresponding variant will be null.
// Returns nullopt if the filter is not a range filter or cannot be converted.
std::optional<std::pair<variant, variant>> extractRangeBounds(
    const common::Filter* filter) {
  VELOX_CHECK_NOT_NULL(filter);

  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      const auto* range = filter->as<common::BigintRange>();
      return std::make_pair(variant(range->lower()), variant(range->upper()));
    }
    case common::FilterKind::kDoubleRange: {
      const auto* range = filter->as<common::DoubleRange>();
      if (range->lowerUnbounded() || range->upperUnbounded()) {
        // Cannot convert unbounded ranges to BetweenCondition.
        return std::nullopt;
      }
      return std::make_pair(variant(range->lower()), variant(range->upper()));
    }
    case common::FilterKind::kFloatRange: {
      const auto* range = filter->as<common::FloatRange>();
      if (range->lowerUnbounded() || range->upperUnbounded()) {
        return std::nullopt;
      }
      return std::make_pair(variant(range->lower()), variant(range->upper()));
    }
    case common::FilterKind::kBytesRange: {
      const auto* range = filter->as<common::BytesRange>();
      if (!range->isSingleValue()) {
        // Only convert bounded range filters.
        return std::make_pair(variant(range->lower()), variant(range->upper()));
      }
      return std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

// Creates an EqualIndexLookupCondition with a constant value.
core::IndexLookupConditionPtr createEqualConditionWithConstant(
    const std::string& columnName,
    const TypePtr& type,
    const variant& value) {
  auto keyExpr = std::make_shared<core::FieldAccessTypedExpr>(type, columnName);
  auto constantExpr = std::make_shared<core::ConstantTypedExpr>(type, value);
  return std::make_shared<core::EqualIndexLookupCondition>(
      std::move(keyExpr), std::move(constantExpr));
}

// Creates a BetweenIndexLookupCondition with constant bounds.
core::IndexLookupConditionPtr createBetweenConditionWithConstants(
    const std::string& columnName,
    const TypePtr& type,
    const variant& lowerValue,
    const variant& upperValue) {
  auto keyExpr = std::make_shared<core::FieldAccessTypedExpr>(type, columnName);
  auto lowerExpr = std::make_shared<core::ConstantTypedExpr>(type, lowerValue);
  auto upperExpr = std::make_shared<core::ConstantTypedExpr>(type, upperValue);
  return std::make_shared<core::BetweenIndexLookupCondition>(
      std::move(keyExpr), std::move(lowerExpr), std::move(upperExpr));
}

// Checks that a HiveColumnHandle is a regular column type.
void checkColumnHandleIsRegular(const FileColumnHandle& handle) {
  VELOX_CHECK_EQ(
      handle.columnType(),
      FileColumnHandle::ColumnType::kRegular,
      "Expected regular column, got {} for column {}",
      FileColumnHandle::columnTypeName(handle.columnType()),
      handle.name());
}

// Gets the table column name from assignments for a given input column name.
// Throws if not found in assignments.
std::string getTableColumnName(
    const std::string& inputColumnName,
    const connector::ColumnHandleMap& assignments) {
  auto it = assignments.find(inputColumnName);
  VELOX_USER_CHECK(
      it != assignments.end(),
      "Column '{}' not found in assignments",
      inputColumnName);
  const auto* handle =
      checkedPointerCast<const HiveColumnHandle>(it->second.get());
  return handle->name();
}

// Creates a new FieldAccessTypedExpr with the given column name but same type.
core::FieldAccessTypedExprPtr renameFieldAccess(
    const core::FieldAccessTypedExprPtr& field,
    const std::string& newName) {
  return std::make_shared<core::FieldAccessTypedExpr>(field->type(), newName);
}

// Converts an index lookup condition's key name from input column name to table
// column name. Returns a new condition with the converted key name.
core::IndexLookupConditionPtr convertConditionKeyName(
    const core::IndexLookupConditionPtr& condition,
    const connector::ColumnHandleMap& assignments) {
  const auto tableColumnName =
      getTableColumnName(condition->key->name(), assignments);
  auto newKey = renameFieldAccess(condition->key, tableColumnName);

  if (auto equalCondition =
          std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
              condition)) {
    return std::make_shared<core::EqualIndexLookupCondition>(
        std::move(newKey), equalCondition->value);
  }
  if (auto betweenCondition =
          std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
              condition)) {
    return std::make_shared<core::BetweenIndexLookupCondition>(
        std::move(newKey), betweenCondition->lower, betweenCondition->upper);
  }

  VELOX_UNREACHABLE(
      "Unsupported IndexLookupCondition type: {}", condition->toString());
}

// Filters input indices based on selected indices from filter evaluation.
BufferPtr filterIndices(
    vector_size_t numRows,
    const BufferPtr& selectedIndices,
    const BufferPtr& inputIndices,
    memory::MemoryPool* pool) {
  auto resultIndices = allocateIndices(numRows, pool);
  auto* rawResultIndices = resultIndices->asMutable<vector_size_t>();
  const auto* rawSelected = selectedIndices->as<vector_size_t>();
  const auto* rawInputIndices = inputIndices->as<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawResultIndices[i] = rawInputIndices[rawSelected[i]];
  }
  return resultIndices;
}
} // namespace

namespace {

// Merges results from multiple split-level ResultIterators in inputHit order.
// Each split independently produces results with sorted inputHits. This
// iterator interleaves rows across splits to maintain the global non-decreasing
// inputHit ordering required by IndexLookupJoin (for left join missed-row
// detection and the check in prepareLookupResult).
//
// Uses a k-way merge: buffers one Result per split, then repeatedly picks
// the split with the smallest current request index and copies all its rows
// for that index before moving on.
class UnionResultIterator : public IndexSource::ResultIterator {
 public:
  UnionResultIterator(
      std::vector<std::shared_ptr<IndexSource::ResultIterator>> splitIters,
      const RowTypePtr& outputType,
      memory::MemoryPool* pool)
      : outputType_(outputType), pool_(pool) {
    VELOX_CHECK_GT(
        splitIters.size(),
        1,
        "UnionResultIterator requires at least two iterators");
    splits_.reserve(splitIters.size());
    for (auto& iter : splitIters) {
      splits_.emplace_back(std::move(iter));
    }
  }

  bool hasNext() override {
    for (const auto& split : splits_) {
      if (!split.hasExhausted()) {
        return true;
      }
    }
    return false;
  }

  std::optional<std::unique_ptr<IndexSource::Result>> next(
      vector_size_t size,
      ContinueFuture& future) override {
    // Fetch results for all non-exhausted splits that need data.
    for (auto& split : splits_) {
      if (split.needFetchResult()) {
        if (!split.fetchResult(size, future)) {
          VELOX_CHECK(future.valid(), "Async return requires a valid future");
          return std::nullopt;
        }
      }
    }

    // Merge rows from all active splits in inputHit order by repeatedly
    // picking the split with the smallest current request index and copying
    // all its rows for that index.
    auto mergedInputHits = allocateIndices(size, pool_);
    auto* rawMergedHits = mergedInputHits->asMutable<vector_size_t>();
    auto mergedOutput = BaseVector::create<RowVector>(outputType_, size, pool_);

    vector_size_t numOutput = 0;
    while (numOutput < size) {
      int minSplitIndex = -1;
      auto minRequestIndex = std::numeric_limits<vector_size_t>::max();
      for (size_t i = 0; i < splits_.size(); ++i) {
        if (splits_[i].hasResult() &&
            splits_[i].currentRequestIndex() < minRequestIndex) {
          minSplitIndex = static_cast<int>(i);
          minRequestIndex = splits_[i].currentRequestIndex();
        }
      }
      if (minSplitIndex < 0) {
        // All splits are exhausted with no buffered data.
        break;
      }

      auto& split = splits_[minSplitIndex];
      VELOX_CHECK_LE(numOutput, size);
      numOutput += split.fillResult(
          minRequestIndex,
          mergedOutput,
          numOutput,
          rawMergedHits,
          size - numOutput);

      // Stop if this split's buffer is consumed but not exhausted. We must
      // refill it before continuing to avoid emitting larger inputHits from
      // other splits that would violate non-decreasing order across next()
      // calls.
      if (split.needFetchResult()) {
        break;
      }
    }

    if (numOutput == 0) {
      return nullptr;
    }

    mergedInputHits->setSize(numOutput * sizeof(vector_size_t));
    mergedOutput->resize(numOutput);
    return std::make_unique<IndexSource::Result>(
        std::move(mergedInputHits), std::move(mergedOutput));
  }

 private:
  // Tracks iteration state for a single split's ResultIterator. Buffers
  // one Result at a time and tracks the current read position within it.
  struct SplitState {
    explicit SplitState(std::shared_ptr<IndexSource::ResultIterator> splitIter)
        : iter(std::move(splitIter)) {}

    const std::shared_ptr<IndexSource::ResultIterator> iter;
    // Current buffered result from this split, or nullptr if not yet fetched.
    std::unique_ptr<IndexSource::Result> result;
    // Next row to read within 'result'.
    vector_size_t resultOffset{0};
    // True when the underlying iterator has no more results.
    bool exhausted{false};

    // Returns true if there are unconsumed rows in the current buffered result.
    bool hasResult() const {
      return result != nullptr && resultOffset < result->size();
    }

    // Returns true if the underlying iterator has no more results.
    bool hasExhausted() const {
      return exhausted;
    }

    // Returns true if the split needs to fetch the next result batch.
    bool needFetchResult() const {
      return !hasResult() && !hasExhausted();
    }

    // Returns the request index (inputHit) of the current row in the buffer.
    vector_size_t currentRequestIndex() const {
      VELOX_CHECK(hasResult());
      return result->inputHits->as<const vector_size_t>()[resultOffset];
    }

    // Copies buffered rows matching the given request index to the output,
    // up to maxRows. Returns the number of rows copied.
    vector_size_t fillResult(
        vector_size_t requestIndex,
        const RowVectorPtr& output,
        vector_size_t outputOffset,
        vector_size_t* rawHits,
        vector_size_t maxRows) {
      VELOX_CHECK(hasResult());
      // Count contiguous rows with the same request index.
      const auto* hits = result->inputHits->as<const vector_size_t>();
      vector_size_t count = 0;
      while (count < maxRows && resultOffset + count < result->size() &&
             hits[resultOffset + count] == requestIndex) {
        ++count;
      }
      if (count > 0) {
        output->copy(result->output.get(), outputOffset, resultOffset, count);
        std::fill(
            rawHits + outputOffset,
            rawHits + outputOffset + count,
            requestIndex);
        resultOffset += count;
      }
      return count;
    }

    // Fetches the next non-empty result from the underlying iterator.
    // Returns true when data is ready (or split is exhausted), false if async.
    bool fetchResult(vector_size_t size, ContinueFuture& future) {
      VELOX_CHECK(
          !hasResult(), "Must consume current result before fetching next");
      while (!exhausted) {
        if (!iter->hasNext()) {
          exhausted = true;
          return true;
        }
        auto resultOpt = iter->next(size, future);
        if (!resultOpt.has_value()) {
          return false;
        }
        auto fetchedResult = std::move(resultOpt).value();
        if (fetchedResult == nullptr) {
          exhausted = true;
          return true;
        }
        // Skip empty results (e.g., when all rows are filtered out by
        // remaining filter) and continue fetching.
        if (fetchedResult->size() == 0) {
          continue;
        }
        result = std::move(fetchedResult);
        resultOffset = 0;
        return true;
      }
      VELOX_UNREACHABLE();
    }
  };

  // Output schema used to allocate merged result vectors.
  const RowTypePtr outputType_;
  memory::MemoryPool* const pool_;
  // Per-split state for buffering and tracking iteration progress. Not const
  // because elements are mutated during iteration (result, resultOffset,
  // exhausted), and const vector makes elements const via const T& access.
  std::vector<SplitState> splits_;
};

} // namespace

/// Iterates over results from a SplitIndexReader and applies HiveIndexSource's
/// format-agnostic orchestration: remaining filter evaluation and output
/// projection.
class HiveLookupIterator : public IndexSource::ResultIterator {
 public:
  HiveLookupIterator(
      std::shared_ptr<HiveIndexSource> indexSource,
      SplitIndexReader* indexReader,
      IndexSource::Request request,
      SplitIndexReader::Options options)
      : indexSource_(std::move(indexSource)),
        indexReader_(indexReader),
        request_(std::move(request)),
        options_(options) {}

  bool hasNext() override {
    return state_ != State::kEnd;
  }

  std::optional<std::unique_ptr<IndexSource::Result>> next(
      vector_size_t size,
      ContinueFuture& /*unused*/) override {
    if (state_ == State::kEnd) {
      return nullptr;
    }

    // Initialize lookup on first call.
    if (state_ == State::kInit) {
      indexReader_->startLookup(request_, options_);
      setState(State::kRead);
    }

    if (!indexReader_->hasNext()) {
      setState(State::kEnd);
      return nullptr;
    }
    return getOutput(size);
  }

 private:
  // State of the iterator.
  enum class State {
    // Initial state after creation.
    kInit,
    // After lookup request has been set in index reader.
    kRead,
    // After all data has been read.
    kEnd,
  };

  // Sets the state with validation of allowed transitions.
  // Allowed transitions:
  //   kInit -> kRead (when request is set)
  //   kInit -> kEnd (when no matches on first call)
  //   kRead -> kEnd (when all data has been read)
  void setState(State newState) {
    switch (state_) {
      case State::kInit:
        VELOX_CHECK(
            newState == State::kRead || newState == State::kEnd,
            "Invalid state transition from {} to {}",
            stateName(state_),
            stateName(newState));
        break;
      case State::kRead:
        VELOX_CHECK(
            newState == State::kEnd,
            "Invalid state transition from {} to {}",
            stateName(state_),
            stateName(newState));
        break;
      case State::kEnd:
        VELOX_FAIL(
            "Invalid state transition from {} to {}",
            stateName(state_),
            stateName(newState));
    }
    state_ = newState;
  }

  static std::string stateName(State state) {
    switch (state) {
      case State::kInit:
        return "kInit";
      case State::kRead:
        return "kRead";
      case State::kEnd:
        return "kEnd";
      default:
        VELOX_UNREACHABLE("Unknown state {}", static_cast<int>(state));
    }
  }

  std::unique_ptr<IndexSource::Result> getOutput(vector_size_t size) {
    auto result = indexReader_->next(size);
    if (result == nullptr) {
      VELOX_CHECK(!indexReader_->hasNext());
      setState(State::kEnd);
      return nullptr;
    }
    if (indexSource_->remainingFilterExprSet_ == nullptr) {
      result->output = indexSource_->projectOutput(
          result->output->size(), nullptr, result->output);
      return result;
    }
    return evaluateRemainingFilter(std::move(result));
  }

  std::unique_ptr<IndexSource::Result> evaluateRemainingFilter(
      std::unique_ptr<IndexSource::Result> result) {
    auto& output = result->output;
    const auto numRemainingRows = indexSource_->evaluateRemainingFilter(output);

    if (numRemainingRows == 0) {
      return getEmptyResult();
    }

    BufferPtr remainingIndices{nullptr};
    if (numRemainingRows != output->size()) {
      remainingIndices = indexSource_->remainingFilterEvalCtx_.selectedIndices;
      result->inputHits = filterIndices(
          numRemainingRows,
          remainingIndices,
          result->inputHits,
          indexSource_->pool_);
    }
    output =
        indexSource_->projectOutput(numRemainingRows, remainingIndices, output);
    return result;
  }

  std::unique_ptr<IndexSource::Result> getEmptyResult() {
    if (emptyResult_ == nullptr) {
      emptyResult_ = std::make_unique<IndexSource::Result>(
          allocateIndices(0, indexSource_->pool_),
          indexSource_->getEmptyOutput());
    }
    return std::make_unique<IndexSource::Result>(
        emptyResult_->inputHits, emptyResult_->output);
  }

  const std::shared_ptr<HiveIndexSource> indexSource_;
  // Raw pointer to index reader for lookup operations.
  SplitIndexReader* const indexReader_;
  const IndexSource::Request request_;
  const SplitIndexReader::Options options_;

  State state_{State::kInit};
  // Cached empty result for reuse when no rows pass the remaining filter.
  std::unique_ptr<IndexSource::Result> emptyResult_;
};

HiveIndexSource::HiveIndexSource(
    const RowTypePtr& requestType,
    const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions,
    const RowTypePtr& outputType,
    HiveTableHandlePtr tableHandle,
    const ColumnHandleMap& columnHandles,
    FileHandleFactory* fileHandleFactory,
    ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig>& hiveConfig,
    folly::Executor* executor)
    : fileHandleFactory_(fileHandleFactory),
      connectorQueryCtx_(connectorQueryCtx),
      hiveConfig_(hiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()),
      maxRowsPerIndexRequest_(hiveConfig_->maxRowsPerIndexRequest(
          connectorQueryCtx_->sessionProperties())),
      tableHandle_(std::move(tableHandle)),
      requestType_(requestType),
      outputType_(outputType),
      executor_(executor),
      ioStatistics_(std::make_shared<io::IoStatistics>()),
      ioStats_(std::make_shared<IoStats>()) {
  init(columnHandles, indexLookupConditions);
}

void HiveIndexSource::initIndexLookupConditions(
    const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions,
    const ColumnHandleMap& assignments) {
  const auto& indexColumns = tableHandle_->indexColumns();
  const auto& dataColumns = tableHandle_->dataColumns();

  // Build a map from index lookup condition key name to condition for quick
  // lookup. The key name in IndexLookupCondition references input column name,
  // we need to convert it to the table column name using assignments.
  folly::F14FastMap<std::string, core::IndexLookupConditionPtr>
      indexLookupConditionMap;
  for (const auto& condition : indexLookupConditions) {
    auto convertedCondition = convertConditionKeyName(condition, assignments);
    const auto& columnName = convertedCondition->key->name();
    VELOX_USER_CHECK(
        indexLookupConditionMap
            .emplace(columnName, std::move(convertedCondition))
            .second,
        "Duplicate lookup key found in indexLookupConditions: {}",
        columnName);
  }

  indexLookupConditions_.reserve(indexColumns.size());
  size_t numValidIndexLookupConditions{0};
  // Process index columns in order, converting filters to index lookup
  // conditions where possible. A range filter/condition stops further
  // processing.
  for (const auto& indexColumn : indexColumns) {
    const common::Subfield subfield(indexColumn);
    const auto filterIt = filters_.find(subfield);
    const bool hasFilter = filterIt != filters_.end();
    const auto conditionIt = indexLookupConditionMap.find(indexColumn);
    const bool hasIndexLookupCondition =
        conditionIt != indexLookupConditionMap.end();

    // Cannot have both a filter and an index lookup condition on the same
    // column.
    VELOX_CHECK(
        !(hasFilter && hasIndexLookupCondition),
        "Cannot have both filter and index lookup condition on index column {}",
        indexColumn);

    if (!hasFilter && !hasIndexLookupCondition) {
      // No filter or index lookup condition on this column - stop processing.
      break;
    }

    // Get column type from data columns.
    const auto typeIdx = dataColumns->getChildIdxIfExists(indexColumn);
    VELOX_CHECK(
        typeIdx.has_value(),
        "Index column {} not found in data columns",
        indexColumn);

    if (hasIndexLookupCondition) {
      // Use the existing index lookup condition as-is.
      const auto& condition = conditionIt->second;
      indexLookupConditions_.push_back(condition);
      VELOX_CHECK(!condition->isFilter());
      ++numValidIndexLookupConditions;

      // Check if this is a range condition (Between) - stops further
      // processing.
      if (std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
              condition)) {
        break;
      }
      continue;
    }

    // Has filter - try to convert to index lookup condition.
    VELOX_CHECK(hasFilter);
    const auto& columnType = dataColumns->childAt(*typeIdx);
    const auto* filter = filterIt->second.get();
    // Try point lookup conversion first.
    auto pointValue = extractPointLookupValue(filter);
    if (pointValue.has_value()) {
      auto condition = createEqualConditionWithConstant(
          indexColumn, columnType, pointValue.value());
      indexLookupConditions_.push_back(condition);
      // Remove converted filter from filters_ map.
      filters_.erase(filterIt);
      continue;
    }

    // Try range conversion.
    auto rangeBounds = extractRangeBounds(filter);
    if (rangeBounds.has_value()) {
      auto condition = createBetweenConditionWithConstants(
          indexColumn, columnType, rangeBounds->first, rangeBounds->second);
      indexLookupConditions_.push_back(condition);
      // Remove converted filter from filters_ map.
      filters_.erase(filterIt);
      // Range condition stops further processing.
      break;
    }

    // Filter cannot be converted - leave it in filters_ map and stop
    // processing.
    break;
  }

  VELOX_CHECK_EQ(
      numValidIndexLookupConditions,
      indexLookupConditions.size(),
      "Not all index lookup conditions were processed");
}

void HiveIndexSource::initRemainingFilter(
    std::vector<std::string>& readColumnNames,
    std::vector<TypePtr>& readColumnTypes) {
  VELOX_CHECK_NULL(remainingFilterExprSet_);
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter = extractFiltersFromRemainingFilter(
      tableHandle_->remainingFilter(),
      expressionEvaluator_,
      filters_,
      sampleRate);
  // TODO: support sample rate later.
  VELOX_CHECK_EQ(
      sampleRate, 1, "Sample rate is not supported for index source");

  if (remainingFilter == nullptr) {
    return;
  }

  remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);

  const auto& remainingFilterExpr = remainingFilterExprSet_->expr(0);
  folly::F14FastMap<std::string, column_index_t> columnNames;
  for (int i = 0; i < readColumnNames.size(); ++i) {
    columnNames[readColumnNames[i]] = i;
  }
  for (const auto* input : remainingFilterExpr->distinctFields()) {
    auto it = columnNames.find(input->field());
    if (it != columnNames.end()) {
      if (shouldEagerlyMaterialize(*remainingFilterExpr, *input)) {
        remainingEagerlyLoadFields_.push_back(it->second);
      }
      continue;
    }
    // Remaining filter may reference columns that are not used otherwise,
    // e.g. are not being projected out and are not used in range filters.
    // Make sure to add these columns to readerOutputType_.
    readColumnNames.push_back(input->field());
    readColumnTypes.push_back(input->type());
  }

  remainingFilterSubfields_ = remainingFilterExpr->extractSubfields();
  for (auto& subfield : remainingFilterSubfields_) {
    const auto& name = getColumnName(subfield);
    auto it = projectedSubfields_.find(name);
    if (it != projectedSubfields_.end()) {
      // Some subfields of the column are already projected out, we append
      // the remainingFilter subfield
      it->second.push_back(&subfield);
    } else if (columnNames.count(name) == 0) {
      // remainingFilter subfield's column is not projected out, we add the
      // column and append the subfield
      projectedSubfields_[name].push_back(&subfield);
    }
  }
}

void HiveIndexSource::init(
    const ColumnHandleMap& assignments,
    const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions) {
  VELOX_CHECK_NOT_NULL(tableHandle_);

  folly::F14FastMap<std::string_view, const HiveColumnHandle*> columnHandles;
  // Column handled keyed on the column alias, the name used in the query.
  for (const auto& [_, columnHandle] : assignments) {
    auto handle = checkedPointerCast<const HiveColumnHandle>(columnHandle);
    const auto [it, unique] =
        columnHandles.emplace(handle->name(), handle.get());
    VELOX_CHECK(unique, "Duplicate column handle for {}", handle->name());
    // NOTE: we only support regular column handle for hive index source.
    checkColumnHandleIsRegular(*handle);
  }

  for (const auto& handle : tableHandle_->filterColumnHandles()) {
    auto it = columnHandles.find(handle->name());
    if (it != columnHandles.end()) {
      checkColumnHandleConsistent(*handle, *it->second);
      continue;
    }
    checkColumnHandleIsRegular(*handle);
  }

  std::vector<std::string> readColumnNames;
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = it->second;
    readColumnNames.push_back(handle->name());
    for (auto& subfield : handle->requiredSubfields()) {
      VELOX_USER_CHECK_EQ(
          getColumnName(subfield),
          handle->name(),
          "Required subfield does not match column name");
      projectedSubfields_[handle->name()].push_back(&subfield);
    }
    VELOX_CHECK_NULL(handle->postProcessor(), "Post processor not supported");
  }

  if (hiveConfig_->isFileColumnNamesReadAsLowerCase(
          connectorQueryCtx_->sessionProperties())) {
    checkColumnNameLowerCase(outputType_);
    checkColumnNameLowerCase(tableHandle_->subfieldFilters(), {});
    checkColumnNameLowerCase(tableHandle_->remainingFilter());
  }

  for (const auto& [subfield, filter] : tableHandle_->subfieldFilters()) {
    filters_.emplace(subfield.clone(), filter);
  }

  initRemainingFilter(readColumnNames, readColumnTypes);

  initIndexLookupConditions(indexLookupConditions, assignments);

  readerOutputType_ =
      ROW(std::move(readColumnNames), std::move(readColumnTypes));
  scanSpec_ = makeScanSpec(
      readerOutputType_,
      projectedSubfields_,
      filters_,
      /*indexColumns=*/{},
      tableHandle_->dataColumns(),
      /*partitionKeys=*/{},
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      hiveConfig_->readStatsBasedFilterReorderDisabled(
          connectorQueryCtx_->sessionProperties()),
      pool_);
}

void HiveIndexSource::addSplits(
    std::vector<std::shared_ptr<ConnectorSplit>> splits) {
  VELOX_CHECK(
      readers_.empty(),
      "addSplits can only be called once for HiveIndexSource");

  auto* registry = IndexReaderFactoryRegistry::getInstance();
  for (auto& split : splits) {
    auto hiveSplit = checkedPointerCast<const HiveConnectorSplit>(split);
    const auto* factory = registry->getFactory(hiveSplit->fileFormat);
    if (factory != nullptr) {
      createCustomIndexReader(*factory, std::move(hiveSplit));
    } else {
      VELOX_CHECK_EQ(
          hiveSplit->fileFormat,
          dwio::common::FileFormat::NIMBLE,
          "No IndexReaderFactory registered for format: {}",
          dwio::common::toString(hiveSplit->fileFormat));
      createFileIndexReader(std::move(hiveSplit));
    }
  }

  VELOX_CHECK(!readers_.empty(), "No index readers created from splits");
}

std::shared_ptr<IndexSource::ResultIterator> HiveIndexSource::lookup(
    const Request& request) {
  VELOX_CHECK(!readers_.empty(), "No index readers available for lookup");

  auto options = SplitIndexReader::Options{
      .maxRowsPerRequest = static_cast<vector_size_t>(maxRowsPerIndexRequest_)};

  if (readers_.size() == 1) {
    return std::make_shared<HiveLookupIterator>(
        shared_from_this(), readers_[0].get(), request, options);
  }

  return createUnionLookupIterator(request, options);
}

std::shared_ptr<IndexSource::ResultIterator>
HiveIndexSource::createUnionLookupIterator(
    const Request& request,
    const SplitIndexReader::Options& options) {
  // Wraps each reader in a HiveLookupIterator and unions their results
  // via UnionResultIterator. Each reader searches its own split
  // independently with the same request.
  VELOX_CHECK_GT(readers_.size(), 1, "Union requires at least two readers");
  std::vector<std::shared_ptr<IndexSource::ResultIterator>> splitIters;
  splitIters.reserve(readers_.size());
  for (auto& reader : readers_) {
    splitIters.push_back(
        std::make_shared<HiveLookupIterator>(
            shared_from_this(), reader.get(), request, options));
  }
  return std::make_shared<UnionResultIterator>(
      std::move(splitIters), outputType_, pool_);
}

std::unordered_map<std::string, RuntimeMetric> HiveIndexSource::runtimeStats() {
  std::unordered_map<std::string, RuntimeMetric> stats;
  if (remainingFilterTimeNs_ != 0) {
    stats[std::string(Connector::kTotalRemainingFilterTime)] =
        RuntimeMetric(remainingFilterTimeNs_, RuntimeCounter::Unit::kNanos);
  }
  // Merge stats from all readers.
  for (auto& reader : readers_) {
    for (auto& [key, metric] : reader->runtimeStats()) {
      auto it = stats.find(key);
      if (it != stats.end()) {
        it->second.merge(metric);
      } else {
        stats.emplace(key, metric);
      }
    }
  }
  return stats;
}

vector_size_t HiveIndexSource::evaluateRemainingFilter(
    RowVectorPtr& rowVector) {
  remainingFilterRows_.resize(rowVector->size());
  for (auto fieldIndex : remainingEagerlyLoadFields_) {
    LazyVector::ensureLoadedRows(
        rowVector->childAt(fieldIndex),
        remainingFilterRows_,
        remainingFilterLazyDecoded_,
        remainingFilterLazyBaseRows_);
  }

  uint64_t filterTimeNs{0};
  vector_size_t numRemainingRows{0};
  {
    NanosecondTimer timer(&filterTimeNs);
    expressionEvaluator_->evaluate(
        remainingFilterExprSet_.get(),
        remainingFilterRows_,
        *rowVector,
        remainingFilterResult_);
    numRemainingRows = exec::processFilterResults(
        remainingFilterResult_,
        remainingFilterRows_,
        remainingFilterEvalCtx_,
        pool_);
  }
  remainingFilterTimeNs_ += filterTimeNs;
  return numRemainingRows;
}

RowVectorPtr HiveIndexSource::projectOutput(
    vector_size_t numRows,
    const BufferPtr& remainingIndices,
    const RowVectorPtr& rowVector) {
  if (outputType_->size() == 0) {
    return exec::wrap(numRows, remainingIndices, rowVector);
  }

  std::vector<VectorPtr> outputColumns;
  outputColumns.reserve(outputType_->size());
  for (int i = 0; i < outputType_->size(); ++i) {
    auto& child = rowVector->childAt(i);
    if (remainingIndices) {
      // Disable dictionary values caching in expression eval so that we
      // don't need to reallocate the result for every batch.
      child->disableMemo();
    }
    auto column = exec::wrapChild(numRows, remainingIndices, child);
    outputColumns.push_back(std::move(column));
  }

  return std::make_shared<RowVector>(
      pool_, outputType_, BufferPtr(nullptr), numRows, outputColumns);
}

void HiveIndexSource::createCustomIndexReader(
    const IndexReaderFactory& factory,
    std::shared_ptr<const HiveConnectorSplit> split) {
  VELOX_CHECK_NOT_NULL(split);
  auto reader = factory(split, tableHandle_, connectorQueryCtx_);
  VELOX_CHECK_NOT_NULL(
      reader,
      "IndexReaderFactory returned null for format: {}",
      dwio::common::toString(split->fileFormat));
  readers_.push_back(std::move(reader));
}

void HiveIndexSource::createFileIndexReader(
    std::shared_ptr<const HiveConnectorSplit> split) {
  VELOX_CHECK_NOT_NULL(split);
  readers_.push_back(
      std::make_unique<FileIndexReader>(
          std::move(split),
          tableHandle_,
          connectorQueryCtx_,
          hiveConfig_,
          scanSpec_,
          indexLookupConditions_,
          requestType_,
          readerOutputType_,
          ioStatistics_,
          ioStats_,
          fileHandleFactory_,
          executor_,
          maxRowsPerIndexRequest_));
}

} // namespace facebook::velox::connector::hive
