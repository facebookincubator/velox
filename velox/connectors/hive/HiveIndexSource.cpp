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
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/LazyVector.h"

namespace facebook::velox::connector::hive {
namespace {
// Checks that a HiveColumnHandle is a regular column type.
void checkColumnHandleIsRegular(const HiveColumnHandle& handle) {
  VELOX_CHECK_EQ(
      handle.columnType(),
      HiveColumnHandle::ColumnType::kRegular,
      "Expected regular column, got {} for column {}",
      HiveColumnHandle::columnTypeName(handle.columnType()),
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

class HiveLookupIterator : public IndexSource::ResultIterator {
 public:
  HiveLookupIterator(
      std::shared_ptr<HiveIndexSource> indexSource,
      HiveIndexReader* indexReader,
      IndexSource::Request request)
      : indexSource_(std::move(indexSource)),
        indexReader_(indexReader),
        request_(std::move(request)) {}

  ~HiveLookupIterator() override = default;

  bool hasNext() override {
    return state_ != State::kEnd;
  }

  std::optional<std::unique_ptr<IndexSource::Result>> next(
      vector_size_t size,
      ContinueFuture& /*unused*/) override {
    if (state_ == State::kEnd) {
      return nullptr;
    }

    // Set the request on first call.
    if (state_ == State::kInit) {
      indexReader_->setRequest(request_);
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

  // Holds the index source to ensure HiveIndexReader lifetime.
  const std::shared_ptr<HiveIndexSource> indexSource_;
  // Raw pointer to index reader for lookup operations.
  HiveIndexReader* const indexReader_;
  const IndexSource::Request request_;

  State state_{State::kInit};
  // Cached empty result for reuse when no rows pass the remaining filter.
  std::unique_ptr<IndexSource::Result> emptyResult_;
};

HiveIndexSource::HiveIndexSource(
    const RowTypePtr& requestType,
    const std::vector<core::IndexLookupConditionPtr>& joinConditions,
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
      tableHandle_(std::move(tableHandle)),
      requestType_(requestType),
      outputType_(outputType),
      executor_(executor),
      ioStatistics_(std::make_shared<io::IoStatistics>()),
      ioStats_(std::make_shared<IoStats>()) {
  init(columnHandles, joinConditions);
}

std::vector<std::string> HiveIndexSource::initJoinConditions(
    const std::vector<core::IndexLookupConditionPtr>& joinConditions,
    const ColumnHandleMap& assignments) {
  const auto& indexColumns = tableHandle_->indexColumns();
  VELOX_USER_CHECK_LE(
      joinConditions.size(),
      indexColumns.size(),
      "joinConditions ({}) exceeds the number of index columns ({})",
      joinConditions.size(),
      indexColumns.size());

  std::vector<std::string> joinColumns;
  joinColumns.reserve(joinConditions.size());
  joinConditions_.reserve(joinConditions.size());

  folly::F14FastSet<std::string> seenJoinKeys;
  // Process original join conditions.
  // Filter conditions (with constant values) are converted to filters for
  // better pushdown.
  for (const auto& condition : joinConditions) {
    const auto columnName =
        getTableColumnName(condition->key->name(), assignments);
    VELOX_USER_CHECK(
        seenJoinKeys.insert(columnName).second,
        "Duplicate join key found in joinConditions: {}",
        columnName);
    const common::Subfield subfield(columnName);
    VELOX_CHECK(
        filters_.find(subfield) == filters_.end(),
        "Unexpected filter found on index column {}",
        columnName);
    VELOX_CHECK(
        !condition->isFilter(),
        "Join condition on index column '{}' cannot be a filter condition",
        columnName);

    joinConditions_.push_back(condition);
    joinColumns.push_back(columnName);
  }
  return joinColumns;
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
    const std::vector<core::IndexLookupConditionPtr>& joinConditions) {
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

  const auto joinIndexColumns = initJoinConditions(joinConditions, assignments);

  readerOutputType_ =
      ROW(std::move(readColumnNames), std::move(readColumnTypes));
  scanSpec_ = makeScanSpec(
      readerOutputType_,
      projectedSubfields_,
      filters_,
      joinIndexColumns,
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
  VELOX_CHECK_NULL(
      indexReader_, "addSplits can only be called once for HiveIndexSource");
  std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits;
  hiveSplits.reserve(splits.size());
  for (auto& split : splits) {
    auto hiveSplit = checkedPointerCast<const HiveConnectorSplit>(split);
    VELOX_CHECK_EQ(
        hiveSplit->fileFormat,
        dwio::common::FileFormat::NIMBLE,
        "HiveIndexSource only supports Nimble file format");
    hiveSplits.push_back(hiveSplit);
  }
  createHiveIndexReader(std::move(hiveSplits));
}

std::shared_ptr<IndexSource::ResultIterator> HiveIndexSource::lookup(
    const Request& request) {
  VELOX_CHECK_NOT_NULL(indexReader_, "No index reader available for lookup");
  return std::make_shared<HiveLookupIterator>(
      shared_from_this(), indexReader_.get(), request);
}

std::unordered_map<std::string, RuntimeMetric> HiveIndexSource::runtimeStats() {
  std::unordered_map<std::string, RuntimeMetric> stats;
  if (remainingFilterTimeNs_ != 0) {
    stats[Connector::kTotalRemainingFilterTime] =
        RuntimeMetric(remainingFilterTimeNs_, RuntimeCounter::Unit::kNanos);
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

void HiveIndexSource::createHiveIndexReader(
    std::vector<std::shared_ptr<const HiveConnectorSplit>> splits) {
  VELOX_CHECK(!splits.empty(), "No splits available");
  indexReader_ = std::make_unique<HiveIndexReader>(
      std::move(splits),
      tableHandle_,
      connectorQueryCtx_,
      hiveConfig_,
      scanSpec_,
      joinConditions_,
      requestType_,
      readerOutputType_,
      ioStatistics_,
      ioStats_,
      fileHandleFactory_,
      executor_);
}

} // namespace facebook::velox::connector::hive
