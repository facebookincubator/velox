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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/expression/FieldReference.h"

#include <cudf/stream_compaction.hpp>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241}, // CudfHive blue,
          std::nullopt,
          fmt::format("[{}]", tableHandle->name())),
      cudfHiveConfig_(cudfHiveConfig),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      outputType_(outputType),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Set up column projection if needed
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    readColumnSet_.emplace(handle->name());
    readColumnNames_.emplace_back(handle->name());
  }

  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Copy subfield filters
  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    subfieldFilters_.emplace(k.clone(), v->clone());
  }

  // Extract additional simple filters from remainingFilter (same as CPU path).
  // This extracts single-column filters like "col = 'X'" or "col <> 'Y'" from
  // complex expressions and adds them to subfieldFilters_ for pushdown.
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter =
      facebook::velox::connector::hive::extractFiltersFromRemainingFilter(
          tableHandle_->remainingFilter(),
          expressionEvaluator_,
          subfieldFilters_,
          sampleRate);

  // Add fields in the filter to the columns to read if not there
  for (const auto& [field, _] : subfieldFilters_) {
    if (readColumnSet_.count(field.toString()) == 0) {
      readColumnSet_.emplace(field.toString());
      readColumnNames_.emplace_back(field.toString());
    }
  }
  if (remainingFilter) {
    remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);
    for (const auto& field : remainingFilterExprSet_->distinctFields()) {
      // Add fields in the filter to the columns to read if not there
      if (readColumnSet_.count(field->name()) == 0) {
        readColumnSet_.emplace(field->name());
        readColumnNames_.emplace_back(field->name());
      }
    }

    const RowTypePtr remainingFilterType_ = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> new_names;
        std::vector<TypePtr> new_types;

        for (const auto& name : readColumnNames_) {
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          new_names.emplace_back(std::move(name));
          new_types.push_back(parsedType);
        }

        return ROW(std::move(new_names), std::move(new_types));
      } else {
        return outputType_;
      }
    }();

    cudfExpressionEvaluator_ = velox::cudf_velox::createCudfExpression(
        remainingFilterExprSet_->exprs()[0], remainingFilterType_);
    // TODO(kn): Get column names and subfields from remaining filter and add to
    // readColumnNames_
  }

  // Build a combined AST for all subfield filters once. This is query-constant
  // and doesn't depend on split-specific state.
  if (!subfieldFilters_.empty()) {
    const RowTypePtr readerFilterType = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> newNames;
        std::vector<TypePtr> newTypes;

        for (const auto& name : readColumnNames_) {
          // Ensure all columns being read are available to the filter.
          auto parsedType = tableHandle_->dataColumns()->findChild(name);
          newNames.emplace_back(std::move(name));
          newTypes.push_back(parsedType);
        }

        return ROW(std::move(newNames), std::move(newTypes));
      } else {
        return outputType_;
      }
    }();

    subfieldFilterExpr_ = &createAstFromSubfieldFilters(
        subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<facebook::velox::IoStats>();

  // Whether to use the experimental cuDF reader
  useExperimentalCudfReader_ =
      cudfHiveConfig_->useExperimentalCudfReaderSession(
          connectorQueryCtx_->sessionProperties());
}

std::unique_ptr<CudfSplitReader> CudfHiveDataSource::createCudfSplitReader() {
  return std::make_unique<CudfSplitReader>(
      split_,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      ioStatistics_,
      ioStats_,
      useExperimentalCudfReader_,
      subfieldFilterExpr_);
}

void CudfHiveDataSource::convertSplit(std::shared_ptr<ConnectorSplit> split) {
  // Dynamic cast split to `CudfHiveConnectorSplit`
  if (std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split)) {
    split_ = std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split);
    return;
  }

  // Convert `HiveConnectorSplit` to `CudfHiveConnectorSplit`
  auto hiveSplit = checkedPointerCast<hive::HiveConnectorSplit>(split);

  VELOX_CHECK_EQ(
      hiveSplit->fileFormat,
      dwio::common::FileFormat::PARQUET,
      "Unsupported file format for conversion from HiveConnectorSplit to CudfHiveConnectorSplit");

  // Remove "file:" prefix from the file path if present
  std::string cleanedPath = hiveSplit->filePath;
  constexpr std::string_view kFilePrefix = "file:";
  constexpr std::string_view kS3APrefix = "s3a:";
  if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    cleanedPath = cleanedPath.substr(kFilePrefix.size());
  } else if (cleanedPath.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
    // KvikIO does not support "s3a:" prefix. We need to translate it to "s3:".
    cleanedPath.erase(kS3APrefix.size() - 2, 1);
  }

  auto cudfHiveSplitBuilder = CudfHiveConnectorSplitBuilder(cleanedPath)
                                  .start(hiveSplit->start)
                                  .length(hiveSplit->length)
                                  .connectorId(hiveSplit->connectorId)
                                  .splitWeight(hiveSplit->splitWeight);
  for (auto const& infoColumn : hiveSplit->infoColumns) {
    cudfHiveSplitBuilder.infoColumn(infoColumn.first, infoColumn.second);
  }
  split_ = cudfHiveSplitBuilder.build();

  VLOG(1) << "Adding split " << split_->toString();
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  // Virtual method for class-specific conversion of the split
  convertSplit(split);

  cudfSplitReader_ = createCudfSplitReader();
  cudfSplitReader_->prepareSplit(runtimeStats_);

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    auto const fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    if (fileHandleCachePtr.get() and fileHandleCachePtr.get()->file) {
      completedBytes_ += fileHandleCachePtr->file->size();
    }
  } catch (const std::exception& e) {
    // Unable to get the file size, log a warning and continue
    LOG(WARNING) << "Failed to get file size for " << split_->filePath << ": "
                 << e.what();
  }
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /* future */) {
  VELOX_CHECK_NOT_NULL(split_, "No split present. Call addSplit() first.");
  VELOX_CHECK_NOT_NULL(cudfSplitReader_, "No split to process.");
  auto chunkOpt = cudfSplitReader_->next(size);
  if (!chunkOpt.has_value()) {
    return nullptr;
  }
  auto cudfTable = std::move(chunkOpt.value());
  auto stream = cudfSplitReader_->stream();

  uint64_t filterTimeUs{0};
  if (remainingFilterExprSet_) {
    MicrosecondTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(cudfTableColumns.size());
    for (auto& col : cudfTableColumns) {
      inputViews.push_back(col->view());
    }
    auto filterResult =
        cudfExpressionEvaluator_->eval(inputViews, stream, get_temp_mr());
    auto originalTable =
        std::make_unique<cudf::table>(std::move(cudfTableColumns));
    cudfTable = cudf::apply_boolean_mask(
        *originalTable, asView(filterResult), stream, get_output_mr());
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  const auto nRows = cudfTable->num_rows();

  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> outputColumns;
    outputColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(outputColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(outputColumns));
  }

  // TODO (dm): Should we only enable table scan if cudf is registered?
  // Earlier we could enable cudf table scans without using other cudf operators
  // We still can, but I'm wondering if this is the right thing to do
  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_, stream, get_temp_mr());
  stream.synchronize();

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto result = runtimeStats_.toRuntimeMetricMap();
  result.insert({
      {std::string(connector::hive::HiveDataSource::kTotalScanTime),
       RuntimeMetric(
           ioStatistics_->totalScanTime(), RuntimeCounter::Unit::kNanos)},
      {std::string(Connector::kTotalRemainingFilterTime),
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  const auto& ioStats = ioStats_->stats();
  for (const auto& storageStats : ioStats) {
    result.emplace(storageStats.first, storageStats.second);
  }
  return result;
}

} // namespace facebook::velox::cudf_velox::connector::hive
