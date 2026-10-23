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
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/Casts.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/FileColumnHandle.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/PartitionValue.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/Filter.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>

#include <algorithm>
#include <limits>
#include <span>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

std::optional<cudf::type_id> parquetIntegerType(
    const cudf::io::parquet::SchemaElement& schema) {
  using LogicalType = cudf::io::parquet::LogicalType;
  using ParquetType = cudf::io::parquet::Type;

  if (schema.logical_type.has_value()) {
    if (schema.logical_type->type != LogicalType::INTEGER) {
      return std::nullopt;
    }
    const auto isSigned = schema.logical_type->is_signed();
    switch (schema.logical_type->bit_width()) {
      case 8:
        return isSigned ? cudf::type_id::INT8 : cudf::type_id::UINT8;
      case 16:
        return isSigned ? cudf::type_id::INT16 : cudf::type_id::UINT16;
      case 32:
        return isSigned ? cudf::type_id::INT32 : cudf::type_id::UINT32;
      case 64:
        return isSigned ? cudf::type_id::INT64 : cudf::type_id::UINT64;
      default:
        VELOX_FAIL(
            "Invalid Parquet integer bit width: {}",
            schema.logical_type->bit_width());
    }
  }
  switch (schema.type) {
    case ParquetType::INT32:
      return cudf::type_id::INT32;
    case ParquetType::INT64:
      return cudf::type_id::INT64;
    default:
      return std::nullopt;
  }
}

std::optional<SubfieldFilterDecimalType> parquetDecimalType(
    const cudf::io::parquet::SchemaElement& schema) {
  using ParquetType = cudf::io::parquet::Type;

  if (!schema.logical_type.has_value() ||
      schema.logical_type->type != cudf::io::parquet::LogicalType::DECIMAL) {
    if (const auto integerType = parquetIntegerType(schema)) {
      return SubfieldFilterDecimalType{*integerType, 0};
    }
    return std::nullopt;
  }

  const auto scale = schema.decimal_scale;
  VELOX_CHECK_EQ(
      schema.logical_type->scale(),
      scale,
      "Parquet decimal logical type and schema element scales differ");
  auto decimalType = [&]() -> cudf::type_id {
    switch (schema.type) {
      case ParquetType::INT32:
        return cudf::type_id::DECIMAL32;
      case ParquetType::INT64:
        return cudf::type_id::DECIMAL64;
      case ParquetType::FIXED_LEN_BYTE_ARRAY:
        VELOX_CHECK_GT(
            schema.type_length,
            0,
            "Invalid fixed-length Parquet decimal width");
        if (std::cmp_less_equal(schema.type_length, sizeof(int32_t))) {
          return cudf::type_id::DECIMAL32;
        }
        if (std::cmp_less_equal(schema.type_length, sizeof(int64_t))) {
          return cudf::type_id::DECIMAL64;
        }
        if (std::cmp_less_equal(schema.type_length, sizeof(int128_t))) {
          return cudf::type_id::DECIMAL128;
        }
        VELOX_FAIL(
            "Unsupported fixed-length Parquet decimal width: {}",
            schema.type_length);
      case ParquetType::BYTE_ARRAY: {
        const auto precision = schema.logical_type->precision();
        VELOX_CHECK_GT(
            precision, 0, "Parquet decimal is missing a valid precision");
        VELOX_CHECK_LE(
            precision,
            LongDecimalType::kMaxPrecision,
            "Parquet decimal precision exceeds the maximum supported precision");
        if (precision <= std::numeric_limits<int32_t>::digits10) {
          return cudf::type_id::DECIMAL32;
        }
        if (precision <= std::numeric_limits<int64_t>::digits10) {
          return cudf::type_id::DECIMAL64;
        }
        return cudf::type_id::DECIMAL128;
      }
      default:
        VELOX_FAIL(
            "Unsupported physical Parquet type for decimal column: {}",
            static_cast<int32_t>(schema.type));
    }
  }();
  return SubfieldFilterDecimalType{decimalType, scale};
}

SubfieldFilterDecimalTypes parquetDecimalTypes(
    std::span<const cudf::io::parquet::SchemaElement> metadataSchema,
    const RowTypePtr& readerSchema,
    bool fileColumnNamesReadAsLowerCase) {
  VELOX_CHECK(
      !metadataSchema.empty(), "Cannot build a filter from an empty schema");

  SubfieldFilterDecimalTypes decimalTypes;
  for (const auto childIndex : metadataSchema.front().children_idx) {
    VELOX_CHECK_LT(
        childIndex,
        metadataSchema.size(),
        "Parquet schema child index out of range");
    const auto& child = metadataSchema[childIndex];
    const auto name = fileColumnNamesReadAsLowerCase
        ? functions::stringImpl::utf8StrToLowerCopy(child.name)
        : child.name;
    if (!readerSchema->containsChild(name)) {
      continue;
    }
    const auto& logicalType = readerSchema->findChild(name);
    if (logicalType->isDecimal()) {
      if (auto decimalType = parquetDecimalType(child)) {
        if (cudf::is_fixed_point(cudf::data_type{decimalType->type})) {
          const auto [_, tableScale] = getDecimalPrecisionScale(*logicalType);
          VELOX_CHECK_LE(
              decimalType->scale,
              tableScale,
              "Parquet decimal scale of '{}' exceeds the table scale",
              name);
        }
        decimalTypes.emplace(name, *decimalType);
      }
    }
  }
  return decimalTypes;
}

// Returns whether a constant-folded filter keeps every row. A null constant
// keeps none, matching SQL three-valued logic.
bool isTrueConstant(
    const core::ConstantTypedExpr& constant,
    memory::MemoryPool* pool) {
  VELOX_USER_CHECK_EQ(
      constant.type()->kind(),
      TypeKind::BOOLEAN,
      "Remaining filter must be a boolean expression: {}",
      constant.toString());
  return !constant.isNull() &&
      constant.toConstantVector(pool)->as<ConstantVector<bool>>()->valueAt(0);
}

// Returns whether a pushed-down filter accepts a partition key value, which
// is NULL when absent.
bool testPartitionKeyFilter(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<std::string>& value,
    bool dateValueDaysSinceEpoch,
    bool timestampValueAsLocalTime) {
  if (!value.has_value()) {
    return filter.testNull();
  }
  return common::applyFilter(
      filter,
      PartitionValue::fromString(
          *value,
          *type,
          timestampValueAsLocalTime ? PartitionValue::TimestampMode::kLocalTime
                                    : PartitionValue::TimestampMode::kUtc,
          dateValueDaysSinceEpoch ? PartitionValue::DateMode::kDaysSinceEpoch
                                  : PartitionValue::DateMode::kIsoString));
}

} // namespace

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig)
    : CudfHiveDataSource(
          outputType,
          tableHandle,
          columnHandles,
          fileHandleFactory,
          executor,
          connectorQueryCtx,
          cudfHiveConfig,
          PartitionKeySource::kDataSource) {}

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    PartitionKeySource partitionKeySource)
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
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()),
      partitionKeySource_(partitionKeySource) {
  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Column handles by table column name, including those of the columns only
  // the filters read.
  std::unordered_map<std::string, const FileColumnHandle*> columnHandlesByName;
  for (const auto& [_, handle] : columnHandles) {
    const auto* fileColumnHandle =
        checkedPointerCast<const FileColumnHandle>(handle).get();
    columnHandlesByName.emplace(fileColumnHandle->name(), fileColumnHandle);
  }
  for (const auto& handle : tableHandle_->hiveFilterColumnHandles()) {
    columnHandlesByName.emplace(handle->name(), handle.get());
  }

  // Returns the handle of a partition key the data source resolves, or null.
  const auto partitionKeyHandle =
      [&](const std::string& name) -> const FileColumnHandle* {
    if (partitionKeySource_ != PartitionKeySource::kDataSource) {
      return nullptr;
    }
    const auto it = columnHandlesByName.find(name);
    return it != columnHandlesByName.end() && it->second->isPartitionKey()
        ? it->second
        : nullptr;
  };
  const auto addPartitionKey = [&](const std::string& name,
                                   const TypePtr& type,
                                   const FileColumnHandle& handle) {
    if (partitionKeys_
            .emplace(
                name,
                PartitionKey{type, handle.isPartitionDateValueDaysSinceEpoch()})
            .second) {
      columnTypes_.emplace(name, type);
    }
  };
  const auto materializePartitionKey = [&](const std::string& name) {
    if (std::find(partitionKeyNames_.begin(), partitionKeyNames_.end(), name) ==
        partitionKeyNames_.end()) {
      partitionKeyNames_.push_back(name);
    }
  };

  // Set up column projection if needed. A partition key is not in the data
  // file, so it is materialized per split instead of being read.
  for (uint32_t i = 0; i < outputType_->size(); ++i) {
    const auto& outputName = outputType_->nameOf(i);
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    const auto& name = handle->name();
    outputSourceNames_.push_back(name);
    if (const auto* partitionKey = partitionKeyHandle(name)) {
      addPartitionKey(name, outputType_->childAt(i), *partitionKey);
      materializePartitionKey(name);
      continue;
    }
    columnTypes_.emplace(name, outputType_->childAt(i));
    readColumnSet_.emplace(name);
    readColumnNames_.emplace_back(name);
  }

  // Copy subfield filters.
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

  // A filter on a partition key decides for the whole split, so it is tested
  // per split instead of being compiled into the reader's filter.
  for (auto it = subfieldFilters_.begin(); it != subfieldFilters_.end();) {
    const auto& subfield = it->first;
    const auto* partitionKey = subfield.path().size() == 1
        ? partitionKeyHandle(subfield.baseName())
        : nullptr;
    if (partitionKey == nullptr) {
      ++it;
      continue;
    }
    addPartitionKey(
        subfield.baseName(), partitionKey->dataType(), *partitionKey);
    partitionKeyFilters_.emplace(subfield.clone(), std::move(it->second));
    it = subfieldFilters_.erase(it);
  }

  // Add fields in the filter to the columns to read if not there
  for (const auto& [field, _] : subfieldFilters_) {
    if (readColumnSet_.count(field.toString()) == 0) {
      readColumnSet_.emplace(field.toString());
      readColumnNames_.emplace_back(field.toString());
    }
  }
  // Optimize (rewrites + constant folding) the remaining filter before
  // evaluator selection so CudfFunctions never see scalar-only operand sets.
  // ConnectorQueryCtx exposes no session QueryConfig, only the session time
  // zone and whether to apply it, so folding, and the initialize() of the
  // evaluators built from the folded filter, run against a local QueryCtx
  // carrying those two settings and defaults for the rest. Folding touches
  // only deterministic functions, so start_time, read by current_date, now and
  // localtime, cannot change a fold; a setting left at its default here, such
  // as the rendering of TIMESTAMP WITH TIME ZONE as VARCHAR, would.
  // TODO: Fold through the session's evaluator once expression::optimize has
  // an ExpressionEvaluator overload (facebookincubator/velox#17898), and take
  // the config createCudfExpression needs from ConnectorQueryCtx once it
  // exposes it (facebookincubator/velox#19476).
  std::unordered_map<std::string, std::string> sessionSettings{
      {core::QueryConfig::kAdjustTimestampToTimezone,
       connectorQueryCtx->adjustTimestampToTimezone() ? "true" : "false"}};
  // An empty time zone is not a valid setting; leaving it unset means UTC.
  if (!connectorQueryCtx->sessionTimezone().empty()) {
    sessionSettings.emplace(
        core::QueryConfig::kSessionTimezone,
        connectorQueryCtx->sessionTimezone());
  }
  optimizeQueryCtx_ = core::QueryCtx::create(
      nullptr, core::QueryConfig{std::move(sessionSettings)});
  optimizedRemainingFilter_ = remainingFilter
      ? expression::optimize(remainingFilter, optimizeQueryCtx_.get(), pool_)
      : nullptr;
  if (const auto constantFilter =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
              optimizedRemainingFilter_)) {
    // A filter that folds to a constant keeps every row or none, so it needs
    // no columns and no evaluation.
    remainingFilterRejectsAllRows_ = !isTrueConstant(*constantFilter, pool_);
    optimizedRemainingFilter_ = nullptr;
  }
  if (optimizedRemainingFilter_) {
    // Add fields referenced by the filter to the columns to read. Collect from
    // the optimized expression since folding may drop branches and the columns
    // they reference. Read-column order does not affect results: the data
    // source projects its output to the requested output type.
    const auto filterFields = referencedInputFields(optimizedRemainingFilter_);
    // The filter is evaluated over the columns read for it, so a filter that
    // reads none has no rows to be evaluated over.
    VELOX_USER_CHECK(
        !filterFields.empty(),
        "Remaining filter that references no column is not supported: {}",
        optimizedRemainingFilter_->toString());
    for (const auto& name : filterFields) {
      // A partition key is materialized before the filter runs.
      if (const auto* partitionKey = partitionKeyHandle(name)) {
        addPartitionKey(name, partitionKey->dataType(), *partitionKey);
        materializePartitionKey(name);
        continue;
      }
      if (readColumnSet_.count(name) == 0) {
        readColumnSet_.emplace(name);
        readColumnNames_.emplace_back(name);
      }
    }

    // TODO: Prune struct columns to the subfields referenced by the remaining
    // filter; currently the whole column is read even if only one field is
    // used.

    // The filter is already optimized and constant folded above, so compile it
    // directly.
    auto const remainingFilterType = getTableRowType();
    cudfRemainingFilterExpression_ = createCudfExpression(
        optimizedRemainingFilter_,
        remainingFilterType,
        pool_,
        optimizeQueryCtx_->queryConfig());
  }

  // Keep a logical-width AST for filters that must run after the Parquet read.
  // A second AST with split-specific physical widths is built after reading
  // each Parquet footer.
  if (!subfieldFilters_.empty()) {
    auto const readerFilterType = getTableRowType();
    hasDecimalSubfieldFilter_ =
        hasDecimalSubfieldFilter(subfieldFilters_, readerFilterType);
    subfieldFilterAst_ = &createAstFromSubfieldFilters(
        subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);
  }

  {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    names.reserve(readColumnNames_.size());
    types.reserve(readColumnNames_.size());
    for (const auto& name : readColumnNames_) {
      names.push_back(name);
      types.push_back(columnType(name));
    }
    readerOutputType_ = ROW(std::move(names), std::move(types));
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<facebook::velox::IoStats>();
}

std::unique_ptr<CudfSplitReader> CudfHiveDataSource::createCudfSplitReader() {
  return std::make_unique<CudfSplitReader>(
      split_,
      tableHandle_,
      readerOutputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      ioStatistics_,
      ioStats_,
      subfieldFilterAst_);
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
                                  .splitWeight(hiveSplit->splitWeight)
                                  .cacheable(hiveSplit->cacheable);
  for (auto const& infoColumn : hiveSplit->infoColumns) {
    cudfHiveSplitBuilder.infoColumn(infoColumn.first, infoColumn.second);
  }
  for (const auto& [name, value] : hiveSplit->partitionKeys) {
    cudfHiveSplitBuilder.partitionKey(name, value);
  }
  split_ = cudfHiveSplitBuilder.build();

  VLOG(1) << "Adding split " << split_->toString();
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  ensureCudaContextForThread();
  // Virtual method for class-specific conversion of the split
  convertSplit(split);
  partitionKeyScalars_.clear();

  // A partition key filter the split fails rejects every row of it, so open
  // nothing.
  splitSkipped_ = !partitionKeyFiltersAcceptSplit();
  if (splitSkipped_) {
    cudfSplitReader_.reset();
    recordSkippedSplit();
    return;
  }

  cudfSplitReader_ = createCudfSplitReader();

  // No row of the split can pass the filter, so open nothing.
  if (remainingFilterRejectsAllRows_) {
    recordSkippedSplit();
    return;
  }

  // An empty split has no footer to build the filter from.
  if (hasDecimalSubfieldFilter_ and
      not cudfSplitReader_->fileMetaDatas().empty()) {
    const auto& fileMetadatas = cudfSplitReader_->fileMetaDatas();
    const auto readerFilterType = getTableRowType();
    const HiveConfig hiveConfig(cudfHiveConfig_->config());
    const auto fileColumnNamesReadAsLowerCase =
        hiveConfig.isFileColumnNamesReadAsLowerCase(
            connectorQueryCtx_->sessionProperties());
    pushdownFilterTree_ = cudf::ast::tree{};
    pushdownFilterScalars_.clear();
    const auto decimalTypes = parquetDecimalTypes(
        fileMetadatas.front().schema,
        readerFilterType,
        fileColumnNamesReadAsLowerCase);
    const auto& filter = createAstFromSubfieldFilters(
        subfieldFilters_,
        pushdownFilterTree_,
        pushdownFilterScalars_,
        readerFilterType,
        &decimalTypes);
    cudfSplitReader_->setPushdownFilter(&filter);
  }
  cudfSplitReader_->prepareSplit(runtimeStats_);

  // An ignored missing file leaves nothing to read.
  if (cudfSplitReader_->emptySplit()) {
    return;
  }

  // Parse the split's partition values once, on the stream next() expands
  // them into columns on.
  partitionKeyScalars_ = makePartitionKeyScalars(cudfSplitReader_->stream());

  // Check if preloaded splits should start pre-fetching the first pass of
  // column chunks.
  const bool isPreloadedSplit = split->dataSource != nullptr;
  if (isPreloadedSplit &&
      cudfHiveConfig_->preloadColumnChunksSession(
          connectorQueryCtx_->sessionProperties())) {
    cudfSplitReader_->startColumnChunkFetch();
  }

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

void CudfHiveDataSource::setFromDataSource(std::unique_ptr<DataSource> source) {
  ensureCudaContextForThread();
  auto* preparedSource = checkedPointerCast<CudfHiveDataSource>(source.get());

  split_ = std::move(preparedSource->split_);
  splitSkipped_ = preparedSource->splitSkipped_;
  runtimeStats_.skippedSplits += preparedSource->runtimeStats_.skippedSplits;
  runtimeStats_.processedSplits +=
      preparedSource->runtimeStats_.processedSplits;
  runtimeStats_.skippedSplitBytes +=
      preparedSource->runtimeStats_.skippedSplitBytes;
  completedBytes_ += preparedSource->completedBytes_;
  completedRows_ += preparedSource->completedRows_;

  // Drop the reader of the previous split before replacing the AST storage it
  // references below.
  cudfSplitReader_.reset();

  // The reader's Parquet filter references the AST expressions and literal
  // scalars owned by 'source', which is freed right after this call. Adopt that
  // storage so the reader keeps pointing at live expressions; the nodes are
  // heap-allocated, so moving the owners does not move the expressions.
  subfieldScalars_ = std::move(preparedSource->subfieldScalars_);
  subfieldTree_ = std::move(preparedSource->subfieldTree_);
  subfieldFilterAst_ = preparedSource->subfieldFilterAst_;
  pushdownFilterScalars_ = std::move(preparedSource->pushdownFilterScalars_);
  pushdownFilterTree_ = std::move(preparedSource->pushdownFilterTree_);

  cudfSplitReader_ = std::move(preparedSource->cudfSplitReader_);
  partitionKeyScalars_ = std::move(preparedSource->partitionKeyScalars_);

  // A split a partition key filter rejected has no reader.
  if (!splitSkipped_) {
    VELOX_CHECK_NOT_NULL(cudfSplitReader_);

    // 'source' owns the query context the reader was prepared with and is
    // freed right after this call.
    cudfSplitReader_->setConnectorQueryCtx(connectorQueryCtx_);

    // Start column chunk fetch if it is not already started
    cudfSplitReader_->startColumnChunkFetch();
  }

  // The adopted reader keeps writing I/O statistics to the objects of
  // 'source', so carry the balance accumulated here over to those.
  preparedSource->ioStatistics_->merge(*ioStatistics_);
  ioStatistics_ = std::move(preparedSource->ioStatistics_);
  preparedSource->ioStats_->merge(*ioStats_);
  ioStats_ = std::move(preparedSource->ioStats_);
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /* future */) {
  VELOX_CHECK_NOT_NULL(split_, "No split present. Call addSplit() first.");
  if (splitSkipped_) {
    return nullptr;
  }
  VELOX_CHECK_NOT_NULL(cudfSplitReader_, "No split to process.");
  if (remainingFilterRejectsAllRows_ or cudfSplitReader_->emptySplit()) {
    cudfSplitReader_->resetSplit();
    return nullptr;
  }
  ensureCudaContextForThread();
  auto chunkOpt = cudfSplitReader_->next(size);
  if (!chunkOpt.has_value()) {
    cudfSplitReader_->resetSplit();
    return nullptr;
  }
  auto nRows = chunkOpt.value().numRows;
  auto cudfTable = std::move(chunkOpt.value().table);
  auto stream = cudfSplitReader_->stream();

  if (!partitionKeyNames_.empty()) {
    // The partition key columns follow the read columns, the layout the
    // remaining filter was compiled against. They are temporaries when that
    // filter copies the rows it keeps.
    auto columns = cudfTable->release();
    VELOX_CHECK_EQ(
        columns.size(),
        readColumnNames_.size(),
        "cuDF table scan read an unexpected number of columns");
    auto partitionKeyColumns = makePartitionKeyColumns(
        nRows,
        stream,
        optimizedRemainingFilter_ ? get_temp_mr() : get_output_mr());
    std::move(
        partitionKeyColumns.begin(),
        partitionKeyColumns.end(),
        std::back_inserter(columns));
    cudfTable = std::make_unique<cudf::table>(std::move(columns));
  }

  uint64_t filterTimeUs{0};
  if (optimizedRemainingFilter_) {
    MicrosecondWallTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(cudfTableColumns.size());
    for (auto& col : cudfTableColumns) {
      inputViews.push_back(col->view());
    }
    auto filterResult =
        cudfRemainingFilterExpression_->eval(inputViews, stream, get_temp_mr());
    auto originalTable =
        std::make_unique<cudf::table>(std::move(cudfTableColumns));
    cudfTable = cudf::apply_retention_mask(
        *originalTable, asView(filterResult), stream, get_output_mr());
    nRows = cudfTable->num_rows();
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  if (!partitionKeyNames_.empty()) {
    // Project to the output columns by name: the table holds the read columns
    // and then the partition keys, while the output interleaves them. A
    // partition key may be projected under several names.
    auto columns = cudfTable->release();
    std::unordered_map<std::string, size_t> columnIndices;
    for (size_t i = 0; i < readColumnNames_.size(); ++i) {
      columnIndices.emplace(readColumnNames_[i], i);
    }
    for (size_t i = 0; i < partitionKeyNames_.size(); ++i) {
      columnIndices.emplace(partitionKeyNames_[i], readColumnNames_.size() + i);
    }
    std::vector<size_t> outputIndices;
    outputIndices.reserve(outputSourceNames_.size());
    std::vector<size_t> remainingUses(columns.size(), 0);
    for (const auto& name : outputSourceNames_) {
      const auto it = columnIndices.find(name);
      VELOX_CHECK(
          it != columnIndices.end(),
          "Output column {} was neither read nor materialized",
          name);
      outputIndices.push_back(it->second);
      ++remainingUses[it->second];
    }
    std::vector<std::unique_ptr<cudf::column>> outputColumns;
    outputColumns.reserve(outputIndices.size());
    for (const auto index : outputIndices) {
      auto& column = columns[index];
      outputColumns.push_back(
          --remainingUses[index] == 0 ? std::move(column)
                                      : std::make_unique<cudf::column>(
                                            *column, stream, get_output_mr()));
    }
    cudfTable = std::make_unique<cudf::table>(std::move(outputColumns));
  } else if (outputType_->size() < cudfTable->num_columns()) {
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
  RowVectorPtr output;
  if (cudfIsRegistered()) {
    output = std::make_shared<CudfVector>(
        pool_, outputType_, nRows, std::move(cudfTable), stream);
  } else if (cudfTable->num_columns() == 0) {
    // There is no device data to convert, only the row count.
    output = std::make_shared<RowVector>(
        pool_, outputType_, nullptr, nRows, std::vector<VectorPtr>{});
  } else {
    output = with_arrow::toVeloxColumn(
        cudfTable->view(), pool_, outputType_, stream, get_temp_mr());
  }
  stream.sync();

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto result = runtimeStats_.toRuntimeMetricMap();
  io::addIoStatsToRuntimeStats(*ioStatistics_, "", result);
  if (const auto it = result.find(std::string(io::kStorageReadBytes));
      it != result.end()) {
    // Preserve the DWIO value before a ReadFile-layer value overrides it.
    // Overread bytes are defined relative to this counter.
    result.emplace(kDwioStorageReadBytes, it->second);
  }
  // Preserve a zero-valued totalScanTime before scan timing is recorded.
  result.insert({
      {std::string(io::kTotalScanTime),
       RuntimeMetric(
           ioStatistics_->totalScanTimeNs(), RuntimeCounter::Unit::kNanos)},
      {std::string(Connector::kTotalRemainingFilterTime),
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  const auto& ioStats = ioStats_->stats();
  for (const auto& [key, value] : ioStats) {
    // Keep the ReadFile-layer value under the established key.
    if (key == io::kStorageReadBytes) {
      result[std::string(key)] = value;
    } else {
      result.emplace(key, value);
    }
  }
  return result;
}

const TypePtr& CudfHiveDataSource::columnType(const std::string& name) const {
  if (const auto it = columnTypes_.find(name); it != columnTypes_.end()) {
    return it->second;
  }
  const auto& dataColumns = tableHandle_->dataColumns();
  VELOX_USER_CHECK_NOT_NULL(
      dataColumns,
      "Table schema is required to resolve the type of filter-only column {}",
      name);
  return dataColumns->findChild(name);
}

std::optional<std::string> CudfHiveDataSource::partitionKeyValue(
    const std::string& name) const {
  const auto it = split_->partitionKeys.find(name);
  return it == split_->partitionKeys.end() ? std::nullopt : it->second;
}

bool CudfHiveDataSource::readTimestampPartitionValueAsLocalTime() const {
  const HiveConfig hiveConfig(cudfHiveConfig_->config());
  return hiveConfig.readTimestampPartitionValueAsLocalTime(
      connectorQueryCtx_->sessionProperties());
}

bool CudfHiveDataSource::partitionKeyFiltersAcceptSplit() const {
  if (partitionKeyFilters_.empty()) {
    return true;
  }
  const bool timestampValueAsLocalTime =
      readTimestampPartitionValueAsLocalTime();
  for (const auto& [subfield, filter] : partitionKeyFilters_) {
    const auto& name = subfield.baseName();
    const auto& partitionKey = partitionKeys_.at(name);
    if (!testPartitionKeyFilter(
            *filter,
            partitionKey.type,
            partitionKeyValue(name),
            partitionKey.dateValueDaysSinceEpoch,
            timestampValueAsLocalTime)) {
      VLOG(1) << "Skipping split " << split_->toString()
              << " because the filter on partition key " << name
              << " rejects its value";
      return false;
    }
  }
  return true;
}

std::vector<std::unique_ptr<cudf::scalar>>
CudfHiveDataSource::makePartitionKeyScalars(cuda::stream_ref stream) const {
  const bool timestampValueAsLocalTime =
      readTimestampPartitionValueAsLocalTime();
  // A TIMESTAMP key takes the type the split reader gives file timestamps
  // rather than the unit cuDF expressions default to, so that an expression
  // can combine the two. Day resolution is no unit a TIMESTAMP scalar can
  // take, so the key keeps the default unit when the reader casts timestamps
  // to days.
  const auto readerTimestampType = cudfHiveConfig_->timestampType().id();
  const std::optional<cudf::type_id> timestampType =
      readerTimestampType == cudf::type_id::TIMESTAMP_DAYS
      ? std::nullopt
      : std::optional<cudf::type_id>{readerTimestampType};
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  scalars.reserve(partitionKeyNames_.size());
  for (const auto& name : partitionKeyNames_) {
    const auto& partitionKey = partitionKeys_.at(name);
    const auto constant = newConstantFromString(
        partitionKey.type,
        partitionKeyValue(name),
        pool_,
        timestampValueAsLocalTime,
        partitionKey.dateValueDaysSinceEpoch);
    scalars.push_back(makeScalarFromConstantExpr(
        std::make_shared<core::ConstantTypedExpr>(constant),
        pool_,
        partitionKey.type->isTimestamp() ? timestampType : std::nullopt,
        stream));
  }
  return scalars;
}

std::vector<std::unique_ptr<cudf::column>>
CudfHiveDataSource::makePartitionKeyColumns(
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_EQ(
      partitionKeyScalars_.size(),
      partitionKeyNames_.size(),
      "The partition key scalars of the split are missing");
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(partitionKeyScalars_.size());
  for (const auto& scalar : partitionKeyScalars_) {
    columns.push_back(
        cudf::make_column_from_scalar(*scalar, numRows, stream, mr));
  }
  return columns;
}

void CudfHiveDataSource::recordSkippedSplit() {
  runtimeStats_.skippedSplits++;
  // An unbounded length means the whole file, whose size the split does not
  // carry, so it contributes no byte count.
  if (split_->length != std::numeric_limits<uint64_t>::max()) {
    runtimeStats_.skippedSplitBytes += static_cast<int64_t>(split_->length);
  }
}

const RowTypePtr CudfHiveDataSource::getTableRowType() {
  if (cachedTableRowType_) {
    return cachedTableRowType_;
  }
  // The read columns and then the materialized partition keys: the layout of
  // the table the remaining filter is evaluated on.
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(readColumnNames_.size() + partitionKeyNames_.size());
  types.reserve(readColumnNames_.size() + partitionKeyNames_.size());
  for (const auto& name : readColumnNames_) {
    names.push_back(name);
    types.push_back(columnType(name));
  }
  for (const auto& name : partitionKeyNames_) {
    names.push_back(name);
    types.push_back(partitionKeys_.at(name).type);
  }
  cachedTableRowType_ = ROW(std::move(names), std::move(types));
  return cachedTableRowType_;
}

} // namespace facebook::velox::cudf_velox::connector::hive
