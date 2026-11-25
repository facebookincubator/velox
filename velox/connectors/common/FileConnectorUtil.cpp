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

#include "velox/connectors/common/FileConnectorUtil.h"

#include <fmt/format.h>
#include <unordered_map>

#include "velox/common/config/Config.h"
#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/common/FileConfig.h"
#include "velox/connectors/common/FileConnectorSplit.h"
#include "velox/connectors/common/FileTableHandle.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/dwrf/common/Config.h"
#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/common/ParquetConfig.h"
#endif

#include "velox/expression/Expr.h"
#include "velox/expression/ExprConstants.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::connector {

FormatScopedConfigs makeFormatScopedConfigs(
    const FileConfig& fileConfig,
    const config::ConfigBase& sessionProperties,
    dwio::common::FileFormat fileFormat) {
  VELOX_CHECK_NE(
      fileFormat,
      dwio::common::FileFormat::UNKNOWN,
      "Cannot build format-specific configs for unknown file format");

  auto connectorConfig = fileConfig.config()->rawConfigsWithPrefix(
      fmt::format(
          "{}{}",
          fileConfig.connectorConfigPrefix(),
          dwio::common::formatConfigPrefix(fileFormat, ".")));
  return {
      config::ConfigBase(std::move(connectorConfig)),
      config::ConfigBase(sessionProperties.rawConfigsWithPrefix(
          dwio::common::formatConfigPrefix(fileFormat, "_")))};
}

namespace {

void validateColumnMappingMode(
    dwio::common::ColumnMappingMode mode,
    dwio::common::FileFormat fileFormat) {
  // kParquetFieldId is format-specific: it matches requested columns against
  // physical Parquet schema field_id metadata. Other readers don't have that
  // metadata, so reject it at split setup time instead of letting a later
  // reader path interpret it as a generic field-id or name/position mode.
  VELOX_USER_CHECK(
      mode != dwio::common::ColumnMappingMode::kParquetFieldId ||
          fileFormat == dwio::common::FileFormat::PARQUET,
      "Column mapping mode {} is not supported for file format {}",
      mode,
      dwio::common::FileFormatName::toName(fileFormat));
}

dwio::common::ColumnMappingMode sessionColumnMappingMode(
    const FileConfig& fileConfig,
    const config::ConfigBase* sessionProperties) {
  return fileConfig.useColumnNames(sessionProperties)
      ? dwio::common::ColumnMappingMode::kName
      : dwio::common::ColumnMappingMode::kPosition;
}

} // namespace

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const FileTableHandlePtr& tableHandle,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    dwio::common::ReaderOptions& readerOptions) {
  configureReaderOptions(
      fileConfig,
      connectorQueryCtx,
      tableHandle->dataColumns(),
      fileSplit,
      tableHandle->tableParameters(),
      readerOptions);
}

void configureReaderOptions(
    const std::shared_ptr<const FileConfig>& fileConfig,
    const ConnectorQueryCtx* connectorQueryCtx,
    const RowTypePtr& fileSchema,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::unordered_map<std::string, std::string>& /*tableParameters*/,
    dwio::common::ReaderOptions& readerOptions) {
  auto sessionProperties = connectorQueryCtx->sessionProperties();
  VELOX_CHECK_NOT_NULL(sessionProperties, "Session properties are null");
  readerOptions.setLoadQuantum(fileConfig->loadQuantum(sessionProperties));
  readerOptions.setMaxCoalesceBytes(
      fileConfig->maxCoalescedBytes(sessionProperties));
  readerOptions.setMaxCoalesceDistance(
      fileConfig->maxCoalescedDistanceBytes(sessionProperties));
  readerOptions.setFileColumnNamesReadAsLowerCase(
      fileConfig->isFileColumnNamesReadAsLowerCase(sessionProperties));
  readerOptions.setAllowEmptyFile(true);
  const auto columnMappingMode = fileSplit->columnMappingMode.value_or(
      sessionColumnMappingMode(*fileConfig, sessionProperties));
  validateColumnMappingMode(columnMappingMode, fileSplit->fileFormat);
  readerOptions.setColumnMappingMode(columnMappingMode);
  readerOptions.setFileSchema(fileSchema);
  readerOptions.setFilePreloadThreshold(fileConfig->filePreloadThreshold());
  readerOptions.setPrefetchRowGroups(fileConfig->prefetchRowGroups());
  readerOptions.setCacheable(fileSplit->cacheable);
  const auto& sessionTzName = connectorQueryCtx->sessionTimezone();
  if (!sessionTzName.empty()) {
    const auto timezone = tz::locateZone(sessionTzName);
    readerOptions.setSessionTimezone(timezone);
  }
  readerOptions.setAdjustTimestampToTimezone(
      connectorQueryCtx->adjustTimestampToTimezone());
  // Prefer connector session property (FileConfig). Fall back to
  // ConnectorQueryCtx (threaded from QueryConfig) for backward compatibility
  // with callers that set it as a query config.
  if (sessionProperties->valueExists(
          FileConfig::kSelectiveNimbleReaderEnabledSession)) {
    readerOptions.setSelectiveNimbleReaderEnabled(
        fileConfig->selectiveNimbleReaderEnabled(sessionProperties));
  } else {
    readerOptions.setSelectiveNimbleReaderEnabled(
        connectorQueryCtx->selectiveNimbleReaderEnabled());
  }
  readerOptions.setNimbleDirectBufferedInputEnabled(
      fileConfig->nimbleDirectBufferedInputEnabled(sessionProperties));
  readerOptions.setNimbleLazyFileStats(
      fileConfig->nimbleLazyFileStats(sessionProperties));
  readerOptions.setCacheMetadata(
      fileConfig->cacheMetadata(sessionProperties) && fileSplit->cacheable);
  readerOptions.setPinMetadata(fileConfig->pinMetadata(sessionProperties));
  readerOptions.setCacheIndex(
      fileConfig->cacheIndex(sessionProperties) && fileSplit->cacheable);
  readerOptions.setPinIndex(fileConfig->pinIndex(sessionProperties));

  // Set footer speculative IO size based on file format.
  switch (fileSplit->fileFormat) {
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::ORC:
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->orcFooterSpeculativeIoSize(sessionProperties));
      break;
    case dwio::common::FileFormat::PARQUET:
      break;
    case dwio::common::FileFormat::NIMBLE:
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->nimbleFooterSpeculativeIoSize(sessionProperties));
      break;
    default:
      // Use ORC default for unknown formats.
      readerOptions.setFooterSpeculativeIoSize(
          fileConfig->orcFooterSpeculativeIoSize(sessionProperties));
      break;
  }

  if (readerOptions.fileFormat() != dwio::common::FileFormat::UNKNOWN) {
    VELOX_CHECK(
        readerOptions.fileFormat() == fileSplit->fileFormat,
        "HiveDataSource received splits of different formats: {} and {}",
        dwio::common::FileFormatName::toName(readerOptions.fileFormat()),
        dwio::common::FileFormatName::toName(fileSplit->fileFormat));
  } else {
    readerOptions.setFileFormat(fileSplit->fileFormat);
  }

  if (!dwio::common::hasReaderFactory(fileSplit->fileFormat)) {
    readerOptions.setFormatSpecificOptions(nullptr);
    return;
  }

  auto formatScopedConfigs = makeFormatScopedConfigs(
      *fileConfig, *sessionProperties, fileSplit->fileFormat);
  readerOptions.setFormatSpecificOptions(
      dwio::common::getReaderFactory(fileSplit->fileFormat)
          ->createFormatOptions(
              formatScopedConfigs.connectorConfig,
              formatScopedConfigs.sessionProperties));
}

void configureRowReaderOptions(
    const std::unordered_map<std::string, std::string>& tableParameters,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    const RowTypePtr& rowType,
    const std::shared_ptr<const FileConnectorSplit>& fileSplit,
    const std::shared_ptr<const FileConfig>& fileConfig,
    const config::ConfigBase* sessionProperties,
    folly::Executor* const ioExecutor,
    dwio::common::RowReaderOptions& rowReaderOptions) {
  auto skipRowsIt =
      tableParameters.find(dwio::common::TableParameter::kSkipHeaderLineCount);
  if (skipRowsIt != tableParameters.end()) {
    rowReaderOptions.setSkipRows(folly::to<uint64_t>(skipRowsIt->second));
  }
  rowReaderOptions.setScanSpec(scanSpec);
  rowReaderOptions.setIOExecutor(ioExecutor);
  rowReaderOptions.setMetadataFilter(std::move(metadataFilter));
  rowReaderOptions.setRequestedType(rowType);
  rowReaderOptions.range(fileSplit->start, fileSplit->length);
  if (fileSplit->splitRowRange.has_value()) {
    VELOX_CHECK(
        fileSplit->fileFormat == dwio::common::FileFormat::NIMBLE ||
            fileSplit->fileFormat == dwio::common::FileFormat::DWRF ||
            fileSplit->fileFormat == dwio::common::FileFormat::ORC ||
            fileSplit->fileFormat == dwio::common::FileFormat::PARQUET ||
            fileSplit->fileFormat == dwio::common::FileFormat::TEXT,
        "Split row ranges are only supported for Nimble, DWRF, ORC, Parquet and text files: {}",
        fileSplit->toString());
    rowReaderOptions.setSplitRowRange(fileSplit->splitRowRange);
  }
  if (fileConfig && sessionProperties) {
    rowReaderOptions.setTimestampPrecision(
        static_cast<TimestampPrecision>(
            fileConfig->readTimestampUnit(sessionProperties)));
    rowReaderOptions.setPreserveFlatMapsInMemory(
        fileConfig->preserveFlatMapsInMemory(sessionProperties));
    rowReaderOptions.setParallelUnitLoadCount(
        fileConfig->parallelUnitLoadCount(sessionProperties));
    rowReaderOptions.setIndexEnabled(
        fileConfig->indexEnabled(sessionProperties));
    rowReaderOptions.setLazyColumnIo(
        fileConfig->nimbleLazyColumnIo(sessionProperties));
    rowReaderOptions.setCollectColumnCpuMetrics(
        fileConfig->readerCollectColumnCpuMetrics(sessionProperties));
    rowReaderOptions.setNimbleFsstDecodeRepeatedPayloads(
        fileConfig->nimbleFsstDecodeRepeatedPayloads(sessionProperties));
    rowReaderOptions.setNimblePreserveDictionaryEncoding(
        fileConfig->nimblePreserveDictionaryEncoding(sessionProperties));
    rowReaderOptions.setNimbleDictionaryAwareReads(
        fileConfig->nimbleDictionaryAwareReads(sessionProperties));
    rowReaderOptions.setNimbleScanFastPath(
        fileConfig->nimbleScanFastPath(sessionProperties));
  }
}

namespace {

struct SubfieldSpec {
  const common::Subfield* subfield;
  bool filterOnly;
};

template <typename T>
void deduplicate(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

// Floating point map key subscripts are truncated toward 0 in Presto.  For
// example given `a' as a map with floating point key, if user queries a[0.99],
// Presto coordinator will generate a required subfield a[0]; for a[-1.99] it
// will generate a[-1]; for anything larger than 9223372036854775807, it
// generates a[9223372036854775807]; for anything smaller than
// -9223372036854775808 it generates a[-9223372036854775808].
template <typename T>
std::unique_ptr<common::Filter> makeFloatingPointMapKeyFilter(
    const std::vector<int64_t>& subscripts) {
  std::vector<std::unique_ptr<common::Filter>> filters;
  for (auto subscript : subscripts) {
    T lower = subscript;
    T upper = subscript;
    bool lowerUnbounded = subscript == std::numeric_limits<int64_t>::min();
    bool upperUnbounded = subscript == std::numeric_limits<int64_t>::max();
    bool lowerExclusive = false;
    bool upperExclusive = false;
    if (lower <= 0 && !lowerUnbounded) {
      if (lower > subscript - 1) {
        lower = subscript - 1;
      } else {
        lower = std::nextafter(lower, -std::numeric_limits<T>::infinity());
      }
      lowerExclusive = true;
    }
    if (upper >= 0 && !upperUnbounded) {
      if (upper < subscript + 1) {
        upper = subscript + 1;
      } else {
        upper = std::nextafter(upper, std::numeric_limits<T>::infinity());
      }
      upperExclusive = true;
    }
    if (lowerUnbounded && upperUnbounded) {
      continue;
    }
    filters.push_back(
        std::make_unique<common::FloatingPointRange<T>>(
            lower,
            lowerUnbounded,
            lowerExclusive,
            upper,
            upperUnbounded,
            upperExclusive,
            false));
  }
  if (filters.size() == 1) {
    return std::move(filters[0]);
  }
  return std::make_unique<common::MultiRange>(std::move(filters), false);
}

// Recursively add subfields to scan spec.
void addSubfields(
    const Type& type,
    std::vector<SubfieldSpec>& subfields,
    int level,
    memory::MemoryPool* pool,
    common::ScanSpec& spec) {
  int newSize = 0;
  for (int i = 0; i < subfields.size(); ++i) {
    if (level < subfields[i].subfield->path().size()) {
      subfields[newSize++] = subfields[i];
    } else if (!subfields[i].filterOnly) {
      spec.addAllChildFields(type);
      return;
    }
  }
  subfields.resize(newSize);

  switch (type.kind()) {
    case TypeKind::ROW: {
      folly::F14FastMap<std::string, std::vector<SubfieldSpec>> required;
      for (auto& subfield : subfields) {
        auto* element = subfield.subfield->path()[level].get();
        auto* nestedField = element->asChecked<common::Subfield::NestedField>();
        required[nestedField->name()].push_back(subfield);
      }
      const auto& rowType = type.asRow();
      for (int i = 0; i < rowType.size(); ++i) {
        auto& childName = rowType.nameOf(i);
        auto& childType = rowType.childAt(i);
        auto* child = spec.addField(childName, i);
        auto it = required.find(childName);
        if (it == required.end()) {
          child->setConstantValue(
              BaseVector::createNullConstant(childType, 1, pool));
        } else {
          addSubfields(*childType, it->second, level + 1, pool, *child);
        }
      }
      break;
    }
    case TypeKind::MAP: {
      auto& keyType = type.childAt(0);
      auto* keys = spec.addMapKeyFieldRecursively(*keyType);
      addSubfields(
          *type.childAt(1),
          subfields,
          level + 1,
          pool,
          *spec.addMapValueField());
      if (subfields.empty()) {
        return;
      }
      bool stringKey = keyType->isVarchar() || keyType->isVarbinary();
      std::vector<std::string> stringSubscripts;
      std::vector<int64_t> longSubscripts;
      for (auto& subfield : subfields) {
        auto* element = subfield.subfield->path()[level].get();
        if (element->is(common::SubfieldKind::kAllSubscripts)) {
          return;
        }
        if (stringKey) {
          auto* subscript = element->as<common::Subfield::StringSubscript>();
          VELOX_CHECK(
              subscript,
              "Unsupported for string map pruning: {}",
              element->toString());
          stringSubscripts.push_back(subscript->index());
        } else {
          auto* subscript = element->as<common::Subfield::LongSubscript>();
          VELOX_CHECK(
              subscript,
              "Unsupported for long map pruning: {}",
              element->toString());
          longSubscripts.push_back(subscript->index());
        }
      }
      std::unique_ptr<common::Filter> filter;
      if (stringKey) {
        deduplicate(stringSubscripts);
        filter = std::make_unique<common::BytesValues>(stringSubscripts, false);
        spec.setFlatMapFeatureSelection(std::move(stringSubscripts));
      } else {
        deduplicate(longSubscripts);
        if (keyType->isReal()) {
          filter = makeFloatingPointMapKeyFilter<float>(longSubscripts);
        } else if (keyType->isDouble()) {
          filter = makeFloatingPointMapKeyFilter<double>(longSubscripts);
        } else {
          filter = common::createBigintValues(longSubscripts, false);
        }
        std::vector<std::string> features;
        for (auto num : longSubscripts) {
          features.push_back(std::to_string(num));
        }
        spec.setFlatMapFeatureSelection(std::move(features));
      }
      keys->setFilter(std::move(filter));
      break;
    }
    case TypeKind::ARRAY: {
      addSubfields(
          *type.childAt(0),
          subfields,
          level + 1,
          pool,
          *spec.addArrayElementField());
      if (subfields.empty()) {
        return;
      }
      constexpr long kMaxIndex = std::numeric_limits<vector_size_t>::max();
      long maxIndex = -1;
      for (auto& subfield : subfields) {
        auto* element = subfield.subfield->path()[level].get();
        if (dynamic_cast<const common::Subfield::AllSubscripts*>(element)) {
          return;
        }
        auto* subscript =
            dynamic_cast<const common::Subfield::LongSubscript*>(element);
        VELOX_CHECK(
            subscript,
            "Unsupported for array pruning: {}",
            element->toString());
        VELOX_USER_CHECK_GT(
            subscript->index(),
            0,
            "Non-positive array subscript cannot be push down");
        maxIndex = std::max(maxIndex, std::min(kMaxIndex, subscript->index()));
      }
      spec.setMaxArrayElementsCount(maxIndex);
      break;
    }
    default:
      break;
  }
}

inline bool isSynthesizedColumn(
    const std::string& name,
    const std::unordered_map<std::string, FileColumnHandlePtr>& infoColumns) {
  return infoColumns.count(name) != 0;
}

bool isSpecialColumn(
    const std::string& name,
    const std::optional<std::string>& specialName) {
  return specialName.has_value() && name == *specialName;
}

} // namespace

namespace {

void processFieldSpec(
    const RowTypePtr& dataColumns,
    const TypePtr& outputType,
    common::ScanSpec& fieldSpec) {
  fieldSpec.visit(*outputType, [](const Type& type, common::ScanSpec& spec) {
    if (type.isMap() && !spec.isConstant()) {
      auto* keys = spec.childByName(common::ScanSpec::kMapKeysFieldName);
      VELOX_CHECK_NOT_NULL(keys);
      if (keys->filter()) {
        VELOX_CHECK(!keys->filter()->testNull());
      } else {
        keys->setFilter(std::make_shared<common::IsNotNull>());
      }
    }
  });
  if (dataColumns) {
    const auto childIdxOpt =
        dataColumns->getChildIdxIfExists(fieldSpec.fieldName());
    if (childIdxOpt.has_value()) {
      if (dataColumns->childAt(*childIdxOpt)->isMap() && outputType->isRow()) {
        fieldSpec.setFlatMapAsStruct(true);
      }
    }
  }
}

} // namespace

std::shared_ptr<common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType,
    const folly::F14FastMap<std::string, std::vector<const common::Subfield*>>&
        outputSubfields,
    const common::SubfieldFilters& subfieldFilters,
    const RowTypePtr& dataColumns,
    const std::unordered_map<std::string, FileColumnHandlePtr>& partitionKeys,
    const std::unordered_map<std::string, FileColumnHandlePtr>& infoColumns,
    const SpecialColumnNames& specialColumns,
    bool disableStatsBasedFilterReorder,
    memory::MemoryPool* pool) {
  return makeScanSpec(
      rowType,
      outputSubfields,
      subfieldFilters,
      /*indexColumns=*/{},
      dataColumns,
      partitionKeys,
      infoColumns,
      specialColumns,
      disableStatsBasedFilterReorder,
      pool);
}

std::shared_ptr<common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType,
    const folly::F14FastMap<std::string, std::vector<const common::Subfield*>>&
        outputSubfields,
    const common::SubfieldFilters& subfieldFilters,
    const std::vector<std::string>& indexColumns,
    const RowTypePtr& dataColumns,
    const std::unordered_map<std::string, FileColumnHandlePtr>& partitionKeys,
    const std::unordered_map<std::string, FileColumnHandlePtr>& infoColumns,
    const SpecialColumnNames& specialColumns,
    bool disableStatsBasedFilterReorder,
    memory::MemoryPool* pool) {
  auto spec = std::make_shared<common::ScanSpec>("root");
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      filterSubfields;
  std::vector<SubfieldSpec> subfieldSpecs;
  for (const auto& [subfield, _] : subfieldFilters) {
    if (auto name = subfield.toString();
        !isSynthesizedColumn(name, infoColumns) &&
        partitionKeys.count(name) == 0) {
      VELOX_CHECK(!isSpecialColumn(name, specialColumns.rowIndex));
      VELOX_CHECK(!isSpecialColumn(name, specialColumns.rowId));
      VELOX_CHECK(!isSpecialColumn(name, specialColumns.readRange));
      filterSubfields[subfield.baseName()].push_back(&subfield);
    }
  }

  // Process columns that will be projected out.
  for (int i = 0; i < rowType->size(); ++i) {
    auto& name = rowType->nameOf(i);
    auto& type = rowType->childAt(i);
    if (isSpecialColumn(name, specialColumns.rowIndex)) {
      VELOX_CHECK(type->isBigint());
      auto* fieldSpec = spec->addField(name, i);
      fieldSpec->setColumnType(common::ScanSpec::ColumnType::kRowIndex);
      continue;
    }
    if (isSpecialColumn(name, specialColumns.rowId)) {
      VELOX_CHECK(type->isRow() && type->size() == 5);
      auto& rowIdType = type->asRow();
      auto* fieldSpec = spec->addFieldRecursively(name, rowIdType, i);
      fieldSpec->setColumnType(common::ScanSpec::ColumnType::kComposite);
      fieldSpec->childByName(rowIdType.nameOf(0))
          ->setColumnType(common::ScanSpec::ColumnType::kRowIndex);
      continue;
    }
    if (isSpecialColumn(name, specialColumns.readRange)) {
      VELOX_USER_CHECK(
          type->kindEquals(ROW({BIGINT(), INTEGER()})),
          "Read range column must be ROW(BIGINT, INTEGER): {}",
          type->toString());
      auto* fieldSpec = spec->addField(name, i);
      fieldSpec->setColumnType(common::ScanSpec::ColumnType::kReadRange);
      continue;
    }
    auto it = outputSubfields.find(name);
    if (it == outputSubfields.end()) {
      auto* fieldSpec = spec->addFieldRecursively(name, *type, i);
      processFieldSpec(dataColumns, type, *fieldSpec);
      filterSubfields.erase(name);
      continue;
    }
    for (auto* subfield : it->second) {
      subfieldSpecs.push_back({subfield, false});
    }
    it = filterSubfields.find(name);
    if (it != filterSubfields.end()) {
      for (auto* subfield : it->second) {
        subfieldSpecs.push_back({subfield, true});
      }
      filterSubfields.erase(it);
    }
    auto* fieldSpec = spec->addField(name, i);
    addSubfields(*type, subfieldSpecs, 1, pool, *fieldSpec);
    processFieldSpec(dataColumns, type, *fieldSpec);
    subfieldSpecs.clear();
  }

  // Now process the columns that will not be projected out.
  if (!filterSubfields.empty()) {
    VELOX_CHECK_NOT_NULL(dataColumns);
    for (auto& [fieldName, subfields] : filterSubfields) {
      for (auto* subfield : subfields) {
        subfieldSpecs.push_back({subfield, true});
      }
      auto& type = dataColumns->findChild(fieldName);
      auto* fieldSpec = spec->getOrCreateChild(fieldName);
      addSubfields(*type, subfieldSpecs, 1, pool, *fieldSpec);
      processFieldSpec(dataColumns, type, *fieldSpec);
      subfieldSpecs.clear();
    }
  }

  // Process index columns from join conditions to ensure they are read.
  // These columns are not projected out, only used for index lookup.
  for (const auto& keyName : indexColumns) {
    VELOX_CHECK_NOT_NULL(dataColumns);
    if (spec->childByName(keyName) == nullptr) {
      // This is required so that we can set filter on the index column in the
      // selective reader later.
      spec->getOrCreateChild(keyName);
    }
  }

  for (auto& pair : subfieldFilters) {
    const auto name = pair.first.toString();
    if (isSynthesizedColumn(name, infoColumns)) {
      continue;
    }
    auto fieldSpec = spec->getOrCreateChild(pair.first);
    VELOX_CHECK_NULL(spec->filter());
    fieldSpec->setFilter(pair.second);
  }

  if (disableStatsBasedFilterReorder) {
    spec->disableStatsBasedFilterReorder();
  }
  return spec;
}

namespace {

core::CallTypedExprPtr replaceInputs(
    const core::CallTypedExpr* call,
    std::vector<core::TypedExprPtr>&& inputs) {
  return std::make_shared<core::CallTypedExpr>(
      call->type(), std::move(inputs), call->name());
}

bool endWith(const std::string& str, const char* suffix) {
  return str.ends_with(suffix);
}

bool isNotExpr(
    const core::TypedExprPtr& expr,
    const core::CallTypedExpr* call,
    core::ExpressionEvaluator* evaluator) {
  if (!endWith(call->name(), "not")) {
    return false;
  }
  auto exprs = evaluator->compile(expr);
  VELOX_CHECK_EQ(exprs->size(), 1);
  auto& compiled = exprs->expr(0);
  return compiled->vectorFunction() &&
      compiled->vectorFunction()->getCanonicalName() ==
      exec::FunctionCanonicalName::kNot;
}

double getPrestoSampleRate(
    const core::TypedExprPtr& expr,
    const core::CallTypedExpr* call,
    core::ExpressionEvaluator* evaluator) {
  if (!endWith(call->name(), "lt")) {
    return -1;
  }
  VELOX_CHECK_EQ(call->inputs().size(), 2);
  auto exprs = evaluator->compile(expr);
  VELOX_CHECK_EQ(exprs->size(), 1);
  auto& lt = exprs->expr(0);
  if (!(lt->vectorFunction() &&
        lt->vectorFunction()->getCanonicalName() ==
            exec::FunctionCanonicalName::kLt)) {
    return -1;
  }
  auto& rand = lt->inputs()[0];
  if (!(rand->inputs().empty() && rand->vectorFunction() &&
        rand->vectorFunction()->getCanonicalName() ==
            exec::FunctionCanonicalName::kRand)) {
    return -1;
  }
  auto* rate =
      dynamic_cast<const core::ConstantTypedExpr*>(call->inputs()[1].get());
  if (!(rate && rate->type()->kind() == TypeKind::DOUBLE)) {
    return -1;
  }
  return std::max(0.0, std::min(1.0, rate->value().value<double>()));
}

core::TypedExprPtr extractFiltersFromRemainingFilter(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    bool negated,
    common::SubfieldFilters& filters,
    double& sampleRate) {
  common::Filter* oldFilter = nullptr;
  try {
    if (auto subfieldAndFilter =
            exec::ExprToSubfieldFilterParser::getInstance()
                ->leafToSubfieldFilter(*expr, evaluator, negated)) {
      auto& [subfield, filter] = subfieldAndFilter.value();
      if (auto it = filters.find(subfield); it != filters.end()) {
        oldFilter = it->second.get();
        filter = filter->mergeWith(oldFilter);
      }
      filters.insert_or_assign(std::move(subfield), std::move(filter));
      return nullptr;
    }
  } catch (const VeloxException&) {
    LOG(WARNING) << "Unexpected failure when extracting filter for: "
                 << expr->toString();
    if (oldFilter) {
      LOG(WARNING) << "Merging with " << oldFilter->toString();
    }
  }

  auto* call = dynamic_cast<const core::CallTypedExpr*>(expr.get());
  if (call == nullptr) {
    return expr;
  }

  if (isNotExpr(expr, call, evaluator)) {
    auto inner = extractFiltersFromRemainingFilter(
        call->inputs()[0], evaluator, !negated, filters, sampleRate);
    return inner ? replaceInputs(call, {inner}) : nullptr;
  }

  if ((call->name() == expression::kAnd && !negated) ||
      (call->name() == expression::kOr && negated)) {
    std::vector<core::TypedExprPtr> args;
    args.reserve(call->inputs().size());
    for (const auto& input : call->inputs()) {
      if (auto arg = extractFiltersFromRemainingFilter(
              input, evaluator, negated, filters, sampleRate)) {
        args.push_back(std::move(arg));
      }
      // If extractFiltersFromRemainingFilter returns nullptr, it means
      // everything in input is converted to filters.
    }
    if (args.empty()) {
      return nullptr;
    }
    if (args.size() == 1) {
      return std::move(args[0]);
    }
    return replaceInputs(call, std::move(args));
  }

  if ((call->name() == expression::kAnd && negated) ||
      (call->name() == expression::kOr && !negated)) {
    std::vector<std::unique_ptr<common::Filter>> disjuncts;
    common::Subfield subfield;

    for (const auto& input : call->inputs()) {
      common::SubfieldFilters tmpFilters;
      double tmpSampleRate = 1;
      auto tmpRemaining = extractFiltersFromRemainingFilter(
          input, evaluator, negated, tmpFilters, tmpSampleRate);

      if (tmpRemaining != nullptr || tmpSampleRate != 1 ||
          tmpFilters.size() != 1) {
        disjuncts.clear();
        break;
      }

      if (disjuncts.empty()) {
        subfield = tmpFilters.begin()->first.clone();
      } else if (!(subfield == tmpFilters.begin()->first)) {
        disjuncts.clear();
        break;
      }

      disjuncts.push_back(tmpFilters.begin()->second->clone());
    }

    if (!disjuncts.empty()) {
      auto filter =
          exec::ExprToSubfieldFilterParser::makeOrFilter(std::move(disjuncts));

      if (filter == nullptr) {
        return expr;
      }

      auto it = filters.find(subfield);
      if (it != filters.end()) {
        filter = filter->mergeWith(it->second.get());
      }

      filters.insert_or_assign(std::move(subfield), std::move(filter));

      return nullptr;
    }
  }

  if (!negated) {
    double rate = getPrestoSampleRate(expr, call, evaluator);
    if (rate != -1) {
      sampleRate *= rate;
      return nullptr;
    }
  }

  return expr;
}
} // namespace

core::TypedExprPtr extractFiltersFromRemainingFilter(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    common::SubfieldFilters& filters,
    double& sampleRate) {
  if (expr == nullptr) {
    return nullptr;
  }
  return extractFiltersFromRemainingFilter(
      expr, evaluator, /*negated=*/false, filters, sampleRate);
}

namespace {
template <TypeKind kind>
std::unique_ptr<common::Filter> createRangeFilterInternal(
    const variant& lower,
    const variant& upper) {
  using T = typename TypeTraits<kind>::NativeType;
  const bool lowerUnbounded = lower.isNull();
  const bool upperUnbounded = upper.isNull();

  if constexpr (
      kind == TypeKind::TINYINT || kind == TypeKind::SMALLINT ||
      kind == TypeKind::INTEGER || kind == TypeKind::BIGINT) {
    return std::make_unique<common::BigintRange>(
        lowerUnbounded ? std::numeric_limits<int64_t>::min() : lower.value<T>(),
        upperUnbounded ? std::numeric_limits<int64_t>::max() : upper.value<T>(),
        /*nullAllowed=*/false);
  } else if constexpr (kind == TypeKind::REAL) {
    return std::make_unique<common::FloatRange>(
        lowerUnbounded ? std::numeric_limits<float>::lowest()
                       : lower.value<T>(),
        lowerUnbounded,
        /*lowerExclusive=*/false,
        upperUnbounded ? std::numeric_limits<float>::max() : upper.value<T>(),
        upperUnbounded,
        /*upperExclusive=*/false,
        /*nullAllowed=*/false);
  } else if constexpr (kind == TypeKind::DOUBLE) {
    return std::make_unique<common::DoubleRange>(
        lowerUnbounded ? std::numeric_limits<double>::lowest()
                       : lower.value<T>(),
        lowerUnbounded,
        /*lowerExclusive=*/false,
        upperUnbounded ? std::numeric_limits<double>::max() : upper.value<T>(),
        upperUnbounded,
        /*upperExclusive=*/false,
        /*nullAllowed=*/false);
  } else if constexpr (
      kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY) {
    return std::make_unique<common::BytesRange>(
        lowerUnbounded ? "" : std::string(lower.value<StringView>()),
        lowerUnbounded,
        /*lowerExclusive=*/false,
        upperUnbounded ? "" : std::string(upper.value<StringView>()),
        upperUnbounded,
        /*upperExclusive=*/false,
        /*nullAllowed=*/false);
  } else if constexpr (kind == TypeKind::BOOLEAN) {
    VELOX_CHECK(
        !lowerUnbounded && !upperUnbounded,
        "Boolean range filter requires both bounds");
    return std::make_unique<common::BoolValue>(
        lower.value<T>(), /*nullAllowed=*/false);
  } else {
    VELOX_UNSUPPORTED(
        "Unsupported type kind for filter creation: {}",
        TypeKindName::toName(kind));
  }
}
} // namespace

std::unique_ptr<common::Filter> createPointFilter(
    const TypePtr& type,
    const variant& value) {
  VELOX_CHECK(!value.isNull(), "Value cannot be null");

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      createRangeFilterInternal, type->kind(), value, value);
}

std::unique_ptr<common::Filter> createRangeFilter(
    const TypePtr& type,
    const variant& lower,
    const variant& upper) {
  VELOX_CHECK(
      !lower.isNull() || !upper.isNull(),
      "At least one of lower or upper bound must be set");

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      createRangeFilterInternal, type->kind(), lower, upper);
}

} // namespace facebook::velox::connector
