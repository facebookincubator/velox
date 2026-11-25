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

namespace facebook::velox::connector {

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

  auto formatScopedConfigs = fileConfig->formatScopedConfigs(
      fileSplit->fileFormat, *sessionProperties);
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

} // namespace facebook::velox::connector
