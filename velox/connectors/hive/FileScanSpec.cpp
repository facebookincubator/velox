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

#include "velox/connectors/hive/FileScanSpec.h"

#include <fmt/ranges.h>
#include <algorithm>
#include "velox/common/Casts.h"
#include "velox/connectors/hive/ExtractionUtils.h"
#include "velox/connectors/hive/FileConfig.h"
#include "velox/connectors/hive/FileScanState.h"
#include "velox/expression/Expr.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::connector::hive {
namespace {
common::SubfieldFilters cloneFilters(const common::SubfieldFilters& filters) {
  common::SubfieldFilters copy;
  for (const auto& [subfield, filter] : filters) {
    copy.emplace(subfield.clone(), filter->clone());
  }
  return copy;
}
} // namespace

void FileScanSpec::processColumnHandle(const FileColumnHandlePtr& handle) {
  switch (handle->columnType()) {
    case FileColumnHandle::ColumnType::kRegular:
      break;
    case FileColumnHandle::ColumnType::kPartitionKey:
      partitionKeys_.emplace(handle->name(), handle);
      break;
    case FileColumnHandle::ColumnType::kSynthesized:
      infoColumns_.emplace(handle->name(), handle);
      break;
    case FileColumnHandle::ColumnType::kRowIndex:
      specialColumns_.rowIndex = handle->name();
      break;
    case FileColumnHandle::ColumnType::kRowId:
      specialColumns_.rowId = handle->name();
      break;
  }
}

FileScanSpec::FileScanSpec(
    const RowTypePtr& outputType,
    const FileTableHandlePtr& tableHandle,
    const ColumnHandleMap& assignments,
    const ConnectorQueryCtx* context,
    const std::shared_ptr<FileConfig>& config)
    : FileScanSpec(
          outputType,
          tableHandle,
          assignments,
          context,
          config,
          Options{}) {}

FileScanSpec::FileScanSpec(
    const RowTypePtr& outputType,
    const FileTableHandlePtr& tableHandle,
    const ColumnHandleMap& assignments,
    const ConnectorQueryCtx* context,
    const std::shared_ptr<FileConfig>& config,
    Options options)
    : outputType_(outputType),
      tableHandle_(tableHandle),
      assignments_(assignments),
      disableStatsBasedFilterReorder_(
          config->readStatsBasedFilterReorderDisabled(
              context->sessionProperties())),
      sampleRate_(tableHandle->sampleRate()) {
  auto* expressionEvaluator = context->expressionEvaluator();
  folly::F14FastMap<std::string_view, const FileColumnHandle*> columnHandles;
  // Column handles keyed on the table column name.
  for (const auto& [_, columnHandle] : assignments) {
    auto handle = checkedPointerCast<const FileColumnHandle>(columnHandle);
    const auto [it, unique] =
        columnHandles.emplace(handle->name(), handle.get());
    if (!unique) {
      // This should not happen normally, but there are cases where we get
      // duplicate assignments for partitioning columns.
      checkColumnHandleConsistent(*handle, *it->second);
      VELOX_CHECK_EQ(
          handle->columnType(),
          FileColumnHandle::ColumnType::kPartitionKey,
          "Cannot map from same table column to different outputs in table scan; a project node should be used instead: {}",
          handle->name());
      continue;
    }
    processColumnHandle(handle);
  }
  for (auto& handle : tableHandle_->filterColumnHandles()) {
    auto it = columnHandles.find(handle->name());
    if (it != columnHandles.end()) {
      checkColumnHandleConsistent(*handle, *it->second);
      continue;
    }
    processColumnHandle(handle);
  }

  std::vector<std::string> readColumnNames;
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = assignments.find(outputName);
    VELOX_CHECK(
        it != assignments.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const FileColumnHandle*>(it->second.get());
    readColumnNames.push_back(handle->name());
    for (auto& subfield : handle->requiredSubfields()) {
      VELOX_USER_CHECK_EQ(
          getColumnName(subfield),
          handle->name(),
          "Required subfield does not match column name");
      subfields_[handle->name()].push_back(&subfield);
    }
    columnPostProcessors_.push_back(handle->postProcessor());
  }

  if (config->isFileColumnNamesReadAsLowerCase(context->sessionProperties())) {
    checkColumnNameLowerCase(outputType_);
    checkColumnNameLowerCase(tableHandle_->subfieldFilters(), infoColumns_);
    checkColumnNameLowerCase(tableHandle_->remainingFilter());
  }

  originalFilters_ = cloneFilters(tableHandle_->subfieldFilters());
  filters_ = cloneFilters(originalFilters_);
  remainingFilter_ = options.extractRemainingFilter
      ? extractFiltersFromRemainingFilter(
            tableHandle_->remainingFilter(),
            expressionEvaluator,
            filters_,
            sampleRate_)
      : tableHandle_->remainingFilter();
  const auto& remainingFilter = remainingFilter_;

  if (remainingFilter) {
    auto remainingFilterExprSet = expressionEvaluator->compile(remainingFilter);
    auto& remainingFilterExpr = remainingFilterExprSet->expr(0);
    folly::F14FastMap<std::string, column_index_t> columnNames;
    for (int i = 0; i < readColumnNames.size(); ++i) {
      columnNames[readColumnNames[i]] = i;
    }
    // Capture top-level column names referenced by the remaining filter.
    // These columns must be loaded eagerly (not lazily) so the filter
    // can evaluate before lazy columns are accessed.
    folly::F14FastSet<std::string> remainingFilterColumns;
    for (auto& input : remainingFilterExpr->distinctFields()) {
      remainingFilterColumns.insert(input->field());
      auto it = columnNames.find(input->field());
      if (it != columnNames.end()) {
        if (shouldEagerlyMaterialize(*remainingFilterExpr, *input)) {
          multiReferencedFields_.push_back(it->second);
        }
        continue;
      }
      // Remaining filter may reference columns that are not used otherwise,
      // e.g. are not being projected out and are not used in range filters.
      // Make sure to add these columns to readerOutputType_.
      readColumnNames.push_back(input->field());
      readColumnTypes.push_back(input->type());
    }
    remainingFilterColumns_ = std::move(remainingFilterColumns);
    remainingFilterSubfields_ = remainingFilterExpr->extractSubfields();
    if (VLOG_IS_ON(1)) {
      VLOG(1) << fmt::format(
          "Extracted subfields from remaining filter: [{}]",
          fmt::join(remainingFilterSubfields_, ", "));
    }
    for (auto& subfield : remainingFilterSubfields_) {
      const auto& name = getColumnName(subfield);
      auto it = subfields_.find(name);
      if (it != subfields_.end()) {
        // Some subfields of the column are already projected out, we append the
        // remainingFilter subfield
        it->second.push_back(&subfield);
      } else if (columnNames.count(name) == 0) {
        // remainingFilter subfield's column is not projected out, we add the
        // column and append the subfield
        subfields_[name].push_back(&subfield);
      }
    }
  }

  readerOutputType_ =
      ROW(std::move(readColumnNames), std::move(readColumnTypes));
  // Detect extraction columns and reconfigure scanSpec_ if needed.
  bool hasExtractions = false;
  readColumnTypes = readerOutputType_->children();
  for (int outputIdx = 0; outputIdx < outputType->size(); ++outputIdx) {
    const auto& outputName = outputType->nameOf(outputIdx);
    auto it = assignments.find(outputName);
    if (it == assignments.end()) {
      continue;
    }
    auto* handle = static_cast<const FileColumnHandle*>(it->second.get());
    if (!handle->extractions().empty()) {
      // Column has extraction chains.  Read with schemaType from file, then
      // apply extraction post-read.  Extractions and requiredSubfields are
      // mutually exclusive (enforced by the column handle constructor).
      auto readerIdx = readerOutputType_->getChildIdxIfExists(handle->name());
      if (readerIdx.has_value()) {
        readColumnTypes[*readerIdx] = handle->schemaType();
        extractionColumns_[*readerIdx] = handle;
        hasExtractions = true;
      }
    }
  }

  if (hasExtractions) {
    // Rebuild readerOutputType_ with schemaType for extraction columns.
    readerOutputType_ =
        ROW(std::vector<std::string>(
                readerOutputType_->names().begin(),
                readerOutputType_->names().end()),
            std::move(readColumnTypes));
  }
}
RowTypePtr FileScanSpec::configureExtractionColumns(
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    const RowTypePtr& readerOutputType,
    memory::MemoryPool* pool) const {
  // Configure extraction columns on the ScanSpec.  For each column with
  // extractions, this:
  // 1. Sets pruning hints so DWRF/Nimble readers skip unneeded sub-streams.
  // 2. Sets a transform function on the ScanSpec node so the reader applies
  //    extraction chains and produces the output type directly.
  for (auto& [colIdx, handle] : extractionColumns_) {
    auto* fieldSpec = scanSpec->childByName(readerOutputType->nameOf(colIdx));
    if (!fieldSpec) {
      continue;
    }
    const auto& extractions = handle->extractions();
    auto extractionOutputType = handle->dataType();

    // For multiple extractions, do NOT call configureExtractionScanSpec --
    // keep ExtractionType as kNone and use full chains in the transform.
    // This ensures the text reader (which does not handle ExtractionType
    // natively) produces correct results.
    if (extractions.size() == 1) {
      configureExtractionScanSpec(
          handle->schemaType(), extractions, *fieldSpec, pool);
    }
    if (extractions.size() == 1) {
      // Store a full-chain transform so hasTransform() returns true.  This
      // signals to the delta update path that extraction is configured.
      // The full chain is captured for PrismSplitReader to replace it.
      fieldSpec->setTransform(
          [fullChain = extractions[0].chain](
              const VectorPtr& input, memory::MemoryPool* pool) -> VectorPtr {
            return applyExtractionChain(input, fullChain, pool);
          },
          extractionOutputType);
    } else {
      // Multiple extractions: do NOT set ExtractionType on the ScanSpec.
      // Use full chains in the transform so the text reader (which does
      // not handle ExtractionType natively) produces correct results.
      // TODO: Optimization: for agreeing multiple extractions, set
      // ExtractionType and use remaining chains.  Requires text reader
      // to handle ExtractionType natively.
      struct ExtractionInfo {
        std::string outputName;
        std::vector<ExtractionPathElementPtr> chain;
      };

      std::vector<ExtractionInfo> infos;
      infos.reserve(extractions.size());
      for (const auto& extraction : extractions) {
        infos.push_back({extraction.outputName, extraction.chain});
      }
      // Always need a transform for multiple extractions to assemble ROW.
      fieldSpec->setTransform(
          [infos = std::move(infos)](
              const VectorPtr& input, memory::MemoryPool* pool) -> VectorPtr {
            std::vector<VectorPtr> children;
            std::vector<std::string> names;
            std::vector<TypePtr> types;
            children.reserve(infos.size());
            names.reserve(infos.size());
            types.reserve(infos.size());
            for (const auto& info : infos) {
              VectorPtr extracted;
              if (info.chain.empty()) {
                extracted = input;
              } else {
                extracted = applyExtractionChain(input, info.chain, pool);
              }
              names.push_back(info.outputName);
              types.push_back(extracted->type());
              children.push_back(std::move(extracted));
            }
            return std::make_shared<RowVector>(
                pool,
                ROW(std::move(names), std::move(types)),
                nullptr,
                input->size(),
                std::move(children));
          },
          extractionOutputType);
    }
  }

  // Build readerProducedType_ -- the actual type the reader will produce.
  // For extraction columns where the reader handles extraction natively
  // (ExtractionType != kNone), the output type differs from schemaType.
  {
    auto names = readerOutputType->names();
    auto types = readerOutputType->children();
    bool needsSeparateType = false;
    for (auto& [colIdx, handle] : extractionColumns_) {
      auto* fieldSpec = scanSpec->childByName(readerOutputType->nameOf(colIdx));
      if (fieldSpec &&
          fieldSpec->extractionType() !=
              common::ScanSpec::ExtractionType::kNone) {
        VELOX_CHECK_LT(static_cast<size_t>(colIdx), types.size());
        types[colIdx] = handle->dataType();
        needsSeparateType = true;
      }
    }
    if (needsSeparateType) {
      return ROW(
          std::vector<std::string>(names.begin(), names.end()),
          std::move(types));
    }
  }
  return nullptr;
}

common::SubfieldFilters FileScanSpec::originalFilters() const {
  return cloneFilters(originalFilters_);
}

common::SubfieldFilters FileScanSpec::filters() const {
  return cloneFilters(filters_);
}

FileScanState FileScanSpec::newFileScanState(
    const ConnectorQueryCtx* context) const {
  return newFileScanState(readerOutputType_, subfields_, filters_, context);
}

FileScanState FileScanSpec::newFileScanState(
    const RowTypePtr& readerOutputType,
    const Subfields& subfields,
    const common::SubfieldFilters& filters,
    const ConnectorQueryCtx* context) const {
  FileScanState state;
  state.filters = cloneFilters(filters);
  state.scanSpec = makeScanSpec(
      readerOutputType,
      subfields,
      state.filters,
      /*indexColumns=*/{},
      tableHandle_->dataColumns(),
      partitionKeys_,
      infoColumns_,
      specialColumns_,
      disableStatsBasedFilterReorder_,
      context->memoryPool());
  state.readerProducedType = configureExtractionColumns(
      state.scanSpec, readerOutputType, context->memoryPool());
  // MetadataFilter installs leaves on this exact ScanSpec. Build it after
  // extraction setup and every time the physical column demands change.
  // Predicates on extraction results cannot use the original column's
  // statistics: e.g. extracting r.x changes the meaning of "r IS NULL".
  // Until predicates can be mapped to equivalent physical subfields, disable
  // metadata filtering for the whole expression if it references extraction.
  // Dropping just an affected disjunct would make OR pruning unsafe.
  const bool filterUsesExtraction = std::any_of(
      extractionColumns_.begin(),
      extractionColumns_.end(),
      [this](const auto& entry) {
        return remainingFilterColumns_.contains(entry.second->name());
      });
  if (remainingFilter_ && !filterUsesExtraction) {
    state.metadataFilter = std::make_shared<common::MetadataFilter>(
        *state.scanSpec, *remainingFilter_, context->expressionEvaluator());
  }
  return state;
}

} // namespace facebook::velox::connector::hive
