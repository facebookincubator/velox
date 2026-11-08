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
#include "velox/connectors/paimon/PaimonSplitReader.h"

#include <limits>

namespace facebook::velox::connector::paimon {

PaimonSplitReader::PaimonSplitReader(
    const std::shared_ptr<const connector::FileConnectorSplit>& fileSplit,
    std::shared_ptr<const PaimonConnectorSplit> paimonSplit,
    const std::shared_ptr<const PaimonTableHandle>& tableHandle,
    const std::unordered_map<std::string, connector::FileColumnHandlePtr>*
        partitionKeys,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const connector::FileConfig>& fileConfig,
    const RowTypePtr& readerOutputType,
    const std::shared_ptr<io::IoStatistics>& dataIoStats,
    const std::shared_ptr<io::IoStatistics>& metadataIoStats,
    const std::shared_ptr<IoStats>& ioStats,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const std::shared_ptr<common::ScanSpec>& scanSpec)
    : connector::FileSplitReader(
          fileSplit,
          tableHandle,
          partitionKeys,
          connectorQueryCtx,
          fileConfig,
          readerOutputType,
          dataIoStats,
          metadataIoStats,
          ioStats,
          fileHandleFactory,
          executor,
          scanSpec),
      paimonSplit_(std::move(paimonSplit)) {
  VELOX_CHECK_NOT_NULL(paimonSplit_);

  // Build FileConnectorSplits for all data files so we can switch between
  // them during multi-file iteration.
  const auto& dataFiles = paimonSplit_->dataFiles();
  fileSplits_.reserve(dataFiles.size());
  for (const auto& dataFile : dataFiles) {
    fileSplits_.emplace_back(makeFileConnectorSplit(dataFile));
    if (!fileReadSchemas_.count(dataFile.schemaId)) {
      fileReadSchemas_.emplace(
          dataFile.schemaId, makeFileReadSchema(dataFile.schemaId));
    }
  }
}

RowTypePtr PaimonSplitReader::makeFileReadSchema(int64_t fileSchemaId) const {
  const auto& table = static_cast<const PaimonTableHandle&>(*tableHandle_);
  const auto& dataType = fileSchemaId == table.schemaId()
      ? table.rowType()
      : table.tableSchemas().at(fileSchemaId).rowType;
  const auto& dataIds = fileSchemaId == table.schemaId()
      ? table.fieldIds()
      : table.tableSchemas().at(fileSchemaId).fieldIds;

  std::unordered_map<int32_t, column_index_t> targetColumns;
  for (column_index_t i = 0; i < table.fieldIds().size(); ++i) {
    targetColumns.emplace(table.fieldIds()[i], i);
  }
  // Only top-level field evolution is handled here; nested fields are treated
  // as unchanged.
  // TODO: Support nested field evolution using historical nested field IDs.
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (column_index_t i = 0; i < dataType->size(); ++i) {
    const auto target = targetColumns.find(dataIds[i]);
    if (target == targetColumns.end()) {
      // A dropped file column must not shadow a new field with the same name.
      auto name = fmt::format("__paimon_unmapped_{}", i);
      while (table.rowType()->containsChild(name) ||
             dataType->containsChild(name)) {
        name.push_back('_');
      }
      names.push_back(std::move(name));
    } else {
      const auto& name = table.rowType()->nameOf(target->second);
      if (const auto* spec = scanSpec_->childByName(name); spec &&
          spec->columnType() == common::ScanSpec::ColumnType::kRegular) {
        const auto& type = table.rowType()->childAt(target->second);
        VELOX_USER_CHECK(
            *type == *dataType->childAt(i),
            "Paimon type evolution is not supported for field {} ('{}'), "
            "file schema {}, target schema {}: {} to {}",
            dataIds[i],
            name,
            fileSchemaId,
            table.schemaId(),
            dataType->childAt(i)->toString(),
            type->toString());
      }
      names.push_back(name);
    }
    types.push_back(dataType->childAt(i));
  }
  return ROW(std::move(names), std::move(types));
}

void PaimonSplitReader::configureBaseReaderOptions() {
  connector::FileSplitReader::configureBaseReaderOptions();
  const auto id = paimonSplit_->dataFiles()[currentFileIndex_].schemaId;
  // Ordinals here come from the file's historical append layout. DWIO
  // exposes the mapped names to the unchanged logical ScanSpec and filters.
  baseReaderOpts_.setFileSchema(fileReadSchemas_.at(id));
  baseReaderOpts_.setColumnMappingMode(
      dwio::common::ColumnMappingMode::kPosition);
}

void PaimonSplitReader::prepareSplit(
    std::shared_ptr<common::MetadataFilter> metadataFilter,
    dwio::common::RuntimeStats& runtimeStats,
    const folly::F14FastMap<std::string, std::string>& /*fileReadOps*/) {
  // Save for re-use when advancing to subsequent files.
  metadataFilter_ = std::move(metadataFilter);
  runtimeStats_ = &runtimeStats;
}

uint64_t PaimonSplitReader::next(uint64_t size, VectorPtr& output) {
  while (ensureFileSplitReader()) {
    // TODO: Apply deletion vectors using original physical file row positions.
    // TODO: Decode internal columns and row kinds for primary-key/changelog
    // files.
    const auto rowsRead = connector::FileSplitReader::next(size, output);
    if (rowsRead > 0) {
      return rowsRead;
    }
    // Current file exhausted, try the next one.
    finishFileSplitReader();
  }
  return 0;
}

bool PaimonSplitReader::ensureFileSplitReader() {
  if (baseReader_ != nullptr) {
    VELOX_CHECK_NOT_NULL(baseRowReader_);
    return true;
  }
  VELOX_CHECK_NULL(baseRowReader_);

  while (currentFileIndex_ < fileSplits_.size()) {
    // Point to the current file.
    fileSplit_ = fileSplits_[currentFileIndex_];

    configureReaderOptions(baseReaderOpts_.randomSkip());
    createReader();
    // The file reader has captured its file-order mapping. Missing columns
    // must now be resolved against the query's target schema, not that map.
    baseReaderOpts_.setFileSchema(tableHandle_->dataColumns());
    if (!emptySplit_) {
      auto rowType = getAdaptedRowType();
      if (!checkIfSplitIsEmpty(*runtimeStats_)) {
        createRowReader(metadataFilter_, std::move(rowType), std::nullopt);
      }
    }
    if (!emptySplit_) {
      return true;
    }
    // Empty preparation may leave a file reader but no row reader. Reset
    // the per-file state before advancing so the next file is prepared.
    baseReader_.reset();
    emptySplit_ = false;
    ++currentFileIndex_;
  }
  return false; // All files exhausted.
}

void PaimonSplitReader::finishFileSplitReader() {
  VELOX_CHECK_LT(currentFileIndex_, fileSplits_.size());
  VELOX_CHECK_NOT_NULL(baseReader_);
  VELOX_CHECK_NOT_NULL(baseRowReader_);
  ++currentFileIndex_;
  baseReader_ = nullptr;
  baseRowReader_ = nullptr;
  emptySplit_ = false;
}

std::shared_ptr<connector::FileConnectorSplit>
PaimonSplitReader::makeFileConnectorSplit(
    const PaimonDataFile& dataFile) const {
  return std::make_shared<connector::FileConnectorSplit>(
      paimonSplit_->connectorId,
      dataFile.path,
      paimonSplit_->fileFormat(),
      /*_start=*/0,
      /*_length=*/std::numeric_limits<uint64_t>::max(),
      /*splitWeight=*/0,
      /*cacheable=*/true,
      /*_properties=*/std::nullopt,
      paimonSplit_->partitionKeys());
}

} // namespace facebook::velox::connector::paimon
